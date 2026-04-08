"""State Store — SQLite-backed persistent memory for ixtract.

Stores run history, deviation diagnoses, heuristic parameters, and
controller state. WAL mode for concurrent read performance.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ixtract.controller import ControllerDecision, ControllerState
from ixtract.diagnosis import DeviationDiagnosis


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           TEXT PRIMARY KEY,
    plan_id          TEXT NOT NULL,
    intent_hash      TEXT NOT NULL,
    source           TEXT NOT NULL,
    object           TEXT NOT NULL,
    strategy         TEXT NOT NULL,
    worker_count     INTEGER NOT NULL,
    effective_workers REAL DEFAULT 0.0,
    start_time       TEXT NOT NULL,
    end_time         TEXT,
    status           TEXT DEFAULT 'running',
    total_rows       INTEGER DEFAULT 0,
    total_bytes      INTEGER DEFAULT 0,
    avg_throughput   REAL DEFAULT 0.0,
    duration_seconds REAL DEFAULT 0.0,
    execution_context_json TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS chunks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id        TEXT NOT NULL,
    run_id          TEXT NOT NULL,
    worker_id       INTEGER,
    start_time      TEXT,
    end_time        TEXT,
    rows            INTEGER DEFAULT 0,
    bytes           INTEGER DEFAULT 0,
    status          TEXT DEFAULT 'pending',
    duration_seconds REAL DEFAULT 0.0,
    query_ms        REAL DEFAULT 0.0,
    write_ms        REAL DEFAULT 0.0,
    output_path     TEXT,
    error           TEXT
);

CREATE TABLE IF NOT EXISTS worker_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    worker_id       INTEGER NOT NULL,
    chunks_processed INTEGER DEFAULT 0,
    total_rows      INTEGER DEFAULT 0,
    total_bytes     INTEGER DEFAULT 0,
    idle_pct        REAL DEFAULT 0.0,
    blocked_pct     REAL DEFAULT 0.0,
    busy_pct        REAL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS deviations (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL,
    deviation_ratio   REAL,
    diagnosed_cause   TEXT,
    corrective_action TEXT,
    reasoning         TEXT,
    chunk_variance    REAL DEFAULT 0.0,
    throughput_change REAL DEFAULT 0.0,
    confidence        TEXT DEFAULT 'full'
);

CREATE TABLE IF NOT EXISTS heuristics (
    source      TEXT NOT NULL,
    object      TEXT NOT NULL,
    metric_key  TEXT NOT NULL,
    metric_value REAL NOT NULL,
    confidence  REAL DEFAULT 1.0,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (source, object, metric_key)
);

CREATE TABLE IF NOT EXISTS controller_state (
    source            TEXT NOT NULL,
    object            TEXT NOT NULL,
    current_workers   INTEGER NOT NULL,
    last_throughput   REAL NOT NULL,
    last_worker_count INTEGER NOT NULL,
    direction         TEXT NOT NULL,
    consecutive_holds INTEGER NOT NULL DEFAULT 0,
    converged         INTEGER NOT NULL DEFAULT 0,
    previous_workers  INTEGER NOT NULL DEFAULT 0,
    previous_avg_throughput REAL NOT NULL DEFAULT 0.0,
    updated_at        TEXT NOT NULL,
    PRIMARY KEY (source, object)
);

CREATE TABLE IF NOT EXISTS profiles (
    source          TEXT NOT NULL,
    object          TEXT NOT NULL,
    profiled_at     TEXT NOT NULL,
    row_estimate    INTEGER DEFAULT 0,
    pk_type         TEXT,
    pk_min          TEXT,
    pk_max          TEXT,
    pk_cv           REAL DEFAULT 0.0,
    avg_row_bytes   INTEGER DEFAULT 0,
    latency_p50     REAL DEFAULT 0.0,
    latency_p95     REAL DEFAULT 0.0,
    profile_json    TEXT DEFAULT '{}',
    PRIMARY KEY (source, object)
);

CREATE INDEX IF NOT EXISTS idx_runs_so ON runs(source, object);
CREATE INDEX IF NOT EXISTS idx_dev_run ON deviations(run_id);
CREATE INDEX IF NOT EXISTS idx_chunks_run ON chunks(run_id);
CREATE INDEX IF NOT EXISTS idx_worker_run ON worker_metrics(run_id);

CREATE TABLE IF NOT EXISTS benchmarks (
    source              TEXT NOT NULL,
    object              TEXT NOT NULL,
    benchmarked_at      TEXT NOT NULL,
    probe_rows          INTEGER NOT NULL,
    ranges_used         INTEGER NOT NULL,
    worker_grid         TEXT NOT NULL,    -- JSON array e.g. [1,2,4]
    throughput_json     TEXT NOT NULL,    -- JSON dict {"1": 118400, "2": 224800, ...}
    optimal_workers     INTEGER NOT NULL,
    confidence          REAL NOT NULL,
    signal_strength     REAL NOT NULL,
    curve_shape         TEXT NOT NULL,
    row_estimate        INTEGER NOT NULL,
    result_json         TEXT NOT NULL,   -- full BenchmarkResult.to_dict() for round-trip
    PRIMARY KEY (source, object)
);
"""


class StateStore:
    def __init__(self, db_path: str | Path = "ixtract_state.db") -> None:
        self.db_path = str(db_path)
        self._init()

    def _init(self) -> None:
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ── Runs ──────────────────────────────────────────────────────

    def record_run_start(self, run_id: str, plan_id: str, intent_hash: str,
                         source: str, obj: str, strategy: str, workers: int,
                         context_json: str = "{}") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO runs (run_id,plan_id,intent_hash,source,object,strategy,"
                "worker_count,start_time,execution_context_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (run_id, plan_id, intent_hash, source, obj, strategy, workers,
                 _now(), context_json),
            )

    def record_run_end(self, run_id: str, status: str, rows: int, bytes_: int,
                       throughput: float, duration: float,
                       effective_workers: float = 0.0) -> None:
        with self._conn() as c:
            c.execute(
                "UPDATE runs SET end_time=?,status=?,total_rows=?,total_bytes=?,"
                "avg_throughput=?,duration_seconds=?,effective_workers=? WHERE run_id=?",
                (_now(), status, rows, bytes_, throughput, duration, effective_workers, run_id),
            )

    def get_recent_runs(self, source: str, obj: str, limit: int = 10) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM runs WHERE source=? AND object=? ORDER BY start_time DESC LIMIT ?",
                (source, obj, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_running_count(self, source: str, exclude_run_id: Optional[str] = None) -> int:
        """Count extractions currently running on this source.

        Used by context measurement to determine concurrent extraction count.
        Excludes the calling run itself (exclude_run_id) to avoid self-counting.
        """
        with self._conn() as c:
            if exclude_run_id:
                row = c.execute(
                    "SELECT COUNT(*) FROM runs WHERE source=? AND status='running' "
                    "AND run_id != ?",
                    (source, exclude_run_id),
                ).fetchone()
            else:
                row = c.execute(
                    "SELECT COUNT(*) FROM runs WHERE source=? AND status='running'",
                    (source,),
                ).fetchone()
            return row[0] if row else 0

    def get_runs_with_context(
        self,
        source: str,
        obj: str,
        limit: int = 50,
    ) -> list[dict]:
        """Return recent completed runs with their execution context.

        Used by the context-weighted estimator to find similar historical runs.
        Returns only successful runs with avg_throughput > 0.
        Ordered oldest-first so EWMA computation gets correct time ordering.

        Returns fields: run_id, avg_throughput, execution_context_json, total_rows,
                        worker_count, start_time.
        """
        with self._conn() as c:
            rows = c.execute(
                "SELECT run_id, avg_throughput, execution_context_json, "
                "total_rows, worker_count, start_time "
                "FROM runs "
                "WHERE source=? AND object=? AND status='success' "
                "AND avg_throughput > 0 "
                "ORDER BY start_time DESC LIMIT ?",
                (source, obj, limit),
            ).fetchall()
        # Return oldest-first for EWMA (reversed from the DESC query)
        return list(reversed([dict(r) for r in rows]))

    # ── Deviations ────────────────────────────────────────────────

    def record_deviation(self, run_id: str, d: DeviationDiagnosis) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO deviations (run_id,deviation_ratio,diagnosed_cause,"
                "corrective_action,reasoning,chunk_variance,throughput_change) "
                "VALUES (?,?,?,?,?,?,?)",
                (run_id, d.deviation_ratio, d.category.value,
                 d.corrective_action, d.reasoning, d.chunk_variance, d.throughput_change_pct),
            )

    # ── Heuristics ────────────────────────────────────────────────

    def get_heuristic(self, source: str, obj: str, key: str) -> Optional[float]:
        with self._conn() as c:
            r = c.execute(
                "SELECT metric_value FROM heuristics WHERE source=? AND object=? AND metric_key=?",
                (source, obj, key),
            ).fetchone()
            return r["metric_value"] if r else None

    def set_heuristic(self, source: str, obj: str, key: str,
                      value: float, confidence: float = 1.0) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO heuristics (source,object,metric_key,metric_value,confidence,updated_at) "
                "VALUES (?,?,?,?,?,?) "
                "ON CONFLICT(source,object,metric_key) "
                "DO UPDATE SET metric_value=excluded.metric_value,"
                "confidence=excluded.confidence,updated_at=excluded.updated_at",
                (source, obj, key, value, confidence, _now()),
            )

    # ── Controller State ──────────────────────────────────────────

    def get_controller_state(self, source: str, obj: str) -> Optional[ControllerState]:
        with self._conn() as c:
            r = c.execute(
                "SELECT * FROM controller_state WHERE source=? AND object=?",
                (source, obj),
            ).fetchone()
            if not r:
                return None
            return ControllerState(
                current_workers=r["current_workers"],
                previous_workers=r["previous_workers"],
                previous_avg_throughput=r["previous_avg_throughput"],
                converged=bool(r["converged"]),
                last_throughput=r["last_throughput"],
                last_worker_count=r["last_worker_count"],
                direction=ControllerDecision(r["direction"]),
                consecutive_holds=r["consecutive_holds"],
            )

    def save_controller_state(self, source: str, obj: str, st: ControllerState) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO controller_state "
                "(source,object,current_workers,last_throughput,last_worker_count,"
                "direction,consecutive_holds,converged,"
                "previous_workers,previous_avg_throughput,updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(source,object) DO UPDATE SET "
                "current_workers=excluded.current_workers,"
                "last_throughput=excluded.last_throughput,"
                "last_worker_count=excluded.last_worker_count,"
                "direction=excluded.direction,"
                "consecutive_holds=excluded.consecutive_holds,"
                "converged=excluded.converged,"
                "previous_workers=excluded.previous_workers,"
                "previous_avg_throughput=excluded.previous_avg_throughput,"
                "updated_at=excluded.updated_at",
                (source, obj, st.current_workers, st.last_throughput,
                 st.last_worker_count, st.direction.value,
                 st.consecutive_holds, int(st.converged),
                 st.previous_workers, st.previous_avg_throughput, _now()),
            )

    # ── Chunk Results ─────────────────────────────────────────────

    def record_chunk(self, run_id: str, chunk_id: str, worker_id: int,
                     rows: int, bytes_: int, status: str, duration: float,
                     query_ms: float = 0.0, write_ms: float = 0.0,
                     output_path: str = "", error: str = "") -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO chunks (chunk_id,run_id,worker_id,start_time,rows,bytes,"
                "status,duration_seconds,query_ms,write_ms,output_path,error) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (chunk_id, run_id, worker_id, _now(), rows, bytes_,
                 status, duration, query_ms, write_ms, output_path, error),
            )

    def get_chunks(self, run_id: str) -> list[dict]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM chunks WHERE run_id=? ORDER BY chunk_id",
                (run_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Worker Metrics ────────────────────────────────────────────

    def record_worker_metrics(self, run_id: str, worker_id: int,
                              chunks_processed: int, total_rows: int,
                              total_bytes: int, idle_pct: float = 0.0,
                              blocked_pct: float = 0.0, busy_pct: float = 0.0) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO worker_metrics "
                "(run_id,worker_id,chunks_processed,total_rows,total_bytes,"
                "idle_pct,blocked_pct,busy_pct) VALUES (?,?,?,?,?,?,?,?)",
                (run_id, worker_id, chunks_processed, total_rows, total_bytes,
                 idle_pct, blocked_pct, busy_pct),
            )

    # ── Profiles ──────────────────────────────────────────────────

    def save_profile(self, source: str, obj: str, profile_json: str,
                     row_estimate: int = 0, pk_type: str = "",
                     pk_min: str = "", pk_max: str = "", pk_cv: float = 0.0,
                     avg_row_bytes: int = 0, latency_p50: float = 0.0,
                     latency_p95: float = 0.0) -> None:
        with self._conn() as c:
            c.execute(
                "INSERT INTO profiles "
                "(source,object,profiled_at,row_estimate,pk_type,pk_min,pk_max,"
                "pk_cv,avg_row_bytes,latency_p50,latency_p95,profile_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(source,object) DO UPDATE SET "
                "profiled_at=excluded.profiled_at,row_estimate=excluded.row_estimate,"
                "pk_type=excluded.pk_type,pk_min=excluded.pk_min,pk_max=excluded.pk_max,"
                "pk_cv=excluded.pk_cv,avg_row_bytes=excluded.avg_row_bytes,"
                "latency_p50=excluded.latency_p50,latency_p95=excluded.latency_p95,"
                "profile_json=excluded.profile_json",
                (source, obj, _now(), row_estimate, pk_type, pk_min, pk_max,
                 pk_cv, avg_row_bytes, latency_p50, latency_p95, profile_json),
            )

    def get_profile(self, source: str, obj: str) -> Optional[dict]:
        with self._conn() as c:
            r = c.execute(
                "SELECT * FROM profiles WHERE source=? AND object=?",
                (source, obj),
            ).fetchone()
            return dict(r) if r else None


    # ── Benchmarks ────────────────────────────────────────────────

    def save_benchmark(self, source: str, obj: str, result: "BenchmarkResult") -> None:
        """Persist a BenchmarkResult. Upserts on (source, object)."""
        from ixtract.benchmarker import BenchmarkResult
        import json
        d = result.to_dict()
        with self._conn() as c:
            c.execute(
                "INSERT INTO benchmarks "
                "(source,object,benchmarked_at,probe_rows,ranges_used,worker_grid,"
                "throughput_json,optimal_workers,confidence,signal_strength,"
                "curve_shape,row_estimate,result_json) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(source,object) DO UPDATE SET "
                "benchmarked_at=excluded.benchmarked_at,"
                "probe_rows=excluded.probe_rows,"
                "ranges_used=excluded.ranges_used,"
                "worker_grid=excluded.worker_grid,"
                "throughput_json=excluded.throughput_json,"
                "optimal_workers=excluded.optimal_workers,"
                "confidence=excluded.confidence,"
                "signal_strength=excluded.signal_strength,"
                "curve_shape=excluded.curve_shape,"
                "row_estimate=excluded.row_estimate,"
                "result_json=excluded.result_json",
                (
                    source, obj,
                    d["benchmarked_at"],
                    d["probe_rows"],
                    d["ranges_used"],
                    json.dumps(d["worker_grid"]),
                    json.dumps(d["throughput_by_workers"]),
                    d["optimal_workers"],
                    d["confidence"],
                    d["signal_strength"],
                    d["curve_shape"],
                    d["row_estimate_at_benchmark"],
                    json.dumps(d),
                ),
            )

    def get_benchmark(self, source: str, obj: str) -> "Optional[BenchmarkResult]":
        """Load a BenchmarkResult from the state store, or None if absent."""
        from ixtract.benchmarker import BenchmarkResult
        import json
        with self._conn() as c:
            r = c.execute(
                "SELECT result_json FROM benchmarks WHERE source=? AND object=?",
                (source, obj),
            ).fetchone()
        if not r:
            return None
        return BenchmarkResult.from_dict(json.loads(r["result_json"]))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
