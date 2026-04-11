"""MySQL connector — Phase 2C source implementation.

Uses SQLAlchemy Core with the pymysql driver (pure Python, no system deps).
Supports metadata discovery, range chunk extraction, latency profiling,
connection monitoring, PK distribution analysis, and InnoDB snapshot isolation.

Snapshot isolation model:
    Uses START TRANSACTION WITH CONSISTENT SNAPSHOT, which must be the FIRST
    statement after connecting — any prior query resets the snapshot point.

    Per-worker consistency (not global):
        Each worker gets its own connection and its own snapshot. There is no
        global snapshot shared across workers. Rows committed between worker
        connection times are visible to later-connecting workers. Accepted
        for Phase 2C. Global snapshot deferred to Phase 3.

    Undo log impact:
        Long-running snapshot transactions hold InnoDB undo logs for the
        duration of the run. Schedule extractions during low-write windows.

Storage engine constraint:
    Only InnoDB is supported. MyISAM and other engines have no MVCC, so
    snapshot isolation is impossible. metadata() raises RuntimeError if
    the table is not InnoDB.

Primary key constraint:
    Tables without a primary key cannot be range-chunked — RuntimeError.
    Tables with composite primary keys are not supported in Phase 2C — RuntimeError.

Row estimate quality:
    information_schema.tables.table_rows is an estimate for InnoDB (exact
    for MyISAM). Errors of 10-30% are common. Treated as weak signal.
    No COUNT(*) fallback in Phase 2C; profiler's conservative bias handles this.

Type handling:
    MySQL types are passed through as strings without normalization. Opaque.

Deferred (Phase 3+):
    - Global snapshot coordinator (consistent read across all workers)
    - COUNT(*) fallback for small tables
    - GTID-aware incremental extraction
    - Binlog-based CDC
    - Composite PK support
    - Time-window chunking for non-integer PKs
"""
from __future__ import annotations

import time
from typing import Any, Iterator, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection

from ixtract.connectors.base import (
    BaseConnector, ColumnInfo, ObjectMetadata, LatencyProfile, SourceConnections,
)

DEFAULT_STREAM_BATCH_SIZE = 10_000
SUPPORTED_ENGINES = {"InnoDB", "innodb"}


class MySQLConnector(BaseConnector):
    """MySQL/MariaDB connector using SQLAlchemy Core with pymysql driver.

    InnoDB only. No composite PKs. Snapshot per-worker (not global).
    """

    def __init__(self) -> None:
        self._engine: Optional[Engine] = None
        self._config: dict[str, Any] = {}

    def connect(self, config: dict[str, Any]) -> None:
        """Connect to MySQL.

        Config keys:
            host, port, database, user, password
            OR connection_string (full SQLAlchemy URL, mysql+pymysql scheme)
            charset:     default utf8mb4
            pool_size:   default 10
        """
        self._config = config
        if "connection_string" in config:
            url = config["connection_string"]
        else:
            host = config.get("host", "localhost")
            port = config.get("port", 3306)
            db = config["database"]
            user = config.get("user", "")
            pw = config.get("password", "")
            charset = config.get("charset", "utf8mb4")
            url = f"mysql+pymysql://{user}:{pw}@{host}:{port}/{db}?charset={charset}"

        pool_size = config.get("pool_size", 10)
        self._engine = create_engine(
            url,
            pool_size=pool_size,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    def metadata(self, object_name: str) -> ObjectMetadata:
        """Query MySQL information_schema for table metadata.

        Raises RuntimeError if:
            - Table not found in database
            - Storage engine is not InnoDB (no MVCC, snapshot impossible)
            - Table has no primary key (cannot range-chunk)
            - Table has a composite primary key (Phase 2C limitation)
        """
        engine = self._require_engine()
        db_name = engine.url.database

        with engine.connect() as conn:
            # Storage engine check — fail before any planning
            engine_row = conn.execute(text(
                "SELECT engine FROM information_schema.tables "
                "WHERE table_schema = :db AND table_name = :table"
            ), {"db": db_name, "table": object_name}).fetchone()

            if not engine_row:
                raise RuntimeError(
                    f"Table '{object_name}' not found in database '{db_name}'."
                )
            storage_engine = engine_row[0]
            if storage_engine not in SUPPORTED_ENGINES:
                raise RuntimeError(
                    f"Table '{object_name}' uses storage engine '{storage_engine}'. "
                    f"Only InnoDB is supported — {storage_engine} has no MVCC "
                    f"and cannot provide snapshot isolation."
                )

            # Row count estimate and size (weak signal for InnoDB)
            size_result = conn.execute(text(
                "SELECT table_rows, data_length + index_length AS total_size "
                "FROM information_schema.tables "
                "WHERE table_schema = :db AND table_name = :table"
            ), {"db": db_name, "table": object_name}).fetchone()

            row_est = int(size_result[0] or 0) if size_result else 0
            size_est = int(size_result[1] or 0) if size_result else 0

            # Column info
            cols_result = conn.execute(text(
                "SELECT column_name, data_type, is_nullable "
                "FROM information_schema.columns "
                "WHERE table_schema = :db AND table_name = :table "
                "ORDER BY ordinal_position"
            ), {"db": db_name, "table": object_name}).fetchall()

            # Primary key — all columns of PRIMARY constraint
            pk_result = conn.execute(text(
                "SELECT column_name "
                "FROM information_schema.key_column_usage "
                "WHERE table_schema = :db AND table_name = :table "
                "AND constraint_name = 'PRIMARY' "
                "ORDER BY ordinal_position"
            ), {"db": db_name, "table": object_name}).fetchall()

            if not pk_result:
                raise RuntimeError(
                    f"Table '{object_name}' has no primary key. "
                    f"ixtract requires a primary key for range-based chunking."
                )
            if len(pk_result) > 1:
                pk_cols = ", ".join(r[0] for r in pk_result)
                raise RuntimeError(
                    f"Table '{object_name}' has a composite primary key ({pk_cols}). "
                    f"Composite primary keys are not supported in Phase 2C. "
                    f"Use a single-column integer PK or add a surrogate key."
                )

            pk_name = pk_result[0][0]

            # PK column type
            pk_type_result = conn.execute(text(
                "SELECT data_type FROM information_schema.columns "
                "WHERE table_schema = :db AND table_name = :table "
                "AND column_name = :col"
            ), {"db": db_name, "table": object_name, "col": pk_name}).fetchone()
            pk_type = pk_type_result[0] if pk_type_result else None

            columns = tuple(
                ColumnInfo(
                    name=row[0],
                    data_type=row[1],
                    nullable=(row[2] == "YES"),
                    is_primary_key=(row[0] == pk_name),
                )
                for row in cols_result
            )

            # PK range
            pk_min = pk_max = None
            range_result = conn.execute(text(
                f"SELECT MIN(`{pk_name}`), MAX(`{pk_name}`) FROM `{object_name}`"
            )).fetchone()
            if range_result:
                pk_min, pk_max = range_result[0], range_result[1]

        return ObjectMetadata(
            object_name=object_name,
            row_estimate=max(0, row_est),
            size_estimate_bytes=max(0, size_est),
            columns=columns,
            primary_key=pk_name,
            primary_key_type=pk_type,
            pk_min=pk_min,
            pk_max=pk_max,
        )

    def extract_chunk(
        self,
        object_name: str,
        chunk_query: str,
        params: dict[str, Any] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Extract a chunk via the given query, yielding batches.

        Non-snapshot version — used when snapshot connection is unavailable.
        """
        engine = self._require_engine()
        batch_size = self._config.get("stream_batch_size", DEFAULT_STREAM_BATCH_SIZE)

        with engine.connect() as conn:
            result = conn.execution_options(stream_results=True).execute(
                text(chunk_query), params or {}
            )
            columns = list(result.keys())
            batch: list[dict[str, Any]] = []

            for row in result:
                batch.append(dict(zip(columns, row)))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

    def extract_chunk_snapshot(
        self,
        conn: Connection,
        chunk_query: str,
        params: dict[str, Any] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Extract a chunk within an existing snapshot connection.

        The connection must have been created via create_snapshot_connection().
        """
        batch_size = self._config.get("stream_batch_size", DEFAULT_STREAM_BATCH_SIZE)
        result = conn.execution_options(stream_results=True).execute(
            text(chunk_query), params or {}
        )
        columns = list(result.keys())
        batch: list[dict[str, Any]] = []

        for row in result:
            batch.append(dict(zip(columns, row)))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def estimate_latency(self, object_name: str) -> LatencyProfile:
        """Measure query latency with 5 lightweight probes."""
        engine = self._require_engine()
        latencies: list[float] = []
        conn_start = time.perf_counter()

        with engine.connect() as conn:
            conn_ms = (time.perf_counter() - conn_start) * 1000
            for _ in range(5):
                start = time.perf_counter()
                conn.execute(text(f"SELECT 1 FROM `{object_name}` LIMIT 1"))
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

        latencies.sort()
        return LatencyProfile(
            p50_ms=latencies[2],
            p95_ms=latencies[4],
            connection_ms=conn_ms,
            sample_count=5,
        )

    def get_connections(self) -> SourceConnections:
        """Query current connection utilization from MySQL status variables."""
        engine = self._require_engine()
        with engine.connect() as conn:
            max_conn_row = conn.execute(text(
                "SHOW VARIABLES LIKE 'max_connections'"
            )).fetchone()
            max_conn = int(max_conn_row[1]) if max_conn_row else 100

            active_row = conn.execute(text(
                "SHOW STATUS LIKE 'Threads_connected'"
            )).fetchone()
            active = int(active_row[1]) if active_row else 0

        return SourceConnections(max_connections=max_conn, active_connections=active)

    def get_pk_distribution(self, object_name: str, num_buckets: int = 10) -> list[int]:
        """Count rows in equal-width PK range buckets for skew detection."""
        engine = self._require_engine()
        meta = self.metadata(object_name)

        if not meta.primary_key or meta.pk_min is None or meta.pk_max is None:
            return []

        pk = meta.primary_key
        pk_min, pk_max = meta.pk_min, meta.pk_max

        if isinstance(pk_min, (int, float)) and isinstance(pk_max, (int, float)):
            bucket_width = (pk_max - pk_min) / num_buckets
            if bucket_width <= 0:
                return [meta.row_estimate]

            counts: list[int] = []
            with engine.connect() as conn:
                for i in range(num_buckets):
                    lo = pk_min + i * bucket_width
                    hi = pk_min + (i + 1) * bucket_width
                    where = (
                        f"`{pk}` >= {lo}" +
                        (f" AND `{pk}` < {hi}" if i < num_buckets - 1 else "")
                    )
                    count = conn.execute(text(
                        f"SELECT COUNT(*) FROM `{object_name}` WHERE {where}"
                    )).scalar() or 0
                    counts.append(count)
            return counts

        return []

    def create_snapshot_connection(self) -> Connection:
        """Create an InnoDB consistent snapshot connection.

        CRITICAL: START TRANSACTION WITH CONSISTENT SNAPSHOT must be the
        FIRST statement executed after connecting. Any prior query — including
        SELECT 1 or SET SESSION — resets the snapshot point and breaks
        consistency guarantees.

        Per-worker isolation (Phase 2C):
            Each worker gets its own connection and snapshot. No global snapshot
            across workers. Global coordinator deferred to Phase 3.

        Undo log impact:
            This transaction holds InnoDB undo logs for the run duration.
            Schedule extractions during low-write windows when possible.
        """
        engine = self._require_engine()
        conn = engine.connect()
        # FIRST statement — no prior queries allowed
        conn.execute(text("START TRANSACTION WITH CONSISTENT SNAPSHOT"))
        return conn

    def close(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def _require_engine(self) -> Engine:
        if not self._engine:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._engine
