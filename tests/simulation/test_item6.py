"""Tests for Item 6: source_type propagation + anomaly detection wiring."""
from __future__ import annotations

import pytest
from click.testing import CliRunner
from ixtract.cli.main import cli, _infer_source_type, _compute_health_verdict, _fmt_trend, _ANOMALY_CAUSES
from ixtract.state import StateStore
from ixtract.diagnosis import detect_anomaly, ANOMALY_MIN_BASELINE


# ── Unit: _infer_source_type ──────────────────────────────────────────

class TestInferSourceType:
    def test_explicit_overrides_everything(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        result = _infer_source_type(store, "orders", explicit="mysql")
        assert result == "mysql"

    def test_infers_from_last_run(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.record_run_start(
            "rx-001", "plan-001", "hash-001",
            "sqlserver", "orders", "range_chunking", 4,
        )
        store.record_run_end("rx-001", "success", 100, 1000, 50000.0, 1.0)
        result = _infer_source_type(store, "orders")
        assert result == "sqlserver"

    def test_raises_when_no_runs_and_no_explicit(self, tmp_path):
        import click
        store = StateStore(str(tmp_path / "test.db"))
        with pytest.raises(click.UsageError) as exc:
            _infer_source_type(store, "nonexistent")
        assert "No runs found" in str(exc.value)
        assert "ixtract execute" in str(exc.value)

    def test_error_message_is_generic_not_connector_specific(self, tmp_path):
        """Error message must not mention a specific connector."""
        import click
        store = StateStore(str(tmp_path / "test.db"))
        with pytest.raises(click.UsageError) as exc:
            _infer_source_type(store, "orders")
        msg = str(exc.value)
        assert "<object>" in msg or "execute <object>" in msg or "ixtract execute" in msg
        # Must NOT be connector-specific
        assert "--database ixtract_test" not in msg

    def test_explicit_takes_priority_over_run_history(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.record_run_start(
            "rx-001", "plan-001", "hash-001",
            "postgresql", "orders", "range_chunking", 8,
        )
        store.record_run_end("rx-001", "success", 1000, 10000, 856000.0, 1.0)
        result = _infer_source_type(store, "orders", explicit="mysql")
        assert result == "mysql"


# ── Unit: get_recent_throughputs ─────────────────────────────────────

class TestGetRecentThroughputs:
    def _seed_runs(self, store, source, obj, throughputs, statuses=None):
        for i, tp in enumerate(throughputs):
            status = (statuses[i] if statuses else "success")
            rid = f"rx-{source[:2]}-{obj[:3]}-{i:03d}"
            store.record_run_start(
                rid, f"plan-{i}", f"hash-{i}",
                source, obj, "range_chunking", 8,
            )
            store.record_run_end(rid, status, 1000000, 50000000, tp, 11.0)

    def test_returns_oldest_first(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        tps = [800_000.0, 850_000.0, 900_000.0]
        self._seed_runs(store, "postgresql", "orders", tps)
        result = store.get_recent_throughputs("postgresql", "orders")
        assert result[0] < result[-1]  # oldest first = ascending

    def test_scoped_by_source_type(self, tmp_path):
        """PostgreSQL and SQL Server baselines must not mix."""
        store = StateStore(str(tmp_path / "test.db"))
        self._seed_runs(store, "postgresql", "orders", [856_000.0] * 10)
        self._seed_runs(store, "sqlserver", "orders", [8_700.0] * 10)

        pg_tps = store.get_recent_throughputs("postgresql", "orders")
        ss_tps = store.get_recent_throughputs("sqlserver", "orders")

        assert all(tp > 500_000 for tp in pg_tps), "pg baseline contaminated"
        assert all(tp < 50_000 for tp in ss_tps), "ss baseline contaminated"
        assert len(pg_tps) == 10
        assert len(ss_tps) == 10

    def test_excludes_failed_runs(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        tps = [856_000.0] * 5
        statuses = ["success", "failed", "success", "failed", "success"]
        self._seed_runs(store, "postgresql", "orders", tps, statuses)
        result = store.get_recent_throughputs("postgresql", "orders")
        assert len(result) == 3

    def test_empty_when_no_runs(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        result = store.get_recent_throughputs("postgresql", "orders")
        assert result == []

    def test_respects_limit(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        self._seed_runs(store, "postgresql", "orders", [856_000.0] * 25)
        result = store.get_recent_throughputs("postgresql", "orders", limit=20)
        assert len(result) == 20


# ── Unit: anomaly detection with scoped baseline ──────────────────────

class TestAnomalyDetectionScoped:
    def test_no_cross_contamination(self):
        """SQL Server baseline must not affect PostgreSQL anomaly detection."""
        pg_baseline = [856_000.0 + i * 1000 for i in range(10)]
        anomaly = detect_anomaly(8_700.0, pg_baseline)
        assert anomaly.is_anomaly
        assert anomaly.direction == "degradation"
        assert anomaly.z_score > 10

        ss_baseline = [8_700.0 + i * 100 for i in range(10)]
        normal = detect_anomaly(8_700.0, ss_baseline)
        assert not normal.is_anomaly

    def test_insufficient_baseline_returns_no_anomaly(self):
        result = detect_anomaly(100.0, [200.0, 210.0])
        assert not result.is_anomaly
        assert result.baseline_run_count == 2

    def test_anomaly_detection_independent_per_source(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))

        for i in range(10):
            rid = f"rx-pg-{i:03d}"
            store.record_run_start(rid, f"p{i}", f"h{i}",
                                   "postgresql", "orders", "range_chunking", 8)
            store.record_run_end(rid, "success", 1000000, 50000000, 856_000.0, 11.0)

        for i in range(10):
            rid = f"rx-ss-{i:03d}"
            store.record_run_start(rid, f"p{i}", f"h{i}",
                                   "sqlserver", "orders", "range_chunking", 4)
            store.record_run_end(rid, "success", 1000000, 50000000, 8_700.0, 115.0)

        pg_baseline = store.get_recent_throughputs("postgresql", "orders")
        ss_baseline = store.get_recent_throughputs("sqlserver", "orders")

        assert detect_anomaly(8_700.0, pg_baseline).is_anomaly
        assert not detect_anomaly(8_700.0, ss_baseline).is_anomaly


# ── CLI: history command source inference ─────────────────────────────

class TestHistorySourceInference:
    def test_history_finds_mysql_runs(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.record_run_start(
            "rx-mysql-001", "plan-001", "hash-001",
            "mysql", "customers", "range_chunking", 4,
        )
        store.record_run_end("rx-mysql-001", "success", 500_000, 25_000_000, 420_000.0, 11.9)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "history", "customers",
            "--state-db", str(tmp_path / "test.db"),
        ])
        assert result.exit_code == 0
        assert "No runs found" not in result.output
        assert "customers" in result.output

    def test_history_no_runs_gives_clear_error(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, [
            "history", "nonexistent",
            "--state-db", str(tmp_path / "test.db"),
        ])
        assert result.exit_code != 0 or "No runs found" in result.output or \
               "ixtract execute" in result.output

    def test_history_header_shows_source(self, tmp_path):
        """History output must include source type in header."""
        store = StateStore(str(tmp_path / "test.db"))
        store.record_run_start(
            "rx-pg-001", "plan-001", "hash-001",
            "postgresql", "orders", "range_chunking", 8,
        )
        store.record_run_end("rx-pg-001", "success", 1_000_000, 50_000_000, 856_000.0, 11.7)

        runner = CliRunner()
        result = runner.invoke(cli, [
            "history", "orders",
            "--state-db", str(tmp_path / "test.db"),
        ])
        assert result.exit_code == 0
        assert "postgresql" in result.output


# ── Regression: existing PostgreSQL behaviour unchanged ───────────────

class TestPostgreSQLRegression:
    def test_baseline_query_returns_correct_throughputs(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        expected = [800_000.0, 826_000.0, 856_000.0]
        for i, tp in enumerate(expected):
            rid = f"rx-pg-{i:03d}"
            store.record_run_start(rid, f"p{i}", f"h{i}",
                                   "postgresql", "orders", "range_chunking", 8)
            store.record_run_end(rid, "success", 1_000_000, 50_000_000, tp, 11.0)
        result = store.get_recent_throughputs("postgresql", "orders")
        assert result == expected  # oldest-first

    def test_anomaly_not_flagged_for_stable_postgresql(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        for i in range(10):
            rid = f"rx-pg-{i:03d}"
            store.record_run_start(rid, f"p{i}", f"h{i}",
                                   "postgresql", "orders", "range_chunking", 8)
            store.record_run_end(rid, "success", 1_000_000, 50_000_000,
                                 856_000.0 + (i * 1000), 11.0)
        baseline = store.get_recent_throughputs("postgresql", "orders")
        result = detect_anomaly(856_000.0, baseline)
        assert not result.is_anomaly


# ── Unit: inspect exit codes ──────────────────────────────────────────

class TestInspectExitCodes:
    def _seed_healthy_run(self, store, source="postgresql", obj="orders"):
        from ixtract.controller import ControllerState, ControllerDecision
        import json
        store.record_run_start(
            "rx-001", "plan-001", "hash-001",
            source, obj, "range_chunking", 8,
        )
        store.record_run_end("rx-001", "success", 1_000_000, 50_000_000, 856_000.0, 11.7)
        ctrl = ControllerState(
            current_workers=8, previous_workers=8,
            previous_avg_throughput=850_000.0, converged=True,
            last_throughput=856_000.0, last_worker_count=8,
            direction=ControllerDecision.HOLD, consecutive_holds=3,
        )
        store.save_controller_state(source, obj, ctrl)
        # Seed a fresh profile so profile_stale=False → HEALTHY verdict
        store.save_profile(
            source, obj,
            profile_json=json.dumps({"row_estimate": 1_000_000}),
            row_estimate=1_000_000,
        )

    def test_healthy_exits_0(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        self._seed_healthy_run(store)
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "orders", "--state-db", str(tmp_path / "test.db")])
        assert result.exit_code == 0
        assert "HEALTHY" in result.output

    def test_degraded_exits_2(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        store.record_run_start(
            "rx-fail-001", "plan-001", "hash-001",
            "postgresql", "orders", "range_chunking", 8,
        )
        store.record_run_end("rx-fail-001", "failed", 0, 0, 0.0, 5.2)
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "orders", "--state-db", str(tmp_path / "test.db")])
        assert result.exit_code == 2
        assert "DEGRADED" in result.output

    def test_no_runs_exits_1(self, tmp_path):
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "nonexistent", "--state-db", str(tmp_path / "test.db")])
        assert result.exit_code != 0

    def test_header_shows_source(self, tmp_path):
        store = StateStore(str(tmp_path / "test.db"))
        self._seed_healthy_run(store)
        runner = CliRunner()
        result = runner.invoke(cli, ["inspect", "orders", "--state-db", str(tmp_path / "test.db")])
        assert "postgresql" in result.output
        assert "Inspect: orders (postgresql)" in result.output
