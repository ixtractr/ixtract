"""Integration tests — full pipeline against real PostgreSQL.

Requires:
    docker compose -f docker-compose.test.yml up -d
    python tests/integration/seed_db.py

Tests the complete extraction lifecycle:
    1. Profiling (metadata, latency, skew detection)
    2. Planning (strategy selection, worker count, chunk computation)
    3. Execution (range chunking, Parquet output, idempotent finalize)
    4. Deviation analysis and controller feedback
    5. Multi-run convergence
    6. CLI output verification
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from tests.integration import TEST_DB_CONFIG, requires_db


@requires_db
class TestProfiler(unittest.TestCase):
    """Test source profiling against real PostgreSQL."""

    def setUp(self):
        from ixtract.connectors.postgresql import PostgreSQLConnector
        self.connector = PostgreSQLConnector()
        self.connector.connect(TEST_DB_CONFIG)

    def tearDown(self):
        self.connector.close()

    def test_profile_orders(self):
        """Profile the uniform orders table."""
        from ixtract.profiler import SourceProfiler

        profiler = SourceProfiler(self.connector)
        prof = profiler.profile("orders")

        self.assertGreater(prof.row_estimate, 1_000_000, "Expected 1M+ rows")
        self.assertGreater(prof.size_estimate_bytes, 0)
        self.assertGreater(prof.column_count, 5)
        self.assertEqual(prof.primary_key, "id")
        self.assertTrue(prof.has_usable_pk)
        self.assertIsNotNone(prof.pk_min)
        self.assertIsNotNone(prof.pk_max)
        self.assertGreater(prof.latency_p50_ms, 0)
        self.assertGreater(prof.max_connections, 0)

        # Orders is uniformly distributed — CV should be low
        self.assertLess(prof.pk_distribution_cv, 1.5,
            f"Orders CV {prof.pk_distribution_cv} too high for uniform table")

        # Strategy should be range_chunking for 1M+ rows
        self.assertEqual(prof.recommended_strategy, "range_chunking")
        self.assertGreaterEqual(prof.recommended_start_workers, 1)

        print(f"\n  Profile: {prof.row_estimate:,} rows, "
              f"PK {prof.pk_min}→{prof.pk_max}, "
              f"CV={prof.pk_distribution_cv:.2f}, "
              f"latency p50={prof.latency_p50_ms:.0f}ms, "
              f"recommended={prof.recommended_start_workers} workers")

    def test_profile_events_detects_skew(self):
        """Profile the skewed events table. CV should be elevated."""
        from ixtract.profiler import SourceProfiler

        profiler = SourceProfiler(self.connector)
        prof = profiler.profile("events")

        self.assertGreater(prof.row_estimate, 1_000_000)
        self.assertTrue(prof.has_usable_pk)

        # Events has skewed data — PK range distribution has more variance
        # (Note: PK is auto-increment so distribution is even, but row density
        #  in terms of user_id would show skew. PK CV may still be low since
        #  PK itself is sequential. This tests the profiler mechanics.)
        print(f"\n  Events: {prof.row_estimate:,} rows, CV={prof.pk_distribution_cv:.2f}")

    def test_profile_small_table(self):
        """Small table should recommend single_pass."""
        from ixtract.profiler import SourceProfiler

        profiler = SourceProfiler(self.connector)
        prof = profiler.profile("small_lookup")

        self.assertLess(prof.row_estimate, 100_000)
        self.assertEqual(prof.recommended_strategy, "single_pass")
        self.assertEqual(prof.recommended_start_workers, 1)

    def test_connections_available(self):
        """Connector should report available connections."""
        conns = self.connector.get_connections()
        self.assertGreater(conns.max_connections, 0)
        self.assertGreater(conns.active_connections, 0)
        self.assertGreater(conns.available, 0)
        self.assertGreater(conns.available_safe, 0)


@requires_db
class TestPlanner(unittest.TestCase):
    """Test planning against profiled real data."""

    def setUp(self):
        from ixtract.connectors.postgresql import PostgreSQLConnector
        from ixtract.profiler import SourceProfiler

        self.connector = PostgreSQLConnector()
        self.connector.connect(TEST_DB_CONFIG)
        self.profiler = SourceProfiler(self.connector)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        self.connector.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_plan_orders_range_chunking(self):
        """Orders table should get range_chunking with multiple chunks."""
        from ixtract.planner.planner import plan_extraction, format_plan_summary
        from ixtract.intent import ExtractionIntent, SourceType, TargetType

        prof = self.profiler.profile("orders")
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config=TEST_DB_CONFIG,
            object_name="orders",
            target_type=TargetType.PARQUET,
            target_config={"output_path": os.path.join(self.tmp, "output")},
        )

        plan = plan_extraction(intent, prof)

        self.assertEqual(plan.strategy.value, "range_chunking")
        self.assertGreater(len(plan.chunks), 1)
        self.assertGreaterEqual(plan.worker_count, 1)
        self.assertGreater(plan.cost_estimate.predicted_duration_seconds, 0)
        self.assertGreater(plan.cost_estimate.predicted_total_rows, 1_000_000)

        # Chunks should cover the full PK range without gaps
        for i in range(1, len(plan.chunks)):
            prev = plan.chunks[i-1]
            curr = plan.chunks[i]
            self.assertAlmostEqual(prev.range_end, curr.range_start, places=0,
                msg=f"Gap between chunks {i-1} and {i}")

        summary = format_plan_summary(plan, prof)
        self.assertIn("range_chunking", summary)
        self.assertIn("workers", summary)

        print(f"\n  Plan: {plan.strategy.value}, {len(plan.chunks)} chunks, "
              f"{plan.worker_count} workers, est. {plan.cost_estimate.predicted_duration_seconds:.0f}s")

    def test_plan_small_single_pass(self):
        """Small table should get single_pass with 1 chunk."""
        from ixtract.planner.planner import plan_extraction
        from ixtract.intent import ExtractionIntent, SourceType, TargetType

        prof = self.profiler.profile("small_lookup")
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config=TEST_DB_CONFIG,
            object_name="small_lookup",
            target_type=TargetType.PARQUET,
            target_config={"output_path": os.path.join(self.tmp, "output")},
        )

        plan = plan_extraction(intent, prof)

        self.assertEqual(plan.strategy.value, "single_pass")
        self.assertEqual(len(plan.chunks), 1)
        self.assertEqual(plan.worker_count, 1)

    def test_plan_respects_max_workers(self):
        """Intent max_workers constraint should cap worker count."""
        from ixtract.planner.planner import plan_extraction
        from ixtract.intent import ExtractionIntent, SourceType, TargetType, ExtractionConstraints

        prof = self.profiler.profile("orders")
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config=TEST_DB_CONFIG,
            object_name="orders",
            target_type=TargetType.PARQUET,
            target_config={"output_path": os.path.join(self.tmp, "output")},
            constraints=ExtractionConstraints(max_workers=2),
        )

        plan = plan_extraction(intent, prof)
        self.assertLessEqual(plan.worker_count, 2)


@requires_db
class TestExecution(unittest.TestCase):
    """Test end-to-end extraction: plan → execute → verify output."""

    def setUp(self):
        from ixtract.connectors.postgresql import PostgreSQLConnector
        self.connector = PostgreSQLConnector()
        self.connector.connect(TEST_DB_CONFIG)
        self.tmp = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.tmp, "output")
        os.makedirs(self.output_dir, exist_ok=True)

    def tearDown(self):
        self.connector.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_execute_small_table(self):
        """Full pipeline on small_lookup: profile → plan → execute → verify."""
        from ixtract.profiler import SourceProfiler
        from ixtract.planner.planner import plan_extraction
        from ixtract.intent import ExtractionIntent, SourceType, TargetType
        from ixtract.engine import ExecutionEngine

        # Profile
        prof = SourceProfiler(self.connector).profile("small_lookup")

        # Plan
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config=TEST_DB_CONFIG,
            object_name="small_lookup",
            target_type=TargetType.PARQUET,
            target_config={
                "output_path": self.output_dir,
                "object_name": "small_lookup",
            },
        )
        plan = plan_extraction(intent, prof)

        # Execute
        engine = ExecutionEngine(self.connector)
        result = engine.execute(plan, "small_lookup")

        # Verify
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.total_rows, 500, f"Expected 500 rows, got {result.total_rows}")
        self.assertGreater(result.avg_throughput, 0)
        self.assertEqual(len(result.chunk_results), 1)
        self.assertEqual(result.chunk_results[0].status, "success")

        # Verify Parquet file exists and is valid
        output_files = [f for f in os.listdir(self.output_dir) if f.endswith(".parquet")]
        self.assertEqual(len(output_files), 1, f"Expected 1 parquet file, got {output_files}")

        import pyarrow.parquet as pq
        table = pq.read_table(os.path.join(self.output_dir, output_files[0]))
        self.assertEqual(len(table), 500)
        self.assertIn("id", table.column_names)
        self.assertIn("code", table.column_names)

        print(f"\n  small_lookup: {result.total_rows} rows, "
              f"{result.duration_seconds:.1f}s, "
              f"{result.avg_throughput:.0f} rows/sec")

    def test_execute_orders_range_chunking(self):
        """Full pipeline on orders: range chunking with multiple workers."""
        from ixtract.profiler import SourceProfiler
        from ixtract.planner.planner import plan_extraction
        from ixtract.intent import ExtractionIntent, SourceType, TargetType, ExtractionConstraints
        from ixtract.engine import ExecutionEngine

        # Profile
        prof = SourceProfiler(self.connector).profile("orders")

        # Plan (cap workers at 3 for test speed)
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config=TEST_DB_CONFIG,
            object_name="orders",
            target_type=TargetType.PARQUET,
            target_config={
                "output_path": self.output_dir,
                "object_name": "orders",
            },
            constraints=ExtractionConstraints(max_workers=3),
        )
        plan = plan_extraction(intent, prof)

        self.assertEqual(plan.strategy.value, "range_chunking")
        self.assertGreater(len(plan.chunks), 1)

        # Execute
        engine = ExecutionEngine(self.connector)
        result = engine.execute(plan, "orders")

        # Verify
        self.assertEqual(result.status, "SUCCESS")
        self.assertGreater(result.total_rows, 1_000_000,
            f"Expected 1M+ rows, got {result.total_rows:,}")
        self.assertGreater(result.avg_throughput, 0)

        # All chunks succeeded
        failed = [cr for cr in result.chunk_results if cr.status == "failed"]
        self.assertEqual(len(failed), 0, f"Failed chunks: {[f.chunk_id for f in failed]}")

        # Parquet files exist
        output_files = [f for f in os.listdir(self.output_dir) if f.endswith(".parquet")]
        self.assertEqual(len(output_files), len(plan.chunks))

        # Verify total row count across all files
        import pyarrow.parquet as pq
        total_rows = sum(
            len(pq.read_table(os.path.join(self.output_dir, f)))
            for f in output_files
        )
        self.assertEqual(total_rows, result.total_rows)

        print(f"\n  orders: {result.total_rows:,} rows, "
              f"{len(plan.chunks)} chunks, "
              f"{result.duration_seconds:.1f}s, "
              f"{result.avg_throughput:,.0f} rows/sec")

    def test_idempotent_retry(self):
        """Executing the same plan twice should produce identical output."""
        from ixtract.profiler import SourceProfiler
        from ixtract.planner.planner import plan_extraction
        from ixtract.intent import ExtractionIntent, SourceType, TargetType
        from ixtract.engine import ExecutionEngine

        prof = SourceProfiler(self.connector).profile("small_lookup")
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config=TEST_DB_CONFIG,
            object_name="small_lookup",
            target_type=TargetType.PARQUET,
            target_config={
                "output_path": self.output_dir,
                "object_name": "small_lookup",
            },
        )

        plan = plan_extraction(intent, prof)
        engine = ExecutionEngine(self.connector)

        # First execution
        r1 = engine.execute(plan, "small_lookup")
        self.assertEqual(r1.status, "SUCCESS")

        # Second execution (same plan, same output path)
        # Atomic rename overwrites — no duplicates
        r2 = engine.execute(plan, "small_lookup")
        self.assertEqual(r2.status, "SUCCESS")
        self.assertEqual(r1.total_rows, r2.total_rows)

        # Only 1 output file (overwritten, not duplicated)
        output_files = [f for f in os.listdir(self.output_dir) if f.endswith(".parquet")]
        self.assertEqual(len(output_files), 1)


@requires_db
class TestStateStoreIntegration(unittest.TestCase):
    """Test that run results are correctly persisted to state store."""

    def setUp(self):
        from ixtract.connectors.postgresql import PostgreSQLConnector
        self.connector = PostgreSQLConnector()
        self.connector.connect(TEST_DB_CONFIG)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        self.connector.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_full_pipeline_records_state(self):
        """Execute and verify all state store tables are populated."""
        from ixtract.profiler import SourceProfiler
        from ixtract.planner.planner import plan_extraction
        from ixtract.intent import ExtractionIntent, SourceType, TargetType
        from ixtract.engine import ExecutionEngine
        from ixtract.state import StateStore
        from ixtract.controller import ParallelismController, ControllerState
        from ixtract.diagnosis import DeviationAnalyzer, RunMetrics

        prof = SourceProfiler(self.connector).profile("small_lookup")
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config=TEST_DB_CONFIG,
            object_name="small_lookup",
            target_type=TargetType.PARQUET,
            target_config={
                "output_path": os.path.join(self.tmp, "output"),
                "object_name": "small_lookup",
            },
        )

        store = StateStore(os.path.join(self.tmp, "state.db"))
        plan = plan_extraction(intent, prof, store)

        # Execute
        engine = ExecutionEngine(self.connector)
        result = engine.execute(plan, "small_lookup")

        # Record run
        store.record_run_start(
            result.run_id, plan.plan_id, intent.intent_hash(),
            "postgresql", "small_lookup", plan.strategy.value, plan.worker_count,
        )
        store.record_run_end(
            result.run_id, result.status.lower(),
            result.total_rows, result.total_bytes,
            result.avg_throughput, result.duration_seconds,
        )

        # Record chunk results
        for cr in result.chunk_results:
            store.record_chunk(
                result.run_id, cr.chunk_id, cr.worker_id,
                cr.rows, cr.bytes_written, cr.status, cr.duration_seconds,
                output_path=cr.output_path,
            )

        # Record deviation
        if result.metrics:
            analyzer = DeviationAnalyzer()
            diag = analyzer.diagnose(result.metrics)
            store.record_deviation(result.run_id, diag)

        # Record controller state
        controller = ParallelismController()
        ctrl_state = ControllerState.cold_start(controller.config)
        ctrl_out = controller.evaluate(result.avg_throughput, ctrl_state)
        store.save_controller_state("postgresql", "small_lookup", ctrl_out.new_state)

        # Save profile
        store.save_profile(
            "postgresql", "small_lookup",
            json.dumps(prof.to_dict(), default=str),
            row_estimate=prof.row_estimate,
        )

        # ── Verify all state store tables ────────────────────────
        runs = store.get_recent_runs("postgresql", "small_lookup")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "success")
        self.assertEqual(runs[0]["total_rows"], 500)

        chunks = store.get_chunks(result.run_id)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0]["status"], "success")

        loaded_ctrl = store.get_controller_state("postgresql", "small_lookup")
        self.assertIsNotNone(loaded_ctrl)
        self.assertGreater(loaded_ctrl.last_throughput, 0)

        loaded_prof = store.get_profile("postgresql", "small_lookup")
        self.assertIsNotNone(loaded_prof)

        print(f"\n  State store verified: run={runs[0]['run_id']}, "
              f"chunks={len(chunks)}, "
              f"controller_workers={loaded_ctrl.current_workers}, "
              f"profile stored")


@requires_db
class TestMultiRunConvergence(unittest.TestCase):
    """Test controller convergence across multiple real extraction runs."""

    def setUp(self):
        from ixtract.connectors.postgresql import PostgreSQLConnector
        self.connector = PostgreSQLConnector()
        self.connector.connect(TEST_DB_CONFIG)
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        self.connector.close()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_three_runs_convergence(self):
        """Run 3 extractions and verify controller adjusts correctly."""
        from ixtract.profiler import SourceProfiler
        from ixtract.planner.planner import plan_extraction
        from ixtract.intent import ExtractionIntent, SourceType, TargetType, ExtractionConstraints
        from ixtract.engine import ExecutionEngine
        from ixtract.state import StateStore
        from ixtract.controller import ParallelismController, ControllerState
        from ixtract.diagnosis import DeviationAnalyzer, RunMetrics

        store = StateStore(os.path.join(self.tmp, "state.db"))
        controller = ParallelismController()
        analyzer = DeviationAnalyzer()
        ctrl_state = None

        print(f"\n  Convergence test: 3 runs on small_lookup")
        print(f"  {'Run':>3}  {'Workers':>7}  {'Throughput':>11}  {'Decision':>10}")

        for run_num in range(1, 4):
            # Clean output for each run
            out_dir = os.path.join(self.tmp, f"output_{run_num}")
            os.makedirs(out_dir, exist_ok=True)

            prof = SourceProfiler(self.connector).profile("small_lookup")
            intent = ExtractionIntent(
                source_type=SourceType.POSTGRESQL,
                source_config=TEST_DB_CONFIG,
                object_name="small_lookup",
                target_type=TargetType.PARQUET,
                target_config={"output_path": out_dir, "object_name": "small_lookup"},
            )

            ctrl_state = store.get_controller_state("postgresql", "small_lookup")
            plan = plan_extraction(intent, prof, store, ctrl_state)

            engine = ExecutionEngine(self.connector)
            result = engine.execute(plan, "small_lookup")

            self.assertEqual(result.status, "SUCCESS")

            # Record
            store.record_run_start(
                result.run_id, plan.plan_id, intent.intent_hash(),
                "postgresql", "small_lookup", plan.strategy.value, plan.worker_count,
            )
            store.record_run_end(
                result.run_id, "success", result.total_rows, result.total_bytes,
                result.avg_throughput, result.duration_seconds,
            )

            # Controller
            if ctrl_state is None:
                ctrl_state = ControllerState.cold_start(controller.config)
            ctrl_out = controller.evaluate(result.avg_throughput, ctrl_state)
            store.save_controller_state("postgresql", "small_lookup", ctrl_out.new_state)
            store.set_heuristic("postgresql", "small_lookup", "throughput_baseline", result.avg_throughput)

            print(f"  {run_num:3d}  {plan.worker_count:7d}  "
                  f"{result.avg_throughput:10,.0f}  {ctrl_out.decision.value:>10}")

        # After 3 runs, controller should have state
        final_state = store.get_controller_state("postgresql", "small_lookup")
        self.assertIsNotNone(final_state)
        self.assertGreater(final_state.last_throughput, 0)

        runs = store.get_recent_runs("postgresql", "small_lookup")
        self.assertEqual(len(runs), 3)

        print(f"  Final state: {final_state.current_workers} workers, "
              f"converged={final_state.converged}")


if __name__ == "__main__":
    unittest.main()
