"""Benchmarker Simulation Tests.

Validates BenchmarkResult computation, confidence model, curve classification,
probe range selection, and Benchmarker.run() logic using a mock connector.
No database required.

Run: python -m unittest tests.simulation.test_benchmarker -v
"""
from __future__ import annotations

import os
import sys
import time
import unittest
from typing import Any, Iterator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.benchmarker import (
    Benchmarker, BenchmarkerConfig, BenchmarkResult, WorkerProbeResult,
    ProbeRange, compute_probe_rows, effective_worker_grid, classify_curve_shape,
    compute_confidence, select_optimal_workers, select_probe_ranges,
    conservative_worker_count, planner_workers_from_benchmark,
    CONFIDENCE_THRESHOLD, SIGNAL_STRENGTH_CONSERVATIVE_THRESHOLD,
    FLAT_CURVE_SIGNAL_FLOOR, PROBE_MIN_ROWS, PROBE_MAX_ROWS,
    BENCHMARK_MIN_TABLE_DURATION_SECONDS,
)
from ixtract.profiler import SourceProfile


# ── Mock infrastructure ───────────────────────────────────────────────

def _make_profile(
    row_estimate: int = 1_200_000,
    size_bytes: int = 200 * 1024 * 1024,  # 200MB — above 12s threshold
    pk_min: int = 1,
    pk_max: int = 1_200_000,
    has_pk: bool = True,
) -> SourceProfile:
    return SourceProfile(
        object_name="orders",
        row_estimate=row_estimate,
        size_estimate_bytes=size_bytes,
        column_count=10, avg_row_bytes=175,
        primary_key="id" if has_pk else None,
        primary_key_type="integer" if has_pk else None,
        pk_min=pk_min if has_pk else None,
        pk_max=pk_max if has_pk else None,
        pk_distribution_cv=0.0, has_usable_pk=has_pk,
        latency_p50_ms=0.5, latency_p95_ms=2.0, connection_ms=1.0,
        max_connections=100, active_connections=2, available_connections_safe=49,
        recommended_start_workers=2, recommended_strategy="range_chunking",
        recommended_scheduling="round_robin",
    )


class MockConnector:
    """Simulates a connector that returns rows with configurable throughput.

    throughput_by_workers: maps worker_count → rows/sec (used to simulate
    realistic parallelism curves). The connector doesn't actually know about
    workers — the Benchmarker controls the single-connection probe calls,
    so we model the curve via total rows returned per call and timing.

    For simulation purposes the connector always returns `batch_rows` rows
    per probe and records the call count, which the test can use to verify
    probe behaviour.
    """

    def __init__(self, batch_rows: int = 10_000, calls: list | None = None) -> None:
        self.batch_rows = batch_rows
        self.calls: list[dict] = calls if calls is not None else []

    def extract_chunk(
        self, object_name: str, query: str, params: dict | None = None
    ) -> Iterator[list[dict[str, Any]]]:
        self.calls.append({"object": object_name, "params": params})
        yield [{"id": i} for i in range(self.batch_rows)]


# ── Unit tests: pure functions ────────────────────────────────────────

class TestComputeProbeRows(unittest.TestCase):

    def test_floor_applied_for_small_table(self):
        self.assertEqual(compute_probe_rows(1_000), PROBE_MIN_ROWS)

    def test_fraction_used_for_medium_table(self):
        # 2% of 3M = 60K, within bounds
        self.assertEqual(compute_probe_rows(3_000_000), 60_000)

    def test_cap_applied_for_large_table(self):
        self.assertEqual(compute_probe_rows(100_000_000), PROBE_MAX_ROWS)

    def test_boundary_at_2_5m_rows(self):
        # 2% of 2.5M = 50K — exactly the floor
        self.assertEqual(compute_probe_rows(2_500_000), PROBE_MIN_ROWS)


class TestEffectiveWorkerGrid(unittest.TestCase):

    def test_full_grid_within_bounds(self):
        self.assertEqual(effective_worker_grid(max_workers=16), (1, 2, 4, 8))

    def test_bounded_by_max_workers(self):
        self.assertEqual(effective_worker_grid(max_workers=4), (1, 2, 4))

    def test_bounded_by_max_workers_2(self):
        self.assertEqual(effective_worker_grid(max_workers=2), (1, 2))

    def test_single_point_grid(self):
        # max_workers=1 → grid=(1,) → benchmarker should skip
        self.assertEqual(effective_worker_grid(max_workers=1), (1,))

    def test_min_workers_filter(self):
        self.assertEqual(effective_worker_grid(max_workers=8, min_workers=2), (2, 4, 8))


class TestClassifyCurveShape(unittest.TestCase):

    def test_plateau(self):
        # Peaks at 4 workers then slightly dips
        self.assertEqual(classify_curve_shape([100, 200, 230, 220]), "plateau")

    def test_increasing(self):
        # Still rising at the last grid point
        self.assertEqual(classify_curve_shape([100, 180, 250, 310]), "increasing")

    def test_non_monotonic(self):
        # Peaks at 2 workers then falls significantly
        self.assertEqual(classify_curve_shape([100, 300, 180, 140]), "non_monotonic")

    def test_single_point_defaults_to_plateau(self):
        self.assertEqual(classify_curve_shape([200_000]), "plateau")

    def test_two_point_increasing(self):
        self.assertEqual(classify_curve_shape([100, 200]), "increasing")

    def test_two_point_plateau(self):
        # Second point lower — peak at index 0, which is last only if len==1
        # With two points: peak at idx 0, last at idx 1 → peak before end → plateau/non_mono
        result = classify_curve_shape([200, 100])
        self.assertIn(result, ("plateau", "non_monotonic"))


class TestComputeConfidence(unittest.TestCase):

    def _probes(self, avgs, cv=0.03):
        return [
            WorkerProbeResult(w, w, (tp * 0.97, tp, tp * 1.03), tp, cv)
            for w, tp in zip([1, 2, 4, 8], avgs)
        ]

    def test_high_confidence_stable_plateau(self):
        probes = self._probes([100_000, 220_000, 230_000, 215_000], cv=0.02)
        conf, sig = compute_confidence(probes, "plateau")
        self.assertGreater(conf, 0.90)
        self.assertGreater(sig, 0.30)

    def test_low_confidence_noisy_probes(self):
        probes = self._probes([100_000, 105_000, 107_000, 106_000], cv=1.5)
        conf, sig = compute_confidence(probes, "plateau")
        self.assertLess(conf, CONFIDENCE_THRESHOLD)

    def test_increasing_curve_reduces_confidence(self):
        probes_p = self._probes([100_000, 200_000, 230_000, 220_000])
        probes_i = self._probes([100_000, 200_000, 230_000, 280_000])
        conf_p, _ = compute_confidence(probes_p, "plateau")
        conf_i, _ = compute_confidence(probes_i, "increasing")
        self.assertGreater(conf_p, conf_i)

    def test_flat_curve_signal_floored(self):
        # All worker counts perform similarly → raw signal near 0 → floored
        probes = self._probes([200_000, 202_000, 201_000, 199_000])
        _, sig = compute_confidence(probes, "plateau")
        self.assertGreaterEqual(sig, FLAT_CURVE_SIGNAL_FLOOR)

    def test_confidence_bounded_0_to_1(self):
        for shape in ("plateau", "increasing", "non_monotonic"):
            probes = self._probes([100_000, 200_000, 220_000, 210_000])
            conf, sig = compute_confidence(probes, shape)
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)
            self.assertGreaterEqual(sig, FLAT_CURVE_SIGNAL_FLOOR)


class TestSelectOptimalWorkers(unittest.TestCase):

    def test_argmax_simple(self):
        self.assertEqual(select_optimal_workers({1: 100, 2: 200, 4: 180}), 2)

    def test_tiebreak_prefers_lower_workers(self):
        # 2 and 4 both within 5% of peak (230K)
        tp = {1: 100_000, 2: 224_000, 4: 230_000, 8: 215_000}
        result = select_optimal_workers(tp)
        self.assertEqual(result, 2)

    def test_clear_winner_no_tiebreak(self):
        # 4 workers clearly better (>5% above 2 workers)
        tp = {1: 100_000, 2: 180_000, 4: 230_000, 8: 220_000}
        result = select_optimal_workers(tp)
        self.assertEqual(result, 4)

    def test_single_entry(self):
        self.assertEqual(select_optimal_workers({2: 200_000}), 2)

    def test_empty_returns_1(self):
        self.assertEqual(select_optimal_workers({}), 1)


class TestSelectProbeRanges(unittest.TestCase):

    def test_three_ranges_for_large_pk_space(self):
        ranges, n = select_probe_ranges(1, 1_200_000, 50_000)
        self.assertEqual(n, 3)
        self.assertEqual(len(ranges), 3)
        labels = [r.label for r in ranges]
        self.assertEqual(labels, ["low", "mid", "high"])

    def test_ranges_are_non_overlapping(self):
        ranges, _ = select_probe_ranges(1, 1_200_000, 50_000)
        for i in range(len(ranges) - 1):
            self.assertLess(ranges[i].pk_end, ranges[i + 1].pk_start)

    def test_fallback_to_two_ranges_for_small_pk_space(self):
        # PK range < 3 * probe_rows → 2 ranges
        _, n = select_probe_ranges(1, 100_000, 50_000)
        self.assertEqual(n, 2)

    def test_ranges_within_pk_bounds(self):
        pk_min, pk_max = 1000, 2_000_000
        ranges, _ = select_probe_ranges(pk_min, pk_max, 50_000)
        for r in ranges:
            self.assertGreaterEqual(r.pk_start, pk_min)
            self.assertLessEqual(r.pk_end, pk_max)

    def test_low_range_starts_at_pk_min(self):
        ranges, _ = select_probe_ranges(1, 1_200_000, 50_000)
        self.assertEqual(ranges[0].pk_start, 1)

    def test_high_range_ends_at_pk_max(self):
        ranges, _ = select_probe_ranges(1, 1_200_000, 50_000)
        self.assertEqual(ranges[-1].pk_end, 1_200_000)


class TestPlannerLogic(unittest.TestCase):

    def _make_result(self, conf, sig, optimal=4):
        from datetime import datetime, timezone
        probes = [WorkerProbeResult(2, 2, (200_000,), 200_000, 0.02)]
        return BenchmarkResult(
            source_type="postgresql", object_name="orders",
            probe_rows=50_000, ranges_used=3,
            worker_grid=(1, 2, 4), probe_results=tuple(probes),
            throughput_by_workers={1: 100_000, 2: 200_000, 4: 230_000},
            optimal_workers=optimal, confidence=conf, signal_strength=sig,
            curve_shape="plateau",
            benchmarked_at=datetime.now(timezone.utc),
            row_estimate_at_benchmark=1_200_000,
        )

    def test_low_confidence_raises(self):
        r = self._make_result(conf=0.3, sig=0.5)
        with self.assertRaises(ValueError):
            planner_workers_from_benchmark(r)

    def test_flat_curve_returns_conservative(self):
        r = self._make_result(conf=0.9, sig=0.15, optimal=4)
        self.assertEqual(planner_workers_from_benchmark(r), 2)

    def test_boundary_signal_at_threshold_is_conservative(self):
        # signal_strength == 0.30 → boundary is inclusive → conservative
        r = self._make_result(conf=0.9, sig=0.30, optimal=4)
        self.assertEqual(planner_workers_from_benchmark(r), 2)

    def test_above_threshold_returns_optimal(self):
        r = self._make_result(conf=0.9, sig=0.50, optimal=4)
        self.assertEqual(planner_workers_from_benchmark(r), 4)

    def test_conservative_worker_cap(self):
        self.assertEqual(conservative_worker_count(8), 2)
        self.assertEqual(conservative_worker_count(4), 2)
        self.assertEqual(conservative_worker_count(2), 2)
        self.assertEqual(conservative_worker_count(1), 1)


# ── Benchmarker.run() integration with mock connector ─────────────────

class TestBenchmarkerRun(unittest.TestCase):

    def _cfg(self, max_workers=4):
        return BenchmarkerConfig(max_workers=max_workers, min_table_duration_seconds=0.0)

    def test_returns_none_for_table_without_pk(self):
        prof = _make_profile(has_pk=False)
        b = Benchmarker(MockConnector(), self._cfg())
        result = b.run("orders", prof, force=True)
        self.assertIsNone(result)

    def test_returns_none_for_single_point_grid(self):
        prof = _make_profile()
        b = Benchmarker(MockConnector(), BenchmarkerConfig(max_workers=1, min_table_duration_seconds=0.0))
        result = b.run("orders", prof, force=True)
        self.assertIsNone(result)

    def test_skips_small_table_without_force(self):
        # Small table (< 12s estimated duration) → returns None without force
        prof = _make_profile(size_bytes=10 * 1024 * 1024)  # 10MB → ~0.4s
        b = Benchmarker(MockConnector(), BenchmarkerConfig(max_workers=4))
        result = b.run("orders", prof, force=False)
        self.assertIsNone(result)

    def test_force_bypasses_size_guard(self):
        prof = _make_profile(size_bytes=10 * 1024 * 1024)
        b = Benchmarker(MockConnector(), self._cfg())
        result = b.run("orders", prof, force=True)
        self.assertIsNotNone(result)

    def test_returns_benchmark_result(self):
        prof = _make_profile()
        b = Benchmarker(MockConnector(batch_rows=50_000), self._cfg())
        result = b.run("orders", prof, force=True)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, BenchmarkResult)

    def test_result_covers_full_grid(self):
        prof = _make_profile()
        b = Benchmarker(MockConnector(batch_rows=50_000), self._cfg(max_workers=4))
        result = b.run("orders", prof, force=True)
        self.assertIsNotNone(result)
        self.assertEqual(set(result.worker_grid), {1, 2, 4})
        self.assertEqual(set(result.throughput_by_workers.keys()), {1, 2, 4})

    def test_optimal_workers_within_grid(self):
        prof = _make_profile()
        b = Benchmarker(MockConnector(batch_rows=50_000), self._cfg())
        result = b.run("orders", prof, force=True)
        self.assertIsNotNone(result)
        self.assertIn(result.optimal_workers, result.worker_grid)

    def test_confidence_bounded(self):
        prof = _make_profile()
        b = Benchmarker(MockConnector(batch_rows=50_000), self._cfg())
        result = b.run("orders", prof, force=True)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.confidence, 0.0)
        self.assertLessEqual(result.confidence, 1.0)

    def test_signal_strength_at_least_floor(self):
        prof = _make_profile()
        b = Benchmarker(MockConnector(batch_rows=50_000), self._cfg())
        result = b.run("orders", prof, force=True)
        self.assertIsNotNone(result)
        self.assertGreaterEqual(result.signal_strength, FLAT_CURVE_SIGNAL_FLOOR)

    def test_probe_calls_include_cache_warm(self):
        # Each range gets a warm pass + measured pass → 2 calls per range per worker count
        prof = _make_profile(pk_max=1_200_000)
        calls = []
        b = Benchmarker(MockConnector(batch_rows=50_000, calls=calls), self._cfg(max_workers=2))
        result = b.run("orders", prof, force=True)
        self.assertIsNotNone(result)
        # grid=[1,2], ranges=3 → 2 workers × 3 ranges × 2 passes = 12 calls
        self.assertEqual(len(calls), 12)

    def test_result_serialization_roundtrip(self):
        prof = _make_profile()
        b = Benchmarker(MockConnector(batch_rows=50_000), self._cfg())
        result = b.run("orders", prof, force=True)
        self.assertIsNotNone(result)
        d = result.to_dict()
        restored = BenchmarkResult.from_dict(d)
        self.assertEqual(restored.optimal_workers, result.optimal_workers)
        self.assertEqual(restored.confidence, result.confidence)
        self.assertEqual(restored.curve_shape, result.curve_shape)
        self.assertEqual(restored.throughput_by_workers, result.throughput_by_workers)

    def test_staleness_by_age(self):
        from datetime import datetime, timezone, timedelta
        prof = _make_profile()
        b = Benchmarker(MockConnector(batch_rows=50_000), self._cfg())
        result = b.run("orders", prof, force=True)
        self.assertIsNotNone(result)
        # Fresh result is not stale
        self.assertFalse(result.is_stale(prof.row_estimate))

    def test_staleness_by_row_growth(self):
        from datetime import datetime, timezone
        prof = _make_profile(row_estimate=1_000_000)
        b = Benchmarker(MockConnector(batch_rows=50_000), self._cfg())
        result = b.run("orders", prof, force=True)
        self.assertIsNotNone(result)
        # 30% growth exceeds 20% threshold → stale
        self.assertTrue(result.is_stale(1_300_000))
        # 10% growth is fine
        self.assertFalse(result.is_stale(1_100_000))


if __name__ == "__main__":
    unittest.main(verbosity=2)


# ── Planner integration tests (no DB needed) ──────────────────────────

class TestPlannerBenchmarkIntegration(unittest.TestCase):
    """Validates all three worker resolution paths in the planner."""

    def _make_intent(self, tmp_dir):
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
        from ixtract.intent import ExtractionIntent, SourceType, TargetType
        return ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config={"host": "localhost", "database": "test",
                           "user": "test", "password": "test"},
            object_name="orders",
            target_type=TargetType.PARQUET,
            target_config={"output_path": tmp_dir, "object_name": "orders"},
        )

    def _make_benchmark(self, optimal=4, confidence=0.85, signal=0.55):
        from datetime import datetime, timezone
        from ixtract.benchmarker import BenchmarkResult, WorkerProbeResult
        probes = [WorkerProbeResult(w, w, (200_000,), 200_000, 0.02)
                  for w in [1, 2, 4]]
        return BenchmarkResult(
            source_type="postgresql", object_name="orders",
            probe_rows=50_000, ranges_used=3,
            worker_grid=(1, 2, 4), probe_results=tuple(probes),
            throughput_by_workers={1: 110_000, 2: 200_000, 4: 230_000},
            optimal_workers=optimal, confidence=confidence,
            signal_strength=signal, curve_shape="plateau",
            benchmarked_at=datetime.now(timezone.utc),
            row_estimate_at_benchmark=1_200_000,
        )

    def test_benchmark_path_uses_optimal_workers(self):
        """When a valid benchmark exists with high signal, planner uses benchmark path."""
        import tempfile
        from ixtract.planner.planner import plan_extraction, format_plan_summary
        from ixtract.state import StateStore

        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(os.path.join(tmp, "state.db"))
            # Use optimal=2 — guaranteed to survive any system cpu_count cap
            benchmark = self._make_benchmark(optimal=2, confidence=0.85, signal=0.55)
            store.save_benchmark("postgresql", "orders", benchmark)

            prof = _make_profile(size_bytes=300 * 1024 * 1024)
            intent = self._make_intent(tmp)
            plan, source = plan_extraction(intent, prof, store=store)

            # Label proves the benchmark path was taken
            self.assertEqual(source, "benchmarked")
            # Worker count is benchmark optimal, capped by system resources
            self.assertGreaterEqual(plan.worker_count, 1)
            self.assertLessEqual(plan.worker_count, 2)

    def test_benchmark_flat_curve_uses_conservative_workers(self):
        """Flat curve (signal_strength <= 0.30) → conservative worker count."""
        import tempfile
        from ixtract.planner.planner import plan_extraction

        with tempfile.TemporaryDirectory() as tmp:
            from ixtract.state import StateStore
            store = StateStore(os.path.join(tmp, "state.db"))
            benchmark = self._make_benchmark(optimal=4, confidence=0.85, signal=0.15)
            store.save_benchmark("postgresql", "orders", benchmark)

            prof = _make_profile(size_bytes=300 * 1024 * 1024)
            intent = self._make_intent(tmp)
            plan, source = plan_extraction(intent, prof, store=store)

            self.assertEqual(source, "benchmarked (conservative)")
            self.assertLessEqual(plan.worker_count, 2)

    def test_low_confidence_benchmark_falls_back_to_profiler(self):
        """Low confidence benchmark → planner ignores it, uses profiler."""
        import tempfile
        from ixtract.planner.planner import plan_extraction

        with tempfile.TemporaryDirectory() as tmp:
            from ixtract.state import StateStore
            store = StateStore(os.path.join(tmp, "state.db"))
            # Below CONFIDENCE_THRESHOLD (0.5)
            benchmark = self._make_benchmark(optimal=8, confidence=0.3, signal=0.5)
            store.save_benchmark("postgresql", "orders", benchmark)

            prof = _make_profile(size_bytes=300 * 1024 * 1024)
            intent = self._make_intent(tmp)
            plan, source = plan_extraction(intent, prof, store=store)

            self.assertEqual(source, "profiler recommended")
            self.assertNotEqual(plan.worker_count, 8)

    def test_no_benchmark_uses_profiler(self):
        """No benchmark in store → profiler path."""
        import tempfile
        from ixtract.planner.planner import plan_extraction

        with tempfile.TemporaryDirectory() as tmp:
            from ixtract.state import StateStore
            store = StateStore(os.path.join(tmp, "state.db"))

            prof = _make_profile()
            intent = self._make_intent(tmp)
            plan, source = plan_extraction(intent, prof, store=store)

            self.assertEqual(source, "profiler recommended")

    def test_stale_benchmark_falls_back_to_profiler(self):
        """Stale benchmark (row growth > 20%) → planner ignores it."""
        import tempfile
        from datetime import datetime, timezone
        from ixtract.benchmarker import BenchmarkResult, WorkerProbeResult
        from ixtract.planner.planner import plan_extraction
        from ixtract.state import StateStore

        with tempfile.TemporaryDirectory() as tmp:
            store = StateStore(os.path.join(tmp, "state.db"))

            # Benchmark recorded when table had 800K rows
            probes = [WorkerProbeResult(w, w, (200_000,), 200_000, 0.02)
                      for w in [1, 2, 4]]
            stale_benchmark = BenchmarkResult(
                source_type="postgresql", object_name="orders",
                probe_rows=50_000, ranges_used=3,
                worker_grid=(1, 2, 4), probe_results=tuple(probes),
                throughput_by_workers={1: 110_000, 2: 200_000, 4: 230_000},
                optimal_workers=4, confidence=0.85, signal_strength=0.55,
                curve_shape="plateau",
                benchmarked_at=datetime.now(timezone.utc),
                row_estimate_at_benchmark=800_000,   # ← was 800K rows
            )
            store.save_benchmark("postgresql", "orders", stale_benchmark)

            # Profile shows 1.5M rows (87% growth from 800K → stale)
            prof = _make_profile(row_estimate=1_500_000, size_bytes=300 * 1024 * 1024)
            intent = self._make_intent(tmp)
            plan, source = plan_extraction(intent, prof, store=store)

            self.assertEqual(source, "profiler recommended")

    def test_format_plan_summary_shows_benchmark_label(self):
        """format_plan_summary renders the benchmark source label correctly."""
        import tempfile
        from ixtract.planner.planner import plan_extraction, format_plan_summary

        with tempfile.TemporaryDirectory() as tmp:
            from ixtract.state import StateStore
            store = StateStore(os.path.join(tmp, "state.db"))
            benchmark = self._make_benchmark(optimal=4, confidence=0.85, signal=0.55)
            store.save_benchmark("postgresql", "orders", benchmark)

            prof = _make_profile(size_bytes=300 * 1024 * 1024)
            intent = self._make_intent(tmp)
            plan, source = plan_extraction(intent, prof, store=store)

            summary = format_plan_summary(plan, prof, worker_source=source)
            self.assertIn("benchmarked", summary)
