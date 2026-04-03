"""Phase 0 Simulation Tests — Controller Convergence Validation.

Validates the control model against all 6 architecture scenarios BEFORE
any real connector is built. Uses stdlib unittest — zero dependencies.

Run: python -m pytest tests/ -v  (if pytest available)
  or: python -m unittest tests.simulation.test_phase0 -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

# Ensure src/ is on the path
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.controller import (
    ControllerConfig, ControllerDecision, ControllerState, ParallelismController,
)
from ixtract.diagnosis import DeviationAnalyzer, DeviationCategory, RunMetrics
from ixtract.simulation import SimulatedSource, SimulationConfig
from ixtract.intent import ExtractionIntent, SourceType, ExtractionMode, TargetType
from ixtract.planner import (
    ExecutionPlan, Strategy, ChunkDefinition, ChunkType,
    CostEstimate, MetadataSnapshot, SchedulingStrategy,
)
from ixtract.state import StateStore


# ── Simulation Runner ─────────────────────────────────────────────────

def _simulate(
    config: SimulationConfig,
    ctrl_cfg: ControllerConfig | None = None,
    num_runs: int = 15,
    start_state: ControllerState | None = None,
) -> list[dict]:
    """Run N extraction cycles and return history."""
    source = SimulatedSource(config)
    ctrl = ParallelismController(ctrl_cfg or ControllerConfig(
        max_workers=config.optimal_workers * 2 + 4,
    ))
    analyzer = DeviationAnalyzer()
    state = start_state or ControllerState.cold_start(ctrl.config)
    history: list[dict] = []

    for i in range(num_runs):
        metrics = source.run(
            worker_count=state.current_workers,
            chunk_count=max(10, state.current_workers * 2),
            previous_throughput=state.last_throughput,
            previous_workers=state.last_worker_count,
        )
        diag = analyzer.diagnose(metrics)
        out = ctrl.evaluate(metrics.avg_throughput_rows_sec, state)

        history.append({
            "run": i + 1,
            "workers": state.current_workers,
            "throughput": metrics.avg_throughput_rows_sec,
            "decision": out.decision.value,
            "recommended": out.recommended_workers,
            "diagnosis": diag.category.value,
            "converged": out.new_state.converged,
            "reasoning": out.reasoning,
        })
        state = out.new_state

    return history


# ── Scenario Tests ────────────────────────────────────────────────────

class TestScenario1_HappyPathConvergence(unittest.TestCase):
    """Logarithmic throughput curve, no skew. Should converge within ~10 runs."""

    def setUp(self):
        self.config = SimulationConfig(
            optimal_workers=6, concurrency_curve="logarithmic",
            latency_jitter_pct=0.02, seed=42,
        )

    def test_converges(self):
        history = _simulate(self.config, num_runs=12)
        converged = [h for h in history if h["converged"]]
        self.assertTrue(len(converged) > 0, "Controller never converged")

    def test_final_workers_near_optimal(self):
        history = _simulate(self.config, num_runs=12)
        final = history[-1]["recommended"]
        self.assertLessEqual(abs(final - 6), 1,
            f"Final workers {final} not within ±1 of optimal 6")


class TestScenario2_OverParallelRecovery(unittest.TestCase):
    """Start at max workers on a source that peaks at 4. Should step down.

    The controller needs a prior throughput observation to compare against.
    We seed the state as if the previous run used 11 workers and achieved
    higher throughput (since optimal=4, fewer workers means more throughput
    once you're past the peak). The controller then sees the drop at 12
    and begins stepping down.
    """

    def test_steps_down(self):
        config = SimulationConfig(
            optimal_workers=4, concurrency_curve="logarithmic",
            latency_jitter_pct=0.02, seed=42,
        )
        ctrl_cfg = ControllerConfig(max_workers=12, min_workers=1)

        # Pre-seed: simulate one run at 11 workers to get a baseline throughput.
        # At optimal=4, logarithmic, 11 workers → throughput ≈ peak * 0.3
        # At 12 workers → throughput ≈ peak * 0.2. Controller will see the drop.
        preseed_source = SimulatedSource(SimulationConfig(
            optimal_workers=4, concurrency_curve="logarithmic",
            latency_jitter_pct=0.02, seed=42,
        ))
        preseed_metrics = preseed_source.run(worker_count=11)

        start = ControllerState(
            current_workers=12,
            last_throughput=preseed_metrics.avg_throughput_rows_sec,
            last_worker_count=11,
            direction=ControllerDecision.INCREASE,
            consecutive_holds=0,
            converged=False,
        )
        history = _simulate(config, ctrl_cfg, num_runs=20, start_state=start)
        final = history[-1]["recommended"]
        self.assertLessEqual(final, 7,
            f"Workers {final} still too high after 20 runs (optimal=4)")


class TestScenario3_LatencySpike(unittest.TestCase):
    """Inject 50% latency spike on run 6. Controller should NOT add workers."""

    def test_no_increase_on_spike(self):
        config = SimulationConfig(
            optimal_workers=6, concurrency_curve="logarithmic",
            latency_jitter_pct=0.02,
            latency_spike_on_run=6, latency_spike_multiplier=1.5,
            seed=42,
        )
        history = _simulate(config, num_runs=10)
        # The spike is run 6 (1-indexed). Check run 7's recommendation.
        if len(history) >= 7:
            pre_spike = history[4]["workers"]   # run 5
            post_spike = history[6]["recommended"]  # run 7
            self.assertLessEqual(post_spike, pre_spike + 1,
                f"Workers jumped from {pre_spike} to {post_spike} after latency spike")


class TestScenario4_DataSkew(unittest.TestCase):
    """Power-law chunk variance. Skew should be detectable in metrics."""

    def test_skew_detectable(self):
        config = SimulationConfig(
            optimal_workers=6, skew_distribution="power_law",
            skew_intensity=0.8, latency_jitter_pct=0.02, seed=42,
        )
        source = SimulatedSource(config)
        metrics = source.run(worker_count=6, chunk_count=20)

        cv = DeviationAnalyzer._chunk_cv(metrics.chunk_durations)
        self.assertGreater(cv, 0.3, f"Chunk CV {cv:.2f} too low for skewed data")


class TestScenario5_GrowthDetection(unittest.TestCase):
    """10% row growth per run. Should be visible in metrics."""

    def test_rows_grow(self):
        config = SimulationConfig(
            total_rows=1_000_000, growth_rate_per_run=0.10,
            optimal_workers=6, seed=42,
        )
        source = SimulatedSource(config)
        counts = []
        for _ in range(5):
            m = source.run(worker_count=6)
            counts.append(m.total_rows)

        for i in range(1, len(counts)):
            growth = (counts[i] - counts[i - 1]) / counts[i - 1]
            self.assertAlmostEqual(growth, 0.10, places=2,
                msg=f"Growth {growth:.2%} not ~10% between runs {i} and {i+1}")


class TestScenario6_OscillationResistance(unittest.TestCase):
    """Noisy throughput (±8%). Controller should not flip-flop."""

    def test_no_oscillation(self):
        config = SimulationConfig(
            optimal_workers=6, concurrency_curve="logarithmic",
            latency_jitter_pct=0.08, seed=42,
        )
        history = _simulate(config, num_runs=15)

        decisions = [h["decision"] for h in history if h["decision"] != "hold"]
        flips = sum(
            1 for i in range(1, len(decisions)) if decisions[i] != decisions[i - 1]
        )
        self.assertLessEqual(flips, 5,
            f"Too many direction changes ({flips}): {decisions}")

    def test_converges_under_noise(self):
        config = SimulationConfig(
            optimal_workers=6, concurrency_curve="logarithmic",
            latency_jitter_pct=0.08, seed=42,
        )
        history = _simulate(config, num_runs=20)
        final = history[-1]["recommended"]
        self.assertLessEqual(abs(final - 6), 2,
            f"Final workers {final} too far from optimal 6 under noise")


# ── Controller Invariant Tests ────────────────────────────────────────

class TestControllerInvariants(unittest.TestCase):

    def test_determinism(self):
        """Same inputs → same output."""
        cfg = SimulationConfig(seed=42, latency_jitter_pct=0.02)
        h1 = _simulate(cfg, num_runs=10)
        h2 = _simulate(cfg, num_runs=10)
        self.assertEqual(h1, h2, "Controller is not deterministic")

    def test_bounds_respected(self):
        """Worker count stays within [min, max]."""
        cfg = SimulationConfig(optimal_workers=6, latency_jitter_pct=0.02, seed=42)
        ctrl = ControllerConfig(min_workers=2, max_workers=10)
        history = _simulate(cfg, ctrl, num_runs=15)
        for h in history:
            self.assertTrue(2 <= h["recommended"] <= 10,
                f"Workers {h['recommended']} outside [2,10] on run {h['run']}")

    def test_step_size_bounded(self):
        """Worker changes are ≤2 (±1 normally, ≤2 for regression revert)."""
        cfg = SimulationConfig(seed=42, latency_jitter_pct=0.02)
        history = _simulate(cfg, num_runs=15)
        for i in range(1, len(history)):
            delta = abs(history[i]["recommended"] - history[i - 1]["recommended"])
            self.assertLessEqual(delta, 2,
                f"Worker delta {delta} on run {history[i]['run']}")


# ── Intent Model Tests ────────────────────────────────────────────────

class TestIntentModel(unittest.TestCase):

    def test_basic_creation(self):
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config={"host": "localhost", "database": "test"},
            object_name="orders",
        )
        self.assertEqual(intent.source_type, SourceType.POSTGRESQL)
        self.assertEqual(intent.object_name, "orders")
        self.assertEqual(intent.mode, ExtractionMode.FULL)

    def test_hash_deterministic(self):
        kwargs = dict(
            source_type=SourceType.POSTGRESQL,
            source_config={"host": "localhost"},
            object_name="orders",
        )
        self.assertEqual(
            ExtractionIntent(**kwargs).intent_hash(),
            ExtractionIntent(**kwargs).intent_hash(),
        )

    def test_incremental_requires_key(self):
        with self.assertRaises(ValueError):
            ExtractionIntent(
                source_type=SourceType.POSTGRESQL,
                source_config={},
                object_name="orders",
                mode=ExtractionMode.INCREMENTAL,
            )

    def test_incremental_with_key_valid(self):
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config={},
            object_name="orders",
            mode=ExtractionMode.INCREMENTAL,
            incremental_key="updated_at",
        )
        self.assertEqual(intent.incremental_key, "updated_at")

    def test_source_object_key(self):
        intent = ExtractionIntent(
            source_type=SourceType.MYSQL,
            source_config={},
            object_name="users",
        )
        self.assertEqual(intent.source_object_key(), "mysql::users")


# ── ExecutionPlan Tests ───────────────────────────────────────────────

class TestExecutionPlan(unittest.TestCase):

    def _make_plan(self, **overrides) -> ExecutionPlan:
        defaults = dict(
            intent_hash="abc123",
            strategy=Strategy.SINGLE_PASS,
            chunks=(ChunkDefinition(
                chunk_id="c1", chunk_type=ChunkType.FULL_TABLE,
                estimated_rows=50000, estimated_bytes=10_000_000,
            ),),
            worker_count=1,
            cost_estimate=CostEstimate(10.0, 5000.0, 50000, 10_000_000),
            metadata_snapshot=MetadataSnapshot(50000, 10_000_000, 8),
        )
        defaults.update(overrides)
        return ExecutionPlan(**defaults)

    def test_frozen(self):
        plan = self._make_plan()
        with self.assertRaises(AttributeError):
            plan.worker_count = 8  # type: ignore

    def test_serialization_roundtrip(self):
        plan = self._make_plan()
        data = plan.to_dict()
        json_str = json.dumps(data)
        restored = ExecutionPlan.from_dict(json.loads(json_str))
        self.assertEqual(restored.strategy, plan.strategy)
        self.assertEqual(len(restored.chunks), 1)
        self.assertEqual(restored.worker_count, plan.worker_count)

    def test_validation_min_chunks(self):
        with self.assertRaises(ValueError):
            ExecutionPlan(
                intent_hash="abc", strategy=Strategy.SINGLE_PASS,
                chunks=(), worker_count=1,
                cost_estimate=CostEstimate(1, 1, 1, 1),
                metadata_snapshot=MetadataSnapshot(1, 1, 1),
            )

    def test_validation_worker_bounds(self):
        with self.assertRaises(ValueError):
            self._make_plan(worker_bounds=(0, 5))


# ── State Store Tests ─────────────────────────────────────────────────

class TestStateStore(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store = StateStore(os.path.join(self.tmp, "test.db"))

    def test_controller_roundtrip(self):
        st = ControllerState(5, 12000.0, 4, ControllerDecision.INCREASE, 0, False)
        self.store.save_controller_state("pg", "orders", st)
        loaded = self.store.get_controller_state("pg", "orders")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.current_workers, 5)
        self.assertEqual(loaded.last_throughput, 12000.0)
        self.assertEqual(loaded.direction, ControllerDecision.INCREASE)

    def test_controller_upsert(self):
        st1 = ControllerState(5, 12000.0, 4, ControllerDecision.INCREASE, 0, False)
        self.store.save_controller_state("pg", "orders", st1)
        st2 = ControllerState(6, 14000.0, 5, ControllerDecision.HOLD, 1, False)
        self.store.save_controller_state("pg", "orders", st2)
        loaded = self.store.get_controller_state("pg", "orders")
        self.assertEqual(loaded.current_workers, 6)

    def test_heuristic_upsert(self):
        self.store.set_heuristic("pg", "orders", "tp_baseline", 10000.0)
        self.assertEqual(self.store.get_heuristic("pg", "orders", "tp_baseline"), 10000.0)
        self.store.set_heuristic("pg", "orders", "tp_baseline", 12000.0)
        self.assertEqual(self.store.get_heuristic("pg", "orders", "tp_baseline"), 12000.0)

    def test_heuristic_missing(self):
        self.assertIsNone(self.store.get_heuristic("pg", "orders", "nonexistent"))

    def test_run_recording(self):
        self.store.record_run_start("r1", "p1", "h1", "pg", "orders", "range_chunking", 4)
        self.store.record_run_end("r1", "success", 100000, 20_000_000, 15000.0, 6.7)
        runs = self.store.get_recent_runs("pg", "orders")
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "success")
        self.assertEqual(runs[0]["total_rows"], 100000)


# ── Deviation Analyzer Tests ──────────────────────────────────────────

class TestDeviationAnalyzer(unittest.TestCase):

    def test_under_parallel(self):
        analyzer = DeviationAnalyzer()
        m = RunMetrics(
            total_rows=100000, total_bytes=20_000_000, duration_seconds=10.0,
            worker_count=5, avg_throughput_rows_sec=10000.0,
            chunk_durations=(2.0, 2.0, 2.0, 2.0, 2.0),
            worker_idle_pcts=(0.05, 0.05, 0.05, 0.05, 0.05),
            previous_throughput_rows_sec=8000.0, previous_worker_count=4,
            worker_count_changed=True,
        )
        d = analyzer.diagnose(m)
        self.assertEqual(d.category, DeviationCategory.UNDER_PARALLEL)

    def test_over_parallel(self):
        analyzer = DeviationAnalyzer()
        m = RunMetrics(
            total_rows=100000, total_bytes=20_000_000, duration_seconds=15.0,
            worker_count=10, avg_throughput_rows_sec=6667.0,
            chunk_durations=(1.5,) * 10,
            worker_idle_pcts=(0.1,) * 10,
            previous_throughput_rows_sec=10000.0, previous_worker_count=6,
            worker_count_changed=True,
        )
        d = analyzer.diagnose(m)
        self.assertEqual(d.category, DeviationCategory.OVER_PARALLEL)

    def test_stable(self):
        analyzer = DeviationAnalyzer()
        m = RunMetrics(
            total_rows=100000, total_bytes=20_000_000, duration_seconds=10.0,
            worker_count=6, avg_throughput_rows_sec=10000.0,
            chunk_durations=(1.7,) * 6,
            worker_idle_pcts=(0.05,) * 6,
            previous_throughput_rows_sec=9800.0, previous_worker_count=6,
            worker_count_changed=False,
        )
        d = analyzer.diagnose(m)
        self.assertEqual(d.category, DeviationCategory.STABLE)


if __name__ == "__main__":
    unittest.main()
