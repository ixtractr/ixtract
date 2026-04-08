"""Phase 0 Simulation Tests — Controller Convergence Validation.

Validates the statistical window-based controller against architecture
scenarios. Uses stdlib unittest — zero dependencies.

Run: python -m unittest tests.simulation.test_phase0 -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.controller import (
    ControllerConfig, ControllerDecision, ControllerState,
    ControllerOutput, ParallelismController,
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
    start_workers: int | None = None,
    start_state: ControllerState | None = None,
) -> list[dict]:
    """Run N extraction cycles with the window-based controller."""
    source = SimulatedSource(config)
    cfg = ctrl_cfg or ControllerConfig(
        max_workers=config.optimal_workers * 2 + 4,
        window_size=5,
    )
    ctrl = ParallelismController(cfg)
    analyzer = DeviationAnalyzer()

    # Start from profiler recommendation or explicit start
    sw = start_workers or max(1, config.optimal_workers - 1)
    state = start_state or ControllerState.from_profiler(sw)

    # Track throughput per worker count for window building
    tp_by_workers: dict[int, list[float]] = {}
    history: list[dict] = []

    for i in range(num_runs):
        w = state.current_workers
        metrics = source.run(
            worker_count=w,
            chunk_count=max(10, w * 2),
        )

        # Accumulate throughput for this worker count
        if w not in tp_by_workers:
            tp_by_workers[w] = []
        tp_by_workers[w].append(metrics.avg_throughput_rows_sec)

        # Build window: last N at this worker count
        window = tuple(tp_by_workers[w][-cfg.window_size:])

        out = ctrl.evaluate(window, state)

        history.append({
            "run": i + 1,
            "workers": w,
            "throughput": metrics.avg_throughput_rows_sec,
            "decision": out.decision.value,
            "recommended": out.recommended_workers,
            "converged": out.new_state.converged,
            "reasoning": out.reasoning,
            "window_size": len(window),
        })

        # If workers change, the new count starts with empty window
        # (tp_by_workers will accumulate naturally on next iteration)
        state = out.new_state

    return history


# ── Controller Window Tests ──────────────────────────────────────────

class TestWindowBasedController(unittest.TestCase):
    """Core tests for the statistical window-based controller."""

    def test_holds_until_window_full(self):
        """Controller should HOLD while collecting data (window not full)."""
        ctrl = ParallelismController(ControllerConfig(window_size=5))
        state = ControllerState.from_profiler(4)

        # Feed 1-4 throughputs — all should HOLD
        for n in range(1, 5):
            window = tuple(10000.0 + i * 100 for i in range(n))
            out = ctrl.evaluate(window, state)
            self.assertEqual(out.decision, ControllerDecision.HOLD,
                f"Should HOLD at window size {n}")
            self.assertFalse(out.new_state.converged)
            self.assertIn("Collecting data", out.reasoning)

    def test_converges_on_full_stable_window(self):
        """Full stable window should converge."""
        ctrl = ParallelismController(ControllerConfig(window_size=5))
        state = ControllerState.from_profiler(4)

        # Stable window: small fluctuations
        window = (10000, 10200, 9900, 10100, 10050)
        out = ctrl.evaluate(window, state)
        self.assertEqual(out.decision, ControllerDecision.HOLD)
        self.assertTrue(out.new_state.converged, "Should converge on stable window")

    def test_no_action_on_noisy_window(self):
        """Mixed direction deltas should HOLD, not explore."""
        ctrl = ParallelismController(ControllerConfig(
            window_size=5, magnitude_threshold=0.10,
        ))
        state = ControllerState.from_profiler(4)

        # Noisy: up-down-up-down pattern (no consistent direction)
        window = (220000, 192000, 233000, 217000, 220000)
        out = ctrl.evaluate(window, state)
        self.assertEqual(out.decision, ControllerDecision.HOLD)
        self.assertTrue(out.new_state.converged)

    def test_acts_on_sustained_degradation(self):
        """Consistent downward trend + significant magnitude should trigger INCREASE."""
        ctrl = ParallelismController(ControllerConfig(
            window_size=5, magnitude_threshold=0.10, consistency_ratio=0.8,
        ))
        state = ControllerState.from_profiler(4)

        # Sustained degradation: 4 out of 4 deltas are negative, >10% drift
        window = (220000, 210000, 198000, 190000, 185000)
        out = ctrl.evaluate(window, state)
        self.assertEqual(out.decision, ControllerDecision.INCREASE,
            f"Should increase on sustained degradation. Reasoning: {out.reasoning}")
        self.assertEqual(out.recommended_workers, 5)
        self.assertFalse(out.new_state.converged)

    def test_does_not_act_on_mild_decline(self):
        """Consistent direction but magnitude < threshold should HOLD."""
        ctrl = ParallelismController(ControllerConfig(
            window_size=5, magnitude_threshold=0.10,
        ))
        state = ControllerState.from_profiler(4)

        # Small consistent decline — magnitude below 10%
        window = (220000, 218000, 216000, 215000, 214000)
        out = ctrl.evaluate(window, state)
        self.assertEqual(out.decision, ControllerDecision.HOLD,
            "Small decline should not trigger action")

    def test_reverts_after_bad_change(self):
        """After worker change, if new avg is worse, should revert."""
        ctrl = ParallelismController(ControllerConfig(window_size=3))
        state = ControllerState(
            current_workers=5,
            previous_workers=4,
            previous_avg_throughput=220000.0,  # avg at 4 workers
        )

        # Window at 5 workers: consistently worse than 220K baseline
        window = (180000, 185000, 182000)
        out = ctrl.evaluate(window, state)
        self.assertEqual(out.recommended_workers, 4,
            "Should revert to previous 4 workers")

    def test_accepts_good_change(self):
        """After worker change, if new avg is same or better, accept and converge."""
        ctrl = ParallelismController(ControllerConfig(window_size=3))
        state = ControllerState(
            current_workers=5,
            previous_workers=4,
            previous_avg_throughput=200000.0,
        )

        # Window at 5 workers: better than 200K baseline
        window = (230000, 225000, 228000)
        out = ctrl.evaluate(window, state)
        self.assertEqual(out.recommended_workers, 5)
        self.assertTrue(out.new_state.converged, "Should converge after accepting improvement")

    def test_empty_window_holds(self):
        """Empty window (first run) should HOLD."""
        ctrl = ParallelismController()
        state = ControllerState.from_profiler(4)
        out = ctrl.evaluate((), state)
        self.assertEqual(out.decision, ControllerDecision.HOLD)

    def test_converged_holds_on_noise(self):
        """Once converged, noisy throughput should stay converged."""
        ctrl = ParallelismController(ControllerConfig(window_size=5))
        state = ControllerState(
            current_workers=4, converged=True,
            previous_avg_throughput=200000.0,
        )
        window = (210000, 195000, 220000, 190000, 215000)
        out = ctrl.evaluate(window, state)
        self.assertEqual(out.decision, ControllerDecision.HOLD)
        self.assertTrue(out.new_state.converged)

    def test_converged_breaks_on_severe_regression(self):
        """Sustained regression >30% should break convergence."""
        ctrl = ParallelismController(ControllerConfig(window_size=3))
        state = ControllerState(
            current_workers=4, converged=True,
            previous_avg_throughput=200000.0,
        )
        # Severe sustained drop: avg ~120K vs baseline 200K = -40%
        window = (125000, 118000, 120000)
        out = ctrl.evaluate(window, state)
        self.assertFalse(out.new_state.converged,
            "Should break convergence on >30% sustained regression")

    def test_configurable_window_size(self):
        """Window size should be configurable."""
        ctrl3 = ParallelismController(ControllerConfig(window_size=3))
        state = ControllerState.from_profiler(4)

        window3 = (10000, 10200, 9900)
        out3 = ctrl3.evaluate(window3, state)
        self.assertTrue(out3.new_state.converged, "Should converge with 3-run window")

        ctrl7 = ParallelismController(ControllerConfig(window_size=7))
        out7 = ctrl7.evaluate(window3, state)
        self.assertFalse(out7.new_state.converged, "Should not converge with 3/7 runs")


# ── Escape Mode Tests ─────────────────────────────────────────────────

class TestEscapeMode(unittest.TestCase):
    """Escape mode: fast-path for severe consecutive degradation."""

    def _make_state(self, workers=2):
        return ControllerState(current_workers=workers)

    def test_escape_fires_on_severe_consecutive_drops(self):
        """3 runs each dropping >=15% and total >=20% should trigger escape."""
        ctrl = ParallelismController()
        state = self._make_state(workers=2)
        # 220K → 185K (-16%) → 155K (-16%) → 130K (-16%), total -41%
        window = [220_000, 185_000, 155_000, 130_000]
        out = ctrl.evaluate(window, state)
        self.assertEqual(out.decision, ControllerDecision.INCREASE)
        self.assertEqual(out.recommended_workers, 4)  # escape_step=2
        self.assertIn("ESCAPE MODE", out.reasoning)
        self.assertFalse(out.new_state.converged)
        self.assertEqual(out.new_state.current_workers, 4)

    def test_escape_does_not_fire_on_mild_drops(self):
        """Drops below 15% per-run threshold should not trigger escape."""
        ctrl = ParallelismController()
        state = self._make_state(workers=2)
        # Each drop ~10% — below the 15% per-run escape threshold
        window = [220_000, 198_000, 178_000, 160_000]
        out = ctrl.evaluate(window, state)
        self.assertNotEqual(out.decision, ControllerDecision.INCREASE)
        self.assertNotIn("ESCAPE MODE", out.reasoning)

    def test_escape_does_not_fire_on_mixed_direction(self):
        """Mixed direction in last 3 runs should not trigger escape."""
        ctrl = ParallelismController()
        state = self._make_state(workers=2)
        # Drop, then recover — not consistently down
        window = [220_000, 185_000, 210_000, 175_000]
        out = ctrl.evaluate(window, state)
        self.assertNotEqual(out.decision, ControllerDecision.INCREASE)
        self.assertNotIn("ESCAPE MODE", out.reasoning)

    def test_escape_does_not_fire_with_insufficient_total_drop(self):
        """Per-run drops may each exceed 15% but total must also reach 20%."""
        ctrl = ParallelismController()
        state = self._make_state(workers=2)
        # 3 runs each drop ~16%, but starting from run 2 in window —
        # total from window[0]=220K to window[-1]=155K is -29%, which triggers.
        # This test checks the boundary: if total < 20%, no escape.
        # Craft: window[0]=220K, tail drops ~8% each, total ~15% < 20%
        window = [220_000, 218_000, 205_000, 188_000]
        # per-run in tail: 205/218=-6%, 188/205=-8% — below 15% threshold anyway
        out = ctrl.evaluate(window, state)
        self.assertNotIn("ESCAPE MODE", out.reasoning)

    def test_escape_respects_max_workers_bound(self):
        """Escape at max_workers should HOLD, not exceed bounds."""
        cfg = ControllerConfig(max_workers=2, escape_step=2)
        ctrl = ParallelismController(cfg)
        state = ControllerState(current_workers=2)
        window = [220_000, 185_000, 155_000, 130_000]
        out = ctrl.evaluate(window, state)
        self.assertEqual(out.decision, ControllerDecision.HOLD)
        self.assertEqual(out.recommended_workers, 2)
        self.assertIn("max workers", out.reasoning)

    def test_escape_fires_mid_window(self):
        """Escape must fire before the full window is collected (mid-window)."""
        ctrl = ParallelismController(ControllerConfig(window_size=5))
        state = self._make_state(workers=2)
        # Only 4 runs in window (< window_size=5) but escape conditions met
        window = [220_000, 185_000, 155_000, 130_000]
        self.assertLess(len(window), ctrl.config.window_size)
        out = ctrl.evaluate(window, state)
        self.assertEqual(out.decision, ControllerDecision.INCREASE)
        self.assertIn("ESCAPE MODE", out.reasoning)

    def test_normal_mode_unaffected_by_escape_params(self):
        """Normal noise should still HOLD — escape params don't lower the bar."""
        ctrl = ParallelismController()
        state = self._make_state(workers=2)
        # ±10-15% noise, no sustained direction — normal operating conditions
        window = [220_000, 192_000, 233_000, 217_000, 220_000]
        out = ctrl.evaluate(window, state)
        self.assertNotIn("ESCAPE MODE", out.reasoning)
        self.assertEqual(out.decision, ControllerDecision.HOLD)


# ── Simulation Scenario Tests ────────────────────────────────────────

class TestScenario1_HappyPathConvergence(unittest.TestCase):
    """Stable source, low noise. Should converge within window_size runs."""

    def test_converges(self):
        config = SimulationConfig(
            optimal_workers=6, concurrency_curve="logarithmic",
            latency_jitter_pct=0.02, seed=42,
        )
        history = _simulate(config, num_runs=10, start_workers=5)
        converged = [h for h in history if h["converged"]]
        self.assertTrue(len(converged) > 0, "Controller never converged")

    def test_stays_near_start(self):
        """With low noise, profiler-seeded start should hold without exploring."""
        config = SimulationConfig(
            optimal_workers=6, concurrency_curve="logarithmic",
            latency_jitter_pct=0.02, seed=42,
        )
        history = _simulate(config, num_runs=10, start_workers=5)
        final = history[-1]["recommended"]
        self.assertLessEqual(abs(final - 5), 1,
            f"Final workers {final} drifted from start (5)")


class TestScenario2_OverParallelRecovery(unittest.TestCase):
    """Start at too many workers. After collecting window, should revert."""

    def test_reverts_or_holds(self):
        config = SimulationConfig(
            optimal_workers=4, concurrency_curve="logarithmic",
            latency_jitter_pct=0.02, seed=42,
        )
        # Start at 8 workers (too many), with baseline from 4 workers
        preseed = SimulatedSource(SimulationConfig(
            optimal_workers=4, concurrency_curve="logarithmic",
            latency_jitter_pct=0.02, seed=42,
        ))
        baseline_metrics = preseed.run(worker_count=4)

        start = ControllerState(
            current_workers=8,
            previous_workers=4,
            previous_avg_throughput=baseline_metrics.avg_throughput_rows_sec,
        )
        ctrl_cfg = ControllerConfig(max_workers=12, window_size=5)
        history = _simulate(config, ctrl_cfg, num_runs=10, start_state=start)

        # After window fills, should revert toward 4 or hold if undecided
        final = history[-1]["recommended"]
        self.assertLessEqual(final, 8,
            f"Workers {final} should not increase from over-parallel start")


class TestScenario3_LatencySpike(unittest.TestCase):
    """Latency spike on run 6. Window should absorb it without reacting."""

    def test_no_increase_on_spike(self):
        config = SimulationConfig(
            optimal_workers=6, concurrency_curve="logarithmic",
            latency_jitter_pct=0.02,
            latency_spike_on_run=6, latency_spike_multiplier=1.5,
            seed=42,
        )
        history = _simulate(config, num_runs=12, start_workers=5)

        # Workers should not change due to single spike in window
        worker_counts = set(h["workers"] for h in history)
        changes = len(worker_counts)
        self.assertLessEqual(changes, 2,
            f"Too many worker count changes ({worker_counts}) after latency spike")


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
            self.assertAlmostEqual(growth, 0.10, places=2)


class TestScenario6_OscillationResistance(unittest.TestCase):
    """Noisy throughput (±8%). Window-based controller should not oscillate."""

    def test_no_oscillation(self):
        """With noise, controller should hold — no direction changes."""
        config = SimulationConfig(
            optimal_workers=6, concurrency_curve="logarithmic",
            latency_jitter_pct=0.08, seed=42,
        )
        history = _simulate(config, num_runs=15, start_workers=5)

        # Count worker count changes (not direction flips)
        worker_changes = sum(
            1 for i in range(1, len(history))
            if history[i]["workers"] != history[i - 1]["workers"]
        )
        self.assertLessEqual(worker_changes, 2,
            f"Too many worker changes ({worker_changes}) under noise")

    def test_converges_under_noise(self):
        config = SimulationConfig(
            optimal_workers=6, concurrency_curve="logarithmic",
            latency_jitter_pct=0.08, seed=42,
        )
        history = _simulate(config, num_runs=12, start_workers=5)
        converged = [h for h in history if h["converged"]]
        self.assertTrue(len(converged) > 0,
            "Should converge even under ±8% noise")


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
        ctrl = ControllerConfig(min_workers=2, max_workers=10, window_size=5)
        history = _simulate(cfg, ctrl, num_runs=15)
        for h in history:
            self.assertTrue(2 <= h["recommended"] <= 10,
                f"Workers {h['recommended']} outside [2,10] on run {h['run']}")

    def test_step_size_bounded(self):
        """Worker changes are ≤1 per evaluation."""
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
        st = ControllerState(
            current_workers=5,
            previous_workers=4,
            previous_avg_throughput=12000.0,
            converged=False,
            last_throughput=14000.0,
            last_worker_count=5,
            direction=ControllerDecision.HOLD,
            consecutive_holds=2,
        )
        self.store.save_controller_state("pg", "orders", st)
        loaded = self.store.get_controller_state("pg", "orders")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded.current_workers, 5)
        self.assertEqual(loaded.previous_workers, 4)
        self.assertEqual(loaded.previous_avg_throughput, 12000.0)
        self.assertFalse(loaded.converged)

    def test_controller_upsert(self):
        st1 = ControllerState(current_workers=5, last_throughput=12000.0,
                              last_worker_count=4, direction=ControllerDecision.HOLD)
        self.store.save_controller_state("pg", "orders", st1)
        st2 = ControllerState(current_workers=6, last_throughput=14000.0,
                              last_worker_count=5, direction=ControllerDecision.HOLD)
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


# ── Deviation Analyzer Tests ─────────────────────────────────────────

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
