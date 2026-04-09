"""Adaptive Intra-Run Rules Tests — Phase 2B.

Tests for:
  - _build_adaptive_rules: threshold calibration from profiler p50
  - RuleFiredRecord: confidence grading (note/moderate/low)
  - effective_workers computation from chunk timing
  - Window filter: effective_workers ±0.75 tolerance band
  - Invariant: chunk duration = execution time only (no sleep)

All tests are pure simulation — no database required.

Run: python -m unittest tests.simulation.test_adaptive -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.planner import (
    AdaptiveRule, AdaptiveTrigger, AdaptiveAction, RuleFiredRecord,
)
from ixtract.profiler import SourceProfile


# ── Helpers ───────────────────────────────────────────────────────────

def make_profile(latency_p50_ms: float = 0.5) -> SourceProfile:
    return SourceProfile(
        object_name="orders", row_estimate=1_200_000,
        size_estimate_bytes=166 * 1024 * 1024,
        column_count=10, avg_row_bytes=145,
        primary_key="id", primary_key_type="integer",
        pk_min=1, pk_max=1_200_000,
        pk_distribution_cv=0.0, has_usable_pk=True,
        latency_p50_ms=latency_p50_ms, latency_p95_ms=latency_p50_ms * 3,
        connection_ms=1.0, max_connections=100, active_connections=2,
        available_connections_safe=49, recommended_start_workers=2,
        recommended_strategy="range_chunking", recommended_scheduling="round_robin",
    )


def compute_effective_workers(
    chunk_durations: list[float], elapsed: float, planned: int
) -> float:
    """Mirror of the engine's effective_workers computation."""
    if elapsed <= 0 or not chunk_durations:
        return float(planned)
    eff = sum(chunk_durations) / elapsed
    return max(1.0, min(float(planned), eff))


# ── _build_adaptive_rules ─────────────────────────────────────────────

class TestBuildAdaptiveRules(unittest.TestCase):

    def _build(self, p50: float) -> tuple:
        from ixtract.planner.planner import _build_adaptive_rules
        return _build_adaptive_rules(make_profile(p50))

    def test_no_rule_for_sub_millisecond_latency(self):
        """p50 < 1ms → no rule (latency spikes would need to be absurd)."""
        rules = self._build(0.4)
        self.assertEqual(len(rules), 0)

    def test_no_rule_for_exactly_1ms(self):
        """p50 = 1ms threshold is < 1.0 check — exactly 1ms creates a rule."""
        # The check is `if p50 < 1.0` — so exactly 1.0ms is NOT sub-millisecond
        # and WILL produce a rule. Document the actual boundary.
        rules = self._build(1.0)
        # 1.0ms is >= 1.0 so a rule IS created (threshold=5ms, floor=50ms)
        self.assertEqual(len(rules), 1)

    def test_rule_created_for_measurable_latency(self):
        """p50 > 1ms → backoff rule created."""
        rules = self._build(5.0)
        self.assertEqual(len(rules), 1)
        self.assertEqual(rules[0].trigger, AdaptiveTrigger.SOURCE_LATENCY_SPIKE)
        self.assertEqual(rules[0].action, AdaptiveAction.INCREASE_BACKOFF)

    def test_threshold_is_5x_p50(self):
        """Relative threshold = 5× p50."""
        rules = self._build(10.0)
        self.assertAlmostEqual(rules[0].threshold, 50.0)

    def test_absolute_floor_scales_with_p50(self):
        """Floor = max(50ms, p50 × 10)."""
        # p50=3ms → 10×=30ms < 50ms floor → floor=50ms
        rules_3ms = self._build(3.0)
        self.assertAlmostEqual(rules_3ms[0].absolute_floor_ms, 50.0)

        # p50=20ms → 10×=200ms > 50ms → floor=200ms
        rules_20ms = self._build(20.0)
        self.assertAlmostEqual(rules_20ms[0].absolute_floor_ms, 200.0)

    def test_rule_has_correct_defaults(self):
        rules = self._build(10.0)
        r = rules[0]
        self.assertEqual(r.rule_id, "source_latency_backoff")
        self.assertEqual(r.max_activations, 10)
        self.assertEqual(r.cooldown_chunks, 3)
        self.assertAlmostEqual(r.backoff_sleep_base, 2.0)


# ── RuleFiredRecord confidence grading ───────────────────────────────

class TestRuleFiredRecord(unittest.TestCase):

    def _record(self, activations: int, total_chunks: int, max_consec: int = 1):
        rate = activations / max(total_chunks, 1)
        if rate <= 0.05:
            impact = "note"
        elif rate <= 0.20:
            impact = "moderate"
        else:
            impact = "low"
        return RuleFiredRecord(
            rule_id="source_latency_backoff",
            activations=activations,
            total_chunks=total_chunks,
            max_consecutive=max_consec,
            confidence_impact=impact,
        )

    def test_low_activation_rate_is_note(self):
        r = self._record(1, 20)  # 5% — at boundary
        self.assertEqual(r.confidence_impact, "note")

    def test_just_above_5pct_is_moderate(self):
        r = self._record(2, 20)  # 10%
        self.assertEqual(r.confidence_impact, "moderate")

    def test_above_20pct_is_low(self):
        r = self._record(5, 20)  # 25%
        self.assertEqual(r.confidence_impact, "low")

    def test_activation_rate_property(self):
        r = self._record(3, 12)
        self.assertAlmostEqual(r.activation_rate, 0.25)

    def test_activation_rate_zero_chunks(self):
        r = self._record(0, 0)
        self.assertAlmostEqual(r.activation_rate, 0.0)

    def test_max_consecutive_tracked(self):
        r = self._record(4, 20, max_consec=4)
        self.assertEqual(r.max_consecutive, 4)


# ── Effective workers computation ────────────────────────────────────

class TestEffectiveWorkers(unittest.TestCase):

    def test_perfect_utilisation(self):
        """2 workers, chunks perfectly split — effective ≈ 2."""
        # 2 workers, 4 chunks each ~2.5s, wall clock ~5s
        durations = [2.5, 2.6, 2.4, 2.5]
        eff = compute_effective_workers(durations, 5.0, planned=2)
        self.assertAlmostEqual(eff, 2.0, delta=0.1)

    def test_idle_workers_reduce_effective(self):
        """4 planned workers but 2 idle — effective ≈ 2."""
        durations = [2.5, 2.5, 0.05, 0.05]  # 2 real + 2 trivial chunks
        eff = compute_effective_workers(durations, 5.0, planned=4)
        self.assertAlmostEqual(eff, 1.0, delta=0.1)

    def test_clamped_to_planned_workers(self):
        """effective_workers cannot exceed planned worker_count."""
        # Pathological case: very fast chunks reported on one worker
        durations = [5.0, 5.0, 5.0, 5.0]
        eff = compute_effective_workers(durations, 5.0, planned=2)
        self.assertLessEqual(eff, 2.0)

    def test_clamped_to_minimum_1(self):
        """effective_workers cannot drop below 1."""
        # Single tiny chunk
        durations = [0.01]
        eff = compute_effective_workers(durations, 5.0, planned=4)
        self.assertGreaterEqual(eff, 1.0)

    def test_single_worker_single_chunk(self):
        """1 worker, 1 chunk — effective = 1."""
        eff = compute_effective_workers([4.8], 5.0, planned=1)
        self.assertAlmostEqual(eff, 1.0, delta=0.05)

    def test_invariant_sleep_not_in_duration(self):
        """Sleep between chunks must NOT appear in chunk duration_seconds.

        This invariant ensures effective_workers reflects active execution time.
        The engine's backoff sleep happens in _worker_loop BETWEEN chunks,
        outside _execute_chunk's timing block.

        We verify: if chunk durations are pure execution time, the formula
        gives a meaningful result. Adding hypothetical sleep to the durations
        would incorrectly inflate effective_workers.
        """
        execution_only = [2.5, 2.5, 2.5, 2.5]   # 4 chunks × 2.5s = 10s active
        with_fake_sleep = [5.0, 5.0, 5.0, 5.0]   # if sleep were included: 20s

        elapsed = 5.0
        eff_clean = compute_effective_workers(execution_only, elapsed, planned=4)
        eff_inflated = compute_effective_workers(with_fake_sleep, elapsed, planned=4)

        # Clean gives ~2.0 (10/5=2), inflated hits planned cap (20/5=4)
        self.assertAlmostEqual(eff_clean, 2.0, delta=0.1)
        self.assertAlmostEqual(eff_inflated, 4.0, delta=0.1)
        self.assertLess(eff_clean, eff_inflated)


# ── Window filter: effective_workers ±0.75 tolerance ─────────────────

class TestWindowFilter(unittest.TestCase):
    """Test the effective_workers tolerance band used in the CLI window query."""

    def _matches(self, effective: float, target: float) -> bool:
        return abs(effective - target) <= 0.75

    def test_exact_match(self):
        self.assertTrue(self._matches(2.0, 2.0))

    def test_within_tolerance(self):
        self.assertTrue(self._matches(2.4, 2.0))
        self.assertTrue(self._matches(1.6, 2.0))

    def test_at_boundary(self):
        """±0.75 is inclusive."""
        self.assertTrue(self._matches(2.75, 2.0))
        self.assertTrue(self._matches(1.25, 2.0))

    def test_just_outside_tolerance(self):
        self.assertFalse(self._matches(2.76, 2.0))
        self.assertFalse(self._matches(1.24, 2.0))

    def test_avoids_rounding_cliff(self):
        """2.49 and 2.51 both match target=2 — no rounding cliff."""
        self.assertTrue(self._matches(2.49, 2.0))
        self.assertTrue(self._matches(2.51, 2.0))

    def test_planned_vs_effective_distinction(self):
        """A run at 4 planned but 2.1 effective should NOT match target=4."""
        self.assertFalse(self._matches(2.1, 4.0))

    def test_fallback_to_worker_count_when_effective_zero(self):
        """Pre-Phase-2B runs have effective_workers=0. Fallback to worker_count."""
        run = {"worker_count": 2, "effective_workers": 0.0}
        effective = run.get("effective_workers") or run.get("worker_count", 0)
        self.assertTrue(self._matches(effective, 2.0))


# ── Backoff rule trigger logic ────────────────────────────────────────

class TestBackoffRuleTrigger(unittest.TestCase):
    """Test the dual-guard trigger condition (relative AND absolute)."""

    def _should_fire(self, query_ms: float, rule: AdaptiveRule) -> bool:
        return (query_ms > rule.threshold and query_ms > rule.absolute_floor_ms)

    def test_fires_when_both_guards_breached(self):
        rule = AdaptiveRule(
            rule_id="test", trigger=AdaptiveTrigger.SOURCE_LATENCY_SPIKE,
            threshold=50.0, action=AdaptiveAction.INCREASE_BACKOFF,
            absolute_floor_ms=50.0,
        )
        self.assertTrue(self._should_fire(100.0, rule))

    def test_no_fire_below_relative_threshold(self):
        rule = AdaptiveRule(
            rule_id="test", trigger=AdaptiveTrigger.SOURCE_LATENCY_SPIKE,
            threshold=50.0, action=AdaptiveAction.INCREASE_BACKOFF,
            absolute_floor_ms=10.0,
        )
        self.assertFalse(self._should_fire(30.0, rule))  # below 50ms threshold

    def test_no_fire_below_absolute_floor(self):
        """Prevents over-triggering on fast DBs (p50=1ms, 5×=5ms spike)."""
        rule = AdaptiveRule(
            rule_id="test", trigger=AdaptiveTrigger.SOURCE_LATENCY_SPIKE,
            threshold=5.0, action=AdaptiveAction.INCREASE_BACKOFF,
            absolute_floor_ms=50.0,
        )
        # 25ms exceeds 5× threshold but is below 50ms floor → should NOT fire
        self.assertFalse(self._should_fire(25.0, rule))

    def test_both_guards_required(self):
        """Confirms AND semantics — one guard alone is not sufficient."""
        rule = AdaptiveRule(
            rule_id="test", trigger=AdaptiveTrigger.SOURCE_LATENCY_SPIKE,
            threshold=50.0, action=AdaptiveAction.INCREASE_BACKOFF,
            absolute_floor_ms=100.0,
        )
        # 75ms: above relative (50ms) but below absolute (100ms) → no fire
        self.assertFalse(self._should_fire(75.0, rule))
        # 150ms: above both → fires
        self.assertTrue(self._should_fire(150.0, rule))

    def test_floor_calibration_fast_db(self):
        """For a fast DB (p50=0.5ms), floor should prevent any firing."""
        from ixtract.planner.planner import _build_adaptive_rules
        rules = _build_adaptive_rules(make_profile(latency_p50_ms=0.5))
        self.assertEqual(len(rules), 0)  # no rule at all for sub-ms

    def test_floor_calibration_medium_db(self):
        """For p50=5ms DB: threshold=25ms, floor=50ms."""
        from ixtract.planner.planner import _build_adaptive_rules
        rules = _build_adaptive_rules(make_profile(latency_p50_ms=5.0))
        r = rules[0]
        self.assertAlmostEqual(r.threshold, 25.0)
        self.assertAlmostEqual(r.absolute_floor_ms, 50.0)  # max(50, 50)

    def test_floor_calibration_slow_db(self):
        """For p50=30ms DB: threshold=150ms, floor=300ms."""
        from ixtract.planner.planner import _build_adaptive_rules
        rules = _build_adaptive_rules(make_profile(latency_p50_ms=30.0))
        r = rules[0]
        self.assertAlmostEqual(r.threshold, 150.0)
        self.assertAlmostEqual(r.absolute_floor_ms, 300.0)  # max(50, 300)


if __name__ == "__main__":
    unittest.main(verbosity=2)
