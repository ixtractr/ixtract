"""Context-Weighted Planning Tests — Steps 1 & 2.

Tests for:
  - ExecutionContext dataclass and measurement helpers
  - Weight redistribution on platforms without system load
  - All 5 dimension similarity functions
  - Composite similarity score (symmetry, bounds, correctness)
  - Row growth guard (symmetric, boundary)
  - Confidence assessment (all levels, all downgrade triggers)
  - Estimation paths (weighted, EWMA, cold start)
  - CLI formatting helpers

All tests are pure simulation — no database required.

Run: python -m unittest tests.simulation.test_context -v
"""
from __future__ import annotations

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.context import (
    ExecutionContext, CONTEXT_SCHEMA_VERSION,
    classify_source_load, classify_network_quality, current_time_band,
    measure_system_load_per_core, _compute_effective_weights,
    format_context_summary,
    SOURCE_LOAD_CATEGORIES, NETWORK_QUALITY_CATEGORIES,
)
from ixtract.context.similarity import (
    similarity_score, DimensionBreakdown,
    sim_source_load, sim_concurrent_extractions, sim_time_band,
    sim_system_load, sim_network_quality,
    row_growth_guard, score_candidates, ScoredRun,
    ConfidenceReason, REASON_DISPLAY_ORDER,
    SCORE_EXCLUSION_THRESHOLD, SCORE_STRONG_MATCH_THRESHOLD,
    SCORE_SINGLE_MATCH_THRESHOLD, DOMINANCE_THRESHOLD,
    THROUGHPUT_CV_THRESHOLD, MAX_CANDIDATE_RUNS,
    _null_context, _zero_breakdown,
)
from ixtract.context.estimator import (
    estimate_throughput, compute_ewma, format_estimate_for_cli,
    CLAMP_LOWER_FACTOR, CLAMP_UPPER_FACTOR, EWMA_ALPHA,
)


# ── Test helpers ──────────────────────────────────────────────────────

def make_ctx(
    source_load="normal",
    concurrent=1,
    time_band=1,
    system_load=0.2,
    network="excellent",
    rows=1_200_000,
    sys_avail=True,
) -> ExecutionContext:
    w = _compute_effective_weights(sys_avail)
    return ExecutionContext(
        schema_version=CONTEXT_SCHEMA_VERSION,
        source_load=source_load,
        concurrent_extractions=concurrent,
        time_band=time_band,
        system_load_per_core=system_load if sys_avail else None,
        network_quality=network,
        row_estimate=rows,
        effective_weights=w,
        system_load_available=sys_avail,
    )


def make_scored_run(run_id, score, throughput, ctx=None) -> ScoredRun:
    if ctx is None:
        ctx = make_ctx()
    bd = DimensionBreakdown(1.0, 1.0, 1.0, 1.0, 1.0, score)
    return ScoredRun(
        run_id=run_id, score=score, throughput=throughput,
        breakdown=bd, context=ctx, excluded=False, exclusion_reason="",
    )


# ── Context dataclass ─────────────────────────────────────────────────

class TestExecutionContext(unittest.TestCase):

    def test_schema_version_is_1(self):
        ctx = make_ctx()
        self.assertEqual(ctx.schema_version, CONTEXT_SCHEMA_VERSION)
        self.assertEqual(ctx.schema_version, 1)

    def test_round_trip_serialization(self):
        ctx = make_ctx("high", 3, 2, 0.8, "degraded", rows=2_000_000)
        d = ctx.to_dict()
        restored = ExecutionContext.from_dict(d)
        self.assertEqual(restored.source_load, ctx.source_load)
        self.assertEqual(restored.concurrent_extractions, ctx.concurrent_extractions)
        self.assertEqual(restored.time_band, ctx.time_band)
        self.assertAlmostEqual(restored.system_load_per_core, ctx.system_load_per_core)
        self.assertEqual(restored.network_quality, ctx.network_quality)
        self.assertEqual(restored.row_estimate, ctx.row_estimate)
        self.assertEqual(restored.schema_version, ctx.schema_version)

    def test_json_round_trip(self):
        ctx = make_ctx()
        restored = ExecutionContext.from_json(ctx.to_json())
        self.assertEqual(restored.source_load, ctx.source_load)
        self.assertEqual(restored.effective_weights, ctx.effective_weights)

    def test_effective_weights_stored_in_context(self):
        """Weights must be stored — not recomputed at comparison time."""
        ctx = make_ctx(sys_avail=True)
        self.assertIn("system_load", ctx.effective_weights)
        self.assertAlmostEqual(ctx.effective_weights["system_load"], 0.15, places=5)

    def test_system_load_unavailable_stores_redistribution(self):
        ctx = make_ctx(sys_avail=False)
        self.assertFalse(ctx.system_load_available)
        self.assertIsNone(ctx.system_load_per_core)
        self.assertNotIn("system_load", ctx.effective_weights)

    def test_format_summary_no_crash(self):
        ctx = make_ctx()
        summary = format_context_summary(ctx)
        self.assertIn("Source load", summary)
        self.assertIn("normal", summary)


# ── Weight redistribution ─────────────────────────────────────────────

class TestWeightRedistribution(unittest.TestCase):

    def test_with_system_load_sums_to_1(self):
        w = _compute_effective_weights(True)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=9)

    def test_without_system_load_sums_to_1(self):
        w = _compute_effective_weights(False)
        self.assertAlmostEqual(sum(w.values()), 1.0, places=9)

    def test_without_system_load_excludes_dimension(self):
        w = _compute_effective_weights(False)
        self.assertNotIn("system_load", w)
        self.assertEqual(len(w), 4)

    def test_relative_ratios_preserved_after_redistribution(self):
        w_with = _compute_effective_weights(True)
        w_without = _compute_effective_weights(False)
        # After redistribution, source_load should still dominate concurrency
        # (both increase proportionally)
        self.assertGreater(
            w_without["source_load"],
            w_without["concurrent_extractions"]
        )

    def test_no_dimension_exceeds_half(self):
        for sys_avail in [True, False]:
            w = _compute_effective_weights(sys_avail)
            for k, v in w.items():
                self.assertLessEqual(v, 0.5, f"Dimension {k} exceeds 0.5: {v}")


# ── Classification helpers ────────────────────────────────────────────

class TestClassifications(unittest.TestCase):

    def test_source_load_categories(self):
        self.assertEqual(classify_source_load(0, 100), "low")
        self.assertEqual(classify_source_load(19, 100), "low")
        self.assertEqual(classify_source_load(20, 100), "normal")
        self.assertEqual(classify_source_load(59, 100), "normal")
        self.assertEqual(classify_source_load(60, 100), "high")
        self.assertEqual(classify_source_load(79, 100), "high")
        self.assertEqual(classify_source_load(80, 100), "critical")
        self.assertEqual(classify_source_load(100, 100), "critical")

    def test_source_load_zero_max(self):
        self.assertEqual(classify_source_load(0, 0), "normal")

    def test_network_quality_categories(self):
        self.assertEqual(classify_network_quality(0.5), "excellent")
        self.assertEqual(classify_network_quality(1.0), "good")   # boundary: ≥ 1ms
        self.assertEqual(classify_network_quality(3.0), "good")
        self.assertEqual(classify_network_quality(5.0), "degraded")
        self.assertEqual(classify_network_quality(10.0), "degraded")
        self.assertEqual(classify_network_quality(20.0), "poor")
        self.assertEqual(classify_network_quality(100.0), "poor")

    def test_time_band(self):
        from datetime import datetime, timezone
        self.assertEqual(current_time_band(datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc)), 0)
        self.assertEqual(current_time_band(datetime(2026, 1, 1, 3, 59, tzinfo=timezone.utc)), 0)
        self.assertEqual(current_time_band(datetime(2026, 1, 1, 4, 0, tzinfo=timezone.utc)), 1)
        self.assertEqual(current_time_band(datetime(2026, 1, 1, 23, 59, tzinfo=timezone.utc)), 5)


# ── Individual dimension similarity ───────────────────────────────────

class TestDimensionSimilarity(unittest.TestCase):

    def test_source_load_same(self):
        a = make_ctx(source_load="normal")
        self.assertAlmostEqual(sim_source_load(a, a), 1.0)

    def test_source_load_adjacent(self):
        a = make_ctx(source_load="normal")
        b = make_ctx(source_load="high")
        self.assertAlmostEqual(sim_source_load(a, b), 0.5)

    def test_source_load_far(self):
        a = make_ctx(source_load="low")
        b = make_ctx(source_load="critical")
        self.assertAlmostEqual(sim_source_load(a, b), 0.1)

    def test_concurrency_same(self):
        a = make_ctx(concurrent=2)
        self.assertAlmostEqual(sim_concurrent_extractions(a, a), 1.0)

    def test_concurrency_decays(self):
        a = make_ctx(concurrent=0)
        b = make_ctx(concurrent=2)
        c = make_ctx(concurrent=4)
        s2 = sim_concurrent_extractions(a, b)
        s4 = sim_concurrent_extractions(a, c)
        self.assertGreater(s2, s4)
        self.assertAlmostEqual(s2, math.exp(-4/8), places=5)

    def test_concurrency_capped_at_20(self):
        a = make_ctx(concurrent=20)
        b = make_ctx(concurrent=100)  # should behave same as 20
        b_capped = make_ctx(concurrent=20)
        self.assertAlmostEqual(
            sim_concurrent_extractions(a, b),
            sim_concurrent_extractions(a, b_capped),
        )

    def test_time_band_same(self):
        a = make_ctx(time_band=2)
        self.assertAlmostEqual(sim_time_band(a, a), 1.0)

    def test_time_band_adjacent(self):
        a = make_ctx(time_band=1)
        b = make_ctx(time_band=2)
        self.assertAlmostEqual(sim_time_band(a, b), 0.6)

    def test_time_band_cyclic_0_and_5(self):
        """Band 0 (00-04) and band 5 (20-24) are adjacent."""
        a = make_ctx(time_band=0)
        b = make_ctx(time_band=5)
        self.assertAlmostEqual(sim_time_band(a, b), 0.6)

    def test_time_band_two_apart(self):
        a = make_ctx(time_band=0)
        b = make_ctx(time_band=2)
        self.assertAlmostEqual(sim_time_band(a, b), 0.2)

    def test_time_band_opposite(self):
        a = make_ctx(time_band=0)
        b = make_ctx(time_band=3)
        self.assertAlmostEqual(sim_time_band(a, b), 0.0)

    def test_system_load_same(self):
        a = make_ctx(system_load=0.5)
        self.assertAlmostEqual(sim_system_load(a, a), 1.0)

    def test_system_load_none_returns_1(self):
        """Unavailable system load returns neutral 1.0 (weight will be 0)."""
        a = make_ctx(sys_avail=False)
        b = make_ctx(sys_avail=False)
        self.assertAlmostEqual(sim_system_load(a, b), 1.0)

    def test_network_same(self):
        a = make_ctx(network="good")
        self.assertAlmostEqual(sim_network_quality(a, a), 1.0)

    def test_network_adjacent(self):
        a = make_ctx(network="excellent")
        b = make_ctx(network="good")
        self.assertAlmostEqual(sim_network_quality(a, b), 0.5)


# ── Composite similarity ──────────────────────────────────────────────

class TestCompositeSimilarity(unittest.TestCase):

    def test_identical_contexts_score_1(self):
        ctx = make_ctx()
        score, _ = similarity_score(ctx, ctx)
        self.assertAlmostEqual(score, 1.0)

    def test_score_bounded_0_to_1(self):
        a = make_ctx("low", 0, 0, 0.0, "excellent")
        b = make_ctx("critical", 20, 3, 2.0, "poor")
        score, _ = similarity_score(a, b)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_symmetry(self):
        """sim(a, b) == sim(b, a) when effective_weights are identical."""
        a = make_ctx("normal", 1, 1, 0.2, "excellent")
        b = make_ctx("high", 3, 2, 0.8, "good")
        score_ab, _ = similarity_score(a, b)
        score_ba, _ = similarity_score(b, a)
        self.assertAlmostEqual(score_ab, score_ba, places=9)

    def test_similar_contexts_score_higher_than_different(self):
        reference = make_ctx("normal", 1, 1, 0.2, "excellent")
        similar   = make_ctx("normal", 2, 1, 0.3, "excellent")
        different = make_ctx("critical", 15, 4, 1.8, "poor")
        s_sim, _ = similarity_score(reference, similar)
        s_dif, _ = similarity_score(reference, different)
        self.assertGreater(s_sim, s_dif)

    def test_cyclic_time_adjacency_in_composite(self):
        a = make_ctx(time_band=0)
        b = make_ctx(time_band=5)
        score, bd = similarity_score(a, b)
        self.assertAlmostEqual(bd.time_band, 0.6)
        self.assertGreater(score, 0.5)   # time diff penalised but other dims same

    def test_breakdown_composite_matches_score(self):
        a = make_ctx()
        b = make_ctx("high", 3, 2, 0.8, "good")
        score, bd = similarity_score(a, b)
        self.assertAlmostEqual(score, bd.composite, places=6)


# ── Row growth guard ──────────────────────────────────────────────────

class TestRowGrowthGuard(unittest.TestCase):

    def test_same_rows_not_excluded(self):
        self.assertFalse(row_growth_guard(1_200_000, 1_200_000))

    def test_small_growth_not_excluded(self):
        self.assertFalse(row_growth_guard(1_200_000, 1_000_000))  # 16.7%

    def test_exactly_50pct_not_excluded(self):
        """Boundary is strict (>0.50), so 50% exactly is not excluded."""
        self.assertFalse(row_growth_guard(1_200_000, 600_000))

    def test_above_50pct_excluded(self):
        self.assertTrue(row_growth_guard(1_200_000, 550_000))  # 54.2%

    def test_symmetric(self):
        """row_growth_guard(a, b) == row_growth_guard(b, a)."""
        self.assertEqual(
            row_growth_guard(1_200_000, 500_000),
            row_growth_guard(500_000, 1_200_000),
        )

    def test_zero_rows_not_excluded(self):
        self.assertFalse(row_growth_guard(0, 0))


# ── Estimation paths ──────────────────────────────────────────────────

class TestEstimationPaths(unittest.TestCase):

    def test_strong_matches_weighted_estimate(self):
        matched = [
            make_scored_run("a", 0.91, 224_000),
            make_scored_run("b", 0.84, 236_000),
            make_scored_run("c", 0.71, 209_000),
        ]
        est = estimate_throughput(matched, [], [224_000, 236_000, 209_000], 0)
        self.assertEqual(est.method, "context_weighted")
        self.assertEqual(est.confidence.level, "high")

    def test_single_match_above_threshold_weighted(self):
        matched = [make_scored_run("d", 0.70, 220_000)]
        est = estimate_throughput(matched, [], [220_000], 0)
        self.assertEqual(est.method, "context_weighted")
        self.assertEqual(est.confidence.level, "low")
        self.assertIn(ConfidenceReason.SPARSE_EVIDENCE, est.confidence.reasons)

    def test_single_match_below_threshold_ewma(self):
        matched = [make_scored_run("e", 0.55, 220_000)]
        est = estimate_throughput(matched, [], [200_000, 210_000, 220_000], 0)
        self.assertEqual(est.method, "ewma")

    def test_no_strong_matches_ewma(self):
        matched = [make_scored_run("f", 0.35, 220_000), make_scored_run("g", 0.28, 210_000)]
        est = estimate_throughput(matched, [], [220_000, 210_000], 0)
        self.assertEqual(est.method, "ewma")

    def test_no_history_cold_start(self):
        est = estimate_throughput([], [], [], 50_000.0)
        self.assertEqual(est.method, "cold_start")
        self.assertAlmostEqual(est.value, 50_000.0)
        self.assertEqual(est.confidence.level, "low")
        self.assertIn(ConfidenceReason.FALLBACK_USED, est.confidence.reasons)

    def test_clamping_applied_on_outlier(self):
        # Historical range: 100K–200K. Weighted estimate of 350K should be clamped.
        matched = [make_scored_run("a", 0.95, 350_000), make_scored_run("b", 0.90, 340_000)]
        hist = [100_000, 150_000, 200_000]
        est = estimate_throughput(matched, [], hist, 0)
        self.assertTrue(est.clamped)
        self.assertLessEqual(est.value, max(hist) * CLAMP_UPPER_FACTOR + 1)

    def test_dominance_downgrades_confidence(self):
        # One run with 95% of total weight
        matched = [make_scored_run("a", 0.95, 300_000), make_scored_run("b", 0.21, 100_000)]
        est = estimate_throughput(matched, [], [300_000, 100_000], 0)
        self.assertIn(ConfidenceReason.DOMINANT_MATCH, est.confidence.reasons)
        self.assertIn(est.confidence.level, ("medium", "low"))

    def test_high_variance_downgrades_confidence(self):
        # CV: (300K - 100K) / 200K = 1.0 → well above 0.20
        matched = [
            make_scored_run("a", 0.90, 300_000),
            make_scored_run("b", 0.85, 100_000),
            make_scored_run("c", 0.80, 200_000),
        ]
        est = estimate_throughput(matched, [], [300_000, 100_000, 200_000], 0)
        self.assertIn(ConfidenceReason.HIGH_VARIANCE, est.confidence.reasons)

    def test_cv_not_computed_for_single_match(self):
        matched = [make_scored_run("a", 0.90, 220_000)]
        est = estimate_throughput(matched, [], [220_000], 0)
        self.assertIsNone(est.confidence.throughput_cv)


# ── EWMA ──────────────────────────────────────────────────────────────

class TestEWMA(unittest.TestCase):

    def test_single_value(self):
        self.assertAlmostEqual(compute_ewma([200_000]), 200_000)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(compute_ewma([]), 0.0)

    def test_recency_bias(self):
        # On an ascending series, EWMA lags — recent values pull it up
        # but not above simple mean (α=0.3 means 70% weight on history).
        # The correct property: EWMA on ascending is below simple mean,
        # on descending is above simple mean (lags the decline).
        asc = [100_000, 150_000, 200_000, 250_000]
        desc = [250_000, 200_000, 150_000, 100_000]
        ewma_asc = compute_ewma(asc)
        ewma_desc = compute_ewma(desc)
        mean = sum(asc) / len(asc)   # same for both
        # Ascending: EWMA < mean (lags upward trend)
        self.assertLess(ewma_asc, mean)
        # Descending: EWMA > mean (lags downward trend)
        self.assertGreater(ewma_desc, mean)


# ── Confidence reasons ordering ───────────────────────────────────────

class TestConfidenceReasons(unittest.TestCase):

    def test_fallback_used_is_highest_priority(self):
        self.assertEqual(REASON_DISPLAY_ORDER[0], ConfidenceReason.FALLBACK_USED)

    def test_dominant_match_is_lowest_priority(self):
        self.assertEqual(REASON_DISPLAY_ORDER[-1], ConfidenceReason.DOMINANT_MATCH)

    def test_fallback_always_shown_first_when_present(self):
        """FALLBACK_USED must appear first regardless of other reasons."""
        from ixtract.context.similarity import ConfidenceAssessment
        ca = ConfidenceAssessment(
            level="low",
            reasons=(
                ConfidenceReason.FALLBACK_USED,
                ConfidenceReason.HIGH_VARIANCE,
                ConfidenceReason.DOMINANT_MATCH,
            ),
            matched_run_count=0, max_similarity_score=0.0,
            throughput_cv=None, dominant_weight=None,
        )
        self.assertEqual(ca.reasons[0], ConfidenceReason.FALLBACK_USED)

    def test_reason_categories_cover_all_reasons(self):
        """Every ConfidenceReason must appear in REASON_CATEGORIES."""
        from ixtract.context.similarity import REASON_CATEGORIES
        for reason in ConfidenceReason:
            self.assertIn(reason, REASON_CATEGORIES,
                f"{reason} missing from REASON_CATEGORIES")

    def test_descriptive_reasons_do_not_degrade(self):
        """SPARSE_EVIDENCE and WEAK_MATCH must be 'descriptive' — not 'degrading'."""
        from ixtract.context.similarity import REASON_CATEGORIES
        self.assertEqual(REASON_CATEGORIES[ConfidenceReason.SPARSE_EVIDENCE], "descriptive")
        self.assertEqual(REASON_CATEGORIES[ConfidenceReason.WEAK_MATCH], "descriptive")

    def test_degrading_reasons_are_correctly_classified(self):
        from ixtract.context.similarity import REASON_CATEGORIES
        self.assertEqual(REASON_CATEGORIES[ConfidenceReason.HIGH_VARIANCE], "degrading")
        self.assertEqual(REASON_CATEGORIES[ConfidenceReason.DOMINANT_MATCH], "degrading")

    def test_canonical_phrases_defined_for_all_reasons(self):
        """Every reason must have a canonical CLI phrase."""
        from ixtract.context.similarity import _reason_label
        for reason in ConfidenceReason:
            label = _reason_label(reason)
            self.assertIsInstance(label, str)
            self.assertGreater(len(label), 0)

    def test_format_cli_max_2_reasons(self):
        from ixtract.context.similarity import ConfidenceAssessment
        ca = ConfidenceAssessment(
            level="low",
            reasons=(
                ConfidenceReason.FALLBACK_USED,
                ConfidenceReason.SPARSE_EVIDENCE,
                ConfidenceReason.HIGH_VARIANCE,
            ),
            matched_run_count=1, max_similarity_score=0.5,
            throughput_cv=0.25, dominant_weight=None,
        )
        formatted = ca.format_cli(max_reasons=2)
        self.assertIn("LOW", formatted)
        self.assertIn("using EWMA (no similar runs)", formatted)
        self.assertIn("+1 additional", formatted)

    def test_format_cli_single_reason_no_additional(self):
        from ixtract.context.similarity import ConfidenceAssessment
        ca = ConfidenceAssessment(
            level="medium",
            reasons=(ConfidenceReason.WEAK_MATCH,),
            matched_run_count=2, max_similarity_score=0.55,
            throughput_cv=None, dominant_weight=None,
        )
        formatted = ca.format_cli(max_reasons=2)
        self.assertNotIn("additional", formatted)


if __name__ == "__main__":
    unittest.main(verbosity=2)
