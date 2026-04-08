"""Similarity Engine — context-weighted historical planning.

Computes similarity scores between ExecutionContexts and produces
confidence-weighted throughput estimates.

Three-layer separation (enforced by module structure):
  context/__init__.py     — how runs relate (similarity)  ← this file
  context/estimator.py    — how throughput is computed
  context/confidence.py   — how much to trust it

All functions are pure — no I/O, no state mutation.

Similarity model (5 dimensions, weights sum to 1.0):
  source_load:            0.30  (category match)
  concurrent_extractions: 0.25  (Gaussian decay)
  time_band:              0.20  (cyclic 4-hour bands)
  system_load:            0.15  (Gaussian on normalised load)
  network_quality:        0.10  (category match)

If system_load is unavailable, weights are redistributed proportionally.
The effective_weights stored in ExecutionContext are used — not the base
weights — ensuring two contexts from different platforms compare correctly.

Similarity is symmetric: sim(a, b) == sim(b, a).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence

from ixtract.context import (
    ExecutionContext,
    SOURCE_LOAD_CATEGORIES,
    NETWORK_QUALITY_CATEGORIES,
)


# ── Exclusion thresholds ──────────────────────────────────────────────

SCORE_EXCLUSION_THRESHOLD     = 0.20   # runs below this are always excluded
SCORE_STRONG_MATCH_THRESHOLD  = 0.50   # at least one run must exceed for weighted estimate
SCORE_SINGLE_MATCH_THRESHOLD  = 0.65   # single run must exceed this to be used
ROW_GROWTH_GUARD_THRESHOLD    = 0.50   # 50% growth → exclude regardless of score
DOMINANCE_THRESHOLD           = 0.70   # one run has >70% total weight → flag
THROUGHPUT_CV_THRESHOLD       = 0.20   # CV among matched runs → flag
MAX_CANDIDATE_RUNS            = 50     # recency cap


# ── Confidence system ─────────────────────────────────────────────────

class ConfidenceReason(str, Enum):
    FALLBACK_USED   = "fallback_used"    # EWMA in use
    SPARSE_EVIDENCE = "sparse_evidence"  # single-run estimate
    WEAK_MATCH      = "weak_match"       # best score < 0.65
    HIGH_VARIANCE   = "high_variance"    # throughput CV > 0.20
    DOMINANT_MATCH  = "dominant_match"   # one run > 70% weight


# Display priority order — index = priority (0 = most prominent)
# INVARIANT: FALLBACK_USED must always appear first when present.
# This is guaranteed by its position at index 0 in REASON_DISPLAY_ORDER.
REASON_DISPLAY_ORDER: tuple[ConfidenceReason, ...] = (
    ConfidenceReason.FALLBACK_USED,
    ConfidenceReason.SPARSE_EVIDENCE,
    ConfidenceReason.WEAK_MATCH,
    ConfidenceReason.HIGH_VARIANCE,
    ConfidenceReason.DOMINANT_MATCH,
)

# Reason categories — internal classification for _assess_confidence() logic.
# structural:  sets the base confidence level (not an annotation)
# descriptive: annotates the level — never used to change it
# degrading:   reduces level by one step (each trigger fires independently)
REASON_CATEGORIES: dict[ConfidenceReason, str] = {
    ConfidenceReason.FALLBACK_USED:   "structural",
    ConfidenceReason.SPARSE_EVIDENCE: "descriptive",
    ConfidenceReason.WEAK_MATCH:      "descriptive",
    ConfidenceReason.HIGH_VARIANCE:   "degrading",
    ConfidenceReason.DOMINANT_MATCH:  "degrading",
}


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Confidence level with structured, ordered reasons."""
    level: str                              # "high" | "medium" | "low"
    reasons: tuple[ConfidenceReason, ...]   # ordered by REASON_DISPLAY_ORDER
    matched_run_count: int
    max_similarity_score: float
    throughput_cv: Optional[float]          # None for single-match (undefined)
    dominant_weight: Optional[float]        # None if no dominance detected

    def format_cli(self, max_reasons: int = 2) -> str:
        """Format for CLI output. Shows max_reasons, summarises the rest.

        Formatting rule: the first displayed reason is always the highest-priority
        reason. This is guaranteed because reasons are stored pre-sorted by
        REASON_DISPLAY_ORDER — slicing from the front preserves that ordering
        regardless of max_reasons or any future filtering.
        """
        level_str = self.level.upper()
        if not self.reasons:
            return level_str
        visible = self.reasons[:max_reasons]
        hidden = len(self.reasons) - max_reasons
        reason_strs = [_reason_label(r) for r in visible]
        if hidden > 0:
            reason_strs.append(f"+{hidden} additional factor{'s' if hidden > 1 else ''}")
        return f"{level_str} ({', '.join(reason_strs)})"


def _reason_label(r: ConfidenceReason) -> str:
    """Canonical CLI phrases for confidence reasons.

    Phrases describe the condition, not data values.
    Dynamic values (scores, weights, CV) are shown in the matched runs
    table and summary lines — not repeated here.
    """
    return {
        ConfidenceReason.FALLBACK_USED:   "using EWMA (no similar runs)",
        ConfidenceReason.SPARSE_EVIDENCE: "based on 1 historical run",
        ConfidenceReason.WEAK_MATCH:      "no strong context match",
        ConfidenceReason.HIGH_VARIANCE:   "performance inconsistent across runs",
        ConfidenceReason.DOMINANT_MATCH:  "single run dominates estimate",
    }[r]


# ── Per-dimension similarity functions ────────────────────────────────

def _category_sim(
    a: str,
    b: str,
    categories: tuple[str, ...],
    same: float = 1.0,
    adjacent: float = 0.5,
    two_apart: float = 0.2,
    opposite: float = 0.0,
) -> float:
    """Generic ordered-category similarity.

    Assumes categories are ordered (e.g. low < normal < high < critical).
    Distance of 0 → same, 1 → adjacent, 2 → two_apart, 3+ → opposite.
    """
    if a not in categories or b not in categories:
        return 0.5   # unknown → neutral
    ia, ib = categories.index(a), categories.index(b)
    dist = abs(ia - ib)
    if dist == 0:
        return same
    if dist == 1:
        return adjacent
    if dist == 2:
        return two_apart
    return opposite


def sim_source_load(a: ExecutionContext, b: ExecutionContext) -> float:
    """Source load similarity (ordered category match)."""
    return _category_sim(
        a.source_load, b.source_load,
        SOURCE_LOAD_CATEGORIES,
        same=1.0, adjacent=0.5, two_apart=0.1, opposite=0.1,
    )


def sim_concurrent_extractions(a: ExecutionContext, b: ExecutionContext) -> float:
    """Concurrent extractions similarity (Gaussian decay).

    exp(-diff² / (2 × σ²)) where σ=2: same=1.0, ±2=~0.61, ±4=~0.14.
    Both values capped at 20 before comparison.
    """
    ca = min(a.concurrent_extractions, 20)
    cb = min(b.concurrent_extractions, 20)
    diff = ca - cb
    return math.exp(-(diff ** 2) / 8.0)


def sim_time_band(a: ExecutionContext, b: ExecutionContext) -> float:
    """Time band similarity with cyclic adjacency.

    6 bands (0–5). Band 0 is adjacent to band 5 (00:00 ↔ 20:00).
    Same=1.0, adjacent=0.6, 2-apart=0.2, 3-apart (opposite)=0.0.
    """
    num_bands = 6
    ia, ib = a.time_band % num_bands, b.time_band % num_bands
    # Cyclic distance
    dist = min(abs(ia - ib), num_bands - abs(ia - ib))
    if dist == 0:
        return 1.0
    if dist == 1:
        return 0.6
    if dist == 2:
        return 0.2
    return 0.0


def sim_system_load(a: ExecutionContext, b: ExecutionContext) -> float:
    """System load similarity (Gaussian on normalised load/core).

    exp(-diff² / 0.5): same=1.0, ±0.5=~0.61, ±1.0=~0.14.
    Returns 1.0 if either context has system_load unavailable
    (the weight will be 0.0 in that environment, so this value is unused).
    """
    if a.system_load_per_core is None or b.system_load_per_core is None:
        return 1.0   # dimension excluded via zero weight
    diff = a.system_load_per_core - b.system_load_per_core
    return math.exp(-(diff ** 2) / 0.5)


def sim_network_quality(a: ExecutionContext, b: ExecutionContext) -> float:
    """Network quality similarity (ordered category match)."""
    return _category_sim(
        a.network_quality, b.network_quality,
        NETWORK_QUALITY_CATEGORIES,
        same=1.0, adjacent=0.5, two_apart=0.2, opposite=0.0,
    )


# ── Composite similarity score ────────────────────────────────────────

@dataclass(frozen=True)
class DimensionBreakdown:
    """Per-dimension similarity values for one pair of contexts."""
    source_load: float
    concurrent_extractions: float
    time_band: float
    system_load: float
    network_quality: float
    composite: float


def similarity_score(
    current: ExecutionContext,
    historical: ExecutionContext,
) -> tuple[float, DimensionBreakdown]:
    """Compute composite similarity score between two contexts.

    Uses effective_weights from the CURRENT context (the environment
    doing the comparison). Both contexts must be v1-schema compatible.

    Returns:
        (score, breakdown) — score ∈ [0, 1], breakdown for CLI display.
    """
    w = current.effective_weights

    sl  = sim_source_load(current, historical)
    ce  = sim_concurrent_extractions(current, historical)
    tb  = sim_time_band(current, historical)
    sys = sim_system_load(current, historical)
    nq  = sim_network_quality(current, historical)

    # System load weight is 0.0 on platforms without getloadavg
    w_sl  = w.get("source_load", 0.30)
    w_ce  = w.get("concurrent_extractions", 0.25)
    w_tb  = w.get("time_band", 0.20)
    w_sys = w.get("system_load", 0.15) if current.system_load_available else 0.0
    w_nq  = w.get("network_quality", 0.10)

    composite = (
        w_sl  * sl  +
        w_ce  * ce  +
        w_tb  * tb  +
        w_sys * sys +
        w_nq  * nq
    )

    breakdown = DimensionBreakdown(
        source_load=round(sl, 3),
        concurrent_extractions=round(ce, 3),
        time_band=round(tb, 3),
        system_load=round(sys, 3),
        network_quality=round(nq, 3),
        composite=round(composite, 4),
    )
    return round(composite, 4), breakdown


# ── Row-growth guard ──────────────────────────────────────────────────

def row_growth_guard(
    current_rows: int,
    historical_rows: int,
) -> bool:
    """True if the row count has changed enough to exclude the historical run.

    Uses symmetric formula: abs(a-b) / max(a, b) > 0.50.
    """
    if max(current_rows, historical_rows) == 0:
        return False
    growth = abs(current_rows - historical_rows) / max(current_rows, historical_rows)
    return growth > ROW_GROWTH_GUARD_THRESHOLD


# ── Candidate scoring ─────────────────────────────────────────────────

@dataclass(frozen=True)
class ScoredRun:
    """A historical run with its similarity score and context breakdown."""
    run_id: str
    score: float
    throughput: float
    breakdown: DimensionBreakdown
    context: ExecutionContext
    excluded: bool
    exclusion_reason: str   # "" if not excluded


def score_candidates(
    current: ExecutionContext,
    candidates: Sequence[dict],
) -> tuple[list[ScoredRun], list[ScoredRun]]:
    """Score all candidate runs against the current context.

    Args:
        current:    Current execution context.
        candidates: List of dicts with keys:
                    run_id, avg_throughput, execution_context_json,
                    total_rows (optional — for row growth guard).
                    Source: state_store.get_runs_with_context().

    Returns:
        (matched, excluded) — matched are runs above SCORE_EXCLUSION_THRESHOLD,
        sorted descending by score. Benchmark-tagged runs are always excluded.
    """
    matched: list[ScoredRun] = []
    excluded: list[ScoredRun] = []

    for run in candidates[:MAX_CANDIDATE_RUNS]:
        run_id = run.get("run_id", "")
        throughput = run.get("avg_throughput", 0.0)

        # Parse context
        ctx_raw = run.get("execution_context_json", "{}")
        try:
            ctx_dict = ctx_raw if isinstance(ctx_raw, dict) else __import__("json").loads(ctx_raw)
        except Exception:
            ctx_dict = {}

        # Skip runs with no context recorded (pre-Phase 2B runs)
        if not ctx_dict or ctx_dict.get("schema_version") is None:
            excluded.append(ScoredRun(
                run_id=run_id, score=0.0, throughput=throughput,
                breakdown=_zero_breakdown(), context=_null_context(),
                excluded=True, exclusion_reason="no_context_recorded",
            ))
            continue

        # Exclude benchmark-tagged runs
        if ctx_dict.get("source") == "benchmark":
            excluded.append(ScoredRun(
                run_id=run_id, score=0.0, throughput=throughput,
                breakdown=_zero_breakdown(), context=_null_context(),
                excluded=True, exclusion_reason="benchmark_run",
            ))
            continue

        try:
            hist_ctx = ExecutionContext.from_dict(ctx_dict)
        except Exception:
            excluded.append(ScoredRun(
                run_id=run_id, score=0.0, throughput=throughput,
                breakdown=_zero_breakdown(), context=_null_context(),
                excluded=True, exclusion_reason="invalid_context",
            ))
            continue

        # Row growth guard (symmetric)
        hist_rows = hist_ctx.row_estimate
        if row_growth_guard(current.row_estimate, hist_rows):
            score, breakdown = similarity_score(current, hist_ctx)
            excluded.append(ScoredRun(
                run_id=run_id, score=score, throughput=throughput,
                breakdown=breakdown, context=hist_ctx,
                excluded=True, exclusion_reason="row_growth",
            ))
            continue

        # Score the run
        score, breakdown = similarity_score(current, hist_ctx)

        if score < SCORE_EXCLUSION_THRESHOLD:
            excluded.append(ScoredRun(
                run_id=run_id, score=score, throughput=throughput,
                breakdown=breakdown, context=hist_ctx,
                excluded=True,
                exclusion_reason=f"score_too_low ({score:.2f} < {SCORE_EXCLUSION_THRESHOLD})",
            ))
        else:
            matched.append(ScoredRun(
                run_id=run_id, score=score, throughput=throughput,
                breakdown=breakdown, context=hist_ctx,
                excluded=False, exclusion_reason="",
            ))

    matched.sort(key=lambda r: r.score, reverse=True)
    return matched, excluded


# ── Helpers ───────────────────────────────────────────────────────────

def _zero_breakdown() -> DimensionBreakdown:
    return DimensionBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def _null_context() -> ExecutionContext:
    from ixtract.context import _compute_effective_weights
    return ExecutionContext(
        schema_version=0,
        source_load="normal", concurrent_extractions=0, time_band=0,
        system_load_per_core=None, network_quality="good",
        row_estimate=0,
        effective_weights=_compute_effective_weights(False),
        system_load_available=False,
    )
