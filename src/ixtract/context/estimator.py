"""Context-Weighted Throughput Estimator.

Computes a throughput estimate from historically similar runs.
Falls back to EWMA when no strong contextual matches are available.

This module owns:
  - Weighted throughput calculation
  - EWMA fallback computation
  - Confidence assessment
  - Clamping to historical bounds

It does NOT own similarity scoring (context/similarity.py) or
context measurement (context/__init__.py).

Hard switch policy (Phase 2B):
  ≥1 matched run with score ≥ 0.50  → weighted estimate (primary)
  1  matched run with score ≥ 0.65  → weighted estimate (low confidence)
  1  matched run with score  < 0.65  → EWMA fallback
  no matched runs with score ≥ 0.50  → EWMA fallback

Blend (weighted + EWMA) is deferred to Phase 3.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

from ixtract.context.similarity import (
    ScoredRun,
    ConfidenceAssessment,
    ConfidenceReason,
    REASON_DISPLAY_ORDER,
    SCORE_STRONG_MATCH_THRESHOLD,
    SCORE_SINGLE_MATCH_THRESHOLD,
    DOMINANCE_THRESHOLD,
    THROUGHPUT_CV_THRESHOLD,
)


# ── Clamping bounds ───────────────────────────────────────────────────
CLAMP_LOWER_FACTOR = 0.75   # hist_min × 0.75
CLAMP_UPPER_FACTOR = 1.25   # hist_max × 1.25


# ── EWMA configuration ────────────────────────────────────────────────
EWMA_ALPHA = 0.3   # weight on most recent run: 0.3, previous: 0.7


@dataclass(frozen=True)
class ThroughputEstimate:
    """Complete throughput estimate with confidence and explanation."""
    value: float                          # rows/sec
    confidence: ConfidenceAssessment
    method: str                           # "context_weighted" | "ewma" | "cold_start"
    matched_runs: tuple[ScoredRun, ...]   # empty if EWMA/cold_start
    excluded_runs: tuple[ScoredRun, ...]
    clamped: bool                         # True if clamp was applied
    clamp_bounds: Optional[tuple[float, float]]  # (lower, upper) if clamped


def estimate_throughput(
    matched: list[ScoredRun],
    excluded: list[ScoredRun],
    all_historical_throughputs: list[float],
    fallback_throughput: float,
) -> ThroughputEstimate:
    """Compute a throughput estimate from context-scored runs.

    Args:
        matched:                     Runs above exclusion threshold, sorted desc by score.
        excluded:                    Runs below exclusion threshold (for CLI display).
        all_historical_throughputs:  All recent throughputs (unfiltered) for EWMA + clamping.
        fallback_throughput:         Used when no history at all (cold start).

    Returns:
        ThroughputEstimate with value, confidence, method, and full explainability data.
    """
    hist_min = min(all_historical_throughputs) if all_historical_throughputs else 0.0
    hist_max = max(all_historical_throughputs) if all_historical_throughputs else float("inf")

    # Single-match must be evaluated before has_strong — a single run at 0.55
    # meets has_strong (≥0.50) but does not meet the single-match threshold (≥0.65).
    single_match = len(matched) == 1
    if single_match:
        single_strong = matched[0].score >= SCORE_SINGLE_MATCH_THRESHOLD
        use_weighted = single_strong
    else:
        has_strong = any(r.score >= SCORE_STRONG_MATCH_THRESHOLD for r in matched)
        use_weighted = has_strong

    if use_weighted:
        return _weighted_estimate(
            matched, excluded,
            hist_min, hist_max,
            all_historical_throughputs,
        )
    else:
        return _ewma_estimate(
            matched, excluded,
            all_historical_throughputs,
            fallback_throughput,
            hist_min, hist_max,
        )


def _weighted_estimate(
    matched: list[ScoredRun],
    excluded: list[ScoredRun],
    hist_min: float,
    hist_max: float,
    all_throughputs: list[float],
) -> ThroughputEstimate:
    """Compute weighted average throughput from matched runs."""
    total_weight = sum(r.score for r in matched)

    if total_weight == 0:
        return _ewma_estimate(matched, excluded, all_throughputs, 0.0, hist_min, hist_max)

    weighted_tp = sum(r.score * r.throughput for r in matched) / total_weight

    # Dominance check
    max_weight = max(r.score / total_weight for r in matched)
    dominant = max_weight > DOMINANCE_THRESHOLD
    dominant_weight = max_weight if dominant else None

    # Throughput CV (undefined for single-match)
    tp_cv: Optional[float] = None
    if len(matched) > 1:
        tps = [r.throughput for r in matched]
        mean_tp = sum(tps) / len(tps)
        if mean_tp > 0:
            variance = sum((t - mean_tp) ** 2 for t in tps) / len(tps)
            tp_cv = math.sqrt(variance) / mean_tp

    # Clamping
    lower_bound = hist_min * CLAMP_LOWER_FACTOR if hist_min > 0 else 0.0
    upper_bound = hist_max * CLAMP_UPPER_FACTOR if hist_max < float("inf") else float("inf")
    clamped_tp = max(lower_bound, min(upper_bound, weighted_tp))
    clamped = abs(clamped_tp - weighted_tp) > 1.0

    # Confidence
    confidence = _assess_confidence(
        matched=matched,
        tp_cv=tp_cv,
        dominant=dominant,
        dominant_weight=dominant_weight,
        is_ewma=False,
    )

    return ThroughputEstimate(
        value=round(clamped_tp, 1),
        confidence=confidence,
        method="context_weighted",
        matched_runs=tuple(matched),
        excluded_runs=tuple(excluded),
        clamped=clamped,
        clamp_bounds=(lower_bound, upper_bound) if clamped else None,
    )


def _ewma_estimate(
    matched: list[ScoredRun],
    excluded: list[ScoredRun],
    all_throughputs: list[float],
    fallback_throughput: float,
    hist_min: float,
    hist_max: float,
) -> ThroughputEstimate:
    """EWMA fallback — uses unweighted recent history."""
    if not all_throughputs:
        # Cold start — no history at all
        value = fallback_throughput
        confidence = ConfidenceAssessment(
            level="low",
            reasons=(ConfidenceReason.FALLBACK_USED, ConfidenceReason.SPARSE_EVIDENCE),
            matched_run_count=0,
            max_similarity_score=0.0,
            throughput_cv=None,
            dominant_weight=None,
        )
        return ThroughputEstimate(
            value=value, confidence=confidence,
            method="cold_start",
            matched_runs=(), excluded_runs=tuple(excluded),
            clamped=False, clamp_bounds=None,
        )

    # EWMA over all_throughputs (ordered oldest-first)
    ewma = all_throughputs[0]
    for tp in all_throughputs[1:]:
        ewma = EWMA_ALPHA * tp + (1 - EWMA_ALPHA) * ewma

    # Clamp EWMA too
    lower_bound = hist_min * CLAMP_LOWER_FACTOR
    upper_bound = hist_max * CLAMP_UPPER_FACTOR
    clamped_ewma = max(lower_bound, min(upper_bound, ewma))
    clamped = abs(clamped_ewma - ewma) > 1.0

    # Confidence: always low when using EWMA
    reasons: list[ConfidenceReason] = [ConfidenceReason.FALLBACK_USED]
    if not matched:
        reasons.append(ConfidenceReason.SPARSE_EVIDENCE)
    elif matched[0].score < SCORE_SINGLE_MATCH_THRESHOLD:
        reasons.append(ConfidenceReason.WEAK_MATCH)

    max_similarity_score = matched[0].score if matched else 0.0
    confidence = ConfidenceAssessment(
        level="low",
        reasons=_sort_reasons(reasons),
        matched_run_count=len(matched),
        max_similarity_score=max_similarity_score,
        throughput_cv=None,
        dominant_weight=None,
    )

    return ThroughputEstimate(
        value=round(clamped_ewma, 1),
        confidence=confidence,
        method="ewma",
        matched_runs=tuple(matched),
        excluded_runs=tuple(excluded),
        clamped=clamped,
        clamp_bounds=(lower_bound, upper_bound) if clamped else None,
    )


def _assess_confidence(
    matched: list[ScoredRun],
    tp_cv: Optional[float],
    dominant: bool,
    dominant_weight: Optional[float],
    is_ewma: bool,
) -> ConfidenceAssessment:
    """Compute confidence level and structured reasons.

    Downgrade triggers (each reduces level by 1, floor at "low"):
      - throughput CV > 0.20 among matched runs
      - dominance > 70% of total weight

    Single-match: always starts at "low" regardless of triggers.
    """
    # INVARIANT: reasons annotate the confidence level — they do not determine it.
    # Downgrade triggers (HIGH_VARIANCE, DOMINANT_MATCH) reduce level by one step.
    # WEAK_MATCH and SPARSE_EVIDENCE are annotations only: they describe *why* the
    # level is what it is, not a mechanism to change it. Never add logic of the form
    # "if WEAK_MATCH in reasons: downgrade level" — that would incorrectly push
    # valid medium cases (e.g. 2 matches at 0.52, 0.51) down to low.
    reasons: list[ConfidenceReason] = []
    n = len(matched)
    max_similarity_score = matched[0].score if matched else 0.0

    # Base level from match count and score
    if n >= 3 and max_similarity_score >= 0.70:
        base_level = "high"
    elif n >= 2 and max_similarity_score >= SCORE_STRONG_MATCH_THRESHOLD:
        base_level = "medium"
    else:
        base_level = "low"
        if n == 1:
            reasons.append(ConfidenceReason.SPARSE_EVIDENCE)

    if max_similarity_score < SCORE_SINGLE_MATCH_THRESHOLD and n >= 1:
        reasons.append(ConfidenceReason.WEAK_MATCH)

    # Downgrade triggers
    level_index = ["high", "medium", "low"].index(base_level)

    if tp_cv is not None and tp_cv > THROUGHPUT_CV_THRESHOLD:
        reasons.append(ConfidenceReason.HIGH_VARIANCE)
        level_index = min(level_index + 1, 2)

    if dominant:
        reasons.append(ConfidenceReason.DOMINANT_MATCH)
        level_index = min(level_index + 1, 2)

    final_level = ["high", "medium", "low"][level_index]

    return ConfidenceAssessment(
        level=final_level,
        reasons=_sort_reasons(reasons),
        matched_run_count=n,
        max_similarity_score=round(max_similarity_score, 3),
        throughput_cv=round(tp_cv, 3) if tp_cv is not None else None,
        dominant_weight=round(dominant_weight, 3) if dominant_weight is not None else None,
    )


def _sort_reasons(reasons: list[ConfidenceReason]) -> tuple[ConfidenceReason, ...]:
    """Sort reasons by display priority (REASON_DISPLAY_ORDER)."""
    order = {r: i for i, r in enumerate(REASON_DISPLAY_ORDER)}
    return tuple(sorted(set(reasons), key=lambda r: order.get(r, 99)))


# ── EWMA utility (used by planner for simple baseline too) ───────────

def compute_ewma(throughputs: list[float], alpha: float = EWMA_ALPHA) -> float:
    """Simple EWMA over a list of throughputs (oldest-first).

    Returns 0.0 for empty input.
    """
    if not throughputs:
        return 0.0
    ewma = throughputs[0]
    for tp in throughputs[1:]:
        ewma = alpha * tp + (1 - alpha) * ewma
    return ewma


# ── CLI formatting ────────────────────────────────────────────────────

def format_estimate_for_cli(estimate: ThroughputEstimate) -> str:
    """Format ThroughputEstimate for plan --standard output."""
    lines = []
    conf = estimate.confidence

    # Method + value
    method_label = {
        "context_weighted": "context-weighted average",
        "ewma": "EWMA fallback",
        "cold_start": "cold-start baseline",
    }.get(estimate.method, estimate.method)

    lines.append(
        f"  Estimate:  {estimate.value:,.0f} rows/sec"
        f"  ({method_label}, {conf.format_cli()})"
    )

    if estimate.clamped and estimate.clamp_bounds:
        lo, hi = estimate.clamp_bounds
        lines.append(
            f"  Clamped:   yes (historical bounds "
            f"{lo:,.0f}–{hi:,.0f} rows/sec)"
        )

    if not estimate.matched_runs:
        return "\n".join(lines)

    # Matched runs table
    lines.append(f"\n  Matched runs ({len(estimate.matched_runs)} of "
                 f"{len(estimate.matched_runs) + len(estimate.excluded_runs)} candidates):")
    lines.append(
        f"    {'Run':<10}  {'Score':>6}  {'Throughput':>14}  "
        f"{'Source load':<12}  {'Time band':<10}  {'Concurrency'}"
    )
    lines.append(
        f"    {'───':<10}  {'─────':>6}  {'──────────':>14}  "
        f"{'───────────':<12}  {'─────────':<10}  {'───────────'}"
    )
    for r in estimate.matched_runs:
        bd = r.breakdown
        marker = " ◀" if r == estimate.matched_runs[0] else ""
        lines.append(
            f"    {r.run_id[:8]:<10}  {r.score:>6.2f}  "
            f"{r.throughput:>13,.0f}/s  "
            f"{r.context.source_load:<12}  "
            f"band {r.context.time_band:<5}  "
            f"{r.context.concurrent_extractions}{marker}"
        )

    if conf.dominant_weight:
        lines.append(
            f"\n  Dominance: {conf.dominant_weight:.0%} "
            f"(one run carries >70% weight)"
        )
    if conf.throughput_cv:
        lines.append(
            f"  Variance:  CV={conf.throughput_cv:.2f} "
            f"({'high' if conf.throughput_cv > THROUGHPUT_CV_THRESHOLD else 'acceptable'})"
        )

    # Exclusions (show first 3, summarise rest)
    if estimate.excluded_runs:
        exc_show = estimate.excluded_runs[:3]
        exc_hidden = len(estimate.excluded_runs) - len(exc_show)
        lines.append(f"\n  Excluded runs ({len(estimate.excluded_runs)}):")
        for r in exc_show:
            lines.append(f"    {r.run_id[:8]:<10}  {r.exclusion_reason}")
        if exc_hidden > 0:
            lines.append(f"    ... and {exc_hidden} more")

    return "\n".join(lines)
