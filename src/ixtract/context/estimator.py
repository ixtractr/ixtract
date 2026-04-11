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


# ── Blend + Time-decay (Phase 3B) ─────────────────────────────────────
TIME_DECAY_LAMBDA = 0.05    # half-life ≈ 14 days
WEIGHT_EPSILON = 1e-6       # below this, total_weight treated as "no effective matches"
DEPTH_RAMP_RUNS = 10        # depth_factor reaches 1.0 at this many runs with context


@dataclass(frozen=True)
class ThroughputEstimate:
    """Complete throughput estimate with confidence and explanation."""
    value: float                          # rows/sec
    confidence: ConfidenceAssessment
    method: str                           # "blended" | "ewma" | "cold_start"
    matched_runs: tuple[ScoredRun, ...]   # empty if EWMA/cold_start
    excluded_runs: tuple[ScoredRun, ...]
    clamped: bool                         # True if clamp was applied
    clamp_bounds: Optional[tuple[float, float]]  # (lower, upper) if clamped
    # Phase 3B blend fields (explainability only)
    blend_weight: Optional[float] = None         # α ∈ [0,1], None if not blended
    time_decay_lambda: Optional[float] = None    # λ, None if not blended


def estimate_throughput(
    matched: list[ScoredRun],
    excluded: list[ScoredRun],
    all_historical_throughputs: list[float],
    fallback_throughput: float,
    runs_with_context: int = 0,
    run_ages: Optional[dict[str, float]] = None,
) -> ThroughputEstimate:
    """Compute a throughput estimate using continuous blend (Phase 3B).

    Replaces the Phase 2B hard switch with a weighted blend:
        estimate = α × context_weighted + (1 - α) × EWMA
    where α = match_quality × depth_factor, clamped to [0, 1].

    Time-decay is applied to context weights: weight = score × exp(-λ × age_days).

    Args:
        matched:                     Runs above exclusion threshold, sorted desc by score.
        excluded:                    Runs below exclusion threshold (for CLI display).
        all_historical_throughputs:  All recent throughputs (unfiltered) for EWMA + clamping.
        fallback_throughput:         Used when no history at all (cold start).
        runs_with_context:           Count of runs with stored ExecutionContext.
        run_ages:                    Dict of run_id → age in days (for time-decay).

    Returns:
        ThroughputEstimate with value, confidence, method, and full explainability data.
    """
    if run_ages is None:
        run_ages = {}

    hist_min = min(all_historical_throughputs) if all_historical_throughputs else 0.0
    hist_max = max(all_historical_throughputs) if all_historical_throughputs else float("inf")

    # ── Cold start (no history at all) ────────────────────────────
    if not all_historical_throughputs:
        confidence = ConfidenceAssessment(
            level="low",
            reasons=(ConfidenceReason.FALLBACK_USED, ConfidenceReason.SPARSE_EVIDENCE),
            matched_run_count=0,
            max_similarity_score=0.0,
            throughput_cv=None,
            dominant_weight=None,
        )
        return ThroughputEstimate(
            value=fallback_throughput, confidence=confidence,
            method="cold_start",
            matched_runs=(), excluded_runs=tuple(excluded),
            clamped=False, clamp_bounds=None,
        )

    # ── EWMA (always computed — used as blend component or standalone) ─
    ewma_tp = _compute_ewma(all_historical_throughputs)

    # ── Context-weighted with time-decay ──────────────────────────
    if not matched:
        # No matched runs → pure EWMA
        return _make_ewma_result(
            ewma_tp, matched, excluded, all_historical_throughputs,
            hist_min, hist_max,
        )

    # Compute time-decayed weights
    decayed_weights = []
    for r in matched:
        age_days = run_ages.get(r.run_id, 30.0)  # default 30 days if missing
        decay = math.exp(-TIME_DECAY_LAMBDA * age_days)
        decayed_weights.append(r.score * decay)

    total_weight = sum(decayed_weights)

    if total_weight < WEIGHT_EPSILON:
        # Matched runs exist but carry no meaningful weight (all very old)
        return _make_ewma_result(
            ewma_tp, matched, excluded, all_historical_throughputs,
            hist_min, hist_max,
        )

    # Context-weighted throughput (time-decayed)
    context_tp = sum(
        w * r.throughput for w, r in zip(decayed_weights, matched)
    ) / total_weight

    # ── Blend ─────────────────────────────────────────────────────
    match_quality = max(r.score for r in matched)
    depth_factor = min(1.0, runs_with_context / DEPTH_RAMP_RUNS)
    blend_weight = max(0.0, min(1.0, match_quality * depth_factor))

    blended_tp = blend_weight * context_tp + (1 - blend_weight) * ewma_tp

    # ── Clamp ─────────────────────────────────────────────────────
    lower_bound = hist_min * CLAMP_LOWER_FACTOR if hist_min > 0 else 0.0
    upper_bound = hist_max * CLAMP_UPPER_FACTOR if hist_max < float("inf") else float("inf")
    clamped_tp = max(lower_bound, min(upper_bound, blended_tp))
    clamped = abs(clamped_tp - blended_tp) > 1.0

    # ── Confidence (uses original scores, not decayed — decay is temporal,
    #    not a quality signal) ─────────────────────────────────────
    # Dominance check (on decayed weights)
    max_w = max(decayed_weights)
    dominant = (max_w / total_weight) > DOMINANCE_THRESHOLD if total_weight > 0 else False
    dominant_weight = (max_w / total_weight) if dominant else None

    # Throughput CV
    tp_cv: Optional[float] = None
    if len(matched) > 1:
        tps = [r.throughput for r in matched]
        mean_tp = sum(tps) / len(tps)
        if mean_tp > 0:
            variance = sum((t - mean_tp) ** 2 for t in tps) / len(tps)
            tp_cv = math.sqrt(variance) / mean_tp

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
        method="blended",
        matched_runs=tuple(matched),
        excluded_runs=tuple(excluded),
        clamped=clamped,
        clamp_bounds=(lower_bound, upper_bound) if clamped else None,
        blend_weight=round(blend_weight, 4),
        time_decay_lambda=TIME_DECAY_LAMBDA,
    )


def _make_ewma_result(
    ewma_tp: float,
    matched: list[ScoredRun],
    excluded: list[ScoredRun],
    all_throughputs: list[float],
    hist_min: float,
    hist_max: float,
) -> ThroughputEstimate:
    """Build an EWMA-only ThroughputEstimate (used when blend falls back to EWMA)."""
    lower_bound = hist_min * CLAMP_LOWER_FACTOR if hist_min > 0 else 0.0
    upper_bound = hist_max * CLAMP_UPPER_FACTOR if hist_max < float("inf") else float("inf")
    clamped_ewma = max(lower_bound, min(upper_bound, ewma_tp))
    clamped = abs(clamped_ewma - ewma_tp) > 1.0

    reasons: list[ConfidenceReason] = [ConfidenceReason.FALLBACK_USED]
    if not matched:
        reasons.append(ConfidenceReason.SPARSE_EVIDENCE)
    elif matched[0].score < SCORE_SINGLE_MATCH_THRESHOLD:
        reasons.append(ConfidenceReason.WEAK_MATCH)

    max_sim = matched[0].score if matched else 0.0
    confidence = ConfidenceAssessment(
        level="low",
        reasons=_sort_reasons(reasons),
        matched_run_count=len(matched),
        max_similarity_score=max_sim,
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


def _compute_ewma(throughputs: list[float], alpha: float = EWMA_ALPHA) -> float:
    """EWMA over throughputs (oldest-first). Returns 0.0 for empty input."""
    if not throughputs:
        return 0.0
    ewma = throughputs[0]
    for tp in throughputs[1:]:
        ewma = alpha * tp + (1 - alpha) * ewma
    return ewma


def _ewma_estimate(
    matched: list[ScoredRun],
    excluded: list[ScoredRun],
    all_throughputs: list[float],
    fallback_throughput: float,
    hist_min: float,
    hist_max: float,
) -> ThroughputEstimate:
    """EWMA fallback — uses unweighted recent history.

    Kept for backward compatibility with planner's no-context path.
    """
    if not all_throughputs:
        confidence = ConfidenceAssessment(
            level="low",
            reasons=(ConfidenceReason.FALLBACK_USED, ConfidenceReason.SPARSE_EVIDENCE),
            matched_run_count=0,
            max_similarity_score=0.0,
            throughput_cv=None,
            dominant_weight=None,
        )
        return ThroughputEstimate(
            value=fallback_throughput, confidence=confidence,
            method="cold_start",
            matched_runs=(), excluded_runs=tuple(excluded),
            clamped=False, clamp_bounds=None,
        )

    ewma_tp = _compute_ewma(all_throughputs)
    return _make_ewma_result(
        ewma_tp, matched, excluded, all_throughputs, hist_min, hist_max,
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
    Public API — delegates to _compute_ewma.
    """
    return _compute_ewma(throughputs, alpha)


# ── CLI formatting ────────────────────────────────────────────────────

def format_estimate_for_cli(estimate: ThroughputEstimate) -> str:
    """Format ThroughputEstimate for plan --standard output."""
    lines = []
    conf = estimate.confidence

    # Method + value
    method_label = {
        "blended": "blended estimate",
        "ewma": "EWMA fallback",
        "cold_start": "cold-start baseline",
    }.get(estimate.method, estimate.method)

    if estimate.method == "blended" and estimate.blend_weight is not None:
        method_label += f" (α={estimate.blend_weight:.2f})"

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
