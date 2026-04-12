"""Deviation Analyzer — structured diagnosis of extraction performance.

Classifies every run deviation into an explicit category with reasoning.
Same inputs → same diagnosis. No probabilistic logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DeviationCategory(str, Enum):
    # Phase 1
    UNDER_PARALLEL = "under_parallel"
    OVER_PARALLEL = "over_parallel"
    STABLE = "stable"
    # Phase 2
    SOURCE_LATENCY = "source_latency"
    DATA_SKEW = "data_skew"
    MODEL_ERROR = "model_error"
    # Phase 3
    ANOMALY = "anomaly"


@dataclass(frozen=True)
class RunMetrics:
    """Aggregate metrics from a completed run."""
    total_rows: int
    total_bytes: int
    duration_seconds: float
    worker_count: int
    avg_throughput_rows_sec: float
    chunk_durations: tuple[float, ...]
    worker_idle_pcts: tuple[float, ...]
    source_query_ms_avg: float = 0.0
    predicted_duration_seconds: float = 0.0
    predicted_throughput_rows_sec: float = 0.0
    previous_throughput_rows_sec: float = 0.0
    previous_worker_count: int = 0
    worker_count_changed: bool = False


@dataclass(frozen=True)
class DeviationDiagnosis:
    """Structured diagnosis output."""
    category: DeviationCategory
    deviation_ratio: float
    throughput_change_pct: float
    reasoning: str
    corrective_action: str
    chunk_variance: float = 0.0
    confidence: float = 1.0


class DeviationAnalyzer:
    """Rule-based deviation classifier (Phase 1 categories)."""

    def __init__(self, noise_threshold: float = 0.05) -> None:
        self.noise_threshold = noise_threshold

    def diagnose(self, metrics: RunMetrics) -> DeviationDiagnosis:
        dev = self._dev_ratio(metrics)
        tpct = self._tp_change(metrics)
        cv = self._chunk_cv(metrics.chunk_durations)

        if metrics.worker_count_changed:
            workers_increased = metrics.worker_count > metrics.previous_worker_count
            workers_decreased = metrics.worker_count < metrics.previous_worker_count
            improved = tpct > self.noise_threshold
            degraded = tpct < -self.noise_threshold

            if workers_increased and improved:
                # More workers helped → room to grow
                return DeviationDiagnosis(
                    DeviationCategory.UNDER_PARALLEL, dev, tpct,
                    f"Throughput {tpct:+.1%} after workers "
                    f"{metrics.previous_worker_count} \u2192 {metrics.worker_count}.",
                    "Controller will consider increasing workers.",
                    cv,
                )
            if workers_increased and degraded:
                # More workers hurt → too many
                return DeviationDiagnosis(
                    DeviationCategory.OVER_PARALLEL, dev, tpct,
                    f"Throughput {tpct:+.1%} after workers "
                    f"{metrics.previous_worker_count} \u2192 {metrics.worker_count}.",
                    "Controller will consider decreasing workers.",
                    cv,
                )
            if workers_decreased and improved:
                # Fewer workers helped → previous level was too high
                return DeviationDiagnosis(
                    DeviationCategory.OVER_PARALLEL, dev, tpct,
                    f"Throughput {tpct:+.1%} after reducing workers "
                    f"{metrics.previous_worker_count} \u2192 {metrics.worker_count}.",
                    "Confirmed over-parallelized at previous level.",
                    cv,
                )
            if workers_decreased and degraded:
                # Fewer workers hurt → previous level was right, now too few
                return DeviationDiagnosis(
                    DeviationCategory.UNDER_PARALLEL, dev, tpct,
                    f"Throughput {tpct:+.1%} after reducing workers "
                    f"{metrics.previous_worker_count} \u2192 {metrics.worker_count}.",
                    "Controller will consider increasing workers.",
                    cv,
                )

        return DeviationDiagnosis(
            DeviationCategory.STABLE, dev, tpct,
            f"Throughput change {tpct:+.1%} within noise band. "
            f"Stable at {metrics.worker_count} workers.",
            "No action. Updating baseline.",
            cv,
        )

    @staticmethod
    def _dev_ratio(m: RunMetrics) -> float:
        return m.duration_seconds / m.predicted_duration_seconds if m.predicted_duration_seconds > 0 else 1.0

    @staticmethod
    def _tp_change(m: RunMetrics) -> float:
        if m.previous_throughput_rows_sec <= 0:
            return 0.0
        return (m.avg_throughput_rows_sec - m.previous_throughput_rows_sec) / m.previous_throughput_rows_sec

    @staticmethod
    def _chunk_cv(durations: tuple[float, ...]) -> float:
        n = len(durations)
        if n < 2:
            return 0.0
        mean = sum(durations) / n
        if mean <= 0:
            return 0.0
        var = sum((d - mean) ** 2 for d in durations) / n
        return (var ** 0.5) / mean


# ── Anomaly Detection (Phase 4A) ─────────────────────────────────────

ANOMALY_BASELINE_WINDOW = 20     # last N successful runs for baseline
ANOMALY_MIN_BASELINE = 5         # minimum runs required for anomaly detection
ANOMALY_Z_THRESHOLD = 2.0        # standard deviations for anomaly
ANOMALY_ZERO_STDDEV_RATIO = 0.20 # 20% threshold when σ ≈ 0


@dataclass(frozen=True)
class AnomalyResult:
    """Result of anomaly detection for a single run.

    Anomaly detection is observational only — it never changes
    controller behavior or plan decisions.
    """
    is_anomaly: bool
    current_throughput: float
    baseline_mean: float
    baseline_stddev: float
    z_score: float                # how many σ from mean (absolute)
    direction: str                # "degradation" | "improvement" | "none"
    baseline_run_count: int
    message: str                  # human-readable summary


def detect_anomaly(
    current_throughput: float,
    baseline_throughputs: list[float],
) -> AnomalyResult:
    """Detect throughput anomaly against a broad baseline.

    Baseline is same source + same table, last N successful runs.
    No context filter — anomaly detection catches unusual events,
    diagnosis explains why.

    Args:
        current_throughput:     Throughput of the current run.
        baseline_throughputs:   Throughputs from recent successful runs (oldest-first).

    Returns:
        AnomalyResult with detection status, z-score, and direction.
    """
    current_throughput = float(current_throughput)
    n = len(baseline_throughputs)

    # Insufficient baseline — cannot detect
    if n < ANOMALY_MIN_BASELINE:
        return AnomalyResult(
            is_anomaly=False,
            current_throughput=current_throughput,
            baseline_mean=0.0,
            baseline_stddev=0.0,
            z_score=0.0,
            direction="none",
            baseline_run_count=n,
            message=f"Insufficient baseline ({n}/{ANOMALY_MIN_BASELINE} runs)",
        )

    mean = sum(baseline_throughputs) / n
    variance = sum((t - mean) ** 2 for t in baseline_throughputs) / n
    stddev = variance ** 0.5

    # σ ≈ 0 guard — fall back to percentage threshold
    if stddev < 1e-6:
        if mean <= 0:
            return AnomalyResult(
                is_anomaly=False, current_throughput=current_throughput,
                baseline_mean=mean, baseline_stddev=0.0, z_score=0.0,
                direction="none", baseline_run_count=n,
                message="Baseline mean is zero — cannot assess",
            )
        ratio = abs(current_throughput - mean) / mean
        is_anomaly = ratio > ANOMALY_ZERO_STDDEV_RATIO
        if is_anomaly:
            direction = "degradation" if current_throughput < mean else "improvement"
        else:
            direction = "none"
        return AnomalyResult(
            is_anomaly=is_anomaly,
            current_throughput=current_throughput,
            baseline_mean=round(mean, 1),
            baseline_stddev=0.0,
            z_score=round(ratio / ANOMALY_ZERO_STDDEV_RATIO, 2) if is_anomaly else 0.0,
            direction=direction,
            baseline_run_count=n,
            message=(
                f"Throughput {current_throughput:,.0f} rows/sec "
                f"deviates {ratio:.0%} from baseline {mean:,.0f} "
                f"(σ≈0, using {ANOMALY_ZERO_STDDEV_RATIO:.0%} threshold)"
            ) if is_anomaly else "Within normal range",
        )

    # Standard z-score detection
    z_score = abs(current_throughput - mean) / stddev

    if z_score > ANOMALY_Z_THRESHOLD:
        direction = "degradation" if current_throughput < mean else "improvement"
        message = (
            f"Throughput {current_throughput:,.0f} rows/sec is "
            f"{z_score:.1f}σ {'below' if direction == 'degradation' else 'above'} "
            f"baseline ({mean:,.0f} ± {stddev:,.0f})"
        )
    else:
        direction = "none"
        message = "Within normal range"

    return AnomalyResult(
        is_anomaly=z_score > ANOMALY_Z_THRESHOLD,
        current_throughput=current_throughput,
        baseline_mean=round(mean, 1),
        baseline_stddev=round(stddev, 1),
        z_score=round(z_score, 2),
        direction=direction,
        baseline_run_count=n,
        message=message,
    )
