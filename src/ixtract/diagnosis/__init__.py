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
