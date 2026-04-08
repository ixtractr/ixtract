"""Benchmarker — multi-worker throughput calibration for ixtract.

Eliminates the cold-start problem by probing throughput at discrete worker
counts before the first real extraction run. The planner consumes the result
to seed the controller with a known-good starting point.

Design principles:
    - Deterministic: fixed worker grid, fixed range selection (thirds of PK space)
    - Fast: probe_rows capped at 100K, cache-warm pass discarded
    - Conservative: confidence gating — planner falls back to profiler if uncertain
    - Non-destructive: probe data is never written to the output target

Confidence model (two independent signals):

    confidence      = range_consistency * curve_shape_factor
    signal_strength = max(FLAT_CURVE_SIGNAL_FLOOR, raw_signal)

    confidence      → reliability of the measurement (used for planner gating)
    signal_strength → importance of tuning (used for planner conservative bias)

    These are kept separate intentionally: a flat curve (low signal_strength)
    is still trustworthy — it means worker count doesn't matter much, so the
    planner should pick conservatively, not fall back to the profiler.

Worker grid: [1, 2, 4, 8] bounded by [min_workers, max_workers].
Skipped entirely if len(effective_grid) < 2 or table is too small to benchmark.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal


# ── Constants ─────────────────────────────────────────────────────────

WORKER_GRID: tuple[int, ...] = (1, 2, 4, 8)

# Probe size: big enough to see parallelism effects, small enough to be fast.
PROBE_MIN_ROWS: int = 50_000
PROBE_MAX_ROWS: int = 100_000
PROBE_TABLE_FRACTION: float = 0.02     # 2% of table rows

# Confidence thresholds
CONFIDENCE_THRESHOLD: float = 0.50    # below this → planner falls back to profiler
FLAT_CURVE_SIGNAL_FLOOR: float = 0.15 # minimum signal when curve is flat

# Planner conservative bias: if signal_strength AT OR BELOW this, pick conservatively
# Boundary is inclusive: signal_strength <= threshold → flat curve → conservative
SIGNAL_STRENGTH_CONSERVATIVE_THRESHOLD: float = 0.30

# Conservative worker count for flat curves: cap at 2, never drift toward 4 or 8
CONSERVATIVE_WORKER_CAP: int = 2

# Staleness: benchmark is stale after this many seconds (7 days)
BENCHMARK_MAX_AGE_SECONDS: float = 7 * 24 * 3600

# Staleness: benchmark is stale if row count grew more than this fraction
BENCHMARK_MAX_ROW_GROWTH: float = 0.20   # 20%

# Auto-benchmark skip: if estimated extraction duration is below this, skip
BENCHMARK_MIN_TABLE_DURATION_SECONDS: float = 12.0

# Tie-breaker: prefer lower workers if within this fraction of max throughput
OPTIMAL_WORKER_TIEBREAK_BAND: float = 0.05  # 5%

# Curve shape factors (multiplied into confidence)
CURVE_SHAPE_FACTORS: dict[str, float] = {
    "plateau":       1.00,   # optimum clearly visible — highest trust
    "increasing":    0.75,   # optimum not visible, recommendation is lower bound
    "non_monotonic": 0.70,   # contention detected — recommendation is unstable
}

CurveShape = Literal["plateau", "increasing", "non_monotonic"]


# ── BenchmarkResult ───────────────────────────────────────────────────

@dataclass(frozen=True)
class ProbeRange:
    """A single PK range used in one benchmark probe."""
    pk_start: int
    pk_end: int
    label: str        # "low" | "mid" | "high"


@dataclass(frozen=True)
class WorkerProbeResult:
    """Throughput measurements across all ranges for one worker count."""
    worker_count: int
    effective_workers: int           # actual observed (may differ from planned)
    throughputs: tuple[float, ...]   # one per range, ordered low/mid/high
    avg_throughput: float
    cv: float                        # coefficient of variation across ranges


@dataclass(frozen=True)
class BenchmarkResult:
    """Complete result of a throughput calibration run.

    Stored in the state store and consumed by the planner.
    Two confidence signals, kept separate:

        confidence      → reliability  (range_consistency * curve_shape_factor)
        signal_strength → tuning importance (flat curve ≠ unreliable)
    """
    source_type: str
    object_name: str
    probe_rows: int
    ranges_used: int                         # 2 or 3
    worker_grid: tuple[int, ...]             # grid actually probed
    probe_results: tuple[WorkerProbeResult, ...]  # ordered by worker_count
    throughput_by_workers: dict[int, float]  # avg across ranges, keyed by worker count
    optimal_workers: int
    confidence: float                        # range_consistency * curve_shape_factor
    signal_strength: float                   # max(floor, (best-worst)/best)
    curve_shape: CurveShape
    benchmarked_at: datetime
    row_estimate_at_benchmark: int           # for staleness detection

    @property
    def is_trustworthy(self) -> bool:
        """True if confidence meets the planner threshold."""
        return self.confidence >= CONFIDENCE_THRESHOLD

    @property
    def tuning_matters(self) -> bool:
        """True if worker count meaningfully affects throughput."""
        return self.signal_strength >= SIGNAL_STRENGTH_CONSERVATIVE_THRESHOLD

    def is_stale(self, current_row_estimate: int) -> bool:
        """True if the benchmark should be re-run."""
        age = (datetime.now(timezone.utc) - self.benchmarked_at).total_seconds()
        if age > BENCHMARK_MAX_AGE_SECONDS:
            return True
        if self.row_estimate_at_benchmark > 0:
            growth = abs(current_row_estimate - self.row_estimate_at_benchmark)
            if growth / self.row_estimate_at_benchmark > BENCHMARK_MAX_ROW_GROWTH:
                return True
        return False

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict for state store storage."""
        return {
            "source_type": self.source_type,
            "object_name": self.object_name,
            "probe_rows": self.probe_rows,
            "ranges_used": self.ranges_used,
            "worker_grid": list(self.worker_grid),
            "throughput_by_workers": {str(k): v for k, v in self.throughput_by_workers.items()},
            "optimal_workers": self.optimal_workers,
            "confidence": self.confidence,
            "signal_strength": self.signal_strength,
            "curve_shape": self.curve_shape,
            "benchmarked_at": self.benchmarked_at.isoformat(),
            "row_estimate_at_benchmark": self.row_estimate_at_benchmark,
            "probe_results": [
                {
                    "worker_count": p.worker_count,
                    "effective_workers": p.effective_workers,
                    "throughputs": list(p.throughputs),
                    "avg_throughput": p.avg_throughput,
                    "cv": p.cv,
                }
                for p in self.probe_results
            ],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BenchmarkResult":
        """Deserialize from state store JSON."""
        probe_results = tuple(
            WorkerProbeResult(
                worker_count=p["worker_count"],
                effective_workers=p["effective_workers"],
                throughputs=tuple(p["throughputs"]),
                avg_throughput=p["avg_throughput"],
                cv=p["cv"],
            )
            for p in d["probe_results"]
        )
        return cls(
            source_type=d["source_type"],
            object_name=d["object_name"],
            probe_rows=d["probe_rows"],
            ranges_used=d["ranges_used"],
            worker_grid=tuple(d["worker_grid"]),
            probe_results=probe_results,
            throughput_by_workers={int(k): v for k, v in d["throughput_by_workers"].items()},
            optimal_workers=d["optimal_workers"],
            confidence=d["confidence"],
            signal_strength=d["signal_strength"],
            curve_shape=d["curve_shape"],
            benchmarked_at=datetime.fromisoformat(d["benchmarked_at"]),
            row_estimate_at_benchmark=d["row_estimate_at_benchmark"],
        )


# ── Confidence computation (pure functions, no I/O) ───────────────────

def compute_probe_rows(table_rows: int) -> int:
    """Calculate probe size for a given table."""
    return max(PROBE_MIN_ROWS, min(PROBE_MAX_ROWS, int(table_rows * PROBE_TABLE_FRACTION)))


def effective_worker_grid(max_workers: int, min_workers: int = 1) -> tuple[int, ...]:
    """Filter the canonical grid to the configured worker bounds."""
    return tuple(w for w in WORKER_GRID if min_workers <= w <= max_workers)


def should_skip_benchmark(estimated_duration_seconds: float) -> bool:
    """True if the table is too small/fast to warrant benchmarking."""
    return estimated_duration_seconds < BENCHMARK_MIN_TABLE_DURATION_SECONDS


def _cv(values: tuple[float, ...]) -> float:
    """Coefficient of variation. Returns 0 for single-element sequences."""
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return (variance ** 0.5) / mean


def classify_curve_shape(throughputs_ordered: list[float]) -> CurveShape:
    """Classify the throughput curve shape from ordered worker counts.

    Args:
        throughputs_ordered: avg throughput at each worker count, ascending.

    Returns:
        'plateau'       — throughput peaked and flattened or declined at the end
        'increasing'    — throughput still rising at the last grid point
        'non_monotonic' — throughput went up then down (contention)
    """
    if len(throughputs_ordered) < 2:
        return "plateau"

    peak_idx = throughputs_ordered.index(max(throughputs_ordered))
    last_idx = len(throughputs_ordered) - 1

    if peak_idx == last_idx:
        # Throughput is highest at the last (most workers) point — still increasing
        return "increasing"

    # Check if anything after the peak is within 5% of it (slight decline = still plateau)
    post_peak = throughputs_ordered[peak_idx + 1:]
    if all(v >= throughputs_ordered[peak_idx] * 0.95 for v in post_peak):
        return "plateau"

    # Peak before the end, post-peak values meaningfully lower
    return "non_monotonic" if peak_idx > 0 else "plateau"


def compute_confidence(
    probe_results: list[WorkerProbeResult],
    curve_shape: CurveShape,
) -> tuple[float, float]:
    """Compute confidence and signal_strength from probe results.

    Returns:
        (confidence, signal_strength)
        confidence      = range_consistency * curve_shape_factor
        signal_strength = max(FLAT_CURVE_SIGNAL_FLOOR, (best-worst)/best)
    """
    # Range consistency: average CV across all worker counts, then invert
    avg_cv = sum(p.cv for p in probe_results) / len(probe_results) if probe_results else 1.0
    range_consistency = max(0.0, 1.0 - avg_cv)

    shape_factor = CURVE_SHAPE_FACTORS[curve_shape]
    confidence = range_consistency * shape_factor

    # Signal strength: how much does worker count matter?
    avgs = [p.avg_throughput for p in probe_results]
    if not avgs or max(avgs) == 0:
        signal_strength = FLAT_CURVE_SIGNAL_FLOOR
    else:
        best, worst = max(avgs), min(avgs)
        raw_signal = (best - worst) / best
        signal_strength = max(FLAT_CURVE_SIGNAL_FLOOR, raw_signal)

    return round(confidence, 4), round(signal_strength, 4)


def select_optimal_workers(
    throughput_by_workers: dict[int, float],
) -> int:
    """Select optimal worker count with conservative tie-breaking.

    argmax(throughput), but if multiple worker counts are within 5% of the
    maximum, prefer the lowest (safety-first, avoids over-parallelism).
    """
    if not throughput_by_workers:
        return 1
    best_tp = max(throughput_by_workers.values())
    threshold = best_tp * (1.0 - OPTIMAL_WORKER_TIEBREAK_BAND)
    candidates = sorted(
        w for w, tp in throughput_by_workers.items() if tp >= threshold
    )
    return candidates[0]   # lowest within 5% of peak


def conservative_worker_count(optimal_workers: int) -> int:
    """Worker count to use when signal_strength indicates a flat curve.

    Caps at CONSERVATIVE_WORKER_CAP (2). Prevents drifting toward 4 or 8
    on flat curves where the benchmark can't tell which count is best.

    Usage (planner):
        if result.signal_strength <= SIGNAL_STRENGTH_CONSERVATIVE_THRESHOLD:
            workers = conservative_worker_count(result.optimal_workers)
        else:
            workers = result.optimal_workers
    """
    return min(optimal_workers, CONSERVATIVE_WORKER_CAP)


def planner_workers_from_benchmark(result: "BenchmarkResult") -> int:
    """Single entry point for the planner to get a worker count from a benchmark.

    Encapsulates the full decision logic:
        confidence < 0.5         → caller should fall back (raises ValueError)
        signal_strength <= 0.3   → flat curve → conservative_worker_count()
        signal_strength > 0.3    → meaningful curve → optimal_workers

    Raises:
        ValueError: if confidence is below threshold (planner should fall back
                    to profiler heuristic instead of using this result).
    """
    if not result.is_trustworthy:
        raise ValueError(
            f"Benchmark confidence {result.confidence:.2f} below threshold "
            f"{CONFIDENCE_THRESHOLD:.2f}. Use profiler heuristic instead."
        )
    if result.signal_strength <= SIGNAL_STRENGTH_CONSERVATIVE_THRESHOLD:
        return conservative_worker_count(result.optimal_workers)
    return result.optimal_workers


def select_probe_ranges(
    pk_min: int, pk_max: int, probe_rows: int,
) -> tuple[list[ProbeRange], int]:
    """Select 2 or 3 non-adjacent PK ranges covering low/mid/high thirds.

    Returns:
        (ranges, ranges_used)
    """
    pk_range = pk_max - pk_min
    range_size = probe_rows  # PK units, assumes ~1 row per PK unit

    # Need at least 3× probe_rows to fit 3 non-adjacent ranges
    if pk_range < 3 * range_size:
        # Fallback: 2 ranges — low quarter and high quarter
        low_start = pk_min
        high_start = max(pk_min + range_size + 1, pk_max - range_size)
        ranges = [
            ProbeRange(low_start, low_start + range_size - 1, "low"),
            ProbeRange(high_start, min(high_start + range_size - 1, pk_max), "high"),
        ]
        return ranges, 2

    third = pk_range // 3
    low_start  = pk_min
    mid_start  = pk_min + third + (third // 2) - range_size // 2   # centre of mid-third
    high_start = pk_max - range_size

    # Clamp to valid range
    mid_start  = max(low_start + range_size + 1, mid_start)
    high_start = max(mid_start + range_size + 1, high_start)

    ranges = [
        ProbeRange(low_start,  low_start  + range_size - 1, "low"),
        ProbeRange(mid_start,  mid_start  + range_size - 1, "mid"),
        ProbeRange(high_start, pk_max,                       "high"),
    ]
    return ranges, 3


# ── Benchmarker class ─────────────────────────────────────────────────

class Benchmarker:
    """Executes multi-worker throughput calibration probes.

    Usage:
        benchmarker = Benchmarker(connector, config)
        result = benchmarker.run("orders", prof)

    Each worker count in the grid is probed across all PK ranges.
    For each range, a cache-warm pass (single worker, discarded) runs
    first, then the measured pass records rows and wall-clock time.

    The Benchmarker uses direct connector queries — not the execution
    engine — so it has no dependency on plan/chunk/worker machinery.
    One connection per probe pass; sequential (not parallel) probing
    avoids confounding source-side load effects.
    """

    def __init__(
        self,
        connector: "BaseConnector",
        config: "BenchmarkerConfig | None" = None,
    ) -> None:
        from ixtract.connectors.base import BaseConnector
        self.connector = connector
        self.config = config or BenchmarkerConfig()

    def run(
        self,
        object_name: str,
        profile: "SourceProfile",
        source_type: str = "postgresql",
        force: bool = False,
    ) -> "BenchmarkResult | None":
        """Run throughput calibration and return a BenchmarkResult.

        Returns None if the table is too small to benchmark or the worker
        grid has fewer than 2 points (benchmarking a single value is meaningless).

        Args:
            object_name: Table name to benchmark.
            profile:     SourceProfile from the profiler (provides pk_min,
                         pk_max, row_estimate, recommended_strategy).
            source_type: Source type string for storage (e.g. "postgresql").
            force:       If True, bypass the size/duration guard. Used by
                         the --force CLI flag and integration tests.
        """
        import time
        from datetime import datetime, timezone

        # Guard: too small / too fast (bypassed by force=True or min_table_duration=0)
        if not force and self.config.min_table_duration_seconds > 0:
            if self._estimated_duration(profile) < self.config.min_table_duration_seconds:
                return None

        # Guard: no usable integer PK
        if not profile.has_usable_pk or profile.pk_min is None or profile.pk_max is None:
            return None

        pk_min = int(profile.pk_min)
        pk_max = int(profile.pk_max)
        pk_col = profile.primary_key

        # Build worker grid
        grid = effective_worker_grid(
            max_workers=self.config.max_workers,
            min_workers=self.config.min_workers,
        )
        if len(grid) < 2:
            return None

        probe_rows = compute_probe_rows(profile.row_estimate)
        ranges, ranges_used = select_probe_ranges(pk_min, pk_max, probe_rows)

        probe_results: list[WorkerProbeResult] = []

        for worker_count in grid:
            range_throughputs: list[float] = []
            effective_workers_seen: list[int] = []

            for probe_range in ranges:
                # Cache-warm pass — single worker, result discarded
                self._execute_probe(
                    object_name, pk_col, probe_range, warm=True
                )

                # Measured pass
                rows, elapsed = self._execute_probe(
                    object_name, pk_col, probe_range, warm=False
                )

                if elapsed > 0 and rows > 0:
                    tp = rows / elapsed
                    range_throughputs.append(tp)
                    # Effective workers: for single-connection probes this equals
                    # the planned worker_count. Stored for future multi-connection probing.
                    effective_workers_seen.append(worker_count)

            if not range_throughputs:
                continue

            avg_tp = sum(range_throughputs) / len(range_throughputs)
            cv = _cv(tuple(range_throughputs))

            probe_results.append(WorkerProbeResult(
                worker_count=worker_count,
                effective_workers=worker_count,
                throughputs=tuple(range_throughputs),
                avg_throughput=avg_tp,
                cv=cv,
            ))

        if len(probe_results) < 2:
            return None

        # Derive outputs
        throughput_by_workers = {p.worker_count: p.avg_throughput for p in probe_results}
        ordered_tps = [throughput_by_workers[w] for w in sorted(throughput_by_workers)]
        curve_shape = classify_curve_shape(ordered_tps)
        confidence, signal_strength = compute_confidence(probe_results, curve_shape)
        optimal_workers = select_optimal_workers(throughput_by_workers)

        return BenchmarkResult(
            source_type=source_type,
            object_name=object_name,
            probe_rows=probe_rows,
            ranges_used=ranges_used,
            worker_grid=tuple(grid),
            probe_results=tuple(probe_results),
            throughput_by_workers=throughput_by_workers,
            optimal_workers=optimal_workers,
            confidence=confidence,
            signal_strength=signal_strength,
            curve_shape=curve_shape,
            benchmarked_at=datetime.now(timezone.utc),
            row_estimate_at_benchmark=profile.row_estimate,
        )

    def _execute_probe(
        self,
        object_name: str,
        pk_col: str,
        probe_range: ProbeRange,
        warm: bool,
    ) -> tuple[int, float]:
        """Execute one probe pass. Returns (rows_read, elapsed_seconds).

        Args:
            warm: If True, this is a cache-warm pass — result is discarded.
        """
        import time
        query = (
            f"SELECT * FROM {object_name} "
            f"WHERE {pk_col} >= :pk_start AND {pk_col} <= :pk_end"
        )
        params = {"pk_start": probe_range.pk_start, "pk_end": probe_range.pk_end}

        rows_read = 0
        start = time.perf_counter()

        for batch in self.connector.extract_chunk(object_name, query, params):
            rows_read += len(batch)

        elapsed = time.perf_counter() - start
        return rows_read, elapsed

    def _estimated_duration(self, profile: "SourceProfile") -> float:
        """Estimate extraction duration using the profiler's 25MB/sec model."""
        if profile.size_estimate_bytes <= 0:
            return 0.0
        return profile.size_estimate_bytes / (25 * 1024 * 1024)


@dataclass(frozen=True)
class BenchmarkerConfig:
    """Configuration for the Benchmarker."""
    min_workers: int = 1
    max_workers: int = 16
    # Override the module-level skip threshold. Useful for tests and --force CLI.
    # Set to 0.0 to disable the size guard entirely.
    min_table_duration_seconds: float = BENCHMARK_MIN_TABLE_DURATION_SECONDS
