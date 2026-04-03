"""Source Profiler — gathers source metadata for intelligent planning.

Runs before the first extraction plan. Produces a SourceProfile that
eliminates blind cold starts and gives the planner real data on day one.

Phase 1: catalog stats, PK analysis, latency probe, skew indicator.
Phase 2 adds: benchmarker (multi-worker throughput calibration).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Optional

from ixtract.connectors.base import BaseConnector, ObjectMetadata, LatencyProfile, SourceConnections


@dataclass(frozen=True)
class SourceProfile:
    """Complete profile of a source object for planning."""
    # Metadata
    object_name: str
    row_estimate: int
    size_estimate_bytes: int
    column_count: int
    avg_row_bytes: int

    # Primary key
    primary_key: Optional[str]
    primary_key_type: Optional[str]
    pk_min: Optional[Any]
    pk_max: Optional[Any]

    # Skew
    pk_distribution_cv: float  # coefficient of variation. >1.0 = significant skew
    has_usable_pk: bool

    # Latency
    latency_p50_ms: float
    latency_p95_ms: float
    connection_ms: float

    # Source capacity
    max_connections: int
    active_connections: int
    available_connections_safe: int

    # Planning recommendation
    recommended_start_workers: int
    recommended_strategy: str  # "single_pass" | "range_chunking"
    recommended_scheduling: str  # "round_robin" | "greedy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# Thresholds
SMALL_TABLE_ROWS = 100_000
SMALL_TABLE_BYTES = 50_000_000  # 50MB
SKEW_CV_THRESHOLD = 1.0  # CV > 1.0 → significant skew


class SourceProfiler:
    """Profiles a source object to produce a SourceProfile.

    Uses the connector to run lightweight diagnostic queries.
    Total profiling time: typically 2-10 seconds.
    """

    def __init__(self, connector: BaseConnector) -> None:
        self._connector = connector

    def profile(self, object_name: str) -> SourceProfile:
        """Run full profiling pass on the given object.

        Steps:
            1. Catalog metadata (row count, size, columns, PK)
            2. Latency probe (5 lightweight queries)
            3. Connection utilization
            4. PK distribution for skew detection
            5. Compute recommendations
        """
        # 1. Metadata
        meta = self._connector.metadata(object_name)

        # 2. Latency
        latency = self._connector.estimate_latency(object_name)

        # 3. Connections
        connections = self._connector.get_connections()

        # 4. PK distribution
        pk_cv = 0.0
        has_usable_pk = bool(meta.primary_key and meta.pk_min is not None)

        if has_usable_pk:
            distribution = self._connector.get_pk_distribution(object_name, num_buckets=10)
            if distribution and len(distribution) > 1:
                pk_cv = self._coefficient_of_variation(distribution)

        # 5. Avg row bytes
        avg_row_bytes = 0
        if meta.row_estimate > 0 and meta.size_estimate_bytes > 0:
            avg_row_bytes = meta.size_estimate_bytes // meta.row_estimate

        # 6. Recommendations
        strategy = self._recommend_strategy(meta, has_usable_pk)
        scheduling = self._recommend_scheduling(pk_cv)
        start_workers = self._recommend_start_workers(
            meta, connections, has_usable_pk
        )

        return SourceProfile(
            object_name=object_name,
            row_estimate=meta.row_estimate,
            size_estimate_bytes=meta.size_estimate_bytes,
            column_count=len(meta.columns),
            avg_row_bytes=avg_row_bytes,
            primary_key=meta.primary_key,
            primary_key_type=meta.primary_key_type,
            pk_min=meta.pk_min,
            pk_max=meta.pk_max,
            pk_distribution_cv=round(pk_cv, 3),
            has_usable_pk=has_usable_pk,
            latency_p50_ms=latency.p50_ms,
            latency_p95_ms=latency.p95_ms,
            connection_ms=latency.connection_ms,
            max_connections=connections.max_connections,
            active_connections=connections.active_connections,
            available_connections_safe=connections.available_safe,
            recommended_start_workers=start_workers,
            recommended_strategy=strategy,
            recommended_scheduling=scheduling,
        )

    def _recommend_strategy(self, meta: ObjectMetadata, has_usable_pk: bool) -> str:
        """Decision tree from architecture Section 11.1."""
        if meta.row_estimate < SMALL_TABLE_ROWS or meta.size_estimate_bytes < SMALL_TABLE_BYTES:
            return "single_pass"
        if has_usable_pk:
            return "range_chunking"
        return "single_pass"  # fallback; offset_chunking is Phase 2

    def _recommend_scheduling(self, pk_cv: float) -> str:
        """Profiler-aware scheduling selection from architecture Section 11.2."""
        if pk_cv > SKEW_CV_THRESHOLD:
            return "greedy"  # work_stealing is Phase 2
        return "round_robin"

    def _recommend_start_workers(
        self, meta: ObjectMetadata, connections: SourceConnections,
        has_usable_pk: bool,
    ) -> int:
        """Conservative first-run worker recommendation.

        Phase 1: no benchmark data, so we estimate from table size
        and available connections.
        """
        if meta.row_estimate < SMALL_TABLE_ROWS:
            return 1  # small table, single pass

        if not has_usable_pk:
            return 1  # can't parallelize without PK

        # Start conservative: min of capacity-based and size-based estimates
        # Capacity: don't use more than half of available connections
        capacity_limit = max(1, connections.available_safe)

        # Size: roughly 1 worker per 2M rows, capped at 8 for first run
        size_based = max(1, min(8, meta.row_estimate // 2_000_000))

        return min(capacity_limit, size_based)

    @staticmethod
    def _coefficient_of_variation(values: list[int]) -> float:
        """CV = stddev / mean. Higher = more skewed."""
        n = len(values)
        if n < 2:
            return 0.0
        mean = sum(values) / n
        if mean <= 0:
            return 0.0
        variance = sum((v - mean) ** 2 for v in values) / n
        return math.sqrt(variance) / mean
