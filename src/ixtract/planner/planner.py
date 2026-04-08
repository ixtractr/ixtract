"""Planner — converts Intent + Profile + History into an ExecutionPlan.

Pure function: same inputs always produce the same plan.
No side effects beyond reading from the state store and profile data.
"""
from __future__ import annotations

import math
import os
from typing import Optional

from ixtract.intent import ExtractionIntent
from ixtract.planner import (
    ExecutionPlan, Strategy, SchedulingStrategy, ChunkDefinition, ChunkType,
    CostEstimate, MetadataSnapshot, RetryPolicy, WriterConfig, AdaptiveRule,
)
from ixtract.profiler import SourceProfile
from ixtract.controller import ControllerState
from ixtract.state import StateStore


# ── Thresholds ────────────────────────────────────────────────────────
SMALL_TABLE_ROWS = 100_000
SMALL_TABLE_BYTES = 50_000_000
DEFAULT_THROUGHPUT = 10_000  # rows/sec fallback when no history
DEFAULT_CHUNK_TARGET_ROWS = 500_000  # target rows per chunk
SNAPSHOT_WARN_MINUTES = 30


def plan_extraction(
    intent: ExtractionIntent,
    profile: SourceProfile,
    store: Optional[StateStore] = None,
    controller_state: Optional[ControllerState] = None,
) -> ExecutionPlan:
    """Generate an ExecutionPlan from intent + profile + history.

    Args:
        intent: What to extract.
        profile: Source profile data.
        store: State store for historical metrics (optional for first run).
        controller_state: Current controller state (None on first run).

    Returns:
        Frozen, immutable ExecutionPlan.
    """
    # ── Strategy selection ────────────────────────────────────────
    strategy = _select_strategy(profile)

    # ── Benchmark lookup ──────────────────────────────────────────
    benchmark = None
    if store:
        try:
            benchmark = store.get_benchmark("postgresql", profile.object_name)
            # Invalidate stale benchmarks so planner falls back to profiler
            if benchmark and benchmark.is_stale(profile.row_estimate):
                benchmark = None
        except Exception:
            benchmark = None  # never let benchmark lookup break planning

    # ── Worker count resolution ───────────────────────────────────
    worker_count, worker_source = _resolve_workers(
        profile, controller_state, intent, benchmark
    )
    worker_bounds = (1, min(16, profile.available_connections_safe * 2))

    # ── Chunk computation ─────────────────────────────────────────
    chunks = _compute_chunks(strategy, profile, worker_count)

    # ── Scheduling ────────────────────────────────────────────────
    scheduling = SchedulingStrategy(profile.recommended_scheduling)

    # ── Cost estimation ───────────────────────────────────────────
    throughput = _estimate_throughput(profile, store, worker_count)
    est_duration = profile.row_estimate / max(throughput, 1)
    cost_estimate = CostEstimate(
        predicted_duration_seconds=round(est_duration, 1),
        predicted_throughput_rows_sec=round(throughput, 1),
        predicted_total_rows=profile.row_estimate,
        predicted_total_bytes=profile.size_estimate_bytes,
    )

    # ── Snapshot warning ──────────────────────────────────────────
    snapshot_warn = est_duration > (SNAPSHOT_WARN_MINUTES * 60)

    # ── Disk space check ──────────────────────────────────────────
    output_path = intent.target_config.get("output_path", "./output")
    _check_disk_space(output_path, profile.size_estimate_bytes)

    # ── Metadata snapshot ─────────────────────────────────────────
    meta_snapshot = MetadataSnapshot(
        row_estimate=profile.row_estimate,
        size_estimate_bytes=profile.size_estimate_bytes,
        column_count=profile.column_count,
        primary_key=profile.primary_key,
        primary_key_type=profile.primary_key_type,
        has_timestamp_column=False,  # detected in Phase 2
    )

    # ── Retry & writer config ─────────────────────────────────────
    max_retries = 3
    retry_policy = RetryPolicy(max_retries=max_retries)
    writer_config = WriterConfig(
        output_format=intent.target_type.value,
        output_path=intent.target_config.get("output_path", "./output"),
        compression=intent.target_config.get("compression", "snappy"),
        naming_pattern=intent.target_config.get(
            "naming_pattern", "{object}_{chunk_id}.parquet"
        ),
    )

    return ExecutionPlan(
        intent_hash=intent.intent_hash(),
        strategy=strategy,
        chunks=tuple(chunks),
        worker_count=worker_count,
        worker_bounds=worker_bounds,
        scheduling=scheduling,
        adaptive_rules=(),  # Phase 2
        retry_policy=retry_policy,
        cost_estimate=cost_estimate,
        writer_config=writer_config,
        metadata_snapshot=meta_snapshot,
    ), worker_source


def format_plan_summary(plan: ExecutionPlan, profile: SourceProfile,
                         controller_state: Optional[ControllerState] = None,
                         worker_source: str = "") -> str:
    """Format plan as CLI summary output."""
    if worker_source:
        source_label = worker_source
    elif controller_state is None:
        source_label = "profiler recommended"
    elif controller_state.converged:
        source_label = "controller converged"
    else:
        source_label = "controller exploring"

    est = plan.cost_estimate
    lines = [
        f"Plan: {profile.object_name}  |  {plan.strategy.value}  |  "
        f"{plan.worker_count} workers  |  {len(plan.chunks)} chunks",
        f"Est. duration: {_fmt_duration(est.predicted_duration_seconds)}  |  "
        f"Est. rows: {_fmt_num(est.predicted_total_rows)}  |  "
        f"Est. bytes: {_fmt_bytes(est.predicted_total_bytes)}",
        f"Workers: {plan.worker_count} ({source_label}; "
        f"capped by source_slots:{profile.available_connections_safe})",
        f"Safety: connections {plan.worker_count}/{profile.max_connections} "
        f"({plan.worker_count*100//profile.max_connections}%) \u2714",
        f"Consistency: snapshot isolation (REPEATABLE READ)",
    ]

    # Snapshot warning
    if est.predicted_duration_seconds > SNAPSHOT_WARN_MINUTES * 60:
        est_min = est.predicted_duration_seconds / 60
        lines.append(
            f"  \u26A0 Est. duration ({est_min:.0f}m) exceeds snapshot threshold "
            f"({SNAPSHOT_WARN_MINUTES}m)."
        )
        lines.append(
            "    Risk: VACUUM delays on source during extraction."
        )
        lines.append(
            "    Recommendations: schedule during low-write window, "
            "or enable chunked_snapshot (Phase 2)"
        )

    return "\n".join(lines)


# ── Internal helpers ──────────────────────────────────────────────────

def _select_strategy(profile: SourceProfile) -> Strategy:
    if profile.row_estimate < SMALL_TABLE_ROWS or profile.size_estimate_bytes < SMALL_TABLE_BYTES:
        return Strategy.SINGLE_PASS
    if profile.has_usable_pk:
        return Strategy.RANGE_CHUNKING
    return Strategy.SINGLE_PASS  # offset_chunking is Phase 2


def _resolve_workers(
    profile: SourceProfile,
    controller_state: Optional[ControllerState],
    intent: ExtractionIntent,
    benchmark: Optional["BenchmarkResult"] = None,
) -> tuple[int, str]:
    """Worker count resolution — returns (count, source_label).

    Priority order (highest first):
        1. Benchmark result (if trustworthy and fresh)
        2. Controller state (if active from prior runs)
        3. Profiler recommendation (first run, no benchmark)

    Resource caps always apply as safety bounds after source selection.

    Returns:
        (worker_count, source_label) — label for CLI display.
    """
    from ixtract.benchmarker import planner_workers_from_benchmark

    # ── Priority 1: Benchmark ─────────────────────────────────────
    if benchmark is not None:
        try:
            base = planner_workers_from_benchmark(benchmark)
            if benchmark.signal_strength <= 0.30:
                label = "benchmarked (conservative)"
            else:
                label = "benchmarked"
        except ValueError:
            # Low confidence — fall through to lower priorities
            benchmark = None

    # ── Priority 2: Controller ────────────────────────────────────
    if benchmark is None:
        if controller_state is not None and controller_state.last_throughput > 0:
            base = controller_state.current_workers
            label = ("controller converged" if controller_state.converged
                     else "controller exploring")
        else:
            # ── Priority 3: Profiler ──────────────────────────────
            base = profile.recommended_start_workers
            label = "profiler recommended"

    # ── Safety caps ───────────────────────────────────────────────
    source_cap = profile.available_connections_safe
    system_cap = os.cpu_count() or 4
    intent_cap = intent.constraints.max_workers or 64

    count = max(1, min(base, source_cap, system_cap, intent_cap))
    return count, label


def _compute_chunks(
    strategy: Strategy, profile: SourceProfile, worker_count: int,
) -> list[ChunkDefinition]:
    """Compute chunk definitions based on strategy."""
    if strategy == Strategy.SINGLE_PASS:
        return [ChunkDefinition(
            chunk_id="chunk_001",
            chunk_type=ChunkType.FULL_TABLE,
            estimated_rows=profile.row_estimate,
            estimated_bytes=profile.size_estimate_bytes,
        )]

    if strategy == Strategy.RANGE_CHUNKING:
        return _range_chunks(profile, worker_count)

    return [ChunkDefinition(
        chunk_id="chunk_001",
        chunk_type=ChunkType.FULL_TABLE,
        estimated_rows=profile.row_estimate,
        estimated_bytes=profile.size_estimate_bytes,
    )]


def _range_chunks(profile: SourceProfile, worker_count: int) -> list[ChunkDefinition]:
    """Compute range-based chunks on the primary key."""
    pk_min = profile.pk_min
    pk_max = profile.pk_max

    if pk_min is None or pk_max is None:
        return [ChunkDefinition(
            chunk_id="chunk_001",
            chunk_type=ChunkType.FULL_TABLE,
            estimated_rows=profile.row_estimate,
            estimated_bytes=profile.size_estimate_bytes,
        )]

    # Target: enough chunks for good load balancing, at least 2x workers
    num_chunks = max(worker_count * 2, profile.row_estimate // DEFAULT_CHUNK_TARGET_ROWS)
    num_chunks = max(num_chunks, 1)
    num_chunks = min(num_chunks, 200)  # cap to avoid excessive overhead

    if isinstance(pk_min, (int, float)) and isinstance(pk_max, (int, float)):
        pk_range = pk_max - pk_min
        if pk_range <= 0:
            return [ChunkDefinition(
                chunk_id="chunk_001",
                chunk_type=ChunkType.FULL_TABLE,
                estimated_rows=profile.row_estimate,
                estimated_bytes=profile.size_estimate_bytes,
            )]

        # For integer PKs, keep boundaries as integers
        pk_is_integer = isinstance(pk_min, int) and isinstance(pk_max, int)
        chunk_width = pk_range / num_chunks
        rows_per_chunk = profile.row_estimate // num_chunks
        bytes_per_chunk = profile.size_estimate_bytes // num_chunks

        chunks = []
        for i in range(num_chunks):
            lo = pk_min + i * chunk_width
            hi = pk_min + (i + 1) * chunk_width

            if pk_is_integer:
                lo = int(round(lo))
                hi = int(round(hi))
                # Ensure last chunk covers through pk_max
                if i == num_chunks - 1:
                    hi = pk_max + 1  # exclusive upper bound

            chunks.append(ChunkDefinition(
                chunk_id=f"chunk_{i+1:03d}",
                chunk_type=ChunkType.RANGE,
                estimated_rows=rows_per_chunk,
                estimated_bytes=bytes_per_chunk,
                range_start=lo,
                range_end=hi,
                priority=i,
            ))
        return chunks

    return [ChunkDefinition(
        chunk_id="chunk_001",
        chunk_type=ChunkType.FULL_TABLE,
        estimated_rows=profile.row_estimate,
        estimated_bytes=profile.size_estimate_bytes,
    )]


def _estimate_throughput(
    profile: SourceProfile,
    store: Optional[StateStore],
    worker_count: int,
) -> float:
    """Estimate throughput from historical data or defaults.

    Phase 1: uses stored baseline from previous runs.
    Phase 2: adds context-weighted matching.
    """
    # Check for stored baseline from previous runs
    # Key must match what the CLI stores: source="postgresql", object=object_name
    if store:
        baseline = store.get_heuristic(
            "postgresql", profile.object_name, "throughput_baseline"
        )
        if baseline and baseline > 0:
            return baseline

    # No history: estimate from profiler data
    if profile.latency_p50_ms > 1.0:
        # Measurable latency: estimate throughput from query speed
        single_worker_tp = 1000 / profile.latency_p50_ms * 100
        return min(single_worker_tp * math.sqrt(worker_count), 500_000)

    # Sub-millisecond latency (local DB, fast network): use size-based estimate
    # 25MB/sec per worker is conservative — accounts for serialization + write overhead
    # (50MB was 2x too optimistic based on real-world measurements: ~218K rows/sec at 2 workers
    #  on local Docker with 138 bytes/row = ~30MB/sec effective)
    if profile.avg_row_bytes > 0 and profile.row_estimate > 0:
        bytes_per_sec = 25_000_000 * math.sqrt(worker_count)
        return min(bytes_per_sec / max(profile.avg_row_bytes, 1), 500_000)

    return DEFAULT_THROUGHPUT


def _check_disk_space(output_path: str, estimated_bytes: int) -> None:
    """Check if target has enough disk space. Raises if insufficient."""
    try:
        os.makedirs(output_path, exist_ok=True)
        stat = os.statvfs(output_path)
        free_bytes = stat.f_bavail * stat.f_frsize
        if free_bytes < estimated_bytes * 1.2:  # 20% safety margin
            raise RuntimeError(
                f"Insufficient disk space. Need ~{estimated_bytes // (1024**2)}MB, "
                f"have {free_bytes // (1024**2)}MB free at {output_path}."
            )
    except OSError:
        pass  # can't check, proceed


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    mins = seconds / 60
    if mins < 60:
        return f"{mins:.0f}m"
    hours = mins / 60
    return f"{hours:.1f}h"


def _fmt_num(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n/1_000_000_000:.1f}B"
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


def _fmt_bytes(b: int) -> str:
    if b >= 1_000_000_000:
        return f"{b/1_000_000_000:.1f}GB"
    if b >= 1_000_000:
        return f"{b/1_000_000:.1f}MB"
    if b >= 1_000:
        return f"{b/1_000:.1f}KB"
    return f"{b}B"
