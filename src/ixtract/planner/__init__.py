"""ExecutionPlan — the central contract of ixtract.

Immutable once created. The Execution Engine reads it; nothing may write
to it after creation. Runtime state is tracked separately in RuntimeState.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class Strategy(str, Enum):
    SINGLE_PASS = "single_pass"
    RANGE_CHUNKING = "range_chunking"
    OFFSET_CHUNKING = "offset_chunking"
    TIME_WINDOW = "time_window"
    ADAPTIVE_BACKOFF = "adaptive_backoff"


class SchedulingStrategy(str, Enum):
    ROUND_ROBIN = "round_robin"
    GREEDY = "greedy"
    WORK_STEALING = "work_stealing"


class ChunkType(str, Enum):
    RANGE = "range"
    OFFSET = "offset"
    TIME_WINDOW = "time_window"
    FULL_TABLE = "full_table"


class AdaptiveTrigger(str, Enum):
    TIMEOUT = "timeout"
    ERROR_RATE = "error_rate"
    THROUGHPUT_DROP = "throughput_drop"
    SOURCE_LATENCY_SPIKE = "source_latency_spike"


class AdaptiveAction(str, Enum):
    REDUCE_CONCURRENCY = "reduce_concurrency"
    INCREASE_BACKOFF = "increase_backoff"
    PAUSE_AND_RETRY = "pause_and_retry"
    SKIP_CHUNK = "skip_chunk"


# ── Component Schemas ─────────────────────────────────────────────────

@dataclass(frozen=True)
class ChunkDefinition:
    """A self-contained unit of work assignable to any worker.
    Chunks must be non-overlapping and collectively exhaustive.
    """
    chunk_id: str
    chunk_type: ChunkType
    estimated_rows: int
    estimated_bytes: int
    range_start: Optional[Any] = None
    range_end: Optional[Any] = None
    offset: Optional[int] = None
    limit: Optional[int] = None
    time_start: Optional[str] = None
    time_end: Optional[str] = None
    priority: int = 0


@dataclass(frozen=True)
class AdaptiveRule:
    """Runtime adaptation rule with trigger, action, and hard bounds.

    Backoff rule fires only when BOTH guards are satisfied:
      - relative: query_ms > threshold (ms) — raw latency threshold
      - absolute: query_ms > absolute_floor_ms — prevents over-triggering on tiny baselines

    E.g. for a p50=1ms DB: threshold=5ms (5×), absolute_floor=50ms → needs both.
    """
    rule_id: str
    trigger: AdaptiveTrigger
    threshold: float           # primary threshold (e.g. 5× p50 in ms)
    action: AdaptiveAction
    step_size: float = 1.0
    max_activations: int = 10
    cooldown_chunks: int = 3   # chunks to skip before re-evaluating after firing
    absolute_floor_ms: float = 50.0  # absolute minimum to prevent over-triggering
    backoff_sleep_base: float = 2.0  # base seconds for sleep (doubles each activation)


@dataclass(frozen=True)
class RuleFiredRecord:
    """Record of a single adaptive rule firing during execution."""
    rule_id: str
    activations: int           # total times fired this run
    total_chunks: int          # total chunks processed (for rate calculation)
    max_consecutive: int       # longest consecutive activation streak
    confidence_impact: str     # "note" | "moderate" | "low"

    @property
    def activation_rate(self) -> float:
        if self.total_chunks == 0:
            return 0.0
        return self.activations / self.total_chunks


@dataclass(frozen=True)
class RetryPolicy:
    max_retries: int = 3
    backoff_base_seconds: float = 1.0
    backoff_max_seconds: float = 60.0
    jitter: bool = True


@dataclass(frozen=True)
class CostEstimate:
    predicted_duration_seconds: float
    predicted_throughput_rows_sec: float
    predicted_total_rows: int
    predicted_total_bytes: int


@dataclass(frozen=True)
class WriterConfig:
    output_format: str = "parquet"
    output_path: str = "./output"
    compression: str = "snappy"
    partition_by: Optional[str] = None
    naming_pattern: str = "{object}_{chunk_id}.{format}"
    temp_path: str = "./output"  # should match output_path; writer uses output dir for atomic rename
    max_file_size_bytes: Optional[int] = None  # Phase 3C: rotation threshold, None = no rotation


@dataclass(frozen=True)
class MetadataSnapshot:
    """Frozen copy of source metadata used during planning."""
    row_estimate: int
    size_estimate_bytes: int
    column_count: int
    primary_key: Optional[str] = None
    primary_key_type: Optional[str] = None
    has_timestamp_column: bool = False
    timestamp_column: Optional[str] = None
    schema_hash: Optional[str] = None


# ── The ExecutionPlan ─────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionPlan:
    """The central immutable contract between Control Plane and Data Plane.

    Once created, this object is frozen. The execution engine reads from it
    but never writes to it.
    """
    # Identity
    intent_hash: str
    strategy: Strategy
    chunks: tuple[ChunkDefinition, ...]
    cost_estimate: CostEstimate
    metadata_snapshot: MetadataSnapshot

    # Parallelism
    worker_count: int = 1
    worker_bounds: tuple[int, int] = (1, 16)
    scheduling: SchedulingStrategy = SchedulingStrategy.GREEDY

    # Adaptation & retry
    adaptive_rules: tuple[AdaptiveRule, ...] = ()
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)

    # Writer
    writer_config: WriterConfig = field(default_factory=WriterConfig)

    # Metadata
    plan_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    plan_version: str = "1.0"
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        if self.worker_count < 1:
            raise ValueError("worker_count must be >= 1")
        if len(self.chunks) < 1:
            raise ValueError("At least one chunk is required")
        lo, hi = self.worker_bounds
        if not (1 <= lo <= hi):
            raise ValueError(f"Invalid worker_bounds: ({lo}, {hi})")

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict (for JSON storage / introspection)."""
        def _convert(obj: Any) -> Any:
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, (list, tuple)):
                return [_convert(i) for i in obj]
            if hasattr(obj, "__dataclass_fields__"):
                return {k: _convert(v) for k, v in asdict(obj).items()}
            return obj
        return _convert(self)

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "ExecutionPlan":
        """Deserialize from a plain dict."""
        chunks = tuple(
            ChunkDefinition(
                chunk_id=c["chunk_id"],
                chunk_type=ChunkType(c["chunk_type"]),
                estimated_rows=c["estimated_rows"],
                estimated_bytes=c["estimated_bytes"],
                range_start=c.get("range_start"),
                range_end=c.get("range_end"),
                offset=c.get("offset"),
                limit=c.get("limit"),
                time_start=c.get("time_start"),
                time_end=c.get("time_end"),
                priority=c.get("priority", 0),
            )
            for c in data["chunks"]
        )
        adaptive = tuple(
            AdaptiveRule(
                rule_id=r["rule_id"],
                trigger=AdaptiveTrigger(r["trigger"]),
                threshold=r["threshold"],
                action=AdaptiveAction(r["action"]),
                step_size=r.get("step_size", 1.0),
                max_activations=r.get("max_activations", 3),
                cooldown_seconds=r.get("cooldown_seconds", 30),
            )
            for r in data.get("adaptive_rules", [])
        )
        return ExecutionPlan(
            intent_hash=data["intent_hash"],
            strategy=Strategy(data["strategy"]),
            chunks=chunks,
            cost_estimate=CostEstimate(**data["cost_estimate"]),
            metadata_snapshot=MetadataSnapshot(**data["metadata_snapshot"]),
            worker_count=data.get("worker_count", 1),
            worker_bounds=tuple(data.get("worker_bounds", (1, 16))),
            scheduling=SchedulingStrategy(data.get("scheduling", "greedy")),
            adaptive_rules=adaptive,
            retry_policy=RetryPolicy(**data.get("retry_policy", {})),
            writer_config=WriterConfig(**data.get("writer_config", {})),
            plan_id=data.get("plan_id", str(uuid.uuid4())),
            plan_version=data.get("plan_version", "1.0"),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
        )
