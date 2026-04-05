"""ExtractionIntent — declarative representation of WHAT to extract.

Intent contains zero execution decisions. No chunk sizes, no worker counts,
no parallelism, no retry logic. Those are planner outputs, not user inputs.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class SourceType(str, Enum):
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"
    SQLSERVER = "sqlserver"
    REST_API = "rest_api"
    FILE = "file"
    S3 = "s3"
    GCS = "gcs"


class ExtractionMode(str, Enum):
    FULL = "full"
    INCREMENTAL = "incremental"


class TargetType(str, Enum):
    PARQUET = "parquet"
    CSV = "csv"
    S3 = "s3"
    GCS = "gcs"
    LOCAL = "local"


@dataclass(frozen=True)
class ExtractionConstraints:
    """Optional user-specified extraction constraints."""
    max_duration_seconds: Optional[int] = None
    max_workers: Optional[int] = None
    max_source_connections: Optional[int] = None
    priority: int = 0


@dataclass(frozen=True)
class ExtractionIntent:
    """Declarative extraction intent.

    Represents WHAT to extract, not HOW. The planner converts this into
    an ExecutionPlan with all execution decisions.
    """
    source_type: SourceType
    source_config: dict[str, Any]
    object_name: str
    mode: ExtractionMode = ExtractionMode.FULL
    incremental_key: Optional[str] = None
    incremental_value: Optional[Any] = None
    target_type: TargetType = TargetType.PARQUET
    target_config: dict[str, Any] = field(default_factory=dict)
    constraints: ExtractionConstraints = field(default_factory=ExtractionConstraints)

    def __post_init__(self) -> None:
        if self.mode == ExtractionMode.INCREMENTAL and not self.incremental_key:
            raise ValueError("incremental_key is required when mode is INCREMENTAL")
        if self.constraints.max_workers is not None:
            if not (1 <= self.constraints.max_workers <= 64):
                raise ValueError("max_workers must be between 1 and 64")

    def intent_hash(self) -> str:
        """Deterministic SHA-256 hash for deduplication and replay."""
        payload = json.dumps({
            "source_type": self.source_type.value,
            "source_config": self.source_config,
            "object_name": self.object_name,
            "mode": self.mode.value,
            "incremental_key": self.incremental_key,
            "incremental_value": self.incremental_value,
            "target_type": self.target_type.value,
            "target_config": self.target_config,
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def source_object_key(self) -> str:
        """Stable key for state store lookups: source_type::object_name."""
        return f"{self.source_type.value}::{self.object_name}"
