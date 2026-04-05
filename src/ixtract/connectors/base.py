"""Abstract connector interface for data sources.

Every source connector implements this interface. The planner and engine
depend only on this abstraction, never on specific connector implementations.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator, Optional


@dataclass(frozen=True)
class ColumnInfo:
    """Metadata for a single column."""
    name: str
    data_type: str
    nullable: bool = True
    is_primary_key: bool = False


@dataclass(frozen=True)
class ObjectMetadata:
    """Source metadata for a single object (table, endpoint, file)."""
    object_name: str
    row_estimate: int
    size_estimate_bytes: int
    columns: tuple[ColumnInfo, ...]
    primary_key: Optional[str] = None
    primary_key_type: Optional[str] = None
    pk_min: Optional[Any] = None
    pk_max: Optional[Any] = None


@dataclass(frozen=True)
class LatencyProfile:
    """Query latency characteristics of the source."""
    p50_ms: float
    p95_ms: float
    connection_ms: float
    sample_count: int


@dataclass(frozen=True)
class SourceConnections:
    """Current connection utilization on the source."""
    max_connections: int
    active_connections: int

    @property
    def available(self) -> int:
        return max(0, self.max_connections - self.active_connections)

    @property
    def available_safe(self) -> int:
        """Available connections with 50% safety factor."""
        return max(1, self.available // 2)


class BaseConnector(ABC):
    """Abstract connector interface.

    All source connectors implement this. The planner and engine depend
    only on this abstraction.
    """

    @abstractmethod
    def connect(self, config: dict[str, Any]) -> None:
        """Establish connection to the data source."""

    @abstractmethod
    def metadata(self, object_name: str) -> ObjectMetadata:
        """Return metadata for the given object."""

    @abstractmethod
    def extract_chunk(
        self, object_name: str, chunk_query: str, params: dict[str, Any] | None = None
    ) -> Iterator[list[dict[str, Any]]]:
        """Extract a chunk of data as an iterator of record batches (list of dicts)."""

    @abstractmethod
    def estimate_latency(self, object_name: str) -> LatencyProfile:
        """Sample the source to build a latency profile."""

    @abstractmethod
    def get_connections(self) -> SourceConnections:
        """Return current connection utilization on the source."""

    @abstractmethod
    def get_pk_distribution(self, object_name: str, num_buckets: int = 10) -> list[int]:
        """Return row counts per equal-width PK range bucket. For skew detection."""

    @abstractmethod
    def close(self) -> None:
        """Release all resources and connections."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
