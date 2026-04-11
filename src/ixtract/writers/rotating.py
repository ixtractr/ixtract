"""File rotation — splits output into multiple files at a size threshold.

RotatingWriter wraps any BaseWriter and creates new file segments
when estimated output exceeds max_file_size_bytes.

Each segment is a complete, valid file (e.g. valid Parquet with footer).
Segment naming: {object}_{chunk_id}_part001.parquet

Thread-safety: one instance per chunk per worker (inherited from BaseWriter).

Configuration keys:
    max_file_size_bytes:  Size threshold for rotation. Default: None (no rotation).
                          Minimum: 1 MB. When set, writer creates new segments.

Usage:
    writer = RotatingWriter(ParquetWriter, max_file_size_bytes=100_000_000)
    writer.open(config, chunk_id)
    writer.write_batch(batch)   # may trigger rotation internally
    writer.finalize()           # finalizes all segments
"""
from __future__ import annotations

from typing import Any, Optional, Type

from ixtract.writers.parquet import BaseWriter, WriteResult, FinalizeResult


MIN_FILE_SIZE_BYTES = 1_000_000  # 1 MB minimum


class RotatingWriter(BaseWriter):
    """Writer wrapper that splits output at a size threshold.

    Creates segments named: {object}_{chunk_id}_part{NNN}.{format}
    Each segment is independently valid (closed with proper footer).
    """

    def __init__(
        self,
        writer_class: Type[BaseWriter],
        max_file_size_bytes: Optional[int] = None,
    ) -> None:
        self._writer_class = writer_class
        self._max_bytes = max_file_size_bytes
        if self._max_bytes is not None and self._max_bytes < MIN_FILE_SIZE_BYTES:
            raise ValueError(
                f"max_file_size_bytes must be >= {MIN_FILE_SIZE_BYTES} "
                f"(1 MB), got {self._max_bytes}"
            )

        self._config: dict[str, Any] = {}
        self._chunk_id: str = ""
        self._schema: Optional[list[dict[str, str]]] = None

        self._current_writer: Optional[BaseWriter] = None
        self._current_bytes: int = 0
        self._segment: int = 1
        self._finalized_segments: list[FinalizeResult] = []
        self._total_rows: int = 0
        self._total_bytes: int = 0

    def open(
        self,
        config: dict[str, Any],
        chunk_id: str,
        schema: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """Open the first segment."""
        self._config = config
        self._chunk_id = chunk_id
        self._schema = schema
        self._segment = 1
        self._finalized_segments = []
        self._total_rows = 0
        self._total_bytes = 0
        self._current_bytes = 0

        self._open_segment()

    def write_batch(self, batch: list[dict[str, Any]]) -> WriteResult:
        """Write batch, rotating if size threshold is exceeded."""
        if not batch:
            return WriteResult(0, 0)

        result = self._current_writer.write_batch(batch)
        self._current_bytes += result.bytes_written
        self._total_rows += result.rows_written

        # Check if rotation needed (only if max_bytes is set)
        if (self._max_bytes is not None
                and self._current_bytes >= self._max_bytes):
            self._rotate()

        return result

    def finalize(self) -> FinalizeResult:
        """Finalize all segments. Returns aggregate result."""
        if self._current_writer:
            seg_result = self._current_writer.finalize()
            self._finalized_segments.append(seg_result)
            self._total_bytes += seg_result.total_bytes
            self._current_writer = None

        if not self._finalized_segments:
            raise RuntimeError("RotatingWriter not properly initialized.")

        # Return first segment path as final_path, but total across all segments
        return FinalizeResult(
            final_path=self._finalized_segments[0].final_path,
            total_rows=self._total_rows,
            total_bytes=self._total_bytes,
        )

    def abort(self) -> None:
        """Abort current segment and clean up all finalized segments."""
        if self._current_writer:
            self._current_writer.abort()
            self._current_writer = None

        # Clean up already-finalized segment files
        import os
        for seg in self._finalized_segments:
            try:
                if os.path.exists(seg.final_path):
                    os.remove(seg.final_path)
            except OSError:
                pass
        self._finalized_segments = []

    @property
    def segment_count(self) -> int:
        """Number of segments created (including current)."""
        count = len(self._finalized_segments)
        if self._current_writer is not None:
            count += 1
        return count

    @property
    def segment_paths(self) -> list[str]:
        """Paths of all finalized segments."""
        return [s.final_path for s in self._finalized_segments]

    # ── Internal ─────────────────────────────────────────────────

    def _open_segment(self) -> None:
        """Open a new segment writer."""
        self._current_writer = self._writer_class()
        segment_config = dict(self._config)

        # Modify naming pattern to include segment number
        if self._max_bytes is not None:
            original_pattern = segment_config.get(
                "naming_pattern", "{object}_{chunk_id}.parquet"
            )
            # Insert _partNNN before extension
            base, ext = _split_pattern_ext(original_pattern)
            segment_config["naming_pattern"] = (
                f"{base}_part{self._segment:03d}.{ext}"
            )

        self._current_writer.open(segment_config, self._chunk_id, self._schema)
        self._current_bytes = 0

    def _rotate(self) -> None:
        """Finalize current segment and open a new one."""
        if self._current_writer:
            seg_result = self._current_writer.finalize()
            self._finalized_segments.append(seg_result)
            self._total_bytes += seg_result.total_bytes

        self._segment += 1
        self._open_segment()


def _split_pattern_ext(pattern: str) -> tuple[str, str]:
    """Split '{object}_{chunk_id}.parquet' → ('{object}_{chunk_id}', 'parquet')."""
    if "." in pattern:
        parts = pattern.rsplit(".", 1)
        return parts[0], parts[1]
    return pattern, "dat"
