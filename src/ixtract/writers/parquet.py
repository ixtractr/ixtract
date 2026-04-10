"""Parquet writer — temp file + atomic rename, idempotent finalize."""
from __future__ import annotations

import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class WriteResult:
    rows_written: int
    bytes_written: int


@dataclass
class FinalizeResult:
    final_path: str
    total_rows: int
    total_bytes: int


class BaseWriter(ABC):
    """Abstract writer interface — open/write_batch/finalize/abort.

    Thread-safety invariant:
        Writer instances are NOT shared between workers. Each worker creates
        its own writer instance per chunk. Workers share only the chunk queue
        and the snapshot connection — never the writer. Implementations may
        assume single-threaded access.

    File visibility invariant:
        Output is invisible at its final path until finalize() succeeds.
        abort() removes all partial state. Re-execution: abort() then restart
        from scratch. No duplicates possible.

    Deferred (Phase 3):
        - File rotation / size-based splitting
        - Partitioned output
        - Compressed archives
    """

    @abstractmethod
    def open(
        self,
        config: dict[str, Any],
        chunk_id: str,
        schema: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """Open writer for a single chunk.

        Args:
            config:   Writer configuration (output_path, compression, etc.)
            chunk_id: Unique chunk identifier used in output filename.
            schema:   Optional column schema [{name: str, type: str}, ...].
                      If None, schema is inferred from first write_batch call.
        """
        ...

    @abstractmethod
    def write_batch(self, batch: list[dict[str, Any]]) -> WriteResult: ...

    @abstractmethod
    def finalize(self) -> FinalizeResult: ...

    @abstractmethod
    def abort(self) -> None: ...


class ParquetWriter(BaseWriter):
    """Parquet writer. File only appears at final path after finalize()."""

    def __init__(self) -> None:
        self._temp_path: Optional[str] = None
        self._final_path: Optional[str] = None
        self._pa_writer: Any = None
        self._total_rows = 0
        self._compression = "snappy"

    def open(
        self,
        config: dict[str, Any],
        chunk_id: str,
        schema: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """Open writer. schema is accepted but Parquet infers types from first batch."""
        output_dir = config.get("output_path", "./output")
        os.makedirs(output_dir, exist_ok=True)

        obj_name = config.get("object_name", "data")
        self._compression = config.get("compression", "snappy")
        naming = config.get("naming_pattern", "{object}_{chunk_id}.parquet")
        filename = naming.format(object=obj_name, chunk_id=chunk_id, format="parquet")
        self._final_path = os.path.join(output_dir, filename)

        # CRITICAL: temp file MUST be in the same directory as final output.
        # os.replace() (atomic rename) fails across filesystem boundaries.
        fd, self._temp_path = tempfile.mkstemp(suffix=".parquet.tmp", dir=output_dir)
        os.close(fd)

        self._total_rows = 0
        self._pa_writer = None

    def write_batch(self, batch: list[dict[str, Any]]) -> WriteResult:
        import pyarrow as pa
        import pyarrow.parquet as pq

        if not batch:
            return WriteResult(0, 0)

        table = pa.Table.from_pylist(batch)

        if self._pa_writer is None:
            self._pa_writer = pq.ParquetWriter(
                self._temp_path, table.schema, compression=self._compression,
            )

        self._pa_writer.write_table(table)
        rows = len(batch)
        self._total_rows += rows
        return WriteResult(rows, table.nbytes)

    def finalize(self) -> FinalizeResult:
        if self._pa_writer:
            self._pa_writer.close()
            self._pa_writer = None

        if self._temp_path and self._final_path:
            os.replace(self._temp_path, self._final_path)
            actual_bytes = os.path.getsize(self._final_path)
            path = self._final_path
            self._temp_path = None
            return FinalizeResult(path, self._total_rows, actual_bytes)

        raise RuntimeError("Writer not properly initialized.")

    def abort(self) -> None:
        if self._pa_writer:
            try:
                self._pa_writer.close()
            except Exception:
                pass
            self._pa_writer = None
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except OSError:
                pass
            self._temp_path = None
