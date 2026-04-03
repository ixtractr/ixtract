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
    @abstractmethod
    def open(self, config: dict[str, Any], chunk_id: str, columns: list[str]) -> None: ...
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

    def open(self, config: dict[str, Any], chunk_id: str, columns: list[str]) -> None:
        output_dir = config.get("output_path", "./output")
        os.makedirs(output_dir, exist_ok=True)

        obj_name = config.get("object_name", "data")
        self._compression = config.get("compression", "snappy")
        naming = config.get("naming_pattern", "{object}_{chunk_id}.parquet")
        filename = naming.format(object=obj_name, chunk_id=chunk_id, format="parquet")
        self._final_path = os.path.join(output_dir, filename)

        temp_dir = config.get("temp_path", output_dir)
        os.makedirs(temp_dir, exist_ok=True)
        fd, self._temp_path = tempfile.mkstemp(suffix=".parquet.tmp", dir=temp_dir)
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
