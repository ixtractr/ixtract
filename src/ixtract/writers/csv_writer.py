"""CSV writer — temp file + atomic rename, idempotent finalize.

Follows the same open/write_batch/finalize/abort contract as ParquetWriter.
One instance per chunk per worker — never shared between workers.

Thread-safety invariant (inherited from BaseWriter contract):
    Writer instances are NOT shared between workers. Each worker creates its
    own writer instance for each chunk. The only shared resource between
    workers is the chunk queue and the snapshot connection — never the writer.

File visibility invariant:
    The output file is invisible at its final path until finalize() succeeds.
    abort() removes the temp file. Re-execution is safe: abort() then
    re-execute from scratch. No duplicates possible.

Configuration keys (passed via config dict in open()):
    output_path:     Directory for output files. Created if absent.
    object_name:     Table/object name used in filename.
    naming_pattern:  Filename template. Default: {object}_{chunk_id}.csv
    delimiter:       Field delimiter. Default: ","
    null_value:      String to write for None/null values. Default: ""
    encoding:        File encoding. Default: "utf-8"
    include_header:  Whether to write a header row. Default: True

Schema handling:
    If schema is provided to open(), headers are written immediately.
    If schema is None, headers are inferred from the first batch's keys.
    Column order is preserved from schema if provided, else from dict
    insertion order of the first batch (Python 3.7+).
"""
from __future__ import annotations

import csv
import os
import tempfile
from typing import Any, Optional

from ixtract.writers.parquet import BaseWriter, WriteResult, FinalizeResult


class CSVWriter(BaseWriter):
    """CSV writer. File only appears at final path after finalize().

    Deferred (Phase 3):
        - File rotation (splitting output at size threshold)
        - Partitioned output
        - Compressed CSV archives
    """

    def __init__(self) -> None:
        self._temp_path: Optional[str] = None
        self._final_path: Optional[str] = None
        self._file: Optional[Any] = None          # open file handle
        self._writer: Optional[csv.DictWriter] = None
        self._total_rows: int = 0
        self._total_bytes: int = 0
        self._columns: Optional[list[str]] = None
        self._delimiter: str = ","
        self._null_value: str = ""
        self._encoding: str = "utf-8"
        self._include_header: bool = True

    def open(
        self,
        config: dict[str, Any],
        chunk_id: str,
        schema: Optional[list[dict[str, str]]] = None,  # [{name, type}, ...] or None
    ) -> None:
        """Open a temp file for this chunk. Header written immediately if schema provided.

        Args:
            config:   Writer configuration dict.
            chunk_id: Unique chunk identifier — used in output filename.
            schema:   Optional column schema. If provided, headers are written
                      now and column order is fixed. If None, headers are
                      inferred from the first batch (first-batch-determines-order).
        """
        output_dir = config.get("output_path", "./output")
        os.makedirs(output_dir, exist_ok=True)

        self._delimiter = config.get("delimiter", ",")
        self._null_value = config.get("null_value", "")
        self._encoding = config.get("encoding", "utf-8")
        self._include_header = config.get("include_header", True)

        obj_name = config.get("object_name", "data")
        naming = config.get("naming_pattern", "{object}_{chunk_id}.csv")
        filename = naming.format(object=obj_name, chunk_id=chunk_id, format="csv")
        self._final_path = os.path.join(output_dir, filename)

        # Temp file in same directory — os.replace() fails across filesystems
        fd, self._temp_path = tempfile.mkstemp(suffix=".csv.tmp", dir=output_dir)
        os.close(fd)

        self._total_rows = 0
        self._total_bytes = 0
        self._file = open(self._temp_path, "w", encoding=self._encoding, newline="")

        # If schema provided, establish column order and write header now
        if schema:
            self._columns = [col["name"] for col in schema]
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._columns,
                delimiter=self._delimiter,
                extrasaction="ignore",  # silently drop extra fields
            )
            if self._include_header:
                self._writer.writeheader()
                self._file.flush()

    def write_batch(self, batch: list[dict[str, Any]]) -> WriteResult:
        """Write a batch of rows. Initialises columns from first batch if schema was None."""
        if not batch:
            return WriteResult(0, 0)

        # First batch: initialise writer if schema was not provided at open()
        if self._writer is None:
            self._columns = list(batch[0].keys())
            self._writer = csv.DictWriter(
                self._file,
                fieldnames=self._columns,
                delimiter=self._delimiter,
                extrasaction="ignore",
            )
            if self._include_header:
                self._writer.writeheader()

        # Normalise nulls and write rows
        rows_before = self._file.tell()
        for row in batch:
            normalised = {
                k: (self._null_value if v is None else v)
                for k, v in row.items()
            }
            self._writer.writerow(normalised)

        self._file.flush()
        bytes_written = self._file.tell() - rows_before
        self._total_rows += len(batch)
        self._total_bytes += bytes_written
        return WriteResult(len(batch), bytes_written)

    def finalize(self) -> FinalizeResult:
        """Flush, close, and atomically rename temp file to final path."""
        if self._file:
            self._file.flush()
            self._file.close()
            self._file = None
            self._writer = None

        if self._temp_path and self._final_path:
            os.replace(self._temp_path, self._final_path)
            actual_bytes = os.path.getsize(self._final_path)
            path = self._final_path
            self._temp_path = None
            return FinalizeResult(path, self._total_rows, actual_bytes)

        raise RuntimeError("CSVWriter not properly initialised — call open() first.")

    def abort(self) -> None:
        """Clean up temp file. Safe to call multiple times. Leaves no partial output."""
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
            self._writer = None
        if self._temp_path and os.path.exists(self._temp_path):
            try:
                os.remove(self._temp_path)
            except OSError:
                pass
            self._temp_path = None
