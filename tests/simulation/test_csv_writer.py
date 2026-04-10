"""CSV Writer Tests — Phase 2C.

Tests for:
  - open/write_batch/finalize/abort contract (same as ParquetWriter)
  - Schema-first vs batch-inferred headers
  - Null value handling
  - Atomic finalize (temp+rename)
  - abort() safety (idempotent, no partial files)
  - Thread-safety invariant (one instance per chunk)
  - Delimiter and encoding configuration
  - BaseWriter signature compatibility

Run: python -m unittest tests.simulation.test_csv_writer -v
"""
from __future__ import annotations

import csv
import os
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.writers.csv_writer import CSVWriter
from ixtract.writers.parquet import BaseWriter, WriteResult, FinalizeResult


# ── Helpers ───────────────────────────────────────────────────────────

def make_config(tmp_dir: str, **overrides) -> dict:
    cfg = {
        "output_path": tmp_dir,
        "object_name": "orders",
        "naming_pattern": "{object}_{chunk_id}.csv",
        "delimiter": ",",
        "null_value": "",
        "encoding": "utf-8",
        "include_header": True,
    }
    cfg.update(overrides)
    return cfg


SAMPLE_BATCH = [
    {"id": 1, "name": "Alice", "amount": 100.0},
    {"id": 2, "name": "Bob",   "amount": 200.0},
    {"id": 3, "name": "Carol", "amount": None},
]


# ── Interface compliance ──────────────────────────────────────────────

class TestBaseWriterCompliance(unittest.TestCase):
    """CSVWriter must implement BaseWriter exactly."""

    def test_is_base_writer_subclass(self):
        self.assertTrue(issubclass(CSVWriter, BaseWriter))

    def test_has_all_abstract_methods(self):
        writer = CSVWriter()
        for method in ("open", "write_batch", "finalize", "abort"):
            self.assertTrue(callable(getattr(writer, method)))

    def test_write_batch_returns_write_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001")
            result = w.write_batch(SAMPLE_BATCH)
            w.abort()
            self.assertIsInstance(result, WriteResult)
            self.assertEqual(result.rows_written, 3)
            self.assertGreater(result.bytes_written, 0)

    def test_finalize_returns_finalize_result(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001")
            w.write_batch(SAMPLE_BATCH)
            result = w.finalize()
            self.assertIsInstance(result, FinalizeResult)
            self.assertGreater(result.total_rows, 0)
            self.assertGreater(result.total_bytes, 0)


# ── Header handling ───────────────────────────────────────────────────

class TestHeaderHandling(unittest.TestCase):

    def _read_csv(self, path: str, delimiter: str = ",") -> list[dict]:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f, delimiter=delimiter))

    def test_header_from_first_batch(self):
        """Without schema, headers are inferred from first batch keys."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001")
            w.write_batch(SAMPLE_BATCH)
            result = w.finalize()
            rows = self._read_csv(result.final_path)
            self.assertEqual(list(rows[0].keys()), ["id", "name", "amount"])

    def test_header_from_schema(self):
        """With schema, headers written immediately at open() with correct order."""
        schema = [
            {"name": "amount", "type": "float"},
            {"name": "name",   "type": "string"},
            {"name": "id",     "type": "integer"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001", schema=schema)
            w.write_batch(SAMPLE_BATCH)
            result = w.finalize()
            rows = self._read_csv(result.final_path)
            # Schema order must be preserved, not batch key order
            self.assertEqual(list(rows[0].keys()), ["amount", "name", "id"])

    def test_no_header_when_disabled(self):
        """include_header=False writes no header row."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp, include_header=False), "chunk_001")
            w.write_batch(SAMPLE_BATCH)
            result = w.finalize()
            with open(result.final_path, encoding="utf-8") as f:
                first_line = f.readline().strip()
            # First line should be data, not header
            self.assertNotIn("id", first_line.split(",")[0])
            self.assertTrue(first_line[0].isdigit())

    def test_multiple_batches_no_duplicate_header(self):
        """Writing multiple batches produces exactly one header row."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001")
            w.write_batch(SAMPLE_BATCH)
            w.write_batch(SAMPLE_BATCH)
            result = w.finalize()
            with open(result.final_path, encoding="utf-8") as f:
                lines = [l for l in f.readlines() if l.strip()]
            header_lines = [l for l in lines if "id,name" in l or "id" == l.split(",")[0]]
            self.assertEqual(len(header_lines), 1)
            # Total: 1 header + 6 data rows
            self.assertEqual(len(lines), 7)


# ── Null handling ─────────────────────────────────────────────────────

class TestNullHandling(unittest.TestCase):

    def _read_raw(self, path: str) -> list[list[str]]:
        with open(path, encoding="utf-8", newline="") as f:
            return list(csv.reader(f))

    def test_none_written_as_empty_string_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001")
            w.write_batch(SAMPLE_BATCH)
            result = w.finalize()
            rows = self._read_raw(result.final_path)
            # Row 3 (Carol) has amount=None → should be empty string
            self.assertEqual(rows[3][2], "")

    def test_custom_null_value(self):
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp, null_value="NULL"), "chunk_001")
            w.write_batch(SAMPLE_BATCH)
            result = w.finalize()
            rows = self._read_raw(result.final_path)
            self.assertEqual(rows[3][2], "NULL")


# ── Atomicity and idempotency ─────────────────────────────────────────

class TestAtomicity(unittest.TestCase):

    def test_output_invisible_before_finalize(self):
        """Final path must not exist until finalize() is called."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            cfg = make_config(tmp)
            w.open(cfg, "chunk_001")
            final_path = os.path.join(tmp, "orders_chunk_001.csv")
            self.assertFalse(os.path.exists(final_path))
            w.write_batch(SAMPLE_BATCH)
            self.assertFalse(os.path.exists(final_path))
            w.finalize()
            self.assertTrue(os.path.exists(final_path))

    def test_abort_removes_temp_file(self):
        """abort() must leave no partial files."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001")
            w.write_batch(SAMPLE_BATCH)
            w.abort()
            # No .tmp files left
            tmp_files = [f for f in os.listdir(tmp) if f.endswith(".tmp")]
            self.assertEqual(tmp_files, [])

    def test_abort_is_idempotent(self):
        """abort() can be called multiple times safely."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001")
            w.write_batch(SAMPLE_BATCH)
            w.abort()
            w.abort()  # second call must not raise

    def test_abort_after_no_batches(self):
        """abort() on an opened but unwritten writer is safe."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001")
            w.abort()  # must not raise

    def test_temp_file_in_same_directory(self):
        """Temp file must be in same dir as output (cross-filesystem rename safety)."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001")
            # While open, a .tmp file should exist in output dir
            tmp_files = [f for f in os.listdir(tmp) if f.endswith(".csv.tmp")]
            self.assertEqual(len(tmp_files), 1)
            w.abort()

    def test_empty_batch_write(self):
        """write_batch([]) returns 0 rows, 0 bytes without error."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001")
            result = w.write_batch([])
            self.assertEqual(result.rows_written, 0)
            self.assertEqual(result.bytes_written, 0)
            w.abort()


# ── Configuration ─────────────────────────────────────────────────────

class TestConfiguration(unittest.TestCase):

    def test_custom_delimiter(self):
        """Tab delimiter produces tab-separated output."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp, delimiter="\t"), "chunk_001")
            w.write_batch(SAMPLE_BATCH)
            result = w.finalize()
            with open(result.final_path, encoding="utf-8") as f:
                first_line = f.readline()
            self.assertIn("\t", first_line)
            self.assertNotIn(",", first_line)

    def test_naming_pattern_applied(self):
        """Output filename follows naming_pattern."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp, naming_pattern="extract_{chunk_id}.csv"), "c001")
            w.write_batch(SAMPLE_BATCH)
            result = w.finalize()
            self.assertEqual(os.path.basename(result.final_path), "extract_c001.csv")

    def test_output_dir_created_if_absent(self):
        """output_path is created if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c")
            w = CSVWriter()
            w.open(make_config(nested), "chunk_001")
            w.write_batch(SAMPLE_BATCH)
            w.finalize()
            self.assertTrue(os.path.isdir(nested))

    def test_finalize_result_row_count_correct(self):
        """FinalizeResult.total_rows matches actual rows written."""
        with tempfile.TemporaryDirectory() as tmp:
            w = CSVWriter()
            w.open(make_config(tmp), "chunk_001")
            w.write_batch(SAMPLE_BATCH)
            w.write_batch(SAMPLE_BATCH[:1])
            result = w.finalize()
            self.assertEqual(result.total_rows, 4)


# ── Thread-safety invariant ───────────────────────────────────────────

class TestThreadSafetyInvariant(unittest.TestCase):
    """Verify that multiple independent writer instances don't interfere.

    The invariant: one writer instance per chunk, per worker.
    Each worker creates its own CSVWriter — they never share one.
    This test runs two writers concurrently to confirm independence.
    """

    def test_concurrent_independent_writers(self):
        """Two writers running concurrently in separate threads produce correct output."""
        results = {}
        errors = []

        def write_chunk(chunk_id: str, tmp_dir: str) -> None:
            try:
                w = CSVWriter()
                w.open(make_config(tmp_dir), chunk_id)
                w.write_batch([{"id": int(chunk_id[-1]), "val": chunk_id}])
                result = w.finalize()
                results[chunk_id] = result
            except Exception as e:
                errors.append(e)

        with tempfile.TemporaryDirectory() as tmp:
            t1 = threading.Thread(target=write_chunk, args=("chunk_001", tmp))
            t2 = threading.Thread(target=write_chunk, args=("chunk_002", tmp))
            t1.start(); t2.start()
            t1.join();  t2.join()

            self.assertEqual(errors, [])
            self.assertIn("chunk_001", results)
            self.assertIn("chunk_002", results)
            self.assertTrue(os.path.exists(results["chunk_001"].final_path))
            self.assertTrue(os.path.exists(results["chunk_002"].final_path))
            # Each file has exactly 1 data row
            for chunk_id, res in results.items():
                with open(res.final_path, encoding="utf-8") as f:
                    data_rows = [l for l in f.readlines() if l.strip()][1:]  # skip header
                self.assertEqual(len(data_rows), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
