"""Extraction Manifest Tests — Phase 2C.

Tests for:
  - ManifestFile and ExtractionManifest dataclasses
  - compute_schema_hash: fingerprint properties
  - compute_file_checksum: SHA-256 correctness
  - build_manifest: accumulate-then-build, failed chunks excluded
  - write_manifest: atomic temp+rename, always written (complete and failed)
  - load_manifest: round-trip serialization
  - Failure semantics: failed runs still produce a manifest
  - Schema field null in Phase 2C
  - Null fields for deferred features (extraction_window, freshness)

Run: python -m unittest tests.simulation.test_manifest -v
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.manifest import (
    ManifestFile, ExtractionManifest,
    compute_file_checksum, compute_schema_hash,
    build_manifest, write_manifest, load_manifest,
)


# ── Helpers ───────────────────────────────────────────────────────────

def make_chunk_result(
    chunk_id: str,
    status: str = "success",
    rows: int = 300_000,
    output_path: str = "",
    bytes_written: int = 45_000_000,
):
    """Minimal ChunkResult-like object for testing."""
    class _CR:
        pass
    cr = _CR()
    cr.chunk_id = chunk_id
    cr.status = status
    cr.rows = rows
    cr.output_path = output_path
    cr.bytes_written = bytes_written
    return cr


def write_temp_file(tmp_dir: str, name: str, content: bytes = b"data") -> str:
    path = os.path.join(tmp_dir, name)
    with open(path, "wb") as f:
        f.write(content)
    return path


# ── ManifestFile ──────────────────────────────────────────────────────

class TestManifestFile(unittest.TestCase):

    def test_to_dict_round_trip(self):
        mf = ManifestFile(
            path="/output/orders_chunk_001.parquet",
            size_bytes=45_000_000,
            row_count=300_000,
            checksum="abc123",
            chunk_id="chunk_001",
        )
        d = mf.to_dict()
        self.assertEqual(d["path"], "/output/orders_chunk_001.parquet")
        self.assertEqual(d["row_count"], 300_000)
        self.assertEqual(d["checksum"], "abc123")
        self.assertEqual(d["chunk_id"], "chunk_001")

    def test_all_fields_present(self):
        mf = ManifestFile("/p", 100, 10, "sha256", "c1")
        d = mf.to_dict()
        for key in ("path", "size_bytes", "row_count", "checksum", "chunk_id"):
            self.assertIn(key, d)


# ── ExtractionManifest ────────────────────────────────────────────────

class TestExtractionManifest(unittest.TestCase):

    def _make_manifest(self, status="complete", n_files=2):
        files = [
            ManifestFile(f"/out/chunk_{i:03d}.parquet", 45_000_000, 300_000, f"sha{i}", f"chunk_{i:03d}")
            for i in range(1, n_files + 1)
        ]
        return ExtractionManifest(
            manifest_id="mfst-001",
            run_id="rx-20260410",
            source="postgresql::orders",
            status=status,
            files=files,
            total_rows=n_files * 300_000,
            total_bytes=n_files * 45_000_000,
            schema=None,
            schema_hash="abc123schema",
            extraction_window=None,
            freshness=None,
            created_at="2026-04-10T12:00:00+00:00",
        )

    def test_schema_is_null(self):
        """Schema field must be null in Phase 2C — not faked."""
        m = self._make_manifest()
        self.assertIsNone(m.schema)
        self.assertIsNone(m.to_dict()["schema"])

    def test_deferred_fields_null(self):
        """extraction_window and freshness are null until implemented."""
        m = self._make_manifest()
        d = m.to_dict()
        self.assertIsNone(d["extraction_window"])
        self.assertIsNone(d["freshness"])

    def test_file_count_property(self):
        m = self._make_manifest(n_files=4)
        self.assertEqual(m.file_count, 4)

    def test_json_round_trip(self):
        m = self._make_manifest()
        restored = ExtractionManifest.from_dict(json.loads(m.to_json()))
        self.assertEqual(restored.manifest_id, m.manifest_id)
        self.assertEqual(restored.run_id, m.run_id)
        self.assertEqual(restored.status, m.status)
        self.assertEqual(restored.total_rows, m.total_rows)
        self.assertEqual(len(restored.files), len(m.files))
        self.assertEqual(restored.files[0].checksum, m.files[0].checksum)

    def test_to_json_is_valid_json(self):
        m = self._make_manifest()
        parsed = json.loads(m.to_json())
        self.assertIsInstance(parsed, dict)
        self.assertIn("manifest_id", parsed)

    def test_status_complete(self):
        m = self._make_manifest(status="complete")
        self.assertEqual(m.status, "complete")

    def test_status_failed(self):
        m = self._make_manifest(status="failed")
        self.assertEqual(m.to_dict()["status"], "failed")


# ── Schema hash ───────────────────────────────────────────────────────

class TestComputeSchemaHash(unittest.TestCase):

    def test_same_inputs_same_hash(self):
        h1 = compute_schema_hash(10, "id", "integer", 145)
        h2 = compute_schema_hash(10, "id", "integer", 145)
        self.assertEqual(h1, h2)

    def test_different_column_count_different_hash(self):
        h1 = compute_schema_hash(10, "id", "integer", 145)
        h2 = compute_schema_hash(12, "id", "integer", 145)
        self.assertNotEqual(h1, h2)

    def test_different_pk_different_hash(self):
        h1 = compute_schema_hash(10, "id", "integer", 145)
        h2 = compute_schema_hash(10, "order_id", "integer", 145)
        self.assertNotEqual(h1, h2)

    def test_different_avg_row_bytes_different_hash(self):
        h1 = compute_schema_hash(10, "id", "integer", 145)
        h2 = compute_schema_hash(10, "id", "integer", 200)
        self.assertNotEqual(h1, h2)

    def test_none_pk_handled_gracefully(self):
        h = compute_schema_hash(10, None, None, 145)
        self.assertIsInstance(h, str)
        self.assertEqual(len(h), 64)  # SHA-256 hex

    def test_hash_is_sha256_length(self):
        h = compute_schema_hash(10, "id", "integer", 145)
        self.assertEqual(len(h), 64)


# ── File checksum ─────────────────────────────────────────────────────

class TestComputeFileChecksum(unittest.TestCase):

    def test_sha256_matches_stdlib(self):
        """compute_file_checksum must match hashlib.sha256 directly."""
        with tempfile.TemporaryDirectory() as tmp:
            content = b"test extraction output data" * 1000
            path = write_temp_file(tmp, "test.parquet", content)
            expected = hashlib.sha256(content).hexdigest()
            actual = compute_file_checksum(path)
            self.assertEqual(actual, expected)

    def test_different_files_different_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = write_temp_file(tmp, "a.parquet", b"data_a")
            p2 = write_temp_file(tmp, "b.parquet", b"data_b")
            self.assertNotEqual(
                compute_file_checksum(p1),
                compute_file_checksum(p2),
            )

    def test_empty_file_has_known_checksum(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_temp_file(tmp, "empty.parquet", b"")
            expected = hashlib.sha256(b"").hexdigest()
            self.assertEqual(compute_file_checksum(path), expected)

    def test_large_file_chunks_correctly(self):
        """Files larger than the 64KB read buffer are hashed correctly."""
        with tempfile.TemporaryDirectory() as tmp:
            content = os.urandom(200_000)  # 200KB
            path = write_temp_file(tmp, "large.parquet", content)
            expected = hashlib.sha256(content).hexdigest()
            self.assertEqual(compute_file_checksum(path), expected)


# ── build_manifest ────────────────────────────────────────────────────

class TestBuildManifest(unittest.TestCase):

    def test_successful_run_status_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            p1 = write_temp_file(tmp, "chunk_001.parquet", b"data1" * 1000)
            p2 = write_temp_file(tmp, "chunk_002.parquet", b"data2" * 1000)
            chunks = [
                make_chunk_result("chunk_001", "success", 300_000, p1),
                make_chunk_result("chunk_002", "success", 300_000, p2),
            ]
            m = build_manifest("rx-001", "postgresql", "orders", "success",
                               chunks, 600_000, 90_000_000, 10, "id", "integer", 145)
            self.assertEqual(m.status, "success")
            self.assertEqual(len(m.files), 2)
            self.assertEqual(m.total_rows, 600_000)

    def test_failed_chunks_excluded_from_files(self):
        """Failed chunks produce no output file — must not appear in manifest."""
        with tempfile.TemporaryDirectory() as tmp:
            p1 = write_temp_file(tmp, "chunk_001.parquet", b"data1")
            chunks = [
                make_chunk_result("chunk_001", "success", 300_000, p1),
                make_chunk_result("chunk_002", "failed",  0, ""),
            ]
            m = build_manifest("rx-001", "postgresql", "orders", "failed",
                               chunks, 300_000, 45_000_000, 10, "id", "integer", 145)
            self.assertEqual(len(m.files), 1)
            self.assertEqual(m.files[0].chunk_id, "chunk_001")

    def test_manifest_written_for_failed_run(self):
        """Failed runs must still produce a manifest (diagnostic artifact)."""
        with tempfile.TemporaryDirectory() as tmp:
            chunks = [make_chunk_result("chunk_001", "failed", 0, "")]
            m = build_manifest("rx-001", "postgresql", "orders", "failed",
                               chunks, 0, 0, 10, "id", "integer", 145)
            self.assertEqual(m.status, "failed")
            # No files (all failed), but manifest exists
            self.assertEqual(len(m.files), 0)
            # Write and confirm it exists
            path = write_manifest(m, tmp)
            self.assertTrue(os.path.exists(path))

    def test_source_field_format(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks = []
            m = build_manifest("rx-001", "postgresql", "orders", "success",
                               chunks, 0, 0, 10, "id", "integer", 145)
            self.assertEqual(m.source, "postgresql::orders")

    def test_schema_is_null(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = build_manifest("rx-001", "postgresql", "orders", "success",
                               [], 0, 0, 10, "id", "integer", 145)
            self.assertIsNone(m.schema)

    def test_schema_hash_is_populated(self):
        m = build_manifest("rx-001", "postgresql", "orders", "success",
                           [], 0, 0, 10, "id", "integer", 145)
        self.assertIsNotNone(m.schema_hash)
        self.assertEqual(len(m.schema_hash), 64)

    def test_file_checksums_computed(self):
        """Each file entry has a valid SHA-256 checksum."""
        with tempfile.TemporaryDirectory() as tmp:
            content = b"parquet data" * 1000
            p = write_temp_file(tmp, "chunk_001.parquet", content)
            chunks = [make_chunk_result("chunk_001", "success", 300_000, p)]
            m = build_manifest("rx-001", "postgresql", "orders", "success",
                               chunks, 300_000, len(content), 10, "id", "integer", 145)
            expected_checksum = hashlib.sha256(content).hexdigest()
            self.assertEqual(m.files[0].checksum, expected_checksum)

    def test_manifest_id_is_unique(self):
        m1 = build_manifest("rx-001", "postgresql", "orders", "success",
                             [], 0, 0, 10, "id", "integer", 145)
        m2 = build_manifest("rx-001", "postgresql", "orders", "success",
                             [], 0, 0, 10, "id", "integer", 145)
        self.assertNotEqual(m1.manifest_id, m2.manifest_id)


# ── write_manifest and load_manifest ─────────────────────────────────

class TestWriteAndLoadManifest(unittest.TestCase):

    def _make_simple_manifest(self, status="complete") -> ExtractionManifest:
        return build_manifest(
            "rx-001", "postgresql", "orders", status,
            [], 0, 0, 10, "id", "integer", 145,
        )

    def test_written_to_manifest_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._make_simple_manifest()
            path = write_manifest(m, tmp)
            self.assertEqual(os.path.basename(path), "_manifest.json")
            self.assertTrue(os.path.exists(path))

    def test_atomic_write_no_temp_files_remaining(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._make_simple_manifest()
            write_manifest(m, tmp)
            tmp_files = [f for f in os.listdir(tmp) if f.endswith(".tmp")]
            self.assertEqual(tmp_files, [])

    def test_load_manifest_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._make_simple_manifest()
            write_manifest(m, tmp)
            loaded = load_manifest(tmp)
            self.assertIsNotNone(loaded)
            self.assertEqual(loaded.manifest_id, m.manifest_id)
            self.assertEqual(loaded.run_id, m.run_id)
            self.assertEqual(loaded.status, m.status)
            self.assertEqual(loaded.schema_hash, m.schema_hash)

    def test_load_manifest_returns_none_if_absent(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = load_manifest(tmp)
            self.assertIsNone(result)

    def test_write_creates_output_dir(self):
        """write_manifest creates output_path if it doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b")
            m = self._make_simple_manifest()
            write_manifest(m, nested)
            self.assertTrue(os.path.exists(os.path.join(nested, "_manifest.json")))

    def test_failed_manifest_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = self._make_simple_manifest(status="failed")
            path = write_manifest(m, tmp)
            with open(path, encoding="utf-8") as f:
                parsed = json.load(f)
            self.assertEqual(parsed["status"], "failed")


if __name__ == "__main__":
    unittest.main(verbosity=2)
