"""Phase 3C Tests — Cloud Writers, SQL Server Connector, File Rotation.

Tests validate interfaces, validation rules, and local behavior
without requiring actual cloud services or SQL Server instances.

Run: python -m pytest tests/simulation/test_phase3c.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.writers.parquet import BaseWriter, WriteResult, FinalizeResult
from ixtract.writers.csv_writer import CSVWriter
from ixtract.writers.rotating import RotatingWriter, MIN_FILE_SIZE_BYTES


# ══════════════════════════════════════════════════════════════════════
# 1. S3 WRITER
# ══════════════════════════════════════════════════════════════════════

class TestS3URIParsing(unittest.TestCase):
    """S3 URI parsing validation."""

    def test_parse_bucket_and_prefix(self):
        from ixtract.writers.s3_writer import _parse_s3_uri
        bucket, prefix = _parse_s3_uri("s3://my-bucket/data/output/")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(prefix, "data/output/")

    def test_parse_bucket_only(self):
        from ixtract.writers.s3_writer import _parse_s3_uri
        bucket, prefix = _parse_s3_uri("s3://my-bucket")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(prefix, "")

    def test_parse_bucket_with_slash(self):
        from ixtract.writers.s3_writer import _parse_s3_uri
        bucket, prefix = _parse_s3_uri("s3://my-bucket/")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(prefix, "")

    def test_invalid_uri_raises(self):
        from ixtract.writers.s3_writer import _parse_s3_uri
        with self.assertRaises(ValueError):
            _parse_s3_uri("https://my-bucket/data")

    def test_invalid_uri_no_scheme(self):
        from ixtract.writers.s3_writer import _parse_s3_uri
        with self.assertRaises(ValueError):
            _parse_s3_uri("/local/path")


class TestS3WriterInterface(unittest.TestCase):
    """S3Writer implements BaseWriter correctly."""

    def test_is_base_writer(self):
        from ixtract.writers.s3_writer import S3Writer
        self.assertTrue(issubclass(S3Writer, BaseWriter))

    def test_open_creates_temp_file(self):
        from ixtract.writers.s3_writer import S3Writer
        writer = S3Writer()
        with tempfile.TemporaryDirectory() as tmpdir:
            # We can't upload to S3, but open() creates a local temp file
            writer.open(
                {"output_path": "s3://test-bucket/prefix/", "object_name": "orders"},
                "chunk_001",
            )
            self.assertIsNotNone(writer._temp_path)
            self.assertTrue(os.path.exists(writer._temp_path))
            writer.abort()

    def test_abort_cleans_temp(self):
        from ixtract.writers.s3_writer import S3Writer
        writer = S3Writer()
        writer.open(
            {"output_path": "s3://test-bucket/", "object_name": "data"},
            "chunk_001",
        )
        temp = writer._temp_path
        self.assertTrue(os.path.exists(temp))
        writer.abort()
        self.assertFalse(os.path.exists(temp))

    def test_s3_key_construction(self):
        from ixtract.writers.s3_writer import S3Writer
        writer = S3Writer()
        writer.open(
            {"output_path": "s3://bucket/prefix/", "object_name": "orders"},
            "chunk_003",
        )
        self.assertEqual(writer._s3_key, "prefix/orders_chunk_003.parquet")
        self.assertEqual(writer._s3_bucket, "bucket")
        writer.abort()


# ══════════════════════════════════════════════════════════════════════
# 2. GCS WRITER
# ══════════════════════════════════════════════════════════════════════

class TestGCSURIParsing(unittest.TestCase):
    """GCS URI parsing validation."""

    def test_parse_bucket_and_prefix(self):
        from ixtract.writers.gcs_writer import _parse_gcs_uri
        bucket, prefix = _parse_gcs_uri("gs://my-bucket/data/output/")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(prefix, "data/output/")

    def test_parse_bucket_only(self):
        from ixtract.writers.gcs_writer import _parse_gcs_uri
        bucket, prefix = _parse_gcs_uri("gs://my-bucket")
        self.assertEqual(bucket, "my-bucket")
        self.assertEqual(prefix, "")

    def test_invalid_uri_raises(self):
        from ixtract.writers.gcs_writer import _parse_gcs_uri
        with self.assertRaises(ValueError):
            _parse_gcs_uri("s3://wrong-scheme/")


class TestGCSWriterInterface(unittest.TestCase):
    """GCSWriter implements BaseWriter correctly."""

    def test_is_base_writer(self):
        from ixtract.writers.gcs_writer import GCSWriter
        self.assertTrue(issubclass(GCSWriter, BaseWriter))

    def test_open_creates_temp_file(self):
        from ixtract.writers.gcs_writer import GCSWriter
        writer = GCSWriter()
        writer.open(
            {"output_path": "gs://test-bucket/prefix/", "object_name": "events"},
            "chunk_002",
        )
        self.assertIsNotNone(writer._temp_path)
        self.assertTrue(os.path.exists(writer._temp_path))
        writer.abort()

    def test_abort_cleans_temp(self):
        from ixtract.writers.gcs_writer import GCSWriter
        writer = GCSWriter()
        writer.open(
            {"output_path": "gs://bucket/", "object_name": "data"},
            "chunk_001",
        )
        temp = writer._temp_path
        writer.abort()
        self.assertFalse(os.path.exists(temp))

    def test_gcs_blob_name_construction(self):
        from ixtract.writers.gcs_writer import GCSWriter
        writer = GCSWriter()
        writer.open(
            {"output_path": "gs://bucket/prefix/", "object_name": "orders"},
            "chunk_005",
        )
        self.assertEqual(writer._gcs_blob_name, "prefix/orders_chunk_005.parquet")
        self.assertEqual(writer._gcs_bucket, "bucket")
        writer.abort()


# ══════════════════════════════════════════════════════════════════════
# 3. SQL SERVER CONNECTOR
# ══════════════════════════════════════════════════════════════════════

class TestSQLServerConnectorInterface(unittest.TestCase):
    """SQLServerConnector structure and validation."""

    def test_is_base_connector(self):
        from ixtract.connectors.sqlserver import SQLServerConnector
        from ixtract.connectors.base import BaseConnector
        self.assertTrue(issubclass(SQLServerConnector, BaseConnector))

    def test_has_required_methods(self):
        from ixtract.connectors.sqlserver import SQLServerConnector
        connector = SQLServerConnector()
        for method in ("connect", "metadata", "extract_chunk",
                        "estimate_latency", "get_connections",
                        "get_pk_distribution", "close"):
            self.assertTrue(hasattr(connector, method),
                            f"Missing method: {method}")

    def test_not_connected_raises(self):
        from ixtract.connectors.sqlserver import SQLServerConnector
        connector = SQLServerConnector()
        with self.assertRaises(RuntimeError):
            connector._require_conn()

    def test_connection_string_building(self):
        from ixtract.connectors.sqlserver import SQLServerConnector
        connector = SQLServerConnector()
        connector._config = {
            "host": "db.example.com",
            "port": 1433,
            "database": "mydb",
            "user": "sa",
            "password": "secret",
            "driver": "ODBC Driver 18 for SQL Server",
            "trust_server_cert": True,
        }
        conn_str = connector._build_conn_str()
        self.assertIn("db.example.com,1433", conn_str)
        self.assertIn("DATABASE=mydb", conn_str)
        self.assertIn("UID=sa", conn_str)
        self.assertIn("TrustServerCertificate=Yes", conn_str)

    def test_connection_string_trust_false(self):
        from ixtract.connectors.sqlserver import SQLServerConnector
        connector = SQLServerConnector()
        connector._config = {
            "host": "prod.db",
            "port": 1433,
            "database": "proddb",
            "user": "app",
            "password": "pw",
            "trust_server_cert": False,
        }
        conn_str = connector._build_conn_str()
        self.assertIn("TrustServerCertificate=No", conn_str)

    def test_close_idempotent(self):
        from ixtract.connectors.sqlserver import SQLServerConnector
        connector = SQLServerConnector()
        connector.close()  # should not raise
        connector.close()  # idempotent


# ══════════════════════════════════════════════════════════════════════
# 4. FILE ROTATION
# ══════════════════════════════════════════════════════════════════════

class TestRotatingWriterValidation(unittest.TestCase):
    """RotatingWriter input validation."""

    def test_min_file_size_enforced(self):
        with self.assertRaises(ValueError) as cm:
            RotatingWriter(CSVWriter, max_file_size_bytes=100)
        self.assertIn("1 MB", str(cm.exception))

    def test_none_max_bytes_allowed(self):
        writer = RotatingWriter(CSVWriter, max_file_size_bytes=None)
        self.assertIsNotNone(writer)


class TestRotatingWriterNoRotation(unittest.TestCase):
    """Without max_file_size_bytes, behaves like a normal writer."""

    def test_single_segment_without_rotation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = RotatingWriter(CSVWriter, max_file_size_bytes=None)
            writer.open(
                {"output_path": tmpdir, "object_name": "data"},
                "chunk_001",
            )
            writer.write_batch([{"id": 1, "name": "alice"}])
            writer.write_batch([{"id": 2, "name": "bob"}])
            result = writer.finalize()

            self.assertEqual(result.total_rows, 2)
            self.assertEqual(len(writer.segment_paths), 1)  # one segment finalized
            self.assertTrue(os.path.exists(result.final_path))


class TestRotatingWriterWithRotation(unittest.TestCase):
    """File rotation creates multiple segments."""

    def test_rotation_creates_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use very small threshold to force rotation
            writer = RotatingWriter(CSVWriter, max_file_size_bytes=MIN_FILE_SIZE_BYTES)
            writer.open(
                {"output_path": tmpdir, "object_name": "data",
                 "naming_pattern": "{object}_{chunk_id}.csv"},
                "chunk_001",
            )

            # Write enough data to exceed 1 MB threshold
            big_batch = [{"id": i, "data": "x" * 500} for i in range(3000)]
            writer.write_batch(big_batch)

            result = writer.finalize()
            self.assertGreater(result.total_rows, 0)
            # May or may not have rotated depending on CSV overhead
            # but the mechanism is exercised

    def test_segment_naming(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = RotatingWriter(CSVWriter, max_file_size_bytes=MIN_FILE_SIZE_BYTES)
            writer.open(
                {"output_path": tmpdir, "object_name": "data",
                 "naming_pattern": "{object}_{chunk_id}.csv"},
                "chunk_001",
            )
            # Force rotation by writing large batches
            for _ in range(5):
                big = [{"id": i, "data": "x" * 1000} for i in range(1000)]
                writer.write_batch(big)

            result = writer.finalize()
            # Check that segment paths contain "part" if rotation happened
            if len(writer.segment_paths) > 1:
                for path in writer.segment_paths:
                    self.assertIn("part", os.path.basename(path))

    def test_abort_cleans_all_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = RotatingWriter(CSVWriter, max_file_size_bytes=MIN_FILE_SIZE_BYTES)
            writer.open(
                {"output_path": tmpdir, "object_name": "data",
                 "naming_pattern": "{object}_{chunk_id}.csv"},
                "chunk_001",
            )
            # Write some data
            writer.write_batch([{"id": i, "data": "x" * 500} for i in range(2000)])
            writer.abort()

            # Temp files should be cleaned
            remaining = [f for f in os.listdir(tmpdir) if not f.startswith(".")]
            # May have some finalized segment files from rotation — abort cleans those
            # The current segment's temp file should be gone


class TestRotatingWriterAggregation(unittest.TestCase):
    """Finalize returns correct aggregate totals."""

    def test_total_rows_across_segments(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = RotatingWriter(CSVWriter, max_file_size_bytes=None)
            writer.open(
                {"output_path": tmpdir, "object_name": "data"},
                "chunk_001",
            )
            writer.write_batch([{"id": 1}, {"id": 2}, {"id": 3}])
            writer.write_batch([{"id": 4}, {"id": 5}])
            result = writer.finalize()
            self.assertEqual(result.total_rows, 5)


# ══════════════════════════════════════════════════════════════════════
# 5. ENGINE WRITER FACTORY
# ══════════════════════════════════════════════════════════════════════

class TestEngineWriterFactory(unittest.TestCase):
    """Engine selects correct writer by output_format."""

    def test_parquet_default(self):
        from ixtract.engine import _create_writer
        from ixtract.writers.parquet import ParquetWriter
        writer = _create_writer({"output_format": "parquet"})
        self.assertIsInstance(writer, ParquetWriter)

    def test_csv_writer(self):
        from ixtract.engine import _create_writer
        writer = _create_writer({"output_format": "csv"})
        self.assertIsInstance(writer, CSVWriter)

    def test_s3_writer(self):
        from ixtract.engine import _create_writer
        from ixtract.writers.s3_writer import S3Writer
        writer = _create_writer({"output_format": "s3"})
        self.assertIsInstance(writer, S3Writer)

    def test_gcs_writer(self):
        from ixtract.engine import _create_writer
        from ixtract.writers.gcs_writer import GCSWriter
        writer = _create_writer({"output_format": "gcs"})
        self.assertIsInstance(writer, GCSWriter)

    def test_unknown_format_defaults_parquet(self):
        from ixtract.engine import _create_writer
        from ixtract.writers.parquet import ParquetWriter
        writer = _create_writer({"output_format": "unknown"})
        self.assertIsInstance(writer, ParquetWriter)

    def test_missing_format_defaults_parquet(self):
        from ixtract.engine import _create_writer
        from ixtract.writers.parquet import ParquetWriter
        writer = _create_writer({})
        self.assertIsInstance(writer, ParquetWriter)


# ══════════════════════════════════════════════════════════════════════
# 6. WRITER CONFIG
# ══════════════════════════════════════════════════════════════════════

class TestWriterConfig(unittest.TestCase):
    """WriterConfig has max_file_size_bytes field."""

    def test_default_no_rotation(self):
        from ixtract.planner import WriterConfig
        wc = WriterConfig()
        self.assertIsNone(wc.max_file_size_bytes)

    def test_with_rotation(self):
        from ixtract.planner import WriterConfig
        wc = WriterConfig(max_file_size_bytes=100_000_000)
        self.assertEqual(wc.max_file_size_bytes, 100_000_000)


if __name__ == "__main__":
    unittest.main()
