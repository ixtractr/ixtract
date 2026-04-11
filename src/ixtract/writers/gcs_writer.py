"""GCS writer — resumable upload with atomic blob creation.

Uses google-cloud-storage resumable upload for reliable large file transfers.
Blob is invisible until upload completes. Aborted uploads are cleaned up.

Thread-safety: one instance per chunk per worker, never shared.

Configuration keys (passed via config dict in open()):
    output_path:     GCS URI: gs://bucket/prefix/
    object_name:     Table/object name used in blob name.
    naming_pattern:  Blob name template. Default: {object}_{chunk_id}.parquet
    compression:     Parquet compression (passed through, not used by writer).
    gcs_project:     GCP project ID. Default: from environment.
    chunk_size_mb:   Resumable upload chunk size in MB. Default: 8.
                     Must be a multiple of 256 KB per GCS requirements.
"""
from __future__ import annotations

import os
import tempfile
from typing import Any, Optional

from ixtract.writers.parquet import BaseWriter, WriteResult, FinalizeResult


DEFAULT_CHUNK_SIZE_MB = 8


class GCSWriter(BaseWriter):
    """GCS writer using resumable upload. Blob only appears after finalize().

    Strategy (same as S3Writer):
        1. open() — write to local temp file
        2. write_batch() — append to local temp file
        3. finalize() — upload temp file to GCS
        4. abort() — clean up temp file
    """

    def __init__(self) -> None:
        self._temp_path: Optional[str] = None
        self._gcs_bucket: Optional[str] = None
        self._gcs_blob_name: Optional[str] = None
        self._pa_writer: Any = None
        self._total_rows: int = 0
        self._compression: str = "snappy"
        self._config: dict[str, Any] = {}

    def open(
        self,
        config: dict[str, Any],
        chunk_id: str,
        schema: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """Open writer. Parses GCS URI and creates local temp file."""
        self._config = config
        output_uri = config.get("output_path", "gs://bucket/output/")
        self._compression = config.get("compression", "snappy")

        bucket, prefix = _parse_gcs_uri(output_uri)
        self._gcs_bucket = bucket

        obj_name = config.get("object_name", "data")
        naming = config.get("naming_pattern", "{object}_{chunk_id}.parquet")
        filename = naming.format(object=obj_name, chunk_id=chunk_id, format="parquet")
        self._gcs_blob_name = f"{prefix}{filename}" if prefix else filename

        # Local temp file for buffering
        fd, self._temp_path = tempfile.mkstemp(suffix=".gcs.tmp")
        os.close(fd)

        self._total_rows = 0
        self._pa_writer = None

    def write_batch(self, batch: list[dict[str, Any]]) -> WriteResult:
        """Write batch to local temp file."""
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
        """Close local file, upload to GCS, delete temp file."""
        if self._pa_writer:
            self._pa_writer.close()
            self._pa_writer = None

        if not self._temp_path or not self._gcs_bucket or not self._gcs_blob_name:
            raise RuntimeError("GCSWriter not properly initialized.")

        file_size = os.path.getsize(self._temp_path)

        # Upload to GCS
        client = _get_gcs_client(self._config)
        bucket = client.bucket(self._gcs_bucket)
        blob = bucket.blob(self._gcs_blob_name)

        chunk_size = self._config.get("chunk_size_mb", DEFAULT_CHUNK_SIZE_MB) * 1024 * 1024
        # GCS requires chunk_size to be a multiple of 256 KB
        chunk_size = max(256 * 1024, (chunk_size // (256 * 1024)) * (256 * 1024))
        blob.chunk_size = chunk_size

        blob.upload_from_filename(self._temp_path)

        gcs_path = f"gs://{self._gcs_bucket}/{self._gcs_blob_name}"
        _safe_remove(self._temp_path)
        self._temp_path = None

        return FinalizeResult(gcs_path, self._total_rows, file_size)

    def abort(self) -> None:
        """Clean up temp file."""
        if self._pa_writer:
            try:
                self._pa_writer.close()
            except Exception:
                pass
            self._pa_writer = None

        _safe_remove(self._temp_path)
        self._temp_path = None


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse gs://bucket/prefix/ → (bucket, prefix)."""
    if not uri.startswith("gs://"):
        raise ValueError(f"Invalid GCS URI (must start with gs://): {uri}")
    path = uri[5:]
    if "/" in path:
        bucket, prefix = path.split("/", 1)
    else:
        bucket, prefix = path, ""
    return bucket, prefix


def _get_gcs_client(config: dict[str, Any]):
    """Create a google.cloud.storage Client from config."""
    from google.cloud import storage

    kwargs: dict[str, Any] = {}
    if config.get("gcs_project"):
        kwargs["project"] = config["gcs_project"]

    return storage.Client(**kwargs)


def _safe_remove(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
