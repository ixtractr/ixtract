"""S3 writer — multipart upload with atomic completion.

Uses boto3 multipart upload for reliable large file transfers.
File is invisible in S3 until CompleteMultipartUpload succeeds.
AbortMultipartUpload on failure — no partial objects left behind.

Thread-safety: one instance per chunk per worker, never shared.

Configuration keys (passed via config dict in open()):
    output_path:     S3 URI: s3://bucket/prefix/
    object_name:     Table/object name used in key.
    naming_pattern:  Key template. Default: {object}_{chunk_id}.parquet
    compression:     Parquet compression (passed through, not used by writer).
    aws_region:      AWS region. Default: from environment.
    aws_profile:     AWS profile name. Default: from environment.
    endpoint_url:    Custom endpoint (for MinIO, LocalStack). Default: None.
    part_size_mb:    Multipart part size in MB. Default: 8.
"""
from __future__ import annotations

import io
import os
import tempfile
from typing import Any, Optional

from ixtract.writers.parquet import BaseWriter, WriteResult, FinalizeResult


# Minimum part size for S3 multipart (5 MB)
MIN_PART_SIZE_BYTES = 5 * 1024 * 1024
DEFAULT_PART_SIZE_MB = 8


class S3Writer(BaseWriter):
    """S3 writer using multipart upload. Object only appears after finalize().

    Strategy:
        1. open() — write to local temp file (same as ParquetWriter)
        2. write_batch() — append to local temp file
        3. finalize() — upload temp file to S3 via multipart upload
        4. abort() — clean up temp file, abort any in-progress upload
    """

    def __init__(self) -> None:
        self._temp_path: Optional[str] = None
        self._s3_bucket: Optional[str] = None
        self._s3_key: Optional[str] = None
        self._pa_writer: Any = None
        self._total_rows: int = 0
        self._compression: str = "snappy"
        self._config: dict[str, Any] = {}
        self._upload_id: Optional[str] = None

    def open(
        self,
        config: dict[str, Any],
        chunk_id: str,
        schema: Optional[list[dict[str, str]]] = None,
    ) -> None:
        """Open writer. Parses S3 URI and creates local temp file."""
        self._config = config
        output_uri = config.get("output_path", "s3://bucket/output/")
        self._compression = config.get("compression", "snappy")

        # Parse s3://bucket/prefix/
        bucket, prefix = _parse_s3_uri(output_uri)
        self._s3_bucket = bucket

        obj_name = config.get("object_name", "data")
        naming = config.get("naming_pattern", "{object}_{chunk_id}.parquet")
        filename = naming.format(object=obj_name, chunk_id=chunk_id, format="parquet")
        self._s3_key = f"{prefix}{filename}" if prefix else filename

        # Local temp file for buffering
        fd, self._temp_path = tempfile.mkstemp(suffix=".s3.tmp")
        os.close(fd)

        self._total_rows = 0
        self._pa_writer = None

    def write_batch(self, batch: list[dict[str, Any]]) -> WriteResult:
        """Write batch to local temp file (same as ParquetWriter)."""
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
        """Close local file, upload to S3 via multipart, delete temp file."""
        if self._pa_writer:
            self._pa_writer.close()
            self._pa_writer = None

        if not self._temp_path or not self._s3_bucket or not self._s3_key:
            raise RuntimeError("S3Writer not properly initialized.")

        file_size = os.path.getsize(self._temp_path)

        # Upload to S3
        s3 = _get_s3_client(self._config)
        part_size = self._config.get("part_size_mb", DEFAULT_PART_SIZE_MB) * 1024 * 1024
        part_size = max(part_size, MIN_PART_SIZE_BYTES)

        if file_size <= part_size:
            # Single PUT for small files
            with open(self._temp_path, "rb") as f:
                s3.put_object(Bucket=self._s3_bucket, Key=self._s3_key, Body=f)
        else:
            # Multipart upload
            mpu = s3.create_multipart_upload(
                Bucket=self._s3_bucket, Key=self._s3_key,
            )
            self._upload_id = mpu["UploadId"]

            parts = []
            part_number = 1
            with open(self._temp_path, "rb") as f:
                while True:
                    chunk = f.read(part_size)
                    if not chunk:
                        break
                    resp = s3.upload_part(
                        Bucket=self._s3_bucket,
                        Key=self._s3_key,
                        UploadId=self._upload_id,
                        PartNumber=part_number,
                        Body=chunk,
                    )
                    parts.append({"PartNumber": part_number, "ETag": resp["ETag"]})
                    part_number += 1

            s3.complete_multipart_upload(
                Bucket=self._s3_bucket,
                Key=self._s3_key,
                UploadId=self._upload_id,
                MultipartUpload={"Parts": parts},
            )
            self._upload_id = None

        # Clean up temp file
        s3_path = f"s3://{self._s3_bucket}/{self._s3_key}"
        _safe_remove(self._temp_path)
        self._temp_path = None

        return FinalizeResult(s3_path, self._total_rows, file_size)

    def abort(self) -> None:
        """Clean up temp file and abort any in-progress multipart upload."""
        if self._pa_writer:
            try:
                self._pa_writer.close()
            except Exception:
                pass
            self._pa_writer = None

        if self._upload_id and self._s3_bucket and self._s3_key:
            try:
                s3 = _get_s3_client(self._config)
                s3.abort_multipart_upload(
                    Bucket=self._s3_bucket,
                    Key=self._s3_key,
                    UploadId=self._upload_id,
                )
            except Exception:
                pass
            self._upload_id = None

        _safe_remove(self._temp_path)
        self._temp_path = None


# ── Helpers ──────────────────────────────────────────────────────────

def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Parse s3://bucket/prefix/ → (bucket, prefix).

    Prefix includes trailing slash if present.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI (must start with s3://): {uri}")
    path = uri[5:]  # strip s3://
    if "/" in path:
        bucket, prefix = path.split("/", 1)
    else:
        bucket, prefix = path, ""
    return bucket, prefix


def _get_s3_client(config: dict[str, Any]):
    """Create a boto3 S3 client from config."""
    import boto3

    kwargs: dict[str, Any] = {}
    if config.get("aws_region"):
        kwargs["region_name"] = config["aws_region"]
    if config.get("endpoint_url"):
        kwargs["endpoint_url"] = config["endpoint_url"]

    if config.get("aws_profile"):
        session = boto3.Session(profile_name=config["aws_profile"])
        return session.client("s3", **kwargs)

    return boto3.client("s3", **kwargs)


def _safe_remove(path: Optional[str]) -> None:
    if path and os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass
