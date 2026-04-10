"""Extraction Manifests — Phase 2C.

An extraction manifest is a structured declaration of what an extraction
run produced. It is the handoff contract between ixtract and downstream
consumers (loaders, quality gates, ixora fleet intelligence).

Design decisions (Phase 2C):

  Accumulate then write:
    File metadata is collected in memory during execution. The manifest
    is written once, atomically, after all chunks complete. No incremental
    writes to _manifest.json — eliminates partial-write corruption risk
    and requires no locking.

  Always write if execution started:
    Manifests are written for complete AND failed runs. A failed manifest
    (status=failed) is still a diagnostic artifact. Absence of a manifest
    would mean no artifact = no diagnosability.

  Schema field is null (Phase 2C):
    Full column schema is deferred to Phase 5 where it can be done properly
    (SourceProfile currently does not carry column-level detail). Faking a
    schema with partial data would require backward-compatibility hacks later.
    schema_hash provides lightweight change detection using available metadata:
    column_count | primary_key | primary_key_type | avg_row_bytes.

  Checksums computed after finalize():
    SHA-256 is computed over the final output file (after atomic rename).
    Never computed on temp files. Computed synchronously in Phase 2C —
    code structured to allow parallelization or opt-out in Phase 3.

  Output location:
    Written to {output_path}/_manifest.json alongside the data files.
    One manifest per run, one run per invocation.

Deferred (Phase 3+):
    - Full column schema in manifest
    - Manifest linking (multi-table consistency sets)
    - extraction_window for incremental extractions
    - freshness (most recent source row timestamp)
    - Parallel/async checksum computation
    - Manifest registry in state store for ixora
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from uuid import uuid4


# ── Manifest dataclasses ──────────────────────────────────────────────

@dataclass
class ManifestFile:
    """Metadata for a single output file produced by one chunk."""
    path: str
    size_bytes: int
    row_count: int
    checksum: str       # SHA-256 hex digest of the final file
    chunk_id: str       # which chunk produced this file

    def to_dict(self) -> dict:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "row_count": self.row_count,
            "checksum": self.checksum,
            "chunk_id": self.chunk_id,
        }


@dataclass
class ExtractionManifest:
    """Complete record of what an extraction run produced.

    Written to {output_path}/_manifest.json after every run.
    status = 'complete' | 'failed' | 'partial'

    schema field is null in Phase 2C — deferred to Phase 5.
    schema_hash is a lightweight fingerprint from available metadata.
    extraction_window and freshness are null until incremental support (Phase 2C).
    """
    manifest_id: str
    run_id: str
    source: str                         # e.g. "postgresql::orders"
    status: str                         # "complete" | "failed" | "partial"
    files: list[ManifestFile]
    total_rows: int
    total_bytes: int
    schema: None                        # null — Phase 5
    schema_hash: str                    # lightweight fingerprint
    extraction_window: None             # null — incremental not yet implemented
    freshness: None                     # null — Phase 5
    created_at: str                     # ISO-8601 UTC

    @property
    def file_count(self) -> int:
        return len(self.files)

    def to_dict(self) -> dict:
        return {
            "manifest_id": self.manifest_id,
            "run_id": self.run_id,
            "source": self.source,
            "status": self.status,
            "files": [f.to_dict() for f in self.files],
            "total_rows": self.total_rows,
            "total_bytes": self.total_bytes,
            "schema": None,
            "schema_hash": self.schema_hash,
            "extraction_window": None,
            "freshness": None,
            "created_at": self.created_at,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, d: dict) -> "ExtractionManifest":
        files = [
            ManifestFile(
                path=f["path"],
                size_bytes=f["size_bytes"],
                row_count=f["row_count"],
                checksum=f["checksum"],
                chunk_id=f.get("chunk_id", ""),
            )
            for f in d.get("files", [])
        ]
        return cls(
            manifest_id=d["manifest_id"],
            run_id=d["run_id"],
            source=d["source"],
            status=d["status"],
            files=files,
            total_rows=d["total_rows"],
            total_bytes=d["total_bytes"],
            schema=None,
            schema_hash=d.get("schema_hash", ""),
            extraction_window=None,
            freshness=None,
            created_at=d["created_at"],
        )


# ── Pure functions (no I/O except checksum and write) ─────────────────

def compute_file_checksum(path: str) -> str:
    """Compute SHA-256 of a file. Called after finalize() — never on temp files.

    Synchronous in Phase 2C. Structured for future parallelization:
    this function is a pure transformation (path → digest) with no
    side effects, so it can be moved to a thread pool without refactoring.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_schema_hash(
    column_count: int,
    primary_key: Optional[str],
    primary_key_type: Optional[str],
    avg_row_bytes: int,
) -> str:
    """Lightweight schema fingerprint from available profiler metadata.

    Phase 2C only — not a full schema hash. Useful for detecting:
    - Column additions/removals (column_count changes)
    - PK changes (primary_key changes)
    - Approximate row width changes (avg_row_bytes changes)

    Does NOT detect: column type changes that don't affect row width,
    column renames, or reorderings. Full schema hash deferred to Phase 5
    when SourceProfile carries column-level detail.

    The fingerprint is documented in the manifest so downstream consumers
    know its limitations.
    """
    fingerprint = (
        f"col_count={column_count}|"
        f"pk={primary_key or ''}|"
        f"pk_type={primary_key_type or ''}|"
        f"avg_row_bytes={avg_row_bytes}"
    )
    return hashlib.sha256(fingerprint.encode()).hexdigest()


def build_manifest(
    run_id: str,
    source_type: str,
    object_name: str,
    status: str,
    chunk_results: list,
    total_rows: int,
    total_bytes: int,
    column_count: int,
    primary_key: Optional[str],
    primary_key_type: Optional[str],
    avg_row_bytes: int,
) -> ExtractionManifest:
    """Build a manifest from execution results. Pure function — no I/O.

    Checksums are computed here for successful chunks with output files.
    Failed chunks produce no file entry (no file was finalized).

    Args:
        chunk_results: list of ChunkResult from ExecutionResult.
        All other args: from ExecutionResult and SourceProfile.
    """
    files: list[ManifestFile] = []
    for cr in chunk_results:
        if cr.status == "success" and cr.output_path:
            # Compute checksum synchronously — file must exist (post-finalize)
            try:
                checksum = compute_file_checksum(cr.output_path)
                size = os.path.getsize(cr.output_path)
            except OSError:
                # File unexpectedly missing — treat as failed chunk
                continue
            files.append(ManifestFile(
                path=cr.output_path,
                size_bytes=size,
                row_count=cr.rows,
                checksum=checksum,
                chunk_id=cr.chunk_id,
            ))

    schema_hash = compute_schema_hash(
        column_count, primary_key, primary_key_type, avg_row_bytes
    )

    return ExtractionManifest(
        manifest_id=str(uuid4()),
        run_id=run_id,
        source=f"{source_type}::{object_name}",
        status=status,
        files=files,
        total_rows=total_rows,
        total_bytes=total_bytes,
        schema=None,
        schema_hash=schema_hash,
        extraction_window=None,
        freshness=None,
        created_at=datetime.now(timezone.utc).isoformat(),
    )


def write_manifest(manifest: ExtractionManifest, output_path: str) -> str:
    """Write manifest to {output_path}/_manifest.json. Returns the manifest path.

    Uses atomic temp+rename to prevent partial writes — same pattern as writers.
    Always called, even for failed runs (status=failed is a diagnostic artifact).
    """
    import tempfile
    os.makedirs(output_path, exist_ok=True)
    manifest_path = os.path.join(output_path, "_manifest.json")

    # Atomic write — temp file in same directory
    fd, tmp_path = tempfile.mkstemp(suffix=".json.tmp", dir=output_path)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(manifest.to_json())
        os.replace(tmp_path, manifest_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise

    return manifest_path


def load_manifest(output_path: str) -> Optional[ExtractionManifest]:
    """Load a manifest from {output_path}/_manifest.json. Returns None if absent."""
    manifest_path = os.path.join(output_path, "_manifest.json")
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return ExtractionManifest.from_dict(json.load(f))
