"""PostgreSQL connector — Phase 1 source implementation.

Uses SQLAlchemy Core (not ORM) for SQL generation.
Supports metadata discovery, range chunk extraction, latency profiling,
connection monitoring, and PK distribution analysis for skew detection.
"""
from __future__ import annotations

import time
from typing import Any, Iterator, Optional

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine, Connection

from ixtract.connectors.base import (
    BaseConnector, ColumnInfo, ObjectMetadata, LatencyProfile, SourceConnections,
)

# Batch size for streaming chunks (rows per batch yielded to engine)
DEFAULT_STREAM_BATCH_SIZE = 10_000


class PostgreSQLConnector(BaseConnector):
    """PostgreSQL connector using SQLAlchemy Core.

    Supports REPEATABLE READ snapshot isolation for consistent extraction.
    """

    def __init__(self) -> None:
        self._engine: Optional[Engine] = None
        self._config: dict[str, Any] = {}

    def connect(self, config: dict[str, Any]) -> None:
        """Connect to PostgreSQL.

        Config keys:
            host, port, database, user, password
            OR connection_string (full SQLAlchemy URL)
        """
        self._config = config
        if "connection_string" in config:
            url = config["connection_string"]
        else:
            host = config.get("host", "localhost")
            port = config.get("port", 5432)
            db = config["database"]
            user = config.get("user", "")
            pw = config.get("password", "")
            url = f"postgresql+psycopg2://{user}:{pw}@{host}:{port}/{db}"

        pool_size = config.get("pool_size", 10)
        self._engine = create_engine(
            url,
            pool_size=pool_size,
            pool_pre_ping=True,
            pool_recycle=3600,
        )

    def metadata(self, object_name: str) -> ObjectMetadata:
        """Query PostgreSQL catalog for table metadata."""
        engine = self._require_engine()

        with engine.connect() as conn:
            # Row count estimate from pg_class (fast, no full scan)
            row_est = conn.execute(text(
                "SELECT reltuples::bigint AS estimate "
                "FROM pg_class WHERE relname = :table"
            ), {"table": object_name}).scalar() or 0

            # Table size estimate
            size_est = conn.execute(text(
                "SELECT pg_total_relation_size(:table)::bigint"
            ), {"table": object_name}).scalar() or 0

            # Column info via information_schema
            cols_result = conn.execute(text(
                "SELECT column_name, data_type, is_nullable "
                "FROM information_schema.columns "
                "WHERE table_name = :table "
                "ORDER BY ordinal_position"
            ), {"table": object_name}).fetchall()

            # Primary key
            pk_result = conn.execute(text(
                "SELECT a.attname, format_type(a.atttypid, a.atttypmod) "
                "FROM pg_index i "
                "JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey) "
                "JOIN pg_class c ON c.oid = i.indrelid "
                "WHERE c.relname = :table AND i.indisprimary "
                "LIMIT 1"
            ), {"table": object_name}).fetchone()

            pk_name = pk_result[0] if pk_result else None
            pk_type = pk_result[1] if pk_result else None

            columns = tuple(
                ColumnInfo(
                    name=row[0],
                    data_type=row[1],
                    nullable=(row[2] == "YES"),
                    is_primary_key=(row[0] == pk_name),
                )
                for row in cols_result
            )

            # PK range (min/max) if PK exists
            pk_min = pk_max = None
            if pk_name:
                range_result = conn.execute(text(
                    f"SELECT MIN({pk_name}), MAX({pk_name}) FROM {object_name}"
                )).fetchone()
                if range_result:
                    pk_min, pk_max = range_result[0], range_result[1]

        return ObjectMetadata(
            object_name=object_name,
            row_estimate=max(0, row_est),
            size_estimate_bytes=max(0, size_est),
            columns=columns,
            primary_key=pk_name,
            primary_key_type=pk_type,
            pk_min=pk_min,
            pk_max=pk_max,
        )

    def extract_chunk(
        self,
        object_name: str,
        chunk_query: str,
        params: dict[str, Any] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Extract a chunk via the given query, yielding batches of rows.

        The caller (execution engine) is responsible for managing the
        snapshot transaction. This method executes within whatever
        transaction context the caller provides.
        """
        engine = self._require_engine()
        batch_size = self._config.get("stream_batch_size", DEFAULT_STREAM_BATCH_SIZE)

        with engine.connect() as conn:
            result = conn.execution_options(stream_results=True).execute(
                text(chunk_query), params or {}
            )
            columns = list(result.keys())
            batch: list[dict[str, Any]] = []

            for row in result:
                batch.append(dict(zip(columns, row)))
                if len(batch) >= batch_size:
                    yield batch
                    batch = []

            if batch:
                yield batch

    def extract_chunk_snapshot(
        self,
        conn: Connection,
        chunk_query: str,
        params: dict[str, Any] | None = None,
    ) -> Iterator[list[dict[str, Any]]]:
        """Extract a chunk within an existing connection/transaction.

        Used by the execution engine for snapshot isolation — all chunks
        share the same REPEATABLE READ transaction.
        """
        batch_size = self._config.get("stream_batch_size", DEFAULT_STREAM_BATCH_SIZE)
        result = conn.execution_options(stream_results=True).execute(
            text(chunk_query), params or {}
        )
        columns = list(result.keys())
        batch: list[dict[str, Any]] = []

        for row in result:
            batch.append(dict(zip(columns, row)))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def estimate_latency(self, object_name: str) -> LatencyProfile:
        """Measure query latency with 5 lightweight probes."""
        engine = self._require_engine()
        latencies: list[float] = []

        with engine.connect() as conn:
            for _ in range(5):
                start = time.perf_counter()
                conn.execute(text(
                    f"SELECT 1 FROM {object_name} LIMIT 1"
                ))
                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies.append(elapsed_ms)

        latencies.sort()
        p50 = latencies[2]  # median of 5
        p95 = latencies[4]  # max of 5 (approximation for small sample)

        # Connection latency
        start = time.perf_counter()
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        conn_ms = (time.perf_counter() - start) * 1000

        return LatencyProfile(
            p50_ms=round(p50, 2),
            p95_ms=round(p95, 2),
            connection_ms=round(conn_ms, 2),
            sample_count=5,
        )

    def get_connections(self) -> SourceConnections:
        """Query current connection utilization."""
        engine = self._require_engine()
        with engine.connect() as conn:
            max_conn = conn.execute(text(
                "SELECT setting::int FROM pg_settings WHERE name = 'max_connections'"
            )).scalar() or 100

            active = conn.execute(text(
                "SELECT count(*) FROM pg_stat_activity WHERE state IS NOT NULL"
            )).scalar() or 0

        return SourceConnections(max_connections=max_conn, active_connections=active)

    def get_pk_distribution(self, object_name: str, num_buckets: int = 10) -> list[int]:
        """Count rows in equal-width PK range buckets for skew detection."""
        engine = self._require_engine()
        meta = self.metadata(object_name)

        if not meta.primary_key or meta.pk_min is None or meta.pk_max is None:
            return []

        pk = meta.primary_key
        pk_min, pk_max = meta.pk_min, meta.pk_max

        if isinstance(pk_min, (int, float)) and isinstance(pk_max, (int, float)):
            bucket_width = (pk_max - pk_min) / num_buckets
            if bucket_width <= 0:
                return [meta.row_estimate]

            counts: list[int] = []
            with engine.connect() as conn:
                for i in range(num_buckets):
                    lo = pk_min + i * bucket_width
                    hi = pk_min + (i + 1) * bucket_width
                    where = f"{pk} >= {lo}" + (f" AND {pk} < {hi}" if i < num_buckets - 1 else "")
                    count = conn.execute(text(
                        f"SELECT count(*) FROM {object_name} WHERE {where}"
                    )).scalar() or 0
                    counts.append(count)
            return counts

        return []

    def create_snapshot_connection(self) -> Connection:
        """Create a connection with REPEATABLE READ isolation for snapshot consistency.

        In SQLAlchemy 2.0, isolation level must be set via execution_options
        before the transaction begins.
        """
        engine = self._require_engine()
        conn = engine.connect().execution_options(
            isolation_level="REPEATABLE READ"
        )
        # Force the transaction to start by executing a lightweight query
        conn.execute(text("SELECT 1"))
        return conn

    def close(self) -> None:
        if self._engine:
            self._engine.dispose()
            self._engine = None

    def _require_engine(self) -> Engine:
        if not self._engine:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._engine
