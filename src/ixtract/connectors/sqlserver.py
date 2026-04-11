"""SQL Server connector — Phase 3C source implementation.

Uses pyodbc directly (not SQLAlchemy) for maximum control over
connection-level isolation and driver behavior.

Snapshot isolation model:
    Uses SET TRANSACTION ISOLATION LEVEL SNAPSHOT, which requires
    ALTER DATABASE ... SET ALLOW_SNAPSHOT_ISOLATION ON at server level.
    Each worker gets its own connection and snapshot.

Primary key constraint:
    Tables without a primary key cannot be range-chunked — RuntimeError.
    Tables with composite primary keys are not supported — RuntimeError.

Row estimate:
    Uses sys.dm_db_partition_stats for row estimates (more reliable than
    information_schema for large tables). Falls back to COUNT(*) for
    small tables (< 100K estimated rows).

Deferred:
    - Global snapshot coordinator (consistent read across all workers)
    - Composite PK support
    - Change tracking / CDC integration
"""
from __future__ import annotations

import time
from typing import Any, Iterator, Optional

from ixtract.connectors.base import (
    BaseConnector, ColumnInfo, ObjectMetadata, LatencyProfile, SourceConnections,
)

DEFAULT_STREAM_BATCH_SIZE = 10_000
SMALL_TABLE_THRESHOLD = 100_000  # below this, use COUNT(*) for exact count


class SQLServerConnector(BaseConnector):
    """SQL Server connector using pyodbc.

    Requires SNAPSHOT isolation enabled on the database.
    No composite PKs. Snapshot per-worker (not global).
    """

    def __init__(self) -> None:
        self._conn: Any = None
        self._config: dict[str, Any] = {}

    def connect(self, config: dict[str, Any]) -> None:
        """Connect to SQL Server.

        Config keys:
            connection_string:  Full ODBC connection string
            OR host, port, database, user, password, driver
            driver:             ODBC driver name. Default: "ODBC Driver 18 for SQL Server"
            trust_server_cert:  Trust self-signed certs. Default: True
            pool_size:          Not used directly (pyodbc manages pooling)
        """
        import pyodbc

        self._config = config

        if "connection_string" in config:
            conn_str = config["connection_string"]
        else:
            host = config.get("host", "localhost")
            port = config.get("port", 1433)
            db = config["database"]
            user = config.get("user", "")
            pw = config.get("password", "")
            driver = config.get("driver", "ODBC Driver 18 for SQL Server")
            trust_cert = config.get("trust_server_cert", True)
            trust_str = "Yes" if trust_cert else "No"

            conn_str = (
                f"DRIVER={{{driver}}};"
                f"SERVER={host},{port};"
                f"DATABASE={db};"
                f"UID={user};"
                f"PWD={pw};"
                f"TrustServerCertificate={trust_str};"
            )

        self._conn = pyodbc.connect(conn_str, autocommit=True)

    def metadata(self, object_name: str) -> ObjectMetadata:
        """Query SQL Server system views for table metadata.

        Raises RuntimeError if:
            - Table not found
            - Table has no primary key
            - Table has a composite primary key
        """
        conn = self._require_conn()
        cursor = conn.cursor()

        # Verify table exists
        cursor.execute(
            "SELECT 1 FROM INFORMATION_SCHEMA.TABLES "
            "WHERE TABLE_NAME = ? AND TABLE_TYPE = 'BASE TABLE'",
            (object_name,)
        )
        if not cursor.fetchone():
            raise RuntimeError(f"Table '{object_name}' not found.")

        # Row estimate from sys.dm_db_partition_stats (more reliable than info schema)
        cursor.execute(
            "SELECT SUM(row_count) FROM sys.dm_db_partition_stats "
            "WHERE object_id = OBJECT_ID(?) AND index_id IN (0, 1)",
            (object_name,)
        )
        row_result = cursor.fetchone()
        row_est = int(row_result[0] or 0) if row_result else 0

        # For small tables, use exact COUNT(*)
        if row_est < SMALL_TABLE_THRESHOLD:
            cursor.execute(f"SELECT COUNT(*) FROM [{object_name}]")
            exact = cursor.fetchone()
            if exact:
                row_est = int(exact[0])

        # Size estimate from sys.dm_db_partition_stats
        cursor.execute(
            "SELECT SUM(used_page_count) * 8 * 1024 "
            "FROM sys.dm_db_partition_stats "
            "WHERE object_id = OBJECT_ID(?) AND index_id IN (0, 1)",
            (object_name,)
        )
        size_result = cursor.fetchone()
        size_est = int(size_result[0] or 0) if size_result else 0

        # Columns
        cursor.execute(
            "SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE "
            "FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE TABLE_NAME = ? ORDER BY ORDINAL_POSITION",
            (object_name,)
        )
        col_rows = cursor.fetchall()

        # Primary key
        cursor.execute(
            "SELECT COLUMN_NAME "
            "FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE kcu "
            "JOIN INFORMATION_SCHEMA.TABLE_CONSTRAINTS tc "
            "ON kcu.CONSTRAINT_NAME = tc.CONSTRAINT_NAME "
            "WHERE tc.TABLE_NAME = ? AND tc.CONSTRAINT_TYPE = 'PRIMARY KEY' "
            "ORDER BY kcu.ORDINAL_POSITION",
            (object_name,)
        )
        pk_rows = cursor.fetchall()

        if not pk_rows:
            raise RuntimeError(
                f"Table '{object_name}' has no primary key. "
                f"ixtract requires a primary key for range-based chunking."
            )
        if len(pk_rows) > 1:
            pk_cols = ", ".join(r[0] for r in pk_rows)
            raise RuntimeError(
                f"Table '{object_name}' has a composite primary key ({pk_cols}). "
                f"Composite primary keys are not supported in Phase 3C."
            )

        pk_name = pk_rows[0][0]

        # PK type
        pk_type = None
        for row in col_rows:
            if row[0] == pk_name:
                pk_type = row[1]
                break

        columns = tuple(
            ColumnInfo(
                name=row[0],
                data_type=row[1],
                nullable=(row[2] == "YES"),
                is_primary_key=(row[0] == pk_name),
            )
            for row in col_rows
        )

        # PK range
        pk_min = pk_max = None
        cursor.execute(f"SELECT MIN([{pk_name}]), MAX([{pk_name}]) FROM [{object_name}]")
        range_result = cursor.fetchone()
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
        self, object_name: str, chunk_query: str, params: dict[str, Any] | None = None
    ) -> Iterator[list[dict[str, Any]]]:
        """Extract data using a per-worker SNAPSHOT isolation connection."""
        import pyodbc

        conn = self._require_conn()
        # Create a new connection for this extraction with SNAPSHOT isolation
        conn_str = conn.getinfo(pyodbc.SQL_DATA_SOURCE_NAME)

        # Use the same connection string from config
        if "connection_string" in self._config:
            worker_conn_str = self._config["connection_string"]
        else:
            worker_conn_str = self._build_conn_str()

        worker_conn = pyodbc.connect(worker_conn_str, autocommit=False)
        try:
            worker_conn.execute("SET TRANSACTION ISOLATION LEVEL SNAPSHOT")
            worker_conn.execute("BEGIN TRANSACTION")

            cursor = worker_conn.cursor()
            cursor.execute(chunk_query)

            columns = [desc[0] for desc in cursor.description]
            batch_size = self._config.get("batch_size", DEFAULT_STREAM_BATCH_SIZE)

            while True:
                rows = cursor.fetchmany(batch_size)
                if not rows:
                    break
                yield [dict(zip(columns, row)) for row in rows]

            worker_conn.commit()
        except Exception:
            try:
                worker_conn.rollback()
            except Exception:
                pass
            raise
        finally:
            worker_conn.close()

    def estimate_latency(self, object_name: str) -> LatencyProfile:
        """Sample the source to build a latency profile."""
        conn = self._require_conn()
        cursor = conn.cursor()

        latencies = []
        for _ in range(5):
            start = time.perf_counter()
            cursor.execute(f"SELECT TOP 1 * FROM [{object_name}]")
            cursor.fetchone()
            latencies.append((time.perf_counter() - start) * 1000)

        latencies.sort()
        return LatencyProfile(
            p50_ms=latencies[len(latencies) // 2],
            p95_ms=latencies[-1],
            connection_ms=latencies[0],
            sample_count=len(latencies),
        )

    def get_connections(self) -> SourceConnections:
        """Return current connection utilization."""
        conn = self._require_conn()
        cursor = conn.cursor()

        cursor.execute(
            "SELECT "
            "(SELECT COUNT(*) FROM sys.dm_exec_sessions WHERE is_user_process = 1), "
            "(SELECT CAST(value_in_use AS INT) FROM sys.configurations "
            "WHERE name = 'user connections')"
        )
        row = cursor.fetchone()
        active = int(row[0]) if row and row[0] else 0
        max_conn = int(row[1]) if row and row[1] else 32767
        # SQL Server default 0 means unlimited — cap at reasonable value
        if max_conn == 0:
            max_conn = 32767

        return SourceConnections(max_connections=max_conn, active_connections=active)

    def get_pk_distribution(self, object_name: str, num_buckets: int = 10) -> list[int]:
        """Return row counts per equal-width PK range bucket."""
        conn = self._require_conn()
        cursor = conn.cursor()

        meta = self.metadata(object_name)
        if not meta.primary_key or meta.pk_min is None or meta.pk_max is None:
            return [meta.row_estimate]

        pk = meta.primary_key
        pk_min, pk_max = meta.pk_min, meta.pk_max

        if not isinstance(pk_min, (int, float)) or not isinstance(pk_max, (int, float)):
            return [meta.row_estimate]

        pk_range = pk_max - pk_min
        if pk_range <= 0:
            return [meta.row_estimate]

        bucket_width = pk_range / num_buckets
        counts = []
        for i in range(num_buckets):
            lo = pk_min + i * bucket_width
            hi = pk_min + (i + 1) * bucket_width
            cursor.execute(
                f"SELECT COUNT(*) FROM [{object_name}] "
                f"WHERE [{pk}] >= ? AND [{pk}] < ?",
                (lo, hi)
            )
            row = cursor.fetchone()
            counts.append(int(row[0]) if row else 0)

        return counts

    def close(self) -> None:
        """Close the connection."""
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── Internal helpers ─────────────────────────────────────────

    def _require_conn(self):
        if self._conn is None:
            raise RuntimeError("Not connected. Call connect() first.")
        return self._conn

    def _build_conn_str(self) -> str:
        config = self._config
        host = config.get("host", "localhost")
        port = config.get("port", 1433)
        db = config["database"]
        user = config.get("user", "")
        pw = config.get("password", "")
        driver = config.get("driver", "ODBC Driver 18 for SQL Server")
        trust_cert = config.get("trust_server_cert", True)
        trust_str = "Yes" if trust_cert else "No"
        return (
            f"DRIVER={{{driver}}};"
            f"SERVER={host},{port};"
            f"DATABASE={db};"
            f"UID={user};"
            f"PWD={pw};"
            f"TrustServerCertificate={trust_str};"
        )
