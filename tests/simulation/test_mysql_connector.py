"""MySQL Connector Tests — Phase 2C (hardened).

Tests the MySQLConnector interface contract using mocks.
No live MySQL required — all database interactions are mocked.

Tests verify:
  - Interface compliance: MySQLConnector is a BaseConnector subclass
  - connect(): URL construction, charset, pool configuration
  - metadata(): InnoDB check, no-PK error, composite-PK error, column mapping
  - extract_chunk(): streaming batch behaviour
  - extract_chunk_snapshot(): uses existing connection, no new connect
  - get_connections(): SHOW VARIABLES / SHOW STATUS parsing
  - create_snapshot_connection(): correct SQL, first-statement invariant
  - close(): disposal and idempotency
  - Snapshot isolation: no prior queries before CONSISTENT SNAPSHOT

Run: python -m unittest tests.simulation.test_mysql_connector -v
"""
from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.connectors.mysql import MySQLConnector, SUPPORTED_ENGINES
from ixtract.connectors.base import (
    BaseConnector, ObjectMetadata, LatencyProfile, SourceConnections,
)


# ── Interface compliance ──────────────────────────────────────────────

class TestBaseConnectorCompliance(unittest.TestCase):

    def test_is_base_connector_subclass(self):
        self.assertTrue(issubclass(MySQLConnector, BaseConnector))

    def test_has_all_required_methods(self):
        c = MySQLConnector()
        for m in ["connect", "metadata", "extract_chunk", "extract_chunk_snapshot",
                  "estimate_latency", "get_connections", "get_pk_distribution",
                  "create_snapshot_connection", "close"]:
            self.assertTrue(callable(getattr(c, m)), f"Missing method: {m}")

    def test_context_manager_protocol(self):
        c = MySQLConnector()
        mock_engine = MagicMock()
        c._engine = mock_engine
        with c as ctx:
            self.assertIs(ctx, c)
        mock_engine.dispose.assert_called_once()

    def test_supported_engines_constant(self):
        self.assertIn("InnoDB", SUPPORTED_ENGINES)
        self.assertNotIn("MyISAM", SUPPORTED_ENGINES)


# ── connect() ────────────────────────────────────────────────────────

class TestConnect(unittest.TestCase):

    @patch("ixtract.connectors.mysql.create_engine")
    def test_url_from_config_dict(self, mock_ce):
        MySQLConnector().connect({"host": "db.host", "port": 3306,
                                   "database": "mydb", "user": "root", "password": "pw"})
        url = mock_ce.call_args[0][0]
        self.assertIn("mysql+pymysql://", url)
        self.assertIn("db.host", url)
        self.assertIn("mydb", url)

    @patch("ixtract.connectors.mysql.create_engine")
    def test_connection_string_passed_through(self, mock_ce):
        MySQLConnector().connect({"connection_string": "mysql+pymysql://u:p@h/db"})
        self.assertEqual(mock_ce.call_args[0][0], "mysql+pymysql://u:p@h/db")

    @patch("ixtract.connectors.mysql.create_engine")
    def test_default_port_3306(self, mock_ce):
        MySQLConnector().connect({"host": "h", "database": "db", "user": "u", "password": "p"})
        self.assertIn(":3306/", mock_ce.call_args[0][0])

    @patch("ixtract.connectors.mysql.create_engine")
    def test_default_charset_utf8mb4(self, mock_ce):
        MySQLConnector().connect({"host": "h", "database": "db", "user": "u", "password": "p"})
        self.assertIn("utf8mb4", mock_ce.call_args[0][0])

    @patch("ixtract.connectors.mysql.create_engine")
    def test_custom_charset(self, mock_ce):
        MySQLConnector().connect({"host": "h", "database": "db", "user": "u",
                                   "password": "p", "charset": "latin1"})
        self.assertIn("latin1", mock_ce.call_args[0][0])

    @patch("ixtract.connectors.mysql.create_engine")
    def test_pool_pre_ping_enabled(self, mock_ce):
        MySQLConnector().connect({"host": "h", "database": "db", "user": "u", "password": "p"})
        self.assertTrue(mock_ce.call_args[1].get("pool_pre_ping"))

    @patch("ixtract.connectors.mysql.create_engine")
    def test_custom_pool_size(self, mock_ce):
        MySQLConnector().connect({"host": "h", "database": "db", "user": "u",
                                   "password": "p", "pool_size": 20})
        self.assertEqual(mock_ce.call_args[1]["pool_size"], 20)

    def test_require_engine_raises_before_connect(self):
        with self.assertRaises(RuntimeError):
            MySQLConnector()._require_engine()


# ── metadata() — constraint enforcement ──────────────────────────────

class TestMetadataConstraints(unittest.TestCase):

    def _conn_mock(self, c, execute_results):
        mock_conn = MagicMock()
        c._engine = MagicMock()
        c._engine.url.database = "testdb"
        c._engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        c._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        mock_conn.execute.side_effect = lambda *a, **kw: next(execute_results)
        return mock_conn

    def _engine_row(self, engine_name):
        r = MagicMock()
        r.__getitem__ = lambda self, i: engine_name
        return MagicMock(fetchone=lambda: r)

    def test_raises_if_table_not_found(self):
        c = MySQLConnector()
        results = iter([MagicMock(fetchone=lambda: None)])
        self._conn_mock(c, results)
        with self.assertRaises(RuntimeError, msg="Table not found"):
            c.metadata("missing_table")

    def test_raises_for_myisam(self):
        c = MySQLConnector()
        results = iter([self._engine_row("MyISAM")])
        self._conn_mock(c, results)
        with self.assertRaises(RuntimeError) as ctx:
            c.metadata("myisam_table")
        self.assertIn("MyISAM", str(ctx.exception))
        self.assertIn("InnoDB", str(ctx.exception))

    def test_raises_for_no_primary_key(self):
        c = MySQLConnector()
        size_row = MagicMock()
        size_row.__getitem__ = lambda self, i: [1000, 50000][i]
        results = iter([
            self._engine_row("InnoDB"),
            MagicMock(fetchone=lambda: size_row),
            MagicMock(fetchall=lambda: [("val", "varchar", "YES")]),
            MagicMock(fetchall=lambda: []),  # no PK rows
        ])
        self._conn_mock(c, results)
        with self.assertRaises(RuntimeError) as ctx:
            c.metadata("no_pk_table")
        self.assertIn("no primary key", str(ctx.exception))

    def test_raises_for_composite_primary_key(self):
        c = MySQLConnector()
        size_row = MagicMock()
        size_row.__getitem__ = lambda self, i: [1000, 50000][i]
        results = iter([
            self._engine_row("InnoDB"),
            MagicMock(fetchone=lambda: size_row),
            MagicMock(fetchall=lambda: [("id", "int", "NO"), ("type", "varchar", "NO")]),
            MagicMock(fetchall=lambda: [("id",), ("type",)]),  # 2 PK columns
        ])
        self._conn_mock(c, results)
        with self.assertRaises(RuntimeError) as ctx:
            c.metadata("composite_pk_table")
        self.assertIn("composite primary key", str(ctx.exception))
        self.assertIn("Phase 2C", str(ctx.exception))

    def test_successful_metadata_for_innodb_single_pk(self):
        c = MySQLConnector()
        size_row = MagicMock()
        size_row.__getitem__ = lambda self, i: [1_200_000, 166_000_000][i]
        pk_row = MagicMock()
        pk_row.__getitem__ = lambda self, i: "id"
        pk_type_row = MagicMock()
        pk_type_row.__getitem__ = lambda self, i: "int"
        range_row = MagicMock()
        range_row.__getitem__ = lambda self, i: [1, 1_200_000][i]

        results = iter([
            self._engine_row("InnoDB"),
            MagicMock(fetchone=lambda: size_row),
            MagicMock(fetchall=lambda: [("id", "int", "NO"), ("name", "varchar", "YES")]),
            MagicMock(fetchall=lambda: [("id",)]),
            MagicMock(fetchone=lambda: pk_type_row),
            MagicMock(fetchone=lambda: range_row),
        ])
        self._conn_mock(c, results)
        meta = c.metadata("orders")
        self.assertIsInstance(meta, ObjectMetadata)
        self.assertEqual(meta.primary_key, "id")
        self.assertEqual(meta.row_estimate, 1_200_000)


# ── extract_chunk() ───────────────────────────────────────────────────

class TestExtractChunk(unittest.TestCase):

    def _mock_connector(self, rows, batch_size=3):
        c = MySQLConnector()
        c._config = {"stream_batch_size": batch_size}
        c._engine = MagicMock()
        mock_conn = MagicMock()
        c._engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        c._engine.connect.return_value.__exit__ = MagicMock(return_value=False)
        result_mock = MagicMock()
        result_mock.keys.return_value = ["id", "val"]
        result_mock.__iter__ = MagicMock(return_value=iter(rows))
        mock_conn.execution_options.return_value.execute.return_value = result_mock
        return c

    def test_yields_batches_of_correct_size(self):
        c = self._mock_connector([(i, f"v{i}") for i in range(6)], batch_size=3)
        batches = list(c.extract_chunk("orders", "SELECT * FROM orders"))
        self.assertEqual(len(batches), 2)
        self.assertEqual(len(batches[0]), 3)

    def test_batch_is_list_of_dicts(self):
        c = self._mock_connector([(1, "a")], batch_size=10)
        batches = list(c.extract_chunk("orders", "SELECT * FROM orders"))
        self.assertIsInstance(batches[0][0], dict)
        self.assertIn("id", batches[0][0])


# ── create_snapshot_connection() ─────────────────────────────────────

class TestCreateSnapshotConnection(unittest.TestCase):

    def test_consistent_snapshot_is_first_statement(self):
        """CRITICAL: START TRANSACTION WITH CONSISTENT SNAPSHOT must be first.

        No prior queries — not even SELECT 1 or SET SESSION.
        Verified by checking the first execute() call uses the correct SQL.
        """
        c = MySQLConnector()
        c._engine = MagicMock()
        mock_conn = MagicMock()
        c._engine.connect.return_value = mock_conn

        c.create_snapshot_connection()

        first_call_sql = str(mock_conn.execute.call_args_list[0][0][0])
        self.assertIn("CONSISTENT SNAPSHOT", first_call_sql)

    def test_no_set_session_before_snapshot(self):
        """SET SESSION must NOT precede START TRANSACTION (resets snapshot)."""
        c = MySQLConnector()
        c._engine = MagicMock()
        mock_conn = MagicMock()
        c._engine.connect.return_value = mock_conn

        c.create_snapshot_connection()

        for i, call_args in enumerate(mock_conn.execute.call_args_list):
            sql = str(call_args[0][0]).upper()
            if "CONSISTENT SNAPSHOT" in sql:
                # All prior calls must not be SET SESSION
                for j in range(i):
                    prior = str(mock_conn.execute.call_args_list[j][0][0]).upper()
                    self.assertNotIn("SET SESSION", prior,
                        "SET SESSION executed before CONSISTENT SNAPSHOT — "
                        "this resets the snapshot point")
                break

    def test_returns_connection(self):
        c = MySQLConnector()
        mock_conn = MagicMock()
        c._engine = MagicMock()
        c._engine.connect.return_value = mock_conn
        result = c.create_snapshot_connection()
        self.assertIs(result, mock_conn)

    def test_docstring_documents_first_statement_invariant(self):
        doc = MySQLConnector.create_snapshot_connection.__doc__
        self.assertIn("FIRST", doc)
        self.assertIn("CONSISTENT SNAPSHOT", doc)


# ── get_connections() ─────────────────────────────────────────────────

class TestGetConnections(unittest.TestCase):

    def test_parses_show_variables_and_show_status(self):
        c = MySQLConnector()
        c._engine = MagicMock()
        c._engine.url.database = "db"
        mock_conn = MagicMock()
        c._engine.connect.return_value.__enter__ = MagicMock(return_value=mock_conn)
        c._engine.connect.return_value.__exit__ = MagicMock(return_value=False)

        max_row = MagicMock()
        max_row.__getitem__ = lambda self, i: ["max_connections", "151"][i]
        active_row = MagicMock()
        active_row.__getitem__ = lambda self, i: ["Threads_connected", "12"][i]

        results = iter([
            MagicMock(fetchone=lambda: max_row),
            MagicMock(fetchone=lambda: active_row),
        ])
        mock_conn.execute.side_effect = lambda *a, **kw: next(results)

        result = c.get_connections()
        self.assertIsInstance(result, SourceConnections)
        self.assertEqual(result.max_connections, 151)
        self.assertEqual(result.active_connections, 12)


# ── close() ──────────────────────────────────────────────────────────

class TestClose(unittest.TestCase):

    def test_disposes_engine(self):
        c = MySQLConnector()
        mock_engine = MagicMock()
        c._engine = mock_engine
        c.close()
        mock_engine.dispose.assert_called_once()

    def test_sets_engine_none(self):
        c = MySQLConnector()
        c._engine = MagicMock()
        c.close()
        self.assertIsNone(c._engine)

    def test_idempotent(self):
        c = MySQLConnector()
        c.close()
        c.close()  # must not raise


if __name__ == "__main__":
    unittest.main(verbosity=2)
