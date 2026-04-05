"""Integration test fixtures.

Requires a running PostgreSQL instance seeded with test data.
Start with:
    docker compose -f docker-compose.test.yml up -d
    python tests/integration/seed_db.py
"""
from __future__ import annotations

import os
import tempfile
import unittest

# Default test connection config — override with env vars
TEST_DB_CONFIG = {
    "host": os.environ.get("IXTRACT_TEST_HOST", "localhost"),
    "port": int(os.environ.get("IXTRACT_TEST_PORT", "5433")),
    "database": os.environ.get("IXTRACT_TEST_DB", "ixtract_test"),
    "user": os.environ.get("IXTRACT_TEST_USER", "ixtract"),
    "password": os.environ.get("IXTRACT_TEST_PASSWORD", "ixtract_test"),
}


def db_available() -> bool:
    """Check if the test database is reachable."""
    try:
        from sqlalchemy import create_engine, text
        url = (
            f"postgresql+psycopg2://{TEST_DB_CONFIG['user']}:{TEST_DB_CONFIG['password']}"
            f"@{TEST_DB_CONFIG['host']}:{TEST_DB_CONFIG['port']}/{TEST_DB_CONFIG['database']}"
        )
        engine = create_engine(url, connect_args={"connect_timeout": 3})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        engine.dispose()
        return True
    except Exception:
        return False


# Skip decorator for tests that need the database
requires_db = unittest.skipUnless(
    db_available(),
    "Test PostgreSQL not available. Run: docker compose -f docker-compose.test.yml up -d && python tests/integration/seed_db.py"
)
