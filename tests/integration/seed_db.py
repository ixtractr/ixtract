"""Seed script — populates test PostgreSQL with realistic extraction targets.

Creates three test tables to validate different extraction scenarios:

  orders        — 1.2M rows, uniform distribution, auto-increment PK.
                  Tests: happy-path range chunking, convergence.

  events        — 2M rows, skewed distribution (80% in 20% of PK range).
                  Tests: skew detection, greedy scheduling.

  small_lookup  — 500 rows. Tests: single-pass strategy selection.

Usage:
    python tests/integration/seed_db.py [--connection-string URL]
    python tests/integration/seed_db.py --host localhost --port 5433 --database ixtract_test --user ixtract --password ixtract_test
"""
from __future__ import annotations

import argparse
import random
import sys
import time

from sqlalchemy import create_engine, text


def build_url(args) -> str:
    if args.connection_string:
        return args.connection_string
    return (
        f"postgresql+psycopg2://{args.user}:{args.password}"
        f"@{args.host}:{args.port}/{args.database}"
    )


def seed_orders(engine, num_rows: int = 1_200_000) -> None:
    """Uniform distribution table. Auto-increment PK."""
    print(f"  Seeding orders ({num_rows:,} rows)...", end="", flush=True)
    start = time.perf_counter()

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS orders CASCADE"))
        conn.execute(text("""
            CREATE TABLE orders (
                id          BIGSERIAL PRIMARY KEY,
                customer_id BIGINT NOT NULL,
                product_id  INTEGER NOT NULL,
                quantity    INTEGER NOT NULL,
                total       NUMERIC(10,2) NOT NULL,
                status      VARCHAR(20) NOT NULL,
                region      VARCHAR(30) NOT NULL,
                created_at  TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at  TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """))
        conn.commit()

        # Batch insert for speed
        batch_size = 10_000
        statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
        regions = ["us-east", "us-west", "eu-west", "eu-central", "ap-south", "ap-east"]
        rng = random.Random(42)

        for batch_start in range(0, num_rows, batch_size):
            batch_end = min(batch_start + batch_size, num_rows)
            rows = []
            for _ in range(batch_end - batch_start):
                rows.append({
                    "customer_id": rng.randint(1, 200_000),
                    "product_id": rng.randint(1, 5000),
                    "quantity": rng.randint(1, 20),
                    "total": round(rng.uniform(5.0, 500.0), 2),
                    "status": rng.choice(statuses),
                    "region": rng.choice(regions),
                })

            values_clause = ",".join(
                f"({r['customer_id']},{r['product_id']},{r['quantity']},"
                f"{r['total']},'{r['status']}','{r['region']}',NOW(),NOW())"
                for r in rows
            )
            conn.execute(text(
                f"INSERT INTO orders (customer_id,product_id,quantity,total,status,region,created_at,updated_at) "
                f"VALUES {values_clause}"
            ))
            conn.commit()

            if (batch_start // batch_size) % 20 == 0 and batch_start > 0:
                print(".", end="", flush=True)

        # Create indexes
        conn.execute(text("CREATE INDEX idx_orders_customer ON orders(customer_id)"))
        conn.execute(text("CREATE INDEX idx_orders_created ON orders(created_at)"))
        conn.commit()

        # Analyze for accurate statistics
        conn.execute(text("ANALYZE orders"))
        conn.commit()

    elapsed = time.perf_counter() - start
    print(f" done ({elapsed:.1f}s)")


def seed_events(engine, num_rows: int = 2_000_000) -> None:
    """Skewed distribution table. 80% of rows in 20% of PK range."""
    print(f"  Seeding events ({num_rows:,} rows, skewed)...", end="", flush=True)
    start = time.perf_counter()

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS events CASCADE"))
        conn.execute(text("""
            CREATE TABLE events (
                id          BIGSERIAL PRIMARY KEY,
                user_id     BIGINT NOT NULL,
                event_type  VARCHAR(50) NOT NULL,
                payload     TEXT,
                created_at  TIMESTAMP NOT NULL DEFAULT NOW()
            )
        """))
        conn.commit()

        batch_size = 10_000
        event_types = ["page_view", "click", "purchase", "login", "logout",
                       "search", "add_to_cart", "remove_from_cart", "signup", "error"]
        rng = random.Random(42)

        for batch_start in range(0, num_rows, batch_size):
            batch_end = min(batch_start + batch_size, num_rows)
            rows = []
            for _ in range(batch_end - batch_start):
                # Skew: 80% of events from 20% of users (IDs 1-40K out of 200K)
                if rng.random() < 0.8:
                    user_id = rng.randint(1, 40_000)
                else:
                    user_id = rng.randint(40_001, 200_000)

                payload_len = rng.randint(20, 500)
                rows.append({
                    "user_id": user_id,
                    "event_type": rng.choice(event_types),
                    "payload": "x" * payload_len,
                })

            values_clause = ",".join(
                f"({r['user_id']},'{r['event_type']}','{r['payload']}',NOW())"
                for r in rows
            )
            conn.execute(text(
                f"INSERT INTO events (user_id,event_type,payload,created_at) "
                f"VALUES {values_clause}"
            ))
            conn.commit()

            if (batch_start // batch_size) % 40 == 0 and batch_start > 0:
                print(".", end="", flush=True)

        conn.execute(text("CREATE INDEX idx_events_user ON events(user_id)"))
        conn.execute(text("CREATE INDEX idx_events_created ON events(created_at)"))
        conn.commit()
        conn.execute(text("ANALYZE events"))
        conn.commit()

    elapsed = time.perf_counter() - start
    print(f" done ({elapsed:.1f}s)")


def seed_small_lookup(engine) -> None:
    """Small table for single-pass strategy testing."""
    print("  Seeding small_lookup (500 rows)...", end="", flush=True)

    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS small_lookup CASCADE"))
        conn.execute(text("""
            CREATE TABLE small_lookup (
                id      SERIAL PRIMARY KEY,
                code    VARCHAR(10) NOT NULL,
                name    VARCHAR(100) NOT NULL,
                active  BOOLEAN DEFAULT TRUE
            )
        """))
        conn.commit()

        rng = random.Random(42)
        for i in range(1, 501):
            conn.execute(text(
                "INSERT INTO small_lookup (code, name, active) VALUES (:code, :name, :active)"
            ), {"code": f"L{i:04d}", "name": f"Lookup item {i}", "active": rng.random() > 0.1})
        conn.commit()
        conn.execute(text("ANALYZE small_lookup"))
        conn.commit()

    print(" done")


def main():
    parser = argparse.ArgumentParser(description="Seed ixtract test database")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default=5433, type=int)
    parser.add_argument("--database", default="ixtract_test")
    parser.add_argument("--user", default="ixtract")
    parser.add_argument("--password", default="ixtract_test")
    parser.add_argument("--connection-string", default=None)
    parser.add_argument("--orders-rows", default=1_200_000, type=int)
    parser.add_argument("--events-rows", default=2_000_000, type=int)
    args = parser.parse_args()

    url = build_url(args)
    print(f"Connecting to {args.host}:{args.port}/{args.database}...")
    engine = create_engine(url)

    # Verify connection
    with engine.connect() as conn:
        version = conn.execute(text("SELECT version()")).scalar()
        print(f"  PostgreSQL: {version.split(',')[0]}")

    print("\nSeeding tables:")
    seed_orders(engine, args.orders_rows)
    seed_events(engine, args.events_rows)
    seed_small_lookup(engine)

    # Verify
    print("\nVerification:")
    with engine.connect() as conn:
        for table in ["orders", "events", "small_lookup"]:
            count = conn.execute(text(f"SELECT count(*) FROM {table}")).scalar()
            size = conn.execute(text(
                f"SELECT pg_total_relation_size('{table}')"
            )).scalar()
            print(f"  {table}: {count:,} rows, {size // (1024*1024)}MB")

    print("\nDone. Database is ready for integration tests.")
    engine.dispose()


if __name__ == "__main__":
    main()
