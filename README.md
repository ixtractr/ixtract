# ixtract

**Deterministic Adaptive Extraction Runtime**

ixtract extracts data from databases to files with automatic parallelism optimization,
feedback-driven learning, and full explainability. It treats extraction as a
closed-loop control problem — not static configuration.

```bash
pip install ixtract
```

[![PyPI version](https://badge.fury.io/py/ixtract.svg)](https://pypi.org/project/ixtract/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

---

## The problem

You tune worker counts by hand. You guess at chunk sizes. When throughput drops,
you don't know why. When a job takes twice as long as yesterday, there's no
explanation — just a number that got worse.

ixtract was built to answer the questions data engineers actually ask:

- *Why is this job slower today than yesterday?*
- *How many workers should I actually use?*
- *Why does adding more workers make it worse?*
- *Am I overloading my source database?*
- *Is it safe to run this during business hours?*
- *Can I trust this to run unattended?*

---

## Quickstart

```bash
pip install ixtract
```

```python
from ixtract import plan, execute, ExtractionIntent

intent = ExtractionIntent(
    source_type="postgresql",
    source_config={"host": "localhost", "database": "mydb", "user": "app"},
    object_name="orders",
)

result = plan(intent)
if result.is_safe:
    execution = execute(result)
    print(f"{execution.rows_extracted:,} rows in {execution.duration_seconds:.1f}s")
```

Or use the CLI:

```bash
# Profile your source (run once before first extraction)
ixtract profile orders --database mydb --user app

# Preview the plan
ixtract plan orders --database mydb --user app

# Extract
ixtract execute orders --database mydb --user app --output ./data

# Check health before next run
ixtract inspect orders
```

---

## How it works

**Profile → Plan → Execute → Learn**

```
Run 1:  Profile source → plan with profiler recommendation → extract → record
Run 2:  Controller adjusts workers based on throughput signal
Run 3+: Controller converges toward optimal worker count
Every:  Deviation diagnosed, anomaly detected, results explained
```

The controller uses direction-aware hill-climbing: if adding a worker helped, try
adding another. If it hurt, reverse. Converges to near-optimal in ≤5 runs under
stable conditions.

---

## CLI Commands

```bash
ixtract profile <table>   --database <db>                    # Profile source
ixtract plan <table>      --database <db>                    # Preview plan
ixtract execute <table>   --database <db> --output <dir>     # Extract
ixtract inspect <table>                                       # Health check
ixtract diagnose          --object <table>                   # Diagnose last run
ixtract explain           --object <table>                   # Explain last run
ixtract history <table>                                       # Run history
ixtract metrics           --object <table>                   # Run metrics
ixtract benchmark <table> --database <db>                    # Calibrate throughput
ixtract replay            --run-id <id> --database <db>      # Replay exact run
```

### `inspect` — operational health

```
$ ixtract inspect orders

Inspect: orders (postgresql)

Last Run          rx-20260414-173011     11.7s    856K/s    SUCCESS
Controller        CONVERGED at 8 workers    stable (drift -3.8%)
Anomalies         None
Profile           ✔ up-to-date

Health
  HEALTHY ✔
  System is stable and operating within expected bounds.
```

Exit codes are a contract:
```bash
ixtract inspect orders
echo $?   # 0=HEALTHY  1=NEEDS ATTENTION  2=DEGRADED
```

### `diagnose` — root cause analysis

```
$ ixtract diagnose --object events

Skew Analysis:
  Severity:   Severe (43.2x max/median)
  Slowest:    chunk_001 (2.07s, 1,502,847 rows)
  Fastest:    chunk_006 (0.03s, 10,241 rows)

Suggestion:
  → Severe skew detected. Work-stealing is active and mitigating.
  → Consider density-aware chunking for a permanent fix.
```

### `replay` — deterministic replay

```
$ ixtract replay --run-id rx-20260408-001 --database mydb

  Decision Check
  Workers        8         8
  Chunks         20        20
  Strategy       range_chunking    range_chunking
  Plan Hash      f6b8048a...       f6b8048a... ✔ identical

  Determinism: ✔ Verified (plan_fingerprint match)
```

---

## RuntimeContext — declare your environment

```bash
# Tell ixtract about your environment — it adjusts the plan accordingly
ixtract execute orders --database mydb \
  --source-load high \
  --network-quality degraded \
  --priority low

# Workers reduced from 8 → 2 via multipliers
# Result: 920K rows/sec at 2 workers (faster than 8 at high load)
```

---

## Sources & Outputs

| Source | Status |
|---|---|
| PostgreSQL | ✅ |
| MySQL | ✅ |
| SQL Server | ✅ |

| Output | Status |
|---|---|
| Parquet (local) | ✅ |
| CSV (local) | ✅ |
| Amazon S3 | ✅ |
| Google Cloud Storage | ✅ |

---

## Real-world validation

Tested across local PostgreSQL and Azure SQL Server:

| Run | Table | Result |
|---|---|---|
| Baseline | pgbench_accounts (10M rows) | 856K rows/sec, 11.7s, 8 workers |
| RuntimeContext | Same, high load declared | 2 workers → 920K rows/sec (faster) |
| Skewed table | skewed_events (CV=2.05) | 43x skew detected, work-stealing active |
| Cloud SQL Server | Azure, p50=30ms latency | 8.7K rows/sec, anomaly flagged at 44.3σ |
| Replay | Run 1 replayed exactly | Plan hash ✔ identical, +0.3% throughput delta |

540 tests passing across 15 simulation suites.

---

## Key guarantees

| Guarantee | Definition |
|---|---|
| Deterministic planning | Same inputs → same plan, every time |
| Explainable decisions | Every plan choice has a structured justification |
| Bounded adaptation | No single adjustment exceeds configured step limits |
| Source safety | Conservative bias under uncertainty — never overloads source |
| Idempotent execution | Retries produce no duplicates |
| Snapshot consistency | REPEATABLE READ — no missing or duplicate rows |
| Deterministic replay | Any run can be re-executed exactly from its stored plan |

---

## Installation

```bash
# Core (PostgreSQL)
pip install ixtract

# With MySQL support
pip install "ixtract[mysql]"

# With SQL Server support
pip install "ixtract[sqlserver]"

# With cloud writers (S3, GCS)
pip install "ixtract[cloud]"

# Everything
pip install "ixtract[all]"
```

---

## Development

```bash
git clone https://github.com/ixtractr/ixtract.git
cd ixtract
pip install -e ".[dev]"
pytest tests/simulation/ -q
```

---

## Product family

| Product | License | Purpose |
|---|---|---|
| **ixtract** | MIT | Extraction runtime — self-tuning, deterministic, explainable |
| **iPoxy** | MIT (coming soon) | Pipeline reinforcement |
| **ixora** | Commercial (coming soon) | Fleet intelligence |

---

## License

MIT — see [LICENSE](LICENSE).
