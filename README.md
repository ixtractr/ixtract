# ixtract

**Deterministic Adaptive Extraction Runtime**

ixtract treats data extraction as a closed-loop control problem, not static configuration.
It profiles your source, plans intelligently, executes with adaptive parallelism,
diagnoses deviations, and converges to optimal performance — all without manual tuning.

## Quickstart

```bash
# Install
pip install -e "."

# Profile your source (run once)
ixtract profile orders --database mydb --user postgres --password secret

# Plan (preview without executing)
ixtract plan orders --database mydb --user postgres --password secret --output ./data

# Extract
ixtract execute orders --database mydb --user postgres --password secret --output ./data

# Explain what happened
ixtract explain --object orders
```

## What It Does

**Run 1:** Profiles the source (row count, PK distribution, latency, connection capacity).
Plans with profiler-recommended workers. Extracts to Parquet. Records metrics.

**Run 2–5:** Controller adjusts workers based on throughput. Direction-aware hill climbing:
if adding a worker helped, add another. If it hurt, reverse. Converges to near-optimal.

**Every run:** Diagnoses deviations (UNDER_PARALLEL, OVER_PARALLEL, STABLE).
Explains why in the CLI. Records per-chunk and per-worker metrics for future analysis.

## Architecture

```
Control Plane                    Data Plane
─────────────                    ──────────
Intent Model                     Execution Engine (ThreadPool)
Source Profiler                   Workers (per-chunk extraction)
Planner (pure function)          Bounded Buffer (backpressure)
Controller (hill-climbing)       Parquet Writer (atomic finalize)
Deviation Analyzer               Metrics Collector
State Store (SQLite WAL)
CLI (explain, plan, execute)
```

## CLI Commands (Phase 1)

| Command | What It Does |
|---------|-------------|
| `ixtract profile <table>` | Profile source: metadata, PK distribution, latency, connections |
| `ixtract plan <table>` | Show plan without executing: strategy, workers, duration, safety |
| `ixtract execute <table>` | Extract end-to-end: profile → plan → execute → diagnose → store |
| `ixtract explain <run_id>` | Explain why a run behaved the way it did |

## Key Guarantees

- **Deterministic:** Same inputs → same plan.
- **Idempotent:** Retries produce no duplicates (temp file + atomic rename).
- **Consistent:** Snapshot isolation (REPEATABLE READ) — no missing/duplicate rows.
- **Bounded:** Worker adjustments ±1 per run. No runaway feedback.
- **Explainable:** Every diagnosis includes reasoning, evidence, and recommendation.

## Project Structure

```
src/ixtract/
  intent/          ExtractionIntent (what to extract)
  planner/         ExecutionPlan (how to extract) + planning logic
  connectors/      PostgreSQL connector (Phase 1)
  profiler/        Source profiling (metadata, latency, skew)
  controller/      Direction-aware hill-climbing optimizer
  diagnosis/       Deviation classification with reasoning chains
  engine/          Execution engine (workers, scheduling, backpressure)
  writers/         Parquet writer (atomic finalize)
  state/           SQLite state store (runs, chunks, workers, deviations)
  simulation/      Synthetic source for controller validation
  cli/             Click-based CLI (profile, plan, execute, explain)
tests/
  simulation/      Phase 0: 28 controller convergence tests
  unit/            Unit tests
```

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run Phase 0 simulation tests
python -m unittest tests.simulation.test_phase0 -v

# Run all tests
pytest tests/ -v
```

## License

MIT — see [LICENSE](LICENSE).
