# Changelog

## v0.1.0 — Phase 1: Vertical MVP (April 2026)

First working release. Extracts from PostgreSQL to Parquet with adaptive parallelism.

### What it does
- **Profile** any PostgreSQL table: row count, PK distribution, latency, connection capacity
- **Plan** extraction: automatic strategy selection (range_chunking vs single_pass), worker count from profiler or controller, duration estimate, safety checks
- **Execute** end-to-end: parallel extraction via ThreadPoolExecutor, streaming to Parquet with atomic finalize, retry with exponential backoff
- **Explain** why a run behaved the way it did: throughput delta, deviation diagnosis, controller recommendation
- **Converge** to optimal worker count over 3-5 runs via direction-aware hill-climbing controller

### Key properties
- **Deterministic**: same inputs → same plan
- **Idempotent**: retries produce no duplicates (temp file + atomic rename)
- **Consistent**: REPEATABLE READ snapshot isolation per worker
- **Bounded**: worker adjustments ±1 per run, no runaway feedback
- **Explainable**: every diagnosis includes reasoning chain and recommendation

### Architecture
- 13 production modules, 2,852 lines
- 28 Phase 0 simulation tests (controller convergence validated)
- 12 Phase 1 integration tests (real PostgreSQL extraction validated)
- SQLite WAL state store with runs, chunks, worker metrics, deviations, profiles
- CLI: `profile`, `plan`, `execute`, `explain` commands

### Test results
- 1.2M rows extracted at 208K rows/sec (range chunking, 5 workers)
- 2M rows extracted at 440K rows/sec (converged at 5 workers in 5 runs)
- 500-row table correctly gets single-pass strategy (1 worker)
- Controller converges and holds stable across repeated runs

### Phase 2 (planned)
Benchmarker, full context-weighted planning, diagnose/history/metrics commands, MySQL/SQL Server, CSV/cloud writers, work-stealing, adaptive intra-run rules.
