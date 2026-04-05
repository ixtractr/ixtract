"""ixtract CLI — the engineer's interface.

Design principle: "I asked a question → I got a clear answer."
Phase 1 implements summary level only.
"""
from __future__ import annotations

import json
import logging
import sys
import time

import click

from ixtract import __version__


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )


@click.group()
@click.version_option(__version__)
def cli():
    """ixtract — Deterministic Adaptive Extraction Runtime."""
    pass


@cli.command()
@click.argument("object_name")
@click.option("--host", default="localhost", help="PostgreSQL host")
@click.option("--port", default=5432, type=int, help="PostgreSQL port")
@click.option("--database", required=True, help="Database name")
@click.option("--user", default="", help="Database user")
@click.option("--password", default="", help="Database password")
@click.option("--connection-string", default=None, help="Full connection string (overrides host/port/database)")
def profile(object_name, host, port, database, user, password, connection_string):
    """Profile a source object. Run before first extraction."""
    from ixtract.connectors.postgresql import PostgreSQLConnector
    from ixtract.profiler import SourceProfiler

    config = _build_config(host, port, database, user, password, connection_string)

    click.echo(f"Profiling {object_name}...")
    connector = PostgreSQLConnector()
    connector.connect(config)

    try:
        profiler = SourceProfiler(connector)
        prof = profiler.profile(object_name)

        click.echo(f"\nSource Profile: postgresql::{object_name}")
        click.echo(f"  Rows: {prof.row_estimate:,}  |  Size: {prof.size_estimate_bytes // (1024*1024)}MB  |  Columns: {prof.column_count}")
        click.echo(f"  PK: {prof.primary_key or 'none'} ({prof.primary_key_type or 'n/a'})")
        if prof.has_usable_pk:
            click.echo(f"  PK range: {prof.pk_min} → {prof.pk_max}")
            click.echo(f"  PK distribution CV: {prof.pk_distribution_cv:.2f}" +
                       (" (skewed)" if prof.pk_distribution_cv > 1.0 else " (uniform)"))
        click.echo(f"  Latency: p50={_fmt_latency(prof.latency_p50_ms)}  p95={_fmt_latency(prof.latency_p95_ms)}")
        click.echo(f"  Connections: {prof.active_connections}/{prof.max_connections} "
                    f"(safe slots: {prof.available_connections_safe})")
        click.echo(f"  Recommended: {prof.recommended_start_workers} workers, "
                    f"{prof.recommended_strategy}, {prof.recommended_scheduling}")

    finally:
        connector.close()


@cli.command()
@click.argument("object_name")
@click.option("--host", default="localhost")
@click.option("--port", default=5432, type=int)
@click.option("--database", required=True)
@click.option("--user", default="")
@click.option("--password", default="")
@click.option("--connection-string", default=None)
@click.option("--output", default="./output", help="Output directory")
@click.option("--compression", default="snappy", help="Parquet compression")
@click.option("--max-workers", default=None, type=int, help="Max worker count cap")
@click.option("--state-db", default="ixtract_state.db", help="State store path")
def plan(object_name, host, port, database, user, password, connection_string,
         output, compression, max_workers, state_db):
    """Show the extraction plan without executing."""
    from ixtract.connectors.postgresql import PostgreSQLConnector
    from ixtract.profiler import SourceProfiler
    from ixtract.planner.planner import plan_extraction, format_plan_summary
    from ixtract.intent import ExtractionIntent, SourceType, TargetType, ExtractionConstraints
    from ixtract.state import StateStore
    from ixtract.controller import ControllerState

    config = _build_config(host, port, database, user, password, connection_string)

    connector = PostgreSQLConnector()
    connector.connect(config)

    try:
        # Profile
        profiler = SourceProfiler(connector)
        prof = profiler.profile(object_name)

        # Build intent
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config=config,
            object_name=object_name,
            target_type=TargetType.PARQUET,
            target_config={"output_path": output, "compression": compression},
            constraints=ExtractionConstraints(max_workers=max_workers),
        )

        # Check state store for controller history
        store = StateStore(state_db)
        ctrl_state = store.get_controller_state("postgresql", object_name)

        # Plan
        exec_plan = plan_extraction(intent, prof, store, ctrl_state)
        summary = format_plan_summary(exec_plan, prof, ctrl_state)
        click.echo(f"\n{summary}")

    finally:
        connector.close()


@cli.command()
@click.argument("object_name")
@click.option("--host", default="localhost")
@click.option("--port", default=5432, type=int)
@click.option("--database", required=True)
@click.option("--user", default="")
@click.option("--password", default="")
@click.option("--connection-string", default=None)
@click.option("--output", default="./output", help="Output directory")
@click.option("--compression", default="snappy")
@click.option("--max-workers", default=None, type=int)
@click.option("--state-db", default="ixtract_state.db")
@click.option("-v", "--verbose", is_flag=True)
def execute(object_name, host, port, database, user, password, connection_string,
            output, compression, max_workers, state_db, verbose):
    """Extract data from source to Parquet."""
    _setup_logging(verbose)

    from ixtract.connectors.postgresql import PostgreSQLConnector
    from ixtract.profiler import SourceProfiler
    from ixtract.planner.planner import plan_extraction, format_plan_summary
    from ixtract.intent import ExtractionIntent, SourceType, TargetType, ExtractionConstraints
    from ixtract.state import StateStore
    from ixtract.controller import ParallelismController, ControllerConfig
    from ixtract.diagnosis import DeviationAnalyzer
    from ixtract.engine import ExecutionEngine

    config = _build_config(host, port, database, user, password, connection_string)

    connector = PostgreSQLConnector()
    connector.connect(config)
    store = StateStore(state_db)

    try:
        # 1. Profile
        profiler = SourceProfiler(connector)
        prof = profiler.profile(object_name)

        # 2. Build intent
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config=config,
            object_name=object_name,
            target_type=TargetType.PARQUET,
            target_config={"output_path": output, "compression": compression,
                           "object_name": object_name},
            constraints=ExtractionConstraints(max_workers=max_workers),
        )

        # 3. Controller state
        ctrl_state = store.get_controller_state("postgresql", object_name)
        controller = ParallelismController()

        # 4. Plan
        exec_plan = plan_extraction(intent, prof, store, ctrl_state)
        click.echo(format_plan_summary(exec_plan, prof, ctrl_state))
        click.echo()

        # 5. Execute
        engine = ExecutionEngine(connector)
        result = engine.execute(exec_plan, object_name)

        # 6. Get previous run BEFORE recording current (so we don't find ourselves)
        recent_runs = store.get_recent_runs("postgresql", object_name, limit=1)

        # 7. Record current run
        store.record_run_start(
            result.run_id, exec_plan.plan_id, intent.intent_hash(),
            "postgresql", object_name, exec_plan.strategy.value,
            exec_plan.worker_count,
        )
        store.record_run_end(
            result.run_id, result.status.lower(),
            result.total_rows, result.total_bytes,
            result.avg_throughput, result.duration_seconds,
        )

        # 7b. Record per-chunk results (for Phase 2 diagnose/metrics commands)
        for cr in result.chunk_results:
            store.record_chunk(
                result.run_id, cr.chunk_id, cr.worker_id,
                cr.rows, cr.bytes_written, cr.status, cr.duration_seconds,
                query_ms=cr.query_ms, write_ms=cr.write_ms,
                output_path=cr.output_path, error=cr.error,
            )

        # 7c. Record per-worker metrics
        worker_chunks: dict[int, list] = {}
        for cr in result.chunk_results:
            worker_chunks.setdefault(cr.worker_id, []).append(cr)
        for wid, chunks in worker_chunks.items():
            store.record_worker_metrics(
                result.run_id, wid,
                chunks_processed=len(chunks),
                total_rows=sum(c.rows for c in chunks),
                total_bytes=sum(c.bytes_written for c in chunks),
            )

        # 8. Deviation analysis
        diag = None
        ctrl_out = None

        if result.metrics:
            analyzer = DeviationAnalyzer()

            # Use ACTUAL previous run's worker count and throughput from state store
            if recent_runs:
                prev_tp = recent_runs[0].get("avg_throughput", 0.0)
                prev_wc = recent_runs[0].get("worker_count", 0)
            else:
                prev_tp = 0.0
                prev_wc = 0

            metrics = result.metrics
            from ixtract.diagnosis import RunMetrics
            metrics = RunMetrics(
                total_rows=metrics.total_rows,
                total_bytes=metrics.total_bytes,
                duration_seconds=metrics.duration_seconds,
                worker_count=metrics.worker_count,
                avg_throughput_rows_sec=metrics.avg_throughput_rows_sec,
                chunk_durations=metrics.chunk_durations,
                worker_idle_pcts=metrics.worker_idle_pcts,
                predicted_duration_seconds=metrics.predicted_duration_seconds,
                predicted_throughput_rows_sec=metrics.predicted_throughput_rows_sec,
                previous_throughput_rows_sec=prev_tp,
                previous_worker_count=prev_wc,
                worker_count_changed=(exec_plan.worker_count != prev_wc and prev_wc > 0),
            )
            diag = analyzer.diagnose(metrics)
            store.record_deviation(result.run_id, diag)

            # 8. Controller evaluation
            from ixtract.controller import ControllerState, ControllerOutput, ControllerDecision

            if ctrl_state is None:
                ctrl_state = ControllerState.cold_start(controller.config)

            ctrl_out = controller.evaluate(result.avg_throughput, ctrl_state)

            # 9. "Revisit lower bound once" rule:
            # If we're over-parallel AND above the profiler's recommendation,
            # try the profiler's value instead of continuing upward. One-shot.
            revisit_applied = store.get_heuristic(
                "postgresql", object_name, "revisit_lower_applied"
            )
            if (not revisit_applied
                and diag.category.value == "over_parallel"
                and exec_plan.worker_count > prof.recommended_start_workers
            ):
                revisit_workers = prof.recommended_start_workers
                revisit_state = ControllerState(
                    current_workers=revisit_workers,
                    last_throughput=result.avg_throughput,
                    last_worker_count=exec_plan.worker_count,
                    direction=ControllerDecision.HOLD,  # observe from here, don't keep going
                    consecutive_holds=0,
                    converged=False,
                )
                ctrl_out = ControllerOutput(
                    recommended_workers=revisit_workers,
                    decision=ControllerDecision.HOLD,
                    reasoning=f"Cold start overshot. Revisiting profiler recommendation ({revisit_workers} workers).",
                    new_state=revisit_state,
                )
                store.set_heuristic("postgresql", object_name, "revisit_lower_applied", 1.0)

            store.save_controller_state("postgresql", object_name, ctrl_out.new_state)

            store.set_heuristic(
                "postgresql", object_name, "throughput_baseline",
                result.avg_throughput,
            )

        # 9. Print summary
        click.echo(f"\n{'='*60}")
        click.echo(f"Run: {result.run_id}  |  {object_name}  |  "
                    f"{_fmt_duration(result.duration_seconds)}")
        click.echo(f"Status: {result.status}")

        # Throughput with delta on one line
        if recent_runs and recent_runs[0].get("avg_throughput", 0) > 0:
            prev_tp_val = recent_runs[0]["avg_throughput"]
            delta_pct = (result.avg_throughput - prev_tp_val) / prev_tp_val * 100
            click.echo(f"Rows: {result.total_rows:,}  |  "
                        f"Throughput: {result.avg_throughput:,.0f} rows/sec "
                        f"({delta_pct:+.1f}% vs previous {_fmt_tp(prev_tp_val)})")
        else:
            click.echo(f"Rows: {result.total_rows:,}  |  "
                        f"Throughput: {result.avg_throughput:,.0f} rows/sec (baseline)")

        # Skew line (chunk duration variance)
        successful_chunks = [cr for cr in result.chunk_results if cr.status == "success"]
        if len(successful_chunks) > 1:
            chunk_durs = sorted(cr.duration_seconds for cr in successful_chunks)
            median_dur = chunk_durs[len(chunk_durs) // 2]
            max_dur = chunk_durs[-1]
            skew_ratio = max_dur / median_dur if median_dur > 0 else 1.0
            if skew_ratio > 3.0:
                skew_label = "HIGH"
            elif skew_ratio > 1.5:
                skew_label = "MODERATE"
            else:
                skew_label = "LOW"
            click.echo(f"Skew (chunk duration): {skew_label} ({skew_ratio:.1f}x max/median)")

        if diag:
            click.echo(f"Diagnosis: {diag.category.value}")
            click.echo(f"  {diag.reasoning}")

        if ctrl_out:
            click.echo(f"Next run: {ctrl_out.decision.value} → "
                        f"{ctrl_out.recommended_workers} workers")
            click.echo(f"  {ctrl_out.reasoning}")

        failed = [cr for cr in result.chunk_results if cr.status == "failed"]
        if failed:
            click.echo(f"\nFailed chunks ({len(failed)}):")
            for fc in failed:
                click.echo(f"  {fc.chunk_id}: {fc.error}")
            click.echo(f"\nRetry: ixtract execute {object_name} --database {database}")
            click.echo(f"  Retry is safe: writer uses atomic finalize, no duplicates.")
            click.echo(f"  To allow partial success: add --allow-partial-failures 5%")

        click.echo(f"{'='*60}")

        sys.exit(0 if result.status == "SUCCESS" else 1)

    finally:
        connector.close()


@cli.command()
@click.argument("run_id", required=False)
@click.option("--object", "object_name", default=None, help="Object name to explain last run")
@click.option("--state-db", default="ixtract_state.db")
def explain(run_id, object_name, state_db):
    """Explain why a run behaved the way it did."""
    from ixtract.state import StateStore

    store = StateStore(state_db)

    if run_id:
        click.echo(f"Explaining run {run_id}...")
        with store._conn() as conn:
            # Try exact match first, then prefix match
            run = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()

            if not run:
                run = conn.execute(
                    "SELECT * FROM runs WHERE run_id LIKE ? ORDER BY start_time DESC LIMIT 1",
                    (f"{run_id}%",)
                ).fetchone()

            if not run:
                click.echo(f"Run {run_id} not found.")
                return

            run = dict(run)

            # Get deviation
            dev = conn.execute(
                "SELECT * FROM deviations WHERE run_id = ?", (run["run_id"],)
            ).fetchone()

            # Get previous run for delta comparison
            prev_run = conn.execute(
                "SELECT avg_throughput, worker_count FROM runs "
                "WHERE source = ? AND object = ? AND start_time < ? "
                "ORDER BY start_time DESC LIMIT 1",
                (run["source"], run["object"], run["start_time"]),
            ).fetchone()

        click.echo(f"\nRun: {run['run_id']}  |  {run['object']}  |  "
                    f"{_fmt_duration(run['duration_seconds'])}")
        click.echo(f"Status: {run['status']}")
        click.echo(f"Rows: {run['total_rows']:,}  |  Workers: {run['worker_count']}")

        # Throughput with delta on one line
        if prev_run and prev_run["avg_throughput"] and prev_run["avg_throughput"] > 0:
            prev_tp_val = prev_run["avg_throughput"]
            delta_pct = (run["avg_throughput"] - prev_tp_val) / prev_tp_val * 100
            click.echo(f"Throughput: {run['avg_throughput']:,.0f} rows/sec "
                        f"({delta_pct:+.1f}% vs previous {_fmt_tp(prev_tp_val)})")
        else:
            click.echo(f"Throughput: {run['avg_throughput']:,.0f} rows/sec (baseline)")

        if dev:
            dev = dict(dev)
            click.echo(f"\nDiagnosis: {dev['diagnosed_cause']}")
            click.echo(f"  {dev['reasoning']}")
            click.echo(f"  {dev['corrective_action']}")

    elif object_name:
        runs = store.get_recent_runs("postgresql", object_name, limit=1)
        if runs:
            # Recurse with the run_id
            ctx = click.get_current_context()
            ctx.invoke(explain, run_id=runs[0]["run_id"], object_name=None, state_db=state_db)
        else:
            click.echo(f"No runs found for {object_name}.")
    else:
        click.echo("Provide a run_id or --object name.")


# ── Helpers ───────────────────────────────────────────────────────────

def _build_config(host, port, database, user, password, connection_string):
    if connection_string:
        return {"connection_string": connection_string}
    return {
        "host": host, "port": port, "database": database,
        "user": user, "password": password,
    }


def _fmt_duration(seconds):
    if seconds is None:
        return "n/a"
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds/60:.1f}m"


def _fmt_latency(ms):
    """Format latency with <1ms floor for sub-millisecond values."""
    if ms is None:
        return "n/a"
    if ms < 1.0:
        return "<1ms"
    return f"{ms:.0f}ms"


def _fmt_tp(rows_sec):
    """Format throughput as human-readable string."""
    if rows_sec is None or rows_sec <= 0:
        return "0"
    if rows_sec >= 1_000_000:
        return f"{rows_sec/1_000_000:.1f}M/s"
    if rows_sec >= 1_000:
        return f"{rows_sec/1_000:.0f}K/s"
    return f"{rows_sec:.0f}/s"


if __name__ == "__main__":
    cli()
