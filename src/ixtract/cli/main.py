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
        click.echo(f"  Latency: p50={prof.latency_p50_ms:.0f}ms  p95={prof.latency_p95_ms:.0f}ms")
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
        result = engine.execute(exec_plan)

        # 6. Record run
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

        # 7. Deviation analysis
        if result.metrics:
            analyzer = DeviationAnalyzer()
            # Set previous run context
            prev_tp = ctrl_state.last_throughput if ctrl_state else 0.0
            prev_wc = ctrl_state.last_worker_count if ctrl_state else 0
            metrics = result.metrics
            # Create updated metrics with previous run data
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
            if ctrl_state is None:
                from ixtract.controller import ControllerState
                ctrl_state = ControllerState.cold_start(controller.config)

            ctrl_out = controller.evaluate(result.avg_throughput, ctrl_state)
            store.save_controller_state("postgresql", object_name, ctrl_out.new_state)

            # Update throughput baseline
            store.set_heuristic(
                "postgresql", object_name, "throughput_baseline",
                result.avg_throughput,
            )

        # 9. Print summary
        click.echo(f"\n{'='*60}")
        click.echo(f"Run: {result.run_id}  |  {object_name}  |  "
                    f"{_fmt_duration(result.duration_seconds)}")
        click.echo(f"Status: {result.status}")
        click.echo(f"Rows: {result.total_rows:,}  |  "
                    f"Throughput: {result.avg_throughput:,.0f} rows/sec")

        if result.metrics and 'diag' in dir():
            click.echo(f"Diagnosis: {diag.category.value}")
            click.echo(f"  {diag.reasoning}")

        if 'ctrl_out' in dir():
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
        # Look up specific run
        click.echo(f"Explaining run {run_id}...")
        # Phase 1: basic explain from state store
        with store._conn() as conn:
            run = conn.execute(
                __import__("sqlite3").connect(":memory:").execute("SELECT 1").description and
                None or
                conn.execute(
                    "SELECT * FROM runs WHERE run_id = ?", (run_id,)
                )
            ).fetchone()
            if not run:
                # Try to find by prefix
                runs = conn.execute(
                    "SELECT * FROM runs WHERE run_id LIKE ? ORDER BY start_time DESC LIMIT 1",
                    (f"{run_id}%",)
                ).fetchone()
                run = runs

            if not run:
                click.echo(f"Run {run_id} not found.")
                return

            run = dict(run)

            # Get deviation
            dev = conn.execute(
                "SELECT * FROM deviations WHERE run_id = ?", (run["run_id"],)
            ).fetchone()

        click.echo(f"\nRun: {run['run_id']}  |  {run['object']}  |  "
                    f"{_fmt_duration(run['duration_seconds'])}")
        click.echo(f"Status: {run['status']}")
        click.echo(f"Throughput: {run['avg_throughput']:,.0f} rows/sec")
        click.echo(f"Rows: {run['total_rows']:,}  |  Workers: {run['worker_count']}")

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


if __name__ == "__main__":
    cli()
