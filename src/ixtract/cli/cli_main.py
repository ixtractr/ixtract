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
@click.option("--standard", "detail_level", flag_value="standard", help="Show worker resolution and estimation reasoning")
@click.option("--state-db", default="ixtract_state.db", help="State store path")
def plan(object_name, host, port, database, user, password, connection_string,
         output, compression, max_workers, detail_level, state_db):
    """Show the extraction plan without executing."""
    from ixtract.connectors.postgresql import PostgreSQLConnector
    from ixtract.profiler import SourceProfiler
    from ixtract.planner.planner import plan_extraction, format_plan_summary
    from ixtract.intent import ExtractionIntent, SourceType, TargetType, ExtractionConstraints
    from ixtract.state import StateStore
    from ixtract.controller import ControllerState
    import os

    config = _build_config(host, port, database, user, password, connection_string)
    standard = detail_level == "standard"

    connector = PostgreSQLConnector()
    connector.connect(config)

    try:
        profiler = SourceProfiler(connector)
        prof = profiler.profile(object_name)

        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config=config,
            object_name=object_name,
            target_type=TargetType.PARQUET,
            target_config={"output_path": output, "compression": compression},
            constraints=ExtractionConstraints(max_workers=max_workers),
        )

        store = StateStore(state_db)
        ctrl_state = store.get_controller_state("postgresql", object_name)

        # Measure context (same logic as execute — planner receives it, not measures it)
        from ixtract.context import measure_context, format_context_summary
        ctx = measure_context(
            connector=connector, store=store, object_name=object_name,
            row_estimate=prof.row_estimate, source="postgresql",
        )

        exec_plan, worker_source, tp_estimate = plan_extraction(
            intent, prof, store, ctrl_state, context=ctx
        )
        # Build scheduling label for plan command (no user override here)
        from ixtract.planner import SchedulingStrategy
        if exec_plan.scheduling == SchedulingStrategy.WORK_STEALING:
            plan_sched_src = f"work_stealing (skew-aware scheduling via LPT, CV={prof.pk_distribution_cv:.2f})"
        else:
            plan_sched_src = exec_plan.scheduling.value
        summary = format_plan_summary(exec_plan, prof, ctrl_state,
                                       worker_source=worker_source,
                                       scheduling_source=plan_sched_src)
        click.echo(f"\n{summary}")

        if standard:
            # ── Worker resolution breakdown ──────────────────────
            click.echo(f"\nWorker Resolution:")
            if ctrl_state and ctrl_state.last_throughput > 0:
                click.echo(f"  Source:        controller ({ctrl_state.current_workers} workers"
                           + (", converged" if ctrl_state.converged else ", exploring") + ")")
            else:
                click.echo(f"  Source:        profiler (first run, {prof.recommended_start_workers} workers)")
            click.echo(f"  Source slots:  {prof.available_connections_safe} (50% of {prof.max_connections - prof.active_connections} available)")
            click.echo(f"  System CPUs:   {os.cpu_count() or 'unknown'}")
            if max_workers:
                click.echo(f"  Intent cap:    {max_workers}")
            click.echo(f"  \u2192 Result:      {exec_plan.worker_count} workers")

            # ── Context ───────────────────────────────────────────
            click.echo(f"\nExecution Context:")
            click.echo(format_context_summary(ctx))

            # ── Context-weighted estimation ───────────────────────
            from ixtract.context.estimator import format_estimate_for_cli
            click.echo(f"\nThroughput Estimation:")
            click.echo(format_estimate_for_cli(tp_estimate))
            click.echo(f"  Est. rows:     {exec_plan.cost_estimate.predicted_total_rows:,}")
            click.echo(f"  Est. duration: {_fmt_duration(exec_plan.cost_estimate.predicted_duration_seconds)}")

            # ── Safety detail ────────────────────────────────────
            click.echo(f"\nSafety:")
            conn_pct = exec_plan.worker_count * 100 // prof.max_connections
            click.echo(f"  Connections:   {exec_plan.worker_count}/{prof.max_connections} ({conn_pct}%) \u2714")
            try:
                stat = os.statvfs(output)
                free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                need_gb = exec_plan.cost_estimate.predicted_total_bytes / (1024**3)
                click.echo(f"  Disk:          {free_gb:.1f}GB free, need ~{need_gb:.1f}GB \u2714")
            except OSError:
                pass
            click.echo(f"  Consistency:   snapshot isolation (REPEATABLE READ)")
            est_min = exec_plan.cost_estimate.predicted_duration_seconds / 60
            if est_min > 30:
                click.echo(f"  \u26A0 Snapshot duration ({est_min:.0f}m) exceeds 30m threshold")

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
@click.option("--window-size", default=5, type=int, help="Controller window size (runs before deciding)")
@click.option("--scheduling", default="auto", type=click.Choice(["auto", "greedy", "work_stealing"]),
              help="Chunk dispatch strategy. auto=planner decides, work_stealing=LPT (skew-aware)")
@click.option("--state-db", default="ixtract_state.db")
@click.option("-v", "--verbose", is_flag=True)
def execute(object_name, host, port, database, user, password, connection_string,
            output, compression, max_workers, window_size, scheduling, state_db, verbose):
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

        # 2. Auto-benchmark on first run (no benchmark in store, table large enough)
        _maybe_run_benchmark(
            connector, prof, store, object_name,
            max_workers=max_workers, force=False,
        )

        # 3. Build intent
        intent = ExtractionIntent(
            source_type=SourceType.POSTGRESQL,
            source_config=config,
            object_name=object_name,
            target_type=TargetType.PARQUET,
            target_config={"output_path": output, "compression": compression,
                           "object_name": object_name},
            constraints=ExtractionConstraints(max_workers=max_workers),
        )

        # 4. Controller state
        ctrl_state = store.get_controller_state("postgresql", object_name)
        controller = ParallelismController(ControllerConfig(window_size=window_size))

        # 5. Measure execution context (once — same logic used at plan and run time)
        from ixtract.context import measure_context
        ctx = measure_context(
            connector=connector, store=store, object_name=object_name,
            row_estimate=prof.row_estimate, source="postgresql",
        )

        # 6. Plan
        exec_plan, worker_source, tp_estimate = plan_extraction(
            intent, prof, store, ctrl_state, context=ctx
        )

        # Apply --scheduling override if user specified (not auto)
        from ixtract.planner import SchedulingStrategy
        if scheduling != "auto":
            scheduling_map = {
                "greedy":        SchedulingStrategy.GREEDY,
                "work_stealing": SchedulingStrategy.WORK_STEALING,
            }
            forced_strategy = scheduling_map[scheduling]
            # Rebuild plan with overridden scheduling — ExecutionPlan is frozen,
            # so we must recreate it. All other fields unchanged.
            import dataclasses
            exec_plan = dataclasses.replace(exec_plan, scheduling=forced_strategy)
            scheduling_source = f"{scheduling} (forced by user)"
        else:
            # Auto: show what planner decided and why
            if exec_plan.scheduling == SchedulingStrategy.WORK_STEALING:
                scheduling_source = f"work_stealing (skew-aware scheduling via LPT, CV={prof.pk_distribution_cv:.2f})"
            elif exec_plan.scheduling == SchedulingStrategy.GREEDY:
                scheduling_source = "greedy"
            else:
                scheduling_source = f"{exec_plan.scheduling.value}"
            # Warn if user forced work_stealing on a uniform table
            # (handled in format_plan_summary via scheduling_source label)

        click.echo(format_plan_summary(exec_plan, prof, ctrl_state,
                                       worker_source=worker_source,
                                       scheduling_source=scheduling_source))
        click.echo()

        # 5. Execute
        engine = ExecutionEngine(connector)
        result = engine.execute(exec_plan, object_name)

        # 6. Get previous run BEFORE recording current (so we don't find ourselves)
        recent_runs = store.get_recent_runs("postgresql", object_name, limit=1)

        # 7. Record current run (with execution context for future similarity matching)
        store.record_run_start(
            result.run_id, exec_plan.plan_id, intent.intent_hash(),
            "postgresql", object_name, exec_plan.strategy.value,
            exec_plan.worker_count,
            context_json=ctx.to_json(),
        )
        store.record_run_end(
            result.run_id, result.status.lower(),
            result.total_rows, result.total_bytes,
            result.avg_throughput, result.duration_seconds,
            effective_workers=result.effective_workers,
            confidence_flag=result.confidence_flag,
            adaptive_rules_json=__import__("json").dumps([
                {"rule_id": r.rule_id, "activations": r.activations,
                 "total_chunks": r.total_chunks, "max_consecutive": r.max_consecutive,
                 "confidence_impact": r.confidence_impact}
                for r in result.adaptive_rules_fired
            ]),
        )

        # 7b. Write extraction manifest (always — even for failed runs)
        from ixtract.manifest import build_manifest, write_manifest
        manifest = build_manifest(
            run_id=result.run_id,
            source_type="postgresql",
            object_name=object_name,
            status=result.status.lower(),
            chunk_results=result.chunk_results,
            total_rows=result.total_rows,
            total_bytes=result.total_bytes,
            column_count=prof.column_count,
            primary_key=prof.primary_key,
            primary_key_type=prof.primary_key_type,
            avg_row_bytes=prof.avg_row_bytes,
        )
        manifest_path = write_manifest(manifest, output)

        # 7c. Record per-chunk results (for Phase 2 diagnose/metrics commands)
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

            # Use ACTUAL previous run's worker count and throughput
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

            # 9. Controller evaluation (window-based)
            from ixtract.controller import ControllerState, ControllerOutput, ControllerDecision

            # Seed controller from profiler on first run (not cold start)
            if ctrl_state is None:
                ctrl_state = ControllerState.from_profiler(exec_plan.worker_count)

            # Build throughput window using effective_workers with ±0.75 tolerance.
            # Uses effective_workers (actual average concurrent active workers) not
            # planned worker_count — a run at 4 planned workers where 2 were idle
            # should not be compared against true 4-worker runs.
            # Tolerance ±0.75 avoids discretization cliffs without requiring rounding.
            target_workers = float(ctrl_state.current_workers)
            all_recent = store.get_recent_runs("postgresql", object_name,
                                                limit=window_size + 5)
            throughput_window = [
                r["avg_throughput"] for r in reversed(all_recent)
                if abs(r.get("effective_workers", r.get("worker_count", 0)) - target_workers) <= 0.75
                   and r.get("avg_throughput", 0) > 0
            ][-window_size:]  # take last window_size entries, oldest first

            ctrl_out = controller.evaluate(throughput_window, ctrl_state)
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
            diag_label = diag.category.value
            if "Confirmed" in diag.corrective_action:
                diag_label += " (confirmed)"
            click.echo(f"Diagnosis: {diag_label}")
            click.echo(f"  {diag.reasoning}")

        # ── Parallelism efficiency ────────────────────────────────
        eff = result.effective_workers
        planned = float(result.worker_count)
        utilisation = eff / planned if planned > 0 else 1.0
        if abs(eff - planned) > 0.5:
            click.echo(
                f"Parallelism: {planned:.0f} planned → "
                f"{eff:.1f} effective ({utilisation:.0%} utilisation)"
            )

        # ── Adaptive rules ────────────────────────────────────────
        for rule_rec in result.adaptive_rules_fired:
            rate_pct = rule_rec.activation_rate * 100
            click.echo(
                f"Adaptive ({rule_rec.rule_id}): "
                f"triggered {rule_rec.activations}× "
                f"({rate_pct:.0f}% of chunks) — {rule_rec.confidence_impact}"
            )
            if rule_rec.max_consecutive >= 3:
                click.echo(
                    f"  ⚠ Sustained: {rule_rec.max_consecutive} consecutive chunks affected"
                )

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

        click.echo(f"Manifest: {manifest_path}  |  "
                   f"{manifest.file_count} file(s)  |  "
                   f"status={manifest.status}")

        click.echo(f"{'='*60}")

        sys.exit(0 if result.status == "SUCCESS" else 1)

    finally:
        connector.close()


@cli.command()
@click.argument("run_id", required=False)
@click.option("--object", "object_name", default=None, help="Object name to explain last run")
@click.option("--standard", "detail_level", flag_value="standard", help="Show evidence table and chunk detail")
@click.option("--state-db", default="ixtract_state.db")
def explain(run_id, object_name, detail_level, state_db):
    """Explain why a run behaved the way it did."""
    from ixtract.state import StateStore

    store = StateStore(state_db)
    standard = detail_level == "standard"

    if run_id:
        click.echo(f"Explaining run {run_id}...")
        with store._conn() as conn:
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

            dev = conn.execute(
                "SELECT * FROM deviations WHERE run_id = ?", (run["run_id"],)
            ).fetchone()

            prev_run = conn.execute(
                "SELECT * FROM runs "
                "WHERE source = ? AND object = ? AND start_time < ? "
                "ORDER BY start_time DESC LIMIT 1",
                (run["source"], run["object"], run["start_time"]),
            ).fetchone()

            chunks = []
            worker_rows = []
            if standard:
                chunks = [dict(c) for c in conn.execute(
                    "SELECT * FROM chunks WHERE run_id = ? ORDER BY chunk_id",
                    (run["run_id"],)
                ).fetchall()]
                worker_rows = [dict(w) for w in conn.execute(
                    "SELECT * FROM worker_metrics WHERE run_id = ? ORDER BY worker_id",
                    (run["run_id"],)
                ).fetchall()]

        # ── Summary (always shown) ───────────────────────────────
        click.echo(f"\nRun: {run['run_id']}  |  {run['object']}  |  "
                    f"{_fmt_duration(run['duration_seconds'])}")
        click.echo(f"Status: {run['status']}")
        click.echo(f"Rows: {run['total_rows']:,}  |  Workers: {run['worker_count']}")

        if prev_run and prev_run["avg_throughput"] and prev_run["avg_throughput"] > 0:
            prev_tp_val = prev_run["avg_throughput"]
            delta_pct = (run["avg_throughput"] - prev_tp_val) / prev_tp_val * 100
            click.echo(f"Throughput: {run['avg_throughput']:,.0f} rows/sec "
                        f"({delta_pct:+.1f}% vs previous {_fmt_tp(prev_tp_val)})")
        else:
            click.echo(f"Throughput: {run['avg_throughput']:,.0f} rows/sec (baseline)")

        if dev:
            dev = dict(dev)
            diag_label = dev["diagnosed_cause"]
            if "Confirmed" in (dev.get("corrective_action") or ""):
                diag_label += " (confirmed)"
            click.echo(f"\nDiagnosis: {diag_label}")
            click.echo(f"  {dev['reasoning']}")
            click.echo(f"  {dev['corrective_action']}")

        # ── Standard: evidence table ─────────────────────────────
        if standard and prev_run:
            prev_run = dict(prev_run)
            click.echo(f"\nEvidence:")
            click.echo(f"  {'Metric':<22} {'Previous':>12} {'This Run':>12} {'Delta':>10}")
            click.echo(f"  {'──────':<22} {'────────':>12} {'────────':>12} {'─────':>10}")
            _evidence_row("Workers", prev_run.get("worker_count"), run["worker_count"], is_int=True)
            _evidence_row("Throughput", prev_run.get("avg_throughput"), run["avg_throughput"], fmt="tp")
            _evidence_row("Duration", prev_run.get("duration_seconds"), run["duration_seconds"], fmt="dur")
            _evidence_row("Rows", prev_run.get("total_rows"), run["total_rows"], is_int=True)

        # ── Standard: chunk detail ───────────────────────────────
        if standard and chunks:
            success_c = [c for c in chunks if c["status"] == "success"]
            if success_c:
                durations = sorted(c["duration_seconds"] for c in success_c)
                click.echo(f"\nChunk Detail:")
                click.echo(f"  {'Chunk':<12} {'Rows':>10} {'Duration':>10} {'Rows/sec':>12} {'Worker':>7}")
                click.echo(f"  {'─────':<12} {'────':>10} {'────────':>10} {'────────':>12} {'──────':>7}")
                for c in sorted(success_c, key=lambda x: x["chunk_id"]):
                    dur = c["duration_seconds"]
                    rows = c.get("rows", 0)
                    tp = rows / dur if dur > 0 else 0
                    click.echo(f"  {c['chunk_id']:<12} {rows:>10,} {dur:>9.2f}s {tp:>11,.0f} {c.get('worker_id', '?'):>7}")
                median_d = durations[len(durations) // 2]
                max_d = durations[-1]
                skew = max_d / median_d if median_d > 0 else 1.0
                click.echo(f"\n  Skew: {skew:.1f}x max/median  |  CV: {_chunk_cv(durations):.2f}")

        # ── Standard: worker detail ──────────────────────────────
        if standard and worker_rows:
            click.echo(f"\nWorker Detail:")
            for wm in worker_rows:
                click.echo(f"  Worker {wm['worker_id']}: {wm['chunks_processed']} chunks, "
                           f"{wm.get('total_rows', 0):,} rows")

    elif object_name:
        runs = store.get_recent_runs("postgresql", object_name, limit=1)
        if runs:
            ctx = click.get_current_context()
            ctx.invoke(explain, run_id=runs[0]["run_id"], object_name=None,
                       detail_level=detail_level, state_db=state_db)
        else:
            click.echo(f"No runs found for {object_name}.")
    else:
        click.echo("Provide a run_id or --object name.")


# ══════════════════════════════════════════════════════════════════════
# diagnose — "What's wrong? Is this skew? What should I do?"
# ══════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument("run_id", required=False)
@click.option("--object", "object_name", default=None, help="Object name (uses last run)")
@click.option("--state-db", default="ixtract_state.db")
def diagnose(run_id, object_name, state_db):
    """Diagnose a run: chunk-level breakdown, skew analysis, worker distribution."""
    from ixtract.state import StateStore

    store = StateStore(state_db)
    run, run_id = _resolve_run(store, run_id, object_name)
    if not run:
        return

    chunks = store.get_chunks(run_id)
    if not chunks:
        click.echo(f"No chunk data recorded for run {run_id}.")
        click.echo("(Chunk recording requires Phase 1 final or later.)")
        return

    # ── Chunk analysis ────────────────────────────────────────────
    success_chunks = [c for c in chunks if c["status"] == "success"]
    failed_chunks = [c for c in chunks if c["status"] == "failed"]

    if not success_chunks:
        click.echo(f"\nAll {len(chunks)} chunks failed.")
        for fc in failed_chunks:
            click.echo(f"  {fc['chunk_id']}: {fc.get('error', 'unknown')}")
        return

    durations = sorted(c["duration_seconds"] for c in success_chunks)
    median_dur = durations[len(durations) // 2]
    max_dur = durations[-1]
    min_dur = durations[0]
    skew_ratio = max_dur / median_dur if median_dur > 0 else 1.0

    if skew_ratio > 3.0:
        skew_label, severity = "HIGH", "Severe"
    elif skew_ratio > 1.5:
        skew_label, severity = "MODERATE", "Moderate"
    else:
        skew_label, severity = "LOW", "Minimal"

    # Find slowest and fastest
    chunks_by_dur = sorted(success_chunks, key=lambda c: c["duration_seconds"], reverse=True)
    slowest = chunks_by_dur[0]
    fastest = chunks_by_dur[-1]

    # Per-worker distribution
    worker_stats: dict[int, dict] = {}
    for c in success_chunks:
        wid = c.get("worker_id", 0)
        if wid not in worker_stats:
            worker_stats[wid] = {"chunks": 0, "rows": 0, "time": 0.0}
        worker_stats[wid]["chunks"] += 1
        worker_stats[wid]["rows"] += c.get("rows", 0)
        worker_stats[wid]["time"] += c.get("duration_seconds", 0)

    # Get deviation
    with store._conn() as conn:
        dev = conn.execute(
            "SELECT * FROM deviations WHERE run_id = ?", (run_id,)
        ).fetchone()

    # ── Output ────────────────────────────────────────────────────
    click.echo(f"\nDiagnosis: {run['object']} (run {run_id})")

    # Deviation
    if dev:
        dev = dict(dev)
        cause = dev["diagnosed_cause"]
        # Polish: confirmed label for recovery cases
        if "Confirmed" in (dev.get("corrective_action") or ""):
            cause += " (confirmed)"
        click.echo(f"\n  Category: {cause}")
        click.echo(f"  {dev['reasoning']}")
        click.echo(f"  {dev['corrective_action']}")

    # Skew
    click.echo(f"\nSkew Analysis:")
    click.echo(f"  Severity:   {severity} ({skew_ratio:.1f}x max/median)")
    click.echo(f"  Fastest:    {fastest['chunk_id']} ({fastest['duration_seconds']:.2f}s, "
               f"{fastest.get('rows', 0):,} rows)")
    click.echo(f"  Slowest:    {slowest['chunk_id']} ({slowest['duration_seconds']:.2f}s, "
               f"{slowest.get('rows', 0):,} rows)")
    if skew_ratio > 1.5:
        click.echo(f"  \u26A0 Top 3 slowest chunks:")
        for c in chunks_by_dur[:3]:
            click.echo(f"      {c['chunk_id']}: {c['duration_seconds']:.2f}s "
                       f"({c.get('rows', 0):,} rows)")

    # Workers
    click.echo(f"\nWorker Distribution:")
    for wid in sorted(worker_stats):
        ws = worker_stats[wid]
        click.echo(f"  Worker {wid}: {ws['chunks']} chunks, "
                   f"{ws['rows']:,} rows, {ws['time']:.1f}s total")

    # Suggestions
    click.echo(f"\nSuggestion:")
    if skew_ratio > 3.0:
        click.echo(f"  \u2192 Severe skew detected. Consider work-stealing scheduling (Phase 2).")
        click.echo(f"  \u2192 Or: increase chunk count to distribute hot ranges.")
    elif skew_ratio > 1.5:
        click.echo(f"  \u2192 Moderate skew. Monitor across runs.")
        click.echo(f"  \u2192 Work-stealing (Phase 2) would improve this.")
    elif failed_chunks:
        click.echo(f"  \u2192 {len(failed_chunks)} chunk(s) failed. Check source connectivity.")
    else:
        click.echo(f"  \u2192 No issues detected. Extraction is balanced.")


# ══════════════════════════════════════════════════════════════════════
# history — "Is it getting better or worse? Has it converged?"
# ══════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument("object_name")
@click.option("--limit", default=10, type=int, help="Number of recent runs to show")
@click.option("--state-db", default="ixtract_state.db")
def history(object_name, limit, state_db):
    """Show extraction history and convergence status for an object."""
    from ixtract.state import StateStore

    store = StateStore(state_db)
    runs = store.get_recent_runs("postgresql", object_name, limit=limit)

    if not runs:
        click.echo(f"No runs found for {object_name}.")
        return

    # Reverse to show oldest first
    runs = list(reversed(runs))

    click.echo(f"\nExtraction History: {object_name} (last {len(runs)} runs)")
    click.echo()

    # Table header
    click.echo(f"  {'Run':<8} {'Workers':>7}  {'Throughput':>11}  {'Duration':>9}  {'Status':>8}  {'Diagnosis'}")
    click.echo(f"  {'───':<8} {'───────':>7}  {'──────────':>11}  {'────────':>9}  {'──────':>8}  {'─────────'}")

    # Table rows
    for run in runs:
        run_short = run["run_id"].split("-")[-1] if run["run_id"] else "?"
        tp_str = _fmt_tp(run.get("avg_throughput", 0))
        dur_str = _fmt_duration(run.get("duration_seconds"))

        # Get diagnosis for this run
        with store._conn() as conn:
            dev = conn.execute(
                "SELECT diagnosed_cause FROM deviations WHERE run_id = ? LIMIT 1",
                (run["run_id"],)
            ).fetchone()
        diag_str = dev["diagnosed_cause"] if dev else "—"

        click.echo(f"  {run_short:<8} {run.get('worker_count', '?'):>7}  "
                   f"{tp_str:>11}  {dur_str:>9}  "
                   f"{run.get('status', '?'):>8}  {diag_str}")

    # Convergence status
    ctrl = store.get_controller_state("postgresql", object_name)
    click.echo()
    if ctrl:
        if ctrl.converged:
            click.echo(f"Convergence: CONVERGED at {ctrl.current_workers} workers")
        else:
            click.echo(f"Convergence: Exploring ({ctrl.consecutive_holds}/3 holds, "
                       f"direction: {ctrl.direction.value})")
    else:
        click.echo(f"Convergence: No controller state (run ixtract execute first)")

    # Trends (if enough data)
    if len(runs) >= 3:
        throughputs = [r.get("avg_throughput", 0) for r in runs if r.get("avg_throughput")]
        rows_list = [r.get("total_rows", 0) for r in runs if r.get("total_rows")]

        if len(throughputs) >= 3:
            tp_first = throughputs[0]
            tp_last = throughputs[-1]
            tp_trend = (tp_last - tp_first) / tp_first * 100 if tp_first > 0 else 0
            trend_dir = "\u2191" if tp_trend > 5 else ("\u2193" if tp_trend < -5 else "\u2192")
            click.echo(f"Throughput trend: {trend_dir} {tp_trend:+.1f}% "
                       f"(first: {_fmt_tp(tp_first)}, latest: {_fmt_tp(tp_last)})")


# ══════════════════════════════════════════════════════════════════════
# metrics — "What actually happened? Where was time spent?"
# ══════════════════════════════════════════════════════════════════════

@cli.command()
@click.argument("run_id", required=False)
@click.option("--object", "object_name", default=None, help="Object name (uses last run)")
@click.option("--state-db", default="ixtract_state.db")
def metrics(run_id, object_name, state_db):
    """Show detailed metrics for a run: throughput, workers, chunks, skew."""
    from ixtract.state import StateStore

    store = StateStore(state_db)
    run, run_id = _resolve_run(store, run_id, object_name)
    if not run:
        return

    chunks = store.get_chunks(run_id)

    # ── Throughput ────────────────────────────────────────────────
    click.echo(f"\nRun Metrics: {run_id}")
    click.echo()
    click.echo(f"Throughput:")
    click.echo(f"  Avg: {run['avg_throughput']:,.0f} rows/sec")

    if chunks:
        success_chunks = [c for c in chunks if c["status"] == "success"]
        if success_chunks:
            # Per-chunk throughput
            chunk_tps = []
            for c in success_chunks:
                dur = c.get("duration_seconds", 0)
                rows = c.get("rows", 0)
                if dur > 0:
                    chunk_tps.append((c["chunk_id"], rows / dur, rows, dur))

            if chunk_tps:
                chunk_tps.sort(key=lambda x: x[1], reverse=True)
                best = chunk_tps[0]
                worst = chunk_tps[-1]
                click.echo(f"  Peak: {best[1]:,.0f} rows/sec ({best[0]})")
                click.echo(f"  Min:  {worst[1]:,.0f} rows/sec ({worst[0]})")

    # ── Workers ───────────────────────────────────────────────────
    click.echo(f"\nWorkers:")
    click.echo(f"  Planned: {run['worker_count']}  |  "
               f"Effective: {run.get('effective_workers', run['worker_count'])}")

    # Load worker metrics
    with store._conn() as conn:
        wm_rows = conn.execute(
            "SELECT * FROM worker_metrics WHERE run_id = ? ORDER BY worker_id",
            (run_id,),
        ).fetchall()

    if wm_rows:
        for wm in wm_rows:
            wm = dict(wm)
            click.echo(f"  Worker {wm['worker_id']}: {wm['chunks_processed']} chunks, "
                       f"{wm.get('total_rows', 0):,} rows"
                       + (f", idle {wm['idle_pct']:.0f}%" if wm.get('idle_pct') else ""))

    # ── Chunks ────────────────────────────────────────────────────
    if chunks:
        success = [c for c in chunks if c["status"] == "success"]
        failed = [c for c in chunks if c["status"] == "failed"]

        click.echo(f"\nChunks:")
        click.echo(f"  Total: {len(chunks)}  |  Succeeded: {len(success)}  |  "
                   f"Failed: {len(failed)}")

        if success:
            durations = sorted(c["duration_seconds"] for c in success)
            median_dur = durations[len(durations) // 2]
            max_dur = durations[-1]
            min_dur = durations[0]
            skew_ratio = max_dur / median_dur if median_dur > 0 else 1.0

            fastest = min(success, key=lambda c: c["duration_seconds"])
            slowest = max(success, key=lambda c: c["duration_seconds"])

            click.echo(f"  Fastest: {fastest['chunk_id']} ({fastest['duration_seconds']:.2f}s)")
            click.echo(f"  Slowest: {slowest['chunk_id']} ({slowest['duration_seconds']:.2f}s)")

            if len(success) > 1:
                cv = _chunk_cv(durations)
                click.echo(f"  Chunk CV: {cv:.2f}  |  "
                           f"Skew ratio: {skew_ratio:.1f}x max/median")

        if failed:
            click.echo(f"\n  Failed chunks:")
            for fc in failed:
                click.echo(f"    {fc['chunk_id']}: {fc.get('error', 'unknown')}")

    # ── Run context ───────────────────────────────────────────────
    click.echo(f"\nRun:")
    click.echo(f"  Duration: {_fmt_duration(run['duration_seconds'])}  |  "
               f"Status: {run['status']}  |  Strategy: {run.get('strategy', 'n/a')}")
    click.echo(f"  Rows: {run['total_rows']:,}  |  Bytes: {run.get('total_bytes', 0):,}")


# ── Shared helpers ────────────────────────────────────────────────────

def _resolve_run(store, run_id, object_name):
    """Find a run by ID or by object name (latest). Returns (run_dict, run_id) or (None, None)."""
    if run_id:
        with store._conn() as conn:
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
            return None, None
        return dict(run), run["run_id"]

    elif object_name:
        runs = store.get_recent_runs("postgresql", object_name, limit=1)
        if runs:
            return runs[0], runs[0]["run_id"]
        click.echo(f"No runs found for {object_name}.")
        return None, None

    click.echo("Provide a run_id or --object name.")
    return None, None


def _chunk_cv(durations):
    """Coefficient of variation for chunk durations."""
    n = len(durations)
    if n < 2:
        return 0.0
    mean = sum(durations) / n
    if mean <= 0:
        return 0.0
    var = sum((d - mean) ** 2 for d in durations) / n
    return (var ** 0.5) / mean


# ── Helpers ───────────────────────────────────────────────────────────

def _evidence_row(label, prev_val, curr_val, is_int=False, fmt=None):
    """Print one row of the evidence table."""
    if fmt == "tp":
        pv = _fmt_tp(prev_val) if prev_val else "—"
        cv = _fmt_tp(curr_val) if curr_val else "—"
    elif fmt == "dur":
        pv = _fmt_duration(prev_val) if prev_val else "—"
        cv = _fmt_duration(curr_val) if curr_val else "—"
    elif is_int:
        pv = f"{prev_val:,}" if isinstance(prev_val, (int, float)) else "—"
        cv = f"{curr_val:,}" if isinstance(curr_val, (int, float)) else "—"
    else:
        pv = str(prev_val) if prev_val is not None else "—"
        cv = str(curr_val) if curr_val is not None else "—"

    if isinstance(prev_val, (int, float)) and isinstance(curr_val, (int, float)) and prev_val > 0:
        if is_int:
            delta = f"{int(curr_val - prev_val):+d}"
        else:
            delta = f"{(curr_val - prev_val) / prev_val * 100:+.1f}%"
    else:
        delta = "—"

    click.echo(f"  {label:<22} {pv:>12} {cv:>12} {delta:>10}")

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


def _maybe_run_benchmark(
    connector, prof, store, object_name: str,
    max_workers: int | None = None,
    force: bool = False,
) -> None:
    """Auto-trigger benchmarker if appropriate. Called from execute and benchmark commands.

    Runs silently in execute (no output on skip), verbosely in benchmark command.
    Stores result in state store for planner to consume.
    """
    from ixtract.benchmarker import (
        Benchmarker, BenchmarkerConfig, should_skip_benchmark,
        effective_worker_grid,
    )
    import os

    effective_max = max_workers or min(16, prof.available_connections_safe * 2, os.cpu_count() or 4)
    cfg = BenchmarkerConfig(
        min_workers=1,
        max_workers=effective_max,
        min_table_duration_seconds=0.0 if force else 12.0,
    )

    # Skip if grid is too small (single-point benchmarks are meaningless)
    grid = effective_worker_grid(cfg.max_workers)
    if len(grid) < 2:
        return

    # Skip if benchmark exists and is fresh (unless force)
    if not force:
        existing = store.get_benchmark("postgresql", object_name)
        if existing and not existing.is_stale(prof.row_estimate):
            return

    benchmarker = Benchmarker(connector, cfg)
    result = benchmarker.run(object_name, prof, source_type="postgresql", force=force)

    if result is not None:
        store.save_benchmark("postgresql", object_name, result)


# ── benchmark command ─────────────────────────────────────────────────

@cli.command()
@click.argument("object_name")
@click.option("--host", default="localhost")
@click.option("--port", default=5432, type=int)
@click.option("--database", required=True)
@click.option("--user", default="")
@click.option("--password", default="")
@click.option("--connection-string", default=None)
@click.option("--max-workers", default=None, type=int, help="Cap worker count in benchmark grid")
@click.option("--force", is_flag=True, help="Run even if table is small or benchmark exists")
@click.option("--state-db", default="ixtract_state.db")
def benchmark(object_name, host, port, database, user, password, connection_string,
              max_workers, force, state_db):
    """Calibrate throughput across worker counts before the first extraction run.

    Probes [1, 2, 4, 8] workers (bounded by --max-workers and source slots)
    across 3 non-adjacent PK ranges. Stores the result so the planner can
    seed the controller with a known-good starting point.

    Run automatically on first execute if the table is large enough (>12s est.).
    Use --force to run manually regardless of table size or cached benchmark.
    """
    from ixtract.connectors.postgresql import PostgreSQLConnector
    from ixtract.profiler import SourceProfiler
    from ixtract.benchmarker import (
        Benchmarker, BenchmarkerConfig, effective_worker_grid,
        compute_probe_rows,
    )
    from ixtract.state import StateStore
    import os

    config = _build_config(host, port, database, user, password, connection_string)
    connector = PostgreSQLConnector()
    connector.connect(config)
    store = StateStore(state_db)

    try:
        profiler = SourceProfiler(connector)
        prof = profiler.profile(object_name)

        effective_max = max_workers or min(16, prof.available_connections_safe * 2, os.cpu_count() or 4)
        cfg = BenchmarkerConfig(
            min_workers=1,
            max_workers=effective_max,
            min_table_duration_seconds=0.0 if force else 12.0,
        )

        grid = effective_worker_grid(cfg.max_workers)
        probe_rows = compute_probe_rows(prof.row_estimate)

        # Pre-flight checks
        if len(grid) < 2 and not force:
            click.echo(f"  Skipping: worker grid has only {len(grid)} point(s). "
                       f"Use --force to override.")
            return

        est_duration = prof.size_estimate_bytes / (25 * 1024 * 1024)
        if est_duration < 12.0 and not force:
            click.echo(f"  Skipping: table too small (est. {est_duration:.1f}s extraction). "
                       f"Controller already converges fast. Use --force to override.")
            return

        existing = store.get_benchmark("postgresql", object_name)
        if existing and not existing.is_stale(prof.row_estimate) and not force:
            click.echo(f"  Benchmark already exists (confidence={existing.confidence:.2f}, "
                       f"optimal={existing.optimal_workers} workers). "
                       f"Use --force to re-run.")
            return

        click.echo(f"\nBenchmarking {object_name}...")
        click.echo(f"  Probe size:  {probe_rows:,} rows  |  "
                   f"Ranges: 3  |  Grid: {list(grid)}")
        click.echo()

        benchmarker = Benchmarker(connector, cfg)
        result = benchmarker.run(object_name, prof, source_type="postgresql", force=force)

        if result is None:
            click.echo("  Benchmark skipped (table too small or insufficient PK range).")
            return

        store.save_benchmark("postgresql", object_name, result)

        # ── Results table ─────────────────────────────────────────
        click.echo(f"  {'Workers':<8}  {'Throughput (avg)':<20}  {'Range Variance'}")
        click.echo(f"  {'───────':<8}  {'────────────────':<20}  {'──────────────'}")
        for probe in result.probe_results:
            cv_label = ("LOW" if probe.cv < 0.05 else
                        "MODERATE" if probe.cv < 0.15 else "HIGH")
            marker = " ◀ optimal" if probe.worker_count == result.optimal_workers else ""
            click.echo(
                f"  {probe.worker_count:<8}  "
                f"{probe.avg_throughput:>14,.0f} rows/sec  "
                f"  {cv_label} (CV={probe.cv:.2f}){marker}"
            )

        click.echo()
        sig_label = "meaningful" if result.signal_strength > 0.30 else "flat (worker count doesn't matter much)"
        click.echo(f"Optimal: {result.optimal_workers} workers  |  "
                   f"Confidence: {result.confidence:.2f}  |  "
                   f"Signal: {result.signal_strength:.2f} ({sig_label})")
        click.echo(f"Curve:   {result.curve_shape}")

        if result.confidence < 0.5:
            click.echo(f"\n  ⚠ Low confidence ({result.confidence:.2f}). "
                       f"Planner will fall back to profiler recommendation.")
        else:
            click.echo(f"\nResult stored. Planner will use this until row count "
                       f"grows >20% or 7 days elapse.")
            click.echo(f"Next execute will start at {result.optimal_workers} workers "
                       f"({'benchmarked' if result.signal_strength > 0.30 else 'benchmarked, conservative'}).")

    finally:
        connector.close()


if __name__ == "__main__":
    cli()
