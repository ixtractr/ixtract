"""ixtract CLI — the engineer's interface.

Design principle: "I asked a question → I got a clear answer."
Phase 1 implements summary level only.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Optional

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
# ── RuntimeContext flags (Phase 3A) ──────────────────────────────
@click.option("--context-file", default=None, help="Path to RuntimeContext JSON file")
@click.option("--network-quality", default=None, type=click.Choice(["good", "degraded", "poor"]))
@click.option("--source-load", default=None, type=click.Choice(["low", "normal", "high"]))
@click.option("--max-source-connections", default=None, type=int)
@click.option("--max-memory-mb", default=None, type=int)
@click.option("--concurrent-extractions", default=None, type=int)
@click.option("--source-maintenance", is_flag=True, default=False, help="Source maintenance is scheduled")
@click.option("--priority", default=None, type=click.Choice(["low", "normal", "critical"]))
@click.option("--target-duration", default=None, type=int, help="Target duration in minutes")
@click.option("--egress-budget-mb", default=None, type=float)
@click.option("--maintenance-window", default=None, type=int, help="Maintenance window in minutes")
@click.option("--disk-available-gb", default=None, type=float)
# ── Cost flags (Phase 4A) ────────────────────────────────────────
@click.option("--cost-file", default=None, help="Path to CostConfig JSON file")
@click.option("--compute-rate", default=None, type=float, help="$/hour compute cost")
@click.option("--egress-rate", default=None, type=float, help="$/GB egress cost")
@click.option("--connection-rate", default=None, type=float, help="$/connection/hour cost")
def plan(object_name, host, port, database, user, password, connection_string,
         output, compression, max_workers, detail_level, state_db,
         context_file, network_quality, source_load, max_source_connections,
         max_memory_mb, concurrent_extractions, source_maintenance,
         priority, target_duration, egress_budget_mb, maintenance_window,
         disk_available_gb,
         cost_file, compute_rate, egress_rate, connection_rate):
    """Show the extraction plan without executing."""
    from ixtract.connectors.postgresql import PostgreSQLConnector
    from ixtract.profiler import SourceProfiler
    from ixtract.planner.planner import plan_extraction, format_plan_summary, compute_runtime_analysis
    from ixtract.intent import ExtractionIntent, SourceType, TargetType, ExtractionConstraints
    from ixtract.state import StateStore
    from ixtract.controller import ControllerState
    from ixtract.context.runtime import (
        RuntimeContext, format_runtime_context_table,
        format_worker_resolution, format_advisories, format_verdict,
    )
    import os

    config = _build_config(host, port, database, user, password, connection_string)
    standard = detail_level == "standard"

    # ── Build RuntimeContext ──────────────────────────────────────
    rt_ctx = None
    try:
        rt_ctx = RuntimeContext.from_cli_args(
            context_file=context_file,
            network_quality=network_quality,
            source_load=source_load,
            max_source_connections=max_source_connections,
            max_memory_mb=max_memory_mb,
            concurrent_extractions=concurrent_extractions,
            source_maintenance_scheduled=source_maintenance or None,
            priority=priority,
            target_duration_minutes=target_duration,
            egress_budget_mb=egress_budget_mb,
            maintenance_window_minutes=maintenance_window,
            disk_available_gb=disk_available_gb,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

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

        # source_type_str derived from intent — no hardcoding
        source_type_str = intent.source_type.value

        store = StateStore(state_db)
        ctrl_state = store.get_controller_state(source_type_str, object_name)

        plan_result, worker_source, tp_estimate = plan_extraction(
            intent, prof, store, ctrl_state, runtime_context=rt_ctx,
        )
        summary = format_plan_summary(plan_result, prof, ctrl_state, worker_source=worker_source)
        click.echo(f"\n{summary}")

        # ── RuntimeContext display (Phase 3A) ─────────────────────
        if rt_ctx is not None:
            analysis = compute_runtime_analysis(
                prof, plan_result, ctrl_state, intent, rt_ctx,
            )
            if analysis:
                ctx_table = format_runtime_context_table(rt_ctx)
                if ctx_table:
                    click.echo(f"\n{ctx_table}")
                click.echo(f"\n{format_worker_resolution(analysis['resolution'])}")
                advs = format_advisories(analysis["advisories"])
                if advs:
                    click.echo(f"\n{advs}")
                click.echo(f"\n{format_verdict(analysis['verdict'])}")

        # ── Cost display (Phase 4A) ───────────────────────────────
        from ixtract.cost import CostConfig, compute_cost, compute_cost_comparison, format_cost_comparison
        cost_cfg = None
        try:
            cost_cfg = CostConfig.from_cli_args(
                cost_file=cost_file,
                compute_rate=compute_rate,
                egress_rate=egress_rate,
                connection_rate=connection_rate,
            )
        except ValueError as e:
            click.echo(f"Cost config error: {e}", err=True)

        if cost_cfg is not None and not cost_cfg.is_zero:
            cost_est = compute_cost(
                plan_result.cost_estimate.predicted_duration_seconds,
                plan_result.cost_estimate.predicted_total_bytes,
                plan_result.worker_count,
                cost_cfg,
            )
            if cost_est.total > 0:
                click.echo(f"\n Estimated Cost: ${cost_est.total:.2f} {cost_est.currency}"
                           f"  (compute ${cost_est.compute:.2f}"
                           f" + egress ${cost_est.egress:.2f}"
                           f" + connections ${cost_est.connections:.2f})")

            tp_per_w = plan_result.cost_estimate.predicted_throughput_rows_sec / max(1, plan_result.worker_count)
            comparison = compute_cost_comparison(
                plan_result.worker_count, tp_per_w,
                plan_result.cost_estimate.predicted_total_rows,
                plan_result.cost_estimate.predicted_total_bytes,
                cost_cfg,
                hard_cap=plan_result.worker_bounds[1] if hasattr(plan_result, 'worker_bounds') else 16,
            )
            comp_text = format_cost_comparison(comparison)
            if comp_text:
                click.echo(f"\n{comp_text}")

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
            click.echo(f"  \u2192 Result:      {plan_result.worker_count} workers")

            # ── Estimation reasoning ─────────────────────────────
            baseline = store.get_heuristic(source_type_str, object_name, "throughput_baseline")
            click.echo(f"\nEstimation:")
            if baseline and baseline > 0:
                click.echo(f"  Throughput:    {baseline:,.0f} rows/sec (from stored baseline)")
                click.echo(f"  Confidence:    high (based on actual previous runs)")
            else:
                click.echo(f"  Throughput:    {plan_result.cost_estimate.predicted_throughput_rows_sec:,.0f} rows/sec (estimated from profiler)")
                click.echo(f"  Confidence:    low (no historical data, first run)")
            click.echo(f"  Est. rows:     {plan_result.cost_estimate.predicted_total_rows:,}")
            click.echo(f"  Est. duration: {_fmt_duration(plan_result.cost_estimate.predicted_duration_seconds)}")

            # ── Safety detail ────────────────────────────────────
            click.echo(f"\nSafety:")
            conn_pct = plan_result.worker_count * 100 // prof.max_connections
            click.echo(f"  Connections:   {plan_result.worker_count}/{prof.max_connections} ({conn_pct}%) \u2714")
            try:
                stat = os.statvfs(output)
                free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                need_gb = plan_result.cost_estimate.predicted_total_bytes / (1024**3)
                click.echo(f"  Disk:          {free_gb:.1f}GB free, need ~{need_gb:.1f}GB \u2714")
            except OSError:
                pass
            click.echo(f"  Consistency:   snapshot isolation (REPEATABLE READ)")
            est_min = plan_result.cost_estimate.predicted_duration_seconds / 60
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
@click.option("--state-db", default="ixtract_state.db")
@click.option("-v", "--verbose", is_flag=True)
@click.option("--force", is_flag=True, help="Skip NOT RECOMMENDED prompt")
# ── RuntimeContext flags (Phase 3A) ──────────────────────────────
@click.option("--context-file", default=None, help="Path to RuntimeContext JSON file")
@click.option("--network-quality", default=None, type=click.Choice(["good", "degraded", "poor"]))
@click.option("--source-load", default=None, type=click.Choice(["low", "normal", "high"]))
@click.option("--max-source-connections", default=None, type=int)
@click.option("--max-memory-mb", default=None, type=int)
@click.option("--concurrent-extractions", default=None, type=int)
@click.option("--source-maintenance", is_flag=True, default=False, help="Source maintenance is scheduled")
@click.option("--priority", default=None, type=click.Choice(["low", "normal", "critical"]))
@click.option("--target-duration", default=None, type=int, help="Target duration in minutes")
@click.option("--egress-budget-mb", default=None, type=float)
@click.option("--maintenance-window", default=None, type=int, help="Maintenance window in minutes")
@click.option("--disk-available-gb", default=None, type=float)
# ── Cost flags (Phase 4A) ────────────────────────────────────────
@click.option("--cost-file", default=None, help="Path to CostConfig JSON file")
@click.option("--compute-rate", default=None, type=float, help="$/hour compute cost")
@click.option("--egress-rate", default=None, type=float, help="$/GB egress cost")
@click.option("--connection-rate", default=None, type=float, help="$/connection/hour cost")
def execute(object_name, host, port, database, user, password, connection_string,
            output, compression, max_workers, window_size, state_db, verbose, force,
            context_file, network_quality, source_load, max_source_connections,
            max_memory_mb, concurrent_extractions, source_maintenance,
            priority, target_duration, egress_budget_mb, maintenance_window,
            disk_available_gb,
            cost_file, compute_rate, egress_rate, connection_rate):
    """Extract data from source to Parquet."""
    _setup_logging(verbose)

    from ixtract.connectors.postgresql import PostgreSQLConnector
    from ixtract.profiler import SourceProfiler
    from ixtract.planner.planner import plan_extraction, format_plan_summary, compute_runtime_analysis
    from ixtract.intent import ExtractionIntent, SourceType, TargetType, ExtractionConstraints
    from ixtract.state import StateStore
    from ixtract.controller import ParallelismController, ControllerConfig
    from ixtract.diagnosis import DeviationAnalyzer
    from ixtract.engine import ExecutionEngine
    from ixtract.context.runtime import (
        RuntimeContext, VerdictStatus,
        format_advisories, format_verdict,
    )

    config = _build_config(host, port, database, user, password, connection_string)

    # ── Build RuntimeContext ──────────────────────────────────────
    rt_ctx = None
    try:
        rt_ctx = RuntimeContext.from_cli_args(
            context_file=context_file,
            network_quality=network_quality,
            source_load=source_load,
            max_source_connections=max_source_connections,
            max_memory_mb=max_memory_mb,
            concurrent_extractions=concurrent_extractions,
            source_maintenance_scheduled=source_maintenance or None,
            priority=priority,
            target_duration_minutes=target_duration,
            egress_budget_mb=egress_budget_mb,
            maintenance_window_minutes=maintenance_window,
            disk_available_gb=disk_available_gb,
        )
    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

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

        # 3. source_type_str derived once from intent — used everywhere, no hardcoding
        source_type_str = intent.source_type.value
        ctrl_state = store.get_controller_state(source_type_str, object_name)
        controller = ParallelismController(ControllerConfig(window_size=window_size))

        # 4. Plan
        exec_plan, worker_source, tp_estimate = plan_extraction(
            intent, prof, store, ctrl_state, runtime_context=rt_ctx,
        )
        click.echo(format_plan_summary(exec_plan, prof, ctrl_state, worker_source=worker_source))

        # 4b. RuntimeContext verdict (Phase 3A)
        if rt_ctx is not None:
            analysis = compute_runtime_analysis(
                prof, exec_plan, ctrl_state, intent, rt_ctx,
            )
            if analysis:
                verdict = analysis["verdict"]
                click.echo(f"\n{format_verdict(verdict)}")
                warn_advisories = [
                    a for a in analysis["advisories"]
                    if a.severity.value in ("fail", "warn")
                ]
                for a in warn_advisories:
                    sym = "\u2717" if a.severity.value == "fail" else "\u26A0"
                    click.echo(f"  {sym} {a.message}")

                if verdict.status == VerdictStatus.NOT_RECOMMENDED and not force:
                    if sys.stdin.isatty():
                        click.echo()
                        if not click.confirm("Continue?", default=False):
                            click.echo("Aborted.")
                            sys.exit(0)
                    else:
                        sys.exit(2)

        click.echo()

        # 5. Execute
        engine = ExecutionEngine(connector)
        result = engine.execute(exec_plan, object_name)

        # 6. Get previous run BEFORE recording current (so we don't find ourselves)
        recent_runs = store.get_recent_runs(source_type_str, object_name, limit=1)

        # 7. Record current run (with plan persistence for replay)
        from ixtract._replay import serialize_plan
        pj, pfp, pv = serialize_plan(exec_plan)
        store.record_run_start(
            result.run_id, exec_plan.plan_id, intent.intent_hash(),
            source_type_str, object_name, exec_plan.strategy.value,
            exec_plan.worker_count,
            runtime_context_json=rt_ctx.to_json() if rt_ctx else None,
            plan_json=pj,
            plan_fingerprint=pfp,
        )
        store.record_run_end(
            result.run_id, result.status.lower(),
            result.total_rows, result.total_bytes,
            result.avg_throughput, result.duration_seconds,
            effective_workers=float(result.worker_count),
        )

        # 7b. Record per-chunk results
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

        # 8. Anomaly detection — wired (Item 6, Problem B)
        # Baseline scoped by (source_type, object). Host/database deferred to Phase 5.
        # Runs BEFORE deviation analysis so anomaly is available for display.
        anomaly_result = None
        if result.status == "SUCCESS" and result.avg_throughput > 0:
            from ixtract.diagnosis import detect_anomaly
            baseline_throughputs = store.get_recent_throughputs(
                source_type_str, object_name
            )
            anomaly_result = detect_anomaly(
                current_throughput=result.avg_throughput,
                baseline_throughputs=baseline_throughputs,
            )

        # 9. Deviation analysis
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

            # 10. Controller evaluation (window-based)
            from ixtract.controller import ControllerState, ControllerOutput, ControllerDecision

            # Seed controller from profiler on first run (not cold start)
            if ctrl_state is None:
                ctrl_state = ControllerState.from_profiler(exec_plan.worker_count)

            # Build throughput window: last N runs at the CURRENT worker count
            all_recent = store.get_recent_runs(source_type_str, object_name,
                                                limit=window_size + 5)
            throughput_window = [
                r["avg_throughput"] for r in reversed(all_recent)
                if r.get("worker_count") == ctrl_state.current_workers
                   and r.get("avg_throughput", 0) > 0
            ][-window_size:]

            ctrl_out = controller.evaluate(throughput_window, ctrl_state)
            store.save_controller_state(source_type_str, object_name, ctrl_out.new_state)

            store.set_heuristic(
                source_type_str, object_name, "throughput_baseline",
                result.avg_throughput,
            )

        # 11. Print summary
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

        # Anomaly display — ⚠ for degradation, ↗ for improvement
        # Symbol legend: ⚠ = below baseline, ↗ = above baseline
        if anomaly_result and anomaly_result.is_anomaly:
            direction = anomaly_result.direction
            sym = "\u26A0" if direction == "degradation" else "\u2197"
            click.echo(f"Anomaly: {sym} {anomaly_result.message}")
            click.echo(f"  Baseline: {anomaly_result.baseline_mean:,.0f} "
                       f"\u00b1 {anomaly_result.baseline_stddev:,.0f} rows/sec "
                       f"({anomaly_result.baseline_run_count} runs, "
                       f"z={anomaly_result.z_score:.1f}\u03c3)")

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
@click.option("--standard", "detail_level", flag_value="standard", help="Show evidence table and chunk detail")
@click.option("--state-db", default="ixtract_state.db")
@click.option("--source-type", default=None,
              help="Source type (postgresql/mysql/sqlserver). Inferred from last run if omitted.")
def explain(run_id, object_name, detail_level, state_db, source_type):
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
        src = _infer_source_type(store, object_name, source_type)
        runs = store.get_recent_runs(src, object_name, limit=1)
        if runs:
            ctx = click.get_current_context()
            ctx.invoke(explain, run_id=runs[0]["run_id"], object_name=None,
                       detail_level=detail_level, state_db=state_db,
                       source_type=source_type)
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
@click.option("--source-type", default=None,
              help="Source type (postgresql/mysql/sqlserver). Inferred from last run if omitted.")
def diagnose(run_id, object_name, state_db, source_type):
    """Diagnose a run: chunk-level breakdown, skew analysis, worker distribution."""
    from ixtract.state import StateStore

    store = StateStore(state_db)
    src = _infer_source_type(store, object_name, source_type) if object_name else None
    run, run_id = _resolve_run(store, run_id, object_name, source_type=src)
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

    # Get deviation via public method — no _conn() in diagnose
    dev = store.get_deviation(run_id)

    # ── Output ────────────────────────────────────────────────────
    click.echo(f"\nDiagnosis: {run['object']} (run {run_id})")

    # RuntimeContext exclusion note
    rt_json = store.get_runtime_context(run_id)
    if rt_json:
        click.echo(f"\n  Note: This run used RuntimeContext constraints.")
        click.echo(f"  It was excluded from controller learning to preserve baseline integrity.")

    # Deviation
    if dev:
        cause = dev["diagnosed_cause"]
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

    # Suggestions — state-aware
    strategy = store.get_run_strategy(run_id) or ""
    work_stealing_active = "work_stealing" in strategy.lower()

    click.echo(f"\nSuggestion:")
    if skew_ratio > 3.0:
        if work_stealing_active:
            click.echo(f"  \u2192 Severe skew detected. Work-stealing is active and mitigating.")
            click.echo(f"  \u2192 Consider density-aware chunking (Phase 5) for a permanent fix.")
            click.echo(f"  \u2192 Or: increase chunk count to distribute hot ranges more finely.")
        else:
            click.echo(f"  \u2192 Severe skew detected. Enable work-stealing scheduling.")
            click.echo(f"  \u2192 Or: increase chunk count to distribute hot ranges.")
    elif skew_ratio > 1.5:
        if work_stealing_active:
            click.echo(f"  \u2192 Moderate skew. Work-stealing is active.")
            click.echo(f"  \u2192 Monitor across runs — if persistent, increase chunk count.")
        else:
            click.echo(f"  \u2192 Moderate skew. Monitor across runs.")
            click.echo(f"  \u2192 Work-stealing would improve chunk balance.")
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
@click.option("--source-type", default=None,
              help="Source type (postgresql/mysql/sqlserver). Inferred from last run if omitted.")
def history(object_name, limit, state_db, source_type):
    """Show extraction history and convergence status for an object."""
    from ixtract.state import StateStore

    store = StateStore(state_db)
    src = _infer_source_type(store, object_name, source_type)
    runs = store.get_recent_runs(src, object_name, limit=limit)

    if not runs:
        click.echo(f"No runs found for {object_name}.")
        return

    # Reverse to show oldest first
    runs = list(reversed(runs))

    click.echo(f"\nExtraction History: {object_name} ({src}, last {len(runs)} runs)")
    click.echo()

    # Table header
    click.echo(f"  {'Run':<8} {'Workers':>7}  {'Throughput':>11}  {'Duration':>9}  {'Status':>8}  {'Diagnosis'}")
    click.echo(f"  {'───':<8} {'───────':>7}  {'──────────':>11}  {'────────':>9}  {'──────':>8}  {'─────────'}")

    # Table rows
    for run in runs:
        run_short = run["run_id"].split("-")[-1] if run["run_id"] else "?"
        tp_str = _fmt_tp(run.get("avg_throughput", 0))
        dur_str = _fmt_duration(run.get("duration_seconds"))

        dev = store.get_deviation(run["run_id"])
        diag_str = dev["diagnosed_cause"] if dev else "—"

        click.echo(f"  {run_short:<8} {run.get('worker_count', '?'):>7}  "
                   f"{tp_str:>11}  {dur_str:>9}  "
                   f"{run.get('status', '?'):>8}  {diag_str}")

    # Convergence status
    ctrl = store.get_controller_state(src, object_name)
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
@click.option("--source-type", default=None,
              help="Source type (postgresql/mysql/sqlserver). Inferred from last run if omitted.")
def metrics(run_id, object_name, state_db, source_type):
    """Show detailed metrics for a run: throughput, workers, chunks, skew."""
    from ixtract.state import StateStore

    store = StateStore(state_db)
    src = _infer_source_type(store, object_name, source_type) if object_name else None
    run, run_id = _resolve_run(store, run_id, object_name, source_type=src)
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


@cli.command()
@click.option("--run-id", required=True, help="Run ID to replay")
@click.option("--host", default="localhost")
@click.option("--port", default=5432, type=int)
@click.option("--database", required=True)
@click.option("--user", default="")
@click.option("--password", default="")
@click.option("--connection-string", default=None)
@click.option("--output-dir", default=None, help="Override output directory")
@click.option("--state-db", default="ixtract_state.db")
@click.option("--force", is_flag=True, help="Replay even if plan version mismatch")
def replay(run_id, host, port, database, user, password, connection_string,
           output_dir, state_db, force):
    """Replay a previous extraction using its stored plan.

    Re-executes the exact same plan — no planner, no profiler, no controller.
    Guarantees identical decision surface. New run links back to original.
    """
    from ixtract.connectors.postgresql import PostgreSQLConnector
    from ixtract.state import StateStore
    from ixtract.engine import ExecutionEngine
    from ixtract._replay import (
        deserialize_plan, validate_plan_integrity, validate_plan_version,
        serialize_plan, PlanCorruptionError, UnsupportedPlanVersion,
    )

    store = StateStore(state_db)

    # 1. Load stored plan
    plan_data = store.load_plan_for_replay(run_id)
    if not plan_data:
        click.echo(f"Run '{run_id}' not found.", err=True)
        sys.exit(1)

    plan_json = plan_data.get("plan_json")
    stored_fp = plan_data.get("plan_fingerprint")

    if not plan_json:
        click.echo(f"Run '{run_id}' has no stored plan (pre-Phase 4B run).", err=True)
        sys.exit(1)

    # 2. Validate integrity
    try:
        if stored_fp:
            validate_plan_integrity(plan_json, stored_fp)
        click.echo(f"Plan integrity: verified \u2714")
    except PlanCorruptionError as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    # 3. Deserialize and validate version
    exec_plan = deserialize_plan(plan_json)
    try:
        validate_plan_version(exec_plan.plan_version)
        click.echo(f"Plan version:   {exec_plan.plan_version} \u2714")
    except UnsupportedPlanVersion as e:
        if not force:
            click.echo(f"Error: {e}", err=True)
            click.echo("Use --force to replay anyway.")
            sys.exit(1)
        click.echo(f"Warning: {e} (--force used, proceeding)")

    # 4. Override output dir if requested
    if output_dir:
        from ixtract.planner import WriterConfig
        wc = exec_plan.writer_config
        new_wc = WriterConfig(
            output_format=wc.output_format,
            output_path=output_dir,
            compression=wc.compression,
            partition_by=wc.partition_by,
            naming_pattern=wc.naming_pattern,
            temp_path=output_dir,
            max_file_size_bytes=wc.max_file_size_bytes,
        )
        from ixtract.planner import ExecutionPlan
        exec_plan = ExecutionPlan(
            intent_hash=exec_plan.intent_hash,
            strategy=exec_plan.strategy,
            chunks=exec_plan.chunks,
            cost_estimate=exec_plan.cost_estimate,
            metadata_snapshot=exec_plan.metadata_snapshot,
            worker_count=exec_plan.worker_count,
            worker_bounds=exec_plan.worker_bounds,
            scheduling=exec_plan.scheduling,
            adaptive_rules=exec_plan.adaptive_rules,
            retry_policy=exec_plan.retry_policy,
            writer_config=new_wc,
            plan_id=exec_plan.plan_id,
            plan_version=exec_plan.plan_version,
            created_at=exec_plan.created_at,
        )

    click.echo(f"\nReplaying run {run_id}")
    click.echo(f"  Workers: {exec_plan.worker_count}  |  Chunks: {len(exec_plan.chunks)}"
               f"  |  Strategy: {exec_plan.strategy.value}")
    click.echo()

    # 5. Execute — no planner, no profiler, no controller
    config = _build_config(host, port, database, user, password, connection_string)
    connector = PostgreSQLConnector()
    connector.connect(config)

    try:
        engine = ExecutionEngine(connector)
        object_name = plan_data.get("object", "unknown")
        result = engine.execute(exec_plan, object_name)

        # 6. Record replay run — use source from stored plan, not hardcoded
        pj, pfp, pv = serialize_plan(exec_plan)
        store.record_run_start(
            result.run_id, exec_plan.plan_id,
            plan_data.get("intent_hash", ""),
            plan_data.get("source", "postgresql"), object_name,
            exec_plan.strategy.value, exec_plan.worker_count,
            plan_json=pj, plan_fingerprint=pfp,
            replay_of=run_id,
        )
        store.record_run_end(
            result.run_id, result.status.lower(),
            result.total_rows, result.total_bytes,
            result.avg_throughput, result.duration_seconds,
            effective_workers=result.effective_workers,
        )

        # 7. Display comparison table
        orig_run = None
        with store._conn() as c:
            row = c.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row:
                orig_run = dict(row)

        click.echo(f"\nReplay Complete")
        click.echo()

        # ── Section 1: Results comparison ─────────────────────────
        col_w = 36
        click.echo(f"  {'':16} {'Original Run':<{col_w}} {'Replay Run':<{col_w}}")
        click.echo(f"  {'':16} {'─' * col_w} {'─' * col_w}")
        click.echo(f"  {'Run ID':<16} {run_id:<{col_w}} {result.run_id:<{col_w}}")

        orig_status = orig_run.get("status", "—").upper() if orig_run else "—"
        click.echo(f"  {'Status':<16} {orig_status:<{col_w}} {result.status:<{col_w}}")

        orig_rows = f"{orig_run['total_rows']:,}" if orig_run else "—"
        click.echo(f"  {'Rows':<16} {orig_rows:<{col_w}} {result.total_rows:,}")

        orig_dur = _fmt_duration(orig_run["duration_seconds"]) if orig_run else "—"
        click.echo(f"  {'Duration':<16} {orig_dur:<{col_w}} {_fmt_duration(result.duration_seconds)}")

        orig_tp = f"{orig_run['avg_throughput']:,.0f} rows/sec" if orig_run else "—"
        click.echo(f"  {'Throughput':<16} {orig_tp:<{col_w}} {result.avg_throughput:,.0f} rows/sec")

        # ── Section 2: Decision check ─────────────────────────────
        click.echo()
        click.echo(f"  Decision Check")
        click.echo(f"  {'─' * 14}")

        orig_workers = str(orig_run.get("worker_count", "—")) if orig_run else "—"
        orig_chunks = "—"
        orig_strategy = "—"
        orig_hash = "—"
        if orig_run and orig_run.get("plan_json"):
            try:
                import json as _json
                orig_plan_dict = _json.loads(orig_run["plan_json"])
                orig_chunks = str(len(orig_plan_dict.get("chunks", [])))
                orig_strategy = orig_plan_dict.get("strategy", "—")
            except Exception:
                pass
        if orig_run and orig_run.get("plan_fingerprint"):
            orig_hash = orig_run["plan_fingerprint"][:8] + "..."

        repl_hash = stored_fp[:8] + "..." if stored_fp else "—"
        hash_match = " \u2714 identical" if stored_fp and orig_run and orig_run.get("plan_fingerprint") == stored_fp else ""

        click.echo(f"  {'Workers':<16} {orig_workers:<{col_w}} {exec_plan.worker_count}")
        click.echo(f"  {'Chunks':<16} {orig_chunks:<{col_w}} {len(exec_plan.chunks)}")
        click.echo(f"  {'Strategy':<16} {orig_strategy:<{col_w}} {exec_plan.strategy.value}")
        click.echo(f"  {'Plan Hash':<16} {orig_hash:<{col_w}} {repl_hash}{hash_match}")

        if hash_match:
            click.echo()
            click.echo(f"  Replay Of    {run_id}")
            click.echo(f"  Determinism: \u2714 Verified (plan_fingerprint match)")

        # ── Section 3: Outcome delta ──────────────────────────────
        if orig_run and orig_run.get("avg_throughput", 0) > 0:
            click.echo()
            click.echo(f"  Outcome Delta")
            click.echo(f"  {'─' * 13}")

            tp_delta = (result.avg_throughput - orig_run["avg_throughput"]) / orig_run["avg_throughput"] * 100
            dur_delta = result.duration_seconds - orig_run["duration_seconds"]
            dur_delta_display = 0.0 if abs(dur_delta) < 0.05 else dur_delta
            click.echo(f"  {'Throughput \u0394':<16} {tp_delta:+.1f}%")
            click.echo(f"  {'Duration \u0394':<16} {dur_delta_display:+.1f}s")

        click.echo()
        click.echo(f"  Note: Replay guarantees identical execution decisions, not identical performance.")
        click.echo()
        click.echo(f"  Audit: ixtract explain --run-id {result.run_id}")

    finally:
        connector.close()


# ══════════════════════════════════════════════════════════════════════
# inspect — "If I run this now, should I trust it?"
# ══════════════════════════════════════════════════════════════════════

# Canonical set of deviation causes that represent real anomalies.
# Using a frozenset avoids string drift across commands.
_ANOMALY_CAUSES = frozenset({
    "THROUGHPUT_DROP_MODERATE",
    "THROUGHPUT_DROP_SEVERE",
    "WORKER_UNDERUTILIZATION",
    "LATENCY_SPIKE",
    "SOURCE_LOAD_HIGH",
    "DATA_SKEW",
    "CHUNK_FAILURE",
    "ANOMALY_DETECTED",
    "anomaly",
})


@cli.command()
@click.argument("object_name")
@click.option("--state-db", default="ixtract_state.db")
@click.option("--source-type", default=None,
              help="Source type (postgresql/mysql/sqlserver). Inferred from last run if omitted.")
@click.option("--profile-stale-hours", default=24, type=int,
              help="Hours after which a profile is considered stale")
def inspect(object_name, state_db, source_type, profile_stale_hours):
    """Show operational health of an extraction target.

    Answers: 'If I run this now, should I trust it?'

    Aggregates signals from last run, controller state, anomalies,
    and profile freshness into a single health verdict.

    Exit codes (contract):
      0 → HEALTHY
      1 → NEEDS ATTENTION
      2 → DEGRADED
    """
    from ixtract.state import StateStore
    from datetime import datetime, timezone, timedelta

    store = StateStore(state_db)

    # ── Resolve source type — delegate to canonical helper ────────
    try:
        active_source = _infer_source_type(store, object_name, source_type)
    except click.UsageError:
        # No runs yet — show helpful message below, continue with unknown
        active_source = source_type or "unknown"

    # ── Gather signals ────────────────────────────────────────────
    runs = store.get_recent_runs(active_source, object_name, limit=5)
    ctrl = store.get_controller_state(active_source, object_name)
    profile = store.get_profile(active_source, object_name)

    if not runs:
        click.echo(f"\nNo runs found for '{object_name}'.")
        click.echo(f"Run `ixtract execute <object> ...` first.")
        sys.exit(1)

    last_run = runs[0]
    now = datetime.now(timezone.utc)

    # ── Last Run section ──────────────────────────────────────────
    click.echo(f"\nInspect: {object_name} ({active_source})")
    click.echo()
    click.echo(f"Last Run")
    click.echo(f"{'─' * 44}")
    click.echo(f"  Run ID        {last_run['run_id']}")
    click.echo(f"  When          {_fmt_age(last_run.get('start_time', ''), now)}")
    click.echo(f"  Duration      {_fmt_duration(last_run.get('duration_seconds'))}")
    click.echo(f"  Throughput    {_fmt_tp(last_run.get('avg_throughput', 0))}")
    click.echo(f"  Workers       {last_run.get('worker_count', '?')}")
    click.echo(f"  Status        {last_run.get('status', '?').upper()}")

    # ── Controller section ────────────────────────────────────────
    click.echo()
    click.echo(f"Controller")
    click.echo(f"{'─' * 44}")

    controller_healthy = True
    if ctrl:
        if ctrl.converged:
            ctrl_state_str = "CONVERGED"
        elif getattr(ctrl, 'escape_mode', False):
            ctrl_state_str = "ESCAPE MODE"
            controller_healthy = False
        else:
            ctrl_state_str = "EXPLORING"
            controller_healthy = False

        click.echo(f"  State         {ctrl_state_str}")
        click.echo(f"  Workers       {ctrl.current_workers}")
        click.echo(f"  Trend         {_fmt_trend(runs)}")
    else:
        click.echo(f"  State         NO STATE")
        click.echo(f"  Workers       —")
        click.echo(f"  Trend         —")
        controller_healthy = False

    # ── Anomalies section ─────────────────────────────────────────
    click.echo()
    click.echo(f"Anomalies")
    click.echo(f"{'─' * 44}")

    anomaly_signals = []
    for run in runs[:3]:
        dev = store.get_deviation(run["run_id"])
        if dev and dev.get("diagnosed_cause") in _ANOMALY_CAUSES:
            anomaly_signals.append({
                "run_id": run["run_id"],
                "cause": dev["diagnosed_cause"],
            })

    if anomaly_signals:
        most_recent = anomaly_signals[0]
        click.echo(f"  Last anomaly  {most_recent['cause']}")
        click.echo(f"  Run           {most_recent['run_id']}")
        click.echo(f"  Details       ixtract diagnose --object {object_name}")
        if len(anomaly_signals) > 1:
            click.echo(f"  Recurrence    {len(anomaly_signals)}/3 recent runs affected")
    else:
        click.echo(f"  None")

    # ── Profile section ───────────────────────────────────────────
    click.echo()
    click.echo(f"Profile")
    click.echo(f"{'─' * 44}")

    profile_stale = False
    if profile:
        profiled_str = profile.get("profiled_at", "")
        profile_stale, profiled_when = _profile_freshness(
            profiled_str, now, profile_stale_hours
        )
        freshness_str = "\u26A0 stale" if profile_stale else "\u2714 up-to-date"
        click.echo(f"  Last profiled {profiled_when}")
        click.echo(f"  Freshness     {freshness_str}")
        row_est = profile.get("row_estimate", 0)
        if row_est:
            click.echo(f"  Row estimate  {row_est:,}")
    else:
        click.echo(f"  Not profiled  (run `ixtract profile {object_name} --database <db>`)")
        profile_stale = True

    # ── Health Verdict ────────────────────────────────────────────
    click.echo()
    click.echo(f"Health")
    click.echo(f"{'─' * 44}")

    verdict, verdict_note = _compute_health_verdict(
        last_run=last_run,
        anomaly_signals=anomaly_signals,
        runs=runs,
        profile_stale=profile_stale,
        controller_healthy=controller_healthy,
    )

    symbol = {"HEALTHY": "\u2714", "NEEDS ATTENTION": "\u26A0", "DEGRADED": "\u2717"}[verdict]
    click.echo(f"  {verdict} {symbol}")
    click.echo(f"  {verdict_note}")
    click.echo()

    # Exit code contract: 0=HEALTHY, 1=NEEDS ATTENTION, 2=DEGRADED
    exit_codes = {"HEALTHY": 0, "NEEDS ATTENTION": 1, "DEGRADED": 2}
    sys.exit(exit_codes[verdict])


# ── benchmark ─────────────────────────────────────────────────────────

@cli.command()
@click.argument("object_name")
@click.option("--host", default="localhost")
@click.option("--port", default=5432, type=int)
@click.option("--database", required=True)
@click.option("--user", default="")
@click.option("--password", default="")
@click.option("--connection-string", default=None)
@click.option("--state-db", default="ixtract_state.db")
@click.option("--probe-rows", default=10000, type=int, help="Rows per probe range")
@click.option("--worker-grid", default="1,2,4,8", help="Comma-separated worker counts to test")
def benchmark(object_name, host, port, database, user, password, connection_string,
              state_db, probe_rows, worker_grid):
    """Calibrate throughput curve for a table across worker counts.

    Runs multi-worker probes and stores the result. The planner uses this
    to pick the optimal worker count instead of relying on the controller alone.
    """
    from ixtract.connectors.postgresql import PostgreSQLConnector
    from ixtract.profiler import SourceProfiler
    from ixtract.benchmarker import Benchmarker, BenchmarkConfig
    from ixtract.state import StateStore

    config = _build_config(host, port, database, user, password, connection_string)

    try:
        grid = [int(w.strip()) for w in worker_grid.split(",") if w.strip()]
    except ValueError:
        click.echo("Error: --worker-grid must be comma-separated integers (e.g. 1,2,4,8)", err=True)
        sys.exit(1)

    connector = PostgreSQLConnector()
    connector.connect(config)

    try:
        profiler = SourceProfiler(connector)
        prof = profiler.profile(object_name)

        click.echo(f"\nBenchmarking {object_name}...")
        click.echo(f"  Worker grid:  {grid}")
        click.echo(f"  Probe rows:   {probe_rows:,} per range")
        click.echo()

        bench_config = BenchmarkConfig(probe_rows=probe_rows, worker_grid=grid)
        benchmarker = Benchmarker(connector, bench_config)
        result = benchmarker.run(prof)

        store = StateStore(state_db)
        store.save_benchmark("postgresql", object_name, result)

        click.echo(f"Benchmark complete")
        click.echo(f"  Optimal workers:  {result.optimal_workers}")
        click.echo(f"  Confidence:       {result.confidence:.2f}")
        click.echo(f"  Curve shape:      {result.curve_shape}")
        click.echo()
        click.echo(f"Throughput by worker count:")
        for w, tp in sorted(result.throughput_by_workers.items()):
            marker = " ← optimal" if w == result.optimal_workers else ""
            click.echo(f"  {w:>3} workers:  {tp:>12,.0f} rows/sec{marker}")

        click.echo(f"\nStored. Next `ixtract plan` will use benchmark data.")

    finally:
        connector.close()


# ── Shared helpers ────────────────────────────────────────────────────

def _infer_source_type(
    store: "StateStore",
    object_name: str,
    explicit: Optional[str] = None,
) -> str:
    """Infer source_type for read-only CLI commands.

    Priority:
      1. Explicit --source-type flag (if provided)
      2. Most recent run in state store for this object
      3. Raise click.UsageError if neither exists

    Keeps CLI commands stateless — no DB connection required for inspection.
    """
    if explicit:
        return explicit
    last = store.get_last_run(object_name)
    if last:
        return last["source"]
    raise click.UsageError(
        f"No runs found for '{object_name}' and no --source-type provided. "
        f"Run `ixtract execute <object> ...` first."
    )


def _resolve_run(store, run_id, object_name, source_type: Optional[str] = None):
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
        src = source_type or _infer_source_type(store, object_name)
        runs = store.get_recent_runs(src, object_name, limit=1)
        if runs:
            return runs[0], runs[0]["run_id"]
        click.echo(f"No runs found for {object_name}.")
        return None, None

    click.echo("Provide a run_id or --object name.")
    return None, None


def _fmt_age(start_str: str, now) -> str:
    """Format a UTC ISO timestamp as a human-readable age string."""
    if not start_str:
        return "unknown"
    try:
        from datetime import datetime, timezone
        start_dt = datetime.fromisoformat(start_str.replace("Z", "+00:00"))
        age = now - start_dt
        secs = age.total_seconds()
        if secs < 60:
            return f"{int(secs)} seconds ago"
        elif secs < 3600:
            return f"{int(secs / 60)} minutes ago"
        elif secs < 86400:
            return f"{int(secs / 3600)} hours ago"
        else:
            return f"{int(secs / 86400)} days ago"
    except Exception:
        return start_str[:19]


def _fmt_trend(runs: list) -> str:
    """Compute trend string from last N runs. Guards against near-zero baselines."""
    throughputs = [
        r.get("avg_throughput", 0) for r in reversed(runs)
        if r.get("avg_throughput", 0) > 0
    ]
    if len(throughputs) < 3:
        return "insufficient history"
    first_tp = throughputs[0]
    last_tp = throughputs[-1]
    if first_tp < 1000:
        return "insufficient signal"
    drift_pct = (last_tp - first_tp) / first_tp * 100
    if abs(drift_pct) < 5:
        return f"stable (drift {drift_pct:+.1f}%)"
    elif drift_pct > 0:
        return f"improving ({drift_pct:+.1f}%)"
    else:
        return f"degrading ({drift_pct:+.1f}%)"


def _profile_freshness(profiled_str: str, now, stale_hours: int):
    """Return (is_stale: bool, human_age: str) for a profile timestamp."""
    from datetime import datetime, timezone, timedelta
    if not profiled_str:
        return True, "unknown"
    try:
        profiled_dt = datetime.fromisoformat(profiled_str.replace("Z", "+00:00"))
        age = now - profiled_dt
        is_stale = age > timedelta(hours=stale_hours)
        secs = age.total_seconds()
        if secs < 3600:
            human = f"{int(secs / 60)} minutes ago"
        elif secs < 86400:
            human = f"{int(secs / 3600)} hours ago"
        else:
            human = f"{int(secs / 86400)} days ago"
        return is_stale, human
    except Exception:
        return True, profiled_str[:19] if profiled_str else "unknown"


def _compute_health_verdict(
    last_run: dict,
    anomaly_signals: list,
    runs: list,
    profile_stale: bool,
    controller_healthy: bool,
) -> tuple:
    """Compute (verdict, note) for inspect health section.

    Priority order (highest to lowest):
      DEGRADED  — last run failed, repeated anomalies, strong negative trend
      NEEDS ATTENTION — single anomaly, stale profile, controller not healthy
      HEALTHY   — none of the above

    Scope rule: Health is determined primarily from the most recent run.
    History informs trend but does not override verdict classification.
    A table with three bad runs followed by a clean run is HEALTHY.
    """
    last_failed = last_run.get("status", "").upper() != "SUCCESS"
    repeated_anomalies = len(anomaly_signals) >= 2

    strong_negative_trend = False
    throughputs = [
        r.get("avg_throughput", 0) for r in reversed(runs)
        if r.get("avg_throughput", 0) > 0
    ]
    if len(throughputs) >= 3 and throughputs[0] >= 1000:
        drift = (throughputs[-1] - throughputs[0]) / throughputs[0]
        strong_negative_trend = drift < -0.20

    if last_failed or repeated_anomalies or strong_negative_trend:
        reasons = []
        if last_failed:
            reasons.append("last run failed")
        if repeated_anomalies:
            reasons.append("repeated anomalies detected")
        if strong_negative_trend:
            reasons.append("strong negative throughput trend")
        note = "Degraded: " + ", ".join(reasons) + ". Investigate before next run."
        return "DEGRADED", note

    # Anomaly takes priority over stale profile in NEEDS ATTENTION tier
    if anomaly_signals:
        return "NEEDS ATTENTION", (
            "Anomaly detected in a recent run. "
            f"Run `ixtract diagnose` for details."
        )

    if profile_stale or not controller_healthy:
        reasons = []
        if profile_stale:
            reasons.append("profile is stale")
        if not controller_healthy:
            reasons.append("controller is exploring or has no state")
        note = "Attention needed: " + ", ".join(reasons) + "."
        return "NEEDS ATTENTION", note

    return "HEALTHY", "System is stable and operating within expected bounds."


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


if __name__ == "__main__":
    cli()
