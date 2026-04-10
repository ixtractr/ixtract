"""Execution Engine — the Data Plane workhorse.

Reads the ExecutionPlan, spawns workers, manages the snapshot transaction,
schedules chunks, streams data through bounded buffers to writers, and
collects metrics. Makes zero planning decisions.
"""
from __future__ import annotations

import logging
import queue
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from dataclasses import dataclass, field
from typing import Any, Optional

from ixtract.connectors.postgresql import PostgreSQLConnector
from ixtract.planner import ExecutionPlan, ChunkDefinition, ChunkType, Strategy, SchedulingStrategy, AdaptiveRule, AdaptiveTrigger, AdaptiveAction, RuleFiredRecord
from ixtract.writers.parquet import ParquetWriter, BaseWriter
from ixtract.diagnosis import RunMetrics

log = logging.getLogger("ixtract.engine")


@dataclass
class ChunkResult:
    """Result of executing a single chunk."""
    chunk_id: str
    worker_id: int
    status: str  # "success" | "failed"
    rows: int = 0
    bytes_written: int = 0
    duration_seconds: float = 0.0
    query_ms: float = 0.0
    write_ms: float = 0.0
    output_path: str = ""
    error: str = ""
    retries: int = 0


@dataclass
class ExecutionResult:
    """Result of a complete extraction run."""
    run_id: str
    status: str  # "SUCCESS" | "FAILED" | "CANCELLED"
    total_rows: int
    total_bytes: int
    duration_seconds: float
    avg_throughput: float
    worker_count: int
    effective_workers: float       # computed from chunk timing — average concurrent active workers
    chunk_results: list[ChunkResult]
    metrics: Optional[RunMetrics] = None
    adaptive_rules_fired: list["RuleFiredRecord"] = field(default_factory=list)
    confidence_flag: str = "full"  # "full" | "moderate" | "low"


class ExecutionEngine:
    """Executes an ExtractionPlan against a PostgreSQL source.

    Phase 1 implementation:
        - ThreadPoolExecutor for worker management
        - Greedy or round-robin chunk scheduling
        - REPEATABLE READ snapshot transaction
        - Bounded buffer per worker (backpressure)
        - Parquet writer with atomic finalize
    """

    def __init__(
        self,
        connector: PostgreSQLConnector,
        max_buffer_batches: int = 4,
        batch_size_rows: int = 10_000,
    ) -> None:
        self._connector = connector
        self._max_buffer = max_buffer_batches
        self._batch_size = batch_size_rows

    def execute(self, plan: ExecutionPlan, object_name: str) -> ExecutionResult:
        """Execute the plan end-to-end.

        Args:
            plan: The immutable ExecutionPlan.
            object_name: The source table/object name (e.g., "orders").

        Returns ExecutionResult with per-chunk details and aggregate metrics.
        """
        run_id = f"rx-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        start_time = time.perf_counter()

        log.info(f"Starting extraction run {run_id}: "
                 f"{plan.strategy.value}, {plan.worker_count} workers, "
                 f"{len(plan.chunks)} chunks")

        chunk_results: list[ChunkResult] = []
        writer_config = {
            "output_path": plan.writer_config.output_path,
            "compression": plan.writer_config.compression,
            "naming_pattern": plan.writer_config.naming_pattern,
            "object_name": object_name,
            "temp_path": plan.writer_config.temp_path,
        }

        # Build chunk work queue
        # WORK_STEALING → LPT (Longest Processing Time First):
        #   Sort chunks by estimated_rows descending before queue insertion.
        #   Largest chunks are dispatched first — when small chunks finish
        #   quickly, large chunks are still in flight for other workers.
        #   On uniform tables (equal estimated_rows), sort is a no-op.
        #
        # Note: chunk order in ExecutionPlan.chunks is always PK order
        # (for reproducibility and explainability). Reordering happens only
        # at dispatch time and is not persisted. Dynamic reordering mid-run
        # is deferred to Phase 3 (coordination complexity with shared state).
        chunk_queue: queue.Queue[ChunkDefinition] = queue.Queue()
        dispatch_chunks = list(plan.chunks)
        if plan.scheduling == SchedulingStrategy.WORK_STEALING:
            dispatch_chunks.sort(key=lambda c: c.estimated_rows, reverse=True)
            log.debug(
                f"  LPT dispatch: {len(dispatch_chunks)} chunks sorted by estimated_rows "
                f"[{', '.join(str(c.estimated_rows) for c in dispatch_chunks[:3])}{'...' if len(dispatch_chunks) > 3 else ''}]"
            )
        for chunk in dispatch_chunks:
            chunk_queue.put(chunk)

        failed = False
        max_retries = plan.retry_policy.max_retries

        # Execute with thread pool
        with ThreadPoolExecutor(max_workers=plan.worker_count) as pool:
            futures: list[Future[ChunkResult]] = []

            for worker_id in range(plan.worker_count):
                fut = pool.submit(
                    self._worker_loop,
                    worker_id=worker_id,
                    chunk_queue=chunk_queue,
                    object_name=object_name,
                    plan=plan,
                    writer_config=writer_config,
                    max_retries=max_retries,
                )
                futures.append(fut)

            for fut in as_completed(futures):
                try:
                    results = fut.result()
                    chunk_results.extend(results)
                except Exception as e:
                    log.error(f"Worker failed: {e}")
                    failed = True

        elapsed = time.perf_counter() - start_time
        total_rows = sum(cr.rows for cr in chunk_results if cr.status == "success")
        total_bytes = sum(cr.bytes_written for cr in chunk_results if cr.status == "success")
        failed_chunks = [cr for cr in chunk_results if cr.status == "failed"]

        if failed_chunks:
            status = "FAILED"
        else:
            status = "SUCCESS"

        throughput = total_rows / elapsed if elapsed > 0 else 0

        # Build RunMetrics for deviation analyzer
        chunk_durations = tuple(
            cr.duration_seconds for cr in chunk_results if cr.status == "success"
        )
        worker_idle = tuple(0.05 for _ in range(plan.worker_count))  # simplified Phase 1

        metrics = RunMetrics(
            total_rows=total_rows,
            total_bytes=total_bytes,
            duration_seconds=elapsed,
            worker_count=plan.worker_count,
            avg_throughput_rows_sec=throughput,
            chunk_durations=chunk_durations,
            worker_idle_pcts=worker_idle,
            predicted_duration_seconds=plan.cost_estimate.predicted_duration_seconds,
            predicted_throughput_rows_sec=plan.cost_estimate.predicted_throughput_rows_sec,
        )

        # ── True effective workers ────────────────────────────────
        # = sum(chunk execution times) / wall-clock elapsed
        # = average concurrent active workers over the run duration.
        # INVARIANT: chunk.duration_seconds captures query+write time only.
        # Backoff sleeps occur between chunks in _worker_loop and are
        # intentionally excluded from chunk timing.
        successful_chunks = [cr for cr in chunk_results if cr.status == "success"]
        if elapsed > 0 and successful_chunks:
            effective_workers = sum(
                cr.duration_seconds for cr in successful_chunks
            ) / elapsed
            # Clamp to [1, plan.worker_count] — can't exceed planned, can't be < 1
            effective_workers = max(1.0, min(float(plan.worker_count), effective_workers))
        else:
            effective_workers = float(plan.worker_count)

        # ── Aggregate adaptive rule firings ───────────────────────
        # Re-scan chunk results to count backoff firings per rule.
        # Each worker tracked independently; we aggregate here.
        rules_fired: list[RuleFiredRecord] = []
        total_chunks = len(chunk_results)

        for rule in plan.adaptive_rules:
            if rule.trigger != AdaptiveTrigger.SOURCE_LATENCY_SPIKE:
                continue
            # Count chunks where latency exceeded threshold (proxy for firings)
            fired = [
                cr for cr in successful_chunks
                if cr.query_ms > rule.threshold and cr.query_ms > rule.absolute_floor_ms
            ]
            if not fired:
                continue

            activations = len(fired)
            # Max consecutive: count longest streak of fired chunks in order
            streak = max_streak = 0
            chunk_ids_fired = {cr.chunk_id for cr in fired}
            for cr in chunk_results:
                if cr.chunk_id in chunk_ids_fired:
                    streak += 1
                    max_streak = max(max_streak, streak)
                else:
                    streak = 0

            rate = activations / max(total_chunks, 1)
            if rate <= 0.05:
                confidence_impact = "note"
            elif rate <= 0.20:
                confidence_impact = "moderate"
            else:
                confidence_impact = "low"

            rules_fired.append(RuleFiredRecord(
                rule_id=rule.rule_id,
                activations=activations,
                total_chunks=total_chunks,
                max_consecutive=max_streak,
                confidence_impact=confidence_impact,
            ))

        # ── Run-level confidence flag ──────────────────────────────
        if any(r.confidence_impact == "low" for r in rules_fired):
            confidence_flag = "low"
        elif any(r.confidence_impact == "moderate" for r in rules_fired):
            confidence_flag = "moderate"
        else:
            confidence_flag = "full"

        log.info(f"Run {run_id} {status}: {total_rows:,} rows, "
                 f"{elapsed:.1f}s, {throughput:,.0f} rows/sec, "
                 f"eff_workers={effective_workers:.1f}")

        if failed_chunks:
            for fc in failed_chunks:
                log.error(f"  Chunk {fc.chunk_id} FAILED: {fc.error}")

        return ExecutionResult(
            run_id=run_id,
            status=status,
            total_rows=total_rows,
            total_bytes=total_bytes,
            duration_seconds=elapsed,
            avg_throughput=throughput,
            worker_count=plan.worker_count,
            effective_workers=effective_workers,
            chunk_results=chunk_results,
            metrics=metrics,
            adaptive_rules_fired=rules_fired,
            confidence_flag=confidence_flag,
        )

    def _worker_loop(
        self,
        worker_id: int,
        chunk_queue: queue.Queue[ChunkDefinition],
        object_name: str,
        plan: ExecutionPlan,
        writer_config: dict[str, Any],
        max_retries: int,
    ) -> list[ChunkResult]:
        """Worker loop: pull chunks from queue, extract, write.

        Each worker opens its own REPEATABLE READ connection for snapshot
        consistency. All chunks processed by this worker see the same
        database state.

        Backoff rule is evaluated BETWEEN chunks — after _execute_chunk
        returns and before the next chunk is pulled. Sleep time is
        intentionally excluded from chunk.duration_seconds so effective_workers
        computation reflects only active execution time.
        """
        results: list[ChunkResult] = []

        # Find backoff rule if present in plan
        backoff_rule: Optional[AdaptiveRule] = None
        for rule in plan.adaptive_rules:
            if rule.trigger == AdaptiveTrigger.SOURCE_LATENCY_SPIKE:
                backoff_rule = rule
                break

        # Per-worker backoff state
        backoff_activations = 0
        chunks_since_last_fire = 0  # for cooldown tracking

        # Open a snapshot-isolated connection for this worker
        try:
            snapshot_conn = self._connector.create_snapshot_connection()
        except Exception as e:
            log.error(f"  Worker {worker_id}: failed to create snapshot connection: {e}")
            snapshot_conn = None

        try:
            while True:
                try:
                    chunk = chunk_queue.get_nowait()
                except queue.Empty:
                    break

                result = self._execute_chunk(
                    worker_id, chunk, object_name, plan, writer_config,
                    max_retries, snapshot_conn,
                )
                results.append(result)
                chunks_since_last_fire += 1

                # ── Backoff rule evaluation ─────────────────────────
                # Fires AFTER chunk completes and BEFORE next chunk pull.
                # Sleep time is NOT included in chunk.duration_seconds.
                if (backoff_rule is not None
                        and result.status == "success"
                        and backoff_activations < backoff_rule.max_activations
                        and chunks_since_last_fire > backoff_rule.cooldown_chunks):

                    latency_ms = result.query_ms
                    relative_breach = latency_ms > backoff_rule.threshold
                    absolute_breach = latency_ms > backoff_rule.absolute_floor_ms

                    if relative_breach and absolute_breach:
                        sleep_secs = min(
                            backoff_rule.backoff_sleep_base ** backoff_activations,
                            30.0,
                        )
                        log.warning(
                            f"  Worker {worker_id}: source latency spike "
                            f"({latency_ms:.0f}ms > threshold {backoff_rule.threshold:.0f}ms). "
                            f"Backoff {sleep_secs:.1f}s before next chunk."
                        )
                        time.sleep(sleep_secs)
                        backoff_activations += 1
                        chunks_since_last_fire = 0

        finally:
            if snapshot_conn:
                try:
                    snapshot_conn.close()
                except Exception:
                    pass

        return results

    def _execute_chunk(
        self,
        worker_id: int,
        chunk: ChunkDefinition,
        object_name: str,
        plan: ExecutionPlan,
        writer_config: dict[str, Any],
        max_retries: int,
        snapshot_conn=None,
    ) -> ChunkResult:
        """Execute a single chunk with retry logic."""
        last_error = ""

        for attempt in range(max_retries + 1):
            writer = ParquetWriter()
            try:
                start = time.perf_counter()

                # Build query for this chunk
                sql = self._build_chunk_query(object_name, chunk, plan)

                # Open writer
                writer.open(writer_config, chunk.chunk_id, schema=None)

                # Extract and write — use snapshot connection if available
                total_rows = 0
                total_bytes = 0
                query_start = time.perf_counter()

                if snapshot_conn:
                    data_iter = self._connector.extract_chunk_snapshot(
                        snapshot_conn, sql
                    )
                else:
                    data_iter = self._connector.extract_chunk(
                        object_name, sql
                    )

                for batch in data_iter:
                    query_ms = (time.perf_counter() - query_start) * 1000

                    write_start = time.perf_counter()
                    wr = writer.write_batch(batch)
                    write_ms = (time.perf_counter() - write_start) * 1000

                    total_rows += wr.rows_written
                    total_bytes += wr.bytes_written
                    query_start = time.perf_counter()

                # Finalize — atomic commit
                result = writer.finalize()
                elapsed = time.perf_counter() - start

                log.debug(f"  Worker {worker_id}: chunk {chunk.chunk_id} done "
                          f"({total_rows:,} rows, {elapsed:.1f}s)")

                return ChunkResult(
                    chunk_id=chunk.chunk_id,
                    worker_id=worker_id,
                    status="success",
                    rows=total_rows,
                    bytes_written=result.total_bytes,
                    duration_seconds=elapsed,
                    output_path=result.final_path,
                    retries=attempt,
                )

            except Exception as e:
                last_error = str(e)
                writer.abort()
                log.warning(f"  Worker {worker_id}: chunk {chunk.chunk_id} "
                            f"attempt {attempt+1}/{max_retries+1} failed: {e}")

                if attempt < max_retries:
                    time.sleep(min(2 ** attempt, 30))  # exponential backoff

        return ChunkResult(
            chunk_id=chunk.chunk_id,
            worker_id=worker_id,
            status="failed",
            error=last_error,
            retries=max_retries,
        )

    def _build_chunk_query(
        self, object_name: str, chunk: ChunkDefinition, plan: ExecutionPlan,
    ) -> str:
        """Build the SQL query for a chunk."""
        if chunk.chunk_type == ChunkType.FULL_TABLE:
            return f"SELECT * FROM {object_name}"

        if chunk.chunk_type == ChunkType.RANGE:
            pk = plan.metadata_snapshot.primary_key or "id"
            # Last chunk: inclusive upper bound
            is_last = chunk.chunk_id == plan.chunks[-1].chunk_id
            if is_last:
                return (
                    f"SELECT * FROM {object_name} "
                    f"WHERE {pk} >= {chunk.range_start}"
                )
            return (
                f"SELECT * FROM {object_name} "
                f"WHERE {pk} >= {chunk.range_start} AND {pk} < {chunk.range_end}"
            )

        return f"SELECT * FROM {object_name}"
