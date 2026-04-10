"""Work-Stealing (LPT) Scheduler Tests — Phase 2C.

Tests for:
  - LPT dispatch: chunks sorted by estimated_rows desc before queue insertion
  - Activation: only when plan.scheduling == WORK_STEALING
  - Greedy/round_robin: no reordering (natural order preserved)
  - Profiler recommendation: work_stealing when CV > 1.0, round_robin otherwise
  - Uniform table: LPT is a no-op (equal estimated_rows → stable sort)
  - Scheduling override: forced work_stealing on uniform table
  - format_plan_summary: scheduling line present on every run
  - Engine dispatch invariant: plan.chunks always PK order, queue order may differ

All tests are pure simulation — no database required.

Run: python -m unittest tests.simulation.test_work_stealing -v
"""
from __future__ import annotations

import os
import queue
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.planner import (
    ExecutionPlan, Strategy, SchedulingStrategy, ChunkDefinition, ChunkType,
    CostEstimate, MetadataSnapshot, RetryPolicy, WriterConfig,
)
from ixtract.planner.planner import format_plan_summary
from ixtract.profiler import SourceProfile


# ── Helpers ───────────────────────────────────────────────────────────

def make_chunk(chunk_id: str, estimated_rows: int, priority: int = 0) -> ChunkDefinition:
    return ChunkDefinition(
        chunk_id=chunk_id,
        chunk_type=ChunkType.RANGE,
        estimated_rows=estimated_rows,
        estimated_bytes=estimated_rows * 150,
        range_start=priority * 100_000,
        range_end=(priority + 1) * 100_000,
        priority=priority,
    )


def make_plan(
    chunks: list[ChunkDefinition],
    scheduling: SchedulingStrategy = SchedulingStrategy.WORK_STEALING,
    worker_count: int = 2,
) -> ExecutionPlan:
    return ExecutionPlan(
        intent_hash="test-hash",
        strategy=Strategy.RANGE_CHUNKING,
        chunks=tuple(chunks),
        worker_count=worker_count,
        worker_bounds=(1, 8),
        scheduling=scheduling,
        adaptive_rules=(),
        retry_policy=RetryPolicy(),
        cost_estimate=CostEstimate(10.0, 100_000.0, 1_000_000, 150_000_000),
        writer_config=WriterConfig("parquet", "./output", "snappy", "{object}_{chunk_id}.parquet"),
        metadata_snapshot=MetadataSnapshot(1_000_000, 150_000_000, 10, "id", "integer"),
    )


def make_profile(cv: float = 0.0) -> SourceProfile:
    return SourceProfile(
        object_name="orders", row_estimate=1_200_000,
        size_estimate_bytes=166 * 1024 * 1024,
        column_count=10, avg_row_bytes=145,
        primary_key="id", primary_key_type="integer",
        pk_min=1, pk_max=1_200_000,
        pk_distribution_cv=cv, has_usable_pk=True,
        latency_p50_ms=0.5, latency_p95_ms=2.0, connection_ms=1.0,
        max_connections=100, active_connections=2, available_connections_safe=49,
        recommended_start_workers=2, recommended_strategy="range_chunking",
        recommended_scheduling="work_stealing" if cv > 1.0 else "round_robin",
    )


def simulated_dispatch_order(plan: ExecutionPlan) -> list[str]:
    """Reproduce the engine's dispatch logic and return chunk_id order."""
    dispatch_chunks = list(plan.chunks)
    if plan.scheduling == SchedulingStrategy.WORK_STEALING:
        dispatch_chunks.sort(key=lambda c: c.estimated_rows, reverse=True)
    q: queue.Queue[ChunkDefinition] = queue.Queue()
    for c in dispatch_chunks:
        q.put(c)
    order = []
    while not q.empty():
        order.append(q.get_nowait().chunk_id)
    return order


# ── LPT dispatch ordering ─────────────────────────────────────────────

class TestLPTDispatch(unittest.TestCase):

    def test_largest_chunk_dispatched_first(self):
        """WORK_STEALING: chunk with most estimated_rows is first in queue."""
        chunks = [
            make_chunk("chunk_001", 100_000),
            make_chunk("chunk_002", 800_000),
            make_chunk("chunk_003", 50_000),
        ]
        plan = make_plan(chunks, SchedulingStrategy.WORK_STEALING)
        order = simulated_dispatch_order(plan)
        self.assertEqual(order[0], "chunk_002")

    def test_descending_order(self):
        """Full dispatch order is strictly descending by estimated_rows."""
        chunks = [
            make_chunk("c1", 100_000),
            make_chunk("c2", 800_000),
            make_chunk("c3", 400_000),
            make_chunk("c4", 50_000),
        ]
        plan = make_plan(chunks, SchedulingStrategy.WORK_STEALING)
        order = simulated_dispatch_order(plan)
        self.assertEqual(order, ["c2", "c3", "c1", "c4"])

    def test_plan_chunks_unchanged(self):
        """plan.chunks must remain in original PK order — only queue is reordered."""
        chunks = [
            make_chunk("c1", 100_000, priority=0),
            make_chunk("c2", 800_000, priority=1),
            make_chunk("c3", 50_000,  priority=2),
        ]
        plan = make_plan(chunks, SchedulingStrategy.WORK_STEALING)
        # plan.chunks always PK order
        self.assertEqual([c.chunk_id for c in plan.chunks], ["c1", "c2", "c3"])
        # dispatch order is LPT
        order = simulated_dispatch_order(plan)
        self.assertEqual(order[0], "c2")

    def test_greedy_preserves_insertion_order(self):
        """GREEDY: no reordering — chunks dispatched in plan order."""
        chunks = [
            make_chunk("c1", 100_000),
            make_chunk("c2", 800_000),
            make_chunk("c3", 50_000),
        ]
        plan = make_plan(chunks, SchedulingStrategy.GREEDY)
        order = simulated_dispatch_order(plan)
        self.assertEqual(order, ["c1", "c2", "c3"])

    def test_round_robin_preserves_insertion_order(self):
        """ROUND_ROBIN: no reordering — chunks dispatched in plan order."""
        chunks = [
            make_chunk("c1", 100_000),
            make_chunk("c2", 800_000),
            make_chunk("c3", 50_000),
        ]
        plan = make_plan(chunks, SchedulingStrategy.ROUND_ROBIN)
        order = simulated_dispatch_order(plan)
        self.assertEqual(order, ["c1", "c2", "c3"])

    def test_uniform_chunks_lpt_is_noop(self):
        """Equal estimated_rows → LPT sort is a stable no-op (insertion order preserved)."""
        chunks = [make_chunk(f"c{i}", 300_000, priority=i) for i in range(4)]
        plan = make_plan(chunks, SchedulingStrategy.WORK_STEALING)
        order = simulated_dispatch_order(plan)
        # Python sort is stable: equal elements maintain relative order
        self.assertEqual(order, ["c0", "c1", "c2", "c3"])

    def test_single_chunk_no_error(self):
        """Single chunk with WORK_STEALING doesn't error."""
        chunks = [make_chunk("c1", 500_000)]
        plan = make_plan(chunks, SchedulingStrategy.WORK_STEALING)
        order = simulated_dispatch_order(plan)
        self.assertEqual(order, ["c1"])

    def test_two_workers_skewed_ideal_case(self):
        """Classic skew scenario: large chunks first → workers stay busy."""
        # Without LPT: [800K, 50K, 750K, 50K] → W0 gets 800K+50K, W1 gets 750K+50K
        # With LPT:    [800K, 750K, 50K, 50K]  → W0 gets 800K+50K, W1 gets 750K+50K
        # Both roughly balanced — the key is large chunks don't pile up at the end
        chunks = [
            make_chunk("c1", 800_000),
            make_chunk("c2", 50_000),
            make_chunk("c3", 750_000),
            make_chunk("c4", 50_000),
        ]
        plan = make_plan(chunks, SchedulingStrategy.WORK_STEALING)
        order = simulated_dispatch_order(plan)
        # Largest two are dispatched first
        self.assertEqual(order[:2], ["c1", "c3"])
        self.assertEqual(set(order[2:]), {"c2", "c4"})


# ── Profiler scheduling recommendation ───────────────────────────────

class TestProfilerSchedulingRecommendation(unittest.TestCase):

    def test_high_cv_recommends_work_stealing(self):
        """CV > 1.0 → profiler recommends work_stealing."""
        from ixtract.profiler import SourceProfiler, SKEW_CV_THRESHOLD
        # Use the static method directly
        from ixtract.profiler import SourceProfiler
        profiler = SourceProfiler.__new__(SourceProfiler)
        result = profiler._recommend_scheduling(1.5)
        self.assertEqual(result, "work_stealing")

    def test_low_cv_recommends_round_robin(self):
        """CV <= 1.0 → profiler recommends round_robin."""
        from ixtract.profiler import SourceProfiler
        profiler = SourceProfiler.__new__(SourceProfiler)
        result = profiler._recommend_scheduling(0.5)
        self.assertEqual(result, "round_robin")

    def test_boundary_cv_exactly_1_is_round_robin(self):
        """CV == 1.0 exactly is NOT > 1.0 → round_robin."""
        from ixtract.profiler import SourceProfiler
        profiler = SourceProfiler.__new__(SourceProfiler)
        result = profiler._recommend_scheduling(1.0)
        self.assertEqual(result, "round_robin")

    def test_just_above_threshold_is_work_stealing(self):
        from ixtract.profiler import SourceProfiler
        profiler = SourceProfiler.__new__(SourceProfiler)
        result = profiler._recommend_scheduling(1.001)
        self.assertEqual(result, "work_stealing")


# ── Plan invariants ───────────────────────────────────────────────────

class TestPlanInvariants(unittest.TestCase):

    def test_scheduling_enum_has_work_stealing(self):
        self.assertEqual(SchedulingStrategy.WORK_STEALING.value, "work_stealing")

    def test_scheduling_enum_has_greedy(self):
        self.assertEqual(SchedulingStrategy.GREEDY.value, "greedy")

    def test_scheduling_enum_has_round_robin(self):
        self.assertEqual(SchedulingStrategy.ROUND_ROBIN.value, "round_robin")

    def test_work_stealing_plan_is_immutable(self):
        """Replacing scheduling via dataclasses.replace works correctly."""
        import dataclasses
        chunks = [make_chunk("c1", 100_000)]
        plan = make_plan(chunks, SchedulingStrategy.ROUND_ROBIN)
        forced = dataclasses.replace(plan, scheduling=SchedulingStrategy.WORK_STEALING)
        self.assertEqual(forced.scheduling, SchedulingStrategy.WORK_STEALING)
        # Original unchanged
        self.assertEqual(plan.scheduling, SchedulingStrategy.ROUND_ROBIN)

    def test_plan_chunks_always_pk_order(self):
        """Chunk order in ExecutionPlan is always PK order, not dispatch order."""
        chunks = [
            make_chunk("c1", 100_000, priority=0),  # PK: 0-100K
            make_chunk("c2", 900_000, priority=1),  # PK: 100K-200K
            make_chunk("c3", 200_000, priority=2),  # PK: 200K-300K
        ]
        plan = make_plan(chunks, SchedulingStrategy.WORK_STEALING)
        # plan.chunks reflects PK order from planner
        self.assertEqual([c.chunk_id for c in plan.chunks], ["c1", "c2", "c3"])


# ── format_plan_summary: scheduling visible on every run ─────────────

class TestPlanSummarySchedulingVisibility(unittest.TestCase):

    def test_scheduling_line_present_in_summary(self):
        """Scheduling decision appears in plan summary header, not just --standard."""
        chunks = [make_chunk("c1", 300_000)]
        plan = make_plan(chunks, SchedulingStrategy.WORK_STEALING)
        prof = make_profile(cv=1.5)
        summary = format_plan_summary(
            plan, prof,
            scheduling_source="work_stealing (skew-aware scheduling via LPT, CV=1.50)"
        )
        self.assertIn("Scheduling:", summary)
        self.assertIn("work_stealing", summary)

    def test_scheduling_line_shows_greedy(self):
        chunks = [make_chunk("c1", 300_000)]
        plan = make_plan(chunks, SchedulingStrategy.GREEDY)
        prof = make_profile()
        summary = format_plan_summary(plan, prof, scheduling_source="greedy")
        self.assertIn("Scheduling:", summary)
        self.assertIn("greedy", summary)

    def test_scheduling_line_fallback_when_not_provided(self):
        """If scheduling_source not supplied, falls back to enum value."""
        chunks = [make_chunk("c1", 300_000)]
        plan = make_plan(chunks, SchedulingStrategy.ROUND_ROBIN)
        prof = make_profile()
        summary = format_plan_summary(plan, prof)
        self.assertIn("Scheduling:", summary)
        self.assertIn("round_robin", summary)

    def test_warning_shown_for_forced_work_stealing_on_uniform_table(self):
        """Forced work_stealing on uniform table (CV<=1.0) shows a no-effect warning."""
        chunks = [make_chunk("c1", 300_000)]
        plan = make_plan(chunks, SchedulingStrategy.WORK_STEALING)
        prof = make_profile(cv=0.0)  # uniform
        summary = format_plan_summary(
            plan, prof,
            scheduling_source="work_stealing (forced by user)"
        )
        self.assertIn("no effect", summary)
        self.assertIn("CV=0.00", summary)

    def test_no_warning_for_auto_work_stealing_on_skewed_table(self):
        """Auto work_stealing on genuinely skewed table has no warning."""
        chunks = [make_chunk("c1", 300_000)]
        plan = make_plan(chunks, SchedulingStrategy.WORK_STEALING)
        prof = make_profile(cv=1.5)
        summary = format_plan_summary(
            plan, prof,
            scheduling_source="work_stealing (skew-aware scheduling via LPT, CV=1.50)"
        )
        self.assertNotIn("no effect", summary)


# ── LPT effectiveness property ────────────────────────────────────────

class TestLPTEffectivenessProperty(unittest.TestCase):
    """Verify that LPT ordering reduces maximum worker load vs random order."""

    def _simulate_makespan(
        self, chunk_rows: list[int], n_workers: int, scheduling: SchedulingStrategy
    ) -> float:
        """Simulate makespan for a given chunk order and worker count.

        Assumes throughput is proportional to rows (1 row = 1 unit time).
        Returns total wall-clock time (makespan).
        """
        chunks = [make_chunk(f"c{i}", r, priority=i) for i, r in enumerate(chunk_rows)]
        plan = make_plan(chunks, scheduling, worker_count=n_workers)
        dispatch_order = simulated_dispatch_order(plan)

        # Greedy assignment: each chunk goes to the worker with least work so far
        worker_load = [0.0] * n_workers
        for chunk_id in dispatch_order:
            chunk_rows_count = next(
                c.estimated_rows for c in chunks if c.chunk_id == chunk_id
            )
            # Assign to least-loaded worker
            w = worker_load.index(min(worker_load))
            worker_load[w] += chunk_rows_count

        return max(worker_load)

    def test_lpt_makespan_le_fifo_on_skewed_workload(self):
        """LPT makespan must be ≤ FIFO makespan for skewed chunk distribution."""
        # Highly skewed: [900K, 50K, 800K, 50K] → skewed if dispatched in order
        rows = [900_000, 50_000, 800_000, 50_000]
        lpt_span  = self._simulate_makespan(rows, n_workers=2, scheduling=SchedulingStrategy.WORK_STEALING)
        fifo_span = self._simulate_makespan(rows, n_workers=2, scheduling=SchedulingStrategy.GREEDY)
        self.assertLessEqual(lpt_span, fifo_span)

    def test_lpt_equal_to_fifo_on_uniform_workload(self):
        """LPT and FIFO are identical on uniform chunks (LPT is a no-op)."""
        rows = [300_000, 300_000, 300_000, 300_000]
        lpt_span  = self._simulate_makespan(rows, n_workers=2, scheduling=SchedulingStrategy.WORK_STEALING)
        fifo_span = self._simulate_makespan(rows, n_workers=2, scheduling=SchedulingStrategy.GREEDY)
        self.assertEqual(lpt_span, fifo_span)


if __name__ == "__main__":
    unittest.main(verbosity=2)
