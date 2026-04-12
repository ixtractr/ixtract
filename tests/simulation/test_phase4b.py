"""Phase 4B Tests — Deterministic Replay.

Covers: canonical JSON, round-trip serialization, fingerprinting,
plan integrity validation, version checks, state store persistence.

Run: python -m pytest tests/simulation/test_phase4b.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.planner import (
    ExecutionPlan, Strategy, SchedulingStrategy, ChunkDefinition, ChunkType,
    CostEstimate, MetadataSnapshot, RetryPolicy, WriterConfig, AdaptiveRule,
    AdaptiveTrigger, AdaptiveAction,
)
from ixtract._replay import (
    canonical_json, plan_fingerprint, serialize_plan, deserialize_plan,
    validate_plan_integrity, validate_plan_version,
    PlanCorruptionError, UnsupportedPlanVersion,
    CURRENT_PLAN_VERSION, _normalize_value,
)


# ── Test fixtures ────────────────────────────────────────────────────

def _make_plan(**overrides) -> ExecutionPlan:
    """Build a minimal valid ExecutionPlan for testing."""
    defaults = dict(
        intent_hash="abc123",
        strategy=Strategy.RANGE_CHUNKING,
        chunks=(
            ChunkDefinition(
                chunk_id="chunk_001", chunk_type=ChunkType.RANGE,
                estimated_rows=500_000, estimated_bytes=140_000_000,
                range_start=1, range_end=500_001,
            ),
            ChunkDefinition(
                chunk_id="chunk_002", chunk_type=ChunkType.RANGE,
                estimated_rows=500_000, estimated_bytes=140_000_000,
                range_start=500_001, range_end=1_000_001,
            ),
        ),
        cost_estimate=CostEstimate(
            predicted_duration_seconds=120.5,
            predicted_throughput_rows_sec=200_000.123456789,
            predicted_total_rows=1_000_000,
            predicted_total_bytes=280_000_000,
        ),
        metadata_snapshot=MetadataSnapshot(
            row_estimate=1_000_000,
            size_estimate_bytes=280_000_000,
            column_count=12,
            primary_key="id",
            primary_key_type="int8",
        ),
        worker_count=4,
        worker_bounds=(1, 16),
        scheduling=SchedulingStrategy.GREEDY,
        adaptive_rules=(
            AdaptiveRule(
                rule_id="source_latency_backoff",
                trigger=AdaptiveTrigger.SOURCE_LATENCY_SPIKE,
                threshold=25.0,
                action=AdaptiveAction.INCREASE_BACKOFF,
                step_size=1.0,
                max_activations=10,
                cooldown_chunks=3,
                absolute_floor_ms=50.0,
                backoff_sleep_base=2.0,
            ),
        ),
        retry_policy=RetryPolicy(max_retries=3, backoff_base_seconds=1.0),
        writer_config=WriterConfig(
            output_format="parquet",
            output_path="./output",
            compression="snappy",
        ),
        plan_version="1.0",
    )
    defaults.update(overrides)
    return ExecutionPlan(**defaults)


# ══════════════════════════════════════════════════════════════════════
# 1. CANONICAL JSON
# ══════════════════════════════════════════════════════════════════════

class TestCanonicalJSON(unittest.TestCase):

    def test_sorted_keys(self):
        d = {"z": 1, "a": 2, "m": 3}
        result = canonical_json(d)
        self.assertEqual(result, '{"a":2,"m":3,"z":1}')

    def test_no_whitespace(self):
        d = {"key": "value"}
        result = canonical_json(d)
        self.assertNotIn(" ", result)

    def test_float_rounding(self):
        d = {"val": 0.1 + 0.2}  # 0.30000000000000004
        result = canonical_json(d)
        parsed = json.loads(result)
        self.assertAlmostEqual(parsed["val"], 0.3, places=6)

    def test_nested_float_rounding(self):
        d = {"outer": {"inner": 1.0000001}}
        result = canonical_json(d)
        parsed = json.loads(result)
        self.assertEqual(parsed["outer"]["inner"], 1.0)

    def test_list_order_preserved(self):
        d = {"items": [3, 1, 2]}
        result = canonical_json(d)
        parsed = json.loads(result)
        self.assertEqual(parsed["items"], [3, 1, 2])

    def test_deterministic(self):
        d = {"b": 2, "a": 1, "c": [3.0, {"z": 9, "y": 8}]}
        r1 = canonical_json(d)
        r2 = canonical_json(d)
        self.assertEqual(r1, r2)

    def test_nan_becomes_zero(self):
        d = {"val": float("nan")}
        result = canonical_json(d)
        parsed = json.loads(result)
        self.assertEqual(parsed["val"], 0.0)

    def test_inf_becomes_zero(self):
        d = {"val": float("inf")}
        result = canonical_json(d)
        parsed = json.loads(result)
        self.assertEqual(parsed["val"], 0.0)

    def test_none_preserved(self):
        d = {"val": None}
        result = canonical_json(d)
        self.assertIn("null", result)


class TestNormalizeValue(unittest.TestCase):

    def test_float_rounded(self):
        self.assertEqual(_normalize_value(0.1234567890), 0.123457)

    def test_int_unchanged(self):
        self.assertEqual(_normalize_value(42), 42)

    def test_string_unchanged(self):
        self.assertEqual(_normalize_value("hello"), "hello")

    def test_nested_dict(self):
        result = _normalize_value({"a": {"b": 1.1111111}})
        self.assertAlmostEqual(result["a"]["b"], 1.111111, places=6)

    def test_list_of_floats(self):
        result = _normalize_value([1.1111111, 2.2222222])
        self.assertAlmostEqual(result[0], 1.111111, places=6)
        self.assertAlmostEqual(result[1], 2.222222, places=6)


# ══════════════════════════════════════════════════════════════════════
# 2. ROUND-TRIP SERIALIZATION (NON-NEGOTIABLE)
# ══════════════════════════════════════════════════════════════════════

class TestRoundTrip(unittest.TestCase):
    """from_dict(to_dict(plan)) == plan — this MUST hold."""

    def test_basic_plan_round_trip(self):
        plan = _make_plan()
        d = plan.to_dict()
        restored = ExecutionPlan.from_dict(d)

        self.assertEqual(restored.intent_hash, plan.intent_hash)
        self.assertEqual(restored.strategy, plan.strategy)
        self.assertEqual(restored.worker_count, plan.worker_count)
        self.assertEqual(restored.scheduling, plan.scheduling)
        self.assertEqual(len(restored.chunks), len(plan.chunks))
        self.assertEqual(restored.plan_version, plan.plan_version)

    def test_chunk_boundaries_preserved(self):
        plan = _make_plan()
        restored = ExecutionPlan.from_dict(plan.to_dict())

        for orig, rest in zip(plan.chunks, restored.chunks):
            self.assertEqual(rest.chunk_id, orig.chunk_id)
            self.assertEqual(rest.range_start, orig.range_start)
            self.assertEqual(rest.range_end, orig.range_end)
            self.assertEqual(rest.estimated_rows, orig.estimated_rows)

    def test_adaptive_rules_round_trip(self):
        plan = _make_plan()
        restored = ExecutionPlan.from_dict(plan.to_dict())

        self.assertEqual(len(restored.adaptive_rules), len(plan.adaptive_rules))
        orig_rule = plan.adaptive_rules[0]
        rest_rule = restored.adaptive_rules[0]
        self.assertEqual(rest_rule.rule_id, orig_rule.rule_id)
        self.assertEqual(rest_rule.cooldown_chunks, orig_rule.cooldown_chunks)
        self.assertEqual(rest_rule.absolute_floor_ms, orig_rule.absolute_floor_ms)
        self.assertEqual(rest_rule.backoff_sleep_base, orig_rule.backoff_sleep_base)
        self.assertEqual(rest_rule.max_activations, orig_rule.max_activations)

    def test_cost_estimate_round_trip(self):
        plan = _make_plan()
        restored = ExecutionPlan.from_dict(plan.to_dict())

        self.assertAlmostEqual(
            restored.cost_estimate.predicted_duration_seconds,
            plan.cost_estimate.predicted_duration_seconds, places=1
        )
        self.assertEqual(
            restored.cost_estimate.predicted_total_rows,
            plan.cost_estimate.predicted_total_rows
        )

    def test_writer_config_round_trip(self):
        plan = _make_plan()
        restored = ExecutionPlan.from_dict(plan.to_dict())

        self.assertEqual(restored.writer_config.output_format, plan.writer_config.output_format)
        self.assertEqual(restored.writer_config.compression, plan.writer_config.compression)
        self.assertEqual(restored.writer_config.output_path, plan.writer_config.output_path)

    def test_no_adaptive_rules_round_trip(self):
        plan = _make_plan(adaptive_rules=())
        restored = ExecutionPlan.from_dict(plan.to_dict())
        self.assertEqual(len(restored.adaptive_rules), 0)

    def test_worker_bounds_round_trip(self):
        plan = _make_plan(worker_bounds=(2, 12))
        restored = ExecutionPlan.from_dict(plan.to_dict())
        self.assertEqual(restored.worker_bounds, (2, 12))


# ══════════════════════════════════════════════════════════════════════
# 3. FINGERPRINTING
# ══════════════════════════════════════════════════════════════════════

class TestFingerprint(unittest.TestCase):

    def test_same_plan_same_fingerprint(self):
        plan = _make_plan()
        d = plan.to_dict()
        fp1 = plan_fingerprint(d)
        fp2 = plan_fingerprint(d)
        self.assertEqual(fp1, fp2)

    def test_different_plan_different_fingerprint(self):
        plan_a = _make_plan(worker_count=4)
        plan_b = _make_plan(worker_count=6)
        fp_a = plan_fingerprint(plan_a.to_dict())
        fp_b = plan_fingerprint(plan_b.to_dict())
        self.assertNotEqual(fp_a, fp_b)

    def test_fingerprint_is_sha256(self):
        plan = _make_plan()
        fp = plan_fingerprint(plan.to_dict())
        self.assertEqual(len(fp), 64)  # SHA-256 hex = 64 chars

    def test_float_precision_doesnt_break_fingerprint(self):
        """Two plans with equivalent floats must have same fingerprint."""
        plan = _make_plan()
        d1 = plan.to_dict()
        d2 = plan.to_dict()
        # Simulate tiny float drift
        d2["cost_estimate"]["predicted_duration_seconds"] = 120.5000001
        fp1 = plan_fingerprint(d1)
        fp2 = plan_fingerprint(d2)
        self.assertEqual(fp1, fp2)  # normalized to same value

    def test_serialize_plan_returns_consistent_fingerprint(self):
        plan = _make_plan()
        pj1, fp1, pv1 = serialize_plan(plan)
        pj2, fp2, pv2 = serialize_plan(plan)
        self.assertEqual(fp1, fp2)
        self.assertEqual(pj1, pj2)


# ══════════════════════════════════════════════════════════════════════
# 4. PLAN INTEGRITY VALIDATION
# ══════════════════════════════════════════════════════════════════════

class TestPlanIntegrity(unittest.TestCase):

    def test_valid_plan_passes(self):
        plan = _make_plan()
        pj, fp, _ = serialize_plan(plan)
        validate_plan_integrity(pj, fp)  # should not raise

    def test_corrupted_plan_raises(self):
        plan = _make_plan()
        pj, fp, _ = serialize_plan(plan)
        # Corrupt the JSON
        corrupted = pj.replace('"worker_count":4', '"worker_count":99')
        with self.assertRaises(PlanCorruptionError):
            validate_plan_integrity(corrupted, fp)

    def test_wrong_fingerprint_raises(self):
        plan = _make_plan()
        pj, _, _ = serialize_plan(plan)
        with self.assertRaises(PlanCorruptionError):
            validate_plan_integrity(pj, "0" * 64)


# ══════════════════════════════════════════════════════════════════════
# 5. VERSION CHECKS
# ══════════════════════════════════════════════════════════════════════

class TestVersionCheck(unittest.TestCase):

    def test_current_version_passes(self):
        validate_plan_version(CURRENT_PLAN_VERSION)  # should not raise

    def test_old_version_raises(self):
        with self.assertRaises(UnsupportedPlanVersion):
            validate_plan_version("0.1")

    def test_future_version_raises(self):
        with self.assertRaises(UnsupportedPlanVersion):
            validate_plan_version("99.0")


# ══════════════════════════════════════════════════════════════════════
# 6. STATE STORE PERSISTENCE
# ══════════════════════════════════════════════════════════════════════

class TestStatePersistence(unittest.TestCase):

    def test_plan_columns_exist(self):
        from ixtract.state import StateStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = StateStore(f.name)
            with store._conn() as c:
                cols = {row[1] for row in c.execute("PRAGMA table_info(runs)").fetchall()}
            self.assertIn("plan_json", cols)
            self.assertIn("plan_fingerprint", cols)
            self.assertIn("replay_of", cols)
        os.unlink(f.name)

    def test_plan_stored_and_loaded(self):
        from ixtract.state import StateStore
        plan = _make_plan()
        pj, fp, pv = serialize_plan(plan)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = StateStore(f.name)
            store.record_run_start(
                "test-run-001", plan.plan_id, "hash123",
                "postgresql", "orders", "range_chunking", 4,
                plan_json=pj, plan_fingerprint=fp,
            )
            loaded = store.load_plan_for_replay("test-run-001")
        os.unlink(f.name)

        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["plan_json"], pj)
        self.assertEqual(loaded["plan_fingerprint"], fp)

    def test_replay_of_stored(self):
        from ixtract.state import StateStore
        plan = _make_plan()
        pj, fp, pv = serialize_plan(plan)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = StateStore(f.name)
            store.record_run_start(
                "original-001", plan.plan_id, "hash123",
                "postgresql", "orders", "range_chunking", 4,
                plan_json=pj, plan_fingerprint=fp,
            )
            store.record_run_start(
                "replay-001", plan.plan_id, "hash123",
                "postgresql", "orders", "range_chunking", 4,
                plan_json=pj, plan_fingerprint=fp,
                replay_of="original-001",
            )
            with store._conn() as c:
                row = c.execute(
                    "SELECT replay_of FROM runs WHERE run_id='replay-001'"
                ).fetchone()
        os.unlink(f.name)

        self.assertEqual(row["replay_of"], "original-001")

    def test_prefix_match_load(self):
        from ixtract.state import StateStore
        plan = _make_plan()
        pj, fp, pv = serialize_plan(plan)

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = StateStore(f.name)
            store.record_run_start(
                "rx-20260412-abc123", plan.plan_id, "hash",
                "postgresql", "orders", "range_chunking", 4,
                plan_json=pj, plan_fingerprint=fp,
            )
            loaded = store.load_plan_for_replay("rx-20260412")
        os.unlink(f.name)

        self.assertIsNotNone(loaded)

    def test_missing_run_returns_none(self):
        from ixtract.state import StateStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = StateStore(f.name)
            loaded = store.load_plan_for_replay("nonexistent")
        os.unlink(f.name)
        self.assertIsNone(loaded)


# ══════════════════════════════════════════════════════════════════════
# 7. DESERIALIZATION
# ══════════════════════════════════════════════════════════════════════

class TestDeserialization(unittest.TestCase):

    def test_deserialize_valid_json(self):
        plan = _make_plan()
        pj, _, _ = serialize_plan(plan)
        restored = deserialize_plan(pj)
        self.assertEqual(restored.worker_count, plan.worker_count)
        self.assertEqual(len(restored.chunks), len(plan.chunks))

    def test_deserialize_invalid_json_raises(self):
        with self.assertRaises(ValueError):
            deserialize_plan("not valid json")

    def test_full_cycle_serialize_deserialize(self):
        """Serialize → store → load → deserialize → execute-ready."""
        plan = _make_plan()
        pj, fp, pv = serialize_plan(plan)

        # Simulate storage round-trip
        stored_json = pj  # as if loaded from DB
        validate_plan_integrity(stored_json, fp)
        restored = deserialize_plan(stored_json)

        # Restored plan is execution-ready
        self.assertEqual(restored.intent_hash, plan.intent_hash)
        self.assertEqual(restored.worker_count, plan.worker_count)
        self.assertEqual(len(restored.chunks), 2)
        self.assertEqual(restored.chunks[0].range_start, 1)
        self.assertEqual(restored.chunks[1].range_end, 1_000_001)


# ══════════════════════════════════════════════════════════════════════
# 8. API EXPORTS
# ══════════════════════════════════════════════════════════════════════

class TestExports(unittest.TestCase):

    def test_replay_importable_from_top_level(self):
        from ixtract import execute_plan, replay
        from ixtract import PlanCorruptionError, UnsupportedPlanVersion
        self.assertTrue(callable(execute_plan))
        self.assertTrue(callable(replay))

    def test_exception_hierarchy(self):
        self.assertTrue(issubclass(PlanCorruptionError, Exception))
        self.assertTrue(issubclass(UnsupportedPlanVersion, Exception))


if __name__ == "__main__":
    unittest.main()
