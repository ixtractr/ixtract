"""Phase 3A RuntimeContext Tests — Validation, Resolution, Advisories, Verdict.

Run: python -m pytest tests/simulation/test_runtime_context.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.context.runtime import (
    RuntimeContext,
    WorkerResolution,
    Advisory,
    AdvisorySeverity,
    Verdict,
    VerdictStatus,
    resolve_workers,
    compute_advisories,
    compute_verdict,
    format_runtime_context_table,
    format_worker_resolution,
    format_advisories,
    format_verdict,
    MIN_ENV_FACTOR,
    MIN_WORKERS,
    PRIORITY_LOW_MULT,
    PRIORITY_CRIT_MULT,
)


# ══════════════════════════════════════════════════════════════════════
# 1. BACKWARD COMPATIBILITY
# ══════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility(unittest.TestCase):
    """RuntimeContext=None → system unchanged from v0.8.0."""

    def test_no_runtime_context_returns_base(self):
        res = resolve_workers(8, None)
        self.assertEqual(res.final_workers, 8)
        self.assertEqual(res.base_workers, 8)
        self.assertFalse(res.floor_applied)
        self.assertEqual(res.warnings, [])

    def test_no_runtime_context_advisories_empty(self):
        advs = compute_advisories(None, 10.0, 1000, 4)
        self.assertEqual(advs, [])

    def test_no_runtime_context_verdict_safe(self):
        res = resolve_workers(8, None)
        verdict = compute_verdict(res, [])
        self.assertEqual(verdict.status, VerdictStatus.SAFE)


# ══════════════════════════════════════════════════════════════════════
# 2. VALIDATION
# ══════════════════════════════════════════════════════════════════════

class TestValidation(unittest.TestCase):

    def test_valid_full_context(self):
        ctx = RuntimeContext.from_dict({
            "network_quality": "poor",
            "source_load": "high",
            "max_source_connections": 6,
            "concurrent_extractions": 2,
            "priority": "critical",
            "target_duration_minutes": 30,
        })
        self.assertEqual(ctx.network_quality, "poor")
        self.assertEqual(ctx.max_source_connections, 6)

    def test_valid_partial_context(self):
        ctx = RuntimeContext.from_dict({"network_quality": "degraded"})
        self.assertEqual(ctx.network_quality, "degraded")
        self.assertIsNone(ctx.source_load)

    def test_unknown_fields_rejected(self):
        with self.assertRaises(ValueError) as cm:
            RuntimeContext.from_dict({"network_quality": "poor", "foo": "bar"})
        self.assertIn("Unknown fields", str(cm.exception))
        self.assertIn("foo", str(cm.exception))

    def test_invalid_network_quality(self):
        with self.assertRaises(ValueError) as cm:
            RuntimeContext.from_dict({"network_quality": "terrible"})
        self.assertIn("Invalid network_quality", str(cm.exception))

    def test_invalid_source_load(self):
        with self.assertRaises(ValueError) as cm:
            RuntimeContext.from_dict({"source_load": "extreme"})
        self.assertIn("Invalid source_load", str(cm.exception))

    def test_invalid_priority(self):
        with self.assertRaises(ValueError) as cm:
            RuntimeContext.from_dict({"priority": "urgent"})
        self.assertIn("Invalid priority", str(cm.exception))

    def test_negative_numeric_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeContext.from_dict({"max_source_connections": -1})

    def test_zero_max_source_connections_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeContext.from_dict({"max_source_connections": 0})

    def test_zero_max_memory_rejected(self):
        with self.assertRaises(ValueError):
            RuntimeContext.from_dict({"max_memory_mb": 0})

    def test_zero_concurrent_extractions_valid(self):
        ctx = RuntimeContext.from_dict({"concurrent_extractions": 0})
        self.assertEqual(ctx.concurrent_extractions, 0)

    def test_boolean_validation(self):
        with self.assertRaises(ValueError):
            RuntimeContext.from_dict({"source_maintenance_scheduled": "yes"})

    def test_valid_json(self):
        ctx = RuntimeContext.from_json('{"network_quality": "poor"}')
        self.assertEqual(ctx.network_quality, "poor")

    def test_invalid_json(self):
        with self.assertRaises(ValueError) as cm:
            RuntimeContext.from_json("not json")
        self.assertIn("Invalid JSON", str(cm.exception))

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"network_quality": "degraded", "priority": "low"}, f)
            f.flush()
            ctx = RuntimeContext.from_file(f.name)
        os.unlink(f.name)
        self.assertEqual(ctx.network_quality, "degraded")
        self.assertEqual(ctx.priority, "low")

    def test_missing_file(self):
        with self.assertRaises(ValueError) as cm:
            RuntimeContext.from_file("/nonexistent/path.json")
        self.assertIn("not found", str(cm.exception))

    def test_inline_flags_override_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"network_quality": "poor", "priority": "low"}, f)
            f.flush()
            ctx = RuntimeContext.from_cli_args(
                context_file=f.name,
                priority="critical",
            )
        os.unlink(f.name)
        self.assertEqual(ctx.network_quality, "poor")     # from file
        self.assertEqual(ctx.priority, "critical")          # overridden

    def test_no_flags_no_file_returns_none(self):
        ctx = RuntimeContext.from_cli_args()
        self.assertIsNone(ctx)

    def test_serialization_roundtrip(self):
        ctx = RuntimeContext.from_dict({
            "network_quality": "poor",
            "max_source_connections": 6,
        })
        d = ctx.to_dict()
        self.assertEqual(d["network_quality"], "poor")
        self.assertEqual(d["max_source_connections"], 6)
        self.assertNotIn("source_load", d)  # None fields excluded


# ══════════════════════════════════════════════════════════════════════
# 3. WORKER RESOLUTION
# ══════════════════════════════════════════════════════════════════════

class TestWorkerResolution(unittest.TestCase):

    # ── Hard caps ────────────────────────────────────────────────
    def test_max_source_connections_caps(self):
        ctx = RuntimeContext(max_source_connections=4)
        res = resolve_workers(8, ctx)
        self.assertEqual(res.after_hard_caps, 4)
        self.assertEqual(res.final_workers, 4)
        self.assertEqual(res.hard_cap_source, "max_source_connections")

    def test_max_memory_caps(self):
        ctx = RuntimeContext(max_memory_mb=500)
        # per_worker_memory_mb=200 → 500/200 = 2
        res = resolve_workers(8, ctx, per_worker_memory_mb=200.0)
        self.assertEqual(res.after_hard_caps, 2)
        self.assertEqual(res.final_workers, 2)

    def test_concurrent_with_max_source(self):
        ctx = RuntimeContext(max_source_connections=6, concurrent_extractions=2)
        res = resolve_workers(8, ctx)
        self.assertEqual(res.after_hard_caps, 4)  # 6-2=4

    def test_concurrent_exceeds_max_source(self):
        ctx = RuntimeContext(max_source_connections=3, concurrent_extractions=3)
        res = resolve_workers(8, ctx)
        self.assertEqual(res.final_workers, 0)
        self.assertIn("No available connections after constraints", res.warnings)

    def test_concurrent_without_max_source_warns(self):
        ctx = RuntimeContext(concurrent_extractions=3)
        res = resolve_workers(8, ctx)
        self.assertEqual(res.final_workers, 8)  # no adjustment
        self.assertTrue(any("cannot determine connection budget" in w for w in res.warnings))

    def test_memory_cap_with_zero_per_worker(self):
        ctx = RuntimeContext(max_memory_mb=500)
        res = resolve_workers(8, ctx, per_worker_memory_mb=0.0)
        self.assertEqual(res.final_workers, 8)  # skipped
        self.assertTrue(any("memory cap skipped" in w for w in res.warnings))

    # ── Soft multipliers ─────────────────────────────────────────
    def test_network_poor(self):
        ctx = RuntimeContext(network_quality="poor")
        res = resolve_workers(8, ctx)
        self.assertEqual(res.final_workers, int(8 * 0.5))  # 4

    def test_network_degraded(self):
        ctx = RuntimeContext(network_quality="degraded")
        res = resolve_workers(8, ctx)
        self.assertEqual(res.final_workers, int(8 * 0.75))  # 6

    def test_source_load_high(self):
        ctx = RuntimeContext(source_load="high")
        res = resolve_workers(8, ctx)
        self.assertEqual(res.final_workers, int(8 * 0.5))  # 4

    def test_source_load_normal_no_effect(self):
        ctx = RuntimeContext(source_load="normal")
        res = resolve_workers(8, ctx)
        self.assertEqual(res.final_workers, 8)

    def test_maintenance_scheduled(self):
        ctx = RuntimeContext(source_maintenance_scheduled=True)
        res = resolve_workers(8, ctx)
        self.assertEqual(res.final_workers, int(8 * 0.5))  # 4

    def test_stacked_multipliers(self):
        ctx = RuntimeContext(network_quality="poor", source_load="high")
        res = resolve_workers(8, ctx)
        # 0.5 * 0.5 = 0.25 → 8 * 0.25 = 2
        self.assertEqual(res.final_workers, 2)
        self.assertAlmostEqual(res.raw_multiplier, 0.25)

    def test_floor_activation(self):
        ctx = RuntimeContext(
            network_quality="poor",
            source_load="high",
            source_maintenance_scheduled=True,
        )
        res = resolve_workers(8, ctx)
        # 0.5 * 0.5 * 0.5 = 0.125 → floored to 0.25 → 8*0.25=2
        self.assertAlmostEqual(res.raw_multiplier, 0.125)
        self.assertAlmostEqual(res.effective_multiplier, MIN_ENV_FACTOR)
        self.assertTrue(res.floor_applied)
        self.assertEqual(res.final_workers, 2)

    # ── Priority ─────────────────────────────────────────────────
    def test_priority_low(self):
        ctx = RuntimeContext(priority="low")
        res = resolve_workers(8, ctx)
        self.assertEqual(res.final_workers, int(8 * PRIORITY_LOW_MULT))  # 6

    def test_priority_critical(self):
        ctx = RuntimeContext(priority="critical")
        res = resolve_workers(8, ctx)
        # min(8, int(8 * 1.25)) = min(8, 10) = 8 (capped by after_hard_caps)
        # Actually: after_hard_caps = 8, int(8*1.25) = 10, min(8,10)=8
        self.assertEqual(res.final_workers, 8)

    def test_priority_critical_with_cap(self):
        ctx = RuntimeContext(max_source_connections=10, priority="critical")
        res = resolve_workers(8, ctx)
        # after_hard_caps=8, after_multipliers=8, int(8*1.25)=10, min(8,10)=8
        self.assertEqual(res.final_workers, 8)

    def test_priority_critical_under_heavy_constraints(self):
        ctx = RuntimeContext(
            network_quality="poor", source_load="high", priority="critical"
        )
        res = resolve_workers(8, ctx)
        # after_hard_caps=8, 0.5*0.5=0.25, 8*0.25=2
        # critical: min(8, int(2*1.25))=min(8,2)=2
        self.assertEqual(res.final_workers, 2)
        self.assertFalse(res.priority_had_effect)
        self.assertTrue(any("Priority 'critical' had no effect" in w for w in res.warnings))

    def test_priority_normal_no_warning(self):
        ctx = RuntimeContext(priority="normal")
        res = resolve_workers(8, ctx)
        self.assertEqual(res.final_workers, 8)
        self.assertFalse(res.priority_had_effect)
        # No warning for normal priority
        self.assertFalse(any("Priority" in w for w in res.warnings))

    # ── Rounding ─────────────────────────────────────────────────
    def test_floor_rounding(self):
        """Verify floor() not round(): 2.9 → 2, not 3."""
        ctx = RuntimeContext(network_quality="degraded")
        # base=4, 4*0.75=3.0 → exactly 3
        res = resolve_workers(4, ctx)
        self.assertEqual(res.final_workers, 3)

        # base=7, 7*0.75=5.25 → floor=5
        res = resolve_workers(7, ctx)
        self.assertEqual(res.final_workers, 5)

    def test_floor_prevents_round_up(self):
        """int(2.9) = 2, not 3."""
        # Construct: after_hard_caps * multiplier = non-integer
        # base=3, network=degraded → 3*0.75=2.25 → int()=2
        ctx = RuntimeContext(network_quality="degraded")
        res = resolve_workers(3, ctx)
        self.assertEqual(res.after_multipliers, 2)
        self.assertEqual(res.final_workers, 2)

    # ── Final clamp ──────────────────────────────────────────────
    def test_final_clamp_min_1(self):
        ctx = RuntimeContext(network_quality="poor", source_load="high",
                             source_maintenance_scheduled=True)
        res = resolve_workers(2, ctx)
        # 0.125 floored to 0.25 → 2*0.25=0.5 → int()=0 → clamp to 1
        self.assertEqual(res.final_workers, 1)

    # ── Order of operations ──────────────────────────────────────
    def test_hard_caps_before_multipliers(self):
        """Caps applied first, then multipliers shape within."""
        ctx = RuntimeContext(max_source_connections=4, network_quality="poor")
        res = resolve_workers(8, ctx)
        # base=8, cap=4, 4*0.5=2
        self.assertEqual(res.after_hard_caps, 4)
        self.assertEqual(res.after_multipliers, 2)
        self.assertEqual(res.final_workers, 2)


# ══════════════════════════════════════════════════════════════════════
# 4. ADVISORIES
# ══════════════════════════════════════════════════════════════════════

class TestAdvisories(unittest.TestCase):

    def test_target_duration_miss(self):
        ctx = RuntimeContext(target_duration_minutes=5)
        advs = compute_advisories(ctx, 10.0, 1000, 2, throughput_per_worker=100.0, total_rows=60000)
        target_adv = [a for a in advs if a.field == "target_duration"]
        self.assertEqual(len(target_adv), 1)
        self.assertEqual(target_adv[0].severity, AdvisorySeverity.WARN)
        self.assertIn("plan estimates", target_adv[0].message)

    def test_target_duration_met(self):
        ctx = RuntimeContext(target_duration_minutes=30)
        advs = compute_advisories(ctx, 10.0, 1000, 2)
        target_adv = [a for a in advs if a.field == "target_duration"]
        self.assertEqual(target_adv[0].severity, AdvisorySeverity.PASS_)
        self.assertIn("margin", target_adv[0].message)

    def test_maintenance_window_breach(self):
        ctx = RuntimeContext(maintenance_window_minutes=5)
        advs = compute_advisories(ctx, 10.0, 1000, 2)
        mw = [a for a in advs if a.field == "maintenance_window"]
        self.assertEqual(mw[0].severity, AdvisorySeverity.FAIL)

    def test_maintenance_window_ok(self):
        ctx = RuntimeContext(maintenance_window_minutes=60)
        advs = compute_advisories(ctx, 10.0, 1000, 2)
        mw = [a for a in advs if a.field == "maintenance_window"]
        self.assertEqual(mw[0].severity, AdvisorySeverity.PASS_)

    def test_disk_breach_fail(self):
        ctx = RuntimeContext(disk_available_gb=0.0001)  # tiny
        advs = compute_advisories(ctx, 10.0, 500_000_000, 2)  # 500MB output
        disk = [a for a in advs if a.field == "disk"]
        self.assertEqual(disk[0].severity, AdvisorySeverity.FAIL)
        self.assertIn("will fail", disk[0].message)

    def test_disk_tight_warn(self):
        # output_gb / available > 0.8 but < 1.0
        # 0.4 GB output, 0.45 GB available → 0.4/0.45 = 0.89 > 0.8
        output_bytes = int(0.4 * 1024**3)
        ctx = RuntimeContext(disk_available_gb=0.45)
        advs = compute_advisories(ctx, 10.0, output_bytes, 2)
        disk = [a for a in advs if a.field == "disk"]
        self.assertEqual(disk[0].severity, AdvisorySeverity.WARN)

    def test_disk_comfortable_pass(self):
        ctx = RuntimeContext(disk_available_gb=100.0)
        advs = compute_advisories(ctx, 10.0, 1000, 2)
        disk = [a for a in advs if a.field == "disk"]
        self.assertEqual(disk[0].severity, AdvisorySeverity.PASS_)

    def test_egress_breach_is_warn(self):
        """Egress breach is WARN, not FAIL."""
        ctx = RuntimeContext(egress_budget_mb=100)
        advs = compute_advisories(ctx, 10.0, 200_000_000, 2)  # ~200MB
        egress = [a for a in advs if a.field == "egress_budget"]
        self.assertEqual(egress[0].severity, AdvisorySeverity.WARN)

    def test_egress_ok(self):
        ctx = RuntimeContext(egress_budget_mb=1000)
        advs = compute_advisories(ctx, 10.0, 1000, 2)
        egress = [a for a in advs if a.field == "egress_budget"]
        self.assertEqual(egress[0].severity, AdvisorySeverity.PASS_)

    def test_severity_sort_order(self):
        ctx = RuntimeContext(
            maintenance_window_minutes=5,    # FAIL
            target_duration_minutes=5,        # WARN
            disk_available_gb=100.0,          # PASS
        )
        advs = compute_advisories(ctx, 10.0, 1000, 2)
        severities = [a.severity for a in advs]
        # FAIL first, then WARN, then PASS
        fail_idx = [i for i, s in enumerate(severities) if s == AdvisorySeverity.FAIL]
        warn_idx = [i for i, s in enumerate(severities) if s == AdvisorySeverity.WARN]
        pass_idx = [i for i, s in enumerate(severities) if s == AdvisorySeverity.PASS_]
        if fail_idx and warn_idx:
            self.assertLess(max(fail_idx), min(warn_idx))
        if warn_idx and pass_idx:
            self.assertLess(max(warn_idx), min(pass_idx))

    def test_target_duration_gap_analysis(self):
        ctx = RuntimeContext(target_duration_minutes=5)
        advs = compute_advisories(
            ctx, 10.0, 1000, 2,
            throughput_per_worker=1000.0, total_rows=600_000
        )
        target_adv = [a for a in advs if a.field == "target_duration"][0]
        self.assertIsNotNone(target_adv.detail)
        self.assertIn("Need", target_adv.detail)
        self.assertIn("workers", target_adv.detail)


# ══════════════════════════════════════════════════════════════════════
# 5. VERDICT
# ══════════════════════════════════════════════════════════════════════

class TestVerdict(unittest.TestCase):

    def test_safe(self):
        res = resolve_workers(8, None)
        verdict = compute_verdict(res, [])
        self.assertEqual(verdict.status, VerdictStatus.SAFE)
        self.assertEqual(verdict.label, "SAFE TO RUN")

    def test_safe_with_warnings(self):
        res = resolve_workers(8, RuntimeContext(priority="normal"))
        advs = [Advisory(AdvisorySeverity.WARN, "egress_budget", "test")]
        verdict = compute_verdict(res, advs)
        self.assertEqual(verdict.status, VerdictStatus.SAFE_WITH_WARNINGS)

    def test_maintenance_breach_not_recommended(self):
        res = resolve_workers(8, RuntimeContext(priority="normal"))
        advs = [Advisory(AdvisorySeverity.FAIL, "maintenance_window", "exceeded")]
        verdict = compute_verdict(res, advs)
        self.assertEqual(verdict.status, VerdictStatus.NOT_RECOMMENDED)
        self.assertIn("maintenance window", verdict.reason)

    def test_disk_breach_not_recommended(self):
        res = resolve_workers(8, RuntimeContext(priority="normal"))
        advs = [Advisory(AdvisorySeverity.FAIL, "disk", "exceeded")]
        verdict = compute_verdict(res, advs)
        self.assertEqual(verdict.status, VerdictStatus.NOT_RECOMMENDED)
        self.assertIn("disk", verdict.reason)

    def test_zero_connections_not_recommended(self):
        ctx = RuntimeContext(max_source_connections=2, concurrent_extractions=2)
        res = resolve_workers(8, ctx)
        verdict = compute_verdict(res, [])
        self.assertEqual(verdict.status, VerdictStatus.NOT_RECOMMENDED)
        self.assertIn("No available connections", verdict.reason)

    def test_verdict_precedence_structural_first(self):
        """Structural infeasibility takes precedence over advisory failures."""
        ctx = RuntimeContext(max_source_connections=1, concurrent_extractions=1)
        res = resolve_workers(8, ctx)
        advs = [Advisory(AdvisorySeverity.FAIL, "maintenance_window", "exceeded")]
        verdict = compute_verdict(res, advs)
        self.assertEqual(verdict.status, VerdictStatus.NOT_RECOMMENDED)
        self.assertIn("connections", verdict.reason)

    def test_not_recommended_includes_reason(self):
        ctx = RuntimeContext(max_source_connections=1, concurrent_extractions=1)
        res = resolve_workers(8, ctx)
        verdict = compute_verdict(res, [])
        self.assertIsNotNone(verdict.reason)


# ══════════════════════════════════════════════════════════════════════
# 6. CLI FORMATTING
# ══════════════════════════════════════════════════════════════════════

class TestFormatting(unittest.TestCase):

    def test_format_resolution_includes_final(self):
        res = resolve_workers(8, RuntimeContext(network_quality="poor"))
        text = format_worker_resolution(res)
        self.assertIn("Final:", text)
        self.assertIn("4", text)

    def test_format_resolution_shows_floor(self):
        ctx = RuntimeContext(
            network_quality="poor", source_load="high",
            source_maintenance_scheduled=True,
        )
        res = resolve_workers(8, ctx)
        text = format_worker_resolution(res)
        self.assertIn("floor applied", text)

    def test_format_advisories_severity_symbols(self):
        advs = [
            Advisory(AdvisorySeverity.FAIL, "disk", "failed"),
            Advisory(AdvisorySeverity.WARN, "egress", "warned"),
            Advisory(AdvisorySeverity.PASS_, "target", "ok"),
        ]
        text = format_advisories(advs)
        self.assertIn("\u2717", text)   # ✗
        self.assertIn("\u26A0", text)   # ⚠
        self.assertIn("\u2713", text)   # ✓

    def test_format_verdict_not_recommended(self):
        verdict = Verdict(VerdictStatus.NOT_RECOMMENDED, "test reason")
        text = format_verdict(verdict)
        self.assertIn("NOT RECOMMENDED", text)
        self.assertIn("test reason", text)

    def test_format_context_table_nonempty(self):
        ctx = RuntimeContext(network_quality="poor", source_load="high")
        text = format_runtime_context_table(ctx)
        self.assertIn("network_quality", text)
        self.assertIn("source_load", text)

    def test_format_context_table_empty_for_no_fields(self):
        ctx = RuntimeContext()
        text = format_runtime_context_table(ctx)
        self.assertEqual(text, "")


# ══════════════════════════════════════════════════════════════════════
# 7. EDGE CASES
# ══════════════════════════════════════════════════════════════════════

class TestEdgeCases(unittest.TestCase):

    def test_base_zero_clamped(self):
        res = resolve_workers(0, RuntimeContext(network_quality="good"))
        self.assertEqual(res.final_workers, 1)

    def test_max_source_1_concurrent_0(self):
        ctx = RuntimeContext(max_source_connections=1, concurrent_extractions=0)
        res = resolve_workers(8, ctx)
        self.assertEqual(res.final_workers, 1)
        self.assertEqual(res.warnings, [])

    def test_max_source_1_concurrent_1(self):
        ctx = RuntimeContext(max_source_connections=1, concurrent_extractions=1)
        res = resolve_workers(8, ctx)
        self.assertEqual(res.final_workers, 0)

    def test_priority_critical_after_multipliers_1(self):
        """int(1 * 1.25) = 1, no effect, warning emitted."""
        ctx = RuntimeContext(network_quality="poor", source_load="high", priority="critical")
        res = resolve_workers(4, ctx)
        # 4 * 0.25 = 1, critical: min(4, int(1*1.25))=min(4,1)=1
        self.assertEqual(res.final_workers, 1)
        self.assertFalse(res.priority_had_effect)

    def test_identical_to_v080_without_context(self):
        """RuntimeContext=None should not change any intermediate values."""
        res = resolve_workers(5, None)
        self.assertEqual(res.base_workers, 5)
        self.assertEqual(res.after_hard_caps, 5)
        self.assertEqual(res.after_multipliers, 5)
        self.assertEqual(res.after_priority, 5)
        self.assertEqual(res.final_workers, 5)
        self.assertIsNone(res.hard_cap_source)
        self.assertEqual(res.raw_multiplier, 1.0)
        self.assertFalse(res.floor_applied)


if __name__ == "__main__":
    unittest.main()
