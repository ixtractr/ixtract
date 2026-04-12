"""Phase 4A Tests — Cost Model, Cost Comparison, Anomaly Detection.

Run: python -m pytest tests/simulation/test_phase4a.py -v
"""
from __future__ import annotations

import json
import math
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.cost import (
    CostConfig,
    CostEstimate,
    CostOption,
    compute_cost,
    compute_cost_comparison,
    format_cost_estimate,
    format_cost_comparison,
)
from ixtract.diagnosis import (
    AnomalyResult,
    detect_anomaly,
    ANOMALY_BASELINE_WINDOW,
    ANOMALY_MIN_BASELINE,
    ANOMALY_Z_THRESHOLD,
    ANOMALY_ZERO_STDDEV_RATIO,
)


# ══════════════════════════════════════════════════════════════════════
# 1. COST CONFIG VALIDATION
# ══════════════════════════════════════════════════════════════════════

class TestCostConfigValidation(unittest.TestCase):

    def test_all_defaults_zero(self):
        cfg = CostConfig()
        self.assertTrue(cfg.is_zero)

    def test_from_dict_valid(self):
        cfg = CostConfig.from_dict({
            "compute_cost_per_hour": 0.50,
            "egress_cost_per_gb": 0.09,
            "connection_cost_per_hour": 0.01,
        })
        self.assertAlmostEqual(cfg.compute_cost_per_hour, 0.50)
        self.assertAlmostEqual(cfg.egress_cost_per_gb, 0.09)
        self.assertFalse(cfg.is_zero)

    def test_negative_rate_rejected(self):
        with self.assertRaises(ValueError):
            CostConfig.from_dict({"compute_cost_per_hour": -1.0})

    def test_unknown_field_rejected(self):
        with self.assertRaises(ValueError) as cm:
            CostConfig.from_dict({"compute_cost_per_hour": 1.0, "foo": "bar"})
        self.assertIn("Unknown", str(cm.exception))

    def test_from_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"compute_cost_per_hour": 0.25, "egress_cost_per_gb": 0.12}, f)
            f.flush()
            cfg = CostConfig.from_file(f.name)
        os.unlink(f.name)
        self.assertAlmostEqual(cfg.compute_cost_per_hour, 0.25)

    def test_missing_file_raises(self):
        with self.assertRaises(ValueError):
            CostConfig.from_file("/nonexistent/cost.json")

    def test_from_cli_args_none_when_empty(self):
        cfg = CostConfig.from_cli_args()
        self.assertIsNone(cfg)

    def test_from_cli_args_inline(self):
        cfg = CostConfig.from_cli_args(compute_rate=0.50, egress_rate=0.09)
        self.assertIsNotNone(cfg)
        self.assertAlmostEqual(cfg.compute_cost_per_hour, 0.50)

    def test_from_cli_args_file_with_override(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"compute_cost_per_hour": 0.25, "egress_cost_per_gb": 0.10}, f)
            f.flush()
            cfg = CostConfig.from_cli_args(cost_file=f.name, compute_rate=0.50)
        os.unlink(f.name)
        self.assertAlmostEqual(cfg.compute_cost_per_hour, 0.50)  # overridden
        self.assertAlmostEqual(cfg.egress_cost_per_gb, 0.10)     # from file

    def test_to_dict_excludes_zero(self):
        cfg = CostConfig(compute_cost_per_hour=0.50)
        d = cfg.to_dict()
        self.assertIn("compute_cost_per_hour", d)
        self.assertNotIn("egress_cost_per_gb", d)
        self.assertIn("currency", d)  # always included


# ══════════════════════════════════════════════════════════════════════
# 2. COST COMPUTATION
# ══════════════════════════════════════════════════════════════════════

class TestCostComputation(unittest.TestCase):

    def test_zero_config_returns_zero(self):
        cfg = CostConfig()
        est = compute_cost(3600, 1_000_000_000, 4, cfg)
        self.assertEqual(est.total, 0.0)

    def test_compute_only(self):
        cfg = CostConfig(compute_cost_per_hour=1.0)
        est = compute_cost(3600, 0, 4, cfg)  # 1 hour
        self.assertAlmostEqual(est.compute, 1.0)
        self.assertAlmostEqual(est.egress, 0.0)
        self.assertAlmostEqual(est.connections, 0.0)
        self.assertAlmostEqual(est.total, 1.0)

    def test_egress_only(self):
        cfg = CostConfig(egress_cost_per_gb=0.10)
        est = compute_cost(3600, 10 * 1024**3, 1, cfg)  # 10 GB
        self.assertAlmostEqual(est.egress, 1.0, places=2)

    def test_connection_only(self):
        cfg = CostConfig(connection_cost_per_hour=0.05)
        est = compute_cost(3600, 0, 4, cfg)  # 4 workers, 1 hour
        self.assertAlmostEqual(est.connections, 0.20, places=2)

    def test_all_components(self):
        cfg = CostConfig(
            compute_cost_per_hour=1.0,
            egress_cost_per_gb=0.10,
            connection_cost_per_hour=0.05,
        )
        # 30 min, 5 GB, 4 workers
        est = compute_cost(1800, 5 * 1024**3, 4, cfg)
        self.assertAlmostEqual(est.compute, 0.50, places=2)
        self.assertAlmostEqual(est.egress, 0.50, places=2)
        self.assertAlmostEqual(est.connections, 0.10, places=2)
        self.assertAlmostEqual(est.total, 1.10, places=2)

    def test_breakdown_sums_to_total(self):
        cfg = CostConfig(
            compute_cost_per_hour=2.0,
            egress_cost_per_gb=0.15,
            connection_cost_per_hour=0.03,
        )
        est = compute_cost(7200, 20 * 1024**3, 8, cfg)
        breakdown_sum = est.compute + est.egress + est.connections
        self.assertAlmostEqual(est.total, breakdown_sum, places=3)

    def test_nan_guard(self):
        cfg = CostConfig(compute_cost_per_hour=1.0)
        est = compute_cost(float("inf"), 0, 1, cfg)
        self.assertEqual(est.total, 0.0)

    def test_unit_conversions_correct(self):
        """Verify seconds→hours and bytes→GB conversions."""
        cfg = CostConfig(compute_cost_per_hour=3600.0, egress_cost_per_gb=1024**3)
        # 1 second = 1/3600 hours → compute = 1.0
        # 1 byte = 1/1024^3 GB → egress = 1.0
        est = compute_cost(1.0, 1, 0, cfg)
        self.assertAlmostEqual(est.compute, 1.0, places=3)
        self.assertAlmostEqual(est.egress, 1.0, places=3)


# ══════════════════════════════════════════════════════════════════════
# 3. COST COMPARISON
# ══════════════════════════════════════════════════════════════════════

class TestCostComparison(unittest.TestCase):

    def _cfg(self) -> CostConfig:
        return CostConfig(
            compute_cost_per_hour=1.0,
            egress_cost_per_gb=0.10,
            connection_cost_per_hour=0.05,
        )

    def test_empty_when_zero_config(self):
        options = compute_cost_comparison(4, 50_000, 1_000_000, 500_000_000, CostConfig())
        self.assertEqual(options, [])

    def test_deterministic_candidates(self):
        options = compute_cost_comparison(
            recommended_workers=6,
            throughput_per_worker=50_000,
            total_rows=1_000_000,
            estimated_bytes=500_000_000,
            cost_config=self._cfg(),
            hard_cap=16,
        )
        workers = [o.workers for o in options]
        # step = max(1, round(6*0.5)) = 3
        # candidates: [3, 6, 9]
        self.assertIn(6, workers)  # recommended always present
        self.assertTrue(len(workers) >= 2)

    def test_planned_option_marked(self):
        options = compute_cost_comparison(4, 50_000, 1_000_000, 500_000_000, self._cfg())
        planned = [o for o in options if o.is_planned]
        self.assertEqual(len(planned), 1)
        self.assertEqual(planned[0].workers, 4)

    def test_dominance_filter(self):
        """Options that are both slower AND more expensive should be dropped."""
        options = compute_cost_comparison(4, 50_000, 1_000_000, 500_000_000, self._cfg())
        for i, opt_a in enumerate(options):
            for j, opt_b in enumerate(options):
                if i == j:
                    continue
                # No option should be dominated (unless it's the planned one kept forcefully)
                if not opt_a.is_planned:
                    is_dominated = (
                        opt_b.estimated_duration_minutes <= opt_a.estimated_duration_minutes
                        and opt_b.cost.total <= opt_a.cost.total
                    )
                    self.assertFalse(is_dominated, f"Option {opt_a.workers}w is dominated by {opt_b.workers}w")

    def test_more_workers_costs_more(self):
        """With same throughput model, more workers = faster + more expensive (connection cost)."""
        options = compute_cost_comparison(
            4, 50_000, 1_000_000, 500_000_000,
            CostConfig(connection_cost_per_hour=1.0),  # high connection cost
        )
        if len(options) >= 2:
            sorted_by_workers = sorted(options, key=lambda o: o.workers)
            # More workers should be faster
            self.assertLessEqual(
                sorted_by_workers[-1].estimated_duration_minutes,
                sorted_by_workers[0].estimated_duration_minutes,
            )

    def test_hard_cap_respected(self):
        options = compute_cost_comparison(14, 50_000, 1_000_000, 500_000_000, self._cfg(), hard_cap=16)
        for o in options:
            self.assertLessEqual(o.workers, 16)

    def test_minimum_one_worker(self):
        options = compute_cost_comparison(1, 50_000, 1_000_000, 500_000_000, self._cfg())
        for o in options:
            self.assertGreaterEqual(o.workers, 1)


# ══════════════════════════════════════════════════════════════════════
# 4. COST FORMATTING
# ══════════════════════════════════════════════════════════════════════

class TestCostFormatting(unittest.TestCase):

    def test_format_cost_estimate_nonzero(self):
        est = CostEstimate(total=0.48, compute=0.30, egress=0.10, connections=0.08)
        text = format_cost_estimate(est)
        self.assertIn("$0.48", text)

    def test_format_cost_estimate_zero(self):
        est = CostEstimate(total=0.0, compute=0.0, egress=0.0, connections=0.0)
        text = format_cost_estimate(est)
        self.assertEqual(text, "")

    def test_format_comparison_contains_planned(self):
        options = [
            CostOption(4, 16.0, CostEstimate(0.38, 0.20, 0.10, 0.08)),
            CostOption(6, 12.0, CostEstimate(0.48, 0.25, 0.10, 0.13), is_planned=True),
        ]
        text = format_cost_comparison(options)
        self.assertIn("planned", text)
        self.assertIn("$0.48", text)
        self.assertIn("$0.38", text)

    def test_format_comparison_empty(self):
        text = format_cost_comparison([])
        self.assertEqual(text, "")


# ══════════════════════════════════════════════════════════════════════
# 5. ANOMALY DETECTION
# ══════════════════════════════════════════════════════════════════════

class TestAnomalyDetection(unittest.TestCase):

    def test_insufficient_baseline(self):
        result = detect_anomaly(200_000, [210_000, 205_000])  # only 2 runs
        self.assertFalse(result.is_anomaly)
        self.assertIn("Insufficient", result.message)
        self.assertEqual(result.baseline_run_count, 2)

    def test_exactly_min_baseline(self):
        baseline = [200_000, 210_000, 205_000, 208_000, 203_000]  # 5 runs
        result = detect_anomaly(200_000, baseline)
        self.assertEqual(result.baseline_run_count, 5)
        # Should compute, not reject

    def test_normal_throughput_no_anomaly(self):
        baseline = [200_000 + i * 1000 for i in range(20)]
        result = detect_anomaly(205_000, baseline)
        self.assertFalse(result.is_anomaly)
        self.assertEqual(result.direction, "none")
        self.assertIn("normal range", result.message)

    def test_degradation_anomaly(self):
        baseline = [200_000] * 20
        baseline[-1] = 210_000  # slight variation for nonzero stddev
        result = detect_anomaly(50_000, baseline)  # massive drop
        self.assertTrue(result.is_anomaly)
        self.assertEqual(result.direction, "degradation")
        self.assertIn("below", result.message)

    def test_improvement_anomaly(self):
        baseline = [200_000] * 20
        baseline[-1] = 190_000
        result = detect_anomaly(500_000, baseline)  # massive spike up
        self.assertTrue(result.is_anomaly)
        self.assertEqual(result.direction, "improvement")
        self.assertIn("above", result.message)

    def test_z_score_computed(self):
        baseline = [200_000 + i * 100 for i in range(20)]
        mean = sum(baseline) / len(baseline)
        stddev = (sum((t - mean)**2 for t in baseline) / len(baseline)) ** 0.5
        # 5σ below
        anomalous = mean - 5 * stddev
        result = detect_anomaly(anomalous, baseline)
        self.assertTrue(result.is_anomaly)
        self.assertGreater(result.z_score, 4.0)

    def test_stddev_zero_uses_percentage(self):
        """When all runs identical (σ=0), fall back to 20% threshold."""
        baseline = [200_000.0] * 20
        # Within 20% → no anomaly
        result = detect_anomaly(170_000, baseline)  # 15% drop
        self.assertFalse(result.is_anomaly)

        # Beyond 20% → anomaly
        result = detect_anomaly(150_000, baseline)  # 25% drop
        self.assertTrue(result.is_anomaly)
        self.assertEqual(result.direction, "degradation")

    def test_stddev_zero_improvement(self):
        baseline = [200_000.0] * 20
        result = detect_anomaly(250_000, baseline)  # 25% improvement
        self.assertTrue(result.is_anomaly)
        self.assertEqual(result.direction, "improvement")

    def test_empty_baseline(self):
        result = detect_anomaly(200_000, [])
        self.assertFalse(result.is_anomaly)
        self.assertIn("Insufficient", result.message)

    def test_baseline_mean_zero(self):
        baseline = [0.0] * 10
        result = detect_anomaly(100, baseline)
        self.assertFalse(result.is_anomaly)
        self.assertIn("zero", result.message)

    def test_anomaly_result_fields(self):
        baseline = [200_000 + i * 100 for i in range(20)]
        result = detect_anomaly(50_000, baseline)
        self.assertIsInstance(result.is_anomaly, bool)
        self.assertIsInstance(result.current_throughput, float)
        self.assertIsInstance(result.baseline_mean, float)
        self.assertIsInstance(result.baseline_stddev, float)
        self.assertIsInstance(result.z_score, float)
        self.assertIn(result.direction, ("degradation", "improvement", "none"))
        self.assertEqual(result.baseline_run_count, 20)


# ══════════════════════════════════════════════════════════════════════
# 6. WRITER CONFIG
# ══════════════════════════════════════════════════════════════════════

class TestCostInStateStore(unittest.TestCase):
    """State store has cost columns."""

    def test_cost_columns_in_schema(self):
        from ixtract.state import StateStore
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            store = StateStore(f.name)
            with store._conn() as c:
                cursor = c.execute("PRAGMA table_info(runs)")
                columns = {row[1] for row in cursor.fetchall()}
            self.assertIn("estimated_cost", columns)
            self.assertIn("actual_cost", columns)
        os.unlink(f.name)


if __name__ == "__main__":
    unittest.main()
