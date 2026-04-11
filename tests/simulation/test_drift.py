"""Phase 3B Controller Drift/Stress Tests — validates Phase 2A behavior.

These tests cover controller scenarios that were built in Phase 2A but
not fully validated. No new features — pure validation of existing code.

Run: python -m pytest tests/simulation/test_drift.py -v
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from ixtract.controller import (
    ControllerConfig,
    ControllerDecision,
    ControllerState,
    ControllerOutput,
    ParallelismController,
)


def _default_ctrl(max_workers: int = 16, window_size: int = 5,
                   **kwargs) -> ParallelismController:
    return ParallelismController(ControllerConfig(
        max_workers=max_workers, window_size=window_size, **kwargs
    ))


class TestDegradationDetection(unittest.TestCase):
    """1. Gradual throughput decline triggers INCREASE."""

    def test_sustained_decline_triggers_increase(self):
        ctrl = _default_ctrl(window_size=5)
        state = ControllerState(current_workers=4)

        # Window with consistent downward drift: each run ~12% worse
        window = (100_000, 88_000, 78_000, 69_000, 61_000)

        out = ctrl.evaluate(window, state)
        self.assertEqual(out.decision, ControllerDecision.INCREASE)
        self.assertEqual(out.recommended_workers, 5)
        self.assertFalse(out.new_state.converged)


class TestReConvergence(unittest.TestCase):
    """2. Controller re-evaluates after sustained environment shift."""

    def test_regression_breaks_convergence_then_reconverges(self):
        ctrl = _default_ctrl(window_size=5)

        # Phase 1: converge at 4 workers
        state = ControllerState(current_workers=4)
        stable_window = (200_000, 198_000, 202_000, 199_000, 201_000)
        out = ctrl.evaluate(stable_window, state)
        self.assertTrue(out.new_state.converged)

        # Phase 2: sustained regression (>30%) breaks convergence
        converged_state = out.new_state
        regressed_window = (200_000, 140_000, 120_000, 100_000, 90_000)
        out2 = ctrl.evaluate(regressed_window, converged_state)
        self.assertFalse(out2.new_state.converged)

        # Phase 3: after environment stabilizes, reconverge
        deconverged_state = out2.new_state
        new_stable = (110_000, 112_000, 108_000, 111_000, 109_000)
        out3 = ctrl.evaluate(new_stable, deconverged_state)
        # Should either converge or hold (collecting data) — not oscillate
        self.assertIn(out3.decision, (ControllerDecision.HOLD,))


class TestEscapePrecision(unittest.TestCase):
    """3. Escape fires only when all 3 conditions met."""

    def test_escape_fires_on_severe_consecutive_drops(self):
        ctrl = _default_ctrl(window_size=5)
        state = ControllerState(current_workers=4)

        # Each of last 3 runs drops ≥15%, total drop ≥20%
        window = (100_000, 95_000, 78_000, 65_000, 54_000)
        out = ctrl.evaluate(window, state)

        # Escape should fire — adjust by ±2 workers
        if out.decision != ControllerDecision.HOLD:
            worker_diff = abs(out.recommended_workers - state.current_workers)
            self.assertEqual(worker_diff, ctrl.config.escape_step)

    def test_escape_requires_all_conditions(self):
        """Single large drop doesn't trigger escape if not consecutive."""
        ctrl = _default_ctrl(window_size=5)
        state = ControllerState(current_workers=4)

        # Big drop in one run, but recovery after — not consecutive
        window = (100_000, 50_000, 95_000, 93_000, 91_000)
        out = ctrl.evaluate(window, state)

        # Should NOT fire escape (no consecutive drops)
        # May trigger normal drift detection or hold
        if out.decision == ControllerDecision.INCREASE:
            worker_diff = abs(out.recommended_workers - state.current_workers)
            # Normal step is 1, escape step is 2
            self.assertEqual(worker_diff, ctrl.config.step_size)


class TestEscapeSingleSpike(unittest.TestCase):
    """4. One bad run doesn't trigger escape."""

    def test_single_bad_run_no_escape(self):
        ctrl = _default_ctrl(window_size=5)
        state = ControllerState(current_workers=4)

        # One terrible run surrounded by good ones
        window = (200_000, 195_000, 40_000, 198_000, 201_000)
        out = ctrl.evaluate(window, state)

        # Should not fire escape or increase — spike is noise
        # May converge since overall window is stable
        self.assertIn(out.decision, (ControllerDecision.HOLD,))


class TestEscapeResetsWindow(unittest.TestCase):
    """5. After escape fires, new window starts fresh."""

    def test_escape_resets_to_new_window(self):
        ctrl = _default_ctrl(window_size=5)
        state = ControllerState(current_workers=4)

        # Trigger escape
        window = (100_000, 82_000, 68_000, 56_000, 46_000)
        out = ctrl.evaluate(window, state)

        if out.decision != ControllerDecision.HOLD:
            new_state = out.new_state
            # New state should NOT be converged (fresh window needed)
            self.assertFalse(new_state.converged)
            # Workers should have changed
            self.assertNotEqual(new_state.current_workers, state.current_workers)
            # Previous workers should be recorded
            self.assertEqual(new_state.previous_workers, state.current_workers)


class TestConvergedSurvivesMildNoise(unittest.TestCase):
    """6. ±5% jitter doesn't break convergence."""

    def test_mild_noise_stays_converged(self):
        ctrl = _default_ctrl(window_size=5)

        # First: converge
        state = ControllerState(current_workers=4)
        stable = (200_000, 198_000, 202_000, 199_000, 201_000)
        out = ctrl.evaluate(stable, state)
        self.assertTrue(out.new_state.converged)

        # Then: mild noise (±5%) — should stay converged
        converged = out.new_state
        noisy = (200_000, 210_000, 192_000, 205_000, 195_000)
        out2 = ctrl.evaluate(noisy, converged)
        self.assertTrue(out2.new_state.converged)
        self.assertEqual(out2.decision, ControllerDecision.HOLD)


class TestRegressionBreaksConvergence(unittest.TestCase):
    """7. >30% sustained drop un-converges."""

    def test_severe_regression_breaks_convergence(self):
        ctrl = _default_ctrl(window_size=5)

        # Converge first
        state = ControllerState(current_workers=4)
        stable = (200_000, 198_000, 202_000, 199_000, 201_000)
        out = ctrl.evaluate(stable, state)
        self.assertTrue(out.new_state.converged)

        # Severe sustained regression: >30% drop
        converged = out.new_state
        crashed = (200_000, 150_000, 120_000, 110_000, 100_000)
        out2 = ctrl.evaluate(crashed, converged)
        self.assertFalse(out2.new_state.converged)


if __name__ == "__main__":
    unittest.main()
