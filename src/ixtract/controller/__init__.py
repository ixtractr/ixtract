"""Parallelism Controller — two-layer statistical optimizer.

Two complementary control layers:

  Normal mode (statistical):
    Evaluates a window of N runs. Acts only when ALL of:
      1. Window is full (enough data collected)
      2. Consistent directional drift (>=80% of deltas same direction)
      3. Magnitude exceeds threshold (>=10%)
    Default is HOLD. Eliminates oscillation from environmental noise.

  Escape mode (corrective):
    A rare, high-confidence fast path for obvious misconfiguration.
    Fires mid-window (before normal evaluation) when ALL of:
      1. Last 3 runs each dropped >= escape_per_run_threshold (15%)
      2. Total drop from window start >= escape_total_threshold (20%)
    Action: adjust by escape_step (2 workers), reset window.
    Returns to normal mode immediately after.

After a worker change, the controller collects a new full window at the
new count, then compares window averages to decide if the change helped.

Guarantees:
    - Normal step size = +/-1. Escape step = +/-2. Never larger.
    - Bounds always respected.
    - Deterministic: same inputs -> same output.
    - No oscillation from noise (window filters it).
    - Convergence = window full + no evidence to change.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ControllerDecision(str, Enum):
    INCREASE = "increase"
    DECREASE = "decrease"
    HOLD = "hold"


@dataclass(frozen=True)
class ControllerConfig:
    min_workers: int = 1
    max_workers: int = 16
    step_size: int = 1
    window_size: int = 5              # runs at same worker count before deciding
    magnitude_threshold: float = 0.10  # 10% drift to trigger action
    consistency_ratio: float = 0.8     # 80% of deltas in same direction
    regression_threshold: float = 0.30 # 30% sustained drop breaks convergence
    cold_start_fraction: float = 0.33  # fallback when no profiler
    # Escape mode — fast path for obvious misconfiguration
    escape_consecutive_runs: int = 3   # how many consecutive drops to require
    escape_per_run_threshold: float = 0.15  # each run must drop >= 15%
    escape_total_threshold: float = 0.20    # total drop from window start >= 20%
    escape_step: int = 2               # adjust by ±2 workers (not ±1)


@dataclass(frozen=True)
class ControllerState:
    """Persistent state between controller evaluations.

    current_workers: what the next run should use.
    previous_workers: workers before last change (0 if never changed).
    previous_avg_throughput: avg throughput at previous_workers (0 if none).
    converged: True if system has settled.
    """
    current_workers: int
    previous_workers: int = 0
    previous_avg_throughput: float = 0.0
    converged: bool = False

    # Legacy fields for state store and CLI compatibility
    last_throughput: float = 0.0
    last_worker_count: int = 0
    direction: ControllerDecision = ControllerDecision.HOLD
    consecutive_holds: int = 0

    @staticmethod
    def cold_start(config: ControllerConfig) -> "ControllerState":
        """Used only when no profiler data is available."""
        start = max(config.min_workers, int(config.max_workers * config.cold_start_fraction))
        return ControllerState(current_workers=start)

    @staticmethod
    def from_profiler(profiler_workers: int) -> "ControllerState":
        """Seed controller from profiler recommendation after first run."""
        return ControllerState(current_workers=profiler_workers)


@dataclass(frozen=True)
class ControllerOutput:
    recommended_workers: int
    decision: ControllerDecision
    reasoning: str
    new_state: ControllerState


class ParallelismController:
    """Statistical window-based parallelism optimizer.

    Usage:
        controller = ParallelismController()
        # window = throughputs from last N runs at current worker count
        output = controller.evaluate(window, state)
    """

    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()

    def evaluate(self, window: tuple[float, ...] | list[float],
                 state: ControllerState) -> ControllerOutput:
        """Evaluate throughput window and decide whether to adjust workers.

        Args:
            window: Recent throughput measurements at state.current_workers.
                    Ordered oldest-first. May be empty or partial.
            state: Current controller state.

        Returns:
            ControllerOutput with recommended_workers and reasoning.
        """
        cfg = self.config
        window = tuple(window)

        # ── Empty window (first run) ──────────────────────────────
        if len(window) == 0:
            return ControllerOutput(
                recommended_workers=state.current_workers,
                decision=ControllerDecision.HOLD,
                reasoning="First run. Collecting data.",
                new_state=state,
            )

        # ── Escape mode — universal fast path ─────────────────────
        # Fires before all other checks, including converged state.
        # Escape is an emergency override: severe consecutive degradation
        # must be acted on regardless of whether the system was settled.
        escape = self._check_escape(window, state)
        if escape is not None:
            return escape

        # ── Already converged ─────────────────────────────────────
        if state.converged:
            return self._handle_converged(window, state)

        # ── Window not full — keep collecting ─────────────────────
        if len(window) < cfg.window_size:
            return ControllerOutput(
                recommended_workers=state.current_workers,
                decision=ControllerDecision.HOLD,
                reasoning=f"Collecting data ({len(window)}/{cfg.window_size} runs).",
                new_state=ControllerState(
                    current_workers=state.current_workers,
                    previous_workers=state.previous_workers,
                    previous_avg_throughput=state.previous_avg_throughput,
                    converged=False,
                    last_throughput=window[-1],
                    last_worker_count=state.current_workers,
                    direction=ControllerDecision.HOLD,
                    consecutive_holds=len(window),
                ),
            )

        # ── Window full — evaluate ────────────────────────────────
        current_avg = sum(window) / len(window)

        # If workers were changed, compare against previous baseline
        if (state.previous_workers > 0
                and state.previous_workers != state.current_workers
                and state.previous_avg_throughput > 0):
            return self._evaluate_after_change(window, current_avg, state)

        # Workers have been stable — check for drift
        return self._evaluate_drift(window, current_avg, state)

    def _check_escape(self, window: tuple[float, ...],
                      state: ControllerState) -> ControllerOutput | None:
        """Escape mode: fast-path for obvious severe misconfiguration.

        Triggers only when ALL conditions are met:
          - At least escape_consecutive_runs entries in window
          - Last N runs each dropped >= escape_per_run_threshold
          - Total drop from window[0] to window[-1] >= escape_total_threshold

        Returns a ControllerOutput if escape fires, None otherwise.
        """
        cfg = self.config
        n = cfg.escape_consecutive_runs
        if len(window) < n:
            return None

        # Check last N consecutive per-run drops
        tail = window[-n:]
        per_run_drops = [_pct(tail[i], tail[i - 1]) for i in range(1, n)]
        if not all(d <= -cfg.escape_per_run_threshold for d in per_run_drops):
            return None

        # Check total drop from start of window to latest
        total_drop = _pct(window[-1], window[0])
        if total_drop > -cfg.escape_total_threshold:
            return None

        # All conditions met — fire escape
        new_w = min(state.current_workers + cfg.escape_step, cfg.max_workers)
        if new_w == state.current_workers:
            # Already at max — can't escape upward, document it and hold
            return ControllerOutput(
                recommended_workers=state.current_workers,
                decision=ControllerDecision.HOLD,
                reasoning=(
                    f"Escape mode triggered (consecutive drops: "
                    f"{', '.join(f'{d:+.1%}' for d in per_run_drops)}, "
                    f"total {total_drop:+.1%}) but already at max workers "
                    f"({cfg.max_workers}). Holding."
                ),
                new_state=ControllerState(
                    current_workers=state.current_workers,
                    previous_workers=state.previous_workers,
                    previous_avg_throughput=state.previous_avg_throughput,
                    converged=False,
                    last_throughput=window[-1],
                    last_worker_count=state.current_workers,
                    direction=ControllerDecision.HOLD,
                    consecutive_holds=0,
                ),
            )

        current_avg = sum(window) / len(window)
        return ControllerOutput(
            recommended_workers=new_w,
            decision=ControllerDecision.INCREASE,
            reasoning=(
                f"ESCAPE MODE: severe consecutive degradation "
                f"({', '.join(f'{d:+.1%}' for d in per_run_drops)}, "
                f"total {total_drop:+.1%} over {len(window)} runs). "
                f"Jumping {state.current_workers}→{new_w} workers (+{cfg.escape_step}). "
                f"Window reset."
            ),
            new_state=ControllerState(
                current_workers=new_w,
                previous_workers=state.current_workers,
                previous_avg_throughput=current_avg,
                converged=False,
                last_throughput=window[-1],
                last_worker_count=state.current_workers,
                direction=ControllerDecision.INCREASE,
                consecutive_holds=0,
            ),
        )

    def _handle_converged(self, window: tuple[float, ...],
                          state: ControllerState) -> ControllerOutput:
        """Handle already-converged state. Only break on severe sustained regression."""
        cfg = self.config

        if len(window) >= cfg.window_size and state.previous_avg_throughput > 0:
            recent_avg = sum(window[-cfg.window_size:]) / cfg.window_size
            shift = _pct(recent_avg, state.previous_avg_throughput)
            if shift < -cfg.regression_threshold:
                return ControllerOutput(
                    recommended_workers=state.current_workers,
                    decision=ControllerDecision.HOLD,
                    reasoning=f"Was converged, sustained regression ({shift:+.1%}). Re-evaluating.",
                    new_state=ControllerState(
                        current_workers=state.current_workers,
                        previous_workers=state.previous_workers,
                        previous_avg_throughput=state.previous_avg_throughput,
                        converged=False,
                        last_throughput=window[-1] if window else 0.0,
                        last_worker_count=state.current_workers,
                    ),
                )

        return ControllerOutput(
            recommended_workers=state.current_workers,
            decision=ControllerDecision.HOLD,
            reasoning="Converged. Holding.",
            new_state=ControllerState(
                current_workers=state.current_workers,
                previous_workers=state.previous_workers,
                previous_avg_throughput=state.previous_avg_throughput,
                converged=True,
                last_throughput=window[-1] if window else state.last_throughput,
                last_worker_count=state.current_workers,
                direction=ControllerDecision.HOLD,
                consecutive_holds=state.consecutive_holds + 1,
            ),
        )

    def _evaluate_after_change(self, window: tuple[float, ...],
                                current_avg: float,
                                state: ControllerState) -> ControllerOutput:
        """Window full after a worker change. Compare against previous baseline."""
        cfg = self.config
        prev_avg = state.previous_avg_throughput
        change = _pct(current_avg, prev_avg)

        if change < -cfg.magnitude_threshold:
            # New worker count is worse — revert
            revert_to = state.previous_workers
            return ControllerOutput(
                recommended_workers=revert_to,
                decision=(ControllerDecision.DECREASE
                          if state.current_workers > revert_to
                          else ControllerDecision.INCREASE),
                reasoning=(
                    f"Worker change {state.previous_workers}\u2192{state.current_workers} "
                    f"degraded throughput ({change:+.1%} avg over {len(window)} runs). "
                    f"Reverting to {revert_to}."
                ),
                new_state=ControllerState(
                    current_workers=revert_to,
                    previous_workers=state.current_workers,
                    previous_avg_throughput=current_avg,
                    converged=False,
                    last_throughput=window[-1],
                    last_worker_count=state.current_workers,
                ),
            )

        # New worker count is same or better — accept and converge
        verdict = "improved" if change > cfg.magnitude_threshold else "neutral"
        return ControllerOutput(
            recommended_workers=state.current_workers,
            decision=ControllerDecision.HOLD,
            reasoning=(
                f"Worker change {state.previous_workers}\u2192{state.current_workers} "
                f"{verdict} ({change:+.1%} avg over {len(window)} runs). "
                f"Converged at {state.current_workers} workers."
            ),
            new_state=ControllerState(
                current_workers=state.current_workers,
                previous_workers=state.previous_workers,
                previous_avg_throughput=current_avg,
                converged=True,
                last_throughput=window[-1],
                last_worker_count=state.current_workers,
                direction=ControllerDecision.HOLD,
                consecutive_holds=cfg.window_size,
            ),
        )

    def _evaluate_drift(self, window: tuple[float, ...],
                        current_avg: float,
                        state: ControllerState) -> ControllerOutput:
        """Window full, workers stable. Check for sustained directional drift."""
        cfg = self.config

        # Compute run-to-run deltas
        deltas = [_pct(window[i], window[i - 1]) for i in range(1, len(window))]
        n_deltas = len(deltas)

        if n_deltas == 0:
            return self._converge(window, current_avg, state, "Single-run window.")

        down_count = sum(1 for d in deltas if d < -0.01)  # 1% floor for jitter
        up_count = sum(1 for d in deltas if d > 0.01)

        consistent_down = down_count / n_deltas >= cfg.consistency_ratio
        # consistent_up not acted on — don't fix what's improving

        # Check magnitude: second half vs first half
        half = len(window) // 2
        first_half_avg = sum(window[:half]) / half
        second_half_avg = sum(window[half:]) / (len(window) - half)
        drift_pct = _pct(second_half_avg, first_half_avg)
        significant = abs(drift_pct) > cfg.magnitude_threshold

        if consistent_down and significant and drift_pct < 0:
            # Sustained degradation — try more workers
            new_w = min(state.current_workers + cfg.step_size, cfg.max_workers)
            if new_w == state.current_workers:
                return self._converge(window, current_avg, state,
                    f"Sustained degradation ({drift_pct:+.1%}) but at max workers.")

            return ControllerOutput(
                recommended_workers=new_w,
                decision=ControllerDecision.INCREASE,
                reasoning=(
                    f"Sustained degradation: {down_count}/{n_deltas} runs declining, "
                    f"drift {drift_pct:+.1%} over window. "
                    f"Trying {new_w} workers."
                ),
                new_state=ControllerState(
                    current_workers=new_w,
                    previous_workers=state.current_workers,
                    previous_avg_throughput=current_avg,
                    converged=False,
                    last_throughput=window[-1],
                    last_worker_count=state.current_workers,
                    direction=ControllerDecision.INCREASE,
                ),
            )

        # No sustained drift — converge
        return self._converge(window, current_avg, state,
            f"Window stable ({len(window)} runs, drift {drift_pct:+.1%}). "
            f"Converged at {state.current_workers} workers.")

    def _converge(self, window: tuple[float, ...], current_avg: float,
                  state: ControllerState, reasoning: str) -> ControllerOutput:
        """Mark as converged."""
        return ControllerOutput(
            recommended_workers=state.current_workers,
            decision=ControllerDecision.HOLD,
            reasoning=reasoning,
            new_state=ControllerState(
                current_workers=state.current_workers,
                previous_workers=state.previous_workers,
                previous_avg_throughput=current_avg,
                converged=True,
                last_throughput=window[-1] if window else 0.0,
                last_worker_count=state.current_workers,
                direction=ControllerDecision.HOLD,
                consecutive_holds=len(window),
            ),
        )


def _pct(cur: float, prev: float) -> float:
    """Percentage change from prev to cur."""
    return (cur - prev) / prev if prev > 0 else 0.0
