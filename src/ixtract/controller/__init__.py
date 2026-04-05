"""Parallelism Controller — direction-aware bounded hill-climbing optimizer.

Tracks the DIRECTION of the last worker change and interprets throughput
relative to that direction. If last move helped → continue. If it hurt → reverse.

Guarantees:
    - Step size = ±1. Never larger.
    - Bounds always respected.
    - Deterministic: same inputs → same output.
    - Regression guard on >30% drop.
    - Stability window freezes after N consecutive HOLDs.
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
    noise_threshold: float = 0.05
    regression_threshold: float = 0.30
    stability_window: int = 3
    cold_start_fraction: float = 0.33


@dataclass(frozen=True)
class ControllerState:
    current_workers: int
    last_throughput: float
    last_worker_count: int
    direction: ControllerDecision
    consecutive_holds: int
    converged: bool

    @staticmethod
    def cold_start(config: ControllerConfig) -> "ControllerState":
        start = max(config.min_workers, int(config.max_workers * config.cold_start_fraction))
        return ControllerState(start, 0.0, start, ControllerDecision.HOLD, 0, False)


@dataclass(frozen=True)
class ControllerOutput:
    recommended_workers: int
    decision: ControllerDecision
    reasoning: str
    new_state: ControllerState


class ParallelismController:
    def __init__(self, config: ControllerConfig | None = None) -> None:
        self.config = config or ControllerConfig()

    def evaluate(self, current_throughput: float, state: ControllerState) -> ControllerOutput:
        cfg = self.config

        # Converged — hold unless large shift.
        if state.converged:
            pct = _pct(current_throughput, state.last_throughput)
            if abs(pct) > cfg.regression_threshold:
                return self._out(state.current_workers, ControllerDecision.HOLD,
                    f"Was converged, throughput shifted {pct:+.1%}. Re-evaluating.",
                    state, current_throughput, consec=0, conv=False)
            return self._out(state.current_workers, ControllerDecision.HOLD,
                "Converged. Holding.", state, current_throughput,
                consec=state.consecutive_holds, conv=True)

        # First run — establish baseline.
        if state.last_throughput <= 0:
            return self._out(state.current_workers, ControllerDecision.HOLD,
                "First run. Establishing baseline.", state, current_throughput)

        pct = _pct(current_throughput, state.last_throughput)

        # Regression guard.
        if pct < -cfg.regression_threshold:
            rev = state.last_worker_count
            return self._out(rev, ControllerDecision.DECREASE,
                f"Throughput dropped {pct:+.1%} (>{cfg.regression_threshold:.0%} guard). "
                f"Reverting {state.current_workers} → {rev}.",
                state, current_throughput)

        # Noise band → hold.
        if abs(pct) <= cfg.noise_threshold:
            nh = state.consecutive_holds + 1
            conv = nh >= cfg.stability_window
            return self._out(state.current_workers, ControllerDecision.HOLD,
                f"Change {pct:+.1%} in noise band. Hold at {state.current_workers}. "
                f"Holds: {nh}/{cfg.stability_window}." + (" Converged." if conv else ""),
                state, current_throughput, consec=nh, conv=conv)

        # ── Direction-aware hill climbing ─────────────────────────
        improved = pct > cfg.noise_threshold
        last_dir = state.direction

        if last_dir == ControllerDecision.INCREASE:
            if improved:
                return self._step(+1, state, current_throughput,
                    f"Last increase helped ({pct:+.1%}). Continuing up.")
            else:
                return self._step(-1, state, current_throughput,
                    f"Last increase hurt ({pct:+.1%}). Reversing down.")

        elif last_dir == ControllerDecision.DECREASE:
            if improved:
                return self._step(-1, state, current_throughput,
                    f"Last decrease helped ({pct:+.1%}). Continuing down.")
            else:
                return self._step(+1, state, current_throughput,
                    f"Last decrease hurt ({pct:+.1%}). Reversing up.")

        else:  # HOLD — no prior direction
            if improved:
                return self._out(state.current_workers, ControllerDecision.HOLD,
                    f"Throughput improved {pct:+.1%} without worker change. Observing.",
                    state, current_throughput)
            else:
                return self._step(+1, state, current_throughput,
                    f"Throughput degraded {pct:+.1%} at rest. Trying increase.")

    def _step(self, direction: int, state: ControllerState,
              throughput: float, reason: str) -> ControllerOutput:
        cfg = self.config
        nw = state.current_workers + (cfg.step_size * direction)
        nw = max(cfg.min_workers, min(cfg.max_workers, nw))
        if nw > state.current_workers:
            d = ControllerDecision.INCREASE
        elif nw < state.current_workers:
            d = ControllerDecision.DECREASE
        else:
            d = ControllerDecision.HOLD
            reason += " At bound, holding."
        if nw != state.current_workers:
            reason += f" Workers {state.current_workers} → {nw}."
        return self._out(nw, d, reason, state, throughput)

    def _out(self, workers: int, decision: ControllerDecision, reasoning: str,
             prev: ControllerState, throughput: float,
             consec: int = 0, conv: bool = False) -> ControllerOutput:
        ns = ControllerState(workers, throughput, prev.current_workers,
                             decision, consec, conv)
        return ControllerOutput(workers, decision, reasoning, ns)


def _pct(cur: float, prev: float) -> float:
    return (cur - prev) / prev if prev > 0 else 0.0
