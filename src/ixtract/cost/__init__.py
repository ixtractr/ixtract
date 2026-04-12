"""Cost Model — extraction cost estimation and comparison.

Cost is an estimate, not a guarantee. All rates are user-declared.
No guessing, no heuristics, no cloud API calls. Backward compatible —
all rates default to zero (cost-unaware mode).

Design principles:
    - Deterministic: same inputs → same cost
    - Explainable: breakdown by component (compute, egress, connections)
    - User-declared rates only — system never assumes pricing
    - Advisory: cost comparison never changes the plan
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional


# ── CostConfig ───────────────────────────────────────────────────────

_VALID_COST_FIELDS = frozenset({
    "compute_cost_per_hour", "egress_cost_per_gb",
    "connection_cost_per_hour", "currency",
})


@dataclass(frozen=True)
class CostConfig:
    """User-declared cost rates. All default to zero (cost-unaware mode).

    Rates are per-unit:
        compute_cost_per_hour:    $/hour of extraction runtime
        egress_cost_per_gb:       $/GB transferred out of source
        connection_cost_per_hour: $/connection/hour held open
        currency:                 Label only (no conversion logic)
    """
    compute_cost_per_hour: float = 0.0
    egress_cost_per_gb: float = 0.0
    connection_cost_per_hour: float = 0.0
    currency: str = "USD"

    @property
    def is_zero(self) -> bool:
        """True if all rates are zero (cost-unaware mode)."""
        return (self.compute_cost_per_hour == 0.0
                and self.egress_cost_per_gb == 0.0
                and self.connection_cost_per_hour == 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v != 0.0 or k == "currency"}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CostConfig":
        """Validated constructor. Raises ValueError on invalid input."""
        errors = []

        unknown = set(d.keys()) - _VALID_COST_FIELDS
        if unknown:
            errors.append(f"Unknown fields: {', '.join(sorted(unknown))}")

        for field in ("compute_cost_per_hour", "egress_cost_per_gb", "connection_cost_per_hour"):
            val = d.get(field)
            if val is not None:
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    errors.append(f"{field} must be a number")
                elif val < 0:
                    errors.append(f"{field} must be >= 0, got {val}")

        if errors:
            raise ValueError("CostConfig validation failed:\n  " + "\n  ".join(errors))

        kwargs = {k: v for k, v in d.items() if k in _VALID_COST_FIELDS}
        return cls(**kwargs)

    @classmethod
    def from_file(cls, path: str | Path) -> "CostConfig":
        p = Path(path)
        if not p.exists():
            raise ValueError(f"Cost config file not found: {p}")
        return cls.from_dict(json.loads(p.read_text(encoding="utf-8")))

    @classmethod
    def from_cli_args(
        cls,
        cost_file: Optional[str] = None,
        *,
        compute_rate: Optional[float] = None,
        egress_rate: Optional[float] = None,
        connection_rate: Optional[float] = None,
    ) -> Optional["CostConfig"]:
        """Build from CLI flags. Inline flags override file values.
        Returns None if no cost config provided.
        """
        base: dict[str, Any] = {}
        if cost_file:
            file_cfg = cls.from_file(cost_file)
            base = file_cfg.to_dict()

        if compute_rate is not None:
            base["compute_cost_per_hour"] = compute_rate
        if egress_rate is not None:
            base["egress_cost_per_gb"] = egress_rate
        if connection_rate is not None:
            base["connection_cost_per_hour"] = connection_rate

        if not base or all(v == 0.0 for k, v in base.items() if k != "currency"):
            return None

        return cls.from_dict(base)


# ── CostEstimate ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class CostEstimate:
    """Estimated extraction cost with breakdown.

    Cost is an estimate, not a guarantee.
    """
    total: float
    compute: float
    egress: float
    connections: float
    currency: str = "USD"

    @property
    def breakdown(self) -> dict[str, float]:
        return {"compute": self.compute, "egress": self.egress, "connections": self.connections}


def compute_cost(
    estimated_duration_seconds: float,
    estimated_bytes: int,
    workers: int,
    cost_config: CostConfig,
) -> CostEstimate:
    """Compute extraction cost estimate from plan data and cost rates.

    Uses explicit unit conversions:
        duration_hours = seconds / 3600
        egress_gb = bytes / (1024^3)
    """
    if cost_config.is_zero:
        return CostEstimate(0.0, 0.0, 0.0, 0.0, cost_config.currency)

    duration_hours = estimated_duration_seconds / 3600.0
    egress_gb = estimated_bytes / (1024 ** 3)

    compute = duration_hours * cost_config.compute_cost_per_hour
    egress = egress_gb * cost_config.egress_cost_per_gb
    connections = workers * duration_hours * cost_config.connection_cost_per_hour

    total = compute + egress + connections

    # Guard: NaN/inf → zero with no breakdown
    if not math.isfinite(total):
        return CostEstimate(0.0, 0.0, 0.0, 0.0, cost_config.currency)

    return CostEstimate(
        total=round(total, 4),
        compute=round(compute, 4),
        egress=round(egress, 4),
        connections=round(connections, 4),
        currency=cost_config.currency,
    )


# ── Cost Comparison ──────────────────────────────────────────────────

@dataclass(frozen=True)
class CostOption:
    """One worker-count option in a cost comparison."""
    workers: int
    estimated_duration_minutes: float
    cost: CostEstimate
    is_planned: bool = False


def compute_cost_comparison(
    recommended_workers: int,
    throughput_per_worker: float,
    total_rows: int,
    estimated_bytes: int,
    cost_config: CostConfig,
    hard_cap: int = 16,
) -> list[CostOption]:
    """Compute cost at 2-3 worker options for comparison.

    Candidate selection (deterministic):
        step = max(1, round(recommended * 0.5))
        candidates = [recommended - step, recommended, recommended + step]

    Dominance filter: drop options that are both slower AND more expensive.
    """
    if cost_config.is_zero or throughput_per_worker <= 0 or total_rows <= 0:
        return []

    step = max(1, round(recommended_workers * 0.5))
    raw_candidates = sorted(set([
        max(1, recommended_workers - step),
        recommended_workers,
        min(hard_cap, recommended_workers + step),
    ]))

    options: list[CostOption] = []
    for w in raw_candidates:
        tp = throughput_per_worker * w
        duration_sec = total_rows / tp if tp > 0 else 0.0
        cost = compute_cost(duration_sec, estimated_bytes, w, cost_config)
        options.append(CostOption(
            workers=w,
            estimated_duration_minutes=round(duration_sec / 60, 1),
            cost=cost,
            is_planned=(w == recommended_workers),
        ))

    # Dominance filter: drop options both slower AND more expensive
    filtered: list[CostOption] = []
    for opt in options:
        dominated = any(
            other.estimated_duration_minutes <= opt.estimated_duration_minutes
            and other.cost.total <= opt.cost.total
            and other is not opt
            for other in options
        )
        if not dominated:
            filtered.append(opt)

    # Ensure planned option is always included
    if not any(o.is_planned for o in filtered):
        planned = next(o for o in options if o.is_planned)
        filtered.append(planned)
        filtered.sort(key=lambda o: o.workers)

    return filtered


# ── CLI Formatting ───────────────────────────────────────────────────

def format_cost_estimate(cost: CostEstimate) -> str:
    """Format cost estimate for CLI inline display."""
    if cost.total == 0.0:
        return ""
    return f"${cost.total:.2f} {cost.currency}"


def format_cost_comparison(options: list[CostOption]) -> str:
    """Format cost comparison table for CLI output."""
    if not options:
        return ""

    lines = [" Cost Comparison (estimated, rates user-declared)"]
    for opt in options:
        marker = "  \u2190 planned" if opt.is_planned else ""
        lines.append(
            f"   {opt.workers:>2} workers:  {opt.estimated_duration_minutes:>5.0f} min"
            f"   ${opt.cost.total:.2f}{marker}"
        )

    # Savings line: compare planned vs cheapest non-dominated option
    planned = next((o for o in options if o.is_planned), None)
    cheapest = min(options, key=lambda o: o.cost.total)
    if planned and cheapest and cheapest is not planned and cheapest.cost.total > 0:
        time_diff = planned.estimated_duration_minutes - cheapest.estimated_duration_minutes
        cost_diff = planned.cost.total - cheapest.cost.total
        cost_pct = (cost_diff / cheapest.cost.total) * 100
        if time_diff < 0:  # planned is faster
            lines.append(
                f"\n   Saving {abs(time_diff):.0f} min costs"
                f" ${abs(cost_diff):.2f} more ({cost_pct:+.0f}%)"
            )

    return "\n".join(lines)
