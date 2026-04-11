"""RuntimeContext — user-supplied environmental hints for the planner.

Design invariants (FROZEN — Phase 3A design doc v1.1):
  - RuntimeContext never overrides the planner — it constrains the search space.
  - RuntimeContext never affects the controller — plan-time only, no run-time mutation.
  - RuntimeContext never feeds learning — stored for history, excluded from similarity/EWMA.
  - RuntimeContext never invents capacity assumptions — no heuristics, no guessing.
  - RuntimeContext may reduce capability but must never increase it beyond what
    the system can justify.
  - The system works identically without RuntimeContext — every field defaults to None.

Epistemic boundary:
  RuntimeContext = user-declared beliefs about the environment.
  ExecutionContext = system-measured reality.
  These must never be conflated.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ── Enums ────────────────────────────────────────────────────────────

class NetworkQuality(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    POOR = "poor"


class SourceLoad(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    CRITICAL = "critical"


# ── Constants (locked — Phase 3A design doc) ─────────────────────────

MIN_ENV_FACTOR = 0.25       # Floor for combined soft multiplier product
MIN_WORKERS = 1             # Absolute minimum workers after all stages
PRIORITY_LOW_MULT = 0.75    # Worker reduction for low priority
PRIORITY_CRIT_MULT = 1.25   # Worker increase for critical priority

# Multiplier tables
_NETWORK_MULTIPLIERS = {
    NetworkQuality.GOOD: 1.0,
    NetworkQuality.DEGRADED: 0.75,
    NetworkQuality.POOR: 0.5,
}
_SOURCE_LOAD_MULTIPLIERS = {
    SourceLoad.LOW: 1.0,
    SourceLoad.NORMAL: 1.0,
    SourceLoad.HIGH: 0.5,
}

# ── Valid field names (for strict validation) ────────────────────────

_VALID_FIELDS = frozenset({
    "network_quality", "source_load", "concurrent_extractions",
    "source_maintenance_scheduled", "priority",
    "max_source_connections", "max_memory_mb",
    "target_duration_minutes", "egress_budget_mb",
    "maintenance_window_minutes", "disk_available_gb",
})


# ── RuntimeContext ───────────────────────────────────────────────────

@dataclass(frozen=True)
class RuntimeContext:
    """User-supplied environmental hints for the planner.

    All fields are optional. None = no user opinion, use defaults.
    Classified into three categories:

    HARD CAPS: max_source_connections, max_memory_mb
    SOFT MULTIPLIERS: network_quality, source_load, concurrent_extractions,
                      source_maintenance_scheduled, priority
    ADVISORY: target_duration_minutes, egress_budget_mb,
              maintenance_window_minutes, disk_available_gb
    """
    # Hard caps
    max_source_connections: Optional[int] = None
    max_memory_mb: Optional[int] = None

    # Soft multipliers
    network_quality: Optional[str] = None
    source_load: Optional[str] = None
    concurrent_extractions: Optional[int] = None
    source_maintenance_scheduled: Optional[bool] = None
    priority: Optional[str] = None

    # Advisory signals
    target_duration_minutes: Optional[int] = None
    egress_budget_mb: Optional[float] = None
    maintenance_window_minutes: Optional[int] = None
    disk_available_gb: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict (only non-None fields)."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RuntimeContext":
        """Validated constructor. Raises ValueError on invalid input.

        This is the ONLY constructor that should be used for external input
        (CLI, context files). Raw RuntimeContext(**d) does not validate.
        """
        errors = []

        # 1. Strict schema — reject unknown fields
        unknown = set(d.keys()) - _VALID_FIELDS
        if unknown:
            errors.append(f"Unknown fields: {', '.join(sorted(unknown))}")

        # 2. Enum validation
        if "network_quality" in d and d["network_quality"] is not None:
            try:
                NetworkQuality(d["network_quality"])
            except ValueError:
                errors.append(
                    f"Invalid network_quality: '{d['network_quality']}'. "
                    f"Must be one of: good, degraded, poor"
                )
        if "source_load" in d and d["source_load"] is not None:
            try:
                SourceLoad(d["source_load"])
            except ValueError:
                errors.append(
                    f"Invalid source_load: '{d['source_load']}'. "
                    f"Must be one of: low, normal, high"
                )
        if "priority" in d and d["priority"] is not None:
            try:
                Priority(d["priority"])
            except ValueError:
                errors.append(
                    f"Invalid priority: '{d['priority']}'. "
                    f"Must be one of: low, normal, critical"
                )

        # 3. Numeric validation
        _int_fields = {
            "max_source_connections": (1, None),  # (min, max or None)
            "max_memory_mb": (1, None),
            "concurrent_extractions": (0, None),
            "target_duration_minutes": (1, None),
            "maintenance_window_minutes": (1, None),
        }
        _float_fields = {
            "egress_budget_mb": (0.0, None),
            "disk_available_gb": (0.0, None),
        }

        for field_name, (minimum, maximum) in _int_fields.items():
            val = d.get(field_name)
            if val is not None:
                if not isinstance(val, int) or isinstance(val, bool):
                    errors.append(f"{field_name} must be an integer, got {type(val).__name__}")
                elif val < minimum:
                    errors.append(f"{field_name} must be >= {minimum}, got {val}")

        for field_name, (minimum, maximum) in _float_fields.items():
            val = d.get(field_name)
            if val is not None:
                if not isinstance(val, (int, float)) or isinstance(val, bool):
                    errors.append(f"{field_name} must be a number, got {type(val).__name__}")
                elif val < minimum:
                    errors.append(f"{field_name} must be >= {minimum}, got {val}")

        # 4. Boolean validation
        if "source_maintenance_scheduled" in d and d["source_maintenance_scheduled"] is not None:
            if not isinstance(d["source_maintenance_scheduled"], bool):
                errors.append(
                    f"source_maintenance_scheduled must be a boolean, "
                    f"got {type(d['source_maintenance_scheduled']).__name__}"
                )

        if errors:
            raise ValueError(
                "RuntimeContext validation failed:\n  " + "\n  ".join(errors)
            )

        # Build with only known fields
        kwargs = {k: v for k, v in d.items() if k in _VALID_FIELDS and v is not None}
        return cls(**kwargs)

    @classmethod
    def from_json(cls, s: str) -> "RuntimeContext":
        """Parse from JSON string. Raises ValueError on invalid JSON or content."""
        try:
            d = json.loads(s)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON: {e}")
        if not isinstance(d, dict):
            raise ValueError("RuntimeContext JSON must be an object")
        return cls.from_dict(d)

    @classmethod
    def from_file(cls, path: str | Path) -> "RuntimeContext":
        """Load from a JSON file. Raises ValueError on invalid content."""
        p = Path(path)
        if not p.exists():
            raise ValueError(f"Context file not found: {p}")
        try:
            text = p.read_text(encoding="utf-8")
        except OSError as e:
            raise ValueError(f"Cannot read context file {p}: {e}")
        return cls.from_json(text)

    @classmethod
    def from_cli_args(
        cls,
        context_file: Optional[str] = None,
        *,
        network_quality: Optional[str] = None,
        source_load: Optional[str] = None,
        max_source_connections: Optional[int] = None,
        max_memory_mb: Optional[int] = None,
        concurrent_extractions: Optional[int] = None,
        source_maintenance_scheduled: Optional[bool] = None,
        priority: Optional[str] = None,
        target_duration_minutes: Optional[int] = None,
        egress_budget_mb: Optional[float] = None,
        maintenance_window_minutes: Optional[int] = None,
        disk_available_gb: Optional[float] = None,
    ) -> Optional["RuntimeContext"]:
        """Build from CLI flags. Inline flags override file values.

        Returns None if no context is provided (no file, no flags).
        """
        # Start from file if provided
        base: dict[str, Any] = {}
        if context_file:
            file_ctx = cls.from_file(context_file)
            base = file_ctx.to_dict()

        # Inline flags override file values
        overrides = {
            "network_quality": network_quality,
            "source_load": source_load,
            "max_source_connections": max_source_connections,
            "max_memory_mb": max_memory_mb,
            "concurrent_extractions": concurrent_extractions,
            "source_maintenance_scheduled": source_maintenance_scheduled,
            "priority": priority,
            "target_duration_minutes": target_duration_minutes,
            "egress_budget_mb": egress_budget_mb,
            "maintenance_window_minutes": maintenance_window_minutes,
            "disk_available_gb": disk_available_gb,
        }
        for k, v in overrides.items():
            if v is not None:
                base[k] = v

        if not base:
            return None

        return cls.from_dict(base)


# ── Worker Resolution ────────────────────────────────────────────────

@dataclass
class WorkerResolution:
    """Complete trace of worker resolution for CLI output."""
    final_workers: int
    base_workers: int
    after_hard_caps: int
    hard_cap_source: Optional[str]
    after_multipliers: int
    raw_multiplier: float
    effective_multiplier: float
    floor_applied: bool
    after_priority: int
    priority_had_effect: bool
    warnings: list[str]


def resolve_workers(
    base: int,
    runtime_ctx: Optional[RuntimeContext],
    per_worker_memory_mb: float = 0.0,
) -> WorkerResolution:
    """Resolve final worker count with RuntimeContext constraints.

    Stages (strict order — design doc Section 4.2):
      1. Base workers
      2. Hard caps (max_source_connections, max_memory_mb)
      3. Soft multipliers (independent, multiplicative, floored at 0.25)
      4. Priority bias (within remaining headroom)
      5. Final clamp (min = 1)

    Rounding: floor() everywhere, ceil() nowhere.
    """
    warnings: list[str] = []

    if runtime_ctx is None:
        return WorkerResolution(
            final_workers=max(MIN_WORKERS, base),
            base_workers=base,
            after_hard_caps=base,
            hard_cap_source=None,
            after_multipliers=base,
            raw_multiplier=1.0,
            effective_multiplier=1.0,
            floor_applied=False,
            after_priority=base,
            priority_had_effect=False,
            warnings=[],
        )

    # ── Stage 1: Base ────────────────────────────────────────────
    R = max(MIN_WORKERS, base)

    # ── Stage 2: Hard caps ───────────────────────────────────────
    cap = R
    cap_source = None

    if runtime_ctx.max_source_connections is not None:
        available = runtime_ctx.max_source_connections
        if runtime_ctx.concurrent_extractions is not None:
            available = available - runtime_ctx.concurrent_extractions
            if available <= 0:
                return WorkerResolution(
                    final_workers=0,
                    base_workers=R,
                    after_hard_caps=0,
                    hard_cap_source="max_source_connections",
                    after_multipliers=0,
                    raw_multiplier=1.0,
                    effective_multiplier=1.0,
                    floor_applied=False,
                    after_priority=0,
                    priority_had_effect=False,
                    warnings=["No available connections after constraints"],
                )
        if available < cap:
            cap = available
            cap_source = "max_source_connections"

    if runtime_ctx.max_memory_mb is not None:
        if per_worker_memory_mb <= 0:
            warnings.append("per_worker_memory_mb=0, memory cap skipped")
        else:
            mem_cap = int(runtime_ctx.max_memory_mb / per_worker_memory_mb)
            if mem_cap < cap:
                cap = mem_cap
                cap_source = "max_memory_mb"

    after_hard_caps = max(MIN_WORKERS, cap)

    # ── Stage 3: Soft multipliers ────────────────────────────────
    factors: list[float] = []

    if runtime_ctx.network_quality is not None:
        nq = NetworkQuality(runtime_ctx.network_quality)
        factors.append(_NETWORK_MULTIPLIERS[nq])

    if runtime_ctx.source_load is not None:
        sl = SourceLoad(runtime_ctx.source_load)
        factors.append(_SOURCE_LOAD_MULTIPLIERS[sl])

    if runtime_ctx.source_maintenance_scheduled is True:
        factors.append(0.5)

    # concurrent_extractions without max_source_connections: NO heuristic
    if (runtime_ctx.concurrent_extractions is not None
            and runtime_ctx.max_source_connections is None):
        warnings.append(
            "concurrent_extractions provided without max_source_connections"
            " \u2014 cannot determine connection budget, no adjustment applied"
        )

    if not factors:
        raw_multiplier = 1.0
    else:
        raw_multiplier = 1.0
        for f in factors:
            raw_multiplier *= f

    floor_applied = raw_multiplier < MIN_ENV_FACTOR
    effective_multiplier = max(MIN_ENV_FACTOR, raw_multiplier)

    after_multipliers = max(MIN_WORKERS, int(after_hard_caps * effective_multiplier))

    # ── Stage 4: Priority bias ───────────────────────────────────
    priority = runtime_ctx.priority or "normal"
    priority_had_effect = False

    if priority == "normal":
        after_priority = after_multipliers
    elif priority == "low":
        target = max(MIN_WORKERS, int(after_multipliers * PRIORITY_LOW_MULT))
        after_priority = target
        priority_had_effect = (target != after_multipliers)
    elif priority == "critical":
        # Upper bound: after_hard_caps (pre-multiplier ceiling)
        upper = after_hard_caps
        target = min(upper, int(after_multipliers * PRIORITY_CRIT_MULT))
        after_priority = target
        priority_had_effect = (target != after_multipliers)
    else:
        after_priority = after_multipliers

    if not priority_had_effect and priority != "normal":
        warnings.append(
            f"Priority '{priority}' had no effect"
            " \u2014 environment constraints dominate"
        )

    # ── Stage 5: Final clamp ─────────────────────────────────────
    final_workers = max(MIN_WORKERS, after_priority)

    return WorkerResolution(
        final_workers=final_workers,
        base_workers=R,
        after_hard_caps=after_hard_caps,
        hard_cap_source=cap_source,
        after_multipliers=after_multipliers,
        raw_multiplier=raw_multiplier,
        effective_multiplier=effective_multiplier,
        floor_applied=floor_applied,
        after_priority=after_priority,
        priority_had_effect=priority_had_effect,
        warnings=warnings,
    )


# ── Advisory System ──────────────────────────────────────────────────

class AdvisorySeverity(str, Enum):
    FAIL = "fail"
    WARN = "warn"
    PASS_ = "pass"   # trailing underscore to avoid shadowing builtin


@dataclass
class Advisory:
    severity: AdvisorySeverity
    field: str
    message: str
    detail: Optional[str] = None


def compute_advisories(
    runtime_ctx: Optional[RuntimeContext],
    estimated_duration_minutes: float,
    estimated_output_bytes: int,
    estimated_workers: int,
    throughput_per_worker: float = 0.0,
    total_rows: int = 0,
) -> list[Advisory]:
    """Evaluate advisory constraints against plan outcomes.

    Returns list sorted by severity: FAIL → WARN → PASS.
    """
    if runtime_ctx is None:
        return []

    advisories: list[Advisory] = []

    # Target duration
    if runtime_ctx.target_duration_minutes is not None:
        target = runtime_ctx.target_duration_minutes
        estimated = estimated_duration_minutes
        if estimated > target:
            detail = None
            if throughput_per_worker > 0 and total_rows > 0:
                required_workers = math.ceil(
                    total_rows / (throughput_per_worker * target * 60)
                )
                detail = (
                    f"Need {required_workers} workers to meet target, "
                    f"safe max is {estimated_workers}"
                )
                if required_workers > estimated_workers:
                    detail += " \u2014 cannot meet target within safe bounds"
            advisories.append(Advisory(
                severity=AdvisorySeverity.WARN,
                field="target_duration",
                message=f"Target {target} min \u2014 plan estimates {estimated:.0f} min",
                detail=detail,
            ))
        else:
            margin = target - estimated
            advisories.append(Advisory(
                severity=AdvisorySeverity.PASS_,
                field="target_duration",
                message=f"Target {target} min \u2014 {margin:.0f} min margin",
            ))

    # Maintenance window
    if runtime_ctx.maintenance_window_minutes is not None:
        window = runtime_ctx.maintenance_window_minutes
        if estimated_duration_minutes > window:
            advisories.append(Advisory(
                severity=AdvisorySeverity.FAIL,
                field="maintenance_window",
                message=(
                    f"Estimated {estimated_duration_minutes:.0f} min exceeds "
                    f"{window} min window"
                ),
            ))
        else:
            margin = window - estimated_duration_minutes
            advisories.append(Advisory(
                severity=AdvisorySeverity.PASS_,
                field="maintenance_window",
                message=f"{window} min \u2014 {margin:.0f} min margin",
            ))

    # Disk — two tiers
    if runtime_ctx.disk_available_gb is not None:
        output_gb = estimated_output_bytes / (1024 ** 3)
        available = runtime_ctx.disk_available_gb
        if output_gb > available:
            advisories.append(Advisory(
                severity=AdvisorySeverity.FAIL,
                field="disk",
                message=(
                    f"Estimated output {output_gb:.1f} GB exceeds "
                    f"{available} GB available \u2014 will fail"
                ),
            ))
        elif output_gb > available * 0.8:
            headroom = available - output_gb
            advisories.append(Advisory(
                severity=AdvisorySeverity.WARN,
                field="disk",
                message=(
                    f"Estimated output {output_gb:.1f} GB \u2014 "
                    f"only {headroom:.1f} GB headroom"
                ),
            ))
        else:
            advisories.append(Advisory(
                severity=AdvisorySeverity.PASS_,
                field="disk",
                message=(
                    f"{available} GB available \u2014 "
                    f"output {output_gb:.1f} GB"
                ),
            ))

    # Egress budget
    if runtime_ctx.egress_budget_mb is not None:
        output_mb = estimated_output_bytes / (1024 ** 2)
        budget = runtime_ctx.egress_budget_mb
        if output_mb > budget:
            pct = round((output_mb / budget - 1) * 100)
            advisories.append(Advisory(
                severity=AdvisorySeverity.WARN,
                field="egress_budget",
                message=(
                    f"Budget {budget:.0f} MB \u2014 estimated output "
                    f"{output_mb:.0f} MB (exceeds by {pct}%)"
                ),
            ))
        else:
            advisories.append(Advisory(
                severity=AdvisorySeverity.PASS_,
                field="egress_budget",
                message=f"Budget {budget:.0f} MB \u2014 output {output_mb:.0f} MB",
            ))

    # Sort: FAIL → WARN → PASS
    _severity_order = {
        AdvisorySeverity.FAIL: 0,
        AdvisorySeverity.WARN: 1,
        AdvisorySeverity.PASS_: 2,
    }
    advisories.sort(key=lambda a: _severity_order[a.severity])

    return advisories


# ── Verdict System ───────────────────────────────────────────────────

class VerdictStatus(str, Enum):
    SAFE = "safe"
    SAFE_WITH_WARNINGS = "safe_with_warnings"
    NOT_RECOMMENDED = "not_recommended"


@dataclass
class Verdict:
    status: VerdictStatus
    reason: Optional[str] = None

    @property
    def label(self) -> str:
        if self.status == VerdictStatus.SAFE:
            return "SAFE TO RUN"
        if self.status == VerdictStatus.SAFE_WITH_WARNINGS:
            return "SAFE TO RUN \u2014 with advisory warnings above"
        return "NOT RECOMMENDED"


def compute_verdict(
    resolution: WorkerResolution,
    advisories: list[Advisory],
) -> Verdict:
    """Compute verdict from resolution + advisories.

    Evaluation order (strict — first match wins):
      1. Structural infeasibility → NOT RECOMMENDED
      2. maintenance_window FAIL → NOT RECOMMENDED
      3. disk FAIL → NOT RECOMMENDED
      4. Any FAIL → SAFE_WITH_WARNINGS
      5. Any WARN → SAFE_WITH_WARNINGS
      6. Else → SAFE
    """
    # 1. Structural infeasibility
    if resolution.final_workers == 0:
        reason = "No available connections after constraints"
        if resolution.warnings:
            reason = resolution.warnings[0]
        return Verdict(VerdictStatus.NOT_RECOMMENDED, reason)

    # 2. Maintenance window breach
    for a in advisories:
        if a.severity == AdvisorySeverity.FAIL and a.field == "maintenance_window":
            return Verdict(
                VerdictStatus.NOT_RECOMMENDED,
                "Estimated duration exceeds maintenance window",
            )

    # 3. Disk breach
    for a in advisories:
        if a.severity == AdvisorySeverity.FAIL and a.field == "disk":
            return Verdict(
                VerdictStatus.NOT_RECOMMENDED,
                "Estimated output exceeds available disk",
            )

    # 4. Any FAIL
    if any(a.severity == AdvisorySeverity.FAIL for a in advisories):
        return Verdict(VerdictStatus.SAFE_WITH_WARNINGS)

    # 5. Any WARN
    if any(a.severity == AdvisorySeverity.WARN for a in advisories):
        return Verdict(VerdictStatus.SAFE_WITH_WARNINGS)

    # 6. Clean
    return Verdict(VerdictStatus.SAFE)


# ── CLI Formatting ───────────────────────────────────────────────────

_SEVERITY_SYMBOLS = {
    AdvisorySeverity.FAIL: "\u2717",
    AdvisorySeverity.WARN: "\u26A0",
    AdvisorySeverity.PASS_: "\u2713",
}


def format_runtime_context_table(ctx: RuntimeContext) -> str:
    """Format the Runtime Context Applied table for CLI output."""
    rows: list[tuple[str, str, str]] = []

    if ctx.max_source_connections is not None:
        rows.append(("max_source_connections", str(ctx.max_source_connections),
                      f"cap: {ctx.max_source_connections}"))
    if ctx.max_memory_mb is not None:
        rows.append(("max_memory_mb", str(ctx.max_memory_mb),
                      f"cap"))
    if ctx.network_quality is not None:
        m = _NETWORK_MULTIPLIERS[NetworkQuality(ctx.network_quality)]
        rows.append(("network_quality", ctx.network_quality, f"\u00D7{m:.2f}"))
    if ctx.source_load is not None:
        m = _SOURCE_LOAD_MULTIPLIERS[SourceLoad(ctx.source_load)]
        rows.append(("source_load", ctx.source_load, f"\u00D7{m:.2f}"))
    if ctx.source_maintenance_scheduled is True:
        rows.append(("source_maintenance_scheduled", "yes", "\u00D70.50"))
    if ctx.concurrent_extractions is not None:
        if ctx.max_source_connections is not None:
            avail = ctx.max_source_connections - ctx.concurrent_extractions
            rows.append(("concurrent_extractions", str(ctx.concurrent_extractions),
                          f"cap: {avail} avail"))
        else:
            rows.append(("concurrent_extractions", str(ctx.concurrent_extractions),
                          "no cap (no max_source_connections)"))
    if ctx.priority is not None and ctx.priority != "normal":
        rows.append(("priority", ctx.priority, "see Worker Resolution"))

    if not rows:
        return ""

    lines = [" Runtime Context Applied"]
    for name, value, effect in rows:
        lines.append(f"   {name:<36} {value:<8} {effect}")
    return "\n".join(lines)


def format_worker_resolution(res: WorkerResolution) -> str:
    """Format the Worker Resolution waterfall for CLI output."""
    lines = [" Worker Resolution"]
    lines.append(f"   Base:                             {res.base_workers}")

    if res.hard_cap_source:
        lines.append(
            f"   After hard caps:                  {res.after_hard_caps}"
            f"  ({res.hard_cap_source})"
        )

    if res.raw_multiplier != 1.0 or res.floor_applied:
        mult_label = f"\u00D7{res.effective_multiplier:.2f}"
        if res.floor_applied:
            mult_label += f", floor applied from \u00D7{res.raw_multiplier:.3f}"
        lines.append(
            f"   After environment multiplier:     {res.after_multipliers}"
            f"  ({mult_label})"
        )

    if res.after_priority != res.after_multipliers:
        lines.append(
            f"   After priority:                   {res.after_priority}"
        )

    for w in res.warnings:
        lines.append(f"   \u26A0 {w}")

    lines.append(f"   Final:                            {res.final_workers}")
    return "\n".join(lines)


def format_advisories(advisories: list[Advisory]) -> str:
    """Format advisories for CLI output. Severity-sorted single list."""
    if not advisories:
        return ""
    lines = [" Advisories"]
    for a in advisories:
        sym = _SEVERITY_SYMBOLS[a.severity]
        lines.append(f"   {sym} {a.message}")
        if a.detail:
            lines.append(f"     \u2192 {a.detail}")
    return "\n".join(lines)


def format_verdict(verdict: Verdict) -> str:
    """Format verdict for CLI output."""
    line = f" Verdict: {verdict.label}"
    if verdict.reason:
        line += f"\n   Reason: {verdict.reason}"
    return line
