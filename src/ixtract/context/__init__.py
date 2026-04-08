"""Execution Context — captures environmental state at plan/run time.

The context is the "when and how" of an extraction run. It enables the
context-weighted historical planner to find runs that happened under
similar conditions and weight their throughput accordingly.

Design invariants:
  - Context is captured ONCE at plan/run start. Never updated mid-run.
  - Identical measurement logic is used at plan-time and run-time.
  - Schema is versioned. Old records remain readable after schema evolution.
  - If a dimension is unavailable (e.g. system load on Windows), weights
    are statically renormalized at capture time and stored in the context.
    Renormalization is fixed per environment, not dynamic per comparison.

Planner receives context as an input parameter — it does not observe the
world directly. "Planner does not observe the world — it receives it."

Context is stored as JSON in runs.execution_context_json. The schema_version
field allows the similarity engine to handle mixed-version history gracefully.
"""
from __future__ import annotations

import json
import math
import os
import platform
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any, Optional

# ── Schema version ────────────────────────────────────────────────────
CONTEXT_SCHEMA_VERSION: int = 1

# ── Source load thresholds (connection utilisation %) ─────────────────
SOURCE_LOAD_LOW      = 0.20   # < 20%
SOURCE_LOAD_NORMAL   = 0.60   # 20–60%
SOURCE_LOAD_HIGH     = 0.80   # 60–80%
# > 80% → critical

# ── Network quality thresholds (p50 latency ms) ───────────────────────
NETWORK_EXCELLENT = 1.0    # < 1ms
NETWORK_GOOD      = 5.0    # 1–5ms
NETWORK_DEGRADED  = 20.0   # 5–20ms
# > 20ms → poor

# ── Canonical dimension weights ───────────────────────────────────────
# Must sum to 1.0. No single dimension exceeds 0.5.
_BASE_WEIGHTS: dict[str, float] = {
    "source_load":             0.30,
    "concurrent_extractions":  0.25,
    "time_band":               0.20,
    "system_load":             0.15,
    "network_quality":         0.10,
}

# ── Time band configuration ───────────────────────────────────────────
TIME_BAND_HOURS = 4   # 4-hour bands: 0=00-04, 1=04-08, ..., 5=20-24


def _detect_system_load_available() -> bool:
    """True if os.getloadavg() is available on this platform."""
    return hasattr(os, "getloadavg") and platform.system() != "Windows"


def _compute_effective_weights(system_load_available: bool) -> dict[str, float]:
    """Compute effective dimension weights for this environment.

    If system_load is unavailable, its 0.15 weight is redistributed
    proportionally to the remaining 4 dimensions — preserving their
    relative ratios and keeping the total at 1.0.

    This redistribution is static (called once at context capture time)
    and stored inside the context object, so two contexts from different
    platforms can always be compared correctly.
    """
    if system_load_available:
        return dict(_BASE_WEIGHTS)

    # Redistribute system_load weight proportionally
    removed = _BASE_WEIGHTS["system_load"]
    remaining = {k: v for k, v in _BASE_WEIGHTS.items() if k != "system_load"}
    total_remaining = sum(remaining.values())
    redistributed = {
        k: round(v + v / total_remaining * removed, 6)
        for k, v in remaining.items()
    }
    # Force exact sum to 1.0 by adjusting the largest key (floating point safety)
    largest = max(redistributed, key=redistributed.__getitem__)
    redistributed[largest] = round(1.0 - sum(
        v for k, v in redistributed.items() if k != largest
    ), 6)
    return redistributed


# ── Source load classification ────────────────────────────────────────

def classify_source_load(
    active_connections: int,
    max_connections: int,
) -> str:
    """Classify source load from connection utilisation ratio."""
    if max_connections <= 0:
        return "normal"
    ratio = active_connections / max_connections
    if ratio < SOURCE_LOAD_LOW:
        return "low"
    if ratio < SOURCE_LOAD_NORMAL:
        return "normal"
    if ratio < SOURCE_LOAD_HIGH:
        return "high"
    return "critical"


SOURCE_LOAD_CATEGORIES = ("low", "normal", "high", "critical")


# ── Network quality classification ─────────────────────────────────────

def classify_network_quality(latency_p50_ms: float) -> str:
    """Classify network quality from p50 query latency."""
    if latency_p50_ms < NETWORK_EXCELLENT:
        return "excellent"
    if latency_p50_ms < NETWORK_GOOD:
        return "good"
    if latency_p50_ms < NETWORK_DEGRADED:
        return "degraded"
    return "poor"


NETWORK_QUALITY_CATEGORIES = ("excellent", "good", "degraded", "poor")


# ── Time band ─────────────────────────────────────────────────────────

def current_time_band(dt: Optional[datetime] = None) -> int:
    """Return the 4-hour time band index (0–5) for the given datetime.

    Band 0: 00:00–04:00
    Band 1: 04:00–08:00
    ...
    Band 5: 20:00–24:00
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.hour // TIME_BAND_HOURS


# ── System load ────────────────────────────────────────────────────────

def measure_system_load_per_core() -> Optional[float]:
    """Return 1-minute load average normalised by CPU count.

    Returns None if not available (Windows or unsupported platform).
    Value is clamped to [0.0, 2.0] to prevent spike distortion.
    """
    if not _detect_system_load_available():
        return None
    try:
        load_1min = os.getloadavg()[0]
        cpu_count = os.cpu_count() or 1
        normalised = load_1min / cpu_count
        return max(0.0, min(2.0, normalised))
    except (OSError, AttributeError):
        return None


# ── Concurrent extractions ────────────────────────────────────────────

def count_concurrent_extractions(
    store: Any,
    source: str,
    object_name: str,
    exclude_run_id: Optional[str] = None,
) -> int:
    """Count currently running extractions on this source (excluding self).

    Reads from the state store. Capped at 20 to prevent outlier distortion.
    """
    try:
        running = store.get_running_count(source, exclude_run_id=exclude_run_id)
        return min(running, 20)
    except Exception:
        return 0   # never let context capture break a run


# ── ExecutionContext dataclass ─────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionContext:
    """Complete environmental snapshot at plan/run start.

    All fields are captured once. The effective_weights dict reflects
    the actual weights used for similarity scoring in this environment
    (may differ from base weights if system_load was unavailable).

    schema_version enables the similarity engine to handle mixed-version
    history without silent corruption.
    """
    schema_version: int

    # Dimension values
    source_load: str              # "low" | "normal" | "high" | "critical"
    concurrent_extractions: int   # 0–20 (capped)
    time_band: int                # 0–5 (4-hour buckets, UTC)
    system_load_per_core: Optional[float]  # None if unavailable
    network_quality: str          # "excellent" | "good" | "degraded" | "poor"

    # Row estimate at capture time (used for row-growth guard)
    row_estimate: int

    # Effective weights for this environment (stored for symmetry)
    effective_weights: dict       # {"source_load": 0.30, ...}

    # Flags
    system_load_available: bool   # False → system_load weight redistributed

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-safe dict for state store storage."""
        return {
            "schema_version": self.schema_version,
            "source_load": self.source_load,
            "concurrent_extractions": self.concurrent_extractions,
            "time_band": self.time_band,
            "system_load_per_core": self.system_load_per_core,
            "network_quality": self.network_quality,
            "row_estimate": self.row_estimate,
            "effective_weights": self.effective_weights,
            "system_load_available": self.system_load_available,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ExecutionContext":
        return cls(
            schema_version=d.get("schema_version", 1),
            source_load=d["source_load"],
            concurrent_extractions=d["concurrent_extractions"],
            time_band=d["time_band"],
            system_load_per_core=d.get("system_load_per_core"),
            network_quality=d["network_quality"],
            row_estimate=d.get("row_estimate", 0),
            effective_weights=d["effective_weights"],
            system_load_available=d.get("system_load_available", True),
        )

    @classmethod
    def from_json(cls, s: str) -> "ExecutionContext":
        return cls.from_dict(json.loads(s))


# ── Context measurement (the single entry point) ──────────────────────

def measure_context(
    connector: Any,
    store: Any,
    object_name: str,
    row_estimate: int,
    source: str = "postgresql",
    at: Optional[datetime] = None,
    exclude_run_id: Optional[str] = None,
) -> ExecutionContext:
    """Measure the current execution context.

    This is the ONLY function that should be called to produce an
    ExecutionContext. Identical logic must be used at plan-time and
    run-time to ensure comparability.

    Args:
        connector:       Live connector (for connection counts + latency).
        store:           State store (for concurrent extraction count).
        object_name:     Table being extracted.
        row_estimate:    Current row estimate from profiler.
        source:          Source type string (e.g. "postgresql").
        at:              Timestamp to use for time band (default: now UTC).
        exclude_run_id:  Run ID to exclude from concurrent count (self).

    Returns:
        ExecutionContext with all dimensions populated.
    """
    # Source load
    try:
        connections = connector.get_connections()
        source_load = classify_source_load(
            connections.active_connections,
            connections.max_connections,
        )
    except Exception:
        source_load = "normal"

    # Network quality — use latency probe if available, else estimate
    try:
        latency = connector.estimate_latency(object_name)
        network_quality = classify_network_quality(latency.p50_ms)
    except Exception:
        network_quality = "good"

    # Time band
    time_band = current_time_band(at)

    # System load
    system_load_available = _detect_system_load_available()
    system_load_per_core = measure_system_load_per_core()

    # Concurrent extractions
    concurrent = count_concurrent_extractions(
        store, source, object_name, exclude_run_id=exclude_run_id
    )

    # Effective weights (static per environment)
    effective_weights = _compute_effective_weights(system_load_available)

    return ExecutionContext(
        schema_version=CONTEXT_SCHEMA_VERSION,
        source_load=source_load,
        concurrent_extractions=concurrent,
        time_band=time_band,
        system_load_per_core=system_load_per_core,
        network_quality=network_quality,
        row_estimate=row_estimate,
        effective_weights=effective_weights,
        system_load_available=system_load_available,
    )


# ── CLI formatting helpers ─────────────────────────────────────────────

def format_context_summary(ctx: ExecutionContext) -> str:
    """Format context for CLI display (plan --standard)."""
    load_pct = ""  # populated by caller if connections available
    lines = [
        f"  Source load:     {ctx.source_load}",
        f"  Concurrency:     {ctx.concurrent_extractions} active extraction(s)",
        f"  Time band:       band {ctx.time_band} "
        f"({ctx.time_band * 4:02d}:00–{min(ctx.time_band * 4 + 4, 24):02d}:00 UTC)",
    ]
    if ctx.system_load_per_core is not None:
        lines.append(f"  System load:     {ctx.system_load_per_core:.2f}/core")
    else:
        lines.append(f"  System load:     unavailable (weights redistributed)")
    lines.append(f"  Network quality: {ctx.network_quality}")
    return "\n".join(lines)
