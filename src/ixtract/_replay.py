"""Deterministic Replay — reproduce any historical extraction exactly.

Replay guarantees identical decision surface and execution structure.
Replay does NOT guarantee identical timing, external system state,
or physical output layout.

Core invariants:
    - ExecutionPlan is fully resolved at persistence time
    - No planner, profiler, controller, or enrichment during replay
    - Plan is the contract — if something is missing, fail, don't guess
    - plan_fingerprint validates integrity on load
    - plan_version gates compatibility

Implementation note:
    Document it as "re-instantiation of a historical decision."
    Implement it as load plan → execute.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Optional


# ── Canonical JSON ───────────────────────────────────────────────────

def _normalize_value(obj: Any) -> Any:
    """Normalize a value for canonical JSON serialization.

    Rules:
        - Floats: round to 6 decimal places (prevents fingerprint drift)
        - Dicts: handled by sort_keys in json.dumps
        - Lists/tuples: preserve order (order is truth)
        - Enums: should already be string values from to_dict()
        - None: preserved as null
    """
    if isinstance(obj, float):
        if not math.isfinite(obj):
            return 0.0  # NaN/inf → 0 (defensive)
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _normalize_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_normalize_value(item) for item in obj]
    return obj


def canonical_json(plan_dict: dict) -> str:
    """Produce canonical JSON from a plan dict.

    Properties:
        - Deterministic: same plan → identical string
        - Sorted keys at all levels
        - No whitespace
        - Floats rounded to 6 decimal places
        - No mutation of input dict

    This is the ONLY serialization function used for fingerprinting.
    """
    normalized = _normalize_value(plan_dict)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def plan_fingerprint(plan_dict: dict) -> str:
    """Compute SHA-256 fingerprint of a plan dict.

    Same plan → same hash. Any change → different hash.
    Uses canonical_json() for deterministic serialization.
    """
    canonical = canonical_json(plan_dict)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ── Plan Persistence ─────────────────────────────────────────────────

CURRENT_PLAN_VERSION = "1.0"


def serialize_plan(plan: "ExecutionPlan") -> tuple[str, str, str]:
    """Serialize an ExecutionPlan for storage.

    Returns:
        (plan_json, fingerprint, plan_version)

    The plan must be fully resolved — no defaults applied later.
    """
    plan_dict = plan.to_dict()
    pj = canonical_json(plan_dict)
    fp = hashlib.sha256(pj.encode("utf-8")).hexdigest()
    pv = plan.plan_version
    return pj, fp, pv


def deserialize_plan(plan_json: str) -> "ExecutionPlan":
    """Deserialize an ExecutionPlan from stored JSON.

    Raises ValueError if JSON is invalid.
    """
    from ixtract.planner import ExecutionPlan
    try:
        plan_dict = json.loads(plan_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid plan JSON: {e}")
    return ExecutionPlan.from_dict(plan_dict)


def validate_plan_integrity(plan_json: str, stored_fingerprint: str) -> None:
    """Validate that stored plan JSON matches its fingerprint.

    Raises PlanCorruptionError on mismatch.
    """
    plan_dict = json.loads(plan_json)
    computed = plan_fingerprint(plan_dict)
    if computed != stored_fingerprint:
        raise PlanCorruptionError(
            f"Plan fingerprint mismatch. "
            f"Stored: {stored_fingerprint[:16]}... "
            f"Computed: {computed[:16]}..."
        )


def validate_plan_version(plan_version: str) -> None:
    """Validate that plan version matches current version.

    Raises UnsupportedPlanVersion on mismatch.
    """
    if plan_version != CURRENT_PLAN_VERSION:
        raise UnsupportedPlanVersion(
            f"Plan version '{plan_version}' is not supported. "
            f"Current version: '{CURRENT_PLAN_VERSION}'. "
            f"No migration available."
        )


# ── Exceptions ───────────────────────────────────────────────────────

class PlanCorruptionError(Exception):
    """Raised when stored plan fingerprint doesn't match computed fingerprint."""
    pass


class UnsupportedPlanVersion(Exception):
    """Raised when plan version doesn't match current version."""
    pass
