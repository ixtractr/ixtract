"""ixtract — Deterministic Adaptive Extraction Runtime."""
__version__ = "0.9.4"

# ── Public API (Phase 3B) ────────────────────────────────────────────
# Usage:
#   from ixtract import plan, execute, RuntimeContext, ExtractionIntent
#   result = plan(intent, runtime_context=ctx)
#   if result.is_safe:
#       execution = execute(result)

from ixtract.api import (
    plan,
    plan_with,
    execute,
    execute_plan,
    replay,
    profile,
    explain,
    PlanResult,
    ExecutionResult,
    IxtractError,
    ValidationError,
    NotRecommendedError,
    ExecutionError,
)
from ixtract.context.runtime import RuntimeContext
from ixtract.intent import ExtractionIntent
from ixtract.cost import CostConfig, CostEstimate
from ixtract._replay import PlanCorruptionError, UnsupportedPlanVersion
