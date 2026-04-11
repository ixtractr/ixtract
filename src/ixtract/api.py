"""ixtract Python API — structured, non-interactive interface.

Mental model:
    Core (planner / engine)
            ↑
       Python API layer     ← this module
            ↑
            CLI             ← separate consumer (not refactored yet)

Contracts (locked):
    - PlanResult is immutable after creation. execute() never modifies it.
    - ExecutionPlan is self-sufficient for execution.
    - No hidden persistent state. state_db path always explicit.
    - Strictly non-interactive: no prompts, no stdout, no exit calls.
    - execute() is not idempotent. Caller manages output.
    - v0.x: evolving API. v1.0: signatures frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ixtract.intent import ExtractionIntent, SourceType
from ixtract.planner import ExecutionPlan
from ixtract.context.runtime import (
    RuntimeContext,
    WorkerResolution,
    Advisory,
    Verdict,
    VerdictStatus,
)
from ixtract.context.estimator import ThroughputEstimate
from ixtract.profiler import SourceProfile


# ── Exceptions ───────────────────────────────────────────────────────

class IxtractError(Exception):
    """Base exception for all ixtract errors."""
    pass


class ValidationError(IxtractError):
    """Raised when input validation fails."""
    pass


class NotRecommendedError(IxtractError):
    """Raised when execute() is called on a NOT RECOMMENDED plan without force=True."""
    def __init__(self, reason: str, verdict: Verdict):
        self.reason = reason
        self.verdict = verdict
        super().__init__(f"Plan not recommended: {reason}")


class ExecutionError(IxtractError):
    """Raised when extraction execution fails."""
    pass


# ── Data Types ───────────────────────────────────────────────────────

@dataclass(frozen=True)
class PlanResult:
    """Complete planning result. Immutable after creation.

    Contains everything needed for execution and display.
    execute() reads from this — no recomputation, no external dependencies.
    """
    execution_plan: ExecutionPlan
    intent: ExtractionIntent
    profile: SourceProfile
    worker_source: str
    throughput_estimate: ThroughputEstimate
    worker_resolution: Optional[WorkerResolution]  # None if no RuntimeContext
    advisories: list[Advisory]
    verdict: Verdict

    @property
    def is_safe(self) -> bool:
        """True if verdict is SAFE or SAFE_WITH_WARNINGS."""
        return self.verdict.status in (VerdictStatus.SAFE, VerdictStatus.SAFE_WITH_WARNINGS)

    @property
    def is_not_recommended(self) -> bool:
        """True if verdict is NOT_RECOMMENDED."""
        return self.verdict.status == VerdictStatus.NOT_RECOMMENDED


@dataclass(frozen=True)
class ExecutionResult:
    """Result of a complete extraction run."""
    run_id: str
    duration_seconds: float
    rows_extracted: int
    bytes_written: int
    effective_workers: float
    manifest_path: Optional[str]
    status: str  # "success" | "failed"


# ── Connector factory ────────────────────────────────────────────────

def _create_connector(intent: ExtractionIntent):
    """Create and connect a source connector based on intent source_type."""
    if intent.source_type == SourceType.POSTGRESQL:
        from ixtract.connectors.postgresql import PostgreSQLConnector
        connector = PostgreSQLConnector()
        connector.connect(intent.source_config)
        return connector
    elif intent.source_type == SourceType.MYSQL:
        from ixtract.connectors.mysql import MySQLConnector
        connector = MySQLConnector()
        connector.connect(intent.source_config)
        return connector
    elif intent.source_type == SourceType.SQLSERVER:
        from ixtract.connectors.sqlserver import SQLServerConnector
        connector = SQLServerConnector()
        connector.connect(intent.source_config)
        return connector
    else:
        raise ValidationError(f"Unsupported source type: {intent.source_type}")


# ── Public API ───────────────────────────────────────────────────────

def profile(intent: ExtractionIntent) -> SourceProfile:
    """Profile a source object.

    Creates a connector, profiles, and closes. No persistent state.

    Args:
        intent: Extraction intent (only source fields are used).

    Returns:
        SourceProfile with row estimates, PK info, latency, etc.

    Raises:
        ValidationError: If source type is unsupported.
        IxtractError: If profiling fails.
    """
    connector = _create_connector(intent)
    try:
        from ixtract.profiler import SourceProfiler
        profiler = SourceProfiler(connector)
        return profiler.profile(intent.object_name)
    except Exception as e:
        if isinstance(e, IxtractError):
            raise
        raise IxtractError(f"Profiling failed: {e}") from e
    finally:
        connector.close()


def plan(
    intent: ExtractionIntent,
    runtime_context: Optional[RuntimeContext] = None,
    state_db: str = "ixtract_state.db",
) -> PlanResult:
    """Plan an extraction. High-level API — manages connector lifecycle.

    Creates connector and state store, profiles, plans, and closes.

    Args:
        intent: What to extract.
        runtime_context: Optional environmental hints (caps, multipliers, advisories).
        state_db: Path to SQLite state store. Explicit, never hidden.

    Returns:
        PlanResult with plan, verdict, advisories, and all explainability data.

    Raises:
        ValidationError: If intent or runtime_context is invalid.
        IxtractError: If planning fails.
    """
    connector = _create_connector(intent)
    try:
        from ixtract.state import StateStore
        store = StateStore(state_db)
        return plan_with(intent, connector, store, runtime_context)
    except Exception as e:
        if isinstance(e, IxtractError):
            raise
        raise IxtractError(f"Planning failed: {e}") from e
    finally:
        connector.close()


def plan_with(
    intent: ExtractionIntent,
    connector: Any,
    store: Any,
    runtime_context: Optional[RuntimeContext] = None,
) -> PlanResult:
    """Plan an extraction. Low-level API — caller manages connector/store lifecycle.

    Args:
        intent: What to extract.
        connector: Pre-connected source connector.
        store: Pre-created state store.
        runtime_context: Optional environmental hints.

    Returns:
        PlanResult with plan, verdict, advisories, and all explainability data.

    Raises:
        ValidationError: If intent or runtime_context is invalid.
        IxtractError: If planning fails.
    """
    from ixtract.profiler import SourceProfiler
    from ixtract.planner.planner import plan_extraction, compute_runtime_analysis
    from ixtract.context.runtime import (
        resolve_workers as _rt_resolve,
        compute_advisories as _rt_advisories,
        compute_verdict as _rt_verdict,
    )

    try:
        # Profile
        profiler = SourceProfiler(connector)
        prof = profiler.profile(intent.object_name)

        # Controller state
        ctrl_state = store.get_controller_state(
            intent.source_type.value, intent.object_name
        )

        # Plan
        exec_plan, worker_source, tp_estimate = plan_extraction(
            intent, prof, store, ctrl_state, runtime_context=runtime_context,
        )

        # RuntimeContext analysis
        worker_resolution = None
        advisories: list[Advisory] = []
        verdict = Verdict(VerdictStatus.SAFE)

        if runtime_context is not None:
            analysis = compute_runtime_analysis(
                prof, exec_plan, ctrl_state, intent, runtime_context,
            )
            if analysis:
                worker_resolution = analysis["resolution"]
                advisories = analysis["advisories"]
                verdict = analysis["verdict"]

        return PlanResult(
            execution_plan=exec_plan,
            intent=intent,
            profile=prof,
            worker_source=worker_source,
            throughput_estimate=tp_estimate,
            worker_resolution=worker_resolution,
            advisories=advisories,
            verdict=verdict,
        )
    except Exception as e:
        if isinstance(e, IxtractError):
            raise
        raise IxtractError(f"Planning failed: {e}") from e


def execute(
    plan_result: PlanResult,
    force: bool = False,
    state_db: str = "ixtract_state.db",
) -> ExecutionResult:
    """Execute a planned extraction. High-level API — manages connector lifecycle.

    Respects the verdict: raises NotRecommendedError if plan is NOT RECOMMENDED
    and force is False.

    Args:
        plan_result: Result from plan() or plan_with(). Immutable — not modified.
        force: If True, execute even if NOT RECOMMENDED. Default False.
        state_db: Path to SQLite state store. Explicit, never hidden.

    Returns:
        ExecutionResult with run_id, rows, bytes, duration, manifest path.

    Raises:
        NotRecommendedError: If verdict is NOT RECOMMENDED and force is False.
        ExecutionError: If extraction fails.
        IxtractError: For other failures.
    """
    # Verdict check — never prompt, raise structured exception
    if plan_result.is_not_recommended and not force:
        raise NotRecommendedError(
            reason=plan_result.verdict.reason or "Plan not recommended",
            verdict=plan_result.verdict,
        )

    intent = plan_result.intent
    exec_plan = plan_result.execution_plan
    connector = _create_connector(intent)

    try:
        from ixtract.state import StateStore
        from ixtract.engine import ExecutionEngine

        store = StateStore(state_db)
        engine = ExecutionEngine(connector)

        # Execute
        result = engine.execute(exec_plan, intent.object_name)

        # Record in state store
        runtime_ctx = None
        if plan_result.worker_resolution is not None:
            # RuntimeContext was used — reconstruct for storage
            # PlanResult has the resolution, but we need the original context JSON
            # For now, store None — the CLI path handles this separately
            pass

        store.record_run_start(
            result.run_id, exec_plan.plan_id, intent.intent_hash(),
            intent.source_type.value, intent.object_name,
            exec_plan.strategy.value, exec_plan.worker_count,
        )
        store.record_run_end(
            result.run_id, result.status.lower(),
            result.total_rows, result.total_bytes,
            result.avg_throughput, result.duration_seconds,
            effective_workers=result.effective_workers,
            confidence_flag=result.confidence_flag,
        )

        # Record per-chunk results
        for cr in result.chunk_results:
            store.record_chunk(
                result.run_id, cr.chunk_id, cr.worker_id,
                cr.rows, cr.bytes_written, cr.status, cr.duration_seconds,
                query_ms=cr.query_ms, write_ms=cr.write_ms,
                output_path=cr.output_path, error=cr.error,
            )

        # Determine manifest path
        manifest_path = None
        output_path = intent.target_config.get("output_path", "./output")
        import os
        manifest_candidate = os.path.join(output_path, "_manifest.json")
        if os.path.exists(manifest_candidate):
            manifest_path = manifest_candidate

        return ExecutionResult(
            run_id=result.run_id,
            duration_seconds=result.duration_seconds,
            rows_extracted=result.total_rows,
            bytes_written=result.total_bytes,
            effective_workers=result.effective_workers,
            manifest_path=manifest_path,
            status=result.status.lower(),
        )

    except Exception as e:
        if isinstance(e, IxtractError):
            raise
        raise ExecutionError(f"Extraction failed: {e}") from e
    finally:
        connector.close()


def explain(plan_result: PlanResult) -> str:
    """Format a PlanResult as human-readable text.

    Returns the same structured output as CLI plan command.
    Useful for logging, debugging, and documentation.

    Args:
        plan_result: Result from plan() or plan_with().

    Returns:
        Multi-line formatted string.
    """
    from ixtract.planner.planner import format_plan_summary
    from ixtract.context.runtime import (
        format_runtime_context_table,
        format_worker_resolution,
        format_advisories,
        format_verdict,
    )
    from ixtract.context.estimator import format_estimate_for_cli

    lines = []

    # Plan summary
    lines.append(format_plan_summary(
        plan_result.execution_plan,
        plan_result.profile,
        worker_source=plan_result.worker_source,
    ))

    # Throughput estimate
    lines.append("")
    lines.append(format_estimate_for_cli(plan_result.throughput_estimate))

    # RuntimeContext sections (only if present)
    if plan_result.worker_resolution is not None:
        # Try to reconstruct RuntimeContext from resolution for display
        # Resolution carries all the data we need for the table
        res_text = format_worker_resolution(plan_result.worker_resolution)
        lines.append("")
        lines.append(res_text)

    if plan_result.advisories:
        adv_text = format_advisories(plan_result.advisories)
        if adv_text:
            lines.append("")
            lines.append(adv_text)

    # Verdict
    lines.append("")
    lines.append(format_verdict(plan_result.verdict))

    return "\n".join(lines)
