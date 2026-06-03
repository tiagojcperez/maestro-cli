"""Maestro CLI — multi-engine AI orchestrator.

Stable public API for programmatic use::

    from maestro_cli import load_plan, run_plan, validate_plan

See the README for YAML plan format and CLI usage.
"""
from __future__ import annotations

__all__ = [
    "__version__",
    # Core workflow
    "load_plan",
    "validate_plan",
    "run_plan",
    "scaffold_plan",
    "blame_run",
    "diff_runs",
    "audit_plan",
    # Specs (input)
    "PlanSpec",
    "TaskSpec",
    "PlanBrief",
    "TaskBrief",
    "EngineDefaults",
    "PlanDefaults",
    "PolicySpec",
    "JudgeSpec",
    # Results (output)
    "PlanRunResult",
    "TaskResult",
    "BlameChain",
    "BlameNode",
    "RunDiff",
    "TaskDiff",
    "AuditFinding",
    # Type aliases
    "EngineName",
    "ExecutionProfile",
    "TaskStatus",
    "EventCallback",
    # Exceptions
    "PlanValidationError",
    "TaskExecutionError",
]

__version__ = "2.4.0"

# --- Core workflow functions ---
from .loader import load_plan, validate_plan
from .scheduler import run_plan
from .scaffold import scaffold_plan
from .blame import blame_run
from .diff import diff_runs, RunDiff, TaskDiff
from .audit import audit_plan, AuditFinding

# --- Data types ---
from .models import (
    PlanSpec,
    TaskSpec,
    PlanBrief,
    TaskBrief,
    EngineDefaults,
    PlanDefaults,
    PolicySpec,
    JudgeSpec,
    PlanRunResult,
    TaskResult,
    BlameChain,
    BlameNode,
    EngineName,
    ExecutionProfile,
    TaskStatus,
    EventCallback,
)

# --- Exceptions ---
from .errors import PlanValidationError, TaskExecutionError
