"""CI Agentic Workflows — automatic failure analysis and remediation.

Extends the CI generator with an agentic layer that analyzes CI
failures, suggests fixes, and can trigger automatic re-runs with
escalated models or adjusted parameters.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .blame import blame_run
from .knowledge import consolidate_knowledge, extract_knowledge, store_knowledge
from .models import PlanRunResult


@dataclass
class CiFailureAnalysis:
    """Analysis of a CI run failure with remediation suggestions."""

    run_path: Path
    failed_tasks: list[str] = field(default_factory=list)
    root_causes: list[str] = field(default_factory=list)
    cascading_failures: list[str] = field(default_factory=list)
    remediation_actions: list[CiRemediationAction] = field(default_factory=list)
    should_retry: bool = False
    retry_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_path": str(self.run_path),
            "failed_tasks": self.failed_tasks,
            "root_causes": self.root_causes,
            "cascading_failures": self.cascading_failures,
            "remediation_actions": [a.to_dict() for a in self.remediation_actions],
            "should_retry": self.should_retry,
            "retry_config": self.retry_config,
        }


@dataclass
class CiRemediationAction:
    """A specific remediation action for a CI failure."""

    task_id: str
    action: str  # escalate_model, increase_timeout, skip_task, add_retry
    reason: str
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "action": self.action,
            "reason": self.reason,
            "params": self.params,
        }


_RETRIABLE_CATEGORIES = {
    "timeout", "rate_limited", "context_exceeded", "runtime_error",
}
_ESCALATABLE_CATEGORIES = {
    "compilation_error", "test_failure", "validation_error",
    "output_format_error",
}


def analyze_ci_failure(
    run_path: Path,
    plan_name: str | None = None,
) -> CiFailureAnalysis:
    """Analyze a failed CI run and produce remediation suggestions.

    Uses blame attribution to identify root causes, then maps
    failure categories to remediation actions.
    """
    analysis = CiFailureAnalysis(run_path=run_path)

    # Load manifest
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        return analysis

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return analysis

    task_results = manifest.get("task_results", {})

    # Identify failed tasks
    for task_id, result in task_results.items():
        if result.get("status") == "failed":
            analysis.failed_tasks.append(task_id)

    if not analysis.failed_tasks:
        return analysis

    # Use blame for causal attribution
    try:
        blame_chain = blame_run(run_path)
        for node in blame_chain.nodes:
            if node.category == "root_cause":
                analysis.root_causes.append(node.task_id)
            elif node.category == "dependency_cascade":
                analysis.cascading_failures.append(node.task_id)
    except Exception:
        # Blame is best-effort
        analysis.root_causes = list(analysis.failed_tasks)

    # Generate remediation actions
    for task_id in analysis.root_causes or analysis.failed_tasks:
        result = task_results.get(task_id, {})
        failure_history = result.get("failure_history", [])

        for fh in failure_history:
            category = fh.get("category", "unknown")

            if category in _RETRIABLE_CATEGORIES:
                analysis.should_retry = True
                if category == "timeout":
                    original = result.get("timeout_sec", 1800)
                    analysis.remediation_actions.append(CiRemediationAction(
                        task_id=task_id,
                        action="increase_timeout",
                        reason=f"Task timed out after {result.get('duration_sec', 0):.0f}s",
                        params={"timeout_sec": int(original * 1.5)},
                    ))
                elif category == "rate_limited":
                    analysis.remediation_actions.append(CiRemediationAction(
                        task_id=task_id,
                        action="add_delay",
                        reason="Rate limited by API provider",
                        params={"retry_delay_sec": 30},
                    ))
                elif category == "context_exceeded":
                    analysis.remediation_actions.append(CiRemediationAction(
                        task_id=task_id,
                        action="reduce_context",
                        reason="Context window exceeded",
                        params={"context_compaction": "progressive"},
                    ))

            if category in _ESCALATABLE_CATEGORIES:
                analysis.should_retry = True
                analysis.remediation_actions.append(CiRemediationAction(
                    task_id=task_id,
                    action="escalate_model",
                    reason=f"Failed with {category} — escalating to more capable model",
                    params={"escalation": ["sonnet", "opus"]},
                ))

    if analysis.should_retry:
        analysis.retry_config = {
            "resume": True,
            "only": analysis.root_causes or analysis.failed_tasks,
        }

    return analysis


def learn_from_ci_run(
    run_result: PlanRunResult,
    source_dir: Path,
) -> int:
    """Extract knowledge from a completed CI run and store it.

    Returns the number of new records stored.
    """
    records = extract_knowledge(run_result)
    if not records:
        return 0
    store_knowledge(run_result.plan_name, source_dir, records)
    return len(records)


def format_ci_analysis(analysis: CiFailureAnalysis) -> str:
    """Format CI failure analysis for human-readable output."""
    lines = ["[maestro] CI Failure Analysis", ""]

    if analysis.root_causes:
        lines.append(f"  Root causes: {', '.join(analysis.root_causes)}")
    if analysis.cascading_failures:
        lines.append(f"  Cascading: {', '.join(analysis.cascading_failures)}")

    if analysis.remediation_actions:
        lines.append("")
        lines.append("  Remediation actions:")
        for action in analysis.remediation_actions:
            lines.append(f"    - [{action.task_id}] {action.action}: {action.reason}")

    if analysis.should_retry:
        lines.append("")
        only = analysis.retry_config.get("only", [])
        lines.append(f"  Recommended: retry with --only {','.join(only)}")

    return "\n".join(lines)
