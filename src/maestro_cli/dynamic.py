"""T2.1 — Dynamic Task Decomposition.

Handles the Phase 2 execution of ``dynamic_group: true`` tasks: builds a
``PlanSpec`` from the LLM-generated ``structured_output``, runs it as a
nested DAG, and merges the results back into the parent task.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, cast

from .models import (
    EngineName,
    ExecutionProfile,
    PlanRunResult,
    PlanSpec,
    TaskResult,
    TaskSpec,
    TokenUsage,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DYNAMIC_MAX_TASKS = 20
_DYNAMIC_MAX_RETRIES = 2
_VALID_ENGINES = frozenset({"codex", "claude", "gemini", "copilot", "qwen", "ollama", "llama"})

# Only these fields are read from LLM-generated task dicts.
# Everything else is IGNORED — security by allowlist.
_ALLOWED_TASK_FIELDS = frozenset({
    "id", "engine", "prompt", "model",
    "depends_on", "description", "tags",
})


# ---------------------------------------------------------------------------
# Build PlanSpec from structured output
# ---------------------------------------------------------------------------

def build_plan_from_output(
    output: dict[str, Any],
    parent_plan: PlanSpec,
    task: TaskSpec,
) -> PlanSpec | None:
    """Convert LLM ``structured_output`` into a validated ``PlanSpec``.

    Returns ``None`` on any validation failure (graceful degradation).
    Only fields in ``_ALLOWED_TASK_FIELDS`` are read from the output.
    """
    from .errors import PlanValidationError
    from .loader import validate_plan as _validate_plan

    if not isinstance(output, dict):
        return None

    tasks_raw = output.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        return None

    # Cap task count
    tasks_raw = tasks_raw[:_DYNAMIC_MAX_TASKS]

    sub_tasks: list[TaskSpec] = []
    seen_ids: set[str] = set()

    for idx, t in enumerate(tasks_raw):
        if not isinstance(t, dict):
            continue

        # -- id (required) --
        task_id = str(t.get("id", f"dyn-{idx}"))
        if task_id in seen_ids:
            task_id = f"{task_id}-{idx}"
        seen_ids.add(task_id)

        # -- engine (required, validated) --
        engine = t.get("engine")
        if not engine or str(engine) not in _VALID_ENGINES:
            continue

        # -- prompt (required, inline only) --
        prompt = t.get("prompt")
        if not prompt or not isinstance(prompt, str):
            continue

        # -- model (optional, inherits defaults if None) --
        model = t.get("model")
        if model is not None:
            model = str(model)

        # -- depends_on (optional, validated by validate_plan) --
        depends_on_raw = t.get("depends_on", [])
        depends_on = (
            [str(d) for d in depends_on_raw]
            if isinstance(depends_on_raw, list)
            else []
        )

        # -- description, tags (optional, harmless) --
        description = str(t.get("description", ""))
        tags_raw = t.get("tags", [])
        tags = (
            [str(tag) for tag in tags_raw]
            if isinstance(tags_raw, list)
            else []
        )

        # Build TaskSpec with safe defaults for everything else
        sub_task = TaskSpec(
            id=task_id,
            engine=cast(EngineName, str(engine)),
            model=model,
            prompt=str(prompt),
            depends_on=depends_on,
            description=description,
            tags=tags,
            # Inherit from parent defaults — LLM cannot override these
            timeout_sec=parent_plan.defaults.timeout_sec,
            max_retries=_DYNAMIC_MAX_RETRIES,
            cache=False,  # dynamic tasks should not be cached
        )
        sub_tasks.append(sub_task)

    if not sub_tasks:
        return None

    plan_name = str(output.get("name", f"{task.id}-dynamic"))

    # Compute remaining budget
    remaining_budget: float | None = None
    if parent_plan.max_cost_usd is not None:
        # Approximate: parent cost not available here, pass full budget
        # (scheduler tracks actual spending; this is a safety ceiling)
        remaining_budget = parent_plan.max_cost_usd

    sub_plan = PlanSpec(
        name=plan_name,
        version=1,
        workspace_root=parent_plan.workspace_root,
        max_parallel=min(parent_plan.max_parallel, _DYNAMIC_MAX_TASKS),
        fail_fast=True,
        max_cost_usd=remaining_budget,
        secrets=parent_plan.secrets,
        secrets_auto=parent_plan.secrets_auto,
        defaults=parent_plan.defaults,
        policies=parent_plan.policies,
        routing_strategy=parent_plan.routing_strategy,
        control_flow_integrity=True,  # ALWAYS — sub-plan is untrusted
        tasks=sub_tasks,
        source_path=parent_plan.source_path,
    )

    try:
        _validate_plan(sub_plan)
    except PlanValidationError:
        return None

    return sub_plan


# ---------------------------------------------------------------------------
# Run the dynamic sub-plan
# ---------------------------------------------------------------------------

def run_dynamic_subplan(
    sub_plan: PlanSpec,
    run_path: Path,
    task_id: str,
    dry_run: bool,
    execution_profile: ExecutionProfile,
    event_callback: Callable[[str, dict[str, object]], None] | None = None,
) -> PlanRunResult:
    """Execute a dynamically-generated sub-plan as a nested DAG."""
    from .scheduler import run_plan

    sub_run_dir = run_path / task_id / "_dynamic"
    sub_run_dir.mkdir(parents=True, exist_ok=True)

    # Wrap callback to tag events with dynamic parent
    wrapped_callback: Callable[[str, dict[str, object]], None] | None = None
    if event_callback is not None:
        def _prefixed_cb(event: str, data: dict[str, object]) -> None:
            tagged = dict(data)  # avoid mutating caller's dict
            tagged["dynamic_parent"] = task_id
            event_callback(event, tagged)
        wrapped_callback = _prefixed_cb

    return run_plan(
        sub_plan,
        dry_run=dry_run,
        execution_profile="safe",  # ALWAYS safe — sub-plan is untrusted
        run_dir_override=str(sub_run_dir),
        verbosity="normal",
        output_mode="text",
        event_callback=wrapped_callback,
    )


# ---------------------------------------------------------------------------
# Merge sub-plan result into parent task result
# ---------------------------------------------------------------------------

def merge_dynamic_result(
    phase1_result: TaskResult,
    sub_result: PlanRunResult,
    task: TaskSpec,
) -> TaskResult:
    """Merge the dynamic sub-plan result into the Phase 1 task result."""
    # Aggregate cost
    phase1_cost = phase1_result.cost_usd or 0.0
    sub_cost = sub_result.total_cost_usd or 0.0
    total_cost = phase1_cost + sub_cost

    # Aggregate tokens — keep Phase 1 per-field breakdown intact.
    # Sub-plan token total is recorded separately in dynamic_subplan_result.
    # We only merge if we can distribute sub-plan tokens reasonably.
    sub_tokens = sub_result.total_tokens or 0
    merged_tokens: TokenUsage | None
    if phase1_result.token_usage:
        # Treat sub-plan tokens as additional input (conservative estimate)
        # that preserves Phase 1's per-field accuracy.
        merged_tokens = TokenUsage(
            input_tokens=phase1_result.token_usage.input_tokens + sub_tokens,
            cached_tokens=phase1_result.token_usage.cached_tokens,
            output_tokens=phase1_result.token_usage.output_tokens,
            cache_creation_tokens=phase1_result.token_usage.cache_creation_tokens,
        ) if sub_tokens else phase1_result.token_usage
    elif sub_tokens:
        merged_tokens = TokenUsage(input_tokens=sub_tokens)
    else:
        merged_tokens = phase1_result.token_usage

    # Status
    if sub_result.success:
        status = phase1_result.status  # keep original (success)
    elif task.allow_failure:
        status = "soft_failed"
    else:
        status = "failed"

    # Sub-plan task summary counts
    ok_count = sum(
        1 for r in sub_result.task_results.values()
        if r.status in {"success", "soft_failed", "dry_run"}
    )
    fail_count = sum(
        1 for r in sub_result.task_results.values()
        if r.status == "failed"
    )
    skip_count = sum(
        1 for r in sub_result.task_results.values()
        if r.status == "skipped"
    )
    sub_summary = (
        f"Dynamic sub-plan: {ok_count} ok / {fail_count} failed / "
        f"{skip_count} skipped"
    )

    # Build meaningful stdout_tail from sub-task outputs
    sub_tails: list[str] = []
    for tid, sub_res in sub_result.task_results.items():
        tail = sub_res.stdout_tail or ""
        if tail:
            sub_tails.append(f"=== {tid} ({sub_res.status}) ===\n{tail}")
    merged_stdout = "\n\n".join(sub_tails) if sub_tails else sub_summary

    # Build structured summary for {{ task.output.sub_tasks }}
    sub_tasks_summary = [
        {
            "id": tid,
            "status": r.status,
            "summary": (r.stdout_tail or "")[:500],
        }
        for tid, r in sub_result.task_results.items()
    ]

    # Update result
    phase1_result.status = status
    phase1_result.cost_usd = total_cost if total_cost > 0 else phase1_result.cost_usd
    phase1_result.token_usage = merged_tokens
    phase1_result.message = (
        f"{phase1_result.message or ''}\n{sub_summary}".strip()
    )
    phase1_result.stdout_tail = merged_stdout
    phase1_result.structured_output = {
        "sub_tasks": sub_tasks_summary,
        "ok": ok_count,
        "failed": fail_count,
        "skipped": skip_count,
    }
    phase1_result.dynamic_subplan_result = {
        "plan_name": sub_result.plan_name,
        "success": sub_result.success,
        "task_count": len(sub_result.task_results),
        "ok_count": ok_count,
        "fail_count": fail_count,
        "skip_count": skip_count,
        "total_cost_usd": sub_result.total_cost_usd,
        "total_tokens": sub_result.total_tokens,
        "run_path": str(sub_result.run_path),
    }

    return phase1_result


# ---------------------------------------------------------------------------
# Write raw output for forensics
# ---------------------------------------------------------------------------

def write_raw_output(
    run_path: Path,
    task_id: str,
    raw_output: dict[str, Any],
) -> None:
    """Write the raw LLM output before filtering for post-incident analysis."""
    forensics_dir = run_path / task_id / "_dynamic"
    forensics_dir.mkdir(parents=True, exist_ok=True)
    forensics_path = forensics_dir / "raw_output.json"
    try:
        forensics_path.write_text(
            json.dumps(raw_output, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    except OSError:
        pass  # never fail the run for forensics
