from __future__ import annotations

import hashlib
import json
import re
import subprocess
import threading
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

from .loader import load_plan
from .scheduler import run_plan
from .models import (
    ExecutionProfile,
    OnRegression,
    OutputMode,
    PlanDefaults,
    PlanRunResult,
    PlanSpec,
    SessionSnapshot,
    SteppingStone,
    TaskSpec,
    Verbosity,
    LessonRecord,
    WatchIteration,
    WatchSpec,
    WatchState,
)
from .blame import blame_run
from .utils import resolve_path, sanitize_dirname

_HISTORY_LIMIT = 10
_ITERATION_EXCERPT_MAX_CHARS = 600
_FIX_SUMMARY_MAX_CHARS = 240
_SESSION_MEMORY_TRIGGER_ITERATION = 8
_SESSION_MEMORY_RECENT_TAIL = 3
_SESSION_MEMORY_MAX_SNAPSHOTS = 5
_SESSION_MEMORY_MAX_CHARS = 4000
_LOG_SECTION_HEADERS = {"[verify_command]", "[guard_command]"}
_EMPTY_SESSION_MEMORY_TEXT = "No durable session memory yet."
_EMPTY_RECENT_OUTPUTS_TEXT = "No recent verbatim iteration outputs yet."
_DEFAULT_CONSOLIDATION_PROMPT = (
    "Review the experiment history below. Identify:\n"
    "1. Which approaches consistently succeed\n"
    "2. Which approaches caused regressions\n"
    "3. What areas remain untested\n"
    "Produce a concise strategy for the next iterations."
)


_DEFAULT_IMPROVE_PROGRAM = """\
## Fix Priority Order (cheapest first)

1. **guard_command / verify_command bugs** -- fix assertion value, path, or pattern ($0)
2. **Timeout too low** -- `timeout_sec *= 2`, max 3600 ($0)
3. **Wrong paths/commands** -- fix the file path or command so it works ($0)
4. **Near-timeout** -- `duration > 0.8 x timeout` -> preemptive increase ($0)
5. **Task decomposition** -- split tasks with 2+ independent actions ($0)
6. **Retries exhausted** -- `max_retries += 1` or add `escalation` (+$1-3)
7. **Model confused** -- escalate model or clarify prompt (+$1-5)

## Rules

- **ONE fix per iteration** -- never mix fix categories
- **PRESERVE all existing values** -- only change the ONE field you are fixing; do NOT rewrite or revert other fields that were already modified in previous iterations
- **Use surgical edits** -- change only the specific value that needs fixing, leave everything else exactly as-is
- **Never remove** `verify_command` or `guard_command` -- fix them or relax them
- **Never set** `timeout_sec > 3600` or `max_retries > 3`
- **Never add** `allow_failure: true` to mask real failures
- **Never reduce** the number of tasks or remove judge blocks
- **Only modify** the plan file -- no other files
- **NEVER modify tasks marked `frozen: true`** -- these are quality gates that must not be weakened
- **Fix root causes first** -- if Task C fails because Task A failed, fix Task A

## Failure Classification

| Signal | Diagnosis | Fix |
|--------|-----------|-----|
| `exit_code: 124` | Timeout | `timeout_sec *= 2` |
| Guard/verify exits non-zero but task output correct | guard/verify bug | Fix assertion |
| `FileNotFoundError` or `No such file` | Wrong path | Fix the path |
| Task skipped because dependency failed | Cascading | Fix upstream first |
| `duration > 0.8 x timeout_sec` and passes | Near-timeout | Preemptive increase |
"""

_IMPROVE_PROMPT_TEMPLATE = """\
Iteration {{ watch.iteration }}. Tasks passed last run: {{ watch.last_metric }} / {{ improve.total_tasks }}
(best so far: {{ watch.best_metric }}).

{{ watch.history }}

{{ watch.experiments_summary }}

{{ watch.program }}

## Session Memory

{{ watch.session_memory }}

## Recent Iteration Outputs

{{ watch.recent_outputs }}

## Failure Analysis (auto-injected from last run)

### Blame (root cause attribution)
{{ watch.blame }}

### Task Results (status, exit codes, messages)
{{ watch.manifest }}

## Lessons From Previous Iterations

{{ watch.lessons }}

## Your Task

1. Read `{{ improve.plan_path }}` (the plan being improved)
2. Use the blame and manifest data above to understand what failed and why
3. Classify failures using the priority table above
4. Apply exactly ONE fix (the cheapest root-cause fix)
5. Write the corrected plan back to `{{ improve.plan_path }}`
6. Do NOT modify any other files

After writing the fix, print a one-line summary:
`FIX: <category> -- <what you changed>`
"""


def _build_improve_plan(
    target_plan: PlanSpec,
    spec: WatchSpec,
    plan_path_rel: str,
) -> PlanSpec:
    """Build an internal 1-task PlanSpec for the improve agent."""
    model = spec.improve_model or "sonnet"
    return PlanSpec(
        name=f"improve-{target_plan.name}",
        version=1,
        workspace_root=target_plan.workspace_root,
        secrets=target_plan.secrets,
        secrets_auto=target_plan.secrets_auto,
        max_cost_usd=spec.max_cost_usd,
        fail_fast=True,
        defaults=PlanDefaults(
            timeout_sec=180,
            env=target_plan.defaults.env if target_plan.defaults.env else {},
        ),
        tasks=[
            TaskSpec(
                id="improve-plan",
                engine="claude",
                model=model,
                prompt=_IMPROVE_PROMPT_TEMPLATE,
                description="Analyze failures and fix the plan",
                timeout_sec=180,
                max_retries=1,
                verify_command=["python", "-m", "maestro_cli", "validate", plan_path_rel],
            ),
        ],
        source_path=target_plan.source_path,
    )


# ---------------------------------------------------------------------------
# Knowledge Archive — lesson extraction and retrieval
# ---------------------------------------------------------------------------

_LESSON_DECAY_HALF_LIFE_DAYS = 30.0


_FIX_LINE_RE = re.compile(r"^FIX:\s*(.+)$", re.MULTILINE)


def _truncate_iteration_excerpt(
    text: str,
    *,
    max_chars: int = _ITERATION_EXCERPT_MAX_CHARS,
) -> str | None:
    """Return a prompt-ready excerpt or None for blank input."""
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return None
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _extract_fix_summary(improve_log: str) -> str | None:
    """Extract the one-line FIX summary emitted by the improve agent."""
    if not improve_log:
        return None
    match = _FIX_LINE_RE.search(improve_log)
    if match is None:
        return None
    return _truncate_iteration_excerpt(
        match.group(1).strip(),
        max_chars=_FIX_SUMMARY_MAX_CHARS,
    )


def _capture_iteration_excerpts(
    run_path: Path | None,
    *,
    improve_log_text: str = "",
    consolidated_summary: str = "",
) -> dict[str, str | None]:
    """Capture prompt-relevant excerpts for persistence in WatchIteration."""
    blame_excerpt: str | None = None
    manifest_excerpt: str | None = None
    if run_path is not None:
        blame_json, manifest_summary = _build_blame_context(run_path)
        blame_excerpt = _truncate_iteration_excerpt(blame_json)
        manifest_excerpt = _truncate_iteration_excerpt(manifest_summary)

    return {
        "fix_summary": _extract_fix_summary(improve_log_text),
        "manifest_excerpt": manifest_excerpt,
        "blame_excerpt": blame_excerpt,
        "consolidated_excerpt": _truncate_iteration_excerpt(consolidated_summary),
    }


def _one_line_excerpt(text: str | None, *, max_chars: int = 160) -> str | None:
    """Collapse an excerpt to a single line for compact snapshot sections."""
    if not text:
        return None
    collapsed = " ".join(text.split())
    if not collapsed:
        return None
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def _build_recent_iteration_outputs(
    iterations: list[WatchIteration],
    *,
    recent_tail_count: int = _SESSION_MEMORY_RECENT_TAIL,
) -> str:
    """Render the most recent stored iteration excerpts verbatim."""
    if not iterations:
        return _EMPTY_RECENT_OUTPUTS_TEXT

    blocks: list[str] = []
    for item in iterations[-recent_tail_count:]:
        sections: list[str] = []
        if item.fix_summary:
            sections.append(f"FIX summary: {item.fix_summary}")
        if item.manifest_excerpt:
            sections.append(f"Manifest excerpt:\n{item.manifest_excerpt}")
        if item.blame_excerpt:
            sections.append(f"Blame excerpt:\n{item.blame_excerpt}")
        if item.consolidated_excerpt:
            sections.append(f"Consolidated excerpt:\n{item.consolidated_excerpt}")
        if not sections:
            continue
        action = item.action or "n/a"
        blocks.append(
            f"### Iteration {item.iteration} ({action})\n"
            + "\n\n".join(sections)
        )

    if not blocks:
        return _EMPTY_RECENT_OUTPUTS_TEXT
    return "\n\n".join(blocks)


def _build_session_memory_snapshot(
    iterations: list[WatchIteration],
    lessons: list[LessonRecord],
    *,
    plan_name: str,
    watch_run_path: str,
    metric_name: str,
    metric_direction: str,
    plateau_count: int,
    plateau_threshold: int,
    consolidate_model: str | None,
    consolidated_summary: str,
    recent_tail_count: int = _SESSION_MEMORY_RECENT_TAIL,
) -> SessionSnapshot | None:
    """Build a deterministic snapshot from older watch iterations."""
    if len(iterations) < _SESSION_MEMORY_TRIGGER_ITERATION:
        return None

    archived = iterations[:-recent_tail_count] if len(iterations) > recent_tail_count else []
    if not archived:
        return None

    successes = [it for it in archived if it.improved]
    failures = [it for it in archived if not it.improved and it.action not in ("baseline", "")]
    latest_archived = archived[-1]

    lines: list[str] = [
        f"Plan: {plan_name}",
        "",
        "### Goal and Metric State",
        f"- Metric: {metric_name} ({metric_direction})",
        f"- Snapshot range: iterations {archived[0].iteration}-{archived[-1].iteration}",
        f"- Best metric so far: {'-' if latest_archived.best_metric is None else f'{latest_archived.best_metric:g}'}",
        f"- Plateau count at snapshot time: {plateau_count}/{plateau_threshold}",
    ]

    lines.append("")
    lines.append("### Best Known Working Approaches")
    if successes:
        for item in successes[-5:]:
            metric_text = "-" if item.metric_value is None else f"{item.metric_value:g}"
            detail = item.fix_summary or item.action or "improved"
            lines.append(f"- Iteration {item.iteration}: {detail} -> metric {metric_text}")
    else:
        lines.append("- None yet.")

    lines.append("")
    lines.append("### Failed Approaches Not To Repeat")
    if failures:
        for item in failures[-5:]:
            metric_text = "-" if item.metric_value is None else f"{item.metric_value:g}"
            detail = item.fix_summary or item.error or item.action or "failed attempt"
            lines.append(f"- Iteration {item.iteration}: {detail} -> metric {metric_text}")
    else:
        lines.append("- No repeated failures recorded yet.")

    blockers: list[str] = []
    for item in reversed(archived):
        if item.error:
            blockers.append(f"Iteration {item.iteration}: {item.error}")
        manifest_hint = _one_line_excerpt(item.manifest_excerpt)
        if manifest_hint:
            blockers.append(f"Iteration {item.iteration} manifest: {manifest_hint}")
        blame_hint = _one_line_excerpt(item.blame_excerpt)
        if blame_hint:
            blockers.append(f"Iteration {item.iteration} blame: {blame_hint}")
        if len(blockers) >= 3:
            break

    lines.append("")
    lines.append("### Current Blockers")
    if blockers:
        for blocker in blockers[:3]:
            lines.append(f"- {blocker}")
    else:
        lines.append("- No clear blockers in archived iterations.")

    lines.append("")
    lines.append("### Lessons")
    if lessons:
        for lesson in lessons[:5]:
            conf = f"{lesson.confidence:.0%}"
            task = f" (task: {lesson.task_id})" if lesson.task_id else ""
            lines.append(f"- [{conf}]{task} {lesson.lesson}")
    else:
        lines.append("- No durable lessons yet.")

    lines.append("")
    lines.append("### Resume Hints")
    latest_fix = next((it.fix_summary for it in reversed(archived) if it.fix_summary), None)
    if latest_fix:
        lines.append(f"- Last archived fix attempt: {latest_fix}")
    consolidated_hint = _one_line_excerpt(consolidated_summary, max_chars=220)
    if consolidated_hint:
        lines.append(f"- Consolidated strategy: {consolidated_hint}")
    lines.append(
        f"- Keep the last {recent_tail_count} iterations verbatim; use this snapshot"
        " only for older strategic continuity."
    )

    snapshot_text = _truncate_iteration_excerpt(
        "\n".join(lines),
        max_chars=_SESSION_MEMORY_MAX_CHARS,
    )
    if not snapshot_text:
        return None

    return SessionSnapshot(
        plan_name=plan_name,
        watch_run_path=watch_run_path,
        snapshot_kind="watch",
        iteration_from=archived[0].iteration,
        iteration_to=archived[-1].iteration,
        best_metric=latest_archived.best_metric,
        snapshot_text=snapshot_text,
        recent_tail_count=recent_tail_count,
        source_type="watch",
        source_id=f"{watch_run_path}:{archived[-1].iteration}",
        metadata={
            "metric_name": metric_name,
            "metric_direction": metric_direction,
            "plateau_count": plateau_count,
            "consolidate_model": consolidate_model or "",
            "summary_method": "deterministic",
        },
    )


def _maybe_extract_session_memory(
    *,
    plan_name: str,
    source_dir: Path,
    watch_run_path: Path,
    iterations: list[WatchIteration],
    lessons: list[LessonRecord],
    metric_name: str,
    metric_direction: str,
    plateau_count: int,
    plateau_threshold: int,
    consolidate_model: str | None,
    consolidated_summary: str,
) -> SessionSnapshot | None:
    """Persist a deterministic snapshot when the archived range grows."""
    snapshot = _build_session_memory_snapshot(
        iterations,
        lessons,
        plan_name=plan_name,
        watch_run_path=str(watch_run_path),
        metric_name=metric_name,
        metric_direction=metric_direction,
        plateau_count=plateau_count,
        plateau_threshold=plateau_threshold,
        consolidate_model=consolidate_model,
        consolidated_summary=consolidated_summary,
    )
    if snapshot is None:
        return None

    from .memory import (
        load_latest_session_snapshot,
        prune_session_snapshots,
        store_session_snapshot,
    )

    latest = load_latest_session_snapshot(
        plan_name,
        source_dir,
        watch_run_path=str(watch_run_path),
    )
    if latest is not None and latest.iteration_to >= snapshot.iteration_to:
        return latest

    store_session_snapshot(plan_name, source_dir, snapshot)
    prune_session_snapshots(
        plan_name,
        source_dir,
        watch_run_path=str(watch_run_path),
        keep=_SESSION_MEMORY_MAX_SNAPSHOTS,
    )
    latest = load_latest_session_snapshot(
        plan_name,
        source_dir,
        watch_run_path=str(watch_run_path),
    )
    return latest or snapshot


def _load_session_memory_text(
    plan_name: str,
    source_dir: Path,
    *,
    watch_run_path: Path,
) -> str:
    """Return the latest non-quarantined session snapshot text for a run."""
    from .memory import load_latest_session_snapshot

    snapshot = load_latest_session_snapshot(
        plan_name,
        source_dir,
        watch_run_path=str(watch_run_path),
    )
    if snapshot is None or not snapshot.snapshot_text.strip():
        return _EMPTY_SESSION_MEMORY_TEXT
    return snapshot.snapshot_text


def _extract_lesson(
    iteration: WatchIteration,
    manifest_summary: str,
    improve_log: str = "",
) -> LessonRecord | None:
    """Extract a semantic lesson from a completed iteration.

    Returns None if no actionable lesson can be derived (e.g. baseline run).
    The *improve_log* parameter receives the improve agent's task log content
    so we can extract the ``FIX: <description>`` line.
    """
    if not iteration.action or iteration.action in ("baseline", "validation_failed"):
        return None

    category = "unknown"
    task_id = ""
    lesson_text = ""

    action = iteration.action
    metric = iteration.metric_value
    improved = iteration.improved

    # Try to extract the FIX: line from the improve agent's log
    fix_description = _extract_fix_summary(improve_log) or ""

    # Identify which task was failing from manifest
    if manifest_summary:
        for line in manifest_summary.splitlines():
            if "failed" in line.lower():
                parts = line.strip().split(":")
                if parts:
                    task_id = parts[0].strip()
                    break

    if improved and fix_description:
        lesson_text = f"Iteration {iteration.iteration}: {fix_description} (metric {metric})"
        category = "successful_fix"
    elif improved:
        lesson_text = f"Iteration {iteration.iteration}: metric improved to {metric}."
        category = "successful_fix"
    elif fix_description:
        lesson_text = f"Iteration {iteration.iteration}: tried '{fix_description}' but no improvement (metric={metric}), {action}."
        category = "failed_attempt"
    else:
        lesson_text = f"Iteration {iteration.iteration}: no improvement (metric={metric}), {action}."
        category = "failed_attempt"

    return LessonRecord(
        iteration=iteration.iteration,
        task_id=task_id,
        category=category,
        lesson=lesson_text,
        confidence=0.9 if improved else 0.5,
        timestamp=iteration.timestamp or datetime.now().isoformat(),
    )


def _write_lesson(lessons_path: Path, lesson: LessonRecord) -> None:
    """Append a lesson to the lessons.jsonl file."""
    with lessons_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(lesson.to_dict()) + "\n")
        fh.flush()


def _load_lessons(lessons_path: Path, max_lessons: int = 20) -> list[LessonRecord]:
    """Load lessons from jsonl, apply time-decay, return most relevant."""
    if not lessons_path.exists():
        return []

    lessons: list[LessonRecord] = []
    now = datetime.now()

    try:
        for line in lessons_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                lr = LessonRecord(
                    iteration=data.get("iteration", 0),
                    task_id=data.get("task_id", ""),
                    category=data.get("category", ""),
                    lesson=data.get("lesson", ""),
                    confidence=data.get("confidence", 0.5),
                    timestamp=data.get("timestamp", ""),
                )
                # Apply time-decay
                if lr.timestamp:
                    try:
                        ts = datetime.fromisoformat(lr.timestamp)
                        age_days = (now - ts).total_seconds() / 86400
                        decay = 0.5 ** (age_days / _LESSON_DECAY_HALF_LIFE_DAYS)
                        lr.confidence *= decay
                    except (ValueError, TypeError):
                        pass
                lessons.append(lr)
            except (json.JSONDecodeError, KeyError):
                continue
    except OSError:
        return []

    # Sort by confidence (highest first), take top N
    lessons.sort(key=lambda l: l.confidence, reverse=True)
    return lessons[:max_lessons]


def _format_lessons(lessons: list[LessonRecord]) -> str:
    """Format lessons for injection into the improve prompt."""
    if not lessons:
        return "No lessons from previous iterations."
    lines: list[str] = []
    for lr in lessons:
        conf = f"{lr.confidence:.0%}"
        task = f" (task: {lr.task_id})" if lr.task_id else ""
        lines.append(f"- [{conf}]{task} {lr.lesson}")
    return "\n".join(lines)


def _watch_improve(
    plan_path: Path,
    plan: PlanSpec,
    spec: WatchSpec,
    *,
    max_parallel_override: int | None = None,
    event_callback: Callable[[str, dict[str, object]], None] | None = None,
    cancel_event: threading.Event | None = None,
    dry_run: bool = False,
    execution_profile: ExecutionProfile = "plan",
    verbosity: Verbosity = "normal",
    output_mode: OutputMode = "text",
    auto_approve: bool = False,
    resume_from: Path | None = None,
    extra_template_vars: dict[str, str] | None = None,
) -> WatchState:
    """Run the improve loop: alternate between improve-agent and target-plan execution."""
    workdir = resolve_path(plan.source_dir, plan.workspace_root) or plan.source_dir.resolve()
    program_text = _load_program(plan, spec) or _DEFAULT_IMPROVE_PROGRAM
    watch_root = resolve_path(plan.source_dir, plan.run_dir)
    if watch_root is None:
        raise ValueError("Unable to resolve run directory")

    plan_path_rel = str(plan_path.relative_to(workdir)) if workdir in plan_path.parents else str(plan_path)

    # Auto-compute target_metric if not set
    if spec.target_metric is None:
        spec = WatchSpec(
            **{**spec.to_dict(), "target_metric": float(len(plan.tasks))}
        )

    # Build the internal improve plan
    improve_plan = _build_improve_plan(plan, spec, plan_path_rel)

    if resume_from is not None:
        watch_run_path = Path(resume_from).resolve()
        state = _resume_watch_state(watch_run_path)
        state.plan_path = str(plan_path)
    else:
        watch_run_path = (
            watch_root
            / f"watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitize_dirname(plan.name)}"
        ).resolve()
        watch_run_path.mkdir(parents=True, exist_ok=True)
        state = WatchState(plan_path=str(plan_path), status="max_iterations")

    experiments_path = watch_run_path / "experiments.jsonl"
    lessons_path = watch_run_path / "lessons.jsonl"

    # Apply best prior stepping stone if available and not resuming
    if spec.stepping_stones and resume_from is None and not state.iterations:
        higher = spec.metric_direction == "higher_is_better"
        prior_stone = _load_best_stepping_stone(
            plan.name, plan.source_dir, spec.metric, higher_is_better=higher,
        )
        if prior_stone is not None:
            applied = _apply_stepping_stone(prior_stone, plan_path)
            if applied:
                state.best_metric = prior_stone.metric_value
                print(
                    f"[maestro] watch improve: applied stepping stone from iteration "
                    f"{prior_stone.iteration} (metric={prior_stone.metric_value:g})"
                )
                _emit(
                    event_callback,
                    "stepping_stone_applied",
                    metric_value=prior_stone.metric_value,
                    source_iteration=prior_stone.iteration,
                    source_run=prior_stone.watch_run_path,
                )
                # Reload plan from applied stone
                plan = load_plan(plan_path)
                improve_plan = _build_improve_plan(plan, spec, plan_path_rel)

    _emit(
        event_callback,
        "watch_start",
        plan_name=plan.name,
        max_iterations=spec.max_iterations,
        metric=spec.metric,
        metric_direction=spec.metric_direction,
        mode="improve",
    )

    print(f"[maestro] watch improve: run dir {watch_run_path}")
    print(f"[maestro] watch improve: target plan {plan_path_rel} ({len(plan.tasks)} tasks)")

    if dry_run:
        _emit(
            event_callback,
            "watch_complete",
            status=state.status,
            best_metric=state.best_metric,
            best_iteration=state.best_iteration,
            total_iterations=state.total_iterations,
            total_cost_usd=state.total_cost_usd,
        )
        return state

    consolidated_summary: str = ""
    last_target_run_path: Path | None = None  # Track target run path directly
    # Snapshot frozen tasks for post-edit validation
    original_frozen_tasks: dict[str, dict[str, object]] = {
        t.id: t.to_dict() for t in plan.tasks if t.frozen
    }
    try:
        start_iteration = state.total_iterations + 1
        for iteration_number in range(start_iteration, spec.max_iterations + 1):
            if cancel_event is not None and cancel_event.is_set():
                state.status = "interrupted"
                break

            if spec.max_cost_usd is not None and state.total_cost_usd >= spec.max_cost_usd:
                state.status = "budget_exceeded"
                break

            if state.plateau_count >= spec.plateau_threshold:
                state.status = "plateau"
                break

            if spec.max_total_steps is not None and state.total_steps >= spec.max_total_steps:
                state.status = "step_limit_reached"
                _emit(
                    event_callback,
                    "watch_step_limit",
                    total_steps=state.total_steps,
                    max_total_steps=spec.max_total_steps,
                )
                break

            _emit(
                event_callback,
                "iteration_start",
                iteration=iteration_number,
                best_metric=state.best_metric,
            )

            # Consolidation agent (reuse existing logic)
            if (
                spec.consolidate_model is not None
                and state.iterations
                and state.total_iterations % spec.consolidate_every == 0
            ):
                consolidated_summary = _run_consolidation(
                    spec,
                    _build_history_text(state.iterations),
                    workdir,
                    plan=plan,
                )

            iteration_started = time.monotonic()
            last_metric = state.iterations[-1].metric_value if state.iterations else None

            # Build blame context from the PREVIOUS iteration's target run
            # (use tracked path directly to avoid name suffix collisions)
            blame_json = ""
            manifest_summary = ""
            if last_target_run_path is not None:
                blame_json, manifest_summary = _build_blame_context(last_target_run_path)

            extra_vars: dict[str, str] = dict(extra_template_vars) if extra_template_vars else {}
            extra_vars.update({
                "watch.iteration": str(iteration_number),
                "watch.best_metric": "" if state.best_metric is None else str(state.best_metric),
                "watch.last_metric": "" if last_metric is None else str(last_metric),
                "watch.history": _build_history_text(state.iterations),
                "watch.experiments_summary": _build_experiments_summary(
                    state.iterations,
                    plateau_count=state.plateau_count,
                    plateau_threshold=spec.plateau_threshold,
                ),
                "watch.program": program_text,
                "watch.consolidated": consolidated_summary,
                "watch.blame": blame_json,
                "watch.manifest": manifest_summary,
                "watch.session_memory": _load_session_memory_text(
                    plan.name,
                    plan.source_dir,
                    watch_run_path=watch_run_path,
                ),
                "watch.recent_outputs": _build_recent_iteration_outputs(state.iterations),
                "improve.plan_path": plan_path_rel,
                "improve.total_tasks": str(len(plan.tasks)),
                "improve.frozen_tasks": ", ".join(original_frozen_tasks.keys()) if original_frozen_tasks else "none",
                "watch.lessons": _format_lessons(_load_lessons(lessons_path)),
            })

            # Phase 0 / Phase 1: Run improve agent (skip on first iteration)
            iteration_cost: float | None = None
            iteration_steps: int = 0
            improve_log_text: str = ""
            if iteration_number == 1 and not state.iterations:
                # First iteration: just run target plan for baseline
                print("[maestro] watch improve: iteration 1 — baseline run")
            else:
                # Run the improve agent
                # Improve agent MUST use yolo profile to write plan files
                improve_result = run_plan(
                    improve_plan,
                    dry_run=False,
                    execution_profile="yolo",
                    max_parallel_override=1,
                    verbosity=verbosity,
                    output_mode=output_mode,
                    auto_approve=auto_approve,
                    event_callback=event_callback,
                    cancel_event=cancel_event,
                    extra_template_vars=extra_vars,
                )
                iteration_cost = improve_result.total_cost_usd
                iteration_steps += _count_executed_tasks(improve_result)

                # Read improve agent's log for lesson extraction
                improve_task_result = improve_result.task_results.get("improve-plan")
                if improve_task_result and improve_task_result.log_path.exists():
                    try:
                        improve_log_text = improve_task_result.log_path.read_text(
                            encoding="utf-8", errors="replace",
                        )[-2000:]  # last 2000 chars is enough for FIX: line
                    except OSError:
                        pass

                # Phase 2: Validate the modified plan
                try:
                    plan = load_plan(plan_path)
                except Exception as exc:
                    print(f"[maestro] watch improve: validation failed after improve agent: {exc}")
                    _git_rollback(workdir, spec.on_regression)
                    # Record failed iteration
                    iteration_excerpts = _capture_iteration_excerpts(
                        None,
                        improve_log_text=improve_log_text,
                        consolidated_summary=consolidated_summary,
                    )
                    wi = WatchIteration(
                        iteration=iteration_number,
                        metric_value=last_metric,
                        best_metric=state.best_metric,
                        improved=False,
                        action="validation_failed",
                        cost_usd=iteration_cost,
                        duration_sec=time.monotonic() - iteration_started,
                        error=str(exc),
                        timestamp=datetime.now().isoformat(),
                        **iteration_excerpts,
                    )
                    state.iterations.append(wi)
                    state.total_iterations = len(state.iterations)
                    state.plateau_count += 1
                    _write_experiment(experiments_path, wi)
                    _maybe_extract_session_memory(
                        plan_name=plan.name,
                        source_dir=plan.source_dir,
                        watch_run_path=watch_run_path,
                        iterations=state.iterations,
                        lessons=_load_lessons(lessons_path),
                        metric_name=spec.metric,
                        metric_direction=spec.metric_direction,
                        plateau_count=state.plateau_count,
                        plateau_threshold=spec.plateau_threshold,
                        consolidate_model=spec.consolidate_model,
                        consolidated_summary=consolidated_summary,
                    )
                    _emit(
                        event_callback,
                        "iteration_complete",
                        iteration=iteration_number,
                        metric_value=last_metric,
                        best_metric=state.best_metric,
                        improved=False,
                        action="validation_failed",
                    )
                    continue

            # Check frozen tasks weren't modified
            if original_frozen_tasks:
                modified_plan = load_plan(plan_path)
                for tid, original_task in original_frozen_tasks.items():
                    for t in modified_plan.tasks:
                        if t.id == tid and t.to_dict() != original_task:
                            print(f"[maestro] watch improve: frozen task '{tid}' was modified — rolling back")
                            _git_rollback(workdir, spec.on_regression)
                            break

            # Phase 3: Run the target plan to measure metric
            target_plan_loaded = load_plan(plan_path)
            target_plan_loaded.fail_fast = False
            target_plan_loaded.watch = None  # prevent recursion

            target_result = run_plan(
                target_plan_loaded,
                dry_run=False,
                execution_profile=execution_profile,
                max_parallel_override=max_parallel_override,
                verbosity=verbosity,
                output_mode=output_mode,
                auto_approve=auto_approve,
                event_callback=event_callback,
                cancel_event=cancel_event,
            )

            duration_sec = time.monotonic() - iteration_started
            target_cost = target_result.total_cost_usd
            total_iter_cost = (iteration_cost or 0.0) + (target_cost or 0.0) if (iteration_cost is not None or target_cost is not None) else None
            if total_iter_cost is not None:
                state.total_cost_usd += total_iter_cost
            iteration_steps += _count_executed_tasks(target_result)
            state.total_steps += iteration_steps

            metric_value = _extract_manifest_metric(target_result)
            last_target_run_path = target_result.run_path  # Track for next iteration's blame
            warmup = iteration_number <= spec.warmup_iterations
            # In improve mode, equal metric = keep (the fix may prepare
            # for the next iteration even if it doesn't increase the count)
            improved = warmup or _is_improvement(metric_value, state.best_metric, spec)
            if not improved and metric_value is not None and state.best_metric is not None:
                if metric_value == state.best_metric:
                    improved = True  # keep lateral fixes

            if metric_value is not None:
                _emit(
                    event_callback,
                    "metric_recorded",
                    iteration=iteration_number,
                    metric=spec.metric,
                    value=metric_value,
                    best=state.best_metric,
                    improved=improved,
                )

            action = "keep"
            git_commit: str | None = None
            error: str | None = None

            if improved:
                action = "warmup_keep" if warmup else "keep"
                git_commit = _git_commit_changes(workdir, iteration_number, spec.metric, metric_value or 0.0)
                state.best_metric = metric_value
                state.best_iteration = iteration_number
                state.plateau_count = 0
                # Save stepping stone on improvement
                if spec.stepping_stones and git_commit and metric_value is not None:
                    cumulative_cost = sum(
                        it.cost_usd for it in state.iterations if it.cost_usd
                    ) + (iteration_cost or 0.0)
                    _stone = _save_stepping_stone(
                        plan_path=plan_path,
                        plan_name=plan.name,
                        metric_value=metric_value,
                        metric_name=spec.metric,
                        iteration=iteration_number,
                        git_commit=git_commit,
                        lessons_path=lessons_path,
                        watch_run_path=str(watch_run_path),
                        total_cost_usd=cumulative_cost,
                    )
                    if _stone is not None:
                        _emit(
                            event_callback,
                            "stepping_stone_saved",
                            iteration=iteration_number,
                            metric_value=metric_value,
                            plan_hash=_stone.plan_hash,
                        )
            else:
                action = spec.on_regression
                state.plateau_count += 1
                _emit(
                    event_callback,
                    "regression_detected",
                    iteration=iteration_number,
                    metric_value=metric_value,
                    best_metric=state.best_metric,
                    action=action,
                )
                rollback_ok = _git_rollback(workdir, spec.on_regression)
                _emit(
                    event_callback,
                    "rollback_executed",
                    iteration=iteration_number,
                    on_regression=spec.on_regression,
                    success=rollback_ok,
                )
                if not rollback_ok:
                    error = f"git {spec.on_regression} failed"
                if state.plateau_count >= spec.plateau_threshold:
                    _emit(
                        event_callback,
                        "plateau_detected",
                        iteration=iteration_number,
                        plateau_count=state.plateau_count,
                        plateau_threshold=spec.plateau_threshold,
                        action=spec.plateau_action,
                    )
                # Reload plan after rollback (file was reverted)
                try:
                    plan = load_plan(plan_path)
                except Exception:
                    pass

            iteration_excerpts = _capture_iteration_excerpts(
                target_result.run_path,
                improve_log_text=improve_log_text,
                consolidated_summary=consolidated_summary,
            )
            watch_iteration = WatchIteration(
                iteration=iteration_number,
                metric_value=metric_value,
                best_metric=state.best_metric,
                improved=improved,
                action=action,
                cost_usd=total_iter_cost,
                duration_sec=duration_sec,
                git_commit=git_commit,
                error=error,
                timestamp=datetime.now().isoformat(),
                **iteration_excerpts,
            )
            state.iterations.append(watch_iteration)
            state.total_iterations = len(state.iterations)
            _write_experiment(experiments_path, watch_iteration)

            # Extract and persist lesson from this iteration
            lesson = _extract_lesson(watch_iteration, manifest_summary, improve_log_text)
            if lesson is not None:
                _write_lesson(lessons_path, lesson)

            _maybe_extract_session_memory(
                plan_name=plan.name,
                source_dir=plan.source_dir,
                watch_run_path=watch_run_path,
                iterations=state.iterations,
                lessons=_load_lessons(lessons_path),
                metric_name=spec.metric,
                metric_direction=spec.metric_direction,
                plateau_count=state.plateau_count,
                plateau_threshold=spec.plateau_threshold,
                consolidate_model=spec.consolidate_model,
                consolidated_summary=consolidated_summary,
            )

            # Record improve agent cost in budget ledger
            if plan.budget_period and iteration_cost and iteration_cost > 0:
                from .budget import record_cost as _record_cost, _DEFAULT_LEDGER_PATH
                _ledger = plan.source_dir / _DEFAULT_LEDGER_PATH
                _record_cost(_ledger, plan.name, f"improve-iter-{iteration_number}", iteration_cost)

            _emit(
                event_callback,
                "iteration_complete",
                iteration=iteration_number,
                metric_value=metric_value,
                best_metric=state.best_metric,
                improved=improved,
                action=action,
                cost_usd=total_iter_cost,
                duration_sec=duration_sec,
            )
            ok = int(metric_value) if metric_value is not None else "?"
            total = len(plan.tasks)
            print(
                f"[maestro] watch improve: iteration {iteration_number}/{spec.max_iterations}, "
                f"tasks={ok}/{total}, best={state.best_metric} [{action}]"
            )

            if _target_reached(metric_value, spec):
                state.status = "target_reached"
                _emit(
                    event_callback,
                    "target_reached",
                    iteration=iteration_number,
                    metric_value=metric_value,
                    target_metric=spec.target_metric,
                )
                break

        else:
            state.status = "max_iterations"

    except KeyboardInterrupt:
        state.status = "interrupted"

    _emit(
        event_callback,
        "watch_complete",
        status=state.status,
        best_metric=state.best_metric,
        best_iteration=state.best_iteration,
        total_iterations=state.total_iterations,
        total_cost_usd=state.total_cost_usd,
    )
    return state


def watch(
    plan_path: str | Path,
    *,
    max_parallel_override: int | None = None,
    event_callback: Callable[[str, dict[str, object]], None] | None = None,
    cancel_event: threading.Event | None = None,
    dry_run: bool = False,
    execution_profile: ExecutionProfile = "plan",
    verbosity: Verbosity = "normal",
    output_mode: OutputMode = "text",
    cache_dir: str | Path | None = None,
    auto_approve: bool = False,
    resume_from: Path | None = None,
    extra_template_vars: dict[str, str] | None = None,
) -> WatchState:
    del cache_dir

    current_plan_path = Path(plan_path).resolve()
    plan: PlanSpec = load_plan(current_plan_path)
    if plan.watch is None:
        raise ValueError("Plan does not define a watch block")

    spec = plan.watch

    # Dispatch to improve mode if configured
    if spec.mode == "improve":
        return _watch_improve(
            plan_path=current_plan_path,
            plan=plan,
            spec=spec,
            max_parallel_override=max_parallel_override,
            event_callback=event_callback,
            cancel_event=cancel_event,
            dry_run=dry_run,
            execution_profile=execution_profile,
            verbosity=verbosity,
            output_mode=output_mode,
            auto_approve=auto_approve,
            resume_from=resume_from,
            extra_template_vars=extra_template_vars,
        )

    workdir = resolve_path(plan.source_dir, plan.workspace_root) or plan.source_dir.resolve()
    program_text = _load_program(plan, spec)
    watch_root = resolve_path(plan.source_dir, plan.run_dir)
    if watch_root is None:
        raise ValueError("Unable to resolve run directory")

    if resume_from is not None:
        watch_run_path = Path(resume_from).resolve()
        state = _resume_watch_state(watch_run_path)
        state.plan_path = str(current_plan_path)
    else:
        watch_run_path = (
            watch_root
            / f"watch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sanitize_dirname(plan.name)}"
        ).resolve()
        watch_run_path.mkdir(parents=True, exist_ok=True)
        state = WatchState(plan_path=str(current_plan_path), status="max_iterations")

    experiments_path = watch_run_path / "experiments.jsonl"
    _emit(
        event_callback,
        "watch_start",
        plan_name=plan.name,
        max_iterations=spec.max_iterations,
        metric=spec.metric,
        metric_direction=spec.metric_direction,
    )

    print(f"[maestro] watch: run dir {watch_run_path}")

    # Resolve blame_plan for {{ watch.blame }} / {{ watch.manifest }} injection
    blame_plan_spec: PlanSpec | None = None
    if spec.blame_plan:
        blame_plan_path = (plan.source_dir / spec.blame_plan).resolve()
        try:
            blame_plan_spec = load_plan(blame_plan_path)
        except Exception:
            pass

    if dry_run:
        _emit(
            event_callback,
            "watch_complete",
            status=state.status,
            best_metric=state.best_metric,
            best_iteration=state.best_iteration,
            total_iterations=state.total_iterations,
            total_cost_usd=state.total_cost_usd,
        )
        return state

    consolidated_summary: str = ""
    try:
        start_iteration = state.total_iterations + 1
        for iteration_number in range(start_iteration, spec.max_iterations + 1):
            if cancel_event is not None and cancel_event.is_set():
                state.status = "interrupted"
                break

            if spec.max_cost_usd is not None and state.total_cost_usd >= spec.max_cost_usd:
                state.status = "budget_exceeded"
                break

            if state.plateau_count >= spec.plateau_threshold:
                state.status = "plateau"
                break

            if spec.max_total_steps is not None and state.total_steps >= spec.max_total_steps:
                state.status = "step_limit_reached"
                _emit(
                    event_callback,
                    "watch_step_limit",
                    total_steps=state.total_steps,
                    max_total_steps=spec.max_total_steps,
                )
                break

            _emit(
                event_callback,
                "iteration_start",
                iteration=iteration_number,
                best_metric=state.best_metric,
            )

            # Consolidation agent: run every N iterations when enabled
            if (
                spec.consolidate_model is not None
                and state.iterations
                and state.total_iterations % spec.consolidate_every == 0
            ):
                consolidated_summary = _run_consolidation(
                    spec,
                    _build_history_text(state.iterations),
                    workdir,
                    plan=plan,
                )

            iteration_started = time.monotonic()
            last_metric = state.iterations[-1].metric_value if state.iterations else None
            # Build blame context from the target plan's most recent run
            blame_json = ""
            manifest_summary = ""
            if blame_plan_spec is not None:
                target_run = _find_latest_target_run(
                    blame_plan_spec.source_dir,
                    blame_plan_spec.run_dir or ".maestro-runs",
                    blame_plan_spec.name,
                )
                if target_run is not None:
                    blame_json, manifest_summary = _build_blame_context(target_run)

            extra_vars: dict[str, str] = dict(extra_template_vars) if extra_template_vars else {}
            extra_vars.update({
                "watch.iteration": str(iteration_number),
                "watch.best_metric": "" if state.best_metric is None else str(state.best_metric),
                "watch.last_metric": "" if last_metric is None else str(last_metric),
                "watch.history": _build_history_text(state.iterations),
                "watch.experiments_summary": _build_experiments_summary(
                    state.iterations,
                    plateau_count=state.plateau_count,
                    plateau_threshold=spec.plateau_threshold,
                ),
                "watch.program": program_text,
                "watch.consolidated": consolidated_summary,
                "watch.blame": blame_json,
                "watch.manifest": manifest_summary,
            })

            result = run_plan(
                plan,
                dry_run=dry_run,
                execution_profile=execution_profile,
                max_parallel_override=max_parallel_override,
                verbosity=verbosity,
                output_mode=output_mode,
                auto_approve=auto_approve,
                event_callback=event_callback,
                cancel_event=cancel_event,
                extra_template_vars=extra_vars,
            )

            duration_sec = time.monotonic() - iteration_started
            cost_usd = result.total_cost_usd
            if cost_usd is not None:
                state.total_cost_usd += cost_usd
            state.total_steps += _count_executed_tasks(result)

            metric_value = _extract_metric(result, spec, plan, result.run_path)
            warmup = iteration_number <= spec.warmup_iterations
            improved = warmup or _is_improvement(metric_value, state.best_metric, spec)

            if metric_value is not None:
                _emit(
                    event_callback,
                    "metric_recorded",
                    iteration=iteration_number,
                    metric=spec.metric,
                    value=metric_value,
                    best=state.best_metric,
                    improved=improved,
                )

            action = "keep"
            git_commit: str | None = None
            error: str | None = None

            if improved:
                action = "warmup_keep" if warmup else "keep"
                git_commit = _git_commit_changes(workdir, iteration_number, spec.metric, metric_value or 0.0)
                state.best_metric = metric_value
                state.best_iteration = iteration_number
                state.plateau_count = 0
            else:
                action = spec.on_regression
                state.plateau_count += 1
                _emit(
                    event_callback,
                    "regression_detected",
                    iteration=iteration_number,
                    metric_value=metric_value,
                    best_metric=state.best_metric,
                    action=action,
                )
                rollback_ok = _git_rollback(workdir, spec.on_regression)
                _emit(
                    event_callback,
                    "rollback_executed",
                    iteration=iteration_number,
                    on_regression=spec.on_regression,
                    success=rollback_ok,
                )
                if not rollback_ok:
                    error = f"git {spec.on_regression} failed"
                if state.plateau_count >= spec.plateau_threshold:
                    _emit(
                        event_callback,
                        "plateau_detected",
                        iteration=iteration_number,
                        plateau_count=state.plateau_count,
                        plateau_threshold=spec.plateau_threshold,
                        action=spec.plateau_action,
                    )

            iteration_excerpts = _capture_iteration_excerpts(
                result.run_path,
                consolidated_summary=consolidated_summary,
            )
            watch_iteration = WatchIteration(
                iteration=iteration_number,
                metric_value=metric_value,
                best_metric=state.best_metric,
                improved=improved,
                action=action,
                cost_usd=cost_usd,
                duration_sec=duration_sec,
                git_commit=git_commit,
                error=error,
                timestamp=datetime.now().isoformat(),
                **iteration_excerpts,
            )
            state.iterations.append(watch_iteration)
            state.total_iterations = len(state.iterations)
            _write_experiment(experiments_path, watch_iteration)

            _emit(
                event_callback,
                "iteration_complete",
                iteration=iteration_number,
                metric_value=metric_value,
                best_metric=state.best_metric,
                improved=improved,
                action=action,
                cost_usd=cost_usd,
                duration_sec=duration_sec,
            )
            print(
                f"[maestro] watch: iteration {iteration_number}/{spec.max_iterations}, "
                f"metric={metric_value}, best={state.best_metric} [{action}]"
            )

            if _target_reached(metric_value, spec):
                state.status = "target_reached"
                _emit(
                    event_callback,
                    "target_reached",
                    iteration=iteration_number,
                    metric_value=metric_value,
                    target_metric=spec.target_metric,
                )
                break

        else:
            state.status = "max_iterations"

    except KeyboardInterrupt:
        state.status = "interrupted"

    _emit(
        event_callback,
        "watch_complete",
        status=state.status,
        best_metric=state.best_metric,
        best_iteration=state.best_iteration,
        total_iterations=state.total_iterations,
        total_cost_usd=state.total_cost_usd,
    )
    return state


def _extract_metric(
    result: PlanRunResult,
    spec: WatchSpec,
    plan: PlanSpec,
    run_path: Path,
) -> float | None:
    if not plan.tasks:
        return None

    # Manifest source works on the full PlanRunResult, not a single task
    if spec.metric_source == "manifest":
        try:
            return _extract_manifest_metric(result)
        except (OSError, ValueError, TypeError):
            return None

    target_task_id = spec.metric_task or plan.tasks[-1].id
    task_result = result.task_results.get(target_task_id)
    if task_result is None:
        return None

    try:
        if spec.metric_source == "stdout_regex":
            if not spec.metric_pattern:
                return None
            match = re.search(spec.metric_pattern, task_result.stdout_tail or "")
            if match is None:
                return None
            return float(match.group(1))

        if spec.metric_source == "verify_command":
            if not task_result.log_path.exists() or not spec.metric_pattern:
                return None
            section_text = _extract_log_section(task_result.log_path, "[verify_command]")
            if section_text is None:
                return None
            match = re.search(spec.metric_pattern, section_text)
            if match is None:
                return None
            return float(match.group(1))

        if spec.metric_source == "guard_command":
            if not task_result.log_path.exists() or not spec.metric_pattern:
                return None
            section_text = _extract_log_section(task_result.log_path, "[guard_command]")
            if section_text is None:
                return None
            match = re.search(spec.metric_pattern, section_text)
            if match is None:
                return None
            return float(match.group(1))

        if spec.metric_source == "json_field":
            if not spec.metric_json_path:
                return None
            payload_path = run_path / f"{target_task_id}.result.json"
            if not payload_path.exists():
                return None
            payload = json.loads(payload_path.read_text(encoding="utf-8"))
            value = _lookup_json_path(payload, spec.metric_json_path)
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                return float(value)
            return None

    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None

    return None


def _extract_manifest_metric(result: PlanRunResult) -> float | None:
    """Count tasks with ``success`` or ``dry_run`` status from a run result."""
    if not result.task_results:
        return None
    ok_count = sum(
        1 for r in result.task_results.values()
        if r.status in {"success", "dry_run"}
    )
    return float(ok_count)


def _find_latest_target_run(
    plan_source_dir: Path,
    plan_run_dir: str,
    plan_name: str,
) -> Path | None:
    """Find the most recent run directory for a target plan.

    Scans the plan's run_dir for directories ending with the sanitized plan
    name, sorted by name (timestamp prefix) descending.
    """
    run_root = plan_source_dir / plan_run_dir
    if not run_root.is_dir():
        return None
    suffix = f"_{sanitize_dirname(plan_name)}"
    candidates = sorted(
        (d for d in run_root.iterdir() if d.is_dir() and d.name.endswith(suffix)),
        key=lambda d: d.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def _build_blame_context(run_path: Path) -> tuple[str, str]:
    """Build blame JSON and compact manifest summary for a target plan run.

    Returns (blame_json, manifest_summary).
    """
    blame_json = ""
    manifest_summary = ""
    try:
        chain = blame_run(run_path)
        blame_json = json.dumps(chain.to_dict(), indent=2)
    except Exception:
        pass

    manifest_path = run_path / "run_manifest.json"
    if manifest_path.is_file():
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            tasks = raw.get("task_results", {})
            lines: list[str] = []
            for tid, tdata in tasks.items():
                status = tdata.get("status", "?")
                exit_code = tdata.get("exit_code", "?")
                duration = tdata.get("duration_sec", "?")
                msg = tdata.get("message", "")
                if msg and len(msg) > 120:
                    msg = msg[:120] + "..."
                line = f"  {tid}: {status} (exit={exit_code}, {duration}s)"
                if msg:
                    line += f" — {msg}"
                lines.append(line)
            manifest_summary = "\n".join(lines)
        except (OSError, json.JSONDecodeError, TypeError):
            pass

    return blame_json, manifest_summary


def _target_reached(current: float | None, spec: WatchSpec) -> bool:
    """Check whether the metric has reached the configured target."""
    if current is None or spec.target_metric is None:
        return False
    if spec.metric_direction == "lower_is_better":
        return current <= spec.target_metric
    return current >= spec.target_metric


def _is_improvement(current: float | None, best: float | None, spec: WatchSpec) -> bool:
    if current is None:
        return False
    if best is None:
        return True
    if spec.metric_direction == "lower_is_better":
        return current < best
    return current > best


def _git_commit_changes(workdir: Path, iteration: int, metric_name: str, metric_value: float) -> str | None:
    try:
        add_result = subprocess.run(
            ["git", "add", "-A"],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        if add_result.returncode != 0:
            return None

        commit_result = subprocess.run(
            ["git", "commit", "-m", f"watch: iteration {iteration}, {metric_name}={metric_value}"],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        if commit_result.returncode != 0:
            return None

        rev_parse = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workdir,
            capture_output=True,
            text=True,
        )
        if rev_parse.returncode != 0:
            return None
        sha = rev_parse.stdout.strip()
        return sha or None
    except OSError:
        return None


def _git_rollback(workdir: Path, on_regression: OnRegression) -> bool:
    if on_regression == "keep":
        return True

    # Regressions are not committed, so rollback should discard the failed
    # iteration's worktree changes while preserving the last good commit.
    command = ["git", "reset", "--hard", "HEAD"]
    if on_regression == "revert":
        command = ["git", "revert", "--no-edit", "HEAD"]

    try:
        completed = subprocess.run(
            command,
            cwd=workdir,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False

    return completed.returncode == 0


def _load_program(plan: PlanSpec, spec: WatchSpec) -> str:
    if not spec.program_md:
        return ""
    program_path = plan.source_dir / spec.program_md
    try:
        return program_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _build_history_text(iterations: list[WatchIteration]) -> str:
    if not iterations:
        return ""

    recent = iterations[-_HISTORY_LIMIT:]
    lines = [
        "iter | metric | best | action",
        "-----|--------|------|-------",
    ]
    for item in recent:
        metric_text = "-" if item.metric_value is None else f"{item.metric_value:g}"
        best_text = "-" if item.best_metric is None else f"{item.best_metric:g}"
        lines.append(f"{item.iteration:>4} | {metric_text} | {best_text} | {item.action}")
    return "\n".join(lines)


def _build_experiments_summary(
    iterations: list[WatchIteration],
    plateau_count: int = 0,
    plateau_threshold: int = 3,
) -> str:
    """Build a semantic analysis of experiments for the adaptive improve prompt.

    Unlike ``_build_history_text`` (a raw data table), this function produces
    an analytical summary of what was tried and what worked/failed, giving the
    improve agent strategic guidance when stuck.
    """
    if not iterations:
        return ""

    recent = iterations[-_HISTORY_LIMIT:]
    successes = [it for it in recent if it.improved]
    failures = [it for it in recent if not it.improved and it.action not in ("baseline", "")]
    regressions = [it for it in recent if it.action in ("rollback", "revert")]

    lines: list[str] = ["## Experiment Analysis"]
    lines.append(f"- Iterations run: {len(iterations)}")
    lines.append(f"- Improvements: {len(successes)} / Failures: {len(failures)} / Regressions: {len(regressions)}")
    if iterations[-1].best_metric is not None:
        lines.append(f"- Best metric so far: {iterations[-1].best_metric:g}")

    # Successful approaches
    if successes:
        lines.append("\n### Approaches that WORKED:")
        for it in successes[-5:]:
            metric_text = f"{it.metric_value:g}" if it.metric_value is not None else "?"
            lines.append(f"- Iteration {it.iteration}: {it.action} → metric {metric_text}")

    # Failed approaches — help the agent avoid repeating them
    if failures:
        lines.append("\n### Approaches that FAILED (do NOT repeat):")
        for it in failures[-5:]:
            metric_text = f"{it.metric_value:g}" if it.metric_value is not None else "?"
            error_hint = f" — {it.error[:80]}" if it.error else ""
            lines.append(f"- Iteration {it.iteration}: {it.action} → metric {metric_text}{error_hint}")

    # Plateau pressure — escalating urgency
    if plateau_count >= 2:
        remaining = max(0, plateau_threshold - plateau_count)
        lines.append("\n### ⚠ Plateau Alert")
        lines.append(f"- Stuck for {plateau_count} iterations without improvement.")
        if remaining <= 1:
            lines.append(
                "- CRITICAL: This is the last chance before the watch loop stops. "
                "Try a fundamentally different approach — change strategy, not parameters."
            )
        else:
            lines.append(
                f"- {remaining} iteration(s) remaining before stop. "
                "Try a DIFFERENT category of fix than what was already attempted."
            )

    return "\n".join(lines)


_STEPPING_STONES_MAX = 20


def _stepping_stones_dir(source_dir: Path | str, plan_name: str) -> Path:
    """Return the stepping stones directory for a given plan."""
    return Path(source_dir) / ".maestro-cache" / "stepping" / sanitize_dirname(plan_name)


def _save_stepping_stone(
    plan_path: Path,
    plan_name: str,
    metric_value: float,
    metric_name: str,
    iteration: int,
    git_commit: str | None,
    lessons_path: Path | None,
    watch_run_path: str,
    total_cost_usd: float,
    *,
    archive_source_dir: Path | None = None,
    lessons: list[dict[str, Any]] | None = None,
    source_type: str = "watch",
    metadata: dict[str, Any] | None = None,
) -> SteppingStone | None:
    """Save a stepping stone snapshot after a successful improvement."""
    try:
        plan_yaml = plan_path.read_text(encoding="utf-8")
    except OSError:
        return None

    plan_hash = hashlib.sha256(plan_yaml.encode("utf-8")).hexdigest()[:16]

    resolved_lessons = list(lessons or [])
    if not resolved_lessons and lessons_path is not None and lessons_path.exists():
        try:
            for line in lessons_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        resolved_lessons.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        except OSError:
            pass

    stone = SteppingStone(
        plan_name=plan_name,
        plan_hash=plan_hash,
        metric_value=metric_value,
        metric_name=metric_name,
        iteration=iteration,
        git_commit=git_commit,
        plan_yaml=plan_yaml,
        lessons=resolved_lessons,
        timestamp=datetime.now().isoformat(),
        watch_run_path=watch_run_path,
        total_cost_usd=total_cost_usd,
        source_type=source_type,
        metadata=dict(metadata or {}),
    )

    stones_dir = _stepping_stones_dir(archive_source_dir or plan_path.parent, plan_name)
    stones_dir.mkdir(parents=True, exist_ok=True)
    stones_path = stones_dir / "stones.jsonl"
    with stones_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stone.to_dict()) + "\n")
        fh.flush()

    # Compact if over the limit — keep best stones only
    _compact_stepping_stones(stones_path, metric_name)

    return stone


def _compact_stepping_stones(
    stones_path: Path,
    metric_name: str,
) -> None:
    """Keep only the top N stepping stones for the given metric."""
    if not stones_path.exists():
        return
    try:
        lines = stones_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= _STEPPING_STONES_MAX:
        return

    stones: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            stones.append(json.loads(line))
        except json.JSONDecodeError:
            continue

    matching = [
        stone
        for stone in stones
        if stone.get("metric_name", metric_name) == metric_name
    ]
    if len(matching) <= _STEPPING_STONES_MAX:
        return
    others = [
        stone
        for stone in stones
        if stone.get("metric_name", metric_name) != metric_name
    ]
    matching.sort(key=lambda s: s.get("metric_value", 0.0), reverse=True)
    kept = matching[:_STEPPING_STONES_MAX]
    final_stones = others + kept

    try:
        stones_path.write_text(
            "\n".join(json.dumps(s) for s in final_stones) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass


def _load_best_stepping_stone(
    plan_name: str,
    source_dir: Path | str,
    metric_name: str,
    higher_is_better: bool = True,
) -> SteppingStone | None:
    """Load the best stepping stone for a plan, respecting metric direction."""
    stones_path = _stepping_stones_dir(source_dir, plan_name) / "stones.jsonl"
    if not stones_path.exists():
        return None

    best: dict[str, Any] | None = None
    try:
        for line in stones_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            if data.get("metric_name", metric_name) != metric_name:
                continue
            if best is None:
                best = data
            else:
                val = data.get("metric_value", 0.0)
                best_val = best.get("metric_value", 0.0)
                if (higher_is_better and val > best_val) or (
                    not higher_is_better and val < best_val
                ):
                    best = data
    except OSError:
        return None

    if best is None:
        return None

    return SteppingStone(
        plan_name=best.get("plan_name", plan_name),
        plan_hash=best.get("plan_hash", ""),
        metric_value=best.get("metric_value", 0.0),
        metric_name=best.get("metric_name", metric_name),
        iteration=best.get("iteration", 0),
        git_commit=best.get("git_commit"),
        plan_yaml=best.get("plan_yaml", ""),
        lessons=best.get("lessons", []),
        timestamp=best.get("timestamp", ""),
        watch_run_path=best.get("watch_run_path", ""),
        total_cost_usd=best.get("total_cost_usd", 0.0),
        source_type=str(best.get("source_type", "watch") or "watch"),
        metadata=best.get("metadata", {}) if isinstance(best.get("metadata"), dict) else {},
    )


def _apply_stepping_stone(
    stone: SteppingStone,
    plan_path: Path,
) -> bool:
    """Apply a stepping stone by writing its plan YAML to disk.

    Returns True if the stone was applied successfully and the resulting
    plan validates. Returns False otherwise (plan left untouched).
    """
    if not stone.plan_yaml:
        return False

    # Backup current plan
    backup = plan_path.read_text(encoding="utf-8") if plan_path.exists() else ""

    try:
        plan_path.write_text(stone.plan_yaml, encoding="utf-8")
        # Validate the restored plan
        load_plan(plan_path)
        return True
    except Exception:
        # Restore backup on validation failure
        if backup:
            try:
                plan_path.write_text(backup, encoding="utf-8")
            except OSError:
                pass
        return False


def _write_experiment(experiments_path: Path, iteration: WatchIteration) -> None:
    experiments_path.parent.mkdir(parents=True, exist_ok=True)
    with experiments_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(iteration.to_dict()) + "\n")


def _resume_watch_state(run_dir: Path) -> WatchState:
    experiments_path = run_dir / "experiments.jsonl"
    state = WatchState(plan_path="", status="max_iterations")
    if not experiments_path.exists():
        return state

    with experiments_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            iteration = WatchIteration(
                iteration=int(payload.get("iteration", 0)),
                metric_value=_coerce_float(payload.get("metric_value")),
                best_metric=_coerce_float(payload.get("best_metric")),
                improved=bool(payload.get("improved", False)),
                action=str(payload.get("action", "")),
                cost_usd=_coerce_float(payload.get("cost_usd")),
                duration_sec=_coerce_float(payload.get("duration_sec")) or 0.0,
                git_commit=_coerce_str(payload.get("git_commit")),
                error=_coerce_str(payload.get("error")),
                timestamp=_coerce_str(payload.get("timestamp")) or "",
                fix_summary=_coerce_str(payload.get("fix_summary")),
                manifest_excerpt=_coerce_str(payload.get("manifest_excerpt")),
                blame_excerpt=_coerce_str(payload.get("blame_excerpt")),
                consolidated_excerpt=_coerce_str(payload.get("consolidated_excerpt")),
            )
            state.iterations.append(iteration)
            state.best_metric = iteration.best_metric
            if iteration.improved:
                state.best_iteration = iteration.iteration
                state.plateau_count = 0
            else:
                state.plateau_count += 1
            if iteration.cost_usd is not None:
                state.total_cost_usd += iteration.cost_usd

    state.total_iterations = len(state.iterations)
    return state


def _run_consolidation(
    watch_spec: WatchSpec,
    history_text: str,
    workspace_root: Path,
    plan: PlanSpec | None = None,
) -> str:
    """Run LLM consolidation of experiment history via ``claude --print``.

    Applies the same firewall policies as tool outputs:
    pass-1 deterministic injection stripping, optional pass-2
    model-based classification (when ``plan.firewall_model`` is set),
    and instructionality scoring with quarantine on high scores.
    """
    from .runners import _strip_injection_patterns
    from .memory import compute_instructionality

    model = watch_spec.consolidate_model or "haiku"
    prompt = watch_spec.consolidate_prompt or _DEFAULT_CONSOLIDATION_PROMPT
    full_prompt = f"{prompt}\n\n## Experiment History\n\n{history_text}"
    cmd = ["claude", "--print", "--model", model, full_prompt]
    raw_output = ""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            cwd=workspace_root,
        )
        if result.returncode == 0 and result.stdout.strip():
            raw_output = result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass

    if not raw_output:
        return ""

    # Pass-1: deterministic injection pattern stripping
    sanitized = _strip_injection_patterns(raw_output)

    # Pass-2: model-based classification (opt-in via plan.firewall_model)
    firewall_model = getattr(plan, "firewall_model", None) if plan else None
    if firewall_model and sanitized:
        try:
            from .runners import _run_firewall_pass2
            decision = _run_firewall_pass2(
                firewall_model, "consolidation_output", sanitized,
                workdir=workspace_root,
            )
            if decision.verdict == "block":
                return ""
        except Exception:
            pass  # fail-open to pass-1 sanitized output

    # Instructionality check
    iscore = compute_instructionality(sanitized)
    if iscore >= 0.4:
        # High instructionality — strip again more aggressively
        sanitized = _strip_injection_patterns(sanitized)

    return sanitized


def _count_executed_tasks(result: PlanRunResult) -> int:
    """Count tasks that actually executed (not skipped)."""
    return sum(
        1 for tr in result.task_results.values()
        if tr.status != "skipped"
    )


def _emit(
    event_callback: Callable[[str, dict[str, object]], None] | None,
    event_type: str,
    **kwargs: object,
) -> None:
    if event_callback:
        event_callback(event_type, kwargs)


def _extract_log_section(log_path: Path, section_name: str) -> str | None:
    try:
        text = log_path.read_text(encoding="utf-8")
    except OSError:
        return None

    lines = text.splitlines()
    start_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == section_name:
            start_index = index + 1
            break
    if start_index is None:
        return None

    buffer: list[str] = []
    for line in lines[start_index:]:
        stripped = line.strip()
        if stripped in _LOG_SECTION_HEADERS and stripped != section_name:
            break
        if stripped.startswith(("status=", "message=")):
            break
        buffer.append(line)
    return "\n".join(buffer).strip() or None


def _lookup_json_path(payload: object, path: str) -> object | None:
    current = payload
    tokens = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    for token in tokens:
        if token.startswith("[") and token.endswith("]"):
            if not isinstance(current, list):
                return None
            index = int(token[1:-1])
            if index >= len(current):
                return None
            current = current[index]
            continue
        if not isinstance(current, dict):
            return None
        if token not in current:
            return None
        current = current[token]
    return current


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None
