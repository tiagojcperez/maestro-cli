from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from .models import (
    PlanSpec, Suggestion, PlanSuggestions, SuggestionCategory,
)


_FAILURE_REMEDIATION: dict[str, str] = {
    "dependency_missing": (
        "Pre-flight check: add pre_command to verify required tools, "
        "or add maestro doctor check"
    ),
    "output_format_error": (
        "Add guard_command to validate output format, "
        "or use judge type: json-schema"
    ),
    "cascading_failure": (
        "Add context_budget_tokens to limit upstream context propagation, "
        "or add guard_command on producer task"
    ),
    "deadlock": (
        "Reduce timeout_sec, ensure approval gates have "
        "--auto-approve fallback for CI"
    ),
    "miscommunication": (
        "Improve prompt clarity with description field, "
        "use context_mode: summarized for cleaner upstream context"
    ),
    "role_confusion": (
        "Scope agent role more narrowly, "
        "add verify_command to enforce output boundaries"
    ),
    "verification_gap": (
        "Strengthen verify_command assertions, "
        "add guard_command as secondary check"
    ),
}


def _safe_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _safe_int(value: object) -> int:
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return 0


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _downgrade_model(model: str | None) -> str | None:
    if not model:
        return None
    lowered = model.lower()
    if "opus" in lowered:
        return "sonnet"
    if "sonnet" in lowered:
        return "haiku"
    return None


def _upgrade_model(model: str | None) -> str | None:
    if not model:
        return None
    lowered = model.lower()
    if "haiku" in lowered:
        return "sonnet"
    if "sonnet" in lowered:
        return "opus"
    return None


def _load_run_history(plan_name: str, run_dir: Path, min_runs: int) -> list[dict[str, Any]]:
    if not run_dir.exists() or not run_dir.is_dir():
        return []

    loaded: list[tuple[str, dict[str, Any]]] = []
    for candidate in run_dir.glob(f"*_{plan_name}"):
        if not candidate.is_dir():
            continue
        manifest_path = candidate / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            loaded.append((candidate.name, payload))

    loaded.sort(key=lambda item: item[0])
    manifests = [manifest for _, manifest in loaded]
    if len(manifests) < min_runs:
        return []
    return manifests


def _analyze_task(task_id: str, task_spec_map: dict[str, Any], runs: list[dict[str, Any]]) -> list[Suggestion]:
    suggestions: list[Suggestion] = []
    if not runs:
        return suggestions

    task_runs: list[dict[str, Any]] = []
    for run in runs:
        task_results = run.get("task_results")
        if not isinstance(task_results, dict):
            continue
        task_payload = task_results.get(task_id)
        if isinstance(task_payload, dict):
            task_runs.append(task_payload)

    if not task_runs:
        return suggestions

    task_spec = task_spec_map.get(task_id)
    model = getattr(task_spec, "model", None)
    max_retries = _safe_int(getattr(task_spec, "max_retries", 0))

    total = len(task_runs)
    success_count = sum(1 for r in task_runs if r.get("status") == "success")
    retry_runs = sum(1 for r in task_runs if _safe_int(r.get("retry_count")) > 0)
    success_rate = success_count / total
    retry_rate = retry_runs / total

    if success_count == total and retry_runs == 0:
        next_model = _downgrade_model(model)
        if next_model:
            suggestions.append(
                Suggestion(
                    task_id=task_id,
                    category="downgrade_model",
                    severity="high",
                    reason=(
                        f"Task passes in {total}/{total} runs with 0 retries; "
                        "downgrading model should preserve quality while reducing cost."
                    ),
                    current_value=str(model),
                    suggested_value=next_model,
                    confidence=success_rate,
                    estimated_savings_pct=15.0,
                )
            )

    judge_scores: list[float] = []
    judge_verdicts: list[str] = []
    for result in task_runs:
        judge = result.get("judge_result")
        if not isinstance(judge, dict):
            continue
        score = judge.get("overall_score")
        verdict = judge.get("verdict")
        if isinstance(score, (int, float)):
            judge_scores.append(float(score))
        if isinstance(verdict, str):
            judge_verdicts.append(verdict)

    if (
        judge_scores
        and len(judge_scores) == total
        and len(judge_verdicts) == total
        and all(v == "pass" for v in judge_verdicts)
    ):
        avg_score = sum(judge_scores) / len(judge_scores)
        if avg_score > 0.9:
            suggestions.append(
                Suggestion(
                    task_id=task_id,
                    category="remove_judge",
                    severity="low",
                    reason=(
                        "Judge always passes with very high scores; "
                        "consider removing judge checks for this task."
                    ),
                    current_value="judge=enabled",
                    suggested_value="judge=disabled",
                    confidence=min(judge_scores),
                    estimated_savings_pct=5.0,
                )
            )

    if retry_rate > 0.5:
        stronger_model = _upgrade_model(model)
        if stronger_model:
            category: SuggestionCategory = "upgrade_model"
            current_value = str(model)
            suggested_value = stronger_model
            reason = (
                f"Retries happen in {retry_runs}/{total} runs (>50%); "
                "a stronger model should reduce retries."
            )
        else:
            category = "add_retry"
            current_value = f"max_retries={max_retries}"
            suggested_value = f"max_retries={max(max_retries + 1, 1)}"
            reason = (
                f"Retries happen in {retry_runs}/{total} runs (>50%); "
                "increase max_retries for better completion reliability."
            )

        suggestions.append(
            Suggestion(
                task_id=task_id,
                category=category,
                severity="medium",
                reason=reason,
                current_value=current_value,
                suggested_value=suggested_value,
                confidence=retry_rate,
                estimated_savings_pct=8.0,
            )
        )

    task_avg_cost = sum(_safe_float(r.get("cost_usd")) for r in task_runs) / total
    plan_avg_cost = 0.0
    if runs:
        plan_avg_cost = sum(_safe_float(run.get("total_cost_usd")) for run in runs) / len(runs)

    if plan_avg_cost > 0 and (task_avg_cost / plan_avg_cost) > 0.4:
        next_model = _downgrade_model(model)
        if next_model:
            category = "downgrade_model"
            current_value = str(model)
            suggested_value = next_model
        else:
            category = "add_review_task"
            current_value = f"avg_task_cost={task_avg_cost:.4f}"
            suggested_value = "split task into smaller subtasks"

        suggestions.append(
            Suggestion(
                task_id=task_id,
                category=category,
                severity="high",
                reason=(
                    f"Task consumes about {task_avg_cost / plan_avg_cost:.0%} of total plan cost; "
                    "split it or downgrade model to reduce spend."
                ),
                current_value=current_value,
                suggested_value=suggested_value,
                confidence=min(task_avg_cost / plan_avg_cost, 1.0),
                estimated_savings_pct=12.0,
            )
        )

    all_durations: list[float] = []
    for run in runs:
        task_results = run.get("task_results")
        if not isinstance(task_results, dict):
            continue
        for payload in task_results.values():
            if isinstance(payload, dict):
                all_durations.append(_safe_float(payload.get("duration_sec")))

    task_avg_duration = sum(_safe_float(r.get("duration_sec")) for r in task_runs) / total
    median_duration = _median(all_durations)
    if median_duration > 0 and task_avg_duration > (3 * median_duration):
        suggestions.append(
            Suggestion(
                task_id=task_id,
                category="add_checkpoint",
                severity="low",
                reason=(
                    "Average duration is over 3x median task duration; "
                    "add checkpointing or adjust timeout."
                ),
                current_value=f"avg_duration={task_avg_duration:.1f}s",
                suggested_value="enable checkpoint or tune timeout_sec",
                confidence=min(task_avg_duration / (3 * median_duration), 1.0),
                estimated_savings_pct=3.0,
            )
        )

    # Detect repeated timeout failures (exit_code == 124)
    timeout_count = sum(
        1 for r in task_runs
        if _safe_int(r.get("exit_code")) == 124
    )
    if timeout_count >= 2:
        timeout_rate = timeout_count / total
        max_duration = max(
            (_safe_float(r.get("duration_sec")) for r in task_runs),
            default=0.0,
        )
        task_timeout = getattr(task_spec, "timeout_sec", None) if task_spec else None
        current_str = f"timeout_sec={task_timeout}" if task_timeout else "timeout_sec=default"
        suggested_timeout = int(max_duration * 1.5) if max_duration > 0 else 600
        suggestions.append(
            Suggestion(
                task_id=task_id,
                category="tune_timeout",
                severity="high" if timeout_rate >= 0.5 else "medium",
                reason=(
                    f"Task timed out in {timeout_count}/{total} runs "
                    f"({timeout_rate:.0%}); max observed duration was "
                    f"{max_duration:.0f}s."
                ),
                current_value=current_str,
                suggested_value=f"timeout_sec={suggested_timeout}",
                confidence=timeout_rate,
                estimated_savings_pct=0.0,
            )
        )

    ratios: list[float] = []
    for result in task_runs:
        ratio = result.get("context_compression_ratio")
        if isinstance(ratio, (int, float)):
            ratios.append(float(ratio))

    if ratios and len(ratios) == total and all(r > 0.8 for r in ratios):
        suggestions.append(
            Suggestion(
                task_id=task_id,
                category="reduce_context_budget",
                severity="low",
                reason=(
                    "Context compression ratio stays above 0.8 across runs; "
                    "context budget appears over-provisioned."
                ),
                current_value="context_compression_ratio>0.8",
                suggested_value="reduce context_budget_tokens",
                confidence=min(ratios),
                estimated_savings_pct=5.0,
            )
        )

    category_counts: dict[str, int] = {}
    for result in task_runs:
        cat = result.get("failure_category")
        if isinstance(cat, str) and cat in _FAILURE_REMEDIATION:
            category_counts[cat] = category_counts.get(cat, 0) + 1

    for failure_cat, count in category_counts.items():
        occurrence_rate = count / total
        if occurrence_rate >= 0.3:
            suggestions.append(
                Suggestion(
                    task_id=task_id,
                    category="fix_failure_pattern",
                    severity="high" if occurrence_rate >= 0.5 else "medium",
                    reason=(
                        f"Failure category '{failure_cat}' observed in "
                        f"{count}/{total} runs ({occurrence_rate:.0%}). "
                        f"{_FAILURE_REMEDIATION[failure_cat]}."
                    ),
                    current_value=f"failure_category={failure_cat}",
                    suggested_value=_FAILURE_REMEDIATION[failure_cat],
                    confidence=occurrence_rate,
                    estimated_savings_pct=10.0,
                )
            )

    return suggestions


def suggest_plan(plan: PlanSpec, run_dir: Path, min_runs: int = 3) -> PlanSuggestions:
    runs = _load_run_history(plan.name, run_dir, min_runs)
    if not runs:
        return PlanSuggestions(
            plan_name=plan.name,
            runs_analyzed=0,
            suggestions=[],
            total_estimated_savings_pct=None,
        )

    task_spec_map = {task.id: task for task in plan.tasks}
    all_suggestions: list[Suggestion] = []
    for task in plan.tasks:
        all_suggestions.extend(_analyze_task(task.id, task_spec_map, runs))

    savings = [s.estimated_savings_pct for s in all_suggestions if s.estimated_savings_pct is not None]
    total_savings = min(sum(savings), 100.0) if savings else None

    return PlanSuggestions(
        plan_name=plan.name,
        runs_analyzed=len(runs),
        suggestions=all_suggestions,
        total_estimated_savings_pct=total_savings,
    )


def format_suggestions(result: PlanSuggestions) -> str:
    if not result.suggestions:
        return "[maestro] No suggestions — plan looks well-optimized"

    estimated = result.total_estimated_savings_pct
    estimated_text = "unknown" if estimated is None else f"{estimated:.1f}%"

    lines = [
        f"[maestro] Analyzed {result.runs_analyzed} runs of \"{result.plan_name}\"",
        "",
        f"Suggestions ({len(result.suggestions)} items, estimated -{estimated_text} cost):",
        "",
    ]

    for item in result.suggestions:
        lines.append(
            f"  [{item.severity.upper()}] task \"{item.task_id}\" -- category: "
            f"{item.current_value} → {item.suggested_value}"
        )
        lines.append(f"         Reason: [{item.category}] {item.reason}")
        if item.estimated_savings_pct is None:
            lines.append("         Est. savings: n/a")
        else:
            lines.append(f"         Est. savings: {item.estimated_savings_pct:.1f}%")
        lines.append("")

    return "\n".join(lines).rstrip()


def format_suggestions_json(result: PlanSuggestions) -> str:
    return json.dumps(result.to_dict(), indent=2)
