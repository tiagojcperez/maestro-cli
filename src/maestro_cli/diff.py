from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, cast

from .models import TaskStatus

_VALID_TASK_STATUSES: set[str] = {"success", "failed", "soft_failed", "skipped", "dry_run"}


@dataclass
class TaskDiff:
    task_id: str
    status_a: TaskStatus | None  # None if task doesn't exist in run A
    status_b: TaskStatus | None  # None if task doesn't exist in run B
    duration_a: float | None
    duration_b: float | None
    cost_a: float | None
    cost_b: float | None
    tokens_a: int | None
    tokens_b: int | None


@dataclass
class RunDiff:
    run_id_a: str
    run_id_b: str
    plan_name_a: str
    plan_name_b: str
    success_a: bool
    success_b: bool
    duration_a: float
    duration_b: float
    cost_a: float | None
    cost_b: float | None
    tokens_a: int | None
    tokens_b: int | None
    task_diffs: list[TaskDiff]
    added_tasks: list[str]  # in B but not A
    removed_tasks: list[str]  # in A but not B
    regressions: list[str]  # success->failed
    fixes: list[str]  # failed->success


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


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, (float, str)):
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    return None


def _parse_iso(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    normalized = value
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _duration_from_timestamps(started_at: object, finished_at: object) -> float | None:
    start_dt = _parse_iso(started_at)
    finish_dt = _parse_iso(finished_at)
    if start_dt is None or finish_dt is None:
        return None
    return max(0.0, (finish_dt - start_dt).total_seconds())


def _task_results(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_task_results = manifest.get("task_results")
    if not isinstance(raw_task_results, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for task_id, payload in raw_task_results.items():
        if isinstance(task_id, str) and isinstance(payload, dict):
            out[task_id] = payload
    return out


def _task_status(task_result: dict[str, Any] | None) -> TaskStatus | None:
    if task_result is None:
        return None

    raw = task_result.get("status")
    if isinstance(raw, str) and raw in _VALID_TASK_STATUSES:
        return cast(TaskStatus, raw)
    return None


def _task_duration(task_result: dict[str, Any] | None) -> float | None:
    if task_result is None:
        return None

    duration = _coerce_float(task_result.get("duration_sec"))
    if duration is not None:
        return duration

    return _duration_from_timestamps(task_result.get("started_at"), task_result.get("finished_at"))


def _task_cost(task_result: dict[str, Any] | None) -> float | None:
    if task_result is None:
        return None
    return _coerce_float(task_result.get("cost_usd"))


def _task_tokens(task_result: dict[str, Any] | None) -> int | None:
    if task_result is None:
        return None

    token_usage = task_result.get("token_usage")
    if not isinstance(token_usage, dict):
        return None

    total = _coerce_int(token_usage.get("total_tokens"))
    if total is not None:
        return total

    input_tokens = _coerce_int(token_usage.get("input_tokens")) or 0
    cached_tokens = _coerce_int(token_usage.get("cached_tokens")) or 0
    output_tokens = _coerce_int(token_usage.get("output_tokens")) or 0

    if input_tokens or cached_tokens or output_tokens:
        return input_tokens + cached_tokens + output_tokens
    return None


def _run_success(manifest: dict[str, Any], task_results: dict[str, dict[str, Any]]) -> bool:
    raw_success = manifest.get("success")
    if isinstance(raw_success, bool):
        return raw_success

    if not task_results:
        return False

    return not any(_task_status(task_result) == "failed" for task_result in task_results.values())


def _run_duration(manifest: dict[str, Any], task_results: dict[str, dict[str, Any]]) -> float:
    duration = _duration_from_timestamps(manifest.get("started_at"), manifest.get("finished_at"))
    if duration is not None:
        return duration

    total = 0.0
    found = False
    for task_result in task_results.values():
        task_duration = _task_duration(task_result)
        if task_duration is None:
            continue
        total += task_duration
        found = True

    return total if found else 0.0


def _run_cost(manifest: dict[str, Any], task_results: dict[str, dict[str, Any]]) -> float | None:
    total = _coerce_float(manifest.get("total_cost_usd"))
    if total is not None:
        return total

    costs = [_task_cost(task_result) for task_result in task_results.values()]
    values = [cost for cost in costs if cost is not None]
    if not values:
        return None
    return sum(values)


def _run_tokens(manifest: dict[str, Any], task_results: dict[str, dict[str, Any]]) -> int | None:
    total = _coerce_int(manifest.get("total_tokens"))
    if total is not None:
        return total

    token_values = [_task_tokens(task_result) for task_result in task_results.values()]
    values = [tokens for tokens in token_values if tokens is not None]
    if not values:
        return None
    return sum(values)


def load_run_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists() or not manifest_path.is_file():
        raise FileNotFoundError(f"run_manifest.json not found in: {run_dir}")

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {manifest_path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError(f"Invalid manifest format in {manifest_path}: expected JSON object")

    return payload


def compare_manifests(
    manifest_a: dict[str, Any],
    manifest_b: dict[str, Any],
    *,
    fallback_run_id_a: str = "run_a",
    fallback_run_id_b: str = "run_b",
) -> RunDiff:
    task_results_a = _task_results(manifest_a)
    task_results_b = _task_results(manifest_b)

    task_ids_a = set(task_results_a)
    task_ids_b = set(task_results_b)

    all_task_ids = sorted(task_ids_a | task_ids_b)
    added_tasks = sorted(task_ids_b - task_ids_a)
    removed_tasks = sorted(task_ids_a - task_ids_b)

    task_diffs: list[TaskDiff] = []
    regressions: list[str] = []
    fixes: list[str] = []

    for task_id in all_task_ids:
        task_a = task_results_a.get(task_id)
        task_b = task_results_b.get(task_id)
        status_a = _task_status(task_a)
        status_b = _task_status(task_b)

        task_diffs.append(
            TaskDiff(
                task_id=task_id,
                status_a=status_a,
                status_b=status_b,
                duration_a=_task_duration(task_a),
                duration_b=_task_duration(task_b),
                cost_a=_task_cost(task_a),
                cost_b=_task_cost(task_b),
                tokens_a=_task_tokens(task_a),
                tokens_b=_task_tokens(task_b),
            )
        )

        if status_a == "success" and status_b == "failed":
            regressions.append(task_id)
        elif status_a == "failed" and status_b == "success":
            fixes.append(task_id)

    return RunDiff(
        run_id_a=str(manifest_a.get("run_id") or fallback_run_id_a),
        run_id_b=str(manifest_b.get("run_id") or fallback_run_id_b),
        plan_name_a=str(manifest_a.get("plan_name") or ""),
        plan_name_b=str(manifest_b.get("plan_name") or ""),
        success_a=_run_success(manifest_a, task_results_a),
        success_b=_run_success(manifest_b, task_results_b),
        duration_a=_run_duration(manifest_a, task_results_a),
        duration_b=_run_duration(manifest_b, task_results_b),
        cost_a=_run_cost(manifest_a, task_results_a),
        cost_b=_run_cost(manifest_b, task_results_b),
        tokens_a=_run_tokens(manifest_a, task_results_a),
        tokens_b=_run_tokens(manifest_b, task_results_b),
        task_diffs=task_diffs,
        added_tasks=added_tasks,
        removed_tasks=removed_tasks,
        regressions=regressions,
        fixes=fixes,
    )


def compare_runs(run_dir_a: Path, run_dir_b: Path) -> RunDiff:
    manifest_a = load_run_manifest(run_dir_a)
    manifest_b = load_run_manifest(run_dir_b)

    return compare_manifests(
        manifest_a,
        manifest_b,
        fallback_run_id_a=run_dir_a.name,
        fallback_run_id_b=run_dir_b.name,
    )


def diff_runs(run_dir_a: Path | str, run_dir_b: Path | str) -> RunDiff:
    """Compare two run directories and return aggregate + task-level differences."""
    return compare_runs(Path(run_dir_a), Path(run_dir_b))


def _format_money_delta(a: float | None, b: float | None) -> str:
    if a is None or b is None:
        return "n/a"
    delta = b - a
    sign = "+" if delta > 0 else ""
    return f"{sign}${delta:.4f}"


def _format_number_delta(a: float | int | None, b: float | int | None) -> str:
    if a is None or b is None:
        return "n/a"
    delta = b - a
    sign = "+" if delta > 0 else ""
    if isinstance(delta, float):
        return f"{sign}{delta:.2f}"
    return f"{sign}{delta}"


def format_diff(diff: RunDiff) -> str:
    """Render a human-readable run diff summary."""
    lines = [
        f"Run A: {diff.run_id_a} (plan={diff.plan_name_a or '-'})",
        f"Run B: {diff.run_id_b} (plan={diff.plan_name_b or '-'})",
        (
            "Summary: "
            f"success {diff.success_a}->{diff.success_b}, "
            f"duration {diff.duration_a:.2f}s->{diff.duration_b:.2f}s "
            f"(delta {_format_number_delta(diff.duration_a, diff.duration_b)}), "
            f"cost {diff.cost_a if diff.cost_a is not None else 'n/a'}"
            f"->{diff.cost_b if diff.cost_b is not None else 'n/a'} "
            f"(delta {_format_money_delta(diff.cost_a, diff.cost_b)}), "
            f"tokens {diff.tokens_a if diff.tokens_a is not None else 'n/a'}"
            f"->{diff.tokens_b if diff.tokens_b is not None else 'n/a'} "
            f"(delta {_format_number_delta(diff.tokens_a, diff.tokens_b)})"
        ),
        (
            "Tasks: "
            f"added={len(diff.added_tasks)}, removed={len(diff.removed_tasks)}, "
            f"regressions={len(diff.regressions)}, fixes={len(diff.fixes)}"
        ),
    ]

    if diff.added_tasks:
        lines.append(f"Added tasks: {', '.join(diff.added_tasks)}")
    if diff.removed_tasks:
        lines.append(f"Removed tasks: {', '.join(diff.removed_tasks)}")
    if diff.regressions:
        lines.append(f"Regressions: {', '.join(diff.regressions)}")
    if diff.fixes:
        lines.append(f"Fixes: {', '.join(diff.fixes)}")

    return "\n".join(lines)


def format_diff_json(diff: RunDiff) -> str:
    """Render a machine-readable JSON representation of a run diff."""
    return json.dumps(asdict(diff), ensure_ascii=True, indent=2)


__all__ = [
    "TaskDiff",
    "RunDiff",
    "diff_runs",
    "format_diff",
    "format_diff_json",
    "load_run_manifest",
    "compare_manifests",
    "compare_runs",
]
