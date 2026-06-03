from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Literal

from .cache import cache_lookup, compute_task_hash
from .diff import load_run_manifest
from .models import PlanSpec

PipelineState = Literal["up-to-date", "stale", "never-run", "failed", "skipped"]

_PIPELINE_STATES: tuple[PipelineState, ...] = (
    "up-to-date",
    "stale",
    "never-run",
    "failed",
    "skipped",
)
_SUCCESS_STATUS = "success"
_FAILED_STATUSES: set[str] = {"failed", "soft_failed"}
_SKIPPED_STATUSES: set[str] = {"skipped", "dry_run"}


@dataclass
class TaskPipelineStatus:
    task_id: str
    state: PipelineState
    last_run_at: str | None = None
    last_duration: float | None = None
    last_cost: float | None = None
    stale_reason: str | None = None


@dataclass
class PlanPipelineStatus:
    plan_name: str
    last_run_id: str | None
    last_run_at: str | None
    tasks: list[TaskPipelineStatus]

    @property
    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {state: 0 for state in _PIPELINE_STATES}
        for task in self.tasks:
            counts[task.state] = counts.get(task.state, 0) + 1
        return counts


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


def _manifest_task_results(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    raw_task_results = manifest.get("task_results")
    if not isinstance(raw_task_results, dict):
        return {}

    out: dict[str, dict[str, Any]] = {}
    for task_id, payload in raw_task_results.items():
        if isinstance(task_id, str) and isinstance(payload, dict):
            out[task_id] = payload
    return out


def _load_result_payload(run_path: Path, task_id: str) -> dict[str, Any] | None:
    result_path = run_path / f"{task_id}.result.json"
    if not result_path.exists() or not result_path.is_file():
        return None
    try:
        data = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _merge_task_payloads(
    file_payload: dict[str, Any] | None,
    manifest_payload: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if file_payload is None and manifest_payload is None:
        return None

    merged: dict[str, Any] = {}
    if manifest_payload is not None:
        merged.update(manifest_payload)
    if file_payload is not None:
        merged.update(file_payload)
    return merged


def _extract_status(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    raw_status = payload.get("status")
    if not isinstance(raw_status, str) or not raw_status:
        return None
    return raw_status


def _extract_task_hash(payload: dict[str, Any] | None) -> str | None:
    if payload is None:
        return None
    raw_hash = payload.get("task_hash")
    if not isinstance(raw_hash, str) or not raw_hash:
        return None
    return raw_hash


def _extract_last_run_at(
    payload: dict[str, Any] | None,
    fallback_last_run_at: str | None,
) -> str | None:
    if payload is not None:
        finished_at = payload.get("finished_at")
        if isinstance(finished_at, str) and finished_at:
            return finished_at
    return fallback_last_run_at


def _extract_last_duration(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    duration = _coerce_float(payload.get("duration_sec"))
    if duration is not None:
        return duration
    return _duration_from_timestamps(payload.get("started_at"), payload.get("finished_at"))


def _extract_last_cost(payload: dict[str, Any] | None) -> float | None:
    if payload is None:
        return None
    return _coerce_float(payload.get("cost_usd"))


def _compute_hashes_for_plan(plan: PlanSpec) -> dict[str, str | None]:
    """Compute current task hashes for all tasks in topological order."""
    task_map = {task.id: task for task in plan.tasks}
    hashes: dict[str, str | None] = {}
    visiting: set[str] = set()

    def _visit(task_id: str) -> str | None:
        if task_id in hashes:
            return hashes[task_id]
        if task_id in visiting:
            hashes[task_id] = None
            return None

        task = task_map[task_id]
        visiting.add(task_id)

        upstream_hashes: dict[str, str] = {}
        for dep_id in sorted(set(task.depends_on)):
            if dep_id in task_map:
                dep_hash = _visit(dep_id)
                upstream_hashes[dep_id] = dep_hash or ""

        try:
            task_hash = compute_task_hash(task, plan, upstream_hashes)
        except Exception:
            task_hash = None

        visiting.discard(task_id)
        hashes[task_id] = task_hash
        return task_hash

    for task in plan.tasks:
        _visit(task.id)

    return hashes


def _base_state_from_status(raw_status: str | None) -> PipelineState:
    if raw_status is None:
        return "never-run"
    if raw_status in _FAILED_STATUSES:
        return "failed"
    if raw_status in _SKIPPED_STATUSES:
        return "skipped"
    if raw_status == _SUCCESS_STATUS:
        return "up-to-date"
    return "never-run"


def _stale_reason(
    *,
    task_id: str,
    cached_hash: str | None,
    current_hash: str | None,
) -> str:
    """Return a human-readable reason why a task is stale."""
    if cached_hash is None:
        return "missing previous task hash"
    if current_hash is None:
        return "cannot compute current task hash"
    if cached_hash != current_hash:
        return "task inputs changed"
    return f"task '{task_id}' is stale"


def plan_status(
    plan: PlanSpec,
    latest_run_path: Path | None,
    cache_dir: Path | None,
) -> PlanPipelineStatus:
    if latest_run_path is None:
        return PlanPipelineStatus(
            plan_name=plan.name,
            last_run_id=None,
            last_run_at=None,
            tasks=[TaskPipelineStatus(task_id=task.id, state="never-run") for task in plan.tasks],
        )

    try:
        manifest = load_run_manifest(latest_run_path)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return PlanPipelineStatus(
            plan_name=plan.name,
            last_run_id=None,
            last_run_at=None,
            tasks=[TaskPipelineStatus(task_id=task.id, state="never-run") for task in plan.tasks],
        )

    manifest_task_results = _manifest_task_results(manifest)
    last_run_id: str | None = None
    last_run_at: str | None = None
    if manifest is not None:
        raw_run_id = manifest.get("run_id")
        if isinstance(raw_run_id, str) and raw_run_id:
            last_run_id = raw_run_id
        else:
            last_run_id = latest_run_path.name if latest_run_path is not None else None

        raw_last_run_at = manifest.get("finished_at")
        if isinstance(raw_last_run_at, str) and raw_last_run_at:
            last_run_at = raw_last_run_at

    current_hashes = _compute_hashes_for_plan(plan)

    task_statuses: list[TaskPipelineStatus] = []
    for task in plan.tasks:
        file_payload = (
            _load_result_payload(latest_run_path, task.id)
            if latest_run_path is not None
            else None
        )
        manifest_payload = manifest_task_results.get(task.id)
        payload = _merge_task_payloads(file_payload, manifest_payload)

        raw_status = _extract_status(payload)
        base_state = _base_state_from_status(raw_status)

        stale_reason: str | None = None
        state = base_state
        if raw_status == _SUCCESS_STATUS:
            previous_hash = _extract_task_hash(payload)
            current_hash = current_hashes.get(task.id)
            if previous_hash is None or current_hash is None or previous_hash != current_hash:
                state = "stale"
                stale_reason = _stale_reason(
                    task_id=task.id,
                    cached_hash=previous_hash,
                    current_hash=current_hash,
                )

        task_statuses.append(
            TaskPipelineStatus(
                task_id=task.id,
                state=state,
                last_run_at=_extract_last_run_at(payload, last_run_at),
                last_duration=_extract_last_duration(payload),
                last_cost=_extract_last_cost(payload),
                stale_reason=stale_reason,
            )
        )

    return PlanPipelineStatus(
        plan_name=plan.name,
        last_run_id=last_run_id,
        last_run_at=last_run_at,
        tasks=task_statuses,
    )


def _format_duration(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}s"


def _format_cost(value: float | None) -> str:
    if value is None:
        return "-"
    return f"${value:.4f}"


def format_status(ps: PlanPipelineStatus) -> str:
    summary = ps.summary
    lines: list[str] = [
        f"Plan: {ps.plan_name}",
        f"Last run: {ps.last_run_id or '-'} at {ps.last_run_at or '-'}",
        (
            "Summary: "
            f"up-to-date={summary['up-to-date']}, "
            f"stale={summary['stale']}, "
            f"never-run={summary['never-run']}, "
            f"failed={summary['failed']}, "
            f"skipped={summary['skipped']}"
        ),
        "",
    ]

    headers = ("Task", "State", "Last Run", "Duration", "Cost", "Reason")
    rows = [
        (
            task.task_id,
            task.state,
            task.last_run_at or "-",
            _format_duration(task.last_duration),
            _format_cost(task.last_cost),
            task.stale_reason or "-",
        )
        for task in ps.tasks
    ]

    widths = [len(header) for header in headers]
    for row in rows:
        for idx, value in enumerate(row):
            widths[idx] = max(widths[idx], len(value))

    lines.append(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    lines.append("-+-".join("-" * width for width in widths))
    for row in rows:
        lines.append(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))

    return "\n".join(lines)


def format_status_json(ps: PlanPipelineStatus) -> str:
    return json.dumps(asdict(ps), ensure_ascii=True, indent=2)


__all__ = [
    "PipelineState",
    "TaskPipelineStatus",
    "PlanPipelineStatus",
    "plan_status",
    "format_status",
    "format_status_json",
]
