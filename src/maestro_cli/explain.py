from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

from .cache import cache_lookup, cache_stats, compute_task_hash
from .eventsource import replay_events
from .models import (
    ContextSelectionEntry,
    ContextTrajectoryReport,
    PlanSpec,
    TaskSpec,
)

ExplainCacheStatus = Literal["hit", "miss", "disabled", "no-cache-dir"]


@dataclass
class TaskExplanation:
    task_id: str
    cache_status: ExplainCacheStatus
    reason: str
    current_hash: str | None = None
    cached_hash: str | None = None


@dataclass
class PlanExplanation:
    plan_name: str
    task_count: int
    cache_entries: int
    cache_size_bytes: int
    tasks: list[TaskExplanation]


def _read_cache_stats(cache_dir: Path | None) -> tuple[int, int]:
    """Return (entry_count, total_size_bytes) for the cache directory."""
    if cache_dir is None:
        return 0, 0
    raw = cache_stats(cache_dir)
    return int(raw.get("entries", 0)), int(raw.get("total_size_bytes", 0))


def _short_hash(value: str) -> str:
    if len(value) <= 12:
        return value
    return f"{value[:12]}..."


def _topological_sort(plan: PlanSpec) -> list[TaskSpec]:
    tasks_by_id = {task.id: task for task in plan.tasks}
    in_degree = {task.id: 0 for task in plan.tasks}
    dependents: dict[str, list[str]] = {task.id: [] for task in plan.tasks}

    for task in plan.tasks:
        for dependency_id in task.depends_on:
            if dependency_id not in tasks_by_id:
                raise ValueError(
                    f"Task '{task.id}' depends on unknown task '{dependency_id}'"
                )
            in_degree[task.id] += 1
            dependents[dependency_id].append(task.id)

    ready: list[str] = [task.id for task in plan.tasks if in_degree[task.id] == 0]
    ordered_ids: list[str] = []
    cursor = 0
    while cursor < len(ready):
        task_id = ready[cursor]
        cursor += 1
        ordered_ids.append(task_id)

        for dependent_id in dependents[task_id]:
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                ready.append(dependent_id)

    if len(ordered_ids) != len(plan.tasks):
        raise ValueError("Dependency cycle detected in plan tasks")

    return [tasks_by_id[task_id] for task_id in ordered_ids]


def explain_plan(plan: PlanSpec, cache_dir: Path | None) -> PlanExplanation:
    ordered_tasks = _topological_sort(plan)
    cache_entries, cache_size_bytes = _read_cache_stats(cache_dir)

    upstream_hashes: dict[str, str] = {}
    explanations: list[TaskExplanation] = []

    for task in ordered_tasks:
        if not task.cache:
            explanations.append(
                TaskExplanation(
                    task_id=task.id,
                    cache_status="disabled",
                    reason="caching disabled",
                )
            )
            continue

        if cache_dir is None:
            explanations.append(
                TaskExplanation(
                    task_id=task.id,
                    cache_status="no-cache-dir",
                    reason="no cache directory configured",
                )
            )
            continue

        try:
            current_hash = compute_task_hash(task, plan, upstream_hashes)
        except Exception:
            explanations.append(
                TaskExplanation(
                    task_id=task.id,
                    cache_status="miss",
                    reason="hash computation failed",
                )
            )
            continue

        upstream_hashes[task.id] = current_hash

        cached_entry = cache_lookup(cache_dir, current_hash)
        if cached_entry is None:
            explanations.append(
                TaskExplanation(
                    task_id=task.id,
                    cache_status="miss",
                    reason="no cached result",
                    current_hash=current_hash,
                )
            )
            continue

        explanations.append(
            TaskExplanation(
                task_id=task.id,
                cache_status="hit",
                reason=f"hash match [{_short_hash(current_hash)}]",
                current_hash=current_hash,
                cached_hash=current_hash,
            )
        )

    return PlanExplanation(
        plan_name=plan.name,
        task_count=len(plan.tasks),
        cache_entries=cache_entries,
        cache_size_bytes=cache_size_bytes,
        tasks=explanations,
    )


def format_explain(explanation: PlanExplanation) -> str:
    lines = [
        f"Plan: {explanation.plan_name}",
        f"Tasks: {explanation.task_count}",
        f"Cache entries: {explanation.cache_entries}",
        f"Cache size: {explanation.cache_size_bytes} bytes",
        "",
    ]

    if not explanation.tasks:
        lines.append("(no tasks)")
        return "\n".join(lines)

    rows = [
        (item.task_id, item.cache_status, item.reason)
        for item in explanation.tasks
    ]

    task_width = max(len("task_id"), *(len(row[0]) for row in rows))
    status_width = max(len("status"), *(len(row[1]) for row in rows))
    reason_width = max(len("reason"), *(len(row[2]) for row in rows))

    lines.append(
        f"{'task_id'.ljust(task_width)} | {'status'.ljust(status_width)} | {'reason'.ljust(reason_width)}"
    )
    lines.append(
        f"{'-' * task_width}-+-{'-' * status_width}-+-{'-' * reason_width}"
    )
    for task_id, status, reason in rows:
        lines.append(
            f"{task_id.ljust(task_width)} | {status.ljust(status_width)} | {reason.ljust(reason_width)}"
        )

    return "\n".join(lines)


def format_explain_json(explanation: PlanExplanation) -> str:
    return json.dumps(asdict(explanation), indent=2)


def _safe_str(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _safe_keywords(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, str)]


def _coerce_selection_entry(raw: object) -> ContextSelectionEntry | None:
    if not isinstance(raw, dict):
        return None

    upstream_id = _safe_str(raw.get("upstream_id") or raw.get("upstream"))
    if not upstream_id:
        return None

    return ContextSelectionEntry(
        upstream_id=upstream_id,
        score=_safe_float(raw.get("score")) or 0.0,
        keywords_matched=_safe_keywords(
            raw.get("keywords_matched") or raw.get("intent_keywords")
        ),
        hop_distance=_safe_int(raw.get("hop_distance")) or 0,
        hop_decay_factor=_safe_float(raw.get("hop_decay_factor")) or 1.0,
        tokens_raw=_safe_int(
            raw.get("tokens_raw") or raw.get("original_tokens")
        ) or 0,
        tokens_final=_safe_int(
            raw.get("tokens_final") or raw.get("trimmed_tokens")
        ) or 0,
        trimmed=bool(raw.get("trimmed", False)),
        trim_reason=_safe_str(raw.get("trim_reason") or raw.get("reason")),
    )


def _merge_selection_entry(
    existing: ContextSelectionEntry | None,
    incoming: ContextSelectionEntry,
) -> ContextSelectionEntry:
    if existing is None:
        return incoming

    score = incoming.score if incoming.score != 0.0 else existing.score
    keywords = incoming.keywords_matched or existing.keywords_matched
    hop_distance = incoming.hop_distance if incoming.hop_distance != 0 else existing.hop_distance
    hop_decay = (
        incoming.hop_decay_factor
        if incoming.hop_decay_factor != 1.0 or existing.hop_decay_factor == 1.0
        else existing.hop_decay_factor
    )
    tokens_raw = incoming.tokens_raw if incoming.tokens_raw > 0 else existing.tokens_raw
    if incoming.tokens_final > 0 or incoming.trimmed or existing.tokens_final == 0:
        tokens_final = incoming.tokens_final
    else:
        tokens_final = existing.tokens_final

    return ContextSelectionEntry(
        upstream_id=existing.upstream_id,
        score=score,
        keywords_matched=keywords,
        hop_distance=hop_distance,
        hop_decay_factor=hop_decay,
        tokens_raw=tokens_raw,
        tokens_final=tokens_final,
        trimmed=incoming.trimmed or existing.trimmed,
        trim_reason=incoming.trim_reason or existing.trim_reason,
    )


def explain_context_trajectory(run_path: Path) -> list[ContextTrajectoryReport]:
    """Read ``events.jsonl`` from a completed run and rebuild context decisions."""
    events_path = run_path / "events.jsonl"
    if not events_path.is_file():
        return []

    reports: dict[str, ContextTrajectoryReport] = {}
    entries_by_task: dict[str, dict[str, ContextSelectionEntry]] = {}

    for record in replay_events(events_path):
        payload = record.payload
        event_name = _safe_str(payload.get("event") or payload.get("type"))
        task_id = _safe_str(payload.get("task_id"))
        if not task_id:
            continue

        if event_name not in {
            "context_selection",
            "context_trajectory",
            "context_budget_trim",
            "context_compression",
        }:
            continue

        report = reports.setdefault(task_id, ContextTrajectoryReport(task_id=task_id))
        task_entries = entries_by_task.setdefault(task_id, {})

        budget = _safe_int(payload.get("budget") or payload.get("budget_tokens"))
        if budget is not None:
            report.budget_tokens = budget

        if event_name in {"context_selection", "context_trajectory"}:
            entry = _coerce_selection_entry(payload)
            if entry is not None:
                task_entries[entry.upstream_id] = _merge_selection_entry(
                    task_entries.get(entry.upstream_id),
                    entry,
                )

        if event_name == "context_budget_trim":
            upstream_id = _safe_str(payload.get("upstream_id") or payload.get("upstream"))
            if upstream_id:
                trim_entry = ContextSelectionEntry(
                    upstream_id=upstream_id,
                    tokens_raw=_safe_int(payload.get("original_tokens")) or 0,
                    tokens_final=_safe_int(payload.get("trimmed_tokens")) or 0,
                    trimmed=True,
                    trim_reason=_safe_str(payload.get("trim_reason")) or "budget_trim",
                )
                task_entries[upstream_id] = _merge_selection_entry(
                    task_entries.get(upstream_id),
                    trim_entry,
                )

        if event_name == "context_compression":
            raw_tokens = _safe_int(payload.get("context_raw_tokens"))
            final_tokens = _safe_int(payload.get("context_final_tokens"))
            if raw_tokens is not None:
                report.total_tokens_raw = raw_tokens
            if final_tokens is not None:
                report.total_tokens_final = final_tokens

            raw_entries = payload.get("entries") or payload.get("selection_entries")
            if isinstance(raw_entries, list):
                for raw_entry in raw_entries:
                    entry = _coerce_selection_entry(raw_entry)
                    if entry is None:
                        continue
                    task_entries[entry.upstream_id] = _merge_selection_entry(
                        task_entries.get(entry.upstream_id),
                        entry,
                    )

    resolved_reports: list[ContextTrajectoryReport] = []
    for task_id in sorted(reports):
        report = reports[task_id]
        entries = list(entries_by_task.get(task_id, {}).values())
        entries.sort(
            key=lambda item: (
                -item.score,
                item.hop_distance,
                item.upstream_id,
            )
        )
        report.entries = entries
        if report.total_tokens_raw == 0:
            report.total_tokens_raw = sum(entry.tokens_raw for entry in entries)
        if report.total_tokens_final == 0:
            report.total_tokens_final = sum(entry.tokens_final for entry in entries)
        report.upstreams_evicted = sum(
            1 for entry in entries if entry.trimmed and entry.tokens_final == 0
        )
        resolved_reports.append(report)

    return resolved_reports


def explain_context(run_path: Path) -> list[ContextTrajectoryReport]:
    """Backward-compatible alias for context trajectory reporting."""
    return explain_context_trajectory(run_path)


def format_context_trajectory(reports: list[ContextTrajectoryReport]) -> str:
    if not reports:
        return "(no context trajectory events found)"

    lines: list[str] = []
    for index, report in enumerate(reports):
        if index:
            lines.append("")
        budget = (
            str(report.budget_tokens)
            if report.budget_tokens is not None
            else "n/a"
        )
        lines.extend([
            f"Task: {report.task_id}",
            (
                f"Totals: raw={report.total_tokens_raw} final={report.total_tokens_final} "
                f"budget={budget} evicted={report.upstreams_evicted}"
            ),
        ])

        if not report.entries:
            lines.append("(no upstream context entries)")
            continue

        rows = [
            (
                entry.upstream_id,
                f"{entry.score:.4f}",
                str(entry.hop_distance),
                f"{entry.hop_decay_factor:.4f}",
                str(entry.tokens_raw),
                str(entry.tokens_final),
                "yes" if entry.trimmed else "no",
                entry.trim_reason or "-",
                ",".join(entry.keywords_matched) or "-",
            )
            for entry in report.entries
        ]
        headers = ("upstream", "score", "hop", "decay", "raw", "final", "trim", "reason", "keywords")
        widths = [
            max(len(header), *(len(row[idx]) for row in rows))
            for idx, header in enumerate(headers)
        ]
        lines.append(
            " | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
        )
        lines.append(
            "-+-".join("-" * width for width in widths)
        )
        for row in rows:
            lines.append(
                " | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row))
            )

    return "\n".join(lines)


def format_context_trajectory_json(reports: list[ContextTrajectoryReport]) -> str:
    return json.dumps([report.to_dict() for report in reports], indent=2)


__all__ = [
    "ContextSelectionEntry",
    "ContextTrajectoryReport",
    "ExplainCacheStatus",
    "TaskExplanation",
    "PlanExplanation",
    "explain_context",
    "explain_context_trajectory",
    "explain_plan",
    "format_context_trajectory",
    "format_context_trajectory_json",
    "format_explain",
    "format_explain_json",
]
