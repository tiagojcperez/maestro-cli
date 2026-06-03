from __future__ import annotations

from dataclasses import replace

from .models import TaskSpec


def build_consistency_group_members(tasks: list[TaskSpec]) -> dict[str, list[str]]:
    members: dict[str, list[str]] = {}
    for task in tasks:
        for group in task.consistency_group:
            bucket = members.setdefault(group, [])
            if task.id not in bucket:
                bucket.append(task.id)
    return members


def resolve_task_dependencies(tasks: list[TaskSpec]) -> dict[str, list[str]]:
    task_map = {task.id: task for task in tasks}
    group_members = build_consistency_group_members(tasks)
    resolved: dict[str, list[str]] = {}

    for task in tasks:
        deps: list[str] = []
        seen: set[str] = set()

        def _add(dep_id: str) -> None:
            if dep_id == task.id or dep_id in seen:
                return
            if dep_id not in task_map:
                return
            seen.add(dep_id)
            deps.append(dep_id)

        for dep_id in task.depends_on:
            if dep_id in seen:
                continue
            seen.add(dep_id)
            deps.append(dep_id)
        for dep_id in task.consumes_contracts:
            _add(dep_id)
        for group in task.reconcile_after:
            for member_id in group_members.get(group, []):
                _add(member_id)

        resolved[task.id] = deps

    return resolved


def clone_tasks_with_resolved_dependencies(tasks: list[TaskSpec]) -> list[TaskSpec]:
    resolved = resolve_task_dependencies(tasks)
    return [
        replace(task, depends_on=list(resolved.get(task.id, task.depends_on)))
        for task in tasks
    ]
