"""In-memory state for active plan runs.

Tracks runs that are currently executing in background threads so the API
and SSE endpoints can report progress while the scheduler is working.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from ..models import PlanRunResult
from ..utils import now_utc


@dataclass
class RunState:
    """Tracks a single active run."""

    run_id: str
    plan_name: str
    task_ids: list[str]
    run_path: Path
    started_at: datetime
    thread: threading.Thread
    execution_profile: str = "plan"
    dry_run: bool = False
    result: PlanRunResult | None = None
    error: str | None = None
    task_graph: dict[str, dict[str, object]] = field(default_factory=dict)

    @property
    def is_finished(self) -> bool:
        return not self.thread.is_alive()

    def to_summary(self) -> dict[str, object]:
        now = now_utc()
        active = not self.is_finished
        profile = self.result.execution_profile if self.result else self.execution_profile
        finished_at = self.result.finished_at if self.result else (now if not active else None)
        duration_sec: float | None = None
        if finished_at is not None:
            duration_sec = max(0.0, (finished_at - self.started_at).total_seconds())
        elif active:
            duration_sec = max(0.0, (now - self.started_at).total_seconds())
        return {
            "run_id": self.run_id,
            "plan_name": self.plan_name,
            "task_ids": self.task_ids,
            "run_path": str(self.run_path),
            "started_at": self.started_at.isoformat(),
            "finished_at": finished_at.isoformat() if finished_at is not None else None,
            "active": active,
            "success": self.result.success if self.result else None,
            "execution_profile": profile,
            "dry_run": self.dry_run,
            "duration_sec": duration_sec,
            "total_cost_usd": self.result.total_cost_usd if self.result else None,
            "error": self.error,
        }


_lock = threading.Lock()
_active_runs: dict[str, RunState] = {}

# Project roots — set at app startup, used for run discovery
_project_roots: list[Path] = [Path(".")]


def _normalize_project_roots(roots: list[Path]) -> list[Path]:
    normalized: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        try:
            resolved = root.resolve()
        except OSError:
            resolved = root
        if resolved in seen:
            continue
        seen.add(resolved)
        normalized.append(resolved)

    if not normalized:
        return [Path(".").resolve()]
    return normalized


def set_project_root(root: Path) -> None:
    set_project_roots([root])


def set_project_roots(roots: list[Path]) -> None:
    global _project_roots
    with _lock:
        _project_roots = _normalize_project_roots(roots)


def get_project_root() -> Path:
    with _lock:
        return _project_roots[0]


def get_project_roots() -> list[Path]:
    with _lock:
        return list(_project_roots)


def register_run(state: RunState) -> None:
    with _lock:
        _active_runs[state.run_id] = state


def get_run(run_id: str) -> RunState | None:
    with _lock:
        return _active_runs.get(run_id)


def list_active_runs() -> list[RunState]:
    with _lock:
        return list(_active_runs.values())


def remove_run(run_id: str) -> None:
    with _lock:
        _active_runs.pop(run_id, None)


def shutdown_active_runs() -> None:
    """Wait for all active runs to finish (called on app shutdown)."""
    with _lock:
        runs = list(_active_runs.values())
    for rs in runs:
        if rs.thread.is_alive():
            rs.thread.join(timeout=5)
    with _lock:
        _active_runs.clear()
