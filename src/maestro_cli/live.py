from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

from rich import box
from rich.live import Live
from rich.table import Table
from rich.text import Text

from .models import PlanSpec, STATUS_STYLES, TERMINAL_STATUSES
from .utils import format_duration, format_cost, humanize_output_line

_STATUS_STYLES = STATUS_STYLES
_TERMINAL_STATUSES = TERMINAL_STATUSES


@dataclass
class _TaskState:
    task_id: str
    status: str = "pending"
    engine: str | None = None
    model: str | None = None
    duration_sec: float | None = None
    cost_usd: float | None = None
    started_at: float | None = None
    reason: str | None = None
    last_line: str = ""
    attempts: int = 0
    max_retries: int = 0
    progress_pct: int | None = None


class _LivePlanDisplay:
    _OUTPUT_THROTTLE_SEC = 0.15

    def __init__(self, plan: PlanSpec) -> None:
        self._plan = plan
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self._run_duration_sec: float | None = None
        self._run_total_cost_usd: float | None = None
        self._budget_warning: str | None = None
        self._last_event_message = "waiting for events"
        self._last_output_update_at = 0.0
        self._pending_output: dict[str, str] = {}
        self._tasks: dict[str, _TaskState] = {
            task.id: _TaskState(task_id=task.id) for task in plan.tasks
        }

    def __rich__(self) -> Table:
        return self._build_table()

    def handle_event(self, event_name: str, payload: dict[str, object]) -> None:
        with self._lock:
            if event_name != "task_output":
                self._flush_pending_output_locked(force=True)
            if event_name == "task_start":
                self._handle_task_start(payload)
            elif event_name == "task_complete":
                self._handle_task_complete(payload)
            elif event_name == "task_skip":
                self._handle_task_skip(payload)
            elif event_name == "task_output":
                self._handle_task_output(payload)
            elif event_name == "task_retry":
                self._handle_task_retry(payload)
            elif event_name == "judge_start":
                self._handle_judge_start(payload)
            elif event_name in {"judge_verdict", "judge_result"}:
                self._handle_judge_verdict(payload)
            elif event_name == "task_escalation":
                self._handle_task_escalation(payload)
            elif event_name == "engine_fallback":
                self._handle_engine_fallback(payload)
            elif event_name == "verify_failure":
                self._handle_verify_failure(payload)
            elif event_name == "task_progress":
                self._handle_task_progress(payload)
            elif event_name == "task_artifact":
                self._handle_task_artifact(payload)
            elif event_name == "budget_warning":
                self._handle_budget_warning(payload)
            elif event_name == "run_complete":
                self._handle_run_complete(payload)
            else:
                self._last_event_message = event_name.replace("_", " ")

    def _handle_task_start(self, payload: dict[str, object]) -> None:
        task = self._get_task_state(payload)
        task.status = "running"
        task.engine = self._as_str(payload.get("engine"))
        task.model = self._as_str(payload.get("model"))
        task.started_at = time.monotonic()
        task.duration_sec = None
        task.cost_usd = None
        task.reason = None
        task.last_line = ""
        task.attempts = 0
        task.max_retries = self._as_int(payload.get("max_retries")) or 0
        self._last_event_message = f"{task.task_id} started"

    def _handle_task_complete(self, payload: dict[str, object]) -> None:
        task = self._get_task_state(payload)
        status = self._as_str(payload.get("status")) or "success"
        task.status = status
        task.duration_sec = self._as_float(payload.get("duration_sec"))
        task.cost_usd = self._as_float(payload.get("cost_usd"))
        task.started_at = None
        task.reason = None
        event_label = "completed" if status in {"success", "dry_run"} else status.replace("_", " ")
        self._last_event_message = (
            f"{task.task_id} {event_label} ({format_duration(task.duration_sec)}, "
            f"{format_cost(task.cost_usd)})"
        )

    def _handle_task_skip(self, payload: dict[str, object]) -> None:
        task = self._get_task_state(payload)
        task.status = "skipped"
        task.reason = self._as_str(payload.get("reason"))
        task.started_at = None
        task.duration_sec = None
        task.cost_usd = None
        reason = f": {task.reason}" if task.reason else ""
        self._last_event_message = f"{task.task_id} skipped{reason}"

    def _handle_task_output(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        line = humanize_output_line(str(payload.get("line", "")).strip())
        if not line:
            return
        now = time.monotonic()
        truncated_line = line[:60]
        if now - self._last_output_update_at >= self._OUTPUT_THROTTLE_SEC:
            state = self._get_or_create_state(task_id)
            state.last_line = truncated_line
            self._last_output_update_at = now
            self._pending_output.pop(task_id, None)
            return
        self._pending_output[task_id] = truncated_line

    def _handle_task_retry(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        attempt = self._as_int(payload.get("attempt")) or 0
        state = self._get_or_create_state(task_id)
        state.attempts = attempt
        state.max_retries = self._as_int(payload.get("max_retries")) or state.max_retries
        self._last_event_message = f"retry {task_id} (attempt {attempt})"

    def _handle_judge_start(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        self._last_event_message = f"[judge] evaluating {task_id}"

    def _handle_judge_verdict(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        verdict = str(payload.get("verdict", ""))
        self._last_event_message = f"[judge] {task_id}: {verdict}"

    def _handle_task_escalation(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        to_model = str(payload.get("to_model", ""))
        state = self._get_or_create_state(task_id)
        state.model = to_model
        self._last_event_message = f"escalated {task_id} -> {to_model}"

    def _handle_engine_fallback(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        to_engine = str(payload.get("to_engine", ""))
        state = self._get_or_create_state(task_id)
        state.engine = to_engine
        self._last_event_message = f"fallback {task_id} -> {to_engine}"

    def _handle_verify_failure(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        exit_code = payload.get("exit_code", "?")
        snippet = str(payload.get("output_snippet", ""))[:60]
        self._last_event_message = f"verify failed {task_id} (exit {exit_code}): {snippet}"

    def _handle_task_progress(self, payload: dict[str, object]) -> None:
        task = self._get_task_state(payload)
        pct = self._as_int(payload.get("pct"))
        step = self._as_str(payload.get("step"))
        if pct is not None:
            task.progress_pct = pct
        if step:
            task.last_line = step
        self._last_event_message = f"{task.task_id} {pct}%"

    def _handle_task_artifact(self, payload: dict[str, object]) -> None:
        task_id = self._as_str(payload.get("task_id"))
        label = self._as_str(payload.get("label")) or self._as_str(payload.get("path"))
        self._last_event_message = f"{task_id} artifact: {label}"

    def _handle_budget_warning(self, payload: dict[str, object]) -> None:
        spent = self._as_float(payload.get("spent"))
        limit = self._as_float(payload.get("limit"))
        pct = self._as_float(payload.get("pct"))
        parts = [f"spent {format_cost(spent)}"]
        if limit is not None:
            parts.append(f"limit {format_cost(limit)}")
        if pct is not None:
            parts.append(f"{pct:.0%}")
        self._budget_warning = "budget warning: " + ", ".join(parts)
        self._last_event_message = self._budget_warning

    def _handle_run_complete(self, payload: dict[str, object]) -> None:
        self._run_duration_sec = self._as_float(payload.get("duration_sec"))
        self._run_total_cost_usd = self._as_float(payload.get("cost_usd"))
        success = payload.get("success")
        status = "success" if success else "failed"
        self._last_event_message = (
            f"run completed: {status} "
            f"({format_duration(self._run_duration_sec)}, "
            f"{format_cost(self._run_total_cost_usd)})"
        )

    def _get_task_state(self, payload: dict[str, object]) -> _TaskState:
        task_id = self._as_str(payload.get("task_id")) or "(unknown)"
        return self._get_or_create_state(task_id)

    def _get_or_create_state(self, task_id: str) -> _TaskState:
        if not task_id:
            task_id = "(unknown)"
        task = self._tasks.get(task_id)
        if task is None:
            task = _TaskState(task_id=task_id)
            self._tasks[task_id] = task
        return task

    def _build_table(self) -> Table:
        with self._lock:
            self._flush_pending_output_locked()
            table = Table(
                box=box.SIMPLE_HEAVY,
                expand=True,
                show_header=False,
                title=self._build_header(),
                title_justify="left",
                caption=self._build_footer(),
                caption_justify="left",
                pad_edge=False,
            )
            table.add_column("Status", width=5, no_wrap=True)
            table.add_column("Task", ratio=3, overflow="ellipsis")
            table.add_column("Engine", ratio=3, overflow="ellipsis")
            table.add_column("Duration", width=8, justify="right", no_wrap=True)
            table.add_column("Cost", width=8, justify="right", no_wrap=True)
            table.add_column("Detail", ratio=2, overflow="ellipsis")

            for task in self._plan.tasks:
                state = self._tasks.get(task.id, _TaskState(task_id=task.id))
                icon, style = _STATUS_STYLES.get(state.status, ("[??]", "bold magenta"))
                table.add_row(
                    Text(icon, style=style),
                    Text(task.id),
                    Text(self._format_engine_model(state), style="dim" if state.status == "pending" else ""),
                    Text(self._format_row_duration(state), style="dim" if state.duration_sec is None and state.status != "running" else ""),
                    Text(format_cost(state.cost_usd), style="dim" if state.cost_usd is None else ""),
                    Text(self._format_detail(state), style="dim" if state.status in {"pending", "skipped"} else ""),
                )
            return table

    def _build_header(self) -> Text:
        completed = sum(
            1 for state in self._tasks.values() if state.status in _TERMINAL_STATUSES
        )
        total = len(self._plan.tasks)
        task_costs = [
            state.cost_usd
            for state in self._tasks.values()
            if state.cost_usd is not None
        ]
        total_cost = (
            self._run_total_cost_usd
            if self._run_total_cost_usd is not None
            else (sum(task_costs) if task_costs else None)
        )
        elapsed_sec = (
            self._run_duration_sec
            if self._run_duration_sec is not None
            else time.monotonic() - self._started_at
        )

        header = Text()
        header.append("MAESTRO", style="bold")
        header.append("  ")
        header.append(self._plan.name, style="bold white")
        header.append("  ")
        header.append(self._progress_bar(completed, total), style="cyan")
        header.append("  ")
        header.append(f"{completed}/{total}", style="bold")
        header.append("  ")
        header.append(format_cost(total_cost), style="green")
        header.append("  ")
        header.append(format_duration(elapsed_sec), style="cyan")
        if self._budget_warning:
            header.append("  ")
            header.append(self._budget_warning, style="bold yellow")
        return header

    def _build_footer(self) -> Text:
        return Text(f"Last: {self._last_event_message}", style="dim")

    def _progress_bar(self, completed: int, total: int) -> str:
        width = 16
        if total <= 0:
            return "─" * width
        filled = min(width, int(width * completed / total))
        return ("━" * filled) + ("─" * (width - filled))

    def _format_engine_model(self, state: _TaskState) -> str:
        if state.engine and state.model:
            return f"{state.engine}/{state.model}"
        if state.engine:
            return state.engine
        if state.model:
            return state.model
        # Completed/skipped tasks without engine info (e.g. resumed from cache)
        if state.status in _TERMINAL_STATUSES:
            return "—"
        return "(pending)"

    def _format_row_duration(self, state: _TaskState) -> str:
        if state.status == "running" and state.started_at is not None:
            return format_duration(time.monotonic() - state.started_at)
        # Resumed tasks have duration_sec=0.0 but no engine — show dash
        if (
            state.status in _TERMINAL_STATUSES
            and state.engine is None
            and state.duration_sec is not None
            and state.duration_sec == 0.0
        ):
            return "—"
        return format_duration(state.duration_sec)

    def _format_detail(self, state: _TaskState) -> str:
        if state.attempts > 0:
            max_retries = state.max_retries or state.attempts
            return f"retry {state.attempts}/{max_retries}"
        if state.status == "running":
            if state.last_line:
                return state.last_line
            return "running"
        if state.status == "skipped" and state.reason:
            return state.reason
        return ""

    def _flush_pending_output_locked(self, force: bool = False) -> None:
        if not self._pending_output:
            return
        now = time.monotonic()
        if not force and now - self._last_output_update_at < self._OUTPUT_THROTTLE_SEC:
            return
        for task_id, line in self._pending_output.items():
            state = self._get_or_create_state(task_id)
            state.last_line = line
        self._pending_output.clear()
        self._last_output_update_at = now

    @staticmethod
    def _as_str(value: object) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _as_float(value: object) -> float | None:
        if isinstance(value, int | float):
            return float(value)
        return None

    @staticmethod
    def _as_int(value: object) -> int | None:
        if isinstance(value, int):
            return value
        return None


def create_live_callback(
    plan: PlanSpec,
) -> tuple[Live, Callable[[str, dict[str, object]], None]]:
    """Create a Rich Live display and its event callback."""
    from rich.console import Console

    display = _LivePlanDisplay(plan)
    console = Console(force_terminal=True)
    live = Live(
        display,
        refresh_per_second=4,
        console=console,
        vertical_overflow="visible",
    )
    return live, display.handle_event
