from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rich.table import Table
from rich.text import Text
from textual.widgets import RichLog, Static

from ..models import PlanSpec, STATUS_STYLES, TERMINAL_STATUSES
from ..utils import format_duration, format_cost, humanize_output_line


# --- Task state (TUI's own, not shared with live.py) ---

@dataclass
class TaskState:
    task_id: str
    status: str = "pending"
    engine: str | None = None
    model: str | None = None
    duration_sec: float | None = None
    cost_usd: float | None = None
    reason: str | None = None
    last_line: str = ""
    started_at: float | None = None
    timeout_sec: int | None = None
    max_retries: int = 0


# --- PlanHeader ---

class PlanHeader(Static):
    """Top bar: plan name, progress, cost, elapsed."""

    def __init__(self, plan: PlanSpec) -> None:
        super().__init__()
        self._plan = plan
        self._completed = 0
        self._total = len(plan.tasks)
        self._total_cost = 0.0
        self._running_count: int = 0
        self._budget_warning: str | None = None
        self._finished = False
        self._started_at: float = time.monotonic()
        self._final_duration: float | None = None
        self._timer: object | None = None

    def on_mount(self) -> None:
        self.update(self.render())
        self._needs_refresh = False
        self._timer = self.set_interval(1.0, self._tick)

    def _tick(self) -> None:
        """Update header every second for elapsed time.

        Only re-renders when state has changed or elapsed time is ticking.
        Uses deferred flag to avoid blocking the keyboard event queue.
        """
        if self._finished and not self._needs_refresh:
            return
        self._needs_refresh = False
        self.update(self.render(), layout=False)

    def render(self) -> Text:
        elapsed = (
            self._final_duration
            if self._final_duration is not None
            else time.monotonic() - self._started_at
        )
        header = Text()
        header.append("MAESTRO", style="bold")
        header.append("  ")
        header.append(self._plan.name, style="bold white")
        header.append("  ")
        header.append(self._progress_bar(), style="cyan")
        header.append("  ")
        header.append(f"{self._completed}/{self._total}", style="bold")
        header.append("  ")
        header.append(format_cost(self._total_cost), style="green")
        header.append("  ")
        header.append(format_duration(elapsed), style="cyan")
        if self._budget_warning:
            header.append("  ")
            header.append(self._budget_warning, style="bold yellow")
        return header

    def increment_running(self) -> None:
        self._running_count += 1
        self._needs_refresh = True

    def task_completed(self, payload: dict[str, object]) -> None:
        self._completed += 1
        self._running_count = max(0, self._running_count - 1)
        cost = payload.get("cost_usd")
        if isinstance(cost, (int, float)):
            self._total_cost += cost
        self._needs_refresh = True

    def show_budget_warning(self, payload: dict[str, object]) -> None:
        spent = payload.get("spent")
        limit = payload.get("limit")
        spent_f = float(spent) if isinstance(spent, (int, float)) else None
        limit_f = float(limit) if isinstance(limit, (int, float)) else None
        self._budget_warning = f"budget: {format_cost(spent_f)}/{format_cost(limit_f)}"
        self._needs_refresh = True

    def run_completed(self, payload: dict[str, object]) -> None:
        self._finished = True
        dur = payload.get("duration_sec")
        if isinstance(dur, (int, float)):
            self._final_duration = dur
        else:
            self._final_duration = time.monotonic() - self._started_at
        cost = payload.get("cost_usd")
        if isinstance(cost, (int, float)):
            self._total_cost = cost
        self._needs_refresh = True

    def _progress_bar(self) -> str:
        width = 20
        if self._total <= 0:
            return "─" * width
        filled = min(width, int(width * self._completed / self._total))
        return ("━" * filled) + ("─" * (width - filled))


# --- DAGPanel ---

class DAGPanel(Static):
    """Task list rendered as a Rich Table (avoids Textual DataTable bugs)."""

    _OUTPUT_THROTTLE_SEC = 0.25  # max 4 table refreshes/sec for output lines
    _FILTER_CYCLE = ["all", "running", "failed", "completed"]

    def __init__(self, plan: PlanSpec) -> None:
        super().__init__()
        self._plan = plan
        self._tasks_by_id = {task.id: task for task in plan.tasks}
        self._descriptions: dict[str, str] = {
            task.id: task.description or "" for task in plan.tasks
        }
        self._states: dict[str, TaskState] = {
            task.id: TaskState(
                task_id=task.id,
                timeout_sec=task.timeout_sec,
                max_retries=task.max_retries,
            )
            for task in plan.tasks
        }
        self._last_output_refresh: float = 0.0
        self._cursor: int = 0
        self._filter: str = "all"
        self._follow: bool = False
        self._needs_refresh: bool = False

    @property
    def row_count(self) -> int:
        return len(self._visible_task_ids())

    def on_mount(self) -> None:
        self.update(self._render_table())
        self._timer = self.set_interval(0.5, self._tick)

    def _tick(self) -> None:
        """Render deferred updates and live elapsed times.

        Runs every 0.25s.  Event handlers set ``_needs_refresh`` instead of
        calling ``self.update()`` directly, so multiple events within a tick
        are batched into a single render.  This keeps the Textual message
        queue free for keyboard events.
        """
        if self._needs_refresh or any(
            s.status == "running" for s in self._states.values()
        ):
            self._needs_refresh = False
            # Re-clamp cursor to visible list bounds before rendering
            visible = self._visible_task_ids()
            if visible:
                self._cursor = max(0, min(len(visible) - 1, self._cursor))
            self._refresh_table()

    def _selected_task_id(self) -> str | None:
        """Return the task_id currently under the cursor (snapshot-safe)."""
        visible = self._visible_task_ids()
        if 0 <= self._cursor < len(visible):
            return visible[self._cursor]
        return None

    def move_cursor(self, delta: int) -> None:
        visible = self._visible_task_ids()
        if not visible:
            return
        self._cursor = max(0, min(len(visible) - 1, self._cursor + delta))
        self._follow = False  # manual navigation disables follow
        self._refresh_table()

    def move_cursor_to(self, task_id: str) -> None:
        visible = self._visible_task_ids()
        for i, visible_task_id in enumerate(visible):
            if visible_task_id == task_id:
                self._cursor = i
                self._refresh_table()
                return
        # Fallback: task not visible (filtered out) — clamp cursor to bounds
        if visible:
            self._cursor = max(0, min(len(visible) - 1, self._cursor))

    def get_cursor_task_id(self) -> str | None:
        visible = self._visible_task_ids()
        if not visible:
            return None
        # Clamp defensively in case cursor drifted
        self._cursor = max(0, min(len(visible) - 1, self._cursor))
        return visible[self._cursor]

    def cycle_filter(self) -> None:
        idx = self._FILTER_CYCLE.index(self._filter)
        self._filter = self._FILTER_CYCLE[(idx + 1) % len(self._FILTER_CYCLE)]
        self._cursor = 0
        self._refresh_table()

    def clear_filter(self) -> None:
        self._filter = "all"
        self._cursor = 0
        self._refresh_table()

    def toggle_follow(self) -> None:
        self._follow = not self._follow

    def _visible_task_ids(self) -> list[str]:
        result: list[str] = []
        for task in self._plan.tasks:
            state = self._states[task.id]
            if self._filter == "all":
                result.append(task.id)
            elif self._filter == "running" and state.status == "running":
                result.append(task.id)
            elif self._filter == "failed" and state.status in ("failed", "soft_failed"):
                result.append(task.id)
            elif self._filter == "completed" and state.status in TERMINAL_STATUSES:
                result.append(task.id)
        return result

    def _render_table(self) -> Table:
        table = Table(box=None, show_header=True, expand=True, padding=(0, 1))
        table.add_column("", width=4, no_wrap=True)
        table.add_column("Task", ratio=3, no_wrap=True)
        table.add_column("Engine", ratio=1, no_wrap=True)
        table.add_column("Duration", width=8, no_wrap=True)
        table.add_column("Cost", width=8, no_wrap=True)
        table.add_column("Info", ratio=1, no_wrap=True)
        # Snapshot: preserve selected task across re-renders
        prev_selected = self._selected_task_id()
        visible_task_ids = self._visible_task_ids()
        if self._filter != "all":
            table.title = f"[filter: {self._filter}]"
        # Try to keep cursor on the same task after state changes
        if prev_selected and prev_selected in visible_task_ids:
            self._cursor = visible_task_ids.index(prev_selected)
        elif visible_task_ids:
            self._cursor = max(0, min(len(visible_task_ids) - 1, self._cursor))
        else:
            self._cursor = 0
        for row_index, task_id in enumerate(visible_task_ids):
            task = self._tasks_by_id[task_id]
            state = self._states[task.id]
            icon, style = STATUS_STYLES.get(state.status, ("[??]", "bold magenta"))
            desc = self._descriptions.get(task.id, "")
            task_label = f"{task.id}  {desc}" if desc else task.id
            engine_str = ""
            info_str = Text("waiting", style="dim")
            dur_str = ""
            cost_str = ""
            if state.status == "running":
                engine_str = f"{state.engine}/{state.model}" if state.engine and state.model else state.engine or "shell"
                if state.started_at is not None:
                    dur_str = format_duration(time.monotonic() - state.started_at)
                info_str = Text(state.last_line or "running...", style="cyan")
            elif state.status in TERMINAL_STATUSES:
                engine_str = f"{state.engine}/{state.model}" if state.engine and state.model else state.engine or "shell"
                # Resumed tasks: duration=0.0 + no engine → show dash
                if state.engine is None and state.duration_sec is not None and state.duration_sec == 0.0:
                    dur_str = "—"
                else:
                    dur_str = format_duration(state.duration_sec)
                cost_str = format_cost(state.cost_usd)
                if state.status in ("success", "dry_run"):
                    info_str = Text("done", style="green")
                elif state.status == "failed":
                    info_str = Text("FAILED", style="bold red")
                elif state.status == "soft_failed":
                    info_str = Text("soft fail", style="yellow")
                elif state.status == "skipped":
                    info_str = Text(state.reason or "skipped", style="dim")
                else:
                    info_str = Text(state.status, style="dim")
            table.add_row(
                Text(icon, style=style),
                Text(task_label),
                Text(engine_str, style="dim"),
                Text(dur_str, style="dim"),
                Text(cost_str, style="dim"),
                info_str,
                style="bold reverse" if row_index == self._cursor else None,
            )
        return table

    def _refresh_table(self) -> None:
        self.update(self._render_table(), layout=False)

    def update_task_start(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        state = self._states.get(task_id)
        if state is None:
            return
        state.status = "running"
        state.engine = self._as_str(payload.get("engine"))
        state.model = self._as_str(payload.get("model"))
        state.started_at = time.monotonic()
        self._needs_refresh = True

    def update_task_complete(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        state = self._states.get(task_id)
        if state is None:
            return
        state.status = str(payload.get("status", "success"))
        state.duration_sec = self._as_float(payload.get("duration_sec"))
        state.cost_usd = self._as_float(payload.get("cost_usd"))
        state.started_at = None
        self._needs_refresh = True

    def update_task_skip(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        state = self._states.get(task_id)
        if state is None:
            return
        state.status = "skipped"
        state.reason = self._as_str(payload.get("reason"))
        self._needs_refresh = True

    def update_task_output(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        line = humanize_output_line(str(payload.get("line", "")).strip())
        state = self._states.get(task_id)
        if state is None or not line:
            return
        state.last_line = line
        # Throttle: only flag refresh at most every _OUTPUT_THROTTLE_SEC
        now = time.monotonic()
        if now - self._last_output_refresh >= self._OUTPUT_THROTTLE_SEC:
            self._last_output_refresh = now
            self._needs_refresh = True

    def update_task_progress(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        state = self._states.get(task_id)
        if state is None:
            return
        pct = payload.get("pct")
        if isinstance(pct, (int, float)):
            pct = max(0, min(100, int(pct)))
            step = str(payload.get("step", ""))
            state.last_line = f"[{pct}%] {step}".strip() if step else f"[{pct}%]"
            self._needs_refresh = True

    def update_task_retry(self, payload: dict[str, object]) -> None:
        task_id = str(payload.get("task_id", ""))
        state = self._states.get(task_id)
        if state is None:
            return
        self._needs_refresh = True

    @staticmethod
    def _as_str(value: object) -> str | None:
        return value if isinstance(value, str) else None

    @staticmethod
    def _as_float(value: object) -> float | None:
        if isinstance(value, (int, float)):
            return float(value)
        return None


# --- DetailPanel ---

class DetailPanel(Static):
    """Right-side panel showing selected task metadata and live log tail."""

    _LOG_POLL_SEC = 0.5
    _LOG_TAIL_LINES = 20

    def __init__(self) -> None:
        super().__init__()
        self._task_id: str | None = None
        self._state: TaskState | None = None
        self._log_path: Path | None = None
        self._last_log_size: int = 0
        self._log_lines: list[str] = []
        self._timer: object | None = None

    def on_mount(self) -> None:
        self._timer = self.set_interval(self._LOG_POLL_SEC, self._poll_log)

    def render(self) -> Text:
        """Render current state — called by Textual on every refresh cycle."""
        if self._task_id is not None and self._state is not None:
            return self._render_detail()
        return self._render_empty()

    def select_task(self, task_id: str, state: TaskState, run_path: Path | None) -> None:
        """Select a task to display details for."""
        self._task_id = task_id
        self._state = state
        self._log_lines = []
        self._last_log_size = 0
        if run_path is not None:
            self._log_path = run_path / f"{task_id}.log"
        else:
            self._log_path = None
        self._poll_log()
        self.refresh(layout=True)

    def clear_selection(self) -> None:
        """Clear task selection."""
        self._task_id = None
        self._state = None
        self._log_path = None
        self._log_lines = []
        self.refresh(layout=True)

    def update_state(self, state: TaskState) -> None:
        """Update the displayed task state (called on events)."""
        if self._state is not None and state.task_id == self._task_id:
            self._state = state
            self.refresh()

    def _poll_log(self) -> None:
        """Read new log content from disk."""
        if self._log_path is None or self._task_id is None:
            return
        try:
            if not self._log_path.exists():
                return
            size = self._log_path.stat().st_size
            if size == self._last_log_size:
                return
            content = self._log_path.read_text(encoding="utf-8", errors="replace")
            self._last_log_size = size
            # Humanize, drop blanks, deduplicate consecutive identical lines
            humanized: list[str] = []
            for raw in content.splitlines():
                ln = humanize_output_line(raw)
                if not ln:
                    continue
                if humanized and humanized[-1] == ln:
                    continue
                humanized.append(ln)
            self._log_lines = humanized[-self._LOG_TAIL_LINES:]
            self.refresh()
        except OSError:
            pass

    def _render_empty(self) -> Text:
        return Text("\u2191\u2193 to navigate, Enter to select task", style="dim italic")

    def _render_detail(self) -> Text:
        if self._state is None or self._task_id is None:
            return self._render_empty()
        s = self._state
        t = Text()
        t.append(" Task: ", style="bold")
        t.append(f"{s.task_id}\n")
        icon, style = STATUS_STYLES.get(s.status, ("[??]", "bold magenta"))
        t.append(" Status: ", style="bold")
        t.append(f"{icon} {s.status}", style=style)
        if s.engine:
            engine_str = f"{s.engine}/{s.model}" if s.model else s.engine
            t.append("    Engine: ", style="bold")
            t.append(engine_str)
        t.append("\n")
        t.append(" Duration: ", style="bold")
        if s.status == "running" and s.started_at is not None:
            elapsed = time.monotonic() - s.started_at
            t.append(format_duration(elapsed))
            # Show timeout proximity warning
            if s.timeout_sec is not None and elapsed > s.timeout_sec * 0.8:
                t.append(" !", style="bold red")
        else:
            t.append(format_duration(s.duration_sec))
        t.append("    Cost: ", style="bold")
        t.append(format_cost(s.cost_usd))
        if s.timeout_sec is not None:
            t.append("    Timeout: ", style="bold")
            t.append(f"{s.timeout_sec}s")
        if s.max_retries > 0:
            t.append("    Retries: ", style="bold")
            t.append(f"{s.max_retries}")
        t.append("\n")
        t.append("─" * 40 + "\n", style="dim")
        if self._log_lines:
            for line in self._log_lines:
                t.append(f" {line}\n", style="dim")
        elif s.status == "running":
            if s.started_at is not None:
                elapsed = time.monotonic() - s.started_at
                t.append(f" Engine working… ({format_duration(elapsed)} elapsed)\n", style="dim italic")
            else:
                t.append(" Engine working…\n", style="dim italic")
            t.append(" Output appears when the engine finishes.\n", style="dim italic")
        else:
            t.append(" (no log output)\n", style="dim italic")
        return t


# --- ApprovalModal ---

class ApprovalModal(Static):
    """Modal overlay for task approval gates."""

    def __init__(self) -> None:
        super().__init__()
        self._task_id: str = ""
        self._message: str = ""
        self._response_event: threading.Event | None = None
        self._approved: bool = False

    def show_approval(
        self,
        task_id: str,
        message: str,
        response_event: threading.Event,
    ) -> None:
        """Show approval dialog for a task."""
        self._task_id = task_id
        self._message = message
        self._response_event = response_event
        self._approved = False
        self.display = True
        self.update(self._render())

    def approve(self) -> None:
        """User approved the task."""
        self._approved = True
        self.display = False
        if self._response_event:
            self._response_event.set()

    def deny(self) -> None:
        """User denied the task."""
        self._approved = False
        self.display = False
        if self._response_event:
            self._response_event.set()

    @property
    def was_approved(self) -> bool:
        return self._approved

    def _render(self) -> Text:  # type: ignore[override]  # Text is a valid Visual at runtime
        t = Text()
        t.append("\n")
        t.append("  APPROVAL REQUIRED\n", style="bold yellow")
        t.append(f"  Task: {self._task_id}\n", style="bold")
        t.append(f"  {self._message}\n\n")
        t.append("  [y] Approve    [n] Deny\n", style="bold")
        return t


# --- EventFeed ---

class EventFeed(RichLog):
    """Scrolling event log at the bottom."""

    can_focus = False

    _OUTPUT_THROTTLE_SEC = 0.15

    def __init__(self) -> None:
        super().__init__(highlight=True, markup=True, max_lines=200)
        self._last_output_time: float = 0.0
        self._pending_line: tuple[str, str] | None = None
        self._last_written: tuple[str, str] | None = None  # dedup

    def write_output(self, payload: dict[str, object]) -> None:
        """Show a live output line from a running task with tree connector."""
        task_id = str(payload.get("task_id", ""))
        line = humanize_output_line(str(payload.get("line", "")).strip(), max_len=120)
        if not line:
            return
        # Dedup consecutive identical output per task
        key = (task_id, line)
        if key == self._last_written:
            return
        now = time.monotonic()
        if now - self._last_output_time < self._OUTPUT_THROTTLE_SEC:
            self._pending_line = key
            return
        self._last_output_time = now
        self._pending_line = None
        self._last_written = key
        self._write_output_line(task_id, line)

    def flush_pending_output(self) -> None:
        """Write any throttled pending line (called on task_complete)."""
        if self._pending_line is not None:
            task_id, line = self._pending_line
            self._pending_line = None
            self._last_written = (task_id, line)
            self._write_output_line(task_id, line)

    def _write_output_line(self, task_id: str, line: str) -> None:
        self.write(Text.assemble(
            ("         ", ""),
            ("├─ ", "dim"),
            (f"[{task_id}] ", "dim cyan"),
            (line, "dim italic"),
        ))

    @staticmethod
    def _format_local_time(ts_value: object) -> str:
        """Convert UTC ISO timestamp to local HH:MM:SS."""
        if not isinstance(ts_value, str) or not ts_value:
            return datetime.now().strftime("%H:%M:%S")
        try:
            ts_str = ts_value.replace("Z", "+00:00")
            dt_utc = datetime.fromisoformat(ts_str)
            dt_local = dt_utc.astimezone()
            return dt_local.strftime("%H:%M:%S")
        except (ValueError, OSError):
            return ts_value[-8:]

    def write_event(self, event_name: str, payload: dict[str, object]) -> None:
        task_id = payload.get("task_id", "")
        ts = self._format_local_time(payload.get("ts"))
        match event_name:
            case "task_start":
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[>>] {task_id}", "bold cyan"), " started"
                ))
            case "task_complete":
                status = str(payload.get("status", ""))
                dur_val = payload.get("duration_sec")
                dur = format_duration(dur_val if isinstance(dur_val, (int, float)) else None)
                cost_val = payload.get("cost_usd")
                cost = format_cost(cost_val if isinstance(cost_val, (int, float)) else None)
                style = "bold green" if status in ("success", "dry_run") else "bold red" if status == "failed" else "bold yellow"
                icon = STATUS_STYLES.get(status, ("[??]", ""))[0]
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"{icon} {task_id}", style), f" ({dur}, {cost})"
                ))
            case "task_skip":
                reason = payload.get("reason", "")
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[--] {task_id}", "dim"), f" skipped: {reason}"
                ))
            case "budget_warning":
                bw_spent = payload.get("spent")
                bw_limit = payload.get("limit")
                bw_spent_f = float(bw_spent) if isinstance(bw_spent, (int, float)) else None
                bw_limit_f = float(bw_limit) if isinstance(bw_limit, (int, float)) else None
                self.write(Text.assemble(
                    (ts, "dim"), " ", ("budget warning", "bold yellow"),
                    f": {format_cost(bw_spent_f)} / {format_cost(bw_limit_f)}"
                ))
            case "run_complete":
                ok = payload.get("ok", 0)
                failed = payload.get("failed", 0)
                style = "bold green" if payload.get("success") else "bold red"
                self.write(Text.assemble(
                    (ts, "dim"), " ", ("run complete", style),
                    f": {ok} ok, {failed} failed"
                ))
            case "task_retry":
                attempt = payload.get("attempt", "?")
                max_r = payload.get("max_retries", "?")
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[..] {task_id}", "bold yellow"),
                    f" retry {attempt}/{max_r}"
                ))
            case "task_escalation":
                from_model = payload.get("from_model", "")
                to_model = payload.get("to_model", "")
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[^^] {task_id}", "bold magenta"),
                    f" escalate {from_model} → {to_model}"
                ))
            case "engine_fallback":
                from_engine = payload.get("from_engine", "")
                to_engine = payload.get("to_engine", "")
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[<>] {task_id}", "bold magenta"),
                    f" fallback {from_engine} → {to_engine}"
                ))
            case "verify_failure":
                exit_code = payload.get("exit_code", "?")
                snippet = str(payload.get("output_snippet", ""))[:60]
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[VF] {task_id}", "bold red"),
                    f" exit={exit_code}: {snippet}"
                ))
            case "judge_start":
                method = payload.get("method", "direct")
                count = payload.get("criteria_count", "?")
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[JJ] {task_id}", "dim cyan"),
                    f" judge ({method}, {count} criteria)"
                ))
            case "judge_verdict":
                jv_verdict = str(payload.get("verdict", ""))
                score = payload.get("score")
                score_str = f" score={score:.2f}" if isinstance(score, (int, float)) else ""
                jv_style = "green" if jv_verdict == "pass" else "red" if jv_verdict == "fail" else "yellow"
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[JJ] {task_id}", "dim cyan"),
                    f" verdict: ", (jv_verdict, jv_style), score_str
                ))
            case "task_tool_call":
                tool = payload.get("tool", "")
                dynamic_parent = payload.get("dynamic_parent")
                if dynamic_parent:
                    self.write(Text.assemble(
                        ("         ", ""),
                        ("├─ ", "dim"),
                        (f"[{dynamic_parent}→{task_id}]", "dim magenta"),
                        f" tool: ", (str(tool), "dim italic"),
                    ))
                else:
                    self.write(Text.assemble(
                        ("         ", ""),
                        ("├─ ", "dim"),
                        (f"[{task_id}]", "dim cyan"),
                        f" tool: ", (str(tool), "dim italic"),
                    ))
            case "context_summarize":
                upstream = payload.get("upstream_id", "")
                self.write(Text.assemble(
                    (ts, "dim"), " ",
                    (f"[CTX] {task_id}", "dim"),
                    f" summarize", f" ← {upstream}" if upstream else "",
                ))
            case "context_compression":
                raw = payload.get("tokens_raw")
                final = payload.get("tokens_final")
                if isinstance(raw, (int, float)) and isinstance(final, (int, float)) and raw > 0:
                    ratio = final / raw
                    self.write(Text.assemble(
                        (ts, "dim"), " ",
                        (f"[CTX] {task_id}", "dim"),
                        f" compression {int(raw)}→{int(final)} ({ratio:.0%})",
                    ))
                else:
                    self.write(Text.assemble(
                        (ts, "dim"), " ",
                        (f"[CTX] {task_id}", "dim"),
                        " compression",
                    ))
            case "watch_start" | "watch_complete" | "iteration_start" | "iteration_complete" | \
                 "metric_recorded" | "regression_detected" | "rollback_executed" | "plateau_detected":
                label = event_name.replace("_", " ")
                detail_parts: list[str] = []
                for key in ("iteration", "metric_value", "best_metric", "status", "action"):
                    val = payload.get(key)
                    if val is not None:
                        detail_parts.append(f"{key}={val}")
                detail_str = " ".join(detail_parts)
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[W] {label}", "bold cyan"),
                    f" {detail_str}" if detail_str else ""
                ))
            case "worktree_create":
                wt_path = payload.get("worktree_path", "")
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[WT] {task_id}", "bold blue"),
                    f" worktree created"
                ))
            case "worktree_merge":
                wt_status = str(payload.get("status", ""))
                files = payload.get("files_changed", [])
                file_count = len(files) if isinstance(files, list) else 0
                wt_review_verdict = str(payload.get("review_verdict", ""))
                overlap_files = payload.get("overlapping_files", [])
                overlap_count = len(overlap_files) if isinstance(overlap_files, list) else 0
                wt_style = "green" if wt_status == "merged" else "red" if wt_status == "conflict" else "dim"
                suffix = f" ({file_count} files)" if file_count else ""
                if overlap_count:
                    suffix += f" [{overlap_count} overlaps]"
                if wt_review_verdict and wt_review_verdict != "safe":
                    suffix += f" review={wt_review_verdict}"
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[WT] {task_id}", "bold blue"),
                    " merge: ", (wt_status, wt_style), suffix,
                ))
            case "worktree_review":
                wr_verdict = str(payload.get("verdict", ""))
                conflicts = payload.get("conflict_files", [])
                conflict_count = len(conflicts) if isinstance(conflicts, list) else 0
                suggestion = str(payload.get("resolution_suggestion", ""))
                wr_style = "yellow" if wr_verdict == "resolvable" else "red" if wr_verdict == "conflict" else "green"
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[WT] {task_id}", "bold blue"),
                    " review: ", (wr_verdict, wr_style),
                    f" ({conflict_count} conflicts)" if conflict_count else "",
                ))
                if suggestion:
                    self.write(Text.assemble(
                        ("    ", "dim"), (suggestion[:140], "dim"),
                    ))
            case "worktree_cleanup":
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[WT] {task_id}", "dim blue"),
                    " worktree cleaned up"
                ))
            case "model_routed":
                resolved = payload.get("resolved", "")
                score = payload.get("complexity_score")
                score_str = f" score={score:.2f}" if isinstance(score, (int, float)) else ""
                self.write(Text.assemble(
                    (ts, "dim"), " ", (f"[AR] {task_id}", "dim green"),
                    f" auto → {resolved}{score_str}"
                ))
            case "dynamic_subplan_start":
                count = payload.get("sub_task_count", "?")
                name = payload.get("sub_plan_name", "")
                self.write(Text.assemble(
                    (ts, "dim"), " ",
                    (f"[DG] {task_id}", "bold magenta"),
                    f" dynamic sub-plan started: {name} ({count} tasks)"
                ))
            case "dynamic_subplan_complete":
                ok = payload.get("success", False)
                count = payload.get("sub_task_count", "?")
                cost_val = payload.get("total_cost_usd")
                cost = format_cost(cost_val if isinstance(cost_val, (int, float)) else None)
                status_str = "OK" if ok else "FAILED"
                style = "bold green" if ok else "bold red"
                self.write(Text.assemble(
                    (ts, "dim"), " ",
                    (f"[DG] {task_id}", "bold magenta"),
                    f" dynamic sub-plan ", (status_str, style),
                    f" ({count} tasks, {cost})"
                ))
            case "task_progress":
                pct = payload.get("pct", "?")
                step = payload.get("step", "")
                suffix = f" — {step}" if step else ""
                self.write(Text.assemble(
                    (ts, "dim"), " ",
                    (f"[>>] {task_id}", "cyan"),
                    f" progress: {pct}%{suffix}",
                ))
            case "task_artifact":
                label = str(payload.get("label") or payload.get("path", ""))
                self.write(Text.assemble(
                    (ts, "dim"), " ",
                    (f"[>>] {task_id}", "cyan"),
                    f" artifact: {label}",
                ))
            case "task_signal_log":
                level = str(payload.get("level", "info"))
                msg = str(payload.get("message", ""))[:120]
                level_style = {"error": "bold red", "warn": "yellow"}.get(level, "dim")
                self.write(Text.assemble(
                    (ts, "dim"), " ",
                    (f"[{level}] {task_id}", level_style),
                    f" {msg}",
                ))
            case "timeout_extended":
                extra = payload.get("additional_sec", "?")
                self.write(Text.assemble(
                    (ts, "dim"), " ",
                    (f"[>>] {task_id}", "yellow"),
                    f" timeout extended +{extra}s",
                ))
            case _:
                label = event_name.replace("_", " ")
                dynamic_parent = payload.get("dynamic_parent")
                if dynamic_parent:
                    self.write(Text.assemble(
                        (ts, "dim"), " ",
                        (f"[{dynamic_parent}→{task_id}]", "dim magenta"),
                        f" {label}" if label else "",
                    ))
                elif task_id:
                    self.write(Text.assemble(
                        (ts, "dim"), " ",
                        (f"{label}", "dim"),
                        (f" {task_id}", "dim cyan"),
                    ))
                elif label:
                    self.write(Text.assemble(
                        (ts, "dim"), " ", (label, "dim"),
                    ))
