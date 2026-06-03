from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal
from textual.message import Message
from textual.widgets import Footer

from ..models import PlanSpec, PlanRunResult, STATUS_STYLES, TERMINAL_STATUSES, ExecutionProfile
from ..utils import format_duration, format_cost
from .widgets import PlanHeader, DAGPanel, DetailPanel, EventFeed, ApprovalModal


class PlanEvent(Message):
    """Non-blocking event bridge from executor thread to main thread.

    Uses ``post_message`` (fire-and-forget) instead of ``call_from_thread``
    (blocking) to avoid starving the keyboard event queue.
    """

    def __init__(self, event_name: str, payload: dict[str, object]) -> None:
        super().__init__()
        self.event_name = event_name
        self.payload = payload


class MaestroApp(App[None]):
    """Maestro CLI interactive TUI."""

    CSS_PATH = "app.tcss"

    BINDINGS = [
        Binding("q", "quit_app", "Quit"),
        Binding("up,k", "cursor_up", "Up", show=False, priority=True),
        Binding("down,j", "cursor_down", "Down", show=False, priority=True),
        Binding("enter", "select_task", "Select", show=False, priority=True),
        Binding("escape", "deselect", "Back", show=False, priority=True),
        Binding("y", "approve", "Approve", show=False),
        Binding("n", "deny", "Deny", show=False),
        Binding("f", "cycle_filter", "Filter"),
        Binding("t", "toggle_follow", "Follow"),
    ]

    def __init__(
        self,
        plan: PlanSpec,
        *,
        dry_run: bool = False,
        execution_profile: str = "plan",
        max_parallel_override: int | None = None,
        run_dir_override: str | None = None,
        auto_approve: bool = False,
        resume_path: Path | None = None,
        cache_dir: Path | None = None,
        only: set[str] | None = None,
        skip: set[str] | None = None,
        tags: set[str] | None = None,
        skip_tags: set[str] | None = None,
        webhook_url: str | None = None,
        extra_template_vars: dict[str, str] | None = None,
    ) -> None:
        super().__init__()
        self._plan = plan
        self._dry_run = dry_run
        self._execution_profile = execution_profile
        self._max_parallel_override = max_parallel_override
        self._run_dir_override = run_dir_override
        self._auto_approve = auto_approve
        self._resume_path = resume_path
        self._cache_dir = cache_dir
        self._only = only
        self._skip = skip
        self._tags = tags
        self._skip_tags = skip_tags
        self._webhook_url = webhook_url
        self._extra_template_vars = extra_template_vars
        self._cancel_event = threading.Event()
        self._result: PlanRunResult | None = None
        self._run_path: Path | None = None
        self._selected_task: str | None = None
        self._quit_requested = False
        self._last_approval_result: bool = False
        self._pending_approval_event: threading.Event | None = None

    def compose(self) -> ComposeResult:
        yield PlanHeader(self._plan)
        with Horizontal(id="main-area"):
            yield DAGPanel(self._plan)
            yield DetailPanel()
        yield EventFeed()
        yield Footer()
        yield ApprovalModal()

    def on_mount(self) -> None:
        self.run_worker(self._run_plan_worker, thread=True, name="plan_executor")
        # Auto-select first task so detail panel is never empty
        self._auto_select_cursor()

    def _run_plan_worker(self) -> PlanRunResult:
        """Blocking worker — runs in a separate OS thread."""
        from ..scheduler import run_plan

        result = run_plan(
            self._plan,
            dry_run=self._dry_run,
            execution_profile=cast(ExecutionProfile, self._execution_profile),
            max_parallel_override=self._max_parallel_override,
            run_dir_override=self._run_dir_override,
            auto_approve=self._auto_approve,
            resume_path=self._resume_path,
            cache_dir=self._cache_dir,
            only=self._only,
            skip=self._skip,
            tags=self._tags,
            skip_tags=self._skip_tags,
            webhook_url=self._webhook_url,
            verbosity="quiet",
            output_mode="text",
            event_callback=self._on_event,
            cancel_event=self._cancel_event,
            approval_handler=self._approval_handler,
            extra_template_vars=self._extra_template_vars,
        )
        self._result = result
        return result

    def _on_event(self, event_name: str, payload: dict[str, object]) -> None:
        """Called from executor thread — bridge to main thread.

        Uses ``post_message`` (non-blocking, fire-and-forget) instead of
        ``call_from_thread`` (synchronous, blocks executor until callback
        completes on main thread).  This prevents the convoy effect where
        rapid events starve the keyboard event queue.
        """
        self.post_message(PlanEvent(event_name, payload))

    def on_plan_event(self, message: PlanEvent) -> None:
        """Handle plan events on the main thread (posted by _on_event)."""
        with self.batch_update():
            self._dispatch_event(message.event_name, message.payload)

    def _approval_handler(self, task_id: str, message: str | None) -> bool:
        """Called from executor thread when task needs approval."""
        if self._auto_approve:
            return True
        response_event = threading.Event()
        self.call_from_thread(
            self._show_approval,
            task_id,
            message or "Approval required.",
            response_event,
        )
        response_event.wait()
        return self._last_approval_result

    def _show_approval(
        self,
        task_id: str,
        message: str,
        response_event: threading.Event,
    ) -> None:
        """Main thread — show the modal."""
        modal = self.query_one(ApprovalModal)
        self._pending_approval_event = response_event
        modal.show_approval(task_id, message, response_event)

    def select_task(self, task_id: str) -> None:
        """Select a task to show in DetailPanel."""
        dag = self.query_one(DAGPanel)
        detail = self.query_one(DetailPanel)
        state = dag._states.get(task_id)
        if state is not None:
            self._selected_task = task_id
            detail.select_task(task_id, state, self._run_path)

    def clear_task_selection(self) -> None:
        """Clear task selection."""
        self._selected_task = None
        self.query_one(DetailPanel).clear_selection()

    def action_cursor_up(self) -> None:
        self.query_one(DAGPanel).move_cursor(-1)
        self._auto_select_cursor()

    def action_cursor_down(self) -> None:
        self.query_one(DAGPanel).move_cursor(1)
        self._auto_select_cursor()

    def _auto_select_cursor(self) -> None:
        """Auto-select the task under the cursor for the detail panel."""
        dag = self.query_one(DAGPanel)
        task_id = dag.get_cursor_task_id()
        if task_id and task_id != self._selected_task:
            self.select_task(task_id)

    def action_select_task(self) -> None:
        dag = self.query_one(DAGPanel)
        task_id = dag.get_cursor_task_id()
        if task_id:
            self.select_task(task_id)

    def action_deselect(self) -> None:
        modal = self.query_one(ApprovalModal)
        if modal.display:
            return
        if self._selected_task:
            self.clear_task_selection()
        else:
            dag = self.query_one(DAGPanel)
            dag.clear_filter()

    def action_cycle_filter(self) -> None:
        self.query_one(DAGPanel).cycle_filter()

    def action_toggle_follow(self) -> None:
        self.query_one(DAGPanel).toggle_follow()

    def action_approve(self) -> None:
        modal = self.query_one(ApprovalModal)
        if modal.display:
            self._last_approval_result = True
            self._pending_approval_event = None
            modal.approve()

    def action_deny(self) -> None:
        modal = self.query_one(ApprovalModal)
        if modal.display:
            self._last_approval_result = False
            self._pending_approval_event = None
            modal.deny()

    def _dispatch_event(self, event_name: str, payload: dict[str, object]) -> None:
        """Now on main thread — safe to update widgets."""
        header = self.query_one(PlanHeader)
        dag = self.query_one(DAGPanel)
        feed = self.query_one(EventFeed)

        match event_name:
            case "run_start":
                run_path = payload.get("run_path")
                if isinstance(run_path, str):
                    self._run_path = Path(run_path)
            case "task_start":
                dag.update_task_start(payload)
                header.increment_running()
                if dag._follow:
                    task_id = str(payload.get("task_id", ""))
                    dag.move_cursor_to(task_id)
            case "task_complete":
                dag.update_task_complete(payload)
                header.task_completed(payload)
                feed.flush_pending_output()
            case "task_skip":
                dag.update_task_skip(payload)
                header.task_completed(payload)
            case "task_retry":
                dag.update_task_retry(payload)
            case "task_output":
                dag.update_task_output(payload)
                feed.write_output(payload)
                return
            case "task_tool_call":
                # Show tool calls as output lines + event feed
                tool = payload.get("tool", "")
                if tool:
                    feed.write_output({
                        "task_id": payload.get("task_id", ""),
                        "line": f"tool: {tool}",
                    })
            case "task_progress":
                dag.update_task_progress(payload)
            case "task_artifact":
                pass  # displayed in feed below
            case "budget_warning":
                header.show_budget_warning(payload)
            case "run_complete":
                header.run_completed(payload)

        # Only update DetailPanel when the event is for the selected task
        if self._selected_task:
            event_task_id = str(payload.get("task_id", ""))
            if event_task_id == self._selected_task:
                detail = self.query_one(DetailPanel)
                state = dag._states.get(self._selected_task)
                if state is not None:
                    detail.update_state(state)

        feed.write_event(event_name, payload)

    def on_worker_state_changed(self, event: Any) -> None:
        """Handle worker completion."""
        if event.worker.name == "plan_executor" and event.worker.is_finished:
            # Plan is done — update header to show final state
            pass

    def action_quit_app(self) -> None:
        """Handle q key."""
        if self._result is not None:
            self.exit(return_code=0 if self._result.success else 1)
        else:
            self._cancel_event.set()
            if self._quit_requested:
                self.exit(return_code=1)
                return
            self.notify("Cancelling plan… press q again to force quit", severity="warning")
            self._quit_requested = True
            self.set_timer(3.0, self._reset_quit_request)

    def _reset_quit_request(self) -> None:
        self._quit_requested = False
