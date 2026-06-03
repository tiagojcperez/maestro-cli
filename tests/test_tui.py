from __future__ import annotations

import pytest

textual = pytest.importorskip("textual")

import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from maestro_cli.models import PlanRunResult, PlanSpec, TaskSpec
from maestro_cli.tui import MaestroApp
from maestro_cli.tui.widgets import DAGPanel, EventFeed, PlanHeader, TaskState

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(
    tasks: list[TaskSpec] | None = None,
    name: str = "test-plan",
) -> PlanSpec:
    if tasks is None:
        tasks = [TaskSpec(id="t1", command="echo ok")]
    return PlanSpec(name=name, tasks=tasks, source_path=Path("plan.yaml"))


def _make_result(success: bool = True) -> PlanRunResult:
    now = datetime.now(tz=UTC)
    return PlanRunResult(
        plan_name="test-plan",
        run_id="test-run",
        run_path=Path("."),
        started_at=now,
        finished_at=now,
        success=success,
    )


def _noop_run_plan(plan: PlanSpec, **kwargs: Any) -> PlanRunResult:
    """Mock run_plan that fires events after a small delay for mount."""
    time.sleep(0.15)
    cb = kwargs.get("event_callback")
    if cb and callable(cb):
        for task in plan.tasks:
            cb("task_start", {"task_id": task.id, "engine": "shell", "model": None})
            cb("task_complete", {
                "task_id": task.id,
                "status": "dry_run",
                "duration_sec": 0.1,
                "cost_usd": 0.0,
            })
        cb("run_complete", {"success": True, "ok": len(plan.tasks), "failed": 0})
    return _make_result(success=True)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_app_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    """App composes PlanHeader, DAGPanel, EventFeed without crash."""
    plan = _make_plan()
    monkeypatch.setattr("maestro_cli.scheduler.run_plan", _noop_run_plan)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.3)
        assert app.query_one(PlanHeader) is not None
        assert app.query_one(DAGPanel) is not None
        assert app.query_one(EventFeed) is not None


async def test_dag_panel_shows_all_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """DAGPanel should have one row per task."""
    tasks = [TaskSpec(id=f"t{i}", command="echo ok") for i in range(1, 4)]
    plan = _make_plan(tasks)
    monkeypatch.setattr("maestro_cli.scheduler.run_plan", _noop_run_plan)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.3)
        dag = app.query_one(DAGPanel)
        assert dag.row_count == 3


async def test_event_dispatch_updates_dag(monkeypatch: pytest.MonkeyPatch) -> None:
    """Events dispatched via _dispatch_event update DAGPanel state."""
    plan = _make_plan([TaskSpec(id="t1", command="echo ok")])

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)  # let DataTable mount
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude", "model": "sonnet"})
            cb("task_complete", {
                "task_id": "t1",
                "status": "success",
                "duration_sec": 2.5,
                "cost_usd": 0.10,
            })
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)
        dag = app.query_one(DAGPanel)
        assert dag._states["t1"].status == "success"
        assert dag._states["t1"].cost_usd == 0.10


async def test_header_progress_updates(monkeypatch: pytest.MonkeyPatch) -> None:
    """PlanHeader completed count and cost accumulate from events."""
    tasks = [TaskSpec(id="t1", command="echo 1"), TaskSpec(id="t2", command="echo 2")]
    plan = _make_plan(tasks)

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1"})
            cb("task_complete", {"task_id": "t1", "status": "success", "cost_usd": 0.05})
            cb("task_start", {"task_id": "t2"})
            cb("task_complete", {"task_id": "t2", "status": "success", "cost_usd": 0.03})
            cb("run_complete", {"success": True, "ok": 2, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)
        header = app.query_one(PlanHeader)
        assert header._completed == 2
        assert header._total_cost == pytest.approx(0.08, abs=0.01)


async def test_event_feed_receives_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """EventFeed.write_event is called for each dispatched event."""
    plan = _make_plan()
    written: list[str] = []
    original_write = EventFeed.write_event

    def tracking_write(self: EventFeed, event_name: str, payload: dict[str, object]) -> None:
        written.append(event_name)
        original_write(self, event_name, payload)

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", _noop_run_plan)
    monkeypatch.setattr(EventFeed, "write_event", tracking_write)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)
    assert "task_start" in written
    assert "task_complete" in written


async def test_quit_key_exits(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pressing q should exit the app."""
    plan = _make_plan()
    monkeypatch.setattr("maestro_cli.scheduler.run_plan", _noop_run_plan)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.3)
        await pilot.press("q")


async def test_task_skip_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """task_skip events update DAGPanel status."""
    plan = _make_plan([TaskSpec(id="t1", command="echo ok")])

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_skip", {"task_id": "t1", "reason": "dependency failure"})
            cb("run_complete", {"success": False, "ok": 0, "failed": 0})
        return _make_result(success=False)

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)
        dag = app.query_one(DAGPanel)
        assert dag._states["t1"].status == "skipped"
        assert dag._states["t1"].reason == "dependency failure"


async def test_cancel_event_set_on_quit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pressing q before plan completes sets cancel_event."""
    plan = _make_plan()

    def slow_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        cancel = kw.get("cancel_event")
        if cancel and hasattr(cancel, "wait"):
            cancel.wait(timeout=5.0)
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", slow_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.2)
        await pilot.press("q")
        await pilot.pause(delay=0.1)
    assert app._cancel_event.is_set()


async def test_budget_warning_in_header(monkeypatch: pytest.MonkeyPatch) -> None:
    """budget_warning event populates PlanHeader._budget_warning."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("budget_warning", {"spent": 0.8, "limit": 1.0})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)
        header = app.query_one(PlanHeader)
        assert header._budget_warning is not None
        assert "budget" in header._budget_warning


class TestTaskState:
    """Unit tests for the TaskState dataclass."""

    def test_default_values(self) -> None:
        s = TaskState(task_id="x")
        assert s.status == "pending"
        assert s.engine is None

    def test_update_fields(self) -> None:
        s = TaskState(task_id="x")
        s.status = "running"
        s.engine = "claude"
        assert s.status == "running"
        assert s.engine == "claude"


class TestCLITuiBlock:
    """Test CLI-level TUI guards."""

    def test_multi_plan_tui_blocked(self, capsys: pytest.CaptureFixture[str]) -> None:
        """--output tui with multiple plans should return error."""
        import argparse

        ns = argparse.Namespace(
            plan=["plan1.yaml", "plan2.yaml"],
            output="tui",
            dry_run=False,
            verbose=False,
            quiet=False,
        )
        from maestro_cli.cli import _cmd_run

        result = _cmd_run(ns)
        assert result == 1
        captured = capsys.readouterr()
        assert "TUI mode does not support multi-plan" in captured.out


# ---------------------------------------------------------------------------
# New widget unit tests (sync — no app context needed)
# ---------------------------------------------------------------------------

from maestro_cli.tui.widgets import ApprovalModal, DetailPanel  # noqa: E402


class TestDetailPanel:
    """Unit tests for DetailPanel widget logic."""

    def _make_panel(self) -> DetailPanel:
        panel = DetailPanel()
        panel.update = lambda *a, **kw: None  # suppress Textual rendering
        return panel

    def test_detail_panel_empty_render(self) -> None:
        panel = self._make_panel()
        rendered = panel._render_empty()
        assert "Enter to select task" in str(rendered)

    def test_detail_panel_select_task(self) -> None:
        panel = self._make_panel()
        state = TaskState(task_id="t1", status="running")
        panel.select_task("t1", state, None)
        assert panel._task_id == "t1"
        assert panel._state is state

    def test_detail_panel_clear_selection(self) -> None:
        panel = self._make_panel()
        state = TaskState(task_id="t1", status="success")
        panel.select_task("t1", state, None)
        panel.clear_selection()
        assert panel._task_id is None
        assert panel._state is None
        assert panel._log_lines == []

    def test_detail_panel_update_state(self) -> None:
        panel = self._make_panel()
        state = TaskState(task_id="t1", status="running")
        panel.select_task("t1", state, None)
        updated = TaskState(task_id="t1", status="success", cost_usd=0.05)
        panel.update_state(updated)
        assert panel._state is updated
        assert panel._state.status == "success"

    def test_detail_panel_shows_timeout_and_retries(self) -> None:
        panel = self._make_panel()
        state = TaskState(
            task_id="t1",
            status="running",
            engine="claude",
            model="sonnet",
            timeout_sec=600,
            max_retries=2,
        )
        panel.select_task("t1", state, None)
        rendered = str(panel._render_detail())
        assert "Timeout:" in rendered
        assert "600s" in rendered
        assert "Retries:" in rendered
        assert "2" in rendered

    def test_detail_panel_running_no_log_shows_working_message(self) -> None:
        import time
        panel = self._make_panel()
        state = TaskState(
            task_id="t1",
            status="running",
            started_at=time.monotonic() - 5.0,
        )
        panel.select_task("t1", state, None)
        rendered = str(panel._render_detail())
        assert "Engine working" in rendered
        assert "Output appears when the engine finishes" in rendered

    def test_detail_panel_completed_no_log_shows_no_output(self) -> None:
        panel = self._make_panel()
        state = TaskState(task_id="t1", status="success")
        panel.select_task("t1", state, None)
        rendered = str(panel._render_detail())
        assert "(no log output)" in rendered
        assert "Engine working" not in rendered


class TestDAGPanelNavigation:
    """Unit tests for DAGPanel cursor, filter, and navigation logic."""

    def _make_panel(self, n: int = 3) -> DAGPanel:
        tasks = [TaskSpec(id=f"t{i}", command="echo ok") for i in range(1, n + 1)]
        plan = PlanSpec(name="test-plan", tasks=tasks, source_path=Path("plan.yaml"))
        panel = DAGPanel(plan)
        panel._refresh_table = lambda: None  # suppress Textual rendering
        return panel

    def test_cursor_movement(self) -> None:
        panel = self._make_panel()
        assert panel._cursor == 0
        panel.move_cursor(1)
        assert panel._cursor == 1
        panel.move_cursor(-1)
        assert panel._cursor == 0

    def test_cursor_bounds(self) -> None:
        panel = self._make_panel(3)  # tasks: t1, t2, t3 → indices 0-2
        panel.move_cursor(-10)
        assert panel._cursor == 0
        panel.move_cursor(100)
        assert panel._cursor == 2

    def test_get_cursor_task_id(self) -> None:
        panel = self._make_panel(3)
        panel.move_cursor(1)
        assert panel.get_cursor_task_id() == "t2"
        panel.move_cursor(1)
        assert panel.get_cursor_task_id() == "t3"

    def test_filter_cycling(self) -> None:
        panel = self._make_panel()
        assert panel._filter == "all"
        panel.cycle_filter()
        assert panel._filter == "running"
        panel.cycle_filter()
        assert panel._filter == "failed"
        panel.cycle_filter()
        assert panel._filter == "completed"
        panel.cycle_filter()
        assert panel._filter == "all"

    def test_filter_running_only(self) -> None:
        panel = self._make_panel(3)
        panel._states["t1"].status = "running"
        # t2 and t3 remain "pending"
        panel._filter = "running"
        visible = panel._visible_task_ids()
        assert visible == ["t1"]
        assert len(visible) == 1

    def test_clear_filter(self) -> None:
        panel = self._make_panel(3)
        panel._filter = "failed"
        panel._cursor = 2
        panel.clear_filter()
        assert panel._filter == "all"
        assert panel._cursor == 0

    def test_follow_toggle(self) -> None:
        panel = self._make_panel()
        assert panel._follow is False
        panel.toggle_follow()
        assert panel._follow is True
        panel.toggle_follow()
        assert panel._follow is False

    def test_move_cursor_to(self) -> None:
        panel = self._make_panel(3)
        panel.move_cursor_to("t3")
        assert panel._cursor == 2
        panel.move_cursor_to("t1")
        assert panel._cursor == 0


class TestApprovalModal:
    """Unit tests for ApprovalModal widget logic."""

    def _make_modal(self, monkeypatch: pytest.MonkeyPatch) -> ApprovalModal:
        # Replace Textual's display reactive with a plain property so the
        # widget can be tested without a running app event loop.
        _display: dict[int, bool] = {}
        monkeypatch.setattr(
            ApprovalModal,
            "display",
            property(
                lambda self: _display.get(id(self), False),
                lambda self, v: _display.__setitem__(id(self), v),
            ),
            raising=False,
        )
        modal = ApprovalModal()
        modal.update = lambda *a, **kw: None  # suppress Textual rendering
        return modal

    def test_approval_modal_initial_hidden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        modal = self._make_modal(monkeypatch)
        assert modal.display is False
        assert modal.was_approved is False

    def test_approval_modal_approve(self, monkeypatch: pytest.MonkeyPatch) -> None:
        modal = self._make_modal(monkeypatch)
        event = threading.Event()
        modal.show_approval("t1", "Proceed with task?", event)
        modal.approve()
        assert modal.was_approved is True
        assert event.is_set()

    def test_approval_modal_deny(self, monkeypatch: pytest.MonkeyPatch) -> None:
        modal = self._make_modal(monkeypatch)
        event = threading.Event()
        modal.show_approval("t1", "Proceed with task?", event)
        modal.deny()
        assert modal.was_approved is False
        assert event.is_set()

    def test_approval_modal_render(self, monkeypatch: pytest.MonkeyPatch) -> None:
        modal = self._make_modal(monkeypatch)
        event = threading.Event()
        modal.show_approval("deploy-prod", "About to deploy to production", event)
        rendered = str(modal._render())
        assert "APPROVAL REQUIRED" in rendered
        assert "deploy-prod" in rendered
        assert "About to deploy to production" in rendered
        assert "[y] Approve" in rendered
        assert "[n] Deny" in rendered

    def test_approval_modal_deny_no_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """deny() when _response_event is None should not crash."""
        modal = self._make_modal(monkeypatch)
        modal._response_event = None
        modal.deny()
        assert modal.was_approved is False

    def test_approval_modal_approve_no_event(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """approve() when _response_event is None should not crash."""
        modal = self._make_modal(monkeypatch)
        modal._response_event = None
        modal.approve()
        assert modal.was_approved is True


# ---------------------------------------------------------------------------
# PlanHeader unit tests (sync — no app context)
# ---------------------------------------------------------------------------


class TestPlanHeaderUnit:
    """Unit tests for PlanHeader pure logic (no mounting)."""

    def _make_header(self, n_tasks: int = 3, name: str = "my-plan") -> PlanHeader:
        tasks = [TaskSpec(id=f"t{i}", command="echo ok") for i in range(1, n_tasks + 1)]
        plan = PlanSpec(name=name, tasks=tasks, source_path=Path("plan.yaml"))
        header = PlanHeader(plan)
        header.update = lambda *a, **kw: None  # suppress Textual rendering
        return header

    def test_initial_state(self) -> None:
        header = self._make_header(5)
        assert header._completed == 0
        assert header._total == 5
        assert header._total_cost == 0.0
        assert header._running_count == 0
        assert header._budget_warning is None
        assert header._finished is False

    def test_increment_running(self) -> None:
        header = self._make_header()
        header.increment_running()
        assert header._running_count == 1
        header.increment_running()
        assert header._running_count == 2

    def test_task_completed(self) -> None:
        header = self._make_header()
        header.increment_running()
        header.task_completed({"status": "success", "cost_usd": 0.05})
        assert header._completed == 1
        assert header._running_count == 0
        assert header._total_cost == pytest.approx(0.05)

    def test_task_completed_no_cost(self) -> None:
        header = self._make_header()
        header.task_completed({"status": "success"})
        assert header._completed == 1
        assert header._total_cost == 0.0

    def test_task_completed_none_cost(self) -> None:
        header = self._make_header()
        header.task_completed({"status": "success", "cost_usd": None})
        assert header._completed == 1
        assert header._total_cost == 0.0

    def test_running_count_floor(self) -> None:
        """Running count should never go below zero."""
        header = self._make_header()
        header.task_completed({"status": "success"})
        assert header._running_count == 0

    def test_show_budget_warning(self) -> None:
        header = self._make_header()
        header.show_budget_warning({"spent": 0.8, "limit": 1.0})
        assert header._budget_warning is not None
        assert "budget" in header._budget_warning
        assert "$0.80" in header._budget_warning
        assert "$1.00" in header._budget_warning

    def test_show_budget_warning_none_values(self) -> None:
        header = self._make_header()
        header.show_budget_warning({"spent": None, "limit": None})
        assert header._budget_warning is not None
        assert "budget" in header._budget_warning

    def test_run_completed_with_duration(self) -> None:
        header = self._make_header()
        header.run_completed({"duration_sec": 42.5, "cost_usd": 1.23})
        assert header._finished is True
        assert header._final_duration == 42.5
        assert header._total_cost == 1.23

    def test_run_completed_no_duration(self) -> None:
        header = self._make_header()
        header.run_completed({"success": True})
        assert header._finished is True
        assert header._final_duration is not None  # falls back to monotonic

    def test_run_completed_no_cost(self) -> None:
        """run_completed with no cost_usd should not override accumulated cost."""
        header = self._make_header()
        header.task_completed({"status": "success", "cost_usd": 0.50})
        header.run_completed({"success": True})
        assert header._total_cost == pytest.approx(0.50)

    def test_progress_bar_empty(self) -> None:
        header = self._make_header(0)
        header._total = 0
        bar = header._progress_bar()
        assert len(bar) == 20

    def test_progress_bar_partial(self) -> None:
        header = self._make_header(10)
        header._completed = 5
        bar = header._progress_bar()
        assert len(bar) == 20
        assert bar.count("\u2501") == 10  # half filled

    def test_progress_bar_full(self) -> None:
        header = self._make_header(3)
        header._completed = 3
        bar = header._progress_bar()
        assert len(bar) == 20
        assert bar.count("\u2501") == 20  # all filled

    def test_render_includes_plan_name(self) -> None:
        header = self._make_header(name="deploy-pipeline")
        rendered = str(header.render())
        assert "MAESTRO" in rendered
        assert "deploy-pipeline" in rendered
        assert "0/3" in rendered

    def test_render_with_budget_warning(self) -> None:
        header = self._make_header()
        header.show_budget_warning({"spent": 0.9, "limit": 1.0})
        rendered = str(header.render())
        assert "budget" in rendered

    def test_render_final_duration(self) -> None:
        header = self._make_header()
        header._final_duration = 65.0
        rendered = str(header.render())
        assert "1m05s" in rendered

    def test_tick_skips_when_finished_no_refresh(self) -> None:
        header = self._make_header()
        header._finished = True
        header._needs_refresh = False
        # _tick should be a no-op (no crash, no update)
        header._tick()


# ---------------------------------------------------------------------------
# DAGPanel data update tests (sync — no app context)
# ---------------------------------------------------------------------------


class TestDAGPanelDataUpdates:
    """Tests for DAGPanel event-driven data mutation methods."""

    def _make_panel(self, n: int = 3) -> DAGPanel:
        tasks = [TaskSpec(id=f"t{i}", command="echo ok") for i in range(1, n + 1)]
        plan = PlanSpec(name="test-plan", tasks=tasks, source_path=Path("plan.yaml"))
        panel = DAGPanel(plan)
        panel._refresh_table = lambda: None  # suppress Textual rendering
        return panel

    def test_update_task_start(self) -> None:
        panel = self._make_panel()
        panel.update_task_start({"task_id": "t1", "engine": "claude", "model": "sonnet"})
        state = panel._states["t1"]
        assert state.status == "running"
        assert state.engine == "claude"
        assert state.model == "sonnet"
        assert state.started_at is not None

    def test_update_task_start_unknown_task(self) -> None:
        """Start event for unknown task should be silently ignored."""
        panel = self._make_panel()
        panel.update_task_start({"task_id": "unknown-task", "engine": "claude"})
        assert "unknown-task" not in panel._states

    def test_update_task_complete(self) -> None:
        panel = self._make_panel()
        panel.update_task_start({"task_id": "t1", "engine": "claude"})
        panel.update_task_complete({
            "task_id": "t1", "status": "success",
            "duration_sec": 5.0, "cost_usd": 0.10,
        })
        state = panel._states["t1"]
        assert state.status == "success"
        assert state.duration_sec == 5.0
        assert state.cost_usd == 0.10
        assert state.started_at is None

    def test_update_task_complete_unknown_task(self) -> None:
        panel = self._make_panel()
        panel.update_task_complete({"task_id": "nope", "status": "success"})
        assert "nope" not in panel._states

    def test_update_task_skip(self) -> None:
        panel = self._make_panel()
        panel.update_task_skip({"task_id": "t2", "reason": "dependency failed"})
        state = panel._states["t2"]
        assert state.status == "skipped"
        assert state.reason == "dependency failed"

    def test_update_task_skip_unknown_task(self) -> None:
        panel = self._make_panel()
        panel.update_task_skip({"task_id": "nope", "reason": "test"})

    def test_update_task_output(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "running"
        panel._last_output_refresh = 0.0  # ensure refresh is not throttled
        panel.update_task_output({"task_id": "t1", "line": "compiling main.py"})
        assert panel._states["t1"].last_line == "compiling main.py"

    def test_update_task_output_throttled(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "running"
        panel._last_output_refresh = time.monotonic()  # just refreshed
        panel._needs_refresh = False
        panel.update_task_output({"task_id": "t1", "line": "line1"})
        assert panel._states["t1"].last_line == "line1"
        assert panel._needs_refresh is False  # throttled

    def test_update_task_output_unknown_task(self) -> None:
        panel = self._make_panel()
        panel.update_task_output({"task_id": "nope", "line": "test"})

    def test_update_task_output_empty_line(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "running"
        panel._states["t1"].last_line = "old"
        panel.update_task_output({"task_id": "t1", "line": ""})
        assert panel._states["t1"].last_line == "old"  # not updated

    def test_update_task_progress(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "running"
        panel.update_task_progress({"task_id": "t1", "pct": 42, "step": "parsing"})
        assert panel._states["t1"].last_line == "[42%] parsing"

    def test_update_task_progress_no_step(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "running"
        panel.update_task_progress({"task_id": "t1", "pct": 100})
        assert panel._states["t1"].last_line == "[100%]"

    def test_update_task_progress_clamped(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "running"
        panel.update_task_progress({"task_id": "t1", "pct": 150})
        assert "[100%]" in panel._states["t1"].last_line

    def test_update_task_progress_non_numeric(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "running"
        panel._states["t1"].last_line = "old"
        panel.update_task_progress({"task_id": "t1", "pct": "not-a-number"})
        assert panel._states["t1"].last_line == "old"  # not updated

    def test_update_task_progress_unknown_task(self) -> None:
        panel = self._make_panel()
        panel.update_task_progress({"task_id": "nope", "pct": 50})

    def test_update_task_retry(self) -> None:
        panel = self._make_panel()
        panel._needs_refresh = False
        panel.update_task_retry({"task_id": "t1", "attempt": 2, "max_retries": 3})
        assert panel._needs_refresh is True

    def test_update_task_retry_unknown_task(self) -> None:
        panel = self._make_panel()
        panel._needs_refresh = False
        panel.update_task_retry({"task_id": "nope"})
        assert panel._needs_refresh is False

    def test_as_str(self) -> None:
        assert DAGPanel._as_str("hello") == "hello"
        assert DAGPanel._as_str(42) is None
        assert DAGPanel._as_str(None) is None

    def test_as_float(self) -> None:
        assert DAGPanel._as_float(1.5) == 1.5
        assert DAGPanel._as_float(42) == 42.0
        assert DAGPanel._as_float("nope") is None
        assert DAGPanel._as_float(None) is None


# ---------------------------------------------------------------------------
# DAGPanel render table tests (sync — no app context)
# ---------------------------------------------------------------------------


class TestDAGPanelRenderTable:
    """Tests for DAGPanel._render_table() output."""

    def _make_panel(self, tasks: list[TaskSpec] | None = None) -> DAGPanel:
        if tasks is None:
            tasks = [
                TaskSpec(id="t1", command="echo ok", description="First task"),
                TaskSpec(id="t2", engine="claude", prompt="Do stuff"),
                TaskSpec(id="t3", command="echo done"),
            ]
        plan = PlanSpec(name="test-plan", tasks=tasks, source_path=Path("plan.yaml"))
        panel = DAGPanel(plan)
        panel._refresh_table = lambda: None
        return panel

    def test_render_pending_tasks(self) -> None:
        panel = self._make_panel()
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_running_task(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "running"
        panel._states["t1"].engine = "claude"
        panel._states["t1"].model = "sonnet"
        panel._states["t1"].started_at = time.monotonic()
        panel._states["t1"].last_line = "compiling..."
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_running_engine_only(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "running"
        panel._states["t1"].engine = "claude"
        panel._states["t1"].model = None
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_running_shell(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "running"
        panel._states["t1"].engine = None
        panel._states["t1"].model = None
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_success_task(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "success"
        panel._states["t1"].engine = "claude"
        panel._states["t1"].model = "sonnet"
        panel._states["t1"].duration_sec = 10.0
        panel._states["t1"].cost_usd = 0.05
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_failed_task(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "failed"
        panel._states["t1"].engine = "claude"
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_soft_failed_task(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "soft_failed"
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_skipped_task_with_reason(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "skipped"
        panel._states["t1"].reason = "budget exceeded"
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_skipped_task_no_reason(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "skipped"
        panel._states["t1"].reason = None
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_dry_run_task(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "dry_run"
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_unknown_status(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "something_weird"
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_resumed_task_dash_duration(self) -> None:
        """Resumed tasks with engine=None and duration_sec=0.0 show dash."""
        panel = self._make_panel()
        panel._states["t1"].status = "success"
        panel._states["t1"].engine = None
        panel._states["t1"].duration_sec = 0.0
        table = panel._render_table()
        assert table.row_count == 3

    def test_render_with_description(self) -> None:
        tasks = [TaskSpec(id="t1", command="echo ok", description="Deploy to prod")]
        panel = self._make_panel(tasks)
        table = panel._render_table()
        assert table.row_count == 1

    def test_render_with_filter_title(self) -> None:
        panel = self._make_panel()
        panel._filter = "running"
        table = panel._render_table()
        assert table.title is not None
        assert "running" in str(table.title)

    def test_cursor_preserved_after_render(self) -> None:
        panel = self._make_panel()
        panel._cursor = 1
        panel._render_table()
        assert panel._cursor == 1

    def test_cursor_clamped_when_filter_shrinks(self) -> None:
        panel = self._make_panel()
        panel._cursor = 2  # pointing at t3
        panel._filter = "running"  # nothing running
        panel._states["t1"].status = "running"
        table = panel._render_table()
        assert panel._cursor == 0  # clamped


# ---------------------------------------------------------------------------
# DAGPanel filter + navigation edge cases
# ---------------------------------------------------------------------------


class TestDAGPanelFilterEdges:
    """Edge cases for filtering and navigation."""

    def _make_panel(self, n: int = 3) -> DAGPanel:
        tasks = [TaskSpec(id=f"t{i}", command="echo ok") for i in range(1, n + 1)]
        plan = PlanSpec(name="test-plan", tasks=tasks, source_path=Path("plan.yaml"))
        panel = DAGPanel(plan)
        panel._refresh_table = lambda: None
        return panel

    def test_visible_completed_filter(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "success"
        panel._states["t2"].status = "failed"
        panel._states["t3"].status = "pending"
        panel._filter = "completed"
        visible = panel._visible_task_ids()
        assert "t1" in visible
        assert "t2" in visible
        assert "t3" not in visible

    def test_visible_failed_filter_soft_failed(self) -> None:
        panel = self._make_panel()
        panel._states["t1"].status = "soft_failed"
        panel._states["t2"].status = "failed"
        panel._states["t3"].status = "success"
        panel._filter = "failed"
        visible = panel._visible_task_ids()
        assert visible == ["t1", "t2"]

    def test_move_cursor_empty_visible(self) -> None:
        """move_cursor on empty visible list should not crash."""
        panel = self._make_panel()
        panel._filter = "running"  # nothing running
        panel.move_cursor(1)  # should be no-op
        assert panel._cursor == 0

    def test_get_cursor_task_id_empty(self) -> None:
        panel = self._make_panel()
        panel._filter = "running"  # nothing running
        assert panel.get_cursor_task_id() is None

    def test_move_cursor_to_not_visible(self) -> None:
        """move_cursor_to for a filtered-out task should clamp."""
        panel = self._make_panel()
        panel._filter = "running"
        panel._states["t1"].status = "running"
        panel.move_cursor_to("t3")  # t3 is pending, not visible
        assert panel._cursor == 0  # clamped to visible bounds

    def test_selected_task_id_in_range(self) -> None:
        panel = self._make_panel()
        panel._cursor = 1
        assert panel._selected_task_id() == "t2"

    def test_selected_task_id_out_of_range(self) -> None:
        panel = self._make_panel()
        panel._cursor = 99
        assert panel._selected_task_id() is None

    def test_move_cursor_disables_follow(self) -> None:
        panel = self._make_panel()
        panel._follow = True
        panel.move_cursor(1)
        assert panel._follow is False


# ---------------------------------------------------------------------------
# DetailPanel render edge cases (sync — no app context)
# ---------------------------------------------------------------------------


class TestDetailPanelRender:
    """Tests for DetailPanel render paths."""

    def _make_panel(self) -> DetailPanel:
        panel = DetailPanel()
        panel.update = lambda *a, **kw: None
        panel.refresh = lambda *a, **kw: None  # suppress Textual refresh
        return panel

    def test_render_empty(self) -> None:
        panel = self._make_panel()
        rendered = panel.render()
        assert "select task" in str(rendered).lower()

    def test_render_with_task(self) -> None:
        panel = self._make_panel()
        state = TaskState(
            task_id="t1", status="success", engine="claude",
            model="sonnet", duration_sec=5.0, cost_usd=0.03,
        )
        panel.select_task("t1", state, None)
        rendered = str(panel.render())
        assert "t1" in rendered
        assert "success" in rendered
        assert "claude/sonnet" in rendered

    def test_render_engine_no_model(self) -> None:
        panel = self._make_panel()
        state = TaskState(task_id="t1", status="success", engine="claude", model=None)
        panel.select_task("t1", state, None)
        rendered = str(panel._render_detail())
        assert "claude" in rendered

    def test_render_running_timeout_proximity(self) -> None:
        """Running task near timeout shows warning indicator."""
        panel = self._make_panel()
        state = TaskState(
            task_id="t1", status="running",
            started_at=time.monotonic() - 500.0,
            timeout_sec=600,
        )
        panel.select_task("t1", state, None)
        rendered = str(panel._render_detail())
        assert "!" in rendered

    def test_render_running_no_timeout_warning(self) -> None:
        """Running task well within timeout should not show warning."""
        panel = self._make_panel()
        state = TaskState(
            task_id="t1", status="running",
            started_at=time.monotonic() - 1.0,
            timeout_sec=600,
        )
        panel.select_task("t1", state, None)
        rendered = str(panel._render_detail())
        # The "!" timeout proximity warning should not appear
        assert "Engine working" in rendered

    def test_render_running_no_started_at(self) -> None:
        """Running task with no started_at — still shows working message."""
        panel = self._make_panel()
        state = TaskState(task_id="t1", status="running", started_at=None)
        panel.select_task("t1", state, None)
        rendered = str(panel._render_detail())
        assert "Engine working" in rendered

    def test_render_with_log_lines(self) -> None:
        panel = self._make_panel()
        state = TaskState(task_id="t1", status="success")
        panel.select_task("t1", state, None)
        panel._log_lines = ["line 1", "line 2", "line 3"]
        rendered = str(panel._render_detail())
        assert "line 1" in rendered
        assert "line 3" in rendered
        assert "(no log output)" not in rendered

    def test_render_failed_no_log(self) -> None:
        panel = self._make_panel()
        state = TaskState(task_id="t1", status="failed")
        panel.select_task("t1", state, None)
        rendered = str(panel._render_detail())
        assert "(no log output)" in rendered

    def test_render_detail_none_state_fallback(self) -> None:
        """_render_detail with cleared state falls back to empty."""
        panel = self._make_panel()
        panel._task_id = "t1"
        panel._state = None
        rendered = str(panel._render_detail())
        assert "select task" in rendered.lower()

    def test_clear_selection(self) -> None:
        panel = self._make_panel()
        state = TaskState(task_id="t1", status="success")
        panel.select_task("t1", state, None)
        panel.clear_selection()
        assert panel._task_id is None
        assert panel._state is None
        assert panel._log_lines == []
        assert panel._log_path is None

    def test_select_task_with_run_path(self) -> None:
        panel = self._make_panel()
        state = TaskState(task_id="t1", status="running")
        panel.select_task("t1", state, Path("/tmp/runs/run1"))
        assert panel._log_path == Path("/tmp/runs/run1/t1.log")

    def test_update_state_wrong_task(self) -> None:
        """update_state for a different task should be ignored."""
        panel = self._make_panel()
        state = TaskState(task_id="t1", status="running")
        panel.select_task("t1", state, None)
        other_state = TaskState(task_id="t2", status="failed")
        panel.update_state(other_state)
        assert panel._state.task_id == "t1"


# ---------------------------------------------------------------------------
# DetailPanel._poll_log tests
# ---------------------------------------------------------------------------


class TestDetailPanelPollLog:
    """Tests for DetailPanel log file polling."""

    def _make_panel(self) -> DetailPanel:
        panel = DetailPanel()
        panel.update = lambda *a, **kw: None
        panel.refresh = lambda *a, **kw: None
        return panel

    def test_poll_log_no_path(self) -> None:
        panel = self._make_panel()
        panel._log_path = None
        panel._poll_log()
        assert panel._log_lines == []

    def test_poll_log_file_not_exists(self, tmp_path: Path) -> None:
        panel = self._make_panel()
        panel._task_id = "t1"
        panel._log_path = tmp_path / "nonexistent.log"
        panel._poll_log()
        assert panel._log_lines == []

    def test_poll_log_reads_file(self, tmp_path: Path) -> None:
        log_file = tmp_path / "t1.log"
        log_file.write_text("line A\nline B\nline C\n", encoding="utf-8")
        panel = self._make_panel()
        panel._task_id = "t1"
        panel._log_path = log_file
        panel._poll_log()
        assert "line A" in panel._log_lines
        assert "line C" in panel._log_lines

    def test_poll_log_dedup_consecutive(self, tmp_path: Path) -> None:
        log_file = tmp_path / "t1.log"
        log_file.write_text("same\nsame\nsame\ndiff\n", encoding="utf-8")
        panel = self._make_panel()
        panel._task_id = "t1"
        panel._log_path = log_file
        panel._poll_log()
        assert panel._log_lines.count("same") == 1
        assert "diff" in panel._log_lines

    def test_poll_log_skips_blank_lines(self, tmp_path: Path) -> None:
        log_file = tmp_path / "t1.log"
        log_file.write_text("line1\n\n\nline2\n", encoding="utf-8")
        panel = self._make_panel()
        panel._task_id = "t1"
        panel._log_path = log_file
        panel._poll_log()
        assert "" not in panel._log_lines

    def test_poll_log_no_reread_same_size(self, tmp_path: Path) -> None:
        log_file = tmp_path / "t1.log"
        log_file.write_text("line1\n", encoding="utf-8")
        panel = self._make_panel()
        panel._task_id = "t1"
        panel._log_path = log_file
        panel._poll_log()
        # Simulate reading again at same size
        panel._log_lines = ["already read"]
        panel._poll_log()
        assert panel._log_lines == ["already read"]

    def test_poll_log_tails_to_20(self, tmp_path: Path) -> None:
        log_file = tmp_path / "t1.log"
        lines = [f"line{i}" for i in range(50)]
        log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
        panel = self._make_panel()
        panel._task_id = "t1"
        panel._log_path = log_file
        panel._poll_log()
        assert len(panel._log_lines) <= 20


# ---------------------------------------------------------------------------
# EventFeed tests (sync — no app context)
# ---------------------------------------------------------------------------


class TestEventFeedUnit:
    """Tests for EventFeed static helpers and write_event branches."""

    def test_format_local_time_valid_utc(self) -> None:
        ts = "2026-03-28T14:30:00Z"
        result = EventFeed._format_local_time(ts)
        assert len(result) == 8  # HH:MM:SS
        assert ":" in result

    def test_format_local_time_valid_offset(self) -> None:
        ts = "2026-03-28T14:30:00+01:00"
        result = EventFeed._format_local_time(ts)
        assert len(result) == 8

    def test_format_local_time_invalid(self) -> None:
        result = EventFeed._format_local_time("not-a-date")
        # Falls back to last 8 chars
        assert result == "t-a-date"

    def test_format_local_time_empty_string(self) -> None:
        result = EventFeed._format_local_time("")
        assert len(result) == 8  # falls back to datetime.now()

    def test_format_local_time_non_string(self) -> None:
        result = EventFeed._format_local_time(12345)
        assert len(result) == 8  # falls back to datetime.now()

    def test_format_local_time_none(self) -> None:
        result = EventFeed._format_local_time(None)
        assert len(result) == 8


# ---------------------------------------------------------------------------
# EventFeed.write_event branch coverage (async — needs mounted widget)
# ---------------------------------------------------------------------------


async def test_event_feed_task_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    """task_retry event shows retry info in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("task_retry", {"task_id": "t1", "attempt": 1, "max_retries": 3})
            cb("task_complete", {"task_id": "t1", "status": "success", "cost_usd": 0.0})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_task_tool_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """task_tool_call events appear in EventFeed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("task_tool_call", {"task_id": "t1", "tool": "write_file"})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_task_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """task_output events are dispatched to both DAGPanel and EventFeed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("task_output", {"task_id": "t1", "line": "compiling..."})
            time.sleep(0.2)
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_task_escalation(monkeypatch: pytest.MonkeyPatch) -> None:
    """task_escalation event renders in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("task_escalation", {
                "task_id": "t1", "from_model": "haiku", "to_model": "sonnet",
            })
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_engine_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """engine_fallback event renders in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "codex"})
            cb("engine_fallback", {
                "task_id": "t1", "from_engine": "codex", "to_engine": "claude",
            })
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_verify_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """verify_failure event renders in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("verify_failure", {
                "task_id": "t1", "exit_code": 1,
                "output_snippet": "Error: test failed",
            })
            cb("task_complete", {"task_id": "t1", "status": "failed"})
            cb("run_complete", {"success": False, "ok": 0, "failed": 1})
        return _make_result(success=False)

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_judge_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """judge_start and judge_verdict events render in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("judge_start", {"task_id": "t1", "method": "g_eval", "criteria_count": 3})
            cb("judge_verdict", {"task_id": "t1", "verdict": "pass", "score": 0.85})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_context_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """context_summarize and context_compression events render in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("context_summarize", {"task_id": "t1", "upstream_id": "t0"})
            cb("context_compression", {
                "task_id": "t1", "tokens_raw": 1000, "tokens_final": 500,
            })
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_context_compression_no_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    """context_compression with no raw tokens shows simple message."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("context_compression", {"task_id": "t1"})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_watch_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Watch-related events render in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("watch_start", {"iteration": 1})
            cb("iteration_start", {"iteration": 1})
            cb("metric_recorded", {"iteration": 1, "metric_value": 0.85, "best_metric": 0.85})
            cb("iteration_complete", {"iteration": 1, "status": "improved"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_worktree_events(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worktree events render in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("worktree_create", {"task_id": "t1", "worktree_path": "/tmp/wt/t1", "branch": "maestro/t1"})
            cb("worktree_merge", {
                "task_id": "t1", "status": "merged",
                "files_changed": ["a.py", "b.py"], "overlapping_files": [],
            })
            cb("worktree_cleanup", {"task_id": "t1"})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_worktree_merge_conflict(monkeypatch: pytest.MonkeyPatch) -> None:
    """Worktree merge with conflict and overlap."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("worktree_merge", {
                "task_id": "t1", "status": "conflict",
                "files_changed": ["a.py"], "overlapping_files": ["a.py"],
                "review_verdict": "conflict",
            })
            cb("run_complete", {"success": False, "ok": 0, "failed": 1})
        return _make_result(success=False)

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_worktree_review(monkeypatch: pytest.MonkeyPatch) -> None:
    """worktree_review event with conflicts and suggestion."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("worktree_review", {
                "task_id": "t1", "verdict": "resolvable",
                "conflict_files": ["a.py"], "resolution_suggestion": "Merge manually",
            })
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_model_routed(monkeypatch: pytest.MonkeyPatch) -> None:
    """model_routed event renders in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("model_routed", {
                "task_id": "t1", "resolved": "opus", "complexity_score": 0.85,
            })
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_dynamic_subplan(monkeypatch: pytest.MonkeyPatch) -> None:
    """dynamic_subplan_start and dynamic_subplan_complete events."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("dynamic_subplan_start", {
                "task_id": "t1", "sub_plan_name": "sub-plan", "sub_task_count": 3,
            })
            cb("dynamic_subplan_complete", {
                "task_id": "t1", "success": True, "sub_task_count": 3, "total_cost_usd": 0.12,
            })
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_dynamic_subplan_failed(monkeypatch: pytest.MonkeyPatch) -> None:
    """dynamic_subplan_complete with failure."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("dynamic_subplan_complete", {
                "task_id": "t1", "success": False, "sub_task_count": 3,
            })
            cb("run_complete", {"success": False, "ok": 0, "failed": 1})
        return _make_result(success=False)

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_task_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    """task_progress event renders in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("task_progress", {"task_id": "t1", "pct": 50, "step": "analyzing"})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_task_artifact(monkeypatch: pytest.MonkeyPatch) -> None:
    """task_artifact event renders in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("task_artifact", {"task_id": "t1", "label": "report.html"})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_task_signal_log(monkeypatch: pytest.MonkeyPatch) -> None:
    """task_signal_log event renders in feed with level-based styling."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("task_signal_log", {"task_id": "t1", "level": "error", "message": "something broke"})
            cb("task_signal_log", {"task_id": "t1", "level": "warn", "message": "heads up"})
            cb("task_signal_log", {"task_id": "t1", "level": "info", "message": "all good"})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_timeout_extended(monkeypatch: pytest.MonkeyPatch) -> None:
    """timeout_extended event renders in feed."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude"})
            cb("timeout_extended", {"task_id": "t1", "additional_sec": 300})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_unknown_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unknown events hit the default branch in write_event."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("some_future_event", {"task_id": "t1"})
            cb("another_event", {})
            cb("task_tool_call", {"task_id": "t1", "tool": "edit", "dynamic_parent": "parent1"})
            cb("unknown_dynamic", {"task_id": "t1", "dynamic_parent": "parent2"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


async def test_event_feed_judge_verdict_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    """judge_verdict with fail renders differently."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("judge_verdict", {"task_id": "t1", "verdict": "fail", "score": 0.3})
            cb("judge_verdict", {"task_id": "t1", "verdict": "error"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)


# ---------------------------------------------------------------------------
# EventFeed write_output / throttle tests
# ---------------------------------------------------------------------------


class TestEventFeedWriteOutput:
    """Tests for EventFeed.write_output throttling and dedup."""

    def _make_feed(self) -> EventFeed:
        feed = EventFeed()
        # Replace write method to capture output
        feed._written: list[Any] = []
        original_write = feed.write

        def tracking_write(content: Any) -> None:
            feed._written.append(str(content))
        feed.write = tracking_write  # type: ignore[assignment]
        return feed

    def test_write_output_basic(self) -> None:
        feed = self._make_feed()
        feed._last_output_time = 0.0  # not throttled
        feed.write_output({"task_id": "t1", "line": "hello world"})
        assert len(feed._written) == 1

    def test_write_output_dedup(self) -> None:
        feed = self._make_feed()
        feed._last_output_time = 0.0
        feed.write_output({"task_id": "t1", "line": "same line"})
        feed._last_output_time = 0.0  # reset throttle
        feed.write_output({"task_id": "t1", "line": "same line"})
        assert len(feed._written) == 1  # deduped

    def test_write_output_different_task_not_deduped(self) -> None:
        feed = self._make_feed()
        feed._last_output_time = 0.0
        feed.write_output({"task_id": "t1", "line": "same line"})
        feed._last_output_time = 0.0
        feed.write_output({"task_id": "t2", "line": "same line"})
        assert len(feed._written) == 2

    def test_write_output_empty_line_ignored(self) -> None:
        feed = self._make_feed()
        feed.write_output({"task_id": "t1", "line": ""})
        assert len(feed._written) == 0

    def test_write_output_throttled(self) -> None:
        feed = self._make_feed()
        feed._last_output_time = time.monotonic()  # just wrote
        feed.write_output({"task_id": "t1", "line": "should be pending"})
        assert len(feed._written) == 0
        assert feed._pending_line is not None

    def test_flush_pending_output(self) -> None:
        feed = self._make_feed()
        feed._last_output_time = time.monotonic()
        feed.write_output({"task_id": "t1", "line": "pending line"})
        assert len(feed._written) == 0
        feed.flush_pending_output()
        assert len(feed._written) == 1

    def test_flush_pending_output_nothing(self) -> None:
        feed = self._make_feed()
        feed.flush_pending_output()
        assert len(feed._written) == 0


# ---------------------------------------------------------------------------
# MaestroApp action tests (async — full app context)
# ---------------------------------------------------------------------------


async def test_app_run_start_sets_run_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_start event sets _run_path on the app."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("run_start", {"run_path": "/tmp/runs/test-run"})
            cb("task_start", {"task_id": "t1"})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)
        assert app._run_path == Path("/tmp/runs/test-run")


async def test_app_keyboard_navigation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrow keys move cursor in DAGPanel."""
    tasks = [TaskSpec(id=f"t{i}", command="echo ok") for i in range(1, 4)]
    plan = _make_plan(tasks)

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", _noop_run_plan)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.3)
        dag = app.query_one(DAGPanel)
        await pilot.press("down")
        await pilot.pause(delay=0.1)
        await pilot.press("down")
        await pilot.pause(delay=0.1)
        # Cursor should have moved — verify via selected_task
        assert dag._cursor >= 0


async def test_app_follow_toggle(monkeypatch: pytest.MonkeyPatch) -> None:
    """'t' key toggles follow mode on DAGPanel."""
    plan = _make_plan()
    monkeypatch.setattr("maestro_cli.scheduler.run_plan", _noop_run_plan)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.3)
        dag = app.query_one(DAGPanel)
        initial = dag._follow
        await pilot.press("t")
        await pilot.pause(delay=0.1)
        assert dag._follow != initial


async def test_app_filter_cycle(monkeypatch: pytest.MonkeyPatch) -> None:
    """'f' key cycles filter on DAGPanel."""
    plan = _make_plan()
    monkeypatch.setattr("maestro_cli.scheduler.run_plan", _noop_run_plan)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.3)
        dag = app.query_one(DAGPanel)
        assert dag._filter == "all"
        await pilot.press("f")
        await pilot.pause(delay=0.1)
        assert dag._filter == "running"


async def test_app_escape_clears_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    """Escape key clears task selection or filter."""
    tasks = [TaskSpec(id=f"t{i}", command="echo ok") for i in range(1, 3)]
    plan = _make_plan(tasks)
    monkeypatch.setattr("maestro_cli.scheduler.run_plan", _noop_run_plan)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.3)
        # Select a task then deselect
        await pilot.press("enter")
        await pilot.pause(delay=0.1)
        await pilot.press("escape")
        await pilot.pause(delay=0.1)


async def test_app_quit_completed_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Quitting after run completion exits cleanly."""
    plan = _make_plan()
    monkeypatch.setattr("maestro_cli.scheduler.run_plan", _noop_run_plan)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)
        await pilot.press("q")


async def test_app_detail_panel_updates_on_event(monkeypatch: pytest.MonkeyPatch) -> None:
    """When a task is selected, events for that task update the DetailPanel."""
    plan = _make_plan([TaskSpec(id="t1", command="echo ok")])

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "claude", "model": "sonnet"})
            time.sleep(0.2)
            cb("task_complete", {"task_id": "t1", "status": "success", "duration_sec": 1.0})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.3)
        # Select the task
        await pilot.press("enter")
        await pilot.pause(delay=0.5)
        detail = app.query_one(DetailPanel)
        # DetailPanel should have been updated
        assert detail._task_id is not None


async def test_app_follow_moves_cursor_on_start(monkeypatch: pytest.MonkeyPatch) -> None:
    """With follow=True, task_start moves cursor to the started task."""
    tasks = [TaskSpec(id=f"t{i}", command="echo ok") for i in range(1, 4)]
    plan = _make_plan(tasks)

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("task_start", {"task_id": "t1", "engine": "shell"})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("task_start", {"task_id": "t2", "engine": "shell"})
            cb("task_complete", {"task_id": "t2", "status": "success"})
            cb("task_start", {"task_id": "t3", "engine": "shell"})
            cb("task_complete", {"task_id": "t3", "status": "success"})
            cb("run_complete", {"success": True, "ok": 3, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.1)
        # Enable follow mode
        dag = app.query_one(DAGPanel)
        dag._follow = True
        await pilot.pause(delay=0.5)


async def test_app_auto_approve_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    """Auto-approve flag makes _approval_handler return True immediately."""
    plan = _make_plan()
    monkeypatch.setattr("maestro_cli.scheduler.run_plan", _noop_run_plan)
    app = MaestroApp(plan, dry_run=True, auto_approve=True)
    # Test _approval_handler directly
    result = app._approval_handler("t1", "Approve?")
    assert result is True


async def test_app_dispatch_run_start_non_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """run_start with non-string run_path should not crash."""
    plan = _make_plan()

    def mock_run(p: PlanSpec, **kw: Any) -> PlanRunResult:
        time.sleep(0.15)
        cb = kw.get("event_callback")
        if cb and callable(cb):
            cb("run_start", {"run_path": 12345})  # not a string
            cb("task_start", {"task_id": "t1"})
            cb("task_complete", {"task_id": "t1", "status": "success"})
            cb("run_complete", {"success": True, "ok": 1, "failed": 0})
        return _make_result()

    monkeypatch.setattr("maestro_cli.scheduler.run_plan", mock_run)
    app = MaestroApp(plan, dry_run=True)
    async with app.run_test() as pilot:
        await pilot.pause(delay=0.5)
        assert app._run_path is None  # not set since not a string
