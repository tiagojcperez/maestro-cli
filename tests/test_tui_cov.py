from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

textual = pytest.importorskip("textual")

from textual.css.query import NoMatches

from maestro_cli.models import PlanSpec, TaskSpec
from maestro_cli.tui import widgets as widgets_mod
from maestro_cli.tui.app import MaestroApp
from maestro_cli.tui.widgets import DAGPanel, DetailPanel, PlanHeader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(n: int = 3, name: str = "cov-plan") -> PlanSpec:
    tasks = [TaskSpec(id=f"t{i}", command="echo ok") for i in range(1, n + 1)]
    return PlanSpec(name=name, tasks=tasks, source_path=Path("plan.yaml"))


def _make_app() -> MaestroApp:
    """Construct a MaestroApp instance without a running event loop.

    The action methods under test only call boundary methods (``query_one``,
    ``call_from_thread``, ``notify``, ``set_timer``, ``exit``,
    ``batch_update``) which the individual tests stub out.
    """
    return MaestroApp(_make_plan(), dry_run=True)


class _FakeModal:
    """Stand-in for ApprovalModal that records calls without Textual."""

    def __init__(self, display: bool = False) -> None:
        self.display = display
        self.approved_called = False
        self.denied_called = False
        self.shown: tuple[str, str, threading.Event] | None = None

    def approve(self) -> None:
        self.approved_called = True

    def deny(self) -> None:
        self.denied_called = True

    def show_approval(
        self, task_id: str, message: str, response_event: threading.Event
    ) -> None:
        self.shown = (task_id, message, response_event)


class _FakeDAG:
    """Stand-in for DAGPanel recording cursor/filter calls."""

    def __init__(self) -> None:
        self.move_delta: int | None = None
        self.clear_filter_called = False

    def move_cursor(self, delta: int) -> None:
        self.move_delta = delta

    def clear_filter(self) -> None:
        self.clear_filter_called = True


# ---------------------------------------------------------------------------
# widgets.PlanHeader._tick — non-skip refresh path (66-67)
# ---------------------------------------------------------------------------


class TestPlanHeaderTickRefresh:
    """Cover the branch of _tick that actually re-renders."""

    def _make_header(self) -> PlanHeader:
        header = PlanHeader(_make_plan())
        calls: list[Any] = []
        header.update = lambda *a, **kw: calls.append((a, kw))  # type: ignore[method-assign]
        header._update_calls = calls  # type: ignore[attr-defined]
        return header

    def test_tick_renders_when_not_finished(self) -> None:
        header = self._make_header()
        header._finished = False
        header._needs_refresh = True
        header._tick()
        # The render path ran and cleared the refresh flag.
        assert header._needs_refresh is False
        assert header._update_calls  # type: ignore[attr-defined]

    def test_tick_renders_when_finished_but_needs_refresh(self) -> None:
        header = self._make_header()
        header._finished = True
        header._needs_refresh = True
        header._tick()
        assert header._needs_refresh is False
        assert header._update_calls  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# widgets.DAGPanel._render_table — defensive else for unknown terminal
# status (301)
# ---------------------------------------------------------------------------


class TestDAGPanelRenderTableDefensive:
    """Drive the defensive else fallback in _render_table.

    The else at the end of the terminal-status block is only reachable when a
    status is a member of TERMINAL_STATUSES but not one of the explicitly
    handled values. The shipped TERMINAL_STATUSES has no such value, so we
    inject a synthetic terminal status to exercise the defensive line.
    """

    def _make_panel(self) -> DAGPanel:
        panel = DAGPanel(_make_plan(1))
        panel._refresh_table = lambda: None  # type: ignore[method-assign]
        return panel

    def test_render_unknown_terminal_status(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        panel = self._make_panel()
        # Inject a synthetic terminal status that none of the explicit
        # branches handle, so the defensive else fires.
        injected = frozenset(set(widgets_mod.TERMINAL_STATUSES) | {"weird_terminal"})
        monkeypatch.setattr(widgets_mod, "TERMINAL_STATUSES", injected)
        panel._states["t1"].status = "weird_terminal"
        table = panel._render_table()
        assert table.row_count == 1


# ---------------------------------------------------------------------------
# widgets.DetailPanel._poll_log — except OSError branch (466-467)
# ---------------------------------------------------------------------------


class TestDetailPanelPollLogOSError:
    """Cover the OSError swallow in _poll_log."""

    def _make_panel(self) -> DetailPanel:
        panel = DetailPanel()
        panel.update = lambda *a, **kw: None  # type: ignore[method-assign]
        panel.refresh = lambda *a, **kw: None  # type: ignore[method-assign]
        return panel

    def test_poll_log_read_raises_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        log_file = tmp_path / "t1.log"
        log_file.write_text("line A\nline B\n", encoding="utf-8")
        panel = self._make_panel()
        panel._task_id = "t1"
        panel._log_path = log_file

        # The exists()/stat() checks run against the real file first; make the
        # subsequent read_text raise OSError to exercise the swallow branch.
        real_read_text = Path.read_text

        def _boom(self: Path, *a: Any, **kw: Any) -> str:
            if self == log_file:
                raise OSError("disk gone")
            return real_read_text(self, *a, **kw)

        monkeypatch.setattr(Path, "read_text", _boom)
        # Should not raise — OSError is swallowed.
        panel._poll_log()
        assert panel._log_lines == []


# ---------------------------------------------------------------------------
# app.MaestroApp._approval_handler / _show_approval (151-159, 168-170)
# ---------------------------------------------------------------------------


class TestApprovalHandler:
    """Cover the interactive (non auto-approve) approval bridge."""

    def test_approval_handler_waits_and_returns_result(self) -> None:
        app = _make_app()
        app._auto_approve = False
        app._last_approval_result = True

        captured: dict[str, Any] = {}

        def fake_call_from_thread(fn: Any, *args: Any, **kwargs: Any) -> None:
            # Simulate the main-thread modal flow: invoke _show_approval and
            # then immediately resolve the response event so wait() returns.
            captured["fn"] = fn
            captured["args"] = args
            event = args[-1]
            assert isinstance(event, threading.Event)
            event.set()

        app.call_from_thread = fake_call_from_thread  # type: ignore[method-assign]

        result = app._approval_handler("t1", None)
        assert result is True
        # Bound methods compare by underlying function, not identity.
        assert captured["fn"].__func__ is MaestroApp._show_approval
        # message defaulted to the standard prompt since None was passed.
        assert "Approval required." in captured["args"]

    def test_approval_handler_returns_false_when_denied(self) -> None:
        app = _make_app()
        app._auto_approve = False
        app._last_approval_result = False

        def fake_call_from_thread(fn: Any, *args: Any, **kwargs: Any) -> None:
            args[-1].set()

        app.call_from_thread = fake_call_from_thread  # type: ignore[method-assign]
        assert app._approval_handler("t2", "Custom message") is False

    def test_show_approval_queries_modal_and_shows(self) -> None:
        app = _make_app()
        modal = _FakeModal()
        app.query_one = lambda *a, **kw: modal  # type: ignore[method-assign]
        event = threading.Event()
        app._show_approval("deploy", "Proceed?", event)
        assert app._pending_approval_event is event
        assert modal.shown == ("deploy", "Proceed?", event)


# ---------------------------------------------------------------------------
# app.MaestroApp.action_cursor_up (187-188)
# ---------------------------------------------------------------------------


class TestActionCursorUp:
    def test_cursor_up_moves_and_auto_selects(self) -> None:
        app = _make_app()
        dag = _FakeDAG()
        app.query_one = lambda *a, **kw: dag  # type: ignore[method-assign]
        selected: list[Any] = []
        app._auto_select_cursor = lambda: selected.append(True)  # type: ignore[method-assign]
        app.action_cursor_up()
        assert dag.move_delta == -1
        assert selected == [True]


# ---------------------------------------------------------------------------
# app.MaestroApp.action_deselect (210, 214-215)
# ---------------------------------------------------------------------------


class TestActionDeselect:
    def test_deselect_returns_early_when_modal_visible(self) -> None:
        app = _make_app()
        modal = _FakeModal(display=True)
        app.query_one = lambda *a, **kw: modal  # type: ignore[method-assign]
        app._selected_task = "t1"
        cleared: list[Any] = []
        app.clear_task_selection = lambda: cleared.append(True)  # type: ignore[method-assign]
        app.action_deselect()
        # Modal visible → early return, selection untouched.
        assert cleared == []
        assert app._selected_task == "t1"

    def test_deselect_clears_filter_when_no_selection(self) -> None:
        app = _make_app()
        modal = _FakeModal(display=False)
        dag = _FakeDAG()

        def fake_query_one(arg: Any, *rest: Any, **kw: Any) -> Any:
            # First call (ApprovalModal) returns the modal; second (DAGPanel)
            # returns the dag stand-in.
            name = getattr(arg, "__name__", "")
            if name == "ApprovalModal":
                return modal
            return dag

        app.query_one = fake_query_one  # type: ignore[method-assign]
        app._selected_task = None
        app.action_deselect()
        assert dag.clear_filter_called is True


# ---------------------------------------------------------------------------
# app.MaestroApp.action_approve / action_deny (224-228, 231-235)
# ---------------------------------------------------------------------------


class TestActionApproveDeny:
    def test_action_approve_when_modal_visible(self) -> None:
        app = _make_app()
        modal = _FakeModal(display=True)
        app.query_one = lambda *a, **kw: modal  # type: ignore[method-assign]
        app._pending_approval_event = threading.Event()
        app.action_approve()
        assert app._last_approval_result is True
        assert app._pending_approval_event is None
        assert modal.approved_called is True

    def test_action_approve_noop_when_modal_hidden(self) -> None:
        app = _make_app()
        modal = _FakeModal(display=False)
        app.query_one = lambda *a, **kw: modal  # type: ignore[method-assign]
        app._last_approval_result = False
        app.action_approve()
        assert modal.approved_called is False
        assert app._last_approval_result is False

    def test_action_deny_when_modal_visible(self) -> None:
        app = _make_app()
        modal = _FakeModal(display=True)
        app.query_one = lambda *a, **kw: modal  # type: ignore[method-assign]
        app._pending_approval_event = threading.Event()
        app.action_deny()
        assert app._last_approval_result is False
        assert app._pending_approval_event is None
        assert modal.denied_called is True

    def test_action_deny_noop_when_modal_hidden(self) -> None:
        app = _make_app()
        modal = _FakeModal(display=False)
        app.query_one = lambda *a, **kw: modal  # type: ignore[method-assign]
        app.action_deny()
        assert modal.denied_called is False


# ---------------------------------------------------------------------------
# app.MaestroApp._dispatch_event — NoMatches except branch (243-247)
# ---------------------------------------------------------------------------


class TestDispatchEventNoMatches:
    def test_dispatch_event_swallows_no_matches(self) -> None:
        app = _make_app()

        def raising_query_one(*a: Any, **kw: Any) -> Any:
            raise NoMatches("panels not mounted")

        app.query_one = raising_query_one  # type: ignore[method-assign]
        # Should return cleanly without raising despite missing widgets.
        app._dispatch_event("task_start", {"task_id": "t1"})


# ---------------------------------------------------------------------------
# app.MaestroApp.action_quit_app force-quit + _reset_quit_request
# (314-315, 321)
# ---------------------------------------------------------------------------


class TestQuitApp:
    def test_quit_force_quits_on_second_press(self) -> None:
        app = _make_app()
        app._result = None
        app._quit_requested = True  # second press scenario
        exits: list[int] = []
        app.exit = lambda *a, **kw: exits.append(kw.get("return_code", 0))  # type: ignore[method-assign]
        notified: list[Any] = []
        app.notify = lambda *a, **kw: notified.append((a, kw))  # type: ignore[method-assign]
        timers: list[Any] = []
        app.set_timer = lambda *a, **kw: timers.append((a, kw))  # type: ignore[method-assign]
        app.action_quit_app()
        assert app._cancel_event.is_set()
        assert exits == [1]
        # Force-quit path returns before notifying / scheduling a timer.
        assert notified == []
        assert timers == []

    def test_quit_first_press_notifies_and_arms_timer(self) -> None:
        app = _make_app()
        app._result = None
        app._quit_requested = False
        exits: list[int] = []
        app.exit = lambda *a, **kw: exits.append(kw.get("return_code", 0))  # type: ignore[method-assign]
        notified: list[Any] = []
        app.notify = lambda *a, **kw: notified.append((a, kw))  # type: ignore[method-assign]
        timers: list[tuple[Any, ...]] = []
        app.set_timer = lambda *a, **kw: timers.append((a, kw))  # type: ignore[method-assign]
        app.action_quit_app()
        assert app._cancel_event.is_set()
        assert exits == []  # did not force quit
        assert app._quit_requested is True
        assert notified  # warned the user
        assert timers  # armed the reset timer

    def test_reset_quit_request(self) -> None:
        app = _make_app()
        app._quit_requested = True
        app._reset_quit_request()
        assert app._quit_requested is False
