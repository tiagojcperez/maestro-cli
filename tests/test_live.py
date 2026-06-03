from __future__ import annotations

import threading
import time
from typing import Any

import pytest

pytest.importorskip("rich")

from rich.console import Console
from rich.live import Live

from maestro_cli.live import _LivePlanDisplay, create_live_callback
from maestro_cli.utils import format_duration, format_cost
from maestro_cli.models import PlanDefaults, PlanSpec, TaskSpec


def _make_plan(tasks: list[TaskSpec] | None = None) -> PlanSpec:
    if tasks is None:
        tasks = [
            TaskSpec(id="t1", description="task 1", command="echo one"),
            TaskSpec(id="t2", description="task 2", command="echo two"),
        ]
    return PlanSpec(
        version=1,
        name="live-test-plan",
        defaults=PlanDefaults(),
        tasks=tasks,
    )


def _get_display(live: Live) -> Any:
    return live.renderable


def _render_text(renderable: Any) -> str:
    console = Console(width=240, record=True)
    console.print(renderable)
    return console.export_text()


def test_create_live_callback_returns_tuple() -> None:
    live, callback = create_live_callback(_make_plan())

    assert isinstance(live, Live)
    assert callable(callback)


def test_callback_handles_task_start() -> None:
    live, callback = create_live_callback(_make_plan())

    callback("task_start", {"task_id": "t1", "engine": "codex", "model": "gpt-5"})

    display = _get_display(live)
    task = display._tasks["t1"]
    assert task.status == "running"
    assert task.engine == "codex"
    assert task.model == "gpt-5"
    assert task.started_at is not None
    assert display._last_event_message == "t1 started"


def test_callback_handles_task_complete() -> None:
    live, callback = create_live_callback(_make_plan())

    callback("task_start", {"task_id": "t1", "engine": "codex"})
    callback(
        "task_complete",
        {"task_id": "t1", "status": "success", "duration_sec": 1.25, "cost_usd": 0.5},
    )

    display = _get_display(live)
    task = display._tasks["t1"]
    assert task.status == "success"
    assert task.duration_sec == 1.25
    assert task.cost_usd == 0.5
    assert task.started_at is None
    assert display._last_event_message == "t1 completed (1.2s, $0.50)"


def test_callback_handles_run_complete() -> None:
    live, callback = create_live_callback(_make_plan())

    callback("run_complete", {"success": True, "duration_sec": 9.5, "cost_usd": 1.75})

    display = _get_display(live)
    assert display._run_duration_sec == 9.5
    assert display._run_total_cost_usd == 1.75
    assert display._last_event_message == "run completed: success (9.5s, $1.75)"


def test_callback_unknown_task_id() -> None:
    live, callback = create_live_callback(_make_plan())

    callback(
        "task_complete",
        {
            "task_id": "missing-task",
            "status": "failed",
            "duration_sec": 0.2,
            "cost_usd": 0.0,
        },
    )

    display = _get_display(live)
    task = display._tasks["missing-task"]
    assert task.status == "failed"
    assert task.duration_sec == 0.2
    assert task.cost_usd == 0.0
    assert display._last_event_message == "missing-task failed (0.2s, $0.00)"


def test_callback_rapid_burst() -> None:
    live, callback = create_live_callback(_make_plan())

    for index in range(20):
        task_id = "t1" if index % 2 == 0 else "t2"
        if index % 3 == 0:
            callback("task_start", {"task_id": task_id, "engine": "codex", "model": "gpt-5"})
        else:
            callback(
                "task_complete",
                {
                    "task_id": task_id,
                    "status": "success",
                    "duration_sec": float(index),
                    "cost_usd": index / 100,
                },
            )

    display = _get_display(live)
    assert set(display._tasks) >= {"t1", "t2"}
    assert display._last_event_message == "t2 completed (19.0s, $0.19)"


def test_live_empty_plan() -> None:
    live, callback = create_live_callback(_make_plan([]))

    assert isinstance(live, Live)
    assert callable(callback)

    callback("run_start", {"goal": "test"})
    callback("run_complete", {"success": True, "duration_sec": 0.0, "cost_usd": 0.0})

    display = _get_display(live)
    assert display._run_duration_sec == 0.0
    assert display._run_total_cost_usd == 0.0
    assert display._last_event_message == "run completed: success (0.0s, $0.00)"


def test_live_single_task() -> None:
    live, callback = create_live_callback(
        _make_plan([TaskSpec(id="only", description="single", command="echo single")])
    )

    callback("task_start", {"task_id": "only", "engine": "codex"})
    callback(
        "task_complete",
        {"task_id": "only", "status": "success", "duration_sec": 0.4, "cost_usd": 0.02},
    )

    display = _get_display(live)
    task = display._tasks["only"]
    assert task.status == "success"
    assert task.duration_sec == 0.4
    assert task.cost_usd == 0.02


def test_live_50_tasks() -> None:
    tasks = [
        TaskSpec(id=f"task-{index:02d}", description=f"task {index}", command="echo ok")
        for index in range(50)
    ]
    live, callback = create_live_callback(_make_plan(tasks))

    for task in tasks:
        callback("task_start", {"task_id": task.id, "engine": "codex"})
        callback(
            "task_complete",
            {"task_id": task.id, "status": "success", "duration_sec": 0.1, "cost_usd": 0.01},
        )

    display = _get_display(live)
    assert len(display._tasks) == 50
    assert all(display._tasks[task.id].status == "success" for task in tasks)
    display._build_table()


def test_live_long_task_ids() -> None:
    long_id = "task-" + ("x" * 210)
    display = _LivePlanDisplay(
        _make_plan([TaskSpec(id=long_id, description="long id", command="echo long")])
    )

    display.handle_event("task_complete", {"task_id": long_id, "status": "success"})
    display._build_table()


def test_live_unicode_task_ids() -> None:
    task_ids = ["任务-1", "🚀-deploy", "café-setup"]
    display = _LivePlanDisplay(
        _make_plan(
            [TaskSpec(id=task_id, description=task_id, command="echo unicode") for task_id in task_ids]
        )
    )

    for task_id in task_ids:
        display.handle_event("task_complete", {"task_id": task_id, "status": "success"})

    display._build_table()


def test_live_budget_warning_event() -> None:
    display = _LivePlanDisplay(_make_plan())

    display.handle_event("budget_warning", {"spent": 7.5, "limit": 10.0, "pct": 0.75})

    assert display._budget_warning == "budget warning: spent $7.50, limit $10.00, 75%"
    header = display._build_header().plain
    assert "budget warning" in header
    assert "$7.50" in header
    assert "$10.00" in header
    assert "75%" in header


def test_live_all_status_combinations() -> None:
    statuses = [
        "pending",
        "running",
        "success",
        "failed",
        "soft_failed",
        "skipped",
        "dry_run",
    ]
    tasks = [
        TaskSpec(id=f"task-{status}", description=status, command="echo status")
        for status in statuses
    ]
    display = _LivePlanDisplay(_make_plan(tasks))

    for task, status in zip(tasks, statuses):
        state = display._tasks[task.id]
        state.status = status
        if status == "running":
            state.started_at = 0.0
        if status == "skipped":
            state.reason = "skip reason"

    table_text = _render_text(display._build_table())
    for icon in ("[..]", "[>>]", "[ok]", "[!!]", "[~~]", "[--]"):
        assert icon in table_text


def test_live_skip_with_reason() -> None:
    display = _LivePlanDisplay(_make_plan([TaskSpec(id="skip-me", description="skip", command="echo")]))

    display.handle_event("task_skip", {"task_id": "skip-me", "reason": "Budget exceeded"})

    state = display._tasks["skip-me"]
    assert state.reason == "Budget exceeded"
    assert display._format_detail(state) == "Budget exceeded"


def test_live_task_output_updates_running_detail_with_truncation() -> None:
    display = _LivePlanDisplay(_make_plan([TaskSpec(id="t1", description="task", command="echo")]))

    display.handle_event("task_start", {"task_id": "t1", "engine": "codex"})
    display.handle_event("task_output", {"task_id": "t1", "line": "x" * 80})

    state = display._tasks["t1"]
    assert state.last_line == "x" * 60
    assert display._format_detail(state) == "x" * 60


def test_live_task_output_throttles_via_pending_buffer() -> None:
    display = _LivePlanDisplay(_make_plan([TaskSpec(id="t1", description="task", command="echo")]))

    display.handle_event("task_output", {"task_id": "t1", "line": "first"})
    display.handle_event("task_output", {"task_id": "t1", "line": "second"})

    assert display._tasks["t1"].last_line == "first"
    assert display._pending_output["t1"] == "second"

    display._last_output_update_at = 0.0
    display._build_table()

    assert display._tasks["t1"].last_line == "second"
    assert display._pending_output == {}


def test_live_task_retry_updates_detail_and_footer() -> None:
    display = _LivePlanDisplay(_make_plan([TaskSpec(id="t1", description="task", command="echo")]))

    display.handle_event("task_retry", {"task_id": "t1", "attempt": 2, "max_retries": 3})

    state = display._tasks["t1"]
    assert state.attempts == 2
    assert state.max_retries == 3
    assert display._format_detail(state) == "retry 2/3"
    assert display._last_event_message == "retry t1 (attempt 2)"


def test_live_judge_events_update_footer() -> None:
    display = _LivePlanDisplay(_make_plan())

    display.handle_event("judge_start", {"task_id": "t1"})
    assert display._last_event_message == "[judge] evaluating t1"

    display.handle_event("judge_verdict", {"task_id": "t1", "verdict": "pass"})
    assert display._last_event_message == "[judge] t1: pass"


def test_live_judge_result_alias_updates_footer() -> None:
    display = _LivePlanDisplay(_make_plan())

    display.handle_event("judge_result", {"task_id": "t1", "verdict": "fail"})

    assert display._last_event_message == "[judge] t1: fail"


def test_live_escalation_and_fallback_update_displayed_engine_model() -> None:
    display = _LivePlanDisplay(_make_plan([TaskSpec(id="t1", description="task", command="echo")]))

    display.handle_event("task_start", {"task_id": "t1", "engine": "claude", "model": "haiku"})
    display.handle_event("task_escalation", {"task_id": "t1", "to_model": "sonnet"})
    assert display._tasks["t1"].model == "sonnet"
    assert display._last_event_message == "escalated t1 -> sonnet"

    display.handle_event("engine_fallback", {"task_id": "t1", "to_engine": "codex"})
    assert display._tasks["t1"].engine == "codex"
    assert display._last_event_message == "fallback t1 -> codex"


def test_live_format_duration_edge_cases() -> None:
    assert format_duration(None) == "--"
    assert format_duration(0.0) == "0.0s"
    assert format_duration(59.9) == "59.9s"
    assert format_duration(60.0) == "1m00s"
    assert format_duration(3599) == "59m59s"
    assert format_duration(3600) == "1h00m"


def test_live_format_cost_edge_cases() -> None:
    assert format_cost(None) == "--"
    assert format_cost(0.0) == "$0.00"
    assert format_cost(0.001) == "$0.00"
    assert format_cost(99.99) == "$99.99"


def test_state_machine_start_then_immediate_complete() -> None:
    display = _LivePlanDisplay(_make_plan())

    assert display._tasks["t1"].status == "pending"

    display.handle_event("task_start", {"task_id": "t1", "engine": "codex"})
    assert display._tasks["t1"].status == "running"

    display.handle_event(
        "task_complete",
        {"task_id": "t1", "status": "success", "duration_sec": 0.0, "cost_usd": 0.0},
    )

    task = display._tasks["t1"]
    assert task.status == "success"
    assert task.started_at is None


def test_state_machine_complete_without_start() -> None:
    display = _LivePlanDisplay(_make_plan())

    display.handle_event(
        "task_complete",
        {"task_id": "never-started", "status": "success", "duration_sec": 0.3, "cost_usd": 0.0},
    )

    task = display._tasks["never-started"]
    assert task.task_id == "never-started"
    assert task.status == "success"
    assert task.duration_sec == 0.3


def test_state_machine_retry_resets_state() -> None:
    display = _LivePlanDisplay(_make_plan())

    display.handle_event("task_start", {"task_id": "t1", "engine": "codex"})
    first_started_at = display._tasks["t1"].started_at
    display.handle_event(
        "task_complete",
        {"task_id": "t1", "status": "failed", "duration_sec": 5.0, "cost_usd": 0.25},
    )

    # The source records started_at via time.monotonic(). On Windows + Python
    # 3.12 time.monotonic() uses GetTickCount64 (~15.6 ms granularity), so a
    # tiny fixed sleep can leave the clock unmoved and make the strict-greater
    # assertion below flaky. Spin until the clock actually advances (bounded)
    # so the assertion stays deterministic on every platform/version.
    assert first_started_at is not None
    deadline = time.monotonic() + 2.0
    while time.monotonic() <= first_started_at and time.monotonic() < deadline:
        time.sleep(0.01)
    display.handle_event("task_start", {"task_id": "t1", "engine": "codex"})
    second_started_at = display._tasks["t1"].started_at
    display.handle_event(
        "task_complete",
        {"task_id": "t1", "status": "success", "duration_sec": 0.2, "cost_usd": 0.01},
    )

    task = display._tasks["t1"]
    assert first_started_at is not None
    assert second_started_at is not None
    assert second_started_at > first_started_at
    assert task.status == "success"
    assert task.duration_sec == 0.2
    assert task.cost_usd == 0.01
    assert task.started_at is None


def test_state_machine_run_complete_with_failure() -> None:
    display = _LivePlanDisplay(_make_plan())

    display.handle_event("run_complete", {"success": False, "duration_sec": 1.0, "cost_usd": 0.0})

    assert "failed" in display._last_event_message


def test_state_machine_progress_bar_edge_cases() -> None:
    display = _LivePlanDisplay(_make_plan())

    assert display._progress_bar(0, 0) == "─" * 16
    assert display._progress_bar(0, 10) == "─" * 16
    assert display._progress_bar(10, 10) == "━" * 16
    assert display._progress_bar(5, 10) == ("━" * 8) + ("─" * 8)
    assert display._progress_bar(1, 100) == ("━" * 0) + ("─" * 16)


def test_state_machine_elapsed_time_running() -> None:
    display = _LivePlanDisplay(_make_plan())

    first_header = display._build_header().plain
    time.sleep(0.11)
    second_header = display._build_header().plain

    assert first_header != second_header


def test_state_machine_thread_safety() -> None:
    tasks = [
        TaskSpec(id=f"thread-{index}", description=f"thread {index}", command="echo ok")
        for index in range(10)
    ]
    display = _LivePlanDisplay(_make_plan(tasks))
    barrier = threading.Barrier(len(tasks))
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(task_id: str) -> None:
        try:
            barrier.wait()
            for attempt in range(10):
                display.handle_event("task_start", {"task_id": task_id, "engine": "codex"})
                display.handle_event(
                    "task_complete",
                    {
                        "task_id": task_id,
                        "status": "success",
                        "duration_sec": attempt / 10,
                        "cost_usd": attempt / 100,
                    },
                )
        except BaseException as exc:  # pragma: no cover - failure capture for threads
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(task.id,)) for task in tasks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert all(display._tasks[task.id].status == "success" for task in tasks)


class TestLiveEnhancements:
    def test_task_output_updates_state(self) -> None:
        display = _LivePlanDisplay(_make_plan([TaskSpec(id="t1", description="task", command="echo")]))

        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event("task_output", {"task_id": "t1", "line": "hello world"})

        assert display._tasks["t1"].last_line == "hello world"

    def test_task_output_truncated(self) -> None:
        display = _LivePlanDisplay(_make_plan([TaskSpec(id="t1", description="task", command="echo")]))

        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event("task_output", {"task_id": "t1", "line": "x" * 80})

        assert display._tasks["t1"].last_line == "x" * 60

    def test_task_retry_updates_attempts(self) -> None:
        display = _LivePlanDisplay(_make_plan([TaskSpec(id="t1", description="task", command="echo")]))

        display.handle_event("task_retry", {"task_id": "t1", "attempt": 2, "max_retries": 3})

        state = display._tasks["t1"]
        assert state.attempts == 2
        assert state.max_retries == 3

    def test_judge_start_footer(self) -> None:
        display = _LivePlanDisplay(_make_plan())

        display.handle_event("judge_start", {"task_id": "t1"})

        assert display._last_event_message == "[judge] evaluating t1"

    def test_judge_verdict_footer(self) -> None:
        display = _LivePlanDisplay(_make_plan())

        display.handle_event("judge_verdict", {"task_id": "t1", "verdict": "pass"})

        assert display._last_event_message == "[judge] t1: pass"

    def test_task_escalation_updates_model(self) -> None:
        display = _LivePlanDisplay(_make_plan([TaskSpec(id="t1", description="task", command="echo")]))

        display.handle_event("task_start", {"task_id": "t1", "engine": "claude", "model": "haiku"})
        display.handle_event("task_escalation", {"task_id": "t1", "to_model": "opus"})

        assert display._tasks["t1"].model == "opus"
        assert display._last_event_message == "escalated t1 -> opus"

    def test_engine_fallback_updates_engine(self) -> None:
        display = _LivePlanDisplay(_make_plan([TaskSpec(id="t1", description="task", command="echo")]))

        display.handle_event("task_start", {"task_id": "t1", "engine": "claude", "model": "sonnet"})
        display.handle_event("engine_fallback", {"task_id": "t1", "to_engine": "codex"})

        assert display._tasks["t1"].engine == "codex"
        assert display._last_event_message == "fallback t1 -> codex"


# ---------------------------------------------------------------------------
# Additional tests (appended)
# ---------------------------------------------------------------------------


class TestLiveTaskStateDataclass:
    """Tests for the _TaskState dataclass defaults and fields."""

    def test_default_values(self) -> None:
        from maestro_cli.live import _TaskState

        state = _TaskState(task_id="x")
        assert state.task_id == "x"
        assert state.status == "pending"
        assert state.engine is None
        assert state.model is None
        assert state.duration_sec is None
        assert state.cost_usd is None
        assert state.started_at is None
        assert state.reason is None
        assert state.last_line == ""
        assert state.attempts == 0
        assert state.max_retries == 0
        assert state.progress_pct is None

    def test_custom_values(self) -> None:
        from maestro_cli.live import _TaskState

        state = _TaskState(
            task_id="t1",
            status="running",
            engine="claude",
            model="sonnet",
            duration_sec=5.0,
            cost_usd=0.12,
            started_at=100.0,
            reason="retry",
            last_line="doing stuff",
            attempts=2,
            max_retries=3,
            progress_pct=50,
        )
        assert state.engine == "claude"
        assert state.progress_pct == 50


class TestLiveDisplayInit:
    """Tests for _LivePlanDisplay initialization."""

    def test_tasks_populated_from_plan(self) -> None:
        tasks = [
            TaskSpec(id="a", description="a", command="echo a"),
            TaskSpec(id="b", description="b", command="echo b"),
            TaskSpec(id="c", description="c", command="echo c"),
        ]
        display = _LivePlanDisplay(_make_plan(tasks))
        assert set(display._tasks.keys()) == {"a", "b", "c"}
        assert all(s.status == "pending" for s in display._tasks.values())

    def test_empty_plan_no_tasks(self) -> None:
        display = _LivePlanDisplay(_make_plan([]))
        assert display._tasks == {}

    def test_initial_event_message(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        assert display._last_event_message == "waiting for events"

    def test_initial_budget_warning_none(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        assert display._budget_warning is None

    def test_initial_run_duration_none(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        assert display._run_duration_sec is None
        assert display._run_total_cost_usd is None


class TestLiveFormatEngineModel:
    """Tests for _format_engine_model helper."""

    def test_engine_and_model(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", engine="claude", model="opus")
        assert display._format_engine_model(state) == "claude/opus"

    def test_engine_only(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", engine="codex")
        assert display._format_engine_model(state) == "codex"

    def test_model_only(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", model="sonnet")
        assert display._format_engine_model(state) == "sonnet"

    def test_pending_no_engine_no_model(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="pending")
        assert display._format_engine_model(state) == "(pending)"

    def test_terminal_no_engine_shows_dash(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="success")
        assert display._format_engine_model(state) == "\u2014"

    def test_skipped_no_engine_shows_dash(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="skipped")
        assert display._format_engine_model(state) == "\u2014"


class TestLiveFormatRowDuration:
    """Tests for _format_row_duration helper."""

    def test_running_task_shows_elapsed(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="running", started_at=time.monotonic() - 5.0)
        dur = display._format_row_duration(state)
        # Should be roughly 5s
        assert "s" in dur
        assert dur != "--"

    def test_completed_task_shows_fixed_duration(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="success", engine="claude", duration_sec=12.5)
        assert display._format_row_duration(state) == "12.5s"

    def test_pending_task_shows_dash(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="pending")
        assert display._format_row_duration(state) == "--"

    def test_resumed_task_zero_duration_no_engine_shows_dash(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="success", engine=None, duration_sec=0.0)
        assert display._format_row_duration(state) == "\u2014"

    def test_completed_task_zero_duration_with_engine(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="success", engine="claude", duration_sec=0.0)
        assert display._format_row_duration(state) == "0.0s"


class TestLiveFormatDetail:
    """Tests for _format_detail helper."""

    def test_retry_detail(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", attempts=1, max_retries=3)
        assert display._format_detail(state) == "retry 1/3"

    def test_retry_without_max_retries_uses_attempts(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", attempts=2, max_retries=0)
        assert display._format_detail(state) == "retry 2/2"

    def test_running_with_last_line(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="running", last_line="compiling...")
        assert display._format_detail(state) == "compiling..."

    def test_running_without_last_line(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="running")
        assert display._format_detail(state) == "running"

    def test_skipped_with_reason(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="skipped", reason="budget exceeded")
        assert display._format_detail(state) == "budget exceeded"

    def test_skipped_without_reason(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="skipped")
        assert display._format_detail(state) == ""

    def test_pending_no_detail(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="pending")
        assert display._format_detail(state) == ""

    def test_success_no_detail(self) -> None:
        from maestro_cli.live import _TaskState

        display = _LivePlanDisplay(_make_plan())
        state = _TaskState(task_id="t", status="success")
        assert display._format_detail(state) == ""


class TestLiveProgressBar:
    """Tests for _progress_bar edge cases."""

    def test_negative_total(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        assert display._progress_bar(0, -1) == "\u2500" * 16

    def test_completed_exceeds_total(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        # Should be capped at width
        bar = display._progress_bar(20, 10)
        assert bar == "\u2501" * 16

    def test_one_of_many(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        bar = display._progress_bar(1, 16)
        assert bar == "\u2501" + "\u2500" * 15


class TestLiveEventHandling:
    """Tests for various event handling edge cases."""

    def test_unknown_event_sets_message(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("custom_event_name", {})
        assert display._last_event_message == "custom event name"

    def test_task_skip_no_reason(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_skip", {"task_id": "t1"})
        state = display._tasks["t1"]
        assert state.status == "skipped"
        assert state.reason is None
        assert display._last_event_message == "t1 skipped"

    def test_task_skip_with_reason(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_skip", {"task_id": "t1", "reason": "dependency failed"})
        assert display._last_event_message == "t1 skipped: dependency failed"

    def test_task_complete_failed_status_label(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "failed", "duration_sec": 2.0, "cost_usd": 0.0},
        )
        assert "failed" in display._last_event_message

    def test_task_complete_soft_failed_status_label(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "soft_failed", "duration_sec": 1.0, "cost_usd": 0.0},
        )
        assert "soft failed" in display._last_event_message

    def test_task_complete_dry_run_label(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "dry_run", "duration_sec": 0.0, "cost_usd": 0.0},
        )
        assert "completed" in display._last_event_message

    def test_task_complete_missing_status_defaults_success(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "duration_sec": 1.0, "cost_usd": 0.0},
        )
        assert display._tasks["t1"].status == "success"

    def test_task_start_resets_previous_state(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        # First run, complete with data
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude", "model": "haiku"})
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "failed", "duration_sec": 5.0, "cost_usd": 1.0},
        )
        # Re-start resets state
        display.handle_event("task_start", {"task_id": "t1", "engine": "codex", "model": "5.4"})
        state = display._tasks["t1"]
        assert state.status == "running"
        assert state.engine == "codex"
        assert state.model == "5.4"
        assert state.duration_sec is None
        assert state.cost_usd is None
        assert state.reason is None
        assert state.last_line == ""
        assert state.started_at is not None

    def test_task_start_max_retries_from_payload(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude", "max_retries": 3})
        assert display._tasks["t1"].max_retries == 3

    def test_task_output_empty_line_ignored(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event("task_output", {"task_id": "t1", "line": ""})
        assert display._tasks["t1"].last_line == ""

    def test_task_output_whitespace_only_ignored(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event("task_output", {"task_id": "t1", "line": "   "})
        assert display._tasks["t1"].last_line == ""

    def test_verify_failure_event(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event(
            "verify_failure",
            {"task_id": "t1", "exit_code": 1, "output_snippet": "assertion failed: x != y"},
        )
        assert "verify failed t1" in display._last_event_message
        assert "exit 1" in display._last_event_message
        assert "assertion failed" in display._last_event_message

    def test_verify_failure_long_snippet_truncated(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event(
            "verify_failure",
            {"task_id": "t1", "exit_code": 2, "output_snippet": "A" * 100},
        )
        # The snippet is truncated to 60 chars
        assert len(display._last_event_message) < 200

    def test_task_progress_with_pct(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event("task_progress", {"task_id": "t1", "pct": 42})
        assert display._tasks["t1"].progress_pct == 42
        assert "42%" in display._last_event_message

    def test_task_progress_with_step(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event("task_progress", {"task_id": "t1", "pct": 75, "step": "compiling"})
        assert display._tasks["t1"].last_line == "compiling"
        assert display._tasks["t1"].progress_pct == 75

    def test_task_artifact_with_label(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_artifact", {"task_id": "t1", "label": "report.html"})
        assert display._last_event_message == "t1 artifact: report.html"

    def test_task_artifact_with_path_fallback(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_artifact", {"task_id": "t1", "path": "/tmp/out.json"})
        assert display._last_event_message == "t1 artifact: /tmp/out.json"

    def test_budget_warning_without_limit(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("budget_warning", {"spent": 3.0})
        assert "$3.00" in display._budget_warning  # type: ignore[operator]
        assert "limit" not in display._budget_warning  # type: ignore[operator]

    def test_budget_warning_without_pct(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("budget_warning", {"spent": 3.0, "limit": 5.0})
        assert "$3.00" in display._budget_warning  # type: ignore[operator]
        assert "$5.00" in display._budget_warning  # type: ignore[operator]
        assert "%" not in display._budget_warning  # type: ignore[operator]

    def test_run_complete_failed(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("run_complete", {"success": False, "duration_sec": 5.0, "cost_usd": 0.0})
        assert "failed" in display._last_event_message
        assert "5.0s" in display._last_event_message

    def test_run_complete_success(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("run_complete", {"success": True, "duration_sec": 10.0, "cost_usd": 2.5})
        assert "success" in display._last_event_message
        assert "$2.50" in display._last_event_message


class TestLiveStaticHelpers:
    """Tests for _as_str, _as_float, _as_int static helpers."""

    def test_as_str_with_string(self) -> None:
        assert _LivePlanDisplay._as_str("hello") == "hello"

    def test_as_str_with_non_string(self) -> None:
        assert _LivePlanDisplay._as_str(42) is None
        assert _LivePlanDisplay._as_str(None) is None
        assert _LivePlanDisplay._as_str(3.14) is None
        assert _LivePlanDisplay._as_str([]) is None

    def test_as_float_with_float(self) -> None:
        assert _LivePlanDisplay._as_float(1.5) == 1.5

    def test_as_float_with_int(self) -> None:
        assert _LivePlanDisplay._as_float(3) == 3.0

    def test_as_float_with_non_numeric(self) -> None:
        assert _LivePlanDisplay._as_float("abc") is None
        assert _LivePlanDisplay._as_float(None) is None
        assert _LivePlanDisplay._as_float([]) is None

    def test_as_int_with_int(self) -> None:
        assert _LivePlanDisplay._as_int(5) == 5

    def test_as_int_with_non_int(self) -> None:
        assert _LivePlanDisplay._as_int(5.5) is None
        assert _LivePlanDisplay._as_int("5") is None
        assert _LivePlanDisplay._as_int(None) is None


class TestLiveGetTaskState:
    """Tests for _get_task_state and _get_or_create_state."""

    def test_get_task_state_known(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        state = display._get_task_state({"task_id": "t1"})
        assert state.task_id == "t1"

    def test_get_task_state_unknown_creates(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        state = display._get_task_state({"task_id": "new-task"})
        assert state.task_id == "new-task"
        assert "new-task" in display._tasks

    def test_get_task_state_missing_task_id(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        state = display._get_task_state({})
        assert state.task_id == "(unknown)"

    def test_get_or_create_empty_string(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        state = display._get_or_create_state("")
        assert state.task_id == "(unknown)"

    def test_get_or_create_idempotent(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        s1 = display._get_or_create_state("foo")
        s2 = display._get_or_create_state("foo")
        assert s1 is s2


class TestLiveBuildHeader:
    """Tests for _build_header content and format."""

    def test_header_contains_plan_name(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        header = display._build_header().plain
        assert "live-test-plan" in header

    def test_header_contains_maestro_label(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        header = display._build_header().plain
        assert "MAESTRO" in header

    def test_header_shows_progress_count(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        header = display._build_header().plain
        assert "0/2" in header

    def test_header_updates_after_task_complete(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "success", "duration_sec": 1.0, "cost_usd": 0.5},
        )
        header = display._build_header().plain
        assert "1/2" in header

    def test_header_cost_from_tasks_when_no_run_total(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "success", "duration_sec": 1.0, "cost_usd": 1.25},
        )
        header = display._build_header().plain
        assert "$1.25" in header

    def test_header_cost_uses_run_total_when_available(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "success", "duration_sec": 1.0, "cost_usd": 1.25},
        )
        display.handle_event("run_complete", {"success": True, "duration_sec": 5.0, "cost_usd": 3.0})
        header = display._build_header().plain
        assert "$3.00" in header

    def test_header_no_cost_data_shows_dash(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        header = display._build_header().plain
        assert "--" in header

    def test_header_includes_budget_warning(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("budget_warning", {"spent": 8.0, "limit": 10.0, "pct": 0.8})
        header = display._build_header().plain
        assert "budget warning" in header


class TestLiveBuildFooter:
    """Tests for _build_footer."""

    def test_footer_shows_last_event(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        footer = display._build_footer().plain
        assert "Last: waiting for events" in footer

    def test_footer_updates_after_event(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        footer = display._build_footer().plain
        assert "Last: t1 started" in footer


class TestLiveBuildTable:
    """Tests for _build_table rendering."""

    def test_table_renders_without_error(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        table = display._build_table()
        text = _render_text(table)
        assert "t1" in text
        assert "t2" in text

    def test_table_with_all_statuses(self) -> None:
        from maestro_cli.live import _TaskState

        tasks = [
            TaskSpec(id="a", description="a", command="echo"),
            TaskSpec(id="b", description="b", command="echo"),
        ]
        display = _LivePlanDisplay(_make_plan(tasks))
        display._tasks["a"].status = "success"
        display._tasks["a"].engine = "claude"
        display._tasks["a"].duration_sec = 2.0
        display._tasks["a"].cost_usd = 0.5
        display._tasks["b"].status = "failed"
        display._tasks["b"].engine = "codex"
        display._tasks["b"].duration_sec = 10.0
        display._tasks["b"].cost_usd = 1.0
        table = display._build_table()
        text = _render_text(table)
        assert "[ok]" in text
        assert "[!!]" in text

    def test_rich_renderable(self) -> None:
        """__rich__ returns the same as _build_table."""
        display = _LivePlanDisplay(_make_plan())
        table_from_rich = display.__rich__()
        text = _render_text(table_from_rich)
        assert "t1" in text

    def test_table_flushes_pending_output(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_output", {"task_id": "t1", "line": "first"})
        display.handle_event("task_output", {"task_id": "t1", "line": "buffered"})
        # "buffered" is in pending
        assert "t1" in display._pending_output
        # Force flush via table build with old timestamp
        display._last_output_update_at = 0.0
        display._build_table()
        assert display._pending_output == {}
        assert display._tasks["t1"].last_line == "buffered"


class TestLiveFlushPendingOutput:
    """Tests for _flush_pending_output_locked."""

    def test_no_pending_is_noop(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display._flush_pending_output_locked()
        # No error, no crash

    def test_force_flush_ignores_throttle(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display._pending_output = {"t1": "hello"}
        display._last_output_update_at = time.monotonic()  # very recent
        display._flush_pending_output_locked(force=True)
        assert display._pending_output == {}
        assert display._tasks["t1"].last_line == "hello"

    def test_non_force_respects_throttle(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display._pending_output = {"t1": "hello"}
        display._last_output_update_at = time.monotonic()  # very recent
        display._flush_pending_output_locked(force=False)
        # Still pending because throttle hasn't elapsed
        assert "t1" in display._pending_output


class TestLiveCostDisplay:
    """Tests for cost display formatting in the live display."""

    def test_zero_cost(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "success", "duration_sec": 1.0, "cost_usd": 0.0},
        )
        assert display._tasks["t1"].cost_usd == 0.0
        assert "$0.00" in display._last_event_message

    def test_real_cost(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "success", "duration_sec": 1.0, "cost_usd": 5.67},
        )
        assert "$5.67" in display._last_event_message

    def test_none_cost(self) -> None:
        display = _LivePlanDisplay(_make_plan())
        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "success", "duration_sec": 1.0},
        )
        assert display._tasks["t1"].cost_usd is None
        assert "--" in display._last_event_message


class TestLiveMultipleEventSequences:
    """Tests for realistic multi-event sequences."""

    def test_full_lifecycle_two_tasks(self) -> None:
        display = _LivePlanDisplay(_make_plan())

        display.handle_event("task_start", {"task_id": "t1", "engine": "claude", "model": "sonnet"})
        display.handle_event("task_output", {"task_id": "t1", "line": "processing..."})
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "success", "duration_sec": 3.0, "cost_usd": 0.5},
        )

        display.handle_event("task_start", {"task_id": "t2", "engine": "codex", "model": "5.4"})
        display.handle_event("task_retry", {"task_id": "t2", "attempt": 1, "max_retries": 2})
        display.handle_event("task_escalation", {"task_id": "t2", "to_model": "opus"})
        display.handle_event(
            "task_complete",
            {"task_id": "t2", "status": "success", "duration_sec": 8.0, "cost_usd": 2.0},
        )

        display.handle_event("run_complete", {"success": True, "duration_sec": 11.0, "cost_usd": 2.5})

        assert display._tasks["t1"].status == "success"
        assert display._tasks["t2"].status == "success"
        assert display._tasks["t2"].model == "opus"
        assert display._run_total_cost_usd == 2.5

    def test_task_with_judge_sequence(self) -> None:
        display = _LivePlanDisplay(_make_plan())

        display.handle_event("task_start", {"task_id": "t1", "engine": "claude"})
        display.handle_event("judge_start", {"task_id": "t1"})
        assert "[judge]" in display._last_event_message
        display.handle_event("judge_result", {"task_id": "t1", "verdict": "pass"})
        assert "pass" in display._last_event_message
        display.handle_event(
            "task_complete",
            {"task_id": "t1", "status": "success", "duration_sec": 5.0, "cost_usd": 1.0},
        )
        assert display._tasks["t1"].status == "success"

    def test_skip_then_run_complete(self) -> None:
        display = _LivePlanDisplay(_make_plan())

        display.handle_event("task_skip", {"task_id": "t1", "reason": "budget"})
        display.handle_event("task_skip", {"task_id": "t2", "reason": "budget"})
        display.handle_event("run_complete", {"success": False, "duration_sec": 0.5, "cost_usd": 0.0})

        assert display._tasks["t1"].status == "skipped"
        assert display._tasks["t2"].status == "skipped"
        header = display._build_header().plain
        assert "2/2" in header  # both terminal
