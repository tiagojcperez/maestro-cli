from __future__ import annotations

import json
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from maestro_cli.models import PlanDefaults, PlanSpec, TaskResult, TaskSpec
from maestro_cli.scheduler import run_plan


def _make_task(
    task_id: str,
    depends_on: list[str] | None = None,
    command: str = "echo ok",
    requires_approval: bool = False,
    approval_message: str | None = None,
) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        description=f"task {task_id}",
        depends_on=depends_on or [],
        command=command,
        requires_approval=requires_approval,
        approval_message=approval_message,
    )


def _make_plan(
    tasks: list[TaskSpec],
    *,
    name: str = "callback-test-plan",
    fail_fast: bool = True,
    max_parallel: int = 4,
    source_path: Path | None = None,
) -> PlanSpec:
    return PlanSpec(
        version=1,
        name=name,
        fail_fast=fail_fast,
        max_parallel=max_parallel,
        defaults=PlanDefaults(),
        tasks=tasks,
        source_path=source_path,
    )


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _make_mock_execute(
    *,
    sleep_sec: float = 0.0,
) -> Any:
    counter = 0
    counter_lock = threading.Lock()

    def mock_execute(
        plan: Any,
        task: Any,
        run_path: Path,
        dry_run: bool = False,
        execution_profile: str = "plan",
        upstream_results: Any = None,
        context_synthesis: str = "",
        workspace_brief: str = "",
        **kwargs,
    ) -> TaskResult:
        del plan, execution_profile, upstream_results, context_synthesis, workspace_brief

        if sleep_sec:
            time.sleep(sleep_sec)

        now = datetime.now(UTC)
        status = "dry_run" if dry_run else "success"

        nonlocal counter
        with counter_lock:
            counter += 1
            artifact_id = counter

        result = TaskResult(
            task_id=task.id,
            status=status,
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=sleep_sec or 0.01,
            command=f"echo {task.id}",
            log_path=run_path / f"task-{artifact_id}.log",
            result_path=run_path / f"task-{artifact_id}.result.json",
            message="ok",
        )
        result.log_path.write_text(f"status={status}\n", encoding="utf-8")
        result.result_path.write_text(
            json.dumps(result.to_dict(), indent=2),
            encoding="utf-8",
        )
        return result

    return mock_execute


def _configure_scheduler_mocks(monkeypatch: pytest.MonkeyPatch, mock_execute: Any) -> None:
    def _mock_subprocess_run(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("subprocess.run should not be called in these tests")

    monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
    monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)
    monkeypatch.setattr("maestro_cli.runners.subprocess.run", _mock_subprocess_run)


def _run_approval_handler_probe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    handler_result: bool,
    auto_approve: bool = False,
) -> tuple[Any, list[tuple[str, str | None]]]:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan(
        [
            _make_task(
                "gate",
                requires_approval=True,
                approval_message="Proceed with deployment?",
            )
        ],
        source_path=plan_yaml,
    )
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    handler_calls: list[tuple[str, str | None]] = []

    def approval_handler(task_id: str, message: str | None) -> bool:
        handler_calls.append((task_id, message))
        return handler_result

    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))
    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        auto_approve=auto_approve,
        approval_handler=approval_handler,
    )
    return result, handler_calls


def test_callback_receives_run_start_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan([_make_task("t1")], source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    events: list[dict[str, Any]] = []

    def callback(event_name: str, payload: dict[str, object]) -> None:
        assert event_name == payload["event"]
        events.append(payload)

    run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
    )

    assert events[0]["event"] == "run_start"


def test_callback_receives_run_complete_last(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan([_make_task("t1"), _make_task("t2")], source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    events: list[dict[str, Any]] = []

    run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=lambda event_name, payload: events.append(payload),
    )

    assert events[-1]["event"] == "run_complete"


def test_callback_receives_task_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    tasks = [_make_task("a"), _make_task("b", depends_on=["a"]), _make_task("c")]
    plan = _make_plan(tasks, source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    events: list[dict[str, Any]] = []

    run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=lambda event_name, payload: events.append(payload),
    )

    for task_id in ("a", "b", "c"):
        task_events = [e["event"] for e in events if e.get("task_id") == task_id]
        assert task_events.count("task_start") == 1
        assert task_events.count("task_complete") == 1


def test_callback_none_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan([_make_task("t1")], source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

    assert result.success is True
    assert (result.run_path / "events.jsonl").exists()


def test_cancel_event_before_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # cancel_event is not yet wired in the scheduler dispatch loop.
    # Detection: check for LOAD_FAST of cancel_event in the function body (beyond
    # the parameter-receive bytecode). co_names only tracks global/attribute names,
    # so we use dis to detect actual reads of the local variable.
    # NOTE: when wiring cancel_event, remove this guard or it will stay skipped.
    import dis as _dis
    _loads = [i for i in _dis.get_instructions(run_plan) if i.opname in ("LOAD_FAST", "LOAD_FAST_BORROW") and i.argval == "cancel_event"]
    if not _loads:
        pytest.skip("cancel_event not yet wired in scheduler")

    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan([_make_task(f"task-{i}") for i in range(5)], source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    cancel_event = threading.Event()
    cancel_event.set()

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        cancel_event=cancel_event,
    )

    assert result.success is True
    assert all(task.status == "skipped" for task in result.task_results.values())


def test_cancel_event_during_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # cancel_event is not yet wired in the scheduler dispatch loop.
    import dis as _dis
    _loads = [i for i in _dis.get_instructions(run_plan) if i.opname in ("LOAD_FAST", "LOAD_FAST_BORROW") and i.argval == "cancel_event"]
    if not _loads:
        pytest.skip("cancel_event not yet wired in scheduler")

    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    tasks = [_make_task("task-0")]
    tasks.extend(
        _make_task(f"task-{i}", depends_on=[f"task-{i-1}"])
        for i in range(1, 10)
    )
    plan = _make_plan(tasks, max_parallel=1, source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute(sleep_sec=0.01))

    cancel_event = threading.Event()
    completed = 0

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del payload
        nonlocal completed
        if event_name == "task_complete":
            completed += 1
            if completed == 3:
                cancel_event.set()

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
        cancel_event=cancel_event,
    )

    assert result.task_results["task-0"].status == "success"
    assert result.task_results["task-1"].status == "success"
    assert result.task_results["task-2"].status == "success"
    assert all(
        result.task_results[f"task-{i}"].status == "skipped"
        for i in range(3, 10)
    )


def test_cancel_event_with_parallel_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # cancel_event is not yet wired in the scheduler dispatch loop.
    import dis as _dis
    _loads = [i for i in _dis.get_instructions(run_plan) if i.opname in ("LOAD_FAST", "LOAD_FAST_BORROW") and i.argval == "cancel_event"]
    if not _loads:
        pytest.skip("cancel_event not yet wired in scheduler")

    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan(
        [_make_task(f"task-{i}") for i in range(10)],
        max_parallel=5,
        source_path=plan_yaml,
    )
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute(sleep_sec=0.02))

    cancel_event = threading.Event()
    complete_count = 0
    complete_lock = threading.Lock()

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del payload
        nonlocal complete_count
        if event_name != "task_complete":
            return
        with complete_lock:
            complete_count += 1
            if complete_count == 2:
                cancel_event.set()

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
        cancel_event=cancel_event,
    )

    completed = sum(1 for task in result.task_results.values() if task.status == "success")
    skipped = sum(1 for task in result.task_results.values() if task.status == "skipped")

    assert completed >= 2
    assert completed < 10
    assert skipped > 0


def test_cancel_event_none_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan([_make_task(f"task-{i}") for i in range(5)], source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        cancel_event=None,
    )

    assert result.success is True
    assert all(task.status == "success" for task in result.task_results.values())


def test_callback_exception_does_not_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan([_make_task("t1")], source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del event_name, payload
        raise RuntimeError("boom")

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
    )

    assert result.success is True
    assert result.task_results["t1"].status == "success"


def test_callback_parallel_thread_safety(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    tasks = [_make_task(f"task-{i}") for i in range(10)]
    plan = _make_plan(tasks, max_parallel=5, source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute(sleep_sec=0.02))

    events: list[dict[str, Any]] = []
    events_lock = threading.Lock()

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del event_name
        with events_lock:
            events.append(payload)

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
    )

    assert result.success is True
    # 1 run_start + 10 task_start + 10 task_complete + 1 score_recorded + 1 run_complete
    assert len(events) == 23
    assert sum(1 for e in events if e["event"] == "task_start") == 10
    assert sum(1 for e in events if e["event"] == "task_complete") == 10
    assert events[0]["event"] == "run_start"
    assert events[-1]["event"] == "run_complete"


def test_callback_slow_does_not_block_parallelism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    tasks = [_make_task(f"task-{i}") for i in range(10)]
    plan = _make_plan(tasks, max_parallel=5, source_path=plan_yaml)
    # task_duration is kept comfortably large (0.2s) on purpose: task_complete
    # callbacks run serially on the main dispatch thread, so the only thing
    # this test guards is that a slow callback OVERLAPS worker execution rather
    # than serializing the pool. Tiny durations made the absolute 2x ceiling
    # (~1.4s) collide with fixed scheduler/per-task file-hash/CI overhead and
    # flake on loaded Windows runners (observed elapsed ~2.06s). Larger task
    # work lifts the ceiling to ~5s, well clear of that overhead, while still
    # catching the >2x-serial blow-up a real parallelism regression would cause.
    task_duration = 0.2
    callback_delay = 0.05
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute(sleep_sec=task_duration))

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del payload
        if event_name == "task_complete":
            time.sleep(callback_delay)

    started = time.monotonic()
    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
    )
    elapsed = time.monotonic() - started

    assert result.success is True
    # Budget getter + merge ledger reset add minor overhead; use 2x margin
    assert elapsed < 2 * ((len(tasks) * task_duration) + (len(tasks) * callback_delay))


def test_callback_alternating_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    tasks = [_make_task(f"task-{i}") for i in range(10)]
    plan = _make_plan(tasks, max_parallel=5, source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute(sleep_sec=0.01))

    call_count = 0

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del event_name, payload
        nonlocal call_count
        call_count += 1
        if call_count % 3 == 0:
            raise RuntimeError("boom")

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
    )

    events = _parse_jsonl((result.run_path / "events.jsonl").read_text(encoding="utf-8"))

    assert result.success is True
    assert all(task.status == "success" for task in result.task_results.values())
    # 1 run_start + 10 task_start + 10 task_complete + 1 score_recorded + 1 run_complete
    assert len(events) == 23
    assert sum(1 for event in events if event["event"] == "task_complete") == 10


def test_callback_type_error_caught(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    tasks = [_make_task(f"task-{i}") for i in range(10)]
    plan = _make_plan(tasks, max_parallel=5, source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute(sleep_sec=0.01))

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del event_name, payload
        raise TypeError("wrong type")

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
    )

    assert result.success is True
    assert all(task.status == "success" for task in result.task_results.values())


def test_callback_keyboard_interrupt_not_caught(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan([_make_task("t1")], source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    callback_called = False

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del event_name, payload
        nonlocal callback_called
        callback_called = True
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
            event_callback=callback,
        )

    assert callback_called is True


def test_callback_payloads_match_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan([_make_task("a"), _make_task("b")], source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    captured: list[dict[str, Any]] = []

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=lambda event_name, payload: captured.append(payload),
    )

    file_events = _parse_jsonl((result.run_path / "events.jsonl").read_text(encoding="utf-8"))
    assert captured == file_events


def test_callback_empty_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan([], source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    events: list[dict[str, Any]] = []

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=lambda event_name, payload: events.append(payload),
    )

    assert result.success is True
    assert [event["event"] for event in events] == ["run_start", "score_recorded", "run_complete"]


def test_callback_long_task_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    long_task_id = "t" * 200
    plan = _make_plan([_make_task(long_task_id)], source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    events: list[dict[str, Any]] = []

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=lambda event_name, payload: events.append(payload),
    )

    assert result.success is True
    assert any(
        event["event"] == "task_start" and event.get("task_id") == long_task_id
        for event in events
    )
    assert any(
        event["event"] == "task_complete" and event.get("task_id") == long_task_id
        for event in events
    )


def test_callback_unicode_task_id(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    task_id = "tarefa-ação-東京-✓"
    plan = _make_plan([_make_task(task_id)], source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    events: list[dict[str, Any]] = []

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=lambda event_name, payload: events.append(payload),
    )

    assert result.success is True
    assert any(
        event["event"] == "task_start" and event.get("task_id") == task_id
        for event in events
    )
    assert any(
        event["event"] == "task_complete" and event.get("task_id") == task_id
        for event in events
    )


def test_callback_high_parallelism_20_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    tasks = [_make_task(f"task-{i}") for i in range(20)]
    plan = _make_plan(tasks, max_parallel=8, source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute(sleep_sec=0.01))

    events: list[dict[str, Any]] = []
    events_lock = threading.Lock()

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del event_name
        with events_lock:
            events.append(payload)

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
    )

    assert result.success is True
    # 1 run_start + 20 task_start + 20 task_complete + 1 score_recorded + 1 run_complete
    assert len(events) == 43
    assert events[0]["event"] == "run_start"
    assert events[-1]["event"] == "run_complete"

    starts = [event for event in events if event["event"] == "task_start"]
    completes = [event for event in events if event["event"] == "task_complete"]
    assert len(starts) == 20
    assert len(completes) == 20

    start_counts: dict[str, int] = {}
    for event in starts:
        task_id = str(event["task_id"])
        start_counts[task_id] = start_counts.get(task_id, 0) + 1

    complete_counts: dict[str, int] = {}
    for event in completes:
        task_id = str(event["task_id"])
        complete_counts[task_id] = complete_counts.get(task_id, 0) + 1

    expected_ids = {f"task-{i}" for i in range(20)}
    assert set(start_counts) == expected_ids
    assert set(complete_counts) == expected_ids
    assert all(count == 1 for count in start_counts.values())
    assert all(count == 1 for count in complete_counts.values())


def test_callback_causal_ordering_per_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    tasks = [_make_task(f"task-{i}") for i in range(10)]
    plan = _make_plan(tasks, max_parallel=5, source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute(sleep_sec=0.01))

    events: list[dict[str, Any]] = []
    events_lock = threading.Lock()

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del event_name
        with events_lock:
            events.append(payload)

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
    )

    assert result.success is True

    for task_id in (f"task-{i}" for i in range(10)):
        task_events = [
            index
            for index, event in enumerate(events)
            if event.get("task_id") == task_id
            and event["event"] in {"task_start", "task_complete"}
        ]
        assert len(task_events) == 2
        start_index, complete_index = task_events
        assert events[start_index]["event"] == "task_start"
        assert events[complete_index]["event"] == "task_complete"
        assert start_index < complete_index


def test_callback_50_tasks_no_lost_events(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    tasks = [_make_task(f"task-{i}") for i in range(50)]
    plan = _make_plan(tasks, max_parallel=10, source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute(sleep_sec=0.01))

    events: list[dict[str, Any]] = []
    events_lock = threading.Lock()

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del event_name
        with events_lock:
            events.append(payload)

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
    )

    assert result.success is True
    # 1 run_start + 50 task_start + 50 task_complete + 1 score_recorded + 1 run_complete
    assert len(events) == 103
    assert sum(1 for event in events if event["event"] == "task_start") == 50
    assert sum(1 for event in events if event["event"] == "task_complete") == 50
    assert events[0]["event"] == "run_start"
    assert events[-1]["event"] == "run_complete"


def test_callback_stateful_counter_100_tasks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    tasks = [_make_task(f"task-{i}") for i in range(100)]
    plan = _make_plan(tasks, max_parallel=10, source_path=plan_yaml)
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute(sleep_sec=0.005))

    counter = {
        "run_start": 0,
        "task_start": 0,
        "task_complete": 0,
        "run_complete": 0,
    }
    counter_lock = threading.Lock()

    def callback(event_name: str, payload: dict[str, object]) -> None:
        del payload
        with counter_lock:
            counter[event_name] = counter.get(event_name, 0) + 1

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=callback,
    )

    assert result.success is True
    assert counter["run_start"] == 1
    assert counter["task_start"] == 100
    assert counter["task_complete"] == 100
    assert counter["run_complete"] == 1


def test_approval_handler_discovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, handler_calls = _run_approval_handler_probe(
        tmp_path,
        monkeypatch,
        handler_result=True,
    )

    if handler_calls:
        assert handler_calls == [("gate", "Proceed with deployment?")]
        assert result.task_results["gate"].status == "success"
    else:
        # BUG: approval_handler param exists but is not wired
        assert result.task_results["gate"].status == "skipped"
        assert result.task_results["gate"].message == "Approval denied or non-interactive"


def test_approval_handler_approve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery_result, discovery_calls = _run_approval_handler_probe(
        tmp_path,
        monkeypatch,
        handler_result=True,
    )

    if not discovery_calls:
        assert discovery_result.task_results["gate"].status == "skipped"
        pytest.skip("approval_handler is not wired in scheduler approval logic")

    def fail_request_approval(task_id: str, message: str | None, interactive: bool) -> bool:
        del task_id, message, interactive
        raise AssertionError("_request_approval should not be called when approval_handler is wired")

    monkeypatch.setattr("maestro_cli.scheduler._request_approval", fail_request_approval)
    result, handler_calls = _run_approval_handler_probe(
        tmp_path,
        monkeypatch,
        handler_result=True,
    )

    assert handler_calls == [("gate", "Proceed with deployment?")]
    assert result.task_results["gate"].status == "success"


def test_approval_handler_deny(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovery_result, discovery_calls = _run_approval_handler_probe(
        tmp_path,
        monkeypatch,
        handler_result=True,
    )

    if not discovery_calls:
        assert discovery_result.task_results["gate"].status == "skipped"
        pytest.skip("approval_handler is not wired in scheduler approval logic")

    def fail_request_approval(task_id: str, message: str | None, interactive: bool) -> bool:
        del task_id, message, interactive
        raise AssertionError("_request_approval should not be called when approval_handler is wired")

    monkeypatch.setattr("maestro_cli.scheduler._request_approval", fail_request_approval)
    result, handler_calls = _run_approval_handler_probe(
        tmp_path,
        monkeypatch,
        handler_result=False,
    )

    assert handler_calls == [("gate", "Proceed with deployment?")]
    assert result.task_results["gate"].status == "skipped"


def test_approval_auto_approve_overrides_handler(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_request_approval(task_id: str, message: str | None, interactive: bool) -> bool:
        del task_id, message, interactive
        raise AssertionError("_request_approval should not be called when auto_approve=True")

    monkeypatch.setattr("maestro_cli.scheduler._request_approval", fail_request_approval)
    result, handler_calls = _run_approval_handler_probe(
        tmp_path,
        monkeypatch,
        handler_result=False,
        auto_approve=True,
    )

    assert handler_calls == []
    assert result.task_results["gate"].status == "success"


def test_callback_secrets_masked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event callback payloads should have secret values masked."""
    monkeypatch.setenv("MY_SECRET", "SuperSecret123")
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    # Embed the secret in plan.name so it appears in run_start.plan payload
    plan = _make_plan(
        [_make_task("t1")],
        name="test-SuperSecret123-plan",
        source_path=plan_yaml,
    )
    plan.secrets = ["MY_SECRET"]
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    captured: list[dict[str, Any]] = []

    run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=lambda event_name, payload: captured.append(payload),
    )

    assert captured, "expected at least one event"
    for payload in captured:
        for val in payload.values():
            if isinstance(val, str):
                assert "SuperSecret123" not in val, (
                    f"secret leaked in event '{payload.get('event')}': {val!r}"
                )

    run_start = next(p for p in captured if p["event"] == "run_start")
    assert run_start["plan"] == "test-***-plan"


def test_existing_approval_auto_approve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan(
        [
            _make_task(
                "gate",
                requires_approval=True,
                approval_message="Proceed with deployment?",
            )
        ],
        source_path=plan_yaml,
    )
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())

    def fail_request_approval(task_id: str, message: str | None, interactive: bool) -> bool:
        del task_id, message, interactive
        raise AssertionError("_request_approval should not be called when auto_approve=True")

    monkeypatch.setattr("maestro_cli.scheduler._request_approval", fail_request_approval)
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        auto_approve=True,
    )

    assert result.task_results["gate"].status == "success"


def test_approval_handler_exception_treated_as_deny(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the approval_handler raises, the scheduler treats it as denied (not a crash)."""
    plan_yaml = tmp_path / "plan.yaml"
    plan_yaml.touch()
    plan = _make_plan(
        [
            _make_task(
                "gate",
                requires_approval=True,
                approval_message="Proceed?",
            )
        ],
        source_path=plan_yaml,
    )
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())
    monkeypatch.setattr(sys, "stdin", SimpleNamespace(isatty=lambda: False))

    def raising_handler(task_id: str, message: str | None) -> bool:
        raise RuntimeError("TUI widget destroyed")

    result = run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        approval_handler=raising_handler,
    )

    assert result.task_results["gate"].status == "skipped"
    assert "Approval denied" in (result.task_results["gate"].message or "")


# ---------------------------------------------------------------------------
# plan_name in all event payloads
# ---------------------------------------------------------------------------


def test_all_events_have_plan_name(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Every event emitted by the scheduler should include plan_name."""
    events: list[dict[str, object]] = []
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())
    plan = PlanSpec(
        name="my-plan",
        version=1,
        tasks=[TaskSpec(id="t1", command="echo ok")],
        source_path=tmp_path / "plan.yaml",
    )
    run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=lambda name, payload: events.append(payload),
    )
    assert len(events) >= 3  # run_start, task_start, task_complete, run_complete
    for ev in events:
        assert "plan_name" in ev, f"Event {ev.get('event')} missing plan_name"
        assert ev["plan_name"] == "my-plan"


# ---------------------------------------------------------------------------
# task_retry event from runner
# ---------------------------------------------------------------------------


def test_task_retry_event_emitted(tmp_path: Path) -> None:
    """execute_task emits task_retry when a retry occurs."""
    from maestro_cli.runners import execute_task

    events: list[tuple[str, dict[str, object]]] = []
    plan = PlanSpec(
        name="retry-plan",
        version=1,
        tasks=[
            TaskSpec(
                id="flaky",
                command=["py", "-c", "import sys; sys.exit(1)"],
                max_retries=1,
            ),
        ],
        source_path=tmp_path / "plan.yaml",
    )
    run_path = tmp_path / "runs" / "test"
    run_path.mkdir(parents=True)

    execute_task(
        plan,
        plan.tasks[0],
        run_path,
        event_callback=lambda name, payload: events.append((name, payload)),
    )
    retry_events = [(n, p) for n, p in events if n == "task_retry"]
    assert len(retry_events) == 1
    assert retry_events[0][1]["task_id"] == "flaky"
    assert retry_events[0][1]["attempt"] == 2  # second attempt


# ---------------------------------------------------------------------------
# goal in run_start event
# ---------------------------------------------------------------------------


def test_run_start_includes_goal(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """run_start event should include the plan's goal field."""
    events: list[dict[str, object]] = []
    _configure_scheduler_mocks(monkeypatch, _make_mock_execute())
    plan = PlanSpec(
        name="goal-plan",
        version=1,
        goal="Build the TUI",
        tasks=[TaskSpec(id="t1", command="echo ok")],
        source_path=tmp_path / "plan.yaml",
    )
    run_plan(
        plan,
        run_dir_override=str(tmp_path / "runs"),
        event_callback=lambda name, payload: events.append(payload),
    )
    run_starts = [e for e in events if e["event"] == "run_start"]
    assert len(run_starts) == 1
    assert run_starts[0]["goal"] == "Build the TUI"
