from __future__ import annotations

import re
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

pytest.importorskip("rich")

from rich.console import Console

from maestro_cli.cli import main
from maestro_cli.live import create_live_callback
from maestro_cli.models import PlanDefaults, PlanSpec, TaskSpec
from maestro_cli.scheduler import run_plan


def _make_task(
    task_id: str,
    *,
    depends_on: list[str] | None = None,
    command: str = "echo ok",
    allow_failure: bool = False,
) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        description=f"task {task_id}",
        depends_on=depends_on or [],
        command=command,
        allow_failure=allow_failure,
    )


def _make_plan(
    tmp_path: Path,
    tasks: list[TaskSpec],
    *,
    name: str = "live-integration-plan",
    goal: str = "Stress-test live output before Phase B TUI",
    fail_fast: bool = True,
    max_parallel: int = 4,
    max_cost_usd: float | None = None,
    budget_warning_pct: float | None = None,
    filename: str = "plan.yaml",
) -> PlanSpec:
    plan_path = tmp_path / filename
    plan_path.write_text(f"name: {name}\n", encoding="utf-8")
    return PlanSpec(
        version=1,
        name=name,
        goal=goal,
        fail_fast=fail_fast,
        max_parallel=max_parallel,
        max_cost_usd=max_cost_usd,
        budget_warning_pct=budget_warning_pct,
        defaults=PlanDefaults(),
        tasks=tasks,
        source_path=plan_path,
    )


def _render_text(renderable: Any, *, styles: bool = False) -> str:
    console = Console(
        width=240,
        record=True,
        force_terminal=styles,
        color_system="standard" if styles else None,
    )
    console.print(renderable)
    return console.export_text(styles=styles)


def _display_text(live_obj: Any, *, styles: bool = False) -> str:
    return _render_text(live_obj.renderable, styles=styles)


def _mock_subprocess_success(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
    del kwargs
    cmd = args[0] if args else ""
    return subprocess.CompletedProcess(cmd, 0, stdout="ok\n", stderr="")


def test_live_with_fail_fast(tmp_path: Path) -> None:
    plan = _make_plan(
        tmp_path,
        [
            _make_task("t1"),
            _make_task("t2", depends_on=["t1"]),
            _make_task("t3"),
            _make_task("t4"),
            _make_task("t5"),
        ],
        fail_fast=True,
    )
    live_obj, callback = create_live_callback(plan)

    callback("task_start", {"task_id": "t1"})
    callback(
        "task_complete",
        {"task_id": "t1", "status": "failed", "duration_sec": 0.2, "cost_usd": 0.0},
    )
    for task_id in ("t2", "t3", "t4", "t5"):
        callback(
            "task_skip",
            {"task_id": task_id, "reason": "fail_fast triggered by task 't1'"},
        )

    text = _display_text(live_obj)

    assert re.search(r"\[!!\]\s+t1(?:\s|$)", text)
    for task_id in ("t2", "t3", "t4", "t5"):
        assert re.search(rf"\[--\]\s+{task_id}(?:\s|$)", text)


def test_live_with_budget_exceeded(tmp_path: Path) -> None:
    plan = _make_plan(
        tmp_path,
        [_make_task("t1"), _make_task("t2"), _make_task("t3")],
        fail_fast=False,
        max_cost_usd=1.0,
        budget_warning_pct=0.8,
    )
    live_obj, callback = create_live_callback(plan)
    display = live_obj.renderable

    callback(
        "task_complete",
        {"task_id": "t1", "status": "success", "duration_sec": 0.3, "cost_usd": 0.80},
    )
    callback("budget_warning", {"spent": 0.80, "limit": 1.0, "pct": 0.8})
    callback(
        "task_complete",
        {"task_id": "t2", "status": "success", "duration_sec": 0.4, "cost_usd": 0.30},
    )
    callback("task_skip", {"task_id": "t3", "reason": "Budget exceeded ($1.10 / $1.00 limit)"})

    header = display._build_header().plain
    text = _display_text(live_obj)

    assert "budget warning" in header
    assert "$0.80" in header
    assert "$1.00" in header
    assert "80%" in header
    assert re.search(r"\[--\]\s+t3(?:\s|$)", text)


def test_live_with_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _make_plan(
        tmp_path,
        [_make_task("t1"), _make_task("t2", depends_on=["t1"]), _make_task("t3")],
    )
    live_obj, callback = create_live_callback(plan)

    def _unexpected_subprocess_run(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise AssertionError("subprocess.run should not be called for dry_run=True")

    monkeypatch.setattr("maestro_cli.runners.subprocess.run", _unexpected_subprocess_run)

    result = run_plan(
        plan,
        dry_run=True,
        run_dir_override=str(tmp_path / "runs"),
        verbosity="quiet",
        event_callback=callback,
    )
    text = _display_text(live_obj)

    assert result.success is True
    for task_id in ("t1", "t2", "t3"):
        assert result.task_results[task_id].status == "dry_run"
        assert re.search(rf"\[ok\]\s+{task_id}(?:\s|$)", text)


def test_live_with_mixed_statuses(tmp_path: Path) -> None:
    plan = _make_plan(
        tmp_path,
        [
            _make_task("t1"),
            _make_task("t2"),
            _make_task("t3", allow_failure=True),
            _make_task("t4"),
            _make_task("t5"),
        ],
    )
    live_obj, callback = create_live_callback(plan)

    callback("task_complete", {"task_id": "t1", "status": "success", "duration_sec": 0.1, "cost_usd": 0.1})
    callback("task_complete", {"task_id": "t2", "status": "failed", "duration_sec": 0.2, "cost_usd": 0.0})
    callback(
        "task_complete",
        {"task_id": "t3", "status": "soft_failed", "duration_sec": 0.3, "cost_usd": 0.0},
    )
    callback("task_skip", {"task_id": "t4", "reason": "dependency failure"})
    callback("task_complete", {"task_id": "t5", "status": "success", "duration_sec": 0.4, "cost_usd": 0.2})

    plain_text = _display_text(live_obj)
    styled_text = _display_text(live_obj, styles=True)

    assert re.search(r"\[ok\]\s+t1(?:\s|$)", plain_text)
    assert re.search(r"\[!!\]\s+t2(?:\s|$)", plain_text)
    assert re.search(r"\[~~\]\s+t3(?:\s|$)", plain_text)
    assert re.search(r"\[--\]\s+t4(?:\s|$)", plain_text)
    assert re.search(r"\[ok\]\s+t5(?:\s|$)", plain_text)


def test_live_callback_receives_all_event_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _make_plan(
        tmp_path,
        [_make_task("t1"), _make_task("t2", depends_on=["t1"])],
        filename="events-plan.yaml",
    )
    events: list[str] = []

    monkeypatch.setattr("maestro_cli.runners.subprocess.run", _mock_subprocess_success)

    result = run_plan(
        plan,
        dry_run=True,
        run_dir_override=str(tmp_path / "runs"),
        verbosity="quiet",
        event_callback=lambda event_name, payload: events.append(event_name),
    )

    assert result.success is True
    assert {"run_start", "task_start", "task_complete", "run_complete"} <= set(events)


def test_live_output_suppresses_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan_file = tmp_path / "cli-live-plan.yaml"
    plan_file.write_text(
        "\n".join(
            [
                "version: 1",
                "name: cli-live-test",
                "goal: verify live output stays quiet",
                "tasks:",
                "  - id: t1",
                '    command: "echo hello"',
            ]
        ),
        encoding="utf-8",
    )

    captured_callback = object()
    mock_run_plan = MagicMock(return_value=SimpleNamespace(success=True))

    class _DummyLive:
        def __enter__(self) -> "_DummyLive":
            return self

        def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
            del exc_type, exc, tb
            return None

    monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run_plan)
    monkeypatch.setattr(
        "maestro_cli.live.create_live_callback",
        lambda plan: (_DummyLive(), captured_callback),
    )

    rc = main(
        [
            "run",
            str(plan_file),
            "--output",
            "live",
            "--run-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert rc == 0
    _, kwargs = mock_run_plan.call_args
    assert kwargs["verbosity"] == "quiet"

