from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.models import PlanDefaults, PlanSpec, TaskResult, TaskSpec, TokenUsage
from maestro_cli.scheduler import run_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(task_id: str, depends_on: list[str] | None = None, command: str = "echo ok") -> TaskSpec:
    return TaskSpec(id=task_id, description=f"task {task_id}", depends_on=depends_on or [], command=command)


def _make_plan(
    tasks: list[TaskSpec],
    name: str = "events-test-plan",
    fail_fast: bool = True,
    max_parallel: int = 4,
    max_cost_usd: float | None = None,
    source_path: Path | None = None,
) -> PlanSpec:
    plan = PlanSpec(
        version=1,
        name=name,
        fail_fast=fail_fast,
        max_parallel=max_parallel,
        defaults=PlanDefaults(),
        tasks=tasks,
        source_path=source_path,
    )
    plan.max_cost_usd = max_cost_usd
    return plan


def _mock_success_execute(
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
    now = datetime.now(UTC)
    status = "dry_run" if dry_run else "success"
    result = TaskResult(
        task_id=task.id,
        status=status,
        exit_code=0,
        started_at=now,
        finished_at=now,
        duration_sec=0.01,
        command=f"echo {task.id}",
        log_path=run_path / f"{task.id}.log",
        result_path=run_path / f"{task.id}.result.json",
        message="ok",
    )
    result.log_path.write_text(f"status={status}\n", encoding="utf-8")
    result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _mock_costly_execute(
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
    now = datetime.now(UTC)
    result = TaskResult(
        task_id=task.id,
        status="success",
        exit_code=0,
        started_at=now,
        finished_at=now,
        duration_sec=0.01,
        command=f"echo {task.id}",
        log_path=run_path / f"{task.id}.log",
        result_path=run_path / f"{task.id}.result.json",
        message="ok",
        cost_usd=2.00,
        token_usage=TokenUsage(input_tokens=500, output_tokens=500),
    )
    result.log_path.write_text("status=success\n", encoding="utf-8")
    result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEventsFileCreated:
    def test_events_file_exists_after_run(self, tmp_path: Path, monkeypatch: Any) -> None:
        """events.jsonl is created in the run directory after a plan run."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events_file = result.run_path / "events.jsonl"
        assert events_file.exists()

    def test_events_file_not_empty(self, tmp_path: Path, monkeypatch: Any) -> None:
        """events.jsonl has content after a run."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events_text = (result.run_path / "events.jsonl").read_text(encoding="utf-8")
        assert events_text.strip() != ""


class TestEventOrdering:
    def test_run_start_is_first_event(self, tmp_path: Path, monkeypatch: Any) -> None:
        """run_start is the first event in events.jsonl."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl((result.run_path / "events.jsonl").read_text(encoding="utf-8"))
        assert events[0]["event"] == "run_start"

    def test_run_complete_is_last_event(self, tmp_path: Path, monkeypatch: Any) -> None:
        """run_complete is the last event in events.jsonl."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan(
            [_make_task("t1"), _make_task("t2")],
            source_path=tmp_path / "plan.yaml",
        )

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl((result.run_path / "events.jsonl").read_text(encoding="utf-8"))
        assert events[-1]["event"] == "run_complete"

    def test_task_start_before_task_complete(self, tmp_path: Path, monkeypatch: Any) -> None:
        """task_start for t1 appears before task_complete for t1."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl((result.run_path / "events.jsonl").read_text(encoding="utf-8"))
        names = [e["event"] for e in events]
        assert names.index("task_start") < names.index("task_complete")


class TestEventFields:
    def test_task_events_include_required_fields(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """task_start and task_complete events carry expected fields."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl((result.run_path / "events.jsonl").read_text(encoding="utf-8"))
        starts = [e for e in events if e["event"] == "task_start"]
        completes = [e for e in events if e["event"] == "task_complete"]

        assert len(starts) == 1
        assert "task_id" in starts[0]
        assert "ts" in starts[0]
        assert "wave" in starts[0]

        assert len(completes) == 1
        assert "task_id" in completes[0]
        assert "status" in completes[0]
        assert "duration_sec" in completes[0]
        assert "cost_usd" in completes[0]
        assert "tokens" in completes[0]
        assert "ts" in completes[0]

    def test_run_start_fields(self, tmp_path: Path, monkeypatch: Any) -> None:
        """run_start event carries plan, run_id, tasks, max_parallel."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan(
            [_make_task("t1"), _make_task("t2")],
            source_path=tmp_path / "plan.yaml",
        )

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl((result.run_path / "events.jsonl").read_text(encoding="utf-8"))
        run_start = next(e for e in events if e["event"] == "run_start")

        assert run_start["plan"] == "events-test-plan"
        assert run_start["tasks"] == 2
        assert "run_id" in run_start
        assert "max_parallel" in run_start

    def test_run_complete_fields(self, tmp_path: Path, monkeypatch: Any) -> None:
        """run_complete event carries success, ok, failed, skipped, duration_sec."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl((result.run_path / "events.jsonl").read_text(encoding="utf-8"))
        run_end = next(e for e in events if e["event"] == "run_complete")

        assert run_end["success"] is True
        assert run_end["ok"] == 1
        assert run_end["failed"] == 0
        assert run_end["skipped"] == 0
        assert isinstance(run_end["duration_sec"], float)


class TestEventJson:
    def test_all_lines_are_valid_json(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Every line in events.jsonl is parseable as JSON."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan(
            [_make_task("t1"), _make_task("t2", depends_on=["t1"])],
            source_path=tmp_path / "plan.yaml",
        )

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events_text = (result.run_path / "events.jsonl").read_text(encoding="utf-8")
        for line in events_text.splitlines():
            if line.strip():
                json.loads(line)  # raises if invalid

    def test_event_timestamps_are_iso8601(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Every event's 'ts' field is a valid ISO 8601 timestamp."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl((result.run_path / "events.jsonl").read_text(encoding="utf-8"))
        for evt in events:
            assert "ts" in evt, f"Missing 'ts' in event: {evt}"
            # fromisoformat raises on invalid format
            datetime.fromisoformat(evt["ts"])


class TestBudgetExceededEvent:
    def test_budget_exceeded_in_events_file(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """budget_exceeded event is recorded in events.jsonl when soft limit is crossed."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_costly_execute)
        plan = _make_plan(
            [_make_task("t1"), _make_task("t2")],
            fail_fast=False,
            max_parallel=1,
            max_cost_usd=0.50,
            source_path=tmp_path / "plan.yaml",
        )

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl((result.run_path / "events.jsonl").read_text(encoding="utf-8"))
        budget_events = [e for e in events if e["event"] == "budget_exceeded"]
        assert len(budget_events) == 1
        assert "spent" in budget_events[0]
        assert "limit" in budget_events[0]
        assert budget_events[0]["limit"] == 0.50

    def test_budget_exceeded_event_before_run_complete(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        """budget_exceeded event precedes run_complete in events.jsonl."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_costly_execute)
        plan = _make_plan(
            [_make_task("t1"), _make_task("t2")],
            fail_fast=False,
            max_parallel=1,
            max_cost_usd=0.50,
            source_path=tmp_path / "plan.yaml",
        )

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl((result.run_path / "events.jsonl").read_text(encoding="utf-8"))
        names = [e["event"] for e in events]
        if "budget_exceeded" in names:
            assert names.index("budget_exceeded") < names.index("run_complete")
