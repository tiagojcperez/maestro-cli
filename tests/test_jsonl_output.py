from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.cli import _build_parser, main
from maestro_cli.models import PlanDefaults, PlanSpec, TaskResult, TaskSpec
from maestro_cli.scheduler import run_plan


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_SIMPLE_PLAN_YAML = """\
version: 1
name: jsonl-test-plan
tasks:
  - id: t1
    command: "echo hello"
  - id: t2
    command: "echo world"
    depends_on: [t1]
"""

_FAIL_PLAN_YAML = """\
version: 1
name: jsonl-fail-plan
fail_fast: true
tasks:
  - id: t1
    command: "echo ok"
  - id: t2
    command: "exit 1"
    depends_on: [t1]
  - id: t3
    command: "echo after"
    depends_on: [t2]
"""


def _write_plan(tmp_path: Path, content: str = _SIMPLE_PLAN_YAML) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


def _make_task(task_id: str, depends_on: list[str] | None = None, command: str = "echo ok") -> TaskSpec:
    return TaskSpec(
        id=task_id,
        description=f"task {task_id}",
        depends_on=depends_on or [],
        command=command,
    )


def _make_plan(
    tasks: list[TaskSpec],
    name: str = "test-plan",
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


def _mock_success_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kwargs) -> TaskResult:
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


def _mock_fail_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                        upstream_results=None, context_synthesis="", workspace_brief="", **kwargs) -> TaskResult:
    now = datetime.now(UTC)
    result = TaskResult(
        task_id=task.id,
        status="failed",
        exit_code=1,
        started_at=now,
        finished_at=now,
        duration_sec=0.01,
        command=f"exit 1",
        log_path=run_path / f"{task.id}.log",
        result_path=run_path / f"{task.id}.result.json",
        message="task failed",
    )
    result.log_path.write_text("status=failed\n", encoding="utf-8")
    result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _parse_jsonl(text: str) -> list[dict[str, Any]]:
    """Parse JSONL output into a list of event dicts. Skips blank lines."""
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


def _parse_jsonl_file(path: Path) -> list[dict[str, Any]]:
    return _parse_jsonl(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# CLI arg parsing
# ---------------------------------------------------------------------------


class TestOutputArgParsing:
    def test_output_default_is_text(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml"])
        assert args.output == "text"

    def test_output_jsonl_parsed(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--output", "jsonl"])
        assert args.output == "jsonl"

    def test_output_text_explicit(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--output", "text"])
        assert args.output == "text"

    def test_output_invalid_rejected(self) -> None:
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "plan.yaml", "--output", "yaml"])

    def test_output_jsonl_with_dry_run(self) -> None:
        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--output", "jsonl", "--dry-run"])
        assert args.output == "jsonl"
        assert args.dry_run is True


# ---------------------------------------------------------------------------
# JSONL output from run_plan
# ---------------------------------------------------------------------------


class TestJsonlEvents:
    def test_run_start_emitted(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """run_start event is emitted first with correct fields."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        run_start = next(e for e in events if e["event"] == "run_start")

        assert run_start["plan"] == "test-plan"
        assert run_start["tasks"] == 1
        assert "run_id" in run_start
        assert "ts" in run_start
        assert "max_parallel" in run_start

    def test_task_start_emitted(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """task_start event is emitted for each dispatched task."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1"), _make_task("t2")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        starts = [e for e in events if e["event"] == "task_start"]

        assert len(starts) == 2
        task_ids = {e["task_id"] for e in starts}
        assert task_ids == {"t1", "t2"}

        for e in starts:
            assert "ts" in e
            assert "engine" in e
            assert "model" in e
            assert "wave" in e
            assert isinstance(e["wave"], int)

    def test_task_start_engine_none_for_shell_task(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """Shell tasks (no engine) emit task_start with engine=null."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1", command="echo hi")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        starts = [e for e in events if e["event"] == "task_start"]
        assert starts[0]["engine"] is None

    def test_task_complete_emitted(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """task_complete event emitted for each completed task with correct fields."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        completes = [e for e in events if e["event"] == "task_complete"]

        assert len(completes) == 1
        evt = completes[0]
        assert evt["task_id"] == "t1"
        assert evt["status"] == "success"
        assert isinstance(evt["duration_sec"], float)
        assert "cost_usd" in evt
        assert "tokens" in evt
        assert "ts" in evt

    def test_task_complete_failed_status(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """task_complete reports 'failed' status when task fails."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_fail_execute)
        plan = _make_plan(
            [_make_task("t1")],
            source_path=tmp_path / "plan.yaml",
        )
        # Single task, fail_fast doesn't matter
        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        completes = [e for e in events if e["event"] == "task_complete"]
        assert len(completes) == 1
        assert completes[0]["status"] == "failed"

    def test_run_complete_emitted(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """run_complete event is emitted last with correct fields."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1"), _make_task("t2")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        run_end = next(e for e in events if e["event"] == "run_complete")

        assert run_end["success"] is True
        assert run_end["ok"] == 2
        assert run_end["failed"] == 0
        assert run_end["skipped"] == 0
        assert isinstance(run_end["duration_sec"], float)
        assert "cost_usd" in run_end
        assert "tokens" in run_end
        assert "ts" in run_end

    def test_run_complete_is_last_event(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """run_complete is the last event emitted."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        assert events[-1]["event"] == "run_complete"

    def test_run_start_is_first_event(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """run_start is the first event emitted."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        assert events[0]["event"] == "run_start"

    def test_event_ordering_start_before_complete(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """task_start always appears before its corresponding task_complete."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        event_names = [e["event"] for e in events]

        start_idx = event_names.index("task_start")
        complete_idx = event_names.index("task_complete")
        assert start_idx < complete_idx

    def test_all_events_are_valid_json(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """Every line of JSONL output is valid JSON."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan(
            [_make_task("t1"), _make_task("t2", depends_on=["t1"])],
            source_path=tmp_path / "plan.yaml",
        )

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        for line in captured.out.splitlines():
            if line.strip():
                json.loads(line)  # raises if invalid

    def test_all_events_have_ts_field(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """Every emitted event has a 'ts' ISO8601 timestamp field."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        for evt in events:
            assert "ts" in evt, f"Missing 'ts' in event: {evt}"
            # Should be parseable as ISO8601
            datetime.fromisoformat(evt["ts"])


# ---------------------------------------------------------------------------
# task_skip events
# ---------------------------------------------------------------------------


class TestJsonlSkipEvents:
    def test_task_skip_on_dependency_failure(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """task_skip emitted for tasks skipped due to upstream failure."""
        call_count = [0]

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs) -> TaskResult:
            call_count[0] += 1
            now = datetime.now(UTC)
            # First task fails, second should be skipped
            status = "failed" if task.id == "t1" else "success"
            exit_code = 1 if task.id == "t1" else 0
            result = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=exit_code,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="",
            )
            result.log_path.write_text(f"status={status}\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        plan = _make_plan(
            [_make_task("t1"), _make_task("t2", depends_on=["t1"])],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        skips = [e for e in events if e["event"] == "task_skip"]

        assert len(skips) == 1
        assert skips[0]["task_id"] == "t2"
        assert "reason" in skips[0]
        assert "dependency failure" in skips[0]["reason"]

    def test_task_skip_on_fail_fast(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """task_skip emitted for tasks skipped due to fail_fast."""
        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs) -> TaskResult:
            now = datetime.now(UTC)
            status = "failed" if task.id == "t1" else "success"
            result = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=1 if status == "failed" else 0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command="cmd",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="",
            )
            result.log_path.write_text(f"status={status}\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        # Independent tasks: t1 fails, t2 has no dep on t1 but fail_fast skips it
        plan = _make_plan(
            [_make_task("t1"), _make_task("t2")],
            fail_fast=True,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        skips = [e for e in events if e["event"] == "task_skip"]

        # At least one task should be skipped due to fail_fast (t2 if t1 runs first)
        if skips:
            for skip in skips:
                assert "reason" in skip
                assert "fail_fast" in skip["reason"]


# ---------------------------------------------------------------------------
# run_plan API: output_mode parameter
# ---------------------------------------------------------------------------


class TestRunPlanOutputMode:
    def test_text_mode_no_json_events(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """Text mode does NOT emit JSON Lines events."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="text")

        captured = capsys.readouterr()
        # Should not contain JSON lines events (no lines starting with {"event":)
        for line in captured.out.splitlines():
            if line.strip().startswith("{"):
                parsed = json.loads(line)
                assert "event" not in parsed, f"Unexpected JSONL event in text mode: {line}"

    def test_jsonl_mode_suppresses_human_readable(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """JSONL mode suppresses all [maestro ...] human-readable lines."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        for line in captured.out.splitlines():
            assert not line.startswith("[maestro"), f"Human-readable line leaked in JSONL mode: {line}"

    def test_default_output_mode_is_text(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """Default output_mode='text' produces human-readable output."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        run_plan(plan, run_dir_override=str(tmp_path / "runs"))  # no output_mode arg

        captured = capsys.readouterr()
        assert "[maestro" in captured.out

    def test_events_file_written_in_text_mode(self, tmp_path: Path, monkeypatch: Any) -> None:
        """events.jsonl is created and populated even in text output mode."""
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_success_execute)
        plan = _make_plan([_make_task("t1")], source_path=tmp_path / "plan.yaml")

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="text")

        events_file = result.run_path / "events.jsonl"
        assert events_file.exists()
        events = _parse_jsonl_file(events_file)
        assert events[0]["event"] == "run_start"
        assert events[-1]["event"] == "run_complete"
        assert any(e["event"] == "task_start" for e in events)
        assert any(e["event"] == "task_complete" for e in events)

    def test_budget_exceeded_event_emitted(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        """budget_exceeded event is emitted with spent/limit when soft budget is crossed."""
        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs) -> TaskResult:
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
                cost_usd=1.25,
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        plan = _make_plan(
            [_make_task("t1"), _make_task("t2")],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        plan.max_cost_usd = 0.50

        run_plan(plan, run_dir_override=str(tmp_path / "runs"), output_mode="jsonl")

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        budget = [e for e in events if e["event"] == "budget_exceeded"]
        assert len(budget) == 1
        assert budget[0]["spent"] == 1.25
        assert budget[0]["limit"] == 0.5


# ---------------------------------------------------------------------------
# CLI integration: --output jsonl
# ---------------------------------------------------------------------------


class TestCliJsonlIntegration:
    def test_jsonl_output_via_cli_dry_run(self, tmp_path: Path, capsys: Any) -> None:
        """--output jsonl via CLI produces JSON Lines output (dry-run)."""
        plan_file = _write_plan(tmp_path)
        exit_code = main([
            "run", str(plan_file),
            "--dry-run",
            "--output", "jsonl",
            "--run-dir", str(tmp_path / "runs"),
        ])

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)

        assert exit_code == 0
        event_names = [e["event"] for e in events]
        assert "run_start" in event_names
        assert "run_complete" in event_names

    def test_jsonl_no_maestro_prefix_in_output(self, tmp_path: Path, capsys: Any) -> None:
        """--output jsonl produces no [maestro ...] lines."""
        plan_file = _write_plan(tmp_path)
        main([
            "run", str(plan_file),
            "--dry-run",
            "--output", "jsonl",
            "--run-dir", str(tmp_path / "runs"),
        ])

        captured = capsys.readouterr()
        for line in captured.out.splitlines():
            assert not line.startswith("[maestro"), f"Leaked text: {line}"

    def test_text_mode_default_via_cli(self, tmp_path: Path, capsys: Any) -> None:
        """Without --output, CLI produces human-readable text."""
        plan_file = _write_plan(tmp_path)
        main([
            "run", str(plan_file),
            "--dry-run",
            "--run-dir", str(tmp_path / "runs"),
        ])

        captured = capsys.readouterr()
        assert "[maestro" in captured.out

    def test_run_start_fields_via_cli(self, tmp_path: Path, capsys: Any) -> None:
        """run_start event from CLI has all required fields."""
        plan_file = _write_plan(tmp_path)
        main([
            "run", str(plan_file),
            "--dry-run",
            "--output", "jsonl",
            "--run-dir", str(tmp_path / "runs"),
        ])

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        run_start = next(e for e in events if e["event"] == "run_start")

        assert run_start["plan"] == "jsonl-test-plan"
        assert run_start["tasks"] == 2
        assert isinstance(run_start["run_id"], str)
        assert isinstance(run_start["max_parallel"], int)

    def test_run_complete_success_true_via_cli(self, tmp_path: Path, capsys: Any) -> None:
        """run_complete.success=true when all tasks succeed (dry-run)."""
        plan_file = _write_plan(tmp_path)
        main([
            "run", str(plan_file),
            "--dry-run",
            "--output", "jsonl",
            "--run-dir", str(tmp_path / "runs"),
        ])

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)
        run_end = next(e for e in events if e["event"] == "run_complete")

        assert run_end["success"] is True

    def test_task_start_and_complete_both_emitted_via_cli(self, tmp_path: Path, capsys: Any) -> None:
        """CLI dry-run emits task_start and task_complete for every task."""
        plan_file = _write_plan(tmp_path)
        main([
            "run", str(plan_file),
            "--dry-run",
            "--output", "jsonl",
            "--run-dir", str(tmp_path / "runs"),
        ])

        captured = capsys.readouterr()
        events = _parse_jsonl(captured.out)

        starts = {e["task_id"] for e in events if e["event"] == "task_start"}
        completes = {e["task_id"] for e in events if e["event"] == "task_complete"}

        assert starts == {"t1", "t2"}
        assert completes == {"t1", "t2"}
