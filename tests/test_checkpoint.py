from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import (
    EngineDefaults,
    JudgeResult,
    JudgeSpec,
    PlanDefaults,
    PlanSpec,
    TaskResult,
    TaskSpec,
)
from maestro_cli.runners import execute_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


def _make_plan(tmp_path: Path, **kwargs: Any) -> PlanSpec:
    defaults = kwargs.pop(
        "defaults",
        PlanDefaults(codex=EngineDefaults(), claude=EngineDefaults()),
    )
    return PlanSpec(
        version=1,
        name=kwargs.pop("name", "test"),
        max_parallel=kwargs.pop("max_parallel", 1),
        fail_fast=kwargs.pop("fail_fast", True),
        run_dir=str(tmp_path / "runs"),
        defaults=defaults,
        tasks=kwargs.pop("tasks", []),
        **kwargs,
    )


def _make_task_result(tmp_path: Path, **kwargs: Any) -> TaskResult:
    now = datetime.now(tz=timezone.utc)
    return TaskResult(
        task_id=kwargs.get("task_id", "t1"),
        status=kwargs.get("status", "success"),
        exit_code=kwargs.get("exit_code", 0),
        started_at=now,
        finished_at=now,
        duration_sec=1.0,
        command="echo",
        log_path=tmp_path / "t1.log",
        result_path=tmp_path / "t1.result.json",
        checkpoint_count=kwargs.get("checkpoint_count", 0),
        judge_result=kwargs.get("judge_result", None),
    )


# ===========================================================================
# TestCheckpointLoader
# ===========================================================================


class TestCheckpointLoader:
    def test_checkpoint_true_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    checkpoint: true
""",
        )
        plan = load_plan(plan_file)
        assert plan.tasks[0].checkpoint is True

    def test_checkpoint_false_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    checkpoint: false
""",
        )
        plan = load_plan(plan_file)
        assert plan.tasks[0].checkpoint is False

    def test_checkpoint_absent_defaults_false(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""",
        )
        plan = load_plan(plan_file)
        assert plan.tasks[0].checkpoint is False

    def test_checkpoint_preserved_on_matrix_expansion(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo {{ matrix.env }}"
    checkpoint: true
    matrix:
      env: [dev, prod]
""",
        )
        plan = load_plan(plan_file)
        assert len(plan.tasks) == 2
        for task in plan.tasks:
            assert task.checkpoint is True

    def test_judge_field_parsed_from_yaml(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["compiles", "tests pass"]
      pass_threshold: 0.8
      on_fail: retry
      model: sonnet
""",
        )
        plan = load_plan(plan_file)
        task = plan.tasks[0]
        assert task.judge is not None
        assert task.judge.criteria == ["compiles", "tests pass"]
        assert task.judge.pass_threshold == pytest.approx(0.8)
        assert task.judge.on_fail == "retry"
        assert task.judge.model == "sonnet"

    def test_judge_absent_defaults_to_none(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""",
        )
        plan = load_plan(plan_file)
        assert plan.tasks[0].judge is None

    def test_judge_empty_criteria_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: []
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E020\].*criteria"):
            load_plan(plan_file)

    def test_judge_criteria_not_a_list_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: "single string"
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E020\].*criteria"):
            load_plan(plan_file)

    def test_judge_pass_threshold_out_of_range_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["c"]
      pass_threshold: 1.5
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E020\].*pass_threshold"):
            load_plan(plan_file)

    def test_judge_invalid_on_fail_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["c"]
      on_fail: explode
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E020\].*on_fail"):
            load_plan(plan_file)

    def test_judge_not_a_dict_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge: "not an object"
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E020\]"):
            load_plan(plan_file)


# ===========================================================================
# TestCheckpointExecution
# ===========================================================================


class TestCheckpointExecution:
    def test_checkpoint_dir_created_when_enabled(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t1", command="echo hello", checkpoint=True)
        execute_task(plan, task, run_path)
        checkpoint_dir = run_path / "t1" / "checkpoints"
        assert checkpoint_dir.exists()
        assert checkpoint_dir.is_dir()

    def test_checkpoint_dir_not_created_when_disabled(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t1", command="echo hello", checkpoint=False)
        execute_task(plan, task, run_path)
        checkpoint_dir = run_path / "t1" / "checkpoints"
        assert not checkpoint_dir.exists()

    def test_checkpoint_count_zero_when_no_files_written(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t1", command="echo hello", checkpoint=True)
        result = execute_task(plan, task, run_path)
        assert result.checkpoint_count == 0

    def test_checkpoint_count_reflects_pre_existing_files(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        # Pre-create checkpoint files in the expected directory.
        # execute_task uses exist_ok=True, so pre-existing dir + files are preserved.
        checkpoint_dir = run_path / "t1" / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "ck1.json").write_text("{}", encoding="utf-8")
        (checkpoint_dir / "ck2.json").write_text("{}", encoding="utf-8")

        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t1", command="echo hello", checkpoint=True)
        result = execute_task(plan, task, run_path)
        assert result.checkpoint_count == 2

    def test_checkpoint_count_zero_when_checkpoint_disabled(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t1", command="echo hello", checkpoint=False)
        result = execute_task(plan, task, run_path)
        assert result.checkpoint_count == 0


# ===========================================================================
# TestCheckpointInResult
# ===========================================================================


class TestCheckpointInResult:
    def test_checkpoint_field_defaults_false_in_task_spec(self) -> None:
        task = TaskSpec(id="t1")
        assert task.checkpoint is False

    def test_checkpoint_count_default_zero_in_task_result(self, tmp_path: Path) -> None:
        result = _make_task_result(tmp_path)
        assert result.checkpoint_count == 0

    def test_checkpoint_count_in_to_dict(self, tmp_path: Path) -> None:
        result = _make_task_result(tmp_path, checkpoint_count=3)
        d = result.to_dict()
        assert d["checkpoint_count"] == 3

    def test_checkpoint_count_zero_in_to_dict(self, tmp_path: Path) -> None:
        result = _make_task_result(tmp_path, checkpoint_count=0)
        d = result.to_dict()
        assert d["checkpoint_count"] == 0

    def test_judge_result_none_serialized_as_none(self, tmp_path: Path) -> None:
        result = _make_task_result(tmp_path, judge_result=None)
        d = result.to_dict()
        assert d["judge_result"] is None

    def test_judge_result_serialized_when_present(self, tmp_path: Path) -> None:
        jr = JudgeResult(verdict="pass", overall_score=0.9, reasoning="good")
        result = _make_task_result(tmp_path, judge_result=jr)
        d = result.to_dict()
        assert d["judge_result"] is not None
        assert d["judge_result"]["verdict"] == "pass"
        assert d["judge_result"]["overall_score"] == pytest.approx(0.9)

    def test_judge_spec_to_dict_round_trip(self) -> None:
        spec = JudgeSpec(
            criteria=["tests pass", "no regressions"],
            pass_threshold=0.85,
            on_fail="warn",
            model="haiku",
        )
        d = spec.to_dict()
        # Verify all fields round-trip through dict
        assert d["criteria"] == ["tests pass", "no regressions"]
        assert d["pass_threshold"] == pytest.approx(0.85)
        assert d["on_fail"] == "warn"
        assert d["model"] == "haiku"
