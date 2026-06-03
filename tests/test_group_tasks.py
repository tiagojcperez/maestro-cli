from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from maestro_cli.errors import E001, E011, PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import (
    PlanDefaults,
    PlanRunResult,
    PlanSpec,
    TaskResult,
    TaskSpec,
    TokenUsage,
)
from maestro_cli.runners import _execute_group_task, execute_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SUB_PLAN_YAML = """\
version: 1
name: sub-plan
tasks:
  - id: sub-task
    command: echo sub
"""


def _write_plan(tmp_path: Path, content: str, filename: str = "plan.yaml") -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


def _make_plan(
    tasks: list[TaskSpec],
    source_path: Path | None = None,
    workspace_root: str | None = None,
) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="test-plan",
        max_parallel=2,
        defaults=PlanDefaults(),
        tasks=tasks,
        source_path=source_path,
        workspace_root=workspace_root,
    )


def _make_group_task(
    task_id: str = "grp",
    group: str = "sub.yaml",
    depends_on: list[str] | None = None,
    allow_failure: bool = False,
) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        group=group,
        depends_on=depends_on or [],
        allow_failure=allow_failure,
    )


def _now() -> datetime:
    return datetime.now(UTC)


def _fake_plan_run_result(
    success: bool = True,
    total_cost_usd: float | None = None,
    total_tokens: int | None = None,
    run_path: Path | None = None,
) -> PlanRunResult:
    now = _now()
    return PlanRunResult(
        plan_name="sub-plan",
        run_id="abc",
        run_path=run_path or Path("/tmp/sub"),
        started_at=now,
        finished_at=now,
        success=success,
        total_cost_usd=total_cost_usd,
        total_tokens=total_tokens,
    )


# ---------------------------------------------------------------------------
# Loader / validation tests
# ---------------------------------------------------------------------------


class TestGroupTaskParsing:
    def test_group_field_parsed(self, tmp_path: Path) -> None:
        sub = _write_plan(tmp_path, _SUB_PLAN_YAML, "sub.yaml")
        parent_yaml = f"""\
version: 1
name: parent
tasks:
  - id: grp
    group: sub.yaml
"""
        plan = load_plan(_write_plan(tmp_path, parent_yaml))
        task = plan.tasks[0]
        assert task.group == "sub.yaml"
        assert task.engine is None
        assert task.command is None

    def test_group_with_command_raises(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: p
tasks:
  - id: grp
    group: sub.yaml
    command: echo hi
"""
        with pytest.raises(PlanValidationError, match=E011):
            load_plan(_write_plan(tmp_path, yaml))

    def test_group_with_engine_raises(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: p
tasks:
  - id: grp
    group: sub.yaml
    engine: claude
    prompt: hi
"""
        with pytest.raises(PlanValidationError, match=E011):
            load_plan(_write_plan(tmp_path, yaml))

    def test_group_with_prompt_raises(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: p
tasks:
  - id: grp
    group: sub.yaml
    prompt: "do stuff"
"""
        with pytest.raises(PlanValidationError, match=E011):
            load_plan(_write_plan(tmp_path, yaml))

    def test_task_without_any_type_raises(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: p
tasks:
  - id: grp
    description: "missing type"
"""
        with pytest.raises(PlanValidationError, match=E001):
            load_plan(_write_plan(tmp_path, yaml))

    def test_group_with_depends_on(self, tmp_path: Path) -> None:
        sub = _write_plan(tmp_path, _SUB_PLAN_YAML, "sub.yaml")
        parent_yaml = """\
version: 1
name: parent
tasks:
  - id: setup
    command: echo setup
  - id: grp
    group: sub.yaml
    depends_on: [setup]
"""
        plan = load_plan(_write_plan(tmp_path, parent_yaml))
        grp = next(t for t in plan.tasks if t.id == "grp")
        assert grp.depends_on == ["setup"]

    def test_group_backslash_warning(self, tmp_path: Path) -> None:
        sub = _write_plan(tmp_path, _SUB_PLAN_YAML, "sub.yaml")
        parent_yaml = """\
version: 1
name: parent
tasks:
  - id: grp
    group: plans\\sub.yaml
"""
        # Should parse but emit a warning (backslash in group path)
        plan = load_plan(_write_plan(tmp_path, parent_yaml))
        assert any("backslash" in w.lower() or "backslashes" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# _execute_group_task unit tests (mock run_plan)
# ---------------------------------------------------------------------------


class TestExecuteGroupTask:
    def _run_path(self, tmp_path: Path) -> Path:
        rp = tmp_path / "run"
        rp.mkdir()
        return rp

    def _sub_plan_file(self, tmp_path: Path, name: str = "sub.yaml") -> Path:
        return _write_plan(tmp_path, _SUB_PLAN_YAML, name)

    def test_missing_sub_plan_returns_failed(self, tmp_path: Path) -> None:
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml")
        task = _make_group_task(group="nonexistent.yaml")

        result = _execute_group_task(plan, task, run_path, dry_run=False, execution_profile="plan")

        assert result.status == "failed"
        assert result.exit_code == 1
        assert "not found" in result.message.lower() or "nonexistent" in result.message

    def test_success_sub_plan_returns_success(self, tmp_path: Path) -> None:
        self._sub_plan_file(tmp_path)
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml")
        task = _make_group_task()

        mock_result = _fake_plan_run_result(success=True, total_cost_usd=1.5)

        with patch("maestro_cli.scheduler.run_plan", return_value=mock_result):
            result = _execute_group_task(plan, task, run_path, dry_run=False, execution_profile="plan")

        assert result.status == "success"
        assert result.exit_code == 0
        assert result.cost_usd == 1.5

    def test_failing_sub_plan_returns_failed(self, tmp_path: Path) -> None:
        self._sub_plan_file(tmp_path)
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml")
        task = _make_group_task()

        mock_result = _fake_plan_run_result(success=False)

        with patch("maestro_cli.scheduler.run_plan", return_value=mock_result):
            result = _execute_group_task(plan, task, run_path, dry_run=False, execution_profile="plan")

        assert result.status == "failed"
        assert result.exit_code == 1

    def test_failing_sub_plan_with_allow_failure_returns_soft_failed(self, tmp_path: Path) -> None:
        self._sub_plan_file(tmp_path)
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml")
        task = _make_group_task(allow_failure=True)

        mock_result = _fake_plan_run_result(success=False)

        with patch("maestro_cli.scheduler.run_plan", return_value=mock_result):
            result = _execute_group_task(plan, task, run_path, dry_run=False, execution_profile="plan")

        assert result.status == "soft_failed"

    def test_dry_run_returns_dry_run_status(self, tmp_path: Path) -> None:
        self._sub_plan_file(tmp_path)
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml")
        task = _make_group_task()

        mock_result = _fake_plan_run_result(success=True)

        with patch("maestro_cli.scheduler.run_plan", return_value=mock_result):
            result = _execute_group_task(plan, task, run_path, dry_run=True, execution_profile="plan")

        assert result.status == "dry_run"
        assert result.exit_code == 0

    def test_token_usage_aggregated(self, tmp_path: Path) -> None:
        self._sub_plan_file(tmp_path)
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml")
        task = _make_group_task()

        mock_result = _fake_plan_run_result(success=True, total_tokens=5000)

        with patch("maestro_cli.scheduler.run_plan", return_value=mock_result):
            result = _execute_group_task(plan, task, run_path, dry_run=False, execution_profile="plan")

        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 5000

    def test_workspace_root_inherited_when_sub_plan_has_none(self, tmp_path: Path) -> None:
        self._sub_plan_file(tmp_path)
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml", workspace_root="/parent-root")
        task = _make_group_task()

        captured_sub_plan: list[PlanSpec] = []

        def _capture_run_plan(sub_plan: PlanSpec, **kwargs: object) -> PlanRunResult:
            captured_sub_plan.append(sub_plan)
            return _fake_plan_run_result(success=True)

        with patch("maestro_cli.scheduler.run_plan", side_effect=_capture_run_plan):
            _execute_group_task(plan, task, run_path, dry_run=False, execution_profile="plan")

        assert captured_sub_plan[0].workspace_root == "/parent-root"

    def test_workspace_root_not_overridden_when_sub_plan_has_own(self, tmp_path: Path) -> None:
        sub_yaml = """\
version: 1
name: sub-plan
workspace_root: /sub-root
tasks:
  - id: sub-task
    command: echo sub
"""
        _write_plan(tmp_path, sub_yaml, "sub.yaml")
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml", workspace_root="/parent-root")
        task = _make_group_task()

        captured: list[PlanSpec] = []

        def _capture(sub_plan: PlanSpec, **kwargs: object) -> PlanRunResult:
            captured.append(sub_plan)
            return _fake_plan_run_result(success=True)

        with patch("maestro_cli.scheduler.run_plan", side_effect=_capture):
            _execute_group_task(plan, task, run_path, dry_run=False, execution_profile="plan")

        assert captured[0].workspace_root == "/sub-root"

    def test_sub_run_directory_created(self, tmp_path: Path) -> None:
        self._sub_plan_file(tmp_path)
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml")
        task = _make_group_task(task_id="my-group")

        mock_result = _fake_plan_run_result(success=True)

        with patch("maestro_cli.scheduler.run_plan", return_value=mock_result):
            _execute_group_task(plan, task, run_path, dry_run=False, execution_profile="plan")

        assert (run_path / "my-group").is_dir()

    def test_result_json_written(self, tmp_path: Path) -> None:
        self._sub_plan_file(tmp_path)
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml")
        task = _make_group_task()

        mock_result = _fake_plan_run_result(success=True)

        with patch("maestro_cli.scheduler.run_plan", return_value=mock_result):
            result = _execute_group_task(plan, task, run_path, dry_run=False, execution_profile="plan")

        assert result.result_path.exists()

    def test_command_string_identifies_group(self, tmp_path: Path) -> None:
        self._sub_plan_file(tmp_path)
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml")
        task = _make_group_task(group="sub.yaml")

        mock_result = _fake_plan_run_result(success=True)

        with patch("maestro_cli.scheduler.run_plan", return_value=mock_result):
            result = _execute_group_task(plan, task, run_path, dry_run=False, execution_profile="plan")

        assert "group" in result.command
        assert "sub.yaml" in result.command

    def test_execute_task_dispatches_group(self, tmp_path: Path) -> None:
        """execute_task() should delegate group tasks to _execute_group_task."""
        self._sub_plan_file(tmp_path)
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml")
        task = _make_group_task()

        mock_result = _fake_plan_run_result(success=True)

        with patch("maestro_cli.runners._execute_group_task", return_value=mock_result) as m:
            result = execute_task(plan, task, run_path)

        m.assert_called_once_with(plan, task, run_path, False, "plan")
        assert result is mock_result

    def test_bad_yaml_sub_plan_returns_failed(self, tmp_path: Path) -> None:
        bad_yaml = tmp_path / "sub.yaml"
        bad_yaml.write_text("this: is: invalid: yaml: [broken", encoding="utf-8")
        run_path = self._run_path(tmp_path)
        plan = _make_plan([], source_path=tmp_path / "parent.yaml")
        task = _make_group_task()

        result = _execute_group_task(plan, task, run_path, dry_run=False, execution_profile="plan")

        assert result.status == "failed"
        assert result.exit_code == 1
