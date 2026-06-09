from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path
from typing import Callable, get_args, get_origin

import pytest

import maestro_cli


# ---------------------------------------------------------------------------
# TestSdkImport
# ---------------------------------------------------------------------------


class TestSdkImport:
    def test_import_works(self) -> None:
        assert maestro_cli is not None

    def test_version_is_current(self) -> None:
        assert maestro_cli.__version__ == "2.5.4"

    def test_version_is_string(self) -> None:
        assert isinstance(maestro_cli.__version__, str)


# ---------------------------------------------------------------------------
# TestSdkAllExports
# ---------------------------------------------------------------------------


class TestSdkAllExports:
    def test_all_names_importable(self) -> None:
        for name in maestro_cli.__all__:
            obj = getattr(maestro_cli, name)
            assert obj is not None, f"{name} resolved to None"

    def test_no_attribute_error(self) -> None:
        for name in maestro_cli.__all__:
            # Should not raise AttributeError
            getattr(maestro_cli, name)

    def test_all_is_nonempty(self) -> None:
        assert len(maestro_cli.__all__) > 0


# ---------------------------------------------------------------------------
# TestSdkFunctionIdentity
# ---------------------------------------------------------------------------


class TestSdkFunctionIdentity:
    def test_load_plan_identity(self) -> None:
        from maestro_cli.loader import load_plan

        assert maestro_cli.load_plan is load_plan

    def test_run_plan_identity(self) -> None:
        from maestro_cli.scheduler import run_plan

        assert maestro_cli.run_plan is run_plan

    def test_validate_plan_identity(self) -> None:
        from maestro_cli.loader import validate_plan

        assert maestro_cli.validate_plan is validate_plan

    def test_scaffold_plan_identity(self) -> None:
        from maestro_cli.scaffold import scaffold_plan

        assert maestro_cli.scaffold_plan is scaffold_plan

    def test_blame_run_identity(self) -> None:
        from maestro_cli.blame import blame_run

        assert maestro_cli.blame_run is blame_run

    def test_diff_runs_identity(self) -> None:
        from maestro_cli.diff import diff_runs

        assert maestro_cli.diff_runs is diff_runs

    def test_audit_plan_identity(self) -> None:
        from maestro_cli.audit import audit_plan

        assert maestro_cli.audit_plan is audit_plan


# ---------------------------------------------------------------------------
# TestSdkDataTypes
# ---------------------------------------------------------------------------


class TestSdkDataTypes:
    def test_plan_spec_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.PlanSpec)

    def test_task_result_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.TaskResult)

    def test_task_spec_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.TaskSpec)

    def test_plan_run_result_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.PlanRunResult)

    def test_blame_chain_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.BlameChain)

    def test_blame_node_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.BlameNode)

    def test_run_diff_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.RunDiff)

    def test_task_diff_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.TaskDiff)

    def test_audit_finding_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.AuditFinding)

    def test_policy_spec_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.PolicySpec)

    def test_judge_spec_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.JudgeSpec)

    def test_plan_brief_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.PlanBrief)

    def test_task_brief_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.TaskBrief)

    def test_engine_defaults_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.EngineDefaults)

    def test_plan_defaults_is_dataclass(self) -> None:
        assert dataclasses.is_dataclass(maestro_cli.PlanDefaults)

    def test_event_callback_is_callable_alias(self) -> None:
        origin = get_origin(maestro_cli.EventCallback)
        assert origin is Callable or callable is not None
        # EventCallback = Callable[[str, dict[str, object]], None]
        args = get_args(maestro_cli.EventCallback)
        assert len(args) == 2  # [[str, dict], None]
        assert args[1] is None

    def test_engine_name_is_literal(self) -> None:
        # Literal types have get_args but no get_origin in older typing,
        # in 3.11+ get_origin returns typing.Literal
        args = get_args(maestro_cli.EngineName)
        assert "claude" in args
        assert "codex" in args
        assert "gemini" in args
        assert "copilot" in args
        assert "qwen" in args
        assert "ollama" in args

    def test_execution_profile_is_literal(self) -> None:
        args = get_args(maestro_cli.ExecutionProfile)
        assert "plan" in args
        assert "safe" in args
        assert "yolo" in args

    def test_task_status_is_literal(self) -> None:
        args = get_args(maestro_cli.TaskStatus)
        assert "success" in args
        assert "failed" in args
        assert "soft_failed" in args
        assert "skipped" in args

    def test_plan_validation_error_is_exception(self) -> None:
        assert issubclass(maestro_cli.PlanValidationError, Exception)

    def test_task_execution_error_is_exception(self) -> None:
        assert issubclass(maestro_cli.TaskExecutionError, Exception)


# ---------------------------------------------------------------------------
# TestSdkPyTyped
# ---------------------------------------------------------------------------


class TestSdkPyTyped:
    def test_py_typed_marker_exists(self) -> None:
        package_dir = Path(maestro_cli.__file__).parent
        py_typed = package_dir / "py.typed"
        assert py_typed.exists(), f"py.typed not found at {py_typed}"

    def test_py_typed_via_importlib(self) -> None:
        # importlib.resources approach (3.9+)
        files = importlib.resources.files("maestro_cli")
        py_typed = files / "py.typed"
        # Joinpath should not raise; check it resolves
        assert py_typed is not None


# ---------------------------------------------------------------------------
# TestSdkBasicWorkflow
# ---------------------------------------------------------------------------


_MINIMAL_PLAN = """\
version: 1
name: sdk-test
tasks:
  - id: hello
    command: "echo hello"
"""

_INVALID_PLAN_BAD_VERSION = """\
version: 999
name: broken
tasks:
  - id: t1
    command: "echo broken"
"""

_INVALID_PLAN_CYCLE = """\
version: 1
name: cycle
tasks:
  - id: a
    command: "echo a"
    depends_on: [b]
  - id: b
    command: "echo b"
    depends_on: [a]
"""


class TestSdkBasicWorkflow:
    def test_load_plan_returns_plan_spec(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "plan.yaml"
        yaml_file.write_text(_MINIMAL_PLAN, encoding="utf-8")
        plan = maestro_cli.load_plan(str(yaml_file))
        assert isinstance(plan, maestro_cli.PlanSpec)

    def test_load_plan_task_count(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "plan.yaml"
        yaml_file.write_text(_MINIMAL_PLAN, encoding="utf-8")
        plan = maestro_cli.load_plan(str(yaml_file))
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "hello"

    def test_validate_plan_no_raise(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "plan.yaml"
        yaml_file.write_text(_MINIMAL_PLAN, encoding="utf-8")
        plan = maestro_cli.load_plan(str(yaml_file))
        # Should not raise
        maestro_cli.validate_plan(plan)

    def test_invalid_plan_bad_version_raises(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(_INVALID_PLAN_BAD_VERSION, encoding="utf-8")
        with pytest.raises(maestro_cli.PlanValidationError):
            maestro_cli.load_plan(str(yaml_file))

    def test_invalid_plan_cycle_raises(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "cycle.yaml"
        yaml_file.write_text(_INVALID_PLAN_CYCLE, encoding="utf-8")
        with pytest.raises(maestro_cli.PlanValidationError):
            maestro_cli.load_plan(str(yaml_file))

    def test_plan_name_matches(self, tmp_path: Path) -> None:
        yaml_file = tmp_path / "plan.yaml"
        yaml_file.write_text(_MINIMAL_PLAN, encoding="utf-8")
        plan = maestro_cli.load_plan(str(yaml_file))
        assert plan.name == "sdk-test"
