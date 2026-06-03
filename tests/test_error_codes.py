from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.errors import (
    E003,
    E004,
    E005,
    E006,
    E007,
    E100,
    PlanValidationError,
    TaskExecutionError,
)
from maestro_cli.loader import load_plan


def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


# ===========================================================================
# PlanValidationError with codes
# ===========================================================================


class TestPlanValidationError:
    def test_validation_error_with_code(self) -> None:
        exc = PlanValidationError("duplicate task id", code=E003)
        assert exc.code == E003
        assert "[E003]" in str(exc)
        assert "duplicate task id" in str(exc)

    def test_validation_error_without_code(self) -> None:
        exc = PlanValidationError("something went wrong")
        assert exc.code is None
        assert str(exc) == "something went wrong"
        assert "[" not in str(exc)

    def test_validation_error_code_attribute(self) -> None:
        exc = PlanValidationError("test message", code=E003)
        assert hasattr(exc, "code")
        assert exc.code == E003

    def test_validation_error_is_exception(self) -> None:
        exc = PlanValidationError("test", code=E003)
        assert isinstance(exc, Exception)


# ===========================================================================
# TaskExecutionError with codes
# ===========================================================================


class TestTaskExecutionError:
    def test_execution_error_with_code(self) -> None:
        exc = TaskExecutionError("prompt file not found", code=E100)
        assert exc.code == E100
        assert "[E100]" in str(exc)
        assert "prompt file not found" in str(exc)

    def test_execution_error_without_code(self) -> None:
        exc = TaskExecutionError("runtime error")
        assert exc.code is None
        assert str(exc) == "runtime error"

    def test_execution_error_code_attribute(self) -> None:
        exc = TaskExecutionError("test message", code=E100)
        assert hasattr(exc, "code")
        assert exc.code == E100

    def test_execution_error_is_exception(self) -> None:
        exc = TaskExecutionError("test", code=E100)
        assert isinstance(exc, Exception)


# ===========================================================================
# Loader raises correct error codes
# ===========================================================================


class TestLoaderErrorCodes:
    def test_duplicate_task_id_has_code(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: echo one
  - id: t1
    command: echo duplicate
""")
        with pytest.raises(PlanValidationError, match=r"\[E003\]"):
            load_plan(plan_file)

    def test_invalid_engine_has_code(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: bad-engine
    prompt: "Do something"
""")
        with pytest.raises(PlanValidationError, match=r"\[E006\]"):
            load_plan(plan_file)

    def test_missing_prompt_has_code(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
""")
        with pytest.raises(PlanValidationError, match=r"\[E007\]"):
            load_plan(plan_file)

    def test_circular_dependency_has_code(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    depends_on: [t2]
    command: echo one
  - id: t2
    depends_on: [t1]
    command: echo two
""")
        with pytest.raises(PlanValidationError, match=r"\[E004\]"):
            load_plan(plan_file)

    def test_unknown_dependency_has_code(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    depends_on: [nonexistent]
    command: echo one
""")
        with pytest.raises(PlanValidationError, match=r"\[E005\]"):
            load_plan(plan_file)

    def test_prompt_file_not_found_has_code(self, tmp_path: Path) -> None:
        """TaskExecutionError with E100 raised when prompt_file doesn't exist."""
        from maestro_cli.errors import TaskExecutionError
        from maestro_cli.models import EngineDefaults, PlanDefaults, PlanSpec, TaskSpec
        from maestro_cli.runners import _load_prompt

        plan = PlanSpec(
            version=1,
            name="test",
            defaults=PlanDefaults(codex=EngineDefaults(), claude=EngineDefaults()),
            tasks=[],
        )
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt_file="nonexistent_prompt.txt",
        )
        with pytest.raises(TaskExecutionError, match=r"\[E100\]"):
            _load_prompt(plan, task)

    def test_duplicate_task_id_code_string(self, tmp_path: Path) -> None:
        """Verify the error code value itself is the string 'E003'."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: dup
    command: echo a
  - id: dup
    command: echo b
""")
        try:
            load_plan(plan_file)
            pytest.fail("Expected PlanValidationError")
        except PlanValidationError as exc:
            assert exc.code == "E003"

    def test_unknown_dep_code_string(self, tmp_path: Path) -> None:
        """Verify the error code value itself is the string 'E005'."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    depends_on: [ghost]
    command: echo one
""")
        try:
            load_plan(plan_file)
            pytest.fail("Expected PlanValidationError")
        except PlanValidationError as exc:
            assert exc.code == "E005"
