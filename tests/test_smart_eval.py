from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan, validate_plan
from maestro_cli.models import ASSERTION_TYPES, PlanSpec, TaskSpec
from maestro_cli.runners import _evaluate_typed_assertion, _run_guard_command


class TestGuardCommand:
    def test_run_guard_command_pass(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        passed, output = _run_guard_command("python -c \"print('ok')\"", "tail", tmp_path, {}, timeout_sec=5)
        assert passed is True
        assert output == "ok\n"

    def test_run_guard_command_fail(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=args[0], returncode=3, stdout="bad\n", stderr="err\n")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        passed, output = _run_guard_command(["fake", "guard"], "tail", tmp_path, {}, timeout_sec=5)
        assert passed is False
        assert "guard_command exited with code 3" in output
        assert "bad\nerr\n" in output

    def test_run_guard_command_receives_stdin(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        captured: dict[str, Any] = {}

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured["input"] = kwargs.get("input")
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        stdout_tail = "hello from task output"
        _run_guard_command(["fake", "guard"], stdout_tail, tmp_path, {}, timeout_sec=5)
        assert captured["input"] == stdout_tail

    def test_run_guard_command_timeout(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
    ) -> None:
        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd=args[0], timeout=7)

        monkeypatch.setattr(subprocess, "run", _mock_run)
        passed, output = _run_guard_command(["fake", "guard"], "tail", tmp_path, {}, timeout_sec=7)
        assert passed is False
        assert output == "guard_command timed out after 7s"


class TestBudgetWarning:
    def test_budget_warning_pct_field(self) -> None:
        plan = PlanSpec(version=1, name="p")
        assert hasattr(plan, "budget_warning_pct")
        assert plan.budget_warning_pct is None

    def test_budget_warning_pct_validation(self) -> None:
        with pytest.raises(PlanValidationError, match="budget_warning_pct must be between 0 and 1"):
            validate_plan(PlanSpec(version=1, name="p", budget_warning_pct=0.0))

        with pytest.raises(PlanValidationError, match="budget_warning_pct must be between 0 and 1"):
            validate_plan(PlanSpec(version=1, name="p", budget_warning_pct=1.0))

        validate_plan(PlanSpec(version=1, name="p", budget_warning_pct=0.5))

    def test_budget_warning_loader_parsing(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            """\
version: 1
name: budget-plan
budget_warning_pct: 0.85
tasks:
  - id: t1
    command: "echo ok"
""",
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        assert plan.budget_warning_pct == pytest.approx(0.85)


class TestMaxIterations:
    def test_max_iterations_field(self) -> None:
        task = TaskSpec(id="t1")
        assert hasattr(task, "max_iterations")
        assert task.max_iterations is None

    def test_max_iterations_validation(self) -> None:
        with pytest.raises(PlanValidationError, match="max_iterations must be >= 1"):
            validate_plan(
                PlanSpec(
                    version=1,
                    name="p",
                    tasks=[TaskSpec(id="t1", command="echo ok", max_iterations=0)],
                )
            )

        validate_plan(
            PlanSpec(
                version=1,
                name="p",
                tasks=[TaskSpec(id="t1", command="echo ok", max_iterations=1)],
            )
        )

    def test_max_iterations_loader_parsing(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            """\
version: 1
name: max-iter-plan
tasks:
  - id: t1
    command: "echo ok"
    max_iterations: 4
""",
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        assert plan.tasks[0].max_iterations == 4


class TestTypedAssertions:
    def test_contains_pass(self) -> None:
        score = _evaluate_typed_assertion({"type": "contains", "value": "hello"}, "hello world", None, 0.0)
        assert score is not None
        assert score.passed is True

    def test_contains_fail(self) -> None:
        score = _evaluate_typed_assertion({"type": "contains", "value": "missing"}, "hello world", None, 0.0)
        assert score is not None
        assert score.passed is False

    def test_regex_pass(self) -> None:
        score = _evaluate_typed_assertion({"type": "regex", "value": r"h.llo"}, "hello world", None, 0.0)
        assert score is not None
        assert score.passed is True

    def test_regex_fail(self) -> None:
        score = _evaluate_typed_assertion({"type": "regex", "value": r"^bye$"}, "hello world", None, 0.0)
        assert score is not None
        assert score.passed is False

    def test_is_json_pass(self) -> None:
        score = _evaluate_typed_assertion({"type": "is-json", "value": "json"}, 'prefix {"a": 1} suffix', None, 0.0)
        assert score is not None
        assert score.passed is True

    def test_is_json_fail(self) -> None:
        score = _evaluate_typed_assertion({"type": "is-json", "value": "json"}, "not json", None, 0.0)
        assert score is not None
        assert score.passed is False

    def test_cost_under_pass(self) -> None:
        score = _evaluate_typed_assertion({"type": "cost_under", "value": 2.0}, "out", 1.5, 0.0)
        assert score is not None
        assert score.passed is True

    def test_cost_under_fail(self) -> None:
        score = _evaluate_typed_assertion({"type": "cost_under", "value": 1.0}, "out", 1.5, 0.0)
        assert score is not None
        assert score.passed is False

    def test_cost_under_no_cost(self) -> None:
        score = _evaluate_typed_assertion({"type": "cost_under", "value": 1.0}, "out", None, 0.0)
        assert score is not None
        assert score.passed is False
        assert "Cost data unavailable" in score.reasoning

    def test_duration_under_pass(self) -> None:
        score = _evaluate_typed_assertion({"type": "duration_under", "value": 5.0}, "out", None, 2.1)
        assert score is not None
        assert score.passed is True

    def test_duration_under_fail(self) -> None:
        score = _evaluate_typed_assertion({"type": "duration_under", "value": 2.0}, "out", None, 2.1)
        assert score is not None
        assert score.passed is False

    def test_llm_rubric_returns_none(self) -> None:
        score = _evaluate_typed_assertion({"type": "llm-rubric", "value": "quality"}, "out", None, 0.0)
        assert score is None

    def test_unknown_type_returns_error(self) -> None:
        score = _evaluate_typed_assertion({"type": "not-real", "value": "x"}, "out", None, 0.0)
        assert score is not None
        assert score.passed is False
        assert "Unsupported assertion type" in score.reasoning

    def test_judge_spec_mixed_criteria(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            """\
version: 1
name: judge-mixed
tasks:
  - id: t1
    engine: claude
    prompt: "Evaluate"
    judge:
      criteria:
        - "output should be concise"
        - type: contains
          value: "PASS"
        - type: duration_under
          value: 10
""",
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        judge = plan.tasks[0].judge
        assert judge is not None
        assert len(judge.criteria) == 3
        assert isinstance(judge.criteria[0], str)
        assert isinstance(judge.criteria[1], dict)
        assert isinstance(judge.criteria[2], dict)

    def test_assertion_types_constant(self) -> None:
        assert len(ASSERTION_TYPES) == 8
