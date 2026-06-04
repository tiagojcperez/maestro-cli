from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.eval import (
    DimensionResult,
    EvalSuiteResult,
    _build_judge_spec,
    _resolve_tasks,
    run_eval,
)
from maestro_cli.models import JudgeResult, JudgeSpec


def _write_eval(tmp_path: Path, text: str) -> Path:
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(text, encoding="utf-8")
    return eval_path


# ---------------------------------------------------------------------------
# _build_judge_spec: the "_to_judge_spec returned None" guard
# ---------------------------------------------------------------------------


class TestBuildJudgeSpecNoneGuard:
    def test_none_from_to_judge_spec_raises_must_be_provided(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The input passes the isinstance(dict) check, but the underlying
        # _to_judge_spec returns None -> the "must be provided" branch fires.
        monkeypatch.setattr(
            "maestro_cli.eval._to_judge_spec", lambda data, field_name: None
        )
        with pytest.raises(PlanValidationError, match="must be provided"):
            _build_judge_spec({"model": "sonnet", "criteria": ["ok"]}, "judge")

    def test_none_guard_uses_custom_field_name(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.eval._to_judge_spec", lambda data, field_name: None
        )
        with pytest.raises(PlanValidationError, match="overrides.task-z must be provided"):
            _build_judge_spec({"model": "sonnet"}, "overrides.task-z")


# ---------------------------------------------------------------------------
# _resolve_tasks: empty normalized include patterns -> default wildcard
# ---------------------------------------------------------------------------


class TestResolveTasksEmptyIncludeDefaults:
    def test_explicit_empty_tasks_list_defaults_to_wildcard(self) -> None:
        # spec.get returns the explicit empty list (not the default), so the
        # normalized include patterns are empty -> falls back to ["*"].
        selected = _resolve_tasks({"tasks": []}, ["task-a", "task-b"])
        assert selected == ["task-a", "task-b"]

    def test_whitespace_only_tasks_defaults_to_wildcard(self) -> None:
        # Whitespace-only patterns are stripped during normalization, leaving
        # an empty include list that must default back to the wildcard.
        selected = _resolve_tasks({"tasks": ["   ", ""]}, ["alpha", "beta"])
        assert selected == ["alpha", "beta"]

    def test_null_tasks_defaults_to_wildcard(self) -> None:
        selected = _resolve_tasks({"tasks": None}, ["only-one"])
        assert selected == ["only-one"]


# ---------------------------------------------------------------------------
# run_eval: multi-dimensional execution path
# ---------------------------------------------------------------------------


class TestRunEvalDimensions:
    def _spec(self, tmp_path: Path) -> Path:
        return _write_eval(
            tmp_path,
            """\
name: dim-suite
tasks: ["*"]
judge:
  model: sonnet
  criteria:
    - "must be correct"
  pass_threshold: 0.7
dimensions:
  - name: correctness
    tasks: ["impl-*"]
  - name: security
    tasks: ["sec-*"]
    exclude: ["sec-skip"]
""",
        )

    def test_dimensions_execute_and_aggregate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = self._spec(tmp_path)
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {
            "task_results": {
                "impl-a": {"stdout_tail": "impl-a out", "cost_usd": 1.0, "duration_sec": 2.0},
                "impl-b": {"stdout_tail": "impl-b out"},
                "sec-a": {"stdout_tail": "sec-a out"},
                "sec-skip": {"stdout_tail": "skipped one"},
            }
        }
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)

        seen: list[str] = []

        def _fake_quorum(
            task_id: str,
            judge: Any,
            stdout_tail: str,
            workdir: Path,
            cost_usd: float | None = None,
            duration_sec: float = 0.0,
            timeout_sec: int = 45,
        ) -> JudgeResult:
            seen.append(task_id)
            # impl-a + sec-a pass; impl-b fails.
            verdict = "fail" if task_id == "impl-b" else "pass"
            score = 0.4 if task_id == "impl-b" else 0.9
            return JudgeResult(verdict=verdict, overall_score=score, reasoning="r")

        monkeypatch.setattr("maestro_cli.runners._run_judge_quorum", _fake_quorum)

        suite = run_eval(eval_path, run_path)

        assert suite.dimensions is not None
        assert len(suite.dimensions) == 2

        correctness = suite.dimensions[0]
        assert correctness.name == "correctness"
        assert {r.task_id for r in correctness.results} == {"impl-a", "impl-b"}
        assert correctness.passed == 1
        assert correctness.failed == 1
        assert correctness.overall_pass is False

        security = suite.dimensions[1]
        assert security.name == "security"
        # sec-skip is filtered out by the dimension's exclude pattern.
        assert {r.task_id for r in security.results} == {"sec-a"}
        assert security.passed == 1
        assert security.overall_pass is True

        # The top-level results loop still ran for every task (no exclude there).
        assert "sec-skip" in seen

    def test_dimension_skips_non_dict_task_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: dim-skip
tasks: ["*"]
judge:
  model: sonnet
  criteria:
    - "ok"
dimensions:
  - name: only
    tasks: ["impl-*"]
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {
            "task_results": {
                "impl-good": {"stdout_tail": "good"},
                "impl-bad": "not-a-dict",
            }
        }
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_quorum",
            lambda *a, **kw: JudgeResult(verdict="pass", overall_score=1.0, reasoning=""),
        )

        suite = run_eval(eval_path, run_path)

        assert suite.dimensions is not None
        dim = suite.dimensions[0]
        assert [r.task_id for r in dim.results] == ["impl-good"]
        assert "impl-bad" in dim.skipped

    def test_dimension_stdout_tail_non_string_coerced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: dim-coerce
tasks: ["*"]
judge:
  model: sonnet
  criteria:
    - "ok"
dimensions:
  - name: only
    tasks: ["t-*"]
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {
            "task_results": {
                "t-a": {"stdout_tail": 98765, "cost_usd": "bad", "duration_sec": 4.0},
            }
        }
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)

        captured: list[dict[str, Any]] = []

        def _capture(
            task_id: str,
            judge: Any,
            stdout_tail: str,
            workdir: Path,
            cost_usd: float | None = None,
            duration_sec: float = 0.0,
            timeout_sec: int = 45,
        ) -> JudgeResult:
            captured.append(
                {
                    "stdout_tail": stdout_tail,
                    "cost_usd": cost_usd,
                    "duration_sec": duration_sec,
                }
            )
            return JudgeResult(verdict="pass", overall_score=1.0, reasoning="")

        monkeypatch.setattr("maestro_cli.runners._run_judge_quorum", _capture)

        run_eval(eval_path, run_path)

        assert captured[0]["stdout_tail"] == "98765"
        # cost_usd "bad" coerces to None.
        assert captured[0]["cost_usd"] is None
        assert captured[0]["duration_sec"] == pytest.approx(4.0)

    def test_dimension_judge_exception_captured_as_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: dim-error
tasks: ["*"]
judge:
  model: sonnet
  criteria:
    - "ok"
dimensions:
  - name: faulty-dim
    tasks: ["t-*"]
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {"task_results": {"t-a": {"stdout_tail": "out"}}}
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)

        def _boom(*args: Any, **kwargs: Any) -> JudgeResult:
            raise RuntimeError("dimension judge exploded")

        monkeypatch.setattr("maestro_cli.runners._run_judge_quorum", _boom)

        suite = run_eval(eval_path, run_path)

        assert suite.dimensions is not None
        dim = suite.dimensions[0]
        assert len(dim.results) == 1
        result = dim.results[0]
        assert result.task_id == "t-a"
        assert result.judge_result.verdict == "error"
        assert result.passed is False
        assert "dimension judge exploded" in result.judge_result.reasoning
        assert "faulty-dim" in result.judge_result.reasoning
        assert dim.errors == 1

    def test_dimension_missing_requested_id_recorded_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # A plain (non-glob) dimension task ID that is absent from the manifest
        # must surface in the dimension's skipped list.
        eval_path = _write_eval(
            tmp_path,
            """\
name: dim-missing
tasks: ["*"]
judge:
  model: sonnet
  criteria:
    - "ok"
dimensions:
  - name: only
    tasks: ["present", "ghost-task"]
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {"task_results": {"present": {"stdout_tail": "here"}}}
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_quorum",
            lambda *a, **kw: JudgeResult(verdict="pass", overall_score=1.0, reasoning=""),
        )

        suite = run_eval(eval_path, run_path)

        assert suite.dimensions is not None
        dim = suite.dimensions[0]
        assert [r.task_id for r in dim.results] == ["present"]
        assert "ghost-task" in dim.skipped

    def test_no_dimensions_leaves_suite_dimensions_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Sanity guard: when no dimensions are declared, the dimension loop is
        # skipped entirely and suite.dimensions stays None.
        eval_path = _write_eval(
            tmp_path,
            """\
name: no-dims
tasks: ["*"]
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {"task_results": {"t-a": {"stdout_tail": "out"}}}
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_quorum",
            lambda *a, **kw: JudgeResult(verdict="pass", overall_score=1.0, reasoning=""),
        )

        suite = run_eval(eval_path, run_path)
        assert suite.dimensions is None


# Sanity: building a valid judge spec still works (control for the None guard).
class TestBuildJudgeSpecValid:
    def test_valid_dict_returns_judge_spec(self) -> None:
        judge = _build_judge_spec({"model": "sonnet", "criteria": ["ok"]}, "judge")
        assert isinstance(judge, JudgeSpec)
        assert judge.model == "sonnet"


# Sanity: EvalSuiteResult.overall_pass with an empty dimension list (control).
class TestSuiteOverallPassEmptyDimensions:
    def test_empty_dimension_list_overall_pass_uses_results(self) -> None:
        suite = EvalSuiteResult(
            name="s",
            run_path=Path("."),
            results=[],
            skipped=[],
            dimensions=[DimensionResult(name="empty", results=[], skipped=[])],
        )
        assert suite.overall_pass is True
