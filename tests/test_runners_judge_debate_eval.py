from __future__ import annotations

"""Coverage-focused tests for ``_run_debate_evaluation`` and
``_run_judge_evaluation`` in ``maestro_cli.runners``.

These tests drive the currently-uncovered branches in those two functions:
partial-result handling in the bull/bear debate loop, the no-scores guard,
the reflection-method delegation, the non-str/non-dict criterion fallback,
the empty-G-Eval-steps fallback, generic subprocess error handling, the
early return on a parser ``error`` verdict, weighted_mean weight extraction
edge cases, and the rubric+deterministic (no-LLM) reasoning branch.

All subprocess / engine calls are mocked — no real CLI is ever invoked.
"""

from pathlib import Path
from typing import Any

import pytest

import maestro_cli.runners as runners
from maestro_cli.models import CriterionScore, JudgeResult, JudgeSpec
from maestro_cli.runners import _run_debate_evaluation, _run_judge_evaluation


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------


class _Proc:
    """Minimal stand-in for a ``subprocess.CompletedProcess``."""

    def __init__(self, stdout: str = "", returncode: int = 0, stderr: str = "") -> None:
        self.stdout = stdout
        self.returncode = returncode
        self.stderr = stderr


def _make_debate_proc_sequence(
    behaviors: list[Any],
) -> Any:
    """Return a fake ``subprocess.run`` that consumes ``behaviors`` in order.

    Each behavior is either a ``_Proc`` (returned) or an Exception instance
    (raised) on successive calls.  Used to script the bull/bear call sequence.
    """
    calls = {"n": 0}

    def _fake_run(*args: Any, **kwargs: Any) -> Any:
        idx = calls["n"]
        calls["n"] += 1
        if idx >= len(behaviors):
            # Default: succeed with a neutral score so the loop keeps going.
            return _Proc(stdout='{"score": 0.5, "assessment": "neutral"}')
        behavior = behaviors[idx]
        if isinstance(behavior, BaseException):
            raise behavior
        return behavior

    _fake_run.calls = calls  # type: ignore[attr-defined]
    return _fake_run


# ---------------------------------------------------------------------------
# _run_debate_evaluation
# ---------------------------------------------------------------------------


class TestRunDebateEvaluation:
    """Drive the partial-result and error branches of the debate judge."""

    def test_bull_failure_round_two_breaks_with_partial_scores(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bull call failing on round 2 (after round 1 produced scores) hits
        the ``if all_scores: break`` path at line 5819 — the function uses the
        scores already gathered instead of aborting."""
        # Round 1: bull ok, bear ok (2 calls). Round 2: bull raises.
        behaviors = [
            _Proc(stdout='{"score": 0.8, "assessment": "good bull"}'),
            _Proc(stdout='{"score": 0.6, "assessment": "ok bear"}'),
            RuntimeError("bull round 2 boom"),
        ]
        monkeypatch.setattr(
            runners.subprocess, "run", _make_debate_proc_sequence(behaviors),
        )

        judge = JudgeSpec(
            criteria=["Overall quality is acceptable"],
            method="debate",
            debate_rounds=3,
            pass_threshold=0.5,
        )
        result = _run_debate_evaluation(
            task_id="t-debate",
            judge=judge,
            stdout_tail="some task output",
            workdir=tmp_path,
        )
        # Round 1 scores (0.8, 0.6) averaged => 0.7 >= 0.5 => pass.
        assert result.verdict == "pass"
        assert result.overall_score == pytest.approx(0.7, abs=1e-3)

    def test_bull_failure_first_round_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bull failing on the very first round (no prior scores) returns the
        error JudgeResult (lines 5820-5824)."""
        behaviors = [ValueError("bull immediate failure")]
        monkeypatch.setattr(
            runners.subprocess, "run", _make_debate_proc_sequence(behaviors),
        )

        judge = JudgeSpec(
            criteria=["x"], method="debate", debate_rounds=2, pass_threshold=0.5,
        )
        result = _run_debate_evaluation(
            task_id="t",
            judge=judge,
            stdout_tail="out",
            workdir=tmp_path,
        )
        assert result.verdict == "error"
        assert "bull call failed" in result.reasoning

    def test_bear_failure_appends_bull_then_breaks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bear failing on round 1 appends the bull score then breaks via the
        ``if all_scores: break`` path (lines 5847-5851).  With exactly one
        bull score, overall == bull_score."""
        behaviors = [
            _Proc(stdout='{"score": 0.9, "assessment": "bullish"}'),
            RuntimeError("bear boom"),
        ]
        monkeypatch.setattr(
            runners.subprocess, "run", _make_debate_proc_sequence(behaviors),
        )

        judge = JudgeSpec(
            criteria=["q"], method="debate", debate_rounds=2, pass_threshold=0.5,
        )
        result = _run_debate_evaluation(
            task_id="t",
            judge=judge,
            stdout_tail="out",
            workdir=tmp_path,
        )
        # Only the bull score (0.9) was recorded.
        assert result.verdict == "pass"
        assert result.overall_score == pytest.approx(0.9, abs=1e-3)

    def test_degenerate_empty_output_round_never_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Degenerate round where bull and bear both return empty stdout.

        Both calls yield ``{}`` => default score 0.5 each.  This exercises the
        normal append + averaging path with empty assessments and confirms the
        function never raises.

        NOTE on the two intentionally-uncovered defensive lines:
          * Line 5852 (the bear-branch ``return ...error``) is unreachable:
            it is preceded by ``all_scores.append(bull_score)`` (5849) and
            ``if all_scores: break`` (5850-5851), so ``all_scores`` is always
            truthy and the ``break`` always fires before 5852.
          * Line 5866 (the ``if not all_scores`` no-scores guard) is
            near-unreachable: ``rounds = max(1, ...)`` forces at least one
            loop iteration, and any non-failing iteration appends scores.
        We do not fake those lines; we cover all of the reachable flow.
        """
        behaviors = [
            _Proc(stdout=""),
            _Proc(stdout=""),
        ]
        monkeypatch.setattr(
            runners.subprocess, "run", _make_debate_proc_sequence(behaviors),
        )
        judge = JudgeSpec(
            criteria=["q"], method="debate", debate_rounds=1, pass_threshold=0.9,
        )
        result = _run_debate_evaluation(
            task_id="t",
            judge=judge,
            stdout_tail="out",
            workdir=tmp_path,
        )
        # Two 0.5 scores => 0.5 < 0.9 => fail (never raised).
        assert result.verdict == "fail"
        assert result.overall_score == pytest.approx(0.5, abs=1e-3)

    def test_debate_dict_criteria_filtered_uses_default_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All criteria being typed-assertion dicts get filtered out, so the
        default ``criteria_text`` fallback is used (lines 5783-5784)."""
        behaviors = [
            _Proc(stdout='{"score": 0.7, "assessment": "a"}'),
            _Proc(stdout='{"score": 0.7, "assessment": "b"}'),
        ]
        monkeypatch.setattr(
            runners.subprocess, "run", _make_debate_proc_sequence(behaviors),
        )
        judge = JudgeSpec(
            criteria=[{"type": "contains", "value": "x"}],
            method="debate",
            debate_rounds=1,
            pass_threshold=0.5,
        )
        result = _run_debate_evaluation(
            task_id="t",
            judge=judge,
            stdout_tail="out",
            workdir=tmp_path,
        )
        assert result.verdict == "pass"


# ---------------------------------------------------------------------------
# _run_judge_evaluation — method dispatch & branches
# ---------------------------------------------------------------------------


class TestRunJudgeEvaluationDispatch:
    """Cover the method-dispatch and criterion-classification branches."""

    def test_debate_method_delegates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``method == 'debate'`` delegates to _run_debate_evaluation
        (lines 5904-5907)."""
        sentinel = JudgeResult(verdict="pass", overall_score=0.95, reasoning="debate")

        def _fake_debate(*args: Any, **kwargs: Any) -> JudgeResult:
            return sentinel

        monkeypatch.setattr(runners, "_run_debate_evaluation", _fake_debate)
        judge = JudgeSpec(criteria=["q"], method="debate", pass_threshold=0.5)
        result = _run_judge_evaluation(
            task_id="t", judge=judge, stdout_tail="x", workdir=tmp_path,
        )
        assert result is sentinel

    def test_reflection_method_delegates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``method == 'reflection'`` delegates to _run_reflection_evaluation
        (line 5911)."""
        sentinel = JudgeResult(verdict="warn", overall_score=0.4, reasoning="reflect")

        def _fake_reflection(*args: Any, **kwargs: Any) -> JudgeResult:
            return sentinel

        monkeypatch.setattr(runners, "_run_reflection_evaluation", _fake_reflection)
        judge = JudgeSpec(criteria=["q"], method="reflection", pass_threshold=0.5)
        result = _run_judge_evaluation(
            task_id="t", judge=judge, stdout_tail="x", workdir=tmp_path,
        )
        assert result is sentinel

    def test_non_str_non_dict_criterion_str_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A criterion that is neither str nor dict (e.g. an int) is coerced
        via ``str(criterion)`` into the plain-LLM list (line 5940)."""
        monkeypatch.setattr(runners, "_resolve_executable", lambda x: [x])

        captured: dict[str, str] = {}

        def _fake_run(cmd: Any, *args: Any, **kwargs: Any) -> _Proc:
            # The last element of cmd is the prompt; capture for assertion.
            captured["prompt"] = cmd[-1] if isinstance(cmd, list) else ""
            return _Proc(
                stdout='{"criteria": [], "overall_score": 0.9, "reasoning": "ok"}',
                returncode=0,
            )

        monkeypatch.setattr(runners.subprocess, "run", _fake_run)
        monkeypatch.setattr(
            runners, "_build_safe_env", lambda a, b: {},
        )

        # The int 123 is neither str nor dict -> str(criterion) appended.
        judge = JudgeSpec(criteria=[123], pass_threshold=0.5)  # type: ignore[list-item]
        result = _run_judge_evaluation(
            task_id="t", judge=judge, stdout_tail="output", workdir=tmp_path,
        )
        assert result.verdict == "pass"
        assert "123" in captured["prompt"]


# ---------------------------------------------------------------------------
# _run_judge_evaluation — G-Eval empty steps fallback
# ---------------------------------------------------------------------------


class TestGEvalEmptyStepsFallback:
    def test_g_eval_empty_steps_falls_back_to_direct(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """g_eval method with empty generated eval steps prints the fallback
        notice (line 5979) and uses the direct judge prompt."""
        monkeypatch.setattr(runners, "_resolve_executable", lambda x: [x])
        monkeypatch.setattr(runners, "_build_safe_env", lambda a, b: {})
        # Force _generate_eval_steps to return no steps.
        monkeypatch.setattr(
            runners, "_generate_eval_steps", lambda **kwargs: [],
        )

        def _fake_run(*args: Any, **kwargs: Any) -> _Proc:
            return _Proc(
                stdout='{"criteria": [], "overall_score": 0.8, "reasoning": "fine"}',
                returncode=0,
            )

        monkeypatch.setattr(runners.subprocess, "run", _fake_run)

        judge = JudgeSpec(
            criteria=["Code quality is high"],
            method="g_eval",
            pass_threshold=0.5,
        )
        result = _run_judge_evaluation(
            task_id="task-x", judge=judge, stdout_tail="some code", workdir=tmp_path,
        )
        captured = capsys.readouterr()
        assert "G-Eval steps empty for task 'task-x'" in captured.out
        assert result.verdict == "pass"


# ---------------------------------------------------------------------------
# _run_judge_evaluation — subprocess error & parser-error branches
# ---------------------------------------------------------------------------


class TestJudgeSubprocessErrorBranches:
    def test_generic_exception_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-timeout exception from subprocess.run returns verdict 'error'
        (lines 6019-6020)."""
        monkeypatch.setattr(runners, "_resolve_executable", lambda x: [x])
        monkeypatch.setattr(runners, "_build_safe_env", lambda a, b: {})

        def _boom(*args: Any, **kwargs: Any) -> _Proc:
            raise OSError("exec format error")

        monkeypatch.setattr(runners.subprocess, "run", _boom)

        judge = JudgeSpec(criteria=["readable"], pass_threshold=0.5)
        result = _run_judge_evaluation(
            task_id="t", judge=judge, stdout_tail="x", workdir=tmp_path,
        )
        assert result.verdict == "error"
        assert "Judge error" in result.reasoning

    def test_parser_error_verdict_returns_early(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the LLM produces non-empty output containing no JSON object,
        _parse_judge_response yields verdict 'error', triggering the early
        return at lines 6026-6027."""
        monkeypatch.setattr(runners, "_resolve_executable", lambda x: [x])
        monkeypatch.setattr(runners, "_build_safe_env", lambda a, b: {})

        def _fake_run(*args: Any, **kwargs: Any) -> _Proc:
            # Non-empty, non-zero-free stdout but NO braces => parser error.
            return _Proc(stdout="this is prose with no json object", returncode=0)

        monkeypatch.setattr(runners.subprocess, "run", _fake_run)

        judge = JudgeSpec(criteria=["readable"], pass_threshold=0.5)
        result = _run_judge_evaluation(
            task_id="t", judge=judge, stdout_tail="x", workdir=tmp_path,
        )
        assert result.verdict == "error"
        assert "No JSON object" in result.reasoning


# ---------------------------------------------------------------------------
# _run_judge_evaluation — weighted_mean weight extraction
# ---------------------------------------------------------------------------


class TestWeightedMeanExtraction:
    def test_weighted_mean_skips_nonstr_name_and_bad_weight(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """weighted_mean aggregation: a rubric criterion whose ``name`` is not
        a string is skipped (line 6044); a rubric criterion whose ``weight``
        cannot be coerced to float falls back to 1.0 (lines 6047-6048)."""
        # Make rubric evaluation deterministic without an LLM call.
        def _fake_rubric(**kwargs: Any) -> list[CriterionScore]:
            return [
                CriterionScore(
                    criterion="good-name", passed=True, score=1.0, reasoning="r1",
                ),
                CriterionScore(
                    criterion="bad-weight", passed=True, score=0.5, reasoning="r2",
                ),
            ]

        monkeypatch.setattr(runners, "_evaluate_rubric_criteria", _fake_rubric)

        judge = JudgeSpec(
            criteria=[
                # name is an int -> skipped at line 6044
                {"type": "rubric", "name": 999, "weight": 2.0,
                 "levels": [{"score": 3, "description": "ok"}]},
                # good string name, bad (non-numeric) weight -> default 1.0
                {"type": "rubric", "name": "good-name", "weight": "not-a-number",
                 "levels": [{"score": 3, "description": "ok"}]},
                {"type": "rubric", "name": "bad-weight", "weight": None,
                 "levels": [{"score": 3, "description": "ok"}]},
            ],
            aggregation="weighted_mean",
            pass_threshold=0.5,
        )
        result = _run_judge_evaluation(
            task_id="t", judge=judge, stdout_tail="output", workdir=tmp_path,
        )
        # No LLM criteria => reasoning comes from the rubric-only branch.
        assert result.verdict in {"pass", "warn", "fail"}
        assert "Rubric assertions passed" in result.reasoning


# ---------------------------------------------------------------------------
# _run_judge_evaluation — rubric + deterministic (no LLM) reasoning branch
# ---------------------------------------------------------------------------


class TestRubricPlusDeterministicReasoning:
    def test_rubric_and_deterministic_no_llm_reasoning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Both rubric and deterministic scores present, but NO plain LLM
        criteria => the elif at lines 6080-6086 builds the combined reasoning
        string."""
        def _fake_rubric(**kwargs: Any) -> list[CriterionScore]:
            return [
                CriterionScore(
                    criterion="style", passed=True, score=1.0, reasoning="r",
                ),
            ]

        monkeypatch.setattr(runners, "_evaluate_rubric_criteria", _fake_rubric)
        # No subprocess should be invoked since there are no plain LLM criteria.

        judge = JudgeSpec(
            criteria=[
                {"type": "contains", "value": "hello"},   # deterministic (pass)
                {"type": "rubric", "name": "style", "min_score": 3,
                 "levels": [{"score": 3, "description": "ok"}]},
            ],
            aggregation="mean",
            pass_threshold=0.5,
        )
        result = _run_judge_evaluation(
            task_id="t",
            judge=judge,
            stdout_tail="hello there world",
            workdir=tmp_path,
        )
        assert "Deterministic assertions passed" in result.reasoning
        assert "Rubric assertions passed" in result.reasoning
        # Both deterministic and rubric passed -> high score -> pass.
        assert result.verdict == "pass"
