from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import maestro_cli.runners as runners
from maestro_cli.models import CriterionScore, JudgeResult, JudgeSpec
from maestro_cli.runners import (
    _aggregate_scores,
    _build_judge_feedback,
    _parse_judge_response,
    _run_judge_evaluation,
    _run_judge_quorum,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_judge_spec(**kwargs: Any) -> JudgeSpec:
    return JudgeSpec(
        criteria=kwargs.get("criteria", ["tests pass", "code quality"]),
        pass_threshold=kwargs.get("pass_threshold", 0.7),
        on_fail=kwargs.get("on_fail", "fail"),
        model=kwargs.get("model", "haiku"),
        method=kwargs.get("method", "direct"),
        aggregation=kwargs.get("aggregation", "mean"),
        quorum=kwargs.get("quorum"),
        quorum_strategy=kwargs.get("quorum_strategy"),
    )


def _make_valid_json(
    overall_score: float = 0.9,
    criteria: list[dict[str, Any]] | None = None,
    reasoning: str = "Looks good",
) -> str:
    if criteria is None:
        criteria = [
            {"criterion": "tests pass", "passed": True, "score": overall_score, "reasoning": "ok"}
        ]
    return json.dumps(
        {"criteria": criteria, "overall_score": overall_score, "reasoning": reasoning}
    )


def _mock_subprocess_run(
    returncode: int = 0, stdout: str = "", stderr: str = ""
) -> Any:
    """Return a callable that mimics subprocess.run with fixed output."""

    def _mock(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=args[0] if args else [], returncode=returncode, stdout=stdout, stderr=stderr
        )

    return _mock


# ===========================================================================
# TestParseJudgeResponse
# ===========================================================================


class TestParseJudgeResponse:
    def test_valid_json_returns_pass_as_placeholder_verdict(self) -> None:
        # _parse_judge_response always sets verdict='pass'; threshold applied by caller
        text = _make_valid_json(overall_score=0.9)
        result = _parse_judge_response(text)
        assert result.verdict == "pass"

    def test_overall_score_extracted(self) -> None:
        text = _make_valid_json(overall_score=0.75)
        result = _parse_judge_response(text)
        assert result.overall_score == pytest.approx(0.75)

    def test_no_json_returns_error_verdict(self) -> None:
        result = _parse_judge_response("no json here at all")
        assert result.verdict == "error"
        assert "No JSON object" in result.reasoning

    def test_invalid_json_returns_error_verdict(self) -> None:
        result = _parse_judge_response("{invalid json!!!}")
        assert result.verdict == "error"
        assert "JSON parse error" in result.reasoning

    def test_empty_text_returns_error_verdict(self) -> None:
        result = _parse_judge_response("")
        assert result.verdict == "error"

    def test_whitespace_only_returns_error_verdict(self) -> None:
        result = _parse_judge_response("   \n\t  ")
        assert result.verdict == "error"

    def test_overall_score_defaults_to_zero_when_missing(self) -> None:
        text = json.dumps({"criteria": [], "reasoning": "ok"})
        result = _parse_judge_response(text)
        assert result.overall_score == pytest.approx(0.0)

    def test_criteria_parsed_correctly(self) -> None:
        criteria = [
            {"criterion": "compiles", "passed": True, "score": 1.0, "reasoning": "yes"},
            {"criterion": "tests pass", "passed": False, "score": 0.3, "reasoning": "partial"},
        ]
        text = _make_valid_json(criteria=criteria)
        result = _parse_judge_response(text)
        assert len(result.criterion_scores) == 2
        assert result.criterion_scores[0].criterion == "compiles"
        assert result.criterion_scores[0].passed is True
        assert result.criterion_scores[1].criterion == "tests pass"
        assert result.criterion_scores[1].passed is False

    def test_passed_flag_parsed_as_bool(self) -> None:
        criteria = [{"criterion": "c", "passed": False, "score": 0.0, "reasoning": "no"}]
        result = _parse_judge_response(_make_valid_json(criteria=criteria))
        assert isinstance(result.criterion_scores[0].passed, bool)
        assert result.criterion_scores[0].passed is False

    def test_score_parsed_as_float_from_string(self) -> None:
        criteria = [{"criterion": "c", "passed": True, "score": "0.75", "reasoning": "ok"}]
        result = _parse_judge_response(_make_valid_json(criteria=criteria))
        assert isinstance(result.criterion_scores[0].score, float)
        assert result.criterion_scores[0].score == pytest.approx(0.75)

    def test_reasoning_extracted(self) -> None:
        result = _parse_judge_response(_make_valid_json(reasoning="Excellent work done"))
        assert result.reasoning == "Excellent work done"

    def test_non_dict_criteria_entries_skipped(self) -> None:
        payload = {
            "criteria": [
                "not a dict — should be skipped",
                {"criterion": "c", "passed": True, "score": 1.0, "reasoning": "ok"},
            ],
            "overall_score": 0.9,
            "reasoning": "ok",
        }
        result = _parse_judge_response(json.dumps(payload))
        assert len(result.criterion_scores) == 1
        assert result.criterion_scores[0].criterion == "c"

    def test_json_embedded_in_prefix_text(self) -> None:
        # LLM might output preamble before the JSON block
        inner = _make_valid_json(overall_score=0.8)
        text = f"Here is my evaluation:\n{inner}\nThat's it."
        result = _parse_judge_response(text)
        assert result.overall_score == pytest.approx(0.8)
        assert result.verdict == "pass"

    def test_empty_criteria_list_in_json(self) -> None:
        text = json.dumps({"criteria": [], "overall_score": 0.5, "reasoning": "nothing to evaluate"})
        result = _parse_judge_response(text)
        assert result.criterion_scores == []
        assert result.overall_score == pytest.approx(0.5)


# ===========================================================================
# TestBuildJudgeFeedback
# ===========================================================================


class TestBuildJudgeFeedback:
    def _make_result(
        self,
        score: float = 0.4,
        reasoning: str = "Too many bugs",
        criterion_scores: list[CriterionScore] | None = None,
    ) -> JudgeResult:
        return JudgeResult(
            verdict="fail",
            overall_score=score,
            criterion_scores=criterion_scores or [],
            reasoning=reasoning,
        )

    def test_includes_judge_feedback_marker(self) -> None:
        result = self._make_result()
        feedback = _build_judge_feedback(result)
        assert "[JUDGE FEEDBACK]" in feedback

    def test_includes_score(self) -> None:
        result = self._make_result(score=0.35)
        feedback = _build_judge_feedback(result)
        assert "0.35" in feedback

    def test_includes_overall_reasoning(self) -> None:
        result = self._make_result(reasoning="Code does not compile")
        feedback = _build_judge_feedback(result)
        assert "Code does not compile" in feedback

    def test_includes_failed_criterion_name(self) -> None:
        scores = [
            CriterionScore(criterion="tests pass", passed=False, score=0.0, reasoning="all fail"),
            CriterionScore(criterion="code quality", passed=True, score=1.0, reasoning="good"),
        ]
        result = self._make_result(criterion_scores=scores)
        feedback = _build_judge_feedback(result)
        assert "tests pass" in feedback

    def test_excludes_passing_criteria(self) -> None:
        scores = [
            CriterionScore(criterion="tests pass", passed=False, score=0.0, reasoning="all fail"),
            CriterionScore(criterion="code quality", passed=True, score=1.0, reasoning="good"),
        ]
        result = self._make_result(criterion_scores=scores)
        feedback = _build_judge_feedback(result)
        # "code quality" passed — should not appear in failed criteria list
        assert "code quality" not in feedback

    def test_no_failed_criteria_shows_fallback_message(self) -> None:
        # All criteria passed but overall verdict is fail (edge case)
        scores = [CriterionScore(criterion="c", passed=True, score=1.0, reasoning="ok")]
        result = self._make_result(criterion_scores=scores)
        feedback = _build_judge_feedback(result)
        assert "no individual criteria" in feedback.lower()

    def test_includes_criterion_score_value(self) -> None:
        scores = [
            CriterionScore(criterion="compiles", passed=False, score=0.1, reasoning="build errors"),
        ]
        result = self._make_result(criterion_scores=scores)
        feedback = _build_judge_feedback(result)
        assert "0.10" in feedback

    def test_includes_criterion_reasoning(self) -> None:
        scores = [
            CriterionScore(criterion="compiles", passed=False, score=0.1, reasoning="build errors here"),
        ]
        result = self._make_result(criterion_scores=scores)
        feedback = _build_judge_feedback(result)
        assert "build errors here" in feedback

    def test_ends_with_please_address_message(self) -> None:
        result = self._make_result()
        feedback = _build_judge_feedback(result)
        assert "Please address" in feedback


# ===========================================================================
# TestJudgeSpecDefaults
# ===========================================================================


class TestJudgeSpecDefaults:
    def test_default_model_is_haiku(self) -> None:
        spec = JudgeSpec()
        assert spec.model == "haiku"

    def test_default_threshold_is_0_7(self) -> None:
        spec = JudgeSpec()
        assert spec.pass_threshold == pytest.approx(0.7)

    def test_default_on_fail_is_fail(self) -> None:
        spec = JudgeSpec()
        assert spec.on_fail == "fail"

    def test_default_criteria_is_empty_list(self) -> None:
        spec = JudgeSpec()
        assert spec.criteria == []

    def test_to_dict_serializes_all_fields(self) -> None:
        spec = JudgeSpec(
            criteria=["c1", "c2"],
            pass_threshold=0.8,
            on_fail="retry",
            model="sonnet",
        )
        d = spec.to_dict()
        assert d["criteria"] == ["c1", "c2"]
        assert d["pass_threshold"] == pytest.approx(0.8)
        assert d["on_fail"] == "retry"
        assert d["model"] == "sonnet"


# ===========================================================================
# TestJudgeResultToDict
# ===========================================================================


class TestJudgeResultToDict:
    def test_serializes_verdict(self) -> None:
        result = JudgeResult(verdict="pass", overall_score=0.9)
        d = result.to_dict()
        assert d["verdict"] == "pass"

    def test_serializes_overall_score(self) -> None:
        result = JudgeResult(verdict="fail", overall_score=0.3)
        d = result.to_dict()
        assert d["overall_score"] == pytest.approx(0.3)

    def test_serializes_reasoning(self) -> None:
        result = JudgeResult(verdict="warn", overall_score=0.6, reasoning="borderline quality")
        d = result.to_dict()
        assert d["reasoning"] == "borderline quality"

    def test_serializes_criterion_scores(self) -> None:
        scores = [CriterionScore(criterion="c", passed=True, score=1.0, reasoning="ok")]
        result = JudgeResult(verdict="pass", overall_score=1.0, criterion_scores=scores)
        d = result.to_dict()
        assert len(d["criterion_scores"]) == 1
        assert d["criterion_scores"][0]["criterion"] == "c"
        assert d["criterion_scores"][0]["passed"] is True

    def test_empty_criterion_scores_serialized_as_empty_list(self) -> None:
        result = JudgeResult(verdict="error", overall_score=0.0)
        d = result.to_dict()
        assert d["criterion_scores"] == []


# ===========================================================================
# TestCriterionScoreToDict
# ===========================================================================


class TestCriterionScoreToDict:
    def test_serializes_all_fields(self) -> None:
        score = CriterionScore(
            criterion="tests pass",
            passed=False,
            score=0.2,
            reasoning="most tests fail",
        )
        d = score.to_dict()
        assert d["criterion"] == "tests pass"
        assert d["passed"] is False
        assert d["score"] == pytest.approx(0.2)
        assert d["reasoning"] == "most tests fail"

    def test_passed_true_serialized(self) -> None:
        score = CriterionScore(criterion="c", passed=True, score=1.0, reasoning="")
        assert score.to_dict()["passed"] is True


# ===========================================================================
# TestAggregateScores
# ===========================================================================


class TestAggregateScores:
    def test_returns_zero_for_empty_scores(self) -> None:
        assert _aggregate_scores([], "mean") == pytest.approx(0.0)

    def test_mean_aggregation(self) -> None:
        scores = [
            CriterionScore("a", True, 1.0, ""),
            CriterionScore("b", True, 0.5, ""),
            CriterionScore("c", False, 0.0, ""),
        ]
        assert _aggregate_scores(scores, "mean") == pytest.approx(0.5)

    def test_min_aggregation(self) -> None:
        scores = [
            CriterionScore("a", True, 1.0, ""),
            CriterionScore("b", True, 0.4, ""),
            CriterionScore("c", False, 0.2, ""),
        ]
        assert _aggregate_scores(scores, "min") == pytest.approx(0.2)

    def test_weighted_mean_aggregation(self) -> None:
        scores = [
            CriterionScore("accuracy", True, 0.5, ""),
            CriterionScore("style", True, 1.0, ""),
        ]
        weights = {"accuracy": 3.0, "style": 1.0}
        assert _aggregate_scores(scores, "weighted_mean", weights) == pytest.approx(0.625)

    def test_unknown_aggregation_falls_back_to_mean(self) -> None:
        scores = [
            CriterionScore("a", True, 1.0, ""),
            CriterionScore("b", False, 0.0, ""),
        ]
        assert _aggregate_scores(scores, "not-a-real-mode") == pytest.approx(0.5)


# ===========================================================================
# TestRunJudgeEvaluation
# ===========================================================================


class TestRunJudgeEvaluation:
    def test_empty_criteria_returns_auto_pass(self, tmp_path: Path) -> None:
        spec = JudgeSpec(criteria=[])
        result = _run_judge_evaluation("t1", spec, "some output", tmp_path)
        assert result.verdict == "pass"
        assert result.overall_score == pytest.approx(1.0)
        assert "No criteria" in result.reasoning

    def test_pass_verdict_when_score_above_threshold(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec(pass_threshold=0.7)
        monkeypatch.setattr(
            subprocess,
            "run",
            _mock_subprocess_run(stdout=_make_valid_json(overall_score=0.9)),
        )
        result = _run_judge_evaluation("t1", spec, "good output", tmp_path)
        assert result.verdict == "pass"

    def test_fail_verdict_when_score_below_half_threshold(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec(pass_threshold=0.7)
        # 0.2 < 0.7 * 0.5 = 0.35 → fail
        monkeypatch.setattr(
            subprocess,
            "run",
            _mock_subprocess_run(stdout=_make_valid_json(overall_score=0.2)),
        )
        result = _run_judge_evaluation("t1", spec, "bad output", tmp_path)
        assert result.verdict == "fail"

    def test_warn_verdict_when_score_between_half_and_threshold(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec(pass_threshold=0.7)
        # 0.4 is between 0.35 (0.7*0.5) and 0.7 → warn
        monkeypatch.setattr(
            subprocess,
            "run",
            _mock_subprocess_run(stdout=_make_valid_json(overall_score=0.4)),
        )
        result = _run_judge_evaluation("t1", spec, "mediocre output", tmp_path)
        assert result.verdict == "warn"

    def test_subprocess_nonzero_exit_returns_error(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec()
        monkeypatch.setattr(
            subprocess,
            "run",
            _mock_subprocess_run(returncode=1, stdout="", stderr="internal error"),
        )
        result = _run_judge_evaluation("t1", spec, "some output", tmp_path)
        assert result.verdict == "error"

    def test_subprocess_empty_stdout_returns_error(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec()
        monkeypatch.setattr(
            subprocess,
            "run",
            _mock_subprocess_run(returncode=0, stdout="", stderr=""),
        )
        result = _run_judge_evaluation("t1", spec, "some output", tmp_path)
        assert result.verdict == "error"

    def test_subprocess_timeout_returns_error(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec()

        def raise_timeout(*args: Any, **kwargs: Any) -> None:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=60)

        monkeypatch.setattr(subprocess, "run", raise_timeout)
        result = _run_judge_evaluation("t1", spec, "some output", tmp_path)
        assert result.verdict == "error"
        assert "timed out" in result.reasoning

    def test_judge_uses_specified_model(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec(model="sonnet")
        captured_cmds: list[list[str]] = []

        def capture(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            cmd = list(args[0]) if args else []
            captured_cmds.append(cmd)
            return subprocess.CompletedProcess(
                args=cmd, returncode=0, stdout=_make_valid_json(0.9), stderr=""
            )

        monkeypatch.setattr(subprocess, "run", capture)
        _run_judge_evaluation("t1", spec, "some output", tmp_path)
        assert captured_cmds
        cmd_str = " ".join(captured_cmds[0])
        assert "sonnet" in cmd_str

    def test_threshold_exactly_at_boundary_passes(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec(pass_threshold=0.7)
        # Exactly 0.7 — should pass (>= threshold)
        monkeypatch.setattr(
            subprocess,
            "run",
            _mock_subprocess_run(stdout=_make_valid_json(overall_score=0.7)),
        )
        result = _run_judge_evaluation("t1", spec, "output", tmp_path)
        assert result.verdict == "pass"

    def test_score_returned_in_result(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec(pass_threshold=0.5)
        monkeypatch.setattr(
            subprocess,
            "run",
            _mock_subprocess_run(stdout=_make_valid_json(overall_score=0.6)),
        )
        result = _run_judge_evaluation("t1", spec, "output", tmp_path)
        assert result.overall_score == pytest.approx(0.6)

    def test_mean_keeps_backward_compatible_mixed_scoring(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec(
            criteria=[{"type": "contains", "value": "ok"}, "quality"],
            aggregation="mean",
        )
        llm_payload = _make_valid_json(
            overall_score=0.2,
            criteria=[
                {
                    "criterion": "quality",
                    "passed": True,
                    "score": 0.9,
                    "reasoning": "high quality",
                }
            ],
        )
        monkeypatch.setattr(
            subprocess,
            "run",
            _mock_subprocess_run(stdout=llm_payload),
        )
        result = _run_judge_evaluation("t1", spec, "ok output", tmp_path)
        # Old behavior: deterministic mean + (LLM overall * llm_count) / total_count
        assert result.overall_score == pytest.approx((1.0 + 0.2) / 2.0)

    def test_weighted_mean_uses_only_rubric_weights(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        rubric_criteria = [
            {
                "type": "rubric",
                "name": "accuracy",
                "weight": 3.0,
                "min_score": 3,
                "levels": [{"score": 1, "description": "poor"}],
            },
            {
                "type": "rubric",
                "name": "style",
                "weight": 1.0,
                "min_score": 3,
                "levels": [{"score": 1, "description": "poor"}],
            },
            {
                "type": "llm-rubric",
                "name": "ignored",
                "weight": 999.0,
                "value": "style check",
            },
        ]
        spec = _make_judge_spec(criteria=rubric_criteria, aggregation="weighted_mean")
        monkeypatch.setattr(
            runners,
            "_evaluate_rubric_criteria",
            lambda **kwargs: [
                CriterionScore("accuracy", True, 0.5, "ok"),
                CriterionScore("style", True, 1.0, "great"),
            ],
        )
        monkeypatch.setattr(
            subprocess,
            "run",
            _mock_subprocess_run(
                stdout=_make_valid_json(
                    overall_score=0.9,
                    criteria=[
                        {
                            "criterion": "style check",
                            "passed": True,
                            "score": 0.9,
                            "reasoning": "good",
                        }
                    ],
                )
            ),
        )
        result = _run_judge_evaluation("t1", spec, "output", tmp_path)
        assert result.overall_score == pytest.approx((0.5 * 3.0 + 1.0 * 1.0 + 0.9) / 5.0)


class TestRunJudgeQuorum:
    def test_quorum_disabled_falls_back_to_single_evaluation(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec()
        calls: list[str] = []

        def _fake_single(*args: Any, **kwargs: Any) -> JudgeResult:
            calls.append("called")
            return JudgeResult(verdict="pass", overall_score=0.9, reasoning="ok")

        monkeypatch.setattr(runners, "_run_judge_evaluation", _fake_single)

        result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"
        assert calls == ["called"]

    def test_majority_strategy_uses_vote_and_average_score(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec(quorum=3, quorum_strategy="majority")
        responses = iter(
            [
                JudgeResult(verdict="pass", overall_score=0.9, reasoning="judge one"),
                JudgeResult(verdict="fail", overall_score=0.2, reasoning="judge two"),
                JudgeResult(verdict="pass", overall_score=0.8, reasoning="judge three"),
            ]
        )

        monkeypatch.setattr(runners, "_run_judge_evaluation", lambda *a, **kw: next(responses))

        result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"
        assert result.overall_score == pytest.approx((0.9 + 0.2 + 0.8) / 3.0)
        assert "Quorum: 2/3 valid pass (majority)" in result.reasoning
        assert "Judge 2: fail" in result.reasoning

    def test_unanimous_strategy_fails_on_warn_vote(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec(quorum=3, quorum_strategy="unanimous")
        responses = iter(
            [
                JudgeResult(verdict="pass", overall_score=0.9, reasoning="judge one"),
                JudgeResult(verdict="warn", overall_score=0.6, reasoning="judge two"),
                JudgeResult(verdict="pass", overall_score=0.8, reasoning="judge three"),
            ]
        )

        monkeypatch.setattr(runners, "_run_judge_evaluation", lambda *a, **kw: next(responses))

        result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"
        assert result.overall_score == pytest.approx((0.9 + 0.6 + 0.8) / 3.0)
        assert "Quorum: 2/3 valid pass (unanimous)" in result.reasoning

    def test_any_strategy_passes_with_single_pass(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec(quorum=3, quorum_strategy="any")
        responses = iter(
            [
                JudgeResult(verdict="fail", overall_score=0.1, reasoning="judge one"),
                JudgeResult(verdict="pass", overall_score=0.9, reasoning="judge two"),
                JudgeResult(verdict="error", overall_score=0.0, reasoning="judge three"),
            ]
        )

        monkeypatch.setattr(runners, "_run_judge_evaluation", lambda *a, **kw: next(responses))

        result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"
        assert result.overall_score == pytest.approx((0.1 + 0.9) / 2.0)
        assert "Quorum: 1/2 valid pass (any, 1 error(s) excluded)" in result.reasoning

    def test_quorum_continues_after_single_judge_exception(
        self, tmp_path: Path, monkeypatch: Any
    ) -> None:
        spec = _make_judge_spec(quorum=3, quorum_strategy="majority")
        calls = {"count": 0}

        def _fake_single(*args: Any, **kwargs: Any) -> JudgeResult:
            calls["count"] += 1
            if calls["count"] == 1:
                raise RuntimeError("boom")
            if calls["count"] == 2:
                return JudgeResult(verdict="pass", overall_score=0.9, reasoning="judge two")
            return JudgeResult(verdict="fail", overall_score=0.2, reasoning="judge three")

        monkeypatch.setattr(runners, "_run_judge_evaluation", _fake_single)

        result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"
        assert result.overall_score == pytest.approx((0.9 + 0.2) / 2.0)
        assert "Judge 1: error" in result.reasoning
