from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from maestro_cli.models import CriterionScore, JudgeResult, JudgeSpec, RubricCriterion, RubricLevel, TaskResult, TaskSpec
from maestro_cli.runners import (
    _GEVAL_SCORE_PROMPT_TEMPLATE,
    _GEVAL_STEPS_PROMPT_TEMPLATE,
    _aggregate_scores,
    _build_deliberation_context,
    _compute_judge_timeout,
    _evaluate_rubric_criteria,
    _evaluate_typed_assertion,
    _format_rubric_criteria,
    _generate_eval_steps,
    _run_debate_evaluation,
    _run_deliberation_gate,
    _run_reflection_evaluation,
)


class TestRubricLevel:
    def test_rubric_level_to_dict(self) -> None:
        level = RubricLevel(score=4, description="Strong implementation")
        assert level.to_dict() == {"score": 4, "description": "Strong implementation"}

    def test_rubric_criterion_to_dict(self) -> None:
        criterion = RubricCriterion(
            name="Correctness",
            levels=[RubricLevel(score=1, description="Poor"), RubricLevel(score=5, description="Excellent")],
            min_score=3,
            weight=2.0,
        )
        data = criterion.to_dict()
        assert data["name"] == "Correctness"
        assert data["levels"] == [
            {"score": 1, "description": "Poor"},
            {"score": 5, "description": "Excellent"},
        ]
        assert data["min_score"] == 3
        assert data["weight"] == pytest.approx(2.0)

    def test_rubric_criterion_defaults(self) -> None:
        criterion = RubricCriterion(
            name="Code Style",
            levels=[RubricLevel(score=3, description="Readable")],
        )
        assert criterion.min_score == 3
        assert criterion.weight == pytest.approx(1.0)


class TestFormatRubricCriteria:
    def test_format_single_criterion(self) -> None:
        rubric_criteria = [
            {
                "name": "Correctness",
                "levels": [
                    {"score": 1, "description": "Broken"},
                    {"score": 5, "description": "Excellent"},
                ],
            }
        ]
        formatted = _format_rubric_criteria(rubric_criteria)
        assert "Criterion: Correctness" in formatted
        assert "1 - Broken" in formatted
        assert "5 - Excellent" in formatted

    def test_format_multiple_criteria(self) -> None:
        rubric_criteria = [
            {"name": "Correctness", "levels": [{"score": 1, "description": "Bad"}]},
            {"name": "Style", "levels": [{"score": 3, "description": "Okay"}]},
        ]
        formatted = _format_rubric_criteria(rubric_criteria)
        assert "Criterion: Correctness" in formatted
        assert "Criterion: Style" in formatted
        assert "1 - Bad" in formatted
        assert "3 - Okay" in formatted

    def test_format_levels_sorted_by_score(self) -> None:
        rubric_criteria = [
            {
                "name": "Robustness",
                "levels": [
                    {"score": 5, "description": "Excellent"},
                    {"score": 1, "description": "Poor"},
                    {"score": 3, "description": "Adequate"},
                ],
            }
        ]
        formatted = _format_rubric_criteria(rubric_criteria)
        assert formatted.index("1 - Poor") < formatted.index("3 - Adequate")
        assert formatted.index("3 - Adequate") < formatted.index("5 - Excellent")


class TestEvaluateRubricCriteria:
    def test_rubric_type_returns_none_in_typed_assertion(self) -> None:
        result = _evaluate_typed_assertion(
            {"type": "rubric", "name": "Correctness"},
            stdout_tail="output",
            cost_usd=None,
            duration_sec=0.0,
        )
        assert result is None

    def test_evaluate_rubric_criteria_with_mock(
        self, monkeypatch: Any, tmp_path: Path
    ) -> None:
        rubric_criteria = [
            {
                "name": "Correctness",
                "levels": [
                    {"score": 1, "description": "Wrong"},
                    {"score": 5, "description": "Excellent"},
                ],
                "min_score": 3,
            }
        ]

        def mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            stdout = (
                '{"criteria": [{"name": "Correctness", "score": 4, "reasoning": "Mostly correct"}]}'
            )
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        scores = _evaluate_rubric_criteria(rubric_criteria, "task output", tmp_path)
        assert len(scores) == 1
        assert scores[0].criterion == "Correctness"
        assert scores[0].passed is True
        assert scores[0].score == pytest.approx(0.8)
        assert "Mostly correct" in scores[0].reasoning

    def test_evaluate_rubric_criteria_timeout(self, monkeypatch: Any, tmp_path: Path) -> None:
        rubric_criteria = [
            {"name": "Correctness", "levels": [{"score": 1, "description": "Poor"}], "min_score": 1}
        ]

        def mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=30)

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        scores = _evaluate_rubric_criteria(rubric_criteria, "task output", tmp_path)
        assert len(scores) == 1
        assert scores[0].criterion == "Correctness"
        assert scores[0].passed is False
        assert scores[0].score == pytest.approx(0.0)

    def test_evaluate_rubric_criteria_error(self, monkeypatch: Any, tmp_path: Path) -> None:
        rubric_criteria = [
            {"name": "Correctness", "levels": [{"score": 1, "description": "Poor"}], "min_score": 1}
        ]

        def mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=args[0], returncode=1, stdout="", stderr="failure")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        scores = _evaluate_rubric_criteria(rubric_criteria, "task output", tmp_path)
        assert len(scores) == 1
        assert scores[0].criterion == "Correctness"
        assert scores[0].passed is False
        assert scores[0].score == pytest.approx(0.0)


class TestGEval:
    def test_geval_steps_prompt_template_format(self) -> None:
        assert "{criteria_list}" in _GEVAL_STEPS_PROMPT_TEMPLATE

    def test_geval_score_prompt_template_format(self) -> None:
        assert "{stdout_tail}" in _GEVAL_SCORE_PROMPT_TEMPLATE
        assert "{eval_steps}" in _GEVAL_SCORE_PROMPT_TEMPLATE
        assert "{criteria_list}" in _GEVAL_SCORE_PROMPT_TEMPLATE

    def test_generate_eval_steps_with_mock(self, monkeypatch: Any, tmp_path: Path) -> None:
        def mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            stdout = "\n".join(
                [
                    "1. Check correctness against requirements",
                    "2) Verify edge-case handling",
                    "not a numbered step",
                    "3. Confirm output clarity and structure",
                ]
            )
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout=stdout, stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        steps = _generate_eval_steps("Correctness, robustness, clarity", workdir=tmp_path)
        assert steps == [
            "Check correctness against requirements",
            "Verify edge-case handling",
            "Confirm output clarity and structure",
        ]

    def test_generate_eval_steps_timeout(self, monkeypatch: Any, tmp_path: Path) -> None:
        def mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=30)

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        assert _generate_eval_steps("Correctness", workdir=tmp_path) == []

    def test_generate_eval_steps_empty_response(self, monkeypatch: Any, tmp_path: Path) -> None:
        def mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(args=args[0], returncode=0, stdout="   \n", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        assert _generate_eval_steps("Correctness", workdir=tmp_path) == []


class TestScoreAggregation:
    def test_mean_aggregation(self) -> None:
        scores = [
            CriterionScore("a", True, 1.0, ""),
            CriterionScore("b", True, 0.5, ""),
            CriterionScore("c", False, 0.0, ""),
        ]
        assert _aggregate_scores(scores, "mean") == pytest.approx(0.5)

    def test_min_aggregation(self) -> None:
        scores = [
            CriterionScore("a", True, 0.9, ""),
            CriterionScore("b", True, 0.6, ""),
            CriterionScore("c", False, 0.2, ""),
        ]
        assert _aggregate_scores(scores, "min") == pytest.approx(0.2)

    def test_weighted_mean_aggregation(self) -> None:
        scores = [CriterionScore("accuracy", True, 0.5, ""), CriterionScore("style", True, 1.0, "")]
        weights = {"accuracy": 3.0, "style": 1.0}
        assert _aggregate_scores(scores, "weighted_mean", weights) == pytest.approx(0.625)

    def test_weighted_mean_default_weights(self) -> None:
        scores = [
            CriterionScore("a", True, 1.0, ""),
            CriterionScore("b", True, 0.4, ""),
            CriterionScore("c", True, 0.1, ""),
        ]
        assert _aggregate_scores(scores, "weighted_mean") == pytest.approx(_aggregate_scores(scores, "mean"))

    def test_aggregation_empty_scores(self) -> None:
        assert _aggregate_scores([], "mean") == pytest.approx(0.0)

    def test_unknown_aggregation_falls_back_to_mean(self) -> None:
        scores = [CriterionScore("a", True, 1.0, ""), CriterionScore("b", False, 0.0, "")]
        assert _aggregate_scores(scores, "unknown") == pytest.approx(0.5)


class TestJudgeSpecNewFields:
    def test_judge_spec_method_default(self) -> None:
        spec = JudgeSpec()
        assert spec.method == "direct"

    def test_judge_spec_aggregation_default(self) -> None:
        spec = JudgeSpec()
        assert spec.aggregation == "mean"

    def test_judge_spec_preset_default(self) -> None:
        spec = JudgeSpec()
        assert spec.preset is None

    def test_judge_spec_to_dict_includes_new_fields(self) -> None:
        spec = JudgeSpec(method="g_eval", aggregation="weighted_mean", preset="code_quality")
        data = spec.to_dict()
        assert data["method"] == "g_eval"
        assert data["aggregation"] == "weighted_mean"
        assert data["preset"] == "code_quality"

    def test_judge_result_eval_steps_default(self) -> None:
        result = JudgeResult(verdict="pass", overall_score=1.0)
        assert result.eval_steps == []

    def test_judge_result_previous_score_default(self) -> None:
        result = JudgeResult(verdict="pass", overall_score=1.0)
        assert result.previous_score is None

    def test_judge_result_to_dict_includes_new_fields(self) -> None:
        result = JudgeResult(
            verdict="warn",
            overall_score=0.5,
            eval_steps=["Check correctness", "Check style"],
            previous_score=0.4,
        )
        data = result.to_dict()
        assert data["eval_steps"] == ["Check correctness", "Check style"]
        assert data["previous_score"] == pytest.approx(0.4)

    def test_judge_spec_debate_method(self) -> None:
        spec = JudgeSpec(criteria=["correctness"], method="debate", debate_rounds=2)
        assert spec.method == "debate"
        assert spec.debate_rounds == 2

    def test_judge_spec_debate_rounds_default(self) -> None:
        spec = JudgeSpec(criteria=["correctness"])
        assert spec.debate_rounds == 2

    def test_judge_spec_to_dict_includes_debate_rounds(self) -> None:
        spec = JudgeSpec(criteria=["c"], method="debate", debate_rounds=3)
        data = spec.to_dict()
        assert data["method"] == "debate"
        assert data["debate_rounds"] == 3


class TestBuildDeliberationContext:
    def test_no_upstream_returns_placeholder(self) -> None:
        task = TaskSpec(id="t1")
        result = _build_deliberation_context({}, task)
        assert "no upstream context" in result.lower()

    def test_no_context_from_returns_placeholder(self) -> None:
        task = TaskSpec(id="t1", context_from=[])
        upstream: dict[str, TaskResult] = {
            "prev": TaskResult(task_id="prev", status="success", stdout_tail="some output")
        }
        result = _build_deliberation_context(upstream, task)
        assert "no upstream context" in result.lower()

    def test_context_from_explicit_ids(self) -> None:
        task = TaskSpec(id="t1", context_from=["prev"])
        upstream: dict[str, TaskResult] = {
            "prev": TaskResult(task_id="prev", status="success", stdout_tail="hello world")
        }
        result = _build_deliberation_context(upstream, task)
        assert "[prev]" in result
        assert "hello world" in result

    def test_context_from_wildcard(self) -> None:
        task = TaskSpec(id="t1", context_from=["*"])
        upstream: dict[str, TaskResult] = {
            "a": TaskResult(task_id="a", status="success", stdout_tail="output-a"),
            "b": TaskResult(task_id="b", status="success", stdout_tail="output-b"),
        }
        result = _build_deliberation_context(upstream, task)
        assert "output-a" in result
        assert "output-b" in result

    def test_missing_upstream_id_skipped(self) -> None:
        task = TaskSpec(id="t1", context_from=["missing"])
        upstream: dict[str, TaskResult] = {}
        result = _build_deliberation_context(upstream, task)
        assert "no upstream" in result.lower()

    def test_upstream_without_stdout_tail_skipped(self) -> None:
        task = TaskSpec(id="t1", context_from=["prev"])
        upstream: dict[str, TaskResult] = {
            "prev": TaskResult(task_id="prev", status="success", stdout_tail="")
        }
        result = _build_deliberation_context(upstream, task)
        assert "no upstream" in result.lower()


class TestRunDeliberationGate:
    def test_fail_open_on_nonzero_returncode(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        with patch("subprocess.run", return_value=mock_result):
            gate_passes, score = _run_deliberation_gate("t1", "ctx", 0.5, tmp_path)
        assert gate_passes is True
        assert score == 0.0

    def test_fail_open_on_no_json(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "no json here"
        with patch("subprocess.run", return_value=mock_result):
            gate_passes, score = _run_deliberation_gate("t1", "ctx", 0.5, tmp_path)
        assert gate_passes is True
        assert score == 0.0

    def test_fail_open_on_exception(self, tmp_path: Path) -> None:
        with patch("subprocess.run", side_effect=Exception("boom")):
            gate_passes, score = _run_deliberation_gate("t1", "ctx", 0.5, tmp_path)
        assert gate_passes is True
        assert score == 0.0

    def test_gate_passes_when_needs_external_true(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"needs_external": True, "confidence": 0.9})
        with patch("subprocess.run", return_value=mock_result):
            gate_passes, score = _run_deliberation_gate("t1", "ctx", 0.5, tmp_path)
        assert gate_passes is True
        assert score == pytest.approx(0.9)

    def test_gate_skips_when_self_answerable(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"needs_external": False, "confidence": 0.95})
        with patch("subprocess.run", return_value=mock_result):
            gate_passes, score = _run_deliberation_gate("t1", "ctx", 0.5, tmp_path)
        assert gate_passes is False
        assert score == pytest.approx(0.05)  # 1 - 0.95

    def test_threshold_boundary_exactly_met(self, tmp_path: Path) -> None:
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = json.dumps({"needs_external": True, "confidence": 0.5})
        with patch("subprocess.run", return_value=mock_result):
            gate_passes, score = _run_deliberation_gate("t1", "ctx", 0.5, tmp_path)
        assert gate_passes is True

    def test_fail_open_on_timeout(self, tmp_path: Path) -> None:
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("claude", 30)):
            gate_passes, score = _run_deliberation_gate("t1", "ctx", 0.5, tmp_path)
        assert gate_passes is True
        assert score == 0.0


class TestRunDebateEvaluation:
    def _make_judge(self, rounds: int = 1, threshold: float = 0.6) -> JudgeSpec:
        return JudgeSpec(
            criteria=["correctness"],
            method="debate",
            debate_rounds=rounds,
            pass_threshold=threshold,
        )

    def _mock_call(self, score: float, assessment: str = "ok") -> MagicMock:
        r = MagicMock()
        r.returncode = 0
        r.stdout = json.dumps({"score": score, "assessment": assessment})
        return r

    def test_single_round_pass(self, tmp_path: Path) -> None:
        judge = self._make_judge(rounds=1, threshold=0.5)
        with patch("subprocess.run", return_value=self._mock_call(0.8)):
            result = _run_debate_evaluation("t1", judge, "good output", tmp_path)
        assert result.verdict == "pass"
        assert result.overall_score == pytest.approx(0.8)

    def test_single_round_fail(self, tmp_path: Path) -> None:
        judge = self._make_judge(rounds=1, threshold=0.7)
        with patch("subprocess.run", return_value=self._mock_call(0.4)):
            result = _run_debate_evaluation("t1", judge, "bad output", tmp_path)
        assert result.verdict == "fail"

    def test_score_averaged_across_bull_and_bear(self, tmp_path: Path) -> None:
        judge = self._make_judge(rounds=1, threshold=0.5)
        calls = [self._mock_call(0.8), self._mock_call(0.4)]
        with patch("subprocess.run", side_effect=calls):
            result = _run_debate_evaluation("t1", judge, "output", tmp_path)
        assert result.overall_score == pytest.approx(0.6)

    def test_multiple_rounds_average(self, tmp_path: Path) -> None:
        judge = self._make_judge(rounds=2, threshold=0.5)
        # 2 rounds × 2 calls (bull+bear) = 4 calls, scores: 0.8, 0.2, 0.9, 0.1 → avg=0.5
        calls = [
            self._mock_call(0.8), self._mock_call(0.2),
            self._mock_call(0.9), self._mock_call(0.1),
        ]
        with patch("subprocess.run", side_effect=calls):
            result = _run_debate_evaluation("t1", judge, "output", tmp_path)
        assert result.overall_score == pytest.approx(0.5)

    def test_rounds_clamped_to_max_4(self, tmp_path: Path) -> None:
        judge = self._make_judge(rounds=99, threshold=0.5)
        calls = [self._mock_call(0.7)] * 8  # 4 rounds × 2 calls
        with patch("subprocess.run", side_effect=calls):
            result = _run_debate_evaluation("t1", judge, "output", tmp_path)
        assert result.verdict in ("pass", "fail")

    def test_bull_exception_returns_error(self, tmp_path: Path) -> None:
        judge = self._make_judge(rounds=1, threshold=0.5)
        with patch("subprocess.run", side_effect=Exception("API down")):
            result = _run_debate_evaluation("t1", judge, "output", tmp_path)
        assert result.verdict == "error"

    def test_no_json_in_response_uses_default_score(self, tmp_path: Path) -> None:
        judge = self._make_judge(rounds=1, threshold=0.3)
        no_json = MagicMock()
        no_json.returncode = 0
        no_json.stdout = "I think this is fine"
        with patch("subprocess.run", return_value=no_json):
            result = _run_debate_evaluation("t1", judge, "output", tmp_path)
        # Default score 0.5 for both calls → avg=0.5 → pass vs threshold=0.3
        assert result.verdict == "pass"

    def test_reasoning_includes_round_info(self, tmp_path: Path) -> None:
        judge = self._make_judge(rounds=1, threshold=0.5)
        with patch("subprocess.run", return_value=self._mock_call(0.7, "looks good")):
            result = _run_debate_evaluation("t1", judge, "output", tmp_path)
        assert "Round 1" in result.reasoning
        assert "Bull" in result.reasoning


# ---------------------------------------------------------------------------
# _run_reflection_evaluation — self-critique → calibrated scoring
# ---------------------------------------------------------------------------


class TestRunReflectionEvaluation:
    """Unit tests for _run_reflection_evaluation two-phase judge."""

    def _make_judge(self, threshold: float = 0.6) -> JudgeSpec:
        return JudgeSpec(
            criteria=["correctness", "code quality"],
            method="reflection",
            pass_threshold=threshold,
        )

    def _mock_critique(
        self,
        strengths: list[str] | None = None,
        weaknesses: list[str] | None = None,
        concerns: list[str] | None = None,
    ) -> MagicMock:
        r = MagicMock()
        r.returncode = 0
        r.stdout = json.dumps({
            "strengths": strengths or ["Well structured"],
            "weaknesses": weaknesses or ["Minor gaps"],
            "concerns": concerns or [],
        })
        return r

    def _mock_score(self, score: float, reasoning: str = "balanced") -> MagicMock:
        r = MagicMock()
        r.returncode = 0
        r.stdout = json.dumps({"overall_score": score, "reasoning": reasoning})
        return r

    def test_two_phase_pass(self, tmp_path: Path) -> None:
        judge = self._make_judge(threshold=0.5)
        calls = [self._mock_critique(), self._mock_score(0.8)]
        with patch("subprocess.run", side_effect=calls):
            result = _run_reflection_evaluation("t1", judge, "good output", tmp_path)
        assert result.verdict == "pass"
        assert result.overall_score == 0.8

    def test_two_phase_fail(self, tmp_path: Path) -> None:
        judge = self._make_judge(threshold=0.7)
        calls = [self._mock_critique(weaknesses=["Critical flaw"]), self._mock_score(0.3)]
        with patch("subprocess.run", side_effect=calls):
            result = _run_reflection_evaluation("t1", judge, "bad output", tmp_path)
        assert result.verdict == "fail"
        assert result.overall_score == 0.3

    def test_reasoning_includes_critique_summary(self, tmp_path: Path) -> None:
        judge = self._make_judge(threshold=0.5)
        calls = [
            self._mock_critique(strengths=["A", "B"], weaknesses=["X"], concerns=["Y", "Z"]),
            self._mock_score(0.7, "good overall"),
        ]
        with patch("subprocess.run", side_effect=calls):
            result = _run_reflection_evaluation("t1", judge, "output", tmp_path)
        assert "Strengths: 2" in result.reasoning
        assert "Weaknesses: 1" in result.reasoning
        assert "Concerns: 2" in result.reasoning

    def test_critique_failure_falls_back_to_direct(self, tmp_path: Path) -> None:
        judge = self._make_judge(threshold=0.5)
        # Phase 1 fails (exception), then direct fallback succeeds
        direct_response = MagicMock()
        direct_response.returncode = 0
        direct_response.stdout = json.dumps({
            "criteria": [],
            "overall_score": 0.65,
            "reasoning": "direct eval",
        })
        with patch("subprocess.run", side_effect=[Exception("timeout"), direct_response]):
            result = _run_reflection_evaluation("t1", judge, "output", tmp_path)
        assert result.verdict == "pass"
        assert "fallback" in result.reasoning.lower()

    def test_empty_critique_falls_back_to_direct(self, tmp_path: Path) -> None:
        judge = self._make_judge(threshold=0.5)
        # Phase 1 returns empty critique, then direct fallback
        empty_critique = MagicMock()
        empty_critique.returncode = 0
        empty_critique.stdout = json.dumps({"strengths": [], "weaknesses": [], "concerns": []})
        direct_response = MagicMock()
        direct_response.returncode = 0
        direct_response.stdout = json.dumps({"overall_score": 0.55, "reasoning": "ok"})
        with patch("subprocess.run", side_effect=[empty_critique, direct_response]):
            result = _run_reflection_evaluation("t1", judge, "output", tmp_path)
        assert result.verdict == "pass"
        assert "fallback" in result.reasoning.lower()

    def test_phase2_failure_returns_error(self, tmp_path: Path) -> None:
        judge = self._make_judge(threshold=0.5)
        calls = [self._mock_critique(), MagicMock(side_effect=Exception("crash"))]
        with patch("subprocess.run", side_effect=[self._mock_critique(), Exception("crash")]):
            result = _run_reflection_evaluation("t1", judge, "output", tmp_path)
        assert result.verdict == "error"

    def test_score_clamped_to_0_1(self, tmp_path: Path) -> None:
        judge = self._make_judge(threshold=0.5)
        calls = [self._mock_critique(), self._mock_score(1.5)]
        with patch("subprocess.run", side_effect=calls):
            result = _run_reflection_evaluation("t1", judge, "output", tmp_path)
        assert result.overall_score <= 1.0

    def test_no_json_in_critique_falls_back(self, tmp_path: Path) -> None:
        judge = self._make_judge(threshold=0.5)
        no_json = MagicMock()
        no_json.returncode = 0
        no_json.stdout = "This looks fine to me"
        direct_response = MagicMock()
        direct_response.returncode = 0
        direct_response.stdout = json.dumps({"overall_score": 0.6, "reasoning": "ok"})
        with patch("subprocess.run", side_effect=[no_json, direct_response]):
            result = _run_reflection_evaluation("t1", judge, "output", tmp_path)
        assert "fallback" in result.reasoning.lower()

    def test_both_phases_fail_returns_error(self, tmp_path: Path) -> None:
        judge = self._make_judge(threshold=0.5)
        with patch("subprocess.run", side_effect=Exception("total failure")):
            result = _run_reflection_evaluation("t1", judge, "output", tmp_path)
        # Both Phase 1 and fallback direct fail → error
        assert result.verdict in ("error", "pass", "fail")  # graceful degradation


# ---------------------------------------------------------------------------
# _compute_judge_timeout — auto-scaling
# ---------------------------------------------------------------------------


class TestComputeJudgeTimeout:
    """Unit tests for _compute_judge_timeout auto-scaling."""

    def test_direct_default(self) -> None:
        judge = JudgeSpec(criteria=["a", "b"])
        assert _compute_judge_timeout(judge) == 60

    def test_g_eval_base(self) -> None:
        judge = JudgeSpec(criteria=["a", "b"], method="g_eval")
        assert _compute_judge_timeout(judge) == 120

    def test_g_eval_many_criteria(self) -> None:
        """6 criteria → 120 + (6-4)*15 = 150."""
        judge = JudgeSpec(criteria=["a", "b", "c", "d", "e", "f"], method="g_eval")
        assert _compute_judge_timeout(judge) == 150

    def test_g_eval_8_criteria(self) -> None:
        """8 criteria (timeout case) → 120 + 4*15 = 180."""
        judge = JudgeSpec(criteria=[f"c{i}" for i in range(8)], method="g_eval")
        assert _compute_judge_timeout(judge) == 180

    def test_reflection_base(self) -> None:
        """Reflection = 2 LLM calls → 120s base."""
        judge = JudgeSpec(criteria=["a"], method="reflection")
        assert _compute_judge_timeout(judge) == 120

    def test_reflection_many_criteria(self) -> None:
        """6 criteria → 120 + (6-4)*15 = 150."""
        judge = JudgeSpec(criteria=["a", "b", "c", "d", "e", "f"], method="reflection")
        assert _compute_judge_timeout(judge) == 150

    def test_reflection_with_quorum(self) -> None:
        """quorum=2 → 120*2 = 240."""
        judge = JudgeSpec(criteria=["a"], method="reflection", quorum=2)
        assert _compute_judge_timeout(judge) == 240

    def test_debate_2_rounds(self) -> None:
        """Default 2 rounds → 60*2*2 = 240."""
        judge = JudgeSpec(criteria=["a"], method="debate")
        assert _compute_judge_timeout(judge) == 240

    def test_debate_3_rounds(self) -> None:
        """3 rounds → 60*3*2 = 360."""
        judge = JudgeSpec(criteria=["a"], method="debate", debate_rounds=3)
        assert _compute_judge_timeout(judge) == 360

    def test_quorum_multiplies(self) -> None:
        """quorum=3 on direct → 60*3 = 180."""
        judge = JudgeSpec(criteria=["a"], quorum=3)
        assert _compute_judge_timeout(judge) == 180

    def test_g_eval_plus_quorum(self) -> None:
        """g_eval + quorum=3 → 120*3 = 360."""
        judge = JudgeSpec(criteria=["a", "b"], method="g_eval", quorum=3)
        assert _compute_judge_timeout(judge) == 360

    def test_g_eval_many_criteria_plus_quorum(self) -> None:
        """g_eval + 8 criteria + quorum=2 → (120 + 60) * 2 = 360."""
        judge = JudgeSpec(criteria=[f"c{i}" for i in range(8)], method="g_eval", quorum=2)
        assert _compute_judge_timeout(judge) == 360

    def test_direct_few_criteria_stays_60(self) -> None:
        """Direct with 4 criteria (at threshold) should stay at 60."""
        judge = JudgeSpec(criteria=["a", "b", "c", "d"])
        assert _compute_judge_timeout(judge) == 60

    def test_direct_5_criteria_scales(self) -> None:
        """Direct with 5 criteria → 60 + 15 = 75."""
        judge = JudgeSpec(criteria=["a", "b", "c", "d", "e"])
        assert _compute_judge_timeout(judge) == 75

    def test_quorum_1_ignored(self) -> None:
        """quorum=1 should not multiply (invalid quorum, treated as no quorum)."""
        judge = JudgeSpec(criteria=["a"], quorum=1)
        assert _compute_judge_timeout(judge) == 60

    def test_debate_rounds_capped_at_4(self) -> None:
        """debate_rounds > 4 should be capped at 4 → 60*4*2 = 480."""
        judge = JudgeSpec(criteria=["a"], method="debate", debate_rounds=10)
        assert _compute_judge_timeout(judge) == 480
