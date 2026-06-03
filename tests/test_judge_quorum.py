from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import maestro_cli.runners as runners
from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import CriterionScore, JudgeResult, JudgeSpec
from maestro_cli.runners import _run_judge_quorum


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _jr(
    verdict: str = "pass",
    score: float = 0.9,
    reasoning: str = "ok",
    criterion_scores: list[CriterionScore] | None = None,
) -> JudgeResult:
    return JudgeResult(
        verdict=verdict,
        overall_score=score,
        reasoning=reasoning,
        criterion_scores=criterion_scores or [],
    )


def _make_spec(
    quorum: int | None = None,
    quorum_strategy: str | None = None,
    pass_threshold: float = 0.7,
    criteria: list[Any] | None = None,
) -> JudgeSpec:
    return JudgeSpec(
        criteria=criteria or ["quality"],
        pass_threshold=pass_threshold,
        quorum=quorum,
        quorum_strategy=quorum_strategy,
    )


def _plan_yaml(*, quorum: Any = None, quorum_strategy: Any = None) -> str:
    """Build a minimal plan YAML with a judge block."""
    lines = [
        "version: 1",
        "name: quorum-test",
        "tasks:",
        "  - id: task-a",
        "    engine: claude",
        '    prompt: "Do something"',
        "    judge:",
        "      criteria: [quality]",
    ]
    if quorum is not None:
        lines.append(f"      quorum: {quorum}")
    if quorum_strategy is not None:
        lines.append(f"      quorum_strategy: {quorum_strategy}")
    return "\n".join(lines) + "\n"


def _write_plan(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(content, encoding="utf-8")
    return p


# ===========================================================================
# TestJudgeSpecQuorumFields
# ===========================================================================


class TestJudgeSpecQuorumFields:
    def test_default_quorum_is_none(self) -> None:
        spec = JudgeSpec()
        assert spec.quorum is None

    def test_default_quorum_strategy_is_none(self) -> None:
        spec = JudgeSpec()
        assert spec.quorum_strategy is None

    def test_to_dict_includes_quorum_field(self) -> None:
        spec = JudgeSpec(quorum=3)
        d = spec.to_dict()
        assert "quorum" in d
        assert d["quorum"] == 3

    def test_to_dict_includes_quorum_strategy_field(self) -> None:
        spec = JudgeSpec(quorum=3, quorum_strategy="unanimous")
        d = spec.to_dict()
        assert "quorum_strategy" in d
        assert d["quorum_strategy"] == "unanimous"

    def test_to_dict_quorum_none_serializes_as_none(self) -> None:
        spec = JudgeSpec(quorum=None)
        d = spec.to_dict()
        assert d["quorum"] is None

    def test_to_dict_quorum_strategy_none_serializes_as_none(self) -> None:
        spec = JudgeSpec(quorum_strategy=None)
        d = spec.to_dict()
        assert d["quorum_strategy"] is None

    def test_quorum_five_rounds_stored(self) -> None:
        spec = JudgeSpec(quorum=5)
        assert spec.quorum == 5

    def test_all_three_strategies_stored(self) -> None:
        for strategy in ("majority", "unanimous", "any"):
            spec = JudgeSpec(quorum=2, quorum_strategy=strategy)
            assert spec.quorum_strategy == strategy


# ===========================================================================
# TestJudgeQuorumLoaderValidation
# ===========================================================================


class TestJudgeQuorumLoaderValidation:
    def test_quorum_less_than_2_raises_e054(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, _plan_yaml(quorum=1))
        with pytest.raises(PlanValidationError, match="E054"):
            load_plan(plan_file)

    def test_quorum_zero_raises_e054(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, _plan_yaml(quorum=0))
        with pytest.raises(PlanValidationError, match="E054"):
            load_plan(plan_file)

    def test_quorum_non_integer_string_raises_e054(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, _plan_yaml(quorum="abc"))
        with pytest.raises(PlanValidationError, match="E054"):
            load_plan(plan_file)

    def test_invalid_quorum_strategy_raises_e055(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, _plan_yaml(quorum=3, quorum_strategy="best-of"))
        with pytest.raises(PlanValidationError, match="E055"):
            load_plan(plan_file)

    def test_quorum_strategy_without_quorum_raises_e056(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, _plan_yaml(quorum_strategy="majority"))
        with pytest.raises(PlanValidationError, match="E056"):
            load_plan(plan_file)

    def test_quorum_2_is_valid_minimum(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, _plan_yaml(quorum=2))
        plan = load_plan(plan_file)
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.quorum == 2

    def test_all_valid_strategies_accepted(self, tmp_path: Path) -> None:
        for strategy in ("majority", "unanimous", "any"):
            plan_file = _write_plan(tmp_path, _plan_yaml(quorum=3, quorum_strategy=strategy))
            plan = load_plan(plan_file)
            assert plan.tasks[0].judge is not None
            assert plan.tasks[0].judge.quorum_strategy == strategy

    def test_quorum_without_strategy_defaults_none_in_spec(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, _plan_yaml(quorum=3))
        plan = load_plan(plan_file)
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.quorum_strategy is None

    def test_quorum_stored_on_judge_spec(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, _plan_yaml(quorum=5, quorum_strategy="unanimous"))
        plan = load_plan(plan_file)
        judge = plan.tasks[0].judge
        assert judge is not None
        assert judge.quorum == 5
        assert judge.quorum_strategy == "unanimous"


# ===========================================================================
# TestRunJudgeQuorumDisabled
# ===========================================================================


class TestRunJudgeQuorumDisabled:
    def test_quorum_none_calls_single_evaluation(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=None)
        fake = _jr(verdict="pass", score=0.85)

        with patch.object(runners, "_run_judge_evaluation", return_value=fake) as mock_eval:
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        mock_eval.assert_called_once()
        assert result.verdict == "pass"
        assert result.overall_score == pytest.approx(0.85)

    def test_quorum_one_calls_single_evaluation(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=1)
        fake = _jr(verdict="fail", score=0.1)

        with patch.object(runners, "_run_judge_evaluation", return_value=fake) as mock_eval:
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        mock_eval.assert_called_once()
        assert result.verdict == "fail"

    def test_single_fallback_passes_all_kwargs(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=None)
        captured: dict[str, Any] = {}

        def _capture(*args: Any, **kwargs: Any) -> JudgeResult:
            captured.update(kwargs)
            return _jr()

        with patch.object(runners, "_run_judge_evaluation", side_effect=_capture):
            _run_judge_quorum("my-task", spec, "stdout content", tmp_path, cost_usd=1.5)

        assert captured["task_id"] == "my-task"
        assert captured["stdout_tail"] == "stdout content"
        assert captured["cost_usd"] == pytest.approx(1.5)


# ===========================================================================
# TestRunJudgeQuorumMajorityStrategy
# ===========================================================================


class TestRunJudgeQuorumMajorityStrategy:
    def test_all_pass_gives_pass_verdict(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        responses = iter([_jr("pass", 0.9), _jr("pass", 0.85), _jr("pass", 0.95)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"

    def test_all_fail_gives_fail_verdict(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        responses = iter([_jr("fail", 0.1), _jr("fail", 0.2), _jr("fail", 0.15)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"

    def test_even_split_4_judges_fails(self, tmp_path: Path) -> None:
        # 4 judges: 2 pass, 2 fail → 2 > 4/2 = 2.0 is False → fail
        spec = _make_spec(quorum=4, quorum_strategy="majority")
        responses = iter([_jr("pass"), _jr("pass"), _jr("fail", 0.1), _jr("fail", 0.1)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"

    def test_3_of_4_pass_gives_pass_verdict(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=4, quorum_strategy="majority")
        responses = iter([_jr("pass"), _jr("pass"), _jr("pass"), _jr("fail", 0.1)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"

    def test_2_judge_minimum_both_pass(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=2, quorum_strategy="majority")
        responses = iter([_jr("pass", 0.9), _jr("pass", 0.8)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"

    def test_2_judge_minimum_one_fail_gives_fail(self, tmp_path: Path) -> None:
        # 2 judges: 1 pass, 1 fail → 1 > 2/2 = 1.0 is False → fail
        spec = _make_spec(quorum=2, quorum_strategy="majority")
        responses = iter([_jr("pass", 0.9), _jr("fail", 0.1)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"

    def test_majority_used_when_strategy_none(self, tmp_path: Path) -> None:
        # quorum_strategy=None → defaults to "majority" at runtime
        spec = _make_spec(quorum=3, quorum_strategy=None)
        responses = iter([_jr("pass", 0.9), _jr("pass", 0.85), _jr("fail", 0.2)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"
        assert "majority" in result.reasoning

    def test_5_judge_quorum_needs_3_to_pass(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=5, quorum_strategy="majority")
        # 3 pass, 2 fail → 3 > 2.5 → pass
        responses = iter([_jr("pass"), _jr("pass"), _jr("pass"), _jr("fail", 0.1), _jr("fail", 0.1)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"

    def test_5_judge_quorum_2_pass_fails(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=5, quorum_strategy="majority")
        # 2 pass, 3 fail → 2 > 2.5 is False → fail
        responses = iter([_jr("pass"), _jr("pass"), _jr("fail", 0.1), _jr("fail", 0.1), _jr("fail", 0.1)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"


# ===========================================================================
# TestRunJudgeQuorumUnanimousStrategy
# ===========================================================================


class TestRunJudgeQuorumUnanimousStrategy:
    def test_all_pass_gives_pass(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="unanimous")
        responses = iter([_jr("pass", 0.9), _jr("pass", 0.85), _jr("pass", 0.95)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"

    def test_single_fail_causes_fail(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="unanimous")
        responses = iter([_jr("pass", 0.9), _jr("fail", 0.1), _jr("pass", 0.8)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"

    def test_single_error_excluded_from_unanimous_vote(self, tmp_path: Path) -> None:
        """Error evaluations are excluded from voting — 2/2 valid pass = unanimous pass."""
        spec = _make_spec(quorum=3, quorum_strategy="unanimous")
        responses = iter([_jr("pass", 0.9), _jr("pass", 0.9), _jr("error", 0.0)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"
        assert "1 error(s) excluded" in result.reasoning

    def test_all_fail_gives_fail(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="unanimous")
        responses = iter([_jr("fail", 0.1), _jr("fail", 0.2), _jr("fail", 0.15)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"

    def test_reasoning_mentions_unanimous(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=2, quorum_strategy="unanimous")
        responses = iter([_jr("pass", 0.9), _jr("fail", 0.2)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert "unanimous" in result.reasoning


# ===========================================================================
# TestRunJudgeQuorumAnyStrategy
# ===========================================================================


class TestRunJudgeQuorumAnyStrategy:
    def test_all_fail_gives_fail(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="any")
        responses = iter([_jr("fail", 0.1), _jr("fail", 0.2), _jr("fail", 0.15)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"

    def test_all_error_gives_fail(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="any")
        responses = iter([_jr("error", 0.0), _jr("error", 0.0), _jr("error", 0.0)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"

    def test_last_judge_passes_gives_pass(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="any")
        responses = iter([_jr("fail", 0.1), _jr("fail", 0.2), _jr("pass", 0.9)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"

    def test_all_pass_gives_pass(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="any")
        responses = iter([_jr("pass"), _jr("pass"), _jr("pass")])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"

    def test_reasoning_mentions_any(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="any")
        responses = iter([_jr("pass"), _jr("fail", 0.1), _jr("fail", 0.1)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert "any" in result.reasoning


# ===========================================================================
# TestRunJudgeQuorumScoreAveraging
# ===========================================================================


class TestRunJudgeQuorumScoreAveraging:
    def test_error_verdicts_excluded_from_average(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        responses = iter([_jr("pass", 1.0), _jr("error", 0.0), _jr("pass", 0.6)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        # Only 1.0 and 0.6 are valid; error (0.0) excluded
        assert result.overall_score == pytest.approx((1.0 + 0.6) / 2.0)

    def test_all_error_verdicts_score_is_zero(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        responses = iter([_jr("error", 0.0), _jr("error", 0.0), _jr("error", 0.0)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.overall_score == pytest.approx(0.0)

    def test_score_averages_all_non_error_verdicts(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=4, quorum_strategy="majority")
        responses = iter([_jr("pass", 0.8), _jr("pass", 0.6), _jr("fail", 0.2), _jr("fail", 0.4)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.overall_score == pytest.approx((0.8 + 0.6 + 0.2 + 0.4) / 4.0)

    def test_single_valid_score_preserved(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="any")
        responses = iter([_jr("error", 0.0), _jr("error", 0.0), _jr("pass", 0.77)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.overall_score == pytest.approx(0.77)

    def test_identical_scores_average_is_same(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="unanimous")
        responses = iter([_jr("pass", 0.75), _jr("pass", 0.75), _jr("pass", 0.75)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.overall_score == pytest.approx(0.75)


# ===========================================================================
# TestRunJudgeQuorumReasoningFormat
# ===========================================================================


class TestRunJudgeQuorumReasoningFormat:
    def test_quorum_summary_line_format(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        responses = iter([_jr("pass", 0.9), _jr("pass", 0.8), _jr("fail", 0.2)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert "Quorum: 2/3 valid pass (majority)" in result.reasoning

    def test_judge_line_format_includes_verdict_and_score(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=2, quorum_strategy="majority")
        responses = iter([_jr("pass", 0.90), _jr("fail", 0.30)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert "Judge 1: pass (score=0.90)" in result.reasoning
        assert "Judge 2: fail (score=0.30)" in result.reasoning

    def test_all_judges_appear_in_reasoning(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=4, quorum_strategy="majority")
        responses = iter([_jr("pass"), _jr("pass"), _jr("fail", 0.1), _jr("warn", 0.5)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        for n in range(1, 5):
            assert f"Judge {n}:" in result.reasoning

    def test_unanimous_summary_format(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=2, quorum_strategy="unanimous")
        responses = iter([_jr("pass"), _jr("fail", 0.1)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert "Quorum: 1/2 valid pass (unanimous)" in result.reasoning

    def test_any_strategy_summary_format(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="any")
        responses = iter([_jr("fail", 0.1), _jr("pass", 0.9), _jr("fail", 0.1)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert "Quorum: 1/3 valid pass (any)" in result.reasoning

    def test_error_judge_appears_in_reasoning(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        responses = iter([_jr("error", 0.0), _jr("pass", 0.9), _jr("pass", 0.8)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert "Judge 1: error" in result.reasoning


# ===========================================================================
# TestRunJudgeQuorumRepresentativeReasoning
# ===========================================================================


class TestRunJudgeQuorumRepresentativeReasoning:
    def test_pass_verdict_uses_pass_judge_reasoning(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        responses = iter([
            _jr("fail", 0.1, "fail reasoning"),
            _jr("pass", 0.9, "pass reasoning"),
            _jr("pass", 0.85, "other pass reasoning"),
        ])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        # Representative reasoning comes from a pass judge
        assert "pass reasoning" in result.reasoning or "other pass reasoning" in result.reasoning

    def test_fail_verdict_uses_fail_judge_reasoning(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        responses = iter([
            _jr("pass", 0.9, "pass reasoning"),
            _jr("fail", 0.1, "fail reasoning A"),
            _jr("fail", 0.15, "fail reasoning B"),
        ])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        # 1 pass, 2 fail → verdict = fail; representative comes from a fail judge
        assert "fail reasoning" in result.reasoning

    def test_criterion_scores_copied_from_representative(self, tmp_path: Path) -> None:
        criterion_scores = [
            CriterionScore("accuracy", True, 0.95, "very accurate"),
            CriterionScore("style", False, 0.3, "poor style"),
        ]
        spec = _make_spec(quorum=2, quorum_strategy="any")
        responses = iter([
            _jr("fail", 0.1, "fail judge"),
            _jr("pass", 0.9, "pass judge", criterion_scores=criterion_scores),
        ])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        # Representative is the pass judge, criterion_scores should be copied
        assert result.criterion_scores is not None
        assert len(result.criterion_scores) == 2
        assert result.criterion_scores[0].criterion == "accuracy"

    def test_representative_reasoning_appended_to_summary(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=2, quorum_strategy="majority")
        responses = iter([
            _jr("pass", 0.9, "detailed assessment here"),
            _jr("pass", 0.85, "other assessment"),
        ])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        # Summary contains pass/judge lines AND representative reasoning
        assert "detailed assessment here" in result.reasoning or "other assessment" in result.reasoning


# ===========================================================================
# TestRunJudgeQuorumErrorHandling
# ===========================================================================


class TestRunJudgeQuorumErrorHandling:
    def test_exception_in_first_judge_recorded_as_error(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        call_count = {"n": 0}

        def _fake(*args: Any, **kwargs: Any) -> JudgeResult:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("network failure")
            return _jr("pass", 0.9)

        with patch.object(runners, "_run_judge_evaluation", side_effect=_fake):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert "Judge 1: error" in result.reasoning
        assert result.verdict == "pass"  # 2/3 pass via majority

    def test_all_exceptions_gives_fail_and_zero_score(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")

        def _raise(*args: Any, **kwargs: Any) -> JudgeResult:
            raise RuntimeError("always fails")

        with patch.object(runners, "_run_judge_evaluation", side_effect=_raise):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"
        assert result.overall_score == pytest.approx(0.0)

    def test_remaining_judges_run_after_exception(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        calls: list[int] = []

        def _fake(*args: Any, **kwargs: Any) -> JudgeResult:
            call_number = len(calls) + 1
            calls.append(call_number)
            if call_number == 2:
                raise ValueError("middle failure")
            return _jr("pass", 0.8)

        with patch.object(runners, "_run_judge_evaluation", side_effect=_fake):
            _run_judge_quorum("t1", spec, "output", tmp_path)

        # All 3 judges attempted despite the exception in judge 2
        assert len(calls) == 3

    def test_exception_error_verdict_excluded_from_score(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="any")
        call_count = {"n": 0}

        def _fake(*args: Any, **kwargs: Any) -> JudgeResult:
            call_count["n"] += 1
            if call_count["n"] == 3:
                raise RuntimeError("fails")
            return _jr("pass", 0.7)

        with patch.object(runners, "_run_judge_evaluation", side_effect=_fake):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        # 2 valid scores of 0.7, error excluded
        assert result.overall_score == pytest.approx(0.7)

    def test_quorum_runs_exact_n_times(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=5, quorum_strategy="majority")
        mock_eval = MagicMock(return_value=_jr("pass", 0.9))

        with patch.object(runners, "_run_judge_evaluation", mock_eval):
            _run_judge_quorum("t1", spec, "output", tmp_path)

        assert mock_eval.call_count == 5

    def test_quorum_passes_same_args_to_each_judge(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        captured_calls: list[dict[str, Any]] = []

        def _capture(*args: Any, **kwargs: Any) -> JudgeResult:
            captured_calls.append(kwargs)
            return _jr()

        with patch.object(runners, "_run_judge_evaluation", side_effect=_capture):
            _run_judge_quorum("my-task", spec, "my output", tmp_path)

        assert len(captured_calls) == 3
        # All calls have same task_id, stdout_tail, and workdir
        for kw in captured_calls:
            assert kw["task_id"] == "my-task"
            assert kw["stdout_tail"] == "my output"
            assert kw["workdir"] == tmp_path


# ===========================================================================
# W24: quorum > 3 warning
# ===========================================================================


class TestW24QuorumWarning:
    def test_quorum_4_emits_w24(self, tmp_path: Path) -> None:
        plan = load_plan(_write_plan(tmp_path, _plan_yaml(quorum=4)))
        w24 = [w for w in plan.validation_warnings if w.startswith("W24:")]
        assert len(w24) == 1
        assert "quorum=4" in w24[0]
        assert "degrades" in w24[0]

    def test_quorum_3_no_w24(self, tmp_path: Path) -> None:
        plan = load_plan(_write_plan(tmp_path, _plan_yaml(quorum=3)))
        w24 = [w for w in plan.validation_warnings if w.startswith("W24:")]
        assert len(w24) == 0

    def test_quorum_2_no_w24(self, tmp_path: Path) -> None:
        plan = load_plan(_write_plan(tmp_path, _plan_yaml(quorum=2)))
        w24 = [w for w in plan.validation_warnings if w.startswith("W24:")]
        assert len(w24) == 0


# ===========================================================================
# Timeout-aware quorum voting
# ===========================================================================


class TestTimeoutAwareQuorum:
    def test_majority_excludes_errors_from_denominator(self, tmp_path: Path) -> None:
        """1 pass + 1 error out of 3 → valid_count=2, pass_count=1 → 1/2 → fail."""
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        responses = iter([_jr("pass", 0.9), _jr("fail", 0.3), _jr("error", 0.0)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"
        assert "1 error(s) excluded" in result.reasoning

    def test_majority_2_pass_1_error_passes(self, tmp_path: Path) -> None:
        """2 pass + 1 error → valid_count=2, pass_count=2 → 2/2 > 1 → pass."""
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        responses = iter([_jr("pass", 0.9), _jr("pass", 0.8), _jr("error", 0.0)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"
        assert "1 error(s) excluded" in result.reasoning

    def test_unanimous_with_errors_passes_if_all_valid_pass(self, tmp_path: Path) -> None:
        """2 pass + 1 error → valid_count=2, pass_count=2 → unanimous pass."""
        spec = _make_spec(quorum=3, quorum_strategy="unanimous")
        responses = iter([_jr("pass", 0.9), _jr("error", 0.0), _jr("pass", 0.85)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"

    def test_unanimous_with_errors_fails_if_any_valid_fail(self, tmp_path: Path) -> None:
        """1 pass + 1 fail + 1 error → valid_count=2, pass_count=1 → not unanimous."""
        spec = _make_spec(quorum=3, quorum_strategy="unanimous")
        responses = iter([_jr("pass", 0.9), _jr("fail", 0.2), _jr("error", 0.0)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"

    def test_all_errors_fails(self, tmp_path: Path) -> None:
        """All errors → valid_count=0 → fail."""
        spec = _make_spec(quorum=3, quorum_strategy="majority")
        responses = iter([_jr("error", 0.0), _jr("error", 0.0), _jr("error", 0.0)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "fail"
        assert result.overall_score == pytest.approx(0.0)

    def test_error_note_not_shown_when_no_errors(self, tmp_path: Path) -> None:
        spec = _make_spec(quorum=2, quorum_strategy="majority")
        responses = iter([_jr("pass", 0.9), _jr("pass", 0.8)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert "excluded" not in result.reasoning


# ===========================================================================
# TestQuorumDiversity
# ===========================================================================


class TestQuorumDiversitySpec:
    """Test quorum_diversity field on JudgeSpec."""

    def test_default_is_false(self) -> None:
        spec = JudgeSpec()
        assert spec.quorum_diversity is False

    def test_to_dict_includes_quorum_diversity(self) -> None:
        spec = JudgeSpec(quorum=3, quorum_diversity=True)
        d = spec.to_dict()
        assert "quorum_diversity" in d
        assert d["quorum_diversity"] is True

    def test_to_dict_false_when_disabled(self) -> None:
        spec = JudgeSpec(quorum=3, quorum_diversity=False)
        d = spec.to_dict()
        assert d["quorum_diversity"] is False


class TestQuorumDiversityLoader:
    """Test loader parsing and W25 warning."""

    def test_parse_quorum_diversity_true(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\nname: qd-test\ntasks:\n"
            "  - id: t1\n    engine: claude\n    prompt: do\n"
            "    judge:\n      criteria: [quality]\n"
            "      quorum: 3\n      quorum_diversity: true\n"
        )
        plan = load_plan(_write_plan(tmp_path, yaml_text))
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.quorum_diversity is True

    def test_parse_quorum_diversity_false_by_default(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\nname: qd-test\ntasks:\n"
            "  - id: t1\n    engine: claude\n    prompt: do\n"
            "    judge:\n      criteria: [quality]\n      quorum: 3\n"
        )
        plan = load_plan(_write_plan(tmp_path, yaml_text))
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.quorum_diversity is False

    def test_w25_diversity_without_quorum(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\nname: qd-test\ntasks:\n"
            "  - id: t1\n    engine: claude\n    prompt: do\n"
            "    judge:\n      criteria: [quality]\n"
            "      quorum_diversity: true\n"
        )
        plan = load_plan(_write_plan(tmp_path, yaml_text))
        w25 = [w for w in plan.validation_warnings if "W25" in w]
        assert len(w25) == 1
        assert "quorum_diversity" in w25[0]

    def test_no_w25_when_quorum_set(self, tmp_path: Path) -> None:
        yaml_text = (
            "version: 1\nname: qd-test\ntasks:\n"
            "  - id: t1\n    engine: claude\n    prompt: do\n"
            "    judge:\n      criteria: [quality]\n"
            "      quorum: 3\n      quorum_diversity: true\n"
        )
        plan = load_plan(_write_plan(tmp_path, yaml_text))
        w25 = [w for w in plan.validation_warnings if "W25" in w]
        assert len(w25) == 0


class TestQuorumDiversityModelCycling:
    """Test model cycling in _run_judge_quorum when diversity is enabled."""

    def test_diversity_uses_different_models(self, tmp_path: Path) -> None:
        """With quorum=3 and diversity, each slot should use a different model."""
        spec = JudgeSpec(
            criteria=["quality"],
            quorum=3,
            quorum_strategy="majority",
            quorum_diversity=True,
        )
        captured_models: list[str] = []

        def mock_eval(*args: Any, **kwargs: Any) -> JudgeResult:
            judge_arg = kwargs.get("judge") or args[1]
            captured_models.append(judge_arg.model)
            return _jr("pass", 0.9)

        with patch.object(runners, "_run_judge_evaluation", side_effect=mock_eval):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert result.verdict == "pass"
        assert captured_models == ["haiku", "sonnet", "opus"]

    def test_diversity_cycles_for_quorum_4(self, tmp_path: Path) -> None:
        """With quorum=4, models cycle: haiku, sonnet, opus, haiku."""
        spec = JudgeSpec(
            criteria=["quality"],
            quorum=4,
            quorum_strategy="majority",
            quorum_diversity=True,
        )
        captured_models: list[str] = []

        def mock_eval(*args: Any, **kwargs: Any) -> JudgeResult:
            judge_arg = kwargs.get("judge") or args[1]
            captured_models.append(judge_arg.model)
            return _jr("pass", 0.85)

        with patch.object(runners, "_run_judge_evaluation", side_effect=mock_eval):
            _run_judge_quorum("t1", spec, "output", tmp_path)

        assert captured_models == ["haiku", "sonnet", "opus", "haiku"]

    def test_no_diversity_uses_same_model(self, tmp_path: Path) -> None:
        """Without diversity, all slots use the same model."""
        spec = JudgeSpec(
            criteria=["quality"],
            quorum=3,
            quorum_strategy="majority",
            quorum_diversity=False,
            model="sonnet",
        )
        captured_models: list[str] = []

        def mock_eval(*args: Any, **kwargs: Any) -> JudgeResult:
            judge_arg = kwargs.get("judge") or args[1]
            captured_models.append(judge_arg.model)
            return _jr("pass", 0.9)

        with patch.object(runners, "_run_judge_evaluation", side_effect=mock_eval):
            _run_judge_quorum("t1", spec, "output", tmp_path)

        assert captured_models == ["sonnet", "sonnet", "sonnet"]

    def test_diversity_reasoning_includes_model_tags(self, tmp_path: Path) -> None:
        """Reasoning should include [model] tags when diversity is on."""
        spec = JudgeSpec(
            criteria=["quality"],
            quorum=3,
            quorum_strategy="majority",
            quorum_diversity=True,
        )
        responses = iter([_jr("pass", 0.9), _jr("pass", 0.8), _jr("fail", 0.4)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert "[haiku]" in result.reasoning
        assert "[sonnet]" in result.reasoning
        assert "[opus]" in result.reasoning
        assert "diverse models" in result.reasoning

    def test_diversity_reasoning_no_model_tags_when_off(self, tmp_path: Path) -> None:
        """No model tags in reasoning when diversity is off."""
        spec = JudgeSpec(
            criteria=["quality"],
            quorum=2,
            quorum_strategy="majority",
        )
        responses = iter([_jr("pass", 0.9), _jr("pass", 0.8)])

        with patch.object(runners, "_run_judge_evaluation", side_effect=lambda *a, **kw: next(responses)):
            result = _run_judge_quorum("t1", spec, "output", tmp_path)

        assert "[haiku]" not in result.reasoning
        assert "diverse models" not in result.reasoning
