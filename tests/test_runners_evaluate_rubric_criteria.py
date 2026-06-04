from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import subprocess

from maestro_cli.models import CriterionScore
from maestro_cli.runners import _evaluate_rubric_criteria


def _make_proc(stdout: str, returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(
        args=["claude"], returncode=returncode, stdout=stdout, stderr=""
    )


def _patch_run(
    monkeypatch: pytest.MonkeyPatch, proc: subprocess.CompletedProcess[str]
) -> None:
    def _fake_run(*_a: Any, **_kw: Any) -> subprocess.CompletedProcess[str]:
        return proc

    monkeypatch.setattr("maestro_cli.runners.subprocess.run", _fake_run)


def _good_levels() -> list[dict[str, Any]]:
    return [
        {"score": 1, "description": "poor"},
        {"score": 3, "description": "ok"},
        {"score": 5, "description": "great"},
    ]


class TestEvaluateRubricFallbacks:
    def test_no_json_braces_returns_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4858: stripped output has no '{' -> start == -1 -> fallback_scores."""
        _patch_run(monkeypatch, _make_proc("no json here at all"))
        rubric = [{"name": "clarity", "levels": _good_levels(), "min_score": 3}]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        assert len(result) == 1
        assert result[0].criterion == "clarity"
        assert result[0].passed is False
        assert "Rubric evaluation failed" in result[0].reasoning

    def test_criteria_not_a_list_returns_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4862: payload['criteria'] is not a list -> fallback_scores."""
        payload = json.dumps({"criteria": {"clarity": 5}})
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [{"name": "clarity", "levels": _good_levels(), "min_score": 3}]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        assert len(result) == 1
        assert "Rubric evaluation failed" in result[0].reasoning

    def test_nonzero_returncode_returns_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_run(monkeypatch, _make_proc('{"criteria": []}', returncode=1))
        rubric = [{"name": "clarity", "levels": _good_levels(), "min_score": 3}]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        assert "Rubric evaluation failed" in result[0].reasoning

    def test_empty_stdout_returns_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _patch_run(monkeypatch, _make_proc("   "))
        rubric = [{"name": "clarity", "levels": _good_levels(), "min_score": 3}]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        assert "Rubric evaluation failed" in result[0].reasoning

    def test_no_evaluated_items_returns_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4915: every item skipped -> evaluated is empty -> fallback_scores.

        Item references an unknown criterion name so matched_criterion is None.
        """
        payload = json.dumps({"criteria": [{"name": "unknown", "score": 5}]})
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [{"name": "clarity", "levels": _good_levels(), "min_score": 3}]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        assert len(result) == 1
        assert result[0].criterion == "clarity"
        assert "Rubric evaluation failed" in result[0].reasoning

    def test_subprocess_exception_returns_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """L4922-4924: subprocess raises -> except -> fallback_scores + E107 print."""

        def _boom(*_a: Any, **_kw: Any) -> subprocess.CompletedProcess[str]:
            raise OSError("boom")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", _boom)
        rubric = [{"name": "clarity", "levels": _good_levels(), "min_score": 3}]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        assert "Rubric evaluation failed" in result[0].reasoning
        assert "[E107] rubric evaluation failed" in capsys.readouterr().out


class TestEvaluateRubricItemSkips:
    def test_item_not_dict_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4867: a non-dict item is skipped; a valid item still evaluates."""
        payload = json.dumps(
            {
                "criteria": [
                    "not-a-dict",
                    {"name": "clarity", "score": 5, "reasoning": "good"},
                ]
            }
        )
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [{"name": "clarity", "levels": _good_levels(), "min_score": 3}]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        scores = {s.criterion: s for s in result}
        assert scores["clarity"].passed is True
        assert scores["clarity"].score == pytest.approx(1.0)

    def test_unknown_criterion_name_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4871: item name not in rubric -> matched_criterion None -> skipped."""
        payload = json.dumps(
            {
                "criteria": [
                    {"name": "phantom", "score": 5},
                    {"name": "clarity", "score": 3, "reasoning": "fine"},
                ]
            }
        )
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [{"name": "clarity", "levels": _good_levels(), "min_score": 3}]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        scores = {s.criterion: s for s in result}
        assert "phantom" not in scores
        assert scores["clarity"].passed is True

    def test_levels_not_a_list_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4875: criterion 'levels' is not a list -> item skipped -> fallback."""
        payload = json.dumps({"criteria": [{"name": "clarity", "score": 5}]})
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [{"name": "clarity", "levels": "notalist", "min_score": 3}]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        # No valid evaluated item -> single fallback returned.
        assert len(result) == 1
        assert result[0].criterion == "clarity"
        assert "Rubric evaluation failed" in result[0].reasoning

    def test_non_dict_level_is_skipped_but_others_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4879: a non-dict level entry is skipped; valid levels still counted."""
        payload = json.dumps(
            {"criteria": [{"name": "clarity", "score": 4, "reasoning": "r"}]}
        )
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [
            {
                "name": "clarity",
                "levels": ["junk", {"score": 4}, {"score": 8}],
                "min_score": 4,
            }
        ]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        scores = {s.criterion: s for s in result}
        # max valid level score is 8, selected 4 -> normalized 0.5, passed (4>=4).
        assert scores["clarity"].score == pytest.approx(0.5)
        assert scores["clarity"].passed is True

    def test_level_with_none_score_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4882: level whose 'score' is None is skipped; others remain."""
        payload = json.dumps(
            {"criteria": [{"name": "clarity", "score": 5, "reasoning": "r"}]}
        )
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [
            {
                "name": "clarity",
                "levels": [{"description": "missing score"}, {"score": 10}],
                "min_score": 1,
            }
        ]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        scores = {s.criterion: s for s in result}
        # only level score is 10, selected 5 -> 0.5.
        assert scores["clarity"].score == pytest.approx(0.5)

    def test_level_score_not_floatable_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4885-4886: non-numeric level score raises ValueError -> continue."""
        payload = json.dumps(
            {"criteria": [{"name": "clarity", "score": 6, "reasoning": "r"}]}
        )
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [
            {
                "name": "clarity",
                "levels": [{"score": "abc"}, {"score": 12}],
                "min_score": 1,
            }
        ]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        scores = {s.criterion: s for s in result}
        # bad level dropped, valid max is 12, selected 6 -> 0.5.
        assert scores["clarity"].score == pytest.approx(0.5)

    def test_no_valid_level_scores_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4888: all levels invalid -> level_scores empty -> item skipped."""
        payload = json.dumps({"criteria": [{"name": "clarity", "score": 5}]})
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [
            {
                "name": "clarity",
                "levels": [{"score": None}, {"score": "x"}],
                "min_score": 1,
            }
        ]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        assert len(result) == 1
        assert "Rubric evaluation failed" in result[0].reasoning

    def test_selected_score_none_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4892: item 'score' is None -> item skipped -> fallback."""
        payload = json.dumps({"criteria": [{"name": "clarity", "reasoning": "r"}]})
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [{"name": "clarity", "levels": _good_levels(), "min_score": 3}]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        assert len(result) == 1
        assert "Rubric evaluation failed" in result[0].reasoning

    def test_selected_score_not_floatable_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4895-4896: item 'score' non-numeric raises -> item skipped -> fallback."""
        payload = json.dumps({"criteria": [{"name": "clarity", "score": "nope"}]})
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [{"name": "clarity", "levels": _good_levels(), "min_score": 3}]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        assert len(result) == 1
        assert "Rubric evaluation failed" in result[0].reasoning


class TestEvaluateRubricMinScoreAndMerge:
    def test_non_floatable_min_score_defaults_to_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4904-4905: min_score not float-able -> except -> min_score = 0.0.

        With min_score forced to 0.0, any selected score passes.
        """
        payload = json.dumps(
            {"criteria": [{"name": "clarity", "score": 1, "reasoning": "low"}]}
        )
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [
            {"name": "clarity", "levels": _good_levels(), "min_score": "not-a-number"}
        ]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        scores = {s.criterion: s for s in result}
        # selected 1 >= min_score 0.0 -> passed True; max level 5 -> 0.2.
        assert scores["clarity"].passed is True
        assert scores["clarity"].score == pytest.approx(0.2)

    def test_missing_criterion_appended_as_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """L4917-4920: criteria not present in evaluated set get fallback appended."""
        payload = json.dumps(
            {"criteria": [{"name": "clarity", "score": 5, "reasoning": "great"}]}
        )
        _patch_run(monkeypatch, _make_proc(payload))
        rubric = [
            {"name": "clarity", "levels": _good_levels(), "min_score": 3},
            {"name": "depth", "levels": _good_levels(), "min_score": 3},
        ]
        result = _evaluate_rubric_criteria(rubric, "output", tmp_path)
        scores = {s.criterion: s for s in result}
        assert set(scores) == {"clarity", "depth"}
        # clarity evaluated successfully.
        assert scores["clarity"].passed is True
        assert scores["clarity"].score == pytest.approx(1.0)
        # depth never evaluated -> appended fallback.
        assert scores["depth"].passed is False
        assert "Rubric evaluation failed" in scores["depth"].reasoning
