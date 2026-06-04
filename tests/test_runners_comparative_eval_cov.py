from __future__ import annotations

from pathlib import Path
import json
import subprocess
from typing import Any

import pytest

from maestro_cli.models import CriterionScore, JudgeResult, JudgeSpec
from maestro_cli.runners import _run_comparative_evaluation


def _fake_proc(stdout: str, returncode: int = 0) -> Any:
    proc = type("R", (), {})()
    proc.returncode = returncode
    proc.stdout = stdout
    proc.stderr = ""
    return proc


class TestRunComparativeEvaluationUncovered:
    """Drive the uncovered branches of _run_comparative_evaluation."""

    def test_parse_error_returns_result_early(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-empty stdout with no JSON object => parse 'error' verdict returned (L6278)."""
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable", lambda x: [x]
        )
        # returncode 0 and non-empty stdout passes the L6264 guard, but contains
        # no '{' so _parse_judge_response returns verdict='error', hitting L6278.
        monkeypatch.setattr(
            "maestro_cli.runners.subprocess.run",
            lambda *a, **kw: _fake_proc("no json object here at all"),
        )
        judge = JudgeSpec(criteria=["Quality"], pass_threshold=0.7)
        result = _run_comparative_evaluation(
            task_id="t1",
            judge=judge,
            current_output="current",
            previous_output="previous",
            previous_score=0.3,
            workdir=tmp_path,
        )
        assert result.verdict == "error"
        assert result.previous_score == 0.3

    def test_non_dict_criteria_item_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-dict entry inside metadata 'criteria' list is skipped (L6299)."""
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable", lambda x: [x]
        )
        # The metadata 'criteria' list mixes a non-dict item (skipped at L6299)
        # with a valid dict item. overall_score 0.9 => verdict 'pass'.
        response = json.dumps({
            "criteria": [
                "this is not a dict -- must be skipped",
                {"criterion": "Quality", "passed": True, "score": 0.9,
                 "improved": True, "reasoning": "better"},
            ],
            "overall_score": 0.9,
            "overall_improved": True,
            "reasoning": "Improvement detected",
        })
        monkeypatch.setattr(
            "maestro_cli.runners.subprocess.run",
            lambda *a, **kw: _fake_proc(response),
        )
        judge = JudgeSpec(criteria=["Quality"], pass_threshold=0.7)
        result = _run_comparative_evaluation(
            task_id="t1",
            judge=judge,
            current_output="improved",
            previous_output="bad",
            previous_score=0.4,
            workdir=tmp_path,
        )
        assert result.verdict == "pass"
        assert result.previous_score == 0.4
        # The valid criterion got an [improved] label prefix.
        assert any("improved" in cs.reasoning for cs in result.criterion_scores)

    def test_improved_none_for_criterion_score_skips_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A criterion_score whose name is absent from the improved map is skipped (L6308).

        The parsed criterion_scores include 'Quality', but the improvement map only
        carries 'SomethingElse'. improved_by_criterion is non-empty (so the labelling
        loop runs), but .get('Quality') is None -> continue at L6308.
        """
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable", lambda x: [x]
        )
        response = json.dumps({
            "criteria": [
                # This dict produces the CriterionScore (criterion='Quality')...
                {"criterion": "Quality", "passed": True, "score": 0.9,
                 "reasoning": "ok"},
                # ...and this dict carries a *different* improved-named entry, so the
                # improvement map = {'SomethingElse': True}, never matching 'Quality'.
                {"criterion": "SomethingElse", "improved": True},
            ],
            "overall_score": 0.9,
            "reasoning": "Fine",
        })
        monkeypatch.setattr(
            "maestro_cli.runners.subprocess.run",
            lambda *a, **kw: _fake_proc(response),
        )
        judge = JudgeSpec(criteria=["Quality"], pass_threshold=0.7)
        result = _run_comparative_evaluation(
            task_id="t1",
            judge=judge,
            current_output="cur",
            previous_output="prev",
            previous_score=0.5,
            workdir=tmp_path,
        )
        assert result.verdict == "pass"
        # 'Quality' had no matching improved entry -> reasoning left unlabelled.
        quality = next(
            cs for cs in result.criterion_scores if cs.criterion == "Quality"
        )
        assert not quality.reasoning.startswith("[")

    def test_metadata_block_exception_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If the metadata re-parse raises, it is swallowed (L6311-6312) and the
        threshold-based verdict assignment still runs (L6314-6319).

        _parse_judge_response calls json.loads once (succeeds). The metadata block
        calls json.loads a second time -- we make that call raise so the
        `except Exception: pass` branch executes. The verdict is then derived from
        overall_score vs pass_threshold (0.9 >= 0.7 => 'pass').
        """
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable", lambda x: [x]
        )
        response = json.dumps({
            "criteria": [
                {"criterion": "Quality", "passed": True, "score": 0.9,
                 "reasoning": "ok"},
            ],
            "overall_score": 0.9,
            "reasoning": "Good",
        })
        monkeypatch.setattr(
            "maestro_cli.runners.subprocess.run",
            lambda *a, **kw: _fake_proc(response),
        )

        real_loads = json.loads
        calls = {"n": 0}

        def _flaky_loads(*args: Any, **kwargs: Any) -> Any:
            calls["n"] += 1
            # First call comes from _parse_judge_response -> let it succeed.
            if calls["n"] == 1:
                return real_loads(*args, **kwargs)
            # Second call is the metadata re-parse -> blow up to hit L6311-6312.
            raise ValueError("boom in metadata parse")

        monkeypatch.setattr("maestro_cli.runners.json.loads", _flaky_loads)

        judge = JudgeSpec(criteria=["Quality"], pass_threshold=0.7)
        result = _run_comparative_evaluation(
            task_id="t1",
            judge=judge,
            current_output="cur",
            previous_output="prev",
            previous_score=0.5,
            workdir=tmp_path,
        )
        # Metadata exception swallowed; threshold logic still assigns the verdict.
        assert result.verdict == "pass"
        assert result.previous_score == 0.5
        assert calls["n"] >= 2

    def test_verdict_warn_from_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """overall_score between 0.5*threshold and threshold => 'warn' (L6316-6317)."""
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable", lambda x: [x]
        )
        # threshold 0.8 -> warn band is [0.4, 0.8). score 0.6 lands in it.
        response = json.dumps({
            "criteria": [
                {"criterion": "Quality", "passed": False, "score": 0.6,
                 "reasoning": "meh"},
            ],
            "overall_score": 0.6,
            "reasoning": "Partial",
        })
        monkeypatch.setattr(
            "maestro_cli.runners.subprocess.run",
            lambda *a, **kw: _fake_proc(response),
        )
        judge = JudgeSpec(criteria=["Quality"], pass_threshold=0.8)
        result = _run_comparative_evaluation(
            task_id="t1",
            judge=judge,
            current_output="cur",
            previous_output="prev",
            previous_score=0.5,
            workdir=tmp_path,
        )
        assert result.verdict == "warn"
        assert result.previous_score == 0.5

    def test_verdict_fail_from_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """overall_score below 0.5*threshold => 'fail' (L6318-6319)."""
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable", lambda x: [x]
        )
        # threshold 0.8 -> fail band is < 0.4. score 0.2 lands below it.
        response = json.dumps({
            "criteria": [
                {"criterion": "Quality", "passed": False, "score": 0.2,
                 "reasoning": "bad"},
            ],
            "overall_score": 0.2,
            "reasoning": "Worse",
        })
        monkeypatch.setattr(
            "maestro_cli.runners.subprocess.run",
            lambda *a, **kw: _fake_proc(response),
        )
        judge = JudgeSpec(criteria=["Quality"], pass_threshold=0.8)
        result = _run_comparative_evaluation(
            task_id="t1",
            judge=judge,
            current_output="cur",
            previous_output="prev",
            previous_score=0.5,
            workdir=tmp_path,
        )
        assert result.verdict == "fail"
        assert result.previous_score == 0.5

    def test_generic_exception_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A non-timeout exception during the subprocess call => generic error (L6328-6329)."""
        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable", lambda x: [x]
        )

        def _raise(*a: Any, **kw: Any) -> None:
            raise RuntimeError("unexpected explosion")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", _raise)
        judge = JudgeSpec(criteria=["Quality"], pass_threshold=0.7)
        result = _run_comparative_evaluation(
            task_id="t1",
            judge=judge,
            current_output="cur",
            previous_output="prev",
            previous_score=0.6,
            workdir=tmp_path,
        )
        assert result.verdict == "error"
        assert result.previous_score == 0.6
        assert "error" in result.reasoning.lower()
        assert "unexpected explosion" in result.reasoning