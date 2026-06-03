from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

import maestro_cli.cache as cache
import maestro_cli.models as models
import maestro_cli.runners as runners
from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import JudgeSpec, PlanDefaults, PlanSpec, TaskSpec


def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


def _build_hash_plan(tmp_path: Path, task: TaskSpec) -> PlanSpec:
    source_path = tmp_path / "plan-hash.yaml"
    source_path.write_text("version: 1\nname: hash\n", encoding="utf-8")
    return PlanSpec(
        version=1,
        name="hash-plan",
        defaults=PlanDefaults(),
        tasks=[task],
        source_path=source_path,
    )


class TestComparativeEval:
    def test_comparative_prompt_template_format(self) -> None:
        prompt = runners._COMPARATIVE_JUDGE_PROMPT_TEMPLATE.format(
            previous_output="previous text",
            current_output="current text",
            previous_score=0.35,
            criteria_list="  1. quality",
        )
        assert "previous text" in prompt
        assert "current text" in prompt
        assert "0.35/1.0" in prompt

    def test_run_comparative_with_mock(self, tmp_path: Path, monkeypatch: Any) -> None:
        payload = {
            "criteria": [
                {
                    "criterion": "quality",
                    "passed": True,
                    "score": 0.9,
                    "improved": True,
                    "reasoning": "better than previous",
                }
            ],
            "overall_score": 0.9,
            "overall_improved": True,
            "reasoning": "much better",
        }

        def mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=args[0] if args else [],
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = runners._run_comparative_evaluation(
            task_id="t1",
            judge=JudgeSpec(criteria=["quality"], pass_threshold=0.7),
            current_output="new output",
            previous_output="old output",
            previous_score=0.4,
            workdir=tmp_path,
        )
        assert result.verdict == "pass"
        assert result.overall_score == pytest.approx(0.9)
        assert result.previous_score == pytest.approx(0.4)
        assert "Overall improvement vs previous: yes." in result.reasoning
        assert result.criterion_scores
        assert "[improved]" in result.criterion_scores[0].reasoning

    def test_run_comparative_timeout(self, tmp_path: Path, monkeypatch: Any) -> None:
        def mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=60)

        monkeypatch.setattr(subprocess, "run", mock_run)
        result = runners._run_comparative_evaluation(
            task_id="t1",
            judge=JudgeSpec(criteria=["quality"], pass_threshold=0.7),
            current_output="new output",
            previous_output="old output",
            previous_score=0.2,
            workdir=tmp_path,
        )
        assert result.verdict == "error"
        assert "timed out" in result.reasoning

    def test_run_comparative_truncates_output(self, tmp_path: Path, monkeypatch: Any) -> None:
        seen_prompt: dict[str, str] = {}
        payload = {
            "criteria": [{"criterion": "quality", "passed": True, "score": 0.8, "reasoning": "ok"}],
            "overall_score": 0.8,
            "reasoning": "ok",
        }

        def mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            cmd = args[0] if args else []
            seen_prompt["value"] = cmd[-1]
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout=json.dumps(payload),
                stderr="",
            )

        monkeypatch.setattr(subprocess, "run", mock_run)

        previous_output = "HEAD_PREV|" + ("p" * 2000) + "|TAIL_PREV"
        current_output = "HEAD_CURR|" + ("c" * 2000) + "|TAIL_CURR"
        runners._run_comparative_evaluation(
            task_id="t1",
            judge=JudgeSpec(criteria=["quality"], pass_threshold=0.7),
            current_output=current_output,
            previous_output=previous_output,
            previous_score=0.4,
            workdir=tmp_path,
        )

        prompt = seen_prompt["value"]
        assert "HEAD_PREV" not in prompt
        assert "HEAD_CURR" not in prompt
        assert "TAIL_PREV" in prompt
        assert "TAIL_CURR" in prompt


class TestJudgePresets:
    def test_code_quality_preset_exists(self) -> None:
        assert "code_quality" in models.JUDGE_PRESETS

    def test_security_audit_preset_exists(self) -> None:
        assert "security_audit" in models.JUDGE_PRESETS

    def test_preset_has_criteria(self) -> None:
        for preset in models.JUDGE_PRESETS.values():
            assert isinstance(preset.get("criteria"), list)
            assert preset["criteria"]

    def test_preset_has_pass_threshold(self) -> None:
        for preset in models.JUDGE_PRESETS.values():
            threshold = preset.get("pass_threshold")
            assert isinstance(threshold, (int, float))
            assert 0.0 <= float(threshold) <= 1.0

    def test_preset_has_aggregation(self) -> None:
        for preset in models.JUDGE_PRESETS.values():
            aggregation = preset.get("aggregation")
            assert isinstance(aggregation, str)
            assert aggregation in models.SCORE_AGGREGATIONS

    def test_preset_criteria_are_rubric_type(self) -> None:
        for preset in models.JUDGE_PRESETS.values():
            for criterion in preset["criteria"]:
                assert criterion.get("type") == "rubric"

    def test_loader_expands_preset_criteria(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: preset-expand
tasks:
  - id: t1
    engine: claude
    prompt: "Judge this"
    judge:
      preset: code_quality
""",
        )
        plan = load_plan(plan_file)
        judge = plan.tasks[0].judge
        assert judge is not None
        assert judge.preset == "code_quality"
        assert judge.criteria == models.JUDGE_PRESETS["code_quality"]["criteria"]

    def test_loader_preset_override(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: preset-override
tasks:
  - id: t1
    engine: claude
    prompt: "Judge this"
    judge:
      preset: code_quality
      criteria:
        - "must pass tests"
""",
        )
        plan = load_plan(plan_file)
        judge = plan.tasks[0].judge
        assert judge is not None
        assert judge.criteria == ["must pass tests"]

    def test_loader_invalid_preset_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: preset-invalid
tasks:
  - id: t1
    engine: claude
    prompt: "Judge this"
    judge:
      preset: unknown_preset
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E020\].*preset"):
            load_plan(plan_file)


class TestJudgeCacheHash:
    def test_different_method_different_hash(self, tmp_path: Path) -> None:
        task_a = TaskSpec(
            id="t1",
            command="echo ok",
            judge=JudgeSpec(criteria=["quality"], method="direct"),
        )
        task_b = TaskSpec(
            id="t1",
            command="echo ok",
            judge=JudgeSpec(criteria=["quality"], method="g_eval"),
        )
        plan_a = _build_hash_plan(tmp_path, task_a)
        plan_b = _build_hash_plan(tmp_path, task_b)
        assert cache.compute_task_hash(task_a, plan_a, {}) != cache.compute_task_hash(task_b, plan_b, {})

    def test_different_aggregation_different_hash(self, tmp_path: Path) -> None:
        task_a = TaskSpec(
            id="t1",
            command="echo ok",
            judge=JudgeSpec(criteria=["quality"], aggregation="mean"),
        )
        task_b = TaskSpec(
            id="t1",
            command="echo ok",
            judge=JudgeSpec(criteria=["quality"], aggregation="min"),
        )
        plan_a = _build_hash_plan(tmp_path, task_a)
        plan_b = _build_hash_plan(tmp_path, task_b)
        assert cache.compute_task_hash(task_a, plan_a, {}) != cache.compute_task_hash(task_b, plan_b, {})

    def test_different_preset_different_hash(self, tmp_path: Path) -> None:
        task_a = TaskSpec(
            id="t1",
            command="echo ok",
            judge=JudgeSpec(
                criteria=["quality"],
                preset="code_quality",
            ),
        )
        task_b = TaskSpec(
            id="t1",
            command="echo ok",
            judge=JudgeSpec(
                criteria=["quality"],
                preset="security_audit",
            ),
        )
        plan_a = _build_hash_plan(tmp_path, task_a)
        plan_b = _build_hash_plan(tmp_path, task_b)
        assert cache.compute_task_hash(task_a, plan_a, {}) != cache.compute_task_hash(task_b, plan_b, {})

    def test_same_judge_same_hash(self, tmp_path: Path) -> None:
        task_a = TaskSpec(
            id="t1",
            command="echo ok",
            judge=JudgeSpec(
                criteria=["quality"],
                method="direct",
                aggregation="mean",
                preset="code_quality",
            ),
        )
        task_b = TaskSpec(
            id="t1",
            command="echo ok",
            judge=JudgeSpec(
                criteria=["quality"],
                method="direct",
                aggregation="mean",
                preset="code_quality",
            ),
        )
        plan_a = _build_hash_plan(tmp_path, task_a)
        plan_b = _build_hash_plan(tmp_path, task_b)
        assert cache.compute_task_hash(task_a, plan_a, {}) == cache.compute_task_hash(task_b, plan_b, {})


class TestLoaderNewValidation:
    def test_invalid_method_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: invalid-method
tasks:
  - id: t1
    engine: claude
    prompt: "Judge this"
    judge:
      criteria: ["quality"]
      method: unknown
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E020\].*method"):
            load_plan(plan_file)

    def test_invalid_aggregation_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: invalid-aggregation
tasks:
  - id: t1
    engine: claude
    prompt: "Judge this"
    judge:
      criteria: ["quality"]
      aggregation: unknown
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E020\].*aggregation"):
            load_plan(plan_file)

    def test_rubric_criterion_missing_name_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: rubric-missing-name
tasks:
  - id: t1
    engine: claude
    prompt: "Judge this"
    judge:
      criteria:
        - type: rubric
          levels:
            - score: 3
              description: "ok"
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E020\].*name is required"):
            load_plan(plan_file)

    def test_rubric_criterion_missing_levels_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: rubric-missing-levels
tasks:
  - id: t1
    engine: claude
    prompt: "Judge this"
    judge:
      criteria:
        - type: rubric
          name: "Correctness"
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E020\].*levels is required"):
            load_plan(plan_file)

    @pytest.mark.parametrize("score", [0, 6])
    def test_rubric_criterion_invalid_score_raises(self, tmp_path: Path, score: int) -> None:
        plan_file = _write_plan(
            tmp_path,
            f"""\
version: 1
name: rubric-invalid-score
tasks:
  - id: t1
    engine: claude
    prompt: "Judge this"
    judge:
      criteria:
        - type: rubric
          name: "Correctness"
          levels:
            - score: {score}
              description: "bad"
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E020\].*score must be an integer 1-5"):
            load_plan(plan_file)
