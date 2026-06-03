from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from maestro_cli.models import PlanSpec, PlanSuggestions, Suggestion, TaskSpec
from maestro_cli.suggest import (
    _FAILURE_REMEDIATION,
    _analyze_task,
    _downgrade_model,
    _load_run_history,
    _median,
    _safe_float,
    _safe_int,
    _upgrade_model,
    format_suggestions,
    format_suggestions_json,
    suggest_plan,
)


def _write_manifest(run_dir: Path, manifest: dict) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def sample_manifest() -> dict:
    return {
        "plan_name": "test-plan",
        "tasks": {
            "task-1": {
                "status": "success",
                "cost_usd": 0.50,
                "duration_sec": 10.0,
                "retry_count": 0,
                "token_usage": {"total_tokens": 1000},
            }
        },
    }


class TestLoadRunHistory:
    def test_no_runs_found(self, tmp_path: Path) -> None:
        runs = _load_run_history("test-plan", tmp_path, min_runs=1)
        assert runs == []

    def test_runs_below_min(self, tmp_path: Path) -> None:
        manifest = {
            "plan_name": "test-plan",
            "task_results": {
                "task-1": {
                    "status": "success",
                    "cost_usd": 0.50,
                    "duration_sec": 10.0,
                    "retry_count": 0,
                    "token_usage": {"total_tokens": 1000},
                }
            },
        }
        _write_manifest(tmp_path / "20260305_120000_test-plan", manifest)

        runs = _load_run_history("test-plan", tmp_path, min_runs=2)
        assert runs == []

    def test_loads_matching_runs(self, tmp_path: Path) -> None:
        manifest = {
            "plan_name": "test-plan",
            "task_results": {
                "task-1": {
                    "status": "success",
                    "cost_usd": 0.50,
                    "duration_sec": 10.0,
                    "retry_count": 0,
                    "token_usage": {"total_tokens": 1000},
                }
            },
        }
        _write_manifest(tmp_path / "20260305_120000_test-plan", manifest)
        _write_manifest(tmp_path / "20260305_120100_test-plan", manifest)
        _write_manifest(tmp_path / "20260305_120200_other-plan", manifest)

        runs = _load_run_history("test-plan", tmp_path, min_runs=2)
        assert len(runs) == 2
        assert all(run["plan_name"] == "test-plan" for run in runs)


class TestAnalyzeTask:
    def test_always_passes_suggests_downgrade(self) -> None:
        task_spec_map = {"task-1": TaskSpec(id="task-1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "task-1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 10.0,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for _ in range(3)
        ]

        suggestions = _analyze_task("task-1", task_spec_map, runs)
        assert any(s.category == "downgrade_model" and "passes in" in s.reason for s in suggestions)

    def test_high_retry_rate_suggests_upgrade(self) -> None:
        task_spec_map = {"task-1": TaskSpec(id="task-1", model="haiku")}
        runs = [
            {
                "task_results": {
                    "task-1": {
                        "status": "success",
                        "retry_count": 1,
                        "cost_usd": 0.0,
                        "duration_sec": 10.0,
                    }
                },
                "total_cost_usd": 0.0,
            },
            {
                "task_results": {
                    "task-1": {
                        "status": "failed",
                        "retry_count": 1,
                        "cost_usd": 0.0,
                        "duration_sec": 9.0,
                    }
                },
                "total_cost_usd": 0.0,
            },
            {
                "task_results": {
                    "task-1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 11.0,
                    }
                },
                "total_cost_usd": 0.0,
            },
        ]

        suggestions = _analyze_task("task-1", task_spec_map, runs)
        assert any(
            s.category == "upgrade_model" and s.suggested_value == "sonnet"
            for s in suggestions
        )

    def test_no_suggestions_for_healthy_task(self) -> None:
        task_spec_map = {"task-1": TaskSpec(id="task-1", model="haiku")}
        runs = [
            {
                "task_results": {
                    "task-1": {
                        "status": "success",
                        "retry_count": 0,
                        "duration_sec": 10.0,
                        "cost_usd": 0.0,
                    }
                }
            },
            {
                "task_results": {
                    "task-1": {
                        "status": "failed",
                        "retry_count": 1,
                        "duration_sec": 9.0,
                        "cost_usd": 0.0,
                    }
                }
            },
            {
                "task_results": {
                    "task-1": {
                        "status": "success",
                        "retry_count": 0,
                        "duration_sec": 11.0,
                        "cost_usd": 0.0,
                    }
                }
            },
        ]

        suggestions = _analyze_task("task-1", task_spec_map, runs)
        assert suggestions == []


class TestSuggestPlan:
    def test_empty_history(self, tmp_path: Path) -> None:
        plan = PlanSpec(version=1, name="test-plan", tasks=[TaskSpec(id="task-1", model="sonnet")])

        result = suggest_plan(plan, tmp_path)

        assert isinstance(result, PlanSuggestions)
        assert result.plan_name == "test-plan"
        assert result.runs_analyzed == 0
        assert result.suggestions == []

    def test_with_run_history(self, tmp_path: Path) -> None:
        plan = PlanSpec(version=1, name="test-plan", tasks=[TaskSpec(id="task-1", model="sonnet")])
        manifest = {
            "plan_name": "test-plan",
            "task_results": {
                "task-1": {
                    "status": "success",
                    "cost_usd": 0.50,
                    "duration_sec": 10.0,
                    "retry_count": 0,
                    "token_usage": {"total_tokens": 1000},
                }
            },
            "total_cost_usd": 1.0,
        }

        _write_manifest(tmp_path / "20260305_120000_test-plan", manifest)
        _write_manifest(tmp_path / "20260305_120100_test-plan", manifest)
        _write_manifest(tmp_path / "20260305_120200_test-plan", manifest)

        result = suggest_plan(plan, tmp_path)

        assert result.runs_analyzed == 3
        assert len(result.suggestions) > 0
        assert any(s.category == "downgrade_model" for s in result.suggestions)


class TestFormatSuggestions:
    def test_no_suggestions_message(self) -> None:
        result = PlanSuggestions(plan_name="test-plan", runs_analyzed=0, suggestions=[])

        output = format_suggestions(result)

        assert "No suggestions" in output

    def test_formatted_output_structure(self) -> None:
        result = PlanSuggestions(
            plan_name="test-plan",
            runs_analyzed=3,
            suggestions=[
                Suggestion(
                    task_id="task-high",
                    category="downgrade_model",
                    severity="high",
                    reason="high reason",
                    current_value="opus",
                    suggested_value="sonnet",
                    confidence=0.95,
                    estimated_savings_pct=15.0,
                ),
                Suggestion(
                    task_id="task-med",
                    category="upgrade_model",
                    severity="medium",
                    reason="medium reason",
                    current_value="haiku",
                    suggested_value="sonnet",
                    confidence=0.75,
                    estimated_savings_pct=8.0,
                ),
                Suggestion(
                    task_id="task-low",
                    category="add_checkpoint",
                    severity="low",
                    reason="low reason",
                    current_value="avg_duration=40s",
                    suggested_value="enable checkpoint",
                    confidence=0.60,
                    estimated_savings_pct=3.0,
                ),
            ],
            total_estimated_savings_pct=26.0,
        )

        output = format_suggestions(result)

        assert "[HIGH]" in output
        assert "[MEDIUM]" in output
        assert "[LOW]" in output

    def test_json_output(self, tmp_path: Path) -> None:
        # Keep tempfile import intentionally exercised while still using tmp_path for file operations.
        assert tempfile.gettempdir()

        result = PlanSuggestions(
            plan_name="test-plan",
            runs_analyzed=2,
            suggestions=[],
            total_estimated_savings_pct=None,
        )

        output = format_suggestions_json(result)
        parsed = json.loads(output)

        assert parsed["plan_name"] == "test-plan"
        assert parsed["runs_analyzed"] == 2
        assert parsed["suggestions"] == []


class TestSafeHelpers:
    @pytest.mark.parametrize("value,expected", [
        (5, 5.0),
        (3.14, 3.14),
        ("abc", 0.0),
        (None, 0.0),
    ])
    def test_safe_float(self, value: object, expected: float) -> None:
        assert _safe_float(value) == expected

    def test_safe_int_with_bool_returns_zero(self) -> None:
        assert _safe_int(True) == 0
        assert _safe_int(False) == 0

    def test_safe_int_with_float_truncates(self) -> None:
        assert _safe_int(3.9) == 3

    @pytest.mark.parametrize("values,expected", [
        ([], 0.0),
        ([7.0], 7.0),
        ([1.0, 3.0, 5.0], 3.0),
        ([1.0, 3.0, 5.0, 7.0], 4.0),
    ])
    def test_median(self, values: list[float], expected: float) -> None:
        assert _median(values) == expected


class TestModelHelpers:
    @pytest.mark.parametrize("model,expected", [
        ("opus", "sonnet"),
        ("sonnet", "haiku"),
        ("haiku", None),
        (None, None),
        ("OPUS", "sonnet"),
    ])
    def test_downgrade_model(self, model: str | None, expected: str | None) -> None:
        assert _downgrade_model(model) == expected

    @pytest.mark.parametrize("model,expected", [
        ("haiku", "sonnet"),
        ("sonnet", "opus"),
        ("opus", None),
        (None, None),
    ])
    def test_upgrade_model(self, model: str | None, expected: str | None) -> None:
        assert _upgrade_model(model) == expected


class TestLoadRunHistoryEdgeCases:
    def test_non_existent_dir(self, tmp_path: Path) -> None:
        result = _load_run_history("plan", tmp_path / "does_not_exist", min_runs=1)
        assert result == []

    def test_invalid_json_skipped(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "20260101_120000_my-plan"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text("not-json{{{", encoding="utf-8")
        result = _load_run_history("my-plan", tmp_path, min_runs=1)
        assert result == []

    def test_manifest_not_a_dict_skipped(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "20260101_120000_my-plan"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        result = _load_run_history("my-plan", tmp_path, min_runs=1)
        assert result == []


class TestAnalyzeTaskEdgeCases:
    def test_empty_runs_returns_no_suggestions(self) -> None:
        assert _analyze_task("t1", {}, []) == []

    def test_task_not_in_any_run_returns_no_suggestions(self) -> None:
        runs = [{"task_results": {"other": {"status": "success"}}}]
        assert _analyze_task("t1", {}, runs) == []

    def test_failure_pattern_at_30_pct_threshold(self) -> None:
        runs = [
            {"task_results": {"t1": {"status": "failed", "failure_category": "output_format_error",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": "output_format_error",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert any(s.category == "fix_failure_pattern" for s in suggestions)

    def test_add_retry_when_no_upgrade_possible(self) -> None:
        """When model is at top tier (opus) and retry rate >50%, suggest add_retry."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="opus", max_retries=0)}
        runs = [
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert any(s.category == "add_retry" for s in suggestions)


class TestFailureRemediation:
    def test_all_known_categories_have_remediation(self) -> None:
        known = {
            "dependency_missing", "output_format_error", "cascading_failure",
            "deadlock", "miscommunication", "role_confusion", "verification_gap",
        }
        assert set(_FAILURE_REMEDIATION.keys()) == known

    def test_all_remediations_non_empty(self) -> None:
        for cat, msg in _FAILURE_REMEDIATION.items():
            assert msg, f"Empty remediation for category '{cat}'"


class TestSafeIntAdditional:
    def test_int_returns_as_is(self) -> None:
        assert _safe_int(7) == 7

    def test_string_returns_zero(self) -> None:
        assert _safe_int("foo") == 0

    def test_none_returns_zero(self) -> None:
        assert _safe_int(None) == 0


class TestAnalyzeTaskJudge:
    def test_judge_all_pass_high_score_suggests_remove(self) -> None:
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"verdict": "pass", "overall_score": 0.95},
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert any(s.category == "remove_judge" for s in suggestions)

    def test_judge_avg_below_09_no_remove(self) -> None:
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"verdict": "pass", "overall_score": 0.85},
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "remove_judge" for s in suggestions)

    def test_judge_verdict_fail_no_remove(self) -> None:
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"verdict": "fail", "overall_score": 0.95},
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "remove_judge" for s in suggestions)


class TestAnalyzeTaskContextAndDuration:
    def test_high_compression_ratio_suggests_reduce_budget(self) -> None:
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "context_compression_ratio": 0.9,
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert any(s.category == "reduce_context_budget" for s in suggestions)

    def test_partial_compression_ratio_no_suggestion(self) -> None:
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "context_compression_ratio": 0.9,
                    }
                },
                "total_cost_usd": 0.0,
            },
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "context_compression_ratio": 0.7,
                    }
                },
                "total_cost_usd": 0.0,
            },
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "reduce_context_budget" for s in suggestions)

    def test_duration_3x_median_suggests_checkpoint(self) -> None:
        # t1=100s; t2..t6=1s each → median of all 6 durations = 1s; 100 > 3*1
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        run = {
            "task_results": {
                "t1": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 100.0},
                "t2": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t3": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t4": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t5": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t6": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
            },
            "total_cost_usd": 0.0,
        }
        suggestions = _analyze_task("t1", task_spec_map, [run])
        assert any(s.category == "add_checkpoint" for s in suggestions)

    def test_high_cost_no_model_suggests_add_review_task(self) -> None:
        # model=None so _downgrade_model returns None → add_review_task category
        task_spec_map = {"t1": TaskSpec(id="t1", model=None)}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.6,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 1.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert any(s.category == "add_review_task" for s in suggestions)


class TestModelHelpersAdditional:
    def test_downgrade_model_empty_string_returns_none(self) -> None:
        assert _downgrade_model("") is None

    def test_upgrade_model_case_insensitive_haiku(self) -> None:
        assert _upgrade_model("HAIKU") == "sonnet"

    def test_upgrade_model_case_insensitive_sonnet(self) -> None:
        assert _upgrade_model("SONNET") == "opus"

    def test_safe_float_bool_treated_as_int(self) -> None:
        # bool is subclass of int; isinstance(True, (int, float)) → True
        assert _safe_float(True) == 1.0

    def test_downgrade_model_unknown_model_returns_none(self) -> None:
        assert _downgrade_model("flash") is None


class TestLoadRunHistoryFileNotDir:
    def test_run_dir_is_a_file_returns_empty(self, tmp_path: Path) -> None:
        file_path = tmp_path / "not_a_dir.txt"
        file_path.write_text("content", encoding="utf-8")
        result = _load_run_history("my-plan", file_path, min_runs=1)
        assert result == []


class TestFailurePatternSeverity:
    def test_high_severity_when_occurrence_at_50_pct(self) -> None:
        """occurrence_rate >= 0.5 → severity 'high'."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "failure_category": "role_confusion",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": "role_confusion",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        fp_suggestions = [s for s in suggestions if s.category == "fix_failure_pattern"]
        assert any(s.severity == "high" for s in fp_suggestions)

    def test_medium_severity_when_occurrence_below_50_pct(self) -> None:
        """occurrence_rate < 0.5 but >= 0.3 → severity 'medium'."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "failure_category": "verification_gap",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            # 3 successes → rate = 1/4 = 0.25 — below threshold, no suggestion
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        # 1/4 = 0.25 < 0.3, so no fix_failure_pattern suggestion
        assert not any(s.category == "fix_failure_pattern" for s in suggestions)

    def test_unknown_failure_category_not_added(self) -> None:
        """failure_category not in _FAILURE_REMEDIATION → ignored."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "failure_category": "unknown_category_xyz",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": "unknown_category_xyz",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "fix_failure_pattern" for s in suggestions)


class TestSuggestPlanSavingsCap:
    def test_total_savings_capped_at_100(self, tmp_path: Path) -> None:
        """Multiple high-savings suggestions should cap at 100%."""
        # Build a plan with many tasks each suggesting 15% savings
        tasks = [TaskSpec(id=f"task-{i}", model="sonnet") for i in range(10)]
        plan = PlanSpec(version=1, name="big-plan", tasks=tasks)
        # Each task always succeeds with 0 retries → downgrade_model (15% savings each)
        # 10 tasks × 15% = 150% > 100%, so should be capped
        task_result = {
            "status": "success",
            "cost_usd": 0.0,
            "duration_sec": 1.0,
            "retry_count": 0,
        }
        manifest = {
            "plan_name": "big-plan",
            "task_results": {f"task-{i}": task_result for i in range(10)},
            "total_cost_usd": 0.5,
        }
        for j in range(3):
            run_dir = tmp_path / f"2026010{j}_120000_big-plan"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        result = suggest_plan(plan, tmp_path)
        assert result.total_estimated_savings_pct is not None
        assert result.total_estimated_savings_pct <= 100.0


class TestFormatSuggestionsAdditional:
    def test_none_total_savings_shows_unknown(self) -> None:
        result = PlanSuggestions(
            plan_name="p",
            runs_analyzed=2,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="downgrade_model",
                    severity="high",
                    reason="passes",
                    current_value="sonnet",
                    suggested_value="haiku",
                    confidence=1.0,
                    estimated_savings_pct=10.0,
                )
            ],
            total_estimated_savings_pct=None,
        )
        output = format_suggestions(result)
        assert "unknown" in output

    def test_none_item_savings_shows_na(self) -> None:
        result = PlanSuggestions(
            plan_name="p",
            runs_analyzed=2,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="downgrade_model",
                    severity="high",
                    reason="passes",
                    current_value="sonnet",
                    suggested_value="haiku",
                    confidence=1.0,
                    estimated_savings_pct=None,
                )
            ],
            total_estimated_savings_pct=None,
        )
        output = format_suggestions(result)
        assert "n/a" in output


class TestAnalyzeTaskHaikuCannotDowngrade:
    def test_haiku_100_success_no_downgrade_suggestion(self) -> None:
        """haiku model → _downgrade_model returns None → no downgrade_model suggestion."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for _ in range(3)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "downgrade_model" for s in suggestions)


class TestAnalyzeTaskHighCostDowngradeable:
    def test_opus_high_cost_suggests_downgrade_not_review(self) -> None:
        """Opus model + cost > 40% of plan → downgrade_model (not add_review_task)."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="opus")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.6,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 1.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        cats = [s.category for s in suggestions]
        assert "downgrade_model" in cats
        assert "add_review_task" not in cats
        downgrade = next(s for s in suggestions if s.category == "downgrade_model")
        assert downgrade.suggested_value == "sonnet"


class TestFormatSuggestionsJsonStructure:
    def test_json_suggestion_items_have_correct_fields(self) -> None:
        result = PlanSuggestions(
            plan_name="p",
            runs_analyzed=3,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="downgrade_model",
                    severity="high",
                    reason="always passes with 0 retries",
                    current_value="sonnet",
                    suggested_value="haiku",
                    confidence=0.9,
                    estimated_savings_pct=15.0,
                )
            ],
            total_estimated_savings_pct=15.0,
        )
        output = format_suggestions_json(result)
        data = json.loads(output)
        assert len(data["suggestions"]) == 1
        item = data["suggestions"][0]
        assert item["task_id"] == "t1"
        assert item["category"] == "downgrade_model"
        assert item["severity"] == "high"
        assert item["suggested_value"] == "haiku"
        assert item["estimated_savings_pct"] == 15.0


class TestSuggestPlanRunsAnalyzedCount:
    def test_runs_analyzed_equals_loaded_run_count(self, tmp_path: Path) -> None:
        plan = PlanSpec(version=1, name="cnt-plan", tasks=[TaskSpec(id="t1", model="sonnet")])
        task_result = {
            "status": "success",
            "cost_usd": 0.0,
            "duration_sec": 1.0,
            "retry_count": 0,
        }
        manifest = {
            "plan_name": "cnt-plan",
            "task_results": {"t1": task_result},
            "total_cost_usd": 0.0,
        }
        for i in range(4):
            run_dir = tmp_path / f"2026030{i}_120000_cnt-plan"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        result = suggest_plan(plan, tmp_path, min_runs=2)
        assert result.runs_analyzed == 4


class TestSafeIntNegative:
    def test_negative_float_truncates_toward_zero(self) -> None:
        assert _safe_int(-3.7) == -3

    def test_negative_int_returned_as_is(self) -> None:
        assert _safe_int(-5) == -5


class TestAnalyzeTaskNoSpec:
    def test_always_succeeds_no_spec_no_downgrade(self) -> None:
        """No entry in task_spec_map → model=None → _downgrade_model(None)=None → no downgrade."""
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for _ in range(3)
        ]
        suggestions = _analyze_task("t1", {}, runs)
        assert not any(s.category == "downgrade_model" for s in suggestions)

    def test_high_retry_no_spec_suggests_add_retry(self) -> None:
        """No entry in task_spec_map → model=None → _upgrade_model(None)=None → add_retry."""
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": 1,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for _ in range(3)
        ]
        suggestions = _analyze_task("t1", {}, runs)
        assert any(s.category == "add_retry" for s in suggestions)


class TestAnalyzeTaskPartialJudge:
    def test_partial_judge_scores_no_remove_suggestion(self) -> None:
        """Only some runs have judge_result; len(scores) != total → remove_judge not suggested."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"verdict": "pass", "overall_score": 0.98},
                    }
                },
                "total_cost_usd": 0.0,
            },
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        # No judge_result — partial coverage
                    }
                },
                "total_cost_usd": 0.0,
            },
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "remove_judge" for s in suggestions)


class TestLoadRunHistorySortOrder:
    def test_runs_returned_in_ascending_directory_name_order(self, tmp_path: Path) -> None:
        """Runs must be sorted alphabetically by dir name (ascending timestamp)."""
        base: dict = {"task_results": {}, "total_cost_usd": 0.0}
        for ts in ["20260303_120000", "20260301_120000", "20260302_120000"]:
            run_dir = tmp_path / f"{ts}_sort-plan"
            run_dir.mkdir()
            manifest = {**base, "run_id": ts}
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        runs = _load_run_history("sort-plan", tmp_path, min_runs=3)
        assert len(runs) == 3
        assert runs[0]["run_id"] == "20260301_120000"
        assert runs[1]["run_id"] == "20260302_120000"
        assert runs[2]["run_id"] == "20260303_120000"


class TestAnalyzeTaskMultipleFailCategories:
    def test_two_distinct_categories_both_above_threshold(self) -> None:
        """Two different failure categories each at >= 0.3 → two fix_failure_pattern suggestions."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "failure_category": "deadlock",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": "deadlock",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": "output_format_error",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": "output_format_error",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        fp = [s for s in suggestions if s.category == "fix_failure_pattern"]
        cats = [s.current_value for s in fp]
        assert any("deadlock" in c for c in cats)
        assert any("output_format_error" in c for c in cats)


class TestAnalyzeTaskZeroPlanCost:
    def test_zero_plan_cost_no_cost_suggestion(self) -> None:
        """All runs have total_cost_usd=0 → plan_avg_cost=0 → no downgrade/review_task from cost path."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.9,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,  # plan cost is zero → ratio skipped
            }
            for _ in range(3)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        # downgrade_model may still appear from the "always passes" check, but NOT from the cost path
        # (which requires plan_avg_cost > 0 to fire). We verify add_review_task is absent.
        assert not any(s.category == "add_review_task" for s in suggestions)


class TestAnalyzeTaskAddRetryIncrements:
    def test_opus_max_retries_already_set_suggests_higher_retry(self) -> None:
        """model=opus (can't upgrade) AND max_retries=2; retry rate >50% → add_retry with max_retries=3."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="opus", max_retries=2)}
        runs = [
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        add_retry = [s for s in suggestions if s.category == "add_retry"]
        assert len(add_retry) == 1
        assert add_retry[0].suggested_value == "max_retries=3"


class TestAnalyzeTaskCompressionPartialRuns:
    def test_missing_ratio_in_some_runs_no_suggestion(self) -> None:
        """Only run 1 has context_compression_ratio; len(ratios)=1 != total=2 → no suggestion."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "context_compression_ratio": 0.95,
                    }
                },
                "total_cost_usd": 0.0,
            },
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        # no context_compression_ratio key
                    }
                },
                "total_cost_usd": 0.0,
            },
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "reduce_context_budget" for s in suggestions)


class TestFormatSuggestionsJsonTotalSavings:
    def test_non_null_savings_appears_in_json(self) -> None:
        result = PlanSuggestions(
            plan_name="plan-x",
            runs_analyzed=5,
            suggestions=[],
            total_estimated_savings_pct=42.5,
        )
        output = format_suggestions_json(result)
        data = json.loads(output)
        assert data["total_estimated_savings_pct"] == 42.5
        assert data["runs_analyzed"] == 5
        assert data["plan_name"] == "plan-x"


class TestAnalyzeTaskCascadingFailure:
    def test_cascading_failure_category_above_threshold(self) -> None:
        """failure_category='cascading_failure' in 2/3 runs (0.67 >= 0.3) → fix_failure_pattern."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "failure_category": "cascading_failure",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": "cascading_failure",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        fp = [s for s in suggestions if s.category == "fix_failure_pattern"]
        assert len(fp) >= 1
        assert any("cascading_failure" in s.current_value for s in fp)


class TestAnalyzeTaskAllSucceedWithRetries:
    def test_all_succeed_with_retries_no_downgrade_from_success_path(self) -> None:
        """All runs succeed but retry_count > 0 → (retry_runs == 0) fails → no downgrade_model
        from the 'always passes' path (retry_rate check may still fire)."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        # All succeed but all have retries — success_count==total but retry_runs==total
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 1,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for _ in range(3)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        # The "always passes with 0 retries" downgrade suggestion must NOT appear
        assert not any(
            s.category == "downgrade_model" and "passes in" in s.reason
            for s in suggestions
        )


class TestLoadRunHistoryNonDirEntry:
    def test_file_matching_glob_is_skipped(self, tmp_path: Path) -> None:
        """A file (not a directory) that matches the plan glob pattern is ignored."""
        # Create one valid run directory
        valid_run = tmp_path / "20260101_120000_skip-plan"
        valid_run.mkdir()
        manifest: dict = {"task_results": {}, "total_cost_usd": 0.0}
        (valid_run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        # Create a FILE also matching the glob (same prefix pattern)
        fake_file = tmp_path / "20260101_120100_skip-plan"
        fake_file.write_text("not a directory", encoding="utf-8")

        runs = _load_run_history("skip-plan", tmp_path, min_runs=1)
        assert len(runs) == 1


class TestMedianEdgeCases:
    def test_negative_values_sorted_correctly(self) -> None:
        # sorted: [-3.0, -2.0, -1.0], mid=1 → -2.0
        assert _median([-3.0, -1.0, -2.0]) == -2.0

    def test_two_equal_values(self) -> None:
        assert _median([3.0, 3.0]) == 3.0


class TestAnalyzeTaskJudgeErrorVerdict:
    def test_error_verdict_no_remove_judge_suggestion(self) -> None:
        """Judge verdict 'error' → not all 'pass' → remove_judge not suggested."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"verdict": "error", "overall_score": 0.98},
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "remove_judge" for s in suggestions)


class TestFormatSuggestionsRunsAnalyzedInHeader:
    def test_runs_analyzed_and_plan_name_in_header(self) -> None:
        result = PlanSuggestions(
            plan_name="my-plan",
            runs_analyzed=7,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="add_retry",
                    severity="medium",
                    reason="retries happen often",
                    current_value="max_retries=0",
                    suggested_value="max_retries=1",
                    confidence=0.6,
                    estimated_savings_pct=8.0,
                )
            ],
            total_estimated_savings_pct=8.0,
        )
        output = format_suggestions(result)
        assert "7" in output
        assert "my-plan" in output


class TestSafeFloatBoolFalse:
    def test_false_returns_zero_float(self) -> None:
        """bool is subclass of int; float(False) == 0.0."""
        assert _safe_float(False) == 0.0


class TestAnalyzeTaskHaikuHighCost:
    def test_haiku_high_cost_suggests_add_review_task(self) -> None:
        """haiku model → _downgrade_model('haiku') is None → add_review_task instead of downgrade."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.6,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 1.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert any(s.category == "add_review_task" for s in suggestions)
        assert not any(s.category == "downgrade_model" for s in suggestions)


class TestAnalyzeTaskAddRetryWithExistingRetries:
    def test_add_retry_suggested_value_increments_existing_max_retries(self) -> None:
        """opus with max_retries=2 and >50% retry rate → add_retry suggestion is max_retries=3."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="opus", max_retries=2)}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": 1,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            },
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": 1,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            },
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            },
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        retry_s = next((s for s in suggestions if s.category == "add_retry"), None)
        assert retry_s is not None
        assert retry_s.suggested_value == "max_retries=3"


class TestAnalyzeTaskZeroMedianDuration:
    def test_zero_median_no_checkpoint_suggestion(self) -> None:
        """All tasks have duration_sec=0.0 → median=0 → checkpoint condition skipped."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 0.0,
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "add_checkpoint" for s in suggestions)


class TestAnalyzeTaskContextRatioBoundary:
    def test_context_ratio_exactly_08_no_suggestion(self) -> None:
        """Compression ratio exactly 0.8 does not satisfy > 0.8 → no reduce_context_budget."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "context_compression_ratio": 0.8,
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "reduce_context_budget" for s in suggestions)


class TestAnalyzeTaskAllFailed:
    def test_all_runs_failed_no_downgrade_from_success_path(self) -> None:
        """success_count=0 → the 'always passes' path is never taken; no downgrade_model."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for _ in range(3)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(
            s.category == "downgrade_model" and "passes in" in s.reason
            for s in suggestions
        )


class TestSafeFloatNegative:
    def test_negative_int_returns_negative_float(self) -> None:
        assert _safe_float(-3) == -3.0

    def test_negative_float_returns_as_is(self) -> None:
        assert _safe_float(-2.5) == -2.5


class TestLoadRunHistoryDirWithoutManifest:
    def test_matching_dir_without_manifest_file_is_skipped(self, tmp_path: Path) -> None:
        """Run directory matching glob pattern exists but has no run_manifest.json → skipped."""
        run_dir = tmp_path / "20260101_120000_ghost-plan"
        run_dir.mkdir()
        # No run_manifest.json inside — exercises the `if not manifest_path.is_file(): continue` branch
        result = _load_run_history("ghost-plan", tmp_path, min_runs=1)
        assert result == []

    def test_valid_and_missing_manifest_dirs_counted_correctly(self, tmp_path: Path) -> None:
        """Only directories with a manifest file count toward min_runs."""
        manifest: dict = {"task_results": {}, "total_cost_usd": 0.0}
        # Valid run
        valid = tmp_path / "20260101_120000_combo-plan"
        valid.mkdir()
        (valid / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        # Directory matching glob but no manifest
        empty = tmp_path / "20260101_120100_combo-plan"
        empty.mkdir()
        # With min_runs=2, only 1 valid manifest → returns []
        result = _load_run_history("combo-plan", tmp_path, min_runs=2)
        assert result == []


class TestAnalyzeTaskNonDictTaskResults:
    def test_non_dict_task_results_runs_are_skipped(self) -> None:
        """Runs where task_results is not a dict are skipped; only valid runs contribute."""
        runs = [
            {"task_results": None},
            {"task_results": ["a", "b"]},
            {
                "task_results": {
                    "t1": {"status": "success", "retry_count": 0,
                            "cost_usd": 0.0, "duration_sec": 1.0}
                },
                "total_cost_usd": 0.0,
            },
            {
                "task_results": {
                    "t1": {"status": "success", "retry_count": 0,
                            "cost_usd": 0.0, "duration_sec": 1.0}
                },
                "total_cost_usd": 0.0,
            },
        ]
        # haiku can't be downgraded → no downgrade suggestion even from 2 valid success runs
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "downgrade_model" for s in suggestions)


class TestAnalyzeTaskNonDictTaskPayload:
    def test_non_dict_payload_values_yield_empty_task_runs(self) -> None:
        """task_results has the task id but value is not a dict → task_runs is empty → no suggestions."""
        runs = [
            {"task_results": {"t1": None}},
            {"task_results": {"t1": "some_string"}},
            {"task_results": {"t1": 42}},
        ]
        suggestions = _analyze_task("t1", {"t1": TaskSpec(id="t1", model="sonnet")}, runs)
        assert suggestions == []


class TestSuggestPlanNoSuggestionsNullSavings:
    def test_runs_exist_but_no_suggestions_gives_null_savings(self, tmp_path: Path) -> None:
        """When runs load successfully but generate zero suggestions, total savings is None."""
        plan = PlanSpec(version=1, name="frugal-plan", tasks=[
            TaskSpec(id="t1", model="haiku"),  # haiku can't be downgraded
        ])
        manifest = {
            "plan_name": "frugal-plan",
            "task_results": {
                "t1": {
                    "status": "success",
                    "cost_usd": 0.0,
                    "duration_sec": 1.0,
                    "retry_count": 0,
                }
            },
            "total_cost_usd": 0.0,
        }
        for i in range(3):
            run_dir = tmp_path / f"2026010{i}_120000_frugal-plan"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        result = suggest_plan(plan, tmp_path)
        assert result.runs_analyzed == 3
        assert result.suggestions == []
        assert result.total_estimated_savings_pct is None


class TestSafeFloatNonNumericTypes:
    def test_list_returns_zero(self) -> None:
        assert _safe_float([1, 2, 3]) == 0.0

    def test_dict_returns_zero(self) -> None:
        assert _safe_float({"key": "val"}) == 0.0

    def test_zero_int_returns_zero_float(self) -> None:
        assert _safe_float(0) == 0.0


class TestSafeIntNonNumericTypes:
    def test_list_returns_zero(self) -> None:
        assert _safe_int([1, 2]) == 0

    def test_dict_returns_zero(self) -> None:
        assert _safe_int({"x": 1}) == 0

    def test_float_zero_returns_int_zero(self) -> None:
        assert _safe_int(0.0) == 0


class TestAnalyzeTaskJudgeNonDictSkipped:
    def test_judge_result_string_is_skipped(self) -> None:
        """When judge_result is a string instead of dict, it should be ignored."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": "not-a-dict",
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "remove_judge" for s in suggestions)

    def test_judge_result_none_is_skipped(self) -> None:
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": None,
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "remove_judge" for s in suggestions)


class TestAnalyzeTaskFailureCategoryEdgeCases:
    def test_non_string_failure_category_ignored(self) -> None:
        """failure_category=42 → not a string → not counted toward fix_failure_pattern."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "failure_category": 42,
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": 42,
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "fix_failure_pattern" for s in suggestions)

    def test_failure_category_missing_key_ignored(self) -> None:
        """Runs with no failure_category key at all → no fix_failure_pattern."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {"task_results": {"t1": {"status": "failed",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0}
            for _ in range(4)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "fix_failure_pattern" for s in suggestions)


class TestSafeFloatStringInput:
    def test_string_returns_zero(self) -> None:
        assert _safe_float("3.14") == 0.0

    def test_empty_string_returns_zero(self) -> None:
        assert _safe_float("") == 0.0


class TestAnalyzeTaskRetryRateBoundary:
    def test_retry_rate_exactly_50pct_no_upgrade_or_retry(self) -> None:
        """retry_rate == 0.5 is NOT > 0.5, so upgrade_model/add_retry not triggered."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category in ("upgrade_model", "add_retry") for s in suggestions)


class TestFormatSuggestionsReasonText:
    def test_category_name_in_reason_line(self) -> None:
        """format_suggestions includes [category] in the Reason: line."""
        result = PlanSuggestions(
            plan_name="p",
            runs_analyzed=2,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="add_retry",
                    severity="medium",
                    reason="retries too frequent",
                    current_value="max_retries=0",
                    suggested_value="max_retries=1",
                    confidence=0.7,
                    estimated_savings_pct=8.0,
                )
            ],
            total_estimated_savings_pct=8.0,
        )
        output = format_suggestions(result)
        assert "[add_retry]" in output
        assert "retries too frequent" in output


class TestFormatSuggestionsJsonNullSavings:
    def test_null_total_savings_appears_in_json(self) -> None:
        result = PlanSuggestions(
            plan_name="myplan",
            runs_analyzed=3,
            suggestions=[],
            total_estimated_savings_pct=None,
        )
        output = format_suggestions_json(result)
        data = json.loads(output)
        assert data["total_estimated_savings_pct"] is None
        assert data["plan_name"] == "myplan"
        assert data["runs_analyzed"] == 3


class TestAnalyzeTaskZeroMedianDuration:
    def test_zero_median_no_checkpoint_suggestion(self) -> None:
        """When all durations are 0 (missing), median_duration==0, no add_checkpoint."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        # no duration_sec → _safe_float returns 0.0
                    }
                },
                "total_cost_usd": 0.0,
            }
            for _ in range(3)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "add_checkpoint" for s in suggestions)


class TestAnalyzeTaskBoundaryConditions:
    def test_judge_avg_exactly_09_no_remove_suggestion(self) -> None:
        """avg_score == 0.9 is not > 0.9, so remove_judge should not be suggested."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {"task_results": {"t1": {
                "status": "success",
                "retry_count": 0,
                "judge_result": {"overall_score": 0.9, "verdict": "pass"},
            }}},
            {"task_results": {"t1": {
                "status": "success",
                "retry_count": 0,
                "judge_result": {"overall_score": 0.9, "verdict": "pass"},
            }}},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "remove_judge" for s in suggestions)

    def test_compression_ratio_exactly_08_no_reduce_suggestion(self) -> None:
        """ratio == 0.8 is not > 0.8, so reduce_context_budget should not be suggested."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {"task_results": {"t1": {
                "status": "success",
                "retry_count": 0,
                "context_compression_ratio": 0.8,
            }}},
            {"task_results": {"t1": {
                "status": "success",
                "retry_count": 0,
                "context_compression_ratio": 0.8,
            }}},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "reduce_context_budget" for s in suggestions)


class TestAnalyzeTaskAddRetryIncrementFromNonZero:
    def test_add_retry_increments_max_retries_from_nonzero(self) -> None:
        """When model=None (can't upgrade) and max_retries=2, retry >50% → max_retries=3."""
        task_spec_map = {"t1": TaskSpec(id="t1", model=None, max_retries=2)}
        runs = [
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        retry_s = next((s for s in suggestions if s.category == "add_retry"), None)
        assert retry_s is not None
        assert retry_s.suggested_value == "max_retries=3"


class TestAnalyzeTaskJudgeScoreWithoutVerdict:
    def test_judge_score_but_no_verdict_key_no_remove(self) -> None:
        """judge_result has overall_score but no verdict key → verdicts list empty → no remove_judge."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"overall_score": 0.95},  # no "verdict" key
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "remove_judge" for s in suggestions)


class TestSuggestPlanDefaultMinRuns:
    def test_two_run_dirs_below_default_min_returns_empty(self, tmp_path: Path) -> None:
        """Default min_runs=3; providing only 2 matching dirs → runs_analyzed=0."""
        plan = PlanSpec(version=1, name="slim-plan", tasks=[TaskSpec(id="t1", model="sonnet")])
        manifest = {
            "plan_name": "slim-plan",
            "task_results": {"t1": {"status": "success", "cost_usd": 0.0,
                                    "duration_sec": 1.0, "retry_count": 0}},
            "total_cost_usd": 0.0,
        }
        for i in range(2):
            run_dir = tmp_path / f"2026010{i}_120000_slim-plan"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        # No explicit min_runs → default is 3; only 2 runs available → empty
        result = suggest_plan(plan, tmp_path)
        assert result.runs_analyzed == 0
        assert result.suggestions == []


class TestSafeFloatBoolSubclass:
    def test_bool_true_returns_one(self) -> None:
        """bool is a subclass of int; isinstance(True, (int, float)) is True → float(True) = 1.0."""
        assert _safe_float(True) == 1.0

    def test_bool_false_returns_zero(self) -> None:
        assert _safe_float(False) == 0.0


class TestModelHelpersFullNames:
    def test_downgrade_full_opus_name(self) -> None:
        """Full model name 'claude-opus-4.6' contains 'opus' → downgrades to 'sonnet'."""
        assert _downgrade_model("claude-opus-4.6") == "sonnet"

    def test_downgrade_full_sonnet_name(self) -> None:
        assert _downgrade_model("claude-sonnet-4.5-20251001") == "haiku"

    def test_upgrade_full_sonnet_name(self) -> None:
        """Full model name 'claude-sonnet-4.5' contains 'sonnet' → upgrades to 'opus'."""
        assert _upgrade_model("claude-sonnet-4.5") == "opus"

    def test_upgrade_full_opus_name_returns_none(self) -> None:
        """Full model name 'claude-opus-4.6' is already at the top tier → None."""
        assert _upgrade_model("claude-opus-4.6") is None


class TestSafeIntNone:
    def test_none_returns_zero(self) -> None:
        """None is not bool/int/float → falls through to return 0."""
        assert _safe_int(None) == 0


class TestFormatSuggestionsZeroSavings:
    def test_zero_estimated_savings_shows_percentage_not_na(self) -> None:
        """estimated_savings_pct=0.0 is not None → shows '0.0%' not 'n/a'."""
        result = PlanSuggestions(
            plan_name="p",
            runs_analyzed=3,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="add_retry",
                    severity="medium",
                    reason="retries too frequent",
                    current_value="max_retries=0",
                    suggested_value="max_retries=1",
                    confidence=0.7,
                    estimated_savings_pct=0.0,
                )
            ],
            total_estimated_savings_pct=0.0,
        )
        output = format_suggestions(result)
        assert "0.0%" in output
        assert "n/a" not in output


class TestUpgradeModelUppercaseOpus:
    def test_opus_uppercase_returns_none(self) -> None:
        """'OPUS' in lower is 'opus'; no tier above opus → None."""
        assert _upgrade_model("OPUS") is None

    def test_sonnet_uppercase_returns_opus(self) -> None:
        """'SONNET' lowercased contains 'sonnet' → upgrade to 'opus'."""
        assert _upgrade_model("SONNET") == "opus"


class TestAnalyzeTaskAddRetryZeroMaxRetriesSuggestsOne:
    def test_suggested_value_is_max_retries_1_when_zero(self) -> None:
        """opus (can't upgrade) + max_retries=0 + retry_rate>0.5 → add_retry with max_retries=1."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="opus", max_retries=0)}
        runs = [
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        add_retry = [s for s in suggestions if s.category == "add_retry"]
        assert len(add_retry) == 1
        # max(0 + 1, 1) = 1
        assert add_retry[0].suggested_value == "max_retries=1"
        assert add_retry[0].current_value == "max_retries=0"


class TestAnalyzeTaskUpgradeModelConfidence:
    def test_confidence_equals_retry_rate(self) -> None:
        """upgrade_model suggestion confidence equals retry_rate = 2/3."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "retry_count": 1,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        upgrade = next((s for s in suggestions if s.category == "upgrade_model"), None)
        assert upgrade is not None
        # 2 out of 3 runs had retries → retry_rate = 2/3
        assert abs(upgrade.confidence - 2 / 3) < 1e-9
        assert upgrade.current_value == "haiku"
        assert upgrade.suggested_value == "sonnet"


class TestDowngradeHaikuFullName:
    def test_haiku_full_name_returns_none(self) -> None:
        """Full 'claude-haiku-4.5' contains neither 'opus' nor 'sonnet' → None."""
        assert _downgrade_model("claude-haiku-4.5") is None

    def test_upgrade_haiku_full_name_returns_sonnet(self) -> None:
        """Full 'claude-haiku-4.5' contains 'haiku' → upgrades to 'sonnet'."""
        assert _upgrade_model("claude-haiku-4.5") == "sonnet"


class TestSafeFloatTuple:
    def test_tuple_returns_zero(self) -> None:
        """Tuples are not int or float → returns 0.0."""
        assert _safe_float((1.0, 2.0)) == 0.0


class TestAnalyzeTaskJudgeScoreNotNumeric:
    def test_judge_score_string_value_not_appended_to_scores(self) -> None:
        """judge_result.overall_score is a non-numeric string → not appended → len(scores) != total → no remove_judge."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"verdict": "pass", "overall_score": "high"},
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "remove_judge" for s in suggestions)


class TestFormatSuggestionsArrowSeparator:
    def test_arrow_between_current_and_suggested_value(self) -> None:
        """format_suggestions shows 'current_value → suggested_value' in each item line."""
        result = PlanSuggestions(
            plan_name="arrow-plan",
            runs_analyzed=3,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="downgrade_model",
                    severity="high",
                    reason="always passes",
                    current_value="opus",
                    suggested_value="sonnet",
                    confidence=1.0,
                    estimated_savings_pct=15.0,
                )
            ],
            total_estimated_savings_pct=15.0,
        )
        output = format_suggestions(result)
        assert "opus" in output
        assert "sonnet" in output
        assert "→" in output


class TestSuggestPlanEmptyTasks:
    def test_plan_with_no_tasks_returns_empty_suggestions(self, tmp_path: Path) -> None:
        """Plan with zero tasks → no suggestions even with sufficient run history."""
        plan = PlanSpec(version=1, name="empty-tasks-plan", tasks=[])
        manifest = {
            "plan_name": "empty-tasks-plan",
            "task_results": {},
            "total_cost_usd": 0.0,
        }
        for i in range(3):
            run_dir = tmp_path / f"2026010{i}_120000_empty-tasks-plan"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        result = suggest_plan(plan, tmp_path)
        assert result.runs_analyzed == 3
        assert result.suggestions == []
        assert result.total_estimated_savings_pct is None


class TestLoadRunHistoryExactMinRunsMatch:
    def test_exactly_min_runs_returns_data(self, tmp_path: Path) -> None:
        """Providing exactly min_runs matching directories → returns them (not empty)."""
        manifest: dict = {"task_results": {}, "total_cost_usd": 0.0}
        for i in range(2):
            run_dir = tmp_path / f"2026010{i}_120000_exact-plan"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        runs = _load_run_history("exact-plan", tmp_path, min_runs=2)
        assert len(runs) == 2

    def test_one_below_min_returns_empty(self, tmp_path: Path) -> None:
        """Providing min_runs - 1 matching directories → returns []."""
        manifest: dict = {"task_results": {}, "total_cost_usd": 0.0}
        run_dir = tmp_path / "20260101_120000_short-plan"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        runs = _load_run_history("short-plan", tmp_path, min_runs=2)
        assert runs == []


class TestAnalyzeTaskCostRatioBoundary:
    def test_cost_ratio_exactly_40_pct_no_cost_suggestion(self) -> None:
        """task_avg_cost / plan_avg_cost == 0.4 is NOT > 0.4 → no cost-ratio suggestion."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.4,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 1.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        # The cost-ratio path says "consumes about X% of total plan cost"
        cost_ratio_suggestions = [
            s for s in suggestions
            if "consumes about" in s.reason
        ]
        assert cost_ratio_suggestions == []


class TestAnalyzeTaskMultipleSuggestionsFromSameTask:
    def test_sonnet_all_succeed_and_high_cost_two_downgrade_suggestions(self) -> None:
        """sonnet model, 100% success rate, AND cost > 40% of plan → two downgrade_model suggestions."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.6,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 1.0,
            }
            for _ in range(3)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        downgrades = [s for s in suggestions if s.category == "downgrade_model"]
        # One from the "always passes" path, one from the cost-ratio path
        assert len(downgrades) == 2
        reasons = {s.reason for s in downgrades}
        assert any("passes in" in r for r in reasons)
        assert any("cost" in r.lower() for r in reasons)


class TestAnalyzeTaskCheckpointConfidenceCapped:
    def test_checkpoint_confidence_capped_at_one(self) -> None:
        """When task_avg_duration is far above 3*median, confidence = min(ratio, 1.0) = 1.0."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        # t1=1000s duration; all other tasks=1s → median≈1s; 1000/(3*1)>>1 → capped
        run = {
            "task_results": {
                "t1": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1000.0},
                "t2": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t3": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
            },
            "total_cost_usd": 0.0,
        }
        suggestions = _analyze_task("t1", task_spec_map, [run])
        cp = next((s for s in suggestions if s.category == "add_checkpoint"), None)
        assert cp is not None
        assert cp.confidence == 1.0


class TestFormatSuggestionsItemCountInOutput:
    def test_items_count_shown_in_header(self) -> None:
        """format_suggestions includes 'N items' in the summary line."""
        result = PlanSuggestions(
            plan_name="cnt-plan",
            runs_analyzed=5,
            suggestions=[
                Suggestion(
                    task_id=f"t{i}",
                    category="add_retry",
                    severity="medium",
                    reason="retries",
                    current_value="max_retries=0",
                    suggested_value="max_retries=1",
                    confidence=0.6,
                    estimated_savings_pct=8.0,
                )
                for i in range(3)
            ],
            total_estimated_savings_pct=24.0,
        )
        output = format_suggestions(result)
        assert "3 items" in output


class TestLoadRunHistoryListManifest:
    def test_json_list_manifest_is_skipped(self, tmp_path: Path) -> None:
        """Manifest JSON that is a list (not a dict) → isinstance(payload, dict) fails → skipped."""
        run_dir = tmp_path / "20260101_120000_list-plan"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text("[1, 2, 3]", encoding="utf-8")
        result = _load_run_history("list-plan", tmp_path, min_runs=1)
        assert result == []

    def test_json_null_manifest_is_skipped(self, tmp_path: Path) -> None:
        """Manifest JSON that is null → isinstance(null, dict) fails → skipped."""
        run_dir = tmp_path / "20260101_120000_null-plan"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text("null", encoding="utf-8")
        result = _load_run_history("null-plan", tmp_path, min_runs=1)
        assert result == []


class TestFormatSuggestionsPlanNameInHeader:
    def test_plan_name_in_analyzed_header(self) -> None:
        """The 'Analyzed N runs of "plan_name"' header includes the actual plan name."""
        result = PlanSuggestions(
            plan_name="my-special-plan",
            runs_analyzed=7,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="downgrade_model",
                    severity="high",
                    reason="always passes",
                    current_value="opus",
                    suggested_value="sonnet",
                    confidence=1.0,
                    estimated_savings_pct=15.0,
                )
            ],
            total_estimated_savings_pct=15.0,
        )
        output = format_suggestions(result)
        assert "my-special-plan" in output
        assert "7" in output

    def test_runs_analyzed_count_in_header(self) -> None:
        """Analyzed header shows the exact runs_analyzed count."""
        result = PlanSuggestions(
            plan_name="p",
            runs_analyzed=12,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="add_retry",
                    severity="medium",
                    reason="x",
                    current_value="max_retries=0",
                    suggested_value="max_retries=1",
                    confidence=0.6,
                    estimated_savings_pct=8.0,
                )
            ],
            total_estimated_savings_pct=8.0,
        )
        output = format_suggestions(result)
        assert "12" in output


class TestAnalyzeTaskContextBudgetConfidence:
    def test_confidence_equals_min_compression_ratio(self) -> None:
        """reduce_context_budget confidence = min(ratios)."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0,
                                     "context_compression_ratio": 0.85}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0,
                                     "context_compression_ratio": 0.9}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        rbc = next((s for s in suggestions if s.category == "reduce_context_budget"), None)
        assert rbc is not None
        assert abs(rbc.confidence - 0.85) < 1e-9  # min(0.85, 0.9) = 0.85


class TestAnalyzeTaskFailureCategoryUnknownString:
    def test_unknown_string_category_not_in_remediation_ignored(self) -> None:
        """failure_category is a valid string but not a key in _FAILURE_REMEDIATION
        (e.g. 'timeout_error') → filtered by `cat in _FAILURE_REMEDIATION` → no fix_failure_pattern."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        cat = "timeout_error"  # valid string, but NOT in _FAILURE_REMEDIATION
        runs = [
            {"task_results": {"t1": {"status": "failed", "failure_category": cat,
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": cat,
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": cat,
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "fix_failure_pattern" for s in suggestions)

    def test_unknown_string_category_mixed_with_known_ignores_unknown(self) -> None:
        """Mix of known and unknown failure categories: only known one triggers suggestion."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "failure_category": "network_error",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": "output_format_error",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": "output_format_error",
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        fix = [s for s in suggestions if s.category == "fix_failure_pattern"]
        # output_format_error appears 2/3 = 67% → should trigger; network_error should not
        assert len(fix) == 1
        assert fix[0].current_value == "failure_category=output_format_error"


class TestModelHelpersEmptyString:
    def test_downgrade_empty_string_returns_none(self) -> None:
        """Empty string is falsy: `if not model` fires → returns None."""
        assert _downgrade_model("") is None

    def test_upgrade_empty_string_returns_none(self) -> None:
        """Empty string is falsy: `if not model` fires → returns None."""
        assert _upgrade_model("") is None


class TestFormatSuggestionsLowSeverity:
    def test_low_severity_shows_low_label(self) -> None:
        """A suggestion with severity='low' should render '[LOW]' in the output."""
        result = PlanSuggestions(
            plan_name="p",
            runs_analyzed=3,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="remove_judge",
                    severity="low",
                    reason="judge always passes at high score",
                    current_value="judge=enabled",
                    suggested_value="judge=disabled",
                    confidence=0.95,
                    estimated_savings_pct=5.0,
                )
            ],
            total_estimated_savings_pct=5.0,
        )
        output = format_suggestions(result)
        assert "[LOW]" in output
        assert "remove_judge" in output


class TestAnalyzeTaskFailurePatternRemediation:
    def test_fix_failure_pattern_suggested_value_matches_remediation(self) -> None:
        """fix_failure_pattern.suggested_value == _FAILURE_REMEDIATION[failure_cat]."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        cat = "output_format_error"
        runs = [
            {"task_results": {"t1": {"status": "failed", "failure_category": cat,
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "failed", "failure_category": cat,
                                     "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
            {"task_results": {"t1": {"status": "success", "retry_count": 0,
                                     "cost_usd": 0.0, "duration_sec": 1.0}},
             "total_cost_usd": 0.0},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        fp = next((s for s in suggestions if s.category == "fix_failure_pattern"), None)
        assert fp is not None
        assert fp.suggested_value == _FAILURE_REMEDIATION[cat]
        assert fp.current_value == f"failure_category={cat}"

    def test_all_failure_remediation_keys_present(self) -> None:
        """_FAILURE_REMEDIATION contains all 7 expected failure category keys."""
        expected_keys = {
            "dependency_missing",
            "output_format_error",
            "cascading_failure",
            "deadlock",
            "miscommunication",
            "role_confusion",
            "verification_gap",
        }
        assert set(_FAILURE_REMEDIATION.keys()) == expected_keys

    def test_fix_failure_pattern_occurrence_exactly_30pct(self) -> None:
        """occurrence_rate exactly 30% (= 0.3) satisfies >= 0.3 → fix_failure_pattern suggested."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        cat = "deadlock"
        # 3 out of 10 = 30%
        runs = (
            [{"task_results": {"t1": {"status": "failed", "failure_category": cat,
                                      "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0}},
              "total_cost_usd": 0.0}] * 3
            + [{"task_results": {"t1": {"status": "success", "retry_count": 0,
                                        "cost_usd": 0.0, "duration_sec": 1.0}},
                "total_cost_usd": 0.0}] * 7
        )
        suggestions = _analyze_task("t1", task_spec_map, runs)
        fp = next((s for s in suggestions if s.category == "fix_failure_pattern"), None)
        assert fp is not None
        assert fp.severity == "medium"  # 0.3 < 0.5 → medium


class TestAnalyzeTaskNonDictTaskResults:
    def test_run_with_task_results_as_list_is_skipped(self) -> None:
        """Run where task_results is a list (not dict) → skipped via isinstance guard."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {"task_results": [{"status": "success"}]},  # list, not dict
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert suggestions == []

    def test_run_with_task_results_as_string_is_skipped(self) -> None:
        """Run where task_results is a string (not dict) → skipped via isinstance guard."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {"task_results": "not-a-dict"},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert suggestions == []

    def test_task_payload_not_dict_is_skipped(self) -> None:
        """Task payload is a string instead of dict → not appended to task_runs."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {"task_results": {"t1": "just-a-string"}},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert suggestions == []


class TestSafeFloatAdditional:
    def test_list_input_returns_zero(self) -> None:
        """list is not int/float → returns 0.0."""
        assert _safe_float([1, 2, 3]) == 0.0

    def test_dict_input_returns_zero(self) -> None:
        assert _safe_float({"key": "val"}) == 0.0

    def test_negative_float_preserved(self) -> None:
        assert _safe_float(-3.14) == -3.14


class TestMedianTwoElements:
    def test_two_element_list_returns_average(self) -> None:
        assert _median([2.0, 8.0]) == 5.0


class TestAnalyzeTaskMissingPayloadFields:
    def test_retry_count_absent_treated_as_zero(self) -> None:
        """Task payload with no retry_count key → _safe_int(None) → 0 → not counted as retry."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        # no retry_count key at all
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for _ in range(3)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        # All success + retry_count treated as 0 → downgrade_model suggestion
        assert any(s.category == "downgrade_model" and "passes in" in s.reason for s in suggestions)
        # retry_rate = 0 → no upgrade/add_retry
        assert not any(s.category in ("upgrade_model", "add_retry") for s in suggestions)

    def test_cost_usd_absent_treated_as_zero(self) -> None:
        """Task payload missing cost_usd key → _safe_float(None) → 0.0 → cost ratio is 0."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="opus")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        # no cost_usd key
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 10.0,  # plan cost is 10, task cost is 0 → ratio 0/10 = 0
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        # cost ratio 0/10 = 0.0, NOT > 0.4 → no cost-ratio downgrade/review
        assert not any("consumes about" in s.reason for s in suggestions)

    def test_total_cost_usd_absent_no_cost_ratio_suggestion(self) -> None:
        """Run dict missing total_cost_usd → _safe_float(None) → 0.0 → plan_avg_cost = 0 → cost check skipped."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="opus")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 5.0,
                        "duration_sec": 1.0,
                    }
                },
                # no total_cost_usd key at all
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        # plan_avg_cost = _safe_float(None) = 0.0 → 0 is NOT > 0 → cost-ratio path skipped
        assert not any("consumes about" in s.reason for s in suggestions)


class TestAnalyzeTaskJudgeIntegerScore:
    def test_integer_score_accepted_as_numeric(self) -> None:
        """judge_result.overall_score is int (1) → isinstance(1, (int, float)) → True → appended."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"verdict": "pass", "overall_score": 1},
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        # score=1 > 0.9, all verdicts="pass", len==total → remove_judge
        assert any(s.category == "remove_judge" for s in suggestions)


class TestAnalyzeTaskCompressionRatioIntegerValue:
    def test_integer_ratio_accepted_as_numeric(self) -> None:
        """context_compression_ratio=1 (int) → isinstance(1, (int, float)) → True → float(1) = 1.0 > 0.8."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "context_compression_ratio": 1,  # integer, not float
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert any(s.category == "reduce_context_budget" for s in suggestions)

    def test_two_equal_elements(self) -> None:
        assert _median([4.0, 4.0]) == 4.0


class TestSuggestPlanNoSuggestionsWithHistory:
    def test_runs_present_but_no_suggestions_gives_none_savings(self, tmp_path: Path) -> None:
        """All tasks healthy with haiku (can't downgrade) → no suggestions, savings=None."""
        plan = PlanSpec(
            version=1, name="healthy-plan",
            tasks=[TaskSpec(id="t1", model="haiku")],
        )
        manifest = {
            "plan_name": "healthy-plan",
            "task_results": {
                "t1": {
                    "status": "success",
                    "retry_count": 0,
                    "cost_usd": 0.0,
                    "duration_sec": 1.0,
                }
            },
            "total_cost_usd": 0.0,
        }
        # Need enough runs and a task that doesn't trigger ANY suggestion:
        # - haiku can't downgrade
        # - 0 retries → no upgrade/add_retry
        # - cost_usd=0, total_cost=0 → no cost-based suggestion
        # - duration not 3x median (only 1 task so median = own duration)
        # BUT: the "100% success 0 retries" path triggers downgrade if model != haiku.
        # haiku → _downgrade_model returns None → no downgrade. Good.
        for i in range(3):
            _write_manifest(tmp_path / f"2026030{i}_120000_healthy-plan", manifest)

        result = suggest_plan(plan, tmp_path)
        assert result.runs_analyzed == 3
        assert result.suggestions == []
        assert result.total_estimated_savings_pct is None


class TestAnalyzeTaskCombinedSuggestions:
    def test_high_retry_and_high_cost_both_suggested(self) -> None:
        """A task with high retry AND high cost should produce multiple suggestions."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku", max_retries=0)}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": 1,
                        "cost_usd": 0.8,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 1.0,
            },
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": 1,
                        "cost_usd": 0.9,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 1.0,
            },
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        categories = {s.category for s in suggestions}
        # retry_rate = 100% > 50% → haiku can upgrade → upgrade_model
        assert "upgrade_model" in categories
        # cost > 40% of plan → haiku can't downgrade → add_review_task
        # (the cost path checks _downgrade_model on haiku → None → add_review_task)
        # Actually haiku → _downgrade_model returns None → add_review_task
        # But wait: success_rate = 0/2 = 0 → 100% success path won't fire
        # The cost path: task_avg_cost = 0.85, plan_avg_cost = 1.0, ratio = 0.85 > 0.4
        # model = haiku → _downgrade_model = None → add_review_task
        assert "add_review_task" in categories


class TestAnalyzeTaskJudgeIntegerScore:
    def test_judge_score_integer_value_accepted(self) -> None:
        """overall_score as int (not float) → isinstance(1, (int, float)) is True → counted."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"verdict": "pass", "overall_score": 1},
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        # int score 1 > 0.9 → remove_judge should be suggested
        assert any(s.category == "remove_judge" for s in suggestions)


class TestAnalyzeTaskCheckpointExactly3xMedian:
    def test_duration_exactly_3x_median_no_checkpoint(self) -> None:
        """task_avg_duration == 3 * median_duration is NOT > 3*median → no add_checkpoint."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        # 3 tasks: t1=3s, t2=1s, t3=1s → all_durations=[3.0, 1.0, 1.0] → median=1.0
        # task_avg_duration for t1 = 3.0; 3*median = 3.0; 3.0 > 3.0 is False
        run = {
            "task_results": {
                "t1": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 3.0},
                "t2": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t3": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
            },
            "total_cost_usd": 0.0,
        }
        suggestions = _analyze_task("t1", task_spec_map, [run])
        assert not any(s.category == "add_checkpoint" for s in suggestions)


class TestAnalyzeTaskCostConfidenceCapped:
    def test_cost_ratio_confidence_capped_at_1(self) -> None:
        """When task_avg_cost > plan_avg_cost, the ratio > 1.0 → confidence = min(ratio, 1.0) = 1.0."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="opus")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 2.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 1.0,  # plan cost < task cost → ratio > 1
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        cost_s = next((s for s in suggestions if "cost" in s.reason.lower()), None)
        assert cost_s is not None
        assert cost_s.confidence == 1.0


class TestFormatSuggestionsJsonConfidenceField:
    def test_confidence_present_in_json_suggestion(self) -> None:
        """Each suggestion in JSON output includes the 'confidence' field."""
        result = PlanSuggestions(
            plan_name="conf-plan",
            runs_analyzed=3,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="upgrade_model",
                    severity="medium",
                    reason="retries happen often",
                    current_value="haiku",
                    suggested_value="sonnet",
                    confidence=0.75,
                    estimated_savings_pct=8.0,
                )
            ],
            total_estimated_savings_pct=8.0,
        )
        output = format_suggestions_json(result)
        data = json.loads(output)
        item = data["suggestions"][0]
        assert "confidence" in item
        assert item["confidence"] == 0.75
        assert item["current_value"] == "haiku"
        assert item["reason"] == "retries happen often"


class TestAnalyzeTaskJudgeVerdictCountMismatch:
    def test_verdict_missing_in_one_run_no_remove_judge(self) -> None:
        """len(judge_verdicts) != total → remove_judge not suggested even if scores are high."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"overall_score": 0.98, "verdict": "pass"},
                    }
                },
                "total_cost_usd": 0.0,
            },
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"overall_score": 0.97},  # no verdict key
                    }
                },
                "total_cost_usd": 0.0,
            },
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "remove_judge" for s in suggestions)


class TestSuggestPlanAggregatesTwoTaskSuggestions:
    def test_two_tasks_each_get_suggestions(self, tmp_path: Path) -> None:
        """Two sonnet tasks that always pass → each gets a downgrade suggestion."""
        plan = PlanSpec(
            version=1,
            name="two-task-plan",
            tasks=[
                TaskSpec(id="t1", model="sonnet"),
                TaskSpec(id="t2", model="sonnet"),
            ],
        )
        task_result: dict = {
            "status": "success",
            "retry_count": 0,
            "cost_usd": 0.0,
            "duration_sec": 1.0,
        }
        manifest = {
            "plan_name": "two-task-plan",
            "task_results": {"t1": task_result, "t2": task_result},
            "total_cost_usd": 0.0,
        }
        for i in range(3):
            run_dir = tmp_path / f"2026010{i}_120000_two-task-plan"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        result = suggest_plan(plan, tmp_path)
        assert result.runs_analyzed == 3
        task_ids = {s.task_id for s in result.suggestions if s.category == "downgrade_model"}
        assert "t1" in task_ids
        assert "t2" in task_ids


class TestLoadRunHistoryExistingDirNoMatchingSubdirs:
    def test_existing_dir_no_matching_subdir_returns_empty(self, tmp_path: Path) -> None:
        """Run dir exists but contains no subdir matching *_planname pattern → returns []."""
        # Create a subdir that does NOT match the plan name
        unrelated = tmp_path / "20260101_120000_other-plan"
        unrelated.mkdir()
        (unrelated / "run_manifest.json").write_text(json.dumps({}), encoding="utf-8")

        result = _load_run_history("my-plan", tmp_path, min_runs=1)
        assert result == []


class TestAnalyzeTaskCheckpointSingleTaskMedian:
    def test_single_task_in_run_no_checkpoint_suggestion(self) -> None:
        """Only t1 appears in task_results → all_durations=[task_duration] → median=task_avg.
        task_avg_duration == median_duration → ratio = 1.0 → NOT > 3.0 → no add_checkpoint."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        run = {
            "task_results": {
                "t1": {
                    "status": "success",
                    "retry_count": 0,
                    "cost_usd": 0.0,
                    "duration_sec": 100.0,
                }
            },
            "total_cost_usd": 0.0,
        }
        suggestions = _analyze_task("t1", task_spec_map, [run])
        assert not any(s.category == "add_checkpoint" for s in suggestions)


class TestSafeIntNearZeroFloat:
    def test_negative_float_between_neg1_and_0_truncates_to_zero(self) -> None:
        """int(-0.9) == 0 in Python (truncation toward zero, not floor)."""
        assert _safe_int(-0.9) == 0

    def test_positive_float_below_1_truncates_to_zero(self) -> None:
        """int(0.9) == 0 (truncation toward zero)."""
        assert _safe_int(0.9) == 0


class TestAnalyzeTaskRetryCountBoolTrue:
    def test_bool_retry_count_treated_as_zero_no_upgrade(self) -> None:
        """retry_count=True (bool) → _safe_int(True)=0 → not counted as retry
        → retry_rate=0.0 ≤ 0.5 → no upgrade_model or add_retry suggestion."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": True,  # bool True → _safe_int → 0
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for _ in range(3)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category in ("upgrade_model", "add_retry") for s in suggestions)


class TestFormatSuggestionsJsonNullItemSavings:
    def test_null_item_savings_serialised_as_null(self) -> None:
        """A suggestion with estimated_savings_pct=None → 'null' in JSON output."""
        result = PlanSuggestions(
            plan_name="null-pct-plan",
            runs_analyzed=3,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="add_retry",
                    severity="medium",
                    reason="retries happen",
                    current_value="max_retries=0",
                    suggested_value="max_retries=1",
                    confidence=0.6,
                    estimated_savings_pct=None,
                )
            ],
            total_estimated_savings_pct=None,
        )
        output = format_suggestions_json(result)
        data = json.loads(output)
        item = data["suggestions"][0]
        assert item["estimated_savings_pct"] is None


class TestAnalyzeTaskContextRatioZero:
    def test_context_ratio_zero_no_reduce_budget_suggestion(self) -> None:
        """context_compression_ratio=0.0 is not > 0.8 → no reduce_context_budget suggestion."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "context_compression_ratio": 0.0,
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "reduce_context_budget" for s in suggestions)


class TestAnalyzeTaskNonNumericContextRatio:
    def test_string_ratio_ignored(self) -> None:
        """context_compression_ratio='high' → not isinstance((int,float)) → filtered out → no suggestion."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "context_compression_ratio": "high",
                    }
                },
                "total_cost_usd": 0.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "reduce_context_budget" for s in suggestions)


class TestAnalyzeTaskRunWithoutTaskResultsKey:
    def test_run_missing_task_results_key_entirely(self) -> None:
        """Run dict with no 'task_results' key → .get() returns None → not isinstance(dict) → skipped."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {"total_cost_usd": 1.0},  # no task_results key at all
            {"total_cost_usd": 0.5},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        # No task_runs found → returns empty
        assert suggestions == []


class TestAnalyzeTaskAllDurationsFilterNonDictPayload:
    def test_non_dict_payload_excluded_from_all_durations(self) -> None:
        """task_results.values() includes a non-dict → filtered by isinstance in duration loop."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        # t1 has a very long duration, but "t2" payload is not a dict → excluded from all_durations
        # With t1=100s and only t1 contributing to all_durations, median = 100s
        # task_avg_duration = 100 → 100 > 3*100 is False → no checkpoint
        run = {
            "task_results": {
                "t1": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 100.0},
                "t2": "not-a-dict",  # filtered out of all_durations
                "t3": None,  # also filtered out
            },
            "total_cost_usd": 0.0,
        }
        suggestions = _analyze_task("t1", task_spec_map, [run])
        # Only t1's duration in all_durations → median=100 → no checkpoint
        assert not any(s.category == "add_checkpoint" for s in suggestions)


class TestFormatSuggestionsOutputStripped:
    def test_output_has_no_trailing_newline(self) -> None:
        """format_suggestions uses .rstrip() → no trailing newline."""
        result = PlanSuggestions(
            plan_name="p",
            runs_analyzed=3,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="add_retry",
                    severity="medium",
                    reason="retries",
                    current_value="max_retries=0",
                    suggested_value="max_retries=1",
                    confidence=0.6,
                    estimated_savings_pct=8.0,
                )
            ],
            total_estimated_savings_pct=8.0,
        )
        output = format_suggestions(result)
        assert not output.endswith("\n")
        assert not output.endswith(" ")


class TestAnalyzeTaskRetryRateBoundary:
    def test_retry_rate_exactly_50_pct_no_upgrade(self) -> None:
        """retry_rate=0.5 is NOT > 0.5 → no upgrade_model or add_retry suggestion."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        # 1 run with retries, 1 without → rate = 1/2 = 0.5 exactly
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 1,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            },
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 0.0,
            },
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "upgrade_model" for s in suggestions)
        assert not any(s.category == "add_retry" for s in suggestions)


class TestAnalyzeTaskCostRatioBoundary:
    def test_cost_ratio_exactly_40_pct_no_cost_suggestion(self) -> None:
        """task_avg_cost / plan_avg_cost == 0.4 exactly → NOT > 0.4 → no cost-path suggestion."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="opus")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.4,
                        "duration_sec": 1.0,
                    }
                },
                "total_cost_usd": 1.0,
            }
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        # Cost ratio is exactly 0.4 → condition is > 0.4 → cost-path not triggered
        # The "always passes" downgrade_model may fire but its reason says "passes in"
        cost_path = [
            s for s in suggestions
            if "consumes" in s.reason.lower()
        ]
        assert cost_path == []


class TestSuggestPlanTasksNotInRuns:
    def test_plan_task_not_in_any_run_produces_no_suggestions(self, tmp_path: Path) -> None:
        """A task in the plan but absent from all run manifests → no crash, no suggestions for it."""
        plan = PlanSpec(
            version=1,
            name="gap-plan",
            tasks=[
                TaskSpec(id="t-missing", model="sonnet"),
                TaskSpec(id="t-present", model="sonnet"),
            ],
        )
        task_result = {
            "status": "success",
            "cost_usd": 0.0,
            "duration_sec": 1.0,
            "retry_count": 0,
        }
        manifest = {
            "plan_name": "gap-plan",
            "task_results": {"t-present": task_result},
            "total_cost_usd": 0.0,
        }
        for i in range(3):
            run_dir = tmp_path / f"2026030{i}_120000_gap-plan"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        result = suggest_plan(plan, tmp_path)
        assert result.runs_analyzed == 3
        # t-missing has no data → no suggestions for it
        assert not any(s.task_id == "t-missing" for s in result.suggestions)


class TestFormatSuggestionsCategoryInOutput:
    def test_category_name_appears_in_reason_line(self) -> None:
        """The suggestion category appears in the formatted output."""
        result = PlanSuggestions(
            plan_name="cat-plan",
            runs_analyzed=5,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="reduce_context_budget",
                    severity="low",
                    reason="context over-provisioned",
                    current_value="context_compression_ratio>0.8",
                    suggested_value="reduce context_budget_tokens",
                    confidence=0.85,
                    estimated_savings_pct=5.0,
                )
            ],
            total_estimated_savings_pct=5.0,
        )
        output = format_suggestions(result)
        assert "reduce_context_budget" in output
        assert "cat-plan" in output
        assert "5.0%" in output


class TestAnalyzeTaskFailurePatternExactThirtyPct:
    def test_exactly_30pct_fires_with_medium_severity(self) -> None:
        """occurrence_rate = 3/10 = 0.3 → fires (>= 0.3); 0.3 < 0.5 → severity 'medium'."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "failure_category": "dependency_missing" if i < 3 else None,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for i in range(10)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        fix_s = [s for s in suggestions if s.category == "fix_failure_pattern"]
        assert len(fix_s) == 1
        assert fix_s[0].severity == "medium"
        assert "dependency_missing" in fix_s[0].reason


class TestAnalyzeTaskFailurePatternTwentyPct:
    def test_20pct_below_threshold_no_suggestion(self) -> None:
        """occurrence_rate = 2/10 = 0.2 → below 0.3 threshold → no fix_failure_pattern."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "failure_category": "output_format_error" if i < 2 else None,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for i in range(10)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "fix_failure_pattern" for s in suggestions)


class TestAnalyzeTaskMultipleFailCategoriesPartialFires:
    def test_only_category_above_threshold_suggested(self) -> None:
        """cascading_failure at 60% fires (high); output_format_error at 20% is silent."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        cats = (
            ["cascading_failure"] * 6  # 60% → fires, severity high
            + ["output_format_error"] * 2  # 20% → below threshold
            + [None, None]
        )
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "failure_category": c,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for c in cats
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        fix_sug = [s for s in suggestions if s.category == "fix_failure_pattern"]
        assert len(fix_sug) == 1
        assert "cascading_failure" in fix_sug[0].reason
        assert fix_sug[0].severity == "high"


class TestAnalyzeTaskFailurePatternHighSeverityBoundary:
    def test_exactly_50pct_gives_high_severity(self) -> None:
        """occurrence_rate = 5/10 = 0.5 → severity 'high' (>= 0.5)."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "failure_category": "verification_gap" if i < 5 else None,
                    }
                },
                "total_cost_usd": 0.0,
            }
            for i in range(10)
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        fix_s = [s for s in suggestions if s.category == "fix_failure_pattern"]
        assert len(fix_s) == 1
        assert fix_s[0].severity == "high"


class TestAnalyzeTaskFailurePatternEmptyCategoryIgnored:
    def test_empty_string_category_not_in_remediation_ignored(self) -> None:
        """failure_category='' → isinstance is True but '' not in _FAILURE_REMEDIATION → not counted."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="haiku")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "failed",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "failure_category": "",
                    }
                },
                "total_cost_usd": 0.0,
            }
        ] * 5  # 5/5 = 100% but empty string filtered → no suggestion
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "fix_failure_pattern" for s in suggestions)


class TestAnalyzeTaskDurationExactly3xMedian:
    def test_duration_exactly_3x_median_no_checkpoint(self) -> None:
        """task_avg_duration == 3 * median_duration → NOT > 3*median → no add_checkpoint."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        run = {
            "task_results": {
                "t1": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 3.0},
                "t2": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t3": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t4": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t5": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
            },
            "total_cost_usd": 0.0,
        }
        # median of [3.0, 1.0, 1.0, 1.0, 1.0] = 1.0; task_avg = 3.0; 3.0 is NOT > 3*1.0
        suggestions = _analyze_task("t1", task_spec_map, [run])
        assert not any(s.category == "add_checkpoint" for s in suggestions)

    def test_duration_just_above_3x_median_triggers_checkpoint(self) -> None:
        """task_avg_duration > 3 * median_duration → add_checkpoint suggested."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        run = {
            "task_results": {
                "t1": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 3.1},
                "t2": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t3": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t4": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
                "t5": {"status": "success", "retry_count": 0, "cost_usd": 0.0, "duration_sec": 1.0},
            },
            "total_cost_usd": 0.0,
        }
        suggestions = _analyze_task("t1", task_spec_map, [run])
        assert any(s.category == "add_checkpoint" for s in suggestions)


class TestSuggestPlanMultipleTasksMixed:
    def test_some_tasks_suggest_some_dont(self, tmp_path: Path) -> None:
        """Plan with two tasks: one always passes (downgrade), one healthy (no suggestions)."""
        tasks = [
            TaskSpec(id="t-downgrade", model="sonnet"),
            TaskSpec(id="t-healthy", model="haiku"),  # haiku can't downgrade
        ]
        plan = PlanSpec(version=1, name="mixed-plan", tasks=tasks)
        task_result_dg = {"status": "success", "cost_usd": 0.0, "duration_sec": 1.0, "retry_count": 0}
        task_result_ok = {"status": "success", "cost_usd": 0.0, "duration_sec": 1.0, "retry_count": 0}
        manifest = {
            "plan_name": "mixed-plan",
            "task_results": {
                "t-downgrade": task_result_dg,
                "t-healthy": task_result_ok,
            },
            "total_cost_usd": 0.0,
        }
        for i in range(3):
            run_dir = tmp_path / f"2026010{i}_120000_mixed-plan"
            run_dir.mkdir()
            (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        result = suggest_plan(plan, tmp_path)
        assert result.runs_analyzed == 3
        # t-downgrade generates downgrade_model; t-healthy generates nothing
        task_ids = {s.task_id for s in result.suggestions}
        assert "t-downgrade" in task_ids
        assert "t-healthy" not in task_ids


class TestAnalyzeTaskJudgeMixedVerdicts:
    def test_mixed_pass_and_fail_verdicts_no_remove_judge(self) -> None:
        """Judge verdicts include 'fail' → not all 'pass' → remove_judge not suggested."""
        task_spec_map = {"t1": TaskSpec(id="t1", model="sonnet")}
        runs = [
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"verdict": "pass", "overall_score": 0.95},
                    }
                },
                "total_cost_usd": 0.0,
            },
            {
                "task_results": {
                    "t1": {
                        "status": "success",
                        "retry_count": 0,
                        "cost_usd": 0.0,
                        "duration_sec": 1.0,
                        "judge_result": {"verdict": "fail", "overall_score": 0.95},
                    }
                },
                "total_cost_usd": 0.0,
            },
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        assert not any(s.category == "remove_judge" for s in suggestions)


class TestFormatSuggestionsConfidenceNotShown:
    def test_confidence_value_not_directly_shown_in_output(self) -> None:
        """format_suggestions does not directly display confidence values in the output text."""
        result = PlanSuggestions(
            plan_name="p",
            runs_analyzed=3,
            suggestions=[
                Suggestion(
                    task_id="t1",
                    category="upgrade_model",
                    severity="medium",
                    reason="retries too frequent",
                    current_value="haiku",
                    suggested_value="sonnet",
                    confidence=0.6789,
                    estimated_savings_pct=8.0,
                )
            ],
            total_estimated_savings_pct=8.0,
        )
        output = format_suggestions(result)
        # confidence is stored but not rendered
        assert "0.6789" not in output
        assert "haiku" in output
        assert "sonnet" in output


class TestTuneTimeoutSuggestion:
    """P2: tune_timeout suggestion for repeated timeout failures."""

    def test_triggered_on_two_timeouts(self) -> None:
        task_spec_map = {"t1": TaskSpec(id="t1", command="echo hi", timeout_sec=300)}
        runs = [
            {"task_results": {"t1": {"status": "failed", "exit_code": 124, "duration_sec": 300.0}}},
            {"task_results": {"t1": {"status": "failed", "exit_code": 124, "duration_sec": 300.0}}},
            {"task_results": {"t1": {"status": "success", "exit_code": 0, "duration_sec": 120.0}}},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        timeout_sugs = [s for s in suggestions if s.category == "tune_timeout"]
        assert len(timeout_sugs) == 1
        sug = timeout_sugs[0]
        assert "timeout_sec=450" in sug.suggested_value  # 300 * 1.5
        assert sug.current_value == "timeout_sec=300"

    def test_not_triggered_on_single_timeout(self) -> None:
        task_spec_map = {"t1": TaskSpec(id="t1", command="echo hi")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "exit_code": 124, "duration_sec": 300.0}}},
            {"task_results": {"t1": {"status": "success", "exit_code": 0, "duration_sec": 120.0}}},
            {"task_results": {"t1": {"status": "success", "exit_code": 0, "duration_sec": 130.0}}},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        timeout_sugs = [s for s in suggestions if s.category == "tune_timeout"]
        assert len(timeout_sugs) == 0

    def test_severity_high_at_50_pct(self) -> None:
        task_spec_map = {"t1": TaskSpec(id="t1", command="echo hi")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "exit_code": 124, "duration_sec": 600.0}}},
            {"task_results": {"t1": {"status": "failed", "exit_code": 124, "duration_sec": 600.0}}},
            {"task_results": {"t1": {"status": "failed", "exit_code": 124, "duration_sec": 600.0}}},
            {"task_results": {"t1": {"status": "success", "exit_code": 0, "duration_sec": 200.0}}},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        timeout_sugs = [s for s in suggestions if s.category == "tune_timeout"]
        assert len(timeout_sugs) == 1
        assert timeout_sugs[0].severity == "high"

    def test_severity_medium_below_50_pct(self) -> None:
        task_spec_map = {"t1": TaskSpec(id="t1", command="echo hi")}
        runs = [
            {"task_results": {"t1": {"status": "failed", "exit_code": 124, "duration_sec": 300.0}}},
            {"task_results": {"t1": {"status": "failed", "exit_code": 124, "duration_sec": 300.0}}},
            {"task_results": {"t1": {"status": "success", "exit_code": 0, "duration_sec": 100.0}}},
            {"task_results": {"t1": {"status": "success", "exit_code": 0, "duration_sec": 100.0}}},
            {"task_results": {"t1": {"status": "success", "exit_code": 0, "duration_sec": 100.0}}},
        ]
        suggestions = _analyze_task("t1", task_spec_map, runs)
        timeout_sugs = [s for s in suggestions if s.category == "tune_timeout"]
        assert len(timeout_sugs) == 1
        assert timeout_sugs[0].severity == "medium"
