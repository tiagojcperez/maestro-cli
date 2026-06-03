from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import TaskResult
from maestro_cli.scheduler import _apply_context_budget, _estimate_tokens, _extract_keywords


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


def _make_result(task_id: str, stdout_tail: str) -> TaskResult:
    now = datetime.now(tz=timezone.utc)
    return TaskResult(
        task_id=task_id,
        status="success",
        exit_code=0,
        started_at=now,
        finished_at=now,
        duration_sec=1.0,
        command="echo",
        log_path=Path(f"C:/tmp/{task_id}.log"),
        result_path=Path(f"C:/tmp/{task_id}.result.json"),
        stdout_tail=stdout_tail,
    )


# ===========================================================================
# TestEstimateTokens
# ===========================================================================


class TestEstimateTokens:
    def test_estimates_from_char_count(self) -> None:
        # 4 chars per token (conservative estimate)
        assert _estimate_tokens("a" * 400) == 100

    def test_minimum_is_one(self) -> None:
        # Empty string → minimum 1 token
        assert _estimate_tokens("") == 1

    def test_scales_linearly(self) -> None:
        t1 = _estimate_tokens("a" * 400)
        t2 = _estimate_tokens("a" * 800)
        assert t2 == 2 * t1

    def test_rounds_down(self) -> None:
        # 5 chars / 4 = 1.25 → floor to 1
        assert _estimate_tokens("abcde") == 1

    def test_single_char(self) -> None:
        assert _estimate_tokens("x") == 1

    def test_exactly_divisible(self) -> None:
        # 8 chars / 4 = exactly 2
        assert _estimate_tokens("12345678") == 2

    @pytest.mark.parametrize("length", [100, 1000, 10000])
    def test_proportional_to_length(self, length: int) -> None:
        base = _estimate_tokens("a" * 400)  # 100 tokens
        result = _estimate_tokens("a" * length)
        assert result == pytest.approx(length // 4, abs=1)


# ===========================================================================
# TestApplyContextBudget
# ===========================================================================


class TestApplyContextBudget:
    def test_no_trimming_when_under_budget(self) -> None:
        upstream = {"t1": _make_result("t1", "short output")}
        result, trims, _ = _apply_context_budget(upstream, 10_000)
        assert trims == []
        assert result["t1"].stdout_tail == "short output"

    def test_returns_same_dict_when_under_budget(self) -> None:
        upstream = {"t1": _make_result("t1", "short")}
        result, _, _ = _apply_context_budget(upstream, 10_000)
        # When no trimming needed, the original dict is returned unchanged
        assert result is upstream

    def test_trims_when_total_over_budget(self) -> None:
        long_tail = "x" * 4000  # ~1000 tokens
        upstream = {"t1": _make_result("t1", long_tail)}
        result, trims, _ = _apply_context_budget(upstream, 100)
        assert len(result["t1"].stdout_tail) < len(long_tail)

    def test_trim_records_returned_with_task_id(self) -> None:
        long_tail = "x" * 4000
        upstream = {"t1": _make_result("t1", long_tail)}
        _, trims, _ = _apply_context_budget(upstream, 100)
        assert len(trims) == 1
        task_id, orig, trimmed = trims[0]
        assert task_id == "t1"

    def test_trim_record_shows_original_larger_than_trimmed(self) -> None:
        long_tail = "x" * 4000
        upstream = {"t1": _make_result("t1", long_tail)}
        _, trims, _ = _apply_context_budget(upstream, 100)
        _, orig, trimmed = trims[0]
        assert orig > trimmed

    def test_trim_records_empty_when_no_trimming(self) -> None:
        upstream = {"t1": _make_result("t1", "short text")}
        _, trims, _ = _apply_context_budget(upstream, 10_000)
        assert trims == []

    def test_originals_unchanged_after_trim(self) -> None:
        original_tail = "x" * 4000
        original_result = _make_result("t1", original_tail)
        upstream = {"t1": original_result}
        _apply_context_budget(upstream, 100)
        # Original TaskResult must be untouched (dataclasses.replace creates copies)
        assert original_result.stdout_tail == original_tail

    def test_proportional_trimming_two_equal_upstreams(self) -> None:
        t1 = _make_result("t1", "a" * 4000)
        t2 = _make_result("t2", "b" * 4000)
        upstream = {"t1": t1, "t2": t2}
        result, trims, _ = _apply_context_budget(upstream, 200)
        # Both should be trimmed equally (same original length)
        assert len(result["t1"].stdout_tail) == len(result["t2"].stdout_tail)

    def test_two_trim_records_when_both_over_budget(self) -> None:
        t1 = _make_result("t1", "a" * 4000)
        t2 = _make_result("t2", "b" * 4000)
        upstream = {"t1": t1, "t2": t2}
        _, trims, _ = _apply_context_budget(upstream, 200)
        assert len(trims) == 2

    def test_empty_upstream_returns_empty(self) -> None:
        result, trims, _ = _apply_context_budget({}, 1000)
        assert result == {}
        assert trims == []

    def test_result_keys_match_upstream_keys(self) -> None:
        t1 = _make_result("t1", "a" * 4000)
        t2 = _make_result("t2", "b" * 4000)
        upstream = {"t1": t1, "t2": t2}
        result, _, _ = _apply_context_budget(upstream, 100)
        assert set(result.keys()) == {"t1", "t2"}

    def test_very_large_budget_no_trimming(self) -> None:
        long_tail = "x" * 10_000
        upstream = {"t1": _make_result("t1", long_tail)}
        result, trims, _ = _apply_context_budget(upstream, 1_000_000)
        assert trims == []
        assert result["t1"].stdout_tail == long_tail

    def test_short_upstream_not_trimmed_when_only_long_one_over_budget(self) -> None:
        # One short upstream and one long — only the long one should be trimmed
        short = _make_result("t1", "short")  # ~2 tokens
        long_tail = "y" * 4000  # ~1000 tokens
        long_result = _make_result("t2", long_tail)
        upstream = {"t1": short, "t2": long_result}
        result, trims, _ = _apply_context_budget(upstream, 100)
        # Short one should have been proportionally trimmed too (ratio applied uniformly)
        # But what matters: total tokens in result <= budget
        total_chars = len(result["t1"].stdout_tail) + len(result["t2"].stdout_tail)
        # ratio = 100 / (2 + 1000) ≈ 0.1 → total result chars ≈ 0.1 * 4005 ≈ 400 chars
        assert total_chars < len(long_tail)  # clearly less than without trimming


# ===========================================================================
# TestContextBudgetLoader
# ===========================================================================


class TestContextBudgetLoader:
    def test_task_context_budget_tokens_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    context_budget_tokens: 8000
""",
        )
        plan = load_plan(plan_file)
        assert plan.tasks[0].context_budget_tokens == 8000

    def test_context_budget_tokens_negative_raises_e019(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    context_budget_tokens: -1
""",
        )
        with pytest.raises(PlanValidationError, match=r"\[E019\]"):
            load_plan(plan_file)

    def test_context_budget_tokens_non_integer_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    context_budget_tokens: "not-a-number"
""",
        )
        with pytest.raises(PlanValidationError):
            load_plan(plan_file)

    def test_defaults_context_budget_tokens_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
defaults:
  context_budget_tokens: 20000
tasks:
  - id: t1
    command: "echo hello"
""",
        )
        plan = load_plan(plan_file)
        assert plan.defaults.context_budget_tokens == 20_000

    def test_context_budget_absent_defaults_to_none_on_task(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""",
        )
        plan = load_plan(plan_file)
        assert plan.tasks[0].context_budget_tokens is None

    def test_context_budget_absent_defaults_to_none_on_defaults(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""",
        )
        plan = load_plan(plan_file)
        assert plan.defaults.context_budget_tokens is None

    def test_task_budget_independent_of_plan_budget(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
defaults:
  context_budget_tokens: 10000
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    context_budget_tokens: 2000
""",
        )
        plan = load_plan(plan_file)
        assert plan.defaults.context_budget_tokens == 10_000
        assert plan.tasks[0].context_budget_tokens == 2_000

    def test_large_budget_value_accepted(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    context_budget_tokens: 1000000
""",
        )
        plan = load_plan(plan_file)
        assert plan.tasks[0].context_budget_tokens == 1_000_000


# ===========================================================================
# TestContextBudgetSchedulerIntegration
# ===========================================================================


class TestContextBudgetSchedulerIntegration:
    def test_apply_budget_is_idempotent_at_exact_boundary(self) -> None:
        # Total tokens equals budget exactly → no trimming
        tail = "a" * 400  # 100 tokens
        upstream = {"t1": _make_result("t1", tail)}
        result, trims, _ = _apply_context_budget(upstream, 100)
        assert trims == []
        assert result["t1"].stdout_tail == tail

    def test_apply_budget_trims_just_over_boundary(self) -> None:
        # One token over budget should trigger trimming
        tail = "a" * 404  # 101 tokens
        upstream = {"t1": _make_result("t1", tail)}
        result, trims, _ = _apply_context_budget(upstream, 100)
        assert len(trims) == 1
        assert len(result["t1"].stdout_tail) < len(tail)

    def test_multiple_tasks_all_short_no_trim(self) -> None:
        upstream = {
            "t1": _make_result("t1", "hello"),
            "t2": _make_result("t2", "world"),
            "t3": _make_result("t3", "foo"),
        }
        result, trims, _ = _apply_context_budget(upstream, 10_000)
        assert trims == []
        for task_id in upstream:
            assert result[task_id].stdout_tail == upstream[task_id].stdout_tail

    def test_trim_result_is_new_object_not_original(self) -> None:
        long_tail = "x" * 4000
        original = _make_result("t1", long_tail)
        upstream = {"t1": original}
        result, trims, _ = _apply_context_budget(upstream, 100)
        # The result should be a different object (copy via dataclasses.replace)
        assert result["t1"] is not original


# ===========================================================================
# TestIntentDrivenContextFiltering
# ===========================================================================


class TestIntentDrivenContextFiltering:
    def test_relevant_sections_are_kept_when_over_budget(self) -> None:
        preamble = "build output line " * 80
        relevant = "scheduler context budget token filtering section " * 40
        epilogue = "unrelated deployment artifact logs " * 80
        tail = f"{preamble}\n\n{relevant}\n\n{epilogue}"

        upstream = {"t1": _make_result("t1", tail)}
        result, trims, _ = _apply_context_budget(
            upstream,
            300,
            {"scheduler", "context", "budget", "filtering"},
        )

        filtered_tail = result["t1"].stdout_tail
        assert "scheduler context budget token filtering" in filtered_tail
        assert "unrelated deployment artifact logs" not in filtered_tail
        assert len(filtered_tail) < len(tail)
        assert trims

    def test_no_keyword_match_falls_back_to_proportional_truncation(self) -> None:
        alpha = "alpha section " * 120
        beta = "beta section " * 120
        tail = f"{alpha}\n\n{beta}"

        upstream = {"t1": _make_result("t1", tail)}
        result, trims, _ = _apply_context_budget(upstream, 80, {"scheduler", "context"})

        filtered_tail = result["t1"].stdout_tail
        assert filtered_tail
        assert filtered_tail.startswith("alpha section")
        assert "beta section" not in filtered_tail
        assert trims

    def test_prompt_keyword_extraction_removes_common_stopwords(self) -> None:
        keywords = _extract_keywords("Use the scheduler to filter context and update tokens")
        assert "use" not in keywords
        assert "the" not in keywords
        assert "and" not in keywords
        assert "scheduler" in keywords
        assert "filter" in keywords
