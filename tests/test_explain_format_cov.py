from __future__ import annotations

from maestro_cli.explain import (
    PlanExplanation,
    TaskExplanation,
    _merge_selection_entry,
    _safe_float,
    _safe_int,
    format_explain,
)
from maestro_cli.models import ContextSelectionEntry


class TestFormatExplain:
    def test_renders_header_and_task_table(self) -> None:
        explanation = PlanExplanation(
            plan_name="demo-plan",
            task_count=2,
            cache_entries=5,
            cache_size_bytes=12345,
            tasks=[
                TaskExplanation(
                    task_id="build",
                    cache_status="hit",
                    reason="hash match [abc123def456]",
                ),
                TaskExplanation(
                    task_id="a-very-long-task-id",
                    cache_status="miss",
                    reason="no cached entry",
                ),
            ],
        )

        rendered = format_explain(explanation)
        lines = rendered.splitlines()

        # Header block
        assert "Plan: demo-plan" in lines
        assert "Tasks: 2" in lines
        assert "Cache entries: 5" in lines
        assert "Cache size: 12345 bytes" in lines

        # Column header row present
        assert any("task_id" in line and "status" in line and "reason" in line for line in lines)
        # Separator row uses dashes and the +- joiner
        assert any(set(line) <= {"-", "+", " "} and "-+-" in line for line in lines)

        # Both task rows are emitted
        assert any("build" in line and "hit" in line for line in lines)
        assert any("a-very-long-task-id" in line and "miss" in line for line in lines)

    def test_column_width_grows_to_fit_longest_value(self) -> None:
        explanation = PlanExplanation(
            plan_name="p",
            task_count=1,
            cache_entries=0,
            cache_size_bytes=0,
            tasks=[
                TaskExplanation(
                    task_id="x",
                    cache_status="disabled",
                    reason="caching disabled for this task entirely",
                ),
            ],
        )

        rendered = format_explain(explanation)
        lines = rendered.splitlines()
        # The data row should be padded so the reason column matches the
        # longest reason value, demonstrating the dynamic width computation.
        data_row = next(line for line in lines if line.startswith("x "))
        assert "caching disabled for this task entirely" in data_row
        assert "disabled" in data_row

    def test_no_tasks_branch(self) -> None:
        explanation = PlanExplanation(
            plan_name="empty",
            task_count=0,
            cache_entries=0,
            cache_size_bytes=0,
            tasks=[],
        )

        rendered = format_explain(explanation)
        lines = rendered.splitlines()

        assert "Plan: empty" in lines
        assert "(no tasks)" in lines
        # No table column header should be emitted when there are no tasks.
        assert not any("status" in line and "reason" in line for line in lines)


class TestSafeInt:
    def test_bool_returns_none(self) -> None:
        # bool is a subclass of int, but must be rejected before the int path.
        assert _safe_int(True) is None
        assert _safe_int(False) is None

    def test_float_is_truncated_to_int(self) -> None:
        assert _safe_int(3.9) == 3
        assert _safe_int(-2.5) == -2

    def test_plain_int_passes_through(self) -> None:
        assert _safe_int(7) == 7

    def test_non_numeric_returns_none(self) -> None:
        assert _safe_int("12") is None
        assert _safe_int(None) is None


class TestSafeFloat:
    def test_bool_returns_none(self) -> None:
        assert _safe_float(True) is None
        assert _safe_float(False) is None

    def test_int_becomes_float(self) -> None:
        result = _safe_float(5)
        assert result == 5.0
        assert isinstance(result, float)

    def test_float_passes_through(self) -> None:
        assert _safe_float(2.5) == 2.5

    def test_non_numeric_returns_none(self) -> None:
        assert _safe_float("3.0") is None
        assert _safe_float(None) is None


class TestMergeSelectionEntry:
    def test_existing_none_returns_incoming(self) -> None:
        incoming = ContextSelectionEntry(upstream_id="u", score=1.0)
        assert _merge_selection_entry(None, incoming) is incoming

    def test_incoming_zero_final_not_trimmed_keeps_existing_final(self) -> None:
        # Drives the else branch: incoming.tokens_final == 0, incoming.trimmed
        # is False, and existing.tokens_final > 0, so the existing value wins.
        existing = ContextSelectionEntry(
            upstream_id="build",
            tokens_final=300,
        )
        incoming = ContextSelectionEntry(
            upstream_id="build",
            tokens_final=0,
            trimmed=False,
        )

        merged = _merge_selection_entry(existing, incoming)
        assert merged.tokens_final == 300

    def test_incoming_final_present_overrides_existing(self) -> None:
        existing = ContextSelectionEntry(upstream_id="build", tokens_final=300)
        incoming = ContextSelectionEntry(upstream_id="build", tokens_final=150)

        merged = _merge_selection_entry(existing, incoming)
        assert merged.tokens_final == 150

    def test_merge_prefers_incoming_signals_over_defaults(self) -> None:
        existing = ContextSelectionEntry(
            upstream_id="build",
            score=0.5,
            keywords_matched=["old"],
            hop_distance=2,
            hop_decay_factor=0.8,
            tokens_raw=100,
            tokens_final=50,
            trimmed=False,
            trim_reason="",
        )
        incoming = ContextSelectionEntry(
            upstream_id="build",
            score=1.5,
            keywords_matched=["new"],
            hop_distance=1,
            hop_decay_factor=0.64,
            tokens_raw=200,
            tokens_final=80,
            trimmed=True,
            trim_reason="budget_trim",
        )

        merged = _merge_selection_entry(existing, incoming)
        assert merged.upstream_id == "build"
        assert merged.score == 1.5
        assert merged.keywords_matched == ["new"]
        assert merged.hop_distance == 1
        assert merged.hop_decay_factor == 0.64
        assert merged.tokens_raw == 200
        assert merged.tokens_final == 80
        assert merged.trimmed is True
        assert "budget_trim" in merged.trim_reason

    def test_merge_falls_back_to_existing_when_incoming_is_default(self) -> None:
        existing = ContextSelectionEntry(
            upstream_id="build",
            score=0.5,
            keywords_matched=["old"],
            hop_distance=2,
            hop_decay_factor=0.8,
            tokens_raw=100,
            tokens_final=50,
            trimmed=True,
            trim_reason="hop_decay",
        )
        incoming = ContextSelectionEntry(upstream_id="build")

        merged = _merge_selection_entry(existing, incoming)
        assert merged.score == 0.5
        assert merged.keywords_matched == ["old"]
        assert merged.hop_distance == 2
        assert merged.hop_decay_factor == 0.8
        assert merged.tokens_raw == 100
        # trimmed in existing stays True via OR
        assert merged.trimmed is True
        assert "hop_decay" in merged.trim_reason
