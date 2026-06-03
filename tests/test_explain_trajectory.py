from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro_cli.explain import (
    explain_context,
    explain_context_trajectory,
    format_context_trajectory,
    format_context_trajectory_json,
)
from maestro_cli.models import ContextSelectionEntry, ContextTrajectoryReport


def _write_events(run_path: Path, events: list[dict]) -> None:
    (run_path / "events.jsonl").write_text(
        "\n".join(json.dumps(event) for event in events),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Empty / missing data
# ---------------------------------------------------------------------------


class TestMissingOrEmptyData:
    def test_missing_events_file_returns_empty(self, tmp_path: Path) -> None:
        result = explain_context_trajectory(tmp_path)
        assert result == []

    def test_empty_events_file_returns_empty(self, tmp_path: Path) -> None:
        (tmp_path / "events.jsonl").write_text("", encoding="utf-8")
        result = explain_context_trajectory(tmp_path)
        assert result == []

    def test_irrelevant_events_are_ignored(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "task_start", "task_id": "build", "ts": "2025-01-01"},
            {"event": "task_complete", "task_id": "build", "status": "success"},
            {"event": "run_complete", "plan_name": "test"},
        ])
        result = explain_context_trajectory(tmp_path)
        assert result == []

    def test_events_without_task_id_are_ignored(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "upstream_id": "build", "score": 1.0},
        ])
        result = explain_context_trajectory(tmp_path)
        assert result == []

    def test_blank_lines_in_events_file_tolerated(self, tmp_path: Path) -> None:
        lines = [
            json.dumps({"event": "task_start", "task_id": "a"}),
            "",
            json.dumps({"event": "context_selection", "task_id": "b", "upstream_id": "a", "score": 1.0, "tokens_raw": 100, "tokens_final": 100}),
            "",
        ]
        (tmp_path / "events.jsonl").write_text("\n".join(lines), encoding="utf-8")
        result = explain_context_trajectory(tmp_path)
        assert len(result) == 1
        assert result[0].task_id == "b"

    def test_malformed_json_lines_tolerated(self, tmp_path: Path) -> None:
        lines = [
            "not json at all }{",
            json.dumps({"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 1.0, "tokens_raw": 50, "tokens_final": 50}),
        ]
        (tmp_path / "events.jsonl").write_text("\n".join(lines), encoding="utf-8")
        result = explain_context_trajectory(tmp_path)
        assert len(result) == 1


# ---------------------------------------------------------------------------
# context_selection events
# ---------------------------------------------------------------------------


class TestContextSelectionEvents:
    def test_single_selection_event_creates_report(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "review",
                "upstream_id": "build",
                "score": 1.5,
                "keywords_matched": ["api", "schema"],
                "hop_distance": 1,
                "hop_decay_factor": 1.0,
                "tokens_raw": 400,
                "tokens_final": 400,
                "trimmed": False,
                "trim_reason": "",
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert len(reports) == 1
        report = reports[0]
        assert report.task_id == "review"
        assert len(report.entries) == 1
        entry = report.entries[0]
        assert entry.upstream_id == "build"
        assert entry.score == pytest.approx(1.5)
        assert entry.keywords_matched == ["api", "schema"]
        assert entry.hop_distance == 1
        assert entry.hop_decay_factor == pytest.approx(1.0)
        assert entry.tokens_raw == 400
        assert entry.tokens_final == 400
        assert entry.trimmed is False

    def test_context_trajectory_event_name_is_alias(self, tmp_path: Path) -> None:
        """'context_trajectory' event is treated the same as 'context_selection'."""
        _write_events(tmp_path, [
            {
                "event": "context_trajectory",
                "task_id": "synthesize",
                "upstream_id": "scan",
                "score": 0.9,
                "hop_distance": 2,
                "hop_decay_factor": 0.8,
                "tokens_raw": 300,
                "tokens_final": 300,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert len(reports) == 1
        entry = reports[0].entries[0]
        assert entry.upstream_id == "scan"
        assert entry.hop_decay_factor == pytest.approx(0.8)

    def test_multiple_upstreams_for_same_task(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "qa", "upstream_id": "build", "score": 2.0, "tokens_raw": 500, "tokens_final": 500},
            {"event": "context_selection", "task_id": "qa", "upstream_id": "lint", "score": 0.5, "tokens_raw": 100, "tokens_final": 100},
        ])
        reports = explain_context_trajectory(tmp_path)
        assert len(reports) == 1
        assert len(reports[0].entries) == 2
        # sorted by score descending
        assert reports[0].entries[0].upstream_id == "build"
        assert reports[0].entries[1].upstream_id == "lint"

    def test_hop_distance_and_decay_factor_recorded(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "final",
                "upstream_id": "step3",
                "score": 0.3,
                "hop_distance": 3,
                "hop_decay_factor": 0.64,
                "tokens_raw": 200,
                "tokens_final": 200,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        entry = reports[0].entries[0]
        assert entry.hop_distance == 3
        assert entry.hop_decay_factor == pytest.approx(0.64)

    def test_budget_tokens_field_from_selection_event(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "review",
                "upstream_id": "build",
                "score": 1.0,
                "budget_tokens": 800,
                "tokens_raw": 300,
                "tokens_final": 300,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].budget_tokens == 800

    def test_budget_alt_field_name(self, tmp_path: Path) -> None:
        """'budget' is accepted alongside 'budget_tokens'."""
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "review",
                "upstream_id": "build",
                "score": 1.0,
                "budget": 600,
                "tokens_raw": 300,
                "tokens_final": 300,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].budget_tokens == 600

    def test_trim_reason_preserved(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "t",
                "upstream_id": "up",
                "score": 0.5,
                "trimmed": True,
                "trim_reason": "evicted_by_budget",
                "tokens_raw": 400,
                "tokens_final": 0,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        entry = reports[0].entries[0]
        assert entry.trimmed is True
        assert entry.trim_reason == "evicted_by_budget"


# ---------------------------------------------------------------------------
# context_budget_trim events
# ---------------------------------------------------------------------------


class TestContextBudgetTrimEvents:
    def test_trim_event_creates_trimmed_entry(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_budget_trim",
                "task_id": "report",
                "upstream_id": "collect",
                "original_tokens": 600,
                "trimmed_tokens": 250,
                "budget": 400,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert len(reports) == 1
        report = reports[0]
        assert report.budget_tokens == 400
        entry = report.entries[0]
        assert entry.upstream_id == "collect"
        assert entry.tokens_raw == 600
        assert entry.tokens_final == 250
        assert entry.trimmed is True
        assert entry.trim_reason == "budget_trim"

    def test_fully_evicted_upstream_counted(self, tmp_path: Path) -> None:
        """trimmed=True AND tokens_final=0 → counted in upstreams_evicted."""
        _write_events(tmp_path, [
            {
                "event": "context_budget_trim",
                "task_id": "final",
                "upstream_id": "noisy",
                "original_tokens": 800,
                "trimmed_tokens": 0,
                "budget": 200,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        report = reports[0]
        assert report.upstreams_evicted == 1
        assert report.entries[0].tokens_final == 0

    def test_partial_trim_not_counted_as_evicted(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_budget_trim",
                "task_id": "final",
                "upstream_id": "partial",
                "original_tokens": 800,
                "trimmed_tokens": 150,
                "budget": 300,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].upstreams_evicted == 0

    def test_trim_event_budget_tokens_alt_field(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_budget_trim",
                "task_id": "task",
                "upstream_id": "up",
                "original_tokens": 100,
                "trimmed_tokens": 50,
                "budget_tokens": 75,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].budget_tokens == 75

    def test_trim_event_upstream_alt_field(self, tmp_path: Path) -> None:
        """'upstream' is accepted instead of 'upstream_id' in budget_trim events."""
        _write_events(tmp_path, [
            {
                "event": "context_budget_trim",
                "task_id": "task",
                "upstream": "build",
                "original_tokens": 500,
                "trimmed_tokens": 200,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].entries[0].upstream_id == "build"

    def test_trim_event_without_upstream_id_ignored(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_budget_trim",
                "task_id": "task",
                "original_tokens": 500,
                "trimmed_tokens": 200,
                # no upstream_id or upstream
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        # report is created (task_id exists), but no entries
        assert len(reports) == 1
        assert reports[0].entries == []


# ---------------------------------------------------------------------------
# context_compression events
# ---------------------------------------------------------------------------


class TestContextCompressionEvents:
    def test_compression_event_sets_totals(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "qa",
                "context_raw_tokens": 1200,
                "context_final_tokens": 600,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert len(reports) == 1
        report = reports[0]
        assert report.total_tokens_raw == 1200
        assert report.total_tokens_final == 600

    def test_compression_entries_populate_report(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "qa",
                "context_raw_tokens": 900,
                "context_final_tokens": 450,
                "budget_tokens": 500,
                "entries": [
                    {
                        "upstream_id": "build",
                        "score": 1.8,
                        "keywords_matched": ["compile", "error"],
                        "hop_distance": 1,
                        "hop_decay_factor": 1.0,
                        "tokens_raw": 600,
                        "tokens_final": 300,
                        "trimmed": True,
                        "trim_reason": "budget_trim",
                    },
                    {
                        "upstream_id": "test",
                        "score": 0.6,
                        "keywords_matched": ["failure"],
                        "hop_distance": 1,
                        "hop_decay_factor": 1.0,
                        "tokens_raw": 300,
                        "tokens_final": 150,
                        "trimmed": False,
                        "trim_reason": "",
                    },
                ],
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        report = reports[0]
        assert report.budget_tokens == 500
        assert len(report.entries) == 2
        assert report.entries[0].upstream_id == "build"   # higher score
        assert report.entries[0].keywords_matched == ["compile", "error"]
        assert report.entries[0].trimmed is True

    def test_compression_without_entries_only_sets_totals(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "qa",
                "context_raw_tokens": 500,
                "context_final_tokens": 200,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        report = reports[0]
        assert report.total_tokens_raw == 500
        assert report.total_tokens_final == 200
        assert report.entries == []

    def test_compression_entries_alt_field_name(self, tmp_path: Path) -> None:
        """'selection_entries' is accepted as alternate field for 'entries'."""
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "task",
                "context_raw_tokens": 200,
                "context_final_tokens": 100,
                "selection_entries": [
                    {"upstream_id": "up", "score": 1.0, "tokens_raw": 200, "tokens_final": 100},
                ],
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert len(reports[0].entries) == 1
        assert reports[0].entries[0].upstream_id == "up"

    def test_malformed_entries_in_compression_skipped(self, tmp_path: Path) -> None:
        """Non-dict entries in the 'entries' list are silently dropped."""
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "task",
                "context_raw_tokens": 200,
                "context_final_tokens": 100,
                "entries": [
                    "not a dict",
                    42,
                    {"upstream_id": "valid", "score": 1.0, "tokens_raw": 200, "tokens_final": 100},
                ],
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert len(reports[0].entries) == 1
        assert reports[0].entries[0].upstream_id == "valid"

    def test_compression_entries_missing_upstream_id_skipped(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "task",
                "context_raw_tokens": 100,
                "context_final_tokens": 50,
                "entries": [
                    {"score": 1.0, "tokens_raw": 100, "tokens_final": 50},  # no upstream_id
                ],
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].entries == []


# ---------------------------------------------------------------------------
# Merge behavior (multiple events for same upstream)
# ---------------------------------------------------------------------------


class TestMergeBehavior:
    def test_selection_then_trim_merged_correctly(self, tmp_path: Path) -> None:
        """Selection event followed by budget_trim for same upstream merges both."""
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "final",
                "upstream_id": "step1",
                "score": 1.2,
                "keywords_matched": ["algo"],
                "hop_distance": 1,
                "hop_decay_factor": 1.0,
                "tokens_raw": 500,
                "tokens_final": 500,
                "trimmed": False,
            },
            {
                "event": "context_budget_trim",
                "task_id": "final",
                "upstream_id": "step1",
                "original_tokens": 500,
                "trimmed_tokens": 150,
            },
        ])
        reports = explain_context_trajectory(tmp_path)
        report = reports[0]
        assert len(report.entries) == 1
        entry = report.entries[0]
        assert entry.upstream_id == "step1"
        assert entry.score == pytest.approx(1.2)       # preserved from selection
        assert entry.keywords_matched == ["algo"]       # preserved from selection
        assert entry.trimmed is True                    # updated by trim event
        assert entry.tokens_final == 150                # updated by trim event

    def test_later_nonzero_score_overwrites_zero(self, tmp_path: Path) -> None:
        """If first event has score=0.0, a later event with non-zero score wins."""
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 0.0, "tokens_raw": 100, "tokens_final": 100},
            {"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 1.5, "tokens_raw": 100, "tokens_final": 100},
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].entries[0].score == pytest.approx(1.5)

    def test_later_nonempty_keywords_override_empty(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 1.0, "keywords_matched": [], "tokens_raw": 100, "tokens_final": 100},
            {"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 1.0, "keywords_matched": ["kw"], "tokens_raw": 100, "tokens_final": 100},
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].entries[0].keywords_matched == ["kw"]

    def test_trim_flag_accumulates_across_events(self, tmp_path: Path) -> None:
        """Once trimmed=True is seen, it stays True regardless of later events."""
        _write_events(tmp_path, [
            {"event": "context_budget_trim", "task_id": "t", "upstream_id": "up", "original_tokens": 200, "trimmed_tokens": 0},
            {"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 1.0, "trimmed": False, "tokens_raw": 200, "tokens_final": 100},
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].entries[0].trimmed is True

    def test_same_upstream_appears_once_in_entries(self, tmp_path: Path) -> None:
        """Multiple events for the same upstream_id produce a single merged entry."""
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 0.5, "tokens_raw": 100, "tokens_final": 100},
            {"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 0.8, "tokens_raw": 100, "tokens_final": 100},
            {"event": "context_budget_trim", "task_id": "t", "upstream_id": "up", "original_tokens": 100, "trimmed_tokens": 40},
        ])
        reports = explain_context_trajectory(tmp_path)
        assert len(reports[0].entries) == 1


# ---------------------------------------------------------------------------
# Sorting and ordering
# ---------------------------------------------------------------------------


class TestSortingAndOrdering:
    def test_entries_sorted_by_score_descending(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "low", "score": 0.3, "tokens_raw": 50, "tokens_final": 50},
            {"event": "context_selection", "task_id": "t", "upstream_id": "high", "score": 2.1, "tokens_raw": 50, "tokens_final": 50},
            {"event": "context_selection", "task_id": "t", "upstream_id": "mid", "score": 1.0, "tokens_raw": 50, "tokens_final": 50},
        ])
        reports = explain_context_trajectory(tmp_path)
        ids = [e.upstream_id for e in reports[0].entries]
        assert ids == ["high", "mid", "low"]

    def test_entries_sorted_by_hop_distance_when_scores_equal(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "far", "score": 1.0, "hop_distance": 3, "tokens_raw": 50, "tokens_final": 50},
            {"event": "context_selection", "task_id": "t", "upstream_id": "near", "score": 1.0, "hop_distance": 1, "tokens_raw": 50, "tokens_final": 50},
        ])
        reports = explain_context_trajectory(tmp_path)
        ids = [e.upstream_id for e in reports[0].entries]
        assert ids == ["near", "far"]

    def test_entries_sorted_by_upstream_id_as_tiebreaker(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "zeta", "score": 1.0, "hop_distance": 1, "tokens_raw": 10, "tokens_final": 10},
            {"event": "context_selection", "task_id": "t", "upstream_id": "alpha", "score": 1.0, "hop_distance": 1, "tokens_raw": 10, "tokens_final": 10},
        ])
        reports = explain_context_trajectory(tmp_path)
        ids = [e.upstream_id for e in reports[0].entries]
        assert ids == ["alpha", "zeta"]

    def test_multiple_tasks_sorted_by_task_id(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "zebra", "upstream_id": "a", "score": 1.0, "tokens_raw": 10, "tokens_final": 10},
            {"event": "context_selection", "task_id": "alpha", "upstream_id": "b", "score": 1.0, "tokens_raw": 10, "tokens_final": 10},
            {"event": "context_selection", "task_id": "mango", "upstream_id": "c", "score": 1.0, "tokens_raw": 10, "tokens_final": 10},
        ])
        reports = explain_context_trajectory(tmp_path)
        task_ids = [r.task_id for r in reports]
        assert task_ids == ["alpha", "mango", "zebra"]


# ---------------------------------------------------------------------------
# Token aggregation
# ---------------------------------------------------------------------------


class TestTokenAggregation:
    def test_compression_totals_take_precedence_over_entry_sum(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "task",
                "context_raw_tokens": 1000,
                "context_final_tokens": 500,
                "entries": [
                    {"upstream_id": "a", "score": 1.0, "tokens_raw": 300, "tokens_final": 150},
                    {"upstream_id": "b", "score": 0.5, "tokens_raw": 200, "tokens_final": 100},
                ],
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        report = reports[0]
        # Totals from compression event (1000/500), not sum of entries (500/250)
        assert report.total_tokens_raw == 1000
        assert report.total_tokens_final == 500

    def test_totals_fallback_sums_entries_when_no_compression(self, tmp_path: Path) -> None:
        """Without a compression event, total_tokens_raw/final are summed from entries."""
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "a", "score": 1.0, "tokens_raw": 300, "tokens_final": 300},
            {"event": "context_selection", "task_id": "t", "upstream_id": "b", "score": 0.5, "tokens_raw": 200, "tokens_final": 100},
        ])
        reports = explain_context_trajectory(tmp_path)
        report = reports[0]
        assert report.total_tokens_raw == 500
        assert report.total_tokens_final == 400

    def test_upstreams_evicted_only_counts_zero_tokens_final_and_trimmed(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_budget_trim", "task_id": "t", "upstream_id": "a", "original_tokens": 500, "trimmed_tokens": 0},    # evicted
            {"event": "context_budget_trim", "task_id": "t", "upstream_id": "b", "original_tokens": 400, "trimmed_tokens": 100},  # partial trim
            {"event": "context_selection", "task_id": "t", "upstream_id": "c", "score": 1.0, "tokens_raw": 200, "tokens_final": 200},  # untouched
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].upstreams_evicted == 1

    def test_budget_tokens_none_when_not_in_events(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 1.0, "tokens_raw": 100, "tokens_final": 100},
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].budget_tokens is None


# ---------------------------------------------------------------------------
# Alternate field names
# ---------------------------------------------------------------------------


class TestAlternateFieldNames:
    def test_upstream_alt_field_in_selection(self, tmp_path: Path) -> None:
        """'upstream' is accepted in place of 'upstream_id'."""
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "task",
                "upstream": "build",
                "score": 1.0,
                "tokens_raw": 100,
                "tokens_final": 100,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].entries[0].upstream_id == "build"

    def test_intent_keywords_alt_field(self, tmp_path: Path) -> None:
        """'intent_keywords' is accepted in place of 'keywords_matched'."""
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "task",
                "upstream_id": "up",
                "score": 1.0,
                "intent_keywords": ["token", "budget"],
                "tokens_raw": 100,
                "tokens_final": 100,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].entries[0].keywords_matched == ["token", "budget"]

    def test_original_tokens_alt_field(self, tmp_path: Path) -> None:
        """'original_tokens' is accepted in place of 'tokens_raw'."""
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "task",
                "upstream_id": "up",
                "score": 1.0,
                "original_tokens": 750,
                "tokens_final": 400,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].entries[0].tokens_raw == 750

    def test_trimmed_tokens_alt_field(self, tmp_path: Path) -> None:
        """'trimmed_tokens' is accepted in place of 'tokens_final'."""
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "task",
                "upstream_id": "up",
                "score": 1.0,
                "tokens_raw": 800,
                "trimmed_tokens": 350,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].entries[0].tokens_final == 350

    def test_reason_alt_field(self, tmp_path: Path) -> None:
        """'reason' is accepted in place of 'trim_reason'."""
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "task",
                "upstream_id": "up",
                "score": 1.0,
                "tokens_raw": 100,
                "tokens_final": 50,
                "trimmed": True,
                "reason": "priority_evict",
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        assert reports[0].entries[0].trim_reason == "priority_evict"


# ---------------------------------------------------------------------------
# Formatter: format_context_trajectory
# ---------------------------------------------------------------------------


class TestFormatContextTrajectory:
    def test_empty_list_returns_fallback_string(self) -> None:
        result = format_context_trajectory([])
        assert result == "(no context trajectory events found)"

    def test_text_contains_task_id_and_upstream(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "review",
                "context_raw_tokens": 800,
                "context_final_tokens": 400,
                "entries": [
                    {
                        "upstream_id": "build",
                        "score": 1.5,
                        "keywords_matched": ["compile"],
                        "hop_distance": 1,
                        "hop_decay_factor": 1.0,
                        "tokens_raw": 800,
                        "tokens_final": 400,
                        "trimmed": True,
                        "trim_reason": "budget_trim",
                    }
                ],
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        text = format_context_trajectory(reports)
        assert "Task: review" in text
        assert "build" in text
        assert "compile" in text
        assert "budget_trim" in text

    def test_totals_line_format(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "task",
                "context_raw_tokens": 1000,
                "context_final_tokens": 500,
                "budget_tokens": 600,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        text = format_context_trajectory(reports)
        assert "raw=1000" in text
        assert "final=500" in text
        assert "budget=600" in text
        assert "evicted=0" in text

    def test_no_budget_shows_na(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 1.0, "tokens_raw": 100, "tokens_final": 100},
        ])
        reports = explain_context_trajectory(tmp_path)
        text = format_context_trajectory(reports)
        assert "budget=n/a" in text

    def test_no_entries_shows_placeholder(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "task",
                "context_raw_tokens": 500,
                "context_final_tokens": 200,
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        text = format_context_trajectory(reports)
        assert "(no upstream context entries)" in text

    def test_multiple_reports_both_present(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "alpha", "upstream_id": "x", "score": 1.0, "tokens_raw": 50, "tokens_final": 50},
            {"event": "context_selection", "task_id": "beta", "upstream_id": "y", "score": 1.0, "tokens_raw": 50, "tokens_final": 50},
        ])
        reports = explain_context_trajectory(tmp_path)
        text = format_context_trajectory(reports)
        assert "Task: alpha" in text
        assert "Task: beta" in text

    def test_empty_keywords_renders_dash(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "t",
                "upstream_id": "up",
                "score": 0.5,
                "keywords_matched": [],
                "tokens_raw": 100,
                "tokens_final": 100,
                "trimmed": False,
                "trim_reason": "",
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        text = format_context_trajectory(reports)
        assert "| -" in text

    def test_empty_trim_reason_renders_dash(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_selection",
                "task_id": "t",
                "upstream_id": "up",
                "score": 0.5,
                "tokens_raw": 100,
                "tokens_final": 100,
                "trimmed": False,
                "trim_reason": "",
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        text = format_context_trajectory(reports)
        assert "| -" in text

    def test_trim_yes_no_rendering(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "trimmed_up", "score": 1.0, "tokens_raw": 200, "tokens_final": 0, "trimmed": True, "trim_reason": "budget_trim"},
            {"event": "context_selection", "task_id": "t", "upstream_id": "kept_up", "score": 0.5, "tokens_raw": 100, "tokens_final": 100, "trimmed": False},
        ])
        reports = explain_context_trajectory(tmp_path)
        text = format_context_trajectory(reports)
        assert "yes" in text
        assert "no" in text

    def test_header_row_present(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 1.0, "tokens_raw": 100, "tokens_final": 100},
        ])
        reports = explain_context_trajectory(tmp_path)
        text = format_context_trajectory(reports)
        assert "upstream" in text
        assert "score" in text
        assert "keywords" in text


# ---------------------------------------------------------------------------
# Formatter: format_context_trajectory_json
# ---------------------------------------------------------------------------


class TestFormatContextTrajectoryJson:
    def test_empty_list_returns_empty_array(self) -> None:
        result = format_context_trajectory_json([])
        payload = json.loads(result)
        assert payload == []

    def test_valid_json_structure(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "qa",
                "context_raw_tokens": 300,
                "context_final_tokens": 150,
                "budget_tokens": 200,
                "entries": [
                    {
                        "upstream_id": "build",
                        "score": 1.2,
                        "keywords_matched": ["test"],
                        "hop_distance": 1,
                        "hop_decay_factor": 1.0,
                        "tokens_raw": 300,
                        "tokens_final": 150,
                        "trimmed": False,
                        "trim_reason": "",
                    }
                ],
            }
        ])
        reports = explain_context_trajectory(tmp_path)
        payload = json.loads(format_context_trajectory_json(reports))
        assert isinstance(payload, list)
        obj = payload[0]
        assert obj["task_id"] == "qa"
        assert obj["total_tokens_raw"] == 300
        assert obj["total_tokens_final"] == 150
        assert obj["budget_tokens"] == 200
        assert obj["upstreams_evicted"] == 0
        entry = obj["entries"][0]
        assert entry["upstream_id"] == "build"
        assert entry["score"] == pytest.approx(1.2)
        assert entry["keywords_matched"] == ["test"]
        assert entry["hop_distance"] == 1
        assert entry["trimmed"] is False

    def test_budget_none_serialized_as_null(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "t", "upstream_id": "up", "score": 1.0, "tokens_raw": 100, "tokens_final": 100},
        ])
        reports = explain_context_trajectory(tmp_path)
        payload = json.loads(format_context_trajectory_json(reports))
        assert payload[0]["budget_tokens"] is None

    def test_multiple_tasks_all_present(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {"event": "context_selection", "task_id": "a", "upstream_id": "x", "score": 1.0, "tokens_raw": 50, "tokens_final": 50},
            {"event": "context_selection", "task_id": "b", "upstream_id": "y", "score": 1.0, "tokens_raw": 50, "tokens_final": 50},
        ])
        reports = explain_context_trajectory(tmp_path)
        payload = json.loads(format_context_trajectory_json(reports))
        assert len(payload) == 2
        task_ids = {obj["task_id"] for obj in payload}
        assert task_ids == {"a", "b"}


# ---------------------------------------------------------------------------
# explain_context alias
# ---------------------------------------------------------------------------


class TestExplainContextAlias:
    def test_explain_context_returns_same_result(self, tmp_path: Path) -> None:
        _write_events(tmp_path, [
            {
                "event": "context_compression",
                "task_id": "alias-test",
                "context_raw_tokens": 400,
                "context_final_tokens": 200,
                "entries": [
                    {"upstream_id": "src", "score": 0.8, "tokens_raw": 400, "tokens_final": 200},
                ],
            }
        ])
        from_trajectory = explain_context_trajectory(tmp_path)
        from_alias = explain_context(tmp_path)
        assert len(from_trajectory) == len(from_alias) == 1
        assert from_trajectory[0].task_id == from_alias[0].task_id
        assert from_trajectory[0].total_tokens_raw == from_alias[0].total_tokens_raw
        assert from_trajectory[0].total_tokens_final == from_alias[0].total_tokens_final


# ---------------------------------------------------------------------------
# ContextTrajectoryReport / ContextSelectionEntry to_dict
# ---------------------------------------------------------------------------


class TestModelToDict:
    def test_context_selection_entry_to_dict(self) -> None:
        entry = ContextSelectionEntry(
            upstream_id="build",
            score=1.25,
            keywords_matched=["api", "test"],
            hop_distance=2,
            hop_decay_factor=0.8,
            tokens_raw=500,
            tokens_final=300,
            trimmed=True,
            trim_reason="budget_trim",
        )
        d = entry.to_dict()
        assert d["upstream_id"] == "build"
        assert d["score"] == pytest.approx(1.25)
        assert d["keywords_matched"] == ["api", "test"]
        assert d["hop_distance"] == 2
        assert d["hop_decay_factor"] == pytest.approx(0.8)
        assert d["tokens_raw"] == 500
        assert d["tokens_final"] == 300
        assert d["trimmed"] is True
        assert d["trim_reason"] == "budget_trim"

    def test_context_trajectory_report_to_dict(self) -> None:
        entry = ContextSelectionEntry(upstream_id="x", score=0.5, tokens_raw=100, tokens_final=50)
        report = ContextTrajectoryReport(
            task_id="qa",
            entries=[entry],
            total_tokens_raw=100,
            total_tokens_final=50,
            budget_tokens=200,
            upstreams_evicted=1,
        )
        d = report.to_dict()
        assert d["task_id"] == "qa"
        assert d["total_tokens_raw"] == 100
        assert d["total_tokens_final"] == 50
        assert d["budget_tokens"] == 200
        assert d["upstreams_evicted"] == 1
        assert len(d["entries"]) == 1

    def test_context_trajectory_report_budget_none_serialized(self) -> None:
        report = ContextTrajectoryReport(task_id="t")
        d = report.to_dict()
        assert d["budget_tokens"] is None
        assert d["entries"] == []
        assert d["upstreams_evicted"] == 0

    def test_score_rounded_to_4_decimals(self) -> None:
        entry = ContextSelectionEntry(upstream_id="up", score=1.23456789)
        d = entry.to_dict()
        assert d["score"] == 1.2346

    def test_hop_decay_factor_rounded_to_4_decimals(self) -> None:
        entry = ContextSelectionEntry(upstream_id="up", hop_decay_factor=0.800001)
        d = entry.to_dict()
        assert d["hop_decay_factor"] == 0.8
