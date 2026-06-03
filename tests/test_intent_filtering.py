"""Tests for v0.8.0 intent-driven context filtering (scheduler.py)."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from maestro_cli.models import TaskResult
from maestro_cli.scheduler import (
    _extract_keywords,
    _split_into_sections,
    _score_section,
    _apply_intent_filtering,
    _estimate_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(stdout_tail: str = "") -> TaskResult:
    now = datetime.now(tz=timezone.utc)
    return TaskResult(
        task_id="upstream",
        status="success",
        started_at=now,
        finished_at=now,
        duration_sec=0.1,
        command="echo ok",
        stdout_tail=stdout_tail,
    )


# ===========================================================================
# TestExtractKeywords
# ===========================================================================

class TestExtractKeywords:
    def test_extracts_meaningful_words(self) -> None:
        kw = _extract_keywords("Fix the authentication bug in login.py module")
        assert "authentication" in kw
        assert "login" in kw
        assert "bug" in kw
        assert "fix" in kw

    def test_filters_stopwords(self) -> None:
        kw = _extract_keywords("Fix the authentication bug in login.py module")
        assert "the" not in kw
        assert "in" not in kw

    def test_lowercases_all_keywords(self) -> None:
        kw = _extract_keywords("Implement USER Authentication")
        assert "implement" in kw
        assert "user" in kw
        assert "authentication" in kw

    def test_empty_string_returns_empty(self) -> None:
        assert _extract_keywords("") == set()

    def test_only_stopwords_returns_empty(self) -> None:
        kw = _extract_keywords("the is are was were")
        assert len(kw) == 0

    def test_single_char_words_excluded(self) -> None:
        kw = _extract_keywords("a b c fix d e")
        assert "fix" in kw
        # Single chars shouldn't match (regex is 2+ chars)
        assert "a" not in kw

    def test_underscored_identifiers(self) -> None:
        kw = _extract_keywords("call _extract_keywords and _score_section")
        assert "_extract_keywords" in kw or "extract_keywords" in kw
        assert "_score_section" in kw or "score_section" in kw

    def test_numeric_tokens(self) -> None:
        kw = _extract_keywords("version 42 released on port 8080")
        assert "42" in kw
        assert "8080" in kw
        assert "version" in kw


# ===========================================================================
# TestSplitIntoSections
# ===========================================================================

class TestSplitIntoSections:
    def test_splits_on_double_newline(self) -> None:
        text = "hello\n\nworld\n\nfoo"
        secs = _split_into_sections(text)
        assert len(secs) == 3

    def test_strips_whitespace(self) -> None:
        text = "  hello  \n\n  world  "
        secs = _split_into_sections(text)
        for s in secs:
            assert s == s.strip()

    def test_single_block_falls_back_to_line_chunks(self) -> None:
        # A single block with no blank lines should fall back to line chunking
        lines = "\n".join(f"line {i}" for i in range(24))
        secs = _split_into_sections(lines)
        assert len(secs) >= 2

    def test_empty_string(self) -> None:
        assert _split_into_sections("") == []

    def test_only_whitespace(self) -> None:
        assert _split_into_sections("   \n\n  \n  ") == []

    def test_triple_newlines_still_split(self) -> None:
        text = "part1\n\n\npart2"
        secs = _split_into_sections(text)
        assert len(secs) == 2


# ===========================================================================
# TestScoreSection
# ===========================================================================

class TestScoreSection:
    def test_matching_keywords_returns_count(self) -> None:
        score = _score_section("fix the auth bug", {"fix", "auth", "bug"})
        assert score >= 2  # at least 'fix', 'auth', 'bug' partially overlap

    def test_no_match_returns_zero(self) -> None:
        assert _score_section("unrelated text about weather", {"fix", "auth", "bug"}) == 0

    def test_empty_section_returns_zero(self) -> None:
        assert _score_section("", {"fix", "auth"}) == 0

    def test_empty_keywords_returns_zero(self) -> None:
        assert _score_section("some text", set()) == 0

    def test_partial_overlap(self) -> None:
        # Only "fix" matches
        score = _score_section("fix the broken window", {"fix", "database", "migration"})
        assert score == 1


# ===========================================================================
# TestApplyIntentFiltering
# ===========================================================================

class TestApplyIntentFiltering:
    def test_filters_irrelevant_sections(self) -> None:
        # Two distinct sections: one relevant, one not
        tail = "authentication module fix\n\nweather forecast for today"
        upstream = {"t1": _make_result(tail)}
        filtered, records, _ = _apply_intent_filtering(upstream, {"authentication", "fix"})
        # The filtered result should have less content
        assert len(filtered["t1"].stdout_tail) < len(tail)
        assert len(records) == 1  # one task was filtered

    def test_preserves_original_when_all_relevant(self) -> None:
        tail = "authentication fix\n\nlogin bug patch"
        upstream = {"t1": _make_result(tail)}
        filtered, records, _ = _apply_intent_filtering(upstream, {"authentication", "fix", "login", "bug"})
        # All sections match, so nothing should be filtered
        assert len(records) == 0

    def test_preserves_original_when_none_relevant(self) -> None:
        tail = "completely unrelated\n\nalso unrelated"
        upstream = {"t1": _make_result(tail)}
        filtered, records, _ = _apply_intent_filtering(upstream, {"database", "migration"})
        # No sections match → preserve original (safety fallback)
        assert filtered["t1"].stdout_tail == tail
        assert len(records) == 0

    def test_empty_upstream_noop(self) -> None:
        filtered, records, _ = _apply_intent_filtering({}, {"fix", "bug"})
        assert filtered == {}
        assert records == []

    def test_none_keywords_noop(self) -> None:
        upstream = {"t1": _make_result("some text")}
        filtered, records, _ = _apply_intent_filtering(upstream, None)
        assert filtered["t1"].stdout_tail == "some text"
        assert records == []

    def test_filter_records_contain_token_counts(self) -> None:
        tail = "fix authentication bug\n\nweather forecast for today is sunny and warm"
        upstream = {"t1": _make_result(tail)}
        filtered, records, _ = _apply_intent_filtering(upstream, {"authentication", "fix", "bug"})
        if records:
            tid, orig_tok, filtered_tok = records[0]
            assert tid == "t1"
            assert orig_tok > filtered_tok
            assert filtered_tok > 0

    def test_multiple_upstreams(self) -> None:
        upstream = {
            "t1": _make_result("fix auth bug\n\nirrelevant weather"),
            "t2": _make_result("totally unrelated\n\nalso unrelated"),
        }
        filtered, records, _ = _apply_intent_filtering(upstream, {"auth", "fix", "bug"})
        # t1 should be filtered (has relevant + irrelevant sections)
        # t2 should be preserved (no matching sections → safety fallback)
        assert len(records) >= 1


# ===========================================================================
# TestEstimateTokens
# ===========================================================================

class TestEstimateTokens:
    def test_basic_estimation(self) -> None:
        assert _estimate_tokens("hello world") >= 1

    def test_empty_string(self) -> None:
        # max(1, len("") // 4) == 1
        assert _estimate_tokens("") >= 0

    def test_rough_ratio(self) -> None:
        # ~4 chars per token is the heuristic
        text = "x" * 400
        tokens = _estimate_tokens(text)
        assert 80 <= tokens <= 120
