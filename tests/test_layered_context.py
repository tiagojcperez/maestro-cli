from __future__ import annotations

import pytest

from maestro_cli.runners import (
    _build_layered_context,
    _extract_l0_summary,
    _extract_l1_sections,
    _format_layered_context_section,
    _L0_TARGET_TOKENS,
    _L1_TARGET_TOKENS,
)


# ---------------------------------------------------------------------------
# Constants sanity
# ---------------------------------------------------------------------------


class TestLayeredContextConstants:
    def test_l0_target_smaller_than_l1(self) -> None:
        assert _L0_TARGET_TOKENS < _L1_TARGET_TOKENS

    def test_l0_reasonable_size(self) -> None:
        # ~50 tokens ≈ 200 chars — fits in a short summary
        assert _L0_TARGET_TOKENS == 50

    def test_l1_reasonable_size(self) -> None:
        # ~200 tokens ≈ 800 chars — fits key sections
        assert _L1_TARGET_TOKENS == 200


# ---------------------------------------------------------------------------
# _extract_l0_summary
# ---------------------------------------------------------------------------


class TestExtractL0Summary:
    def test_returns_first_meaningful_line(self) -> None:
        text = "JWT validation fails on expired refresh tokens.\nSome extra detail."
        result = _extract_l0_summary(text)
        assert "JWT validation fails" in result

    def test_skips_empty_lines(self) -> None:
        text = "\n\nActual content here."
        result = _extract_l0_summary(text)
        assert "Actual content here" in result

    def test_skips_structural_markers(self) -> None:
        # Lines that are only structural chars should be skipped
        text = "---\n```\nReal content line."
        result = _extract_l0_summary(text)
        assert "Real content line" in result

    def test_skips_short_lines(self) -> None:
        # Lines shorter than 10 chars are skipped
        text = "ok\nshort\nThis is a sufficiently long meaningful sentence."
        result = _extract_l0_summary(text)
        assert "sufficiently long" in result

    def test_skips_punctuation_only_lines(self) -> None:
        text = "---***---\nMeaningful text follows here nicely."
        result = _extract_l0_summary(text)
        assert "Meaningful text" in result

    def test_truncates_long_line(self) -> None:
        long_line = "A" * 500
        result = _extract_l0_summary(long_line)
        # Max chars = _L0_TARGET_TOKENS * 4 = 200
        assert len(result) <= _L0_TARGET_TOKENS * 4

    def test_empty_text_returns_fallback(self) -> None:
        result = _extract_l0_summary("")
        assert result == "(empty output)"

    def test_all_structural_lines_falls_back(self) -> None:
        text = "---\n```\n{}\n[]"
        result = _extract_l0_summary(text)
        # Should still return something (fallback truncation or empty marker)
        assert result

    def test_single_good_line(self) -> None:
        text = "Schema mismatch on /sessions endpoint detected."
        result = _extract_l0_summary(text)
        assert "Schema mismatch" in result

    def test_result_is_stripped(self) -> None:
        text = "   padded content line here   "
        result = _extract_l0_summary(text)
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_first_meaningful_wins_over_longer_later_line(self) -> None:
        text = "First valid line of output.\nSecond line is much longer and different."
        result = _extract_l0_summary(text)
        assert "First valid line" in result

    @pytest.mark.parametrize(
        "marker",
        ["{", "}", "[", "]", "```", "---"],
    )
    def test_each_structural_marker_skipped(self, marker: str) -> None:
        text = f"{marker}\nReal content follows here."
        result = _extract_l0_summary(text)
        assert "Real content follows" in result


# ---------------------------------------------------------------------------
# _extract_l1_sections
# ---------------------------------------------------------------------------


class TestExtractL1Sections:
    def test_captures_markdown_headings(self) -> None:
        text = "## Summary\nThis is the summary content.\n## Details\nSome details."
        result = _extract_l1_sections(text)
        assert "## Summary" in result
        assert "## Details" in result

    def test_captures_next_line_after_heading(self) -> None:
        text = "## Auth Issues\nJWT tokens expire too fast.\nMore verbose info."
        result = _extract_l1_sections(text)
        assert "JWT tokens expire too fast" in result

    def test_does_not_capture_second_body_line_under_heading(self) -> None:
        text = "## Section\nFirst body line.\nSecond body line."
        result = _extract_l1_sections(text)
        # The second body line after the first one should not be captured
        assert "Second body line" not in result

    def test_captures_error_prefix(self) -> None:
        text = "Error: connection timeout after 30s\nNot a signal line."
        result = _extract_l1_sections(text)
        assert "Error: connection timeout" in result

    def test_captures_result_prefix(self) -> None:
        text = "Result: 3 tests failed\nSome explanation."
        result = _extract_l1_sections(text)
        assert "Result: 3 tests failed" in result

    def test_captures_output_prefix(self) -> None:
        text = "Output: success\nFull log here."
        result = _extract_l1_sections(text)
        assert "Output: success" in result

    def test_captures_status_prefix(self) -> None:
        text = "Status: degraded\nAll systems partial."
        result = _extract_l1_sections(text)
        assert "Status: degraded" in result

    def test_captures_bullet_dash(self) -> None:
        text = "- Fixed authentication bug\n- Added retry logic"
        result = _extract_l1_sections(text)
        assert "- Fixed authentication bug" in result

    def test_captures_bullet_star(self) -> None:
        text = "* Rate limit hit at 100 RPS\n* Backoff enabled"
        result = _extract_l1_sections(text)
        assert "* Rate limit hit at 100 RPS" in result

    def test_respects_max_chars_budget(self) -> None:
        # Generate a document with many headings
        text = "\n".join(f"## Section {i}\nContent line for section {i}." for i in range(50))
        result = _extract_l1_sections(text, max_chars=100)
        assert len(result) <= 100

    def test_empty_text_returns_empty_marker(self) -> None:
        result = _extract_l1_sections("")
        assert result == "(empty output)"

    def test_no_signal_lines_falls_back_to_truncation(self) -> None:
        # Plain prose with no headings or signal prefixes
        text = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        result = _extract_l1_sections(text, max_chars=200)
        # Should still return something (fallback truncation)
        assert result
        assert len(result) <= 200

    def test_heading_indents_following_body_line(self) -> None:
        text = "## My Section\nThe body of the section."
        result = _extract_l1_sections(text)
        # Body line should be indented by two spaces
        assert "  The body of the section." in result

    def test_does_not_include_plain_prose_lines(self) -> None:
        text = "This is just plain prose.\nMore plain prose.\n## Found It\nCapture this."
        result = _extract_l1_sections(text)
        assert "just plain prose" not in result
        assert "## Found It" in result

    def test_default_max_chars_matches_l1_target(self) -> None:
        # Default max_chars should be _L1_TARGET_TOKENS * 4 = 800
        large_text = "\n".join(f"## Heading {i}\nBody {i}" for i in range(200))
        result = _extract_l1_sections(large_text)
        assert len(result) <= _L1_TARGET_TOKENS * 4


# ---------------------------------------------------------------------------
# _format_layered_context_section
# ---------------------------------------------------------------------------


class TestFormatLayeredContextSection:
    def test_basic_format(self) -> None:
        result = _format_layered_context_section("task-a", "some content")
        assert result == "--- task-a ---\nsome content"

    def test_header_separator(self) -> None:
        result = _format_layered_context_section("upstream-id", "body text")
        assert result.startswith("--- upstream-id ---\n")

    def test_body_preserved_verbatim(self) -> None:
        body = "line1\nline2\nline3"
        result = _format_layered_context_section("x", body)
        assert body in result

    def test_empty_body(self) -> None:
        result = _format_layered_context_section("task-x", "")
        assert result == "--- task-x ---\n"

    def test_multiline_id_not_expected_but_safe(self) -> None:
        result = _format_layered_context_section("my-task", "content")
        assert "--- my-task ---" in result


# ---------------------------------------------------------------------------
# _build_layered_context — core algorithm
# ---------------------------------------------------------------------------


class TestBuildLayeredContextEmpty:
    def test_empty_upstream_dict(self) -> None:
        result = _build_layered_context({}, budget_tokens=1000)
        assert result == ""

    def test_zero_budget(self) -> None:
        result = _build_layered_context({"a": "content"}, budget_tokens=0)
        assert result == ""

    def test_negative_budget(self) -> None:
        result = _build_layered_context({"a": "content"}, budget_tokens=-10)
        assert result == ""


class TestBuildLayeredContextSingleUpstream:
    def test_tiny_budget_produces_l0(self) -> None:
        # L0 ≈ 50 tokens ≈ 200 chars; budget = 60 tokens = 240 chars
        text = "## Auth\nJWT fails on refresh.\n## Details\nLong verbose explanation " * 20
        result = _build_layered_context({"auth": text}, budget_tokens=60)
        # Should be short (L0 or truncated L1)
        assert len(result) <= 60 * 4 + 20  # +20 for header overhead
        assert "--- auth ---" in result

    def test_medium_budget_includes_sections(self) -> None:
        # L1 ≈ 200 tokens ≈ 800 chars; give enough budget for L1
        text = "## Summary\nJWT validation fails.\n## Details\n" + "verbose " * 100
        result = _build_layered_context({"auth": text}, budget_tokens=300)
        assert "## Summary" in result or "JWT validation fails" in result
        assert "--- auth ---" in result

    def test_large_budget_includes_full_output(self) -> None:
        # L2 = full text; give generous budget
        text = "## Summary\nJWT validation fails on expired refresh tokens.\n## Details\nFull details here."
        result = _build_layered_context({"auth": text}, budget_tokens=5000)
        assert "Full details here" in result
        assert "--- auth ---" in result

    def test_section_header_format(self) -> None:
        result = _build_layered_context({"scan-auth": "important result"}, budget_tokens=500)
        assert "--- scan-auth ---" in result

    def test_empty_upstream_content_shows_marker(self) -> None:
        result = _build_layered_context({"empty-task": ""}, budget_tokens=500)
        assert "--- empty-task ---" in result
        # Should contain the empty output marker
        assert "(empty output)" in result


class TestBuildLayeredContextMultipleUpstreams:
    def test_sections_separated_by_double_newline(self) -> None:
        upstreams = {
            "task-a": "Content from task A.",
            "task-b": "Content from task B.",
        }
        result = _build_layered_context(upstreams, budget_tokens=5000)
        assert "--- task-a ---" in result
        assert "--- task-b ---" in result
        # Sections are separated by \n\n
        assert "\n\n" in result

    def test_both_sections_present_with_generous_budget(self) -> None:
        upstreams = {
            "scan-auth": "JWT validation fails on expired refresh tokens.",
            "scan-api": "Schema mismatch on /sessions endpoint.",
        }
        result = _build_layered_context(upstreams, budget_tokens=5000)
        assert "JWT validation fails" in result
        assert "Schema mismatch" in result

    def test_total_size_within_budget(self) -> None:
        # Budget = 200 tokens = 800 chars
        upstreams = {
            f"task-{i}": f"Task {i} output: " + "x" * 300
            for i in range(5)
        }
        budget_tokens = 200
        result = _build_layered_context(upstreams, budget_tokens=budget_tokens)
        assert len(result) <= budget_tokens * 4 + 100  # +100 for header overhead

    def test_compressed_output_smaller_than_raw(self) -> None:
        long_text = "verbose content line\n" * 200
        upstreams = {"a": long_text, "b": long_text}
        raw_size = len(long_text) * 2
        result = _build_layered_context(upstreams, budget_tokens=500)
        assert len(result) < raw_size

    def test_three_upstreams_all_included_with_big_budget(self) -> None:
        upstreams = {
            "t1": "First task result.",
            "t2": "Second task result.",
            "t3": "Third task result.",
        }
        result = _build_layered_context(upstreams, budget_tokens=5000)
        assert "--- t1 ---" in result
        assert "--- t2 ---" in result
        assert "--- t3 ---" in result


class TestBuildLayeredContextScoreOrdering:
    def test_high_score_upstream_appears_first(self) -> None:
        upstreams = {
            "low-relevance": "Low relevance content.",
            "high-relevance": "High relevance content.",
        }
        scores = {"low-relevance": 0.1, "high-relevance": 0.9}
        result = _build_layered_context(upstreams, budget_tokens=5000, scores=scores)
        pos_high = result.find("--- high-relevance ---")
        pos_low = result.find("--- low-relevance ---")
        assert pos_high < pos_low

    def test_score_ordering_affects_budget_priority(self) -> None:
        # Tight budget — only the most relevant upstream should be kept
        long_text = "x" * 400
        upstreams = {
            "irrelevant": long_text,
            "relevant": "The critical finding is: buffer overflow on line 42.",
        }
        scores = {"irrelevant": 0.0, "relevant": 1.0}
        # Budget just enough for one full upstream + headers
        result = _build_layered_context(upstreams, budget_tokens=50, scores=scores)
        # The relevant one should appear; irrelevant may be trimmed
        assert "--- relevant ---" in result

    def test_no_scores_uses_alphabetical_fallback(self) -> None:
        upstreams = {"beta": "Beta content.", "alpha": "Alpha content."}
        result = _build_layered_context(upstreams, budget_tokens=5000, scores=None)
        pos_alpha = result.find("--- alpha ---")
        pos_beta = result.find("--- beta ---")
        # Without scores, sorted alphabetically
        assert pos_alpha < pos_beta

    def test_equal_scores_uses_id_as_tiebreaker(self) -> None:
        upstreams = {"zzz": "Content Z.", "aaa": "Content A."}
        scores = {"zzz": 0.5, "aaa": 0.5}
        result = _build_layered_context(upstreams, budget_tokens=5000, scores=scores)
        # With equal scores, sort by id (ascending)
        pos_aaa = result.find("--- aaa ---")
        pos_zzz = result.find("--- zzz ---")
        assert pos_aaa < pos_zzz


class TestBuildLayeredContextTierProgression:
    def test_l0_then_l1_upgrade_when_budget_grows(self) -> None:
        text_with_heading = "## Summary\nJWT fails on refresh.\n" + "verbose detail " * 50
        # Very tight budget → L0 only
        result_l0 = _build_layered_context({"a": text_with_heading}, budget_tokens=20)
        # Moderate budget → L1 (includes heading)
        result_l1 = _build_layered_context({"a": text_with_heading}, budget_tokens=120)
        # L1 should contain more content
        assert len(result_l1) >= len(result_l0)

    def test_l1_then_l2_upgrade_when_budget_grows(self) -> None:
        text = "## Section\nKey finding.\n" + "full verbose output " * 30
        result_l1 = _build_layered_context({"a": text}, budget_tokens=120)
        result_l2 = _build_layered_context({"a": text}, budget_tokens=2000)
        # L2 should contain the verbose output
        assert len(result_l2) > len(result_l1)
        assert "full verbose output" in result_l2

    def test_budget_tight_evicts_low_score_upstreams(self) -> None:
        # Three upstreams; budget only fits ~1-2
        upstreams = {
            "high": "Important: critical buffer overflow found in auth module.",
            "mid": "Some moderate finding here.",
            "low": "Low priority info with minor note.",
        }
        scores = {"high": 1.0, "mid": 0.5, "low": 0.1}
        # Very tight budget — 30 tokens ≈ 120 chars
        result = _build_layered_context(upstreams, budget_tokens=30, scores=scores)
        assert len(result) <= 30 * 4 + 60  # generous overhead for headers
        # High-score upstream should be present
        assert "--- high ---" in result

    def test_all_l2_with_very_large_budget(self) -> None:
        text_a = "## Report\nJWT expired.\n" + "detail " * 50
        text_b = "## Findings\nSQL injection risk.\n" + "evidence " * 50
        upstreams = {"scan-a": text_a, "scan-b": text_b}
        result = _build_layered_context(upstreams, budget_tokens=10_000)
        # L2: full text should be in the result
        assert "detail " in result
        assert "evidence " in result

    def test_tier_promotion_is_score_ordered(self) -> None:
        # Two upstreams with different scores; medium budget fits L1 for one, L0 for the other
        text = "## Section\nBody of section.\n" + "more content " * 30
        upstreams = {"low": text, "high": text}
        scores = {"low": 0.1, "high": 0.9}
        # Budget enough for ~1.5× L1 overhead
        result = _build_layered_context(upstreams, budget_tokens=150, scores=scores)
        # High-score upstream should have richer content (L1+) than low-score
        high_start = result.find("--- high ---")
        low_start = result.find("--- low ---")
        assert high_start != -1
        # Both present or at least high is present
        if low_start != -1:
            high_section = result[high_start:low_start] if high_start < low_start else result[high_start:]
            low_section = result[low_start:high_start] if low_start < high_start else result[low_start:]
            # High-score section should be at least as long as low-score
            assert len(high_section) >= len(low_section)


class TestBuildLayeredContextEdgeCases:
    def test_single_char_content_still_returns_section(self) -> None:
        result = _build_layered_context({"t": "x" * 15}, budget_tokens=100)
        assert "--- t ---" in result

    def test_very_long_single_upstream_gets_truncated_to_budget(self) -> None:
        huge = "x" * 100_000
        budget_tokens = 100
        result = _build_layered_context({"giant": huge}, budget_tokens=budget_tokens)
        # Must fit within budget (with some header overhead tolerance)
        assert len(result) <= budget_tokens * 4 + 30

    def test_scores_dict_can_be_empty(self) -> None:
        result = _build_layered_context({"a": "content"}, budget_tokens=500, scores={})
        assert "--- a ---" in result

    def test_whitespace_only_content_replaced_with_marker(self) -> None:
        result = _build_layered_context({"t": "   \n\t\n  "}, budget_tokens=500)
        assert "--- t ---" in result
        assert "(empty output)" in result

    def test_upstream_with_only_structural_markers(self) -> None:
        result = _build_layered_context({"t": "---\n```\n{}"}, budget_tokens=500)
        # Should still include section header; content may be L0 fallback
        assert "--- t ---" in result

    def test_output_never_contains_double_double_newline(self) -> None:
        upstreams = {"a": "content a", "b": "content b", "c": "content c"}
        result = _build_layered_context(upstreams, budget_tokens=5000)
        # Sections are separated by exactly \n\n, not \n\n\n
        assert "\n\n\n" not in result

    def test_many_upstreams_all_within_budget(self) -> None:
        # 10 small upstreams with generous budget
        upstreams = {f"t{i}": f"Result {i}: all good." for i in range(10)}
        result = _build_layered_context(upstreams, budget_tokens=5000)
        for i in range(10):
            assert f"--- t{i} ---" in result

    @pytest.mark.parametrize("budget_tokens", [1, 5, 10, 25, 50, 100, 200, 500, 1000])
    def test_output_stays_within_budget_across_sizes(self, budget_tokens: int) -> None:
        upstreams = {
            "a": "## Summary\nAuthentication failed.\n" + "detail " * 100,
            "b": "## Findings\nSQL injection risk.\n" + "evidence " * 100,
            "c": "## Notes\nMinor style issues.\n" + "note " * 100,
        }
        result = _build_layered_context(upstreams, budget_tokens=budget_tokens)
        # Result must fit in budget (+generous header tolerance)
        assert len(result) <= budget_tokens * 4 + 100


class TestBuildLayeredContextCompressionRatio:
    def test_compression_vs_raw_multi_upstream(self) -> None:
        """Layered context achieves meaningful token savings vs concatenated raw output."""
        upstreams = {
            "scan-auth": "## JWT Analysis\nExpiry check missing.\n" + "verbose " * 200,
            "scan-api": "## API Audit\nSchema drift detected.\n" + "details " * 200,
            "scan-db": "## DB Review\nIndex missing on users table.\n" + "info " * 200,
        }
        raw_size = sum(len(v) for v in upstreams.values())
        # 500 token budget ≈ 2000 chars — well under raw size of ~4800+
        result = _build_layered_context(upstreams, budget_tokens=500)
        assert len(result) < raw_size
        # Should still contain all three section headers
        for key in upstreams:
            assert f"--- {key} ---" in result

    def test_lossless_when_budget_exceeds_total_size(self) -> None:
        """When budget is ample, the full content of all upstreams is included."""
        text_a = "JWT validation fails on expired refresh tokens."
        text_b = "Schema mismatch on /sessions endpoint."
        upstreams = {"a": text_a, "b": text_b}
        result = _build_layered_context(upstreams, budget_tokens=10_000)
        assert text_a in result
        assert text_b in result
