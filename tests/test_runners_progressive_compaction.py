from __future__ import annotations

from pathlib import Path

import pytest

import maestro_cli.runners as runners
from maestro_cli.models import StructuredContext
from maestro_cli.runners import _apply_progressive_compaction


@pytest.fixture(autouse=True)
def _reset_summarization_breaker() -> None:
    """Ensure the module-level summarization circuit breaker starts closed.

    ``_apply_progressive_compaction`` reads the global
    ``_summarization_consecutive_failures`` to decide whether to run Stage 2.5.
    Other tests in the suite may have tripped it; reset for determinism.
    """
    runners._summarization_consecutive_failures = 0


# ---------------------------------------------------------------------------
# Stage 2.5 — LLM summarization branch (lines 1149-1164)
# ---------------------------------------------------------------------------


class TestStage25Summarization:
    def test_summarization_brings_within_budget(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stage 2.5 runs when workdir is set + breaker closed, and a short
        summary returned by ``_run_summarization`` drops total under budget,
        returning at stage 2 (line 1164).
        """
        # Force Stage 1 and Stage 2 to be no-ops so they cannot reach budget,
        # guaranteeing we fall through to Stage 2.5.
        monkeypatch.setattr(runners, "_compact_context", lambda text: text)
        monkeypatch.setattr(
            runners, "_prune_low_signal_sections", lambda text, target: text
        )

        captured: list[tuple[str, str]] = []

        def fake_summarize(
            task_id: str,
            stdout_tail: str,
            structured: StructuredContext,
            workdir: Path,
        ) -> str:
            # Validate the stub StructuredContext built at lines 1152-1157.
            assert structured.task_id == task_id
            assert structured.status == "success"
            assert structured.exit_code == 0
            assert structured.duration_sec == 0.0
            captured.append((task_id, stdout_tail))
            return "SUM"  # tiny summary, shorter than the input

        monkeypatch.setattr(runners, "_run_summarization", fake_summarize)

        # Each upstream is large and well above per-upstream budget so the
        # ``len(...) > per_upstream_budget`` guard at line 1151 is True.
        big = "alpha beta gamma delta epsilon zeta\n" * 200
        upstream = {"u1": big, "u2": big}

        # Small budget -> Stages 1/2 (no-op) cannot fit; Stage 2.5 collapses it.
        result, stage = _apply_progressive_compaction(
            upstream,
            budget_tokens=50,
            scores={"u1": 0.1, "u2": 0.9},
            workdir=Path("."),
        )

        assert stage == 2
        # Summarization was actually invoked (covers lines 1158-1162).
        assert ("u1", big) in captured
        # The short summary replaced the original (line 1162).
        assert result["u1"] == "SUM"

    def test_summary_not_shorter_is_kept_original(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the summary is NOT shorter than the input, the replacement at
        line 1162 is skipped (the ``len(summary) < len(...)`` guard is False),
        but the Stage 2.5 loop body still executes (lines 1149-1160).
        """
        monkeypatch.setattr(runners, "_compact_context", lambda text: text)
        monkeypatch.setattr(
            runners, "_prune_low_signal_sections", lambda text, target: text
        )

        # Summary longer than the input -> guard at 1161 False, no replacement.
        monkeypatch.setattr(
            runners,
            "_run_summarization",
            lambda task_id, stdout_tail, structured, workdir: stdout_tail + "X" * 10,
        )

        big = "line of mostly text content here\n" * 150
        upstream = {"only": big}

        result, stage = _apply_progressive_compaction(
            upstream,
            budget_tokens=40,
            scores={"only": 0.5},
            workdir=Path("."),
        )

        # Original preserved (no shorter summary), and since Stage 2.5 could not
        # fit, we progress past it to a later truncation stage.
        assert result["only"].startswith("line of mostly text")
        assert stage >= 3

    def test_breaker_open_skips_stage_25(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the circuit breaker is open, Stage 2.5 is skipped entirely even
        though workdir is set — ``_run_summarization`` must never be called.
        """
        monkeypatch.setattr(runners, "_compact_context", lambda text: text)
        monkeypatch.setattr(
            runners, "_prune_low_signal_sections", lambda text, target: text
        )

        def boom(*args: object, **kwargs: object) -> str:  # pragma: no cover
            raise AssertionError("Stage 2.5 must be skipped when breaker is open")

        monkeypatch.setattr(runners, "_run_summarization", boom)
        runners._summarization_consecutive_failures = (
            runners._SUMMARIZATION_CIRCUIT_BREAKER_THRESHOLD
        )

        big = "some content line goes here padding\n" * 120
        result, stage = _apply_progressive_compaction(
            {"u": big},
            budget_tokens=40,
            scores={"u": 0.5},
            workdir=Path("."),
        )

        # Skipped 2.5 and fell through to truncation / later stages.
        assert stage >= 3
        assert "u" in result

    def test_no_workdir_skips_stage_25(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without a workdir, the Stage 2.5 guard (line 1146) is False so the
        summarization branch is never entered.
        """
        monkeypatch.setattr(runners, "_compact_context", lambda text: text)
        monkeypatch.setattr(
            runners, "_prune_low_signal_sections", lambda text, target: text
        )

        def boom(*args: object, **kwargs: object) -> str:  # pragma: no cover
            raise AssertionError("Stage 2.5 must not run when workdir is None")

        monkeypatch.setattr(runners, "_run_summarization", boom)

        big = "padding content text here line\n" * 120
        _result, stage = _apply_progressive_compaction(
            {"u": big},
            budget_tokens=40,
            scores={"u": 0.5},
            workdir=None,
        )
        assert stage >= 3


# ---------------------------------------------------------------------------
# Post-compact restoration block (lines 1197-1214)
#
# DEAD-CODE NOTE on lines 1201-1214:
#   The restoration body is guarded by ``if max_stage >= 3 ... if remaining >
#   400``.  ``max_stage`` only reaches 3/4/5 when the corresponding stage loop
#   completes WITHOUT an early ``return`` — i.e. ``_total_chars() <=
#   budget_chars`` was False after compacting every upstream in that stage.
#   But ``remaining = budget_chars - _total_chars()`` uses the SAME running
#   total, so reaching line 1199 implies ``_total_chars() > budget_chars`` and
#   therefore ``remaining < 0``.  ``remaining > 400`` can never be True at that
#   point, so lines 1201-1214 are unreachable through the public function.
#   This was verified empirically (200 randomized shapes never satisfied the
#   condition).  The reachable parts of the block are lines 1197-1200 (the
#   guard evaluation), which the tests below exercise for both the
#   ``max_stage >= 3`` fall-through (guard present, inner branch skipped) and
#   the ``max_stage < 3`` early-return path (block never entered).
# ---------------------------------------------------------------------------


class TestPostCompactRestoration:
    def test_fallthrough_to_stage5_evaluates_restoration_guard(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Drive compaction all the way through Stage 5 fall-through so the
        restoration guard at lines 1197-1200 is evaluated.  Because reaching
        line 1199 implies ``_total_chars() > budget_chars`` (see DEAD-CODE NOTE),
        ``remaining > 400`` is False and the inner re-injection is skipped; the
        function returns the fully-collapsed L0 result at stage 5.
        """
        # Make Stages 1/2/3 no-ops so nothing fits before Stage 4/5.
        monkeypatch.setattr(runners, "_compact_context", lambda text: text)
        monkeypatch.setattr(
            runners, "_prune_low_signal_sections", lambda text, target: text
        )
        monkeypatch.setattr(
            runners, "_truncate_with_markers", lambda text, target: text
        )
        # Stage 4 L1 returns text unchanged (still over budget) -> push to 5.
        monkeypatch.setattr(
            runners, "_extract_l1_sections", lambda text, max_chars=0: text
        )
        # Stage 5 L0 returns a per-upstream block large enough that, summed
        # across upstreams, the total stays ABOVE budget even after the last
        # collapse -> the loop never early-returns and falls through to
        # ``max_stage = 5`` and the restoration guard.
        monkeypatch.setattr(
            runners, "_extract_l0_summary", lambda text: "L0BLOCK" * 200
        )

        big = "# Heading one\nbody content line that is meaningful here\n" * 80
        upstream = {"u1": big, "u2": big}
        originals = {
            "u1": "# Important Finding\n- result detail alpha\n" * 30,
            "u2": "# Secondary Note\n- minor detail\n" * 30,
        }

        result, stage = _apply_progressive_compaction(
            upstream,
            budget_tokens=100,  # 400 chars; two ~1400-char L0 blocks exceed it
            scores={"u1": 0.95, "u2": 0.05},
            original_texts=originals,
            workdir=None,
        )

        assert stage == 5
        # Restoration branch skipped (remaining < 0): every upstream stays as
        # its collapsed L0 block, never re-inflated from originals.
        assert result["u1"] == "L0BLOCK" * 200
        assert result["u2"] == "L0BLOCK" * 200

    def test_restoration_block_uses_upstream_texts_when_no_originals(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Covers line 1197 (``originals = original_texts or upstream_texts``)
        when ``original_texts`` is not supplied: the fallback to ``upstream_texts``
        is taken and the guard is still evaluated at Stage 5 fall-through.
        """
        monkeypatch.setattr(runners, "_compact_context", lambda text: text)
        monkeypatch.setattr(
            runners, "_prune_low_signal_sections", lambda text, target: text
        )
        monkeypatch.setattr(
            runners, "_truncate_with_markers", lambda text, target: text
        )
        monkeypatch.setattr(
            runners, "_extract_l1_sections", lambda text, max_chars=0: text
        )
        monkeypatch.setattr(
            runners, "_extract_l0_summary", lambda text: "BLOCK" * 250
        )

        big = "content line that is reasonably long and meaningful\n" * 60
        result, stage = _apply_progressive_compaction(
            {"a": big, "b": big},
            budget_tokens=100,
            scores={"a": 0.4, "b": 0.6},
            original_texts=None,  # forces the `or upstream_texts` fallback
            workdir=None,
        )

        assert stage == 5
        assert result["a"] == "BLOCK" * 250

    def test_no_restoration_when_stage_below_3(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When compaction finishes at stage < 3, the restoration block at
        line 1198 is not entered (guard ``max_stage >= 3`` is False).
        """
        # Stage 1 (compact) reduces enough to fit -> returns stage 1 before any
        # restoration logic.
        monkeypatch.setattr(runners, "_compact_context", lambda text: "x")

        big = "padding content line that is long enough\n" * 100
        result, stage = _apply_progressive_compaction(
            {"u1": big, "u2": big},
            budget_tokens=100,
            scores={"u1": 0.9, "u2": 0.1},
            original_texts={"u1": "# Orig\n- detail\n" * 50},
            workdir=None,
        )

        assert stage == 1
        assert result["u1"] == "x"


# ---------------------------------------------------------------------------
# Early-return guards (sanity baseline)
# ---------------------------------------------------------------------------


class TestEarlyReturns:
    def test_empty_input_returns_stage_zero(self) -> None:
        result, stage = _apply_progressive_compaction({}, budget_tokens=100)
        assert result == {}
        assert stage == 0

    def test_already_within_budget_returns_stage_zero(self) -> None:
        upstream = {"u": "short"}
        result, stage = _apply_progressive_compaction(upstream, budget_tokens=1000)
        assert result == upstream
        assert stage == 0
