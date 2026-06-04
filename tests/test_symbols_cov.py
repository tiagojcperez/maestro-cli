"""Coverage tests for maestro_cli.symbols targeting specific uncovered branches.

Targets:
  - import-symbol extraction via name_group == 0 (uses full stripped line)
  - _score_chunk early-return on empty target set
  - build_structural_context no-symbols fallback truncation loop break
  - build_structural_context greedy-selection budget-skip continue
"""

from __future__ import annotations

from maestro_cli.symbols import (
    _score_chunk,
    build_structural_context,
    extract_symbols,
)


# ===========================================================================
# extract_symbols: name_group == 0 path (import without named capture)
# ===========================================================================


class TestExtractSymbolsImportNameGroupZero:
    def test_javascript_import_uses_full_stripped_line(self) -> None:
        # The javascript "import" pattern matches `^\s*import\s+` with
        # name_group == 0, so the symbol name is taken from the full
        # stripped line rather than a capture group.
        text = "    import foo from 'bar';\nconst x = function() {};\n"
        symbols = extract_symbols(text, language="javascript")

        names = [s.name for s in symbols]
        kinds = {s.kind for s in symbols}
        # The import line collapses leading whitespace and yields the
        # stripped source line as the symbol name.
        assert "import foo from 'bar';" in names
        assert "import" in kinds
        # The matched import symbol carries the language tag.
        import_syms = [s for s in symbols if s.kind == "import"]
        assert import_syms
        assert import_syms[0].language == "javascript"
        assert import_syms[0].line_start == 1

    def test_go_import_line_captured_as_full_line(self) -> None:
        # Go's import pattern also uses name_group == 0.
        text = "import \"fmt\"\nfunc Main() {}\n"
        symbols = extract_symbols(text, language="go")
        import_names = [s.name for s in symbols if s.kind == "import"]
        assert any("import" in n for n in import_names)
        # The stored name is the full stripped line.
        assert 'import "fmt"' in import_names


# ===========================================================================
# _score_chunk: empty target_symbols -> 0.0
# ===========================================================================


class TestScoreChunkEmptyTargets:
    def test_empty_target_symbols_returns_zero(self) -> None:
        # With no target symbols, the function short-circuits to 0.0.
        score = _score_chunk("def foo(): pass", set())
        assert score == 0.0

    def test_nonempty_targets_scores_proportionally(self) -> None:
        # Sanity contrast: a hit produces a positive score.
        score = _score_chunk("call foo() and bar()", {"foo", "bar"})
        assert score > 0.0


# ===========================================================================
# build_structural_context: no-symbols fallback truncation loop break
# ===========================================================================


class TestBuildStructuralContextNoSymbolsFallback:
    def test_truncation_loop_breaks_when_budget_exhausted(self) -> None:
        # Plain prose with no detectable language and no extractable symbols
        # forces the no-symbols fallback (simple truncation). A tiny budget
        # exhausts after the first upstream header so the loop breaks before
        # emitting the second upstream.
        upstream_texts = {
            "a": "lorem ipsum dolor sit amet plain text no code here at all",
            "b": "another block of completely plain prose with nothing codey",
        }
        # budget_tokens=3 -> budget_chars=12; first header "--- a ---\n" is
        # 10 chars, leaving 2 chars; the second header (10) exceeds the
        # remaining budget and the loop breaks.
        result = build_structural_context(upstream_texts, budget_tokens=3)

        assert "--- a ---" in result
        # Second upstream never emitted because the loop broke.
        assert "--- b ---" not in result

    def test_fallback_returns_truncated_first_upstream(self) -> None:
        # Single upstream, no symbols: returns the header + truncated body.
        upstream_texts = {"only": "just some plain english prose with no symbols"}
        result = build_structural_context(upstream_texts, budget_tokens=5)
        assert result.startswith("--- only ---")


# ===========================================================================
# build_structural_context: greedy selection budget-skip continue
# ===========================================================================


class TestBuildStructuralContextGreedySkip:
    def test_oversized_chunk_is_skipped_via_continue(self) -> None:
        # A diff with many added named functions guarantees a populated
        # changed-symbol set (so we reach the greedy selection phase) and
        # several scored chunks. The first (high-scoring, large) chunks
        # overflow the tiny budget and are skipped via `continue`; the final
        # small chunk fits and is selected.
        lines = ["diff --git a/mod.py b/mod.py", "+++ b/mod.py"]
        for i in range(60):
            lines.append(f"+def fn{i}():")
            lines.append(f"+    return fn{i}()")
        diff = "\n".join(lines)

        # budget_tokens=25 -> budget_chars=100. Full-size chunks (~489 chars
        # incl. header) overflow and trigger the skip branch; only the small
        # trailing chunk fits.
        result = build_structural_context({"u": diff}, budget_tokens=25)

        # Something was selected, but well under the char budget — proving
        # the larger chunks were skipped rather than included.
        assert result
        assert "--- u ---" in result
        assert len(result) <= 100

    def test_zero_budget_returns_empty(self) -> None:
        # Guard branch: non-positive budget short-circuits to "".
        assert build_structural_context({"u": "def f(): pass"}, budget_tokens=0) == ""

    def test_extract_changed_symbols_path_populates_changed_set(self) -> None:
        # A diff upstream yields changed symbols (non-import) so the symbol
        # path (not the fallback) is taken; the result should contain the
        # upstream marker.
        diff = (
            "diff --git a/mod.py b/mod.py\n"
            "+++ b/mod.py\n"
            "+def changed_fn():\n"
            "+    return changed_fn()\n"
        )
        result = build_structural_context({"d": diff}, budget_tokens=200)
        assert "--- d ---" in result
        assert isinstance(result, str)
