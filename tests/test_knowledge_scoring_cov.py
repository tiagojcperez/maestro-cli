from __future__ import annotations

import math

from maestro_cli.knowledge import (
    _KNOWLEDGE_INDEX_SUMMARY_MAX_CHARS,
    _compute_idf,
    _extract_keywords,
    _one_line_summary,
    _score_document,
)


# ---------------------------------------------------------------------------
# _compute_idf
# ---------------------------------------------------------------------------

def test_compute_idf_empty_documents_returns_empty_dict() -> None:
    """No documents -> early-return empty mapping (drives the empty-input guard)."""
    assert _compute_idf([]) == {}


def test_compute_idf_returns_term_weights_for_nonempty_documents() -> None:
    """Non-empty input builds doc-frequency then IDF weights via the log formula."""
    documents = [
        "alpha beta gamma",
        "alpha beta",
        "alpha",
    ]
    idf = _compute_idf(documents)

    # Every non-stopword term should appear as a key.
    assert "alpha" in idf
    assert "beta" in idf
    assert "gamma" in idf

    total_docs = len(documents)
    # "alpha" occurs in all three docs (freq=3), "gamma" only in one (freq=1).
    assert idf["alpha"] == math.log(total_docs / (1 + 3))
    assert idf["gamma"] == math.log(total_docs / (1 + 1))
    # Rarer terms must score higher than common ones.
    assert idf["gamma"] > idf["alpha"]


# ---------------------------------------------------------------------------
# _score_document
# ---------------------------------------------------------------------------

def test_score_document_empty_document_returns_zero() -> None:
    """Empty document short-circuits to 0.0 regardless of intent keywords."""
    assert _score_document("", {"alpha"}, idf=None, avg_doc_len=1.0) == 0.0


def test_score_document_empty_intent_keywords_returns_zero() -> None:
    """Empty intent keyword set short-circuits to 0.0."""
    assert _score_document("alpha beta", set(), idf=None, avg_doc_len=1.0) == 0.0


def test_score_document_idf_none_returns_keyword_intersection_count() -> None:
    """With idf=None the score is the size of the keyword intersection (float)."""
    document = "alpha beta gamma"
    intent = {"alpha", "gamma", "missing"}
    result = _score_document(document, intent, idf=None, avg_doc_len=5.0)

    # alpha + gamma overlap; "missing" is absent => 2 matches.
    expected = float(len(_extract_keywords(document) & intent))
    assert result == expected
    assert result == 2.0
    assert isinstance(result, float)


def test_score_document_no_words_after_tokenize_returns_zero() -> None:
    """idf supplied but document tokenizes to no words -> 0.0 branch.

    Single-character / punctuation-only content yields no keyword tokens
    (the keyword regex requires runs of length >= 2), so the word list is
    empty even though the document string is truthy.
    """
    idf = {"alpha": 1.0}
    # "a" and "!" produce no >=2-char tokens; the string is non-empty so the
    # earlier empty-document guard does not fire.
    result = _score_document("a ! ?", {"alpha"}, idf=idf, avg_doc_len=4.0)
    assert result == 0.0


def test_score_document_bm25_path_scores_matching_terms() -> None:
    """idf supplied with real tokens drives the full BM25 accumulation path."""
    idf = {"alpha": 2.0, "beta": 1.0}
    score = _score_document(
        "alpha alpha beta gamma",
        {"alpha", "beta"},
        idf=idf,
        avg_doc_len=4.0,
    )
    assert score > 0.0


# ---------------------------------------------------------------------------
# _one_line_summary
# ---------------------------------------------------------------------------

def test_one_line_summary_short_text_passes_through() -> None:
    """Text within the limit is normalized (whitespace collapsed) but not truncated."""
    assert _one_line_summary("hello   world\n  again") == "hello world again"


def test_one_line_summary_truncates_long_text_with_ellipsis() -> None:
    """Text longer than max_chars is truncated to max_chars-3 plus an ellipsis."""
    text = "x" * 200
    result = _one_line_summary(text)

    assert result.endswith("...")
    assert len(result) == _KNOWLEDGE_INDEX_SUMMARY_MAX_CHARS
    assert result == "x" * (_KNOWLEDGE_INDEX_SUMMARY_MAX_CHARS - 3) + "..."


def test_one_line_summary_truncation_rstrips_before_ellipsis() -> None:
    """Trailing whitespace at the cut point is stripped before appending the ellipsis."""
    cut = 10
    keep = cut - 3  # slice length used by the implementation: summary[:cut-3]
    # Arrange a space at the very end of the kept slice so rstrip has work to do,
    # while keeping the overall text long enough to trigger truncation.
    text = "a" * (keep - 1) + " " + "b" * 50
    result = _one_line_summary(text, max_chars=cut)

    assert result.endswith("...")
    # The space right before the ellipsis must have been rstripped away.
    assert " ..." not in result
    assert result == "a" * (keep - 1) + "..."
