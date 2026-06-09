"""Zero-dependency full-text ranked search backed by SQLite FTS5.

SQLite ships an FTS5 extension that provides an indexed, BM25-ranked full-text
search.  It is part of the Python standard library's ``sqlite3`` module on the
vast majority of builds, so this gives Maestro a *better* lexical retriever than
the hand-rolled Python BM25 in :mod:`maestro_cli.knowledge` and
:mod:`maestro_cli.scheduler` — without adding a single dependency.

The public surface is intentionally tiny and stateless: callers hand in a list
of documents and a free-text query and receive ranked hits back.  Everything is
built in an in-memory database per call, so there is no index lifecycle to
manage and nothing to clean up.

Graceful degradation is a first-class contract: when FTS5 is unavailable, the
query has no usable terms, or nothing matches, :func:`rank_documents` returns an
empty list.  Callers MUST treat an empty result as "fall back to your own
ranking" rather than "no relevant documents".
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass

# FTS5's query language treats a bare word as a token but reserves AND/OR/NOT/
# NEAR and a handful of punctuation characters as operators.  We sidestep all of
# that by extracting plain word tokens and quoting each one as a string literal,
# so user text can never produce an invalid MATCH expression.
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Cap the number of OR terms so a pathologically long prompt cannot build a
# multi-thousand-clause MATCH expression.
_MAX_QUERY_TERMS = 64

_fts5_available: bool | None = None


@dataclass(frozen=True)
class FtsHit:
    """A single ranked document.

    ``index`` is the position of the document in the list passed to
    :func:`rank_documents`.  ``score`` is a relevance value where **higher means
    more relevant** (it is the negated FTS5 ``bm25()`` rank, which is natively
    "lower is better").
    """

    index: int
    score: float


def fts5_available() -> bool:
    """Return ``True`` when this interpreter's sqlite3 build includes FTS5.

    The probe runs once and is cached for the lifetime of the process.
    """
    global _fts5_available
    if _fts5_available is None:
        conn = sqlite3.connect(":memory:")
        try:
            conn.execute("CREATE VIRTUAL TABLE _probe USING fts5(x)")
            _fts5_available = True
        except sqlite3.OperationalError:
            _fts5_available = False
        finally:
            conn.close()
    return _fts5_available


def _build_match_expression(query: str) -> str:
    """Turn free text into a safe FTS5 OR-of-terms MATCH expression.

    Each unique token is lowercased and wrapped as a quoted string literal so
    reserved words (AND/OR/NOT/NEAR) and punctuation cannot break parsing.
    Returns an empty string when the query yields no usable terms.
    """
    seen: set[str] = set()
    terms: list[str] = []
    for match in _TOKEN_RE.finditer(query):
        token = match.group(0).lower()
        if token in seen:
            continue
        seen.add(token)
        terms.append(f'"{token}"')
        if len(terms) >= _MAX_QUERY_TERMS:
            break
    return " OR ".join(terms)


def rank_documents(
    documents: Sequence[str],
    query: str,
    *,
    limit: int | None = None,
) -> list[FtsHit]:
    """Rank ``documents`` against ``query`` using SQLite FTS5 BM25.

    Returns hits ordered most-relevant first; only documents that match at
    least one query term are included.  Returns an empty list when FTS5 is
    unavailable, the query has no usable terms, or nothing matches — callers
    should treat that as a signal to fall back to their own ranking.

    The match expression is bound as a query parameter, so document and query
    text can never be interpreted as SQL.
    """
    if not documents or not query or not query.strip():
        return []
    if not fts5_available():
        return []

    match_expr = _build_match_expression(query)
    if not match_expr:
        return []

    conn = sqlite3.connect(":memory:")
    try:
        # Pin remove_diacritics=2: the unicode61 default changed across SQLite
        # releases (1 before 3.27.0, 2 after), which would make accent folding —
        # and therefore the *set* of matched documents — build-dependent.
        conn.execute(
            "CREATE VIRTUAL TABLE docs "
            "USING fts5(body, tokenize='unicode61 remove_diacritics 2')"
        )
        conn.executemany(
            "INSERT INTO docs(rowid, body) VALUES (?, ?)",
            [(index, doc or "") for index, doc in enumerate(documents)],
        )
        # Secondary `rowid` key makes tie ordering deterministic and
        # build-independent: bm25 ties (e.g. identical documents) resolve to
        # ascending insertion order rather than SQLite's unspecified emission
        # order, so identical inputs always rank identically across platforms.
        # Both branches pass a string *literal* to execute() with the match
        # expression (and limit) always bound as parameters, so there is no
        # injection surface — and a static analyser can confirm it directly.
        if limit is not None and limit > 0:
            rows = conn.execute(
                "SELECT rowid, bm25(docs) AS rank FROM docs "
                "WHERE docs MATCH ? ORDER BY rank, rowid LIMIT ?",
                (match_expr, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT rowid, bm25(docs) AS rank FROM docs "
                "WHERE docs MATCH ? ORDER BY rank, rowid",
                (match_expr,),
            ).fetchall()
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()

    # bm25() is negative with "lower == more relevant"; flip the sign so the
    # public contract is the intuitive "higher == more relevant".
    return [FtsHit(index=int(rowid), score=-float(rank)) for rowid, rank in rows]


def relevance_by_rank(
    documents: Sequence[str],
    query: str,
    *,
    limit: int | None = None,
) -> dict[int, float]:
    """Rank ``documents`` and return ``{index: relevance}`` in ``(0.0, 1.0]``.

    Relevance is derived from rank *position* rather than the raw BM25 score:
    the top hit gets ``1.0`` and each lower hit a proportionally smaller value,
    with the last hit still strictly positive (``1 / len(hits)``).  Position is
    far more robust than FTS5's raw magnitudes, which are mushy and corpus-size
    dependent for small document sets.  Non-matching documents are simply absent
    from the mapping.  Returns ``{}`` whenever :func:`rank_documents` does, so it
    composes with the same fall-back contract.
    """
    hits = rank_documents(documents, query, limit=limit)
    if not hits:
        return {}
    count = len(hits)
    return {hit.index: (count - position) / count for position, hit in enumerate(hits)}
