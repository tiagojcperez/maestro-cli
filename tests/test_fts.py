from __future__ import annotations

import pytest

from maestro_cli import fts
from maestro_cli.fts import (
    FtsHit,
    fts5_available,
    fts_enabled,
    fts_prefix_enabled,
    rank_documents,
    relevance_by_rank,
)


class TestFts5Available:
    def test_returns_bool(self) -> None:
        assert isinstance(fts5_available(), bool)

    def test_available_on_this_build(self) -> None:
        # Every CPython build the test suite runs on ships FTS5; if this ever
        # fails it is a genuine environment signal, not a flaky test.
        assert fts5_available() is True

    def test_result_is_cached(self) -> None:
        # Two calls must agree (the probe is memoised in a module global).
        assert fts5_available() == fts5_available()


class TestFtsEnabled:
    def test_default_enabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MAESTRO_FTS", raising=False)
        assert fts_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", "False", " no "])
    def test_disabled_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("MAESTRO_FTS", value)
        assert fts_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "anything"])
    def test_enabled_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("MAESTRO_FTS", value)
        assert fts_enabled() is True


class TestRankDocuments:
    def test_returns_only_matching_documents(self) -> None:
        docs = [
            "alpha beta gamma",
            "gamma delta epsilon",
            "alpha omega",
        ]
        hits = rank_documents(docs, "alpha")
        indices = {hit.index for hit in hits}
        assert indices == {0, 2}

    def test_higher_term_frequency_ranks_first(self) -> None:
        docs = [
            "alpha beta gamma",  # 0: one 'alpha'
            "alpha alpha alpha",  # 1: three 'alpha', same length
        ]
        hits = rank_documents(docs, "alpha")
        assert {hit.index for hit in hits} == {0, 1}
        assert hits[0].index == 1  # denser match ranks first

    def test_multi_term_query_prefers_full_match(self) -> None:
        docs = [
            "sqlite fts5 search",  # 0: both terms
            "sqlite only here",    # 1: one term
            "unrelated content",   # 2: no terms
        ]
        hits = rank_documents(docs, "sqlite fts5")
        assert {hit.index for hit in hits} == {0, 1}
        assert hits[0].index == 0

    def test_score_is_higher_is_better_and_sorted_descending(self) -> None:
        docs = ["alpha alpha", "alpha beta gamma delta", "alpha"]
        hits = rank_documents(docs, "alpha")
        scores = [hit.score for hit in hits]
        assert scores == sorted(scores, reverse=True)

    def test_case_insensitive(self) -> None:
        docs = ["The SQLite Database", "nothing here"]
        hits = rank_documents(docs, "sqlite")
        assert {hit.index for hit in hits} == {0}

    def test_limit_caps_result_count(self) -> None:
        docs = [f"alpha doc number {n}" for n in range(10)]
        hits = rank_documents(docs, "alpha", limit=3)
        assert len(hits) == 3

    def test_returns_fthit_instances(self) -> None:
        hits = rank_documents(["alpha beta"], "alpha")
        assert all(isinstance(hit, FtsHit) for hit in hits)

    def test_tie_scores_resolve_to_insertion_order(self) -> None:
        # Identical documents tie on bm25; the secondary rowid sort key must
        # make the order deterministic (ascending insertion index) and
        # build-independent rather than relying on SQLite's emission order.
        docs = ["alpha beta"] * 5
        hits = rank_documents(docs, "alpha")
        assert [hit.index for hit in hits] == [0, 1, 2, 3, 4]

    # ----- empty / no-match contract -----

    def test_empty_documents_returns_empty(self) -> None:
        assert rank_documents([], "alpha") == []

    def test_empty_query_returns_empty(self) -> None:
        assert rank_documents(["alpha beta"], "") == []

    def test_whitespace_query_returns_empty(self) -> None:
        assert rank_documents(["alpha beta"], "   ") == []

    def test_no_match_returns_empty(self) -> None:
        assert rank_documents(["alpha beta gamma"], "zzz") == []

    def test_query_with_only_punctuation_returns_empty(self) -> None:
        assert rank_documents(["alpha beta"], "!@#$%^&*()") == []

    def test_handles_empty_document_strings(self) -> None:
        docs = ["", "alpha here", ""]
        hits = rank_documents(docs, "alpha")
        assert {hit.index for hit in hits} == {1}

    # ----- FTS5 query-syntax hardening -----

    def test_reserved_words_do_not_break_query(self) -> None:
        # AND/OR/NOT/NEAR are FTS5 operators; we quote them as literals.
        docs = ["sqlite is great", "nothing relevant"]
        hits = rank_documents(docs, "AND OR NOT NEAR sqlite")
        assert {hit.index for hit in hits} == {0}

    def test_punctuation_in_query_is_safe(self) -> None:
        docs = ["sqlite-vec extension for fts5", "plain text"]
        hits = rank_documents(docs, "sqlite-vec (FTS5)!! @columns:")
        assert 0 in {hit.index for hit in hits}

    def test_sql_injection_in_query_is_inert(self) -> None:
        docs = ["a normal document about cats", "another about dogs"]
        # Must not raise and must not corrupt the ephemeral table.
        hits = rank_documents(docs, "'); DROP TABLE docs; --")
        assert isinstance(hits, list)

    def test_sql_injection_in_document_is_inert(self) -> None:
        docs = ["'); DROP TABLE docs; -- alpha", "beta gamma"]
        hits = rank_documents(docs, "alpha")
        assert {hit.index for hit in hits} == {0}

    def test_duplicate_query_terms_are_deduplicated(self) -> None:
        docs = ["alpha beta", "gamma delta"]
        hits = rank_documents(docs, "alpha alpha alpha alpha")
        assert {hit.index for hit in hits} == {0}

    def test_accented_terms_match(self) -> None:
        docs = ["café mocha latte", "plain water"]
        hits = rank_documents(docs, "café")
        assert 0 in {hit.index for hit in hits}

    # ----- graceful degradation -----

    def test_returns_empty_when_fts5_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fts, "fts5_available", lambda: False)
        assert rank_documents(["alpha beta"], "alpha") == []


class TestFtsPrefixEnabled:
    def test_default_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("MAESTRO_FTS_PREFIX", raising=False)
        assert fts_prefix_enabled() is False

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "ON"])
    def test_enabled_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("MAESTRO_FTS_PREFIX", value)
        assert fts_prefix_enabled() is True

    @pytest.mark.parametrize("value", ["0", "false", "off", "no", ""])
    def test_disabled_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("MAESTRO_FTS_PREFIX", value)
        assert fts_prefix_enabled() is False


class TestPrefixMatching:
    def test_exact_mode_does_not_match_longer_token(self) -> None:
        # Default (exact): "auth" must NOT match "authentication".
        assert rank_documents(["authentication module"], "auth") == []

    def test_prefix_mode_matches_longer_tokens(self) -> None:
        docs = ["authentication module", "authorize the user", "unrelated text"]
        hits = rank_documents(docs, "auth", prefix=True)
        assert {h.index for h in hits} == {0, 1}

    def test_short_token_stays_exact_in_prefix_mode(self) -> None:
        # "id" (2 chars, < _MIN_PREFIX_LEN) must stay exact, not match "identifier".
        docs = ["identifier resolution", "the id card"]
        hits = rank_documents(docs, "id", prefix=True)
        assert {h.index for h in hits} == {1}

    def test_relevance_by_rank_honours_prefix(self) -> None:
        rel = relevance_by_rank(["configuration loader"], "config", prefix=True)
        assert rel == {0: 1.0}
        assert relevance_by_rank(["configuration loader"], "config") == {}


class TestRelevanceByRank:
    def test_top_hit_is_one(self) -> None:
        docs = ["alpha beta", "alpha", "beta gamma"]
        rel = relevance_by_rank(docs, "alpha")
        assert max(rel.values()) == pytest.approx(1.0)

    def test_values_in_open_unit_interval(self) -> None:
        docs = [f"alpha number {n}" for n in range(5)]
        rel = relevance_by_rank(docs, "alpha")
        assert all(0.0 < value <= 1.0 for value in rel.values())

    def test_only_matching_indices_present(self) -> None:
        docs = ["alpha beta", "gamma delta", "alpha omega"]
        rel = relevance_by_rank(docs, "alpha")
        assert set(rel) == {0, 2}

    def test_relevance_decreases_with_rank(self) -> None:
        docs = ["alpha alpha alpha", "alpha beta gamma delta epsilon"]
        rel = relevance_by_rank(docs, "alpha")
        # The denser, shorter doc (index 0) outranks the sparse, long one.
        assert rel[0] > rel[1]

    def test_last_hit_strictly_positive(self) -> None:
        docs = ["alpha one", "alpha two", "alpha three", "alpha four"]
        rel = relevance_by_rank(docs, "alpha")
        assert min(rel.values()) == pytest.approx(1.0 / 4.0)

    def test_no_match_returns_empty_mapping(self) -> None:
        assert relevance_by_rank(["alpha beta"], "zzz") == {}

    def test_empty_when_fts5_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(fts, "fts5_available", lambda: False)
        assert relevance_by_rank(["alpha beta"], "alpha") == {}

    def test_limit_is_respected(self) -> None:
        docs = [f"alpha doc {n}" for n in range(8)]
        rel = relevance_by_rank(docs, "alpha", limit=2)
        assert len(rel) == 2
