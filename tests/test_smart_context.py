from __future__ import annotations

import math

import pytest

from maestro_cli.models import TaskResult, TaskSpec
from maestro_cli.scheduler import (
    _apply_intent_filtering,
    _apply_context_budget,
    _apply_hop_decay,
    _compute_hop_distances,
    _compute_idf,
    _estimate_tokens,
    _score_section,
)


def _make_result(task_id: str, stdout_tail: str) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        status="success",
        exit_code=0,
        duration_sec=1.0,
        stdout_tail=stdout_tail,
    )


class TestBM25Scoring:
    def test_score_section_backward_compat(self) -> None:
        section = "fix auth fix bug"
        keywords = {"fix", "auth", "db"}
        score = _score_section(section, keywords)
        assert score == 2

    def test_score_section_with_idf(self) -> None:
        sections = [
            "common rare",
            "common common",
            "common",
        ]
        idf = _compute_idf(sections)

        rare_score = _score_section("rare", {"rare", "common"}, idf=idf)
        common_score = _score_section("common", {"rare", "common"}, idf=idf)

        assert rare_score > common_score

    def test_score_section_tf_saturation(self) -> None:
        idf = {"fix": 8.0}
        single = _score_section("fix", {"fix"}, idf=idf, avg_section_len=10.0)
        repeated = _score_section(" ".join(["fix"] * 40), {"fix"}, idf=idf, avg_section_len=10.0)

        assert repeated > single
        assert repeated < single * 3

    def test_score_section_empty_inputs(self) -> None:
        assert _score_section("", {"auth"}) == 0
        assert _score_section("auth appears", set()) == 0

    def test_compute_idf_basic(self) -> None:
        sections = ["alpha beta", "alpha gamma", "alpha"]
        idf = _compute_idf(sections)

        assert set(idf) == {"alpha", "beta", "gamma"}
        assert idf["beta"] == pytest.approx(math.log(3 / 2), rel=1e-6)
        assert idf["gamma"] == pytest.approx(math.log(3 / 2), rel=1e-6)

    def test_compute_idf_rare_term_higher(self) -> None:
        sections = ["shared rare", "shared", "shared"]
        idf = _compute_idf(sections)

        assert idf["rare"] > idf["shared"]


class TestPriorityEviction:
    def test_intent_filtering_collects_selection_metadata(self) -> None:
        upstream = {
            "relevant": _make_result(
                "relevant",
                "api auth schema\n\nrelease note",
            ),
            "noise": _make_result(
                "noise",
                "gardening weather sports",
            ),
        }

        filtered, trims, selection_meta = _apply_intent_filtering(
            upstream,
            {"api", "auth", "schema"},
        )

        assert filtered["relevant"].stdout_tail == "api auth schema"
        assert trims == [("relevant", _estimate_tokens("api auth schema\n\nrelease note"), _estimate_tokens("api auth schema"))]
        assert selection_meta["relevant"]["upstream_id"] == "relevant"
        assert selection_meta["relevant"]["score"] > 0
        assert selection_meta["relevant"]["keywords_matched"] == ["api", "auth", "schema"]
        assert selection_meta["noise"]["score"] == 0.0
        assert selection_meta["noise"]["keywords_matched"] == []

    def test_eviction_preserves_relevant(self) -> None:
        relevant_a = ("auth bug fix " * 70).strip()
        relevant_b = ("auth patch login " * 70).strip()
        irrelevant = ("weather forecast sports " * 70).strip()

        upstream = {
            "rel-a": _make_result("rel-a", relevant_a),
            "rel-b": _make_result("rel-b", relevant_b),
            "noise": _make_result("noise", irrelevant),
        }
        budget = _estimate_tokens(relevant_a) + _estimate_tokens(relevant_b)

        result, trims, _selection_meta = _apply_context_budget(
            upstream,
            budget,
            {"auth", "bug", "patch", "login"},
        )

        assert result["rel-a"].stdout_tail == relevant_a
        assert result["rel-b"].stdout_tail == relevant_b
        assert len(result["noise"].stdout_tail) < len(irrelevant)
        assert any(tid == "noise" for tid, _, _ in trims)

    def test_eviction_trims_irrelevant_first(self) -> None:
        relevant = ("scheduler context budget token " * 80).strip()
        irrelevant = ("gardening weather cooking travel " * 80).strip()

        upstream = {
            "core": _make_result("core", relevant),
            "noise": _make_result("noise", irrelevant),
        }
        budget = _estimate_tokens(relevant) + (_estimate_tokens(irrelevant) // 4)

        result, _trims, _selection_meta = _apply_context_budget(
            upstream,
            budget,
            {"scheduler", "context", "budget", "token"},
        )

        assert result["core"].stdout_tail == relevant
        assert len(result["noise"].stdout_tail) < len(irrelevant)

    def test_eviction_under_budget_no_change(self) -> None:
        upstream = {"a": _make_result("a", "short output")}

        result, trims, selection_meta = _apply_context_budget(upstream, 10_000, {"auth"})

        assert result is upstream
        assert trims == []
        assert selection_meta["a"]["upstream_id"] == "a"
        assert selection_meta["a"]["score"] == 0.0

    def test_eviction_without_intent_keywords(self) -> None:
        left = "a" * 4000
        right = "b" * 4000
        upstream = {
            "left": _make_result("left", left),
            "right": _make_result("right", right),
        }

        result, trims, selection_meta = _apply_context_budget(
            upstream,
            200,
            intent_keywords=None,
        )

        assert len(trims) == 2
        assert len(result["left"].stdout_tail) == len(result["right"].stdout_tail)
        assert selection_meta == {}


class TestGraphDecay:
    def test_hop_distances_direct(self) -> None:
        tasks = {
            "a": TaskSpec(id="a"),
            "b": TaskSpec(id="b", depends_on=["a"]),
        }

        distances = _compute_hop_distances("b", ["a"], tasks)

        assert distances == {"a": 1}

    def test_hop_distances_transitive(self) -> None:
        tasks = {
            "a": TaskSpec(id="a"),
            "b": TaskSpec(id="b", depends_on=["a"]),
            "c": TaskSpec(id="c", depends_on=["b"]),
        }

        distances = _compute_hop_distances("c", ["a"], tasks)

        assert distances == {"a": 2}

    def test_hop_distances_wildcard(self) -> None:
        tasks = {
            "a": TaskSpec(id="a"),
            "b": TaskSpec(id="b"),
            "c": TaskSpec(id="c", depends_on=["a", "b"]),
        }

        distances = _compute_hop_distances("c", ["*"], tasks)

        assert distances == {"a": 1, "b": 1}

    def test_apply_hop_decay_direct(self) -> None:
        tail = "x" * 100
        upstream = {"a": _make_result("a", tail)}

        result = _apply_hop_decay(upstream, {"a": 1})

        assert result["a"].stdout_tail == tail

    def test_apply_hop_decay_transitive(self) -> None:
        tail = "x" * 100
        upstream = {"a": _make_result("a", tail)}

        result = _apply_hop_decay(upstream, {"a": 2})

        assert len(result["a"].stdout_tail) == 80

    def test_apply_hop_decay_no_change(self) -> None:
        upstream = {
            "a": _make_result("a", "x" * 60),
            "b": _make_result("b", "y" * 60),
        }

        result = _apply_hop_decay(upstream, {"a": 1, "b": 1})

        assert result["a"].stdout_tail == upstream["a"].stdout_tail
        assert result["b"].stdout_tail == upstream["b"].stdout_tail
