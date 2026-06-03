"""Tests for v1.28.0 features:
- Adaptive Temporal Routing (trend detection, cross-task affinity)
- Population-Based Search
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from maestro_cli.loader import load_plan
from maestro_cli.models import (
    POPULATION_STRATEGIES,
    ModelRecord,
    PlanSpec,
    PopulationSpec,
    TaskHistory,
    TaskSpec,
)
from maestro_cli.routing import (
    _apply_historical_signal,
    _compute_task_similarity,
    _detect_trend,
    apply_cross_task_routing,
)


def _write_plan(tmp_path: Path, yaml_text: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    return p


# ===========================================================================
# Feature 8: Adaptive Temporal Routing
# ===========================================================================


class TestDetectTrend:
    def test_stable(self) -> None:
        outcomes = ["success", "success", "success", "success", "success", "success"]
        assert _detect_trend(outcomes) == "stable"

    def test_degrading(self) -> None:
        # First 3 succeed, last 3 fail
        outcomes = ["success", "success", "success", "failed", "failed", "failed"]
        assert _detect_trend(outcomes) == "degrading"

    def test_improving(self) -> None:
        # First 3 fail, last 3 succeed
        outcomes = ["failed", "failed", "failed", "success", "success", "success"]
        assert _detect_trend(outcomes) == "improving"

    def test_too_few_outcomes(self) -> None:
        assert _detect_trend(["success", "failed"]) == "stable"

    def test_mixed_stable(self) -> None:
        # Alternating pattern — recent window has 1/3 success, same as previous
        outcomes = ["success", "failed", "success", "success", "failed", "success"]
        assert _detect_trend(outcomes) == "stable"


class TestComputeTaskSimilarity:
    def test_identical_tasks(self) -> None:
        a = TaskSpec(id="a", engine="claude", tags=["security"], context_mode="raw")
        b = TaskSpec(id="b", engine="claude", tags=["security"], context_mode="raw")
        sim = _compute_task_similarity(a, b)
        assert sim > 0.8

    def test_different_engines(self) -> None:
        a = TaskSpec(id="a", engine="claude")
        b = TaskSpec(id="b", engine="codex")
        sim = _compute_task_similarity(a, b)
        assert sim < 0.6

    def test_overlapping_tags(self) -> None:
        a = TaskSpec(id="a", engine="claude", tags=["security", "review"])
        b = TaskSpec(id="b", engine="claude", tags=["security", "audit"])
        sim = _compute_task_similarity(a, b)
        assert sim > 0.5

    def test_no_tags(self) -> None:
        a = TaskSpec(id="a", engine="claude")
        b = TaskSpec(id="b", engine="claude")
        sim = _compute_task_similarity(a, b)
        assert sim > 0.5  # engine + context_mode match

    def test_completely_different(self) -> None:
        a = TaskSpec(id="a", engine="claude", tags=["security"], context_mode="recursive")
        b = TaskSpec(id="b", engine="codex", tags=["docs"], context_mode="raw")
        sim = _compute_task_similarity(a, b)
        assert sim < 0.5


class TestApplyCrossTaskRouting:
    def test_no_histories(self) -> None:
        task = TaskSpec(id="t1", engine="claude")
        score = apply_cross_task_routing(task, {"t1": task}, {}, "claude", 0.5)
        assert score == 0.5

    def test_similar_task_success_pushes_down(self) -> None:
        t1 = TaskSpec(id="t1", engine="claude", tags=["review"])
        t2 = TaskSpec(id="t2", engine="claude", tags=["review"])
        histories = {
            "t2": TaskHistory(
                task_id="t2",
                total_runs=5,
                records={
                    "haiku": ModelRecord(
                        model="haiku", runs=5, successes=5, failures=0,
                        timeouts=0, avg_duration_sec=10.0, avg_cost_usd=0.01,
                    ),
                },
            ),
        }
        score = apply_cross_task_routing(
            t1, {"t1": t1, "t2": t2}, histories, "claude", 0.5,
        )
        assert score < 0.5  # pushed cheaper

    def test_similar_task_failure_pushes_up(self) -> None:
        t1 = TaskSpec(id="t1", engine="claude", tags=["security"])
        t2 = TaskSpec(id="t2", engine="claude", tags=["security"])
        histories = {
            "t2": TaskHistory(
                task_id="t2",
                total_runs=5,
                records={
                    "haiku": ModelRecord(
                        model="haiku", runs=5, successes=1, failures=4,
                        timeouts=0, avg_duration_sec=10.0, avg_cost_usd=0.01,
                    ),
                },
            ),
        }
        score = apply_cross_task_routing(
            t1, {"t1": t1, "t2": t2}, histories, "claude", 0.5,
        )
        assert score > 0.5  # pushed stronger

    def test_dissimilar_task_ignored(self) -> None:
        t1 = TaskSpec(id="t1", engine="claude", tags=["security"])
        t2 = TaskSpec(id="t2", engine="codex", tags=["docs"])
        histories = {
            "t2": TaskHistory(
                task_id="t2",
                total_runs=5,
                records={
                    "5-mini": ModelRecord(
                        model="5-mini", runs=5, successes=5, failures=0,
                        timeouts=0, avg_duration_sec=5.0, avg_cost_usd=0.005,
                    ),
                },
            ),
        }
        score = apply_cross_task_routing(
            t1, {"t1": t1, "t2": t2}, histories, "claude", 0.5,
        )
        assert score == 0.5  # no change


class TestTrendInHistoricalSignal:
    def test_degrading_trend_pushes_up(self) -> None:
        history = TaskHistory(
            task_id="t1",
            total_runs=10,
            records={
                "haiku": ModelRecord(
                    model="haiku", runs=6, successes=3, failures=3,
                    timeouts=0, avg_duration_sec=10.0, avg_cost_usd=0.01,
                    recent_outcomes=[
                        "success", "success", "success",
                        "failed", "failed", "failed",
                    ],
                ),
            },
        )
        base = 0.5
        adjusted = _apply_historical_signal(base, history, "claude")
        assert adjusted > base  # degrading → push up

    def test_improving_trend_on_cheap_model(self) -> None:
        history = TaskHistory(
            task_id="t1",
            total_runs=10,
            records={
                "haiku": ModelRecord(
                    model="haiku", runs=6, successes=3, failures=3,
                    timeouts=0, avg_duration_sec=10.0, avg_cost_usd=0.01,
                    recent_outcomes=[
                        "failed", "failed", "failed",
                        "success", "success", "success",
                    ],
                ),
            },
        )
        base = 0.5
        adjusted = _apply_historical_signal(base, history, "claude")
        # Improving on cheap model + 50% failure rate → signals compete
        # (Rule 2 pushes up for failure, Rule 5 pushes down for improving)
        # Net should be moderate — not wildly higher than base
        assert adjusted <= base + 0.20


# ===========================================================================
# Feature 9: Population-Based Search
# ===========================================================================


class TestPopulationSpec:
    def test_defaults(self) -> None:
        spec = PopulationSpec(candidates=["haiku", "sonnet"])
        assert spec.strategy == "best"
        assert spec.parallel is True

    def test_to_dict(self) -> None:
        spec = PopulationSpec(
            candidates=["haiku", "sonnet", "opus"],
            strategy="first_passing",
            parallel=False,
        )
        d = spec.to_dict()
        assert d["candidates"] == ["haiku", "sonnet", "opus"]
        assert d["strategy"] == "first_passing"
        assert d["parallel"] is False

    def test_strategies_constant(self) -> None:
        assert POPULATION_STRATEGIES == {"best", "first_passing", "majority"}


class TestPopulationLoader:
    def test_parse_basic(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                population:
                  candidates: [haiku, sonnet, opus]
                  strategy: best
        """)
        plan = load_plan(p)
        pop = plan.tasks[0].population
        assert pop is not None
        assert pop.candidates == ["haiku", "sonnet", "opus"]
        assert pop.strategy == "best"

    def test_parse_first_passing(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                population:
                  candidates: [haiku, sonnet]
                  strategy: first_passing
                  parallel: false
        """)
        plan = load_plan(p)
        pop = plan.tasks[0].population
        assert pop is not None
        assert pop.strategy == "first_passing"
        assert pop.parallel is False

    def test_too_few_candidates_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                population:
                  candidates: [haiku]
        """)
        with pytest.raises(Exception, match="E012"):
            load_plan(p)

    def test_empty_candidates_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                population:
                  candidates: []
        """)
        with pytest.raises(Exception):
            load_plan(p)

    def test_invalid_strategy_raises(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
                population:
                  candidates: [haiku, sonnet]
                  strategy: random
        """)
        with pytest.raises(Exception, match="E008"):
            load_plan(p)

    def test_no_population(self, tmp_path: Path) -> None:
        p = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: t1
                engine: claude
                prompt: "do stuff"
        """)
        plan = load_plan(p)
        assert plan.tasks[0].population is None


class TestModelRecordOutcomes:
    def test_recent_outcomes_serialized(self) -> None:
        rec = ModelRecord(
            model="haiku",
            runs=3,
            successes=2,
            failures=1,
            timeouts=0,
            avg_duration_sec=5.0,
            avg_cost_usd=0.01,
            recent_outcomes=["success", "success", "failed"],
        )
        d = rec.to_dict()
        assert d["recent_outcomes"] == ["success", "success", "failed"]

    def test_recent_outcomes_capped(self) -> None:
        rec = ModelRecord(
            model="haiku",
            runs=20,
            successes=15,
            failures=5,
            timeouts=0,
            avg_duration_sec=5.0,
            avg_cost_usd=0.01,
            recent_outcomes=["success"] * 20,
        )
        d = rec.to_dict()
        assert len(d["recent_outcomes"]) == 10  # capped at 10

    def test_empty_outcomes(self) -> None:
        rec = ModelRecord(
            model="haiku", runs=0, successes=0, failures=0,
            timeouts=0, avg_duration_sec=0.0, avg_cost_usd=None,
        )
        d = rec.to_dict()
        assert d["recent_outcomes"] == []
