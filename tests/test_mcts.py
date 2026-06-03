from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from maestro_cli.knowledge import store_score_history
from maestro_cli.mcts import (
    apply_variant_historical_fitness_prior,
    apply_variant_knowledge_prior,
    apply_variant_novelty_prior,
    append_tree_node,
    apply_historical_pruning,
    backpropagate_variant,
    create_workflow_variant,
    load_tree_index,
    select_expansion_parent,
    select_variant_from_pool,
    simulate_variant,
)
from maestro_cli.models import PlanDefaults, PlanRunResult, PlanSpec, ScoreRecord, TaskSpec


def _make_plan(tmp_path: Path, name: str = "demo") -> PlanSpec:
    source_path = tmp_path / f"{name}.yaml"
    source_path.write_text(f"version: 1\nname: {name}\n", encoding="utf-8")
    return PlanSpec(
        version=1,
        name=name,
        defaults=PlanDefaults(),
        tasks=[TaskSpec(id="a", command="echo a")],
        source_path=source_path,
    )


def _make_run_result(
    plan_name: str,
    *,
    success: bool,
    run_id: str,
    plan_hash: str | None = None,
) -> PlanRunResult:
    now = datetime.now(timezone.utc)
    return PlanRunResult(
        plan_name=plan_name,
        run_id=run_id,
        run_path=Path("/tmp/fake"),
        started_at=now,
        finished_at=now,
        success=success,
        plan_hash=plan_hash,
    )


class _StubRandom:
    def __init__(self, random_value: float) -> None:
        self._random_value = random_value

    def random(self) -> float:
        return self._random_value

    def choice(self, values: list[object]) -> object:
        return values[0]


class TestWorkflowVariantCreation:
    def test_classifies_draft_debug_and_improve_nodes(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)

        root = create_workflow_variant(plan, node_id="root")
        assert root.variant_type == "draft"

        debug_child = create_workflow_variant(plan, parent=root, node_id="debug-child")
        assert debug_child.variant_type == "debug"
        assert root.children == [debug_child]

        valid_root = create_workflow_variant(
            plan,
            node_id="valid-root",
            run_result=_make_run_result(plan.name, success=True, run_id="run-valid-root"),
            is_valid=True,
        )
        improve_child = create_workflow_variant(plan, parent=valid_root, node_id="improve-child")
        assert improve_child.variant_type == "improve"


class TestExpansionSelection:
    def test_selects_invalid_leaf_for_debug_path(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        root = create_workflow_variant(
            plan,
            node_id="root",
            run_result=_make_run_result(plan.name, success=True, run_id="root-run"),
            is_valid=True,
        )
        invalid_leaf = create_workflow_variant(
            plan,
            parent=root,
            node_id="bad",
            run_result=_make_run_result(plan.name, success=False, run_id="bad-run"),
            score=0.2,
            is_valid=False,
        )
        create_workflow_variant(
            plan,
            parent=root,
            node_id="good",
            run_result=_make_run_result(plan.name, success=True, run_id="good-run"),
            score=0.9,
            is_valid=True,
        )

        selected = select_expansion_parent(root, debug_prob=1.0, rng=_StubRandom(0.0))
        assert selected is invalid_leaf

    def test_selects_best_valid_leaf_for_improve_path(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        root = create_workflow_variant(
            plan,
            node_id="root",
            run_result=_make_run_result(plan.name, success=True, run_id="root-run"),
            is_valid=True,
        )
        create_workflow_variant(
            plan,
            parent=root,
            node_id="low",
            run_result=_make_run_result(plan.name, success=True, run_id="low-run"),
            score=0.3,
            is_valid=True,
        )
        high = create_workflow_variant(
            plan,
            parent=root,
            node_id="high",
            run_result=_make_run_result(plan.name, success=True, run_id="high-run"),
            score=0.8,
            is_valid=True,
        )

        selected = select_expansion_parent(root, debug_prob=0.0, rng=_StubRandom(0.9))
        assert selected is high

    def test_ucb1_prefers_less_visited_leaf_when_exploration_matters(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        root = create_workflow_variant(
            plan,
            node_id="root",
            run_result=_make_run_result(plan.name, success=True, run_id="root-run"),
            is_valid=True,
        )
        root.visits = 12

        frequently_visited = create_workflow_variant(
            plan,
            parent=root,
            node_id="steady",
            run_result=_make_run_result(plan.name, success=True, run_id="steady-run"),
            score=0.65,
            is_valid=True,
        )
        frequently_visited.visits = 10

        lightly_visited = create_workflow_variant(
            plan,
            parent=root,
            node_id="explore",
            run_result=_make_run_result(plan.name, success=True, run_id="explore-run"),
            score=0.45,
            is_valid=True,
        )
        lightly_visited.visits = 1

        selected = select_expansion_parent(
            root,
            selection_policy="ucb1",
            exploration_constant=1.4,
        )

        assert selected is lightly_visited

    def test_ucb1_with_zero_exploration_prefers_highest_score(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        root = create_workflow_variant(
            plan,
            node_id="root",
            run_result=_make_run_result(plan.name, success=True, run_id="root-run"),
            is_valid=True,
        )
        root.visits = 8

        high_score = create_workflow_variant(
            plan,
            parent=root,
            node_id="high",
            run_result=_make_run_result(plan.name, success=True, run_id="high-run"),
            score=0.8,
            is_valid=True,
        )
        high_score.visits = 7

        low_score = create_workflow_variant(
            plan,
            parent=root,
            node_id="low",
            run_result=_make_run_result(plan.name, success=True, run_id="low-run"),
            score=0.6,
            is_valid=True,
        )
        low_score.visits = 1

        selected = select_expansion_parent(
            root,
            selection_policy="ucb1",
            exploration_constant=0.0,
        )

        assert selected is high_score

    def test_select_variant_from_pool_uses_debug_prob_logic(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        invalid = create_workflow_variant(
            plan,
            node_id="bad",
            run_result=_make_run_result(plan.name, success=False, run_id="bad-run"),
            score=0.1,
            is_valid=False,
        )
        valid = create_workflow_variant(
            plan,
            node_id="good",
            run_result=_make_run_result(plan.name, success=True, run_id="good-run"),
            score=0.9,
            is_valid=True,
        )

        selected = select_variant_from_pool(
            [invalid, valid],
            selection_policy="debug_prob",
            debug_prob=1.0,
            rng=_StubRandom(0.0),
        )

        assert selected is invalid

    def test_select_variant_from_pool_uses_ucb1_logic(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        parent = create_workflow_variant(
            plan,
            node_id="root",
            run_result=_make_run_result(plan.name, success=True, run_id="root-run"),
            is_valid=True,
        )
        parent.visits = 10
        first = create_workflow_variant(
            plan,
            parent=parent,
            node_id="first",
            run_result=_make_run_result(plan.name, success=True, run_id="first-run"),
            score=0.6,
            is_valid=True,
        )
        first.visits = 8
        second = create_workflow_variant(
            plan,
            parent=parent,
            node_id="second",
            run_result=_make_run_result(plan.name, success=True, run_id="second-run"),
            score=0.45,
            is_valid=True,
        )
        second.visits = 1

        selected = select_variant_from_pool(
            [first, second],
            selection_policy="ucb1",
            exploration_constant=1.4,
        )

        assert selected is second


class TestSimulationAndHistory:
    def test_simulate_variant_updates_run_state(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        expected = _make_run_result(
            plan.name,
            success=True,
            run_id="sim-1",
            plan_hash="sim-hash",
        )
        captured: list[PlanSpec] = []

        def _fake_run_plan(plan_spec: PlanSpec, **_: object) -> PlanRunResult:
            captured.append(plan_spec)
            return expected

        monkeypatch.setattr("maestro_cli.mcts.run_plan", _fake_run_plan)
        result = simulate_variant(variant, dry_run=True)

        assert result is expected
        assert captured == [plan]
        assert variant.run_result is expected
        assert variant.is_valid is True
        assert variant.plan_hash == "sim-hash"

    def test_apply_historical_pruning_marks_variant(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        now = datetime.now(timezone.utc).isoformat()

        for idx in range(5):
            store_score_history(
                plan.name,
                tmp_path,
                ScoreRecord(
                    plan_name=plan.name,
                    plan_hash=variant.plan_hash,
                    run_id=f"run-{idx}",
                    success=(idx == 4),
                    cost_usd=0.0,
                    quality_score=1.0 if idx == 4 else 0.0,
                    duration_sec=0.0,
                    timestamp=now,
                    valid_from=now,
                    recorded_at=now,
                    source_id=f"run-{idx}:score",
                    metadata={},
                ),
            )

        decision = apply_historical_pruning(variant)

        assert decision.prune is True
        assert variant.pruned is True
        assert variant.metadata["historical_pruning"]["sample_size"] == 5

    def test_backpropagate_variant_blends_current_and_history(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        root = create_workflow_variant(
            plan,
            node_id="root",
            run_result=_make_run_result(plan.name, success=True, run_id="root-run"),
            is_valid=True,
        )
        child = create_workflow_variant(
            plan,
            parent=root,
            node_id="child",
            run_result=_make_run_result(plan.name, success=True, run_id="child-run"),
            is_valid=True,
        )
        anchor = datetime.now(timezone.utc)
        timestamp = anchor.isoformat()

        store_score_history(
            plan.name,
            tmp_path,
            ScoreRecord(
                plan_name=plan.name,
                plan_hash=child.plan_hash,
                run_id="history-fail",
                success=False,
                cost_usd=0.0,
                quality_score=0.0,
                duration_sec=0.0,
                timestamp=timestamp,
                valid_from=timestamp,
                recorded_at=timestamp,
                source_id="history-fail:score",
                metadata={},
            ),
        )

        current = ScoreRecord(
            plan_name=plan.name,
            plan_hash=child.plan_hash,
            run_id="current-pass",
            success=True,
            cost_usd=0.0,
            quality_score=1.0,
            duration_sec=0.0,
            timestamp=timestamp,
            valid_from=timestamp,
            recorded_at=timestamp,
            source_id="current-pass:score",
            metadata={},
        )

        score = backpropagate_variant(
            child,
            score_record=current,
            history_limit=20,
            horizon_days=3650,
            anchor_time=anchor,
        )

        assert score == pytest.approx(0.575)
        assert child.visits == 1
        assert root.visits == 1

    def test_apply_variant_knowledge_prior_sets_metadata(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")

        bonus = apply_variant_knowledge_prior(
            variant,
            bonus=0.08,
            signals=[{"kind": "failure_pattern", "bonus": 0.08}],
        )

        assert bonus == pytest.approx(0.08)
        assert variant.score == pytest.approx(0.08)
        assert variant.metadata["knowledge_prior"] == pytest.approx(0.08)
        assert variant.metadata["knowledge_signals"][0]["kind"] == "failure_pattern"

    def test_backpropagate_variant_applies_knowledge_prior(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        apply_variant_knowledge_prior(variant, bonus=0.05)
        anchor = datetime.now(timezone.utc)
        timestamp = anchor.isoformat()
        current = ScoreRecord(
            plan_name=plan.name,
            plan_hash=variant.plan_hash,
            run_id="current-pass",
            success=True,
            cost_usd=0.0,
            quality_score=1.0,
            duration_sec=0.0,
            timestamp=timestamp,
            valid_from=timestamp,
            recorded_at=timestamp,
            source_id="current-pass:score",
            metadata={},
        )

        score = backpropagate_variant(
            variant,
            score_record=current,
            history_limit=20,
            horizon_days=3650,
            anchor_time=anchor,
        )

        assert score == pytest.approx(1.0)
        assert variant.score == pytest.approx(1.0)
        assert variant.visits == 1
        assert variant.metadata["knowledge_prior"] == pytest.approx(0.05)
        assert variant.metadata["aggregate_score"] == pytest.approx(1.0)

    def test_backpropagate_variant_applies_score_discount(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        anchor = datetime.now(timezone.utc)
        timestamp = anchor.isoformat()
        current = ScoreRecord(
            plan_name=plan.name,
            plan_hash=variant.plan_hash,
            run_id="current-pass",
            success=True,
            cost_usd=0.0,
            quality_score=1.0,
            duration_sec=0.0,
            timestamp=timestamp,
            valid_from=timestamp,
            recorded_at=timestamp,
            source_id="current-pass:score",
            metadata={},
        )

        score = backpropagate_variant(
            variant,
            score_record=current,
            history_limit=20,
            horizon_days=3650,
            anchor_time=anchor,
            score_discount=0.9,
        )

        assert score == pytest.approx(0.9)
        assert variant.score == pytest.approx(0.9)
        assert variant.metadata["score_discount"] == pytest.approx(0.9)

    def test_apply_variant_novelty_prior_sets_metadata(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")

        bonus = apply_variant_novelty_prior(
            variant,
            bonus=0.04,
            signals=[{"nearest_similarity": 0.2}],
        )

        assert bonus == pytest.approx(0.04)
        assert variant.score == pytest.approx(0.04)
        assert variant.metadata["novelty_prior"] == pytest.approx(0.04)
        assert variant.metadata["novelty_signals"][0]["nearest_similarity"] == pytest.approx(0.2)

    def test_apply_variant_historical_fitness_prior_sets_metadata(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")

        bonus = apply_variant_historical_fitness_prior(
            variant,
            bonus=0.06,
            signals=[{"matched_records": 2, "bootstrapped_fitness": 0.8}],
        )

        assert bonus == pytest.approx(0.06)
        assert variant.score == pytest.approx(0.06)
        assert variant.metadata["historical_fitness_prior"] == pytest.approx(0.06)
        assert variant.metadata["historical_fitness_signals"][0]["matched_records"] == 2

    def test_backpropagate_variant_sums_knowledge_and_novelty_priors(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        apply_variant_knowledge_prior(variant, bonus=0.03)
        apply_variant_novelty_prior(variant, bonus=0.04)
        anchor = datetime.now(timezone.utc)
        timestamp = anchor.isoformat()
        current = ScoreRecord(
            plan_name=plan.name,
            plan_hash=variant.plan_hash,
            run_id="current-pass",
            success=True,
            cost_usd=0.0,
            quality_score=0.9,
            duration_sec=0.0,
            timestamp=timestamp,
            valid_from=timestamp,
            recorded_at=timestamp,
            source_id="current-pass:score",
            metadata={},
        )

        score = backpropagate_variant(
            variant,
            score_record=current,
            history_limit=20,
            horizon_days=3650,
            anchor_time=anchor,
        )

        assert score == pytest.approx(1.0)
        assert variant.score == pytest.approx(1.0)
        assert variant.metadata["knowledge_prior"] == pytest.approx(0.03)
        assert variant.metadata["novelty_prior"] == pytest.approx(0.04)

    def test_backpropagate_variant_sums_history_knowledge_and_novelty_priors(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        apply_variant_knowledge_prior(variant, bonus=0.02)
        apply_variant_novelty_prior(variant, bonus=0.03)
        apply_variant_historical_fitness_prior(variant, bonus=0.04)
        anchor = datetime.now(timezone.utc)
        timestamp = anchor.isoformat()
        current = ScoreRecord(
            plan_name=plan.name,
            plan_hash=variant.plan_hash,
            run_id="current-pass",
            success=True,
            cost_usd=0.0,
            quality_score=0.9,
            duration_sec=0.0,
            timestamp=timestamp,
            valid_from=timestamp,
            recorded_at=timestamp,
            source_id="current-pass:score",
            metadata={},
        )

        score = backpropagate_variant(
            variant,
            score_record=current,
            history_limit=20,
            horizon_days=3650,
            anchor_time=anchor,
        )

        assert score == pytest.approx(1.0)
        assert variant.score == pytest.approx(1.0)
        assert variant.metadata["knowledge_prior"] == pytest.approx(0.02)
        assert variant.metadata["novelty_prior"] == pytest.approx(0.03)
        assert variant.metadata["historical_fitness_prior"] == pytest.approx(0.04)


class TestTreePersistence:
    def test_append_tree_node_writes_parent_links(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        root = create_workflow_variant(plan, node_id="root", mutation_desc="draft root")
        child = create_workflow_variant(
            plan,
            parent=root,
            node_id="child",
            mutation_desc="fix failing step",
        )

        run_dir = tmp_path / "run"
        append_tree_node(run_dir, root)
        append_tree_node(run_dir, child)
        rows = load_tree_index(run_dir)

        assert len(rows) == 2
        assert rows[0]["node_id"] == "root"
        assert rows[0]["plan_spec"]["tasks"][0]["id"] == "a"
        assert rows[1]["node_id"] == "child"
        assert rows[1]["parent_id"] == "root"
        assert rows[1]["variant_type"] == "debug"
