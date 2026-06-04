from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

import maestro_cli.memory as memory_mod
from maestro_cli.knowledge import (
    get_historical_pruning_decision,
    load_score_history,
    plan_topology_signature,
    store_score_history,
)
from maestro_cli.models import (
    BatchSpec,
    HistoricalPruningDecision,
    JudgeSpec,
    PlanDefaults,
    PlanSpec,
    ScoreRecord,
    TaskSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(
    tmp_path: Path,
    *tasks: TaskSpec,
    routing_strategy: str | None = None,
    firewall_model: str | None = None,
) -> PlanSpec:
    source_path = tmp_path / "plan.yaml"
    source_path.write_text("version: 1\nname: demo\n", encoding="utf-8")
    return PlanSpec(
        version=1,
        name="demo",
        defaults=PlanDefaults(),
        tasks=list(tasks),
        source_path=source_path,
        routing_strategy=routing_strategy,
        firewall_model=firewall_model,
    )


def _make_score(plan_hash: str = "hash123") -> ScoreRecord:
    ts = datetime.now(timezone.utc).isoformat()
    return ScoreRecord(
        plan_name="demo",
        plan_hash=plan_hash,
        run_id="20260603_010101_000000",
        success=True,
        cost_usd=0.5,
        quality_score=0.9,
        duration_sec=12.0,
        timestamp=ts,
        valid_from=ts,
        recorded_at=ts,
        source_id="run:score",
        metadata={"quality_source": "outcome"},
    )


# ---------------------------------------------------------------------------
# plan_topology_signature — optional-field branches
# ---------------------------------------------------------------------------

class TestPlanTopologySignature:
    def test_routing_strategy_branch(self, tmp_path: Path) -> None:
        """routing_strategy set -> 'plan.routing:<value>' term is emitted."""
        plan = _make_plan(
            tmp_path,
            TaskSpec(id="a", command="echo a"),
            routing_strategy="cost_optimized",
        )
        sig = plan_topology_signature(plan)
        assert "plan.routing:cost_optimized" in sig

    def test_firewall_model_branch(self, tmp_path: Path) -> None:
        """firewall_model set -> 'plan.firewall:1' term is emitted."""
        plan = _make_plan(
            tmp_path,
            TaskSpec(id="a", command="echo a"),
            firewall_model="haiku",
        )
        sig = plan_topology_signature(plan)
        assert "plan.firewall:1" in sig

    def test_no_optional_plan_fields(self, tmp_path: Path) -> None:
        """Without routing/firewall, neither optional plan term appears."""
        plan = _make_plan(tmp_path, TaskSpec(id="a", command="echo a"))
        sig = plan_topology_signature(plan)
        assert not any(term.startswith("plan.routing:") for term in sig)
        assert "plan.firewall:1" not in sig

    def test_agent_branch(self, tmp_path: Path) -> None:
        """task.agent set -> 'task.agent:<value>' term."""
        task = TaskSpec(id="a", engine="claude", prompt="do it", agent="reviewer")
        plan = _make_plan(tmp_path, task)
        sig = plan_topology_signature(plan)
        assert "task.agent:reviewer" in sig

    def test_dynamic_group_branch(self, tmp_path: Path) -> None:
        """dynamic_group True -> 'task.dynamic_group:1' term."""
        task = TaskSpec(id="a", engine="claude", prompt="p", dynamic_group=True)
        plan = _make_plan(tmp_path, task)
        sig = plan_topology_signature(plan)
        assert "task.dynamic_group:1" in sig

    def test_deliberation_branch(self, tmp_path: Path) -> None:
        """deliberation True -> 'task.deliberation:1' term."""
        task = TaskSpec(id="a", engine="claude", prompt="p", deliberation=True)
        plan = _make_plan(tmp_path, task)
        sig = plan_topology_signature(plan)
        assert "task.deliberation:1" in sig

    def test_group_branch(self, tmp_path: Path) -> None:
        """group set -> 'task.group:1' term."""
        task = TaskSpec(id="a", group="sub-plan.yaml")
        plan = _make_plan(tmp_path, task)
        sig = plan_topology_signature(plan)
        assert "task.group:1" in sig

    def test_batch_branch(self, tmp_path: Path) -> None:
        """batch set -> 'task.batch:1' term."""
        task = TaskSpec(
            id="a",
            engine="claude",
            prompt="p",
            batch=BatchSpec(items=["x", "y"], template="process {{ batch.item }}"),
        )
        plan = _make_plan(tmp_path, task)
        sig = plan_topology_signature(plan)
        assert "task.batch:1" in sig

    def test_judge_branch(self, tmp_path: Path) -> None:
        """judge set -> 'task.judge:1' term."""
        task = TaskSpec(
            id="a",
            engine="claude",
            prompt="p",
            judge=JudgeSpec(criteria=["is good"], pass_threshold=0.7),
        )
        plan = _make_plan(tmp_path, task)
        sig = plan_topology_signature(plan)
        assert "task.judge:1" in sig

    def test_all_optional_task_branches_together(self, tmp_path: Path) -> None:
        """A task exercising several optional branches at once."""
        task = TaskSpec(
            id="MixedCase",
            engine="claude",
            prompt="p",
            agent="qa",
            dynamic_group=True,
            deliberation=True,
            batch=BatchSpec(items=["a"], template="t {{ batch.item }}"),
            judge=JudgeSpec(criteria=["ok"]),
        )
        plan = _make_plan(
            tmp_path,
            task,
            routing_strategy="balanced",
            firewall_model="sonnet",
        )
        sig = plan_topology_signature(plan)
        assert "plan.routing:balanced" in sig
        assert "plan.firewall:1" in sig
        assert "task.agent:qa" in sig
        assert "task.dynamic_group:1" in sig
        assert "task.deliberation:1" in sig
        assert "task.batch:1" in sig
        assert "task.judge:1" in sig
        # task id is lowercased into the signature
        assert "task.id:mixedcase" in sig


# ---------------------------------------------------------------------------
# store_score_history — exception fallback
# ---------------------------------------------------------------------------

class TestStoreScoreHistoryFallback:
    def test_store_returns_false_on_backend_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the memory backend raises, store_score_history returns False."""

        def _boom(*args: object, **kwargs: object) -> bool:
            raise RuntimeError("backend unavailable")

        monkeypatch.setattr(memory_mod, "store_score_record", _boom)
        result = store_score_history("demo", tmp_path, _make_score())
        assert result is False


# ---------------------------------------------------------------------------
# load_score_history — exception fallback
# ---------------------------------------------------------------------------

class TestLoadScoreHistoryFallback:
    def test_load_returns_empty_on_backend_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If the memory backend raises, load_score_history returns []."""

        def _boom(*args: object, **kwargs: object) -> list[ScoreRecord]:
            raise ValueError("corrupt store")

        monkeypatch.setattr(memory_mod, "load_score_records", _boom)
        result = load_score_history("demo", tmp_path, plan_hash="hash123")
        assert result == []


# ---------------------------------------------------------------------------
# get_historical_pruning_decision — exception fallback
# ---------------------------------------------------------------------------

class TestHistoricalPruningDecisionFallback:
    def test_decision_safe_default_on_backend_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Backend error -> conservative no-prune decision is returned."""

        def _boom(*args: object, **kwargs: object) -> HistoricalPruningDecision:
            raise RuntimeError("query failed")

        monkeypatch.setattr(memory_mod, "historical_pruning_decision", _boom)
        decision = get_historical_pruning_decision(
            "demo",
            tmp_path,
            "hash123",
            threshold=0.75,
            min_runs=4,
            recent_runs=15,
            horizon_days=10,
        )
        assert isinstance(decision, HistoricalPruningDecision)
        assert decision.prune is False
        assert decision.sample_size == 0
        assert decision.failures == 0
        assert decision.failure_rate == 0.0
        # Caller-supplied parameters are echoed into the safe default.
        assert decision.plan_hash == "hash123"
        assert decision.threshold == 0.75
        assert decision.min_runs == 4
        assert decision.recent_runs == 15
        assert decision.horizon_days == 10
