from __future__ import annotations

import pytest

from maestro_cli.models import PlanSpec, TaskSpec
from maestro_cli.routing import _score_task_complexity, resolve_auto_model


# ---------------------------------------------------------------------------
# Minimal fixtures
# ---------------------------------------------------------------------------

def _task(
    id: str = "t1",
    prompt: str = "Do something",
    tags: list[str] | None = None,
    **kwargs,
) -> TaskSpec:
    return TaskSpec(id=id, prompt=prompt, tags=tags or [], **kwargs)


def _plan() -> PlanSpec:
    return PlanSpec(name="test-plan")


# ---------------------------------------------------------------------------
# TestCostWeights
# ---------------------------------------------------------------------------

class TestCostWeights:
    def test_cost_optimized_increases_score(self) -> None:
        """cost_optimized should push the raw score up → cheaper (lower-tier) model."""
        task = _task(prompt="x" * 600)  # medium-range prompt
        plan = _plan()
        score_balanced = _score_task_complexity(task, plan, routing_strategy="balanced")
        score_cost = _score_task_complexity(task, plan, routing_strategy="cost_optimized")
        # cost_optimized raises the adjusted score → same or higher
        assert score_cost >= score_balanced

    def test_quality_first_decreases_score(self) -> None:
        """quality_first should push the raw score down → more capable (higher-tier) model."""
        task = _task(prompt="x" * 600)
        plan = _plan()
        score_balanced = _score_task_complexity(task, plan, routing_strategy="balanced")
        score_quality = _score_task_complexity(task, plan, routing_strategy="quality_first")
        # quality_first lowers the adjusted score → same or lower
        assert score_quality <= score_balanced

    def test_balanced_same_as_none(self) -> None:
        """routing_strategy='balanced' must produce the same score as routing_strategy=None."""
        task = _task(prompt="Fix this bug")
        plan = _plan()
        score_none = _score_task_complexity(task, plan, routing_strategy=None)
        score_balanced = _score_task_complexity(task, plan, routing_strategy="balanced")
        assert score_none == score_balanced

    def test_unknown_strategy_defaults_balanced(self) -> None:
        """An unrecognised strategy should fall back to the balanced (no-op) weights."""
        task = _task(prompt="Fix this bug")
        plan = _plan()
        score_balanced = _score_task_complexity(task, plan, routing_strategy="balanced")
        score_unknown = _score_task_complexity(task, plan, routing_strategy="totally_made_up")
        assert score_unknown == score_balanced


# ---------------------------------------------------------------------------
# TestStructuralScoring
# ---------------------------------------------------------------------------

class TestStructuralScoring:
    def test_high_fan_out_boosts_score(self) -> None:
        """fan_out > 3 should raise the complexity score."""
        task = _task()
        plan = _plan()
        score_no_meta = _score_task_complexity(task, plan)
        score_fan_out = _score_task_complexity(
            task,
            plan,
            dag_metadata={"fan_out": 5, "depth": 0, "upstream_failure_rate": 0.0},
        )
        assert score_fan_out > score_no_meta

    def test_deep_task_boosts_score(self) -> None:
        """depth > 4 should raise the complexity score."""
        task = _task()
        plan = _plan()
        score_no_meta = _score_task_complexity(task, plan)
        score_deep = _score_task_complexity(
            task,
            plan,
            dag_metadata={"fan_out": 0, "depth": 6, "upstream_failure_rate": 0.0},
        )
        assert score_deep > score_no_meta

    def test_upstream_failures_boost_score(self) -> None:
        """upstream_failure_rate > 0.3 should raise the complexity score."""
        task = _task()
        plan = _plan()
        score_no_meta = _score_task_complexity(task, plan)
        score_failures = _score_task_complexity(
            task,
            plan,
            dag_metadata={"fan_out": 0, "depth": 0, "upstream_failure_rate": 0.5},
        )
        assert score_failures > score_no_meta

    def test_no_metadata_unchanged(self) -> None:
        """dag_metadata=None must not alter the score compared to omitting it."""
        task = _task()
        plan = _plan()
        score_omitted = _score_task_complexity(task, plan)
        score_none = _score_task_complexity(task, plan, dag_metadata=None)
        assert score_omitted == score_none


# ---------------------------------------------------------------------------
# TestResolveAutoModelBackwardCompat
# ---------------------------------------------------------------------------

class TestResolveAutoModelBackwardCompat:
    def test_no_new_params_same_result(self) -> None:
        """Calling resolve_auto_model without the new params must return the same
        model as the original two-argument behaviour."""
        task = _task(prompt="Review and fix the algorithm")
        plan = _plan()

        result_new = resolve_auto_model(task, plan, "claude")
        result_compat = resolve_auto_model(
            task, plan, "claude", routing_strategy=None, dag_metadata=None
        )
        assert result_new == result_compat

    def test_with_strategy_returns_model(self) -> None:
        """A valid call with all parameters must succeed and return a non-empty model string."""
        task = _task(prompt="Audit the authentication flow", tags=["security"])
        plan = _plan()

        model = resolve_auto_model(
            task,
            plan,
            "claude",
            routing_strategy="quality_first",
            dag_metadata={"fan_out": 2, "depth": 3, "upstream_failure_rate": 0.0},
        )
        assert isinstance(model, str)
        assert model  # non-empty

    @pytest.mark.parametrize("engine", ["claude", "codex", "gemini", "copilot", "qwen", "ollama"])
    def test_all_engines_return_known_model(self, engine: str) -> None:
        """Every supported engine must resolve 'auto' to a concrete, non-empty model."""
        task = _task()
        plan = _plan()
        model = resolve_auto_model(task, plan, engine)
        assert isinstance(model, str)
        assert model != "auto"

    def test_unknown_engine_returns_auto(self) -> None:
        """An engine not in the tier table must return 'auto' unchanged."""
        task = _task()
        plan = _plan()
        result = resolve_auto_model(task, plan, "nonexistent_engine")
        assert result == "auto"
