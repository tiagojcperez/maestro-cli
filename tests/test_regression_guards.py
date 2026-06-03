"""Regression guards: reference plans that must always validate.

Each plan in tests/fixtures/plans/ exercises a specific feature surface
of the Maestro YAML schema. If a change to loader.py, models.py, or
validate_plan() breaks any of these, the test catches it immediately.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.loader import load_plan

_PLANS_DIR = Path(__file__).resolve().parent / "fixtures" / "plans"


def _plan_files() -> list[Path]:
    """Collect all .yaml files in fixtures/plans/."""
    plans = sorted(_PLANS_DIR.glob("*.yaml"))
    assert plans, f"No reference plans found in {_PLANS_DIR}"
    return plans


@pytest.mark.parametrize(
    "plan_path",
    _plan_files(),
    ids=lambda p: p.stem,
)
def test_reference_plan_loads(plan_path: Path) -> None:
    """Reference plan loads and validates without errors."""
    plan = load_plan(plan_path)
    assert plan.name, f"Plan {plan_path.name} has no name"
    assert plan.tasks, f"Plan {plan_path.name} has no tasks"


@pytest.mark.parametrize(
    "plan_path",
    _plan_files(),
    ids=lambda p: p.stem,
)
def test_reference_plan_no_unexpected_warnings(plan_path: Path) -> None:
    """Reference plans should not produce unexpected warnings.

    Some warnings are expected (e.g. no-timeout, env var refs).
    This test filters those out and catches anything new.
    """
    plan = load_plan(plan_path)
    warnings = getattr(plan, "validation_warnings", []) or []
    # Filter out expected warnings that are legitimate for test plans:
    unexpected = [
        w for w in warnings
        if not any(skip in w for skip in [
            "no explicit timeout_sec",
            "more task(s) without explicit timeout",
            "not in the env allowlist",
            "shell=True",         # W1: string commands on Windows
            "contains pipe",      # pipe warning on Windows
            "bin/bash",           # W1: wrong bash binary
            "may not be valid",   # model validation advisory
            "W17:",                              # high dependency density advisory (v1.14.0)
            "W18:",                              # deep DAG without max_parallel advisory
            "W19:",                              # deliberation without context_from
            "W20:",                              # retries without escape valve advisory (consolidated 2026-04-26)
            "has verify_command but max_retries=0",  # verify without retry
            "has context_from but no context_budget_tokens",  # budget advisory
            "judge on_fail='retry' without max_iterations",  # iteration advisory
            "has assert rules but max_retries=0",  # assert without retry
            "observation_block: true has no effect",  # observation advisory
            "worktree: true but only one worktree",  # W16: single worktree
            "judge 'contains' assertion on engine",  # judge advisory
            "judge 'regex' assertion on engine",  # judge advisory
            "W22:",                              # judge timeout auto-scale advisory
            "W29:",                              # codex without fallback in fail_fast
            "W30:",                              # tsc --noEmit gate advisory
        ])
    ]
    assert not unexpected, (
        f"Unexpected warnings for {plan_path.name}:\n"
        + "\n".join(f"  - {w}" for w in unexpected)
    )


class TestPlanFeatureCoverage:
    """Verify that reference plans actually exercise the features they claim."""

    def test_all_engines_covered(self) -> None:
        """all_engines.yaml has one task per engine."""
        plan = load_plan(_PLANS_DIR / "all_engines.yaml")
        engines = {t.engine for t in plan.tasks if t.engine}
        assert engines == {"codex", "claude", "gemini", "copilot", "qwen", "ollama", "llama"}

    def test_all_context_modes_covered(self) -> None:
        """deps_and_context.yaml exercises all 5 context modes."""
        plan = load_plan(_PLANS_DIR / "deps_and_context.yaml")
        modes = {t.context_mode for t in plan.tasks if t.context_mode}
        assert modes == {"raw", "summarized", "map_reduce", "recursive", "layered"}

    def test_judge_features(self) -> None:
        """judge_and_quality.yaml has judge with assertions + quorum."""
        plan = load_plan(_PLANS_DIR / "judge_and_quality.yaml")
        judges = [t for t in plan.tasks if t.judge]
        assert len(judges) == 2
        # First task has rubric + typed assertions
        j1 = judges[0].judge
        assert j1 is not None
        assert j1.method == "direct"
        assert j1.aggregation == "weighted_mean"
        # Second task has quorum
        j2 = judges[1].judge
        assert j2 is not None
        assert j2.quorum == 3
        assert j2.quorum_strategy == "majority"

    def test_matrix_expansion(self) -> None:
        """matrix_expansion.yaml expands to 3 matrix tasks."""
        plan = load_plan(_PLANS_DIR / "matrix_expansion.yaml")
        assert len(plan.tasks) == 3  # 3 expanded from matrix
        ids = {t.id for t in plan.tasks}
        # Matrix IDs use dot-separated format: parent.key-value
        assert any("loader" in tid for tid in ids), f"No loader task in {ids}"
        assert any("runners" in tid for tid in ids), f"No runners task in {ids}"
        assert any("scheduler" in tid for tid in ids), f"No scheduler task in {ids}"

    def test_resilience_features(self) -> None:
        """resilience.yaml has escalation, fallback, circuit_breaker, when."""
        plan = load_plan(_PLANS_DIR / "resilience.yaml")
        primary = next(t for t in plan.tasks if t.id == "primary")
        assert primary.escalation == ["haiku", "sonnet", "opus"]
        assert primary.fallback_engine == "gemini"
        assert primary.fallback_model == "flash"
        assert primary.retry_strategy == "exponential"
        assert primary.allow_failure is True
        assert plan.circuit_breaker is not None
        assert plan.circuit_breaker.max_total_failures == 3
        conditional = next(t for t in plan.tasks if t.id == "conditional")
        assert conditional.when is not None

    def test_policies(self) -> None:
        """policies.yaml has 3 policies + routing_strategy + auto model."""
        plan = load_plan(_PLANS_DIR / "policies.yaml")
        assert len(plan.policies) == 3
        actions = {p.action for p in plan.policies}
        assert actions == {"warn", "block", "audit"}
        assert plan.routing_strategy == "balanced"
        impl = next(t for t in plan.tasks if t.id == "impl")
        assert impl.model == "auto"

    def test_watch_block(self) -> None:
        """watch_block.yaml has full watch spec."""
        plan = load_plan(_PLANS_DIR / "watch_block.yaml")
        w = plan.watch
        assert w is not None
        assert w.metric == "test_count"
        assert w.metric_direction == "higher_is_better"
        assert w.max_iterations == 10
        assert w.target_metric == 100.0
        assert w.plateau_action == "stop"
        assert w.on_regression == "rollback"

    def test_approval_and_worktree(self) -> None:
        """approval_and_worktree.yaml has worktree + approval gate."""
        plan = load_plan(_PLANS_DIR / "approval_and_worktree.yaml")
        safe = next(t for t in plan.tasks if t.id == "safe-change")
        assert safe.worktree is True
        dangerous = next(t for t in plan.tasks if t.id == "dangerous-change")
        assert dangerous.requires_approval is True
        assert dangerous.approval_message is not None

    def test_secrets_and_cfi(self) -> None:
        """secrets_and_cfi.yaml has secrets: auto + CFI + observation_block."""
        plan = load_plan(_PLANS_DIR / "secrets_and_cfi.yaml")
        assert plan.secrets_auto is True  # secrets: auto → secrets_auto=True
        assert plan.control_flow_integrity is True
        process = next(t for t in plan.tasks if t.id == "process")
        assert process.observation_block is True
