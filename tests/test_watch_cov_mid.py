from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.models import (
    PlanRunResult,
    PlanSpec,
    TaskResult,
    TaskSpec,
    WatchIteration,
    WatchSpec,
)
from maestro_cli.watch import _watch_improve, watch


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _improve_plan(
    tmp_path: Path,
    *,
    spec_overrides: dict[str, Any] | None = None,
    tasks: list[TaskSpec] | None = None,
    budget_period: str | None = None,
) -> PlanSpec:
    """Build a target plan configured for mode: improve, written to disk."""
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("version: 1\nname: cov-target\n", encoding="utf-8")
    base: dict[str, Any] = {
        "metric": "tasks_passed",
        "mode": "improve",
        "metric_source": "manifest",
        "metric_direction": "higher_is_better",
        "warmup_iterations": 0,
        "plateau_threshold": 3,
        "max_iterations": 3,
    }
    if spec_overrides:
        base.update(spec_overrides)
    plan = PlanSpec(
        name="cov-target",
        source_path=plan_path,
        workspace_root=".",
        run_dir=".maestro-runs",
        tasks=tasks or [TaskSpec(id="t1"), TaskSpec(id="t2")],
        watch=WatchSpec(**base),
    )
    if budget_period is not None:
        plan.budget_period = budget_period
    return plan


def _run_result(
    tmp_path: Path,
    *,
    name: str,
    n_success: int,
    cost: float = 0.1,
    run_subdir: str = "run",
) -> PlanRunResult:
    """Build a PlanRunResult with n_success success tasks (rest failed)."""
    run_path = tmp_path / run_subdir
    run_path.mkdir(parents=True, exist_ok=True)
    results: dict[str, TaskResult] = {}
    for i in range(2):
        status = "success" if i < n_success else "failed"
        results[f"t{i + 1}"] = TaskResult(
            task_id=f"t{i + 1}",
            status=status,
            exit_code=0 if status == "success" else 1,
            duration_sec=1.0,
            cost_usd=cost,
            log_path=run_path / f"t{i + 1}.log",
            result_path=run_path / f"t{i + 1}.json",
        )
    return PlanRunResult(
        plan_name=name,
        run_id=run_subdir,
        run_path=run_path,
        started_at=datetime.now(),
        finished_at=datetime.now(),
        success=n_success == 2,
        task_results=results,
        total_cost_usd=cost,
    )


def _stub_git_and_blame(monkeypatch: pytest.MonkeyPatch, *, rollback_ok: bool = True) -> None:
    """Stub out git operations and blame to keep tests offline/deterministic."""
    monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
    monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: rollback_ok)
    monkeypatch.setattr(
        "maestro_cli.watch.blame_run",
        lambda _p: type("C", (), {"to_dict": lambda self: {"nodes": []}})(),
    )


# ---------------------------------------------------------------------------
# Loop entry-guards: budget and plateau breaks
# ---------------------------------------------------------------------------


class TestImproveLoopGuards:
    def test_budget_exceeded_break(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When total_cost already >= max_cost_usd, loop sets budget_exceeded."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(
            tmp_path,
            spec_overrides={"max_cost_usd": 1.0, "max_iterations": 5},
        )

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            # Iteration 1 (baseline) returns big cost so iteration 2 trips budget
            return _run_result(tmp_path, name=plan_arg.name, n_success=1, cost=2.0)

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)
        _stub_git_and_blame(monkeypatch)

        state = _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
        )
        assert state.status == "budget_exceeded"

    def test_plateau_break(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When plateau_count reaches threshold, loop sets status plateau."""
        monkeypatch.chdir(tmp_path)
        # threshold 1 so a single regression after baseline trips plateau.
        # target_metric high so we never short-circuit on target_reached.
        plan = _improve_plan(
            tmp_path,
            spec_overrides={
                "plateau_threshold": 1,
                "max_iterations": 5,
                "on_regression": "rollback",
                "target_metric": 5.0,
            },
        )

        # baseline target -> metric 1.0 (improve over None). Improve agent + later
        # targets regress to 0 -> regression -> plateau_count reaches threshold.
        seq = iter([1, 0, 0, 0])

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            if plan_arg.name.startswith("improve-"):
                return _run_result(
                    tmp_path, name=plan_arg.name, n_success=1,
                    run_subdir=f"imp-{id(kwargs)}",
                )
            n = next(seq, 0)
            return _run_result(
                tmp_path, name=plan_arg.name, n_success=n, run_subdir=f"run-{n}-{id(kwargs)}"
            )

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)
        _stub_git_and_blame(monkeypatch)

        state = _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
        )
        assert state.status == "plateau"


# ---------------------------------------------------------------------------
# Improvement path: lateral keep, stepping stones, blame from prior run
# ---------------------------------------------------------------------------


class TestImproveImprovementPath:
    def test_lateral_fix_kept_when_metric_equals_best(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """metric == best_metric (not strictly better) is still kept as a lateral fix."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(tmp_path, spec_overrides={"max_iterations": 2})

        # baseline=1, iteration2=1 (equal -> lateral keep), also exercises blame ctx
        seq = iter([1, 1])

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            if plan_arg.name.startswith("improve-"):
                return _run_result(tmp_path, name=plan_arg.name, n_success=1, run_subdir="imp")
            n = next(seq, 1)
            return _run_result(tmp_path, name=plan_arg.name, n_success=n, run_subdir=f"tgt-{n}-{id(kwargs)}")

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)
        _stub_git_and_blame(monkeypatch)

        events: list[tuple[str, dict[str, object]]] = []
        state = _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
            event_callback=lambda n, p: events.append((n, p)),
        )
        # iteration 2 should be improved=True via lateral keep (metric == best)
        assert len(state.iterations) == 2
        assert state.iterations[1].improved is True
        assert "keep" in state.iterations[1].action

    def test_stepping_stone_saved_on_improvement(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With stepping_stones enabled, an improvement saves a stone + emits event."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(
            tmp_path,
            spec_overrides={"max_iterations": 2, "stepping_stones": True},
        )

        # baseline=1, iteration2=2 (strict improvement)
        seq = iter([1, 2])

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            if plan_arg.name.startswith("improve-"):
                return _run_result(tmp_path, name=plan_arg.name, n_success=1, run_subdir="imp2")
            n = next(seq, 2)
            return _run_result(tmp_path, name=plan_arg.name, n_success=n, run_subdir=f"t-{n}-{id(kwargs)}")

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)
        # Avoid touching the stepping-stone archive / disabling it would skip the branch.
        fake_stone = type("S", (), {"plan_hash": "abc123"})()
        monkeypatch.setattr("maestro_cli.watch._save_stepping_stone", lambda **_kw: fake_stone)
        _stub_git_and_blame(monkeypatch)

        events: list[tuple[str, dict[str, object]]] = []
        state = _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
            event_callback=lambda n, p: events.append((n, p)),
        )
        names = [n for n, _ in events]
        assert "stepping_stone_saved" in names
        stone_evt = next(p for n, p in events if n == "stepping_stone_saved")
        assert stone_evt["plan_hash"] == "abc123"
        assert state.best_metric == 2.0


# ---------------------------------------------------------------------------
# Regression path: rollback emit, rollback failure, plateau detected, reload
# ---------------------------------------------------------------------------


class TestImproveRegressionPath:
    def test_regression_rollback_failure_records_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A regression with a failing rollback records an error + plateau event."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(
            tmp_path,
            spec_overrides={
                "max_iterations": 2,
                "plateau_threshold": 1,
                "on_regression": "rollback",
                "target_metric": 5.0,
            },
        )

        # baseline=1, iteration2 target=0 (regression below best)
        seq = iter([1, 0])

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            if plan_arg.name.startswith("improve-"):
                return _run_result(tmp_path, name=plan_arg.name, n_success=1, run_subdir="impr")
            n = next(seq, 0)
            return _run_result(tmp_path, name=plan_arg.name, n_success=n, run_subdir=f"reg-{n}-{id(kwargs)}")

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)
        # Rollback fails -> error recorded; blame stubbed
        _stub_git_and_blame(monkeypatch, rollback_ok=False)

        events: list[tuple[str, dict[str, object]]] = []
        state = _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
            event_callback=lambda n, p: events.append((n, p)),
        )
        names = [n for n, _ in events]
        assert "regression_detected" in names
        assert "rollback_executed" in names
        assert "plateau_detected" in names
        # iteration 2 is the regression; it should carry an error string
        reg_iter = state.iterations[1]
        assert reg_iter.improved is False
        assert reg_iter.error is not None
        assert "failed" in reg_iter.error

    def test_regression_reload_exception_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If reloading the plan after a rollback raises, it is swallowed (loop survives)."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(
            tmp_path,
            spec_overrides={
                "max_iterations": 2,
                "plateau_threshold": 5,
                "on_regression": "rollback",
                "target_metric": 5.0,
            },
        )

        # baseline=1, iteration2 target=0 (regression)
        seq = iter([1, 0])

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            if plan_arg.name.startswith("improve-"):
                return _run_result(tmp_path, name=plan_arg.name, n_success=1, run_subdir="regrl-imp")
            n = next(seq, 0)
            return _run_result(tmp_path, name=plan_arg.name, n_success=n, run_subdir=f"regrl-{n}-{id(kwargs)}")

        # In-loop load_plan calls, in order:
        #   1: iter1 target reload (succeed)
        #   2: iter2 validation reload (succeed)
        #   3: iter2 target reload (succeed)
        #   4: iter2 post-rollback reload (raise -> swallowed)
        call = {"n": 0}

        def _flaky_load_plan(_p: Path) -> PlanSpec:
            call["n"] += 1
            if call["n"] == 4:
                raise RuntimeError("file vanished after rollback")
            return plan

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", _flaky_load_plan)
        _stub_git_and_blame(monkeypatch, rollback_ok=True)

        state = _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
        )
        # Loop completed both iterations despite the post-rollback reload failure.
        assert state.total_iterations == 2
        assert call["n"] >= 4


# ---------------------------------------------------------------------------
# Improve-agent phase: validation failure, OSError reading log, frozen tasks
# ---------------------------------------------------------------------------


class TestImproveAgentPhase:
    def test_validation_failed_after_improve_agent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """load_plan raising after the improve agent records a validation_failed iteration."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(tmp_path, spec_overrides={"max_iterations": 2})

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            return _run_result(tmp_path, name=plan_arg.name, n_success=1, run_subdir=f"v-{id(kwargs)}")

        # Iteration 1 (baseline) reloads the target plan once (must succeed).
        # Iteration 2 runs the improve agent, then reloads the plan for validation
        # (this 2nd in-loop call must raise to drive the validation-failed branch).
        call = {"n": 0}

        def _flaky_load_plan(_p: Path) -> PlanSpec:
            call["n"] += 1
            if call["n"] >= 2:
                raise RuntimeError("bad yaml after edit")
            return plan

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", _flaky_load_plan)
        _stub_git_and_blame(monkeypatch)

        events: list[tuple[str, dict[str, object]]] = []
        state = _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
            event_callback=lambda n, p: events.append((n, p)),
        )
        # baseline iteration 1 keeps; iteration 2 hits validation failure.
        actions = [it.action for it in state.iterations]
        assert "validation_failed" in actions
        failed_iter = next(it for it in state.iterations if it.action == "validation_failed")
        assert failed_iter.error is not None
        assert "bad yaml" in failed_iter.error
        # iteration_complete with validation_failed action must be emitted
        complete_evts = [
            p for n, p in events
            if n == "iteration_complete" and p.get("action") == "validation_failed"
        ]
        assert complete_evts

    def test_improve_log_oserror_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An OSError reading the improve-agent log is caught and does not crash."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(tmp_path, spec_overrides={"max_iterations": 2})

        # Build an improve-result whose log_path.read_text raises OSError.
        class _BadPath(type(tmp_path)):  # type: ignore[misc]
            pass

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            if plan_arg.name.startswith("improve-"):
                run_path = tmp_path / "imp-os"
                run_path.mkdir(parents=True, exist_ok=True)
                log = run_path / "improve.log"
                log.write_text("FIX: something\n", encoding="utf-8")

                class _Raising:
                    def exists(self) -> bool:
                        return True

                    def read_text(self, *a: Any, **kw: Any) -> str:
                        raise OSError("disk gone")

                tr = TaskResult(
                    task_id="improve-plan",
                    status="success",
                    exit_code=0,
                    duration_sec=1.0,
                    cost_usd=0.1,
                    log_path=run_path / "improve.log",
                    result_path=run_path / "improve.json",
                )
                # Swap log_path with an object whose read_text raises OSError.
                object.__setattr__(tr, "log_path", _Raising())
                return PlanRunResult(
                    plan_name=plan_arg.name,
                    run_id="imp-os",
                    run_path=run_path,
                    started_at=datetime.now(),
                    finished_at=datetime.now(),
                    success=True,
                    task_results={"improve-plan": tr},
                    total_cost_usd=0.1,
                )
            return _run_result(tmp_path, name=plan_arg.name, n_success=1, run_subdir=f"osr-{id(kwargs)}")

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)
        _stub_git_and_blame(monkeypatch)

        state = _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
        )
        # The loop completed without raising; both iterations are recorded.
        assert len(state.iterations) == 2

    def test_frozen_task_modified_triggers_rollback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A modified frozen task is detected and rolls back."""
        monkeypatch.chdir(tmp_path)
        frozen_task = TaskSpec(id="gate", frozen=True, timeout_sec=30)
        plan = _improve_plan(
            tmp_path,
            spec_overrides={"max_iterations": 2},
            tasks=[frozen_task, TaskSpec(id="t1")],
        )

        # After the improve agent, load_plan returns a plan whose frozen task changed.
        modified_task = TaskSpec(id="gate", frozen=True, timeout_sec=999)
        modified_plan = _improve_plan(
            tmp_path,
            spec_overrides={"max_iterations": 2},
            tasks=[modified_task, TaskSpec(id="t1")],
        )

        rollback_calls: list[Any] = []

        def _record_rollback(*a: Any, **kw: Any) -> bool:
            rollback_calls.append(a)
            return True

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            return _run_result(tmp_path, name=plan_arg.name, n_success=1, run_subdir=f"fz-{id(kwargs)}")

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        # First in-loop load_plan call (validation reload) returns the modified plan,
        # which is also used by the frozen-task check.
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: modified_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", _record_rollback)
        monkeypatch.setattr(
            "maestro_cli.watch.blame_run",
            lambda _p: type("C", (), {"to_dict": lambda self: {"nodes": []}})(),
        )

        _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
        )
        # The frozen-task guard must have invoked rollback at least once.
        assert rollback_calls


# ---------------------------------------------------------------------------
# Consolidation + budget ledger + terminal statuses
# ---------------------------------------------------------------------------


class TestImproveConsolidationAndBudget:
    def test_consolidation_invoked_on_schedule(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Consolidation runs when consolidate_model set and iteration count divisible."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(
            tmp_path,
            spec_overrides={
                "max_iterations": 3,
                "consolidate_model": "haiku",
                "consolidate_every": 1,
            },
        )

        consolidation_calls: list[Any] = []

        def _fake_consolidate(*a: Any, **kw: Any) -> str:
            consolidation_calls.append((a, kw))
            return "consolidated strategy text"

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            return _run_result(tmp_path, name=plan_arg.name, n_success=1, run_subdir=f"con-{id(kwargs)}")

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.watch._run_consolidation", _fake_consolidate)
        _stub_git_and_blame(monkeypatch)

        _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
        )
        # Consolidation should have been triggered at least once (iteration 2+).
        assert consolidation_calls

    def test_budget_ledger_records_improve_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When plan.budget_period is set and improve cost > 0, the ledger records it."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(
            tmp_path,
            spec_overrides={"max_iterations": 2},
            budget_period="daily",
        )

        recorded: list[tuple[Any, ...]] = []

        def _fake_record_cost(ledger: Any, name: str, label: str, cost: float) -> None:
            recorded.append((name, label, cost))

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            # improve-agent run carries cost so the ledger branch fires
            return _run_result(tmp_path, name=plan_arg.name, n_success=1, cost=0.5, run_subdir=f"bud-{id(kwargs)}")

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)
        monkeypatch.setattr("maestro_cli.budget.record_cost", _fake_record_cost)
        _stub_git_and_blame(monkeypatch)

        _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
        )
        # iteration 2 has improve-agent cost -> ledger recorded
        assert recorded
        assert recorded[0][1].startswith("improve-iter-")

    def test_max_iterations_terminal_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Completing the loop without breaking sets status max_iterations (for/else)."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(
            tmp_path,
            spec_overrides={"max_iterations": 2, "target_metric": 99.0},
        )

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            return _run_result(tmp_path, name=plan_arg.name, n_success=1, run_subdir=f"mx-{id(kwargs)}")

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)
        _stub_git_and_blame(monkeypatch)

        state = _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
        )
        assert state.status == "max_iterations"

    def test_keyboard_interrupt_sets_interrupted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A KeyboardInterrupt during the loop sets status interrupted."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(tmp_path, spec_overrides={"max_iterations": 3})

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            raise KeyboardInterrupt()

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)
        _stub_git_and_blame(monkeypatch)

        state = _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
        )
        assert state.status == "interrupted"


# ---------------------------------------------------------------------------
# Blame context from prior iteration's target run
# ---------------------------------------------------------------------------


class TestImproveBlameContext:
    def test_blame_context_built_from_prior_target_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second iteration builds blame context from the previous target run path."""
        monkeypatch.chdir(tmp_path)
        plan = _improve_plan(tmp_path, spec_overrides={"max_iterations": 2})

        # Write a manifest into the first target run dir so _build_blame_context
        # produces a non-empty manifest_summary on the second iteration.
        captured: list[dict[str, str]] = []
        counter = {"n": 0}

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            counter["n"] += 1
            if plan_arg.name.startswith("improve-"):
                captured.append(dict(kwargs.get("extra_template_vars", {})))
                return _run_result(tmp_path, name=plan_arg.name, n_success=1, run_subdir=f"impb-{counter['n']}")
            # target run with a manifest file for blame
            run_path = tmp_path / f"tgtb-{counter['n']}"
            run_path.mkdir(parents=True, exist_ok=True)
            manifest = {
                "task_results": {
                    "t1": {"status": "failed", "exit_code": 1, "duration_sec": 2.0, "message": "boom"},
                }
            }
            (run_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            results = {
                "t1": TaskResult(
                    task_id="t1", status="success", exit_code=0, duration_sec=1.0,
                    cost_usd=0.1, log_path=run_path / "t1.log", result_path=run_path / "t1.json",
                ),
            }
            return PlanRunResult(
                plan_name=plan_arg.name, run_id="tgtb", run_path=run_path,
                started_at=datetime.now(), finished_at=datetime.now(),
                success=True, task_results=results, total_cost_usd=0.1,
            )

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)
        _stub_git_and_blame(monkeypatch)

        _watch_improve(
            plan_path=plan.source_path,  # type: ignore[arg-type]
            plan=plan,
            spec=plan.watch,
        )
        # The improve agent on iteration 2 received a manifest excerpt referencing t1.
        assert captured, "improve agent should have run on iteration 2"
        assert "t1" in captured[-1].get("watch.manifest", "")


# ---------------------------------------------------------------------------
# watch() top-level guard branches (custom mode)
# ---------------------------------------------------------------------------


class TestWatchTopLevelGuards:
    def test_unresolvable_run_dir_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """watch() raises ValueError when the run directory cannot be resolved."""
        monkeypatch.chdir(tmp_path)
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text("version: 1\nname: nodir\n", encoding="utf-8")
        plan = PlanSpec(
            name="nodir",
            source_path=plan_path,
            workspace_root=".",
            run_dir=".maestro-runs",
            tasks=[TaskSpec(id="t1")],
            watch=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="t1",
                warmup_iterations=0,
                plateau_threshold=2,
                max_iterations=2,
            ),
        )
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _p: plan)

        # Force the watch-root resolution to fail (custom-mode branch).
        real_resolve = __import__("maestro_cli.watch", fromlist=["resolve_path"]).resolve_path

        def _fake_resolve(base: Path, value: Any) -> Path | None:
            if value == ".maestro-runs":
                return None
            return real_resolve(base, value)

        monkeypatch.setattr("maestro_cli.watch.resolve_path", _fake_resolve)

        with pytest.raises(ValueError, match="run directory"):
            watch(plan_path)

    def test_blame_plan_load_exception_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing blame_plan load in watch() is swallowed and the run proceeds."""
        monkeypatch.chdir(tmp_path)
        # Create a blame plan target path that load_plan will choke on.
        blame_target = tmp_path / "blame.yaml"
        blame_target.write_text("version: 1\nname: blame-target\n", encoding="utf-8")

        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text("version: 1\nname: with-blame\n", encoding="utf-8")
        plan = PlanSpec(
            name="with-blame",
            source_path=plan_path,
            workspace_root=".",
            run_dir=".maestro-runs",
            tasks=[TaskSpec(id="t1")],
            watch=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="t1",
                warmup_iterations=0,
                plateau_threshold=2,
                max_iterations=1,
                blame_plan="blame.yaml",
            ),
        )

        def _flaky_load_plan(p: Path) -> PlanSpec:
            if Path(p).name == "blame.yaml":
                raise RuntimeError("cannot load blame plan")
            return plan

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            return _run_result(tmp_path, name=plan_arg.name, n_success=1, run_subdir=f"bl-{id(kwargs)}")

        monkeypatch.setattr("maestro_cli.watch.load_plan", _flaky_load_plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        _stub_git_and_blame(monkeypatch)

        # dry_run=True keeps the run short; the blame_plan load happens before dry_run check.
        state = watch(plan_path, dry_run=True)
        # No exception means the except-branch swallowed the blame load failure.
        assert state is not None
