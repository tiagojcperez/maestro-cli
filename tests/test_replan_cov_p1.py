from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from maestro_cli.audit import AuditFinding
from maestro_cli.models import (
    PlanRunResult,
    PlanSpec,
    ReplanState,
    ScoreRecord,
    TaskResult,
    WorkflowVariant,
)
from maestro_cli.replan import (
    _GeneratedPlanSecurityDecision,
    _record_simulation_cache_hit,
    _rehydrate_cached_run_result,
    replan,
)


# ──────────────────────────── shared helpers ─────────────────────────────────

_VALID_PLAN = (
    "version: 1\n"
    "name: test\n"
    "tasks:\n"
    "  - id: t1\n"
    "    command: echo hello\n"
)


def _fail_result(tmp_path: Path, *, run_id: str = "r1", message: str = "boom") -> PlanRunResult:
    now = datetime.now()
    return PlanRunResult(
        plan_name="test",
        run_id=run_id,
        run_path=tmp_path,
        started_at=now,
        finished_at=now,
        success=False,
        task_results={
            "t1": TaskResult(
                task_id="t1",
                status="failed",
                exit_code=1,
                message=message,
            ),
        },
    )


def _make_score_record(
    *,
    plan_name: str = "test",
    plan_hash: str = "deadbeef",
    run_id: str = "cached-run",
    success: bool = True,
    timestamp: str = "2026-04-02T00:00:00+00:00",
    metadata: dict[str, object] | None = None,
) -> ScoreRecord:
    return ScoreRecord(
        plan_name=plan_name,
        plan_hash=plan_hash,
        run_id=run_id,
        success=success,
        cost_usd=0.1,
        quality_score=1.0 if success else 0.0,
        duration_sec=1.0,
        timestamp=timestamp,
        valid_from=timestamp,
        recorded_at=timestamp,
        source_id=f"{run_id}:score",
        metadata=metadata or {},
    )


# ──────────────────────── _GeneratedPlanSecurityDecision.to_dict ─────────────


class TestGeneratedPlanSecurityDecisionToDict:
    def test_to_dict_includes_firewall_category_and_reason(self) -> None:
        # Drives the `if self.firewall_category` and
        # `if self.firewall_reason` branches.
        decision = _GeneratedPlanSecurityDecision(
            blocked=True,
            introduced_findings=[],
            firewall_verdict="block",
            firewall_category="policy_bypass",
            firewall_reason="attempted override",
        )
        data = decision.to_dict()
        assert data["blocked"] is True
        assert data["firewall_verdict"] == "block"
        assert data["firewall_category"] == "policy_bypass"
        assert data["firewall_reason"] == "attempted override"

    def test_to_dict_omits_empty_firewall_fields(self) -> None:
        decision = _GeneratedPlanSecurityDecision(
            blocked=False,
            introduced_findings=[
                AuditFinding(
                    severity="warning",
                    rule="SEC001",
                    message="no budget",
                    task_id="",
                ),
            ],
        )
        data = decision.to_dict()
        assert "firewall_category" not in data
        assert "firewall_reason" not in data
        assert data["introduced_findings"][0]["rule"] == "SEC001"


# ──────────────────────── _rehydrate_cached_run_result ───────────────────────


class TestRehydrateCachedRunResult:
    def _variant(self) -> WorkflowVariant:
        plan = PlanSpec(name="test")
        return WorkflowVariant(node_id="n1", plan_spec=plan, plan_hash="variant-hash")

    def test_naive_timestamp_is_assumed_utc(self) -> None:
        # Naive ISO timestamp (no tzinfo) drives the `tzinfo is None` branch
        # at -> replace(tzinfo=timezone.utc).
        record = _make_score_record(timestamp="2026-04-02T00:00:00")
        result = _rehydrate_cached_run_result(self._variant(), record)
        assert result.started_at.tzinfo is not None
        assert result.started_at.utcoffset() == timezone.utc.utcoffset(None)
        assert result.execution_profile == "plan"

    def test_execution_profile_safe(self) -> None:
        # Drives : execution_profile == "safe".
        record = _make_score_record(metadata={"execution_profile": "safe"})
        result = _rehydrate_cached_run_result(self._variant(), record)
        assert result.execution_profile == "safe"

    def test_execution_profile_yolo(self) -> None:
        # Drives : execution_profile == "yolo".
        record = _make_score_record(metadata={"execution_profile": "yolo"})
        result = _rehydrate_cached_run_result(self._variant(), record)
        assert result.execution_profile == "yolo"

    def test_aware_timestamp_is_converted_to_utc(self) -> None:
        record = _make_score_record(timestamp="2026-04-02T03:00:00+03:00")
        result = _rehydrate_cached_run_result(self._variant(), record)
        # 03:00 +03:00 -> 00:00 UTC
        assert result.started_at.hour == 0
        assert result.started_at.utcoffset() == timezone.utc.utcoffset(None)


# ──────────────────────── _record_simulation_cache_hit ──────────────────────


class TestRecordSimulationCacheHit:
    def test_sets_search_tree_path_when_unset(self, tmp_path: Path) -> None:
        # Drives : state.search_tree_path is None -> assign.
        plan = PlanSpec(name="test")
        variant = WorkflowVariant(node_id="n1", plan_spec=plan, plan_hash="vh")
        state = ReplanState(plan_path="p.yaml", max_attempts=1)
        assert state.search_tree_path is None
        record = _make_score_record(run_id="src-run", plan_hash="ph")

        _record_simulation_cache_hit(
            state,
            variant,
            record,
            search_tree_dir=tmp_path,
            score_discount=0.9,
            simulation_plan_hash="sim-hash",
        )

        assert state.search_tree_path == str(tmp_path / "tree.jsonl")
        assert variant.run_result is not None
        assert variant.metadata["simulation_cache"]["hit"] is True
        assert variant.metadata["simulation_cache"]["source_run_id"] == "src-run"
        assert (tmp_path / "tree.jsonl").is_file()

    def test_leaves_existing_search_tree_path(self, tmp_path: Path) -> None:
        plan = PlanSpec(name="test")
        variant = WorkflowVariant(node_id="n2", plan_spec=plan, plan_hash="vh2")
        existing = str(tmp_path / "already" / "tree.jsonl")
        state = ReplanState(
            plan_path="p.yaml",
            max_attempts=1,
            search_tree_path=existing,
        )
        record = _make_score_record(run_id="src-run-2", plan_hash="ph2")

        _record_simulation_cache_hit(
            state,
            variant,
            record,
            search_tree_dir=tmp_path,
            score_discount=0.9,
            simulation_plan_hash="sim-hash-2",
        )

        assert state.search_tree_path == existing


# ──────────────────────────── replan() validation ───────────────────────────


class TestReplanValidation:
    def test_variants_below_one(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")
        with pytest.raises(ValueError, match="variants must be >= 1"):
            replan(plan_path, variants=0)

    def test_debug_prob_out_of_range_low(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")
        with pytest.raises(ValueError, match="debug_prob must be between"):
            replan(plan_path, debug_prob=-0.1)

    def test_debug_prob_out_of_range_high(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")
        with pytest.raises(ValueError, match="debug_prob must be between"):
            replan(plan_path, debug_prob=1.5)

    def test_invalid_selection_policy(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")
        with pytest.raises(ValueError, match="selection_policy must be"):
            replan(plan_path, selection_policy="bogus")  # type: ignore[arg-type]

    def test_negative_exploration_constant(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")
        with pytest.raises(ValueError, match="exploration_constant must be >= 0.0"):
            replan(plan_path, exploration_constant=-1.0)

    def test_invalid_population_strategy(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")
        with pytest.raises(ValueError, match="population_strategy must be"):
            replan(plan_path, population_strategy="random")  # type: ignore[arg-type]

    def test_tournament_size_below_one(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")
        with pytest.raises(ValueError, match="tournament_size must be >= 1"):
            replan(plan_path, tournament_size=0)

    def test_negative_elite_count(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")
        with pytest.raises(ValueError, match="elite_count must be >= 0"):
            replan(plan_path, elite_count=-1)

    def test_diversity_floor_out_of_range(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")
        with pytest.raises(ValueError, match="diversity_floor must be between"):
            replan(plan_path, diversity_floor=2.0)


# ──────────────────── _replan_single: trusted firewall pickup ────────────────


class TestReplanSingleTrustedFirewall:
    def test_trusted_firewall_model_picked_up_from_corrected_plan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The baseline plan has no firewall_model, but the corrected (accepted)
        # plan declares one. This drives : trusted_firewall_model is set
        # from candidate_plan.firewall_model after a clean security gate. The
        # firewall is then used on the next attempt's analysis -> we assert
        # pass-2 ran with that model.
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")

        monkeypatch.setattr(
            "maestro_cli.replan.run_plan",
            lambda *a, **kw: _fail_result(tmp_path, message="attempt failure"),
        )

        # First attempt's corrected plan introduces firewall_model: haiku.
        # The DAG fails again so a second attempt runs and re-loads the corrected
        # plan; that second attempt's analysis triggers a pass-2 firewall call
        # using the trusted model.
        corrected_with_firewall = (
            "version: 1\n"
            "name: test\n"
            "firewall_model: haiku\n"
            "max_parallel: 2\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
        )
        corrected_alt = (
            "version: 1\n"
            "name: test\n"
            "firewall_model: haiku\n"
            "max_parallel: 3\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
        )

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                return f"```yaml\n{corrected_with_firewall}```"
            return f"```yaml\n{corrected_alt}```"

        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        firewall_models: list[str] = []

        class _AllowDecision:
            verdict = "allow"
            category = ""
            reason = ""

        def _fake_firewall(model: str, *a: object, **kw: object) -> _AllowDecision:
            firewall_models.append(model)
            return _AllowDecision()

        monkeypatch.setattr("maestro_cli.replan._run_firewall_pass2", _fake_firewall)

        state = replan(plan_path, max_attempts=2, auto_approve=True)

        assert state.final_success is False
        # Second attempt re-loaded the corrected plan whose firewall_model is now
        # the trusted model and used it on the pass-2 gate.
        assert "haiku" in firewall_models


# ──────────────────── _replan_search: circuit breaker (466-468) ──────────────


class TestReplanSearchCircuitBreaker:
    def test_circuit_breaker_on_repeated_failed_task_set(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")

        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=tmp_path / "runs" / run_dir,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        run_counter = {"n": 0}

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            # All candidate simulations keep failing on the same task set so the
            # selected variant carries the identical failed_task_ids across
            # attempts, eventually tripping _detect_exit_loop at the top of the
            # search loop .
            return _result(f"cand-{run_counter['n']}", f"cand-{run_counter['n']}", False, "still failing")

        # Each attempt produces a distinct candidate so it is selectable, but
        # all candidates fail the same task.
        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                f"max_parallel: {analysis_calls['n'] + 1}\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "```"
            )

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        state = replan(
            plan_path,
            max_attempts=8,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
        )

        out = capsys.readouterr().out
        assert state.status == "circuit_breaker"
        assert "Circuit breaker" in out


# ──────────────── _replan_search: firewall pickup + load failure ─────────────


class TestReplanSearchLoadAndFirewall:
    def test_load_failure_records_error_and_continues(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # The plan file becomes unreadable when re-loaded inside the search loop.
        # We monkeypatch load_plan to raise on every call so the except block
        # runs on the very first attempt; with max_attempts=1
        # the loop ends and the search returns. This also exercises the search
        # entry path (variants>1) before any run_plan call.
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")

        from maestro_cli.errors import PlanValidationError

        def _boom(*a: object, **kw: object) -> PlanSpec:
            raise PlanValidationError("forced load failure")

        monkeypatch.setattr("maestro_cli.replan.load_plan", _boom)

        state = replan(plan_path, max_attempts=1, auto_approve=True, variants=2)

        out = capsys.readouterr().out
        assert state.final_success is False
        assert len(state.attempts) == 1
        assert "Failed to load plan" in state.attempts[0].error_summary
        assert "Failed to load plan" in out

    def test_trusted_firewall_model_from_root_plan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Root plan declares firewall_model -> drives (trusted firewall
        # pickup at the top of the search loop) and (search_tree_path
        # set from the first run's run_path). A blocking candidate firewall
        # verdict with a category exercises (firewall_category captured
        # in the candidate summary).
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            "version: 1\n"
            "name: test\n"
            "firewall_model: haiku\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n",
            encoding="utf-8",
        )

        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=tmp_path / "runs" / run_dir,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        run_counter = {"n": 0}

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-2" in source_name:
                return _result("cand-2", "cand-2", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                return (
                    "```yaml\n"
                    "version: 1\n"
                    "name: test\n"
                    "firewall_model: haiku\n"
                    "max_parallel: 2\n"
                    "tasks:\n"
                    "  - id: t1\n"
                    "    command: echo hello\n"
                    "```"
                )
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "firewall_model: haiku\n"
                "max_parallel: 3\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "```"
            )

        class _BlockedDecision:
            verdict = "block"
            category = "policy_bypass"
            reason = "attempted override"

        # First candidate is blocked (with a category), second is allowed and
        # succeeds.
        firewall_calls = {"n": 0}

        class _AllowDecision:
            verdict = "allow"
            category = ""
            reason = ""

        def _fake_firewall(model: str, label: str, content: str, *a: object, **kw: object):
            firewall_calls["n"] += 1
            if "max_parallel: 2" in content:
                return _BlockedDecision()
            return _AllowDecision()

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)
        monkeypatch.setattr("maestro_cli.replan._run_firewall_pass2", _fake_firewall)

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
        )

        assert state.final_success is True
        assert state.status == "success"
        attempt = state.attempts[0]
        assert len(attempt.candidate_variants) == 2
        blocked = next(c for c in attempt.candidate_variants if c.get("security_blocked"))
        assert blocked["firewall_verdict"] == "block"
        assert blocked["firewall_category"] == "policy_bypass"
        assert state.search_tree_path is not None
        # search_tree_path derives from the first (root) run's run_path .
        assert str(tmp_path / "runs" / "root-run") in state.search_tree_path


# ──────────────── _replan_search: approval gate sets search_approved ─────────


class TestReplanSearchApprovalApproved:
    def test_search_approved_set_after_yes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # auto_approve=False but user answers "y" -> drives
        # (search_approved = True) and the input prompt is asked only once.
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")

        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=tmp_path / "runs" / run_dir,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        run_counter = {"n": 0}

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-2" in source_name:
                return _result("cand-2", "cand-2", True)
            return _result("cand-other", "cand-other", False, "still failing")

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                f"max_parallel: {analysis_calls['n'] + 1}\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "```"
            )

        input_calls = {"n": 0}

        def _fake_input(_prompt: str) -> str:
            input_calls["n"] += 1
            return "y"

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)
        monkeypatch.setattr("builtins.input", _fake_input)

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=False,
            variants=2,
            debug_prob=0.0,
        )

        assert state.final_success is True
        assert state.status == "success"
        assert state.attempts[0].approved is True
        # Prompt is asked exactly once, then search_approved stays True.
        assert input_calls["n"] == 1


# ──────────────── _replan_search: per-candidate error paths ──────────────────


class TestReplanSearchCandidateErrorPaths:
    """Drives candidate-loop continue branches: analysis error (611-614),
    no YAML fence (618-620), invalid corrected plan (631-634)."""

    def _plan_path(self, tmp_path: Path) -> Path:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(_VALID_PLAN, encoding="utf-8")
        return plan_path

    def _root_then_success(self, tmp_path: Path, run_counter: dict[str, int]):
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=tmp_path / "runs" / run_dir,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            # The "good" candidate succeeds.
            return _result("cand-good", "cand-good", True)

        return _fake_run_plan

    def test_candidate_analysis_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = self._plan_path(tmp_path)
        run_counter = {"n": 0}
        monkeypatch.setattr(
            "maestro_cli.replan.run_plan",
            self._root_then_success(tmp_path, run_counter),
        )

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                raise RuntimeError("API down")
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "max_parallel: 3\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "```"
            )

        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
        )

        attempt = state.attempts[0]
        assert len(attempt.candidate_variants) == 2
        errored = attempt.candidate_variants[0]
        assert "Analysis failed" in errored["error"]
        assert state.final_success is True

    def test_candidate_no_yaml_fence(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = self._plan_path(tmp_path)
        run_counter = {"n": 0}
        monkeypatch.setattr(
            "maestro_cli.replan.run_plan",
            self._root_then_success(tmp_path, run_counter),
        )

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                return "I forgot the fence, here is plain text only."
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "max_parallel: 3\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "```"
            )

        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
        )

        attempt = state.attempts[0]
        errored = attempt.candidate_variants[0]
        assert "No YAML code fence" in errored["error"]
        assert state.final_success is True

    def test_candidate_invalid_corrected_plan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = self._plan_path(tmp_path)
        run_counter = {"n": 0}
        monkeypatch.setattr(
            "maestro_cli.replan.run_plan",
            self._root_then_success(tmp_path, run_counter),
        )

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                # Invalid schema version -> load_plan raises PlanValidationError.
                return "```yaml\nversion: 99\nname: bad\n```"
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "max_parallel: 3\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "```"
            )

        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
        )

        attempt = state.attempts[0]
        errored = attempt.candidate_variants[0]
        assert "Corrected plan invalid" in errored["error"]
        assert state.final_success is True
