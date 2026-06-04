from __future__ import annotations

import pytest
from datetime import datetime, UTC
from pathlib import Path

from maestro_cli.models import (
    ASSERTION_TYPES,
    CODEX_MODEL_ALIASES,
    CONTEXT_MODES,
    CONTEXT_WINDOWS,
    COPILOT_MODEL_ALIASES,
    EDIT_POLICIES,
    GEMINI_MODEL_ALIASES,
    JUDGE_METHODS,
    JUDGE_ON_FAIL_VALUES,
    JUDGE_PRESETS,
    MAX_RETRIES_LIMIT,
    OLLAMA_MODEL_ALIASES,
    QWEN_MODEL_ALIASES,
    RECURSIVE_CONTEXT_STAGES,
    SCORE_AGGREGATIONS,
    STATUS_STYLES,
    TERMINAL_STATUSES,
    BlameChain,
    BlameNode,
    CircuitBreakerSpec,
    DualVerificationResult,
    ContextSelectionEntry,
    CriterionScore,
    EngineDefaults,
    EventRecord,
    FailureRecord,
    HistoricalPruningDecision,
    HandoffReport,
    JudgeResult,
    JudgeSpec,
    MultiPlanResult,
    PlanBrief,
    PlanDefaults,
    PlanImport,
    PlanRunResult,
    PlanSpec,
    PlanSuggestions,
    PolicySpec,
    PolicyViolation,
    ReplanAttempt,
    ReplanState,
    RubricCriterion,
    RubricLevel,
    ScoreRecord,
    SessionSnapshot,
    StructuredContext,
    Suggestion,
    TaskBrief,
    TaskResult,
    TaskSpec,
    TokenUsage,
    WatchIteration,
    WatchSpec,
    WatchState,
    WorkspaceBrief,
    WorkspaceExtraction,
    WorktreeMergeResult,
)


class TestTaskSpecContextFrom:
    def test_default_empty(self) -> None:
        task = TaskSpec(id="t1")
        assert task.context_from == []

    def test_set_explicit(self) -> None:
        task = TaskSpec(id="t1", context_from=["a", "b"])
        assert task.context_from == ["a", "b"]

    def test_wildcard(self) -> None:
        task = TaskSpec(id="t1", context_from=["*"])
        assert task.context_from == ["*"]


class TestTaskResultStdoutTail:
    def test_default_empty(self) -> None:
        now = datetime.now(UTC)
        result = TaskResult(
            task_id="t1",
            status="success",
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=0.0,
            command="echo hello",
            log_path=Path("/tmp/t1.log"),
            result_path=Path("/tmp/t1.result.json"),
        )
        assert result.stdout_tail == ""

    def test_set_explicit(self) -> None:
        now = datetime.now(UTC)
        result = TaskResult(
            task_id="t1",
            status="success",
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=1.5,
            command="echo hello",
            log_path=Path("/tmp/t1.log"),
            result_path=Path("/tmp/t1.result.json"),
            stdout_tail="line1\nline2\n",
        )
        assert result.stdout_tail == "line1\nline2\n"

    def test_to_dict_includes_stdout_tail(self) -> None:
        now = datetime.now(UTC)
        result = TaskResult(
            task_id="t1",
            status="success",
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=1.0,
            command="echo hello",
            log_path=Path("/tmp/t1.log"),
            result_path=Path("/tmp/t1.result.json"),
            stdout_tail="output text\n",
        )
        d = result.to_dict()
        assert "stdout_tail" in d
        assert d["stdout_tail"] == "output text\n"


class TestTaskBrief:
    def test_defaults(self) -> None:
        brief = TaskBrief(id="t1")
        assert brief.task_type == "implementation"
        assert brief.depends_on == []
        assert brief.engine is None
        assert brief.prompt_hint == ""

    def test_set_fields(self) -> None:
        brief = TaskBrief(
            id="sec",
            description="Security audit",
            task_type="security-audit",
            agent="security-engineer",
        )
        assert brief.task_type == "security-audit"
        assert brief.agent == "security-engineer"


class TestPlanBrief:
    def test_defaults(self) -> None:
        brief = PlanBrief(name="test")
        assert brief.max_parallel == 3
        assert brief.fail_fast is True
        assert brief.include_quality_gates is True
        assert brief.include_build_verify is True
        assert brief.topology == "pipeline"
        assert brief.tasks == []

    def test_with_tasks(self) -> None:
        brief = PlanBrief(
            name="test",
            tasks=[TaskBrief(id="a"), TaskBrief(id="b", depends_on=["a"])],
        )
        assert len(brief.tasks) == 2
        assert brief.tasks[1].depends_on == ["a"]


class TestPlanDefaultsStdoutTailLines:
    """Tests for PlanDefaults.stdout_tail_lines field."""

    def test_default_is_50(self) -> None:
        defaults = PlanDefaults()
        assert defaults.stdout_tail_lines == 50

    def test_custom_value(self) -> None:
        defaults = PlanDefaults(stdout_tail_lines=200)
        assert defaults.stdout_tail_lines == 200


class TestTaskResultCostUsd:
    """Tests for TaskResult.cost_usd field."""

    def test_default_none(self) -> None:
        now = datetime.now(UTC)
        result = TaskResult(
            task_id="t1",
            status="success",
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=1.0,
            command="echo hello",
            log_path=Path("/tmp/t1.log"),
            result_path=Path("/tmp/t1.result.json"),
        )
        assert result.cost_usd is None

    def test_to_dict_includes_cost_usd(self) -> None:
        now = datetime.now(UTC)
        result = TaskResult(
            task_id="t1",
            status="success",
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=1.0,
            command="echo hello",
            log_path=Path("/tmp/t1.log"),
            result_path=Path("/tmp/t1.result.json"),
            cost_usd=2.50,
        )
        d = result.to_dict()
        assert "cost_usd" in d
        assert d["cost_usd"] == 2.50

    def test_to_dict_cost_usd_none(self) -> None:
        now = datetime.now(UTC)
        result = TaskResult(
            task_id="t1",
            status="success",
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=1.0,
            command="echo hello",
            log_path=Path("/tmp/t1.log"),
            result_path=Path("/tmp/t1.result.json"),
        )
        d = result.to_dict()
        assert "cost_usd" in d
        assert d["cost_usd"] is None


class TestPlanRunResultMetrics:
    """Tests for PlanRunResult parallelism and cost metric fields."""

    def test_default_values(self) -> None:
        now = datetime.now(UTC)
        result = PlanRunResult(
            plan_name="test",
            run_id="abc",
            run_path=Path("/tmp/runs/abc"),
            started_at=now,
            finished_at=now,
            success=True,
        )
        assert result.sequential_duration_sec == 0.0
        assert result.parallelism_savings_pct == 0.0
        assert result.total_cost_usd is None

    def test_to_dict_includes_new_fields(self) -> None:
        now = datetime.now(UTC)
        result = PlanRunResult(
            plan_name="test",
            run_id="abc",
            run_path=Path("/tmp/runs/abc"),
            started_at=now,
            finished_at=now,
            success=True,
            sequential_duration_sec=10.0,
            parallelism_savings_pct=33.3,
            total_cost_usd=5.25,
        )
        d = result.to_dict()
        assert d["sequential_duration_sec"] == 10.0
        assert d["parallelism_savings_pct"] == 33.3
        assert d["total_cost_usd"] == 5.25

    def test_to_dict_includes_optional_plan_score_fields(self) -> None:
        now = datetime.now(UTC)
        result = PlanRunResult(
            plan_name="test",
            run_id="abc",
            run_path=Path("/tmp/runs/abc"),
            started_at=now,
            finished_at=now,
            success=True,
            plan_hash="hash123",
            quality_score=0.75,
        )
        d = result.to_dict()
        assert d["plan_hash"] == "hash123"
        assert d["quality_score"] == pytest.approx(0.75)


class TestScoreRecord:
    def test_to_dict(self) -> None:
        record = ScoreRecord(
            plan_name="demo",
            plan_hash="hash123",
            run_id="run-1",
            success=True,
            cost_usd=1.5,
            quality_score=0.9,
            duration_sec=12.0,
            timestamp="2026-04-01T12:00:00+00:00",
            valid_from="2026-04-01T12:00:00+00:00",
            recorded_at="2026-04-01T12:00:00+00:00",
            source_id="run-1:score",
            metadata={"quality_source": "judge"},
        )
        d = record.to_dict()
        assert d["plan_hash"] == "hash123"
        assert d["quality_score"] == pytest.approx(0.9)
        assert d["metadata"]["quality_source"] == "judge"


class TestHistoricalPruningDecision:
    def test_to_dict(self) -> None:
        decision = HistoricalPruningDecision(
            plan_hash="hash123",
            sample_size=5,
            failures=4,
            failure_rate=0.8,
            threshold=0.8,
            min_runs=5,
            prune=True,
            horizon_days=30,
            recent_runs=20,
        )
        d = decision.to_dict()
        assert d["plan_hash"] == "hash123"
        assert d["failure_rate"] == pytest.approx(0.8)
        assert d["prune"] is True


class TestEditPolicyDefaults:
    """Tests for edit_policy and append_system_prompt field defaults."""

    def test_engine_defaults_append_system_prompt_none(self) -> None:
        ed = EngineDefaults()
        assert ed.append_system_prompt is None

    def test_engine_defaults_append_system_prompt_set(self) -> None:
        ed = EngineDefaults(append_system_prompt="Use Portuguese")
        assert ed.append_system_prompt == "Use Portuguese"

    def test_plan_defaults_edit_policy_default(self) -> None:
        pd = PlanDefaults()
        assert pd.edit_policy == "default"

    def test_plan_defaults_edit_policy_set(self) -> None:
        pd = PlanDefaults(edit_policy="strict")
        assert pd.edit_policy == "strict"

    def test_task_spec_edit_policy_none(self) -> None:
        task = TaskSpec(id="t1")
        assert task.edit_policy is None
        assert task.append_system_prompt is None

    def test_task_spec_edit_policy_set(self) -> None:
        task = TaskSpec(id="t1", edit_policy="efficient", append_system_prompt="Be precise")
        assert task.edit_policy == "efficient"
        assert task.append_system_prompt == "Be precise"


class TestTokenUsage:
    def test_total_tokens_sums_three_fields(self) -> None:
        tu = TokenUsage(input_tokens=100, cached_tokens=50, output_tokens=30, cache_creation_tokens=10)
        assert tu.total_tokens == 180  # input + cached + output (not cache_creation)

    def test_total_tokens_zeros(self) -> None:
        tu = TokenUsage()
        assert tu.total_tokens == 0

    def test_to_dict_includes_total_tokens(self) -> None:
        tu = TokenUsage(input_tokens=10, cached_tokens=5, output_tokens=3, cache_creation_tokens=2)
        d = tu.to_dict()
        assert d["input_tokens"] == 10
        assert d["cached_tokens"] == 5
        assert d["output_tokens"] == 3
        assert d["cache_creation_tokens"] == 2
        assert d["total_tokens"] == 18


class TestFailureRecord:
    def test_to_dict_all_fields(self) -> None:
        rec = FailureRecord(attempt=2, category="timeout", exit_code=124, message="timed out")
        d = rec.to_dict()
        assert d == {"attempt": 2, "category": "timeout", "exit_code": 124, "message": "timed out"}

    def test_to_dict_none_exit_code(self) -> None:
        rec = FailureRecord(attempt=1, category="unknown", exit_code=None, message="error")
        assert rec.to_dict()["exit_code"] is None


class TestJudgeSpec:
    def test_defaults(self) -> None:
        js = JudgeSpec()
        assert js.pass_threshold == 0.7
        assert js.on_fail == "fail"
        assert js.model == "haiku"
        assert js.method == "direct"
        assert js.aggregation == "mean"
        assert js.preset is None
        assert js.criteria == []

    def test_to_dict_has_all_keys(self) -> None:
        js = JudgeSpec(criteria=["Output is correct"], pass_threshold=0.8, on_fail="retry", preset="code_quality")
        d = js.to_dict()
        assert d["criteria"] == ["Output is correct"]
        assert d["pass_threshold"] == 0.8
        assert d["on_fail"] == "retry"
        assert d["preset"] == "code_quality"
        assert "model" in d
        assert "method" in d
        assert "aggregation" in d

    def test_mutable_default_independence(self) -> None:
        js1 = JudgeSpec()
        js2 = JudgeSpec()
        js1.criteria.append("x")
        assert js2.criteria == []


class TestCriterionScore:
    def test_to_dict(self) -> None:
        cs = CriterionScore(criterion="Correctness", passed=True, score=0.9, reasoning="Looks good")
        d = cs.to_dict()
        assert d == {"criterion": "Correctness", "passed": True, "score": 0.9, "reasoning": "Looks good"}


class TestJudgeResult:
    def test_to_dict_no_criterion_scores(self) -> None:
        jr = JudgeResult(verdict="pass", overall_score=0.85)
        d = jr.to_dict()
        assert d["verdict"] == "pass"
        assert d["overall_score"] == 0.85
        assert d["criterion_scores"] == []
        assert d["previous_score"] is None

    def test_to_dict_with_criterion_scores_and_previous(self) -> None:
        cs = CriterionScore(criterion="Style", passed=True, score=0.8, reasoning="Fine")
        jr = JudgeResult(verdict="fail", overall_score=0.4, criterion_scores=[cs], previous_score=0.5)
        d = jr.to_dict()
        assert len(d["criterion_scores"]) == 1
        assert d["criterion_scores"][0]["criterion"] == "Style"
        assert d["previous_score"] == 0.5


class TestHandoffReport:
    def test_defaults(self) -> None:
        hr = HandoffReport()
        assert hr.failure_category == "runtime_error"
        assert hr.partial_output == ""
        assert hr.summary == ""

    def test_to_dict(self) -> None:
        hr = HandoffReport(failure_category="timeout", partial_output="some output", summary="Task timed out")
        d = hr.to_dict()
        assert d == {"failure_category": "timeout", "partial_output": "some output", "summary": "Task timed out"}


class TestWorkspaceExtraction:
    def test_to_dict(self) -> None:
        we = WorkspaceExtraction(
            relevant_files=["a.py", "b.py"],
            snippets={"a.py": "def foo(): ..."},
            reasoning="These files are relevant",
            token_estimate=500,
        )
        d = we.to_dict()
        assert d["relevant_files"] == ["a.py", "b.py"]
        assert d["snippets"] == {"a.py": "def foo(): ..."}
        assert d["token_estimate"] == 500

    def test_mutable_default_independence(self) -> None:
        w1 = WorkspaceExtraction()
        w2 = WorkspaceExtraction()
        w1.relevant_files.append("x.py")
        assert w2.relevant_files == []


class TestWorkspaceBrief:
    def test_to_dict(self) -> None:
        wb = WorkspaceBrief(brief_text="Context doc", token_estimate=200, files_referenced=["src/foo.py"])
        d = wb.to_dict()
        assert d["brief_text"] == "Context doc"
        assert d["token_estimate"] == 200
        assert d["files_referenced"] == ["src/foo.py"]


class TestStructuredContext:
    def test_to_dict_all_fields(self) -> None:
        sc = StructuredContext(
            task_id="t1",
            status="success",
            exit_code=0,
            duration_sec=1.5,
            files_changed=["foo.py"],
            errors=["err1"],
            warnings=["warn1"],
            decisions=["chose X"],
            cost_usd=0.01,
            result_text="done",
            summary="All good",
        )
        d = sc.to_dict()
        assert d["task_id"] == "t1"
        assert d["files_changed"] == ["foo.py"]
        assert d["cost_usd"] == 0.01
        assert d["summary"] == "All good"


class TestRubricLevelAndCriterion:
    def test_rubric_level_to_dict(self) -> None:
        rl = RubricLevel(score=3, description="Adequate")
        assert rl.to_dict() == {"score": 3, "description": "Adequate"}

    def test_rubric_criterion_to_dict(self) -> None:
        levels = [RubricLevel(score=1, description="Bad"), RubricLevel(score=5, description="Great")]
        rc = RubricCriterion(name="Quality", levels=levels, min_score=3, weight=2.0)
        d = rc.to_dict()
        assert d["name"] == "Quality"
        assert len(d["levels"]) == 2
        assert d["levels"][0] == {"score": 1, "description": "Bad"}
        assert d["min_score"] == 3
        assert d["weight"] == 2.0


class TestWatchSpec:
    def test_defaults(self) -> None:
        ws = WatchSpec(metric="test_pass_rate")
        assert ws.max_iterations == 100
        assert ws.metric_direction == "lower_is_better"
        assert ws.metric_source == "stdout_regex"
        assert ws.on_regression == "rollback"
        assert ws.warmup_iterations == 1
        assert ws.plateau_threshold == 5
        assert ws.plateau_action == "stop"
        assert ws.max_cost_usd is None

    def test_to_dict_required_and_optional(self) -> None:
        ws = WatchSpec(metric="score", metric_pattern=r"Score: (\d+\.?\d*)", metric_task="run-tests")
        d = ws.to_dict()
        assert d["metric"] == "score"
        assert d["metric_pattern"] == r"Score: (\d+\.?\d*)"
        assert d["metric_task"] == "run-tests"
        assert d["max_cost_usd"] is None

    def test_to_dict_with_max_cost(self) -> None:
        ws = WatchSpec(metric="latency", max_cost_usd=5.0)
        assert ws.to_dict()["max_cost_usd"] == 5.0


class TestWatchIteration:
    def test_to_dict_with_optional_fields(self) -> None:
        wi = WatchIteration(
            iteration=3,
            metric_value=42.5,
            best_metric=40.0,
            improved=False,
            action="keep",
            cost_usd=0.05,
            duration_sec=10.2,
            git_commit="abc123",
            error=None,
            timestamp="2026-01-01T00:00:00",
            fix_summary="increase timeout",
            manifest_excerpt="task-a: success",
            blame_excerpt="root cause",
            consolidated_excerpt="try path fix next",
        )
        d = wi.to_dict()
        assert d["iteration"] == 3
        assert d["metric_value"] == 42.5
        assert d["git_commit"] == "abc123"
        assert d["error"] is None
        assert d["fix_summary"] == "increase timeout"
        assert d["manifest_excerpt"] == "task-a: success"
        assert d["blame_excerpt"] == "root cause"
        assert d["consolidated_excerpt"] == "try path fix next"

    def test_to_dict_defaults(self) -> None:
        wi = WatchIteration(iteration=1)
        d = wi.to_dict()
        assert d["improved"] is False
        assert d["metric_value"] is None
        assert d["cost_usd"] is None
        assert d["fix_summary"] is None
        assert d["manifest_excerpt"] is None
        assert d["blame_excerpt"] is None
        assert d["consolidated_excerpt"] is None


class TestWatchState:
    def test_defaults(self) -> None:
        ws = WatchState()
        assert ws.status == "max_iterations"
        assert ws.plateau_count == 0
        assert ws.best_metric is None
        assert ws.total_cost_usd == 0.0

    def test_to_dict_empty_iterations(self) -> None:
        ws = WatchState(plan_path="plan.yaml", total_iterations=5)
        d = ws.to_dict()
        assert d["plan_path"] == "plan.yaml"
        assert d["iterations"] == []
        assert d["total_iterations"] == 5

    def test_to_dict_with_nested_iterations(self) -> None:
        ws = WatchState(iterations=[WatchIteration(iteration=1, metric_value=10.0)])
        d = ws.to_dict()
        assert len(d["iterations"]) == 1
        assert d["iterations"][0]["metric_value"] == 10.0


class TestSessionSnapshotLite:
    def test_to_dict_defaults(self) -> None:
        snapshot = SessionSnapshot()
        d = snapshot.to_dict()
        assert d["snapshot_kind"] == "watch"
        assert d["iteration_from"] == 0
        assert d["iteration_to"] == 0
        assert d["metadata"] == {}


class TestWorktreeMergeResult:
    def test_to_dict_merged(self) -> None:
        wmr = WorktreeMergeResult(
            status="merged",
            files_changed=["a.py", "b.py"],
            merge_commit="deadbeef",
        )
        d = wmr.to_dict()
        assert d["status"] == "merged"
        assert d["files_changed"] == ["a.py", "b.py"]
        assert d["merge_commit"] == "deadbeef"
        assert d["conflict_files"] == []
        assert d["error"] is None

    @pytest.mark.parametrize("status", ["merged", "conflict", "empty", "error"])
    def test_to_dict_all_statuses(self, status: str) -> None:
        wmr = WorktreeMergeResult(status=status)  # type: ignore[arg-type]
        assert wmr.to_dict()["status"] == status


class TestPlanImport:
    def test_to_dict_without_overrides(self) -> None:
        pi = PlanImport(path="common.yaml", prefix="shared")
        d = pi.to_dict()
        assert d == {"path": "common.yaml", "prefix": "shared"}
        assert "overrides" not in d

    def test_to_dict_with_overrides(self) -> None:
        pi = PlanImport(path="common.yaml", prefix="shared", overrides={"model": "opus"})
        d = pi.to_dict()
        assert d["overrides"] == {"model": "opus"}


class TestSuggestion:
    def test_to_dict(self) -> None:
        s = Suggestion(
            task_id="build",
            category="downgrade_model",
            severity="medium",
            reason="Consistently passing with cheap model",
            current_value="opus",
            suggested_value="haiku",
            confidence=0.9,
            estimated_savings_pct=40.0,
        )
        d = s.to_dict()
        assert d["task_id"] == "build"
        assert d["category"] == "downgrade_model"
        assert d["estimated_savings_pct"] == 40.0


class TestPlanSpec:
    def test_defaults(self) -> None:
        ps = PlanSpec(name="my-plan")
        assert ps.version == 1
        assert ps.firewall_model is None
        assert ps.max_parallel == 1
        assert ps.fail_fast is True
        assert ps.run_dir == ".maestro-runs"
        assert ps.tasks == []
        assert ps.imports == []
        assert ps.watch is None

    def test_source_dir_with_source_path(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "subdir" / "plan.yaml"
        ps = PlanSpec(name="p", source_path=plan_file)
        assert ps.source_dir == tmp_path / "subdir"

    def test_source_dir_without_source_path(self) -> None:
        ps = PlanSpec(name="p")
        assert ps.source_dir == Path.cwd()

    def test_to_dict_without_watch(self) -> None:
        ps = PlanSpec(name="p", max_parallel=2)
        d = ps.to_dict()
        assert d["name"] == "p"
        assert d["max_parallel"] == 2
        assert "watch" not in d

    def test_to_dict_with_watch(self) -> None:
        ps = PlanSpec(name="p", watch=WatchSpec(metric="score"))
        d = ps.to_dict()
        assert "watch" in d
        assert d["watch"]["metric"] == "score"


class TestReplanAttempt:
    def test_defaults(self) -> None:
        ra = ReplanAttempt(attempt_number=1)
        assert ra.plan_yaml == ""
        assert ra.corrected_plan_yaml is None
        assert ra.approved is False
        assert ra.run_result is None
        assert ra.failed_task_ids == []
        assert ra.candidate_variants == []
        assert ra.selected_candidate_id is None

    def test_to_dict_without_run_result(self) -> None:
        ra = ReplanAttempt(attempt_number=2, error_summary="failed", approved=True)
        d = ra.to_dict()
        assert d["attempt_number"] == 2
        assert d["error_summary"] == "failed"
        assert d["approved"] is True
        assert d["run_result"] is None

    def test_to_dict_with_run_result(self) -> None:
        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name="p", run_id="r1", run_path=Path("/tmp/r1"),
            started_at=now, finished_at=now, success=True,
        )
        ra = ReplanAttempt(attempt_number=1, run_result=run_result)
        d = ra.to_dict()
        assert d["run_result"] is not None
        assert d["run_result"]["plan_name"] == "p"

    def test_mutable_default_independence(self) -> None:
        ra1 = ReplanAttempt(attempt_number=1)
        ra2 = ReplanAttempt(attempt_number=2)
        ra1.failed_task_ids.append("t1")
        assert ra2.failed_task_ids == []
        ra1.candidate_variants.append({"node_id": "x"})
        assert ra2.candidate_variants == []


class TestReplanState:
    def test_defaults(self) -> None:
        rs = ReplanState()
        assert rs.max_attempts == 3
        assert rs.status == "max_attempts_exceeded"
        assert rs.final_success is False
        assert rs.total_cost_usd == 0.0
        assert rs.analysis_model == "opus"
        assert rs.attempts == []
        assert rs.search_tree_path is None

    def test_to_dict_empty_attempts(self) -> None:
        rs = ReplanState(plan_path="plan.yaml", max_attempts=5, final_success=True)
        d = rs.to_dict()
        assert d["plan_path"] == "plan.yaml"
        assert d["max_attempts"] == 5
        assert d["final_success"] is True
        assert d["attempts"] == []

    def test_to_dict_with_nested_attempts(self) -> None:
        ra = ReplanAttempt(attempt_number=1, error_summary="err")
        rs = ReplanState(attempts=[ra])
        d = rs.to_dict()
        assert len(d["attempts"]) == 1
        assert d["attempts"][0]["attempt_number"] == 1
        assert d["attempts"][0]["error_summary"] == "err"


class TestMultiPlanResult:
    def test_defaults(self) -> None:
        mpr = MultiPlanResult()
        assert mpr.plan_results == []
        assert mpr.total_cost_usd is None
        assert mpr.total_tokens is None
        assert mpr.budget_exceeded is False
        assert mpr.success is True

    def test_to_dict_empty(self) -> None:
        mpr = MultiPlanResult(total_cost_usd=1.5, total_tokens=1000, success=False)
        d = mpr.to_dict()
        assert d["plan_results"] == []
        assert d["total_cost_usd"] == 1.5
        assert d["total_tokens"] == 1000
        assert d["success"] is False
        assert "started_at" in d
        assert "finished_at" in d

    def test_to_dict_with_nested_plan_results(self) -> None:
        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name="sub", run_id="r1", run_path=Path("/tmp/r1"),
            started_at=now, finished_at=now, success=True,
        )
        mpr = MultiPlanResult(plan_results=[run_result], success=True)
        d = mpr.to_dict()
        assert len(d["plan_results"]) == 1
        assert d["plan_results"][0]["plan_name"] == "sub"


class TestPlanSuggestions:
    def test_defaults(self) -> None:
        ps = PlanSuggestions(plan_name="my-plan", runs_analyzed=3)
        assert ps.suggestions == []
        assert ps.total_estimated_savings_pct is None

    def test_to_dict_no_suggestions(self) -> None:
        ps = PlanSuggestions(plan_name="p", runs_analyzed=2, total_estimated_savings_pct=25.0)
        d = ps.to_dict()
        assert d["plan_name"] == "p"
        assert d["runs_analyzed"] == 2
        assert d["suggestions"] == []
        assert d["total_estimated_savings_pct"] == 25.0

    def test_to_dict_with_suggestions(self) -> None:
        s = Suggestion(
            task_id="t1", category="downgrade_model", severity="low",
            reason="cheap enough", current_value="opus", suggested_value="haiku",
            confidence=0.8,
        )
        ps = PlanSuggestions(plan_name="p", runs_analyzed=5, suggestions=[s])
        d = ps.to_dict()
        assert len(d["suggestions"]) == 1
        assert d["suggestions"][0]["task_id"] == "t1"


class TestConstants:
    def test_assertion_types_contains_all(self) -> None:
        expected = {"contains", "regex", "is-json", "json-schema", "llm-rubric", "cost_under", "duration_under", "rubric"}
        assert ASSERTION_TYPES == expected

    def test_judge_presets_has_both_keys(self) -> None:
        assert "code_quality" in JUDGE_PRESETS
        assert "security_audit" in JUDGE_PRESETS

    def test_judge_presets_structure(self) -> None:
        cq = JUDGE_PRESETS["code_quality"]
        assert "criteria" in cq
        assert "pass_threshold" in cq
        assert "aggregation" in cq
        assert isinstance(cq["criteria"], list)
        assert len(cq["criteria"]) > 0

    def test_terminal_statuses_subset_of_status_styles(self) -> None:
        for status in TERMINAL_STATUSES:
            assert status in STATUS_STYLES

    def test_codex_model_aliases_coverage(self) -> None:
        assert "5.4" in CODEX_MODEL_ALIASES
        assert CODEX_MODEL_ALIASES["5.4"] == "gpt-5.4-codex"
        assert "5-mini" in CODEX_MODEL_ALIASES

    def test_context_windows_has_claude_models(self) -> None:
        for model in ("haiku", "sonnet", "opus"):
            assert model in CONTEXT_WINDOWS
            assert CONTEXT_WINDOWS[model] == 200_000

    def test_context_windows_has_gemini_models(self) -> None:
        assert "gemini-2.5-flash" in CONTEXT_WINDOWS
        assert CONTEXT_WINDOWS["gemini-2.5-flash"] == 1_000_000

    def test_gemini_model_aliases_coverage(self) -> None:
        assert GEMINI_MODEL_ALIASES["flash"] == "gemini-2.5-flash"
        assert GEMINI_MODEL_ALIASES["pro"] == "gemini-2.5-pro"
        assert GEMINI_MODEL_ALIASES["flash-lite"] == "gemini-2.5-flash-lite"
        assert "flash-3" in GEMINI_MODEL_ALIASES
        assert "pro-3" in GEMINI_MODEL_ALIASES

    def test_copilot_model_aliases_claude_and_gpt(self) -> None:
        assert COPILOT_MODEL_ALIASES["haiku"] == "claude-haiku-4.5"
        assert COPILOT_MODEL_ALIASES["sonnet"] == "claude-sonnet-4.6"
        assert COPILOT_MODEL_ALIASES["opus"] == "claude-opus-4.6"
        assert COPILOT_MODEL_ALIASES["gemini-pro"] == "gemini-2.5-pro"
        assert COPILOT_MODEL_ALIASES["gemini-3-pro"] == "gemini-3-pro-preview"

    def test_qwen_model_aliases_coverage(self) -> None:
        assert QWEN_MODEL_ALIASES["coder"] == "qwen-coder-plus"
        assert QWEN_MODEL_ALIASES["max"] == "qwen-max"
        assert QWEN_MODEL_ALIASES["qwq"] == "qwq-plus"
        assert "coder-turbo" in QWEN_MODEL_ALIASES

    def test_validation_constants(self) -> None:
        assert EDIT_POLICIES == {"default", "efficient", "strict"}
        assert CONTEXT_MODES == {"raw", "summarized", "map_reduce", "recursive", "layered", "selective", "structural", "council", "knowledge_graph"}
        assert MAX_RETRIES_LIMIT == 3


class TestRecursiveContext:
    def test_defaults(self) -> None:
        from maestro_cli.models import RecursiveContext
        rc = RecursiveContext()
        assert rc.stages == []
        assert rc.index is None
        assert rc.extraction is None
        assert rc.brief is None
        assert rc.workspace_brief == ""
        assert rc.duration_sec == 0.0
        assert rc.reused_index is False

    def test_to_dict_all_none_nested(self) -> None:
        from maestro_cli.models import RecursiveContext
        rc = RecursiveContext(stages=["index", "extract"], workspace_brief="ctx", duration_sec=1.5, reused_index=True)
        d = rc.to_dict()
        assert d["stages"] == ["index", "extract"]
        assert d["workspace_brief"] == "ctx"
        assert d["duration_sec"] == 1.5
        assert d["reused_index"] is True
        assert d["index"] is None
        assert d["extraction"] is None
        assert d["brief"] is None

    def test_to_dict_with_nested_extraction_and_brief(self) -> None:
        from maestro_cli.models import RecursiveContext
        extraction = WorkspaceExtraction(relevant_files=["a.py"], token_estimate=100)
        brief = WorkspaceBrief(brief_text="summary", token_estimate=50)
        rc = RecursiveContext(extraction=extraction, brief=brief)
        d = rc.to_dict()
        assert d["extraction"]["relevant_files"] == ["a.py"]
        assert d["brief"]["brief_text"] == "summary"

    def test_mutable_stages_independence(self) -> None:
        from maestro_cli.models import RecursiveContext
        rc1 = RecursiveContext()
        rc2 = RecursiveContext()
        rc1.stages.append("index")
        assert rc2.stages == []


class TestTaskSpecDefaults:
    def test_cache_true_by_default(self) -> None:
        task = TaskSpec(id="t1")
        assert task.cache is True

    def test_context_mode_raw_by_default(self) -> None:
        task = TaskSpec(id="t1")
        assert task.context_mode == "raw"

    def test_max_retries_zero_by_default(self) -> None:
        task = TaskSpec(id="t1")
        assert task.max_retries == 0

    def test_worktree_false_by_default(self) -> None:
        task = TaskSpec(id="t1")
        assert task.worktree is False

    def test_checkpoint_false_by_default(self) -> None:
        task = TaskSpec(id="t1")
        assert task.checkpoint is False

    def test_requires_approval_false_by_default(self) -> None:
        task = TaskSpec(id="t1")
        assert task.requires_approval is False

    @pytest.mark.parametrize("field,value", [
        ("cache", False),
        ("worktree", True),
        ("checkpoint", True),
        ("requires_approval", True),
    ])
    def test_bool_field_override(self, field: str, value: bool) -> None:
        task = TaskSpec(id="t1", **{field: value})
        assert getattr(task, field) == value


class TestTaskSpecToDict:
    def test_to_dict_includes_all_keys(self) -> None:
        task = TaskSpec(id="impl", engine="claude", model="sonnet", prompt="Do X")
        d = task.to_dict()
        assert d["id"] == "impl"
        assert d["engine"] == "claude"
        assert d["model"] == "sonnet"
        assert d["prompt"] == "Do X"
        assert d["cache"] is True
        assert d["worktree"] is False
        assert d["context_mode"] == "raw"
        assert d["judge"] is None

    def test_to_dict_with_judge(self) -> None:
        js = JudgeSpec(criteria=["Correct"], pass_threshold=0.9)
        task = TaskSpec(id="t1", judge=js)
        d = task.to_dict()
        assert d["judge"] is not None
        assert d["judge"]["criteria"] == ["Correct"]
        assert d["judge"]["pass_threshold"] == 0.9

    def test_to_dict_escalation_and_fallback(self) -> None:
        task = TaskSpec(id="t1", escalation=["haiku", "sonnet"], fallback_engine="gemini", fallback_model="flash")
        d = task.to_dict()
        assert d["escalation"] == ["haiku", "sonnet"]
        assert d["fallback_engine"] == "gemini"
        assert d["fallback_model"] == "flash"


class TestTaskResultAutoRoutedModel:
    def test_auto_routed_model_absent_by_default(self) -> None:
        from datetime import datetime, UTC
        now = datetime.now(UTC)
        result = TaskResult(
            task_id="t1", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=1.0,
            command="echo hi", log_path=Path("/tmp/t.log"), result_path=Path("/tmp/t.json"),
        )
        assert result.auto_routed_model is None
        assert "auto_routed_model" not in result.to_dict()

    def test_auto_routed_model_included_when_set(self) -> None:
        from datetime import datetime, UTC
        now = datetime.now(UTC)
        result = TaskResult(
            task_id="t1", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=1.0,
            command="echo hi", log_path=Path("/tmp/t.log"), result_path=Path("/tmp/t.json"),
            auto_routed_model="sonnet",
        )
        d = result.to_dict()
        assert "auto_routed_model" in d
        assert d["auto_routed_model"] == "sonnet"


class TestEngineDefaultsResilienceFields:
    def test_defaults_are_empty(self) -> None:
        ed = EngineDefaults()
        assert ed.escalation == []
        assert ed.fallback_engine is None
        assert ed.fallback_model is None

    def test_escalation_set(self) -> None:
        ed = EngineDefaults(escalation=["haiku", "sonnet", "opus"])
        assert ed.escalation == ["haiku", "sonnet", "opus"]

    def test_fallback_engine_and_model(self) -> None:
        ed = EngineDefaults(fallback_engine="gemini", fallback_model="flash")
        assert ed.fallback_engine == "gemini"
        assert ed.fallback_model == "flash"

    def test_mutable_escalation_independence(self) -> None:
        ed1 = EngineDefaults()
        ed2 = EngineDefaults()
        ed1.escalation.append("opus")
        assert ed2.escalation == []


class TestOllamaAndJudgeConstants:
    def test_ollama_model_aliases_coverage(self) -> None:
        assert OLLAMA_MODEL_ALIASES["llama3"] == "llama3"
        assert OLLAMA_MODEL_ALIASES["codellama"] == "codellama"
        assert OLLAMA_MODEL_ALIASES["deepseek-coder"] == "deepseek-coder"
        assert "mistral" in OLLAMA_MODEL_ALIASES
        assert "mixtral" in OLLAMA_MODEL_ALIASES

    def test_ollama_extended_aliases(self) -> None:
        assert OLLAMA_MODEL_ALIASES["llama3.1"] == "llama3.1"
        assert OLLAMA_MODEL_ALIASES["llama3.2"] == "llama3.2"
        assert OLLAMA_MODEL_ALIASES["qwen2.5-coder"] == "qwen2.5-coder"
        assert OLLAMA_MODEL_ALIASES["deepseek-coder-v2"] == "deepseek-coder-v2"
        assert OLLAMA_MODEL_ALIASES["starcoder2"] == "starcoder2"

    def test_judge_set_constants(self) -> None:
        assert JUDGE_ON_FAIL_VALUES == {"fail", "warn", "retry"}
        assert JUDGE_METHODS == {"direct", "g_eval", "debate", "reflection"}
        assert SCORE_AGGREGATIONS == {"mean", "min", "weighted_mean"}
        assert RECURSIVE_CONTEXT_STAGES == {"index", "extract", "brief"}


class TestClaudeModelAndEffortConstants:
    def test_claude_models_set(self) -> None:
        from maestro_cli.models import CLAUDE_MODELS
        assert CLAUDE_MODELS == {"haiku", "sonnet", "opus", "opusplan"}

    def test_claude_reasoning_efforts(self) -> None:
        from maestro_cli.models import CLAUDE_REASONING_EFFORTS
        # 2026-04-27: Opus 4.7 added `xhigh` (recommended for coding/agentic),
        # `max` was already supported on Opus 4.5/4.6/4.7 and Sonnet 4.6.
        assert CLAUDE_REASONING_EFFORTS == {"low", "medium", "high", "xhigh", "max"}

    def test_codex_reasoning_efforts(self) -> None:
        from maestro_cli.models import CODEX_REASONING_EFFORTS
        # GPT-5.5 supports `none` (no reasoning) alongside the existing levels.
        assert CODEX_REASONING_EFFORTS == {
            "none", "minimal", "low", "medium", "high", "xhigh",
        }

    def test_copilot_models_is_set_of_aliases(self) -> None:
        from maestro_cli.models import COPILOT_MODELS
        # COPILOT_MODELS is set(COPILOT_MODEL_ALIASES) — all alias keys
        assert isinstance(COPILOT_MODELS, set)
        assert "sonnet" in COPILOT_MODELS
        assert "opus" in COPILOT_MODELS
        assert "haiku" in COPILOT_MODELS
        assert "gemini-3-pro" in COPILOT_MODELS


class TestPlanSpecToDictWithImports:
    def test_to_dict_with_imports(self) -> None:
        pi = PlanImport(path="common.yaml", prefix="shared")
        ps = PlanSpec(name="p", imports=[pi])
        d = ps.to_dict()
        assert "imports" in d
        assert len(d["imports"]) == 1
        assert d["imports"][0]["path"] == "common.yaml"
        assert d["imports"][0]["prefix"] == "shared"

    def test_to_dict_empty_imports(self) -> None:
        ps = PlanSpec(name="p")
        d = ps.to_dict()
        assert d["imports"] == []


class TestWatchSpecJsonFieldSource:
    def test_json_field_source_and_path(self) -> None:
        ws = WatchSpec(
            metric="coverage",
            metric_source="json_field",
            metric_json_path="summary.line_rate",
            metric_task="run-coverage",
        )
        assert ws.metric_source == "json_field"
        assert ws.metric_json_path == "summary.line_rate"
        d = ws.to_dict()
        assert d["metric_source"] == "json_field"
        assert d["metric_json_path"] == "summary.line_rate"

    @pytest.mark.parametrize("source", ["stdout_regex", "verify_command", "guard_command", "json_field"])
    def test_all_metric_sources(self, source: str) -> None:
        ws = WatchSpec(metric="m", metric_source=source)  # type: ignore[arg-type]
        assert ws.metric_source == source


class TestPlanDefaultsMutableIndependence:
    def test_env_dict_independence(self) -> None:
        pd1 = PlanDefaults()
        pd2 = PlanDefaults()
        pd1.env["KEY"] = "value"
        assert pd2.env == {}

    def test_workspace_index_exclude_independence(self) -> None:
        pd1 = PlanDefaults()
        pd2 = PlanDefaults()
        pd1.workspace_index_exclude.append("*.log")
        assert pd2.workspace_index_exclude == []


class TestTaskResultConditionalToDict:
    def test_to_dict_with_worktree_merge(self) -> None:
        now = datetime.now(UTC)
        wmr = WorktreeMergeResult(status="merged", files_changed=["x.py"], merge_commit="abc")
        result = TaskResult(
            task_id="t1", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=1.0,
            command="claude", log_path=Path("/tmp/t.log"), result_path=Path("/tmp/t.json"),
            worktree_merge=wmr,
        )
        d = result.to_dict()
        assert "worktree_merge" in d
        assert d["worktree_merge"]["status"] == "merged"
        assert d["worktree_merge"]["files_changed"] == ["x.py"]

    def test_to_dict_with_workspace_brief(self) -> None:
        now = datetime.now(UTC)
        wb = WorkspaceBrief(brief_text="context summary", token_estimate=200)
        result = TaskResult(
            task_id="t2", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=2.0,
            command="claude", log_path=Path("/tmp/t.log"), result_path=Path("/tmp/t.json"),
            workspace_brief=wb,
        )
        d = result.to_dict()
        assert "workspace_brief" in d
        assert d["workspace_brief"]["brief_text"] == "context summary"
        assert d["workspace_brief"]["token_estimate"] == 200


class TestTaskSpecMutableDefaultIndependence:
    """Mutable field defaults on TaskSpec must be independent between instances."""

    @pytest.mark.parametrize("field_name", ["tags", "depends_on", "args", "context_from", "escalation"])
    def test_list_fields_are_independent(self, field_name: str) -> None:
        t1 = TaskSpec(id="t1")
        t2 = TaskSpec(id="t2")
        getattr(t1, field_name).append("x")
        assert getattr(t2, field_name) == []

    def test_env_dict_independence(self) -> None:
        t1 = TaskSpec(id="t1")
        t2 = TaskSpec(id="t2")
        t1.env["KEY"] = "val"
        assert t2.env == {}

    def test_workspace_index_exclude_independence(self) -> None:
        t1 = TaskSpec(id="t1")
        t2 = TaskSpec(id="t2")
        t1.workspace_index_exclude.append("*.log")
        assert t2.workspace_index_exclude == []


class TestPlanSpecToDictGoalAndWebhook:
    """PlanSpec.to_dict() must include goal, webhook_url, secrets*, workspace_root."""

    def test_goal_included(self) -> None:
        ps = PlanSpec(name="p", goal="Make it fast")
        d = ps.to_dict()
        assert d["goal"] == "Make it fast"

    def test_firewall_model_included(self) -> None:
        ps = PlanSpec(name="p", firewall_model="haiku")
        d = ps.to_dict()
        assert d["firewall_model"] == "haiku"

    def test_webhook_url_included(self) -> None:
        ps = PlanSpec(name="p", webhook_url="https://example.com/hook")
        d = ps.to_dict()
        assert d["webhook_url"] == "https://example.com/hook"

    def test_secrets_and_secrets_auto_included(self) -> None:
        ps = PlanSpec(name="p", secrets=["MY_KEY"], secrets_auto=True)
        d = ps.to_dict()
        assert d["secrets"] == ["MY_KEY"]
        assert d["secrets_auto"] is True

    def test_workspace_root_included(self) -> None:
        ps = PlanSpec(name="p", workspace_root="/repo")
        d = ps.to_dict()
        assert d["workspace_root"] == "/repo"

    def test_max_cost_usd_and_budget_warning_pct(self) -> None:
        ps = PlanSpec(name="p", max_cost_usd=5.0, budget_warning_pct=0.8)
        d = ps.to_dict()
        assert d["max_cost_usd"] == 5.0
        assert d["budget_warning_pct"] == 0.8


class TestWatchSpecHigherIsBetterAndBudget:
    """WatchSpec with non-default metric direction and iteration_budget_sec."""

    def test_higher_is_better(self) -> None:
        ws = WatchSpec(metric="accuracy", metric_direction="higher_is_better")
        assert ws.metric_direction == "higher_is_better"
        assert ws.to_dict()["metric_direction"] == "higher_is_better"

    def test_iteration_budget_sec(self) -> None:
        ws = WatchSpec(metric="score", iteration_budget_sec=300.0)
        d = ws.to_dict()
        assert d["iteration_budget_sec"] == 300.0

    def test_iteration_budget_sec_none_by_default(self) -> None:
        ws = WatchSpec(metric="score")
        assert ws.iteration_budget_sec is None


class TestJudgeResultEvalSteps:
    """JudgeResult.eval_steps must round-trip through to_dict()."""

    def test_eval_steps_default_empty(self) -> None:
        jr = JudgeResult(verdict="pass", overall_score=0.9)
        assert jr.eval_steps == []
        assert jr.to_dict()["eval_steps"] == []

    def test_eval_steps_populated(self) -> None:
        jr = JudgeResult(
            verdict="pass",
            overall_score=0.8,
            eval_steps=["Step 1: check output", "Step 2: verify style"],
        )
        d = jr.to_dict()
        assert d["eval_steps"] == ["Step 1: check output", "Step 2: verify style"]

    def test_eval_steps_mutable_independence(self) -> None:
        jr1 = JudgeResult(verdict="pass", overall_score=1.0)
        jr2 = JudgeResult(verdict="fail", overall_score=0.0)
        jr1.eval_steps.append("step")
        assert jr2.eval_steps == []


class TestTokenUsage:
    def test_total_tokens_property(self) -> None:
        tu = TokenUsage(input_tokens=100, cached_tokens=50, output_tokens=200)
        assert tu.total_tokens == 350

    def test_total_tokens_excludes_cache_creation(self) -> None:
        tu = TokenUsage(input_tokens=10, cached_tokens=0, output_tokens=5, cache_creation_tokens=1000)
        assert tu.total_tokens == 15

    def test_to_dict_includes_all_fields(self) -> None:
        tu = TokenUsage(input_tokens=100, cached_tokens=50, output_tokens=200, cache_creation_tokens=10)
        d = tu.to_dict()
        assert d["input_tokens"] == 100
        assert d["cached_tokens"] == 50
        assert d["output_tokens"] == 200
        assert d["cache_creation_tokens"] == 10
        assert d["total_tokens"] == 350

    def test_zero_values(self) -> None:
        tu = TokenUsage()
        assert tu.total_tokens == 0
        d = tu.to_dict()
        assert d["total_tokens"] == 0
        assert d["cache_creation_tokens"] == 0


class TestFailureRecord:
    def test_to_dict_with_exit_code(self) -> None:
        fr = FailureRecord(attempt=1, category="timeout", exit_code=124, message="timed out")
        d = fr.to_dict()
        assert d["attempt"] == 1
        assert d["category"] == "timeout"
        assert d["exit_code"] == 124
        assert d["message"] == "timed out"

    def test_to_dict_with_null_exit_code(self) -> None:
        fr = FailureRecord(attempt=2, category="rate_limited", exit_code=None, message="rate limit")
        d = fr.to_dict()
        assert d["exit_code"] is None
        assert d["category"] == "rate_limited"

    @pytest.mark.parametrize("category", [
        "timeout", "compilation_error", "test_failure",
        "validation_error", "permission_error", "runtime_error",
        "context_exceeded", "rate_limited", "unknown",
    ])
    def test_all_failure_categories(self, category: str) -> None:
        fr = FailureRecord(attempt=1, category=category, exit_code=1, message="err")  # type: ignore[arg-type]
        assert fr.to_dict()["category"] == category

    def test_to_dict_omits_empty_verify_tail_and_zero_duration(self) -> None:
        # PM3.1 — backward compat: serialised payloads from before 2026-04-27
        # don't carry verify_tail / duration_sec, so to_dict must omit them
        # when defaulted (otherwise older parsers see unknown keys).
        fr = FailureRecord(attempt=1, category="timeout", exit_code=124, message="x")
        d = fr.to_dict()
        assert "verify_tail" not in d
        assert "duration_sec" not in d

    def test_to_dict_includes_verify_tail_and_duration_when_set(self) -> None:
        fr = FailureRecord(
            attempt=1,
            category="test_failure",
            exit_code=1,
            message="pytest exited 1",
            verify_tail="FAILED tests/test_x.py::test_y\nAssertionError",
            duration_sec=140.5,
        )
        d = fr.to_dict()
        assert d["verify_tail"].startswith("FAILED")
        assert d["duration_sec"] == 140.5


class TestCriterionScore:
    def test_to_dict(self) -> None:
        cs = CriterionScore(criterion="Correctness", passed=True, score=0.9, reasoning="Looks good")
        d = cs.to_dict()
        assert d["criterion"] == "Correctness"
        assert d["passed"] is True
        assert d["score"] == 0.9
        assert d["reasoning"] == "Looks good"

    def test_to_dict_failed(self) -> None:
        cs = CriterionScore(criterion="Style", passed=False, score=0.3, reasoning="Too verbose")
        d = cs.to_dict()
        assert d["passed"] is False
        assert d["score"] == 0.3


class TestHandoffReport:
    def test_defaults(self) -> None:
        hr = HandoffReport()
        assert hr.failure_category == "runtime_error"
        assert hr.partial_output == ""
        assert hr.summary == ""

    def test_to_dict(self) -> None:
        hr = HandoffReport(failure_category="timeout", partial_output="partial...", summary="timed out after 60s")
        d = hr.to_dict()
        assert d["failure_category"] == "timeout"
        assert d["partial_output"] == "partial..."
        assert d["summary"] == "timed out after 60s"

    def test_to_dict_all_fields_present(self) -> None:
        hr = HandoffReport()
        d = hr.to_dict()
        assert set(d.keys()) == {"failure_category", "partial_output", "summary"}


class TestJudgeSpecDefaultsAndToDict:
    def test_defaults(self) -> None:
        js = JudgeSpec()
        assert js.pass_threshold == 0.7
        assert js.on_fail == "fail"
        assert js.model == "haiku"
        assert js.method == "direct"
        assert js.aggregation == "mean"
        assert js.preset is None
        assert js.criteria == []

    def test_to_dict_defaults(self) -> None:
        js = JudgeSpec()
        d = js.to_dict()
        assert d["pass_threshold"] == 0.7
        assert d["on_fail"] == "fail"
        assert d["model"] == "haiku"
        assert d["method"] == "direct"
        assert d["aggregation"] == "mean"
        assert d["preset"] is None

    def test_to_dict_with_preset(self) -> None:
        js = JudgeSpec(preset="code_quality", on_fail="retry", method="g_eval")
        d = js.to_dict()
        assert d["preset"] == "code_quality"
        assert d["on_fail"] == "retry"
        assert d["method"] == "g_eval"

    def test_criteria_independence(self) -> None:
        js1 = JudgeSpec()
        js2 = JudgeSpec()
        js1.criteria.append("Quality")
        assert js2.criteria == []


class TestJudgeResultPreviousScore:
    def test_previous_score_none_by_default(self) -> None:
        jr = JudgeResult(verdict="pass", overall_score=0.8)
        assert jr.previous_score is None
        assert jr.to_dict()["previous_score"] is None

    def test_previous_score_set(self) -> None:
        jr = JudgeResult(verdict="pass", overall_score=0.9, previous_score=0.7)
        d = jr.to_dict()
        assert d["previous_score"] == 0.7

    def test_to_dict_with_criterion_scores(self) -> None:
        scores = [CriterionScore(criterion="C1", passed=True, score=1.0, reasoning="ok")]
        jr = JudgeResult(verdict="pass", overall_score=1.0, criterion_scores=scores)
        d = jr.to_dict()
        assert len(d["criterion_scores"]) == 1
        assert d["criterion_scores"][0]["criterion"] == "C1"


class TestTaskResultTokenUsageAndCompression:
    """TaskResult fields: token_usage, context compression metrics."""

    def _make_result(self, **kwargs: object) -> TaskResult:
        from datetime import datetime, UTC
        now = datetime.now(UTC)
        return TaskResult(
            task_id="t1", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=1.0,
            command="claude", log_path=Path("/tmp/t.log"), result_path=Path("/tmp/t.json"),
            **kwargs,  # type: ignore[arg-type]
        )

    def test_token_usage_none_by_default(self) -> None:
        r = self._make_result()
        assert r.token_usage is None
        assert r.to_dict()["token_usage"] is None

    def test_token_usage_serialized_when_set(self) -> None:
        tu = TokenUsage(input_tokens=100, cached_tokens=20, output_tokens=50)
        r = self._make_result(token_usage=tu)
        d = r.to_dict()
        assert d["token_usage"] is not None
        assert d["token_usage"]["input_tokens"] == 100
        assert d["token_usage"]["output_tokens"] == 50
        assert d["token_usage"]["total_tokens"] == 170

    def test_context_compression_fields_default_zero(self) -> None:
        r = self._make_result()
        d = r.to_dict()
        assert d["context_raw_tokens"] == 0
        assert d["context_final_tokens"] == 0
        assert d["context_compression_ratio"] == 0.0

    def test_context_compression_fields_set(self) -> None:
        r = self._make_result(context_raw_tokens=1000, context_final_tokens=400, context_compression_ratio=0.6)
        d = r.to_dict()
        assert d["context_raw_tokens"] == 1000
        assert d["context_final_tokens"] == 400
        assert d["context_compression_ratio"] == 0.6


class TestPlanRunResultTotalTokensAndBudget:
    """PlanRunResult fields: total_tokens, budget_exceeded, execution_profile."""

    def _make_result(self, **kwargs: object) -> PlanRunResult:
        from datetime import datetime, UTC
        now = datetime.now(UTC)
        return PlanRunResult(
            plan_name="p", run_id="r1", run_path=Path("/tmp/r1"),
            started_at=now, finished_at=now, success=True,
            **kwargs,  # type: ignore[arg-type]
        )

    def test_total_tokens_none_by_default(self) -> None:
        r = self._make_result()
        assert r.total_tokens is None
        assert r.to_dict()["total_tokens"] is None

    def test_total_tokens_set(self) -> None:
        r = self._make_result(total_tokens=5000)
        assert r.to_dict()["total_tokens"] == 5000

    def test_budget_exceeded_false_by_default(self) -> None:
        r = self._make_result()
        assert r.budget_exceeded is False
        assert r.to_dict()["budget_exceeded"] is False

    def test_budget_exceeded_true(self) -> None:
        r = self._make_result(budget_exceeded=True, total_cost_usd=12.5)
        d = r.to_dict()
        assert d["budget_exceeded"] is True
        assert d["total_cost_usd"] == 12.5

    @pytest.mark.parametrize("profile", ["plan", "safe", "yolo"])
    def test_execution_profile_variants(self, profile: str) -> None:
        r = self._make_result(execution_profile=profile)  # type: ignore[arg-type]
        assert r.to_dict()["execution_profile"] == profile


class TestWatchStatusAndProgramMd:
    """WatchSpec.program_md field and WatchStatus Literal values."""

    def test_program_md_none_by_default(self) -> None:
        ws = WatchSpec(metric="score")
        assert ws.program_md is None
        assert ws.to_dict()["program_md"] is None

    def test_program_md_set(self) -> None:
        ws = WatchSpec(metric="score", program_md="path/to/program.md")
        assert ws.program_md == "path/to/program.md"
        assert ws.to_dict()["program_md"] == "path/to/program.md"

    @pytest.mark.parametrize("status", ["improved", "plateau", "budget_exceeded", "max_iterations", "interrupted", "error"])
    def test_watch_state_all_statuses(self, status: str) -> None:
        from maestro_cli.models import WatchState
        ws = WatchState(status=status)  # type: ignore[arg-type]
        assert ws.to_dict()["status"] == status


class TestTaskResultFailureHistoryAndStructuredContext:
    """TaskResult failure_history list and structured_context serialization."""

    def _make_result(self, **kwargs: object) -> TaskResult:
        from datetime import datetime, UTC
        now = datetime.now(UTC)
        return TaskResult(
            task_id="t1", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=1.0,
            command="claude", log_path=Path("/tmp/t.log"), result_path=Path("/tmp/t.json"),
            **kwargs,  # type: ignore[arg-type]
        )

    def test_failure_history_empty_by_default(self) -> None:
        r = self._make_result()
        assert r.failure_history == []
        assert r.to_dict()["failure_history"] == []

    def test_failure_history_serialized(self) -> None:
        fr = FailureRecord(attempt=1, category="timeout", exit_code=124, message="timed out")
        r = self._make_result(failure_history=[fr])
        d = r.to_dict()
        assert len(d["failure_history"]) == 1
        assert d["failure_history"][0]["attempt"] == 1
        assert d["failure_history"][0]["category"] == "timeout"

    def test_structured_context_none_by_default(self) -> None:
        r = self._make_result()
        assert r.structured_context is None
        assert r.to_dict()["structured_context"] is None

    def test_structured_context_serialized_when_set(self) -> None:
        sc = StructuredContext(task_id="upstream", status="success", exit_code=0, duration_sec=2.0)
        r = self._make_result(structured_context=sc)
        d = r.to_dict()
        assert d["structured_context"] is not None
        assert d["structured_context"]["task_id"] == "upstream"
        assert d["structured_context"]["status"] == "success"


class TestWatchIteration:
    """WatchIteration.to_dict() — all fields including optional git_commit/error."""

    def test_to_dict_minimal(self) -> None:
        wi = WatchIteration(iteration=1)
        d = wi.to_dict()
        assert d["iteration"] == 1
        assert d["metric_value"] is None
        assert d["best_metric"] is None
        assert d["improved"] is False
        assert d["action"] == ""
        assert d["cost_usd"] is None
        assert d["duration_sec"] == 0.0
        assert d["git_commit"] is None
        assert d["error"] is None
        assert d["timestamp"] == ""
        assert d["fix_summary"] is None
        assert d["manifest_excerpt"] is None
        assert d["blame_excerpt"] is None
        assert d["consolidated_excerpt"] is None

    def test_to_dict_with_all_fields(self) -> None:
        wi = WatchIteration(
            iteration=3,
            metric_value=0.85,
            best_metric=0.85,
            improved=True,
            action="keep",
            cost_usd=0.05,
            duration_sec=12.3,
            git_commit="abc1234",
            error=None,
            timestamp="2026-03-08T10:00:00",
            fix_summary="increase timeout",
            manifest_excerpt="task-a: success",
            blame_excerpt="{\"root\":\"task-a\"}",
            consolidated_excerpt="avoid rollback loop",
        )
        d = wi.to_dict()
        assert d["iteration"] == 3
        assert d["metric_value"] == 0.85
        assert d["improved"] is True
        assert d["git_commit"] == "abc1234"
        assert d["timestamp"] == "2026-03-08T10:00:00"
        assert d["fix_summary"] == "increase timeout"
        assert d["manifest_excerpt"] == "task-a: success"
        assert d["blame_excerpt"] == "{\"root\":\"task-a\"}"
        assert d["consolidated_excerpt"] == "avoid rollback loop"

    def test_to_dict_with_error(self) -> None:
        wi = WatchIteration(iteration=2, error="subprocess failed", action="rollback")
        d = wi.to_dict()
        assert d["error"] == "subprocess failed"
        assert d["action"] == "rollback"

    def test_mutable_default_independence(self) -> None:
        wi1 = WatchIteration(iteration=1)
        wi2 = WatchIteration(iteration=2)
        # iterations are plain values, verify they are independent instances
        wi1.action = "keep"
        assert wi2.action == ""


class TestWatchState:
    """WatchState.to_dict() — nested iterations list and all fields."""

    def test_defaults(self) -> None:
        ws = WatchState()
        assert ws.plan_path == ""
        assert ws.iterations == []
        assert ws.status == "max_iterations"
        assert ws.best_metric is None
        assert ws.best_iteration is None
        assert ws.total_cost_usd == 0.0
        assert ws.total_iterations == 0
        assert ws.plateau_count == 0

    def test_to_dict_empty(self) -> None:
        ws = WatchState(plan_path="plan.yaml", total_iterations=0)
        d = ws.to_dict()
        assert d["plan_path"] == "plan.yaml"
        assert d["iterations"] == []
        assert d["status"] == "max_iterations"
        assert d["best_metric"] is None

    def test_to_dict_with_iterations(self) -> None:
        wi = WatchIteration(iteration=1, metric_value=0.9, improved=True)
        ws = WatchState(
            plan_path="p.yaml",
            iterations=[wi],
            status="improved",
            best_metric=0.9,
            best_iteration=1,
            total_cost_usd=0.10,
            total_iterations=1,
            plateau_count=0,
        )
        d = ws.to_dict()
        assert len(d["iterations"]) == 1
        assert d["iterations"][0]["iteration"] == 1
        assert d["iterations"][0]["metric_value"] == 0.9
        assert d["best_metric"] == 0.9
        assert d["best_iteration"] == 1
        assert d["total_cost_usd"] == 0.10

    def test_mutable_default_independence(self) -> None:
        ws1 = WatchState()
        ws2 = WatchState()
        ws1.iterations.append(WatchIteration(iteration=1))
        assert ws2.iterations == []


class TestSessionSnapshot:
    def test_to_dict_with_all_fields(self) -> None:
        snapshot = SessionSnapshot(
            id=7,
            plan_name="demo",
            watch_run_path="watch-run-1",
            snapshot_kind="watch",
            iteration_from=2,
            iteration_to=8,
            best_metric=0.91,
            snapshot_text="Best known state.",
            recent_tail_count=3,
            recorded_at="2026-04-09T10:00:00+00:00",
            source_type="watch",
            source_id="watch-run-1",
            trust_label="trusted",
            instructionality_score=0.0,
            metadata={"metric_name": "score"},
        )
        d = snapshot.to_dict()
        assert d["id"] == 7
        assert d["plan_name"] == "demo"
        assert d["iteration_to"] == 8
        assert d["best_metric"] == 0.91
        assert d["metadata"]["metric_name"] == "score"


class TestWorktreeMergeResult:
    """WorktreeMergeResult.to_dict() — status, files_changed, conflict_files."""

    def test_to_dict_merged(self) -> None:
        r = WorktreeMergeResult(
            status="merged",
            files_changed=["src/foo.py"],
            merge_commit="deadbeef",
        )
        d = r.to_dict()
        assert d["status"] == "merged"
        assert d["files_changed"] == ["src/foo.py"]
        assert d["conflict_files"] == []
        assert d["merge_commit"] == "deadbeef"
        assert d["error"] is None

    def test_to_dict_conflict(self) -> None:
        r = WorktreeMergeResult(
            status="conflict",
            files_changed=["a.py", "b.py"],
            conflict_files=["b.py"],
            error="Merge conflict in b.py",
        )
        d = r.to_dict()
        assert d["status"] == "conflict"
        assert d["conflict_files"] == ["b.py"]
        assert d["error"] == "Merge conflict in b.py"

    def test_to_dict_empty(self) -> None:
        r = WorktreeMergeResult(status="empty")
        d = r.to_dict()
        assert d["status"] == "empty"
        assert d["files_changed"] == []
        assert d["merge_commit"] is None

    def test_mutable_default_independence(self) -> None:
        r1 = WorktreeMergeResult(status="merged")
        r2 = WorktreeMergeResult(status="merged")
        r1.files_changed.append("x.py")
        assert r2.files_changed == []


class TestRubricLevelAndCriterion:
    """RubricLevel and RubricCriterion to_dict() serialization."""

    def test_rubric_level_to_dict(self) -> None:
        lvl = RubricLevel(score=4, description="Mostly correct")
        d = lvl.to_dict()
        assert d["score"] == 4
        assert d["description"] == "Mostly correct"

    def test_rubric_criterion_to_dict(self) -> None:
        levels = [
            RubricLevel(score=1, description="Poor"),
            RubricLevel(score=3, description="Acceptable"),
            RubricLevel(score=5, description="Excellent"),
        ]
        rc = RubricCriterion(name="Clarity", levels=levels, min_score=3, weight=1.5)
        d = rc.to_dict()
        assert d["name"] == "Clarity"
        assert len(d["levels"]) == 3
        assert d["levels"][1]["score"] == 3
        assert d["min_score"] == 3
        assert d["weight"] == 1.5

    def test_rubric_criterion_defaults(self) -> None:
        rc = RubricCriterion(name="Quality", levels=[RubricLevel(score=3, description="ok")])
        assert rc.min_score == 3
        assert rc.weight == 1.0

    def test_rubric_criterion_nested_levels_serialized(self) -> None:
        lvl = RubricLevel(score=5, description="Top")
        rc = RubricCriterion(name="Score", levels=[lvl])
        d = rc.to_dict()
        assert d["levels"][0] == {"score": 5, "description": "Top"}

    def test_rubric_criterion_mutable_default_independence(self) -> None:
        rc1 = RubricCriterion(name="A", levels=[])
        rc2 = RubricCriterion(name="B", levels=[])
        rc1.levels.append(RubricLevel(score=1, description="bad"))
        assert rc2.levels == []


class TestMissingConstants:
    """Constants not yet covered: RECURSIVE_CONTEXT_STAGES, JUDGE_METHODS,
    JUDGE_ON_FAIL_VALUES, SCORE_AGGREGATIONS, OLLAMA_MODEL_ALIASES."""

    def test_recursive_context_stages(self) -> None:
        assert RECURSIVE_CONTEXT_STAGES == {"index", "extract", "brief"}

    def test_judge_methods(self) -> None:
        assert JUDGE_METHODS == {"direct", "g_eval", "debate", "reflection"}

    def test_judge_on_fail_values(self) -> None:
        assert JUDGE_ON_FAIL_VALUES == {"fail", "warn", "retry"}

    def test_score_aggregations(self) -> None:
        assert SCORE_AGGREGATIONS == {"mean", "min", "weighted_mean"}

    def test_ollama_model_aliases_coverage(self) -> None:
        assert OLLAMA_MODEL_ALIASES["llama3"] == "llama3"
        assert OLLAMA_MODEL_ALIASES["codellama"] == "codellama"
        assert "deepseek-coder" in OLLAMA_MODEL_ALIASES
        assert "mistral" in OLLAMA_MODEL_ALIASES


class TestPlanDefaultsAllFields:
    """PlanDefaults fields beyond stdout_tail_lines and edit_policy."""

    def test_timeout_sec_none_by_default(self) -> None:
        pd = PlanDefaults()
        assert pd.timeout_sec is None

    def test_context_budget_tokens_none_by_default(self) -> None:
        pd = PlanDefaults()
        assert pd.context_budget_tokens is None

    def test_budget_warning_pct_none_by_default(self) -> None:
        pd = PlanDefaults()
        assert pd.budget_warning_pct is None

    def test_secrets_empty_list_by_default(self) -> None:
        pd = PlanDefaults()
        assert pd.secrets == []
        assert pd.secrets_auto is False

    def test_workspace_index_exclude_empty_by_default(self) -> None:
        pd = PlanDefaults()
        assert pd.workspace_index_exclude == []

    def test_requires_clean_worktree_false_by_default(self) -> None:
        pd = PlanDefaults()
        assert pd.requires_clean_worktree is False

    def test_mutable_defaults_independent(self) -> None:
        pd1 = PlanDefaults()
        pd2 = PlanDefaults()
        pd1.secrets.append("MY_KEY")
        pd1.workspace_index_exclude.append("*.log")
        assert pd2.secrets == []
        assert pd2.workspace_index_exclude == []


class TestTaskSpecMatrixFields:
    """TaskSpec matrix fields appear correctly in to_dict()."""

    def test_matrix_none_by_default(self) -> None:
        task = TaskSpec(id="t1")
        assert task.matrix is None
        assert task.matrix_parent is None
        assert task.matrix_values is None

    def test_to_dict_matrix_fields_present(self) -> None:
        task = TaskSpec(
            id="t1@env=prod",
            matrix={"env": ["dev", "prod"]},
            matrix_parent="t1",
            matrix_values={"env": "prod"},
        )
        d = task.to_dict()
        assert d["matrix"] == {"env": ["dev", "prod"]}
        assert d["matrix_parent"] == "t1"
        assert d["matrix_values"] == {"env": "prod"}

    def test_to_dict_matrix_none_when_not_set(self) -> None:
        task = TaskSpec(id="plain")
        d = task.to_dict()
        assert d["matrix"] is None
        assert d["matrix_parent"] is None
        assert d["matrix_values"] is None


class TestSuggestionNullSavings:
    """Suggestion.to_dict() with estimated_savings_pct=None."""

    def test_to_dict_null_savings(self) -> None:
        s = Suggestion(
            task_id="lint",
            category="downgrade_model",
            severity="info",
            reason="No failures observed",
            current_value="sonnet",
            suggested_value="haiku",
            confidence=0.5,
            estimated_savings_pct=None,
        )
        d = s.to_dict()
        assert d["estimated_savings_pct"] is None
        assert d["confidence"] == 0.5
        assert d["severity"] == "info"


class TestPlanRunResultTaskResults:
    """PlanRunResult.to_dict() serializes task_results as nested dicts."""

    def _make_run_result(self, **kwargs: object) -> PlanRunResult:
        from datetime import datetime, UTC
        now = datetime.now(UTC)
        return PlanRunResult(
            plan_name="p", run_id="r1", run_path=Path("/tmp/r1"),
            started_at=now, finished_at=now, success=True,
            **kwargs,  # type: ignore[arg-type]
        )

    def _make_task_result(self, task_id: str) -> TaskResult:
        from datetime import datetime, UTC
        now = datetime.now(UTC)
        return TaskResult(
            task_id=task_id, status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=1.0,
            command="echo hi", log_path=Path(f"/tmp/{task_id}.log"),
            result_path=Path(f"/tmp/{task_id}.json"),
        )

    def test_empty_task_results(self) -> None:
        r = self._make_run_result()
        assert r.task_results == {}
        assert r.to_dict()["task_results"] == {}

    def test_task_results_serialized_as_nested_dicts(self) -> None:
        tr = self._make_task_result("build")
        r = self._make_run_result(task_results={"build": tr})
        d = r.to_dict()
        assert "build" in d["task_results"]
        assert d["task_results"]["build"]["task_id"] == "build"
        assert d["task_results"]["build"]["status"] == "success"

    def test_multiple_task_results(self) -> None:
        tr_a = self._make_task_result("task-a")
        tr_b = self._make_task_result("task-b")
        r = self._make_run_result(task_results={"task-a": tr_a, "task-b": tr_b})
        d = r.to_dict()
        assert set(d["task_results"].keys()) == {"task-a", "task-b"}

    def test_task_graph_serialized_when_present(self) -> None:
        r = self._make_run_result(task_graph={
            "task-a": {
                "id": "task-a",
                "description": "Analyse repo",
                "depends_on": [],
                "agent": "architect",
            },
        })
        d = r.to_dict()
        assert d["task_graph"]["task-a"]["agent"] == "architect"

    def test_task_graph_omitted_when_empty(self) -> None:
        r = self._make_run_result()
        assert "task_graph" not in r.to_dict()


class TestWatchSpecRegressionAndPlateauActions:
    """WatchSpec non-default on_regression and plateau_action values."""

    @pytest.mark.parametrize("on_regression", ["rollback", "revert", "keep"])
    def test_on_regression_variants(self, on_regression: str) -> None:
        ws = WatchSpec(metric="score", on_regression=on_regression)  # type: ignore[arg-type]
        assert ws.on_regression == on_regression
        assert ws.to_dict()["on_regression"] == on_regression

    @pytest.mark.parametrize("plateau_action", ["stop", "escalate_model", "notify"])
    def test_plateau_action_variants(self, plateau_action: str) -> None:
        ws = WatchSpec(metric="score", plateau_action=plateau_action)  # type: ignore[arg-type]
        assert ws.plateau_action == plateau_action
        assert ws.to_dict()["plateau_action"] == plateau_action

    def test_warmup_and_plateau_threshold_in_to_dict(self) -> None:
        ws = WatchSpec(metric="m", warmup_iterations=3, plateau_threshold=10)
        d = ws.to_dict()
        assert d["warmup_iterations"] == 3
        assert d["plateau_threshold"] == 10


class TestTaskSpecWhenAndApprovalAndGuard:
    """TaskSpec fields: when, approval_message, guard_command in to_dict()."""

    def test_when_included_in_to_dict(self) -> None:
        task = TaskSpec(id="t1", when="{{ upstream.status }} == 'success'")
        d = task.to_dict()
        assert d["when"] == "{{ upstream.status }} == 'success'"

    def test_approval_message_in_to_dict(self) -> None:
        task = TaskSpec(id="t1", requires_approval=True, approval_message="Review before deploy")
        d = task.to_dict()
        assert d["requires_approval"] is True
        assert d["approval_message"] == "Review before deploy"

    def test_guard_command_string_in_to_dict(self) -> None:
        task = TaskSpec(id="t1", guard_command="python validate.py")
        d = task.to_dict()
        assert d["guard_command"] == "python validate.py"

    def test_guard_command_list_in_to_dict(self) -> None:
        task = TaskSpec(id="t1", guard_command=["python", "validate.py", "--strict"])
        d = task.to_dict()
        assert d["guard_command"] == ["python", "validate.py", "--strict"]


class TestGeminiModelsConstant:
    """GEMINI_MODELS set — validate membership of all documented aliases."""

    def test_gemini_models_is_set_with_expected_members(self) -> None:
        from maestro_cli.models import GEMINI_MODELS
        assert isinstance(GEMINI_MODELS, set)
        for alias in ("flash", "flash-lite", "pro", "flash-3", "pro-3", "pro-3.1", "auto"):
            assert alias in GEMINI_MODELS

    def test_gemini_models_matches_aliases_keys(self) -> None:
        from maestro_cli.models import GEMINI_MODELS
        # Every alias in GEMINI_MODEL_ALIASES must be in GEMINI_MODELS
        for alias in GEMINI_MODEL_ALIASES:
            assert alias in GEMINI_MODELS


# ---------------------------------------------------------------------------
# EdgeL3 — 110+ additional edge-case and coverage tests
# ---------------------------------------------------------------------------

class TestEdgeL3TokenUsageArithmetic:
    """Edge cases for TokenUsage arithmetic and to_dict round-trip."""

    def test_large_values(self) -> None:
        tu = TokenUsage(input_tokens=10_000_000, cached_tokens=5_000_000, output_tokens=2_000_000)
        assert tu.total_tokens == 17_000_000

    def test_only_input_tokens(self) -> None:
        tu = TokenUsage(input_tokens=999)
        assert tu.total_tokens == 999

    def test_only_cached_tokens(self) -> None:
        tu = TokenUsage(cached_tokens=42)
        assert tu.total_tokens == 42

    def test_only_output_tokens(self) -> None:
        tu = TokenUsage(output_tokens=7)
        assert tu.total_tokens == 7

    def test_cache_creation_does_not_affect_total(self) -> None:
        tu = TokenUsage(cache_creation_tokens=9999)
        assert tu.total_tokens == 0

    def test_to_dict_round_trip_preserves_all_fields(self) -> None:
        tu = TokenUsage(input_tokens=1, cached_tokens=2, output_tokens=3, cache_creation_tokens=4)
        d = tu.to_dict()
        tu2 = TokenUsage(
            input_tokens=d["input_tokens"],
            cached_tokens=d["cached_tokens"],
            output_tokens=d["output_tokens"],
            cache_creation_tokens=d["cache_creation_tokens"],
        )
        assert tu2.total_tokens == tu.total_tokens


class TestEdgeL3EventRecord:
    """EventRecord construction and to_dict key mapping."""

    def test_construction(self) -> None:
        er = EventRecord(
            sequence=0, event_type="run_start",
            timestamp="2026-01-01T00:00:00Z",
            payload={"plan_name": "p"},
            prev_hash="0" * 64,
            event_hash="a" * 64,
        )
        assert er.sequence == 0
        assert er.event_type == "run_start"

    def test_to_dict_key_mapping(self) -> None:
        """to_dict uses compact keys: seq, type, ts, hash."""
        er = EventRecord(
            sequence=5, event_type="task_complete",
            timestamp="2026-03-01T12:00:00Z",
            payload={"task_id": "t1"},
            prev_hash="abc",
            event_hash="def",
        )
        d = er.to_dict()
        assert d["seq"] == 5
        assert d["type"] == "task_complete"
        assert d["ts"] == "2026-03-01T12:00:00Z"
        assert d["payload"] == {"task_id": "t1"}
        assert d["prev_hash"] == "abc"
        assert d["hash"] == "def"

    def test_to_dict_empty_payload(self) -> None:
        er = EventRecord(
            sequence=1, event_type="x", timestamp="t",
            payload={}, prev_hash="", event_hash="",
        )
        assert er.to_dict()["payload"] == {}


class TestEdgeL3CircuitBreakerSpec:
    """CircuitBreakerSpec defaults and to_dict."""

    def test_defaults(self) -> None:
        cb = CircuitBreakerSpec()
        assert cb.max_total_failures == 5
        assert cb.action == "fail"

    def test_custom_values(self) -> None:
        cb = CircuitBreakerSpec(max_total_failures=10, action="pause")
        assert cb.max_total_failures == 10
        assert cb.action == "pause"

    def test_to_dict(self) -> None:
        cb = CircuitBreakerSpec(max_total_failures=3, action="fail")
        d = cb.to_dict()
        assert d == {"max_total_failures": 3, "action": "fail"}


class TestEdgeL3PolicySpecAndViolation:
    """PolicySpec and PolicyViolation serialization."""

    def test_policy_spec_defaults(self) -> None:
        ps = PolicySpec(name="p1", rule="model == 'opus'")
        assert ps.action == "warn"
        assert ps.message == ""

    def test_policy_spec_to_dict(self) -> None:
        ps = PolicySpec(name="budget", rule="cost_usd < 5", action="block", message="Too expensive")
        d = ps.to_dict()
        assert d == {
            "name": "budget",
            "rule": "cost_usd < 5",
            "action": "block",
            "message": "Too expensive",
        }

    def test_policy_violation_to_dict(self) -> None:
        pv = PolicyViolation(policy_name="p1", task_id="t1", action="block", message="blocked")
        d = pv.to_dict()
        assert d["policy_name"] == "p1"
        assert d["task_id"] == "t1"
        assert d["action"] == "block"
        assert d["message"] == "blocked"

    def test_policy_violation_audit_action(self) -> None:
        pv = PolicyViolation(policy_name="log", task_id="t2", action="audit", message="logged")
        assert pv.to_dict()["action"] == "audit"


class TestEdgeL3BlameNodeAndChain:
    """BlameNode and BlameChain construction and serialization."""

    def test_blame_node_defaults(self) -> None:
        bn = BlameNode(task_id="t1", category="root_cause", confidence=0.9, message="failed")
        assert bn.caused_by is None
        assert bn.evidence == []

    def test_blame_node_to_dict_with_evidence(self) -> None:
        bn = BlameNode(
            task_id="t1", category="timeout_propagation",
            confidence=0.7, message="timed out",
            caused_by="t0", evidence=["exit_code=124", "duration > timeout"],
        )
        d = bn.to_dict()
        assert d["caused_by"] == "t0"
        assert len(d["evidence"]) == 2

    def test_blame_chain_empty(self) -> None:
        bc = BlameChain()
        assert bc.root_task_id == ""
        assert bc.nodes == []
        assert bc.suggested_fixes == []

    def test_blame_chain_to_dict_with_nodes(self) -> None:
        node = BlameNode(task_id="t2", category="dependency_cascade", confidence=0.8, message="dep failed")
        bc = BlameChain(root_task_id="t1", nodes=[node], suggested_fixes=["retry t1"])
        d = bc.to_dict()
        assert d["root_task_id"] == "t1"
        assert len(d["nodes"]) == 1
        assert d["nodes"][0]["task_id"] == "t2"
        assert d["suggested_fixes"] == ["retry t1"]

    def test_blame_chain_mutable_independence(self) -> None:
        bc1 = BlameChain()
        bc2 = BlameChain()
        bc1.nodes.append(BlameNode(task_id="x", category="unknown", confidence=0.5, message=""))
        bc1.suggested_fixes.append("fix")
        assert bc2.nodes == []
        assert bc2.suggested_fixes == []

    @pytest.mark.parametrize("category", [
        "root_cause", "dependency_cascade", "context_corruption",
        "timeout_propagation", "budget_exhaustion", "unknown",
    ])
    def test_all_blame_categories(self, category: str) -> None:
        bn = BlameNode(task_id="t", category=category, confidence=0.6, message="m")  # type: ignore[arg-type]
        assert bn.to_dict()["category"] == category


class TestEdgeL3ContextSelectionAndTrajectory:
    """ContextSelectionEntry and ContextTrajectoryReport."""

    def test_context_selection_entry_defaults(self) -> None:
        cse = ContextSelectionEntry(upstream_id="t0")
        assert cse.score == 0.0
        assert cse.keywords_matched == []
        assert cse.hop_distance == 0
        assert cse.hop_decay_factor == 1.0
        assert cse.tokens_raw == 0
        assert cse.tokens_final == 0
        assert cse.trimmed is False
        assert cse.trim_reason == ""

    def test_context_selection_entry_to_dict_rounds_score(self) -> None:
        cse = ContextSelectionEntry(upstream_id="u", score=0.123456789, hop_decay_factor=0.876543)
        d = cse.to_dict()
        assert d["score"] == 0.1235
        assert d["hop_decay_factor"] == 0.8765

    def test_context_selection_entry_trimmed(self) -> None:
        cse = ContextSelectionEntry(
            upstream_id="u1", trimmed=True, trim_reason="budget_eviction",
            tokens_raw=500, tokens_final=0,
        )
        d = cse.to_dict()
        assert d["trimmed"] is True
        assert d["trim_reason"] == "budget_eviction"

    def test_context_trajectory_report_defaults(self) -> None:
        from maestro_cli.models import ContextTrajectoryReport
        ctr = ContextTrajectoryReport(task_id="t1")
        assert ctr.entries == []
        assert ctr.total_tokens_raw == 0
        assert ctr.total_tokens_final == 0
        assert ctr.budget_tokens is None
        assert ctr.upstreams_evicted == 0

    def test_context_trajectory_report_to_dict(self) -> None:
        from maestro_cli.models import ContextTrajectoryReport
        entry = ContextSelectionEntry(upstream_id="u", score=1.5)
        ctr = ContextTrajectoryReport(
            task_id="t1", entries=[entry],
            total_tokens_raw=100, total_tokens_final=80,
            budget_tokens=200, upstreams_evicted=1,
        )
        d = ctr.to_dict()
        assert d["task_id"] == "t1"
        assert len(d["entries"]) == 1
        assert d["upstreams_evicted"] == 1
        assert d["budget_tokens"] == 200

    def test_context_trajectory_mutable_independence(self) -> None:
        from maestro_cli.models import ContextTrajectoryReport
        a = ContextTrajectoryReport(task_id="a")
        b = ContextTrajectoryReport(task_id="b")
        a.entries.append(ContextSelectionEntry(upstream_id="x"))
        assert b.entries == []


class TestEdgeL3BatchSpecAndItemResult:
    """BatchSpec and BatchItemResult dataclasses."""

    def test_batch_spec_defaults(self) -> None:
        from maestro_cli.models import BatchSpec
        bs = BatchSpec()
        assert bs.items == []
        assert bs.template == ""
        assert bs.max_per_call == 5

    def test_batch_spec_to_dict(self) -> None:
        from maestro_cli.models import BatchSpec
        bs = BatchSpec(items=["a", "b"], template="Process {{ batch.item }}", max_per_call=2)
        d = bs.to_dict()
        assert d["items"] == ["a", "b"]
        assert d["template"] == "Process {{ batch.item }}"
        assert d["max_per_call"] == 2

    def test_batch_spec_mutable_independence(self) -> None:
        from maestro_cli.models import BatchSpec
        b1 = BatchSpec()
        b2 = BatchSpec()
        b1.items.append("x")
        assert b2.items == []

    def test_batch_item_result_defaults(self) -> None:
        from maestro_cli.models import BatchItemResult
        bir = BatchItemResult(item="file.py", chunk_index=0)
        assert bir.output == ""

    def test_batch_item_result_to_dict(self) -> None:
        from maestro_cli.models import BatchItemResult
        bir = BatchItemResult(item="file.py", chunk_index=2, output="done")
        d = bir.to_dict()
        assert d == {"item": "file.py", "chunk_index": 2, "output": "done"}


class TestEdgeL3TaskSignal:
    """TaskSignal dataclass."""

    def test_construction(self) -> None:
        from maestro_cli.models import TaskSignal
        ts = TaskSignal(signal_type="progress", timestamp="2026-01-01T00:00:00Z")
        assert ts.payload == {}

    def test_to_dict(self) -> None:
        from maestro_cli.models import TaskSignal
        ts = TaskSignal(
            signal_type="metric", timestamp="2026-01-01T00:00:00Z",
            payload={"name": "accuracy", "value": 0.95},
        )
        d = ts.to_dict()
        assert d["signal_type"] == "metric"
        assert d["payload"]["value"] == 0.95

    def test_mutable_payload_independence(self) -> None:
        from maestro_cli.models import TaskSignal
        a = TaskSignal(signal_type="log", timestamp="t")
        b = TaskSignal(signal_type="log", timestamp="t")
        a.payload["key"] = "val"
        assert b.payload == {}


class TestEdgeL3ModelRecordAndTaskHistory:
    """ModelRecord and TaskHistory dataclasses."""

    def test_model_record_to_dict_rounds_values(self) -> None:
        from maestro_cli.models import ModelRecord
        mr = ModelRecord(
            model="sonnet", runs=10, successes=8, failures=1, timeouts=1,
            avg_duration_sec=12.3456789, avg_cost_usd=0.12345678,
        )
        d = mr.to_dict()
        assert d["avg_duration_sec"] == 12.346
        assert d["avg_cost_usd"] == 0.1235

    def test_model_record_none_cost(self) -> None:
        from maestro_cli.models import ModelRecord
        mr = ModelRecord(
            model="llama3", runs=5, successes=5, failures=0, timeouts=0,
            avg_duration_sec=1.0, avg_cost_usd=None,
        )
        assert mr.to_dict()["avg_cost_usd"] is None

    def test_task_history_defaults(self) -> None:
        from maestro_cli.models import TaskHistory
        th = TaskHistory(task_id="t1", total_runs=0)
        assert th.records == {}

    def test_task_history_to_dict_with_records(self) -> None:
        from maestro_cli.models import ModelRecord, TaskHistory
        mr = ModelRecord(model="haiku", runs=3, successes=3, failures=0, timeouts=0,
                         avg_duration_sec=2.0, avg_cost_usd=0.01)
        th = TaskHistory(task_id="build", total_runs=3, records={"haiku": mr})
        d = th.to_dict()
        assert d["task_id"] == "build"
        assert "haiku" in d["records"]
        assert d["records"]["haiku"]["runs"] == 3

    def test_task_history_mutable_independence(self) -> None:
        from maestro_cli.models import TaskHistory
        a = TaskHistory(task_id="a", total_runs=0)
        b = TaskHistory(task_id="b", total_runs=0)
        a.records["x"] = None  # type: ignore[assignment]
        assert b.records == {}


class TestEdgeL3TaskContract:
    """TaskContract construction and to_dict."""

    def test_to_dict(self) -> None:
        from maestro_cli.models import TaskContract
        tc = TaskContract(
            producer_task_id="setup",
            contract_type="sql-schema",
            summary="3 tables",
            body="CREATE TABLE ...",
            content_hash="abc123",
            metadata={"table_count": 3},
        )
        d = tc.to_dict()
        assert d["producer_task_id"] == "setup"
        assert d["contract_type"] == "sql-schema"
        assert d["content_hash"] == "abc123"
        assert d["metadata"]["table_count"] == 3

    def test_metadata_default_empty(self) -> None:
        from maestro_cli.models import TaskContract
        tc = TaskContract(
            producer_task_id="x", contract_type="file-inventory",
            summary="s", body="b", content_hash="h",
        )
        assert tc.metadata == {}

    def test_metadata_mutable_independence(self) -> None:
        from maestro_cli.models import TaskContract
        a = TaskContract(producer_task_id="a", contract_type="c", summary="", body="", content_hash="")
        b = TaskContract(producer_task_id="b", contract_type="c", summary="", body="", content_hash="")
        a.metadata["k"] = "v"
        assert b.metadata == {}


class TestEdgeL3LessonRecord:
    """LessonRecord construction and to_dict."""

    def test_defaults(self) -> None:
        from maestro_cli.models import LessonRecord
        lr = LessonRecord(iteration=1, task_id="t1", category="timeout_fix", lesson="Increased timeout to 600s")
        assert lr.confidence == 0.8
        assert lr.timestamp == ""

    def test_to_dict(self) -> None:
        from maestro_cli.models import LessonRecord
        lr = LessonRecord(
            iteration=3, task_id="t2", category="guard_fix",
            lesson="Added JSON guard", confidence=0.95,
            timestamp="2026-03-01T00:00:00Z",
        )
        d = lr.to_dict()
        assert d["iteration"] == 3
        assert d["task_id"] == "t2"
        assert d["category"] == "guard_fix"
        assert d["confidence"] == 0.95


class TestEdgeL3KnowledgeRecord:
    """KnowledgeRecord construction and to_dict."""

    def test_to_dict_rounds_confidence(self) -> None:
        from maestro_cli.models import KnowledgeRecord
        kr = KnowledgeRecord(
            task_id="build", kind="failure_pattern",
            insight="Fails on Windows with path issues",
            confidence=0.777777, occurrences=5,
            first_seen="2026-01-01T00:00:00Z", last_seen="2026-03-01T00:00:00Z",
        )
        d = kr.to_dict()
        assert d["confidence"] == 0.778
        assert d["occurrences"] == 5
        assert d["kind"] == "failure_pattern"

    @pytest.mark.parametrize("kind", [
        "failure_pattern", "timeout_hint", "success_pattern",
        "cost_pattern", "duration_pattern", "retry_pattern", "model_pattern",
    ])
    def test_all_knowledge_kinds(self, kind: str) -> None:
        from maestro_cli.models import KnowledgeRecord
        kr = KnowledgeRecord(
            task_id="t", kind=kind, insight="i",  # type: ignore[arg-type]
            confidence=0.5, occurrences=1,
            first_seen="2026-01-01", last_seen="2026-01-01",
        )
        assert kr.to_dict()["kind"] == kind


class TestEdgeL3MergeReview:
    """MergeReview and MergeOverlap to_dict."""

    def test_merge_review_minimal(self) -> None:
        from maestro_cli.models import MergeReview
        mr = MergeReview(verdict="safe")
        d = mr.to_dict()
        assert d["verdict"] == "safe"
        assert d["overlapping_files"] == []
        assert d["conflict_files"] == []
        assert d["auto_resolved"] is False
        assert "resolution_suggestion" not in d
        assert "review_model" not in d

    def test_merge_review_full(self) -> None:
        from maestro_cli.models import MergeOverlap, MergeReview
        overlap = MergeOverlap(file="src/main.py", merged_by=["t1", "t2"])
        mr = MergeReview(
            verdict="resolvable",
            overlapping_files=[overlap],
            conflict_files=["src/main.py"],
            resolution_suggestion="Use t2's version",
            auto_resolved=True,
            review_model="haiku",
            review_duration_sec=2.567,
            review_cost_usd=0.001,
        )
        d = mr.to_dict()
        assert d["verdict"] == "resolvable"
        assert d["overlapping_files"][0]["file"] == "src/main.py"
        assert d["overlapping_files"][0]["merged_by"] == ["t1", "t2"]
        assert d["resolution_suggestion"] == "Use t2's version"
        assert d["review_model"] == "haiku"
        assert d["review_duration_sec"] == 2.57
        assert d["review_cost_usd"] == 0.001

    def test_merge_review_zero_duration_excluded(self) -> None:
        from maestro_cli.models import MergeReview
        mr = MergeReview(verdict="safe", review_duration_sec=0.0)
        d = mr.to_dict()
        assert "review_duration_sec" not in d

    @pytest.mark.parametrize("verdict", ["safe", "resolvable", "conflict", "error"])
    def test_all_merge_verdicts(self, verdict: str) -> None:
        from maestro_cli.models import MergeReview
        mr = MergeReview(verdict=verdict)  # type: ignore[arg-type]
        assert mr.to_dict()["verdict"] == verdict


class TestEdgeL3WorktreeMergeResultWithReview:
    """WorktreeMergeResult with embedded MergeReview."""

    def test_without_review(self) -> None:
        wmr = WorktreeMergeResult(status="merged")
        d = wmr.to_dict()
        assert "review" not in d

    def test_with_review(self) -> None:
        from maestro_cli.models import MergeReview
        review = MergeReview(verdict="safe")
        wmr = WorktreeMergeResult(status="merged", review=review)
        d = wmr.to_dict()
        assert "review" in d
        assert d["review"]["verdict"] == "safe"

    def test_error_status_with_error_message(self) -> None:
        wmr = WorktreeMergeResult(status="error", error="merge conflict on main.py")
        d = wmr.to_dict()
        assert d["status"] == "error"
        assert d["error"] == "merge conflict on main.py"


class TestEdgeL3PlanRunResultAggregation:
    """PlanRunResult total_tokens, total_cost_usd, budget_exceeded."""

    def test_total_tokens_default_none(self) -> None:
        now = datetime.now(UTC)
        prr = PlanRunResult(
            plan_name="p", run_id="r", run_path=Path("/tmp"),
            started_at=now, finished_at=now, success=True,
        )
        assert prr.total_tokens is None
        assert prr.total_cost_usd is None
        assert prr.budget_exceeded is False

    def test_to_dict_budget_exceeded(self) -> None:
        now = datetime.now(UTC)
        prr = PlanRunResult(
            plan_name="p", run_id="r", run_path=Path("/tmp"),
            started_at=now, finished_at=now, success=False,
            budget_exceeded=True, total_cost_usd=10.5,
        )
        d = prr.to_dict()
        assert d["budget_exceeded"] is True
        assert d["total_cost_usd"] == 10.5

    def test_to_dict_with_task_results(self) -> None:
        now = datetime.now(UTC)
        tr = TaskResult(
            task_id="t1", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=1.0,
            command="echo", log_path=Path("/t.log"), result_path=Path("/t.json"),
        )
        prr = PlanRunResult(
            plan_name="p", run_id="r", run_path=Path("/tmp"),
            started_at=now, finished_at=now, success=True,
            task_results={"t1": tr},
        )
        d = prr.to_dict()
        assert "t1" in d["task_results"]
        assert d["task_results"]["t1"]["status"] == "success"

    def test_execution_profile_default(self) -> None:
        now = datetime.now(UTC)
        prr = PlanRunResult(
            plan_name="p", run_id="r", run_path=Path("/tmp"),
            started_at=now, finished_at=now, success=True,
        )
        assert prr.execution_profile == "plan"
        assert prr.to_dict()["execution_profile"] == "plan"


class TestEdgeL3TaskResultConditionalFields:
    """TaskResult to_dict conditional inclusion of optional fields."""

    def _make_result(self, **kwargs: object) -> TaskResult:
        now = datetime.now(UTC)
        defaults = dict(
            task_id="t1", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=1.0,
            command="echo", log_path=Path("/t.log"), result_path=Path("/t.json"),
        )
        defaults.update(kwargs)
        return TaskResult(**defaults)  # type: ignore[arg-type]

    def test_batch_results_included_when_non_empty(self) -> None:
        from maestro_cli.models import BatchItemResult
        bir = BatchItemResult(item="a.py", chunk_index=0, output="ok")
        result = self._make_result(batch_results=[bir], batch_chunks_total=1, batch_items_total=1)
        d = result.to_dict()
        assert "batch_results" in d
        assert d["batch_chunks_total"] == 1
        assert d["batch_items_total"] == 1

    def test_batch_results_excluded_when_empty(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        assert "batch_results" not in d

    def test_deliberation_skipped_included_when_true(self) -> None:
        result = self._make_result(deliberation_skipped=True)
        d = result.to_dict()
        assert d["deliberation_skipped"] is True

    def test_deliberation_skipped_excluded_when_false(self) -> None:
        result = self._make_result(deliberation_skipped=False)
        d = result.to_dict()
        assert "deliberation_skipped" not in d

    def test_structured_output_included_when_set(self) -> None:
        result = self._make_result(structured_output={"key": "value"})
        d = result.to_dict()
        assert d["structured_output"] == {"key": "value"}

    def test_structured_output_excluded_when_none(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        assert "structured_output" not in d

    def test_dynamic_subplan_result_included_when_set(self) -> None:
        result = self._make_result(dynamic_subplan_result={"tasks": 3, "success": True})
        d = result.to_dict()
        assert d["dynamic_subplan_result"]["tasks"] == 3

    def test_tainted_included_when_true(self) -> None:
        result = self._make_result(tainted=True)
        d = result.to_dict()
        assert d["tainted"] is True

    def test_tainted_excluded_when_false(self) -> None:
        result = self._make_result(tainted=False)
        d = result.to_dict()
        assert "tainted" not in d

    def test_tool_failure_count_included_when_non_zero(self) -> None:
        result = self._make_result(tool_failure_count=2)
        d = result.to_dict()
        assert d["tool_failure_count"] == 2

    def test_tool_failure_count_excluded_when_zero(self) -> None:
        result = self._make_result(tool_failure_count=0)
        d = result.to_dict()
        assert "tool_failure_count" not in d

    def test_signals_received_included_when_non_empty(self) -> None:
        from maestro_cli.models import TaskSignal
        sig = TaskSignal(signal_type="progress", timestamp="t", payload={"pct": 50})
        result = self._make_result(signals_received=[sig])
        d = result.to_dict()
        assert "signals_received" in d
        assert d["signals_received"][0]["signal_type"] == "progress"

    def test_signals_received_excluded_when_empty(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        assert "signals_received" not in d

    def test_artifacts_included_when_non_empty(self) -> None:
        result = self._make_result(artifacts=[{"path": "/out/report.html"}])
        d = result.to_dict()
        assert d["artifacts"] == [{"path": "/out/report.html"}]

    def test_artifacts_excluded_when_empty(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        assert "artifacts" not in d

    def test_last_progress_pct_included_when_set(self) -> None:
        result = self._make_result(last_progress_pct=75)
        d = result.to_dict()
        assert d["last_progress_pct"] == 75

    def test_last_progress_pct_excluded_when_none(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        assert "last_progress_pct" not in d

    def test_context_trajectory_included_when_set(self) -> None:
        from maestro_cli.models import ContextTrajectoryReport
        ctr = ContextTrajectoryReport(task_id="t1", total_tokens_raw=100)
        result = self._make_result(context_trajectory=ctr)
        d = result.to_dict()
        assert "context_trajectory" in d
        assert d["context_trajectory"]["total_tokens_raw"] == 100

    def test_context_trajectory_excluded_when_none(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        assert "context_trajectory" not in d

    def test_token_usage_in_to_dict(self) -> None:
        tu = TokenUsage(input_tokens=100, output_tokens=50)
        result = self._make_result(token_usage=tu)
        d = result.to_dict()
        assert d["token_usage"]["total_tokens"] == 150

    def test_token_usage_none_in_to_dict(self) -> None:
        result = self._make_result()
        d = result.to_dict()
        assert d["token_usage"] is None

    def test_handoff_report_in_to_dict(self) -> None:
        hr = HandoffReport(failure_category="timeout", summary="oops")
        result = self._make_result(handoff_report=hr)
        d = result.to_dict()
        assert d["handoff_report"]["failure_category"] == "timeout"

    def test_judge_result_in_to_dict(self) -> None:
        jr = JudgeResult(verdict="pass", overall_score=0.9)
        result = self._make_result(judge_result=jr)
        d = result.to_dict()
        assert d["judge_result"]["verdict"] == "pass"

    def test_failure_history_in_to_dict(self) -> None:
        fr = FailureRecord(attempt=1, category="timeout", exit_code=124, message="timeout")
        result = self._make_result(failure_history=[fr], retry_count=1)
        d = result.to_dict()
        assert d["retry_count"] == 1
        assert len(d["failure_history"]) == 1

    def test_produced_contract_in_to_dict(self) -> None:
        from maestro_cli.models import TaskContract
        tc = TaskContract(
            producer_task_id="t1", contract_type="sql-schema",
            summary="s", body="b", content_hash="h",
        )
        result = self._make_result(produced_contract=tc)
        d = result.to_dict()
        assert d["produced_contract"]["contract_type"] == "sql-schema"


class TestEdgeL3TaskSpecNewFields:
    """TaskSpec fields added in later versions."""

    def test_frozen_default_false(self) -> None:
        task = TaskSpec(id="t1")
        assert task.frozen is False

    def test_compress_before_default_false(self) -> None:
        task = TaskSpec(id="t1")
        assert task.compress_before is False

    def test_honeypot_default_false(self) -> None:
        task = TaskSpec(id="t1")
        assert task.honeypot is False

    def test_deliberation_default_false(self) -> None:
        task = TaskSpec(id="t1")
        assert task.deliberation is False

    def test_deliberation_threshold_default(self) -> None:
        task = TaskSpec(id="t1")
        assert task.deliberation_threshold == 0.5

    def test_dynamic_group_default_false(self) -> None:
        task = TaskSpec(id="t1")
        assert task.dynamic_group is False

    def test_signals_default_false(self) -> None:
        task = TaskSpec(id="t1")
        assert task.signals is False

    def test_reminders_default_none(self) -> None:
        task = TaskSpec(id="t1")
        assert task.reminders is None

    def test_output_schema_default_none(self) -> None:
        task = TaskSpec(id="t1")
        assert task.output_schema is None

    def test_context_trust_default_none(self) -> None:
        task = TaskSpec(id="t1")
        assert task.context_trust is None

    def test_observation_block_default_false(self) -> None:
        task = TaskSpec(id="t1")
        assert task.observation_block is False

    def test_retry_strategy_default_none(self) -> None:
        task = TaskSpec(id="t1")
        assert task.retry_strategy is None

    def test_negative_cache_ttl_default_none(self) -> None:
        task = TaskSpec(id="t1")
        assert task.negative_cache_ttl_sec is None

    def test_batch_default_none(self) -> None:
        task = TaskSpec(id="t1")
        assert task.batch is None

    def test_to_dict_includes_new_bool_fields(self) -> None:
        task = TaskSpec(
            id="t1", frozen=True, compress_before=True,
            honeypot=True, deliberation=True, dynamic_group=True,
            signals=True,
        )
        d = task.to_dict()
        assert d["frozen"] is True
        assert d["deliberation"] is True
        assert d["dynamic_group"] is True
        assert d["signals"] is True

    def test_to_dict_with_batch(self) -> None:
        from maestro_cli.models import BatchSpec
        bs = BatchSpec(items=["a", "b"], template="do {{ batch.item }}", max_per_call=1)
        task = TaskSpec(id="t1", batch=bs)
        d = task.to_dict()
        assert d["batch"]["items"] == ["a", "b"]
        assert d["batch"]["max_per_call"] == 1

    def test_to_dict_negative_cache_ttl(self) -> None:
        task = TaskSpec(id="t1", negative_cache_ttl_sec=120)
        d = task.to_dict()
        assert d["negative_cache_ttl_sec"] == 120

    def test_to_dict_batch_none(self) -> None:
        task = TaskSpec(id="t1")
        assert task.to_dict()["batch"] is None

    def test_to_dict_reminders(self) -> None:
        task = TaskSpec(id="t1", reminders=[{"trigger": "timeout", "message": "Try increasing timeout"}])
        d = task.to_dict()
        assert d["reminders"] == [{"trigger": "timeout", "message": "Try increasing timeout"}]

    def test_to_dict_output_schema(self) -> None:
        schema = {"type": "object", "properties": {"status": {"type": "string"}}}
        task = TaskSpec(id="t1", output_schema=schema)
        d = task.to_dict()
        assert "output_schema" not in d  # output_schema not in to_dict keys


class TestEdgeL3PlanSpecConditionalToDict:
    """PlanSpec.to_dict() conditional fields."""

    def test_control_flow_integrity_excluded_when_false(self) -> None:
        ps = PlanSpec(name="p", control_flow_integrity=False)
        d = ps.to_dict()
        assert "control_flow_integrity" not in d

    def test_control_flow_integrity_included_when_true(self) -> None:
        ps = PlanSpec(name="p", control_flow_integrity=True)
        d = ps.to_dict()
        assert d["control_flow_integrity"] is True

    def test_routing_strategy_excluded_when_none(self) -> None:
        ps = PlanSpec(name="p")
        d = ps.to_dict()
        assert "routing_strategy" not in d

    def test_routing_strategy_included_when_set(self) -> None:
        ps = PlanSpec(name="p", routing_strategy="cost_optimized")
        d = ps.to_dict()
        assert d["routing_strategy"] == "cost_optimized"

    def test_policies_excluded_when_empty(self) -> None:
        ps = PlanSpec(name="p")
        d = ps.to_dict()
        assert "policies" not in d

    def test_policies_included_when_present(self) -> None:
        pol = PolicySpec(name="cost", rule="cost_usd < 1", action="block")
        ps = PlanSpec(name="p", policies=[pol])
        d = ps.to_dict()
        assert len(d["policies"]) == 1
        assert d["policies"][0]["name"] == "cost"

    def test_circuit_breaker_excluded_when_none(self) -> None:
        ps = PlanSpec(name="p")
        d = ps.to_dict()
        assert "circuit_breaker" not in d

    def test_circuit_breaker_included_when_set(self) -> None:
        from maestro_cli.models import CircuitBreakerSpec
        cb = CircuitBreakerSpec(max_total_failures=2, action="pause")
        ps = PlanSpec(name="p", circuit_breaker=cb)
        d = ps.to_dict()
        assert d["circuit_breaker"]["max_total_failures"] == 2
        assert d["circuit_breaker"]["action"] == "pause"


class TestEdgeL3PlanSpecMutableFields:
    """PlanSpec mutable field independence."""

    def test_secrets_independence(self) -> None:
        a = PlanSpec(name="a")
        b = PlanSpec(name="b")
        a.secrets.append("KEY")
        assert b.secrets == []

    def test_tasks_independence(self) -> None:
        a = PlanSpec(name="a")
        b = PlanSpec(name="b")
        a.tasks.append(TaskSpec(id="t"))
        assert b.tasks == []

    def test_imports_independence(self) -> None:
        a = PlanSpec(name="a")
        b = PlanSpec(name="b")
        a.imports.append(PlanImport(path="x.yaml", prefix="x"))
        assert b.imports == []

    def test_policies_independence(self) -> None:
        a = PlanSpec(name="a")
        b = PlanSpec(name="b")
        a.policies.append(PolicySpec(name="p", rule="r"))
        assert b.policies == []

    def test_audit_packs_independence(self) -> None:
        a = PlanSpec(name="a")
        b = PlanSpec(name="b")
        a.audit_packs.append("pack1")
        assert b.audit_packs == []

    def test_validation_warnings_independence(self) -> None:
        a = PlanSpec(name="a")
        b = PlanSpec(name="b")
        a.validation_warnings.append("W1")
        assert b.validation_warnings == []


class TestEdgeL3ConstantsCompleteness:
    """Verify constant sets are non-empty and consistent."""

    def test_signal_types_is_frozenset(self) -> None:
        from maestro_cli.models import SIGNAL_TYPES
        assert isinstance(SIGNAL_TYPES, frozenset)
        assert "progress" in SIGNAL_TYPES
        assert "compress" in SIGNAL_TYPES

    def test_workspace_assertion_types(self) -> None:
        from maestro_cli.models import WORKSPACE_ASSERTION_TYPES
        assert "file_contains" in WORKSPACE_ASSERTION_TYPES
        assert "file_contains_count" in WORKSPACE_ASSERTION_TYPES
        assert "glob_exists" in WORKSPACE_ASSERTION_TYPES
        assert len(WORKSPACE_ASSERTION_TYPES) == 9

    def test_contract_types(self) -> None:
        from maestro_cli.models import CONTRACT_TYPES
        expected = {"sql-schema", "dependency-manifest", "conventions-doc",
                    "file-inventory", "api-schema", "test-manifest"}
        assert CONTRACT_TYPES == expected

    def test_quorum_strategies(self) -> None:
        from maestro_cli.models import QUORUM_STRATEGIES
        assert QUORUM_STRATEGIES == {"majority", "unanimous", "any"}

    def test_context_trust_values(self) -> None:
        from maestro_cli.models import CONTEXT_TRUST_VALUES
        assert CONTEXT_TRUST_VALUES == {"trusted", "untrusted"}

    def test_status_styles_covers_running_and_pending(self) -> None:
        assert "pending" in STATUS_STYLES
        assert "running" in STATUS_STYLES
        # pending and running are NOT terminal
        assert "pending" not in TERMINAL_STATUSES
        assert "running" not in TERMINAL_STATUSES

    def test_status_styles_values_are_tuples(self) -> None:
        for key, val in STATUS_STYLES.items():
            assert isinstance(val, tuple), f"STATUS_STYLES[{key}] should be a tuple"
            assert len(val) == 2

    def test_codex_model_aliases_keys_are_strings(self) -> None:
        for alias, full in CODEX_MODEL_ALIASES.items():
            assert isinstance(alias, str)
            assert isinstance(full, str)
            assert full.startswith("gpt-")

    def test_copilot_model_aliases_size(self) -> None:
        # At least Claude + GPT + Gemini
        assert len(COPILOT_MODEL_ALIASES) >= 15

    def test_judge_presets_ai_slop_detection(self) -> None:
        assert "ai_slop_detection" in JUDGE_PRESETS
        slop = JUDGE_PRESETS["ai_slop_detection"]
        assert len(slop["criteria"]) == 5
        assert slop["aggregation"] == "weighted_mean"


class TestEdgeL3WatchSpecAllFields:
    """WatchSpec full field coverage in to_dict."""

    def test_all_fields_present_in_to_dict(self) -> None:
        ws = WatchSpec(
            metric="coverage",
            max_iterations=50,
            iteration_budget_sec=120,
            metric_direction="higher_is_better",
            metric_source="manifest",
            metric_pattern=None,
            metric_task="run-tests",
            metric_json_path=None,
            on_regression="keep",
            program_md="PROGRAM.md",
            warmup_iterations=2,
            plateau_threshold=3,
            plateau_action="escalate_model",
            max_cost_usd=10.0,
            consolidate_model="opus",
            consolidate_every=5,
            consolidate_prompt="Summarize",
            target_metric=95.0,
            blame_plan="target.yaml",
            mode="improve",
            improve_model="sonnet",
            max_total_steps=500,
        )
        d = ws.to_dict()
        assert d["metric"] == "coverage"
        assert d["max_iterations"] == 50
        assert d["iteration_budget_sec"] == 120
        assert d["metric_direction"] == "higher_is_better"
        assert d["metric_source"] == "manifest"
        assert d["on_regression"] == "keep"
        assert d["program_md"] == "PROGRAM.md"
        assert d["warmup_iterations"] == 2
        assert d["plateau_threshold"] == 3
        assert d["plateau_action"] == "escalate_model"
        assert d["consolidate_model"] == "opus"
        assert d["consolidate_every"] == 5
        assert d["consolidate_prompt"] == "Summarize"
        assert d["target_metric"] == 95.0
        assert d["blame_plan"] == "target.yaml"
        assert d["mode"] == "improve"
        assert d["improve_model"] == "sonnet"
        assert d["max_total_steps"] == 500


class TestEdgeL3WatchStateAllFields:
    """WatchState full field coverage."""

    def test_total_steps_default(self) -> None:
        ws = WatchState()
        assert ws.total_steps == 0

    def test_best_iteration_default(self) -> None:
        ws = WatchState()
        assert ws.best_iteration is None

    def test_to_dict_all_fields(self) -> None:
        ws = WatchState(
            plan_path="p.yaml",
            status="target_reached",
            best_metric=42.0,
            best_iteration=7,
            total_cost_usd=3.5,
            total_iterations=10,
            plateau_count=2,
            total_steps=50,
        )
        d = ws.to_dict()
        assert d["status"] == "target_reached"
        assert d["best_iteration"] == 7
        assert d["total_steps"] == 50


class TestEdgeL3ReplanAttemptAllFields:
    """ReplanAttempt full field coverage."""

    def test_analysis_fields(self) -> None:
        ra = ReplanAttempt(
            attempt_number=1,
            analysis_response="Need to fix timeout",
            analysis_error=None,
            diff_summary="--- a\n+++ b",
            candidate_variants=[{"node_id": "v1", "score": 0.8}],
            selected_candidate_id="v1",
        )
        d = ra.to_dict()
        assert d["analysis_response"] == "Need to fix timeout"
        assert d["analysis_error"] is None
        assert d["diff_summary"] == "--- a\n+++ b"
        assert d["candidate_variants"][0]["node_id"] == "v1"
        assert d["selected_candidate_id"] == "v1"

    def test_corrected_plan_yaml(self) -> None:
        ra = ReplanAttempt(
            attempt_number=2,
            corrected_plan_yaml="version: 1\nname: fixed",
        )
        d = ra.to_dict()
        assert d["corrected_plan_yaml"] == "version: 1\nname: fixed"


class TestEdgeL3ReplanStateAllFields:
    """ReplanState total_tokens field."""

    def test_total_tokens_default(self) -> None:
        rs = ReplanState()
        assert rs.total_tokens == 0

    def test_total_tokens_in_to_dict(self) -> None:
        rs = ReplanState(total_tokens=5000)
        d = rs.to_dict()
        assert d["total_tokens"] == 5000

    def test_search_tree_path_in_to_dict(self) -> None:
        rs = ReplanState(search_tree_path="C:/tmp/tree.jsonl")
        d = rs.to_dict()
        assert d["search_tree_path"] == "C:/tmp/tree.jsonl"


class TestEdgeL3PlanDefaultsEngineFields:
    """PlanDefaults per-engine EngineDefaults independence."""

    def test_each_engine_is_independent_instance(self) -> None:
        pd = PlanDefaults()
        pd.codex.model = "5.4"
        assert pd.claude.model is None
        assert pd.gemini.model is None

    def test_engine_defaults_context_model(self) -> None:
        ed = EngineDefaults(context_model="haiku")
        assert ed.context_model == "haiku"

    def test_engine_defaults_context_model_default_none(self) -> None:
        ed = EngineDefaults()
        assert ed.context_model is None

    def test_plan_defaults_signals_default(self) -> None:
        pd = PlanDefaults()
        assert pd.signals is False

    def test_plan_defaults_requires_clean_worktree(self) -> None:
        pd = PlanDefaults(requires_clean_worktree=True)
        assert pd.requires_clean_worktree is True

    def test_plan_defaults_secrets_auto(self) -> None:
        pd = PlanDefaults(secrets_auto=True, secrets=["MY_KEY"])
        assert pd.secrets_auto is True
        assert pd.secrets == ["MY_KEY"]


class TestEdgeL3ContextExtractionAlias:
    """ContextExtraction is a backward-compatible alias for WorkspaceExtraction."""

    def test_alias_is_same_class(self) -> None:
        from maestro_cli.models import ContextExtraction
        assert ContextExtraction is WorkspaceExtraction

    def test_alias_creates_workspace_extraction(self) -> None:
        from maestro_cli.models import ContextExtraction
        ce = ContextExtraction(relevant_files=["x.py"], token_estimate=10)
        assert isinstance(ce, WorkspaceExtraction)
        assert ce.relevant_files == ["x.py"]


class TestEdgeL3WorkspaceExtractionEdgeCases:
    """WorkspaceExtraction edge cases."""

    def test_to_dict_creates_copies(self) -> None:
        we = WorkspaceExtraction(relevant_files=["a.py"], snippets={"a.py": "code"})
        d = we.to_dict()
        # Mutating the dict should not affect the original
        d["relevant_files"].append("b.py")
        d["snippets"]["b.py"] = "more"
        assert we.relevant_files == ["a.py"]
        assert "b.py" not in we.snippets

    def test_empty_snippets(self) -> None:
        we = WorkspaceExtraction()
        assert we.snippets == {}
        assert we.reasoning == ""
        assert we.token_estimate == 0


class TestEdgeL3WorkspaceBriefEdgeCases:
    """WorkspaceBrief edge cases."""

    def test_to_dict_creates_copy_of_files(self) -> None:
        wb = WorkspaceBrief(files_referenced=["a.py"])
        d = wb.to_dict()
        d["files_referenced"].append("b.py")
        assert wb.files_referenced == ["a.py"]

    def test_defaults(self) -> None:
        wb = WorkspaceBrief()
        assert wb.brief_text == ""
        assert wb.token_estimate == 0
        assert wb.files_referenced == []


class TestEdgeL3PlanImportEdgeCases:
    """PlanImport edge cases."""

    def test_overrides_empty_dict_excluded(self) -> None:
        pi = PlanImport(path="x.yaml", prefix="x", overrides={})
        d = pi.to_dict()
        # Empty dict is falsy, so "overrides" should not be in dict
        assert "overrides" not in d

    def test_overrides_mutable_independence(self) -> None:
        a = PlanImport(path="a.yaml", prefix="a")
        b = PlanImport(path="b.yaml", prefix="b")
        a.overrides["model"] = "opus"
        assert b.overrides == {}


class TestEdgeL3SuggestionEdgeCases:
    """Suggestion edge cases."""

    def test_estimated_savings_none(self) -> None:
        s = Suggestion(
            task_id="t1", category="add_judge", severity="info",
            reason="No quality gate", current_value="none",
            suggested_value="judge", confidence=0.5,
        )
        d = s.to_dict()
        assert d["estimated_savings_pct"] is None

    @pytest.mark.parametrize("category", [
        "downgrade_model", "upgrade_model", "add_judge", "remove_judge",
        "add_retry", "reduce_retry", "adjust_effort", "add_review_task",
        "add_checkpoint", "reduce_context_budget", "fix_failure_pattern",
        "tune_timeout",
    ])
    def test_all_suggestion_categories(self, category: str) -> None:
        s = Suggestion(
            task_id="t", category=category, severity="low",  # type: ignore[arg-type]
            reason="r", current_value="a", suggested_value="b", confidence=0.5,
        )
        assert s.to_dict()["category"] == category


class TestEdgeL3JudgeSpecQuorumFields:
    """JudgeSpec quorum and debate fields."""

    def test_quorum_defaults(self) -> None:
        js = JudgeSpec()
        assert js.quorum is None
        assert js.quorum_strategy is None
        assert js.debate_rounds == 2

    def test_quorum_in_to_dict(self) -> None:
        js = JudgeSpec(quorum=3, quorum_strategy="unanimous")
        d = js.to_dict()
        assert d["quorum"] == 3
        assert d["quorum_strategy"] == "unanimous"

    def test_timeout_sec_in_to_dict(self) -> None:
        js = JudgeSpec(timeout_sec=120)
        d = js.to_dict()
        assert d["timeout_sec"] == 120

    def test_debate_rounds_in_to_dict(self) -> None:
        js = JudgeSpec(method="debate", debate_rounds=5)
        d = js.to_dict()
        assert d["method"] == "debate"
        assert d["debate_rounds"] == 5


class TestEdgeL3MultiPlanResultTimestamps:
    """MultiPlanResult timestamps in to_dict."""

    def test_timestamps_are_iso_strings(self) -> None:
        mpr = MultiPlanResult()
        d = mpr.to_dict()
        # started_at and finished_at should be ISO format strings
        assert isinstance(d["started_at"], str)
        assert isinstance(d["finished_at"], str)
        assert "T" in d["started_at"]


class TestEdgeL3TaskBriefAllFields:
    """TaskBrief field coverage."""

    def test_auto_split_default_true(self) -> None:
        tb = TaskBrief(id="t1")
        assert tb.auto_split is True

    def test_workdir_default_none(self) -> None:
        tb = TaskBrief(id="t1")
        assert tb.workdir is None

    def test_all_fields_set(self) -> None:
        tb = TaskBrief(
            id="sec", description="Audit", task_type="security-audit",
            depends_on=["build"], engine="claude", agent="security-engineer",
            workdir="/repo", prompt_hint="Focus on auth", auto_split=False,
        )
        assert tb.id == "sec"
        assert tb.task_type == "security-audit"
        assert tb.engine == "claude"
        assert tb.auto_split is False


class TestEdgeL3PlanBriefAllFields:
    """PlanBrief field coverage."""

    def test_goal_default_empty(self) -> None:
        pb = PlanBrief(name="p")
        assert pb.goal == ""

    def test_workspace_root_and_branch(self) -> None:
        pb = PlanBrief(name="p", workspace_root="/repo", branch_name="feature/x")
        assert pb.workspace_root == "/repo"
        assert pb.branch_name == "feature/x"

    @pytest.mark.parametrize("topology", ["linear", "fan-out", "diamond", "pipeline"])
    def test_all_topologies(self, topology: str) -> None:
        pb = PlanBrief(name="p", topology=topology)  # type: ignore[arg-type]
        assert pb.topology == topology


# ===========================================================================
# CWE Security Profiles — JUDGE_PRESETS + CWE_SECURITY_PROFILES constant
# ===========================================================================


class TestCWESecurityProfiles:
    """Validate CWE security profile presets and their structure."""

    _CWE_NAMES = {"cwe_injection", "cwe_auth", "cwe_data_exposure", "cwe_top_25"}

    def test_cwe_security_profiles_contains_all_four(self) -> None:
        from maestro_cli.models import CWE_SECURITY_PROFILES
        assert CWE_SECURITY_PROFILES == self._CWE_NAMES

    def test_all_cwe_profiles_in_judge_presets(self) -> None:
        for name in self._CWE_NAMES:
            assert name in JUDGE_PRESETS, f"{name} missing from JUDGE_PRESETS"

    @pytest.mark.parametrize("profile", ["cwe_injection", "cwe_auth", "cwe_data_exposure", "cwe_top_25"])
    def test_cwe_preset_valid_structure(self, profile: str) -> None:
        preset = JUDGE_PRESETS[profile]
        assert isinstance(preset["criteria"], list)
        assert len(preset["criteria"]) > 0
        assert isinstance(preset["pass_threshold"], float)
        assert isinstance(preset["aggregation"], str)

    @pytest.mark.parametrize("profile", ["cwe_injection", "cwe_auth", "cwe_data_exposure", "cwe_top_25"])
    def test_cwe_preset_criteria_rubric_structure(self, profile: str) -> None:
        for criterion in JUDGE_PRESETS[profile]["criteria"]:
            assert criterion["type"] == "rubric"
            assert isinstance(criterion["name"], str)
            assert len(criterion["name"]) > 0
            assert isinstance(criterion["levels"], list)
            assert len(criterion["levels"]) >= 2
            assert isinstance(criterion["min_score"], int)
            assert isinstance(criterion["weight"], float)

    @pytest.mark.parametrize("profile", ["cwe_injection", "cwe_auth", "cwe_data_exposure", "cwe_top_25"])
    def test_cwe_presets_use_min_aggregation(self, profile: str) -> None:
        assert JUDGE_PRESETS[profile]["aggregation"] == "min"

    @pytest.mark.parametrize("profile", ["cwe_injection", "cwe_auth", "cwe_data_exposure", "cwe_top_25"])
    def test_cwe_presets_high_pass_threshold(self, profile: str) -> None:
        assert JUDGE_PRESETS[profile]["pass_threshold"] >= 0.7

    @pytest.mark.parametrize("profile", ["cwe_injection", "cwe_auth", "cwe_data_exposure", "cwe_top_25"])
    def test_cwe_rubric_levels_scores_between_1_and_5(self, profile: str) -> None:
        for criterion in JUDGE_PRESETS[profile]["criteria"]:
            for level in criterion["levels"]:
                assert 1 <= level["score"] <= 5

    def test_cwe_injection_has_four_criteria(self) -> None:
        criteria = JUDGE_PRESETS["cwe_injection"]["criteria"]
        assert len(criteria) == 4
        names = {c["name"] for c in criteria}
        assert "SQL Injection (CWE-89)" in names
        assert "Command Injection (CWE-78)" in names
        assert "XSS (CWE-79)" in names
        assert "Path Traversal (CWE-22)" in names

    def test_cwe_auth_has_four_criteria(self) -> None:
        criteria = JUDGE_PRESETS["cwe_auth"]["criteria"]
        assert len(criteria) == 4
        names = {c["name"] for c in criteria}
        assert "Broken Authentication (CWE-287)" in names
        assert "Broken Access Control (CWE-284)" in names
        assert "Credential Storage (CWE-256)" in names
        assert "Session Management (CWE-384)" in names

    def test_cwe_top_25_has_five_criteria(self) -> None:
        criteria = JUDGE_PRESETS["cwe_top_25"]["criteria"]
        assert len(criteria) == 5


class TestDualVerificationResult:
    """Tests for DualVerificationResult dataclass."""

    def test_default_field_values(self) -> None:
        result = DualVerificationResult(verified=True)
        assert result.verified is True
        assert result.files_in_diff == []
        assert result.files_claimed == []
        assert result.unclaimed_files == []
        assert result.phantom_files == []
        assert result.overlap_ratio == 0.0

    def test_to_dict_serialization(self) -> None:
        result = DualVerificationResult(
            verified=True,
            files_in_diff=["a.py"],
            files_claimed=["a.py"],
            unclaimed_files=[],
            phantom_files=[],
            overlap_ratio=1.0,
        )
        d = result.to_dict()
        assert d["verified"] is True
        assert d["files_in_diff"] == ["a.py"]
        assert d["files_claimed"] == ["a.py"]
        assert d["unclaimed_files"] == []
        assert d["phantom_files"] == []
        assert d["overlap_ratio"] == 1.0

    def test_verified_true_all_fields(self) -> None:
        result = DualVerificationResult(
            verified=True,
            files_in_diff=["src/a.py", "src/b.py"],
            files_claimed=["src/a.py", "src/b.py"],
            unclaimed_files=[],
            phantom_files=[],
            overlap_ratio=1.0,
        )
        assert result.verified is True
        assert len(result.files_in_diff) == 2
        assert len(result.files_claimed) == 2

    def test_verified_false_with_gaps(self) -> None:
        result = DualVerificationResult(
            verified=False,
            files_in_diff=["src/a.py", "src/b.py"],
            files_claimed=["src/a.py", "ghost.py"],
            unclaimed_files=["src/b.py"],
            phantom_files=["ghost.py"],
            overlap_ratio=0.33,
        )
        assert result.verified is False
        assert result.unclaimed_files == ["src/b.py"]
        assert result.phantom_files == ["ghost.py"]

    def test_overlap_ratio_roundtrip(self) -> None:
        result = DualVerificationResult(verified=True, overlap_ratio=0.756)
        d = result.to_dict()
        assert d["overlap_ratio"] == 0.756

    def test_worktree_merge_result_verification_default_none(self) -> None:
        result = WorktreeMergeResult(status="merged")
        assert result.verification is None

    def test_worktree_merge_result_to_dict_includes_verification(self) -> None:
        verification = DualVerificationResult(
            verified=True, overlap_ratio=1.0,
        )
        result = WorktreeMergeResult(status="merged", verification=verification)
        d = result.to_dict()
        assert "verification" in d
        assert d["verification"]["verified"] is True

    def test_worktree_merge_result_to_dict_omits_verification_when_none(self) -> None:
        result = WorktreeMergeResult(status="merged")
        d = result.to_dict()
        assert "verification" not in d

    def test_empty_lists(self) -> None:
        result = DualVerificationResult(
            verified=True,
            files_in_diff=[],
            files_claimed=[],
            unclaimed_files=[],
            phantom_files=[],
            overlap_ratio=0.0,
        )
        d = result.to_dict()
        assert d["files_in_diff"] == []
        assert d["files_claimed"] == []

    def test_overlap_ratio_bounds(self) -> None:
        r0 = DualVerificationResult(verified=False, overlap_ratio=0.0)
        r1 = DualVerificationResult(verified=True, overlap_ratio=1.0)
        assert r0.overlap_ratio == 0.0
        assert r1.overlap_ratio == 1.0


# ---------------------------------------------------------------------------
# OutputEnvelope dataclass tests
# ---------------------------------------------------------------------------


class TestOutputEnvelope:
    """Tests for the OutputEnvelope dataclass and its integration with TaskResult/TaskSpec."""

    def test_output_envelope_default_fields(self) -> None:
        from maestro_cli.models import OutputEnvelope
        env = OutputEnvelope(output_hash="abc1234567890def")
        assert env.output_hash == "abc1234567890def"
        assert env.scope_declared == []
        assert env.scope_violations == []
        assert env.scope_verified is True

    def test_output_envelope_to_dict(self) -> None:
        from maestro_cli.models import OutputEnvelope
        env = OutputEnvelope(
            output_hash="a1b2c3d4e5f6a7b8",
            scope_declared=["src/*.py"],
            scope_violations=[],
            scope_verified=True,
        )
        d = env.to_dict()
        assert d == {
            "output_hash": "a1b2c3d4e5f6a7b8",
            "scope_declared": ["src/*.py"],
            "scope_violations": [],
            "scope_verified": True,
        }

    def test_output_envelope_with_violations(self) -> None:
        from maestro_cli.models import OutputEnvelope
        env = OutputEnvelope(
            output_hash="deadbeef12345678",
            scope_declared=["src/*.py"],
            scope_violations=["lib/hack.js", "docs/secret.md"],
            scope_verified=False,
        )
        assert len(env.scope_violations) == 2
        assert "lib/hack.js" in env.scope_violations
        assert env.scope_verified is False

    def test_output_envelope_scope_verified_true_when_no_violations(self) -> None:
        from maestro_cli.models import OutputEnvelope
        env = OutputEnvelope(
            output_hash="1234567890abcdef",
            scope_declared=["src/**/*.py"],
            scope_violations=[],
            scope_verified=True,
        )
        assert env.scope_verified is True

    def test_output_envelope_scope_verified_false_when_violations_exist(self) -> None:
        from maestro_cli.models import OutputEnvelope
        env = OutputEnvelope(
            output_hash="1234567890abcdef",
            scope_declared=["src/**/*.py"],
            scope_violations=["config/settings.yaml"],
            scope_verified=False,
        )
        assert env.scope_verified is False

    def test_task_result_output_envelope_default_is_none(self) -> None:
        result = TaskResult(task_id="t1", status="success")
        assert result.output_envelope is None

    def test_task_result_to_dict_includes_envelope_when_set(self) -> None:
        from maestro_cli.models import OutputEnvelope
        env = OutputEnvelope(
            output_hash="aabbccdd11223344",
            scope_declared=["src/*.py"],
            scope_violations=[],
            scope_verified=True,
        )
        result = TaskResult(task_id="t1", status="success", output_envelope=env)
        d = result.to_dict()
        assert "output_envelope" in d
        assert d["output_envelope"]["output_hash"] == "aabbccdd11223344"
        assert d["output_envelope"]["scope_verified"] is True

    def test_task_result_to_dict_omits_envelope_when_none(self) -> None:
        result = TaskResult(task_id="t1", status="success")
        d = result.to_dict()
        assert "output_envelope" not in d

    def test_task_spec_output_scope_default_empty(self) -> None:
        spec = TaskSpec(id="t1", command="echo hi")
        assert spec.output_scope == []

    def test_task_spec_output_scope_populated(self) -> None:
        spec = TaskSpec(id="t1", command="echo hi", output_scope=["src/*.py", "tests/*.py"])
        assert spec.output_scope == ["src/*.py", "tests/*.py"]
