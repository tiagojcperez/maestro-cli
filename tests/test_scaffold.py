from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from maestro_cli.errors import PlanValidationError
from maestro_cli.models import PlanBrief, TaskBrief
from maestro_cli.scaffold import (
    WORKFLOW_LIBRARY_NAMES,
    _detect_large_files,
    _generate_branch_task,
    _generate_build_verify,
    _generate_quality_gates,
    _generate_split_tasks,
    _load_library,
    _merge_library_into_brief,
    _route_agent,
    _route_model,
    list_workflow_libraries,
    load_brief,
    scaffold_plan,
    validate_plan_cost_safety,
)


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------

class TestRouteModel:
    def test_implementation_routes_to_sonnet(self) -> None:
        engine, model, reasoning = _route_model("implementation")
        assert engine == "claude"
        assert model == "sonnet"
        assert reasoning is None

    def test_security_audit_routes_to_opus(self) -> None:
        engine, model, reasoning = _route_model("security-audit")
        assert engine == "claude"
        assert model == "opus"
        assert reasoning == "high"

    def test_shell_routes_to_none(self) -> None:
        engine, model, reasoning = _route_model("shell")
        assert engine is None
        assert model is None

    def test_trivial_fix_routes_to_haiku(self) -> None:
        engine, model, reasoning = _route_model("trivial-fix")
        assert engine == "claude"
        assert model == "haiku"

    def test_build_verify_routes_to_none(self) -> None:
        engine, model, reasoning = _route_model("build-verify")
        assert engine is None


class TestRouteAgent:
    def test_code_review_default_agent(self) -> None:
        assert _route_agent("code-review", None) == "code-reviewer"

    def test_explicit_agent_overrides(self) -> None:
        assert _route_agent("code-review", "my-reviewer") == "my-reviewer"

    def test_implementation_has_no_default_agent(self) -> None:
        assert _route_agent("implementation", None) is None


# ---------------------------------------------------------------------------
# Task generators
# ---------------------------------------------------------------------------

class TestGenerateBranchTask:
    def test_basic_branch_task(self) -> None:
        task = _generate_branch_task("feature/test", None)
        assert task["id"] == "w0-branch"
        assert "feature/test" in task["command"]
        assert "workdir" not in task

    def test_branch_task_with_workdir(self) -> None:
        task = _generate_branch_task("feature/x", "C:/project")
        assert task["workdir"] == "C:/project"


class TestGenerateQualityGates:
    def test_generates_review_and_qa(self) -> None:
        gates = _generate_quality_gates(["t1", "t2"], None, "test-plan", "test goal")
        assert len(gates) == 2
        ids = [g["id"] for g in gates]
        assert "code-review" in ids
        assert "qa-verification" in ids

    def test_depends_on_impl_tasks(self) -> None:
        gates = _generate_quality_gates(["t1", "t2"], None, "test-plan", "test goal")
        for gate in gates:
            assert gate["depends_on"] == ["t1", "t2"]

    def test_context_from_wildcard(self) -> None:
        gates = _generate_quality_gates(["t1"], None, "test-plan", "test goal")
        for gate in gates:
            assert gate["context_from"] == ["*"]

    def test_workdir_set_when_provided(self) -> None:
        gates = _generate_quality_gates(["t1"], "C:/proj", "test-plan", "test goal")
        for gate in gates:
            assert gate["workdir"] == "C:/proj"


class TestQualityGatePromptContent:
    def test_review_prompt_includes_plan_objective(self) -> None:
        gates = _generate_quality_gates(["t1", "t2"], None, "my-plan", "add auth")
        review = next(g for g in gates if g["id"] == "code-review")
        assert "my-plan" in review["prompt"]
        assert "add auth" in review["prompt"]
        assert "Plan Objective" in review["prompt"]

    def test_review_prompt_includes_simplicity_check(self) -> None:
        gates = _generate_quality_gates(["t1"], None, "p", "g")
        review = next(g for g in gates if g["id"] == "code-review")
        assert "simpler approach" in review["prompt"]
        assert "over-engineering" in review["prompt"]

    def test_qa_prompt_includes_plan_objective(self) -> None:
        gates = _generate_quality_gates(["t1"], None, "my-plan", "add auth")
        qa = next(g for g in gates if g["id"] == "qa-verification")
        assert "my-plan" in qa["prompt"]
        assert "add auth" in qa["prompt"]

    def test_plan_goal_fallback_when_empty(self) -> None:
        gates = _generate_quality_gates(["t1"], None, "p", "")
        review = next(g for g in gates if g["id"] == "code-review")
        assert "see task descriptions" in review["prompt"]


class TestGenerateBuildVerify:
    def test_basic_build_verify(self) -> None:
        task = _generate_build_verify(["review", "qa"], None)
        assert task["id"] == "build-verify"
        assert task["depends_on"] == ["review", "qa"]
        assert "command" in task

    def test_build_verify_with_workdir(self) -> None:
        task = _generate_build_verify(["t1"], "C:/proj")
        assert task["workdir"] == "C:/proj"

    def test_build_verify_includes_verify_command(self) -> None:
        task = _generate_build_verify(["t1"], None)
        assert "verify_command" in task
        assert "TODO" in task["verify_command"]


# ---------------------------------------------------------------------------
# scaffold_plan
# ---------------------------------------------------------------------------

class TestScaffoldPlan:
    def _basic_brief(self) -> PlanBrief:
        return PlanBrief(
            name="test-plan",
            goal="Test goal",
            workspace_root="C:/test/project",
            branch_name="feature/test",
            max_parallel=3,
            tasks=[
                TaskBrief(
                    id="db-migration",
                    description="Create migration",
                    task_type="implementation",
                    prompt_hint="Create tables...",
                ),
                TaskBrief(
                    id="api-endpoint",
                    description="Add REST endpoint",
                    task_type="implementation",
                    depends_on=["db-migration"],
                    prompt_hint="Add GET /api/thing",
                ),
            ],
        )

    def test_generates_valid_yaml(self) -> None:
        brief = self._basic_brief()
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        assert parsed["version"] == 1
        assert parsed["name"] == "test-plan"

    def test_includes_branch_task(self) -> None:
        brief = self._basic_brief()
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task_ids = [t["id"] for t in parsed["tasks"]]
        assert "w0-branch" in task_ids

    def test_no_branch_task_without_branch_name(self) -> None:
        brief = PlanBrief(
            name="no-branch",
            tasks=[TaskBrief(id="t1", task_type="implementation")],
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task_ids = [t["id"] for t in parsed["tasks"]]
        assert "w0-branch" not in task_ids

    def test_includes_quality_gates(self) -> None:
        brief = self._basic_brief()
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task_ids = [t["id"] for t in parsed["tasks"]]
        assert "code-review" in task_ids
        assert "qa-verification" in task_ids

    def test_excludes_quality_gates_when_disabled(self) -> None:
        brief = self._basic_brief()
        brief.include_quality_gates = False
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task_ids = [t["id"] for t in parsed["tasks"]]
        assert "code-review" not in task_ids
        assert "qa-verification" not in task_ids

    def test_includes_build_verify(self) -> None:
        brief = self._basic_brief()
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task_ids = [t["id"] for t in parsed["tasks"]]
        assert "build-verify" in task_ids

    def test_excludes_build_verify_when_disabled(self) -> None:
        brief = self._basic_brief()
        brief.include_build_verify = False
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task_ids = [t["id"] for t in parsed["tasks"]]
        assert "build-verify" not in task_ids

    def test_impl_tasks_depend_on_branch(self) -> None:
        brief = self._basic_brief()
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        db_task = next(t for t in parsed["tasks"] if t["id"] == "db-migration")
        assert "w0-branch" in db_task.get("depends_on", [])

    def test_explicit_deps_preserved(self) -> None:
        brief = self._basic_brief()
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        api_task = next(t for t in parsed["tasks"] if t["id"] == "api-endpoint")
        assert "db-migration" in api_task.get("depends_on", [])

    def test_security_audit_gets_opus_model(self) -> None:
        brief = PlanBrief(
            name="sec-test",
            tasks=[
                TaskBrief(id="audit", task_type="security-audit", description="Audit auth"),
            ],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        audit = next(t for t in parsed["tasks"] if t["id"] == "audit")
        assert audit.get("model") == "opus"
        assert audit.get("reasoning_effort") == "high"

    def test_prompt_hint_in_output(self) -> None:
        brief = self._basic_brief()
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        db_task = next(t for t in parsed["tasks"] if t["id"] == "db-migration")
        assert "Create tables..." in db_task["prompt"]

    def test_max_parallel_from_brief(self) -> None:
        brief = self._basic_brief()
        brief.max_parallel = 5
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        assert parsed["max_parallel"] == 5

    def test_workspace_root_in_plan(self) -> None:
        brief = self._basic_brief()
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        assert parsed["workspace_root"] == "C:/test/project"

    def test_impl_tasks_have_anti_stalling_prompt(self) -> None:
        brief = self._basic_brief()
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        impl_task = next(t for t in parsed["tasks"] if t["id"] == "db-migration")
        assert "append_system_prompt" in impl_task
        assert "5+ files" in impl_task["append_system_prompt"]

    def test_non_impl_tasks_no_anti_stalling(self) -> None:
        brief = self._basic_brief()
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        review = next(t for t in parsed["tasks"] if t["id"] == "code-review")
        qa = next(t for t in parsed["tasks"] if t["id"] == "qa-verification")
        assert "append_system_prompt" not in review
        assert "append_system_prompt" not in qa


# ---------------------------------------------------------------------------
# load_brief
# ---------------------------------------------------------------------------

class TestLoadBrief:
    def test_load_valid_brief(self, sample_brief_yaml: Path) -> None:
        brief = load_brief(sample_brief_yaml)
        assert brief.name == "test-feature"
        assert brief.goal == "Add a new feature"
        assert brief.workspace_root == "C:/test/project"
        assert brief.branch_name == "feature/test"
        assert len(brief.tasks) == 3

    def test_task_types_parsed(self, sample_brief_yaml: Path) -> None:
        brief = load_brief(sample_brief_yaml)
        assert brief.tasks[0].task_type == "implementation"
        assert brief.tasks[2].task_type == "security-audit"

    def test_depends_on_parsed(self, sample_brief_yaml: Path) -> None:
        brief = load_brief(sample_brief_yaml)
        assert brief.tasks[1].depends_on == ["db-migration"]
        assert brief.tasks[2].depends_on == ["api-endpoint"]

    def test_prompt_hint_parsed(self, sample_brief_yaml: Path) -> None:
        brief = load_brief(sample_brief_yaml)
        assert "Create tables" in brief.tasks[0].prompt_hint

    def test_missing_file_raises(self) -> None:
        with pytest.raises(PlanValidationError, match="Brief file not found"):
            load_brief("/nonexistent/file.yaml")

    def test_missing_name_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("goal: test\ntasks: []\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must have a 'name'"):
            load_brief(f)

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("{{invalid yaml", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="Invalid brief YAML"):
            load_brief(f)

    def test_non_dict_root_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="root must be an object"):
            load_brief(f)

    def test_tasks_not_list_raises(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.yaml"
        f.write_text("name: test\ntasks: not-a-list\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="tasks must be a list"):
            load_brief(f)

    def test_task_without_id_raises(self, tmp_path: Path) -> None:
        content = "name: test\ntasks:\n  - description: no id\n"
        f = tmp_path / "bad.yaml"
        f.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="id is required"):
            load_brief(f)

    def test_defaults_applied(self, tmp_path: Path) -> None:
        content = "name: minimal\ntasks:\n  - id: t1\n"
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.max_parallel == 3
        assert brief.fail_fast is True
        assert brief.include_quality_gates is True
        assert brief.topology == "pipeline"

    def test_depends_on_as_string(self, tmp_path: Path) -> None:
        content = "name: test\ntasks:\n  - id: t1\n  - id: t2\n    depends_on: t1\n"
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.tasks[1].depends_on == ["t1"]


# ---------------------------------------------------------------------------
# validate_plan_cost_safety
# ---------------------------------------------------------------------------

class TestValidatePlanCostSafety:
    def test_no_warnings_for_good_plan(self) -> None:
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "good-plan",
            "tasks": [
                {"id": "impl", "engine": "claude", "prompt": "do stuff"},
                {"id": "code-review", "engine": "claude", "agent": "code-reviewer", "prompt": "review"},
                {"id": "qa-verification", "engine": "claude", "agent": "qa-engineer", "prompt": "test"},
                {"id": "build-verify", "command": "npm run build"},
            ],
        })
        warnings = validate_plan_cost_safety(plan_yaml)
        assert warnings == []

    def test_warns_too_many_opus(self) -> None:
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "expensive",
            "tasks": [
                {"id": "t1", "engine": "claude", "model": "opus", "prompt": "a"},
                {"id": "t2", "engine": "claude", "model": "opus", "prompt": "b"},
                {"id": "t3", "engine": "claude", "model": "opus", "prompt": "c"},
                {"id": "code-review", "engine": "claude", "agent": "code-reviewer", "prompt": "r"},
                {"id": "qa-verification", "engine": "claude", "agent": "qa-engineer", "prompt": "q"},
                {"id": "build-verify", "command": "build"},
            ],
        })
        warnings = validate_plan_cost_safety(plan_yaml)
        assert any("opus" in w.lower() or "Opus" in w for w in warnings)

    def test_warns_missing_review(self) -> None:
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "no-review",
            "tasks": [
                {"id": "impl", "engine": "claude", "prompt": "do stuff"},
                {"id": "qa-check", "engine": "claude", "agent": "qa-engineer", "prompt": "test"},
                {"id": "build-verify", "command": "build"},
            ],
        })
        warnings = validate_plan_cost_safety(plan_yaml)
        assert any("review" in w.lower() for w in warnings)

    def test_warns_missing_qa(self) -> None:
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "no-qa",
            "tasks": [
                {"id": "impl", "engine": "claude", "prompt": "do stuff"},
                {"id": "code-review", "engine": "claude", "agent": "code-reviewer", "prompt": "rev"},
                {"id": "build-verify", "command": "build"},
            ],
        })
        warnings = validate_plan_cost_safety(plan_yaml)
        assert any("qa" in w.lower() for w in warnings)

    def test_warns_missing_build(self) -> None:
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "no-build",
            "tasks": [
                {"id": "impl", "engine": "claude", "prompt": "do stuff"},
                {"id": "code-review", "engine": "claude", "agent": "code-reviewer", "prompt": "rev"},
                {"id": "qa-verification", "engine": "claude", "agent": "qa-engineer", "prompt": "q"},
            ],
        })
        warnings = validate_plan_cost_safety(plan_yaml)
        assert any("build" in w.lower() for w in warnings)

    def test_warns_reasoning_effort_on_non_opus(self) -> None:
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "bad-effort",
            "tasks": [
                {
                    "id": "impl",
                    "engine": "claude",
                    "model": "sonnet",
                    "reasoning_effort": "high",
                    "prompt": "do stuff",
                },
                {"id": "code-review", "engine": "claude", "agent": "code-reviewer", "prompt": "r"},
                {"id": "qa-verification", "engine": "claude", "agent": "qa-engineer", "prompt": "q"},
                {"id": "build-verify", "command": "build"},
            ],
        })
        warnings = validate_plan_cost_safety(plan_yaml)
        assert any("reasoning_effort" in w for w in warnings)

    def test_invalid_plan_root(self) -> None:
        warnings = validate_plan_cost_safety("- not an object")
        assert any("not a valid object" in w for w in warnings)


# ---------------------------------------------------------------------------
# T0.4 — Auto-split heuristic
# ---------------------------------------------------------------------------

class TestDetectLargeFiles:
    def test_no_file_paths_in_prompt(self, tmp_path: Path) -> None:
        result = _detect_large_files("no paths here just plain text", str(tmp_path))
        assert result == []

    def test_file_mentioned_but_not_found(self, tmp_path: Path) -> None:
        result = _detect_large_files("edit src/main.py to add logging", str(tmp_path))
        assert result == []

    def test_small_file_not_returned(self, tmp_path: Path) -> None:
        f = tmp_path / "small.py"
        f.write_text("\n".join(f"# line {i}" for i in range(50)), encoding="utf-8")
        result = _detect_large_files("edit small.py", str(tmp_path))
        assert result == []

    def test_large_file_returned(self, tmp_path: Path) -> None:
        f = tmp_path / "large.py"
        f.write_text("\n".join(f"# line {i}" for i in range(350)), encoding="utf-8")
        result = _detect_large_files("edit large.py to add logging", str(tmp_path))
        assert "large.py" in result

    def test_threshold_exact_boundary(self, tmp_path: Path) -> None:
        """Exactly 300 lines meets the threshold."""
        f = tmp_path / "exact.py"
        f.write_text("\n".join(f"# line {i}" for i in range(300)), encoding="utf-8")
        result = _detect_large_files("edit exact.py", str(tmp_path))
        assert "exact.py" in result

    def test_file_in_subdirectory(self, tmp_path: Path) -> None:
        subdir = tmp_path / "src"
        subdir.mkdir()
        f = subdir / "runners.py"
        f.write_text("\n".join(f"# line {i}" for i in range(400)), encoding="utf-8")
        result = _detect_large_files("edit src/runners.py to fix timeout", str(tmp_path))
        assert any("runners.py" in r for r in result)

    def test_multiple_large_files(self, tmp_path: Path) -> None:
        for name in ("alpha.py", "beta.py"):
            (tmp_path / name).write_text("\n".join(f"# {i}" for i in range(350)), encoding="utf-8")
        result = _detect_large_files("update alpha.py and beta.py", str(tmp_path))
        assert "alpha.py" in result
        assert "beta.py" in result

    def test_dedup_same_path_mentioned_twice(self, tmp_path: Path) -> None:
        f = tmp_path / "models.py"
        f.write_text("\n".join(f"# {i}" for i in range(350)), encoding="utf-8")
        result = _detect_large_files("edit models.py and models.py again", str(tmp_path))
        assert result.count("models.py") == 1

    def test_custom_threshold(self, tmp_path: Path) -> None:
        f = tmp_path / "medium.py"
        f.write_text("\n".join(f"# {i}" for i in range(100)), encoding="utf-8")
        assert _detect_large_files("edit medium.py", str(tmp_path), threshold=50) == ["medium.py"]
        assert _detect_large_files("edit medium.py", str(tmp_path), threshold=200) == []


class TestGenerateSplitTasks:
    def _tb(self, **kw: object) -> "TaskBrief":
        from maestro_cli.models import TaskBrief
        return TaskBrief(id="update-models", task_type="implementation", **kw)  # type: ignore[arg-type]

    def test_returns_two_tasks_and_final_id(self) -> None:
        tb = self._tb(prompt_hint="Update models.py")
        tasks, final_id = _generate_split_tasks(tb, ["models.py"], [], None)
        assert len(tasks) == 2
        assert final_id == "update-models-apply"

    def test_read_plan_id_and_apply_id(self) -> None:
        tb = self._tb(prompt_hint="Update models.py")
        tasks, _ = _generate_split_tasks(tb, ["models.py"], [], None)
        assert tasks[0]["id"] == "update-models-read-plan"
        assert tasks[1]["id"] == "update-models-apply"

    def test_read_plan_uses_haiku(self) -> None:
        tb = self._tb(prompt_hint="Update models.py")
        tasks, _ = _generate_split_tasks(tb, ["models.py"], [], None)
        assert tasks[0]["model"] == "haiku"
        assert tasks[0]["engine"] == "claude"

    def test_apply_has_no_explicit_model(self) -> None:
        """Apply task inherits default (sonnet) — no model key set."""
        tb = self._tb(prompt_hint="Update models.py")
        tasks, _ = _generate_split_tasks(tb, ["models.py"], [], None)
        assert "model" not in tasks[1]

    def test_apply_depends_on_read_plan(self) -> None:
        tb = self._tb(prompt_hint="Update x.py")
        tasks, _ = _generate_split_tasks(tb, ["x.py"], [], None)
        assert "update-models-read-plan" in tasks[1]["depends_on"]

    def test_original_deps_go_to_read_plan(self) -> None:
        tb = self._tb(prompt_hint="Update x.py")
        tasks, _ = _generate_split_tasks(tb, ["x.py"], ["w0-branch"], None)
        assert "w0-branch" in tasks[0]["depends_on"]

    def test_no_deps_when_empty(self) -> None:
        tb = self._tb(prompt_hint="Update x.py")
        tasks, _ = _generate_split_tasks(tb, ["x.py"], [], None)
        assert "depends_on" not in tasks[0]

    def test_workdir_propagated_to_both_tasks(self) -> None:
        tb = self._tb(prompt_hint="Update x.py")
        tasks, _ = _generate_split_tasks(tb, ["x.py"], [], "C:/project")
        assert tasks[0]["workdir"] == "C:/project"
        assert tasks[1]["workdir"] == "C:/project"

    def test_no_workdir_when_none(self) -> None:
        tb = self._tb(prompt_hint="Update x.py")
        tasks, _ = _generate_split_tasks(tb, ["x.py"], [], None)
        assert "workdir" not in tasks[0]
        assert "workdir" not in tasks[1]

    def test_read_plan_prompt_contains_file_list(self) -> None:
        tb = self._tb(prompt_hint="Update models.py")
        tasks, _ = _generate_split_tasks(tb, ["models.py", "loader.py"], [], None)
        assert "models.py" in tasks[0]["prompt"]
        assert "loader.py" in tasks[0]["prompt"]

    def test_apply_prompt_contains_context_variable(self) -> None:
        tb = self._tb(prompt_hint="Update models.py")
        tasks, _ = _generate_split_tasks(tb, ["models.py"], [], None)
        assert "{{ update-models-read-plan.stdout_tail }}" in tasks[1]["prompt"]

    def test_apply_has_anti_stalling_prompt(self) -> None:
        tb = self._tb(prompt_hint="Update models.py")
        tasks, _ = _generate_split_tasks(tb, ["models.py"], [], None)
        assert "append_system_prompt" in tasks[1]


class TestScaffoldStrictDefaults:
    """`--strict-defaults` injects sane first-run config (internal post-mortem)."""

    def _brief(self, *, strict_defaults: bool = False) -> PlanBrief:
        return PlanBrief(
            name="test-plan",
            tasks=[
                TaskBrief(
                    id="impl",
                    task_type="implementation",
                    prompt_hint="Do the thing",
                ),
            ],
            strict_defaults=strict_defaults,
        )

    def test_off_keeps_legacy_defaults(self) -> None:
        result = yaml.safe_load(scaffold_plan(self._brief()))
        assert result["defaults"]["timeout_sec"] == 600
        assert "retry_delay_sec" not in result["defaults"]
        assert "max_cost_usd" not in result
        assert "budget_warning_pct" not in result

    def test_on_lifts_timeout_above_w20_threshold(self) -> None:
        result = yaml.safe_load(scaffold_plan(self._brief(strict_defaults=True)))
        # Must clear W20's 900s tight-timeout threshold.
        assert result["defaults"]["timeout_sec"] >= 900
        assert result["defaults"]["timeout_sec"] == 1500

    def test_on_adds_progressive_retry_delay(self) -> None:
        result = yaml.safe_load(scaffold_plan(self._brief(strict_defaults=True)))
        delay = result["defaults"]["retry_delay_sec"]
        assert isinstance(delay, list)
        assert len(delay) >= 2
        assert all(d > 0 for d in delay)

    def test_on_adds_budget_caps(self) -> None:
        result = yaml.safe_load(scaffold_plan(self._brief(strict_defaults=True)))
        assert result["max_cost_usd"] == 10.0
        assert result["budget_warning_pct"] == 0.8

    def test_strict_defaults_silences_w20(self, tmp_path: Path) -> None:
        # End-to-end: a strict-defaults scaffold + an author who opts into
        # retries should NOT trigger W20 because the plan default
        # retry_delay_sec acts as an escape valve.
        from maestro_cli.loader import load_plan

        yaml_text = scaffold_plan(self._brief(strict_defaults=True))
        # Inject max_retries on the impl task to simulate an author opting in.
        plan_dict = yaml.safe_load(yaml_text)
        for task in plan_dict["tasks"]:
            if task["id"] == "impl":
                task["max_retries"] = 1
                break
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml.safe_dump(plan_dict), encoding="utf-8")
        plan = load_plan(plan_path)
        assert not any("W20" in w for w in plan.validation_warnings)


class TestScaffoldAutoSplit:
    def _make_large_file(self, tmp_path: Path, name: str, lines: int = 400) -> Path:
        f = tmp_path / name
        f.write_text("\n".join(f"# line {i}" for i in range(lines)), encoding="utf-8")
        return f

    def test_splits_impl_task_with_large_file(self, tmp_path: Path) -> None:
        self._make_large_file(tmp_path, "models.py")
        brief = PlanBrief(
            name="test-split",
            workspace_root=str(tmp_path),
            include_quality_gates=False,
            include_build_verify=False,
            tasks=[TaskBrief(id="update-models", task_type="implementation",
                             prompt_hint="Update models.py to add new fields")],
        )
        parsed = yaml.safe_load(scaffold_plan(brief))
        task_ids = [t["id"] for t in parsed["tasks"]]
        assert "update-models-read-plan" in task_ids
        assert "update-models-apply" in task_ids
        assert "update-models" not in task_ids

    def test_no_split_for_small_file(self, tmp_path: Path) -> None:
        f = tmp_path / "models.py"
        f.write_text("\n".join(f"# {i}" for i in range(50)), encoding="utf-8")
        brief = PlanBrief(
            name="test-no-split",
            workspace_root=str(tmp_path),
            include_quality_gates=False,
            include_build_verify=False,
            tasks=[TaskBrief(id="update-models", task_type="implementation",
                             prompt_hint="Update models.py to add new fields")],
        )
        parsed = yaml.safe_load(scaffold_plan(brief))
        task_ids = [t["id"] for t in parsed["tasks"]]
        assert "update-models" in task_ids
        assert "update-models-read-plan" not in task_ids

    def test_no_split_without_workspace_root(self) -> None:
        brief = PlanBrief(
            name="test-no-workspace",
            workspace_root=None,
            include_quality_gates=False,
            include_build_verify=False,
            tasks=[TaskBrief(id="update-models", task_type="implementation",
                             prompt_hint="Update models.py")],
        )
        parsed = yaml.safe_load(scaffold_plan(brief))
        task_ids = [t["id"] for t in parsed["tasks"]]
        assert "update-models" in task_ids
        assert "update-models-read-plan" not in task_ids

    def test_no_split_when_auto_split_false(self, tmp_path: Path) -> None:
        self._make_large_file(tmp_path, "models.py")
        brief = PlanBrief(
            name="test-opt-out",
            workspace_root=str(tmp_path),
            include_quality_gates=False,
            include_build_verify=False,
            tasks=[TaskBrief(id="update-models", task_type="implementation",
                             prompt_hint="Update models.py", auto_split=False)],
        )
        parsed = yaml.safe_load(scaffold_plan(brief))
        task_ids = [t["id"] for t in parsed["tasks"]]
        assert "update-models" in task_ids
        assert "update-models-read-plan" not in task_ids

    def test_no_split_for_trivial_fix(self, tmp_path: Path) -> None:
        self._make_large_file(tmp_path, "models.py")
        brief = PlanBrief(
            name="test-trivial",
            workspace_root=str(tmp_path),
            include_quality_gates=False,
            include_build_verify=False,
            tasks=[TaskBrief(id="fix-typo", task_type="trivial-fix",
                             prompt_hint="Fix typo in models.py")],
        )
        parsed = yaml.safe_load(scaffold_plan(brief))
        task_ids = [t["id"] for t in parsed["tasks"]]
        assert "fix-typo" in task_ids
        assert "fix-typo-read-plan" not in task_ids

    def test_quality_gates_depend_on_apply_task(self, tmp_path: Path) -> None:
        self._make_large_file(tmp_path, "models.py")
        brief = PlanBrief(
            name="test-qg",
            workspace_root=str(tmp_path),
            include_quality_gates=True,
            include_build_verify=False,
            tasks=[TaskBrief(id="update-models", task_type="implementation",
                             prompt_hint="Update models.py")],
        )
        parsed = yaml.safe_load(scaffold_plan(brief))
        review = next(t for t in parsed["tasks"] if t["id"] == "code-review")
        assert "update-models-apply" in review["depends_on"]
        assert "update-models" not in review["depends_on"]

    def test_load_brief_parses_auto_split_false(self, tmp_path: Path) -> None:
        content = (
            "name: test\n"
            "tasks:\n"
            "  - id: t1\n"
            "    auto_split: false\n"
        )
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.tasks[0].auto_split is False

    def test_load_brief_auto_split_defaults_true(self, tmp_path: Path) -> None:
        content = "name: test\ntasks:\n  - id: t1\n"
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.tasks[0].auto_split is True


# ---------------------------------------------------------------------------
# Additional load_brief tests — edge cases
# ---------------------------------------------------------------------------

class TestLoadBriefEdgeCases:
    def test_invalid_task_type_raises(self, tmp_path: Path) -> None:
        content = "name: test\ntasks:\n  - id: t1\n    task_type: nonexistent\n"
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="is not valid"):
            load_brief(f)

    def test_invalid_topology_raises(self, tmp_path: Path) -> None:
        content = "name: test\ntopology: star\ntasks:\n  - id: t1\n"
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="topology.*is not valid"):
            load_brief(f)

    def test_task_item_not_dict_raises(self, tmp_path: Path) -> None:
        content = "name: test\ntasks:\n  - just-a-string\n"
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="tasks\\[0\\] must be an object"):
            load_brief(f)

    def test_empty_task_id_raises(self, tmp_path: Path) -> None:
        content = 'name: test\ntasks:\n  - id: ""\n'
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="id is required"):
            load_brief(f)

    def test_extra_fields_ignored(self, tmp_path: Path) -> None:
        content = (
            "name: test\n"
            "extra_field: foo\n"
            "another_unknown: 42\n"
            "tasks:\n  - id: t1\n"
        )
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.name == "test"
        assert len(brief.tasks) == 1

    def test_empty_tasks_list(self, tmp_path: Path) -> None:
        content = "name: test\ntasks: []\n"
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.tasks == []

    def test_whitespace_only_name_raises(self, tmp_path: Path) -> None:
        content = 'name: "   "\ntasks:\n  - id: t1\n'
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must have a 'name'"):
            load_brief(f)

    def test_depends_on_as_list_of_strings(self, tmp_path: Path) -> None:
        content = (
            "name: test\n"
            "tasks:\n"
            "  - id: t1\n"
            "  - id: t2\n"
            "    depends_on: [t1]\n"
        )
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.tasks[1].depends_on == ["t1"]

    def test_explicit_engine_override_in_brief(self, tmp_path: Path) -> None:
        content = (
            "name: test\n"
            "tasks:\n"
            "  - id: t1\n"
            "    engine: gemini\n"
        )
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.tasks[0].engine == "gemini"

    def test_explicit_agent_in_brief(self, tmp_path: Path) -> None:
        content = (
            "name: test\n"
            "tasks:\n"
            "  - id: t1\n"
            "    agent: custom-agent\n"
        )
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.tasks[0].agent == "custom-agent"

    def test_explicit_workdir_in_brief(self, tmp_path: Path) -> None:
        content = (
            "name: test\n"
            "tasks:\n"
            "  - id: t1\n"
            "    workdir: C:/some/dir\n"
        )
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.tasks[0].workdir == "C:/some/dir"

    def test_goal_and_workspace_root_parsed(self, tmp_path: Path) -> None:
        content = (
            "name: test\n"
            "goal: Build the feature\n"
            "workspace_root: C:/project\n"
            "tasks:\n  - id: t1\n"
        )
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.goal == "Build the feature"
        assert brief.workspace_root == "C:/project"

    def test_all_valid_task_types(self, tmp_path: Path) -> None:
        """Every valid task type should be accepted."""
        from maestro_cli.scaffold import _VALID_TASK_TYPES

        for i, tt in enumerate(sorted(_VALID_TASK_TYPES)):
            content = f"name: test\ntasks:\n  - id: t{i}\n    task_type: {tt}\n"
            f = tmp_path / f"brief_{i}.yaml"
            f.write_text(content, encoding="utf-8")
            brief = load_brief(f)
            assert brief.tasks[0].task_type == tt

    def test_all_valid_topologies(self, tmp_path: Path) -> None:
        from maestro_cli.scaffold import _VALID_TOPOLOGIES

        for topo in sorted(_VALID_TOPOLOGIES):
            content = f"name: test\ntopology: {topo}\ntasks:\n  - id: t1\n"
            f = tmp_path / f"brief_{topo}.yaml"
            f.write_text(content, encoding="utf-8")
            brief = load_brief(f)
            assert brief.topology == topo

    def test_max_parallel_override(self, tmp_path: Path) -> None:
        content = "name: test\nmax_parallel: 8\ntasks:\n  - id: t1\n"
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.max_parallel == 8

    def test_fail_fast_false(self, tmp_path: Path) -> None:
        content = "name: test\nfail_fast: false\ntasks:\n  - id: t1\n"
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.fail_fast is False

    def test_include_quality_gates_false(self, tmp_path: Path) -> None:
        content = "name: test\ninclude_quality_gates: false\ntasks:\n  - id: t1\n"
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.include_quality_gates is False

    def test_include_build_verify_false(self, tmp_path: Path) -> None:
        content = "name: test\ninclude_build_verify: false\ntasks:\n  - id: t1\n"
        f = tmp_path / "brief.yaml"
        f.write_text(content, encoding="utf-8")
        brief = load_brief(f)
        assert brief.include_build_verify is False


# ---------------------------------------------------------------------------
# Additional scaffold_plan tests
# ---------------------------------------------------------------------------

class TestScaffoldPlanAdditional:
    def test_single_task_plan(self) -> None:
        brief = PlanBrief(
            name="single",
            tasks=[TaskBrief(id="only-task", task_type="implementation")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        assert parsed["version"] == 1
        assert parsed["name"] == "single"
        assert len(parsed["tasks"]) == 1
        assert parsed["tasks"][0]["id"] == "only-task"

    def test_shell_task_generates_command_not_engine(self) -> None:
        brief = PlanBrief(
            name="shell-test",
            tasks=[TaskBrief(id="run-cmd", task_type="shell")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task = parsed["tasks"][0]
        assert "command" in task
        assert "engine" not in task

    def test_code_review_task_type_gets_agent(self) -> None:
        brief = PlanBrief(
            name="review-test",
            tasks=[TaskBrief(id="review", task_type="code-review")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task = parsed["tasks"][0]
        assert task.get("agent") == "code-reviewer"

    def test_qa_task_type_gets_agent(self) -> None:
        brief = PlanBrief(
            name="qa-test",
            tasks=[TaskBrief(id="qa", task_type="qa-verification")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task = parsed["tasks"][0]
        assert task.get("agent") == "qa-engineer"

    def test_explicit_engine_override_in_scaffold(self) -> None:
        brief = PlanBrief(
            name="engine-override",
            tasks=[TaskBrief(id="t1", task_type="implementation", engine="codex")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task = parsed["tasks"][0]
        assert task["engine"] == "codex"

    def test_no_workspace_root_omitted_from_plan(self) -> None:
        brief = PlanBrief(
            name="no-ws",
            workspace_root=None,
            tasks=[TaskBrief(id="t1", task_type="implementation")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        assert "workspace_root" not in parsed

    def test_no_workdir_when_no_workspace_root_and_no_task_workdir(self) -> None:
        brief = PlanBrief(
            name="no-wd",
            workspace_root=None,
            tasks=[TaskBrief(id="t1", task_type="implementation")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        assert "workdir" not in parsed["tasks"][0]

    def test_task_workdir_overrides_workspace_root(self) -> None:
        brief = PlanBrief(
            name="wd-override",
            workspace_root="C:/default",
            tasks=[TaskBrief(id="t1", task_type="implementation", workdir="C:/override")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        assert parsed["tasks"][0]["workdir"] == "C:/override"

    def test_build_verify_depends_on_quality_gates_when_present(self) -> None:
        brief = PlanBrief(
            name="full-plan",
            tasks=[TaskBrief(id="impl", task_type="implementation")],
            include_quality_gates=True,
            include_build_verify=True,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        build_task = next(t for t in parsed["tasks"] if t["id"] == "build-verify")
        assert "code-review" in build_task["depends_on"]
        assert "qa-verification" in build_task["depends_on"]

    def test_build_verify_depends_on_impl_when_no_quality_gates(self) -> None:
        brief = PlanBrief(
            name="no-gates",
            tasks=[TaskBrief(id="impl", task_type="implementation")],
            include_quality_gates=False,
            include_build_verify=True,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        build_task = next(t for t in parsed["tasks"] if t["id"] == "build-verify")
        assert "impl" in build_task["depends_on"]

    def test_build_verify_depends_on_last_task_when_no_impl(self) -> None:
        brief = PlanBrief(
            name="no-impl",
            tasks=[TaskBrief(id="setup-env", task_type="shell")],
            include_quality_gates=False,
            include_build_verify=True,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        build_task = next(t for t in parsed["tasks"] if t["id"] == "build-verify")
        assert "setup-env" in build_task["depends_on"]

    def test_many_tasks_plan(self) -> None:
        tasks = [
            TaskBrief(id=f"task-{i}", task_type="implementation", description=f"Task {i}")
            for i in range(10)
        ]
        brief = PlanBrief(
            name="big-plan",
            tasks=tasks,
            include_quality_gates=True,
            include_build_verify=True,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task_ids = [t["id"] for t in parsed["tasks"]]
        # 10 impl tasks + code-review + qa-verification + build-verify = 13
        assert len(task_ids) == 13
        assert "code-review" in task_ids
        assert "qa-verification" in task_ids
        assert "build-verify" in task_ids

    def test_no_prompt_hint_generates_todo_prompt(self) -> None:
        brief = PlanBrief(
            name="no-hint",
            tasks=[TaskBrief(id="task-x", task_type="implementation", description="Do stuff")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task = parsed["tasks"][0]
        assert "TODO" in task["prompt"]
        assert "task-x" in task["prompt"]

    def test_trivial_fix_uses_haiku(self) -> None:
        brief = PlanBrief(
            name="trivial",
            tasks=[TaskBrief(id="fix", task_type="trivial-fix")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task = parsed["tasks"][0]
        assert task.get("model") == "haiku"

    def test_trivial_fix_gets_anti_stalling_prompt(self) -> None:
        brief = PlanBrief(
            name="trivial",
            tasks=[TaskBrief(id="fix", task_type="trivial-fix")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        task = parsed["tasks"][0]
        assert "append_system_prompt" in task

    def test_review_context_mode_map_reduce_for_three_plus_upstreams(self) -> None:
        """When 3+ impl tasks exist, code-review uses map_reduce context mode."""
        tasks = [
            TaskBrief(id=f"impl-{i}", task_type="implementation")
            for i in range(3)
        ]
        brief = PlanBrief(
            name="mr-test",
            tasks=tasks,
            include_quality_gates=True,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        review = next(t for t in parsed["tasks"] if t["id"] == "code-review")
        assert review["context_mode"] == "map_reduce"

    def test_review_context_mode_summarized_for_fewer_upstreams(self) -> None:
        """When <3 impl tasks exist, code-review uses summarized context mode."""
        brief = PlanBrief(
            name="summ-test",
            tasks=[
                TaskBrief(id="impl-0", task_type="implementation"),
                TaskBrief(id="impl-1", task_type="implementation"),
            ],
            include_quality_gates=True,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        review = next(t for t in parsed["tasks"] if t["id"] == "code-review")
        assert review["context_mode"] == "summarized"

    def test_fail_fast_from_brief(self) -> None:
        brief = PlanBrief(
            name="ff-test",
            fail_fast=False,
            tasks=[TaskBrief(id="t1", task_type="shell")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        assert parsed["fail_fast"] is False

    def test_defaults_section_structure(self) -> None:
        brief = PlanBrief(
            name="defaults-test",
            tasks=[TaskBrief(id="t1", task_type="implementation")],
            include_quality_gates=False,
            include_build_verify=False,
        )
        result = scaffold_plan(brief)
        parsed = yaml.safe_load(result)
        defaults = parsed["defaults"]
        assert "env" in defaults
        assert defaults["env"]["PYTHONUTF8"] == "1"
        assert defaults["timeout_sec"] == 600
        assert "claude" in defaults
        assert defaults["claude"]["model"] == "sonnet"


# ---------------------------------------------------------------------------
# Additional validate_plan_cost_safety tests
# ---------------------------------------------------------------------------

class TestValidatePlanCostSafetyAdditional:
    def test_two_opus_tasks_no_cost_warning(self) -> None:
        """Exactly 2 opus tasks should not trigger the opus cost warning."""
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "two-opus",
            "tasks": [
                {"id": "t1", "engine": "claude", "model": "opus", "prompt": "a"},
                {"id": "t2", "engine": "claude", "model": "opus", "prompt": "b"},
                {"id": "code-review", "engine": "claude", "agent": "code-reviewer", "prompt": "r"},
                {"id": "qa-check", "engine": "claude", "agent": "qa-engineer", "prompt": "q"},
                {"id": "build-verify", "command": "build"},
            ],
        })
        warnings = validate_plan_cost_safety(plan_yaml)
        assert not any("opus" in w.lower() for w in warnings)

    def test_no_tasks_list_returns_warning(self) -> None:
        plan_yaml = yaml.dump({"version": 1, "name": "bad", "tasks": "not-a-list"})
        warnings = validate_plan_cost_safety(plan_yaml)
        assert any("no valid tasks list" in w for w in warnings)

    def test_non_dict_task_items_skipped(self) -> None:
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "mixed",
            "tasks": [
                "not-a-dict",
                {"id": "code-review", "engine": "claude", "agent": "code-reviewer", "prompt": "r"},
                {"id": "qa-check", "engine": "claude", "agent": "qa-engineer", "prompt": "q"},
                {"id": "build-verify", "command": "build"},
            ],
        })
        # Should not crash, just skip the non-dict item
        warnings = validate_plan_cost_safety(plan_yaml)
        assert isinstance(warnings, list)

    def test_reasoning_effort_on_opus_no_warning(self) -> None:
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "opus-effort",
            "tasks": [
                {"id": "audit", "engine": "claude", "model": "opus",
                 "reasoning_effort": "high", "prompt": "audit"},
                {"id": "code-review", "engine": "claude", "agent": "code-reviewer", "prompt": "r"},
                {"id": "qa-check", "engine": "claude", "agent": "qa-engineer", "prompt": "q"},
                {"id": "build-verify", "command": "build"},
            ],
        })
        warnings = validate_plan_cost_safety(plan_yaml)
        assert not any("reasoning_effort" in w for w in warnings)

    def test_reasoning_effort_on_haiku_warns(self) -> None:
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "haiku-effort",
            "tasks": [
                {"id": "t1", "engine": "claude", "model": "haiku",
                 "reasoning_effort": "high", "prompt": "x"},
                {"id": "code-review", "engine": "claude", "agent": "code-reviewer", "prompt": "r"},
                {"id": "qa-check", "engine": "claude", "agent": "qa-engineer", "prompt": "q"},
                {"id": "build-verify", "command": "build"},
            ],
        })
        warnings = validate_plan_cost_safety(plan_yaml)
        assert any("reasoning_effort" in w and "haiku" in w for w in warnings)

    def test_non_engine_task_no_reasoning_warning(self) -> None:
        """A non-claude engine task with reasoning_effort should not trigger the warning."""
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "codex-effort",
            "tasks": [
                {"id": "t1", "engine": "codex", "model": "5.4",
                 "reasoning_effort": "high", "prompt": "x"},
                {"id": "code-review", "engine": "claude", "agent": "code-reviewer", "prompt": "r"},
                {"id": "qa-check", "engine": "claude", "agent": "qa-engineer", "prompt": "q"},
                {"id": "build-verify", "command": "build"},
            ],
        })
        warnings = validate_plan_cost_safety(plan_yaml)
        assert not any("reasoning_effort" in w for w in warnings)

    def test_task_with_review_in_id_counts_as_review(self) -> None:
        plan_yaml = yaml.dump({
            "version": 1,
            "name": "id-check",
            "tasks": [
                {"id": "my-review-task", "engine": "claude", "prompt": "r"},
                {"id": "qa-check", "engine": "claude", "agent": "qa-engineer", "prompt": "q"},
                {"id": "build-verify", "command": "build"},
            ],
        })
        warnings = validate_plan_cost_safety(plan_yaml)
        assert not any("review" in w.lower() and "consider adding" in w.lower() for w in warnings)

    def test_empty_yaml_string(self) -> None:
        warnings = validate_plan_cost_safety("")
        assert any("not a valid object" in w for w in warnings)


# ---------------------------------------------------------------------------
# Workflow Libraries
# ---------------------------------------------------------------------------

class TestWorkflowLibraries:
    """Tests for built-in workflow library catalog and merge logic."""

    # -- list_workflow_libraries --

    def test_list_returns_all_builtin_libs(self) -> None:
        libs = list_workflow_libraries()
        assert len(libs) == 5
        names = {lib["name"] for lib in libs}
        assert names == {"rest-api", "refactor", "security-review", "bug-fix", "test-backfill"}

    def test_list_returns_sorted_by_name(self) -> None:
        libs = list_workflow_libraries()
        names = [lib["name"] for lib in libs]
        assert names == sorted(names)

    def test_list_entries_have_name_and_description(self) -> None:
        for lib in list_workflow_libraries():
            assert "name" in lib
            assert "description" in lib
            assert isinstance(lib["description"], str)
            assert len(lib["description"]) > 0

    # -- WORKFLOW_LIBRARY_NAMES constant --

    def test_workflow_library_names_contains_all_three(self) -> None:
        assert WORKFLOW_LIBRARY_NAMES == {"rest-api", "refactor", "security-review", "bug-fix", "test-backfill"}

    # -- _load_library: built-in --

    def test_load_library_rest_api(self) -> None:
        lib = _load_library("rest-api")
        assert "tasks" in lib
        assert isinstance(lib["tasks"], list)
        assert len(lib["tasks"]) >= 3

    def test_load_library_refactor(self) -> None:
        lib = _load_library("refactor")
        assert "tasks" in lib
        assert lib["description"] == "Code refactoring (analyse + implement + verify)"

    def test_load_library_security_review(self) -> None:
        lib = _load_library("security-review")
        assert "tasks" in lib
        assert lib["include_quality_gates"] is False

    # -- _load_library: structure checks --

    def test_builtin_libraries_have_valid_task_types(self) -> None:
        from maestro_cli.scaffold import _VALID_TASK_TYPES
        for name in WORKFLOW_LIBRARY_NAMES:
            lib = _load_library(name)
            for task in lib["tasks"]:
                tt = task.get("task_type")
                if tt is not None:
                    assert tt in _VALID_TASK_TYPES, f"{name}/{task['id']}: invalid type {tt}"

    def test_builtin_libraries_have_valid_structure(self) -> None:
        for name in WORKFLOW_LIBRARY_NAMES:
            lib = _load_library(name)
            assert isinstance(lib["tasks"], list)
            assert "description" in lib
            for task in lib["tasks"]:
                assert "id" in task, f"{name}: task missing id"

    # -- _load_library: external file --

    def test_load_library_from_external_yaml(self, tmp_path: Path) -> None:
        lib_file = tmp_path / "custom-lib.yaml"
        lib_file.write_text(yaml.dump({
            "description": "Custom library",
            "tasks": [
                {"id": "step-1", "task_type": "shell"},
                {"id": "step-2", "task_type": "implementation", "depends_on": ["step-1"]},
            ],
        }), encoding="utf-8")
        lib = _load_library(str(lib_file))
        assert len(lib["tasks"]) == 2
        assert lib["tasks"][0]["id"] == "step-1"

    def test_load_library_external_with_custom_tasks(self, tmp_path: Path) -> None:
        lib_file = tmp_path / "ext.yaml"
        lib_file.write_text(yaml.dump({
            "description": "External",
            "goal": "Custom goal",
            "topology": "fan-out",
            "tasks": [
                {"id": "a", "task_type": "implementation", "prompt_hint": "do A"},
            ],
        }), encoding="utf-8")
        lib = _load_library(str(lib_file))
        assert lib["goal"] == "Custom goal"
        assert lib["topology"] == "fan-out"

    # -- _load_library: error paths --

    def test_load_library_unknown_name_raises(self) -> None:
        with pytest.raises(PlanValidationError, match="not found"):
            _load_library("does-not-exist")

    def test_load_library_missing_file_raises(self) -> None:
        with pytest.raises(PlanValidationError, match="not found"):
            _load_library("/nonexistent/path/lib.yaml")

    def test_load_library_invalid_yaml_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text(":\n  - :\n    {{invalid", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="Invalid workflow library YAML"):
            _load_library(str(bad))

    def test_load_library_non_dict_root_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "list.yaml"
        bad.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="root must be an object"):
            _load_library(str(bad))

    def test_load_library_missing_tasks_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "notasks.yaml"
        bad.write_text(yaml.dump({"description": "no tasks key"}), encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must have a 'tasks' list"):
            _load_library(str(bad))

    def test_load_library_tasks_not_list_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "badtasks.yaml"
        bad.write_text(yaml.dump({"tasks": "not-a-list"}), encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must have a 'tasks' list"):
            _load_library(str(bad))

    # -- _merge_library_into_brief --

    def test_merge_no_overrides_keeps_all_lib_tasks(self) -> None:
        lib = _load_library("refactor")
        merged_tasks, _ = _merge_library_into_brief(lib, [], {})
        lib_ids = [t["id"] for t in lib["tasks"]]
        merged_ids = [t["id"] for t in merged_tasks]
        assert merged_ids == lib_ids

    def test_merge_override_existing_task_by_id(self) -> None:
        lib = _load_library("refactor")
        override = [{"id": "analyse-code", "prompt_hint": "Custom analysis prompt"}]
        merged_tasks, _ = _merge_library_into_brief(lib, override, {})
        analyse = next(t for t in merged_tasks if t["id"] == "analyse-code")
        assert analyse["prompt_hint"] == "Custom analysis prompt"
        # Library defaults still present
        assert analyse["task_type"] == "code-review"

    def test_merge_add_extra_tasks(self) -> None:
        lib = _load_library("refactor")
        extra = [{"id": "extra-lint", "task_type": "shell"}]
        merged_tasks, _ = _merge_library_into_brief(lib, extra, {})
        ids = [t["id"] for t in merged_tasks]
        assert "extra-lint" in ids
        # Extra tasks come after library tasks
        assert ids.index("extra-lint") > ids.index("verify-refactor")

    def test_merge_both_override_and_add(self) -> None:
        lib = _load_library("refactor")
        brief_tasks = [
            {"id": "analyse-code", "prompt_hint": "Override prompt"},
            {"id": "new-task", "task_type": "implementation"},
        ]
        merged_tasks, _ = _merge_library_into_brief(lib, brief_tasks, {})
        ids = [t["id"] for t in merged_tasks]
        assert "analyse-code" in ids
        assert "new-task" in ids
        analyse = next(t for t in merged_tasks if t["id"] == "analyse-code")
        assert analyse["prompt_hint"] == "Override prompt"

    def test_merge_metadata_defaults_from_library(self) -> None:
        lib = _load_library("rest-api")
        _, merged_meta = _merge_library_into_brief(lib, [], {})
        assert merged_meta["goal"] == "Implement a REST API service with quality gates"
        assert merged_meta["topology"] == "diamond"
        assert merged_meta["include_quality_gates"] is True

    def test_merge_brief_metadata_overrides_library(self) -> None:
        lib = _load_library("rest-api")
        brief_raw = {"goal": "My custom goal", "topology": "linear"}
        _, merged_meta = _merge_library_into_brief(lib, [], brief_raw)
        assert merged_meta["goal"] == "My custom goal"
        assert merged_meta["topology"] == "linear"

    def test_merge_prompt_hint_override(self) -> None:
        lib = _load_library("rest-api")
        override = [{"id": "implement-models", "prompt_hint": "Use SQLAlchemy ORM"}]
        merged_tasks, _ = _merge_library_into_brief(lib, override, {})
        models_task = next(t for t in merged_tasks if t["id"] == "implement-models")
        assert models_task["prompt_hint"] == "Use SQLAlchemy ORM"
        # Library depends_on still present
        assert models_task["depends_on"] == ["setup-project"]

    def test_merge_depends_on_override(self) -> None:
        lib = _load_library("rest-api")
        override = [{"id": "implement-endpoints", "depends_on": ["setup-project"]}]
        merged_tasks, _ = _merge_library_into_brief(lib, override, {})
        endpoints = next(t for t in merged_tasks if t["id"] == "implement-endpoints")
        assert endpoints["depends_on"] == ["setup-project"]

    # -- load_brief with library --

    def test_load_brief_with_library_field_in_yaml(self, tmp_path: Path) -> None:
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "my-api",
            "library": "rest-api",
        }), encoding="utf-8")
        brief = load_brief(str(brief_file))
        assert brief.library == "rest-api"
        assert len(brief.tasks) >= 4  # rest-api has 4 tasks

    def test_load_brief_with_library_override_kwarg(self, tmp_path: Path) -> None:
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "my-refactor",
        }), encoding="utf-8")
        brief = load_brief(str(brief_file), library_override="refactor")
        assert brief.library == "refactor"
        assert len(brief.tasks) >= 3

    def test_load_brief_library_override_takes_precedence(self, tmp_path: Path) -> None:
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "my-plan",
            "library": "rest-api",
        }), encoding="utf-8")
        brief = load_brief(str(brief_file), library_override="security-review")
        assert brief.library == "security-review"
        task_ids = [t.id for t in brief.tasks]
        # Should have security-review tasks, not rest-api
        assert "dependency-scan" in task_ids
        assert "setup-project" not in task_ids

    def test_load_brief_library_no_brief_tasks(self, tmp_path: Path) -> None:
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "lib-only",
            "library": "refactor",
        }), encoding="utf-8")
        brief = load_brief(str(brief_file))
        task_ids = [t.id for t in brief.tasks]
        assert "analyse-code" in task_ids
        assert "implement-refactor" in task_ids
        assert "verify-refactor" in task_ids

    def test_load_brief_library_with_override_tasks(self, tmp_path: Path) -> None:
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "custom-refactor",
            "library": "refactor",
            "tasks": [
                {"id": "analyse-code", "prompt_hint": "Focus on module X"},
            ],
        }), encoding="utf-8")
        brief = load_brief(str(brief_file))
        analyse = next(t for t in brief.tasks if t.id == "analyse-code")
        assert analyse.prompt_hint == "Focus on module X"
        # Still has other library tasks
        assert any(t.id == "implement-refactor" for t in brief.tasks)

    def test_load_brief_library_with_extra_tasks(self, tmp_path: Path) -> None:
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "extended-refactor",
            "library": "refactor",
            "tasks": [
                {"id": "custom-lint", "task_type": "shell"},
            ],
        }), encoding="utf-8")
        brief = load_brief(str(brief_file))
        task_ids = [t.id for t in brief.tasks]
        assert "custom-lint" in task_ids
        # Library tasks still present
        assert "analyse-code" in task_ids

    def test_load_brief_library_injects_metadata_defaults(self, tmp_path: Path) -> None:
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "api-defaults",
            "library": "rest-api",
        }), encoding="utf-8")
        brief = load_brief(str(brief_file))
        assert brief.goal == "Implement a REST API service with quality gates"
        assert brief.topology == "diamond"
        assert brief.include_quality_gates is True

    def test_load_brief_library_metadata_overridden_by_brief(self, tmp_path: Path) -> None:
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "api-custom",
            "library": "rest-api",
            "goal": "Build payment API",
            "topology": "linear",
        }), encoding="utf-8")
        brief = load_brief(str(brief_file))
        assert brief.goal == "Build payment API"
        assert brief.topology == "linear"

    def test_plan_brief_library_field_preserved(self) -> None:
        brief = PlanBrief(name="test", library="rest-api")
        assert brief.library == "rest-api"

    def test_plan_brief_library_field_default_none(self) -> None:
        brief = PlanBrief(name="test")
        assert brief.library is None

    def test_load_brief_external_library(self, tmp_path: Path) -> None:
        lib_file = tmp_path / "mylib.yaml"
        lib_file.write_text(yaml.dump({
            "description": "External lib",
            "goal": "External goal",
            "topology": "fan-out",
            "tasks": [
                {"id": "ext-step", "task_type": "implementation", "prompt_hint": "Do ext"},
            ],
        }), encoding="utf-8")
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "ext-plan",
            "library": str(lib_file),
        }), encoding="utf-8")
        brief = load_brief(str(brief_file))
        assert len(brief.tasks) == 1
        assert brief.tasks[0].id == "ext-step"
        assert brief.goal == "External goal"

    # -- scaffold_plan with library brief --

    def test_scaffold_plan_with_library_produces_valid_yaml(self, tmp_path: Path) -> None:
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "scaffolded-api",
            "library": "rest-api",
        }), encoding="utf-8")
        brief = load_brief(str(brief_file))
        plan_yaml = scaffold_plan(brief)
        plan = yaml.safe_load(plan_yaml)
        assert plan["version"] == 1
        assert plan["name"] == "scaffolded-api"
        task_ids = [t["id"] for t in plan["tasks"]]
        assert "setup-project" in task_ids
        assert "implement-models" in task_ids

    def test_scaffold_plan_library_with_quality_gates(self, tmp_path: Path) -> None:
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "api-with-gates",
            "library": "rest-api",
        }), encoding="utf-8")
        brief = load_brief(str(brief_file))
        assert brief.include_quality_gates is True
        plan_yaml = scaffold_plan(brief)
        plan = yaml.safe_load(plan_yaml)
        task_ids = [t["id"] for t in plan["tasks"]]
        assert "code-review" in task_ids
        assert "qa-verification" in task_ids

    def test_scaffold_plan_library_without_quality_gates(self, tmp_path: Path) -> None:
        brief_file = tmp_path / "brief.yaml"
        brief_file.write_text(yaml.dump({
            "name": "sec-review",
            "library": "security-review",
        }), encoding="utf-8")
        brief = load_brief(str(brief_file))
        assert brief.include_quality_gates is False
        plan_yaml = scaffold_plan(brief)
        plan = yaml.safe_load(plan_yaml)
        task_ids = [t["id"] for t in plan["tasks"]]
        assert "code-review" not in task_ids
        assert "qa-verification" not in task_ids
        # But build-verify should still be present
        assert "build-verify" in task_ids
