"""Tests for T2.1 — Dynamic Task Decomposition (dynamic_group)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.dynamic import (
    _ALLOWED_TASK_FIELDS,
    _DYNAMIC_MAX_TASKS,
    _VALID_ENGINES,
    build_plan_from_output,
    merge_dynamic_result,
    write_raw_output,
)
from maestro_cli.errors import E063, E064, PlanValidationError
from maestro_cli.models import (
    FailureRecord,
    JudgeResult,
    PlanRunResult,
    PlanSpec,
    TaskResult,
    TaskSpec,
    TokenUsage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_plan(**kwargs: Any) -> PlanSpec:
    defaults = {
        "name": "parent-plan",
        "tasks": [TaskSpec(id="planner", engine="claude", prompt="plan it",
                           dynamic_group=True, output_schema={"type": "object"})],
        "max_cost_usd": 5.0,
    }
    defaults.update(kwargs)
    return PlanSpec(**defaults)


def _make_task(**kwargs: Any) -> TaskSpec:
    defaults = {
        "id": "planner",
        "engine": "claude",
        "prompt": "plan it",
        "dynamic_group": True,
        "output_schema": {"type": "object"},
    }
    defaults.update(kwargs)
    return TaskSpec(**defaults)


def _make_output(tasks: list[dict[str, Any]] | None = None, **kwargs: Any) -> dict[str, Any]:
    """Build a minimal valid structured output for dynamic_group."""
    if tasks is None:
        tasks = [
            {"id": "t1", "engine": "claude", "prompt": "Do thing A"},
            {"id": "t2", "engine": "claude", "prompt": "Do thing B", "depends_on": ["t1"]},
        ]
    result = {"name": "dynamic-plan", "tasks": tasks}
    result.update(kwargs)
    return result


def _make_task_result(
    task_id: str = "planner",
    status: str = "success",
    **kwargs: Any,
) -> TaskResult:
    defaults = {
        "task_id": task_id,
        "status": status,
        "exit_code": 0,
        "duration_sec": 10.0,
        "cost_usd": 0.05,
        "token_usage": TokenUsage(input_tokens=100, output_tokens=50),
    }
    defaults.update(kwargs)
    return TaskResult(**defaults)


def _make_sub_result(
    task_results: dict[str, TaskResult] | None = None,
    success: bool = True,
) -> PlanRunResult:
    if task_results is None:
        task_results = {
            "t1": _make_task_result("t1", "success", stdout_tail="output A"),
            "t2": _make_task_result("t2", "success", stdout_tail="output B"),
        }
    return PlanRunResult(
        plan_name="dynamic-plan",
        run_id="20260319_120000_000000",
        run_path=Path("/tmp/fake"),
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        success=success,
        execution_profile="safe",
        task_results=task_results,
        sequential_duration_sec=20.0,
        parallelism_savings_pct=0.0,
        total_cost_usd=0.10,
        total_tokens=500,
        budget_exceeded=False,
    )


# ---------------------------------------------------------------------------
# Loader / Validation tests
# ---------------------------------------------------------------------------

class TestLoaderValidation:
    def test_dynamic_group_parsed(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\n"
            "name: test\n"
            "max_cost_usd: 5.0\n"
            "tasks:\n"
            "  - id: planner\n"
            "    engine: claude\n"
            "    dynamic_group: true\n"
            "    output_schema:\n"
            "      type: object\n"
            "    prompt: plan it\n",
            encoding="utf-8",
        )
        plan = load_plan(plan_yaml)
        assert plan.tasks[0].dynamic_group is True
        assert plan.tasks[0].cache is False  # forced

    def test_dynamic_group_without_engine_raises_E063(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\n"
            "name: test\n"
            "tasks:\n"
            "  - id: planner\n"
            "    command: echo hi\n"
            "    dynamic_group: true\n"
            "    output_schema:\n"
            "      type: object\n",
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match="E063"):
            load_plan(plan_yaml)

    def test_dynamic_group_without_output_schema_raises_E063(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\n"
            "name: test\n"
            "tasks:\n"
            "  - id: planner\n"
            "    engine: claude\n"
            "    dynamic_group: true\n"
            "    prompt: plan it\n",
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match="E063"):
            load_plan(plan_yaml)

    def test_dynamic_group_with_group_raises_validation_error(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan
        plan_yaml = tmp_path / "plan.yaml"
        # group + engine + dynamic_group → E064 (or E011 for group+prompt conflict)
        plan_yaml.write_text(
            "version: 1\n"
            "name: test\n"
            "tasks:\n"
            "  - id: planner\n"
            "    engine: claude\n"
            "    group: sub.yaml\n"
            "    dynamic_group: true\n"
            "    output_schema:\n"
            "      type: object\n"
            "    prompt: plan it\n",
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError):
            load_plan(plan_yaml)

    def test_dynamic_group_with_batch_raises_E064(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\n"
            "name: test\n"
            "tasks:\n"
            "  - id: planner\n"
            "    engine: claude\n"
            "    dynamic_group: true\n"
            "    output_schema:\n"
            "      type: object\n"
            "    prompt: plan it\n"
            "    batch:\n"
            "      items: [a, b]\n"
            "      template: '{{ batch.item }}'\n",
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match="E064"):
            load_plan(plan_yaml)

    def test_dynamic_group_default_false(self) -> None:
        task = TaskSpec(id="t1", engine="claude", prompt="hi")
        assert task.dynamic_group is False

    def test_dynamic_group_to_dict(self) -> None:
        task = _make_task()
        d = task.to_dict()
        assert d["dynamic_group"] is True


# ---------------------------------------------------------------------------
# build_plan_from_output tests
# ---------------------------------------------------------------------------

class TestBuildPlanFromOutput:
    def test_valid_output_builds_plan(self) -> None:
        plan = _make_plan()
        task = _make_task()
        output = _make_output()
        sub_plan = build_plan_from_output(output, plan, task)
        assert sub_plan is not None
        assert len(sub_plan.tasks) == 2
        assert sub_plan.tasks[0].id == "t1"
        assert sub_plan.tasks[1].id == "t2"

    def test_empty_tasks_returns_none(self) -> None:
        result = build_plan_from_output({"tasks": []}, _make_plan(), _make_task())
        assert result is None

    def test_no_tasks_key_returns_none(self) -> None:
        result = build_plan_from_output({"name": "x"}, _make_plan(), _make_task())
        assert result is None

    def test_command_tasks_filtered_out(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "command": "rm -rf /", "prompt": "bad"},
            {"id": "t2", "engine": "claude", "prompt": "good"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == 1
        assert sub_plan.tasks[0].id == "t2"

    def test_invalid_engine_filtered_out(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "evil_engine", "prompt": "bad"},
            {"id": "t2", "engine": "claude", "prompt": "good"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == 1

    def test_tasks_capped_at_max(self) -> None:
        tasks = [{"id": f"t{i}", "engine": "claude", "prompt": f"task {i}"}
                 for i in range(30)]
        output = _make_output(tasks=tasks)
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == _DYNAMIC_MAX_TASKS

    def test_duplicate_ids_deduplicated(self) -> None:
        output = _make_output(tasks=[
            {"id": "dup", "engine": "claude", "prompt": "A"},
            {"id": "dup", "engine": "claude", "prompt": "B"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        ids = [t.id for t in sub_plan.tasks]
        assert len(ids) == len(set(ids))  # no duplicates

    def test_inherits_workspace_root(self) -> None:
        plan = _make_plan(workspace_root="/ws")
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.workspace_root == "/ws"

    def test_inherits_secrets(self) -> None:
        plan = _make_plan(secrets=["API_KEY"])
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.secrets == ["API_KEY"]

    def test_forces_control_flow_integrity(self) -> None:
        plan = _make_plan()
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.control_flow_integrity is True

    def test_inherits_max_cost(self) -> None:
        plan = _make_plan(max_cost_usd=3.0)
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.max_cost_usd == 3.0

    def test_caps_max_parallel(self) -> None:
        plan = _make_plan(max_parallel=2)
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.max_parallel <= 2

    def test_no_prompt_tasks_filtered(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude"},  # no prompt
            {"id": "t2", "engine": "claude", "prompt": "good"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == 1

    def test_validation_failure_returns_none(self) -> None:
        # Circular deps → validate_plan() fails → returns None
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "A", "depends_on": ["t2"]},
            {"id": "t2", "engine": "claude", "prompt": "B", "depends_on": ["t1"]},
        ])
        result = build_plan_from_output(output, _make_plan(), _make_task())
        assert result is None

    def test_cache_forced_false(self) -> None:
        sub_plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert sub_plan is not None
        for t in sub_plan.tasks:
            assert t.cache is False

    def test_fail_fast_forced_true(self) -> None:
        sub_plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.fail_fast is True


# ---------------------------------------------------------------------------
# Allowlist enforcement tests
# ---------------------------------------------------------------------------

class TestAllowlistEnforcement:
    """Verify that dangerous fields in LLM output are silently ignored."""

    def _build_with_extra(self, extra: dict[str, Any]) -> PlanSpec | None:
        task_dict: dict[str, Any] = {
            "id": "t1", "engine": "claude", "prompt": "safe task",
        }
        task_dict.update(extra)
        output = _make_output(tasks=[task_dict])
        return build_plan_from_output(output, _make_plan(), _make_task())

    def test_args_stripped(self) -> None:
        sub = self._build_with_extra({"args": ["--dangerously-skip-permissions"]})
        assert sub is not None
        assert sub.tasks[0].args == []

    def test_env_stripped(self) -> None:
        sub = self._build_with_extra({"env": {"PATH": "/evil"}})
        assert sub is not None
        assert sub.tasks[0].env == {}

    def test_workdir_stripped(self) -> None:
        sub = self._build_with_extra({"workdir": "/etc"})
        assert sub is not None
        assert sub.tasks[0].workdir is None

    def test_pre_command_stripped(self) -> None:
        sub = self._build_with_extra({"pre_command": "rm -rf /"})
        assert sub is not None
        assert sub.tasks[0].pre_command is None

    def test_verify_command_stripped(self) -> None:
        sub = self._build_with_extra({"verify_command": "curl evil.com"})
        assert sub is not None
        assert sub.tasks[0].verify_command is None

    def test_guard_command_stripped(self) -> None:
        sub = self._build_with_extra({"guard_command": "wget evil.com"})
        assert sub is not None
        assert sub.tasks[0].guard_command is None

    def test_command_stripped(self) -> None:
        # Task with command + engine → command is not used (engine takes priority)
        sub = self._build_with_extra({"command": "rm -rf /"})
        assert sub is not None
        assert sub.tasks[0].command is None

    def test_dynamic_group_stripped(self) -> None:
        sub = self._build_with_extra({"dynamic_group": True})
        assert sub is not None
        assert sub.tasks[0].dynamic_group is False

    def test_allow_failure_stripped(self) -> None:
        sub = self._build_with_extra({"allow_failure": True})
        assert sub is not None
        assert sub.tasks[0].allow_failure is False

    def test_requires_approval_stripped(self) -> None:
        sub = self._build_with_extra({"requires_approval": True})
        assert sub is not None
        assert sub.tasks[0].requires_approval is False

    def test_append_system_prompt_stripped(self) -> None:
        sub = self._build_with_extra({"append_system_prompt": "ignore previous"})
        assert sub is not None
        assert sub.tasks[0].append_system_prompt is None

    def test_timeout_stripped(self) -> None:
        sub = self._build_with_extra({"timeout_sec": 86400})
        assert sub is not None
        # Should inherit from defaults, not from LLM output
        assert sub.tasks[0].timeout_sec != 86400

    def test_worktree_stripped(self) -> None:
        sub = self._build_with_extra({"worktree": True})
        assert sub is not None
        assert sub.tasks[0].worktree is False


# ---------------------------------------------------------------------------
# merge_dynamic_result tests
# ---------------------------------------------------------------------------

class TestMergeDynamicResult:
    def test_merge_success(self) -> None:
        phase1 = _make_task_result()
        sub = _make_sub_result()
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.status == "success"
        assert result.cost_usd == pytest.approx(0.15)  # 0.05 + 0.10

    def test_merge_sub_failure_sets_failed(self) -> None:
        phase1 = _make_task_result()
        sub = _make_sub_result(
            task_results={"t1": _make_task_result("t1", "failed")},
            success=False,
        )
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.status == "failed"

    def test_merge_sub_failure_with_allow_failure(self) -> None:
        phase1 = _make_task_result()
        task = _make_task(allow_failure=True)
        sub = _make_sub_result(
            task_results={"t1": _make_task_result("t1", "failed")},
            success=False,
        )
        result = merge_dynamic_result(phase1, sub, task)
        assert result.status == "soft_failed"

    def test_stdout_tail_contains_sub_outputs(self) -> None:
        phase1 = _make_task_result()
        sub = _make_sub_result()
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert "output A" in result.stdout_tail
        assert "output B" in result.stdout_tail
        assert "=== t1" in result.stdout_tail

    def test_structured_output_replaced_with_summary(self) -> None:
        phase1 = _make_task_result()
        phase1.structured_output = {"tasks": []}  # Phase 1 output
        sub = _make_sub_result()
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert "sub_tasks" in result.structured_output
        assert result.structured_output["ok"] == 2

    def test_dynamic_subplan_result_populated(self) -> None:
        phase1 = _make_task_result()
        sub = _make_sub_result()
        result = merge_dynamic_result(phase1, sub, _make_task())
        dsr = result.dynamic_subplan_result
        assert dsr is not None
        assert dsr["success"] is True
        assert dsr["task_count"] == 2
        assert dsr["plan_name"] == "dynamic-plan"

    def test_tokens_aggregated(self) -> None:
        phase1 = _make_task_result(token_usage=TokenUsage(input_tokens=100, output_tokens=50))
        sub = _make_sub_result()
        sub.total_tokens = 500
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 600  # 100 + 500


# ---------------------------------------------------------------------------
# write_raw_output tests
# ---------------------------------------------------------------------------

class TestWriteRawOutput:
    def test_writes_json_file(self, tmp_path: Path) -> None:
        output = {"tasks": [{"id": "t1"}]}
        write_raw_output(tmp_path, "planner", output)
        path = tmp_path / "planner" / "_dynamic" / "raw_output.json"
        assert path.is_file()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["tasks"][0]["id"] == "t1"


# ---------------------------------------------------------------------------
# Policy integration
# ---------------------------------------------------------------------------

class TestPolicyIntegration:
    def test_dynamic_group_in_safe_fields(self) -> None:
        from maestro_cli.policy import _SAFE_TASK_FIELDS
        assert "dynamic_group" in _SAFE_TASK_FIELDS

    def test_policy_can_evaluate_dynamic_group(self) -> None:
        from maestro_cli.policy import compile_policy
        from maestro_cli.models import PolicySpec
        policy = PolicySpec(
            name="block-dynamic",
            rule="task.dynamic_group == True",
            action="block",
            message="No dynamic groups allowed",
        )
        compiled = compile_policy(policy)
        assert compiled is not None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_allowed_fields_are_7(self) -> None:
        assert len(_ALLOWED_TASK_FIELDS) == 7

    def test_valid_engines_match_known(self) -> None:
        assert "claude" in _VALID_ENGINES
        assert "codex" in _VALID_ENGINES
        assert "gemini" in _VALID_ENGINES
        assert "copilot" in _VALID_ENGINES
        assert "qwen" in _VALID_ENGINES
        assert "ollama" in _VALID_ENGINES
        assert "evil" not in _VALID_ENGINES


# ---------------------------------------------------------------------------
# build_plan_from_output — extended edge cases
# ---------------------------------------------------------------------------

class TestBuildPlanEdgeCases:
    """Additional edge cases for build_plan_from_output."""

    # 1. Non-dict output
    def test_string_output_returns_none(self) -> None:
        result = build_plan_from_output("just a string", _make_plan(), _make_task())  # type: ignore[arg-type]
        assert result is None

    def test_list_output_returns_none(self) -> None:
        result = build_plan_from_output([1, 2, 3], _make_plan(), _make_task())  # type: ignore[arg-type]
        assert result is None

    def test_none_output_returns_none(self) -> None:
        result = build_plan_from_output(None, _make_plan(), _make_task())  # type: ignore[arg-type]
        assert result is None

    def test_int_output_returns_none(self) -> None:
        result = build_plan_from_output(42, _make_plan(), _make_task())  # type: ignore[arg-type]
        assert result is None

    # 2. Empty tasks list (already tested in TestBuildPlanFromOutput, but explicit here)
    def test_empty_tasks_list_returns_none(self) -> None:
        result = build_plan_from_output({"tasks": []}, _make_plan(), _make_task())
        assert result is None

    # 3. Tasks with missing engine — skipped, others kept
    def test_task_missing_engine_skipped(self) -> None:
        output = _make_output(tasks=[
            {"id": "no-engine", "prompt": "I have no engine"},
            {"id": "good", "engine": "claude", "prompt": "I'm fine"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == 1
        assert sub_plan.tasks[0].id == "good"

    # 4. Tasks with invalid engine — skipped
    def test_task_invalid_engine_skipped(self) -> None:
        output = _make_output(tasks=[
            {"id": "bad", "engine": "invalid_engine", "prompt": "nope"},
            {"id": "ok", "engine": "gemini", "prompt": "fine"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == 1
        assert sub_plan.tasks[0].id == "ok"
        assert sub_plan.tasks[0].engine == "gemini"

    # 5. Tasks with missing prompt — skipped
    def test_task_missing_prompt_skipped(self) -> None:
        output = _make_output(tasks=[
            {"id": "no-prompt", "engine": "claude"},
            {"id": "has-prompt", "engine": "claude", "prompt": "good"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == 1
        assert sub_plan.tasks[0].id == "has-prompt"

    # 6. Tasks with non-string prompt — skipped
    def test_task_non_string_prompt_skipped(self) -> None:
        output = _make_output(tasks=[
            {"id": "num-prompt", "engine": "claude", "prompt": 123},
            {"id": "list-prompt", "engine": "claude", "prompt": ["a", "b"]},
            {"id": "ok", "engine": "claude", "prompt": "real prompt"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == 1
        assert sub_plan.tasks[0].id == "ok"

    # 7. Duplicate task IDs — second gets "-{idx}" suffix
    def test_duplicate_ids_get_suffix(self) -> None:
        output = _make_output(tasks=[
            {"id": "dup", "engine": "claude", "prompt": "first"},
            {"id": "dup", "engine": "claude", "prompt": "second"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        ids = [t.id for t in sub_plan.tasks]
        assert ids[0] == "dup"
        assert ids[1] == "dup-1"  # second is at index 1

    # 8. More than _DYNAMIC_MAX_TASKS tasks — capped
    def test_tasks_capped_at_dynamic_max(self) -> None:
        from maestro_cli.dynamic import _DYNAMIC_MAX_TASKS
        tasks = [
            {"id": f"task-{i}", "engine": "claude", "prompt": f"do {i}"}
            for i in range(_DYNAMIC_MAX_TASKS + 10)
        ]
        output = _make_output(tasks=tasks)
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == _DYNAMIC_MAX_TASKS

    # 9. Non-dict items in tasks list — skipped without crash
    def test_non_dict_items_in_tasks_skipped(self) -> None:
        output = _make_output(tasks=[
            "just a string",
            42,
            None,
            ["a", "list"],
            {"id": "valid", "engine": "claude", "prompt": "ok"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == 1
        assert sub_plan.tasks[0].id == "valid"

    # 10. depends_on as non-list — falls back to []
    def test_depends_on_non_list_fallback(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "A", "depends_on": "not-a-list"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[0].depends_on == []

    def test_depends_on_int_fallback(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "A", "depends_on": 42},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[0].depends_on == []

    # 11. tags as non-list — falls back to []
    def test_tags_non_list_fallback(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "A", "tags": "not-a-list"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[0].tags == []

    def test_tags_int_fallback(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "A", "tags": 99},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[0].tags == []

    # 12. model is None — kept as None
    def test_model_none_kept(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "A", "model": None},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[0].model is None

    def test_model_absent_is_none(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "A"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[0].model is None

    # 13. model is non-string — converted to string
    def test_model_int_converted_to_string(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "A", "model": 42},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[0].model == "42"

    def test_model_float_converted_to_string(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "A", "model": 5.4},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert isinstance(sub_plan.tasks[0].model, str)

    # 14. Custom plan name in output — used as sub-plan name
    def test_custom_plan_name_used(self) -> None:
        output = _make_output(name="my-custom-plan")
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.name == "my-custom-plan"

    # 15. No plan name in output — defaults to "{task_id}-dynamic"
    def test_no_plan_name_defaults(self) -> None:
        output = {"tasks": [{"id": "t1", "engine": "claude", "prompt": "A"}]}
        task = _make_task(id="my-planner")
        sub_plan = build_plan_from_output(output, _make_plan(), task)
        assert sub_plan is not None
        assert sub_plan.name == "my-planner-dynamic"

    # 16. Parent plan has no max_cost_usd — remaining_budget is None
    def test_no_parent_budget_gives_none(self) -> None:
        plan = _make_plan(max_cost_usd=None)
        output = _make_output()
        sub_plan = build_plan_from_output(output, plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.max_cost_usd is None

    # 17. Parent plan has max_cost_usd — budget passed through
    def test_parent_budget_passed_through(self) -> None:
        plan = _make_plan(max_cost_usd=7.5)
        output = _make_output()
        sub_plan = build_plan_from_output(output, plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.max_cost_usd == 7.5

    # 18. All tasks filtered out (all invalid) — returns None
    def test_all_tasks_invalid_returns_none(self) -> None:
        output = _make_output(tasks=[
            {"id": "a", "engine": "evil_engine", "prompt": "bad"},
            {"id": "b", "prompt": "no engine"},
            {"id": "c", "engine": "claude"},  # no prompt
            "just a string",
            42,
        ])
        result = build_plan_from_output(output, _make_plan(), _make_task())
        assert result is None

    # 19. control_flow_integrity is always True on sub-plan
    def test_cfi_always_true(self) -> None:
        plan = _make_plan()
        plan.control_flow_integrity = False  # parent says False
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.control_flow_integrity is True

    # 20. fail_fast is always True on sub-plan
    def test_fail_fast_always_true(self) -> None:
        plan = _make_plan(fail_fast=False)
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.fail_fast is True

    # 21. cache is always False on sub-tasks
    def test_cache_always_false_on_all_tasks(self) -> None:
        tasks = [
            {"id": f"t{i}", "engine": "claude", "prompt": f"do {i}"}
            for i in range(5)
        ]
        output = _make_output(tasks=tasks)
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        for t in sub_plan.tasks:
            assert t.cache is False

    # 22. max_retries is always _DYNAMIC_MAX_RETRIES on sub-tasks
    def test_max_retries_forced(self) -> None:
        from maestro_cli.dynamic import _DYNAMIC_MAX_RETRIES
        output = _make_output()
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        for t in sub_plan.tasks:
            assert t.max_retries == _DYNAMIC_MAX_RETRIES

    # 23. Sub-plan inherits parent workspace_root
    def test_inherits_workspace_root_value(self) -> None:
        plan = _make_plan(workspace_root="/my/workspace")
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.workspace_root == "/my/workspace"

    def test_inherits_workspace_root_none(self) -> None:
        plan = _make_plan(workspace_root=None)
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.workspace_root is None

    # 24. Sub-plan inherits parent policies
    def test_inherits_parent_policies(self) -> None:
        from maestro_cli.models import PolicySpec
        policy = PolicySpec(name="test-policy", rule="task.engine == 'claude'", action="warn")
        plan = _make_plan(policies=[policy])
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert len(sub_plan.policies) == 1
        assert sub_plan.policies[0].name == "test-policy"

    def test_inherits_empty_policies(self) -> None:
        plan = _make_plan(policies=[])
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.policies == []

    # 25. Validation catches cycles in depends_on — returns None
    def test_cycle_in_depends_on_returns_none(self) -> None:
        output = _make_output(tasks=[
            {"id": "a", "engine": "claude", "prompt": "A", "depends_on": ["b"]},
            {"id": "b", "engine": "claude", "prompt": "B", "depends_on": ["a"]},
        ])
        result = build_plan_from_output(output, _make_plan(), _make_task())
        assert result is None

    def test_self_dependency_returns_none(self) -> None:
        output = _make_output(tasks=[
            {"id": "self-ref", "engine": "claude", "prompt": "X", "depends_on": ["self-ref"]},
        ])
        result = build_plan_from_output(output, _make_plan(), _make_task())
        assert result is None

    # Extra: sub-plan inherits routing_strategy
    def test_inherits_routing_strategy(self) -> None:
        plan = _make_plan(routing_strategy="cost_optimized")
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.routing_strategy == "cost_optimized"

    # Extra: sub-plan inherits secrets_auto
    def test_inherits_secrets_auto(self) -> None:
        plan = _make_plan(secrets_auto=True)
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.secrets_auto is True

    # Extra: sub-plan inherits defaults
    def test_inherits_parent_defaults(self) -> None:
        from maestro_cli.models import PlanDefaults
        defaults = PlanDefaults(timeout_sec=120)
        plan = _make_plan(defaults=defaults)
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.defaults.timeout_sec == 120

    # Extra: sub-plan inherits source_path
    def test_inherits_source_path(self) -> None:
        plan = _make_plan(source_path=Path("/plans/my-plan.yaml"))
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.source_path == Path("/plans/my-plan.yaml")

    # Extra: auto-generated task ID when missing
    def test_auto_generated_id_when_missing(self) -> None:
        output = _make_output(tasks=[
            {"engine": "claude", "prompt": "no id here"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[0].id == "dyn-0"

    # Extra: description field preserved
    def test_description_preserved(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "go", "description": "My desc"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[0].description == "My desc"

    # Extra: tags field preserved when valid list
    def test_tags_preserved_when_list(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "go", "tags": ["web", "api"]},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[0].tags == ["web", "api"]

    # Extra: depends_on with valid references preserved
    def test_depends_on_valid_references(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "first"},
            {"id": "t2", "engine": "claude", "prompt": "second", "depends_on": ["t1"]},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[1].depends_on == ["t1"]

    # Extra: unknown depends_on reference causes validation failure
    def test_unknown_dep_returns_none(self) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": "claude", "prompt": "A", "depends_on": ["nonexistent"]},
        ])
        result = build_plan_from_output(output, _make_plan(), _make_task())
        assert result is None

    # Extra: max_parallel capped at _DYNAMIC_MAX_TASKS
    def test_max_parallel_capped(self) -> None:
        from maestro_cli.dynamic import _DYNAMIC_MAX_TASKS
        plan = _make_plan(max_parallel=100)
        sub_plan = build_plan_from_output(_make_output(), plan, _make_task())
        assert sub_plan is not None
        assert sub_plan.max_parallel <= _DYNAMIC_MAX_TASKS

    # Extra: all six engines accepted
    @pytest.mark.parametrize("engine", ["codex", "claude", "gemini", "copilot", "qwen", "ollama"])
    def test_all_valid_engines_accepted(self, engine: str) -> None:
        output = _make_output(tasks=[
            {"id": "t1", "engine": engine, "prompt": "go"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert sub_plan.tasks[0].engine == engine

    # Extra: empty prompt string treated as missing
    def test_empty_prompt_string_skipped(self) -> None:
        output = _make_output(tasks=[
            {"id": "empty", "engine": "claude", "prompt": ""},
            {"id": "ok", "engine": "claude", "prompt": "real"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == 1
        assert sub_plan.tasks[0].id == "ok"

    # Extra: empty engine string treated as missing
    def test_empty_engine_string_skipped(self) -> None:
        output = _make_output(tasks=[
            {"id": "empty-eng", "engine": "", "prompt": "go"},
            {"id": "ok", "engine": "claude", "prompt": "real"},
        ])
        sub_plan = build_plan_from_output(output, _make_plan(), _make_task())
        assert sub_plan is not None
        assert len(sub_plan.tasks) == 1
        assert sub_plan.tasks[0].id == "ok"


# ---------------------------------------------------------------------------
# merge_dynamic_result — extended edge cases
# ---------------------------------------------------------------------------

class TestMergeDynamicResultEdgeCases:
    """Additional edge cases for merge_dynamic_result."""

    # 26. Phase 1 has no cost — only sub-plan cost counted
    def test_phase1_no_cost(self) -> None:
        phase1 = _make_task_result(cost_usd=None)
        sub = _make_sub_result()
        sub.total_cost_usd = 0.20
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.cost_usd == pytest.approx(0.20)

    # 27. Sub-plan has no cost — only phase1 cost counted
    def test_sub_plan_no_cost(self) -> None:
        phase1 = _make_task_result(cost_usd=0.05)
        sub = _make_sub_result()
        sub.total_cost_usd = None
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.cost_usd == pytest.approx(0.05)

    # 28. Both have no cost — cost stays None
    def test_both_no_cost(self) -> None:
        phase1 = _make_task_result(cost_usd=None)
        sub = _make_sub_result()
        sub.total_cost_usd = None
        result = merge_dynamic_result(phase1, sub, _make_task())
        # total_cost = 0.0 + 0.0 = 0.0 → not > 0 → keeps original (None)
        assert result.cost_usd is None

    # 29. Sub-plan tokens with no phase1 tokens — creates new TokenUsage
    def test_sub_tokens_no_phase1_tokens(self) -> None:
        phase1 = _make_task_result(token_usage=None)
        sub = _make_sub_result()
        sub.total_tokens = 300
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 300

    # 30. Sub-plan tokens with phase1 tokens — merged into input_tokens
    def test_sub_tokens_merged_into_input(self) -> None:
        phase1 = _make_task_result(
            token_usage=TokenUsage(input_tokens=200, output_tokens=100, cached_tokens=50),
        )
        sub = _make_sub_result()
        sub.total_tokens = 400
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 600  # 200 + 400
        assert result.token_usage.output_tokens == 100  # preserved
        assert result.token_usage.cached_tokens == 50  # preserved

    # 31. No sub-plan tokens — phase1 tokens preserved unchanged
    def test_no_sub_tokens_preserves_phase1(self) -> None:
        phase1 = _make_task_result(
            token_usage=TokenUsage(input_tokens=150, output_tokens=75),
        )
        sub = _make_sub_result()
        sub.total_tokens = 0
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 150
        assert result.token_usage.output_tokens == 75

    def test_no_sub_tokens_none_preserves_phase1(self) -> None:
        phase1 = _make_task_result(
            token_usage=TokenUsage(input_tokens=150, output_tokens=75),
        )
        sub = _make_sub_result()
        sub.total_tokens = None
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 150
        assert result.token_usage.output_tokens == 75

    # 32. Sub-plan fails with allow_failure — status becomes "soft_failed"
    def test_sub_fails_allow_failure_soft_failed(self) -> None:
        phase1 = _make_task_result(status="success")
        task = _make_task(allow_failure=True)
        sub = _make_sub_result(
            task_results={"t1": _make_task_result("t1", "failed")},
            success=False,
        )
        result = merge_dynamic_result(phase1, sub, task)
        assert result.status == "soft_failed"

    # 33. Sub-plan fails without allow_failure — status becomes "failed"
    def test_sub_fails_no_allow_failure_failed(self) -> None:
        phase1 = _make_task_result(status="success")
        task = _make_task(allow_failure=False)
        sub = _make_sub_result(
            task_results={"t1": _make_task_result("t1", "failed")},
            success=False,
        )
        result = merge_dynamic_result(phase1, sub, task)
        assert result.status == "failed"

    # 34. Sub-plan succeeds — keeps phase1 status
    def test_sub_succeeds_keeps_phase1_status(self) -> None:
        phase1 = _make_task_result(status="success")
        sub = _make_sub_result(success=True)
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.status == "success"

    # 35. Sub-task stdout_tail included in merged output
    def test_sub_task_stdout_in_merged_output(self) -> None:
        phase1 = _make_task_result()
        sub = _make_sub_result(
            task_results={
                "t1": _make_task_result("t1", "success", stdout_tail="alpha output"),
                "t2": _make_task_result("t2", "failed", stdout_tail="beta output"),
            },
        )
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert "alpha output" in result.stdout_tail
        assert "beta output" in result.stdout_tail
        assert "=== t1 (success) ===" in result.stdout_tail
        assert "=== t2 (failed) ===" in result.stdout_tail

    # 36. Empty sub-task outputs — sub_summary used instead
    def test_empty_sub_outputs_uses_summary(self) -> None:
        phase1 = _make_task_result()
        sub = _make_sub_result(
            task_results={
                "t1": _make_task_result("t1", "success", stdout_tail=""),
                "t2": _make_task_result("t2", "success", stdout_tail=""),
            },
        )
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert "Dynamic sub-plan:" in result.stdout_tail
        assert "2 ok" in result.stdout_tail

    # 37. structured_output has correct shape
    def test_structured_output_shape(self) -> None:
        phase1 = _make_task_result()
        sub = _make_sub_result(
            task_results={
                "t1": _make_task_result("t1", "success", stdout_tail="out1"),
                "t2": _make_task_result("t2", "failed", stdout_tail="out2"),
                "t3": _make_task_result("t3", "skipped", stdout_tail=""),
            },
            success=False,
        )
        result = merge_dynamic_result(phase1, sub, _make_task(allow_failure=True))
        so = result.structured_output
        assert so is not None
        assert "sub_tasks" in so
        assert so["ok"] == 1  # only "success" and "soft_failed" and "dry_run"
        assert so["failed"] == 1
        assert so["skipped"] == 1
        assert len(so["sub_tasks"]) == 3
        # Each sub_task has id, status, summary
        for st in so["sub_tasks"]:
            assert "id" in st
            assert "status" in st
            assert "summary" in st

    # 38. dynamic_subplan_result has correct shape
    def test_dynamic_subplan_result_shape(self) -> None:
        phase1 = _make_task_result()
        sub = _make_sub_result()
        sub.total_cost_usd = 0.25
        sub.total_tokens = 999
        result = merge_dynamic_result(phase1, sub, _make_task())
        dsr = result.dynamic_subplan_result
        assert dsr is not None
        assert dsr["plan_name"] == "dynamic-plan"
        assert dsr["success"] is True
        assert dsr["task_count"] == 2
        assert dsr["ok_count"] == 2
        assert dsr["fail_count"] == 0
        assert dsr["skip_count"] == 0
        assert dsr["total_cost_usd"] == 0.25
        assert dsr["total_tokens"] == 999
        assert "run_path" in dsr

    # Extra: message includes sub_summary appended
    def test_message_includes_sub_summary(self) -> None:
        phase1 = _make_task_result()
        phase1.message = "Phase 1 done"
        sub = _make_sub_result()
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert "Phase 1 done" in result.message
        assert "Dynamic sub-plan:" in result.message

    def test_message_empty_phase1(self) -> None:
        phase1 = _make_task_result()
        phase1.message = ""
        sub = _make_sub_result()
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert "Dynamic sub-plan:" in result.message

    # Extra: soft_failed counted in ok
    def test_soft_failed_counted_in_ok(self) -> None:
        phase1 = _make_task_result()
        sub = _make_sub_result(
            task_results={
                "t1": _make_task_result("t1", "soft_failed"),
            },
            success=True,
        )
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.structured_output is not None
        assert result.structured_output["ok"] == 1

    # Extra: dry_run counted in ok
    def test_dry_run_counted_in_ok(self) -> None:
        phase1 = _make_task_result()
        sub = _make_sub_result(
            task_results={
                "t1": _make_task_result("t1", "dry_run"),
            },
            success=True,
        )
        result = merge_dynamic_result(phase1, sub, _make_task())
        assert result.structured_output is not None
        assert result.structured_output["ok"] == 1

    # Extra: sub_task summary truncated at 500 chars
    def test_sub_task_summary_truncated(self) -> None:
        long_output = "x" * 1000
        phase1 = _make_task_result()
        sub = _make_sub_result(
            task_results={
                "t1": _make_task_result("t1", "success", stdout_tail=long_output),
            },
        )
        result = merge_dynamic_result(phase1, sub, _make_task())
        so = result.structured_output
        assert so is not None
        assert len(so["sub_tasks"][0]["summary"]) == 500

    # Extra: both token_usage None and sub_tokens zero
    def test_both_tokens_none_and_zero(self) -> None:
        phase1 = _make_task_result(token_usage=None)
        sub = _make_sub_result()
        sub.total_tokens = 0
        result = merge_dynamic_result(phase1, sub, _make_task())
        # Both None → merged stays None
        assert result.token_usage is None


# ---------------------------------------------------------------------------
# write_raw_output — extended edge cases
# ---------------------------------------------------------------------------

class TestWriteRawOutputEdgeCases:
    """Additional edge cases for write_raw_output."""

    # 39. Creates directory structure
    def test_creates_directory_structure(self, tmp_path: Path) -> None:
        write_raw_output(tmp_path, "my-task", {"key": "value"})
        forensics_dir = tmp_path / "my-task" / "_dynamic"
        assert forensics_dir.is_dir()
        raw_file = forensics_dir / "raw_output.json"
        assert raw_file.is_file()

    # 40. Handles OSError gracefully
    def test_oserror_does_not_raise(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fail_write_text(*args: Any, **kwargs: Any) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", _fail_write_text)
        # Should not raise
        write_raw_output(tmp_path, "task-x", {"test": True})

    # 41. JSON is properly formatted
    def test_json_formatted_with_indent(self, tmp_path: Path) -> None:
        data = {"tasks": [{"id": "t1"}], "unicode": "cafe\u0301"}
        write_raw_output(tmp_path, "fmt-test", data)
        raw_file = tmp_path / "fmt-test" / "_dynamic" / "raw_output.json"
        content = raw_file.read_text(encoding="utf-8")
        # Check indent=2 formatting
        assert "  " in content  # indented
        parsed = json.loads(content)
        assert parsed == data
        # Check ensure_ascii=False: the unicode should be present as-is
        assert "cafe\u0301" in content

    def test_nested_dirs_created(self, tmp_path: Path) -> None:
        """Ensure parent directories created even for deeply nested task IDs."""
        write_raw_output(tmp_path, "deep-task", {"x": 1})
        path = tmp_path / "deep-task" / "_dynamic" / "raw_output.json"
        assert path.is_file()


# ---------------------------------------------------------------------------
# run_dynamic_subplan tests
# ---------------------------------------------------------------------------

class TestRunDynamicSubplan:
    """Tests for run_dynamic_subplan function."""

    def _mock_run_plan(
        self,
        monkeypatch: pytest.MonkeyPatch,
        return_result: PlanRunResult | None = None,
    ) -> list[dict[str, Any]]:
        """Mock scheduler.run_plan and capture call args."""
        calls: list[dict[str, Any]] = []

        def _fake_run_plan(plan: Any, **kwargs: Any) -> PlanRunResult:
            kwargs["plan"] = plan
            calls.append(kwargs)
            if return_result is not None:
                return return_result
            return _make_sub_result()

        monkeypatch.setattr("maestro_cli.scheduler.run_plan", _fake_run_plan)
        return calls

    # 42. event_callback wrapping — verify dynamic_parent tag added
    def test_event_callback_wrapping(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.dynamic import run_dynamic_subplan

        received_events: list[tuple[str, dict[str, object]]] = []

        def _original_cb(event: str, data: dict[str, object]) -> None:
            received_events.append((event, data))

        calls = self._mock_run_plan(monkeypatch)
        plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert plan is not None

        run_dynamic_subplan(
            plan,
            run_path=tmp_path,
            task_id="planner",
            dry_run=False,
            execution_profile="plan",
            event_callback=_original_cb,
        )

        # The mock captured the wrapped_callback — invoke it to check tagging
        assert len(calls) == 1
        wrapped_cb = calls[0].get("event_callback")
        assert wrapped_cb is not None
        wrapped_cb("test_event", {"task_id": "t1"})
        assert len(received_events) == 1
        event_name, event_data = received_events[0]
        assert event_name == "test_event"
        assert event_data["task_id"] == "t1"
        assert event_data["dynamic_parent"] == "planner"

    def test_event_callback_does_not_mutate_original_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.dynamic import run_dynamic_subplan

        calls = self._mock_run_plan(monkeypatch)
        plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert plan is not None

        def _noop_cb(event: str, data: dict[str, object]) -> None:
            pass

        run_dynamic_subplan(
            plan,
            run_path=tmp_path,
            task_id="planner",
            dry_run=False,
            execution_profile="plan",
            event_callback=_noop_cb,
        )

        wrapped_cb = calls[0]["event_callback"]
        original_data: dict[str, object] = {"task_id": "t1"}
        wrapped_cb("evt", original_data)
        # Original dict should NOT be mutated
        assert "dynamic_parent" not in original_data

    # 43. execution_profile forced to "safe"
    def test_execution_profile_forced_safe(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.dynamic import run_dynamic_subplan

        calls = self._mock_run_plan(monkeypatch)
        plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert plan is not None

        run_dynamic_subplan(
            plan,
            run_path=tmp_path,
            task_id="planner",
            dry_run=False,
            execution_profile="yolo",  # passed as yolo, should be overridden
        )

        assert len(calls) == 1
        assert calls[0]["execution_profile"] == "safe"

    # 44. run_dir_override set correctly
    def test_run_dir_override_set(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.dynamic import run_dynamic_subplan

        calls = self._mock_run_plan(monkeypatch)
        plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert plan is not None

        run_dynamic_subplan(
            plan,
            run_path=tmp_path,
            task_id="my-planner",
            dry_run=False,
            execution_profile="plan",
        )

        assert len(calls) == 1
        expected_dir = str(tmp_path / "my-planner" / "_dynamic")
        assert calls[0]["run_dir_override"] == expected_dir

    def test_sub_run_dir_created(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.dynamic import run_dynamic_subplan

        self._mock_run_plan(monkeypatch)
        plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert plan is not None

        run_dynamic_subplan(
            plan,
            run_path=tmp_path,
            task_id="planner",
            dry_run=False,
            execution_profile="plan",
        )

        sub_dir = tmp_path / "planner" / "_dynamic"
        assert sub_dir.is_dir()

    # 45. No event_callback — wrapped_callback is None
    def test_no_event_callback_gives_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.dynamic import run_dynamic_subplan

        calls = self._mock_run_plan(monkeypatch)
        plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert plan is not None

        run_dynamic_subplan(
            plan,
            run_path=tmp_path,
            task_id="planner",
            dry_run=False,
            execution_profile="plan",
            event_callback=None,
        )

        assert len(calls) == 1
        assert calls[0]["event_callback"] is None

    # Extra: dry_run passed through
    def test_dry_run_passed_through(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.dynamic import run_dynamic_subplan

        calls = self._mock_run_plan(monkeypatch)
        plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert plan is not None

        run_dynamic_subplan(
            plan,
            run_path=tmp_path,
            task_id="planner",
            dry_run=True,
            execution_profile="plan",
        )

        assert len(calls) == 1
        assert calls[0]["dry_run"] is True

    # Extra: verbosity always "normal"
    def test_verbosity_always_normal(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.dynamic import run_dynamic_subplan

        calls = self._mock_run_plan(monkeypatch)
        plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert plan is not None

        run_dynamic_subplan(
            plan,
            run_path=tmp_path,
            task_id="planner",
            dry_run=False,
            execution_profile="plan",
        )

        assert len(calls) == 1
        assert calls[0]["verbosity"] == "normal"

    # Extra: output_mode always "text"
    def test_output_mode_always_text(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.dynamic import run_dynamic_subplan

        calls = self._mock_run_plan(monkeypatch)
        plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert plan is not None

        run_dynamic_subplan(
            plan,
            run_path=tmp_path,
            task_id="planner",
            dry_run=False,
            execution_profile="plan",
        )

        assert len(calls) == 1
        assert calls[0]["output_mode"] == "text"

    # Extra: result is returned from run_plan
    def test_returns_run_plan_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.dynamic import run_dynamic_subplan

        expected = _make_sub_result()
        expected.plan_name = "custom-result"
        self._mock_run_plan(monkeypatch, return_result=expected)
        plan = build_plan_from_output(_make_output(), _make_plan(), _make_task())
        assert plan is not None

        result = run_dynamic_subplan(
            plan,
            run_path=tmp_path,
            task_id="planner",
            dry_run=False,
            execution_profile="plan",
        )

        assert result.plan_name == "custom-result"
