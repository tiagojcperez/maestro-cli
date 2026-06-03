from __future__ import annotations

import json

import pytest

from maestro_cli.models import JudgeSpec, PlanSpec, PolicySpec, PolicyViolation, TaskSpec
from maestro_cli.policy import compile_policy, evaluate_policies, format_violations, format_violations_json


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def plan() -> PlanSpec:
    return PlanSpec(name="test-plan")


@pytest.fixture()
def codex_task() -> TaskSpec:
    return TaskSpec(id="task-codex", engine="codex", timeout_sec=30, tags=["fast"])


@pytest.fixture()
def claude_task() -> TaskSpec:
    return TaskSpec(id="task-claude", engine="claude", timeout_sec=90, tags=["prod", "deploy"])


def _policy(rule: str, action: str = "warn", name: str = "p1", message: str = "") -> PolicySpec:
    return PolicySpec(name=name, rule=rule, action=action, message=message)  # type: ignore[arg-type]


def _task(**kwargs: object) -> TaskSpec:
    kwargs.setdefault("id", "t1")
    return TaskSpec(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestCompilePolicy
# ---------------------------------------------------------------------------


class TestCompilePolicy:
    def test_field_equals(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        ev = compile_policy(_policy('task.engine == "codex"'))
        assert ev(codex_task, plan) is True

    def test_field_equals_no_match(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        ev = compile_policy(_policy('task.engine == "claude"'))
        assert ev(codex_task, plan) is False

    def test_field_not_equals(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        ev = compile_policy(_policy('task.engine != "claude"'))
        assert ev(codex_task, plan) is True

    def test_numeric_less_than(self, plan: PlanSpec) -> None:
        task = _task(id="t-timeout", timeout_sec=30)
        ev = compile_policy(_policy("task.timeout_sec < 60"))
        assert ev(task, plan) is True

    def test_numeric_less_than_no_match(self, plan: PlanSpec) -> None:
        task = _task(id="t-timeout", timeout_sec=120)
        ev = compile_policy(_policy("task.timeout_sec < 60"))
        assert ev(task, plan) is False

    def test_in_tags(self, plan: PlanSpec, claude_task: TaskSpec) -> None:
        ev = compile_policy(_policy('"prod" in task.tags'))
        assert ev(claude_task, plan) is True

    def test_not_in_tags(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        ev = compile_policy(_policy('"prod" not in task.tags'))
        assert ev(codex_task, plan) is True

    def test_not_in_tags_no_match(self, plan: PlanSpec, claude_task: TaskSpec) -> None:
        ev = compile_policy(_policy('"prod" not in task.tags'))
        assert ev(claude_task, plan) is False

    def test_has_judge_true(self, plan: PlanSpec) -> None:
        task = _task(id="t-judge", judge=JudgeSpec(criteria=["test"]))
        ev = compile_policy(_policy("task.has_judge"))
        assert ev(task, plan) is True

    def test_has_judge_false(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        ev = compile_policy(_policy("not task.has_judge"))
        assert ev(codex_task, plan) is True

    def test_has_judge_with_judge_is_false(self, plan: PlanSpec) -> None:
        task = _task(id="t-judge", judge=JudgeSpec(criteria=["test"]))
        ev = compile_policy(_policy("not task.has_judge"))
        assert ev(task, plan) is False

    def test_compound_and(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        ev = compile_policy(_policy('task.engine == "codex" and not task.has_judge'))
        assert ev(codex_task, plan) is True

    def test_compound_and_fails_when_has_judge(self, plan: PlanSpec) -> None:
        task = _task(id="t-codex-judge", engine="codex", judge=JudgeSpec(criteria=["x"]))
        ev = compile_policy(_policy('task.engine == "codex" and not task.has_judge'))
        assert ev(task, plan) is False

    def test_compound_or_first(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        ev = compile_policy(_policy('task.engine == "codex" or task.engine == "gemini"'))
        assert ev(codex_task, plan) is True

    def test_compound_or_second(self, plan: PlanSpec) -> None:
        task = _task(id="t-gemini", engine="gemini")
        ev = compile_policy(_policy('task.engine == "codex" or task.engine == "gemini"'))
        assert ev(task, plan) is True

    def test_compound_or_neither(self, plan: PlanSpec, claude_task: TaskSpec) -> None:
        ev = compile_policy(_policy('task.engine == "codex" or task.engine == "gemini"'))
        assert ev(claude_task, plan) is False

    def test_invalid_syntax_raises(self) -> None:
        spec = _policy("task.engine ==== bad")
        with pytest.raises((ValueError, SyntaxError)):
            compile_policy(spec)

    def test_no_eval_injection(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        # ast.parse succeeds but eval must reject the Call node
        spec = _policy("__import__('os').system('echo hacked')")
        ev = compile_policy(spec)
        with pytest.raises((ValueError, SyntaxError)):
            ev(codex_task, plan)

    def test_unknown_task_field_raises(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        ev = compile_policy(_policy("task.nonexistent_field == 1"))
        with pytest.raises(ValueError, match="unknown task field"):
            ev(codex_task, plan)

    def test_unknown_plan_field_raises(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        ev = compile_policy(_policy('plan.nonexistent == "x"'))
        with pytest.raises(ValueError, match="unknown plan field"):
            ev(codex_task, plan)

    def test_plan_field_name(self, codex_task: TaskSpec) -> None:
        plan = PlanSpec(name="my-plan")
        ev = compile_policy(_policy('plan.name == "my-plan"'))
        assert ev(codex_task, plan) is True

    def test_allow_failure_bool(self, plan: PlanSpec) -> None:
        task = _task(id="t-af", allow_failure=True)
        ev = compile_policy(_policy("task.allow_failure"))
        assert ev(task, plan) is True

    def test_max_retries_gte(self, plan: PlanSpec) -> None:
        task = _task(id="t-retries", max_retries=3)
        ev = compile_policy(_policy("task.max_retries >= 2"))
        assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# TestEvaluatePolicies
# ---------------------------------------------------------------------------


class TestEvaluatePolicies:
    def test_no_policies(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        violations = evaluate_policies([], codex_task, plan)
        assert violations == []

    def test_block_violation(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        policies = [_policy('task.engine == "codex"', action="block", name="no-codex")]
        violations = evaluate_policies(policies, codex_task, plan)
        assert len(violations) == 1
        assert violations[0].action == "block"
        assert violations[0].policy_name == "no-codex"
        assert violations[0].task_id == "task-codex"

    def test_warn_violation(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        policies = [_policy('task.engine == "codex"', action="warn", name="warn-codex")]
        violations = evaluate_policies(policies, codex_task, plan)
        assert len(violations) == 1
        assert violations[0].action == "warn"

    def test_audit_violation(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        policies = [_policy('task.engine == "codex"', action="audit", name="audit-codex")]
        violations = evaluate_policies(policies, codex_task, plan)
        assert len(violations) == 1
        assert violations[0].action == "audit"

    def test_no_match_no_violation(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        policies = [_policy('task.engine == "claude"', action="block", name="no-claude")]
        violations = evaluate_policies(policies, codex_task, plan)
        assert violations == []

    def test_multiple_policies_one_matches(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        policies = [
            _policy('task.engine == "codex"', action="warn", name="p-codex"),
            _policy('task.engine == "claude"', action="block", name="p-claude"),
        ]
        violations = evaluate_policies(policies, codex_task, plan)
        assert len(violations) == 1
        assert violations[0].policy_name == "p-codex"

    def test_multiple_policies_all_match(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        policies = [
            _policy('task.engine == "codex"', action="warn", name="p1"),
            _policy("not task.has_judge", action="audit", name="p2"),
        ]
        violations = evaluate_policies(policies, codex_task, plan)
        assert len(violations) == 2

    def test_multiple_policies_none_match(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        policies = [
            _policy('task.engine == "claude"', action="block", name="p1"),
            _policy('"prod" in task.tags', action="warn", name="p2"),
        ]
        violations = evaluate_policies(policies, codex_task, plan)
        assert violations == []

    def test_bad_rule_skipped(self, plan: PlanSpec, codex_task: TaskSpec, capsys: pytest.CaptureFixture[str]) -> None:
        policies = [
            _policy("task.engine ==== bad", action="block", name="bad-rule"),
            _policy('task.engine == "codex"', action="warn", name="good-rule"),
        ]
        violations = evaluate_policies(policies, codex_task, plan)
        # bad-rule skipped, good-rule still evaluated
        assert len(violations) == 1
        assert violations[0].policy_name == "good-rule"
        captured = capsys.readouterr()
        assert "bad-rule" in captured.out

    def test_custom_message_used(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        policies = [_policy('task.engine == "codex"', name="p1", message="custom violation message")]
        violations = evaluate_policies(policies, codex_task, plan)
        assert violations[0].message == "custom violation message"

    def test_default_message_generated(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        policies = [_policy('task.engine == "codex"', name="p1", message="")]
        violations = evaluate_policies(policies, codex_task, plan)
        msg = violations[0].message
        assert "p1" in msg
        assert "task-codex" in msg

    def test_with_result_cost_usd(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        from maestro_cli.models import TaskResult
        result = TaskResult(task_id="task-codex", status="success", cost_usd=5.0)
        policies = [_policy("task.cost_usd > 3.0", action="warn", name="cost-check")]
        violations = evaluate_policies(policies, codex_task, plan, result=result)
        assert len(violations) == 1

    def test_without_result_cost_defaults_zero(self, plan: PlanSpec, codex_task: TaskSpec) -> None:
        policies = [_policy("task.cost_usd > 3.0", action="warn", name="cost-check")]
        violations = evaluate_policies(policies, codex_task, plan, result=None)
        assert violations == []

    def test_unknown_field_policy_skipped(self, plan: PlanSpec, codex_task: TaskSpec, capsys: pytest.CaptureFixture[str]) -> None:
        policies = [
            _policy("task.totally_unknown == 1", action="block", name="bad-field"),
            _policy('task.engine == "codex"', action="warn", name="good"),
        ]
        violations = evaluate_policies(policies, codex_task, plan)
        assert len(violations) == 1
        assert violations[0].policy_name == "good"
        captured = capsys.readouterr()
        assert "bad-field" in captured.out


# ---------------------------------------------------------------------------
# TestFormatViolations
# ---------------------------------------------------------------------------


class TestFormatViolations:
    def _violation(
        self,
        policy_name: str = "my-policy",
        task_id: str = "task-1",
        action: str = "warn",
        message: str = "something violated",
    ) -> PolicyViolation:
        return PolicyViolation(
            policy_name=policy_name,
            task_id=task_id,
            action=action,  # type: ignore[arg-type]
            message=message,
        )

    def test_human_format_contains_policy_name(self) -> None:
        v = self._violation(policy_name="no-codex")
        output = format_violations([v])
        assert "no-codex" in output

    def test_human_format_contains_action(self) -> None:
        v = self._violation(action="block")
        output = format_violations([v])
        assert "block" in output

    def test_human_format_contains_task_id(self) -> None:
        v = self._violation(task_id="my-task")
        output = format_violations([v])
        assert "my-task" in output

    def test_human_format_multiple_violations(self) -> None:
        violations = [
            self._violation(policy_name="p1", task_id="t1"),
            self._violation(policy_name="p2", task_id="t2"),
        ]
        output = format_violations(violations)
        assert "p1" in output
        assert "p2" in output
        assert "t1" in output
        assert "t2" in output

    def test_human_format_empty(self) -> None:
        output = format_violations([])
        assert output == ""

    def test_json_format_valid_json(self) -> None:
        v = self._violation()
        output = format_violations_json([v])
        parsed = json.loads(output)
        assert isinstance(parsed, list)
        assert len(parsed) == 1

    def test_json_format_fields(self) -> None:
        v = self._violation(policy_name="p1", task_id="t1", action="audit", message="msg")
        parsed = json.loads(format_violations_json([v]))
        item = parsed[0]
        assert item["policy_name"] == "p1"
        assert item["task_id"] == "t1"
        assert item["action"] == "audit"
        assert item["message"] == "msg"

    def test_json_format_empty(self) -> None:
        output = format_violations_json([])
        parsed = json.loads(output)
        assert parsed == []

    def test_json_format_multiple(self) -> None:
        violations = [
            self._violation(policy_name="p1"),
            self._violation(policy_name="p2"),
        ]
        parsed = json.loads(format_violations_json(violations))
        assert len(parsed) == 2
        names = {item["policy_name"] for item in parsed}
        assert names == {"p1", "p2"}


# ---------------------------------------------------------------------------
# Additional compile_policy tests — operator coverage
# ---------------------------------------------------------------------------

class TestCompilePolicyOperatorCoverage:
    """Comprehensive coverage of all whitelisted operators."""

    def test_less_than_or_equal(self, plan: PlanSpec) -> None:
        task = _task(id="t1", timeout_sec=60)
        ev = compile_policy(_policy("task.timeout_sec <= 60"))
        assert ev(task, plan) is True

    def test_less_than_or_equal_exceeds(self, plan: PlanSpec) -> None:
        task = _task(id="t1", timeout_sec=61)
        ev = compile_policy(_policy("task.timeout_sec <= 60"))
        assert ev(task, plan) is False

    def test_greater_than(self, plan: PlanSpec) -> None:
        task = _task(id="t1", timeout_sec=120)
        ev = compile_policy(_policy("task.timeout_sec > 60"))
        assert ev(task, plan) is True

    def test_greater_than_no_match(self, plan: PlanSpec) -> None:
        task = _task(id="t1", timeout_sec=30)
        ev = compile_policy(_policy("task.timeout_sec > 60"))
        assert ev(task, plan) is False

    def test_greater_than_or_equal(self, plan: PlanSpec) -> None:
        task = _task(id="t1", timeout_sec=60)
        ev = compile_policy(_policy("task.timeout_sec >= 60"))
        assert ev(task, plan) is True

    def test_greater_than_or_equal_below(self, plan: PlanSpec) -> None:
        task = _task(id="t1", timeout_sec=59)
        ev = compile_policy(_policy("task.timeout_sec >= 60"))
        assert ev(task, plan) is False

    def test_not_operator(self, plan: PlanSpec) -> None:
        task = _task(id="t1", allow_failure=False)
        ev = compile_policy(_policy("not task.allow_failure"))
        assert ev(task, plan) is True

    def test_not_operator_true_value(self, plan: PlanSpec) -> None:
        task = _task(id="t1", allow_failure=True)
        ev = compile_policy(_policy("not task.allow_failure"))
        assert ev(task, plan) is False

    def test_in_with_none_right(self, plan: PlanSpec) -> None:
        """'in' with right=None should return False (not crash)."""
        task = _task(id="t1", tags=[])
        ev = compile_policy(_policy('"x" in task.model'))
        # task.model is None by default
        assert ev(task, plan) is False

    def test_not_in_with_none_right(self, plan: PlanSpec) -> None:
        """'not in' with right=None should return False (conservative)."""
        task = _task(id="t1")
        ev = compile_policy(_policy('"x" not in task.model'))
        # task.model is None → right is None → returns False
        assert ev(task, plan) is False

    def test_chained_comparison(self, plan: PlanSpec) -> None:
        """Python chained comparisons like 30 < timeout_sec < 120."""
        task = _task(id="t1", timeout_sec=60)
        ev = compile_policy(_policy("30 < task.timeout_sec < 120"))
        assert ev(task, plan) is True

    def test_chained_comparison_out_of_range(self, plan: PlanSpec) -> None:
        task = _task(id="t1", timeout_sec=150)
        ev = compile_policy(_policy("30 < task.timeout_sec < 120"))
        assert ev(task, plan) is False

    def test_constant_true(self, plan: PlanSpec) -> None:
        task = _task(id="t1")
        ev = compile_policy(_policy("True"))
        assert ev(task, plan) is True

    def test_constant_false(self, plan: PlanSpec) -> None:
        task = _task(id="t1")
        ev = compile_policy(_policy("False"))
        assert ev(task, plan) is False

    def test_string_equality(self, plan: PlanSpec) -> None:
        task = _task(id="my-task")
        ev = compile_policy(_policy('task.id == "my-task"'))
        assert ev(task, plan) is True

    def test_string_not_equals(self, plan: PlanSpec) -> None:
        task = _task(id="my-task")
        ev = compile_policy(_policy('task.id != "other"'))
        assert ev(task, plan) is True

    def test_integer_constant_comparison(self, plan: PlanSpec) -> None:
        task = _task(id="t1", max_retries=2)
        ev = compile_policy(_policy("task.max_retries == 2"))
        assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# Additional compile_policy tests — context_trust field
# ---------------------------------------------------------------------------

class TestPolicyContextTrust:
    def test_context_trust_untrusted(self, plan: PlanSpec) -> None:
        task = _task(id="t1", context_trust="untrusted")
        ev = compile_policy(_policy('task.context_trust == "untrusted"'))
        assert ev(task, plan) is True

    def test_context_trust_trusted(self, plan: PlanSpec) -> None:
        task = _task(id="t1", context_trust="trusted")
        ev = compile_policy(_policy('task.context_trust == "trusted"'))
        assert ev(task, plan) is True

    def test_context_trust_none_by_default(self, plan: PlanSpec) -> None:
        task = _task(id="t1")
        ev = compile_policy(_policy('task.context_trust == "untrusted"'))
        assert ev(task, plan) is False

    def test_context_trust_not_none_check(self, plan: PlanSpec) -> None:
        """A task with context_trust set should not be None."""
        task = _task(id="t1", context_trust="untrusted")
        # Not equals None — but we check for something truthy
        ev = compile_policy(_policy('task.context_trust != "trusted"'))
        assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# Additional compile_policy tests — plan-level fields
# ---------------------------------------------------------------------------

class TestPolicyPlanFields:
    def test_plan_name(self) -> None:
        plan = PlanSpec(name="my-plan")
        task = _task(id="t1")
        ev = compile_policy(_policy('plan.name == "my-plan"'))
        assert ev(task, plan) is True

    def test_plan_max_cost_usd_none(self) -> None:
        plan = PlanSpec(name="p")
        task = _task(id="t1")
        # max_cost_usd is None by default; checking if it is None
        ev = compile_policy(_policy("plan.max_cost_usd == None"))
        # ast.Constant(None)
        assert ev(task, plan) is True

    def test_plan_max_cost_usd_set(self) -> None:
        plan = PlanSpec(name="p", max_cost_usd=10.0)
        task = _task(id="t1")
        ev = compile_policy(_policy("plan.max_cost_usd > 5.0"))
        assert ev(task, plan) is True

    def test_plan_max_parallel(self) -> None:
        plan = PlanSpec(name="p", max_parallel=4)
        task = _task(id="t1")
        ev = compile_policy(_policy("plan.max_parallel >= 4"))
        assert ev(task, plan) is True

    def test_plan_fail_fast(self) -> None:
        plan = PlanSpec(name="p", fail_fast=True)
        task = _task(id="t1")
        ev = compile_policy(_policy("plan.fail_fast"))
        assert ev(task, plan) is True

    def test_plan_fail_fast_false(self) -> None:
        plan = PlanSpec(name="p", fail_fast=False)
        task = _task(id="t1")
        ev = compile_policy(_policy("not plan.fail_fast"))
        assert ev(task, plan) is True

    def test_plan_execution_profile_defaults_to_plan(self) -> None:
        """PlanSpec doesn't have execution_profile; getattr returns 'plan'."""
        plan = PlanSpec(name="p")
        task = _task(id="t1")
        ev = compile_policy(_policy('task.execution_profile == "plan"'))
        assert ev(task, plan) is True

    def test_unknown_plan_field_raises(self) -> None:
        plan = PlanSpec(name="p")
        task = _task(id="t1")
        ev = compile_policy(_policy("plan.unknown_field == 1"))
        with pytest.raises(ValueError, match="unknown plan field"):
            ev(task, plan)


# ---------------------------------------------------------------------------
# Additional compile_policy tests — forbidden / invalid AST
# ---------------------------------------------------------------------------

class TestPolicyForbiddenAST:
    def test_function_call_rejected(self, plan: PlanSpec) -> None:
        task = _task(id="t1")
        ev = compile_policy(_policy("len(task.tags) > 0"))
        with pytest.raises(ValueError, match="unsupported AST node"):
            ev(task, plan)

    def test_import_rejected(self) -> None:
        with pytest.raises(SyntaxError):
            compile_policy(_policy("import os"))

    def test_lambda_rejected(self, plan: PlanSpec) -> None:
        task = _task(id="t1")
        ev = compile_policy(_policy("(lambda: True)()"))
        with pytest.raises(ValueError):
            ev(task, plan)

    def test_nested_attribute_rejected(self, plan: PlanSpec) -> None:
        task = _task(id="t1")
        ev = compile_policy(_policy("task.tags.append"))
        with pytest.raises(ValueError, match="nested attribute access"):
            ev(task, plan)

    def test_unknown_object_name_rejected(self, plan: PlanSpec) -> None:
        task = _task(id="t1")
        ev = compile_policy(_policy('env.PATH == "/usr/bin"'))
        with pytest.raises(ValueError, match="unknown object"):
            ev(task, plan)

    def test_unsupported_name_rejected(self, plan: PlanSpec) -> None:
        task = _task(id="t1")
        ev = compile_policy(_policy("some_variable == 1"))
        with pytest.raises(ValueError, match="unsupported name"):
            ev(task, plan)

    def test_bare_true_name_accepted(self, plan: PlanSpec) -> None:
        task = _task(id="t1")
        ev = compile_policy(_policy("true"))
        assert ev(task, plan) is True

    def test_bare_false_name_accepted(self, plan: PlanSpec) -> None:
        task = _task(id="t1")
        ev = compile_policy(_policy("false"))
        assert ev(task, plan) is False

    def test_empty_rule_raises(self) -> None:
        with pytest.raises(SyntaxError):
            compile_policy(_policy(""))


# ---------------------------------------------------------------------------
# Additional evaluate_policies tests
# ---------------------------------------------------------------------------

class TestEvaluatePoliciesAdditional:
    def test_multiple_violations_for_same_task(self, plan: PlanSpec) -> None:
        task = _task(id="t1", engine="codex", timeout_sec=10)
        policies = [
            _policy('task.engine == "codex"', action="warn", name="p-engine"),
            _policy("task.timeout_sec < 30", action="audit", name="p-timeout"),
        ]
        violations = evaluate_policies(policies, task, plan)
        assert len(violations) == 2
        names = {v.policy_name for v in violations}
        assert names == {"p-engine", "p-timeout"}

    def test_block_action_in_violations(self, plan: PlanSpec) -> None:
        task = _task(id="t1", engine="codex")
        policies = [_policy('task.engine == "codex"', action="block", name="no-codex")]
        violations = evaluate_policies(policies, task, plan)
        assert violations[0].action == "block"

    def test_cost_usd_with_result_none_cost(self, plan: PlanSpec) -> None:
        """When result exists but cost_usd is None, defaults to 0.0."""
        from maestro_cli.models import TaskResult

        task = _task(id="t1")
        result = TaskResult(task_id="t1", status="success", cost_usd=None)
        policies = [_policy("task.cost_usd > 0", action="warn", name="cost")]
        violations = evaluate_policies(policies, task, plan, result=result)
        assert violations == []

    def test_dynamic_group_field(self, plan: PlanSpec) -> None:
        task = _task(id="t1", dynamic_group=True)
        ev = compile_policy(_policy("task.dynamic_group"))
        assert ev(task, plan) is True

    def test_dynamic_group_false(self, plan: PlanSpec) -> None:
        task = _task(id="t1", dynamic_group=False)
        ev = compile_policy(_policy("not task.dynamic_group"))
        assert ev(task, plan) is True

    def test_description_field(self, plan: PlanSpec) -> None:
        task = _task(id="t1", description="My task description")
        ev = compile_policy(_policy('task.description == "My task description"'))
        assert ev(task, plan) is True

    def test_cache_field_true(self, plan: PlanSpec) -> None:
        task = _task(id="t1", cache=True)
        ev = compile_policy(_policy("task.cache"))
        assert ev(task, plan) is True

    def test_cache_field_false(self, plan: PlanSpec) -> None:
        task = _task(id="t1", cache=False)
        ev = compile_policy(_policy("not task.cache"))
        assert ev(task, plan) is True

    def test_requires_approval_field(self, plan: PlanSpec) -> None:
        task = _task(id="t1", requires_approval=True)
        ev = compile_policy(_policy("task.requires_approval"))
        assert ev(task, plan) is True

    def test_syntax_error_policy_skipped_gracefully(
        self, plan: PlanSpec, capsys: pytest.CaptureFixture[str],
    ) -> None:
        task = _task(id="t1")
        policies = [
            _policy("this is not valid python", action="block", name="broken"),
        ]
        violations = evaluate_policies(policies, task, plan)
        assert violations == []
        captured = capsys.readouterr()
        assert "broken" in captured.out

    def test_runtime_error_policy_skipped_gracefully(
        self, plan: PlanSpec, capsys: pytest.CaptureFixture[str],
    ) -> None:
        task = _task(id="t1")
        policies = [
            _policy("task.nonexistent == 1", action="block", name="bad-field"),
        ]
        violations = evaluate_policies(policies, task, plan)
        assert violations == []
        captured = capsys.readouterr()
        assert "bad-field" in captured.out


# ---------------------------------------------------------------------------
# Additional format tests
# ---------------------------------------------------------------------------

class TestFormatViolationsAdditional:
    def _violation(
        self,
        policy_name: str = "p",
        task_id: str = "t",
        action: str = "warn",
        message: str = "msg",
    ) -> PolicyViolation:
        return PolicyViolation(
            policy_name=policy_name,
            task_id=task_id,
            action=action,  # type: ignore[arg-type]
            message=message,
        )

    def test_human_format_contains_message(self) -> None:
        v = self._violation(message="custom message text")
        output = format_violations([v])
        assert "custom message text" in output

    def test_human_format_prefix(self) -> None:
        v = self._violation()
        output = format_violations([v])
        assert output.startswith("[maestro policy]")

    def test_json_format_to_dict_fields(self) -> None:
        v = self._violation(policy_name="abc", task_id="xyz", action="block", message="blk")
        parsed = json.loads(format_violations_json([v]))
        item = parsed[0]
        assert item["policy_name"] == "abc"
        assert item["task_id"] == "xyz"
        assert item["action"] == "block"
        assert item["message"] == "blk"

    def test_multiple_violations_newline_separated(self) -> None:
        violations = [
            self._violation(policy_name="p1", task_id="t1"),
            self._violation(policy_name="p2", task_id="t2"),
            self._violation(policy_name="p3", task_id="t3"),
        ]
        output = format_violations(violations)
        lines = output.strip().split("\n")
        assert len(lines) == 3
