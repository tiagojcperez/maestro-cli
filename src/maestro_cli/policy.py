from __future__ import annotations

import ast
import json
import operator
from typing import Any, Callable

from .models import (
    PolicyAction,
    PolicySpec,
    PolicyViolation,
    PlanSpec,
    TaskResult,
    TaskSpec,
)

# Whitelisted task fields accessible via `task.<field>` in policy rules
_SAFE_TASK_FIELDS: frozenset[str] = frozenset(
    {
        "id",
        "engine",
        "model",
        "tags",
        "timeout_sec",
        "max_retries",
        "allow_failure",
        "requires_approval",
        "cache",
        "description",
        # computed
        "cost_usd",
        "has_judge",
        "execution_profile",
        "dynamic_group",
        "context_trust",
        "contract_type",
        "has_consistency_group",
        "allowed_tools",
        "has_allowed_tools",
    }
)

# Whitelisted plan fields accessible via `plan.<field>` in policy rules
_SAFE_PLAN_FIELDS: frozenset[str] = frozenset(
    {
        "name",
        "max_cost_usd",
        "max_parallel",
        "execution_profile",
        "fail_fast",
    }
)

_OPS: dict[type, Callable[[Any, Any], bool]] = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


class _SafeEvaluator:
    """Walk an AST expression and evaluate it against task/plan context."""

    def __init__(
        self,
        task: TaskSpec,
        plan: PlanSpec,
        result: TaskResult | None,
    ) -> None:
        self._task = task
        self._plan = plan
        self._result = result

    def _resolve_task_field(self, field: str) -> Any:
        if field not in _SAFE_TASK_FIELDS:
            raise ValueError(f"Policy rule references unknown task field '{field}'")
        if field == "cost_usd":
            return self._result.cost_usd if self._result and self._result.cost_usd is not None else 0.0
        if field == "has_judge":
            return self._task.judge is not None
        if field == "execution_profile":
            return getattr(self._plan, "execution_profile", "plan") or "plan"
        if field == "contract_type":
            return self._task.contract_type or ""
        if field == "has_consistency_group":
            return bool(self._task.consistency_group)
        if field == "has_allowed_tools":
            return self._task.allowed_tools is not None
        if field == "allowed_tools":
            return self._task.allowed_tools or []
        return getattr(self._task, field, None)

    def _resolve_plan_field(self, field: str) -> Any:
        if field not in _SAFE_PLAN_FIELDS:
            raise ValueError(f"Policy rule references unknown plan field '{field}'")
        return getattr(self._plan, field, None)

    def eval(self, node: ast.expr) -> Any:  # noqa: A003
        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            name = node.id
            if name in ("True", "true"):
                return True
            if name in ("False", "false"):
                return False
            raise ValueError(f"Policy rule references unsupported name '{name}'")

        if isinstance(node, ast.Attribute):
            if not isinstance(node.value, ast.Name):
                raise ValueError("Policy rule: nested attribute access is not allowed")
            obj_name = node.value.id
            if obj_name == "task":
                return self._resolve_task_field(node.attr)
            if obj_name == "plan":
                return self._resolve_plan_field(node.attr)
            raise ValueError(f"Policy rule: unknown object '{obj_name}' (use 'task' or 'plan')")

        if isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.Not):
                return not self.eval(node.operand)
            raise ValueError(f"Policy rule: unsupported unary op '{type(node.op).__name__}'")

        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self.eval(v) for v in node.values)
            if isinstance(node.op, ast.Or):
                return any(self.eval(v) for v in node.values)
            raise ValueError(f"Policy rule: unsupported bool op '{type(node.op).__name__}'")

        if isinstance(node, ast.Compare):
            left = self.eval(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = self.eval(comparator)
                if isinstance(op, tuple(_OPS.keys())):
                    if not _OPS[type(op)](left, right):
                        return False
                    left = right
                elif isinstance(op, ast.In):
                    if right is None or left not in right:
                        return False
                    left = right
                elif isinstance(op, ast.NotIn):
                    if right is None or left in right:
                        return False
                    left = right
                else:
                    raise ValueError(f"Policy rule: unsupported comparator '{type(op).__name__}'")
            return True

        raise ValueError(f"Policy rule: unsupported AST node '{type(node).__name__}'")


def compile_policy(
    spec: PolicySpec,
) -> Callable[[TaskSpec, PlanSpec, TaskResult | None], bool]:
    """Parse spec.rule via ast and return a safe evaluator callable.

    Returns a function that returns True when the rule matches (= violation detected).
    Raises ValueError for unsupported AST nodes or invalid expressions.
    """
    tree = ast.parse(spec.rule, mode="eval")
    body = tree.body

    def _evaluate(task: TaskSpec, plan: PlanSpec, result: TaskResult | None = None) -> bool:
        evaluator = _SafeEvaluator(task, plan, result)
        return bool(evaluator.eval(body))

    return _evaluate


def evaluate_policies(
    policies: list[PolicySpec],
    task: TaskSpec,
    plan: PlanSpec,
    result: TaskResult | None = None,
) -> list[PolicyViolation]:
    """Compile and evaluate each policy against the given task.

    Policies that match (rule returns True) produce a PolicyViolation.
    SyntaxError / ValueError per policy are caught and logged as warnings —
    execution continues.
    """
    violations: list[PolicyViolation] = []
    for spec in policies:
        try:
            evaluator = compile_policy(spec)
            matched = evaluator(task, plan, result)
        except (SyntaxError, ValueError) as exc:
            print(f"[maestro] warning: policy '{spec.name}' rule error: {exc}")
            continue
        if matched:
            msg = spec.message or f"policy '{spec.name}' matched for task '{task.id}'"
            violations.append(
                PolicyViolation(
                    policy_name=spec.name,
                    task_id=task.id,
                    action=spec.action,
                    message=msg,
                )
            )
    return violations


def format_violations(violations: list[PolicyViolation]) -> str:
    """Return a human-readable string for a list of policy violations."""
    lines = [
        f"[maestro policy] {v.action}: [{v.policy_name}] {v.message} (task: {v.task_id})"
        for v in violations
    ]
    return "\n".join(lines)


def format_violations_json(violations: list[PolicyViolation]) -> str:
    """Return JSON representation of policy violations."""
    return json.dumps([v.to_dict() for v in violations], indent=2)
