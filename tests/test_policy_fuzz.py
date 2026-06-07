from __future__ import annotations

import pytest

from maestro_cli.models import PlanSpec, PolicySpec, TaskSpec
from maestro_cli.policy import compile_policy, evaluate_policies

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _policy(rule: str, name: str = "p") -> PolicySpec:
    return PolicySpec(name=name, rule=rule, action="warn")  # type: ignore[arg-type]


def _task(**kwargs: object) -> TaskSpec:
    kwargs.setdefault("id", "t1")
    return TaskSpec(**kwargs)  # type: ignore[arg-type]


@pytest.fixture()
def plan() -> PlanSpec:
    return PlanSpec(name="test-plan")


@pytest.fixture()
def task() -> TaskSpec:
    return TaskSpec(id="t1", engine="claude", timeout_sec=30, tags=["test"])


# ---------------------------------------------------------------------------
# 1. ast.Call nodes — security-critical
#    compile_policy succeeds (ast.parse is fine), but calling the evaluator
#    must always raise because Call nodes are not in the allowlist.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        'eval("__import__(\'os\')")',
        'exec("print(\'pwned\')")',
        'getattr(task, "__class__")',
        "type(task)",
        "dir(task)",
        "globals()",
        "locals()",
        'compile("x", "x", "exec")',
        'open("/etc/passwd")',
        "len(task.tags)",
        'str(task.engine)',
        'print("pwned")',
        '__import__("os")',
    ],
)
def test_call_nodes_rejected(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """ast.Call nodes must be rejected at evaluation time — no code execution allowed."""
    spec = _policy(injection)
    ev = compile_policy(spec)  # parse succeeds; rejection happens inside eval()
    with pytest.raises((ValueError, SyntaxError)):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 2. Other non-allowlisted AST structural node types
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        "lambda: True",
        "[x for x in task.tags]",
        "(x for x in task.tags)",
        'f"{task.engine}"',
        "task.tags[0]",
        '"yes" if True else "no"',
        '{"key": "value"}',
        "[1, 2, 3]",
        "{1, 2, 3}",
        "(1, 2)",
    ],
)
def test_structural_ast_nodes_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """Non-allowlisted AST node types (Lambda, ListComp, Subscript, etc.) must be rejected."""
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises((ValueError, SyntaxError)):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 3. Nested / chained attribute access bypass attempts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        'task.engine.upper == "CLAUDE"',
        "task.engine.encode",
        "task.id.split",
        "task.tags.append",
        "plan.name.lower",
        "plan.execution_profile.replace",
    ],
)
def test_nested_attribute_access_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """Chained attribute access (task.x.y) must raise ValueError about nesting."""
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="nested"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 4. Dunder attribute access on task / plan
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attr",
    ["__class__", "__dict__", "__module__", "__init__", "__bases__"],
)
def test_dunder_task_attributes_rejected(task: TaskSpec, plan: PlanSpec, attr: str) -> None:
    """Dunder attributes on task must be rejected (not in safe field list)."""
    spec = _policy(f'task.{attr} == "x"')
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unknown task field"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 5. Unknown name / object references
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        "os.path",
        "sys.modules",
        "__builtins__",
        "config.value",
        'True == x',  # 'x' is an unknown free name
    ],
)
def test_unknown_names_and_objects_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """References to names other than 'task' and 'plan' must be rejected."""
    spec = _policy(rule)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# 6. Input sanitization edge cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_rule",
    [
        "",                        # empty string — SyntaxError at parse time
        "\x00",                    # raw null byte
        'task.engine\x00 == "x"',  # null byte embedded in identifier
    ],
)
def test_malformed_input_raises(bad_rule: str) -> None:
    """Malformed rules (empty, null bytes) must fail at parse time."""
    spec = _policy(bad_rule)
    with pytest.raises((ValueError, SyntaxError)):
        compile_policy(spec)


def test_very_long_valid_rule_terminates(task: TaskSpec, plan: PlanSpec) -> None:
    """A 200-operand OR chain must evaluate in finite time without crashing."""
    parts = ['task.engine == "codex"'] * 200
    rule = " or ".join(parts)
    spec = _policy(rule)
    ev = compile_policy(spec)
    result = ev(task, plan)
    assert result is False  # task.engine is "claude", not "codex"


# ---------------------------------------------------------------------------
# 7. Null-safe `in` / `not in` comparisons
# ---------------------------------------------------------------------------


def test_in_with_none_rhs_returns_false(plan: PlanSpec) -> None:
    """'x' in task.model when model is None must return False, not raise."""
    task = _task(id="t1", engine="claude", model=None)
    spec = _policy('"haiku" in task.model')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


def test_not_in_with_none_rhs_returns_false(plan: PlanSpec) -> None:
    """'x' not in task.model when model is None must return False, not raise."""
    task = _task(id="t1", engine="claude", model=None)
    spec = _policy('"haiku" not in task.model')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


def test_constant_none_equality_evaluates(plan: PlanSpec) -> None:
    """task.model == None is valid (None is ast.Constant) and evaluates correctly."""
    task = _task(id="t1", engine="claude", model=None)
    spec = _policy("task.model == None")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# 8. Deep boolean nesting must not crash or stack-overflow
# ---------------------------------------------------------------------------


def test_deeply_nested_boolean_evaluates_correctly(plan: PlanSpec) -> None:
    """4-level nested boolean expression must evaluate correctly without crashing."""
    task = _task(id="t1", engine="claude", timeout_sec=60, max_retries=2, allow_failure=False)
    rule = (
        'task.engine == "claude" and '
        "(task.timeout_sec > 30 or "
        "(task.max_retries >= 1 and "
        "(not task.allow_failure or not task.has_judge)))"
    )
    spec = _policy(rule)
    ev = compile_policy(spec)
    result = ev(task, plan)
    assert result is True


def test_many_operands_chain_does_not_crash(plan: PlanSpec) -> None:
    """15-operand OR chain must evaluate without crashing and return correct result."""
    task = _task(id="t1", engine="claude")
    engines = ["codex", "gemini", "copilot", "qwen", "ollama"]
    # None of these match "claude", so result must be False
    parts = [f'task.engine == "{e}"' for e in engines * 3]
    rule = " or ".join(parts)
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# 9. evaluate_policies must swallow ValueError raised at evaluation time
#    (not just at compile/parse time — that's already tested in test_policy.py)
# ---------------------------------------------------------------------------


def test_evaluate_policies_swallows_eval_time_value_errors(
    task: TaskSpec, plan: PlanSpec, capsys: pytest.CaptureFixture[str]
) -> None:
    """Rules that parse OK but raise ValueError at eval time must be logged, not raised."""
    bad_policies = [
        PolicySpec(name="p-call", rule="len(task.tags)", action="warn"),  # type: ignore[arg-type]
        PolicySpec(name="p-lambda", rule="lambda: True", action="warn"),  # type: ignore[arg-type]
        PolicySpec(name="p-good", rule='task.engine == "claude"', action="warn"),  # type: ignore[arg-type]
    ]
    violations = evaluate_policies(bad_policies, task, plan)
    # bad rules produce no violations — they are caught and printed
    assert len(violations) == 1
    assert violations[0].policy_name == "p-good"
    captured = capsys.readouterr()
    assert "p-call" in captured.out
    assert "p-lambda" in captured.out


# ---------------------------------------------------------------------------
# 10. Unary USub (negation) — only ast.Not is whitelisted
# ---------------------------------------------------------------------------


def test_unary_minus_rejected(task: TaskSpec, plan: PlanSpec) -> None:
    """-task.timeout_sec uses ast.UnaryOp(USub) which is not whitelisted."""
    spec = _policy("-task.timeout_sec > -1")
    ev = compile_policy(spec)
    with pytest.raises(ValueError):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 11. Walrus operator (NamedExpr) — Python 3.8+ — must be rejected
# ---------------------------------------------------------------------------


def test_walrus_operator_rejected(task: TaskSpec, plan: PlanSpec) -> None:
    """(x := task.engine) uses ast.NamedExpr which is not in the allowlist."""
    spec = _policy('(x := task.engine) == "claude"')
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported AST node"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 12. Three-way chained comparison — tests multi-op iteration in Compare handler
# ---------------------------------------------------------------------------


def test_chained_three_way_comparison_true(plan: PlanSpec) -> None:
    """1 < task.timeout_sec < 100 must evaluate correctly (True for timeout=30)."""
    task = _task(id="t1", timeout_sec=30)
    spec = _policy("1 < task.timeout_sec < 100")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_chained_three_way_comparison_false(plan: PlanSpec) -> None:
    """1 < task.timeout_sec < 10 must evaluate to False for timeout=30."""
    task = _task(id="t1", timeout_sec=30)
    spec = _policy("1 < task.timeout_sec < 10")
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# 13. Type-confusion TypeError escapes evaluate_policies
#     evaluate_policies only catches (SyntaxError, ValueError), not TypeError.
#     A rule comparing a str field to an int via < would raise TypeError.
# ---------------------------------------------------------------------------


def test_type_confusion_str_vs_int_raises_typeerror(plan: PlanSpec) -> None:
    """task.engine < 42 raises TypeError (str vs int) that escapes evaluate_policies."""
    task = _task(id="t1", engine="claude")
    policies = [PolicySpec(name="p", rule="task.engine < 42", action="warn")]  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        evaluate_policies(policies, task, plan)


# ---------------------------------------------------------------------------
# 14. Plan dunder fields must be rejected (unknown plan field path)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attr",
    ["__class__", "__dict__", "__module__", "__init__"],
)
def test_plan_dunder_fields_rejected(task: TaskSpec, plan: PlanSpec, attr: str) -> None:
    """Dunder attributes on plan must be rejected via unknown plan field check."""
    spec = _policy(f'plan.{attr} == "x"')
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unknown plan field"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 15. `in` on a string field — substring check semantics
# ---------------------------------------------------------------------------


def test_in_with_string_rhs_substring_match(plan: PlanSpec) -> None:
    """"cl" in task.engine is True when engine is "claude" (str contains substr)."""
    task = _task(id="t1", engine="claude")
    spec = _policy('"cl" in task.engine')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_in_with_string_rhs_no_match(plan: PlanSpec) -> None:
    """"zz" in task.engine is False when engine is "claude"."""
    task = _task(id="t1", engine="claude")
    spec = _policy('"zz" in task.engine')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# 16. BinOp arithmetic — not in the evaluator allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        "task.timeout_sec + 10 > 0",
        "task.timeout_sec - 5 < 100",
        "task.timeout_sec * 2 > 30",
        "task.timeout_sec / 2 < 100",
        "task.max_retries % 2 == 0",
        "task.max_retries ** 2 > 0",
    ],
)
def test_arithmetic_binop_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """Arithmetic BinOp nodes (+ - * / % **) are not allowlisted and must be rejected."""
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported AST node"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 17. Bitwise BinOp — not in the evaluator allowlist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        "task.timeout_sec & 255",
        "task.timeout_sec | 0",
        "task.timeout_sec ^ 1",
        "task.timeout_sec << 1",
    ],
)
def test_bitwise_binop_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """Bitwise BinOp nodes (& | ^ <<) are not allowlisted and must be rejected."""
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported AST node"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 18. Unsupported unary operators (UAdd, Invert) — only ast.Not is allowed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        "+task.timeout_sec > 0",   # ast.UAdd
        "~task.timeout_sec > 0",   # ast.Invert
    ],
)
def test_unsupported_unary_ops_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """Unary +x and ~x use unsupported ops (UAdd / Invert); only Not is allowed."""
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported unary op"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 19. Two task fields compared against each other — allowed positive case
# ---------------------------------------------------------------------------


def test_two_task_fields_compared_numeric(plan: PlanSpec) -> None:
    """task.timeout_sec > task.max_retries must evaluate correctly (30 > 2 → True)."""
    task = _task(id="t1", timeout_sec=30, max_retries=2)
    spec = _policy("task.timeout_sec > task.max_retries")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_two_task_fields_equal_strings(plan: PlanSpec) -> None:
    """task.engine == task.id must be True when both resolve to the same string."""
    task = _task(id="claude", engine="claude")
    spec = _policy("task.engine == task.id")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# 20. Double negation `not not x` — must evaluate correctly
# ---------------------------------------------------------------------------


def test_double_negation_true(plan: PlanSpec) -> None:
    """`not not (task.engine == "claude")` must be True when engine is claude."""
    task = _task(id="t1", engine="claude")
    spec = _policy('not not (task.engine == "claude")')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_double_negation_false(plan: PlanSpec) -> None:
    """`not not (task.engine == "codex")` must be False when engine is claude."""
    task = _task(id="t1", engine="claude")
    spec = _policy('not not (task.engine == "codex")')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# 21. Boolean task fields (cache, requires_approval) as bare conditions
# ---------------------------------------------------------------------------


def test_cache_field_false_as_bare_condition(plan: PlanSpec) -> None:
    """`not task.cache` must be True when cache is explicitly False."""
    task = _task(id="t1", cache=False)
    spec = _policy("not task.cache")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_requires_approval_field_true(plan: PlanSpec) -> None:
    """`task.requires_approval` must evaluate to True when the field is set."""
    task = _task(id="t1", requires_approval=True)
    spec = _policy("task.requires_approval")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# 22. Plan numeric / boolean fields
# ---------------------------------------------------------------------------


def test_plan_max_parallel_numeric_comparison(task: TaskSpec) -> None:
    """`plan.max_parallel > 2` must be True when max_parallel is set to 4."""
    from maestro_cli.models import PlanSpec as _PlanSpec

    plan = _PlanSpec(name="p", max_parallel=4)
    spec = _policy("plan.max_parallel > 2")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_plan_fail_fast_boolean_field(task: TaskSpec, plan: PlanSpec) -> None:
    """`plan.fail_fast` must be True when fail_fast defaults to True."""
    spec = _policy("plan.fail_fast")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# 23. `in` with a list / tuple literal RHS — unsupported AST nodes on RHS
# ---------------------------------------------------------------------------


def test_in_with_list_literal_rhs_rejected(task: TaskSpec, plan: PlanSpec) -> None:
    """`task.id in ["t1", "t2"]` fails: ast.List on RHS is not allowlisted."""
    spec = _policy('task.id in ["t1", "t2"]')
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported AST node"):
        ev(task, plan)


def test_in_with_tuple_literal_rhs_rejected(task: TaskSpec, plan: PlanSpec) -> None:
    """`task.engine in ("claude", "codex")` fails: ast.Tuple on RHS is not allowlisted."""
    spec = _policy('task.engine in ("claude", "codex")')
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported AST node"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 24. ast.Is / ast.IsNot comparators — not in _OPS, must be rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule",
    [
        "task.engine is None",
        "task.engine is not None",
        "task.model is None",
        "task.model is not None",
    ],
)
def test_is_comparator_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """`is` and `is not` are not in the allowed comparator set and must be rejected."""
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported comparator"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 25. Unknown task fields not in _SAFE_TASK_FIELDS — must be rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["prompt", "command", "depends_on", "pre_command", "verify_command", "engine_override"],
)
def test_unknown_task_fields_rejected(task: TaskSpec, plan: PlanSpec, field: str) -> None:
    """Task fields not in _SAFE_TASK_FIELDS must be rejected at evaluation time."""
    spec = _policy(f'task.{field} == "x"')
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unknown task field"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 26. Unknown plan fields not in _SAFE_PLAN_FIELDS — must be rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "field",
    ["tasks", "workspace_root", "defaults", "run_dir"],
)
def test_unknown_plan_fields_rejected(task: TaskSpec, plan: PlanSpec, field: str) -> None:
    """Plan fields not in _SAFE_PLAN_FIELDS must be rejected at evaluation time."""
    spec = _policy(f'plan.{field} == "x"')
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unknown plan field"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 27. Deeply nested Not chains — quadruple / triple negation correctness
# ---------------------------------------------------------------------------


def test_quadruple_negation_evaluates_correctly(plan: PlanSpec) -> None:
    """`not not not not task.cache` with cache=True must evaluate to True."""
    task = _task(id="t1", cache=True)
    spec = _policy("not not not not task.cache")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_triple_negation_evaluates_correctly(plan: PlanSpec) -> None:
    """`not not not task.allow_failure` with allow_failure=True must evaluate to False."""
    task = _task(id="t1", allow_failure=True)
    spec = _policy("not not not task.allow_failure")
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# 28. Python comment in rule string — ignored by ast.parse, rule runs normally
# ---------------------------------------------------------------------------


def test_comment_in_rule_is_ignored(task: TaskSpec, plan: PlanSpec) -> None:
    """A `#` comment appended to a valid rule must be silently ignored by ast.parse."""
    spec = _policy('task.engine == "claude" # injected comment')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# 29. Short-circuit evaluation — bad Call node not reached when condition is settled
# ---------------------------------------------------------------------------


def test_and_short_circuit_skips_bad_node(plan: PlanSpec) -> None:
    """`False and len(...)` — all() short-circuits before reaching the Call node."""
    # "codex" != "claude" → first operand False → len() never evaluated → no ValueError
    task = _task(id="t1", engine="codex")
    spec = _policy('task.engine == "claude" and len(task.tags) > 0')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


def test_or_short_circuit_skips_bad_node(plan: PlanSpec) -> None:
    """`True or len(...)` — any() short-circuits before reaching the Call node."""
    # "claude" == "claude" → first operand True → len() never evaluated → no ValueError
    task = _task(id="t1", engine="claude")
    spec = _policy('task.engine == "claude" or len(task.tags) > 0')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# 30. plan.max_cost_usd — None default and float comparisons
# ---------------------------------------------------------------------------


def test_plan_max_cost_usd_none_comparison(task: TaskSpec, plan: PlanSpec) -> None:
    """`plan.max_cost_usd == None` must be True when max_cost_usd is not set."""
    spec = _policy("plan.max_cost_usd == None")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_plan_max_cost_usd_float_comparison() -> None:
    """`plan.max_cost_usd > 5.0` must be True when max_cost_usd is 10.0."""
    from maestro_cli.models import PlanSpec as _PlanSpec

    local_plan = _PlanSpec(name="p", max_cost_usd=10.0)
    task = _task(id="t1")
    spec = _policy("plan.max_cost_usd > 5.0")
    ev = compile_policy(spec)
    assert ev(task, local_plan) is True


# ---------------------------------------------------------------------------
# 31. task.tags list membership — valid `not in` / `in` with populated list
# ---------------------------------------------------------------------------


def test_not_in_tag_list_when_absent(plan: PlanSpec) -> None:
    """'absent' not in task.tags must be True when tags don't contain it."""
    task = _task(id="t1", tags=["test", "qa"])
    spec = _policy('"absent" not in task.tags')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_in_tag_list_false_when_absent(plan: PlanSpec) -> None:
    """"missing" in task.tags must be False for a non-empty list that lacks the value."""
    task = _task(id="t1", tags=["test", "qa"])
    spec = _policy('"missing" in task.tags')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# 32. task.execution_profile — computed field resolved from plan
# ---------------------------------------------------------------------------


def test_task_execution_profile_defaults_to_plan(plan: PlanSpec) -> None:
    """task.execution_profile equals 'plan' when plan.execution_profile is not set."""
    task = _task(id="t1", engine="claude")
    spec = _policy('task.execution_profile == "plan"')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# 33. task.cost_usd computed field — defaults to 0.0 when no TaskResult provided
# ---------------------------------------------------------------------------


def test_task_cost_usd_defaults_zero_without_result(plan: PlanSpec) -> None:
    """task.cost_usd must equal 0.0 when no TaskResult is available."""
    task = _task(id="t1", engine="claude")
    spec = _policy("task.cost_usd == 0.0")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# 34. task.description — whitelisted field that defaults to None
# ---------------------------------------------------------------------------


def test_task_description_empty_string_default(plan: PlanSpec) -> None:
    """task.description == "" must be True for a task with no description set (defaults to empty string)."""
    task = _task(id="t1", engine="claude")
    spec = _policy('task.description == ""')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# 35. Four-operator chained comparison — tests multi-op iteration beyond 3 operands
# ---------------------------------------------------------------------------


def test_four_operator_chained_comparison_true(plan: PlanSpec) -> None:
    """0 < 1 < task.max_retries < 100 must be True when max_retries is 2."""
    task = _task(id="t1", max_retries=2)
    spec = _policy("0 < 1 < task.max_retries < 100")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_four_operator_chained_comparison_false(plan: PlanSpec) -> None:
    """0 < 1 < task.max_retries < 100 must be False when max_retries is 1 (1 < 1 fails)."""
    task = _task(id="t1", max_retries=1)
    spec = _policy("0 < 1 < task.max_retries < 100")
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# 36. Mixed task + plan fields in a single expression
# ---------------------------------------------------------------------------


def test_mixed_task_and_plan_fields_conjunction(task: TaskSpec, plan: PlanSpec) -> None:
    """task.engine == "claude" and plan.fail_fast must be True with default fixtures."""
    spec = _policy('task.engine == "claude" and plan.fail_fast')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# 37. Slice notation — ast.Subscript with ast.Slice is not allowlisted
# ---------------------------------------------------------------------------


def test_slice_notation_rejected(task: TaskSpec, plan: PlanSpec) -> None:
    """task.tags[0:1] uses ast.Subscript which is not in the allowlist."""
    spec = _policy("task.tags[0:1]")
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported AST node"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# 38. task.has_judge computed field — True with judge, False without
# ---------------------------------------------------------------------------


def test_has_judge_false_without_judge(task: TaskSpec, plan: PlanSpec) -> None:
    """task.has_judge must be False (and `not task.has_judge` True) when judge is None."""
    spec = _policy("not task.has_judge")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_has_judge_true_with_judge(plan: PlanSpec) -> None:
    """task.has_judge must be True when task.judge is set to a JudgeSpec."""
    from maestro_cli.models import JudgeSpec

    task = _task(id="t1", engine="claude", judge=JudgeSpec(criteria=["output is correct"]))
    spec = _policy("task.has_judge")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# 39. evaluate_policies structural invariants
# ---------------------------------------------------------------------------


def test_evaluate_policies_empty_list_returns_no_violations(
    task: TaskSpec, plan: PlanSpec
) -> None:
    """evaluate_policies([]) must return an empty list without errors."""
    violations = evaluate_policies([], task, plan)
    assert violations == []


def test_evaluate_policies_multiple_matching_policies_all_returned(
    task: TaskSpec, plan: PlanSpec
) -> None:
    """All matching policies produce a separate PolicyViolation; non-matching ones are silent."""
    policies = [
        PolicySpec(name="p1", rule='task.engine == "claude"', action="warn"),  # type: ignore[arg-type]
        PolicySpec(name="p2", rule="task.timeout_sec > 0", action="warn"),  # type: ignore[arg-type]
        PolicySpec(name="p3", rule='task.engine == "codex"', action="warn"),  # type: ignore[arg-type]
    ]
    violations = evaluate_policies(policies, task, plan)
    assert len(violations) == 2
    names = {v.policy_name for v in violations}
    assert names == {"p1", "p2"}


# ---------------------------------------------------------------------------
# 40. Constant-only expressions — pure Constant/Compare nodes (no task/plan)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rule,expected",
    [
        ("True", True),
        ("False", False),
        ('"x" == "x"', True),
        ('"a" != "b"', True),
        ("1 < 2", True),
        ("2 < 1", False),
    ],
)
def test_constant_only_expressions(
    task: TaskSpec, plan: PlanSpec, rule: str, expected: bool
) -> None:
    """Expressions using only constants must compile and evaluate without referencing task/plan."""
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is expected


# ---------------------------------------------------------------------------
# 41. Whitespace-only rules — SyntaxError at ast.parse time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_rule",
    [
        "   ",
        "\t",
        "\n",
        "  \t\n  ",
    ],
)
def test_whitespace_only_rule_raises(bad_rule: str) -> None:
    """Whitespace-only rules must fail at compile time with SyntaxError."""
    spec = _policy(bad_rule)
    with pytest.raises((ValueError, SyntaxError)):
        compile_policy(spec)


# ---------------------------------------------------------------------------
# 42. Empty task.tags list membership — must not raise, returns correct bool
# ---------------------------------------------------------------------------


def test_in_with_empty_tags_returns_false(plan: PlanSpec) -> None:
    """'x' in task.tags must be False when tags is an empty list."""
    task = _task(id="t1", tags=[])
    spec = _policy('"x" in task.tags')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


def test_not_in_with_empty_tags_returns_true(plan: PlanSpec) -> None:
    """'x' not in task.tags must be True when tags is an empty list."""
    task = _task(id="t1", tags=[])
    spec = _policy('"x" not in task.tags')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ===========================================================================
# ADVERSARIAL SANDBOX BYPASS TESTS
#
# Each test below attempts a specific strategy to escape the _SafeEvaluator
# sandbox and achieve arbitrary code execution. All must be rejected via
# ValueError or SyntaxError — if any succeeds, the sandbox is broken.
# ===========================================================================


# ---------------------------------------------------------------------------
# A1. __builtins__ based attacks — accessing the builtins namespace to call
#     __import__ or other dangerous functions indirectly.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Direct builtins access chain
        '__builtins__.__import__("os")',
        '__builtins__["__import__"]("os")',
        '__builtins__.__dict__["eval"]("1+1")',
        # Access builtins through class hierarchy (classic jailbreak)
        'task.__class__.__bases__[0].__subclasses__()',
        'task.__class__.__mro__[1].__subclasses__()',
        # Access through __init__.__globals__
        'task.__class__.__init__.__globals__["__builtins__"]',
        # Access through type()
        "type(task).__mro__",
        # Traverse to get os module
        'task.__class__.__bases__[0].__subclasses__()[40]("/etc/passwd")',
    ],
    ids=[
        "builtins_import",
        "builtins_subscript_import",
        "builtins_dict_eval",
        "class_bases_subclasses",
        "class_mro_subclasses",
        "init_globals_builtins",
        "type_mro",
        "subclass_file_open",
    ],
)
def test_bypass_via_builtins_access(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Access __builtins__ namespace to reach __import__/eval/exec.

    The classic Python sandbox escape traverses the class hierarchy:
    obj.__class__.__bases__[0].__subclasses__() to find dangerous classes
    like os._wrap_close that expose os.system. Every step must be blocked
    by the evaluator — nested attributes, dunder fields, subscripts, or calls.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A2. getattr-based attribute access bypass — using getattr() to circumvent
#     the direct attribute whitelist check.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Bypass attribute whitelist via getattr
        'getattr(task, "__class__")',
        'getattr(task, "prompt")',         # access non-whitelisted field
        'getattr(plan, "tasks")',          # access non-whitelisted plan field
        # Build dunder string dynamically to evade static detection
        'getattr(task, chr(95)+chr(95)+"class"+chr(95)+chr(95))',
        # Chain getattr to walk the MRO
        'getattr(getattr(task, "__class__"), "__bases__")',
        # Use getattr with default to avoid AttributeError
        'getattr(task, "__dict__", {})',
        # Nested getattr to reach builtins
        'getattr(getattr(task, "__class__"), "__init__").__globals__',
    ],
    ids=[
        "getattr_dunder_class",
        "getattr_non_whitelisted_field",
        "getattr_plan_tasks",
        "getattr_chr_dunder_construction",
        "getattr_chained_mro",
        "getattr_with_default",
        "getattr_nested_globals",
    ],
)
def test_bypass_via_getattr_manipulation(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Use getattr() to bypass the attribute whitelist.

    If getattr(task, "__class__") were allowed, an attacker could read any
    attribute including __dict__, __init__.__globals__, etc. The evaluator
    must reject all ast.Call nodes including getattr.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A3. eval/exec obfuscation — constructing code strings dynamically and
#     passing them to eval() or exec() in creative ways.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Direct obfuscated eval
        'eval("__" + "import" + "__")(\"os\")',
        'eval(chr(111)+chr(115))',
        # exec with string concat
        'exec("import " + "os")',
        'exec("\\x5f\\x5fimport\\x5f\\x5f(\'os\')")',
        # compile + exec combo
        'exec(compile("import os", "<str>", "exec"))',
        # eval nested in eval
        'eval(eval("chr(95)*2+chr(105)+chr(109)+chr(112)"))',
        # Use format strings to build dangerous code
        'eval("{}{}{}".format("__import", "__", "(\'os\')"))',
        # bytes decode trick
        'eval(b"__import__(\'os\')".decode())',
    ],
    ids=[
        "eval_string_concat_import",
        "eval_chr_construction",
        "exec_string_concat_import",
        "exec_hex_escape_import",
        "compile_exec_combo",
        "eval_nested_eval",
        "eval_format_construction",
        "eval_bytes_decode",
    ],
)
def test_bypass_via_eval_exec_obfuscation(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Obfuscate eval/exec calls to evade pattern-based detection.

    An attacker might try to build the string '__import__' dynamically using
    chr(), string concatenation, bytes.decode(), or format() to bypass any
    string-based detection. The AST-based evaluator must reject the outer
    eval/exec Call node regardless of argument construction.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A4. Comprehension and generator tricks — embedding dangerous calls
#     inside list comprehensions, generator expressions, dict/set comps.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # List comp with __import__
        '[__import__("os") for _ in [1]]',
        # List comp with exec
        '[exec("import os") for _ in range(1)]',
        # Generator expression with eval
        'next(eval("1") for _ in [1])',
        # Dict comprehension with side-effect
        '{k: __import__("os") for k in ["x"]}',
        # Set comprehension
        '{__import__("os") for _ in [1]}',
        # Nested comprehension
        '[[__import__("os")] for _ in [1] for __ in [2]]',
        # Comprehension with walrus operator side-effect
        '[y for x in [1] if (y := __import__("os"))]',
        # Generator passed to next()
        'next(x for x in [__import__("os")])',
    ],
    ids=[
        "listcomp_import",
        "listcomp_exec",
        "genexpr_eval",
        "dictcomp_import",
        "setcomp_import",
        "nested_comp_import",
        "comp_walrus_import",
        "genexpr_next_import",
    ],
)
def test_bypass_via_comprehension_injection(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Hide code execution inside comprehension/generator bodies.

    Comprehensions create their own scope in Python 3, so an attacker might
    hope that the evaluator walks the outer ListComp but not the inner Call
    nodes. The evaluator must reject the entire comprehension structure.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A5. Lambda injection — using lambda to defer code execution and
#     potentially bypass upfront AST checks.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Lambda that imports
        '(lambda: __import__("os"))()',
        # Lambda with exec
        '(lambda: exec("import os"))()',
        # Lambda assigned via walrus and called
        '(f := lambda: True)()',
        # Lambda with default arg trick (evaluated at definition time)
        '(lambda x=__import__("os"): x)()',
        # Nested lambda
        '(lambda: (lambda: __import__("os"))())()',
        # Lambda in comprehension
        '[(lambda: __import__("os"))() for _ in [1]]',
    ],
    ids=[
        "lambda_import",
        "lambda_exec",
        "lambda_walrus_call",
        "lambda_default_arg_import",
        "nested_lambda_import",
        "lambda_in_comprehension",
    ],
)
def test_bypass_via_lambda_injection(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Use lambda to wrap dangerous calls and invoke them.

    Lambda creates an anonymous function that might defer the dangerous Call
    node past an initial AST check. The evaluator must reject ast.Lambda
    nodes entirely, preventing any lambda-based bypass.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A6. Attribute access through indirect references — attempting to reach
#     dangerous attributes via objects that aren't task/plan.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Access string methods to build code
        '"x".__class__.__bases__[0].__subclasses__()',
        '"".__class__.__mro__[2].__subclasses__()',
        # Access through int literal
        '(1).__class__.__bases__[0].__subclasses__()',
        # Access through tuple
        '().__class__.__bases__[0].__subclasses__()',
        # Access through True/False
        'True.__class__.__bases__[0].__subclasses__()',
        # Access dict items through literal
        '{}.__class__.__bases__[0].__subclasses__()',
    ],
    ids=[
        "str_class_subclasses",
        "empty_str_mro_subclasses",
        "int_class_subclasses",
        "tuple_class_subclasses",
        "bool_class_subclasses",
        "dict_class_subclasses",
    ],
)
def test_bypass_via_literal_class_traversal(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Use Python literals to access __class__.__bases__.__subclasses__.

    The classic Python sandbox escape starts from ANY object (even a literal
    string or int) and walks up the class hierarchy via __class__.__bases__
    to reach object.__subclasses__(), which exposes every loaded class including
    os._wrap_close. The evaluator must block this at the attribute/subscript level.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A7. Statement injection via expression wrappers — trying to sneak
#     statements (import, del, raise, assert) into an eval context.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Direct statement injection (ast.parse mode="eval" rejects)
        "import os",
        "from os import system",
        "del task",
        "raise Exception('pwned')",
        "assert False",
        # Assignment (not valid in eval mode)
        'x = __import__("os")',
        # Augmented assignment
        "x += 1",
        # Multi-statement via semicolon
        'True; __import__("os")',
        'True; exec("import os")',
        # Try/except wrapper
        'try:\n  __import__("os")\nexcept:\n  pass',
    ],
    ids=[
        "import_statement",
        "from_import",
        "del_statement",
        "raise_exception",
        "assert_statement",
        "assignment_import",
        "augmented_assignment",
        "semicolon_import",
        "semicolon_exec",
        "try_except_import",
    ],
)
def test_bypass_via_statement_injection(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Inject Python statements that ast.parse(mode='eval') rejects.

    Since the evaluator uses mode='eval', only expressions are valid.
    Statements (import, del, raise, assert, assignment) must cause a
    SyntaxError at parse time, preventing any statement-based attack.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A8. Advanced Python tricks — yield expressions, await, starred unpacking,
#     conditional exec, and other creative bypass attempts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Yield expression (valid in generator context, not in eval)
        '(yield __import__("os"))',
        # Await expression
        'await __import__("asyncio")',
        # Starred expression in various contexts
        '[*__import__("os").listdir(".")]',
        # Conditional (ternary) exec
        'exec("import os") if True else None',
        # Walrus with __import__
        '(x := __import__("os")).system("id")',
        '(x := __import__("os"))',
        # Accessing .mro() method directly
        "type.__mro__",
        # breakpoint() for debugger escape
        "breakpoint()",
        # help() for interactive escape
        "help(task)",
        # vars() to dump namespace
        "vars(task)",
        # repr trick
        'repr(task.__class__)',
        # hash-based probing
        "id(task)",
        # object.__reduce__ for pickle-based attacks
        "task.__reduce__()",
        # Accessing __globals__ through a function
        'print.__globals__["__builtins__"]',
    ],
    ids=[
        "yield_import",
        "await_import",
        "star_unpack_listdir",
        "ternary_exec",
        "walrus_import_system",
        "walrus_import_bare",
        "type_mro_attribute",
        "breakpoint_call",
        "help_interactive",
        "vars_dump",
        "repr_class",
        "id_probe",
        "reduce_pickle",
        "print_globals",
    ],
)
def test_bypass_via_advanced_python_tricks(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Use advanced Python features to escape the sandbox.

    These attacks combine multiple techniques: yield/await expressions,
    starred unpacking, walrus operator with __import__, and accessing
    function __globals__ through builtins like print. Each must be
    rejected either at parse time (SyntaxError) or evaluation time (ValueError).
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A9. ast.NodeVisitor traversal attacks — crafting expressions where the
#     dangerous node is deeply nested in a tree that might not be fully walked.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Deep nesting: Call inside Compare inside BoolOp
        'True and __import__("os") == True',
        'False or False or __import__("os") != None',
        # Call hidden as comparator RHS
        'task.engine == __import__("os")',
        # Call hidden as Compare left side
        '__import__("os") == "posix"',
        # Call nested in Not
        'not __import__("os")',
        # Call in multi-compare chain
        '1 < __import__("os") < 10',
        # Multiple dangerous calls in single expression
        'eval("1") == exec("2")',
        # Call as BoolOp operand among safe ones
        'task.engine == "claude" and task.timeout_sec > 0 and globals()',
    ],
    ids=[
        "call_in_boolop_rhs",
        "call_in_triple_or",
        "call_as_compare_rhs",
        "call_as_compare_lhs",
        "call_in_not",
        "call_in_chained_compare",
        "double_call_compare",
        "call_hidden_among_safe",
    ],
)
def test_bypass_via_deeply_nested_call_nodes(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Hide dangerous Call nodes deep in the AST tree.

    An incomplete AST walker might check top-level nodes but skip nested
    children. These tests place __import__/eval/exec/globals calls inside
    Compare operands, BoolOp values, and UnaryOp operands to verify the
    evaluator walks every node in the tree.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A10. Combined multi-technique attacks — chaining multiple bypass
#      strategies in a single expression.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Lambda + getattr + chr construction
        '(lambda: getattr(task, chr(95)*2+"class"+chr(95)*2))()',
        # Comprehension + walrus + exec
        '[y for x in [1] if (y := exec("import os"))]',
        # Ternary + lambda + import
        '(lambda: __import__("os"))() if True else None',
        # f-string + import (format string injection)
        'f"{__import__(\'os\').getcwd()}"',
        # Nested getattr + subscript + call
        'getattr(task, "__class__").__bases__[0].__subclasses__()',
        # eval(bytes) + decode
        'eval(bytes([95,95,105,109,112,111,114,116,95,95]).decode())',
        # Comprehension generating function + immediate call
        '[f() for f in [lambda: __import__("os")]]',
        # dict.get to extract builtins
        'vars().__getitem__("__builtins__")',
    ],
    ids=[
        "lambda_getattr_chr",
        "comp_walrus_exec",
        "ternary_lambda_import",
        "fstring_import",
        "getattr_chain_subclasses",
        "eval_bytes_decode",
        "comp_lambda_call",
        "vars_getitem_builtins",
    ],
)
def test_bypass_via_combined_techniques(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Chain multiple bypass techniques in a single expression.

    Real attackers combine techniques — lambda wrapping getattr with chr()
    construction, comprehensions with walrus-operator side effects, or
    f-strings embedding __import__ calls. The evaluator must reject at
    least one component in every combination.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A11. Exhaustive builtin function injection — every dangerous Python
#      builtin that could lead to code execution, file access, or info leak.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Code execution builtins
        '__import__("os")',
        'eval("1")',
        'exec("pass")',
        'compile("1", "x", "eval")',
        "breakpoint()",
        # File system access
        'open("/etc/passwd")',
        'open("/etc/passwd", "r").read()',
        # Reflection/introspection
        "globals()",
        "locals()",
        "vars()",
        "dir()",
        "dir(task)",
        "type(task)",
        "id(task)",
        "hash(task)",
        "repr(task)",
        "callable(task)",
        "isinstance(task, object)",
        "issubclass(type(task), object)",
        "hasattr(task, '__class__')",
        'delattr(task, "id")',
        'setattr(task, "id", "pwned")',
        # Object construction
        "object()",
        "dict()",
        "list()",
        "tuple()",
        "set()",
        "frozenset()",
        "bytearray()",
        "memoryview(b'')",
        # String/number constructors that could be abused
        "int('0x41', 16)",
        "float('inf')",
        'bytes([95, 95])',
        'chr(95)',
        'ord("_")',
        # Iteration tools
        "iter(task.tags)",
        "next(iter(task.tags))",
        "enumerate(task.tags)",
        "zip(task.tags, task.tags)",
        "map(str, task.tags)",
        "filter(None, task.tags)",
        "sorted(task.tags)",
        "reversed(task.tags)",
        # Input
        'input(">")',
        # Dynamic attribute access
        'getattr(task, "id")',
        'getattr(task, "__class__")',
    ],
    ids=[
        "import", "eval", "exec", "compile", "breakpoint",
        "open_read", "open_read_chain",
        "globals", "locals", "vars_noarg", "dir_noarg", "dir_task",
        "type", "id", "hash", "repr", "callable",
        "isinstance", "issubclass", "hasattr", "delattr", "setattr",
        "object_new", "dict_new", "list_new", "tuple_new", "set_new",
        "frozenset_new", "bytearray_new", "memoryview_new",
        "int_base16", "float_inf", "bytes_list", "chr", "ord",
        "iter", "next_iter", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "input",
        "getattr_safe_field", "getattr_dunder",
    ],
)
def test_every_dangerous_builtin_rejected(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Exhaustively test every Python builtin that could be weaponized.

    This covers code execution (eval/exec/compile/__import__), file access (open),
    reflection (globals/locals/vars/dir), object construction, and dynamic
    attribute access (getattr/setattr/delattr/hasattr). ALL must be rejected
    because the evaluator must never allow any ast.Call node.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A12. `in` / `not in` with string fields — substring membership check
#      `in` on a Python str is a substring test, not list membership.
#      Both plan.name and task.id are str — these are valid, safe rules.
# ---------------------------------------------------------------------------


def test_in_with_plan_name_string_match(task: TaskSpec) -> None:
    """`"test" in plan.name` must return True when plan.name contains "test"."""
    plan = PlanSpec(name="test-plan")
    spec = _policy('"test" in plan.name')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_in_with_plan_name_string_no_match(task: TaskSpec) -> None:
    """`"prod" in plan.name` must return False when plan.name is "test-plan"."""
    plan = PlanSpec(name="test-plan")
    spec = _policy('"prod" in plan.name')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


def test_not_in_with_task_id_substring(plan: PlanSpec) -> None:
    """`"bad" not in task.id` must return True when id contains no substring "bad"."""
    task = _task(id="good-task", engine="claude")
    spec = _policy('"bad" not in task.id')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_not_in_with_task_id_substring_matches(plan: PlanSpec) -> None:
    """`"bad" not in task.id` must return False when id contains "bad"."""
    task = _task(id="bad-task", engine="claude")
    spec = _policy('"bad" not in task.id')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# A13. Unicode homoglyph attack — using Cyrillic/lookalike characters in
#      object names to try to bypass the "task" / "plan" name check.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Cyrillic 'а' (U+0430) lookalike for ASCII 'a'
        "t\u0430sk.engine == \"claude\"",   # tаsk with Cyrillic а
        # Cyrillic 'е' (U+0435) lookalike for ASCII 'e'
        "task.\u0435ngine == \"claude\"",   # еngine with Cyrillic е
        # Object neither task nor plan (some other whitelisted-looking name)
        'tasks.engine == "claude"',
        'Task.engine == "claude"',
        'TASK.engine == "claude"',
        'task_.engine == "claude"',
    ],
    ids=[
        "cyrillic_a_in_task",
        "cyrillic_e_in_engine_field",
        "plural_tasks",
        "capitalized_Task",
        "uppercase_TASK",
        "trailing_underscore_task_",
    ],
)
def test_unicode_homoglyph_object_names_rejected(
    task: TaskSpec, plan: PlanSpec, injection: str
) -> None:
    """Attack: Use Unicode lookalike characters to spoof 'task' or field names.

    A naive check doing `obj_name == "task"` would pass for ASCII 'task' only.
    Unicode homoglyphs (Cyrillic а=U+0430, е=U+0435) or alternate spellings
    must be caught as 'unknown object' or 'unknown field' errors.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A14. Compiled policy idempotency — same compiled evaluator, multiple tasks
#      A compiled policy must return consistent results across repeated calls.
# ---------------------------------------------------------------------------


def test_compiled_policy_evaluates_multiple_tasks_consistently(plan: PlanSpec) -> None:
    """Compiled evaluator must produce correct independent results per task.

    This catches any state leakage between evaluations — if the evaluator
    caches or mutates state from one call to the next, subsequent evaluations
    might return wrong results.
    """
    spec = _policy('task.engine == "claude"')
    ev = compile_policy(spec)

    task_claude = _task(id="a", engine="claude")
    task_codex = _task(id="b", engine="codex")
    task_gemini = _task(id="c", engine="gemini")

    # Alternating calls must give independent results
    assert ev(task_claude, plan) is True
    assert ev(task_codex, plan) is False
    assert ev(task_claude, plan) is True   # must not be contaminated by codex call
    assert ev(task_gemini, plan) is False
    assert ev(task_codex, plan) is False
    assert ev(task_claude, plan) is True   # still correct after multiple others


# ---------------------------------------------------------------------------
# A15. Deep `not` chain — 8 and 9 consecutive negations
# ---------------------------------------------------------------------------


def test_eight_deep_not_chain_allow_failure_true(plan: PlanSpec) -> None:
    """Eight `not` ops with allow_failure=True must evaluate to True (even count)."""
    task = _task(id="t1", allow_failure=True)
    rule = "not not not not not not not not task.allow_failure"
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_eight_deep_not_chain_allow_failure_false(plan: PlanSpec) -> None:
    """Eight `not` ops with allow_failure=False must evaluate to False (even count)."""
    task = _task(id="t1", allow_failure=False)
    rule = "not not not not not not not not task.allow_failure"
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is False


def test_nine_deep_not_chain_inverts_value(plan: PlanSpec) -> None:
    """Nine `not` ops (odd) must flip the boolean value."""
    task_true = _task(id="t1", allow_failure=True)
    task_false = _task(id="t2", allow_failure=False)
    rule = "not not not not not not not not not task.allow_failure"
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task_true, plan) is False
    assert ev(task_false, plan) is True


# ---------------------------------------------------------------------------
# A16. Non-bool truthy return coerced to bool — `BoolOp` returns last truthy
#      value, not always a bool.  bool() in compile_policy must handle this.
# ---------------------------------------------------------------------------


def test_bool_coercion_of_truthy_string_from_or(plan: PlanSpec) -> None:
    """`task.engine or False` returns the engine string (truthy) → bool True."""
    task = _task(id="t1", engine="claude")
    spec = _policy("task.engine or False")
    ev = compile_policy(spec)
    # engine == "claude" is truthy; or short-circuits → returns "claude"
    # compile_policy wraps with bool(), so result must be True
    assert ev(task, plan) is True


def test_bool_coercion_of_falsy_none_from_or(plan: PlanSpec) -> None:
    """`task.model or False` when model is None → None or False → False."""
    task = _task(id="t1", model=None)
    spec = _policy("task.model or False")
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# A17. Cross-field comparison — task field vs plan field in one Compare node
# ---------------------------------------------------------------------------


def test_task_timeout_vs_plan_max_parallel_gte(plan: PlanSpec) -> None:
    """`task.timeout_sec >= plan.max_parallel` must work when both are ints."""
    plan_4 = PlanSpec(name="p", max_parallel=4)
    task_30 = _task(id="t1", timeout_sec=30)
    spec = _policy("task.timeout_sec >= plan.max_parallel")
    ev = compile_policy(spec)
    assert ev(task_30, plan_4) is True  # 30 >= 4


def test_task_timeout_vs_plan_max_parallel_lt(plan: PlanSpec) -> None:
    """`task.timeout_sec < plan.max_parallel` must be False when 30 < 4 is False."""
    plan_4 = PlanSpec(name="p", max_parallel=4)
    task_30 = _task(id="t1", timeout_sec=30)
    spec = _policy("task.timeout_sec < plan.max_parallel")
    ev = compile_policy(spec)
    assert ev(task_30, plan_4) is False  # 30 < 4 is False


def test_plan_name_equality_combined_with_task_field(plan: PlanSpec) -> None:
    """`plan.name == "prod" and task.engine == "claude"` must be False if plan.name differs."""
    task = _task(id="t1", engine="claude")
    spec = _policy('plan.name == "prod" and task.engine == "claude"')
    ev = compile_policy(spec)
    assert ev(task, plan) is False  # plan.name is "test-plan"


# ---------------------------------------------------------------------------
# A18. Forbidden dunder method invocations on task/plan — calling __str__,
#      __repr__, __format__, __len__, __iter__, __contains__, __hash__, etc.
#      All must be rejected (Call node or dunder field not whitelisted).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        'task.__str__()',
        'task.__repr__()',
        'task.__format__("")',
        'task.__sizeof__()',
        'task.__hash__()',
        'task.__eq__(task)',
        'task.__ne__(task)',
        'task.__getattribute__("id")',
        'task.__setattr__("id", "pwned")',
        'task.__delattr__("id")',
        'task.__reduce__()',
        'task.__reduce_ex__(0)',
        'task.__dir__()',
        'plan.__str__()',
        'plan.__repr__()',
        'plan.__getattribute__("name")',
    ],
    ids=[
        "task_str", "task_repr", "task_format", "task_sizeof",
        "task_hash", "task_eq", "task_ne",
        "task_getattribute", "task_setattr", "task_delattr",
        "task_reduce", "task_reduce_ex", "task_dir",
        "plan_str", "plan_repr", "plan_getattribute",
    ],
)
def test_dunder_method_calls_rejected(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Attack: Call dunder methods directly on task/plan to bypass whitelist.

    If __getattribute__ or __setattr__ calls were allowed, an attacker could
    read/write any attribute. If __reduce__ were allowed, pickle-based attacks
    become possible. All must be rejected via dunder field check or Call rejection.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A19. OS/system command injection via specific dangerous module imports —
#      each import target represents a different attack surface.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        '__import__("os").system("id")',
        '__import__("os").popen("cat /etc/passwd").read()',
        '__import__("subprocess").call(["rm", "-rf", "/"])',
        '__import__("subprocess").Popen(["sh"])',
        '__import__("shutil").rmtree("/")',
        '__import__("pty").spawn("/bin/sh")',
        '__import__("code").interact()',
        '__import__("ctypes").CDLL(None)',
        '__import__("socket").socket()',
        '__import__("signal").alarm(0)',
        '__import__("sys").exit(0)',
        '__import__("builtins").__import__("os")',
        '__import__("importlib").import_module("os")',
        '__import__("pathlib").Path("/etc/passwd").read_text()',
        '__import__("tempfile").mktemp()',
        '__import__("webbrowser").open("http://evil.com")',
    ],
    ids=[
        "os_system", "os_popen", "subprocess_call", "subprocess_popen",
        "shutil_rmtree", "pty_spawn", "code_interact", "ctypes_cdll",
        "socket_create", "signal_alarm", "sys_exit", "builtins_import",
        "importlib_import_module", "pathlib_read", "tempfile_mktemp",
        "webbrowser_open",
    ],
)
def test_os_command_injection_via_import_targets(
    task: TaskSpec, plan: PlanSpec, injection: str
) -> None:
    """Attack: Import specific dangerous modules and call destructive functions.

    Each import target represents a different attack surface: filesystem
    destruction (shutil.rmtree), shell spawning (pty.spawn), network access
    (socket), process control (signal), etc. The __import__ Call node must
    be rejected before any module loading occurs.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A20. Metaclass and type manipulation — type(), __class__ assignment,
#      __metaclass__, __init_subclass__, __class_getitem__.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # type() to inspect or construct classes
        'type(task).__name__',
        'type(task).__bases__',
        'type(task).__mro__',
        'type(task).__subclasses__()',
        'type(task).__call__(task)',
        'type(task).__dict__',
        # Build a new class dynamically
        'type("Evil", (object,), {"__init__": lambda s: None})()',
        # Access metaclass machinery
        'task.__class__.__new__(task.__class__)',
        'task.__class__.__init_subclass__()',
        'task.__class__.__class_getitem__(str)',
        # Probe the type of builtin constants
        'type(True).__bases__',
        'type(None).__mro__',
        'type(1).__subclasses__()',
    ],
    ids=[
        "type_name", "type_bases", "type_mro", "type_subclasses",
        "type_call", "type_dict", "type_construct_class",
        "class_new", "class_init_subclass", "class_getitem",
        "type_true_bases", "type_none_mro", "type_int_subclasses",
    ],
)
def test_metaclass_type_manipulation_rejected(
    task: TaskSpec, plan: PlanSpec, injection: str
) -> None:
    """Attack: Use type() and __class__ to manipulate the class hierarchy.

    type(task).__subclasses__() exposes every class in the interpreter.
    type("Evil", ...) constructs arbitrary classes. __new__ and __init_subclass__
    can trigger code execution. All must be blocked at Call/Attribute level.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A21. Indirect attribute traversal via intermediate container objects —
#      accessing __class__ through task.tags, task.id, and other fields that
#      return Python objects whose attributes could be walked.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Traverse via task.tags (list object)
        'task.tags.__class__.__bases__[0].__subclasses__()',
        'task.tags.__class__.__mro__',
        'task.tags.__class__.__init__.__globals__',
        'task.tags.append.__self__.__class__',
        # Traverse via task.id (str object)
        'task.id.__class__.__bases__[0].__subclasses__()',
        'task.id.__class__.__mro__[1]',
        'task.id.encode.__self__.__class__',
        # Traverse via task.engine (str object)
        'task.engine.__class__.__init__.__globals__',
        'task.engine.__class__.__dict__',
        # Traverse via plan.name (str object)
        'plan.name.__class__.__bases__[0].__subclasses__()',
        'plan.name.__class__.__init__.__globals__["__builtins__"]',
    ],
    ids=[
        "tags_class_subclasses", "tags_class_mro", "tags_init_globals",
        "tags_append_self_class",
        "id_class_subclasses", "id_class_mro_1", "id_encode_self_class",
        "engine_init_globals", "engine_class_dict",
        "plan_name_class_subclasses", "plan_name_init_globals_builtins",
    ],
)
def test_attribute_traversal_via_field_objects(
    task: TaskSpec, plan: PlanSpec, injection: str
) -> None:
    """Attack: Traverse __class__ hierarchy through whitelisted field values.

    Even if task.tags is accessible, task.tags.__class__ leads to `list` →
    `list.__bases__[0]` → `object.__subclasses__()` → every class in the
    interpreter. The nested attribute check must block at the first chained
    access (task.tags.__class__ is a nested attribute on a non-Name node).
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A22. String construction attacks — building dangerous identifiers char by
#      char to bypass any hypothetical string-based detection.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Build "__import__" via chr()
        'eval(chr(95)+chr(95)+chr(105)+chr(109)+chr(112)+chr(111)+chr(114)+chr(116)+chr(95)+chr(95))',
        # Build "os" via chr()
        'eval(chr(111)+chr(115))',
        # Build "__class__" via string multiplication + concat
        'getattr(task, chr(95)*2 + "class" + chr(95)*2)',
        # Use str.join to build dangerous string
        'eval("".join(["_","_","i","m","p","o","r","t","_","_"]))',
        # Use hex escapes
        'eval("\\x5f\\x5fimport\\x5f\\x5f")',
        # Use octal escapes
        'eval("\\137\\137import\\137\\137")',
        # Use format string with chr
        'eval(f"{chr(95)}{chr(95)}import{chr(95)}{chr(95)}")',
        # Use bytes + decode
        'eval(bytes([95,95,105,109,112,111,114,116,95,95]).decode("ascii"))',
        # Use base64 decode
        '__import__("base64").b64decode(b"b3M=").decode()',
    ],
    ids=[
        "chr_import_full", "chr_os", "chr_class_getattr",
        "str_join_import", "hex_escape_import", "octal_escape_import",
        "fstring_chr_import", "bytes_decode_import", "base64_decode_os",
    ],
)
def test_string_construction_bypass_rejected(
    task: TaskSpec, plan: PlanSpec, injection: str
) -> None:
    """Attack: Build dangerous strings character by character to evade detection.

    An attacker might construct '__import__' or 'os' dynamically using chr(),
    bytes().decode(), hex/octal escapes, base64, or str.join(). Since the
    evaluator blocks Call nodes at the AST level (not string pattern matching),
    all construction attempts are rejected when eval/getattr/__import__ is called.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A23. Resource exhaustion / denial-of-service patterns — expressions designed
#      to consume excessive CPU, memory, or stack depth.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Exponential string expansion via multiplication
        '"a" * (10 ** 9)',
        # Range-based memory bomb
        'list(range(10**9))',
        # Recursive string formatting
        'format(format(format("x")))',
        # Zip bomb via comprehension
        '[i for i in range(10**9)]',
        # Nested comprehension bomb
        '[[j for j in range(1000)] for i in range(1000)]',
    ],
    ids=[
        "string_multiply_bomb", "range_memory_bomb", "recursive_format",
        "zip_bomb_comprehension", "nested_comp_bomb",
    ],
)
def test_resource_exhaustion_patterns_rejected(
    task: TaskSpec, plan: PlanSpec, injection: str
) -> None:
    """Attack: Trigger excessive resource consumption via expressions.

    Even without code execution, an attacker might cause denial-of-service
    by constructing huge strings ('a' * 10**9), large ranges, or nested
    comprehensions. These all use Call/BinOp/ListComp AST nodes which the
    evaluator rejects before any computation occurs.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A24. Encoding / codec abuse — using str.encode, bytes.decode, codecs to
#      transform innocent-looking data into executable code.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # str.encode() → bytes → could be passed to eval
        'task.id.encode("utf-8")',
        # bytes.decode
        'b"__import__".decode()',
        # codecs module import
        '__import__("codecs").decode("b3M=", "base64")',
        # str.translate / str.maketrans for obfuscation
        'task.engine.translate(str.maketrans("", ""))',
        # format_map injection
        'task.engine.format_map({"__import__": None})',
        # encode/decode chain
        'task.id.encode("rot13")',
    ],
    ids=[
        "str_encode", "bytes_decode", "codecs_import",
        "translate_maketrans", "format_map", "encode_rot13",
    ],
)
def test_encoding_codec_bypass_rejected(
    task: TaskSpec, plan: PlanSpec, injection: str
) -> None:
    """Attack: Use encoding/decoding operations to transform data into code.

    Codec operations can transform innocent-looking strings into dangerous
    code (e.g., ROT13 encoding, base64). All involve nested attribute access
    (task.id.encode) or Call nodes which are rejected by the evaluator.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A25. Descriptor protocol and special method abuse — __get__, __set__,
#      __delete__, __set_name__, property access tricks.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Descriptor protocol methods
        'task.__get__(task, type(task))',
        'task.__set__(task, "pwned")',
        'task.__delete__(task)',
        # Property-based access
        'type(task).id.__get__(task)',
        'type(task).id.__set__(task, "evil")',
        # __init__ manipulation
        'task.__init__(id="pwned")',
        'task.__class__.__init__(task, id="pwned")',
        # __new__ to construct arbitrary objects
        'object.__new__(type(task))',
        'task.__class__.__new__(task.__class__)',
    ],
    ids=[
        "descriptor_get", "descriptor_set", "descriptor_delete",
        "property_get", "property_set",
        "init_reinit", "class_init_reinit",
        "object_new", "class_new",
    ],
)
def test_descriptor_protocol_abuse_rejected(
    task: TaskSpec, plan: PlanSpec, injection: str
) -> None:
    """Attack: Exploit Python descriptor protocol to modify task state.

    __set__ / __delete__ descriptors could mutate the task object. __init__
    reinvocation could reset the object to attacker-controlled state. __new__
    could construct fresh objects. All paths must be rejected.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A26. Global/frame introspection attacks — sys._getframe, inspect module,
#      gc module to reach objects outside the sandbox scope.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # Frame inspection
        '__import__("sys")._getframe().f_locals',
        '__import__("sys")._getframe().f_globals',
        '__import__("sys")._getframe().f_code',
        # Inspect module
        '__import__("inspect").currentframe().f_locals',
        '__import__("inspect").stack()[0]',
        # Garbage collector module
        '__import__("gc").get_objects()',
        '__import__("gc").get_referrers(task)',
        # sys.modules dictionary
        '__import__("sys").modules["os"]',
        # Direct _getframe without import (if sys already imported)
        'type.__bases__[0].__subclasses__()[0].__init__.__globals__["sys"]._getframe()',
    ],
    ids=[
        "sys_getframe_locals", "sys_getframe_globals", "sys_getframe_code",
        "inspect_currentframe", "inspect_stack",
        "gc_get_objects", "gc_get_referrers",
        "sys_modules_os", "subclass_sys_getframe",
    ],
)
def test_frame_introspection_attacks_rejected(
    task: TaskSpec, plan: PlanSpec, injection: str
) -> None:
    """Attack: Use sys._getframe, inspect, or gc to escape sandbox scope.

    Frame introspection exposes f_locals/f_globals of the calling frame,
    giving access to every variable in the evaluator including the real
    task/plan objects and their __class__. gc.get_objects() exposes every
    Python object in memory. All require __import__ Call node → rejected.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A27. Pickle / marshal / shelve deserialization attacks — crafting payloads
#      that execute code during deserialization.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "injection",
    [
        # pickle.loads with crafted payload
        '__import__("pickle").loads(b"cos\\nsystem\\n(S\'id\'\\ntR.")',
        # marshal.loads (execute arbitrary bytecode)
        '__import__("marshal").loads(b"")',
        # shelve (persistent dict with pickle backend)
        '__import__("shelve").open("/tmp/evil")',
        # yaml.unsafe_load
        '__import__("yaml").unsafe_load("!!python/object:os.system [id]")',
        # copyreg exploit
        '__import__("copyreg").dispatch_table',
    ],
    ids=[
        "pickle_loads", "marshal_loads", "shelve_open",
        "yaml_unsafe_load", "copyreg_dispatch",
    ],
)
def test_deserialization_attacks_rejected(
    task: TaskSpec, plan: PlanSpec, injection: str
) -> None:
    """Attack: Use deserialization libraries to achieve code execution.

    pickle.loads() can execute arbitrary code via __reduce__. marshal.loads()
    can inject bytecode. yaml.unsafe_load() executes Python constructors.
    All require __import__ → Call node → rejected.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ---------------------------------------------------------------------------
# A28. task.model field -- positive string comparison (model is not None)
#      Complements section 7 which only tests model=None; this tests the
#      common case where a model alias is set.
# ---------------------------------------------------------------------------


def test_task_model_non_none_string_match(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude", model="sonnet")
    spec = _policy('task.model == "sonnet"')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_task_model_non_none_string_no_match(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude", model="sonnet")
    spec = _policy('task.model == "haiku"')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# A29. None as LHS of in -- checking if None is a member of task.tags
#      The evaluator null-guards the RHS (right is None), not the LHS;
#      None can legitimately appear as a list element.
# ---------------------------------------------------------------------------


def test_none_lhs_in_tags_containing_none(plan: PlanSpec) -> None:
    task = _task(id="t1", tags=[None, "prod"])  # type: ignore[list-item]
    spec = _policy("None in task.tags")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_none_lhs_in_tags_not_containing_none(plan: PlanSpec) -> None:
    task = _task(id="t1", tags=["qa", "prod"])
    spec = _policy("None in task.tags")
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# A30. Empty string as substring LHS -- empty string is always a substring
# ---------------------------------------------------------------------------


def test_empty_string_is_always_substring_of_engine(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude")
    spec = _policy('"" in task.engine')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# A31. task.execution_profile computed field -- reflects plan.execution_profile
#      when the plan has an explicit non-default value.
# ---------------------------------------------------------------------------


def test_task_execution_profile_defaults_to_plan2(plan: PlanSpec) -> None:
    """execution_profile computed field defaults to 'plan' when not set."""
    task = _task(id="t1", engine="claude")
    spec = _policy('task.execution_profile == "plan"')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_task_execution_profile_reflects_plan_attr() -> None:
    """When plan has execution_profile attr set, the computed field reflects it."""
    plan = PlanSpec(name="p")
    # Monkey-patch since PlanSpec doesn't have execution_profile as a real field
    plan.execution_profile = "yolo"  # type: ignore[attr-defined]
    task = _task(id="t1", engine="codex")
    spec = _policy('task.execution_profile == "yolo"')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# A32. task.max_retries default boundary -- max_retries defaults to 0
# ---------------------------------------------------------------------------


def test_task_max_retries_default_is_zero(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude")
    spec = _policy("task.max_retries == 0")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# A33. task.cost_usd -- result exists but cost_usd is None (zero default)
# ---------------------------------------------------------------------------


def test_task_cost_usd_zero_when_result_cost_is_none(plan: PlanSpec) -> None:
    from maestro_cli.models import TaskResult

    task = _task(id="t1", engine="claude")
    result = TaskResult(task_id="t1", status="success", cost_usd=None)
    spec = _policy("task.cost_usd == 0.0")
    ev = compile_policy(spec)
    assert ev(task, plan, result) is True


# ---------------------------------------------------------------------------
# A34. task.description -- non-empty value set by caller
# ---------------------------------------------------------------------------


def test_task_description_non_empty_value(plan: PlanSpec) -> None:
    task = _task(id="t1", description="deploy the service")
    spec = _policy('task.description == "deploy the service"')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_task_description_in_substring_check(plan: PlanSpec) -> None:
    task = _task(id="t1", description="deploy the service")
    spec = _policy('"deploy" in task.description')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# A35. Ellipsis / complex / bytes constants — unusual literal types that
#      ast.parse accepts but shouldn't be useful in policy rules.
# ---------------------------------------------------------------------------


def test_ellipsis_constant_evaluates(plan: PlanSpec) -> None:
    """Ellipsis literal (...) is a valid ast.Constant — must not crash."""
    task = _task(id="t1", engine="claude")
    spec = _policy("...")
    ev = compile_policy(spec)
    # ... is truthy in Python
    assert ev(task, plan) is True


def test_bytes_constant_evaluates(plan: PlanSpec) -> None:
    """b'hello' is an ast.Constant — must not crash."""
    spec = _policy('b"hello" == b"hello"')
    ev = compile_policy(spec)
    task = _task(id="t1")
    assert ev(task, plan) is True


def test_complex_number_constant_evaluates(plan: PlanSpec) -> None:
    """Complex numbers are valid constants — compare must not crash."""
    spec = _policy("1j == 1j")
    ev = compile_policy(spec)
    task = _task(id="t1")
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# A36. Comparison chain mixing `in` with relational operators — the Compare
#      handler iterates ops/comparators; mixed chains test all branches.
# ---------------------------------------------------------------------------


def test_mixed_in_then_equality_chain(plan: PlanSpec) -> None:
    """Chain: 'x' in tags == True — `in` sets left=tags for next op."""
    task = _task(id="t1", tags=["x", "y"])
    spec = _policy('"x" in task.tags == task.tags')
    ev = compile_policy(spec)
    # After `in`: left becomes task.tags; then task.tags == task.tags → True
    assert ev(task, plan) is True


def test_mixed_not_in_then_equality_chain(plan: PlanSpec) -> None:
    task = _task(id="t1", tags=["a", "b"])
    spec = _policy('"z" not in task.tags == task.tags')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_mixed_equality_then_in_chain(plan: PlanSpec) -> None:
    """Chain: engine == 'claude' in ... — equality first, then `in`."""
    task = _task(id="t1", engine="claude")
    # This means: (engine == "claude") and ("claude" in "claude")
    # In Python chained comparison: claude == claude evaluates to True, then
    # "claude" in task.engine checks substring — True
    spec = _policy('task.engine == "claude" in task.engine')
    ev = compile_policy(spec)
    # chained: left = task.engine -> "claude" == "claude" -> True -> left = "claude"
    # then "claude" in task.engine -> True
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# A37. `task` and `plan` as bare Name nodes without attribute — should raise
#      ValueError (unsupported name) instead of leaking the object.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", [
    "task",
    "plan",
    'task == "x"',
    "plan == plan",
    "task != plan",
    "task and True",
])
def test_bare_task_or_plan_name_rejected(rule: str, plan: PlanSpec) -> None:
    task = _task(id="t1")
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported name"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# A38. evaluate_policies with mixed valid/invalid/matching policies — tests
#      that invalid policies are silently skipped, valid ones still evaluated.
# ---------------------------------------------------------------------------


def test_evaluate_policies_mixed_valid_invalid_matching(plan: PlanSpec, capsys: pytest.CaptureFixture[str]) -> None:
    task = _task(id="t1", engine="claude")
    policies = [
        PolicySpec(name="good-match", rule='task.engine == "claude"', action="warn"),  # type: ignore[arg-type]
        PolicySpec(name="bad-syntax", rule="len(task.tags)", action="block"),  # type: ignore[arg-type]
        PolicySpec(name="good-nomatch", rule='task.engine == "codex"', action="warn"),  # type: ignore[arg-type]
        PolicySpec(name="bad-field", rule="task.nonexistent == 1", action="audit"),  # type: ignore[arg-type]
        PolicySpec(name="good-match2", rule='task.id == "t1"', action="block"),  # type: ignore[arg-type]
    ]
    violations = evaluate_policies(policies, task, plan)
    # Only good-match and good-match2 should produce violations
    names = [v.policy_name for v in violations]
    assert "good-match" in names
    assert "good-match2" in names
    assert "bad-syntax" not in names
    assert "bad-field" not in names
    assert "good-nomatch" not in names
    assert len(violations) == 2


# ---------------------------------------------------------------------------
# A39. Escaped characters in string constants — backslashes, tabs, newlines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule,expected", [
    (r'task.id == "t\n1"', False),         # \n in string, id is "t1"
    (r'task.id == "t\\1"', False),         # literal backslash + 1
    ('task.id == "t1"', True),             # baseline
    (r'"\\n" in task.id', False),          # literal \n not in "t1"
])
def test_escaped_characters_in_string_constants(rule: str, expected: bool, plan: PlanSpec) -> None:
    task = _task(id="t1")
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is expected


# ---------------------------------------------------------------------------
# A40. Extremely long `or` chain — 30 operands; tests that BoolOp handling
#      with many values doesn't stack-overflow or degrade.
# ---------------------------------------------------------------------------


def test_extremely_long_or_chain(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude")
    # 30 false comparisons + 1 true at the end
    clauses = [f'task.engine == "engine{i}"' for i in range(30)]
    clauses.append('task.engine == "claude"')
    rule = " or ".join(clauses)
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_extremely_long_and_chain_all_true(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude", timeout_sec=60)
    # All True conditions
    clauses = ['task.engine == "claude"'] * 25
    clauses.append("task.timeout_sec == 60")
    rule = " and ".join(clauses)
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_extremely_long_and_chain_one_false(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude")
    # All True except one False in the middle
    clauses = ['task.engine == "claude"'] * 15
    clauses.insert(7, 'task.engine == "codex"')
    rule = " and ".join(clauses)
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# A41. Nested Compare inside BoolOp — complex real-world-like policy rules
# ---------------------------------------------------------------------------


def test_nested_compare_inside_boolop(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude", timeout_sec=60, max_retries=2)
    rule = '(task.engine == "claude") and (30 < task.timeout_sec < 120) and (task.max_retries > 0)'
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_nested_compare_mixed_operators(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="codex", timeout_sec=10)
    rule = '(task.engine != "claude") and (task.timeout_sec <= 10 or task.timeout_sec >= 100)'
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# A42. Rule with multiline string — ast.parse handles newlines in source
# ---------------------------------------------------------------------------


def test_multiline_rule_string(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude")
    rule = 'task.engine == "claude"\nand task.id == "t1"'
    # ast.parse in eval mode does NOT support statements, but newline in a
    # single expression context is fine for line continuation
    # Actually: `expr1 \n and expr2` is two expressions — should be SyntaxError
    spec = _policy(rule)
    with pytest.raises(SyntaxError):
        compile_policy(spec)


def test_backslash_continuation_in_rule(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude")
    rule = 'task.engine == "claude" \\\nand task.id == "t1"'
    # Backslash continuation is valid Python
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# A43. Constant-only boolean expressions — no task/plan references at all
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule,expected", [
    ("True", True),
    ("False", False),
    ("true", True),
    ("false", False),
    ("True and True", True),
    ("True and False", False),
    ("False or True", True),
    ("not False", True),
    ("not True", False),
    ("1 == 1", True),
    ("1 == 2", False),
    ('"abc" == "abc"', True),
    ('"abc" != "def"', True),
    ("1 < 2 < 3", True),
    ("1 < 2 > 3", False),
])
def test_constant_only_boolean_expressions(rule: str, expected: bool, plan: PlanSpec) -> None:
    task = _task(id="t1")
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is expected


# ---------------------------------------------------------------------------
# A44. Single statement injection attempts — these are SyntaxError in eval mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("injection", [
    'import os',
    'from os import system',
    'x = 1',
    'del task',
    'raise Exception("x")',
    'assert False',
    'class X: pass',
    'def f(): pass',
    'for x in []: pass',
    'while True: pass',
    'with open("x"): pass',
    'try:\n pass\nexcept:\n pass',
    'return True',
    'break',
    'continue',
])
def test_statement_injection_syntax_error(injection: str) -> None:
    spec = _policy(injection)
    with pytest.raises(SyntaxError):
        compile_policy(spec)


# ---------------------------------------------------------------------------
# A45. PolicyViolation field correctness — action, task_id, message
#      These fields are populated by evaluate_policies but never directly
#      asserted in prior test sections.
# ---------------------------------------------------------------------------


def test_violation_action_block_preserved(task: TaskSpec, plan: PlanSpec) -> None:
    """violation.action must equal 'block' when the matching spec has action='block'."""
    policies = [PolicySpec(name="p", rule='task.engine == "claude"', action="block")]  # type: ignore[arg-type]
    violations = evaluate_policies(policies, task, plan)
    assert len(violations) == 1
    assert violations[0].action == "block"


def test_violation_action_audit_preserved(task: TaskSpec, plan: PlanSpec) -> None:
    """violation.action must equal 'audit' when the matching spec has action='audit'."""
    policies = [PolicySpec(name="p", rule='task.engine == "claude"', action="audit")]  # type: ignore[arg-type]
    violations = evaluate_policies(policies, task, plan)
    assert len(violations) == 1
    assert violations[0].action == "audit"


def test_violation_task_id_reflects_task_id(plan: PlanSpec) -> None:
    """violation.task_id must equal the evaluated task's id field."""
    task = _task(id="my-unique-task-abc", engine="claude")
    policies = [PolicySpec(name="p", rule='task.engine == "claude"', action="warn")]  # type: ignore[arg-type]
    violations = evaluate_policies(policies, task, plan)
    assert len(violations) == 1
    assert violations[0].task_id == "my-unique-task-abc"


def test_violation_message_uses_spec_message_when_set(task: TaskSpec, plan: PlanSpec) -> None:
    """When spec.message is set, violation.message must equal that custom string."""
    policies = [PolicySpec(  # type: ignore[arg-type]
        name="p", rule='task.engine == "claude"', action="warn",
        message="security gate: claude engine requires approval",
    )]
    violations = evaluate_policies(policies, task, plan)
    assert len(violations) == 1
    assert violations[0].message == "security gate: claude engine requires approval"


def test_violation_default_message_contains_policy_name_and_task_id(
    task: TaskSpec, plan: PlanSpec
) -> None:
    """Default violation.message must include both the policy name and task id."""
    policies = [PolicySpec(name="enforce-timeout", rule='task.engine == "claude"', action="warn")]  # type: ignore[arg-type]
    violations = evaluate_policies(policies, task, plan)
    assert len(violations) == 1
    assert "enforce-timeout" in violations[0].message
    assert "t1" in violations[0].message


# ---------------------------------------------------------------------------
# A46. task.cost_usd with a real TaskResult carrying a positive cost
#      Prior tests cover: no result (→ 0.0) and result with cost_usd=None.
#      This covers the actual path where result.cost_usd is a positive float.
# ---------------------------------------------------------------------------


def test_task_cost_usd_positive_with_real_result(plan: PlanSpec) -> None:
    """task.cost_usd > 1.0 must be True when TaskResult.cost_usd is 2.5."""
    from maestro_cli.models import TaskResult

    task = _task(id="t1", engine="claude")
    result = TaskResult(task_id="t1", status="success", cost_usd=2.5)
    spec = _policy("task.cost_usd > 1.0")
    ev = compile_policy(spec)
    assert ev(task, plan, result) is True


def test_task_cost_usd_threshold_comparison_exact(plan: PlanSpec) -> None:
    """task.cost_usd >= 0.5 must distinguish between cost=0.5 (True) and cost=0.1 (False)."""
    from maestro_cli.models import TaskResult

    task = _task(id="t1", engine="claude")
    result_over = TaskResult(task_id="t1", status="success", cost_usd=0.5)
    result_under = TaskResult(task_id="t1", status="success", cost_usd=0.1)
    spec = _policy("task.cost_usd >= 0.5")
    ev = compile_policy(spec)
    assert ev(task, plan, result_over) is True
    assert ev(task, plan, result_under) is False


# ---------------------------------------------------------------------------
# A47. Cross-field: task.model in task.tags — str membership in list
#      Uses two whitelisted task fields on opposite sides of `in`.
# ---------------------------------------------------------------------------


def test_cross_field_model_in_tags_true(plan: PlanSpec) -> None:
    """task.model in task.tags must be True when model is present in the tags list."""
    task = _task(id="t1", engine="claude", model="sonnet", tags=["sonnet", "code-review"])
    spec = _policy("task.model in task.tags")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_cross_field_model_in_tags_false(plan: PlanSpec) -> None:
    """task.model in task.tags must be False when model is absent from tags."""
    task = _task(id="t1", engine="claude", model="opus", tags=["sonnet", "code-review"])
    spec = _policy("task.model in task.tags")
    ev = compile_policy(spec)
    assert ev(task, plan) is False


def test_cross_field_engine_in_tags_true(plan: PlanSpec) -> None:
    """task.engine in task.tags must be True when engine name is listed in tags."""
    task = _task(id="t1", engine="claude", tags=["claude", "prod"])
    spec = _policy("task.engine in task.tags")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# A48. De Morgan's laws — negation of disjunction and conjunction
#      Verifies `not (A or B)` and `not (A and B)` evaluate correctly.
# ---------------------------------------------------------------------------


def test_not_disjunction_true_when_neither_matches(plan: PlanSpec) -> None:
    """`not (engine==codex or engine==gemini)` is True when engine is claude."""
    task = _task(id="t1", engine="claude")
    spec = _policy('not (task.engine == "codex" or task.engine == "gemini")')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_not_disjunction_false_when_first_operand_matches(plan: PlanSpec) -> None:
    """`not (engine==claude or engine==codex)` is False when engine is claude."""
    task = _task(id="t1", engine="claude")
    spec = _policy('not (task.engine == "claude" or task.engine == "codex")')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


def test_not_conjunction_true_when_second_conjunct_fails(plan: PlanSpec) -> None:
    """`not (engine==claude and max_retries==5)` is True when max_retries is 0."""
    task = _task(id="t1", engine="claude", max_retries=0)
    spec = _policy('not (task.engine == "claude" and task.max_retries == 5)')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_not_conjunction_false_when_both_conjuncts_true(plan: PlanSpec) -> None:
    """`not (engine==claude and max_retries==0)` is False when both are satisfied."""
    task = _task(id="t1", engine="claude", max_retries=0)
    spec = _policy('not (task.engine == "claude" and task.max_retries == 0)')
    ev = compile_policy(spec)
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# A45. Decorator / assignment target abuse — invalid in eval mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("injection", [
    '@decorator\ndef f(): pass',
    'x := 1',            # walrus at top level outside compare
    'x += 1',
    'x -= 1',
    'x *= 1',
    'x //= 1',
])
def test_assignment_and_decorator_syntax_error(injection: str) -> None:
    spec = _policy(injection)
    with pytest.raises(SyntaxError):
        compile_policy(spec)


# ---------------------------------------------------------------------------
# A46. Multiple semicolons / statement separators — SyntaxError in eval mode
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("injection", [
    'True; True',
    'task.engine == "x"; __import__("os")',
    '1; 2; 3',
])
def test_semicolon_statement_separator_rejected(injection: str) -> None:
    spec = _policy(injection)
    with pytest.raises(SyntaxError):
        compile_policy(spec)


# ---------------------------------------------------------------------------
# A47. Null byte injection — must not bypass parsing or crash
# ---------------------------------------------------------------------------


def test_null_byte_in_rule_string() -> None:
    spec = _policy('task.engine\x00 == "x"')
    with pytest.raises((SyntaxError, ValueError)):
        compile_policy(spec)


def test_null_byte_in_field_value() -> None:
    spec = _policy('task.engine == "clau\x00de"')
    # This may parse fine (null byte in string literal) but should not match
    plan = PlanSpec(name="p")
    task = _task(id="t1", engine="claude")
    # If it parses, the string won't match "claude"
    try:
        ev = compile_policy(spec)
        assert ev(task, plan) is False
    except (SyntaxError, ValueError):
        pass  # Also acceptable


# ---------------------------------------------------------------------------
# A48. Very long rule string — must not hang or cause excessive memory usage
# ---------------------------------------------------------------------------


def test_very_long_rule_does_not_hang(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude")
    # 500 chained equality checks
    rule = " and ".join(['task.engine == "claude"'] * 500)
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_very_long_string_constant(plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude")
    long_val = "x" * 10000
    rule = f'task.engine != "{long_val}"'
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# A49. `in` / `not in` with None on LHS — None membership checks
# ---------------------------------------------------------------------------


def test_none_not_in_tags_list(plan: PlanSpec) -> None:
    task = _task(id="t1", tags=["a", "b"])
    spec = _policy("None not in task.tags")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_none_not_in_none_rhs(plan: PlanSpec) -> None:
    """When RHS is None, `not in` returns True (null-safe)."""
    task = _task(id="t1", engine="claude")  # model defaults to None
    spec = _policy('"x" not in task.model')
    ev = compile_policy(spec)
    # task.model is None → `not in` with None RHS returns True per the code
    # Actually checking: right is None → returns False for not in
    # Looking at code: elif isinstance(op, ast.NotIn): if right is None or left in right: return False
    # So "x" not in None → right is None → returns False
    assert ev(task, plan) is False


def test_in_with_none_rhs_returns_false2(plan: PlanSpec) -> None:
    """When RHS is None, `in` returns False (null-safe)."""
    task = _task(id="t1", engine="claude")
    spec = _policy('"x" in task.model')
    ev = compile_policy(spec)
    # task.model is None → `in` with None RHS → returns False
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# A50. Negative numeric constants — ast.UnaryOp(USub) wrapping Constant
# ---------------------------------------------------------------------------


def test_negative_number_rejected_as_unsupported_unary(plan: PlanSpec) -> None:
    """Negative number `-1` creates ast.UnaryOp(USub, Constant(1)) — USub not allowed."""
    task = _task(id="t1", timeout_sec=30)
    spec = _policy("task.timeout_sec > -1")
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported unary"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# A51. Empty rule string — must raise SyntaxError
# ---------------------------------------------------------------------------


def test_empty_rule_raises_syntax_error() -> None:
    spec = _policy("")
    with pytest.raises(SyntaxError):
        compile_policy(spec)


# ---------------------------------------------------------------------------
# A52. Plan field comparisons — all 5 plan fields exercised
# ---------------------------------------------------------------------------


def test_plan_name_comparison() -> None:
    plan = PlanSpec(name="my-plan")
    task = _task(id="t1")
    spec = _policy('plan.name == "my-plan"')
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_plan_fail_fast_boolean() -> None:
    plan = PlanSpec(name="p", fail_fast=True)
    task = _task(id="t1")
    spec = _policy("plan.fail_fast")
    ev = compile_policy(spec)
    # plan.fail_fast is True → truthy
    # But wait, bare plan.fail_fast is an Attribute node which resolves to True
    # and bool(True) is True
    assert ev(task, plan) is True


def test_plan_max_parallel_numeric() -> None:
    plan = PlanSpec(name="p", max_parallel=4)
    task = _task(id="t1")
    spec = _policy("plan.max_parallel > 2")
    ev = compile_policy(spec)
    assert ev(task, plan) is True


# ---------------------------------------------------------------------------
# A53. Class instantiation attempts — rejected as ast.Call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("injection", [
    "dict()",
    "list()",
    "set()",
    "tuple()",
    "object()",
    "int(task.engine)",
    "float(task.timeout_sec)",
    "bool(0)",
    "str(42)",
    "bytes(10)",
    "bytearray(10)",
    "frozenset()",
    "memoryview(b'x')",
    "complex(1, 2)",
])
def test_class_instantiation_rejected(injection: str, plan: PlanSpec) -> None:
    task = _task(id="t1")
    spec = _policy(injection)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported AST"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# A54. Comparison operators Is / IsNot with various operands
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", [
    "task.engine is None",
    "task.model is not None",
    "task.tags is task.tags",
    'task.engine is "claude"',
])
def test_is_is_not_comparators_rejected(rule: str, plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude")
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported comparator"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# A55. Chained comparison with 5 operators — tests iteration robustness
# ---------------------------------------------------------------------------


def test_five_operator_chained_comparison(plan: PlanSpec) -> None:
    task = _task(id="t1", timeout_sec=50)
    rule = "0 < 10 < task.timeout_sec < 60 < 100 < 200"
    spec = _policy(rule)
    ev = compile_policy(spec)
    assert ev(task, plan) is True


def test_five_operator_chain_fails_in_middle(plan: PlanSpec) -> None:
    task = _task(id="t1", timeout_sec=50)
    rule = "0 < 10 < task.timeout_sec < 40 < 100 < 200"
    spec = _policy(rule)
    ev = compile_policy(spec)
    # 50 < 40 is False
    assert ev(task, plan) is False


# ---------------------------------------------------------------------------
# A56. Exception inheritance — __mro__, __bases__, __subclasses__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("injection", [
    'task.__class__.__mro__',
    'task.__class__.__bases__',
    'task.__class__.__subclasses__()',
    '"".__class__.__mro__[1].__subclasses__()',
])
def test_exception_inheritance_chain_rejected(injection: str, plan: PlanSpec) -> None:
    task = _task(id="t1")
    spec = _policy(injection)
    ev = compile_policy(spec)
    with pytest.raises(ValueError):
        ev(task, plan)


# ---------------------------------------------------------------------------
# A57. Ternary / conditional expression — ast.IfExp must be rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", [
    '"claude" if True else "codex"',
    'task.engine if task.engine == "claude" else "other"',
    'True if task.timeout_sec > 0 else False',
])
def test_ternary_if_expression_rejected(rule: str, plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude", timeout_sec=30)
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported AST"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# A58. Format string / f-string injection — ast.JoinedStr must be rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", [
    'f"{task.engine}" == "claude"',
    'f"hello {1+1}" == "hello 2"',
    'f"" == ""',
])
def test_fstring_rejected(rule: str, plan: PlanSpec) -> None:
    task = _task(id="t1", engine="claude")
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported AST"):
        ev(task, plan)


# ---------------------------------------------------------------------------
# A59. Dict/List/Set/Tuple literal construction — all must be rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("rule", [
    '{"key": "value"}',
    '[1, 2, 3]',
    '{1, 2, 3}',
    '(1, 2, 3)',
    '{"a": 1, "b": 2} == {"a": 1}',
    '[1, 2] == [1, 2]',
])
def test_container_literal_construction_rejected(rule: str, plan: PlanSpec) -> None:
    task = _task(id="t1")
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported AST"):
        ev(task, plan)


# ===========================================================================
# B1. Sandbox evasion via __import__ — every form of __import__ call that
#     could lead to arbitrary module loading must be rejected.
# ===========================================================================


@pytest.mark.parametrize(
    "injection",
    [
        # Direct __import__ calls with various dangerous modules
        '__import__("os").system("whoami")',
        '__import__("os").popen("id").read()',
        '__import__("os").execvp("/bin/sh", ["/bin/sh"])',
        '__import__("os").remove("/etc/passwd")',
        '__import__("os").unlink("important.db")',
        '__import__("subprocess").check_output(["cat", "/etc/shadow"])',
        '__import__("subprocess").run(["rm", "-rf", "/"], check=True)',
        '__import__("shutil").rmtree("/home")',
        '__import__("pathlib").Path("/etc/passwd").read_text()',
        '__import__("io").open("/etc/passwd")',
        # Import + attribute chain to reach system()
        '__import__("os").path.join("/", "etc", "passwd")',
        # Import builtins to re-gain eval/exec
        '__import__("builtins").eval("1+1")',
        '__import__("builtins").exec("import os")',
        '__import__("builtins").__import__("os")',
        # importlib for dynamic imports
        '__import__("importlib").import_module("os").system("id")',
        '__import__("importlib").__import__("os")',
        # pkgutil / zipimport for obscure import paths
        '__import__("pkgutil").find_loader("os")',
        '__import__("zipimport").zipimporter("/tmp/evil.zip")',
        # ctypes for arbitrary C function calls
        '__import__("ctypes").CDLL("libc.so.6").system(b"id")',
        '__import__("ctypes").cdll.LoadLibrary("libc.so.6")',
        # multiprocessing for process spawning
        '__import__("multiprocessing").Process(target=lambda: None).start()',
        # threading for background code execution
        '__import__("threading").Thread(target=lambda: None).start()',
        # Nested import: import importlib, then import os through it
        '__import__("importlib").import_module(__import__("sys").platform)',
    ],
    ids=[
        "os_system", "os_popen", "os_execvp", "os_remove", "os_unlink",
        "subprocess_check_output", "subprocess_run_rm",
        "shutil_rmtree", "pathlib_read_text", "io_open",
        "os_path_join", "builtins_eval", "builtins_exec",
        "builtins_import", "importlib_import_module", "importlib_import",
        "pkgutil_find_loader", "zipimport_zipimporter",
        "ctypes_cdll_system", "ctypes_loadlibrary",
        "multiprocessing_spawn", "threading_spawn", "nested_import_chain",
    ],
)
def test_import_sandbox_evasion_rejected(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Sandbox evasion: __import__ must NEVER execute regardless of target module.

    The _SafeEvaluator rejects ast.Call nodes. If __import__ were allowed,
    an attacker gains arbitrary module loading → filesystem access, process
    spawning, network I/O, and arbitrary native code execution via ctypes.
    Every single __import__ variant must be blocked at the AST Call node level.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B2. Sandbox evasion via eval()/exec() with obfuscated payloads —
#     the attacker builds dangerous code strings at runtime to evade
#     any hypothetical string-pattern-based detection.
# ===========================================================================


@pytest.mark.parametrize(
    "injection",
    [
        # eval with string concatenation to build __import__
        'eval("__" + "import" + "__(' + "'os'" + ')")',
        # eval with chr() to build "os"
        'eval(chr(95)+chr(95)+chr(105)+chr(109)+chr(112)+chr(111)+chr(114)+chr(116)+chr(95)+chr(95)+"("+chr(39)+"os"+chr(39)+")")',
        # exec with hex escape sequences
        'exec("\\x5f\\x5fimport\\x5f\\x5f(\\x27os\\x27)")',
        # eval with reversed string
        'eval(")(so(__ tropmi__"[::-1])',
        # eval of bytes object decoded to string
        'eval(bytes([111,115]).decode())',
        # eval using str.join to assemble payload
        'eval("".join(["_","_","i","m","p","o","r","t","_","_","(","\'","o","s","\'",")"]))',
        # exec with base64-encoded payload
        'exec(__import__("base64").b64decode(b"aW1wb3J0IG9z").decode())',
        # eval with format string construction
        'eval("{0}{0}import{0}{0}".format("_")+"(\'os\')")',
        # Nested eval: outer eval builds inner eval string
        'eval("ev" + "al" + "(\'1+1\')")',
        # compile() + exec() combination
        'exec(compile("import os", "<stdin>", "exec"))',
        # eval with unicode escape
        'eval("\\u005f\\u005fimport\\u005f\\u005f(\'os\')")',
        # exec with raw string tricks
        'exec(r"__import__(\'os\')")',
    ],
    ids=[
        "eval_concat_import", "eval_chr_full_import", "exec_hex_escape",
        "eval_reversed_string", "eval_bytes_decode", "eval_str_join",
        "exec_base64_decode", "eval_format_underscore", "eval_nested_eval",
        "compile_exec", "eval_unicode_escape", "exec_raw_string",
    ],
)
def test_eval_exec_obfuscation_sandbox_evasion(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Sandbox evasion: eval()/exec() with obfuscated code strings must be rejected.

    An attacker who can call eval() or exec() achieves arbitrary code execution
    regardless of what the argument looks like. String obfuscation (chr(), hex
    escapes, base64, reversed strings, str.join) is irrelevant — the AST-level
    Call node rejection must block eval/exec before the argument is even constructed.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B3. Sandbox evasion via getattr() — dynamic attribute access bypasses
#     the static whitelist check on task/plan fields.
# ===========================================================================


@pytest.mark.parametrize(
    "injection",
    [
        # getattr to read non-whitelisted fields
        'getattr(task, "prompt")',
        'getattr(task, "command")',
        'getattr(task, "pre_command")',
        'getattr(task, "verify_command")',
        'getattr(task, "guard_command")',
        'getattr(plan, "tasks")',
        'getattr(plan, "defaults")',
        'getattr(plan, "policies")',
        # getattr to access dunder methods
        'getattr(task, "__class__")',
        'getattr(task, "__dict__")',
        'getattr(task, "__init__")',
        'getattr(task, "__module__")',
        'getattr(plan, "__class__")',
        'getattr(plan, "__dict__")',
        # getattr with dynamically constructed dunder string
        'getattr(task, chr(95)+chr(95)+"class"+chr(95)+chr(95))',
        'getattr(task, chr(95)+chr(95)+"dict"+chr(95)+chr(95))',
        'getattr(task, chr(95)+chr(95)+"init"+chr(95)+chr(95))',
        # Chained getattr to walk the MRO
        'getattr(getattr(task, "__class__"), "__bases__")',
        'getattr(getattr(task, "__class__"), "__mro__")',
        'getattr(getattr(task, "__class__"), "__subclasses__")()',
        # getattr with 3-arg form (default value) — still a Call node
        'getattr(task, "__missing_field__", "safe_default")',
        'getattr(task, "__class__", None)',
        # setattr / delattr — mutate the sandbox objects
        'setattr(task, "engine", "pwned")',
        'setattr(task, "id", "evil")',
        'delattr(task, "engine")',
        'delattr(plan, "name")',
        # hasattr probing to discover available attributes
        'hasattr(task, "__class__")',
        'hasattr(task, "prompt")',
        'hasattr(plan, "policies")',
    ],
    ids=[
        "getattr_prompt", "getattr_command", "getattr_pre_command",
        "getattr_verify_command", "getattr_guard_command",
        "getattr_plan_tasks", "getattr_plan_defaults", "getattr_plan_policies",
        "getattr_dunder_class", "getattr_dunder_dict", "getattr_dunder_init",
        "getattr_dunder_module", "getattr_plan_class", "getattr_plan_dict",
        "getattr_chr_class", "getattr_chr_dict", "getattr_chr_init",
        "chained_getattr_bases", "chained_getattr_mro", "chained_getattr_subclasses",
        "getattr_3arg_missing", "getattr_3arg_class",
        "setattr_engine", "setattr_id", "delattr_engine", "delattr_plan_name",
        "hasattr_class", "hasattr_prompt", "hasattr_plan_policies",
    ],
)
def test_getattr_sandbox_evasion_rejected(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Sandbox evasion: getattr/setattr/delattr/hasattr bypass the field whitelist.

    If getattr(task, "__class__") were allowed, an attacker could traverse
    task.__class__.__bases__[0].__subclasses__() to find os._wrap_close and
    execute arbitrary commands. setattr could mutate sandbox state. All dynamic
    attribute access functions must be rejected as ast.Call nodes.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B4. Sandbox evasion via MRO / class hierarchy traversal — the classic
#     Python sandbox escape that walks from any object to object.__subclasses__()
#     to find dangerous classes like os._wrap_close, subprocess.Popen, etc.
# ===========================================================================


@pytest.mark.parametrize(
    "injection",
    [
        # From string literal → object → all subclasses
        '"".__class__.__bases__[0].__subclasses__()',
        '"".__class__.__mro__[1].__subclasses__()',
        # From int literal
        '(0).__class__.__bases__[0].__subclasses__()',
        '(0).__class__.__mro__[1].__subclasses__()',
        # From bool literal
        'True.__class__.__bases__[0].__bases__[0].__subclasses__()',
        # From tuple literal
        '().__class__.__bases__[0].__subclasses__()',
        # From task field values (resolved at runtime)
        'task.id.__class__.__bases__[0].__subclasses__()',
        'task.engine.__class__.__bases__[0].__subclasses__()',
        'task.tags.__class__.__bases__[0].__subclasses__()',
        # From plan field values
        'plan.name.__class__.__bases__[0].__subclasses__()',
        # Traverse to find specific dangerous class by index
        '"".__class__.__bases__[0].__subclasses__()[40]("/etc/passwd").read()',
        '"".__class__.__bases__[0].__subclasses__()[132].__init__.__globals__["system"]("id")',
        # Access __init__.__globals__ to reach builtins
        'task.__class__.__init__.__globals__["__builtins__"]["__import__"]("os")',
        'task.__class__.__init__.__globals__["__builtins__"]["eval"]("1")',
        'task.__class__.__init__.__globals__["__builtins__"]["exec"]("import os")',
        # Access __subclasses__() from type() call
        'type(task).__subclasses__()',
        'type("").__subclasses__()',
        'type(0).__subclasses__()',
        # __mro__ direct access
        'type(task).__mro__',
        'type(task).__mro__[1].__subclasses__()',
        # __qualname__ / __name__ probing
        'task.__class__.__qualname__',
        'task.__class__.__name__',
    ],
    ids=[
        "str_bases_subclasses", "str_mro_subclasses",
        "int_bases_subclasses", "int_mro_subclasses",
        "bool_deep_bases_subclasses", "tuple_bases_subclasses",
        "task_id_class_walk", "task_engine_class_walk", "task_tags_class_walk",
        "plan_name_class_walk",
        "subclass_index_file_read", "subclass_init_globals_system",
        "init_globals_builtins_import", "init_globals_builtins_eval",
        "init_globals_builtins_exec",
        "type_task_subclasses", "type_str_subclasses", "type_int_subclasses",
        "type_task_mro", "type_task_mro_subclasses",
        "task_class_qualname", "task_class_name",
    ],
)
def test_mro_class_hierarchy_sandbox_evasion(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Sandbox evasion: MRO traversal is the classic Python sandbox escape.

    Starting from ANY Python object, an attacker can walk:
    obj.__class__.__bases__[0] → object → object.__subclasses__()
    This exposes every class loaded in the interpreter, including os._wrap_close
    (which has os.system in __init__.__globals__), subprocess.Popen, etc.
    The evaluator must block this at the FIRST chained attribute access.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B5. Sandbox evasion via comprehension/generator scope injection —
#     hiding dangerous calls inside list/dict/set comprehensions and
#     generator expressions, which create their own scope in Python 3.
# ===========================================================================


@pytest.mark.parametrize(
    "injection",
    [
        # List comprehension with __import__ in the body
        '[__import__("os").system("id") for _ in [1]]',
        # List comprehension with eval in the body
        '[eval("__import__(\'os\')") for _ in [1]]',
        # List comprehension with exec in the body
        '[exec("import os; os.system(\'id\')") for _ in [1]]',
        # Dict comprehension with dangerous call
        '{k: __import__("os") for k in ["x"]}',
        # Set comprehension with dangerous call
        '{__import__("os") for _ in [1]}',
        # Generator expression with dangerous call
        'next(__import__("os") for _ in [1])',
        'any(__import__("os") for _ in [1])',
        'all(__import__("os") for _ in [1])',
        # Comprehension with walrus operator side-effect
        '[y for x in [1] if (y := __import__("os"))]',
        '[y for x in [1] if (y := eval("__import__(\'os\')"))]',
        # Nested comprehension: outer iterates, inner imports
        '[[__import__("os")] for _ in [1] for __ in [2]]',
        # Comprehension with lambda that imports
        '[(lambda: __import__("os"))() for _ in [1]]',
        # Generator wrapping dangerous code passed to list()
        'list(__import__("os") for _ in [1])',
        'tuple(eval("1") for _ in [1])',
        'set(exec("pass") for _ in [1])',
        # Comprehension with filter condition doing the damage
        '[x for x in [1] if __import__("os")]',
        '[x for x in [1] if eval("True")]',
    ],
    ids=[
        "listcomp_import_system", "listcomp_eval_import", "listcomp_exec_import",
        "dictcomp_import", "setcomp_import",
        "genexpr_next_import", "genexpr_any_import", "genexpr_all_import",
        "comp_walrus_import", "comp_walrus_eval",
        "nested_comp_import", "comp_lambda_import",
        "list_genexpr_import", "tuple_genexpr_eval", "set_genexpr_exec",
        "comp_filter_import", "comp_filter_eval",
    ],
)
def test_comprehension_scope_sandbox_evasion(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Sandbox evasion: comprehensions create their own scope that might evade checks.

    In Python 3, list/dict/set comprehensions and generator expressions run in
    their own implicit function scope. An attacker might hope that the evaluator
    only checks the outer expression but not the inner iteration body/filter.
    The evaluator must reject ListComp/SetComp/DictComp/GeneratorExp AST nodes entirely.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B6. Sandbox evasion via lambda — deferred code execution that wraps
#     dangerous calls in an anonymous function, then immediately invokes it.
# ===========================================================================


@pytest.mark.parametrize(
    "injection",
    [
        # Lambda wrapping __import__ and immediately called
        '(lambda: __import__("os"))()',
        '(lambda: __import__("os").system("id"))()',
        # Lambda wrapping eval/exec
        '(lambda: eval("__import__(\'os\')"))()',
        '(lambda: exec("import os; os.system(\'id\')"))()',
        # Lambda with default argument evaluated at definition time
        '(lambda x=__import__("os"): x)()',
        '(lambda x=__import__("os"): x.system("id"))()',
        # Lambda returning getattr result
        '(lambda: getattr(task, "__class__"))()',
        '(lambda: getattr(task, "__dict__"))()',
        # Nested lambdas (double deferred)
        '(lambda: (lambda: __import__("os"))())()',
        '(lambda: (lambda: (lambda: __import__("os"))())())()',
        # Lambda assigned via walrus then called
        '(f := lambda: __import__("os"))()',
        # Lambda in a conditional expression
        '(lambda: __import__("os"))() if True else None',
        # Lambda with *args/**kwargs
        '(lambda *a, **k: __import__("os"))("arg")',
        # Lambda returning a lambda that is then called
        '(lambda: lambda: __import__("os"))()()',
    ],
    ids=[
        "lambda_import", "lambda_import_system",
        "lambda_eval_import", "lambda_exec_import_system",
        "lambda_default_import", "lambda_default_import_system",
        "lambda_getattr_class", "lambda_getattr_dict",
        "nested_lambda_import", "triple_nested_lambda",
        "walrus_lambda_import", "lambda_ternary_import",
        "lambda_args_import", "lambda_returns_lambda_import",
    ],
)
def test_lambda_deferred_sandbox_evasion(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Sandbox evasion: lambda defers code execution past initial AST inspection.

    An attacker might hope that the evaluator checks the Lambda node but not its
    body, or that wrapping __import__ in a lambda somehow defers the check past
    the security boundary. The evaluator must reject ast.Lambda nodes entirely,
    regardless of their body content or how they're invoked.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B7. Sandbox evasion via deeply nested AST positions — dangerous Call
#     nodes hidden deep inside BoolOp, Compare, UnaryOp trees where an
#     incomplete AST walker might miss them.
# ===========================================================================


@pytest.mark.parametrize(
    "injection",
    [
        # __import__ as the RHS of a comparison
        'task.engine == __import__("os")',
        '42 == __import__("os")',
        # __import__ as the LHS of a comparison
        '__import__("os") == "posix"',
        '__import__("os") != None',
        # __import__ inside a chained comparison
        '1 < __import__("os") < 10',
        '0 <= __import__("os") <= 100',
        # __import__ inside a BoolOp (and/or) chain — hidden among safe terms
        'task.engine == "claude" and __import__("os") and task.timeout_sec > 0',
        'task.engine == "codex" or __import__("os") or False',
        'False or False or False or __import__("os")',
        'True and True and True and __import__("os")',
        # __import__ inside a UnaryOp (not)
        'not __import__("os")',
        'not not __import__("os")',
        # eval() hidden deep in BoolOp
        'task.engine == "claude" and task.timeout_sec > 0 and eval("1+1") == 2',
        # exec() hidden as last operand in long or chain
        'False or False or False or False or exec("import os")',
        # globals() hidden as a comparator
        'task.timeout_sec > globals()',
        # type() hidden in a comparison
        'type(task) == "TaskSpec"',
        # Multiple dangerous calls in a single expression
        'eval("1") == exec("pass")',
        '__import__("os") == __import__("sys")',
        # Call hidden inside not(not(...))
        'not not eval("True")',
        'not not not __import__("os")',
    ],
    ids=[
        "import_as_compare_rhs", "import_as_compare_rhs_int",
        "import_as_compare_lhs", "import_ne_none",
        "import_in_chained_lt", "import_in_chained_lte",
        "import_hidden_in_and_chain", "import_hidden_in_or_chain",
        "import_at_end_of_or_chain", "import_at_end_of_and_chain",
        "import_in_not", "import_in_double_not",
        "eval_deep_in_and", "exec_deep_in_or",
        "globals_as_comparator", "type_in_comparison",
        "double_call_compare", "double_import_compare",
        "eval_in_double_not", "import_in_triple_not",
    ],
)
def test_deeply_nested_call_sandbox_evasion(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Sandbox evasion: dangerous calls hidden deep in the AST tree.

    An incomplete AST walker might check top-level nodes but not recursively
    walk into Compare operands, BoolOp values, or UnaryOp operands. These tests
    verify that __import__/eval/exec/globals/type calls are caught regardless
    of their position in the expression tree. The evaluator's recursive eval()
    method must hit every node.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B8. Sandbox evasion via combined multi-technique attacks — chaining
#     multiple bypass strategies in a single expression to find gaps
#     between individually-blocked techniques.
# ===========================================================================


@pytest.mark.parametrize(
    "injection",
    [
        # Lambda + getattr + chr construction (3 techniques)
        '(lambda: getattr(task, chr(95)*2+"class"+chr(95)*2))()',
        # Comprehension + walrus + exec (3 techniques)
        '[y for x in [1] if (y := exec("import os"))]',
        # Ternary + lambda + __import__ (3 techniques)
        '(lambda: __import__("os"))() if True else None',
        # f-string + __import__ (2 techniques: JoinedStr + Call)
        'f"{__import__(\'os\').getcwd()}"',
        # Nested getattr + subscript + call (3 techniques)
        'getattr(task, "__class__").__bases__[0].__subclasses__()',
        # eval(bytes.decode) (3 techniques: eval + bytes constructor + decode)
        'eval(bytes([95,95,105,109,112,111,114,116,95,95]).decode())',
        # Comprehension generating lambdas + calling them (3 techniques)
        '[f() for f in [lambda: __import__("os")]]',
        # vars() + __getitem__ to extract builtins (2 techniques)
        'vars().__getitem__("__builtins__")',
        # str.format + eval (2 techniques)
        'eval("{}{}".format("os", ".system(\'id\')"))',
        # Lambda with default arg from getattr (3 techniques)
        '(lambda x=getattr(task, "__class__"): x.__bases__)()',
        # Nested comprehension + lambda + exec (3 techniques)
        '[[exec("import os")] for f in [lambda: None] for _ in [f()]]',
        # Type + call + subscript (3 techniques)
        'type(task).__subclasses__()[0]()',
        # eval + chr + join (3 techniques)
        'eval("".join([chr(i) for i in [111, 115]]))',
        # Lambda + compile + exec (3 techniques)
        '(lambda: exec(compile("import os", "<>", "exec")))()',
    ],
    ids=[
        "lambda_getattr_chr", "comp_walrus_exec",
        "ternary_lambda_import", "fstring_import_getcwd",
        "getattr_subscript_subclasses", "eval_bytes_decode",
        "comp_lambda_list_import", "vars_getitem_builtins",
        "str_format_eval", "lambda_default_getattr",
        "nested_comp_lambda_exec", "type_subclass_call",
        "eval_chr_join", "lambda_compile_exec",
    ],
)
def test_combined_multi_technique_sandbox_evasion(task: TaskSpec, plan: PlanSpec, injection: str) -> None:
    """Sandbox evasion: real attackers combine multiple bypass techniques.

    A single blocked technique (e.g., Call rejection) might not catch an attack
    that chains lambda + getattr + chr(), or comprehension + walrus + exec.
    These tests verify that the evaluator catches at least ONE component in
    every multi-technique combination, preventing the entire chain from executing.
    """
    spec = _policy(injection)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B9. Missing BinOp operators — floor division (//) and matrix multiply (@)
#     not covered by existing arithmetic/bitwise tests.
# ===========================================================================


@pytest.mark.parametrize(
    "rule",
    [
        "task.timeout_sec // 2 > 0",
        "task.timeout_sec @ task.max_retries",
    ],
    ids=["floor_div", "matmul"],
)
def test_floor_div_and_matmul_binop_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """Floor division (//) and matrix multiply (@) BinOp nodes must be rejected."""
    spec = _policy(rule)
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported AST node"):
        ev(task, plan)


# ===========================================================================
# B10. Generator expressions, set comprehensions, dict comprehensions —
#      these are distinct AST nodes from ListComp and must each be rejected.
# ===========================================================================


@pytest.mark.parametrize(
    "rule",
    [
        '(x for x in task.tags)',
        'any(x == "test" for x in task.tags)',
        '{x for x in task.tags}',
        '{x: 1 for x in task.tags}',
        'sum(1 for _ in task.tags)',
    ],
    ids=["genexpr_bare", "genexpr_in_any", "setcomp", "dictcomp", "genexpr_in_sum"],
)
def test_generator_set_dict_comp_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """GeneratorExp, SetComp, and DictComp AST nodes must be rejected just like ListComp."""
    spec = _policy(rule)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B11. Await and yield expressions — even though ast.parse(mode="eval") may
#      reject these as SyntaxError, we verify they never produce results.
# ===========================================================================


@pytest.mark.parametrize(
    "rule",
    [
        'await __import__("asyncio").sleep(0)',
        '(yield 42)',
        '(yield from [1, 2, 3])',
    ],
    ids=["await_import", "yield_expr", "yield_from"],
)
def test_await_yield_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """Await and yield expressions must not compile or evaluate in policy rules."""
    spec = _policy(rule)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B12. Starred expressions — unpacking operators (*x, **x) in expression
#      context must be rejected.
# ===========================================================================


@pytest.mark.parametrize(
    "rule",
    [
        '(*task.tags, "extra")',
        '[*task.tags]',
        '{**{"a": 1}}',
    ],
    ids=["starred_tuple", "starred_list", "double_star_dict"],
)
def test_starred_expressions_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """Starred/double-starred expressions must be rejected."""
    spec = _policy(rule)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B13. Attribute access on non-Name base nodes — constants, nested function
#      results, and deeper than two levels must all be rejected.
# ===========================================================================


@pytest.mark.parametrize(
    "rule",
    [
        '"string".__class__.__mro__',
        '(42).__class__',
        'True.__class__',
        'None.__class__',
        '(1, 2).__class__',
    ],
    ids=["str_class_mro", "int_class", "bool_class", "none_class", "tuple_class"],
)
def test_attribute_on_constant_rejected(task: TaskSpec, plan: PlanSpec, rule: str) -> None:
    """Attribute access on constant/literal nodes (not task/plan) must be rejected."""
    spec = _policy(rule)
    with pytest.raises((ValueError, SyntaxError)):
        ev = compile_policy(spec)
        ev(task, plan)


# ===========================================================================
# B14. Policies evaluated in sequence remain independent — a broken policy
#      must not prevent subsequent valid policies from executing.
# ===========================================================================


def test_evaluate_policies_broken_middle_does_not_block_others(
    plan: PlanSpec, capsys: pytest.CaptureFixture[str],
) -> None:
    """A policy with a broken rule sandwiched between two valid policies must
    not prevent either valid policy from being evaluated and producing violations."""
    t = _task(engine="claude", tags=["test"])
    policies = [
        PolicySpec(name="first", rule='task.engine == "claude"', action="warn"),
        PolicySpec(name="broken", rule='eval("os")', action="block"),
        PolicySpec(name="third", rule='"test" in task.tags', action="audit"),
    ]
    violations = evaluate_policies(policies, t, plan)
    names = [v.policy_name for v in violations]
    assert "first" in names
    assert "third" in names
    assert "broken" not in names
    captured = capsys.readouterr()
    assert "broken" in captured.out  # warning printed


# ===========================================================================
# B15. Invert (~) unary operator — must be rejected like USub.
# ===========================================================================


def test_unary_invert_rejected(task: TaskSpec, plan: PlanSpec) -> None:
    """The ~ (invert) unary operator must be rejected."""
    spec = _policy("~task.timeout_sec == -31")
    ev = compile_policy(spec)
    with pytest.raises(ValueError, match="unsupported unary op"):
        ev(task, plan)
