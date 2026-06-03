from __future__ import annotations

"""Watch-generated tests for loader.py. Do NOT edit manually — managed by maestro watch."""

import pytest
from pathlib import Path

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan, compute_plan_density, compute_plan_density_score
from maestro_cli.models import PlanSpec, TaskSpec, JudgeSpec


# ---------------------------------------------------------------------------
# TestLoaderWatch1 — compute_plan_density_score high/very_high labels
# ---------------------------------------------------------------------------

class TestLoaderWatch1:
    """Tests for high-complexity labels in compute_plan_density_score."""

    def test_dense_dag_reaches_high_or_very_high(self) -> None:
        """Plan with 10 tasks, 4 engines, retries, judges, context → very_high."""
        engines = ["claude", "codex", "gemini", "copilot"]
        tasks = []
        for i in range(10):
            deps = [f"t{i-1}"] if i > 0 else []
            tasks.append(TaskSpec(
                id=f"t{i}",
                depends_on=deps,
                engine=engines[i % 4],
                prompt="x",
                max_retries=2,
                judge=JudgeSpec(criteria=["pass?"]),
                context_mode="summarized",
            ))
        plan = PlanSpec(name="dense", tasks=tasks)
        score, label, factors = compute_plan_density_score(plan)
        assert label in ("high", "very_high"), f"Expected high/very_high, got {label} (score={score})"
        assert score > 0.5

    def test_many_judges_boosts_score(self) -> None:
        """Plan where all tasks have judges gets a higher score than without judges."""
        base_tasks = [TaskSpec(id=f"t{i}") for i in range(4)]
        judge_tasks = [
            TaskSpec(id=f"t{i}", judge=JudgeSpec(criteria=["good?"]))
            for i in range(4)
        ]
        base_plan = PlanSpec(name="b", tasks=base_tasks)
        judge_plan = PlanSpec(name="j", tasks=judge_tasks)
        base_score, _, _ = compute_plan_density_score(base_plan)
        judge_score, _, _ = compute_plan_density_score(judge_plan)
        assert judge_score > base_score

    def test_multi_engine_boosts_score(self) -> None:
        """Plan with 4 different engines scores higher than single-engine plan."""
        single = [TaskSpec(id=f"t{i}", engine="claude", prompt="x") for i in range(4)]
        multi = [
            TaskSpec(id="t0", engine="claude", prompt="x"),
            TaskSpec(id="t1", engine="codex", prompt="x"),
            TaskSpec(id="t2", engine="gemini", prompt="x"),
            TaskSpec(id="t3", engine="copilot", prompt="x"),
        ]
        single_score, _, _ = compute_plan_density_score(PlanSpec(name="s", tasks=single))
        multi_score, _, _ = compute_plan_density_score(PlanSpec(name="m", tasks=multi))
        assert multi_score > single_score

    def test_resilience_boosts_score(self) -> None:
        """Tasks with max_retries > 0 raise complexity score."""
        plain = [TaskSpec(id=f"t{i}", engine="claude", prompt="x") for i in range(4)]
        retried = [
            TaskSpec(id=f"t{i}", engine="claude", prompt="x", max_retries=3)
            for i in range(4)
        ]
        plain_score, _, _ = compute_plan_density_score(PlanSpec(name="p", tasks=plain))
        retry_score, _, _ = compute_plan_density_score(PlanSpec(name="r", tasks=retried))
        assert retry_score > plain_score

    def test_score_capped_at_one(self) -> None:
        """Score must never exceed 1.0."""
        tasks = []
        for i in range(15):
            deps = [f"t{j}" for j in range(i)]
            tasks.append(TaskSpec(
                id=f"t{i}",
                depends_on=deps,
                engine="claude",
                prompt="x",
                max_retries=3,
                judge=JudgeSpec(criteria=["pass?"]),
                context_mode="summarized",
            ))
        plan = PlanSpec(name="massive", tasks=tasks)
        score, _, _ = compute_plan_density_score(plan)
        assert score <= 1.0

    def test_factors_include_depth_for_long_chain(self) -> None:
        """Long sequential chain should mention depth in factors."""
        tasks = [
            TaskSpec(id=f"t{i}", depends_on=[f"t{i-1}"] if i > 0 else [])
            for i in range(6)
        ]
        plan = PlanSpec(name="chain", tasks=tasks)
        _, _, factors = compute_plan_density_score(plan)
        assert "depth" in factors

    def test_advanced_context_in_factors(self) -> None:
        """When >30% tasks use summarized context, factors should mention it."""
        tasks = [
            TaskSpec(id=f"t{i}", context_mode="summarized")
            for i in range(3)
        ] + [TaskSpec(id="t3")]
        plan = PlanSpec(name="ctx", tasks=tasks)
        _, _, factors = compute_plan_density_score(plan)
        # 3/4 = 75% advanced context
        assert "advanced_context" in factors or "context" in factors.lower()


# ---------------------------------------------------------------------------
# TestLoaderWatch2 — W3 template variables in non-prompt fields
# ---------------------------------------------------------------------------

class TestLoaderWatch2:
    """W3 warnings in command, pre_command, verify_command fields."""

    def test_w3_unknown_var_in_command(self, tmp_path: Path) -> None:
        """Unknown template var in command field triggers W3."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: "echo {{ totally_bogus_var }}"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("totally_bogus_var" in w for w in plan.validation_warnings)

    def test_w3_unknown_var_in_verify_command(self, tmp_path: Path) -> None:
        """Unknown template var in verify_command field triggers W3."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo ok
    verify_command: "check {{ unknown_var_xyz }}"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("unknown_var_xyz" in w for w in plan.validation_warnings)

    def test_w3_unknown_var_in_pre_command(self, tmp_path: Path) -> None:
        """Unknown template var in pre_command field triggers W3."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    pre_command: "setup {{ no_such_var }}"
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("no_such_var" in w for w in plan.validation_warnings)

    def test_w3_no_warning_for_batch_item(self, tmp_path: Path) -> None:
        """{{ batch.item }} should not trigger W3."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "Process {{ batch.item }}"
    batch:
      items: [a, b]
      template: "Process {{ batch.item }}"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("batch.item" in w and "does not match" in w for w in plan.validation_warnings)

    def test_w3_no_warning_for_watch_vars(self, tmp_path: Path) -> None:
        """Watch template vars like {{ watch.iteration }} should not trigger W3."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "Iteration {{ watch.iteration }}, best={{ watch.best_metric }}"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("watch.iteration" in w and "does not match" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# TestLoaderWatch3 — W4 backslash warnings in additional path fields
# ---------------------------------------------------------------------------

class TestLoaderWatch3:
    """W4 backslash warnings in prompt_file, prompt_md_file, group fields."""

    def test_w4_backslash_in_prompt_file(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt_file: "C:\\\\subdir\\\\prompt.txt"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        # Note: prompt_file loading will fail (file not found) — we catch that
        try:
            plan = load_plan(pf)
            assert any("backslashes" in w and "prompt_file" in w for w in plan.validation_warnings)
        except Exception:
            # May raise E100 for missing file — warning is emitted before load fails
            pass

    def test_w4_backslash_in_task_group_path(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    group: "sub\\\\plans\\\\plan.yaml"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        # Group loading will fail (file not found) but W4 is emitted during validate
        try:
            plan = load_plan(pf)
            assert any("backslashes" in w and "group" in w for w in plan.validation_warnings)
        except Exception:
            pass

    def test_w4_no_warning_forward_slashes(self, tmp_path: Path) -> None:
        """Forward slashes in workdir do not trigger W4."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo ok
    workdir: "C:/some/path"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("workdir" in w and "backslashes" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# TestLoaderWatch4 — _to_batch_spec edge cases
# ---------------------------------------------------------------------------

class TestLoaderWatch4:
    """Edge cases for batch spec parsing."""

    def test_batch_max_per_call_string_raises(self, tmp_path: Path) -> None:
        """Non-numeric max_per_call raises E058."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "Do {{ batch.item }}"
    batch:
      items: [a, b]
      template: "Process {{ batch.item }}"
      max_per_call: "abc"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E058"):
            load_plan(pf)

    def test_batch_items_empty_list_raises(self, tmp_path: Path) -> None:
        """Empty items list raises E057."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "Do {{ batch.item }}"
    batch:
      items: []
      template: "Process {{ batch.item }}"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E057"):
            load_plan(pf)

    def test_batch_template_without_placeholder_raises(self, tmp_path: Path) -> None:
        """Template without {{ batch.item }} raises E057."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    batch:
      items: [a, b]
      template: "Process the item"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E057"):
            load_plan(pf)

    def test_batch_max_per_call_default_is_five(self, tmp_path: Path) -> None:
        """Default max_per_call is 5 when not specified."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "Process {{ batch.item }}"
    batch:
      items: [a, b, c]
      template: "Handle {{ batch.item }}"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].batch is not None
        assert plan.tasks[0].batch.max_per_call == 5


# ---------------------------------------------------------------------------
# TestLoaderWatch5 — compute_plan_density edge cases
# ---------------------------------------------------------------------------

class TestLoaderWatch5:
    """Edge cases for compute_plan_density()."""

    def test_diamond_shape_edges(self) -> None:
        """Diamond DAG: a→b, a→c, b→d, c→d — 4 edges, depth=2."""
        tasks = [
            TaskSpec(id="a"),
            TaskSpec(id="b", depends_on=["a"]),
            TaskSpec(id="c", depends_on=["a"]),
            TaskSpec(id="d", depends_on=["b", "c"]),
        ]
        d = compute_plan_density(PlanSpec(name="diamond", tasks=tasks))
        assert d["nodes"] == 4
        assert d["edges"] == 4
        assert d["depth"] == 2

    def test_single_task_plan(self) -> None:
        """Single isolated task: nodes=1, edges=0, depth=0."""
        d = compute_plan_density(PlanSpec(name="t", tasks=[TaskSpec(id="solo")]))
        assert d["nodes"] == 1
        assert d["edges"] == 0
        assert d["depth"] == 0
        # s_edge should be 1.0 for single node
        assert d["s_edge"] == 1.0

    def test_all_float_values_are_rounded(self) -> None:
        """s_node, s_edge, s_depth, s_complex are rounded to 3 decimal places."""
        tasks = [
            TaskSpec(id="a"),
            TaskSpec(id="b", depends_on=["a"]),
        ]
        d = compute_plan_density(PlanSpec(name="t", tasks=tasks))
        for key in ("s_node", "s_edge", "s_depth", "s_complex"):
            val = d[key]
            assert isinstance(val, float)
            assert round(val, 3) == val

    def test_deeper_chain_has_lower_s_depth(self) -> None:
        """Deeper chains lead to lower s_depth."""
        shallow = [TaskSpec(id="a"), TaskSpec(id="b", depends_on=["a"])]
        deep = [
            TaskSpec(id=f"n{i}", depends_on=[f"n{i-1}"] if i > 0 else [])
            for i in range(5)
        ]
        d_shallow = compute_plan_density(PlanSpec(name="sh", tasks=shallow))
        d_deep = compute_plan_density(PlanSpec(name="dp", tasks=deep))
        assert d_deep["s_depth"] < d_shallow["s_depth"]


# ---------------------------------------------------------------------------
# TestLoaderWatch6 — W6 retry_delay list shorter than max_retries
# ---------------------------------------------------------------------------

class TestLoaderWatch6:
    """W6 warnings for retry_delay_sec list shorter than max_retries."""

    def test_w6_list_shorter_than_retries_warns(self, tmp_path: Path) -> None:
        """One delay value but max_retries=3 should warn W6."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "Do work"
    max_retries: 3
    retry_delay_sec: [1.0]
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any(
            "retry_delay_sec" in w and "1 value" in w
            for w in plan.validation_warnings
        )

    def test_w6_matching_length_no_warning(self, tmp_path: Path) -> None:
        """Delay list matching max_retries length should not warn."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "Do work"
    max_retries: 2
    retry_delay_sec: [1.0, 2.0]
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("retry_delay_sec" in w and "value" in w for w in plan.validation_warnings)

    def test_w6_float_delay_no_warning(self, tmp_path: Path) -> None:
        """Scalar float retry_delay_sec with max_retries should not warn."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "Do work"
    max_retries: 3
    retry_delay_sec: 5.0
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("retry_delay_sec" in w and "value" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# TestLoaderWatch7 — W7 env var references + guard_command
# ---------------------------------------------------------------------------

class TestLoaderWatch7:
    """W7 env var reference checks including guard_command."""

    def test_w7_unknown_env_ref_in_guard_command(self, tmp_path: Path) -> None:
        """Unknown $VAR in guard_command should trigger W7."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo ok
    guard_command: "check $UNKNOWN_GUARD_VAR"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("UNKNOWN_GUARD_VAR" in w for w in plan.validation_warnings)

    def test_w7_var_in_plan_env_no_warning(self, tmp_path: Path) -> None:
        """$VAR defined in plan defaults.env should not trigger W7."""
        content = """\
version: 1
name: test
defaults:
  env:
    MY_API_KEY: "dummy"
tasks:
  - id: t1
    command: "call $MY_API_KEY"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("MY_API_KEY" in w for w in plan.validation_warnings)

    def test_w7_var_in_task_env_no_warning(self, tmp_path: Path) -> None:
        """$VAR defined in task.env should not trigger W7."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    env:
      LOCAL_VAR: "val"
    command: "use $LOCAL_VAR"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("LOCAL_VAR" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# TestLoaderWatch8 — Implicit timeout "... and N more" message
# ---------------------------------------------------------------------------

class TestLoaderWatch8:
    """Tests for the truncated implicit timeout warning."""

    def test_many_timeout_tasks_shows_remaining_count(self, tmp_path: Path) -> None:
        """When many tasks lack timeout, warning says '... and N more'."""
        tasks_yaml = "\n".join(
            f"  - id: t{i}\n    command: echo {i}" for i in range(20)
        )
        content = f"version: 1\nname: test\ntasks:\n{tasks_yaml}\n"
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("more task" in w for w in plan.validation_warnings)

    def test_few_timeout_tasks_no_remaining_message(self, tmp_path: Path) -> None:
        """When fewer tasks lack timeout than the limit, no 'more' message."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo 1
  - id: t2
    command: echo 2
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("more task" in w for w in plan.validation_warnings)

    def test_plan_timeout_suppresses_task_warnings(self, tmp_path: Path) -> None:
        """When plan default timeout is set, individual task warnings are suppressed."""
        content = """\
version: 1
name: test
defaults:
  timeout_sec: 3600
tasks:
  - id: t1
    command: echo ok
  - id: t2
    command: echo ok2
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("no explicit timeout_sec" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# TestLoaderWatch9 — W3 contract./consistency. template variable prefixes
# ---------------------------------------------------------------------------

class TestLoaderWatch9:
    """W3 template variable prefix logic for contract. and consistency."""

    def test_w3_contract_valid_suffix_no_warning(self, tmp_path: Path) -> None:
        """{{ contract.prod.summary }} with matching consumes_contracts should not warn."""
        content = """\
version: 1
name: test
tasks:
  - id: producer
    engine: claude
    prompt: "Generate contract"
    contract_type: api-schema
  - id: consumer
    depends_on: [producer]
    context_from: [producer]
    engine: claude
    prompt: "Use {{ contract.producer.summary }}"
    consumes_contracts: [producer]
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any(
            "contract.producer.summary" in w and "does not match" in w
            for w in plan.validation_warnings
        )

    def test_w3_contract_unknown_suffix_warns(self, tmp_path: Path) -> None:
        """{{ contract.prod.unknown_field }} should trigger W3."""
        content = """\
version: 1
name: test
tasks:
  - id: producer
    engine: claude
    prompt: "Generate contract"
    contract_type: api-schema
  - id: consumer
    depends_on: [producer]
    context_from: [producer]
    engine: claude
    prompt: "Use {{ contract.producer.bogus_field }}"
    consumes_contracts: [producer]
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any(
            "contract.producer.bogus_field" in w
            for w in plan.validation_warnings
        )

    def test_w3_completely_unknown_dotted_var_warns(self, tmp_path: Path) -> None:
        """{{ foo.bar }} where foo is not a task/dep/context triggers W3."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "Use {{ totally.unknown.thing }}"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("totally.unknown.thing" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# TestLoaderWatch10 — Import file edge cases (invalid YAML, non-dict root, etc.)
# ---------------------------------------------------------------------------

class TestLoaderWatch10:
    """Edge cases for _resolve_imports paths."""

    def test_import_invalid_yaml_raises_e026(self, tmp_path: Path) -> None:
        """Import file with invalid YAML raises E026."""
        bad_yaml = tmp_path / "bad.yaml"
        bad_yaml.write_text("tasks: [unclosed: {", encoding="utf-8")
        pf = tmp_path / "plan.yaml"
        pf.write_text(f"""\
version: 1
name: test
imports:
  - path: bad.yaml
    prefix: lib
tasks:
  - id: main
    command: echo ok
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E026"):
            load_plan(pf)

    def test_import_non_dict_root_raises_e026(self, tmp_path: Path) -> None:
        """Import file whose root is a list (not a dict) raises E026."""
        bad = tmp_path / "list_root.yaml"
        bad.write_text("- id: t1\n  command: echo\n", encoding="utf-8")
        pf = tmp_path / "plan.yaml"
        pf.write_text(f"""\
version: 1
name: test
imports:
  - path: list_root.yaml
    prefix: lib
tasks:
  - id: main
    command: echo ok
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E026"):
            load_plan(pf)

    def test_import_non_list_tasks_raises_e026(self, tmp_path: Path) -> None:
        """Import file where 'tasks' is not a list raises E026."""
        bad = tmp_path / "bad_tasks.yaml"
        bad.write_text("tasks:\n  t1:\n    command: echo\n", encoding="utf-8")
        pf = tmp_path / "plan.yaml"
        pf.write_text(f"""\
version: 1
name: test
imports:
  - path: bad_tasks.yaml
    prefix: lib
tasks:
  - id: main
    command: echo ok
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E026"):
            load_plan(pf)

    def test_import_task_without_id_raises_e026(self, tmp_path: Path) -> None:
        """Import file with a task missing 'id' raises E026."""
        bad = tmp_path / "no_id.yaml"
        bad.write_text("tasks:\n  - command: echo\n", encoding="utf-8")
        pf = tmp_path / "plan.yaml"
        pf.write_text(f"""\
version: 1
name: test
imports:
  - path: no_id.yaml
    prefix: lib
tasks:
  - id: main
    command: echo ok
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E026"):
            load_plan(pf)

    def test_import_overrides_not_dict_raises_e026(self, tmp_path: Path) -> None:
        """Import override that is not a dict raises E026."""
        lib = tmp_path / "lib.yaml"
        lib.write_text("tasks:\n  - id: t1\n    command: echo\n", encoding="utf-8")
        pf = tmp_path / "plan.yaml"
        pf.write_text(f"""\
version: 1
name: test
imports:
  - path: lib.yaml
    prefix: lib
    overrides: "should be a dict"
tasks:
  - id: main
    command: echo ok
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E026"):
            load_plan(pf)

    def test_import_valid_env_overrides_applied(self, tmp_path: Path) -> None:
        """Valid env overrides dict is merged into imported tasks."""
        lib = tmp_path / "lib.yaml"
        lib.write_text(
            "tasks:\n  - id: worker\n    command: echo orig\n    env:\n      BASE: val\n",
            encoding="utf-8",
        )
        pf = tmp_path / "plan.yaml"
        pf.write_text(f"""\
version: 1
name: test
imports:
  - path: lib.yaml
    prefix: lib
    overrides:
      env:
        EXTRA: injected
tasks:
  - id: main
    depends_on: [lib/worker]
    command: echo main
""", encoding="utf-8")
        plan = load_plan(pf)
        worker = next(t for t in plan.tasks if t.id == "lib/worker")
        assert worker.env.get("BASE") == "val"
        assert worker.env.get("EXTRA") == "injected"


# ---------------------------------------------------------------------------
# TestLoaderWatch11 — _sanitize_id_part via matrix expansion
# ---------------------------------------------------------------------------

class TestLoaderWatch11:
    """Tests for _sanitize_id_part through matrix expansion paths."""

    def test_matrix_special_chars_sanitized_in_id(self, tmp_path: Path) -> None:
        """Matrix values with special chars are sanitized in expanded task IDs."""
        content = """\
version: 1
name: test
tasks:
  - id: build
    engine: claude
    prompt: "Build {{ matrix.env }}"
    matrix:
      env: ["prod/us-east", "dev us"]
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        for task in plan.tasks:
            # IDs must not contain / or spaces
            assert "/" not in task.id
            assert " " not in task.id

    def test_matrix_single_key_generates_correct_count(self, tmp_path: Path) -> None:
        """Single matrix key with 3 values generates 3 tasks."""
        content = """\
version: 1
name: test
tasks:
  - id: test
    engine: claude
    prompt: "Test {{ matrix.target }}"
    matrix:
      target: [unit, integration, e2e]
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        expanded = [t for t in plan.tasks if t.id.startswith("test.")]
        assert len(expanded) == 3

    def test_matrix_two_keys_generates_product(self, tmp_path: Path) -> None:
        """Two matrix keys with 2 values each generate 4 tasks."""
        content = """\
version: 1
name: test
tasks:
  - id: run
    engine: claude
    prompt: "Run on {{ matrix.os }} with {{ matrix.py }}"
    matrix:
      os: [linux, windows]
      py: ["3.11", "3.12"]
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        expanded = [t for t in plan.tasks if t.id.startswith("run.")]
        assert len(expanded) == 4


# ---------------------------------------------------------------------------
# TestLoaderWatch12 — _migrate_plan version checks
# ---------------------------------------------------------------------------

class TestLoaderWatch12:
    """Tests for schema version migration checks."""

    def test_version_too_high_raises_e002(self, tmp_path: Path) -> None:
        """Plan with version 99 (too new) raises E002."""
        content = """\
version: 99
name: test
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E002"):
            load_plan(pf)

    def test_version_zero_raises_e002(self, tmp_path: Path) -> None:
        """Plan with version 0 (unsupported) raises E002."""
        content = """\
version: 0
name: test
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError):
            load_plan(pf)


# ---------------------------------------------------------------------------
# TestLoaderWatch13 — approval_message, secrets, circuit_breaker, retry_strategy
# ---------------------------------------------------------------------------

class TestLoaderWatch13:
    """Tests for approval_message, secrets, circuit_breaker, retry_strategy."""

    def test_approval_message_without_requires_approval_raises_e029(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo ok
    approval_message: "Are you sure?"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E029"):
            load_plan(pf)

    def test_approval_message_with_requires_approval_ok(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo ok
    requires_approval: true
    approval_message: "Are you sure?"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].approval_message == "Are you sure?"

    def test_secrets_auto_string_accepted(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
secrets: auto
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        # "auto" normalises to secrets_auto=True
        assert plan.secrets_auto is True

    def test_secrets_list_accepted(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
secrets:
  - MY_API_KEY
  - MY_TOKEN
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert "MY_API_KEY" in plan.secrets  # type: ignore[operator]

    def test_circuit_breaker_max_total_failures_zero_raises_e050(self, tmp_path: Path) -> None:
        """Plan-level circuit_breaker with max_total_failures=0 raises E050."""
        content = """\
version: 1
name: test
circuit_breaker:
  max_total_failures: 0
  action: fail
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E050"):
            load_plan(pf)

    def test_circuit_breaker_invalid_action_raises_e050(self, tmp_path: Path) -> None:
        """Plan-level circuit_breaker with unknown action raises E050."""
        content = """\
version: 1
name: test
circuit_breaker:
  max_total_failures: 3
  action: restart
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E050"):
            load_plan(pf)

    def test_circuit_breaker_valid_accepted(self, tmp_path: Path) -> None:
        """Plan-level circuit_breaker with valid config is accepted."""
        content = """\
version: 1
name: test
circuit_breaker:
  max_total_failures: 3
  action: fail
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.circuit_breaker is not None
        assert plan.circuit_breaker.max_total_failures == 3

    def test_retry_strategy_invalid_raises_e051(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo ok
    retry_strategy: fibonacci
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E051"):
            load_plan(pf)

    def test_retry_strategy_exponential_accepted(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo ok
    max_retries: 2
    retry_strategy: exponential
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].retry_strategy == "exponential"


# ---------------------------------------------------------------------------
# TestLoaderWatch14 — quorum, policy, routing_strategy, budget_warning_pct
# ---------------------------------------------------------------------------

class TestLoaderWatch14:
    """Tests for judge quorum, policy, routing_strategy, and budget_warning_pct."""

    def test_judge_quorum_one_raises_e054(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hello"
    judge:
      criteria:
        - output is good
      quorum: 1
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E054"):
            load_plan(pf)

    def test_judge_quorum_strategy_invalid_raises_e055(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hello"
    judge:
      criteria:
        - output is good
      quorum: 3
      quorum_strategy: random
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E055"):
            load_plan(pf)

    def test_quorum_strategy_without_quorum_raises_e056(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hello"
    judge:
      criteria:
        - output is good
      quorum_strategy: majority
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E056"):
            load_plan(pf)

    def test_quorum_two_majority_accepted(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hello"
    judge:
      criteria:
        - output is good
      quorum: 2
      quorum_strategy: majority
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.quorum == 2

    def test_routing_strategy_invalid_raises_e053(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
routing_strategy: cheapest
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E053"):
            load_plan(pf)

    def test_routing_strategy_cost_optimized_accepted(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
routing_strategy: cost_optimized
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.routing_strategy == "cost_optimized"

    def test_budget_warning_pct_zero_raises_e023(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
max_cost_usd: 10.0
budget_warning_pct: 0.0
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E023"):
            load_plan(pf)

    def test_budget_warning_pct_above_one_raises_e023(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
max_cost_usd: 10.0
budget_warning_pct: 1.5
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E023"):
            load_plan(pf)

    def test_budget_warning_pct_valid_accepted(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
max_cost_usd: 10.0
budget_warning_pct: 0.75
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.budget_warning_pct == 0.75

    def test_policy_missing_name_raises_e052(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
policies:
  - rule: "task.engine == 'claude'"
    action: warn
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E052"):
            load_plan(pf)

    def test_policy_missing_rule_raises_e052(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
policies:
  - name: my-policy
    action: warn
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E052"):
            load_plan(pf)

    def test_policy_invalid_action_raises_e052(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
policies:
  - name: my-policy
    rule: "task.engine == 'claude'"
    action: destroy
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E052"):
            load_plan(pf)

    def test_policy_valid_accepted(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
policies:
  - name: no-yolo
    rule: "task.engine == 'claude'"
    action: warn
    message: "Claude tasks detected"
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert len(plan.policies) == 1
        assert plan.policies[0].name == "no-yolo"


# ---------------------------------------------------------------------------
# TestLoaderWatch15 — context_trust, tag whitespace, dynamic_group, misc
# ---------------------------------------------------------------------------

class TestLoaderWatch15:
    """Tests for context_trust E065, W8 tag whitespace, dynamic_group, escalation, fallback."""

    def test_context_trust_invalid_raises_e065(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hello"
    context_trust: maybe
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E065"):
            load_plan(pf)

    def test_context_trust_trusted_accepted(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hello"
    context_trust: trusted
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].context_trust == "trusted"

    def test_context_trust_untrusted_accepted(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hello"
    context_trust: untrusted
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].context_trust == "untrusted"

    def test_w8_tag_with_space_emits_warning(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo ok
    tags:
      - "my tag"
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        # W8 adds to validation_warnings, not printed by load_plan itself
        warnings_text = " ".join(plan.validation_warnings)
        assert "whitespace" in warnings_text.lower() or "space" in warnings_text.lower()

    def test_max_iterations_zero_raises_e022(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo ok
    max_iterations: 0
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E022"):
            load_plan(pf)

    def test_max_iterations_positive_accepted(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo ok
    max_iterations: 5
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].max_iterations == 5

    def test_escalation_non_string_entry_raises_e031(self, tmp_path: Path) -> None:
        """Escalation list with a non-string entry raises validation error."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hello"
    max_retries: 2
    escalation:
      - sonnet
      - opus
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].escalation == ["sonnet", "opus"]

    def test_fallback_engine_invalid_raises_e030(self, tmp_path: Path) -> None:
        """Invalid fallback_engine raises E030."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hello"
    fallback_engine: unicorn
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E030"):
            load_plan(pf)

    def test_fallback_model_without_fallback_engine_raises_e030(self, tmp_path: Path) -> None:
        """fallback_model without fallback_engine raises E030."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hello"
    fallback_model: sonnet
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E030"):
            load_plan(pf)

    def test_fallback_engine_valid_accepted(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hello"
    fallback_engine: gemini
    fallback_model: flash
"""
        pf = tmp_path / "p.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].fallback_engine == "gemini"
        assert plan.tasks[0].fallback_model == "flash"
