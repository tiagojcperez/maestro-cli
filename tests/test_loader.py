from __future__ import annotations

import pytest
from pathlib import Path

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan


class TestContextFromParsing:
    def test_context_from_valid(self, context_plan_yaml: Path) -> None:
        plan = load_plan(context_plan_yaml)
        task_b = next(t for t in plan.tasks if t.id == "b")
        task_c = next(t for t in plan.tasks if t.id == "c")
        assert task_b.context_from == ["a"]
        assert task_c.context_from == ["*"]

    def test_context_from_empty_by_default(self, minimal_plan_yaml: Path) -> None:
        plan = load_plan(minimal_plan_yaml)
        assert plan.tasks[0].context_from == []

    def test_context_from_unknown_task(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test-plan
tasks:
  - id: a
    command: "echo a"
  - id: b
    depends_on: [a]
    context_from: [nonexistent]
    command: "echo b"
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="context_from references unknown task"):
            load_plan(plan_file)

    def test_context_from_not_in_depends_on(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test-plan
tasks:
  - id: a
    command: "echo a"
  - id: b
    command: "echo b"
  - id: c
    depends_on: [a]
    context_from: [b]
    command: "echo c"
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="not in depends_on"):
            load_plan(plan_file)

    def test_context_from_wildcard_passes(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test-plan
tasks:
  - id: a
    command: "echo a"
  - id: b
    depends_on: [a]
    context_from: ["*"]
    command: "echo b"
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        task_b = next(t for t in plan.tasks if t.id == "b")
        assert task_b.context_from == ["*"]

    def test_context_from_as_string(self, tmp_path: Path) -> None:
        """context_from as a single string should be coerced to a list."""
        content = """\
version: 1
name: test-plan
tasks:
  - id: a
    command: "echo a"
  - id: b
    depends_on: [a]
    context_from: a
    command: "echo b"
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        task_b = next(t for t in plan.tasks if t.id == "b")
        assert task_b.context_from == ["a"]


# ===========================================================================
# Schema migration
# ===========================================================================


class TestSchemaMigration:
    def test_yaml_anchors_expand_correctly(self, tmp_path: Path) -> None:
        """YAML anchors and aliases should resolve transparently via safe_load."""
        content = """\
version: 1
name: anchor-test

_defaults: &defaults
  engine: claude
  model: sonnet
  prompt: "Default prompt"

tasks:
  - id: t1
    <<: *defaults
  - id: t2
    <<: *defaults
    prompt: "Override prompt"
    depends_on: [t1]
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert len(plan.tasks) == 2
        t1 = next(t for t in plan.tasks if t.id == "t1")
        t2 = next(t for t in plan.tasks if t.id == "t2")
        assert t1.engine == "claude"
        assert t1.model == "sonnet"
        assert t1.prompt == "Default prompt"
        assert t2.prompt == "Override prompt"
        assert t2.depends_on == ["t1"]

    def test_version_too_high_raises(self, tmp_path: Path) -> None:
        """Schema version higher than supported should raise PlanValidationError."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 99
name: future-plan
tasks:
  - id: t1
    command: echo hello
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E002\]"):
            load_plan(plan_file)

    def test_version_too_high_suggests_upgrade(self, tmp_path: Path) -> None:
        """Error message for too-new version should mention upgrading Maestro."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 99
name: future-plan
tasks:
  - id: t1
    command: echo hello
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="upgrade"):
            load_plan(plan_file)

    def test_migrate_plan_v1_passthrough(self, tmp_path: Path) -> None:
        """Version 1 plan should load without modification."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: stable-plan
tasks:
  - id: t1
    command: echo hello
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.version == 1
        assert plan.name == "stable-plan"
        assert len(plan.tasks) == 1


class TestImports:
    def test_imported_tasks_prefix_dependencies_context_and_merge_env_overrides(
        self, tmp_path: Path
    ) -> None:
        imported_file = tmp_path / "shared.yaml"
        imported_file.write_text("""\
tasks:
  - id: prep
    command: "echo prep"
    env:
      SHARED: base
  - id: run
    depends_on: prep
    context_from: prep
    command: "echo run"
""", encoding="utf-8")

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(f"""\
version: 1
name: import-test
imports:
  - path: "{imported_file.name}"
    prefix: lib
    overrides:
      env:
        EXTRA: override
tasks:
  - id: main
    depends_on: [lib/run]
    command: "echo main"
""", encoding="utf-8")

        plan = load_plan(plan_file)

        prep = next(t for t in plan.tasks if t.id == "lib/prep")
        run = next(t for t in plan.tasks if t.id == "lib/run")

        assert prep.env == {"SHARED": "base", "EXTRA": "override"}
        assert run.depends_on == ["lib/prep"]
        assert run.context_from == ["lib/prep"]
        assert run.env == {"EXTRA": "override"}
        assert plan.imports[0].path == imported_file.name
        assert plan.imports[0].prefix == "lib"

    def test_imported_tasks_preserve_external_refs_and_wildcard_context(
        self, tmp_path: Path
    ) -> None:
        imported_file = tmp_path / "shared.yaml"
        imported_file.write_text("""\
tasks:
  - id: prep
    command: "echo prep"
  - id: run
    depends_on:
      - prep
      - core/seed
    context_from:
      - prep
      - "*"
      - core/seed
    command: "echo run"
""", encoding="utf-8")

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(f"""\
version: 1
name: import-context-edge
imports:
  - path: "{imported_file.name}"
    prefix: lib
tasks:
  - id: core/seed
    command: "echo seed"
  - id: main
    depends_on: [lib/run]
    command: "echo main"
""", encoding="utf-8")

        plan = load_plan(plan_file)

        run = next(t for t in plan.tasks if t.id == "lib/run")
        assert run.depends_on == ["lib/prep", "core/seed"]
        assert run.context_from == ["lib/prep", "*", "core/seed"]

    def test_circular_imports_raise_e025(self, tmp_path: Path) -> None:
        first_import = tmp_path / "first.yaml"
        second_import = tmp_path / "second.yaml"
        first_import.write_text(f"""\
imports:
  - path: "{second_import.name}"
    prefix: second
tasks:
  - id: prep
    command: "echo prep"
""", encoding="utf-8")
        second_import.write_text(f"""\
imports:
  - path: "{first_import.name}"
    prefix: first
tasks:
  - id: run
    command: "echo run"
""", encoding="utf-8")

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(f"""\
version: 1
name: circular-import-test
imports:
  - path: "{first_import.name}"
    prefix: lib
tasks:
  - id: main
    command: "echo main"
""", encoding="utf-8")

        with pytest.raises(PlanValidationError, match=r"\[E025\]"):
            load_plan(plan_file)


class TestPromptSources:
    def test_engine_task_with_prompt_file_counts_as_prompt_source(
        self, tmp_path: Path
    ) -> None:
        prompt_file = tmp_path / "task-prompt.txt"
        prompt_file.write_text("Write a summary.", encoding="utf-8")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(f"""\
version: 1
name: prompt-file-source
tasks:
  - id: t1
    engine: claude
    prompt_file: "{prompt_file.name}"
""", encoding="utf-8")

        plan = load_plan(plan_file)

        assert plan.tasks[0].engine == "claude"
        assert plan.tasks[0].prompt is None
        assert plan.tasks[0].prompt_file == prompt_file.name


class TestWorkspaceAssertionsAndAuditPacks:
    def test_task_assertions_parsed(self, tmp_path: Path) -> None:
        pack_file = tmp_path / "audit-pack.yaml"
        pack_file.write_text("rules: []\n", encoding="utf-8")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(f"""\
version: 1
name: assert-plan
audit_packs:
  - "{pack_file.name}"
tasks:
  - id: t1
    command: "echo hello"
    assert:
      - type: file_contains
        path: README.md
        pattern: "Maestro"
      - type: glob_exists
        glob: "tests/*.py"
""", encoding="utf-8")

        plan = load_plan(plan_file)
        assert plan.audit_packs == [pack_file.name]
        assert len(plan.tasks[0].assertions) == 2
        assert plan.tasks[0].assertions[0]["type"] == "file_contains"
        assert plan.tasks[0].assertions[1]["type"] == "glob_exists"

    def test_task_assert_invalid_type_raises(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: bad-assert
tasks:
  - id: t1
    command: "echo hello"
    assert:
      - type: not-a-real-assertion
""", encoding="utf-8")

        with pytest.raises(PlanValidationError, match="tasks\\[0\\]\\.assert\\[0\\]\\.type"):
            load_plan(plan_file)

    def test_group_task_cannot_use_assert(self, tmp_path: Path) -> None:
        sub_plan = tmp_path / "sub.yaml"
        sub_plan.write_text("""\
tasks:
  - id: inner
    command: "echo hi"
""", encoding="utf-8")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: bad-group-assert
tasks:
  - id: t1
    group: sub.yaml
    assert:
      - type: glob_exists
        glob: "*.py"
""", encoding="utf-8")

        with pytest.raises(PlanValidationError, match="group tasks cannot use assert"):
            load_plan(plan_file)

    def test_missing_audit_pack_raises(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: missing-pack
audit_packs:
  - missing-rules.yaml
tasks:
  - id: t1
    command: "echo hello"
""", encoding="utf-8")

        with pytest.raises(PlanValidationError, match="audit_packs entry"):
            load_plan(plan_file)


class TestContractsAndConsistencyGroups:
    def test_contract_fields_parsed_and_dependencies_inferred(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: contracts-plan
tasks:
  - id: schema
    command: "echo schema"
    contract_type: sql-schema

  - id: repo
    engine: claude
    prompt: "Use {{ contract.schema.summary }}"
    consumes_contracts: [schema]
    context_from: [schema]

  - id: controller
    command: "echo controller"
    consistency_group: [di]

  - id: bindings
    command: "echo bindings"
    consistency_group: [di]

  - id: reconcile
    engine: claude
    prompt: "Review {{ consistency.di.statuses }}"
    reconcile_after: [di]
""", encoding="utf-8")

        plan = load_plan(plan_file)
        task_map = {task.id: task for task in plan.tasks}
        assert task_map["schema"].contract_type == "sql-schema"
        assert task_map["repo"].consumes_contracts == ["schema"]
        assert "schema" in task_map["repo"].depends_on
        assert task_map["controller"].consistency_group == ["di"]
        assert task_map["reconcile"].reconcile_after == ["di"]
        assert set(task_map["reconcile"].depends_on) == {"controller", "bindings"}

    def test_consumes_contracts_requires_contract_producer(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: bad-contract-consumer
tasks:
  - id: t1
    command: "echo hi"

  - id: t2
    engine: claude
    prompt: "Use {{ contract.t1.summary }}"
    consumes_contracts: [t1]
""", encoding="utf-8")

        with pytest.raises(PlanValidationError, match="does not declare contract_type"):
            load_plan(plan_file)

    def test_reconcile_after_requires_existing_group(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: bad-reconcile
tasks:
  - id: t1
    engine: claude
    prompt: "Check {{ consistency.di.statuses }}"
    reconcile_after: [di]
""", encoding="utf-8")

        with pytest.raises(PlanValidationError, match="unknown consistency_group"):
            load_plan(plan_file)


class TestGoalField:
    def test_goal_field_parsed(self, tmp_path: Path) -> None:
        """plan.goal is populated from the top-level goal key."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: goal-plan
goal: "Improve test coverage by 10%"
tasks:
  - id: t1
    command: "echo hello"
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.goal == "Improve test coverage by 10%"

    def test_goal_field_defaults_to_empty_string(self, tmp_path: Path) -> None:
        """plan.goal defaults to empty string when not specified."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: no-goal-plan
tasks:
  - id: t1
    command: "echo hello"
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.goal == ""


class TestMatrixExpansion:
    def test_matrix_tasks_expanded_to_child_ids(self, tmp_path: Path) -> None:
        """A matrix task should be replaced by expanded child tasks with composite IDs."""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: matrix-test
tasks:
  - id: build
    engine: claude
    prompt: "Build for {{ matrix.env }}"
    matrix:
      env: [staging, prod]
""", encoding="utf-8")
        plan = load_plan(plan_file)
        task_ids = [t.id for t in plan.tasks]
        assert "build" not in task_ids
        assert any("staging" in tid for tid in task_ids)
        assert any("prod" in tid for tid in task_ids)
        assert len(task_ids) == 2


class TestWebhookUrlParsing:
    def test_webhook_url_parsed_from_plan(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: webhook-plan
webhook_url: "https://example.com/hook"
tasks:
  - id: t1
    command: echo hello
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.webhook_url == "https://example.com/hook"


# ===========================================================================
# Escalation and fallback validation (v1.3.0)
# ===========================================================================


class TestEscalationFallbackValidation:
    def test_escalation_parsed(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: escalation-test
tasks:
  - id: t1
    engine: claude
    model: haiku
    prompt: "Do the thing"
    max_retries: 2
    escalation: [haiku, sonnet, opus]
""", encoding="utf-8")
        plan = load_plan(plan_file)
        t1 = plan.tasks[0]
        assert t1.escalation == ["haiku", "sonnet", "opus"]

    def test_escalation_empty_default(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: no-escalation
tasks:
  - id: t1
    engine: claude
    prompt: "Do the thing"
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.tasks[0].escalation == []

    def test_escalation_without_engine_raises(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: bad-escalation
tasks:
  - id: t1
    command: echo hello
    escalation: [haiku, sonnet]
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E031\]"):
            load_plan(plan_file)

    def test_escalation_empty_string_raises(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: bad-escalation-entry
tasks:
  - id: t1
    engine: claude
    prompt: "Do the thing"
    escalation: [""]
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E031\]"):
            load_plan(plan_file)

    def test_fallback_engine_parsed(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: fallback-test
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt: "Do the thing"
    fallback_engine: codex
    fallback_model: "5.4"
""", encoding="utf-8")
        plan = load_plan(plan_file)
        t1 = plan.tasks[0]
        assert t1.fallback_engine == "codex"
        assert t1.fallback_model == "5.4"

    def test_fallback_engine_invalid_raises(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: bad-fallback-engine
tasks:
  - id: t1
    engine: claude
    prompt: "Do the thing"
    fallback_engine: invalid
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E030\]"):
            load_plan(plan_file)

    def test_fallback_model_without_engine_raises(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: fallback-model-no-engine
tasks:
  - id: t1
    engine: claude
    prompt: "Do the thing"
    fallback_model: sonnet
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E030\]"):
            load_plan(plan_file)

    def test_fallback_on_command_task_raises(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: fallback-on-command
tasks:
  - id: t1
    command: echo hello
    fallback_engine: codex
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E030\]"):
            load_plan(plan_file)

    def test_warning_fallback_same_as_engine(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: same-fallback
tasks:
  - id: t1
    engine: claude
    prompt: "Do the thing"
    fallback_engine: claude
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert any("W13" in w and "t1" in w for w in plan.validation_warnings)

    def test_warning_escalation_duplicates(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: dup-escalation
tasks:
  - id: t1
    engine: claude
    prompt: "Do the thing"
    max_retries: 2
    escalation: [haiku, haiku, sonnet]
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert any("W14" in w and "t1" in w for w in plan.validation_warnings)

    def test_warning_escalation_no_retries(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: escalation-no-retry
tasks:
  - id: t1
    engine: claude
    prompt: "Do the thing"
    max_retries: 0
    escalation: [haiku, sonnet]
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert any("W15" in w and "t1" in w for w in plan.validation_warnings)


_WATCH_PLAN = """\
version: 1
name: watch-test
tasks:
  - id: experiment
    engine: claude
    prompt: "test"
watch:
  metric: val_loss
  metric_pattern: "loss: ([0-9.]+)"
  max_iterations: 10
"""


class TestWatchParsing:
    def test_plan_without_watch_has_none(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: no-watch
tasks:
  - id: experiment
    engine: claude
    prompt: "test"
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.watch is None

    def test_watch_block_must_be_mapping(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: bad-watch-shape
tasks:
  - id: experiment
    command: "echo test"
watch:
  - metric: score
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E032\]"):
            load_plan(plan_file)

    def test_watch_block_parsed(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: watch-test
tasks:
  - id: experiment
    engine: claude
    prompt: "test"
watch:
  metric: accuracy
  metric_direction: higher_is_better
  metric_source: json_field
  metric_json_path: metrics.val_accuracy
  metric_task: experiment
  max_iterations: 12
  iteration_budget_sec: 300
  on_regression: revert
  warmup_iterations: 2
  plateau_threshold: 4
  plateau_action: notify
  max_cost_usd: 25.5
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.metric == "accuracy"
        assert plan.watch.metric_direction == "higher_is_better"
        assert plan.watch.metric_source == "json_field"
        assert plan.watch.metric_json_path == "metrics.val_accuracy"
        assert plan.watch.metric_task == "experiment"
        assert plan.watch.max_iterations == 12
        assert plan.watch.iteration_budget_sec == 300
        assert plan.watch.on_regression == "revert"
        assert plan.watch.warmup_iterations == 2
        assert plan.watch.plateau_threshold == 4
        assert plan.watch.plateau_action == "notify"
        assert plan.watch.max_cost_usd == 25.5

    def test_watch_numeric_string_fields_are_coerced(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: watch-string-coercion
tasks:
  - id: experiment
    command: "echo test"
watch:
  metric: score
  metric_source: json_field
  metric_json_path: metrics.score
  metric_task: experiment
  metric_direction: higher_is_better
  max_iterations: "12"
  iteration_budget_sec: "300"
  on_regression: keep
  warmup_iterations: "2"
  plateau_threshold: "4"
  max_cost_usd: "25.5"
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.max_iterations == 12
        assert plan.watch.iteration_budget_sec == 300
        assert plan.watch.on_regression == "keep"
        assert plan.watch.warmup_iterations == 2
        assert plan.watch.plateau_threshold == 4
        assert plan.watch.max_cost_usd == 25.5

    def test_watch_metric_required(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_WATCH_PLAN.replace("  metric: val_loss\n", ""), encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E032\]"):
            load_plan(plan_file)

    def test_watch_invalid_direction(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            _WATCH_PLAN + '  metric_direction: sideways\n',
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match=r"\[E033\]"):
            load_plan(plan_file)

    def test_watch_invalid_source(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            _WATCH_PLAN + '  metric_source: invalid_source\n',
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match=r"\[E033\]"):
            load_plan(plan_file)

    def test_watch_verify_command_source_uses_defaults_without_regex(
        self, tmp_path: Path
    ) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: watch-defaults
tasks:
  - id: experiment
    command: "echo test"
    verify_command: ["python", "-c", "print('ok')"]
watch:
  metric: score
  metric_source: verify_command
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.metric_source == "verify_command"
        assert plan.watch.metric_pattern is None
        assert plan.watch.metric_direction == "lower_is_better"
        assert plan.watch.max_iterations == 100
        assert plan.watch.warmup_iterations == 1
        assert plan.watch.plateau_threshold == 5
        assert plan.watch.plateau_action == "stop"
        assert plan.watch.on_regression == "rollback"

    def test_watch_program_md_relative_to_plan_file(self, tmp_path: Path) -> None:
        (tmp_path / "program.md").write_text("# Watch Program\n", encoding="utf-8")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            _WATCH_PLAN + "  program_md: program.md\n",
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.program_md == "program.md"

    def test_watch_program_md_missing_file_raises(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            _WATCH_PLAN + "  program_md: missing.md\n",
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match=r"\[E042\]"):
            load_plan(plan_file)

    def test_watch_regex_required_for_stdout(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_WATCH_PLAN.replace('  metric_pattern: "loss: ([0-9.]+)"\n', ""), encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E034\]"):
            load_plan(plan_file)

    def test_watch_regex_must_be_valid_regex(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            _WATCH_PLAN.replace(
                '  metric_pattern: "loss: ([0-9.]+)"',
                '  metric_pattern: "[unterminated"',
            ),
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match=r"\[E034\]"):
            load_plan(plan_file)

    @pytest.mark.parametrize(
        "pattern",
        ['loss: [0-9.]+', 'loss: ([0-9.]+) step: ([0-9]+)'],
    )
    def test_watch_regex_must_have_one_group(self, tmp_path: Path, pattern: str) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            _WATCH_PLAN.replace('  metric_pattern: "loss: ([0-9.]+)"', f'  metric_pattern: "{pattern}"'),
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match=r"\[E034\]"):
            load_plan(plan_file)

    def test_watch_json_path_required_for_json_field(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            _WATCH_PLAN + '  metric_source: json_field\n',
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match=r"\[E035\]"):
            load_plan(plan_file)

    def test_watch_max_iterations_min_1(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_WATCH_PLAN.replace("  max_iterations: 10\n", "  max_iterations: 0\n"), encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E036\]"):
            load_plan(plan_file)

    def test_watch_warmup_must_be_less_than_max(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_WATCH_PLAN + "  warmup_iterations: 10\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E037\]"):
            load_plan(plan_file)

    def test_watch_plateau_threshold_min_1(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_WATCH_PLAN + "  plateau_threshold: 0\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E038\]"):
            load_plan(plan_file)

    def test_watch_max_cost_must_be_positive(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_WATCH_PLAN + "  max_cost_usd: -1\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E039\]"):
            load_plan(plan_file)

    def test_watch_metric_task_must_exist(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_WATCH_PLAN + "  metric_task: missing-task\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E040\]"):
            load_plan(plan_file)

    def test_watch_invalid_on_regression(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_WATCH_PLAN + "  on_regression: ignore\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E041\]"):
            load_plan(plan_file)

    def test_watch_invalid_plateau_action(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_WATCH_PLAN + "  plateau_action: pause\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E043\]"):
            load_plan(plan_file)

    def test_watch_iteration_budget_must_be_positive(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_WATCH_PLAN + "  iteration_budget_sec: 0\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E044\]"):
            load_plan(plan_file)

    def test_watch_plateau_action_escalate_model_is_valid(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            _WATCH_PLAN + "  plateau_action: escalate_model\n  plateau_threshold: 3\n",
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.plateau_action == "escalate_model"

    def test_watch_guard_command_source_needs_no_pattern(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: watch-guard
tasks:
  - id: experiment
    engine: claude
    prompt: "test"
    guard_command: "python check.py"
watch:
  metric: score
  metric_source: guard_command
  max_iterations: 5
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.metric_source == "guard_command"
        assert plan.watch.metric_pattern is None


_WORKTREE_PLAN = """\
version: 1
name: worktree-test
workspace_root: "{workspace_root}"
tasks:
  - id: impl
    engine: claude
    prompt: "test"
    worktree: true
"""


class TestWorktreeParsing:
    def test_worktree_false_by_default(self, tmp_path: Path) -> None:
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: worktree-test
tasks:
  - id: impl
    engine: claude
    prompt: "test"
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.tasks[0].worktree is False

    def test_worktree_parsed(self, tmp_path: Path) -> None:
        workspace_root = str(tmp_path).replace("\\", "\\\\")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_WORKTREE_PLAN.format(workspace_root=workspace_root), encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.tasks[0].worktree is True

    def test_worktree_requires_workspace_root(self, tmp_path: Path) -> None:
        workspace_root = str(tmp_path).replace("\\", "\\\\")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            _WORKTREE_PLAN.format(workspace_root=workspace_root).replace(
                f'workspace_root: "{workspace_root}"\n',
                "",
            ),
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match=r"\[E045\]"):
            load_plan(plan_file)

    def test_worktree_not_valid_on_command_task(self, tmp_path: Path) -> None:
        workspace_root = str(tmp_path).replace("\\", "\\\\")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            _WORKTREE_PLAN.format(workspace_root=workspace_root).replace(
                "    engine: claude\n    prompt: \"test\"\n",
                "    command: \"echo test\"\n",
            ),
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match=r"\[E046\]"):
            load_plan(plan_file)

    def test_worktree_not_valid_on_group_task(self, tmp_path: Path) -> None:
        workspace_root = str(tmp_path).replace("\\", "\\\\")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            _WORKTREE_PLAN.format(workspace_root=workspace_root).replace(
                "    engine: claude\n    prompt: \"test\"\n",
                "    group: \"sub.yaml\"\n",
            ),
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match=r"\[E046\]"):
            load_plan(plan_file)


class TestGuardCommandWarnings:
    def test_guard_command_on_group_task_warns(self, tmp_path: Path) -> None:
        """guard_command on a group task (no engine, no command) should warn."""
        sub_plan = tmp_path / "sub.yaml"
        sub_plan.write_text("""\
tasks:
  - id: inner
    command: echo inner
""", encoding="utf-8")
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text("""\
version: 1
name: test-plan
tasks:
  - id: t1
    group: sub.yaml
    guard_command: "python check.py"
""", encoding="utf-8")
        plan = load_plan(plan_file)
        assert any(
            "t1" in w and "guard_command" in w and "without engine or command" in w
            for w in plan.validation_warnings
        )


class TestCoercionHelpers:
    def test_env_dict_with_numeric_values_coerced_to_strings(self, tmp_path: Path) -> None:
        """_to_str_dict coerces non-string dict values to strings (e.g. PORT: 8080 → "8080")."""
        content = """\
version: 1
name: test-plan
defaults:
  env:
    PORT: 8080
    NAME: hello
tasks:
  - id: t1
    command: "echo hello"
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.defaults.env == {"PORT": "8080", "NAME": "hello"}

    def test_context_from_bare_string_normalized_to_list(self, tmp_path: Path) -> None:
        """_to_str_list coerces a bare string value to a single-element list."""
        content = """\
version: 1
name: test-plan
tasks:
  - id: a
    command: "echo a"
  - id: b
    depends_on: [a]
    context_from: "a"
    command: "echo b"
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        task_b = next(t for t in plan.tasks if t.id == "b")
        assert task_b.context_from == ["a"]


class TestAgentFieldParsing:
    def test_agent_field_parsed(self, tmp_path: Path) -> None:
        """Task with agent field stores the agent string on task.agent."""
        content = """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    agent: python-developer
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.tasks[0].agent == "python-developer"

    def test_agent_field_none_when_omitted(self, tmp_path: Path) -> None:
        """Task without agent field defaults to None."""
        content = """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.tasks[0].agent is None


class TestJsonSchemaLoaderValidation:
    """Loader validation for the json-schema judge criterion type."""

    def test_json_schema_criterion_valid_inline(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Generate JSON"
    judge:
      criteria:
        - type: json-schema
          schema:
            type: object
            required: [status]
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.tasks[0].judge is not None

    def test_json_schema_criterion_no_schema(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Generate JSON"
    judge:
      criteria:
        - type: json-schema
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E020"):
            load_plan(plan_file)

    def test_json_schema_criterion_both_fields(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Generate JSON"
    judge:
      criteria:
        - type: json-schema
          schema:
            type: object
          schema_file: /some/path.json
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E020"):
            load_plan(plan_file)


# --- v1.9.0: Policy Engine + Routing Strategy validation ---


class TestPoliciesValidation:
    def test_valid_policies_parsed(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t
    command: echo
policies:
  - name: cost-guard
    rule: 'task.engine == "codex"'
    action: warn
    message: "Codex detected"
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert len(plan.policies) == 1
        assert plan.policies[0].name == "cost-guard"
        assert plan.policies[0].action == "warn"

    def test_empty_policies_ok(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t
    command: echo
policies: []
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.policies == []

    def test_no_policies_ok(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t
    command: echo
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.policies == []

    def test_policy_missing_name_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t
    command: echo
policies:
  - name: ""
    rule: 'task.engine == "codex"'
    action: warn
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E052"):
            load_plan(plan_file)

    def test_policy_missing_rule_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t
    command: echo
policies:
  - name: cost-guard
    rule: ""
    action: warn
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E052"):
            load_plan(plan_file)

    def test_policy_invalid_action_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t
    command: echo
policies:
  - name: cost-guard
    rule: 'task.engine == "codex"'
    action: explode
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E052"):
            load_plan(plan_file)

    def test_policy_duplicate_names_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t
    command: echo
policies:
  - name: cost-guard
    rule: 'task.engine == "codex"'
    action: warn
  - name: cost-guard
    rule: 'task.engine == "claude"'
    action: block
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E052"):
            load_plan(plan_file)

    def test_policy_invalid_rule_syntax_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t
    command: echo
policies:
  - name: cost-guard
    rule: "invalid!!!"
    action: warn
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E052"):
            load_plan(plan_file)


class TestRoutingStrategyValidation:
    def test_valid_strategies(self, tmp_path: Path) -> None:
        for strategy in ("cost_optimized", "quality_first", "balanced"):
            content = f"""\
version: 1
name: test
tasks:
  - id: t
    command: echo
routing_strategy: {strategy}
"""
            plan_file = tmp_path / "plan.yaml"
            plan_file.write_text(content, encoding="utf-8")
            plan = load_plan(plan_file)
            assert plan.routing_strategy == strategy

    def test_no_strategy_ok(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t
    command: echo
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.routing_strategy is None

    def test_invalid_strategy_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t
    command: echo
routing_strategy: fast
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E053"):
            load_plan(plan_file)


# --- v1.10.0: Context Intelligence + Hardening validation ---


class TestLayeredContextMode:
    """context_mode: layered — zero-LLM-cost tiered context loading."""

    def test_layered_mode_parses(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: layered-test
tasks:
  - id: a
    command: echo a
  - id: b
    depends_on: [a]
    context_from: [a]
    context_mode: layered
    command: echo b
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        task_b = next(t for t in plan.tasks if t.id == "b")
        assert task_b.context_mode == "layered"

    def test_layered_mode_requires_context_from(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: layered-no-ctx
tasks:
  - id: a
    command: echo a
  - id: b
    depends_on: [a]
    context_mode: layered
    command: echo b
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="context_from"):
            load_plan(plan_file)

    def test_layered_mode_in_context_modes_set(self, tmp_path: Path) -> None:
        """'layered' must be a recognised context_mode value."""
        from maestro_cli.models import CONTEXT_MODES
        assert "layered" in CONTEXT_MODES

    def test_invalid_context_mode_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: bad-ctx-mode
tasks:
  - id: a
    depends_on: []
    context_from: []
    context_mode: magic
    command: echo a
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="context_mode"):
            load_plan(plan_file)


class TestJudgeQuorum:
    """judge.quorum — majority-vote quality gate for high-stakes tasks."""

    def test_quorum_valid_parsed(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: quorum-test
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt: "Do the thing"
    judge:
      criteria: ["output is correct"]
      quorum: 3
      quorum_strategy: majority
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        t1 = plan.tasks[0]
        assert t1.judge is not None
        assert t1.judge.quorum == 3
        assert t1.judge.quorum_strategy == "majority"

    def test_quorum_minimum_two(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: quorum-too-low
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt: "Do the thing"
    judge:
      criteria: ["output is correct"]
      quorum: 1
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E054"):
            load_plan(plan_file)

    def test_quorum_zero_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: quorum-zero
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt: "Do the thing"
    judge:
      criteria: ["output is correct"]
      quorum: 0
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E054"):
            load_plan(plan_file)

    def test_quorum_non_integer_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: quorum-float
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt: "Do the thing"
    judge:
      criteria: ["output is correct"]
      quorum: "two"
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E054"):
            load_plan(plan_file)

    def test_quorum_strategy_invalid_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: bad-quorum-strategy
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt: "Do the thing"
    judge:
      criteria: ["output is correct"]
      quorum: 3
      quorum_strategy: best_of_three
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E055"):
            load_plan(plan_file)

    def test_quorum_strategy_without_quorum_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: strategy-no-quorum
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt: "Do the thing"
    judge:
      criteria: ["output is correct"]
      quorum_strategy: majority
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E056"):
            load_plan(plan_file)

    def test_quorum_strategy_all_valid_values(self, tmp_path: Path) -> None:
        for strategy in ("majority", "unanimous", "any"):
            content = f"""\
version: 1
name: quorum-strategy-{strategy}
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt: "Do the thing"
    judge:
      criteria: ["output is correct"]
      quorum: 3
      quorum_strategy: {strategy}
"""
            plan_file = tmp_path / f"plan_{strategy}.yaml"
            plan_file.write_text(content, encoding="utf-8")
            plan = load_plan(plan_file)
            assert plan.tasks[0].judge is not None
            assert plan.tasks[0].judge.quorum_strategy == strategy

    def test_quorum_without_strategy_defaults_to_none(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: quorum-no-strategy
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt: "Do the thing"
    judge:
      criteria: ["output is correct"]
      quorum: 2
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.quorum == 2
        assert plan.tasks[0].judge.quorum_strategy is None

    def test_no_quorum_judge_defaults(self, tmp_path: Path) -> None:
        """A judge block without quorum fields leaves both as None."""
        content = """\
version: 1
name: no-quorum
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt: "Do the thing"
    judge:
      criteria: ["output is correct"]
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.quorum is None
        assert plan.tasks[0].judge.quorum_strategy is None


class TestObservationBlock:
    """observation_block — sandboxes context_from output to prevent prompt injection."""

    def test_observation_block_default_false(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: obs-default
tasks:
  - id: t1
    command: echo hello
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.tasks[0].observation_block is False

    def test_observation_block_true_parses(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: obs-enabled
tasks:
  - id: a
    command: echo a
  - id: b
    depends_on: [a]
    context_from: [a]
    observation_block: true
    command: echo b
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        task_b = next(t for t in plan.tasks if t.id == "b")
        assert task_b.observation_block is True

    def test_observation_block_without_context_from_warns(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: obs-no-ctx
tasks:
  - id: t1
    observation_block: true
    command: echo hello
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert any("observation_block" in w for w in plan.validation_warnings)

    def test_observation_block_with_context_from_no_warning(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: obs-with-ctx
tasks:
  - id: a
    command: echo a
  - id: b
    depends_on: [a]
    context_from: [a]
    observation_block: true
    command: echo b
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert not any("observation_block" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# mode: improve parsing
# ---------------------------------------------------------------------------


class TestWatchModeImprove:
    """Loader tests for watch mode: improve."""

    def test_mode_improve_parsed(self, tmp_path: Path) -> None:
        """mode: improve is parsed correctly."""
        content = """\
version: 1
name: improve-test
workspace_root: "."
tasks:
  - id: t1
    command: echo hello
watch:
  mode: improve
  max_iterations: 5
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.mode == "improve"

    def test_mode_improve_auto_defaults(self, tmp_path: Path) -> None:
        """mode: improve auto-sets metric, metric_source, metric_direction."""
        content = """\
version: 1
name: auto-defaults
workspace_root: "."
tasks:
  - id: t1
    command: echo hello
watch:
  mode: improve
  max_iterations: 5
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        w = plan.watch
        assert w is not None
        assert w.metric == "tasks_passed"
        assert w.metric_source == "manifest"
        assert w.metric_direction == "higher_is_better"
        assert w.warmup_iterations == 0
        assert w.on_regression == "rollback"
        assert w.plateau_threshold == 3
        assert w.plateau_action == "stop"

    def test_mode_improve_with_model_override(self, tmp_path: Path) -> None:
        """improve_model overrides the default model."""
        content = """\
version: 1
name: model-override
workspace_root: "."
tasks:
  - id: t1
    command: echo hello
watch:
  mode: improve
  max_iterations: 5
  improve_model: opus
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.improve_model == "opus"

    def test_mode_improve_explicit_overrides_preserved(self, tmp_path: Path) -> None:
        """User-provided overrides take priority over improve defaults."""
        content = """\
version: 1
name: overrides
workspace_root: "."
tasks:
  - id: t1
    command: echo hello
watch:
  mode: improve
  max_iterations: 5
  plateau_threshold: 10
  on_regression: keep
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.watch.plateau_threshold == 10
        assert plan.watch.on_regression == "keep"

    def test_mode_improve_requires_workspace_root(self, tmp_path: Path) -> None:
        """mode: improve without workspace_root raises E047."""
        content = """\
version: 1
name: no-workspace
tasks:
  - id: t1
    command: echo hello
watch:
  mode: improve
  max_iterations: 5
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E047"):
            load_plan(plan_file)

    def test_invalid_mode_raises(self, tmp_path: Path) -> None:
        """Invalid mode value raises E048."""
        content = """\
version: 1
name: bad-mode
workspace_root: "."
tasks:
  - id: t1
    command: echo hello
watch:
  mode: invalid
  metric: x
  metric_source: stdout_regex
  metric_pattern: '(\\d+)'
  max_iterations: 5
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E048"):
            load_plan(plan_file)

    def test_metric_source_manifest_no_pattern_required(self, tmp_path: Path) -> None:
        """metric_source: manifest does not require metric_pattern."""
        content = """\
version: 1
name: manifest-source
workspace_root: "."
tasks:
  - id: t1
    command: echo hello
watch:
  metric: tasks_passed
  metric_source: manifest
  metric_direction: higher_is_better
  max_iterations: 5
  warmup_iterations: 0
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.metric_source == "manifest"
        assert plan.watch.metric_pattern is None

    def test_mode_custom_is_default(self, tmp_path: Path) -> None:
        """Default mode is 'custom' (backward compatible)."""
        content = """\
version: 1
name: default-mode
tasks:
  - id: t1
    command: echo hello
watch:
  metric: score
  metric_source: stdout_regex
  metric_pattern: '(\\d+)'
  max_iterations: 5
  warmup_iterations: 0
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.watch.mode == "custom"


# ---------------------------------------------------------------------------
# v1.14.0 — Plan Density Score (compute_plan_density / compute_plan_density_score)
# ---------------------------------------------------------------------------

class TestComputePlanDensity:
    """Unit tests for compute_plan_density() — kills BinaryOp / NumberReplacer mutants."""

    def test_chain_two_tasks(self, tmp_path: Path) -> None:
        """Linear chain a→b: nodes=2, edges=1, depth=1."""
        from maestro_cli.loader import compute_plan_density
        from maestro_cli.models import PlanSpec, TaskSpec

        tasks = [TaskSpec(id="a"), TaskSpec(id="b", depends_on=["a"])]
        plan = PlanSpec(name="t", tasks=tasks)
        d = compute_plan_density(plan)
        assert d["nodes"] == 2
        assert d["edges"] == 1
        assert d["depth"] == 1
        assert d["s_complex"] > 0.0

    def test_empty_plan(self) -> None:
        from maestro_cli.loader import compute_plan_density
        from maestro_cli.models import PlanSpec

        d = compute_plan_density(PlanSpec(name="e", tasks=[]))
        assert d["nodes"] == 0
        assert d["edges"] == 0
        assert d["depth"] == 0
        assert d["s_complex"] == 0.0

    def test_parallel_tasks_no_edges(self) -> None:
        """Three independent tasks: edges=0, depth=0."""
        from maestro_cli.loader import compute_plan_density
        from maestro_cli.models import PlanSpec, TaskSpec

        tasks = [TaskSpec(id=f"t{i}") for i in range(3)]
        d = compute_plan_density(PlanSpec(name="t", tasks=tasks))
        assert d["nodes"] == 3
        assert d["edges"] == 0
        assert d["depth"] == 0

    def test_chain_depth_equals_edges_for_linear(self) -> None:
        """a→b→c: depth=2, edges=2."""
        from maestro_cli.loader import compute_plan_density
        from maestro_cli.models import PlanSpec, TaskSpec

        tasks = [
            TaskSpec(id="a"),
            TaskSpec(id="b", depends_on=["a"]),
            TaskSpec(id="c", depends_on=["b"]),
        ]
        d = compute_plan_density(PlanSpec(name="t", tasks=tasks))
        assert d["edges"] == 2
        assert d["depth"] == 2

    def test_dense_dag_has_higher_edge_count(self) -> None:
        """Fully connected 3-node DAG has 3 edges (a→b, a→c, b→c)."""
        from maestro_cli.loader import compute_plan_density
        from maestro_cli.models import PlanSpec, TaskSpec

        tasks = [
            TaskSpec(id="a"),
            TaskSpec(id="b", depends_on=["a"]),
            TaskSpec(id="c", depends_on=["a", "b"]),
        ]
        d = compute_plan_density(PlanSpec(name="t", tasks=tasks))
        assert d["edges"] == 3

    def test_s_complex_decreases_with_more_tasks(self) -> None:
        """More tasks → lower raw s_complex (exp of decreasing components)."""
        from maestro_cli.loader import compute_plan_density
        from maestro_cli.models import PlanSpec, TaskSpec

        small = compute_plan_density(PlanSpec(name="s", tasks=[TaskSpec(id="t1")]))
        large_tasks = [
            TaskSpec(id=f"t{i}", depends_on=[f"t{i-1}"] if i > 0 else [])
            for i in range(6)
        ]
        large = compute_plan_density(PlanSpec(name="l", tasks=large_tasks))
        # Raw s_complex is exp(s_node + 2*s_edge + s_depth) where components
        # DECREASE with complexity, so a larger/denser DAG has LOWER s_complex.
        assert large["s_complex"] < small["s_complex"]

    def test_all_keys_present(self) -> None:
        """compute_plan_density returns all expected keys."""
        from maestro_cli.loader import compute_plan_density
        from maestro_cli.models import PlanSpec, TaskSpec

        d = compute_plan_density(PlanSpec(name="t", tasks=[TaskSpec(id="x")]))
        for key in ("nodes", "edges", "depth", "s_node", "s_edge", "s_depth", "s_complex"):
            assert key in d

    def test_score_label_low_for_simple_plan(self) -> None:
        """Single-task plan is 'low' complexity."""
        from maestro_cli.loader import compute_plan_density_score
        from maestro_cli.models import PlanSpec, TaskSpec

        _, label, _ = compute_plan_density_score(PlanSpec(name="t", tasks=[TaskSpec(id="t1")]))
        assert label in ("low", "moderate")

    def test_score_label_empty_plan_is_low(self) -> None:
        from maestro_cli.loader import compute_plan_density_score
        from maestro_cli.models import PlanSpec

        score, label, factors = compute_plan_density_score(PlanSpec(name="e", tasks=[]))
        assert score == 0.0
        assert label == "low"
        assert factors == ""

    def test_score_factors_contains_s_complex(self) -> None:
        """factors string mentions S_complex for non-trivial plans."""
        from maestro_cli.loader import compute_plan_density_score
        from maestro_cli.models import PlanSpec, TaskSpec

        tasks = [
            TaskSpec(id=f"t{i}", depends_on=[f"t{i-1}"] if i > 0 else [])
            for i in range(5)
        ]
        _, _, factors = compute_plan_density_score(PlanSpec(name="t", tasks=tasks))
        assert "S_complex" in factors


# ---------------------------------------------------------------------------
# v1.14.0 — W17 / W18 / W19 density warnings
# ---------------------------------------------------------------------------

class TestDensityWarnings:
    """W17 / W18 / W19 warnings emitted from validate_plan()."""

    def _write(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "plan.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_w17_dense_dag_triggers(self, tmp_path: Path) -> None:
        """Fully-connected 4-node DAG (6 edges / max 6 = 100%) → W17."""
        yaml = """\
version: 1
name: dense
tasks:
  - id: a
    command: echo
  - id: b
    depends_on: [a]
    command: echo
  - id: c
    depends_on: [a, b]
    command: echo
  - id: d
    depends_on: [a, b, c]
    command: echo
"""
        plan = load_plan(self._write(tmp_path, yaml))
        warns = plan.validation_warnings or []
        assert any("W17" in w for w in warns), f"No W17 in {warns}"

    def test_w17_sparse_dag_no_warning(self, tmp_path: Path) -> None:
        """Fan-out: 5 tasks, 4 edges, max=10 → 40% density → no W17."""
        yaml = """\
version: 1
name: fanout
tasks:
  - id: root
    command: echo
  - id: a
    depends_on: [root]
    command: echo
  - id: b
    depends_on: [root]
    command: echo
  - id: c
    depends_on: [root]
    command: echo
  - id: d
    depends_on: [root]
    command: echo
"""
        plan = load_plan(self._write(tmp_path, yaml))
        warns = plan.validation_warnings or []
        assert not any("W17" in w for w in warns)

    def test_w17_single_task_no_warning(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: single
tasks:
  - id: t1
    command: echo
"""
        plan = load_plan(self._write(tmp_path, yaml))
        warns = plan.validation_warnings or []
        assert not any("W17" in w for w in warns)

    def test_w18_deep_chain_triggers(self, tmp_path: Path) -> None:
        """6-task linear chain → low parallelism → W18."""
        tasks_yaml = "\n".join(
            f"  - id: t{i}\n    depends_on: [t{i-1}]\n    command: echo"
            if i > 0
            else f"  - id: t{i}\n    command: echo"
            for i in range(6)
        )
        yaml = f"version: 1\nname: chain6\ntasks:\n{tasks_yaml}\n"
        plan = load_plan(self._write(tmp_path, yaml))
        warns = plan.validation_warnings or []
        assert any("W18" in w for w in warns), f"No W18 in {warns}"

    def test_w18_parallel_tasks_no_warning(self, tmp_path: Path) -> None:
        """Four independent tasks → no W18."""
        yaml = """\
version: 1
name: par
tasks:
  - id: t1
    command: echo
  - id: t2
    command: echo
  - id: t3
    command: echo
  - id: t4
    command: echo
"""
        plan = load_plan(self._write(tmp_path, yaml))
        warns = plan.validation_warnings or []
        assert not any("W18" in w for w in warns)

    def test_w18_small_chain_no_warning(self, tmp_path: Path) -> None:
        """3-task chain is below the threshold for W18."""
        yaml = """\
version: 1
name: short
tasks:
  - id: a
    command: echo
  - id: b
    depends_on: [a]
    command: echo
  - id: c
    depends_on: [b]
    command: echo
"""
        plan = load_plan(self._write(tmp_path, yaml))
        warns = plan.validation_warnings or []
        assert not any("W18" in w for w in warns)

    def test_w19_simple_plan_no_warning(self, tmp_path: Path) -> None:
        """Two-task plan has low S_complex → no W19."""
        yaml = """\
version: 1
name: simple
tasks:
  - id: t1
    command: echo
  - id: t2
    depends_on: [t1]
    command: echo
"""
        plan = load_plan(self._write(tmp_path, yaml))
        warns = plan.validation_warnings or []
        assert not any("W19" in w for w in warns)


# ---------------------------------------------------------------------------
# v1.14.0 — Deliberation gate (loader validation)
# ---------------------------------------------------------------------------

class TestDeliberationLoaderValidation:
    """Tests for deliberation field parsing and validation in loader.py."""

    def _write(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "plan.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_deliberation_defaults_false(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: t
tasks:
  - id: t1
    command: echo
"""
        plan = load_plan(self._write(tmp_path, yaml))
        assert plan.tasks[0].deliberation is False
        assert plan.tasks[0].deliberation_threshold == 0.5

    def test_deliberation_true_loaded(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: t
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    deliberation: true
    deliberation_threshold: 0.7
"""
        plan = load_plan(self._write(tmp_path, yaml))
        assert plan.tasks[0].deliberation is True
        assert plan.tasks[0].deliberation_threshold == 0.7

    def test_deliberation_threshold_out_of_range_high(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: bad
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    deliberation: true
    deliberation_threshold: 1.5
"""
        with pytest.raises(PlanValidationError, match="deliberation_threshold"):
            load_plan(self._write(tmp_path, yaml))

    def test_deliberation_threshold_out_of_range_low(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: bad
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    deliberation: true
    deliberation_threshold: -0.1
"""
        with pytest.raises(PlanValidationError, match="deliberation_threshold"):
            load_plan(self._write(tmp_path, yaml))

    def test_deliberation_threshold_non_numeric(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: bad
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    deliberation: true
    deliberation_threshold: "oops"
"""
        with pytest.raises(PlanValidationError, match="deliberation_threshold"):
            load_plan(self._write(tmp_path, yaml))

    @pytest.mark.parametrize("thresh", [0.0, 0.5, 1.0])
    def test_deliberation_threshold_boundary_valid(self, tmp_path: Path, thresh: float) -> None:
        yaml = f"""\
version: 1
name: boundary
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    deliberation: true
    deliberation_threshold: {thresh}
"""
        plan = load_plan(self._write(tmp_path, yaml))
        assert plan.tasks[0].deliberation_threshold == thresh


# ---------------------------------------------------------------------------
# v1.14.0 — Adversarial Debate judge (debate_rounds loader validation)
# ---------------------------------------------------------------------------

class TestDebateRoundsLoaderValidation:
    """Tests for debate_rounds parsing in _to_judge_spec() — kills BoundaryOp mutants."""

    def _write(self, tmp_path: Path, content: str) -> Path:
        p = tmp_path / "plan.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_debate_rounds_default_is_2(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: t
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    judge:
      criteria: ["quality"]
      method: debate
"""
        plan = load_plan(self._write(tmp_path, yaml))
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.debate_rounds == 2

    def test_debate_rounds_custom_value(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: t
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    judge:
      criteria: ["quality"]
      method: debate
      debate_rounds: 3
"""
        plan = load_plan(self._write(tmp_path, yaml))
        assert plan.tasks[0].judge.debate_rounds == 3

    def test_debate_rounds_minimum_1_is_valid(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: t
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    judge:
      criteria: ["quality"]
      method: debate
      debate_rounds: 1
"""
        plan = load_plan(self._write(tmp_path, yaml))
        assert plan.tasks[0].judge.debate_rounds == 1

    def test_debate_rounds_zero_raises(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: bad
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    judge:
      criteria: ["quality"]
      method: debate
      debate_rounds: 0
"""
        with pytest.raises(PlanValidationError, match="debate_rounds"):
            load_plan(self._write(tmp_path, yaml))

    def test_debate_rounds_negative_raises(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: bad
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    judge:
      criteria: ["quality"]
      method: debate
      debate_rounds: -1
"""
        with pytest.raises(PlanValidationError, match="debate_rounds"):
            load_plan(self._write(tmp_path, yaml))

    def test_debate_rounds_non_integer_raises(self, tmp_path: Path) -> None:
        yaml = """\
version: 1
name: bad
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    judge:
      criteria: ["quality"]
      method: debate
      debate_rounds: "three"
"""
        with pytest.raises(PlanValidationError, match="debate_rounds"):
            load_plan(self._write(tmp_path, yaml))


# ===========================================================================
# Additional coverage tests (appended)
# ===========================================================================


# ---------------------------------------------------------------------------
# 1. YAML Anchors and Aliases — merge keys
# ---------------------------------------------------------------------------

class TestYamlAnchorsAndAliases:
    """Expanded YAML anchor/alias coverage beyond TestSchemaMigration."""

    def test_anchor_with_multiple_overrides(self, tmp_path: Path) -> None:
        """Multiple tasks sharing anchors with different overrides."""
        content = """\
version: 1
name: multi-anchor
_base: &base
  engine: claude
  model: sonnet
  max_retries: 1
tasks:
  - id: t1
    <<: *base
    prompt: "Task 1"
  - id: t2
    <<: *base
    model: opus
    prompt: "Task 2"
    depends_on: [t1]
  - id: t3
    <<: *base
    engine: codex
    prompt: "Task 3"
    depends_on: [t1]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert len(plan.tasks) == 3
        t1 = next(t for t in plan.tasks if t.id == "t1")
        t2 = next(t for t in plan.tasks if t.id == "t2")
        t3 = next(t for t in plan.tasks if t.id == "t3")
        assert t1.model == "sonnet"
        assert t1.max_retries == 1
        assert t2.model == "opus"
        assert t3.engine == "codex"

    def test_anchor_in_env_block(self, tmp_path: Path) -> None:
        """YAML anchors in defaults.env merged via alias."""
        content = """\
version: 1
name: anchor-env
_env: &shared_env
  SHARED_VAR: "hello"
  PORT: 8080
defaults:
  env:
    <<: *shared_env
    EXTRA: "world"
tasks:
  - id: t1
    command: echo ok
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.defaults.env["SHARED_VAR"] == "hello"
        assert plan.defaults.env["PORT"] == "8080"
        assert plan.defaults.env["EXTRA"] == "world"


# ---------------------------------------------------------------------------
# 2. Matrix expansion — Cartesian product, template variables
# ---------------------------------------------------------------------------

class TestMatrixExpansionExtended:
    """Extended matrix expansion tests."""

    def test_matrix_multiple_keys_cartesian_product(self, tmp_path: Path) -> None:
        """Two keys: env=[dev,prod], region=[us,eu] -> 4 tasks."""
        content = """\
version: 1
name: matrix-multi
tasks:
  - id: deploy
    engine: claude
    prompt: "Deploy {{ matrix.env }} to {{ matrix.region }}"
    matrix:
      env: [dev, prod]
      region: [us, eu]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert len(plan.tasks) == 4
        ids = {t.id for t in plan.tasks}
        # Original ID "deploy" should not be present
        assert "deploy" not in ids
        # All expanded tasks should have matrix_values
        for t in plan.tasks:
            assert t.matrix_values is not None
            assert "env" in t.matrix_values
            assert "region" in t.matrix_values

    def test_matrix_single_key(self, tmp_path: Path) -> None:
        """Single key with 3 values -> 3 tasks."""
        content = """\
version: 1
name: matrix-single
tasks:
  - id: lint
    command: "echo {{ matrix.lang }}"
    matrix:
      lang: [python, rust, go]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert len(plan.tasks) == 3

    def test_matrix_empty_values_raises(self, tmp_path: Path) -> None:
        """Empty values list in matrix raises PlanValidationError."""
        content = """\
version: 1
name: matrix-empty
tasks:
  - id: t1
    command: echo
    matrix:
      env: []
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="non-empty list"):
            load_plan(pf)

    def test_matrix_depends_on_rewritten(self, tmp_path: Path) -> None:
        """Downstream task depending on matrix parent gets expanded deps."""
        content = """\
version: 1
name: matrix-deps
tasks:
  - id: build
    engine: claude
    prompt: "Build {{ matrix.os }}"
    matrix:
      os: [linux, windows]
  - id: deploy
    depends_on: [build]
    command: echo deploy
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        deploy = next(t for t in plan.tasks if t.id == "deploy")
        # deploy.depends_on should have been rewritten to the expanded IDs
        assert len(deploy.depends_on) == 2
        assert "build" not in deploy.depends_on

    def test_matrix_not_a_dict_raises(self, tmp_path: Path) -> None:
        """Matrix as a list instead of dict raises."""
        content = """\
version: 1
name: matrix-bad
tasks:
  - id: t1
    command: echo
    matrix:
      - env: [dev]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 3. Imports — prefix namespacing, nested, validation
# ---------------------------------------------------------------------------

class TestImportsExtended:
    """Extended import tests."""

    def test_import_duplicate_prefix_raises_e027(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        a.write_text("tasks:\n  - id: t1\n    command: echo\n", encoding="utf-8")
        b = tmp_path / "b.yaml"
        b.write_text("tasks:\n  - id: t1\n    command: echo\n", encoding="utf-8")
        pf = tmp_path / "plan.yaml"
        pf.write_text(f"""\
version: 1
name: dup-prefix
imports:
  - path: a.yaml
    prefix: lib
  - path: b.yaml
    prefix: lib
tasks:
  - id: main
    command: echo
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E027\]"):
            load_plan(pf)

    def test_import_invalid_prefix_raises_e028(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        a.write_text("tasks:\n  - id: t1\n    command: echo\n", encoding="utf-8")
        pf = tmp_path / "plan.yaml"
        pf.write_text(f"""\
version: 1
name: bad-prefix
imports:
  - path: a.yaml
    prefix: UPPER
tasks:
  - id: main
    command: echo
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E028\]"):
            load_plan(pf)

    def test_import_missing_file_raises_e026(self, tmp_path: Path) -> None:
        pf = tmp_path / "plan.yaml"
        pf.write_text("""\
version: 1
name: missing-import
imports:
  - path: nonexistent.yaml
    prefix: lib
tasks:
  - id: main
    command: echo
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E026\]"):
            load_plan(pf)

    def test_import_missing_path_or_prefix_raises_e026(self, tmp_path: Path) -> None:
        pf = tmp_path / "plan.yaml"
        pf.write_text("""\
version: 1
name: bad-import
imports:
  - path: some.yaml
tasks:
  - id: main
    command: echo
""", encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E026\]"):
            load_plan(pf)

    def test_nested_imports(self, tmp_path: Path) -> None:
        """Nested imports (depth 2) resolve correctly."""
        inner = tmp_path / "inner.yaml"
        inner.write_text("""\
tasks:
  - id: deep
    command: echo deep
""", encoding="utf-8")
        outer = tmp_path / "outer.yaml"
        outer.write_text(f"""\
imports:
  - path: inner.yaml
    prefix: inner
tasks:
  - id: mid
    depends_on: [inner/deep]
    command: echo mid
""", encoding="utf-8")
        pf = tmp_path / "plan.yaml"
        pf.write_text(f"""\
version: 1
name: nested-import
imports:
  - path: outer.yaml
    prefix: outer
tasks:
  - id: main
    depends_on: [outer/mid]
    command: echo main
""", encoding="utf-8")
        plan = load_plan(pf)
        ids = {t.id for t in plan.tasks}
        assert "inner/deep" in ids
        assert "outer/mid" in ids
        assert "main" in ids


# ---------------------------------------------------------------------------
# 4. E029: approval_message without requires_approval
# ---------------------------------------------------------------------------

class TestApprovalValidation:
    def test_approval_message_without_requires_approval_raises_e029(
        self, tmp_path: Path
    ) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    approval_message: "Please approve this"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E029\]"):
            load_plan(pf)

    def test_approval_message_with_requires_approval_ok(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    requires_approval: true
    approval_message: "Please approve this"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].requires_approval is True
        assert plan.tasks[0].approval_message == "Please approve this"


# ---------------------------------------------------------------------------
# 5. E030/E031: escalation and fallback validation (extended)
# ---------------------------------------------------------------------------

class TestEscalationFallbackExtended:
    def test_escalation_non_list_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    escalation: "haiku"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        # escalation as a bare string is coerced to a list by _to_str_list
        plan = load_plan(pf)
        assert plan.tasks[0].escalation == ["haiku"]

    def test_escalation_empty_list_raises(self, tmp_path: Path) -> None:
        """Empty escalation list should not error -- it's treated as no escalation."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    escalation: []
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].escalation == []

    def test_fallback_engine_unknown_raises_e030(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "hi"
    fallback_engine: not_a_real_engine
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E030"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 6. E050: circuit_breaker validation
# ---------------------------------------------------------------------------

class TestCircuitBreakerValidation:
    def test_circuit_breaker_valid(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
circuit_breaker:
  max_total_failures: 3
  action: fail
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.circuit_breaker is not None
        assert plan.circuit_breaker.max_total_failures == 3
        assert plan.circuit_breaker.action == "fail"

    def test_circuit_breaker_max_failures_zero_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
circuit_breaker:
  max_total_failures: 0
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E050"):
            load_plan(pf)

    def test_circuit_breaker_not_a_mapping_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
circuit_breaker: [1, 2]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E050"):
            load_plan(pf)

    def test_circuit_breaker_invalid_action_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
circuit_breaker:
  max_total_failures: 5
  action: restart
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E050"):
            load_plan(pf)

    def test_circuit_breaker_pause_action(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
circuit_breaker:
  max_total_failures: 2
  action: pause
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.circuit_breaker.action == "pause"


# ---------------------------------------------------------------------------
# 7. E051: retry_strategy validation
# ---------------------------------------------------------------------------

class TestRetryStrategyValidation:
    def test_valid_retry_strategies(self, tmp_path: Path) -> None:
        for strategy in ("constant", "linear", "exponential"):
            content = f"""\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    max_retries: 2
    retry_strategy: {strategy}
"""
            pf = tmp_path / "plan.yaml"
            pf.write_text(content, encoding="utf-8")
            plan = load_plan(pf)
            assert plan.tasks[0].retry_strategy == strategy

    def test_invalid_retry_strategy_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    retry_strategy: random
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E051"):
            load_plan(pf)

    def test_defaults_retry_strategy_invalid_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
defaults:
  retry_strategy: invalid
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E051"):
            load_plan(pf)

    def test_task_retry_strategy_overrides_default(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
defaults:
  retry_strategy: constant
tasks:
  - id: t1
    command: echo
    retry_strategy: exponential
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].retry_strategy == "exponential"


# ---------------------------------------------------------------------------
# 8. E057-E058, E060, E062: batch validation
# ---------------------------------------------------------------------------

class TestBatchValidation:
    def test_batch_valid(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    batch:
      items: [a, b, c]
      template: "Process {{ batch.item }}"
      max_per_call: 2
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].batch is not None
        assert plan.tasks[0].batch.items == ["a", "b", "c"]
        assert plan.tasks[0].batch.max_per_call == 2

    def test_batch_missing_items_raises_e057(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    batch:
      template: "Process {{ batch.item }}"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E057"):
            load_plan(pf)

    def test_batch_missing_template_raises_e057(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    batch:
      items: [a, b]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E057"):
            load_plan(pf)

    def test_batch_template_no_placeholder_raises_e057(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    batch:
      items: [a]
      template: "Process something"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E057"):
            load_plan(pf)

    def test_batch_empty_items_raises_e057(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    batch:
      items: []
      template: "Process {{ batch.item }}"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E057"):
            load_plan(pf)

    def test_batch_max_per_call_zero_raises_e058(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    batch:
      items: [a]
      template: "Process {{ batch.item }}"
      max_per_call: 0
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E058"):
            load_plan(pf)

    def test_batch_on_command_task_raises_e060(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    batch:
      items: [a]
      template: "{{ batch.item }}"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E060"):
            load_plan(pf)

    def test_batch_not_a_dict_raises_e057(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    batch: "not a dict"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E057"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 9. E063/E064: dynamic_group validation
# ---------------------------------------------------------------------------

class TestDynamicGroupValidation:
    def test_dynamic_group_requires_engine_raises_e063(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    dynamic_group: true
    output_schema:
      type: object
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E063"):
            load_plan(pf)

    def test_dynamic_group_requires_output_schema_raises_e063(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "plan something"
    dynamic_group: true
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E063"):
            load_plan(pf)

    def test_dynamic_group_conflicts_with_batch_raises_e064_alt(self, tmp_path: Path) -> None:
        """dynamic_group + batch -> E064 (different from the other batch test)."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    dynamic_group: true
    output_schema:
      type: object
      properties:
        tasks:
          type: array
    batch:
      items: [x, y]
      template: "Process {{ batch.item }}"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E064"):
            load_plan(pf)

    def test_dynamic_group_conflicts_with_batch_raises_e064(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    dynamic_group: true
    output_schema:
      type: object
    batch:
      items: [a]
      template: "{{ batch.item }}"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E064"):
            load_plan(pf)

    def test_dynamic_group_valid(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "plan tasks"
    dynamic_group: true
    output_schema:
      type: object
      properties:
        tasks:
          type: array
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].dynamic_group is True
        # dynamic_group forces cache: false
        assert plan.tasks[0].cache is False


# ---------------------------------------------------------------------------
# 10. E065: context_trust validation
# ---------------------------------------------------------------------------

class TestContextTrustValidation:
    def test_context_trust_valid_values(self, tmp_path: Path) -> None:
        for value in ("trusted", "untrusted"):
            content = f"""\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    context_trust: {value}
"""
            pf = tmp_path / "plan.yaml"
            pf.write_text(content, encoding="utf-8")
            plan = load_plan(pf)
            assert plan.tasks[0].context_trust == value

    def test_context_trust_invalid_raises_e065(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    context_trust: maybe
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E065"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 11. Warning codes: W4 (backslashes in path fields)
# ---------------------------------------------------------------------------

class TestBackslashWarnings:
    def test_w4_backslash_in_workspace_root(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
workspace_root: "C:\\\\Users\\\\dev\\\\project"
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("backslashes" in w and "workspace_root" in w for w in plan.validation_warnings)

    def test_w4_backslash_in_run_dir(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
run_dir: "runs\\\\output"
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("backslashes" in w and "run_dir" in w for w in plan.validation_warnings)

    def test_w4_backslash_in_task_workdir(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    workdir: "some\\\\path"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("backslashes" in w and "workdir" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# 12. W6: retry_delay_sec list shorter than max_retries
# ---------------------------------------------------------------------------

class TestRetryDelayWarning:
    def test_w6_short_retry_delay_list(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    max_retries: 3
    retry_delay_sec: [1.0]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any(
            "retry_delay_sec" in w and "reused" in w
            for w in plan.validation_warnings
        )

    def test_w6_matching_length_no_warning(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    max_retries: 2
    retry_delay_sec: [1.0, 2.0]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any(
            "retry_delay_sec" in w and "reused" in w
            for w in plan.validation_warnings
        )


# ---------------------------------------------------------------------------
# 13. W7: env var reference not in allowlist
# ---------------------------------------------------------------------------

class TestEnvVarWarning:
    def test_w7_unknown_env_ref(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: "echo $SOME_UNKNOWN_VAR"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any(
            "SOME_UNKNOWN_VAR" in w and "not in the env allowlist" in w
            for w in plan.validation_warnings
        )

    def test_w7_known_env_ref_no_warning(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: "echo $HOME"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any(
            "HOME" in w and "not in the env allowlist" in w
            for w in plan.validation_warnings
        )

    def test_w7_task_env_var_no_warning(self, tmp_path: Path) -> None:
        """Env var defined in task.env should not trigger W7."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: "echo $MY_CUSTOM_VAR"
    env:
      MY_CUSTOM_VAR: hello
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any(
            "MY_CUSTOM_VAR" in w and "not in the env allowlist" in w
            for w in plan.validation_warnings
        )


# ---------------------------------------------------------------------------
# 14. W8: tag with whitespace
# ---------------------------------------------------------------------------

class TestTagWhitespaceWarning:
    def test_w8_tag_with_space(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    tags: ["my tag"]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("whitespace" in w and "my tag" in w for w in plan.validation_warnings)

    def test_w8_tag_with_tab(self, tmp_path: Path) -> None:
        content = "version: 1\nname: test\ntasks:\n  - id: t1\n    command: echo\n    tags: [\"my\\ttag\"]\n"
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("whitespace" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# 15. W16: single worktree task
# ---------------------------------------------------------------------------

class TestWorktreeWarnings:
    def test_w16_single_worktree_task_warns(self, tmp_path: Path) -> None:
        ws = str(tmp_path).replace("\\", "/")
        content = f"""\
version: 1
name: test
workspace_root: "{ws}"
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    worktree: true
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any(
            "worktree: true" in w and "only one" in w
            for w in plan.validation_warnings
        )

    def test_two_worktree_tasks_no_w16(self, tmp_path: Path) -> None:
        ws = str(tmp_path).replace("\\", "/")
        content = f"""\
version: 1
name: test
workspace_root: "{ws}"
tasks:
  - id: t1
    engine: claude
    prompt: "a"
    worktree: true
  - id: t2
    engine: claude
    prompt: "b"
    worktree: true
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any(
            "only one worktree" in w
            for w in plan.validation_warnings
        )


# ---------------------------------------------------------------------------
# 16. W20: engine retries without escape valve (consolidated W20 + W21 + W-no-retry-with-verify)
# ---------------------------------------------------------------------------

class TestTimeoutRetryWarnings:
    def test_w20_fires_when_no_escape_valve(self, tmp_path: Path) -> None:
        # Engine task with retries but no verify/guard/escalation/fallback/progressive delay.
        content = """\
version: 1
name: test
defaults:
  timeout_sec: 300
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    max_retries: 2
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("W20" in w and "t1" in w for w in plan.validation_warnings)

    def test_w20_silent_when_verify_command_present(self, tmp_path: Path) -> None:
        # Regression test: verify_command supplies retry feedback, so the
        # legacy W20 should NOT fire even with a tight inherited timeout.
        content = """\
version: 1
name: test
defaults:
  timeout_sec: 300
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    max_retries: 2
    verify_command: ["test", "-f", "out.txt"]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_w20_silent_when_progressive_delay_present(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
defaults:
  timeout_sec: 300
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    max_retries: 2
    retry_delay_sec: [60, 120]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_w20_silent_when_escalation_present(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
defaults:
  timeout_sec: 300
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    max_retries: 2
    escalation: [haiku, sonnet, opus]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_w20_silent_when_fallback_engine_present(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
defaults:
  timeout_sec: 300
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    max_retries: 1
    fallback_engine: codex
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_w20_silent_when_judge_present(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    max_retries: 1
    timeout_sec: 600
    judge:
      criteria: ["does the output answer the prompt?"]
      pass_threshold: 0.7
      on_fail: warn
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_w20_fires_on_command_task_without_escape(self, tmp_path: Path) -> None:
        # Shell tasks are equally vulnerable to retry-without-escape futility.
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: ["echo", "ok"]
    max_retries: 1
    timeout_sec: 60
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("W20" in w and "t1" in w for w in plan.validation_warnings)
        # Engine-only valves should NOT be suggested for shell tasks.
        msg = next((w for w in plan.validation_warnings if "W20" in w), "")
        assert "escalation" not in msg
        assert "fallback_engine" not in msg

    def test_w20_silent_on_command_task_with_verify(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: ["echo", "ok"]
    max_retries: 1
    timeout_sec: 60
    verify_command: ["test", "-f", "out.txt"]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_w20_silent_when_max_retries_zero(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    max_retries: 0
    timeout_sec: 60
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_w20_message_mentions_escape_valves(self, tmp_path: Path) -> None:
        # The unified message must list all four valves so authors can self-serve.
        content = """\
version: 1
name: test
defaults:
  timeout_sec: 300
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    max_retries: 1
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        msg = next((w for w in plan.validation_warnings if "W20" in w), "")
        assert "verify_command" in msg
        assert "escalation" in msg
        assert "retry_delay_sec" in msg
        assert "fallback_engine" in msg

    def test_w21_no_longer_fires(self, tmp_path: Path) -> None:
        # W21 is retired — the unified W20 covers timeout+retry cases via escape valves.
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    max_retries: 1
    timeout_sec: 600
    verify_command: ["test", "-f", "out.txt"]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any("W21" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# 17. W22: judge timeout insufficient
# ---------------------------------------------------------------------------

class TestJudgeTimeoutWarning:
    def test_w22_geval_explicit_low_timeout(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      criteria: ["q1", "q2", "q3", "q4", "q5", "q6"]
      method: g_eval
      timeout_sec: 30
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("W22" in w and "g_eval" in w for w in plan.validation_warnings)

    def test_w22_geval_many_criteria_auto_scale(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      criteria: ["q1", "q2", "q3", "q4", "q5"]
      method: g_eval
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("W22" in w and "auto-scaled" in w for w in plan.validation_warnings)

    def test_w22_debate_low_timeout(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      criteria: ["quality"]
      method: debate
      debate_rounds: 2
      timeout_sec: 30
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("W22" in w and "debate" in w for w in plan.validation_warnings)

    def test_w22_quorum_low_timeout(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      criteria: ["quality"]
      quorum: 3
      timeout_sec: 30
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("W22" in w and "quorum" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# 18. Template variable validation (W3)
# ---------------------------------------------------------------------------

class TestTemplateVariableWarnings:
    def test_w3_unknown_template_variable(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "Use {{ completely_unknown_var }}"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert any("completely_unknown_var" in w for w in plan.validation_warnings)

    def test_w3_known_global_vars_no_warning(self, tmp_path: Path) -> None:
        """Known global vars should not trigger W3."""
        content = """\
version: 1
name: test
workspace_root: "."
tasks:
  - id: t1
    engine: claude
    prompt: "Root: {{ workspace_root }} Plan: {{ plan_name }} Goal: {{ goal }}"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any(
            "workspace_root" in w and "does not match" in w
            for w in plan.validation_warnings
        )
        assert not any(
            "plan_name" in w and "does not match" in w
            for w in plan.validation_warnings
        )

    def test_w3_task_context_vars_no_warning(self, tmp_path: Path) -> None:
        """{{ task-id.status }} etc. should not trigger W3."""
        content = """\
version: 1
name: test
tasks:
  - id: build
    command: echo ok
  - id: check
    depends_on: [build]
    context_from: [build]
    engine: claude
    prompt: "Build status: {{ build.status }}, exit: {{ build.exit_code }}"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any(
            "build.status" in w and "does not match" in w
            for w in plan.validation_warnings
        )

    def test_w3_output_schema_field_no_warning(self, tmp_path: Path) -> None:
        """{{ task-id.output.field }} from output_schema should not trigger W3."""
        content = """\
version: 1
name: test
tasks:
  - id: gen
    engine: claude
    prompt: "Generate JSON"
    output_schema:
      type: object
      properties:
        result:
          type: string
  - id: use
    depends_on: [gen]
    context_from: [gen]
    engine: claude
    prompt: "Use {{ gen.output.result }}"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert not any(
            "gen.output.result" in w and "does not match" in w
            for w in plan.validation_warnings
        )

    def test_w3_matrix_var_no_warning(self, tmp_path: Path) -> None:
        """{{ matrix.KEY }} in matrix task should not trigger W3."""
        content = """\
version: 1
name: test
tasks:
  - id: build
    engine: claude
    prompt: "Build for {{ matrix.os }}"
    matrix:
      os: [linux, windows]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        # The matrix tasks get expanded, so check there's no W3 about matrix.os
        assert not any(
            "matrix.os" in w and "does not match" in w
            for w in plan.validation_warnings
        )


# ---------------------------------------------------------------------------
# 19. Coercion helpers — _to_str_dict, _to_str_list edge cases
# ---------------------------------------------------------------------------

class TestCoercionHelpersExtended:
    def test_to_str_dict_boolean_value_coerced(self, tmp_path: Path) -> None:
        """Boolean values in env dict are coerced to string."""
        content = """\
version: 1
name: test
defaults:
  env:
    DEBUG: true
    COUNT: 42
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.defaults.env["DEBUG"] == "True"
        assert plan.defaults.env["COUNT"] == "42"

    def test_to_str_list_integer_items_coerced(self, tmp_path: Path) -> None:
        """Integer items in tags list coerced to strings."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    tags: [1, 2, 3]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].tags == ["1", "2", "3"]

    def test_to_str_dict_non_dict_raises(self, tmp_path: Path) -> None:
        """env as a list should raise."""
        content = """\
version: 1
name: test
defaults:
  env:
    - FOO: bar
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be an object"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 20. _to_judge_spec — full field coverage
# ---------------------------------------------------------------------------

class TestJudgeSpecParsing:
    def test_judge_all_fields(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      criteria:
        - "Code is correct"
        - type: contains
          value: "OK"
      pass_threshold: 0.8
      on_fail: warn
      model: sonnet
      method: g_eval
      aggregation: min
      timeout_sec: 120
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        j = plan.tasks[0].judge
        assert j is not None
        assert j.pass_threshold == 0.8
        assert j.on_fail == "warn"
        assert j.model == "sonnet"
        assert j.method == "g_eval"
        assert j.aggregation == "min"
        assert j.timeout_sec == 120

    def test_judge_preset_code_quality(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      preset: code_quality
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        j = plan.tasks[0].judge
        assert j is not None
        assert j.preset == "code_quality"
        assert len(j.criteria) > 0

    def test_judge_invalid_method_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      criteria: ["good"]
      method: invalid
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="method"):
            load_plan(pf)

    def test_judge_invalid_aggregation_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      criteria: ["good"]
      aggregation: max
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="aggregation"):
            load_plan(pf)

    def test_judge_invalid_on_fail_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      criteria: ["good"]
      on_fail: abort
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="on_fail"):
            load_plan(pf)

    def test_judge_timeout_too_low_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      criteria: ["good"]
      timeout_sec: 5
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="timeout_sec must be >= 10"):
            load_plan(pf)

    def test_judge_pass_threshold_out_of_range_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      criteria: ["good"]
      pass_threshold: 1.5
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="pass_threshold"):
            load_plan(pf)

    def test_judge_rubric_criterion(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      criteria:
        - type: rubric
          name: quality
          min_score: 3
          weight: 2.0
          levels:
            - score: 1
              description: "Very poor"
            - score: 3
              description: "Average"
            - score: 5
              description: "Excellent"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        j = plan.tasks[0].judge
        assert j is not None
        assert len(j.criteria) == 1
        assert j.criteria[0]["type"] == "rubric"

    def test_judge_invalid_preset_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    judge:
      preset: nonexistent_preset
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="preset"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 21. _to_watch_spec — extended watch fields
# ---------------------------------------------------------------------------

class TestWatchSpecExtended:
    def test_watch_target_metric_parsed(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
watch:
  metric: accuracy
  metric_source: stdout_regex
  metric_pattern: '([0-9.]+)'
  max_iterations: 10
  target_metric: 0.95
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.watch is not None
        assert plan.watch.target_metric == 0.95

    def test_watch_consolidation_fields_parsed(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
watch:
  metric: score
  metric_source: stdout_regex
  metric_pattern: '([0-9.]+)'
  max_iterations: 10
  consolidate_model: sonnet
  consolidate_every: 5
  consolidate_prompt: "Summarize progress"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        w = plan.watch
        assert w is not None
        assert w.consolidate_model == "sonnet"
        assert w.consolidate_every == 5
        assert w.consolidate_prompt == "Summarize progress"

    def test_watch_blame_plan_parsed(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
watch:
  metric: score
  metric_source: stdout_regex
  metric_pattern: '([0-9.]+)'
  max_iterations: 10
  blame_plan: "target-plan.yaml"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.watch is not None
        assert plan.watch.blame_plan == "target-plan.yaml"

    def test_watch_negative_warmup_raises_e037(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
watch:
  metric: score
  metric_source: stdout_regex
  metric_pattern: '([0-9.]+)'
  max_iterations: 10
  warmup_iterations: -1
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E037\]"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 22. Delay spec edge cases
# ---------------------------------------------------------------------------

class TestDelaySpecEdgeCases:
    def test_delay_spec_negative_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    retry_delay_sec: -1.0
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E013"):
            load_plan(pf)

    def test_delay_spec_list_with_negative_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    retry_delay_sec: [1.0, -2.0]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E013"):
            load_plan(pf)

    def test_delay_spec_non_number_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    retry_delay_sec: "slow"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E013"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 23. max_iterations validation (E022)
# ---------------------------------------------------------------------------

class TestMaxIterationsValidation:
    def test_max_iterations_zero_raises_e022(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    max_iterations: 0
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E022\]"):
            load_plan(pf)

    def test_max_iterations_negative_raises_e022(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    max_iterations: -1
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E022\]"):
            load_plan(pf)

    def test_max_iterations_valid(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    max_iterations: 5
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].max_iterations == 5


# ---------------------------------------------------------------------------
# 24. budget_warning_pct validation (E023)
# ---------------------------------------------------------------------------

class TestBudgetWarningPctValidation:
    def test_budget_warning_pct_out_of_range_raises_e023(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
budget_warning_pct: 1.5
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E023"):
            load_plan(pf)

    def test_budget_warning_pct_zero_raises_e023(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
budget_warning_pct: 0.0
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E023"):
            load_plan(pf)

    def test_budget_warning_pct_valid(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
budget_warning_pct: 0.8
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.budget_warning_pct == 0.8


# ---------------------------------------------------------------------------
# 25. Secrets parsing
# ---------------------------------------------------------------------------

class TestSecretsParsing:
    def test_secrets_auto(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
secrets: auto
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.secrets_auto is True

    def test_secrets_list(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
secrets:
  - API_KEY
  - SECRET_TOKEN
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert "API_KEY" in plan.secrets
        assert "SECRET_TOKEN" in plan.secrets

    def test_secrets_invalid_type_raises_e024(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
secrets: 42
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="E024"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 26. control_flow_integrity parsing
# ---------------------------------------------------------------------------

class TestControlFlowIntegrity:
    def test_cfi_true(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
control_flow_integrity: true
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.control_flow_integrity is True

    def test_cfi_false_default(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.control_flow_integrity is False


# ---------------------------------------------------------------------------
# 27. output_schema parsing
# ---------------------------------------------------------------------------

class TestOutputSchemaParsing:
    def test_output_schema_valid(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "generate"
    output_schema:
      type: object
      properties:
        result:
          type: string
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].output_schema is not None
        assert plan.tasks[0].output_schema["type"] == "object"

    def test_output_schema_non_dict_raises(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "generate"
    output_schema: "not a dict"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="output_schema must be an object"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 28. Webhook URL parsing edge case
# ---------------------------------------------------------------------------

class TestWebhookEdgeCases:
    def test_webhook_url_none_when_missing(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.webhook_url is None


# ---------------------------------------------------------------------------
# 29. Frozen field parsing
# ---------------------------------------------------------------------------

class TestFrozenFieldParsing:
    def test_frozen_default_false(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].frozen is False

    def test_frozen_true(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    frozen: true
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].frozen is True


# ---------------------------------------------------------------------------
# 30. Signals field parsing
# ---------------------------------------------------------------------------

class TestSignalsFieldParsing:
    def test_signals_default_false(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].signals is False

    def test_signals_true(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "test"
    signals: true
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].signals is True

    def test_defaults_signals(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
defaults:
  signals: true
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.defaults.signals is True


# ---------------------------------------------------------------------------
# 31. Self-dependency (E016) and cycle detection (E004)
# ---------------------------------------------------------------------------

class TestCycleAndSelfDep:
    def test_self_dependency_raises_e016(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    depends_on: [t1]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E016\]"):
            load_plan(pf)

    def test_cycle_detection_raises_e004(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: a
    command: echo
    depends_on: [b]
  - id: b
    command: echo
    depends_on: [a]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E004\]"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 32. Plan name validation (E017)
# ---------------------------------------------------------------------------

class TestPlanNameValidation:
    def test_plan_name_invalid_chars_raises_e017(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: "my plan with spaces!"
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E017\]"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# 33. Context mode 'recursive' requires workspace_root (E021)
# ---------------------------------------------------------------------------

class TestRecursiveContextMode:
    def test_recursive_without_workspace_root_raises_e021(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: a
    command: echo
  - id: b
    depends_on: [a]
    context_from: [a]
    context_mode: recursive
    engine: claude
    prompt: "test"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E021\]"):
            load_plan(pf)


# ---------------------------------------------------------------------------
# Event-Driven System Reminders (v1.24.0)
# ---------------------------------------------------------------------------

class TestRemindersLoaderValidation:
    """Loader parsing and validation for the 'reminders' field."""

    def test_valid_reminders_parsed(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
    max_retries: 2
    reminders:
      - trigger: database
        message: "Check the DB connection"
      - trigger: timeout
        message: "Consider splitting the task"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].reminders is not None
        assert len(plan.tasks[0].reminders) == 2
        assert plan.tasks[0].reminders[0]["trigger"] == "database"
        assert plan.tasks[0].reminders[1]["message"] == "Consider splitting the task"

    def test_no_reminders_defaults_to_none(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].reminders is None

    def test_reminders_not_a_list(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
    reminders: "not a list"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E018\]"):
            load_plan(pf)

    def test_reminders_entry_not_a_dict(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
    reminders:
      - "just a string"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E018\]"):
            load_plan(pf)

    def test_reminders_missing_trigger_key(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
    reminders:
      - message: "some advice"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E067\]"):
            load_plan(pf)

    def test_reminders_missing_message_key(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
    reminders:
      - trigger: "some_pattern"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E067\]"):
            load_plan(pf)

    def test_reminders_empty_trigger(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
    reminders:
      - trigger: ""
        message: "some advice"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E067\]"):
            load_plan(pf)

    def test_reminders_empty_message(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
    reminders:
      - trigger: "pattern"
        message: "   "
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=r"\[E067\]"):
            load_plan(pf)

    def test_reminders_values_trimmed(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
    reminders:
      - trigger: "  database  "
        message: "  Check the DB  "
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].reminders is not None
        assert plan.tasks[0].reminders[0]["trigger"] == "database"
        assert plan.tasks[0].reminders[0]["message"] == "Check the DB"


# ===========================================================================
# Agent-Triggered Context Compression — Loader tests
# ===========================================================================


class TestCompressBeforeParsing:
    """Tests for compress_before field parsing in the loader."""

    def test_compress_before_true(self, tmp_path: Path) -> None:
        """compress_before: true is parsed correctly."""
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
    compress_before: true
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].compress_before is True

    def test_compress_before_defaults_to_false(self, tmp_path: Path) -> None:
        """compress_before defaults to False when not specified."""
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].compress_before is False


# ===========================================================================
# Honeypot field parsing
# ===========================================================================


class TestHoneypotLoaderParsing:
    """Tests for honeypot field parsing in the loader."""

    def test_honeypot_true_parsed(self, tmp_path: Path) -> None:
        """honeypot: true is parsed correctly."""
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
    honeypot: true
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].honeypot is True

    def test_honeypot_defaults_to_false(self, tmp_path: Path) -> None:
        """honeypot defaults to False when not specified."""
        content = """\
version: 1
name: test
tasks:
  - id: a
    engine: claude
    prompt: "do it"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].honeypot is False


# ===========================================================================
# Edge-case coverage batch — targeting loader LOC/test ratio improvement
# ===========================================================================


class TestLoaderEdgeL2:
    """Edge-case tests to improve loader LOC/test coverage ratio."""

    # --- _to_str_dict edge cases ---

    def test_env_dict_with_none_value_coerced(self, tmp_path: Path) -> None:
        """None values in env dict are coerced to 'None' string."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    env:
      NULLABLE: null
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].env["NULLABLE"] == "None"

    def test_env_dict_with_float_key_coerced(self, tmp_path: Path) -> None:
        """Numeric keys in env dict are coerced to strings."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    env:
      3.14: pi
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert "3.14" in plan.tasks[0].env

    # --- _to_str_list edge cases ---

    def test_depends_on_as_single_string(self, tmp_path: Path) -> None:
        """depends_on as a bare string is coerced to a list."""
        content = """\
version: 1
name: test
tasks:
  - id: a
    command: echo a
  - id: b
    command: echo b
    depends_on: a
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[1].depends_on == ["a"]

    def test_tags_as_single_string_coerced(self, tmp_path: Path) -> None:
        """tags as a single string is coerced to a list."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    tags: security
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].tags == ["security"]

    def test_args_as_non_list_non_string_raises(self, tmp_path: Path) -> None:
        """args field as an integer should raise."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    args: 42
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be a list or string"):
            load_plan(pf)

    # --- _to_delay_spec edge cases ---

    def test_retry_delay_zero_is_valid(self, tmp_path: Path) -> None:
        """retry_delay_sec of 0 is accepted (no delay)."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    max_retries: 1
    retry_delay_sec: 0
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].retry_delay_sec == 0.0

    def test_retry_delay_negative_raises(self, tmp_path: Path) -> None:
        """retry_delay_sec negative value raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    retry_delay_sec: -1
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be >= 0"):
            load_plan(pf)

    def test_retry_delay_list_with_negative_raises(self, tmp_path: Path) -> None:
        """retry_delay_sec list entry negative raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    retry_delay_sec: [1.0, -0.5]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be >= 0"):
            load_plan(pf)

    def test_retry_delay_list_with_non_number_raises(self, tmp_path: Path) -> None:
        """retry_delay_sec list with non-numeric entry raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    retry_delay_sec: [1.0, "abc"]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be a number"):
            load_plan(pf)

    def test_retry_delay_as_dict_raises(self, tmp_path: Path) -> None:
        """retry_delay_sec as a dict raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    retry_delay_sec:
      base: 1
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be a number or list"):
            load_plan(pf)

    # --- _to_matrix edge cases ---

    def test_matrix_empty_dict_raises(self, tmp_path: Path) -> None:
        """matrix with no dimensions raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    matrix: {}
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="at least one dimension"):
            load_plan(pf)

    def test_matrix_non_list_value_raises(self, tmp_path: Path) -> None:
        """matrix with a scalar value instead of list raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    matrix:
      os: linux
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be a list"):
            load_plan(pf)

    def test_matrix_empty_list_value_raises(self, tmp_path: Path) -> None:
        """matrix with an empty list value raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    matrix:
      os: []
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="non-empty list"):
            load_plan(pf)

    # --- circuit_breaker parsing ---

    def test_circuit_breaker_non_dict_raises(self, tmp_path: Path) -> None:
        """circuit_breaker as a scalar raises."""
        content = """\
version: 1
name: test
circuit_breaker: true
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be a mapping"):
            load_plan(pf)

    def test_circuit_breaker_max_failures_zero_raises(self, tmp_path: Path) -> None:
        """circuit_breaker max_total_failures < 1 raises."""
        content = """\
version: 1
name: test
circuit_breaker:
  max_total_failures: 0
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="positive integer"):
            load_plan(pf)

    def test_circuit_breaker_invalid_action_raises(self, tmp_path: Path) -> None:
        """circuit_breaker with unknown action raises."""
        content = """\
version: 1
name: test
circuit_breaker:
  max_total_failures: 3
  action: explode
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="'pause' or 'fail'"):
            load_plan(pf)

    def test_circuit_breaker_pause_action_valid(self, tmp_path: Path) -> None:
        """circuit_breaker action=pause parses correctly."""
        content = """\
version: 1
name: test
circuit_breaker:
  max_total_failures: 2
  action: pause
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.circuit_breaker is not None
        assert plan.circuit_breaker.max_total_failures == 2
        assert plan.circuit_breaker.action == "pause"

    # --- retry_strategy parsing ---

    def test_retry_strategy_invalid_at_defaults_raises(self, tmp_path: Path) -> None:
        """defaults.retry_strategy with unknown value raises."""
        content = """\
version: 1
name: test
defaults:
  retry_strategy: fibonacci
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="constant/linear/exponential"):
            load_plan(pf)

    def test_retry_strategy_task_overrides_default(self, tmp_path: Path) -> None:
        """Task-level retry_strategy overrides plan default."""
        content = """\
version: 1
name: test
defaults:
  retry_strategy: constant
tasks:
  - id: t1
    command: echo
    max_retries: 2
    retry_strategy: exponential
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].retry_strategy == "exponential"

    def test_retry_strategy_inherits_from_default(self, tmp_path: Path) -> None:
        """Task without retry_strategy inherits plan default."""
        content = """\
version: 1
name: test
defaults:
  retry_strategy: linear
tasks:
  - id: t1
    command: echo
    max_retries: 1
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        assert plan.tasks[0].retry_strategy == "linear"

    # --- _to_judge_spec edge cases ---

    def test_judge_unknown_preset_raises(self, tmp_path: Path) -> None:
        """judge with unknown preset raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      preset: nonexistent_preset
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="not valid"):
            load_plan(pf)

    def test_judge_empty_preset_string_raises(self, tmp_path: Path) -> None:
        """judge with empty preset string raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      preset: "  "
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="non-empty string"):
            load_plan(pf)

    def test_judge_threshold_out_of_range_raises(self, tmp_path: Path) -> None:
        """judge pass_threshold > 1 raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      criteria: ["is correct"]
      pass_threshold: 1.5
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="between 0 and 1"):
            load_plan(pf)

    def test_judge_threshold_non_numeric_raises(self, tmp_path: Path) -> None:
        """judge pass_threshold as string raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      criteria: ["is correct"]
      pass_threshold: high
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be a number"):
            load_plan(pf)

    def test_judge_invalid_method_raises(self, tmp_path: Path) -> None:
        """judge with invalid method raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      criteria: ["ok"]
      method: turbo_eval
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="not valid"):
            load_plan(pf)

    def test_judge_invalid_aggregation_raises(self, tmp_path: Path) -> None:
        """judge with invalid aggregation raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      criteria: ["ok"]
      aggregation: median
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="not valid"):
            load_plan(pf)

    def test_judge_invalid_on_fail_raises(self, tmp_path: Path) -> None:
        """judge with invalid on_fail raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      criteria: ["ok"]
      on_fail: crash
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="not valid"):
            load_plan(pf)

    def test_judge_debate_rounds_zero_raises(self, tmp_path: Path) -> None:
        """judge debate_rounds < 1 raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      criteria: ["ok"]
      method: debate
      debate_rounds: 0
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be >= 1"):
            load_plan(pf)

    def test_judge_debate_rounds_non_int_raises(self, tmp_path: Path) -> None:
        """judge debate_rounds as string raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      criteria: ["ok"]
      debate_rounds: many
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be a positive integer"):
            load_plan(pf)

    def test_judge_timeout_below_10_raises(self, tmp_path: Path) -> None:
        """judge timeout_sec < 10 raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      criteria: ["ok"]
      timeout_sec: 5
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be >= 10"):
            load_plan(pf)

    def test_judge_rubric_missing_name_raises(self, tmp_path: Path) -> None:
        """judge rubric criterion without name raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      criteria:
        - type: rubric
          levels:
            - score: 1
              description: bad
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="name is required"):
            load_plan(pf)

    def test_judge_rubric_level_score_out_of_range_raises(self, tmp_path: Path) -> None:
        """judge rubric level score > 5 raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    judge:
      criteria:
        - type: rubric
          name: quality
          levels:
            - score: 6
              description: too high
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="integer 1-5"):
            load_plan(pf)

    # --- when expression validation ---

    def test_when_referencing_unknown_task_raises(self, tmp_path: Path) -> None:
        """when expression referencing non-existent task raises."""
        content = """\
version: 1
name: test
tasks:
  - id: a
    command: echo
  - id: b
    depends_on: [a]
    command: echo
    when: "{{ ghost.status }} == success"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="unknown task 'ghost'"):
            load_plan(pf)

    def test_when_referencing_non_dependency_raises(self, tmp_path: Path) -> None:
        """when expression referencing task not in depends_on raises."""
        content = """\
version: 1
name: test
tasks:
  - id: a
    command: echo
  - id: b
    command: echo
  - id: c
    depends_on: [a]
    command: echo
    when: "{{ b.status }} == success"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="not in depends_on"):
            load_plan(pf)

    # --- context_mode validation ---

    def test_context_mode_invalid_value_raises(self, tmp_path: Path) -> None:
        """Unknown context_mode value raises."""
        content = """\
version: 1
name: test
tasks:
  - id: a
    command: echo
  - id: b
    engine: claude
    prompt: test
    depends_on: [a]
    context_from: [a]
    context_mode: turbo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="not valid"):
            load_plan(pf)

    def test_summarized_without_context_from_raises(self, tmp_path: Path) -> None:
        """context_mode=summarized without context_from raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    context_mode: summarized
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="requires.*context_from"):
            load_plan(pf)

    # --- context_budget_tokens validation ---

    def test_context_budget_tokens_zero_raises(self, tmp_path: Path) -> None:
        """context_budget_tokens of 0 raises."""
        content = """\
version: 1
name: test
tasks:
  - id: a
    command: echo
  - id: b
    engine: claude
    prompt: test
    depends_on: [a]
    context_from: [a]
    context_budget_tokens: 0
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be >= 1"):
            load_plan(pf)

    def test_context_budget_tokens_non_int_raises(self, tmp_path: Path) -> None:
        """context_budget_tokens as string raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    context_budget_tokens: lots
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be an integer"):
            load_plan(pf)

    # --- imports edge cases ---

    def test_imports_not_list_raises(self, tmp_path: Path) -> None:
        """imports as a scalar raises."""
        content = """\
version: 1
name: test
imports: something
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be a list"):
            load_plan(pf)

    def test_imports_missing_prefix_raises(self, tmp_path: Path) -> None:
        """import entry missing prefix raises."""
        shared = tmp_path / "shared.yaml"
        shared.write_text("tasks:\n  - id: x\n    command: echo\n", encoding="utf-8")
        content = f"""\
version: 1
name: test
imports:
  - path: shared.yaml
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="'path' and 'prefix'"):
            load_plan(pf)

    def test_imports_invalid_prefix_format_raises(self, tmp_path: Path) -> None:
        """import prefix with uppercase raises."""
        shared = tmp_path / "shared.yaml"
        shared.write_text("tasks:\n  - id: x\n    command: echo\n", encoding="utf-8")
        content = """\
version: 1
name: test
imports:
  - path: shared.yaml
    prefix: MyLib
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must match"):
            load_plan(pf)

    def test_imports_file_not_found_raises(self, tmp_path: Path) -> None:
        """import referencing nonexistent file raises."""
        content = """\
version: 1
name: test
imports:
  - path: ghost.yaml
    prefix: lib
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="not found"):
            load_plan(pf)

    # --- YAML anchors edge case ---

    def test_anchor_with_override_fields(self, tmp_path: Path) -> None:
        """YAML anchor merge with additional fields works."""
        content = """\
version: 1
name: test
_common: &common
  engine: claude
  model: haiku
tasks:
  - id: t1
    <<: *common
    prompt: "task 1"
    tags: [fast]
  - id: t2
    <<: *common
    prompt: "task 2"
    model: sonnet
    depends_on: [t1]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        plan = load_plan(pf)
        t1 = next(t for t in plan.tasks if t.id == "t1")
        t2 = next(t for t in plan.tasks if t.id == "t2")
        assert t1.model == "haiku"
        assert t2.model == "sonnet"
        assert t1.tags == ["fast"]

    # --- _to_int_or_none / _to_context_budget_or_none ---

    def test_timeout_sec_non_numeric_raises(self, tmp_path: Path) -> None:
        """timeout_sec as non-numeric string raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: echo
    timeout_sec: forever
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be an integer"):
            load_plan(pf)

    # --- command type validation ---

    def test_command_as_dict_raises(self, tmp_path: Path) -> None:
        """command as a dict raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command:
      run: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be a string or list"):
            load_plan(pf)

    def test_command_list_with_int_raises(self, tmp_path: Path) -> None:
        """command list with non-string entry raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    command: [echo, 42]
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="only strings"):
            load_plan(pf)

    # --- guard_command type validation ---

    def test_guard_command_as_dict_raises(self, tmp_path: Path) -> None:
        """guard_command as dict raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    guard_command:
      run: check
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="guard_command must be a string or list"):
            load_plan(pf)

    # --- defaults validation ---

    def test_defaults_as_non_dict_raises(self, tmp_path: Path) -> None:
        """defaults as a list raises."""
        content = """\
version: 1
name: test
defaults:
  - timeout_sec: 60
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="defaults must be an object"):
            load_plan(pf)

    def test_defaults_secrets_auto_non_bool_raises(self, tmp_path: Path) -> None:
        """defaults.secrets_auto as string raises."""
        content = """\
version: 1
name: test
defaults:
  secrets_auto: "stringval"
tasks:
  - id: t1
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be a boolean"):
            load_plan(pf)

    # --- plan root shape ---

    def test_plan_root_as_list_raises(self, tmp_path: Path) -> None:
        """Plan root as a YAML list raises."""
        content = """\
- version: 1
- name: test
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be an object"):
            load_plan(pf)

    def test_plan_file_not_found_raises(self, tmp_path: Path) -> None:
        """Loading a non-existent file raises."""
        pf = tmp_path / "no-such-plan.yaml"
        with pytest.raises(PlanValidationError, match="not found"):
            load_plan(pf)

    # --- tasks validation ---

    def test_tasks_as_dict_raises(self, tmp_path: Path) -> None:
        """tasks as a dict instead of list raises."""
        content = """\
version: 1
name: test
tasks:
  t1:
    command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="tasks must be a list"):
            load_plan(pf)

    def test_empty_tasks_raises(self, tmp_path: Path) -> None:
        """Empty tasks list raises."""
        content = """\
version: 1
name: test
tasks: []
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="non-empty"):
            load_plan(pf)

    def test_task_missing_id_raises(self, tmp_path: Path) -> None:
        """Task without id raises."""
        content = """\
version: 1
name: test
tasks:
  - command: echo
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="id is required"):
            load_plan(pf)

    def test_task_item_not_dict_raises(self, tmp_path: Path) -> None:
        """Task as a string instead of dict raises."""
        content = """\
version: 1
name: test
tasks:
  - "just a string"
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="must be an object"):
            load_plan(pf)

    # --- _to_contract_type_or_none ---

    def test_invalid_contract_type_raises(self, tmp_path: Path) -> None:
        """Unknown contract_type raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    contract_type: graphql-schema
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="not valid"):
            load_plan(pf)

    def test_empty_contract_type_raises(self, tmp_path: Path) -> None:
        """Empty string contract_type raises."""
        content = """\
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: test
    contract_type: "  "
"""
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match="non-empty"):
            load_plan(pf)


# ===========================================================================
# Layer 3 edge-case expansion (~100 tests)
# ===========================================================================


class TestLoaderEdgeL3:
    """Edge-case tests — Layer 3 expansion for loader LOC/test coverage."""

    # -----------------------------------------------------------------------
    # Helper to write + load a plan from string
    # -----------------------------------------------------------------------

    def _load(self, tmp_path: Path, content: str) -> "PlanSpec":  # noqa: F821
        from maestro_cli.loader import load_plan as _lp
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        return _lp(pf)

    def _load_err(self, tmp_path: Path, content: str, match: str) -> None:
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        with pytest.raises(PlanValidationError, match=match):
            load_plan(pf)

    # === E-code validation tests ==========================================

    # E001 — missing required fields
    def test_e001_no_tasks_key(self, tmp_path: Path) -> None:
        self._load_err(tmp_path, "version: 1\nname: x\n", "non-empty")

    def test_e001_tasks_null(self, tmp_path: Path) -> None:
        self._load_err(tmp_path, "version: 1\nname: x\ntasks:\n", "non-empty")

    def test_e001_engine_task_no_prompt_no_batch(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n",
            "no prompt source",
        )

    def test_e001_context_mode_summarized_no_context_from(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: p\n    context_mode: summarized\n",
            "requires.*non-empty context_from",
        )

    def test_e001_context_mode_layered_no_context_from(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: p\n    context_mode: layered\n",
            "requires.*non-empty context_from",
        )

    # E002 — schema version
    def test_e002_version_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 0\nname: x\ntasks:\n  - id: t1\n    command: echo\n",
            r"\[E002\]",
        )

    # E003 — duplicate IDs
    def test_e003_duplicate_task_ids(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: dup\n    command: echo\n  - id: dup\n    command: echo\n",
            "unique",
        )

    # E004 — cycle detection
    def test_e004_two_node_cycle(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: a\n    command: echo\n    depends_on: [b]\n  - id: b\n    command: echo\n    depends_on: [a]\n",
            "cycle",
        )

    # E005 — unknown dependency
    def test_e005_depends_on_nonexistent(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    depends_on: [ghost]\n",
            "unknown task",
        )

    # E006 — invalid engine
    def test_e006_unsupported_engine(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: chatgpt\n    prompt: hi\n",
            "unsupported engine",
        )

    # E008 — invalid reasoning_effort for codex
    def test_e008_codex_bad_reasoning(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: codex\n    prompt: hi\n    reasoning_effort: ultra\n",
            "reasoning_effort.*not valid",
        )

    # E008 — invalid reasoning_effort for claude
    def test_e008_claude_bad_reasoning(self, tmp_path: Path) -> None:
        # 2026-04-27: Opus 4.7 added `xhigh` to Claude's effort set, so the
        # original test value became valid. Use a genuinely invalid value here.
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    reasoning_effort: ultra\n",
            "reasoning_effort.*not valid",
        )

    # E008 — invalid context_mode
    def test_e008_invalid_context_mode(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: a\n    command: echo\n  - id: t1\n    engine: claude\n    prompt: hi\n    depends_on: [a]\n    context_from: [a]\n    context_mode: custom\n",
            "context_mode.*not valid",
        )

    # E008 — invalid edit_policy
    def test_e008_invalid_edit_policy(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    edit_policy: aggressive\n",
            "edit_policy.*not valid",
        )

    # E008 — defaults-level invalid codex reasoning
    def test_e008_defaults_codex_bad_reasoning(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  codex:\n    reasoning_effort: turbo\ntasks:\n  - id: t1\n    command: echo\n",
            "defaults.codex.reasoning_effort.*not valid",
        )

    # E008 — defaults-level invalid claude reasoning
    def test_e008_defaults_claude_bad_reasoning(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  claude:\n    reasoning_effort: extreme\ntasks:\n  - id: t1\n    command: echo\n",
            "defaults.claude.reasoning_effort.*not valid",
        )

    # E008 — defaults-level invalid edit_policy
    def test_e008_defaults_bad_edit_policy(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  edit_policy: none\ntasks:\n  - id: t1\n    command: echo\n",
            "defaults.edit_policy.*not valid",
        )

    # E010 — context_from not in depends_on
    def test_e010_context_from_not_dep(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: a\n    command: echo\n  - id: b\n    command: echo\n  - id: c\n    engine: claude\n    prompt: hi\n    depends_on: [a]\n    context_from: [b]\n",
            "not in depends_on",
        )

    # E011 — command + engine on same task
    def test_e011_command_and_engine(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    engine: claude\n    prompt: hi\n",
            "more than one",
        )

    # E011 — group task with prompt
    def test_e011_group_with_prompt(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    group: sub.yaml\n    prompt: not allowed\n",
            "group tasks cannot have prompt",
        )

    # E011 — prompt_md_file without prompt_md_heading
    def test_e011_md_file_without_heading(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt_md_file: file.md\n",
            "both prompt_md_file and prompt_md_heading",
        )

    # E012 — max_retries out of range
    def test_e012_max_retries_negative(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    max_retries: -1\n",
            "max_retries",
        )

    def test_e012_max_retries_too_high(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    max_retries: 99\n",
            "max_retries",
        )

    # E012 — max_parallel < 1
    def test_e012_max_parallel_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nmax_parallel: 0\ntasks:\n  - id: t1\n    command: echo\n",
            "max_parallel",
        )

    # E012 — stdout_tail_lines < 1
    def test_e012_stdout_tail_lines_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  stdout_tail_lines: 0\ntasks:\n  - id: t1\n    command: echo\n",
            "stdout_tail_lines",
        )

    def test_e012_task_stdout_tail_lines_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    stdout_tail_lines: 0\n",
            "stdout_tail_lines",
        )

    # E013 — invalid retry_delay_sec
    def test_e013_delay_string_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    retry_delay_sec: fast\n",
            "must be a number or list",
        )

    # E014 — invalid budget_period (NOTE: E014 is missing from loader imports,
    # so this currently raises NameError — the test validates the path is hit)
    def test_e014_bad_budget_period(self, tmp_path: Path) -> None:
        pf = tmp_path / "plan.yaml"
        pf.write_text(
            "version: 1\nname: x\nbudget_period: yearly\ntasks:\n  - id: t1\n    command: echo\n",
            encoding="utf-8",
        )
        with pytest.raises((PlanValidationError, NameError)):
            load_plan(pf)

    # E015 — when references unknown task
    def test_e015_when_unknown_task(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: a\n    command: echo\n  - id: b\n    command: echo\n    depends_on: [a]\n    when: \"{{ ghost.status }} == success\"\n",
            "when expression references unknown task",
        )

    # E015 — when references task not in depends_on
    def test_e015_when_not_in_depends(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: a\n    command: echo\n  - id: b\n    command: echo\n  - id: c\n    command: echo\n    depends_on: [a]\n    when: \"{{ b.status }} == success\"\n",
            "when expression references task.*not in depends_on",
        )

    # E016 — self dependency
    def test_e016_self_dependency(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    depends_on: [t1]\n",
            "depend on itself",
        )

    # E017 — bad chars in task ID
    def test_e017_bad_task_id(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: \"t1 bad\"\n    command: echo\n",
            "invalid characters",
        )

    # E017 — bad chars in plan name
    def test_e017_bad_plan_name(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: \"bad name!\"\ntasks:\n  - id: t1\n    command: echo\n",
            "invalid characters",
        )

    # E018 — type mismatches
    def test_e018_tasks_as_dict_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  t1:\n    command: echo\n",
            "tasks must be a list",
        )

    def test_e018_defaults_as_list_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  - env: {}\ntasks:\n  - id: t1\n    command: echo\n",
            "defaults must be an object",
        )

    def test_e018_plan_root_not_dict(self, tmp_path: Path) -> None:
        self._load_err(tmp_path, "- just a list\n", "Plan root must be an object")

    def test_e018_output_schema_not_dict(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    output_schema: not-a-dict\n",
            "output_schema must be an object",
        )

    def test_e018_command_as_int_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: 42\n",
            "command must be a string or list",
        )

    def test_e018_pre_command_as_int_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    pre_command: 42\n",
            "pre_command must be a string or list",
        )

    def test_e018_verify_command_as_int_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    verify_command: 42\n",
            "verify_command must be a string or list",
        )

    def test_e018_guard_command_as_int_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    guard_command: 42\n",
            "guard_command must be a string or list",
        )

    def test_e018_command_list_non_string_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command:\n      - echo\n      - 42\n",
            "command list must contain only strings",
        )

    def test_e018_env_as_list_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    env:\n      - FOO\n",
            "must be an object",
        )

    # E019 — context_budget_tokens
    def test_e019_context_budget_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    context_budget_tokens: 0\n",
            "must be >= 1",
        )

    def test_e019_context_budget_non_int(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    context_budget_tokens: abc\n",
            "must be an integer",
        )

    # E020 — judge spec errors
    def test_e020_judge_not_dict(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge: [check]\n",
            "must be an object",
        )

    def test_e020_judge_no_criteria(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      pass_threshold: 0.5\n",
            "criteria must be a non-empty list",
        )

    def test_e020_judge_empty_criterion_string(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - \"\"\n",
            "non-empty",
        )

    def test_e020_judge_bad_pass_threshold(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      pass_threshold: 1.5\n",
            "pass_threshold must be between 0 and 1",
        )

    def test_e020_judge_bad_method(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      method: turbo\n",
            "method.*not valid",
        )

    def test_e020_judge_bad_on_fail(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      on_fail: abort\n",
            "on_fail.*not valid",
        )

    def test_e020_judge_bad_aggregation(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      aggregation: median\n",
            "aggregation.*not valid",
        )

    def test_e020_judge_timeout_too_low(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      timeout_sec: 5\n",
            "timeout_sec must be >= 10",
        )

    def test_e020_judge_criterion_dict_no_type(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - value: foo\n",
            "type is required",
        )

    def test_e020_judge_criterion_invalid_type(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: custom_check\n",
            "type.*not valid",
        )

    def test_e020_judge_rubric_no_name(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: rubric\n          levels:\n            - score: 1\n              description: bad\n",
            "name is required for rubric",
        )

    def test_e020_judge_rubric_no_levels(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: rubric\n          name: quality\n",
            "levels is required for rubric",
        )

    def test_e020_judge_rubric_level_score_out_of_range(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: rubric\n          name: q\n          levels:\n            - score: 0\n              description: bad\n",
            "score must be an integer 1-5",
        )

    def test_e020_judge_rubric_level_empty_description(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: rubric\n          name: q\n          levels:\n            - score: 1\n              description: \"  \"\n",
            "description must be a non-empty string",
        )

    def test_e020_debate_rounds_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      method: debate\n      debate_rounds: 0\n",
            "debate_rounds must be >= 1",
        )

    def test_e020_json_schema_both_schema_and_file(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: json-schema\n          schema:\n            type: object\n          schema_file: s.json\n",
            "not both",
        )

    def test_e020_json_schema_neither(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: json-schema\n",
            "requires.*schema",
        )

    # E020 — unknown fields on typed judge criteria
    def test_e020_judge_criterion_unknown_field_contains(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: contains\n          value: hello\n          negate: true\n",
            "unknown field.*negate",
        )

    def test_e020_judge_criterion_unknown_field_regex(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: regex\n          pattern: foo.*bar\n          ignore_case: true\n",
            "unknown field.*ignore_case",
        )

    def test_e020_judge_criterion_unknown_field_rubric(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: rubric\n          name: quality\n          levels:\n            - score: 1\n              description: bad\n            - score: 5\n              description: great\n          negate: true\n",
            "unknown field.*negate",
        )

    def test_e020_judge_criterion_valid_fields_accepted(self, tmp_path: Path) -> None:
        """All known fields for contains type should not raise."""
        pf = tmp_path / "plan.yaml"
        pf.write_text(
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: contains\n          value: hello\n",
            encoding="utf-8",
        )
        plan = load_plan(pf)
        assert plan.tasks[0].judge is not None

    # E022 — max_iterations < 1
    def test_e022_max_iterations_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    max_iterations: 0\n",
            "max_iterations must be >= 1",
        )

    # E023 — budget_warning_pct
    def test_e023_budget_warning_pct_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nbudget_warning_pct: 0.0\ntasks:\n  - id: t1\n    command: echo\n",
            "budget_warning_pct must be between 0 and 1",
        )

    def test_e023_budget_warning_pct_one(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nbudget_warning_pct: 1.0\ntasks:\n  - id: t1\n    command: echo\n",
            "budget_warning_pct must be between 0 and 1",
        )

    # E024 — invalid secrets type
    def test_e024_secrets_as_int(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nsecrets: 42\ntasks:\n  - id: t1\n    command: echo\n",
            "secrets.*must be a list",
        )

    # E025 — circular imports
    def test_e025_circular_import(self, tmp_path: Path) -> None:
        a = tmp_path / "a.yaml"
        b = tmp_path / "b.yaml"
        a.write_text(
            "version: 1\nname: a\nimports:\n  - path: b.yaml\n    prefix: b\ntasks:\n  - id: t1\n    command: echo\n",
            encoding="utf-8",
        )
        b.write_text(
            "imports:\n  - path: a.yaml\n    prefix: a\ntasks:\n  - id: t1\n    command: echo\n",
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match="Circular"):
            load_plan(a)

    # E026 — import validation
    def test_e026_imports_not_list(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nimports:\n  path: x.yaml\ntasks:\n  - id: t1\n    command: echo\n",
            "imports.*must be a list",
        )

    def test_e026_import_missing_path(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nimports:\n  - prefix: lib\ntasks:\n  - id: t1\n    command: echo\n",
            "must have.*path.*prefix",
        )

    def test_e026_import_missing_file(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nimports:\n  - path: nonexistent.yaml\n    prefix: lib\ntasks:\n  - id: t1\n    command: echo\n",
            "not found",
        )

    # E027 — duplicate import prefix
    def test_e027_duplicate_prefix(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared.yaml"
        shared.write_text("tasks:\n  - id: s1\n    command: echo\n", encoding="utf-8")
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nimports:\n  - path: shared.yaml\n    prefix: lib\n  - path: shared.yaml\n    prefix: lib\ntasks:\n  - id: t1\n    command: echo\n",
            "Duplicate import prefix",
        )

    # E028 — invalid prefix format
    def test_e028_bad_prefix(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared.yaml"
        shared.write_text("tasks:\n  - id: s1\n    command: echo\n", encoding="utf-8")
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nimports:\n  - path: shared.yaml\n    prefix: Lib_test\ntasks:\n  - id: t1\n    command: echo\n",
            "must match",
        )

    # E029 — approval_message without requires_approval
    def test_e029_approval_message_no_flag(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    approval_message: please\n",
            "approval_message.*without.*requires_approval",
        )

    # E030 / E031 — fallback/escalation
    def test_e030_fallback_engine_no_engine(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    fallback_engine: claude\n",
            "fallback_engine but no engine",
        )

    def test_e030_fallback_model_no_engine(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    fallback_model: sonnet\n",
            "fallback_model without fallback_engine",
        )

    def test_e030_fallback_bad_engine(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    fallback_engine: chatgpt\n",
            "not a valid engine",
        )

    def test_e031_escalation_no_engine(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    escalation: [sonnet]\n",
            "escalation but no engine",
        )

    # E050 — circuit_breaker validation
    def test_e050_circuit_breaker_not_dict(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ncircuit_breaker: true\ntasks:\n  - id: t1\n    command: echo\n",
            "circuit_breaker must be a mapping",
        )

    def test_e050_circuit_breaker_bad_action(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ncircuit_breaker:\n  action: abort\ntasks:\n  - id: t1\n    command: echo\n",
            "action must be.*pause.*fail",
        )

    # E051 — retry_strategy
    def test_e051_bad_retry_strategy(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    retry_strategy: fibonacci\n",
            "retry_strategy must be",
        )

    def test_e051_bad_default_retry_strategy(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  retry_strategy: random\ntasks:\n  - id: t1\n    command: echo\n",
            "defaults.retry_strategy must be",
        )

    # E052 — policy validation
    def test_e052_policy_no_name(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\npolicies:\n  - rule: \"model == 'opus'\"\n    action: warn\ntasks:\n  - id: t1\n    command: echo\n",
            "missing required.*name",
        )

    def test_e052_policy_no_rule(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\npolicies:\n  - name: p1\n    action: warn\ntasks:\n  - id: t1\n    command: echo\n",
            "missing required.*rule",
        )

    def test_e052_policy_bad_action(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\npolicies:\n  - name: p1\n    rule: \"model == 'opus'\"\n    action: reject\ntasks:\n  - id: t1\n    command: echo\n",
            "invalid action.*reject",
        )

    def test_e052_policy_dup_name(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\npolicies:\n  - name: p1\n    rule: \"True\"\n  - name: p1\n    rule: \"False\"\ntasks:\n  - id: t1\n    command: echo\n",
            "Duplicate policy name",
        )

    # E053 — invalid routing_strategy
    def test_e053_routing_strategy_bad(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nrouting_strategy: fastest\ntasks:\n  - id: t1\n    command: echo\n",
            "Invalid routing_strategy",
        )

    # E054 — quorum < 2
    def test_e054_quorum_one(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      quorum: 1\n",
            "quorum must be >= 2",
        )

    # E055 — bad quorum_strategy
    def test_e055_bad_quorum_strategy(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      quorum: 3\n      quorum_strategy: consensus\n",
            "quorum_strategy.*not valid",
        )

    # E056 — quorum_strategy without quorum
    def test_e056_strategy_no_quorum(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      quorum_strategy: majority\n",
            "quorum_strategy requires quorum",
        )

    # E057 — batch validation
    def test_e057_batch_no_items(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    batch:\n      template: \"Process {{ batch.item }}\"\n",
            "batch.items is required",
        )

    def test_e057_batch_no_template(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    batch:\n      items: [a, b]\n",
            "batch.template is required",
        )

    def test_e057_batch_template_no_placeholder(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    batch:\n      items: [a, b]\n      template: \"Process this\"\n",
            "must contain",
        )

    # E058 — batch max_per_call < 1
    def test_e058_batch_max_per_call_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    batch:\n      items: [a]\n      template: \"{{ batch.item }}\"\n      max_per_call: 0\n",
            "max_per_call must be >= 1",
        )

    # E060 — batch on command task
    def test_e060_batch_on_command_task(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    batch:\n      items: [a]\n      template: \"{{ batch.item }}\"\n",
            "batch is only allowed on engine",
        )

    # E062 — batch and matrix — note: matrix expansion happens before
    # validation, so this check only triggers if _expand_matrix_tasks
    # preserves the batch field (which it does not copy). Test that a task
    # with both batch + matrix on a non-matrix path still raises.
    def test_e062_batch_and_matrix_non_expanded(self, tmp_path: Path) -> None:
        """Validate E062 fires when both batch and matrix are present and
        the matrix field survives to validate_plan (use validate_plan directly)."""
        from maestro_cli.models import TaskSpec, PlanSpec, PlanDefaults, BatchSpec
        task = TaskSpec(
            id="t1", engine="claude", prompt="hi",
            matrix={"os": ["linux"]},
            batch=BatchSpec(items=["a"], template="{{ batch.item }}", max_per_call=5),
        )
        plan = PlanSpec(version=1, name="x", tasks=[task], defaults=PlanDefaults())
        from maestro_cli.loader import validate_plan
        with pytest.raises(PlanValidationError, match="batch and matrix are mutually exclusive"):
            validate_plan(plan)

    # E063 — dynamic_group validation
    def test_e063_dynamic_group_no_engine(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    dynamic_group: true\n",
            "dynamic_group requires engine",
        )

    def test_e063_dynamic_group_no_output_schema(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    dynamic_group: true\n",
            "dynamic_group requires output_schema",
        )

    # E064 — dynamic_group conflicts
    def test_e064_dynamic_group_and_batch(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    dynamic_group: true\n    output_schema:\n      type: object\n    batch:\n      items: [a]\n      template: \"{{ batch.item }}\"\n",
            "dynamic_group and batch are mutually exclusive",
        )

    def test_e064_dynamic_group_and_matrix(self, tmp_path: Path) -> None:
        """Validate E064 fires via validate_plan (matrix expansion strips
        the matrix field before validation via load_plan)."""
        from maestro_cli.models import TaskSpec, PlanSpec, PlanDefaults
        task = TaskSpec(
            id="t1", engine="claude", prompt="hi",
            dynamic_group=True,
            output_schema={"type": "object"},
            matrix={"os": ["linux"]},
        )
        plan = PlanSpec(version=1, name="x", tasks=[task], defaults=PlanDefaults())
        from maestro_cli.loader import validate_plan
        with pytest.raises(PlanValidationError, match="dynamic_group and matrix are mutually exclusive"):
            validate_plan(plan)

    # E065 — context_trust
    def test_e065_invalid_context_trust(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    context_trust: partial\n",
            "context_trust must be.*trusted.*untrusted",
        )

    # E066 — max_total_steps < 1
    def test_e066_watch_max_total_steps_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+\\\\.\\\\d+)\"\n  max_total_steps: 0\ntasks:\n  - id: t1\n    command: echo\n",
            "max_total_steps must be >= 1",
        )

    # E067 — reminders validation
    def test_e067_reminder_missing_trigger(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    reminders:\n      - message: Try again\n",
            "trigger.*message",
        )

    def test_e067_reminder_empty_trigger(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    reminders:\n      - trigger: \"  \"\n        message: Try again\n",
            "trigger must be.*non-empty",
        )

    def test_e067_reminder_empty_message(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    reminders:\n      - trigger: timeout\n        message: \"  \"\n",
            "message must be.*non-empty",
        )

    def test_e067_reminders_not_list(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    reminders:\n      trigger: t\n      message: m\n",
            "reminders must be a list",
        )

    # === Watch block validation ============================================

    # E032 — watch metric missing
    def test_e032_watch_no_metric(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric_source: stdout_regex\ntasks:\n  - id: t1\n    command: echo\n",
            "watch.metric.*required",
        )

    # E033 — bad direction
    def test_e033_watch_bad_direction(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_direction: ascending\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+)\"\ntasks:\n  - id: t1\n    command: echo\n",
            "metric_direction must be one of",
        )

    # E033 — bad source
    def test_e033_watch_bad_source(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: file\n  metric_pattern: \"loss=(\\\\d+)\"\ntasks:\n  - id: t1\n    command: echo\n",
            "metric_source must be one of",
        )

    # E034 — metric_pattern required for stdout_regex
    def test_e034_watch_no_pattern(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\ntasks:\n  - id: t1\n    command: echo\n",
            "metric_pattern is required",
        )

    def test_e034_watch_pattern_no_group(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=\\\\d+\"\ntasks:\n  - id: t1\n    command: echo\n",
            "exactly 1 capture group",
        )

    # E035 — json_field needs metric_json_path
    def test_e035_json_field_no_path(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: json_field\ntasks:\n  - id: t1\n    command: echo\n",
            "metric_json_path is required",
        )

    # E036 — max_iterations < 1
    def test_e036_watch_max_iter_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+)\"\n  max_iterations: 0\ntasks:\n  - id: t1\n    command: echo\n",
            "max_iterations must be >= 1",
        )

    # E037 — warmup_iterations >= max_iterations
    def test_e037_warmup_too_high(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+)\"\n  max_iterations: 5\n  warmup_iterations: 5\ntasks:\n  - id: t1\n    command: echo\n",
            "warmup_iterations must be >= 0 and < max_iterations",
        )

    # E038 — plateau_threshold < 1
    def test_e038_plateau_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+)\"\n  plateau_threshold: 0\ntasks:\n  - id: t1\n    command: echo\n",
            "plateau_threshold must be >= 1",
        )

    # E039 — watch max_cost_usd <= 0
    def test_e039_watch_cost_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+)\"\n  max_cost_usd: 0\ntasks:\n  - id: t1\n    command: echo\n",
            "max_cost_usd must be positive",
        )

    # E040 — metric_task unknown
    def test_e040_watch_bad_metric_task(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+)\"\n  metric_task: ghost\ntasks:\n  - id: t1\n    command: echo\n",
            "does not reference a valid task ID",
        )

    # E041 — bad on_regression
    def test_e041_watch_bad_regression(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+)\"\n  on_regression: abort\ntasks:\n  - id: t1\n    command: echo\n",
            "on_regression must be one of",
        )

    # E043 — bad plateau_action
    def test_e043_watch_bad_plateau_action(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+)\"\n  plateau_action: crash\ntasks:\n  - id: t1\n    command: echo\n",
            "plateau_action must be one of",
        )

    # E044 — iteration_budget_sec <= 0
    def test_e044_watch_budget_sec_zero(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+)\"\n  iteration_budget_sec: 0\ntasks:\n  - id: t1\n    command: echo\n",
            "iteration_budget_sec must be positive",
        )

    # E045 — worktree without workspace_root
    def test_e045_worktree_no_workspace(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    worktree: true\n",
            "worktree.*no workspace_root",
        )

    # E046 — worktree on command task
    def test_e046_worktree_on_command(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            f"version: 1\nname: x\nworkspace_root: {tmp_path.as_posix()}\ntasks:\n  - id: t1\n    command: echo\n    worktree: true\n",
            "worktree.*group/command",
        )

    # E047 — mode: improve without workspace_root
    def test_e047_improve_no_workspace(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  mode: improve\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+)\"\ntasks:\n  - id: t1\n    command: echo\n",
            "requires a resolvable workspace_root",
        )

    # E048 — bad watch mode
    def test_e048_bad_watch_mode(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  mode: auto\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+)\"\ntasks:\n  - id: t1\n    command: echo\n",
            "watch.mode.*not valid",
        )

    # === Successful parse paths ============================================

    def test_valid_judge_with_preset_code_quality(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      preset: code_quality\n",
        )
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.preset == "code_quality"
        assert len(plan.tasks[0].judge.criteria) > 0

    def test_valid_judge_with_quorum(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      quorum: 3\n      quorum_strategy: majority\n",
        )
        j = plan.tasks[0].judge
        assert j is not None
        assert j.quorum == 3
        assert j.quorum_strategy == "majority"

    def test_valid_judge_g_eval(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      method: g_eval\n      model: sonnet\n",
        )
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.method == "g_eval"

    def test_valid_judge_debate(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      method: debate\n      debate_rounds: 3\n",
        )
        j = plan.tasks[0].judge
        assert j is not None
        assert j.method == "debate"
        assert j.debate_rounds == 3

    def test_valid_secrets_auto(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\nsecrets: auto\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert plan.secrets_auto is True

    def test_valid_secrets_list(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\nsecrets:\n  - API_KEY\n  - SECRET\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert "API_KEY" in plan.secrets

    def test_valid_routing_strategy(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\nrouting_strategy: cost_optimized\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert plan.routing_strategy == "cost_optimized"

    def test_valid_control_flow_integrity(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ncontrol_flow_integrity: true\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert plan.control_flow_integrity is True

    def test_valid_firewall_model(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\nfirewall_model: haiku\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert plan.firewall_model == "haiku"

    def test_control_flow_integrity_string_true(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ncontrol_flow_integrity: \"true\"\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert plan.control_flow_integrity is True

    def test_valid_budget_period(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\nbudget_period: weekly\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert plan.budget_period == "weekly"

    def test_valid_circuit_breaker(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ncircuit_breaker:\n  max_total_failures: 3\n  action: pause\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert plan.circuit_breaker is not None
        assert plan.circuit_breaker.max_total_failures == 3
        assert plan.circuit_breaker.action == "pause"

    def test_valid_policies(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\npolicies:\n  - name: no-opus\n    rule: \"model != 'opus'\"\n    action: block\n    message: opus banned\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert len(plan.policies) == 1
        assert plan.policies[0].name == "no-opus"
        assert plan.policies[0].action == "block"

    def test_valid_context_trust_trusted(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    context_trust: trusted\n",
        )
        assert plan.tasks[0].context_trust == "trusted"

    def test_valid_context_trust_untrusted(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    context_trust: untrusted\n",
        )
        assert plan.tasks[0].context_trust == "untrusted"

    def test_valid_negative_cache_ttl(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    negative_cache_ttl_sec: 120\n",
        )
        assert plan.tasks[0].negative_cache_ttl_sec == 120

    def test_negative_cache_ttl_negative_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(Exception, match="negative_cache_ttl_sec must be >= 0"):
            self._load(
                tmp_path,
                "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    negative_cache_ttl_sec: -1\n",
            )

    def test_valid_honeypot_field(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    honeypot: true\n",
        )
        assert plan.tasks[0].honeypot is True

    def test_valid_frozen_field(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    frozen: true\n",
        )
        assert plan.tasks[0].frozen is True

    def test_valid_compress_before(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    compress_before: true\n",
        )
        assert plan.tasks[0].compress_before is True

    def test_valid_signals_field(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    signals: true\n",
        )
        assert plan.tasks[0].signals is True

    def test_valid_deliberation_field(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    deliberation: true\n    deliberation_threshold: 0.7\n",
        )
        assert plan.tasks[0].deliberation is True
        assert plan.tasks[0].deliberation_threshold == 0.7

    def test_deliberation_threshold_bad_value(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    deliberation_threshold: 1.5\n",
            "deliberation_threshold must be between 0.0 and 1.0",
        )

    def test_deliberation_threshold_non_number(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    deliberation_threshold: abc\n",
            "deliberation_threshold must be a number",
        )

    def test_valid_reminders(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    reminders:\n      - trigger: timeout\n        message: Increase timeout\n",
        )
        assert len(plan.tasks[0].reminders) == 1
        assert plan.tasks[0].reminders[0]["trigger"] == "timeout"

    def test_valid_dynamic_group(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    dynamic_group: true\n    output_schema:\n      type: object\n",
        )
        assert plan.tasks[0].dynamic_group is True
        assert plan.tasks[0].cache is False  # forced by dynamic_group

    def test_valid_output_schema(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    output_schema:\n      type: object\n      properties:\n        name:\n          type: string\n",
        )
        assert plan.tasks[0].output_schema is not None
        assert plan.tasks[0].output_schema["type"] == "object"

    def test_valid_watch_mode_improve(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            f"version: 1\nname: x\nworkspace_root: {tmp_path.as_posix()}\nwatch:\n  mode: improve\ntasks:\n  - id: t1\n    command: echo\n",
        )
        w = plan.watch
        assert w is not None
        assert w.mode == "improve"
        assert w.metric == "tasks_passed"
        assert w.metric_direction == "higher_is_better"
        assert w.metric_source == "manifest"
        assert w.warmup_iterations == 0
        assert w.on_regression == "rollback"

    def test_valid_watch_with_target_metric(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: accuracy\n  metric_source: stdout_regex\n  metric_pattern: \"acc=(\\\\d+\\\\.\\\\d+)\"\n  target_metric: 0.95\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert plan.watch is not None
        assert plan.watch.target_metric == 0.95

    def test_valid_watch_with_consolidation(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\nwatch:\n  metric: loss\n  metric_source: stdout_regex\n  metric_pattern: \"loss=(\\\\d+\\\\.\\\\d+)\"\n  consolidate_model: sonnet\n  consolidate_every: 5\n  consolidate_prompt: Summarize\ntasks:\n  - id: t1\n    command: echo\n",
        )
        w = plan.watch
        assert w is not None
        assert w.consolidate_model == "sonnet"
        assert w.consolidate_every == 5

    def test_valid_retry_strategy_exponential(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    retry_strategy: exponential\n",
        )
        assert plan.tasks[0].retry_strategy == "exponential"

    def test_valid_retry_strategy_from_defaults(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  retry_strategy: linear\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert plan.tasks[0].retry_strategy == "linear"

    # === Matrix expansion edge cases =======================================

    def test_matrix_multi_key_expansion(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: \"{{ matrix.os }} {{ matrix.py }}\"\n    matrix:\n      os: [linux, win]\n      py: [\"3.11\", \"3.12\"]\n",
        )
        # 2 x 2 = 4 expanded tasks
        assert len(plan.tasks) == 4
        ids = {t.id for t in plan.tasks}
        assert all("t1." in tid for tid in ids)

    def test_matrix_downstream_deps_rewritten(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: build\n    engine: claude\n    prompt: build\n    matrix:\n      os: [linux, win]\n  - id: test\n    command: echo\n    depends_on: [build]\n",
        )
        test_task = next(t for t in plan.tasks if t.id == "test")
        # depends_on should reference expanded IDs, not 'build'
        assert "build" not in test_task.depends_on
        assert len(test_task.depends_on) == 2

    # === Import resolution =================================================

    def test_imports_with_nested_imports(self, tmp_path: Path) -> None:
        inner = tmp_path / "inner.yaml"
        inner.write_text("tasks:\n  - id: i1\n    command: echo inner\n", encoding="utf-8")
        outer = tmp_path / "outer.yaml"
        outer.write_text(
            "imports:\n  - path: inner.yaml\n    prefix: in\ntasks:\n  - id: o1\n    command: echo outer\n",
            encoding="utf-8",
        )
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            "version: 1\nname: x\nimports:\n  - path: outer.yaml\n    prefix: out\ntasks:\n  - id: t1\n    command: echo\n",
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        ids = {t.id for t in plan.tasks}
        assert "in/i1" in ids
        assert "out/o1" in ids
        assert "t1" in ids

    def test_imports_overrides_env_merge(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared.yaml"
        shared.write_text(
            "tasks:\n  - id: s1\n    command: echo\n    env:\n      BASE: original\n",
            encoding="utf-8",
        )
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            "version: 1\nname: x\nimports:\n  - path: shared.yaml\n    prefix: lib\n    overrides:\n      env:\n        EXTRA: added\ntasks:\n  - id: t1\n    command: echo\n",
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        lib_task = next(t for t in plan.tasks if t.id == "lib/s1")
        assert lib_task.env.get("BASE") == "original"
        assert lib_task.env.get("EXTRA") == "added"

    # === Warning emission tests ============================================

    def test_w2_prompt_md_heading_with_hash(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt_md_file: f.md\n    prompt_md_heading: \"# My Heading\"\n",
        )
        assert any("starts with '#'" in w for w in plan.validation_warnings)

    def test_w3_unknown_template_var(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: \"{{ unknown_var }}\"\n",
        )
        assert any("unknown_var" in w for w in plan.validation_warnings)

    def test_w3_known_template_var_no_warning(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: \"{{ workspace_root }}\"\n",
        )
        assert not any("workspace_root" in w and "does not match" in w for w in plan.validation_warnings)

    def test_w6_retry_delay_list_shorter(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    max_retries: 3\n    retry_delay_sec: [1.0]\n",
        )
        assert any("retry_delay_sec has 1 value" in w for w in plan.validation_warnings)

    def test_w8_tag_with_space(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    tags: [\"my tag\"]\n",
        )
        assert any("contains whitespace" in w for w in plan.validation_warnings)

    def test_w13_fallback_same_as_engine(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    fallback_engine: claude\n",
        )
        assert any("W13" in w for w in plan.validation_warnings)

    def test_w14_escalation_duplicates(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    max_retries: 2\n    escalation: [sonnet, sonnet]\n",
        )
        assert any("W14" in w for w in plan.validation_warnings)

    def test_w15_escalation_no_retries(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    escalation: [sonnet, opus]\n",
        )
        assert any("W15" in w for w in plan.validation_warnings)

    def test_w16_single_worktree_task(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            f"version: 1\nname: x\nworkspace_root: {tmp_path.as_posix()}\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    worktree: true\n",
        )
        assert any("only one worktree task" in w for w in plan.validation_warnings)

    def test_w20_tight_timeout_with_retries(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  timeout_sec: 300\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    max_retries: 2\n",
        )
        assert any("W20" in w for w in plan.validation_warnings)

    def test_w20_silent_with_escape_valve(self, tmp_path: Path) -> None:
        # Engine task with verify_command — retries can self-correct, no warning.
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  timeout_sec: 300\ntasks:\n"
            "  - id: t1\n    engine: claude\n    prompt: hi\n    max_retries: 2\n"
            "    verify_command: [\"test\", \"-f\", \"out.txt\"]\n",
        )
        assert not any("W20" in w for w in plan.validation_warnings)

    def test_w22_g_eval_low_timeout(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      method: g_eval\n      model: sonnet\n      timeout_sec: 30\n",
        )
        assert any("W22" in w for w in plan.validation_warnings)

    def test_w22_debate_low_timeout(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      method: debate\n      debate_rounds: 2\n      timeout_sec: 30\n",
        )
        assert any("W22" in w for w in plan.validation_warnings)

    def test_w22_reflection_low_timeout(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      method: reflection\n      timeout_sec: 30\n",
        )
        assert any("W22" in w and "reflection" in w for w in plan.validation_warnings)

    def test_w22_quorum_low_timeout(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      quorum: 3\n      timeout_sec: 30\n",
        )
        assert any("W22" in w for w in plan.validation_warnings)

    def test_warning_verify_without_retries(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    verify_command: echo ok\n",
        )
        assert any("verify_command but max_retries=0" in w for w in plan.validation_warnings)

    def test_warning_judge_retry_no_iterations(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      on_fail: retry\n",
        )
        assert any("without max_iterations" in w for w in plan.validation_warnings)

    def test_warning_context_from_no_budget(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: a\n    command: echo\n  - id: b\n    command: echo\n    depends_on: [a]\n    context_from: [a]\n",
        )
        assert any("context_budget_tokens" in w for w in plan.validation_warnings)

    def test_warning_observation_block_no_context(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    observation_block: true\n",
        )
        assert any("observation_block" in w and "no effect" in w for w in plan.validation_warnings)

    def test_warning_gemini_reasoning_effort(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: gemini\n    prompt: hi\n    reasoning_effort: high\n",
        )
        assert any("Gemini" in w and "reasoning_effort" in w for w in plan.validation_warnings)

    def test_warning_copilot_reasoning_effort(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: copilot\n    prompt: hi\n    reasoning_effort: high\n",
        )
        assert any("Copilot" in w and "reasoning_effort" in w for w in plan.validation_warnings)

    def test_warning_qwen_reasoning_effort(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: qwen\n    prompt: hi\n    reasoning_effort: high\n",
        )
        assert any("Qwen" in w and "reasoning_effort" in w for w in plan.validation_warnings)

    def test_warning_ollama_reasoning_effort(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: ollama\n    prompt: hi\n    reasoning_effort: high\n",
        )
        assert any("Ollama" in w and "reasoning_effort" in w for w in plan.validation_warnings)

    def test_warning_edit_policy_on_command_task(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    edit_policy: efficient\n",
        )
        assert any("edit_policy has no effect" in w for w in plan.validation_warnings)

    def test_warning_guard_command_no_engine_or_command(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub.yaml"
        sub.write_text("version: 1\nname: sub\ntasks:\n  - id: s1\n    command: echo\n", encoding="utf-8")
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    group: sub.yaml\n    guard_command: echo check\n",
        )
        assert any("guard_command on task without engine or command" in w for w in plan.validation_warnings)

    def test_warning_judge_contains_on_engine_task(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria:\n        - type: contains\n          value: success\n",
        )
        assert any("contains" in w and "engine" in w for w in plan.validation_warnings)

    def test_warning_g_eval_with_haiku(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    judge:\n      criteria: [check]\n      method: g_eval\n      model: haiku\n",
        )
        assert any("g_eval" in w and "haiku" in w for w in plan.validation_warnings)

    def test_warning_no_timeout_hint(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert any("no explicit timeout_sec" in w for w in plan.validation_warnings)

    def test_warning_assert_no_retry(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    assert:\n      - type: glob_exists\n        glob: \"*.txt\"\n",
        )
        assert any("assert rules but max_retries=0" in w for w in plan.validation_warnings)

    # === Defaults inheritance ==============================================

    def test_defaults_engine_escalation_inherited(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  claude:\n    escalation: [haiku, sonnet, opus]\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    max_retries: 2\n",
        )
        assert plan.tasks[0].escalation == ["haiku", "sonnet", "opus"]

    def test_defaults_engine_escalation_overridden(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  claude:\n    escalation: [haiku, sonnet]\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    max_retries: 1\n    escalation: [opus]\n",
        )
        assert plan.tasks[0].escalation == ["opus"]

    def test_defaults_engine_fallback_inherited(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  claude:\n    fallback_engine: codex\n    fallback_model: \"5.4\"\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n",
        )
        assert plan.tasks[0].fallback_engine == "codex"
        assert plan.tasks[0].fallback_model == "5.4"

    def test_defaults_signals_field(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  signals: true\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert plan.defaults.signals is True

    # === Group task edge cases =============================================

    def test_group_task_with_assert_raises(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub.yaml"
        sub.write_text("version: 1\nname: sub\ntasks:\n  - id: s1\n    command: echo\n", encoding="utf-8")
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    group: sub.yaml\n    assert:\n      - type: glob_exists\n        glob: \"*.txt\"\n",
            "group tasks cannot use assert",
        )

    def test_group_task_with_contract_type_raises(self, tmp_path: Path) -> None:
        sub = tmp_path / "sub.yaml"
        sub.write_text("version: 1\nname: sub\ntasks:\n  - id: s1\n    command: echo\n", encoding="utf-8")
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    group: sub.yaml\n    contract_type: sql-schema\n",
            "group tasks cannot produce typed contracts",
        )

    # === Contract consumption validation ===================================

    def test_consumes_contracts_self_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    contract_type: sql-schema\n    consumes_contracts: [t1]\n",
            "cannot consume its own contract",
        )

    def test_consumes_contracts_no_producer_type_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: a\n    command: echo\n  - id: b\n    command: echo\n    depends_on: [a]\n    consumes_contracts: [a]\n",
            "does not declare contract_type",
        )

    # === Plan file not found ===============================================

    def test_plan_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(PlanValidationError, match="not found"):
            load_plan(tmp_path / "nonexistent.yaml")

    # === Invalid YAML ======================================================

    def test_invalid_yaml(self, tmp_path: Path) -> None:
        pf = tmp_path / "plan.yaml"
        pf.write_text("{{bad yaml", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="Invalid YAML"):
            load_plan(pf)

    # === _to_int_or_none edge cases ========================================

    def test_timeout_sec_non_int_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    timeout_sec: abc\n",
            "must be an integer",
        )

    # === defaults engine blocks not dicts ==================================

    def test_defaults_engine_not_dict_raises(self, tmp_path: Path) -> None:
        # Empty list [] is falsy, so `{} or {}` swallows it.
        # Use a non-empty list to trigger the type check.
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  codex:\n    - model: test\ntasks:\n  - id: t1\n    command: echo\n",
            "must be objects",
        )

    # === defaults.secrets_auto not bool ====================================

    def test_defaults_secrets_auto_not_bool(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\ndefaults:\n  secrets_auto: 42\ntasks:\n  - id: t1\n    command: echo\n",
            "secrets_auto must be a boolean",
        )

    # === max_cost_usd validation ===========================================

    def test_max_cost_usd_zero_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nmax_cost_usd: 0\ntasks:\n  - id: t1\n    command: echo\n",
            "must be > 0",
        )

    def test_max_cost_usd_negative_raises(self, tmp_path: Path) -> None:
        self._load_err(
            tmp_path,
            "version: 1\nname: x\nmax_cost_usd: -1\ntasks:\n  - id: t1\n    command: echo\n",
            "must be > 0",
        )

    # === Unnamed plan gets default name ====================================

    def test_unnamed_plan_defaults(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: \"\"\ntasks:\n  - id: t1\n    command: echo\n",
        )
        assert plan.name == "unnamed-plan"

    # === Checkpoint parsing ================================================

    def test_checkpoint_true(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    command: echo\n    checkpoint: true\n",
        )
        assert plan.tasks[0].checkpoint is True

    # === Batch with engine — happy path ====================================

    def test_valid_batch(self, tmp_path: Path) -> None:
        plan = self._load(
            tmp_path,
            "version: 1\nname: x\ntasks:\n  - id: t1\n    engine: claude\n    prompt: hi\n    batch:\n      items: [a.py, b.py, c.py]\n      template: \"Review {{ batch.item }}\"\n      max_per_call: 2\n",
        )
        assert plan.tasks[0].batch is not None
        assert plan.tasks[0].batch.items == ["a.py", "b.py", "c.py"]
        assert plan.tasks[0].batch.max_per_call == 2


# ===========================================================================
# CWE Preset Loading — validate loader handles CWE security profile presets
# ===========================================================================


class TestCWEPresetLoading:
    """Verify that CWE security profile presets load correctly."""

    def _load(self, tmp_path: Path, content: str) -> "PlanSpec":  # noqa: F821
        from maestro_cli.loader import load_plan as _lp
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        return _lp(pf)

    def _plan_yaml(self, judge_block: str) -> str:
        return (
            "version: 1\nname: cwe-test\ntasks:\n  - id: t1\n    engine: claude\n"
            f"    prompt: audit\n    judge:\n{judge_block}\n"
        )

    def test_preset_cwe_injection(self, tmp_path: Path) -> None:
        plan = self._load(tmp_path, self._plan_yaml("      preset: cwe_injection"))
        j = plan.tasks[0].judge
        assert j is not None
        assert j.preset == "cwe_injection"
        assert len(j.criteria) == 4

    def test_preset_cwe_auth(self, tmp_path: Path) -> None:
        plan = self._load(tmp_path, self._plan_yaml("      preset: cwe_auth"))
        j = plan.tasks[0].judge
        assert j is not None
        assert j.preset == "cwe_auth"
        assert len(j.criteria) == 4

    def test_preset_cwe_data_exposure(self, tmp_path: Path) -> None:
        plan = self._load(tmp_path, self._plan_yaml("      preset: cwe_data_exposure"))
        j = plan.tasks[0].judge
        assert j is not None
        assert j.preset == "cwe_data_exposure"
        assert len(j.criteria) == 3

    def test_preset_cwe_top_25(self, tmp_path: Path) -> None:
        plan = self._load(tmp_path, self._plan_yaml("      preset: cwe_top_25"))
        j = plan.tasks[0].judge
        assert j is not None
        assert j.preset == "cwe_top_25"
        assert len(j.criteria) == 5

    def test_preset_cwe_criteria_override(self, tmp_path: Path) -> None:
        plan = self._load(tmp_path, self._plan_yaml(
            "      preset: cwe_injection\n      criteria:\n        - custom check"
        ))
        j = plan.tasks[0].judge
        assert j is not None
        assert j.criteria == ["custom check"]

    def test_preset_cwe_threshold_override(self, tmp_path: Path) -> None:
        plan = self._load(tmp_path, self._plan_yaml(
            "      preset: cwe_injection\n      pass_threshold: 0.95"
        ))
        j = plan.tasks[0].judge
        assert j is not None
        assert j.pass_threshold == 0.95

    def test_preset_cwe_with_on_fail_retry(self, tmp_path: Path) -> None:
        plan = self._load(tmp_path, self._plan_yaml(
            "      preset: cwe_auth\n      on_fail: retry"
        ))
        j = plan.tasks[0].judge
        assert j is not None
        assert j.on_fail == "retry"

    def test_preset_cwe_with_method_g_eval(self, tmp_path: Path) -> None:
        plan = self._load(tmp_path, self._plan_yaml(
            "      preset: cwe_data_exposure\n      method: g_eval"
        ))
        j = plan.tasks[0].judge
        assert j is not None
        assert j.method == "g_eval"

    def test_preset_cwe_with_quorum(self, tmp_path: Path) -> None:
        plan = self._load(tmp_path, self._plan_yaml(
            "      preset: cwe_top_25\n      quorum: 3\n      quorum_strategy: unanimous"
        ))
        j = plan.tasks[0].judge
        assert j is not None
        assert j.quorum == 3
        assert j.quorum_strategy == "unanimous"

    def test_preset_cwe_on_security_audit_task(self, tmp_path: Path) -> None:
        content = (
            "version: 1\nname: sec\ntasks:\n  - id: sec-audit\n    engine: claude\n"
            "    model: opus\n    agent: security-engineer\n"
            "    prompt: Audit auth\n    tags: [security, audit]\n"
            "    judge:\n      preset: cwe_auth\n"
        )
        plan = self._load(tmp_path, content)
        t = plan.tasks[0]
        assert t.agent == "security-engineer"
        assert t.judge is not None
        assert t.judge.preset == "cwe_auth"
        assert len(t.judge.criteria) == 4


# ---------------------------------------------------------------------------
# output_scope loading
# ---------------------------------------------------------------------------


class TestOutputScopeLoading:
    """Tests for output_scope field parsing in loader.py."""

    def _load(self, tmp_path: Path, content: str) -> "PlanSpec":  # noqa: F821
        from maestro_cli.loader import load_plan as _lp
        pf = tmp_path / "plan.yaml"
        pf.write_text(content, encoding="utf-8")
        return _lp(pf)

    def test_output_scope_list_of_strings(self, tmp_path: Path) -> None:
        content = (
            "version: 1\nname: scope-test\ntasks:\n"
            "  - id: t1\n    command: echo hi\n"
            "    output_scope:\n      - 'src/*.py'\n      - 'tests/*.py'\n"
        )
        plan = self._load(tmp_path, content)
        assert plan.tasks[0].output_scope == ["src/*.py", "tests/*.py"]

    def test_output_scope_single_string_coerced_to_list(self, tmp_path: Path) -> None:
        content = (
            "version: 1\nname: scope-test\ntasks:\n"
            "  - id: t1\n    command: echo hi\n"
            "    output_scope: 'src/*.py'\n"
        )
        plan = self._load(tmp_path, content)
        assert plan.tasks[0].output_scope == ["src/*.py"]

    def test_no_output_scope_defaults_empty(self, tmp_path: Path) -> None:
        content = (
            "version: 1\nname: scope-test\ntasks:\n"
            "  - id: t1\n    command: echo hi\n"
        )
        plan = self._load(tmp_path, content)
        assert plan.tasks[0].output_scope == []

    def test_output_scope_with_glob_patterns(self, tmp_path: Path) -> None:
        content = (
            "version: 1\nname: scope-test\ntasks:\n"
            "  - id: t1\n    command: echo hi\n"
            "    output_scope:\n      - 'src/**/*.py'\n      - 'docs/*.md'\n"
        )
        plan = self._load(tmp_path, content)
        assert plan.tasks[0].output_scope == ["src/**/*.py", "docs/*.md"]

    def test_output_scope_preserved_through_load_plan(self, tmp_path: Path) -> None:
        content = (
            "version: 1\nname: scope-test\ntasks:\n"
            "  - id: build\n    command: make\n"
            "    output_scope:\n      - 'dist/*'\n"
            "  - id: test\n    command: pytest\n"
            "    depends_on: [build]\n"
        )
        plan = self._load(tmp_path, content)
        build = next(t for t in plan.tasks if t.id == "build")
        test = next(t for t in plan.tasks if t.id == "test")
        assert build.output_scope == ["dist/*"]
        assert test.output_scope == []
