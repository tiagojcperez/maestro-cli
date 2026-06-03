from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.plugins import EnginePlugin


def _write_plan(tmp_path: Path, content: str) -> Path:
    """Helper to write a plan YAML and return its path."""
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


class TestCommandEngineConflict:
    def test_task_with_both_command_and_engine_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    engine: claude
    prompt: "Do something"
""")
        with pytest.raises(PlanValidationError, match="more than one of"):
            load_plan(plan_file)


class TestImportValidation:
    def test_duplicate_import_prefix_raises_e027(self, tmp_path: Path) -> None:
        imported_file = tmp_path / "shared.yaml"
        imported_file.write_text("""\
tasks:
  - id: imported
    command: "echo imported"
""", encoding="utf-8")
        plan_file = _write_plan(tmp_path, f"""\
version: 1
name: test-plan
imports:
  - path: "{imported_file.name}"
    prefix: lib
  - path: "{imported_file.name}"
    prefix: lib
tasks:
  - id: main
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"\[E027\]"):
            load_plan(plan_file)

    def test_import_prefix_with_invalid_characters_raises_e028(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
imports:
  - path: "shared.yaml"
    prefix: Bad_Prefix
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"\[E028\]"):
            load_plan(plan_file)

    def test_import_overrides_must_be_object_raises_e026(self, tmp_path: Path) -> None:
        imported_file = tmp_path / "shared.yaml"
        imported_file.write_text("""\
tasks:
  - id: imported
    command: "echo imported"
""", encoding="utf-8")
        plan_file = _write_plan(tmp_path, f"""\
version: 1
name: test-plan
imports:
  - path: "{imported_file.name}"
    prefix: lib
    overrides: invalid
tasks:
  - id: main
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"\[E026\]"):
            load_plan(plan_file)

    def test_import_override_env_requires_imported_task_env_object(self, tmp_path: Path) -> None:
        imported_file = tmp_path / "shared.yaml"
        imported_file.write_text("""\
tasks:
  - id: imported
    command: "echo imported"
    env: not-an-object
""", encoding="utf-8")
        plan_file = _write_plan(tmp_path, f"""\
version: 1
name: test-plan
imports:
  - path: "{imported_file.name}"
    prefix: lib
    overrides:
      env:
        EXTRA: value
tasks:
  - id: main
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"\[E026\]"):
            load_plan(plan_file)

    def test_imported_file_without_tasks_list_raises_e026(self, tmp_path: Path) -> None:
        imported_file = tmp_path / "shared.yaml"
        imported_file.write_text("name: shared-fragment\n", encoding="utf-8")
        plan_file = _write_plan(tmp_path, f"""\
version: 1
name: test-plan
imports:
  - path: "{imported_file.name}"
    prefix: lib
tasks:
  - id: main
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"\[E026\]"):
            load_plan(plan_file)


class TestReasoningEffortValidation:
    def test_valid_codex_effort_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: codex
    reasoning_effort: medium
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].reasoning_effort == "medium"

    def test_invalid_codex_effort_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: codex
    reasoning_effort: invalid
    prompt: "Do something"
""")
        with pytest.raises(PlanValidationError, match="reasoning_effort 'invalid' is not valid for Codex"):
            load_plan(plan_file)

    def test_valid_claude_effort_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    model: opus
    reasoning_effort: high
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].reasoning_effort == "high"

    def test_invalid_claude_effort_unknown_raises(self, tmp_path: Path) -> None:
        """xhigh became valid for Claude with Opus 4.7 (2026-04-27).
        Use a genuinely unknown value to exercise the validator."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    model: opus
    reasoning_effort: ultra
    prompt: "Do something"
""")
        with pytest.raises(PlanValidationError, match="reasoning_effort 'ultra' is not valid for Claude"):
            load_plan(plan_file)

    def test_all_codex_efforts_accepted(self, tmp_path: Path) -> None:
        for effort in ("minimal", "low", "medium", "high", "xhigh"):
            plan_file = _write_plan(tmp_path, f"""\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: codex
    reasoning_effort: {effort}
    prompt: "Do something"
""")
            plan = load_plan(plan_file)
            assert plan.tasks[0].reasoning_effort == effort

    def test_all_claude_efforts_accepted(self, tmp_path: Path) -> None:
        for effort in ("low", "medium", "high"):
            plan_file = _write_plan(tmp_path, f"""\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    model: opus
    reasoning_effort: {effort}
    prompt: "Do something"
""")
            plan = load_plan(plan_file)
            assert plan.tasks[0].reasoning_effort == effort

    def test_qwen_task_reasoning_effort_warns(self, tmp_path: Path) -> None:
        """reasoning_effort on a qwen task should produce a warning (not an error)."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: qwen
    reasoning_effort: medium
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any(
            "t1" in w and "Qwen" in w and "reasoning_effort" in w
            for w in plan.validation_warnings
        )

    def test_ollama_task_reasoning_effort_warns(self, tmp_path: Path) -> None:
        """reasoning_effort on an ollama task should produce a warning (not an error)."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: ollama
    reasoning_effort: high
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any(
            "t1" in w and "Ollama" in w and "reasoning_effort" in w
            for w in plan.validation_warnings
        )

    def test_copilot_task_reasoning_effort_warns(self, tmp_path: Path) -> None:
        """reasoning_effort on a copilot task should produce a warning (not an error)."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: copilot
    reasoning_effort: low
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any(
            "t1" in w and "Copilot" in w and "reasoning_effort" in w
            for w in plan.validation_warnings
        )


class TestDefaultsReasoningEffort:
    def test_invalid_defaults_codex_effort_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  codex:
    reasoning_effort: bogus
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match="defaults.codex.reasoning_effort 'bogus' is not valid"):
            load_plan(plan_file)

    def test_invalid_defaults_claude_effort_raises(self, tmp_path: Path) -> None:
        # 2026-04-27: xhigh became valid for Claude with Opus 4.7.
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  claude:
    reasoning_effort: ultra
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match="defaults.claude.reasoning_effort 'ultra' is not valid"):
            load_plan(plan_file)

    def test_valid_defaults_codex_effort_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  codex:
    reasoning_effort: high
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.codex.reasoning_effort == "high"

    def test_valid_defaults_claude_effort_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  claude:
    reasoning_effort: low
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.claude.reasoning_effort == "low"


class TestResilienceDefaultsPropagation:
    def test_engine_defaults_propagate_resilience_fields_to_task(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  claude:
    escalation: sonnet
    fallback_engine: codex
    fallback_model: gpt-5.4
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        task = plan.tasks[0]
        assert task.escalation == ["sonnet"]
        assert task.fallback_engine == "codex"
        assert task.fallback_model == "gpt-5.4"

    def test_task_resilience_fields_override_engine_defaults(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  claude:
    escalation: [haiku, sonnet]
    fallback_engine: codex
    fallback_model: gpt-5.4
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    escalation: [opus]
    fallback_engine: gemini
    fallback_model: gemini-2.0-flash
""")
        plan = load_plan(plan_file)
        task = plan.tasks[0]
        assert task.escalation == ["opus"]
        assert task.fallback_engine == "gemini"
        assert task.fallback_model == "gemini-2.0-flash"


class TestClaudeModelValidation:
    def test_unknown_claude_model_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    model: gpt-4
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any("Claude model 'gpt-4' may not be valid" in w for w in plan.validation_warnings)

    def test_valid_claude_model_no_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    model: sonnet
    prompt: "Do something"
""")
        # No warning should be emitted for known models
        plan = load_plan(plan_file)
        assert plan.tasks[0].model == "sonnet"
        assert not any("Claude model" in w for w in plan.validation_warnings)


class TestValidationWarnings:
    def test_context_from_without_context_budget_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: a
    command: "echo a"
  - id: b
    depends_on: [a]
    context_from: [a]
    command: "echo b"
""")
        plan = load_plan(plan_file)
        assert any(
            "Task 'b': has context_from but no context_budget_tokens" in warning
            for warning in plan.validation_warnings
        )

    def test_verify_command_without_retries_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    verify_command: ["pytest", "-q"]
""")
        plan = load_plan(plan_file)
        assert any(
            "Task 't1': has verify_command but max_retries=0" in warning
            for warning in plan.validation_warnings
        )

    def test_single_worktree_task_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, f"""\
version: 1
name: test-plan
workspace_root: "{tmp_path.as_posix()}"
tasks:
  - id: impl
    engine: claude
    prompt: "Do something"
    worktree: true
""")
        plan = load_plan(plan_file)
        assert any(
            "Task 'impl': worktree: true but only one worktree task in plan" in warning
            for warning in plan.validation_warnings
        )

    def test_judge_contains_on_engine_task_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria:
        - {type: contains, value: "done"}
      pass_threshold: 0.5
""")
        plan = load_plan(plan_file)
        assert any(
            "judge 'contains' assertion on engine task" in w
            for w in plan.validation_warnings
        )

    def test_judge_regex_on_engine_task_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria:
        - {type: regex, pattern: "ok.*"}
      pass_threshold: 0.5
""")
        plan = load_plan(plan_file)
        assert any(
            "judge 'regex' assertion on engine task" in w
            for w in plan.validation_warnings
        )

    def test_retry_delay_list_shorter_than_max_retries_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: ["echo", "hi"]
    max_retries: 3
    retry_delay_sec: [1.0]
""")
        plan = load_plan(plan_file)
        assert any(
            "retry_delay_sec has 1 value(s) but max_retries is 3" in w
            for w in plan.validation_warnings
        )

    def test_judge_on_fail_retry_without_max_iterations_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["good output"]
      on_fail: retry
""")
        plan = load_plan(plan_file)
        assert any(
            "judge on_fail='retry' without max_iterations" in w
            for w in plan.validation_warnings
        )


class TestExistingValidations:
    def test_duplicate_task_ids_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo a"
  - id: t1
    command: "echo b"
""")
        with pytest.raises(PlanValidationError, match="unique"):
            load_plan(plan_file)

    def test_circular_dependency_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: a
    depends_on: [c]
    command: "echo a"
  - id: b
    depends_on: [a]
    command: "echo b"
  - id: c
    depends_on: [b]
    command: "echo c"
""")
        with pytest.raises(PlanValidationError, match="cycle"):
            load_plan(plan_file)

    def test_self_dependency_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    depends_on: [t1]
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match="depend on itself"):
            load_plan(plan_file)

    def test_missing_task_id_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match="id is required"):
            load_plan(plan_file)

    def test_unknown_engine_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: openai
    prompt: "Do something"
""")
        with pytest.raises(PlanValidationError, match="unsupported engine"):
            load_plan(plan_file)

    def test_engine_without_prompt_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
""")
        with pytest.raises(PlanValidationError, match="no prompt source"):
            load_plan(plan_file)

    def test_prompt_md_file_without_heading_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt_md_file: "prompts.md"
""")
        with pytest.raises(PlanValidationError, match="both prompt_md_file and prompt_md_heading"):
            load_plan(plan_file)

    def test_prompt_md_heading_without_file_raises(self, tmp_path: Path) -> None:
        # Also provide a prompt so we get past the "no prompt source" check
        # and hit the prompt_md_file/prompt_md_heading pairing validation.
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Some prompt"
    prompt_md_heading: "My Section"
""")
        with pytest.raises(PlanValidationError, match="both prompt_md_file and prompt_md_heading"):
            load_plan(plan_file)

    def test_prompt_md_heading_hash_prefix_and_non_ascii_warn(
        self, tmp_path: Path
    ) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt_md_file: "prompts.md"
    prompt_md_heading: "# Résumé"
""")
        plan = load_plan(plan_file)
        assert any("prompt_md_heading starts with '#'" in w for w in plan.validation_warnings)
        assert any("prompt_md_heading contains non-ASCII" in w for w in plan.validation_warnings)

    def test_group_task_with_prompt_file_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    group: "subplan.yaml"
    prompt_file: "prompts/task.txt"
""")
        with pytest.raises(PlanValidationError, match=r"\[E011\].*group tasks cannot have prompt fields"):
            load_plan(plan_file)

    def test_version_not_1_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 2
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"schema version 2|version: 1|\[E002\]"):
            load_plan(plan_file)

    def test_max_parallel_less_than_1_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
max_parallel: 0
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match="max_parallel"):
            load_plan(plan_file)

    def test_invalid_plan_name_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: "invalid name with spaces!"
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match="invalid characters"):
            load_plan(plan_file)

    def test_no_command_or_engine_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    description: "Task with nothing"
""")
        with pytest.raises(PlanValidationError, match="must define 'command', 'engine', or 'group'"):
            load_plan(plan_file)

    def test_depends_on_unknown_task_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    depends_on: [nonexistent]
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match="unknown task"):
            load_plan(plan_file)

    def test_valid_minimal_plan_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.name == "test-plan"
        assert len(plan.tasks) == 1
        assert plan.tasks[0].id == "t1"


class TestStdoutTailLinesValidation:
    """Tests for configurable stdout_tail_lines at defaults and task level."""

    def test_defaults_stdout_tail_lines_parsed(self, tmp_path: Path) -> None:
        """Plan with defaults.stdout_tail_lines: 200 parses to 200."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  stdout_tail_lines: 200
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.stdout_tail_lines == 200

    def test_defaults_stdout_tail_lines_default_50(self, tmp_path: Path) -> None:
        """Plan without stdout_tail_lines defaults to 50."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.stdout_tail_lines == 50

    def test_defaults_stdout_tail_lines_zero_raises(self, tmp_path: Path) -> None:
        """defaults.stdout_tail_lines: 0 raises PlanValidationError."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  stdout_tail_lines: 0
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match="stdout_tail_lines must be >= 1"):
            load_plan(plan_file)

    def test_task_stdout_tail_lines_parsed(self, tmp_path: Path) -> None:
        """Task with stdout_tail_lines: 100 parses to 100."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    stdout_tail_lines: 100
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].stdout_tail_lines == 100

    def test_task_stdout_tail_lines_zero_raises(self, tmp_path: Path) -> None:
        """Task stdout_tail_lines: 0 raises PlanValidationError."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    stdout_tail_lines: 0
""")
        with pytest.raises(PlanValidationError, match="stdout_tail_lines must be >= 1"):
            load_plan(plan_file)

    def test_task_stdout_tail_lines_none_by_default(self, tmp_path: Path) -> None:
        """Task without stdout_tail_lines defaults to None."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].stdout_tail_lines is None


class TestEditPolicyValidation:
    """Tests for edit_policy at defaults and task level."""

    def test_defaults_edit_policy_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  edit_policy: efficient
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.edit_policy == "efficient"

    def test_defaults_edit_policy_strict(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  edit_policy: strict
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.edit_policy == "strict"

    def test_defaults_edit_policy_default_when_omitted(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.edit_policy == "default"

    def test_defaults_edit_policy_invalid_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  edit_policy: aggressive
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match="edit_policy.*not valid"):
            load_plan(plan_file)

    def test_task_edit_policy_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    edit_policy: strict
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].edit_policy == "strict"

    def test_task_edit_policy_none_when_omitted(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].edit_policy is None

    def test_task_edit_policy_invalid_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    edit_policy: turbo
""")
        with pytest.raises(PlanValidationError, match="edit_policy.*not valid"):
            load_plan(plan_file)

    def test_task_edit_policy_on_shell_task_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    edit_policy: efficient
""")
        plan = load_plan(plan_file)
        assert any("no effect on shell" in w for w in plan.validation_warnings)


class TestAppendSystemPromptParsing:
    """Tests for append_system_prompt at engine defaults and task level."""

    def test_claude_defaults_append_system_prompt(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  claude:
    append_system_prompt: "Always use Portuguese"
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.claude.append_system_prompt == "Always use Portuguese"

    def test_codex_defaults_append_system_prompt(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  codex:
    append_system_prompt: "Prefer minimal diffs"
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.codex.append_system_prompt == "Prefer minimal diffs"

    def test_defaults_append_system_prompt_none_when_omitted(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.claude.append_system_prompt is None
        assert plan.defaults.codex.append_system_prompt is None

    def test_task_append_system_prompt(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    append_system_prompt: "Be extra careful"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].append_system_prompt == "Be extra careful"

    def test_task_append_system_prompt_none_when_omitted(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].append_system_prompt is None


class TestVerifyCommandValidation:
    def test_verify_command_string_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    verify_command: "pytest -v"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].verify_command == "pytest -v"

    def test_verify_command_list_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    verify_command: ["pytest", "-v"]
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].verify_command == ["pytest", "-v"]

    def test_verify_command_invalid_type_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    verify_command: 42
""")
        with pytest.raises(PlanValidationError, match="verify_command must be a string"):
            load_plan(plan_file)

    def test_verify_command_list_non_string_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    verify_command: [1, 2]
""")
        with pytest.raises(PlanValidationError, match="verify_command list must contain only strings"):
            load_plan(plan_file)

    def test_verify_command_default_none(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].verify_command is None


class TestContextIntelligenceParsing:
    def test_defaults_context_budget_tokens_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  context_budget_tokens: 12000
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.context_budget_tokens == 12000

    def test_defaults_workspace_index_exclude_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  workspace_index_exclude:
    - .git/**
    - node_modules/**
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.workspace_index_exclude == [".git/**", "node_modules/**"]

    def test_task_checkpoint_context_budget_and_judge_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    checkpoint: true
    context_budget_tokens: 4000
    judge:
      criteria: ["compiles", "tests pass"]
      pass_threshold: 0.8
      on_fail: retry
      model: sonnet
""")
        plan = load_plan(plan_file)
        task = plan.tasks[0]
        assert task.checkpoint is True
        assert task.context_budget_tokens == 4000
        assert task.judge is not None
        assert task.judge.criteria == ["compiles", "tests pass"]
        assert task.judge.pass_threshold == 0.8
        assert task.judge.on_fail == "retry"
        assert task.judge.model == "sonnet"

    def test_task_workspace_index_exclude_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    workspace_index_exclude:
      - build/**
      - dist/**
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].workspace_index_exclude == ["build/**", "dist/**"]

    def test_context_budget_tokens_invalid_raises_e019(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    context_budget_tokens: 0
""")
        with pytest.raises(PlanValidationError, match=r"\[E019\].*context_budget_tokens"):
            load_plan(plan_file)

    def test_context_budget_tokens_non_numeric_string_raises_e019(self, tmp_path: Path) -> None:
        """_to_context_budget_or_none raises E019 on non-integer string (ValueError path)."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    context_budget_tokens: "abc"
""")
        with pytest.raises(PlanValidationError, match=r"\[E019\]"):
            load_plan(plan_file)

    def test_judge_invalid_on_fail_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["good quality"]
      on_fail: ignore
""")
        with pytest.raises(PlanValidationError, match=r"\[E020\].*judge.on_fail"):
            load_plan(plan_file)

    def test_judge_invalid_pass_threshold_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["good quality"]
      pass_threshold: 1.5
""")
        with pytest.raises(PlanValidationError, match=r"\[E020\].*pass_threshold"):
            load_plan(plan_file)


class TestMaxIterationsValidation:
    def test_max_iterations_zero_raises_e022(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    max_iterations: 0
""")
        with pytest.raises(PlanValidationError, match=r"\[E022\].*max_iterations"):
            load_plan(plan_file)

    def test_max_iterations_one_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    max_iterations: 1
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].max_iterations == 1


class TestApprovalMessageValidation:
    def test_approval_message_without_requires_approval_raises_e029(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    approval_message: "Please confirm"
""")
        with pytest.raises(PlanValidationError, match=r"\[E029\]"):
            load_plan(plan_file)

    def test_approval_message_with_requires_approval_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    requires_approval: true
    approval_message: "Please confirm before running"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].approval_message == "Please confirm before running"


class TestContextModeRecursiveValidation:
    def test_recursive_without_workspace_root_raises_e021(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Analyse the workspace"
    context_mode: recursive
""")
        with pytest.raises(PlanValidationError, match=r"\[E021\].*recursive"):
            load_plan(plan_file)


class TestTagsWithWhitespaceWarning:
    def test_tag_with_space_produces_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    tags: ["good tag", "another"]
""")
        plan = load_plan(plan_file)
        warnings = plan.validation_warnings
        assert any("contains whitespace" in w for w in warnings)

    def test_tag_without_space_no_whitespace_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    tags: ["good-tag", "another-tag"]
""")
        plan = load_plan(plan_file)
        assert not any("contains whitespace" in w for w in plan.validation_warnings)


class TestPluginEngines:
    def test_custom_engine_from_plugin_passes_validation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plugin = EnginePlugin(
            name="custom",
            build_command=lambda ctx: (["custom-engine", ctx.prompt_text], False),
        )
        monkeypatch.setattr("maestro_cli.loader.get_engine_plugin", lambda _name: plugin)

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: custom
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].engine == "custom"


class TestBudgetWarningPctBoundary:
    def test_budget_warning_pct_exactly_one_raises_e023(self, tmp_path: Path) -> None:
        # 1.0 is excluded — the valid range is (0.0, 1.0) exclusive
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
budget_warning_pct: 1.0
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"\[E023\]"):
            load_plan(plan_file)

    def test_budget_warning_pct_valid_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
budget_warning_pct: 0.9
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.budget_warning_pct == 0.9


class TestJudgeTypedAssertions:
    def test_cost_under_assertion_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do a thing"
    judge:
      criteria:
        - type: cost_under
          value: 0.5
      pass_threshold: 1.0
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].judge is not None
        criteria = plan.tasks[0].judge.criteria
        assert len(criteria) == 1
        assert criteria[0]["type"] == "cost_under"
        assert criteria[0]["value"] == 0.5

    def test_rubric_level_score_out_of_range_raises_e020(self, tmp_path: Path) -> None:
        """Rubric level score must be 1-5; score=6 should raise E020."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do a thing"
    judge:
      criteria:
        - type: rubric
          name: quality
          levels:
            - score: 6
              description: "Perfect"
      pass_threshold: 0.8
""")
        with pytest.raises(PlanValidationError, match=r"\[E020\].*score"):
            load_plan(plan_file)

    def test_rubric_criterion_missing_name_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do a thing"
    judge:
      criteria:
        - type: rubric
          levels:
            - score: 1
              description: "Poor"
            - score: 5
              description: "Excellent"
      pass_threshold: 0.8
""")
        with pytest.raises(PlanValidationError, match=r"\[E020\]"):
            load_plan(plan_file)


class TestSecretsValidation:
    def test_secrets_list_format_parses(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
secrets:
  - AWS_KEY
  - DB_PASSWORD
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.secrets == ["AWS_KEY", "DB_PASSWORD"]
        assert plan.secrets_auto is False

    def test_secrets_auto_mode_parses(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
secrets: auto
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.secrets_auto is True
        assert plan.secrets == []

    def test_invalid_secrets_type_raises_e024(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
secrets: 42
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"\[E024\]"):
            load_plan(plan_file)


class TestUnknownTemplateVarWarning:
    def test_unknown_template_var_in_prompt_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something with {{ typo_var }}"
""")
        plan = load_plan(plan_file)
        warnings = plan.validation_warnings
        assert any("typo_var" in w for w in warnings)
        assert any("does not match any known pattern" in w for w in warnings)

    def test_known_global_var_does_not_warn(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Root is {{ workspace_root }} and plan is {{ plan_name }}"
""")
        plan = load_plan(plan_file)
        assert not any("does not match any known pattern" in w for w in plan.validation_warnings)


class TestTimeoutWarnings:
    def test_task_without_timeout_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert any("t1" in w and "timeout_sec" in w for w in plan.validation_warnings)

    def test_plan_timeout_default_suppresses_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  timeout_sec: 600
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert not any("timeout_sec" in w for w in plan.validation_warnings)

    def test_timeout_with_retries_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    timeout_sec: 60
    max_retries: 2
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert any("t1" in w and "timeout" in w and "retries" in w.lower() for w in plan.validation_warnings)


class TestBudgetWarningPctLowerBoundary:
    def test_budget_warning_pct_exactly_zero_raises_e023(self, tmp_path: Path) -> None:
        # 0.0 is excluded — the valid range is (0.0, 1.0) exclusive
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
budget_warning_pct: 0.0
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"\[E023\]"):
            load_plan(plan_file)


class TestMatrixValidation:
    def test_matrix_empty_dict_raises_e018(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Run {{ matrix.env }}"
    matrix: {}
""")
        with pytest.raises(PlanValidationError, match="at least one dimension"):
            load_plan(plan_file)

    def test_matrix_key_with_empty_list_raises_e018(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Run {{ matrix.env }}"
    matrix:
      env: []
""")
        with pytest.raises(PlanValidationError, match="non-empty list"):
            load_plan(plan_file)


class TestContextModeSummarizedRequiresContextFrom:
    def test_summarized_without_context_from_raises(self, tmp_path: Path) -> None:
        """context_mode: summarized without context_from should raise E001."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Summarize upstream"
    context_mode: summarized
""")
        with pytest.raises(PlanValidationError, match="context_mode 'summarized' requires"):
            load_plan(plan_file)

    def test_map_reduce_without_context_from_raises(self, tmp_path: Path) -> None:
        """context_mode: map_reduce without context_from should raise E001."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Synthesize upstream"
    context_mode: map_reduce
""")
        with pytest.raises(PlanValidationError, match="context_mode 'map_reduce' requires"):
            load_plan(plan_file)


class TestWhenExpressionValidation:
    def test_when_references_task_not_in_depends_on_raises_e015(self, tmp_path: Path) -> None:
        """when expression referencing a valid task that is not in depends_on raises E015."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: build
    command: "echo build"
    timeout_sec: 30
  - id: deploy
    command: "echo deploy"
    timeout_sec: 30
    when: "{{ build.status }} == success"
""")
        with pytest.raises(PlanValidationError, match=r"\[E015\]"):
            load_plan(plan_file)


class TestJudgeGEvalHaikuWarning:
    def test_g_eval_with_haiku_warns(self, tmp_path: Path) -> None:
        """judge.method: g_eval with model: haiku should emit a warning."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["output is coherent"]
      method: g_eval
      model: haiku
      pass_threshold: 0.7
""")
        plan = load_plan(plan_file)
        assert any("g_eval" in w and "haiku" in w for w in plan.validation_warnings)


class TestRetryDelaySecInvalidType:
    def test_retry_delay_sec_dict_raises_e013(self, tmp_path: Path) -> None:
        """retry_delay_sec with a dict value (not a number or list) should raise E013."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    max_retries: 2
    retry_delay_sec:
      delay: 5
""")
        with pytest.raises(PlanValidationError, match=r"\[E013\]"):
            load_plan(plan_file)


class TestJudgeMethodValidation:
    def test_invalid_judge_method_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["looks good"]
      method: chain_of_thought
""")
        with pytest.raises(PlanValidationError, match=r"\[E020\]"):
            load_plan(plan_file)

    def test_rubric_level_score_out_of_range_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do a thing"
    judge:
      criteria:
        - type: rubric
          name: quality
          levels:
            - score: 0
              description: "Unacceptable"
            - score: 5
              description: "Excellent"
      pass_threshold: 0.8
""")
        with pytest.raises(PlanValidationError, match=r"\[E020\].*score must be an integer 1-5"):
            load_plan(plan_file)


class TestBackslashPathWarnings:
    def test_workspace_root_backslash_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
workspace_root: "C:\\\\path\\\\to\\\\dir"
tasks:
  - id: t1
    command: "echo hello"
    timeout_sec: 30
""")
        plan = load_plan(plan_file)
        assert any("workspace_root" in w and "backslash" in w for w in plan.validation_warnings)

    def test_workdir_backslash_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    workdir: "C:\\\\Users\\\\dev\\\\project"
    timeout_sec: 30
""")
        plan = load_plan(plan_file)
        assert any("t1" in w and "workdir" in w and "backslash" in w for w in plan.validation_warnings)


class TestRetryDelaySecNegative:
    def test_retry_delay_sec_negative_float_raises_e013(self, tmp_path: Path) -> None:
        """retry_delay_sec with a negative value should raise PlanValidationError."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    max_retries: 1
    retry_delay_sec: -1.0
""")
        with pytest.raises(PlanValidationError, match=r"\[E013\]"):
            load_plan(plan_file)

    def test_retry_delay_sec_negative_in_list_raises_e013(self, tmp_path: Path) -> None:
        """retry_delay_sec list with a negative entry should raise PlanValidationError."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    max_retries: 2
    retry_delay_sec: [1.0, -0.5]
""")
        with pytest.raises(PlanValidationError, match=r"\[E013\]"):
            load_plan(plan_file)


class TestGuardCommandValidation:
    def test_guard_command_list_parses(self, tmp_path: Path) -> None:
        """guard_command as a list of strings should parse correctly."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    guard_command: ["python", "check.py"]
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].guard_command == ["python", "check.py"]

    def test_guard_command_list_with_non_string_raises(self, tmp_path: Path) -> None:
        """guard_command list with a non-string element should raise PlanValidationError."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    guard_command: ["python", 42]
""")
        with pytest.raises(PlanValidationError, match="guard_command"):
            load_plan(plan_file)


class TestMatrixScalarValueRaises:
    def test_matrix_key_with_scalar_value_raises_e018(self, tmp_path: Path) -> None:
        """matrix key whose value is a scalar (not a list) should raise PlanValidationError."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Run {{ matrix.env }}"
    matrix:
      env: prod
""")
        with pytest.raises(PlanValidationError, match="must be a list"):
            load_plan(plan_file)


class TestEnvVarReferenceWarnings:
    def test_unknown_env_var_in_command_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo $MY_CUSTOM_SECRET"
    timeout_sec: 30
""")
        plan = load_plan(plan_file)
        assert any("MY_CUSTOM_SECRET" in w for w in plan.validation_warnings)

    def test_env_var_defined_in_plan_defaults_suppresses_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  env:
    MY_CUSTOM_SECRET: "hunter2"
tasks:
  - id: t1
    command: "echo $MY_CUSTOM_SECRET"
    timeout_sec: 30
""")
        plan = load_plan(plan_file)
        assert not any("MY_CUSTOM_SECRET" in w for w in plan.validation_warnings)


class TestImportMissingFields:
    def test_import_missing_path_raises_e026(self, tmp_path: Path) -> None:
        """Import entry without 'path' field raises E026."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
imports:
  - prefix: lib
tasks:
  - id: main
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"\[E026\]"):
            load_plan(plan_file)

    def test_import_missing_prefix_raises_e026(self, tmp_path: Path) -> None:
        """Import entry without 'prefix' field raises E026."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
imports:
  - path: "shared.yaml"
tasks:
  - id: main
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"\[E026\]"):
            load_plan(plan_file)


class TestJudgePresetOverrides:
    def test_judge_preset_explicit_pass_threshold_overrides_preset(self, tmp_path: Path) -> None:
        """Explicit pass_threshold overrides the preset's default (code_quality preset default is 0.6)."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      preset: code_quality
      pass_threshold: 0.9
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.pass_threshold == pytest.approx(0.9)

    def test_judge_preset_explicit_aggregation_overrides_preset(self, tmp_path: Path) -> None:
        """Explicit aggregation overrides the preset's default (code_quality preset default is weighted_mean)."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      preset: code_quality
      aggregation: mean
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].judge is not None
        assert plan.tasks[0].judge.aggregation == "mean"


class TestDefaultsEnvValidation:
    def test_defaults_env_parsed_correctly(self, tmp_path: Path) -> None:
        """defaults.env dict is parsed and stored on plan.defaults.env."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  env:
    MY_VAR: "hello"
    OTHER_VAR: "world"
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.env == {"MY_VAR": "hello", "OTHER_VAR": "world"}

    def test_defaults_env_non_dict_raises_e018(self, tmp_path: Path) -> None:
        """defaults.env set to a string (not a dict) raises PlanValidationError [E018]."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  env: "not_a_dict"
tasks:
  - id: t1
    command: "echo hello"
""")
        with pytest.raises(PlanValidationError, match=r"\[E018\]"):
            load_plan(plan_file)


class TestRetryDelaySecValidList:
    def test_retry_delay_sec_valid_list_parses(self, tmp_path: Path) -> None:
        """retry_delay_sec as a list of floats is accepted and stored correctly."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: echo hello
    max_retries: 3
    retry_delay_sec: [1.0, 2.5, 5.0]
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].retry_delay_sec == [1.0, 2.5, 5.0]

    def test_retry_delay_sec_list_equal_to_max_retries_does_not_warn(
        self, tmp_path: Path
    ) -> None:
        """retry_delay_sec list with exactly max_retries entries should not trigger W6."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: echo hello
    max_retries: 2
    retry_delay_sec: [1.0, 2.0]
""")
        plan = load_plan(plan_file)
        assert not any("retry_delay_sec has" in w for w in plan.validation_warnings)


class TestContextModeSummarizedValid:
    def test_context_mode_summarized_with_context_from_passes(
        self, tmp_path: Path
    ) -> None:
        """context_mode: summarized with context_from is valid and loads cleanly."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: setup
    command: echo done
  - id: impl
    engine: claude
    prompt: Implement feature
    depends_on: [setup]
    context_from: [setup]
    context_mode: summarized
    context_budget_tokens: 4000
""")
        plan = load_plan(plan_file)
        assert plan.tasks[1].context_mode == "summarized"


class TestDefaultsContextBudgetSuppressesWarning:
    def test_defaults_context_budget_suppresses_no_budget_warning(
        self, tmp_path: Path
    ) -> None:
        """defaults.context_budget_tokens suppresses the W-context-no-budget warning."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  context_budget_tokens: 6000
tasks:
  - id: setup
    command: echo done
  - id: impl
    engine: claude
    prompt: Implement feature
    depends_on: [setup]
    context_from: [setup]
""")
        plan = load_plan(plan_file)
        assert not any("context_budget_tokens" in w for w in plan.validation_warnings)


class TestDefaultsReasoningEffortWarnings:
    def test_defaults_gemini_reasoning_effort_warns(self, tmp_path: Path) -> None:
        """defaults.gemini.reasoning_effort warns that Gemini CLI does not support it."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  gemini:
    reasoning_effort: medium
tasks:
  - id: t1
    engine: gemini
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any(
            "gemini" in w.lower() and "reasoning_effort" in w
            for w in plan.validation_warnings
        )


class TestJudgeDurationUnderAssertion:
    def test_duration_under_assertion_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do a thing"
    judge:
      criteria:
        - type: duration_under
          value: 30.0
      pass_threshold: 1.0
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].judge is not None
        criteria = plan.tasks[0].judge.criteria
        assert len(criteria) == 1
        assert criteria[0]["type"] == "duration_under"
        assert criteria[0]["value"] == 30.0


class TestJudgePresetInvalidName:
    def test_invalid_preset_name_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do a thing"
    judge:
      preset: nonexistent_preset
""")
        with pytest.raises(PlanValidationError, match=r"\[E020\]"):
            load_plan(plan_file)


class TestJudgeInvalidAggregation:
    def test_invalid_aggregation_raises_e020(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do a thing"
    judge:
      criteria:
        - "Output is correct"
      aggregation: median
""")
        with pytest.raises(PlanValidationError, match=r"\[E020\]"):
            load_plan(plan_file)


class TestDefaultsCopilotReasoningEffortWarning:
    def test_defaults_copilot_reasoning_effort_warns(self, tmp_path: Path) -> None:
        """defaults.copilot.reasoning_effort warns that Copilot CLI does not support it."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  copilot:
    reasoning_effort: medium
tasks:
  - id: t1
    engine: copilot
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any(
            "copilot" in w.lower() and "reasoning_effort" in w
            for w in plan.validation_warnings
        )


class TestDefaultsQwenReasoningEffortWarning:
    def test_defaults_qwen_reasoning_effort_warns(self, tmp_path: Path) -> None:
        """defaults.qwen.reasoning_effort warns that Qwen CLI does not support it."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  qwen:
    reasoning_effort: high
tasks:
  - id: t1
    engine: qwen
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any(
            "qwen" in w.lower() and "reasoning_effort" in w
            for w in plan.validation_warnings
        )


class TestContextCompactParsing:
    def test_context_compact_true_parses(self, tmp_path: Path) -> None:
        """context_compact: true on an engine task is parsed as True."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: setup
    command: echo done
  - id: impl
    engine: claude
    prompt: Implement feature
    depends_on: [setup]
    context_from: [setup]
    context_budget_tokens: 4000
    context_compact: true
""")
        plan = load_plan(plan_file)
        assert plan.tasks[1].context_compact is True

    def test_context_compact_defaults_false(self, tmp_path: Path) -> None:
        """context_compact defaults to False when not specified."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do a thing"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].context_compact is False


class TestCommandTypeValidation:
    """Tests for E018 on command/pre_command/guard_command invalid types."""

    def test_command_as_dict_raises(self, tmp_path: Path) -> None:
        """command as a mapping object (dict) should raise E018."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command:
      key: value
""")
        with pytest.raises(PlanValidationError, match="command must be a string or list"):
            load_plan(plan_file)

    def test_pre_command_invalid_type_raises(self, tmp_path: Path) -> None:
        """pre_command as a bare integer should raise E018."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo main"
    pre_command: 99
""")
        with pytest.raises(PlanValidationError, match="pre_command must be a string or list"):
            load_plan(plan_file)

    def test_guard_command_invalid_type_raises(self, tmp_path: Path) -> None:
        """guard_command as a bare integer should raise E018."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo main"
    guard_command: 42
""")
        with pytest.raises(PlanValidationError, match="guard_command must be a string or list"):
            load_plan(plan_file)

    def test_watch_minimal_verify_command_source_defaults(self, tmp_path: Path) -> None:
        """Watch block with metric_source: verify_command needs no metric_pattern and uses all field defaults."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: run
    command: "echo hello"
    verify_command: "python -c 'print(42)'"
watch:
  metric: score
  metric_source: verify_command
  metric_task: run
""")
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.metric == "score"
        assert plan.watch.metric_source == "verify_command"
        assert plan.watch.metric_pattern is None
        assert plan.watch.metric_direction == "lower_is_better"
        assert plan.watch.on_regression == "rollback"
        assert plan.watch.warmup_iterations == 1
        assert plan.watch.plateau_threshold == 5
        assert plan.watch.plateau_action == "stop"
        assert plan.watch.max_iterations == 100
        assert plan.watch.max_cost_usd is None


# ---------------------------------------------------------------------------
# W22: Judge timeout insufficient for method/criteria/quorum
# ---------------------------------------------------------------------------


class TestW22JudgeTimeoutWarning:
    """W22 warns when explicit judge.timeout_sec is too low for the config."""

    def test_g_eval_explicit_low_timeout_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["a", "b", "c"]
      method: g_eval
      timeout_sec: 60
      pass_threshold: 0.7
""")
        plan = load_plan(plan_file)
        assert any("W22" in w and "g_eval" in w for w in plan.validation_warnings)

    def test_g_eval_many_criteria_auto_scale_info(self, tmp_path: Path) -> None:
        """g_eval with 6 criteria and no explicit timeout should emit info warning."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["a", "b", "c", "d", "e", "f"]
      method: g_eval
      pass_threshold: 0.7
""")
        plan = load_plan(plan_file)
        w22 = [w for w in plan.validation_warnings if "W22" in w]
        assert len(w22) == 1
        assert "auto-scaled" in w22[0]

    def test_g_eval_few_criteria_no_warning(self, tmp_path: Path) -> None:
        """g_eval with 3 criteria and default timeout should NOT warn."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["a", "b", "c"]
      method: g_eval
      pass_threshold: 0.7
""")
        plan = load_plan(plan_file)
        assert not any("W22" in w for w in plan.validation_warnings)

    def test_g_eval_sufficient_explicit_timeout_no_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["a", "b", "c"]
      method: g_eval
      timeout_sec: 200
      pass_threshold: 0.7
""")
        plan = load_plan(plan_file)
        assert not any("W22" in w for w in plan.validation_warnings)

    def test_debate_explicit_low_timeout_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["a", "b"]
      method: debate
      timeout_sec: 60
      pass_threshold: 0.7
""")
        plan = load_plan(plan_file)
        assert any("W22" in w and "debate" in w for w in plan.validation_warnings)

    def test_quorum_explicit_low_timeout_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["a", "b"]
      quorum: 3
      timeout_sec: 60
      pass_threshold: 0.7
""")
        plan = load_plan(plan_file)
        assert any("W22" in w and "quorum" in w for w in plan.validation_warnings)

    def test_direct_method_no_w22(self, tmp_path: Path) -> None:
        """Direct method with default timeout should never trigger W22."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    judge:
      criteria: ["a"]
      method: direct
      pass_threshold: 0.7
""")
        plan = load_plan(plan_file)
        assert not any("W22" in w for w in plan.validation_warnings)
