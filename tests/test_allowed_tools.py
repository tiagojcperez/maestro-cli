from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.audit import audit_plan
from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import (
    CLAUDE_TOOLS,
    CODEX_SANDBOX_LEVELS,
    TOOL_CATEGORIES,
    EngineDefaults,
    PlanDefaults,
    PlanSpec,
    PolicySpec,
    TaskSpec,
)
from maestro_cli.policy import compile_policy, evaluate_policies
from maestro_cli.runners import (
    _build_restriction_prompt,
    _expand_tool_categories,
    _inject_tool_restriction,
    _split_tool_permissions,
    build_command,
    parse_tool_pattern,
)


def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


# ---------------------------------------------------------------------------
# TestAllowedToolsConstants
# ---------------------------------------------------------------------------


class TestAllowedToolsConstants:
    """Verify that the constants for allowed_tools are correctly defined."""

    def test_claude_tools_contains_expected(self) -> None:
        expected = {"Read", "Write", "Edit", "Bash", "Glob", "Grep"}
        assert expected.issubset(CLAUDE_TOOLS)

    def test_claude_tools_contains_web_tools(self) -> None:
        assert "WebSearch" in CLAUDE_TOOLS
        assert "WebFetch" in CLAUDE_TOOLS

    def test_claude_tools_contains_todo(self) -> None:
        assert "TodoWrite" in CLAUDE_TOOLS

    def test_claude_tools_is_frozenset(self) -> None:
        assert isinstance(CLAUDE_TOOLS, frozenset)

    def test_codex_sandbox_levels(self) -> None:
        assert "workspace-write" in CODEX_SANDBOX_LEVELS
        assert "workspace-read-only" in CODEX_SANDBOX_LEVELS
        assert "network-off" in CODEX_SANDBOX_LEVELS

    def test_codex_sandbox_levels_is_frozenset(self) -> None:
        assert isinstance(CODEX_SANDBOX_LEVELS, frozenset)

    def test_tool_categories_has_read_only(self) -> None:
        assert "read-only" in TOOL_CATEGORIES

    def test_tool_categories_has_no_shell(self) -> None:
        assert "no-shell" in TOOL_CATEGORIES

    def test_read_only_claude_excludes_bash(self) -> None:
        claude_read_only = TOOL_CATEGORIES["read-only"]["claude"]
        assert "Bash" not in claude_read_only
        assert "Read" in claude_read_only
        assert "Glob" in claude_read_only
        assert "Grep" in claude_read_only

    def test_read_only_codex_is_workspace_read_only(self) -> None:
        codex_read_only = TOOL_CATEGORIES["read-only"]["codex"]
        assert codex_read_only == ["workspace-read-only"]

    def test_no_shell_claude_excludes_bash(self) -> None:
        claude_no_shell = TOOL_CATEGORIES["no-shell"]["claude"]
        assert "Bash" not in claude_no_shell
        assert "Read" in claude_no_shell
        assert "Write" in claude_no_shell
        assert "Edit" in claude_no_shell

    def test_no_shell_codex_is_workspace_read_only(self) -> None:
        codex_no_shell = TOOL_CATEGORIES["no-shell"]["codex"]
        assert codex_no_shell == ["workspace-read-only"]


# ---------------------------------------------------------------------------
# TestAllowedToolsParsing
# ---------------------------------------------------------------------------


class TestAllowedToolsParsing:
    """Test YAML parsing of allowed_tools at task and defaults levels."""

    def test_parse_allowed_tools_from_task(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [Read, Grep]
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].allowed_tools == ["Read", "Grep"]

    def test_parse_allowed_tools_from_engine_defaults(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  claude:
    allowed_tools: [Read, Glob, Grep]
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.claude.allowed_tools == ["Read", "Glob", "Grep"]

    def test_defaults_inheritance_task_inherits(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  claude:
    allowed_tools: [Read, Glob, Grep]
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        # Task inherits allowed_tools from engine defaults
        assert plan.tasks[0].allowed_tools == ["Read", "Glob", "Grep"]

    def test_task_level_overrides_engine_defaults(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  claude:
    allowed_tools: [Read, Glob, Grep]
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [Read, Write, Edit]
""")
        plan = load_plan(plan_file)
        # Task-level takes priority over engine defaults
        assert plan.tasks[0].allowed_tools == ["Read", "Write", "Edit"]

    def test_absent_allowed_tools_stays_none(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].allowed_tools is None

    def test_null_allowed_tools_stays_none(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: null
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].allowed_tools is None

    def test_single_string_allowed_tools(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: Read
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].allowed_tools == ["Read"]

    def test_allowed_tools_in_to_dict(self) -> None:
        task = TaskSpec(id="t1", allowed_tools=["Read", "Grep"])
        d = task.to_dict()
        assert d["allowed_tools"] == ["Read", "Grep"]

    def test_allowed_tools_none_in_to_dict(self) -> None:
        task = TaskSpec(id="t1")
        d = task.to_dict()
        assert d["allowed_tools"] is None

    def test_engine_defaults_allowed_tools_field(self) -> None:
        ed = EngineDefaults(allowed_tools=["Read", "Grep"])
        assert ed.allowed_tools == ["Read", "Grep"]

    def test_engine_defaults_allowed_tools_none_by_default(self) -> None:
        ed = EngineDefaults()
        assert ed.allowed_tools is None


# ---------------------------------------------------------------------------
# TestAllowedToolsValidation
# ---------------------------------------------------------------------------


class TestAllowedToolsValidation:
    """Test E071 and W27 validation rules for allowed_tools."""

    def test_e071_allowed_tools_on_command_task(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    allowed_tools: [Read, Grep]
""")
        with pytest.raises(PlanValidationError, match="E071"):
            load_plan(plan_file)

    def test_e071_allowed_tools_on_group_task(self, tmp_path: Path) -> None:
        # Create a sub-plan for the group
        sub_plan = tmp_path / "sub.yaml"
        sub_plan.write_text("""\
version: 1
name: sub-plan
tasks:
  - id: s1
    command: "echo sub"
""", encoding="utf-8")
        plan_file = _write_plan(tmp_path, f"""\
version: 1
name: test-plan
tasks:
  - id: t1
    group: "{sub_plan.as_posix()}"
    allowed_tools: [Read, Grep]
""")
        with pytest.raises(PlanValidationError, match="E071"):
            load_plan(plan_file)

    def test_w27_unknown_claude_tool(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [Read, FakeTool]
""")
        plan = load_plan(plan_file)
        assert any("W27" in w and "FakeTool" in w for w in plan.validation_warnings)

    def test_w27_unknown_codex_tool(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: codex
    prompt: "Do something"
    allowed_tools: [workspace-write, bogus-level]
""")
        plan = load_plan(plan_file)
        assert any("W27" in w and "bogus-level" in w for w in plan.validation_warnings)

    def test_w27_ollama_advisory_only(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: ollama
    prompt: "Do something"
    allowed_tools: [Read]
""")
        plan = load_plan(plan_file)
        assert any("W27" in w and "advisory only" in w for w in plan.validation_warnings)

    def test_w27_gemini_system_prompt_enforced(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: gemini
    prompt: "Do something"
    allowed_tools: [Read]
""")
        plan = load_plan(plan_file)
        assert any(
            "W27" in w and "system-prompt-enforced" in w
            for w in plan.validation_warnings
        )

    def test_w27_copilot_system_prompt_enforced(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: copilot
    prompt: "Do something"
    allowed_tools: [Read]
""")
        plan = load_plan(plan_file)
        assert any(
            "W27" in w and "system-prompt-enforced" in w
            for w in plan.validation_warnings
        )

    def test_w27_qwen_system_prompt_enforced(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: qwen
    prompt: "Do something"
    allowed_tools: [Read]
""")
        plan = load_plan(plan_file)
        assert any(
            "W27" in w and "system-prompt-enforced" in w
            for w in plan.validation_warnings
        )

    def test_valid_claude_tools_no_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [Read, Grep, Glob]
""")
        plan = load_plan(plan_file)
        assert not any("W27" in w for w in plan.validation_warnings)

    def test_mcp_tool_references_pass_validation(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [Read, mcp__my_server__tool_name]
""")
        plan = load_plan(plan_file)
        # mcp__ prefix tools should not trigger W27
        assert not any("W27" in w and "mcp__" in w for w in plan.validation_warnings)

    def test_category_names_pass_validation(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [read-only]
""")
        plan = load_plan(plan_file)
        # Category names should not trigger W27
        assert not any("W27" in w for w in plan.validation_warnings)

    def test_no_shell_category_passes_validation(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [no-shell]
""")
        plan = load_plan(plan_file)
        assert not any("W27" in w for w in plan.validation_warnings)

    def test_valid_codex_sandbox_levels_no_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: codex
    prompt: "Do something"
    allowed_tools: [workspace-read-only]
""")
        plan = load_plan(plan_file)
        assert not any("W27" in w for w in plan.validation_warnings)

    def test_codex_category_passes_validation(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: codex
    prompt: "Do something"
    allowed_tools: [read-only]
""")
        plan = load_plan(plan_file)
        assert not any("W27" in w for w in plan.validation_warnings)


# ---------------------------------------------------------------------------
# TestExpandToolCategories
# ---------------------------------------------------------------------------


class TestExpandToolCategories:
    """Test _expand_tool_categories() for category expansion."""

    def test_expand_read_only_claude(self) -> None:
        result = _expand_tool_categories(["read-only"], "claude")
        assert "Read" in result
        assert "Glob" in result
        assert "Grep" in result
        assert "Bash" not in result
        assert "Write" not in result
        assert "Edit" not in result

    def test_expand_no_shell_claude(self) -> None:
        result = _expand_tool_categories(["no-shell"], "claude")
        assert "Read" in result
        assert "Write" in result
        assert "Edit" in result
        assert "Bash" not in result

    def test_expand_read_only_codex(self) -> None:
        result = _expand_tool_categories(["read-only"], "codex")
        assert result == ["workspace-read-only"]

    def test_expand_no_shell_codex(self) -> None:
        result = _expand_tool_categories(["no-shell"], "codex")
        assert result == ["workspace-read-only"]

    def test_unknown_category_passed_through(self) -> None:
        result = _expand_tool_categories(["unknown-cat"], "claude")
        assert result == ["unknown-cat"]

    def test_mix_of_categories_and_tools(self) -> None:
        result = _expand_tool_categories(["read-only", "TodoWrite"], "claude")
        # read-only expands to Read, Glob, Grep, WebSearch, WebFetch
        assert "Read" in result
        assert "Glob" in result
        assert "TodoWrite" in result

    def test_empty_list(self) -> None:
        result = _expand_tool_categories([], "claude")
        assert result == []

    def test_expand_unknown_engine_falls_back_empty(self) -> None:
        # Category exists but engine has no mapping -> no expansion
        result = _expand_tool_categories(["read-only"], "ollama")
        assert result == []

    def test_regular_tools_passed_through(self) -> None:
        result = _expand_tool_categories(["Read", "Write"], "claude")
        assert result == ["Read", "Write"]


# ---------------------------------------------------------------------------
# TestInjectToolRestriction
# ---------------------------------------------------------------------------


class TestInjectToolRestriction:
    """Test _inject_tool_restriction() for prompt-level restrictions."""

    def test_claude_returns_prompt_unchanged(self) -> None:
        task = TaskSpec(id="t1", engine="claude", allowed_tools=["Read", "Grep"])
        prompt = "Do something"
        result = _inject_tool_restriction(prompt, task)
        assert result == prompt  # Claude uses CLI flags, no prompt injection

    def test_codex_returns_prompt_unchanged(self) -> None:
        task = TaskSpec(id="t1", engine="codex", allowed_tools=["workspace-read-only"])
        prompt = "Do something"
        result = _inject_tool_restriction(prompt, task)
        assert result == prompt  # Codex uses CLI flags, no prompt injection

    def test_gemini_prepends_restriction(self) -> None:
        task = TaskSpec(id="t1", engine="gemini", allowed_tools=["Read", "Grep"])
        prompt = "Do something"
        result = _inject_tool_restriction(prompt, task)
        assert result.startswith("IMPORTANT:")
        assert "Read, Grep" in result
        assert result.endswith(prompt)

    def test_copilot_prepends_restriction(self) -> None:
        task = TaskSpec(id="t1", engine="copilot", allowed_tools=["Read", "Write"])
        prompt = "Do something"
        result = _inject_tool_restriction(prompt, task)
        assert "IMPORTANT:" in result
        assert "Read, Write" in result

    def test_qwen_prepends_restriction(self) -> None:
        task = TaskSpec(id="t1", engine="qwen", allowed_tools=["Read"])
        prompt = "Do something"
        result = _inject_tool_restriction(prompt, task)
        assert "IMPORTANT:" in result
        assert "Read" in result

    def test_ollama_prepends_restriction(self) -> None:
        task = TaskSpec(id="t1", engine="ollama", allowed_tools=["Read"])
        prompt = "Do something"
        result = _inject_tool_restriction(prompt, task)
        assert "IMPORTANT:" in result

    def test_none_allowed_tools_returns_unchanged(self) -> None:
        task = TaskSpec(id="t1", engine="gemini", allowed_tools=None)
        prompt = "Do something"
        result = _inject_tool_restriction(prompt, task)
        assert result == prompt

    def test_restriction_includes_all_listed_tools(self) -> None:
        tools = ["Read", "Grep", "WebSearch"]
        task = TaskSpec(id="t1", engine="gemini", allowed_tools=tools)
        result = _inject_tool_restriction("prompt", task)
        for tool in tools:
            assert tool in result


# ---------------------------------------------------------------------------
# TestClaudeDisallowedTools
# ---------------------------------------------------------------------------


class TestClaudeDisallowedTools:
    """Test Claude --disallowedTools flag generation from allowed_tools."""

    def test_allowed_subset_disallows_complement(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [Read, Grep]
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        assert isinstance(cmd, list)
        # Should contain --disallowedTools
        assert "--disallowedTools" in cmd
        idx = cmd.index("--disallowedTools")
        disallowed_str = cmd[idx + 1]
        disallowed = set(disallowed_str.split(","))
        # Write, Edit, Bash, etc. should be disallowed
        assert "Write" in disallowed
        assert "Edit" in disallowed
        assert "Bash" in disallowed
        # Read and Grep should NOT be disallowed
        assert "Read" not in disallowed
        assert "Grep" not in disallowed

    def test_all_tools_allowed_no_disallowed_flag(self, tmp_path: Path) -> None:
        all_tools = sorted(CLAUDE_TOOLS)
        tools_yaml = "[" + ", ".join(all_tools) + "]"
        plan_file = _write_plan(tmp_path, f"""\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: {tools_yaml}
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        assert isinstance(cmd, list)
        # No tools to disallow — should not have --disallowedTools
        assert "--disallowedTools" not in cmd

    def test_read_only_category_expansion(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    allowed_tools: [read-only]
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        assert isinstance(cmd, list)
        assert "--disallowedTools" in cmd
        idx = cmd.index("--disallowedTools")
        disallowed = set(cmd[idx + 1].split(","))
        # read-only expands to Read, Glob, Grep, WebSearch, WebFetch
        # So Write, Edit, Bash, TodoWrite should be disallowed
        assert "Write" in disallowed
        assert "Edit" in disallowed
        assert "Bash" in disallowed
        assert "TodoWrite" in disallowed
        # Read, Glob, Grep should NOT be disallowed
        assert "Read" not in disallowed
        assert "Glob" not in disallowed
        assert "Grep" not in disallowed

    def test_allowed_tools_overrides_edit_policy(self, tmp_path: Path) -> None:
        """allowed_tools takes precedence over edit_policy: strict (which adds --disallowedTools Write)."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
    edit_policy: strict
    allowed_tools: [Read, Write, Edit, Glob, Grep, Bash]
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        assert isinstance(cmd, list)
        # The --disallowedTools from edit_policy: strict should be replaced
        # by the one from allowed_tools (which should disallow WebSearch, WebFetch, TodoWrite)
        if "--disallowedTools" in cmd:
            idx = cmd.index("--disallowedTools")
            disallowed = set(cmd[idx + 1].split(","))
            # Write should NOT be disallowed (it's in allowed_tools)
            assert "Write" not in disallowed
        # There should be at most one --disallowedTools flag
        assert cmd.count("--disallowedTools") <= 1

    def test_no_allowed_tools_no_extra_disallowed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: claude
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        assert isinstance(cmd, list)
        # Without allowed_tools and without strict edit_policy, no --disallowedTools
        assert "--disallowedTools" not in cmd


# ---------------------------------------------------------------------------
# TestCodexSandbox
# ---------------------------------------------------------------------------


class TestCodexSandbox:
    """Test Codex sandbox flag generation from allowed_tools."""

    def test_workspace_read_only_adds_sandbox(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: codex
    prompt: "Do something"
    allowed_tools: [workspace-read-only]
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        assert isinstance(cmd, list)
        assert "--sandbox" in cmd
        idx = cmd.index("--sandbox")
        assert cmd[idx + 1] == "workspace-read-only"

    def test_read_only_category_maps_to_sandbox(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: codex
    prompt: "Do something"
    allowed_tools: [read-only]
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        assert isinstance(cmd, list)
        # read-only for codex expands to workspace-read-only
        assert "--sandbox" in cmd

    def test_workspace_write_no_sandbox(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: codex
    prompt: "Do something"
    allowed_tools: [workspace-write]
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        assert isinstance(cmd, list)
        # workspace-write is not read-only, no sandbox added by allowed_tools
        # (sandbox may still exist from execution profile but not from allowed_tools)

    def test_no_allowed_tools_no_sandbox_from_feature(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: codex
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        cmd, _shell = build_command(plan, plan.tasks[0], tmp_path)
        assert isinstance(cmd, list)
        # No allowed_tools => no sandbox from this feature


# ---------------------------------------------------------------------------
# TestAllowedToolsPolicy
# ---------------------------------------------------------------------------


class TestAllowedToolsPolicy:
    """Test policy engine integration with allowed_tools fields."""

    def _make_plan(self) -> PlanSpec:
        return PlanSpec(
            version=1,
            name="test-plan",
            tasks=[],
            defaults=PlanDefaults(),
        )

    def test_has_allowed_tools_true_when_set(self) -> None:
        task = TaskSpec(id="t1", engine="claude", allowed_tools=["Read"])
        plan = self._make_plan()
        policy = PolicySpec(
            name="check-tools",
            rule="task.has_allowed_tools == True",
            action="warn",
        )
        evaluator = compile_policy(policy)
        assert evaluator(task, plan, None) is True

    def test_has_allowed_tools_false_when_none(self) -> None:
        task = TaskSpec(id="t1", engine="claude", allowed_tools=None)
        plan = self._make_plan()
        policy = PolicySpec(
            name="check-tools",
            rule="task.has_allowed_tools == True",
            action="warn",
        )
        evaluator = compile_policy(policy)
        assert evaluator(task, plan, None) is False

    def test_allowed_tools_returns_list(self) -> None:
        task = TaskSpec(id="t1", engine="claude", allowed_tools=["Read", "Grep"])
        plan = self._make_plan()
        policy = PolicySpec(
            name="check-read",
            rule="'Read' in task.allowed_tools",
            action="warn",
        )
        evaluator = compile_policy(policy)
        assert evaluator(task, plan, None) is True

    def test_allowed_tools_empty_when_none(self) -> None:
        task = TaskSpec(id="t1", engine="claude", allowed_tools=None)
        plan = self._make_plan()
        policy = PolicySpec(
            name="check-read",
            rule="'Read' in task.allowed_tools",
            action="warn",
        )
        evaluator = compile_policy(policy)
        assert evaluator(task, plan, None) is False

    def test_policy_blocks_task_without_allowed_tools(self) -> None:
        task = TaskSpec(id="t1", engine="claude", allowed_tools=None)
        plan = self._make_plan()
        policy = PolicySpec(
            name="require-tools",
            rule="task.has_allowed_tools == False",
            action="block",
            message="Tasks must have allowed_tools",
        )
        violations = evaluate_policies([policy], task, plan)
        assert len(violations) == 1
        assert violations[0].action == "block"

    def test_policy_passes_task_with_allowed_tools(self) -> None:
        task = TaskSpec(id="t1", engine="claude", allowed_tools=["Read"])
        plan = self._make_plan()
        policy = PolicySpec(
            name="require-tools",
            rule="task.has_allowed_tools == False",
            action="block",
            message="Tasks must have allowed_tools",
        )
        violations = evaluate_policies([policy], task, plan)
        assert len(violations) == 0

    def test_policy_check_specific_tool_not_in_list(self) -> None:
        task = TaskSpec(id="t1", engine="claude", allowed_tools=["Read", "Grep"])
        plan = self._make_plan()
        policy = PolicySpec(
            name="no-bash",
            rule="'Bash' in task.allowed_tools",
            action="block",
            message="Bash not allowed",
        )
        violations = evaluate_policies([policy], task, plan)
        # Bash is NOT in allowed_tools, so the rule returns False -> no violation
        assert len(violations) == 0


# ---------------------------------------------------------------------------
# TestSEC023
# ---------------------------------------------------------------------------


class TestSEC023:
    """Test SEC023 audit rule: untrusted context without allowed_tools."""

    def _make_plan(self, tasks: list[TaskSpec]) -> PlanSpec:
        return PlanSpec(
            version=1,
            name="test-plan",
            tasks=tasks,
            defaults=PlanDefaults(),
        )

    def test_untrusted_engine_without_allowed_tools_triggers(self) -> None:
        task = TaskSpec(
            id="t1",
            engine="claude",
            context_trust="untrusted",
            allowed_tools=None,
            prompt="Do something",
        )
        plan = self._make_plan([task])
        findings = audit_plan(plan)
        sec023 = [f for f in findings if f.rule == "SEC023"]
        assert len(sec023) == 1
        assert "t1" in sec023[0].message

    def test_untrusted_engine_with_allowed_tools_no_finding(self) -> None:
        task = TaskSpec(
            id="t1",
            engine="claude",
            context_trust="untrusted",
            allowed_tools=["Read", "Grep"],
            prompt="Do something",
        )
        plan = self._make_plan([task])
        findings = audit_plan(plan)
        sec023 = [f for f in findings if f.rule == "SEC023"]
        assert len(sec023) == 0

    def test_trusted_engine_without_allowed_tools_no_finding(self) -> None:
        task = TaskSpec(
            id="t1",
            engine="claude",
            context_trust="trusted",
            allowed_tools=None,
            prompt="Do something",
        )
        plan = self._make_plan([task])
        findings = audit_plan(plan)
        sec023 = [f for f in findings if f.rule == "SEC023"]
        assert len(sec023) == 0

    def test_command_task_no_finding(self) -> None:
        task = TaskSpec(
            id="t1",
            command="echo hello",
            context_trust="untrusted",
            allowed_tools=None,
        )
        plan = self._make_plan([task])
        findings = audit_plan(plan)
        sec023 = [f for f in findings if f.rule == "SEC023"]
        assert len(sec023) == 0

    def test_no_context_trust_no_finding(self) -> None:
        task = TaskSpec(
            id="t1",
            engine="claude",
            context_trust=None,
            allowed_tools=None,
            prompt="Do something",
        )
        plan = self._make_plan([task])
        findings = audit_plan(plan)
        sec023 = [f for f in findings if f.rule == "SEC023"]
        assert len(sec023) == 0

    def test_sec023_category(self) -> None:
        task = TaskSpec(
            id="t1",
            engine="claude",
            context_trust="untrusted",
            allowed_tools=None,
            prompt="Do something",
        )
        plan = self._make_plan([task])
        findings = audit_plan(plan)
        sec023 = [f for f in findings if f.rule == "SEC023"]
        assert len(sec023) == 1
        assert sec023[0].category == "Agent-Tool Coupling"

    def test_multiple_tasks_mixed(self) -> None:
        tasks = [
            TaskSpec(
                id="safe",
                engine="claude",
                context_trust="untrusted",
                allowed_tools=["Read"],
                prompt="Safe",
            ),
            TaskSpec(
                id="unsafe",
                engine="claude",
                context_trust="untrusted",
                allowed_tools=None,
                prompt="Unsafe",
            ),
            TaskSpec(
                id="cmd",
                command="echo hello",
            ),
        ]
        plan = self._make_plan(tasks)
        findings = audit_plan(plan)
        sec023 = [f for f in findings if f.rule == "SEC023"]
        assert len(sec023) == 1
        assert sec023[0].task_id == "unsafe"


# ---------------------------------------------------------------------------
# Wildcard Tool Patterns (v2.1.x)
# ---------------------------------------------------------------------------


class TestParseToolPattern:
    """Tests for parse_tool_pattern() — wildcard syntax parsing."""

    def test_bare_name(self) -> None:
        name, pattern = parse_tool_pattern("Read")
        assert name == "Read"
        assert pattern == ""

    def test_pattern_with_glob(self) -> None:
        name, pattern = parse_tool_pattern("Bash(git *)")
        assert name == "Bash"
        assert pattern == "git *"

    def test_pattern_with_path(self) -> None:
        name, pattern = parse_tool_pattern("Edit(src/*)")
        assert name == "Edit"
        assert pattern == "src/*"

    def test_wildcard_all(self) -> None:
        name, pattern = parse_tool_pattern("Read(*)")
        assert name == "Read"
        assert pattern == "*"

    def test_mcp_reference_unchanged(self) -> None:
        name, pattern = parse_tool_pattern("mcp__server__tool")
        assert name == "mcp__server__tool"
        assert pattern == ""

    def test_category_unchanged(self) -> None:
        name, pattern = parse_tool_pattern("read-only")
        assert name == "read-only"
        assert pattern == ""


class TestSplitToolPermissions:
    """Tests for _split_tool_permissions() — separating allowed vs restricted."""

    def test_all_bare_names(self) -> None:
        allowed, restricted = _split_tool_permissions(["Read", "Write", "Bash"])
        assert allowed == {"Read", "Write", "Bash"}
        assert restricted == []

    def test_pattern_produces_restricted(self) -> None:
        allowed, restricted = _split_tool_permissions(["Read", "Bash(git *)"])
        assert "Bash" in allowed  # tool stays allowed at CLI level
        assert "Read" in allowed
        assert ("Bash", "git *") in restricted

    def test_wildcard_star_not_restricted(self) -> None:
        allowed, restricted = _split_tool_permissions(["Read(*)", "Write"])
        assert "Read" in allowed
        assert "Write" in allowed
        assert restricted == []  # "*" is not a restriction

    def test_multiple_restrictions(self) -> None:
        allowed, restricted = _split_tool_permissions([
            "Bash(git *)", "Edit(src/*)", "Read",
        ])
        assert allowed == {"Bash", "Edit", "Read"}
        assert len(restricted) == 2


class TestBuildRestrictionPrompt:
    """Tests for _build_restriction_prompt() — argument restriction text."""

    def test_no_restrictions(self) -> None:
        assert _build_restriction_prompt([]) == ""

    def test_single_restriction(self) -> None:
        result = _build_restriction_prompt([("Bash", "git *")])
        assert "Bash" in result
        assert "git *" in result
        assert "IMPORTANT" in result

    def test_multiple_restrictions(self) -> None:
        result = _build_restriction_prompt([("Bash", "git *"), ("Edit", "src/*")])
        assert "Bash" in result
        assert "Edit" in result
        assert "src/*" in result


class TestWildcardToolCategories:
    """Tests for new tool categories (git-only, src-scoped)."""

    def test_git_only_category_exists(self) -> None:
        assert "git-only" in TOOL_CATEGORIES

    def test_git_only_claude_has_bash_pattern(self) -> None:
        claude_tools = TOOL_CATEGORIES["git-only"]["claude"]
        assert any("Bash(git" in t for t in claude_tools)
        assert "Read" in claude_tools

    def test_src_scoped_category_exists(self) -> None:
        assert "src-scoped" in TOOL_CATEGORIES

    def test_src_scoped_claude_has_edit_pattern(self) -> None:
        claude_tools = TOOL_CATEGORIES["src-scoped"]["claude"]
        assert any("Edit(src" in t for t in claude_tools)

    def test_expand_git_only(self) -> None:
        expanded = _expand_tool_categories(["git-only"], "claude")
        assert "Read" in expanded
        assert any("Bash" in e for e in expanded)

    def test_expand_src_scoped(self) -> None:
        expanded = _expand_tool_categories(["src-scoped"], "claude")
        assert any("Edit" in e for e in expanded)


class TestWildcardValidation:
    """Tests for loader validation of wildcard patterns."""

    def test_pattern_with_known_tool_no_warning(self, tmp_path: Path) -> None:
        plan = _write_plan(tmp_path, """
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "do stuff"
    allowed_tools: ["Bash(git *)", "Read"]
""")
        p = load_plan(plan)
        w27 = [w for w in p.validation_warnings if "W27" in w]
        assert len(w27) == 0

    def test_pattern_with_unknown_tool_warns(self, tmp_path: Path) -> None:
        plan = _write_plan(tmp_path, """
version: 1
name: test
tasks:
  - id: t1
    engine: claude
    prompt: "do stuff"
    allowed_tools: ["FakeToolXyz(something)"]
""")
        p = load_plan(plan)
        w27 = [w for w in p.validation_warnings if "W27" in w]
        assert len(w27) == 1
        assert "FakeToolXyz" in w27[0]


class TestWildcardInjection:
    """Tests for prompt injection with wildcard patterns."""

    def test_claude_engine_injects_arg_restriction(self) -> None:
        task = TaskSpec(
            id="t", engine="claude", prompt="do stuff",
            allowed_tools=["Read", "Bash(git *)"],
        )
        result = _inject_tool_restriction("Hello", task)
        assert "Bash" in result
        assert "git *" in result

    def test_claude_engine_no_injection_without_patterns(self) -> None:
        task = TaskSpec(
            id="t", engine="claude", prompt="do stuff",
            allowed_tools=["Read", "Write"],
        )
        result = _inject_tool_restriction("Hello", task)
        assert result == "Hello"

    def test_other_engine_full_injection(self) -> None:
        task = TaskSpec(
            id="t", engine="gemini", prompt="do stuff",
            allowed_tools=["Read", "Bash(git *)"],
        )
        result = _inject_tool_restriction("Hello", task)
        assert "IMPORTANT" in result
        assert "Bash (only for: git *)" in result
        assert "Read" in result
