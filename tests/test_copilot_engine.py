from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import EngineDefaults, PlanDefaults, PlanSpec, TaskSpec
from maestro_cli.runners import (
    _COPILOT_MODEL_ALIASES,
    _apply_execution_profile,
    _normalize_copilot_args,
    _resolve_copilot_model,
    build_command,
)


def _write_plan(tmp_path: Path, content: str) -> Path:
    """Helper to write a plan YAML and return its path."""
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


# ---------------------------------------------------------------------------
# 1. Model resolution
# ---------------------------------------------------------------------------

class TestCopilotModelResolution:
    def test_alias_sonnet_resolves(self) -> None:
        assert _resolve_copilot_model("sonnet") == "claude-sonnet-4.6"

    def test_alias_opus_resolves(self) -> None:
        assert _resolve_copilot_model("opus") == "claude-opus-4.6"

    def test_alias_haiku_resolves(self) -> None:
        assert _resolve_copilot_model("haiku") == "claude-haiku-4.5"

    def test_full_model_name_passthrough(self) -> None:
        assert _resolve_copilot_model("claude-sonnet-4.6") == "claude-sonnet-4.6"

    def test_none_returns_none(self) -> None:
        assert _resolve_copilot_model(None) is None

    def test_unknown_model_passthrough(self) -> None:
        assert _resolve_copilot_model("future-model-99") == "future-model-99"

    @pytest.mark.parametrize("alias,full_name", list(_COPILOT_MODEL_ALIASES.items()))
    def test_all_aliases_resolve(self, alias: str, full_name: str) -> None:
        assert _resolve_copilot_model(alias) == full_name


# ---------------------------------------------------------------------------
# 2. Arg normalization
# ---------------------------------------------------------------------------

class TestNormalizeCopilotArgs:
    def test_empty_list(self) -> None:
        assert _normalize_copilot_args([]) == []

    def test_yolo_passthrough(self) -> None:
        assert _normalize_copilot_args(["--yolo"]) == ["--yolo"]

    def test_allow_all_normalized_to_yolo(self) -> None:
        assert _normalize_copilot_args(["--allow-all"]) == ["--yolo"]

    def test_deduplicates_yolo(self) -> None:
        result = _normalize_copilot_args(["--yolo", "--yolo"])
        assert result == ["--yolo"]

    def test_deduplicates_mixed_yolo_allow_all(self) -> None:
        result = _normalize_copilot_args(["--yolo", "--allow-all"])
        assert result == ["--yolo"]

    def test_mixed_args_preserved(self) -> None:
        result = _normalize_copilot_args(["--verbose", "--model", "opus"])
        assert result == ["--verbose", "--model", "opus"]

    def test_yolo_then_extra_args_ordering(self) -> None:
        result = _normalize_copilot_args(["--yolo", "--verbose"])
        assert result == ["--yolo", "--verbose"]


# ---------------------------------------------------------------------------
# 3. Execution profiles
# ---------------------------------------------------------------------------

class TestApplyExecutionProfileCopilot:
    def test_plan_profile_passthrough(self) -> None:
        args = ["--yolo", "--verbose"]
        result = _apply_execution_profile("copilot", args, "plan")
        assert result == args

    def test_safe_removes_yolo(self) -> None:
        args = ["--yolo", "--verbose"]
        result = _apply_execution_profile("copilot", args, "safe")
        assert "--yolo" not in result
        assert "--allow-all" not in result
        assert "--verbose" in result

    def test_safe_removes_allow_all(self) -> None:
        args = ["--allow-all"]
        result = _apply_execution_profile("copilot", args, "safe")
        assert "--allow-all" not in result

    def test_safe_removes_allow_all_tools(self) -> None:
        args = ["--allow-all-tools", "--verbose"]
        result = _apply_execution_profile("copilot", args, "safe")
        assert "--allow-all-tools" not in result
        assert "--verbose" in result

    def test_safe_removes_allow_all_paths(self) -> None:
        args = ["--allow-all-paths"]
        result = _apply_execution_profile("copilot", args, "safe")
        assert "--allow-all-paths" not in result

    def test_yolo_ensures_yolo_flag(self) -> None:
        args = ["--verbose"]
        result = _apply_execution_profile("copilot", args, "yolo")
        assert "--yolo" in result
        assert "--verbose" in result

    def test_yolo_no_duplicate_when_already_set(self) -> None:
        args = ["--yolo"]
        result = _apply_execution_profile("copilot", args, "yolo")
        assert result.count("--yolo") == 1

    def test_yolo_no_duplicate_with_allow_all(self) -> None:
        args = ["--allow-all"]
        result = _apply_execution_profile("copilot", args, "yolo")
        assert "--allow-all" in result or "--yolo" in result
        # Should not add duplicate
        total = result.count("--yolo") + result.count("--allow-all")
        assert total == 1


# ---------------------------------------------------------------------------
# 4. Command building
# ---------------------------------------------------------------------------

class TestBuildCommandCopilot:
    def _make_plan(self, **kwargs) -> PlanSpec:
        return PlanSpec(
            version=1,
            name="p",
            defaults=PlanDefaults(**kwargs),
            tasks=[],
        )

    def test_basic_copilot_task_no_model(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="copilot", prompt="Do stuff")
        cmd, shell = build_command(plan, task, Path("/tmp"))
        assert isinstance(cmd, list)
        assert not shell
        assert "copilot" in cmd
        assert "--autopilot" in cmd
        assert "--silent" in cmd
        assert "--no-color" in cmd
        assert "-p" in cmd
        idx = cmd.index("-p")
        assert cmd[idx + 1] == "Do stuff"

    def test_model_alias_resolved_in_command(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="copilot", model="sonnet", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-sonnet-4.6"

    def test_full_model_name_in_command(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="copilot", model="claude-opus-4.6", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4.6"

    def test_plan_defaults_model_applied(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan(copilot=EngineDefaults(model="opus"))
        task = TaskSpec(id="t", engine="copilot", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "--model" in cmd
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4.6"

    def test_plan_defaults_args_applied(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan(copilot=EngineDefaults(args=["--max-autopilot-continues", "10"]))
        task = TaskSpec(id="t", engine="copilot", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "--max-autopilot-continues" in cmd
        assert "10" in cmd

    def test_agent_flag_applied(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="copilot", agent="code-review", prompt="Review")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "--agent" in cmd
        idx = cmd.index("--agent")
        assert cmd[idx + 1] == "code-review"

    def test_no_model_flag_when_none(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="copilot", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "--model" not in cmd

    def test_system_prompt_prepended_to_prompt_text(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan(copilot=EngineDefaults(append_system_prompt="Be concise"))
        task = TaskSpec(id="t", engine="copilot", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        idx = cmd.index("-p")
        prompt_arg = cmd[idx + 1]
        assert "[System Instructions]" in prompt_arg
        assert "Be concise" in prompt_arg
        assert "[Task]" in prompt_arg
        assert "Do stuff" in prompt_arg

    def test_retry_feedback_prepended_to_prompt(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="copilot", prompt="Do stuff")
        feedback = "Fix the broken test"
        cmd, _ = build_command(plan, task, Path("/tmp"), retry_feedback=feedback)
        idx = cmd.index("-p")
        prompt_arg = cmd[idx + 1]
        assert "[System Instructions]" in prompt_arg
        assert "Fix the broken test" in prompt_arg
        assert "[Task]" in prompt_arg
        assert "Do stuff" in prompt_arg

    def test_edit_policy_efficient_injects_system_instructions(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan(edit_policy="efficient")
        task = TaskSpec(id="t", engine="copilot", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        idx = cmd.index("-p")
        prompt_arg = cmd[idx + 1]
        assert "[System Instructions]" in prompt_arg
        assert "surgical" in prompt_arg

    def test_task_model_overrides_plan_default(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan(copilot=EngineDefaults(model="sonnet"))
        task = TaskSpec(id="t", engine="copilot", model="opus", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        idx = cmd.index("--model")
        assert cmd[idx + 1] == "claude-opus-4.6"


# ---------------------------------------------------------------------------
# 5. Loader — defaults parsing
# ---------------------------------------------------------------------------

class TestLoaderCopilotDefaults:
    def test_copilot_model_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  copilot:
    model: sonnet
tasks:
  - id: t1
    engine: copilot
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.copilot.model == "sonnet"

    def test_copilot_args_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  copilot:
    args: ["--max-autopilot-continues", "10"]
tasks:
  - id: t1
    engine: copilot
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.copilot.args == ["--max-autopilot-continues", "10"]

    def test_without_copilot_defaults_uses_engine_defaults(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: copilot
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.copilot.model is None
        assert plan.defaults.copilot.args == []
        assert plan.defaults.copilot.append_system_prompt is None

    def test_copilot_append_system_prompt_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  copilot:
    append_system_prompt: "Always be concise"
tasks:
  - id: t1
    engine: copilot
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.copilot.append_system_prompt == "Always be concise"


# ---------------------------------------------------------------------------
# 6. Loader — validation
# ---------------------------------------------------------------------------

class TestLoaderCopilotValidation:
    def test_valid_copilot_task_loads(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  timeout_sec: 1800
tasks:
  - id: t1
    engine: copilot
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].engine == "copilot"
        assert len(plan.validation_warnings) == 0

    def test_unknown_model_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  timeout_sec: 1800
tasks:
  - id: t1
    engine: copilot
    model: unknown-future-model
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any("not a known alias" in w for w in plan.validation_warnings)

    def test_known_model_no_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  timeout_sec: 1800
tasks:
  - id: t1
    engine: copilot
    model: opus
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        model_warnings = [w for w in plan.validation_warnings if "Copilot model" in w]
        assert len(model_warnings) == 0

    def test_reasoning_effort_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  timeout_sec: 1800
tasks:
  - id: t1
    engine: copilot
    reasoning_effort: high
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any("does not support reasoning_effort" in w for w in plan.validation_warnings)

    def test_defaults_reasoning_effort_warns(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  timeout_sec: 1800
  copilot:
    reasoning_effort: high
tasks:
  - id: t1
    engine: copilot
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any("defaults.copilot.reasoning_effort" in w for w in plan.validation_warnings)

    def test_copilot_task_requires_prompt(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: copilot
""")
        with pytest.raises(PlanValidationError, match="no prompt source"):
            load_plan(plan_file)


# ---------------------------------------------------------------------------
# 7. Cache — engine config hashing
# ---------------------------------------------------------------------------

class TestCacheCopilot:
    def test_effective_engine_config_resolves_model(self) -> None:
        from maestro_cli.cache import _effective_engine_config

        plan = PlanSpec(
            version=1,
            name="p",
            defaults=PlanDefaults(copilot=EngineDefaults(model="sonnet")),
            tasks=[],
        )
        task = TaskSpec(id="t", engine="copilot", prompt="Do stuff")
        config = _effective_engine_config(task, plan)
        assert config["engine"] == "copilot"
        assert config["model"] == "claude-sonnet-4.6"
        assert config["reasoning_effort"] is None

    def test_effective_engine_config_task_model_overrides(self) -> None:
        from maestro_cli.cache import _effective_engine_config

        plan = PlanSpec(
            version=1,
            name="p",
            defaults=PlanDefaults(copilot=EngineDefaults(model="sonnet")),
            tasks=[],
        )
        task = TaskSpec(id="t", engine="copilot", model="opus", prompt="Do stuff")
        config = _effective_engine_config(task, plan)
        assert config["model"] == "claude-opus-4.6"

    def test_effective_engine_config_normalizes_args(self) -> None:
        from maestro_cli.cache import _effective_engine_config

        plan = PlanSpec(
            version=1,
            name="p",
            defaults=PlanDefaults(copilot=EngineDefaults(args=["--yolo", "--allow-all"])),
            tasks=[],
        )
        task = TaskSpec(id="t", engine="copilot", prompt="Do stuff")
        config = _effective_engine_config(task, plan)
        assert config["args"].count("--yolo") == 1


# ---------------------------------------------------------------------------
# 8. Cost backfill — engine detection
# ---------------------------------------------------------------------------

class TestCostBackfillCopilot:
    def test_infer_engine_detects_copilot(self) -> None:
        from maestro_cli.cost_backfill import _infer_engine

        assert _infer_engine("copilot --autopilot -p foo") == "copilot"
        assert _infer_engine("COPILOT --silent") == "copilot"

    def test_infer_engine_copilot_startswith(self) -> None:
        from maestro_cli.cost_backfill import _infer_engine

        assert _infer_engine("copilot") == "copilot"
