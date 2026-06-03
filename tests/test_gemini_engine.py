from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import EngineDefaults, PlanDefaults, PlanSpec, TaskSpec
from maestro_cli.runners import (
    _ENV_ALLOWLIST,
    _GEMINI_MODEL_ALIASES,
    _apply_execution_profile,
    _normalize_gemini_args,
    _resolve_gemini_model,
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

class TestGeminiModelResolution:
    def test_alias_pro_resolves(self) -> None:
        assert _resolve_gemini_model("pro") == "gemini-2.5-pro"

    def test_full_model_name_passthrough(self) -> None:
        assert _resolve_gemini_model("gemini-2.5-pro") == "gemini-2.5-pro"

    def test_none_returns_none(self) -> None:
        assert _resolve_gemini_model(None) is None

    def test_unknown_model_passthrough(self) -> None:
        assert _resolve_gemini_model("gemini-99-future") == "gemini-99-future"

    @pytest.mark.parametrize("alias,full_name", list(_GEMINI_MODEL_ALIASES.items()))
    def test_all_aliases_resolve(self, alias: str, full_name: str) -> None:
        assert _resolve_gemini_model(alias) == full_name


# ---------------------------------------------------------------------------
# 2. Arg normalization
# ---------------------------------------------------------------------------

class TestNormalizeGeminiArgs:
    def test_empty_list(self) -> None:
        assert _normalize_gemini_args([]) == []

    def test_yolo_expands_to_approval_mode(self) -> None:
        result = _normalize_gemini_args(["--yolo"])
        assert result == ["--approval-mode", "yolo"]

    def test_approval_mode_yolo_unchanged(self) -> None:
        result = _normalize_gemini_args(["--approval-mode", "yolo"])
        assert result == ["--approval-mode", "yolo"]

    def test_deduplicates_approval_mode(self) -> None:
        result = _normalize_gemini_args(
            ["--approval-mode", "yolo", "--approval-mode", "yolo"]
        )
        assert result == ["--approval-mode", "yolo"]

    def test_mixed_args_preserved(self) -> None:
        result = _normalize_gemini_args(["--verbose", "--output-format", "json"])
        assert result == ["--verbose", "--output-format", "json"]

    def test_yolo_then_extra_args_ordering(self) -> None:
        result = _normalize_gemini_args(["--yolo", "--verbose"])
        assert result == ["--approval-mode", "yolo", "--verbose"]


# ---------------------------------------------------------------------------
# 3. Execution profiles
# ---------------------------------------------------------------------------

class TestApplyExecutionProfileGemini:
    def test_plan_profile_passthrough(self) -> None:
        args = ["--approval-mode", "yolo", "--verbose"]
        result = _apply_execution_profile("gemini", args, "plan")
        assert result == args

    def test_safe_removes_approval_mode_adds_sandbox(self) -> None:
        args = ["--approval-mode", "yolo"]
        result = _apply_execution_profile("gemini", args, "safe")
        assert "--approval-mode" not in result
        assert "yolo" not in result
        assert "--sandbox" in result

    def test_safe_removes_yolo_shorthand_adds_sandbox(self) -> None:
        args = ["--yolo", "--verbose"]
        result = _apply_execution_profile("gemini", args, "safe")
        assert "--yolo" not in result
        assert "--sandbox" in result
        assert "--verbose" in result

    def test_yolo_ensures_approval_mode_yolo(self) -> None:
        args = ["--verbose"]
        result = _apply_execution_profile("gemini", args, "yolo")
        assert "--approval-mode" in result
        idx = result.index("--approval-mode")
        assert result[idx + 1] == "yolo"

    def test_yolo_no_duplicate_when_already_set(self) -> None:
        args = ["--approval-mode", "yolo"]
        result = _apply_execution_profile("gemini", args, "yolo")
        assert result.count("--approval-mode") == 1


# ---------------------------------------------------------------------------
# 4. Command building
# ---------------------------------------------------------------------------

class TestBuildCommandGemini:
    def _make_plan(self, **kwargs) -> PlanSpec:
        return PlanSpec(
            version=1,
            name="p",
            defaults=PlanDefaults(**kwargs),
            tasks=[],
        )

    def test_basic_gemini_task_no_model(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="gemini", prompt="Do stuff")
        cmd, shell = build_command(plan, task, Path("/tmp"))
        assert isinstance(cmd, list)
        assert not shell
        assert "gemini" in cmd
        assert "--output-format" in cmd
        assert "json" in cmd
        assert "-m" not in cmd
        assert "Do stuff" in cmd

    def test_model_alias_resolved_in_command(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="gemini", model="flash", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "-m" in cmd
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "gemini-2.5-flash"

    def test_full_model_name_in_command(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="gemini", model="gemini-2.5-pro", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "-m" in cmd
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "gemini-2.5-pro"

    def test_plan_defaults_model_applied(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan(gemini=EngineDefaults(model="pro"))
        task = TaskSpec(id="t", engine="gemini", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "-m" in cmd
        idx = cmd.index("-m")
        assert cmd[idx + 1] == "gemini-2.5-pro"

    def test_plan_defaults_args_applied(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan(gemini=EngineDefaults(args=["--verbose"]))
        task = TaskSpec(id="t", engine="gemini", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "--verbose" in cmd

    def test_system_prompt_prepended_to_prompt_text(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan(gemini=EngineDefaults(append_system_prompt="Be concise"))
        task = TaskSpec(id="t", engine="gemini", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        last_arg = cmd[-1]
        assert "[System Instructions]" in last_arg
        assert "Be concise" in last_arg
        assert "[Task]" in last_arg
        assert "Do stuff" in last_arg

    def test_retry_feedback_prepended_to_prompt(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="gemini", prompt="Do stuff")
        feedback = "Fix the broken test"
        cmd, _ = build_command(plan, task, Path("/tmp"), retry_feedback=feedback)
        last_arg = cmd[-1]
        assert "[System Instructions]" in last_arg
        assert "Fix the broken test" in last_arg
        assert "[Task]" in last_arg
        assert "Do stuff" in last_arg

    def test_edit_policy_efficient_injects_system_instructions(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan(edit_policy="efficient")
        task = TaskSpec(id="t", engine="gemini", prompt="Do stuff")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        last_arg = cmd[-1]
        assert "[System Instructions]" in last_arg
        assert "surgical" in last_arg

    def test_append_system_prompt_from_defaults(self, monkeypatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan(gemini=EngineDefaults(append_system_prompt="Gemini rules"))
        task = TaskSpec(id="t", engine="gemini", prompt="Do task")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        last_arg = cmd[-1]
        assert "Gemini rules" in last_arg


# ---------------------------------------------------------------------------
# 5. Loader — defaults parsing
# ---------------------------------------------------------------------------

class TestLoaderGeminiDefaults:
    def test_gemini_model_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  gemini:
    model: flash
tasks:
  - id: t1
    engine: gemini
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.gemini.model == "flash"

    def test_gemini_args_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  gemini:
    args: ["--verbose", "--debug"]
tasks:
  - id: t1
    engine: gemini
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.gemini.args == ["--verbose", "--debug"]

    def test_without_gemini_defaults_uses_engine_defaults(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: gemini
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.gemini.model is None
        assert plan.defaults.gemini.args == []
        assert plan.defaults.gemini.append_system_prompt is None

    def test_invalid_gemini_defaults_not_dict_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  gemini: "not-a-dict"
tasks:
  - id: t1
    engine: gemini
    prompt: "Do something"
""")
        with pytest.raises(
            PlanValidationError,
            match=r"defaults\.codex, defaults\.claude, defaults\.gemini, defaults\.copilot, defaults\.qwen, defaults\.ollama and defaults\.llama must be objects",
        ):
            load_plan(plan_file)

    def test_gemini_append_system_prompt_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  gemini:
    append_system_prompt: "Always be concise"
tasks:
  - id: t1
    engine: gemini
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.gemini.append_system_prompt == "Always be concise"


# ---------------------------------------------------------------------------
# 6. Loader — validation
# ---------------------------------------------------------------------------

class TestLoaderGeminiValidation:
    def test_valid_gemini_task_loads(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
defaults:
  timeout_sec: 1800
tasks:
  - id: t1
    engine: gemini
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].engine == "gemini"
        assert len(plan.validation_warnings) == 0

    def test_unknown_gemini_model_generates_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: gemini
    model: "totally-unknown-model"
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any("may not be valid" in w for w in plan.validation_warnings)

    def test_known_gemini_model_no_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: gemini
    model: flash
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        model_warnings = [w for w in plan.validation_warnings if "may not be valid" in w]
        assert len(model_warnings) == 0

    def test_full_gemini_model_name_no_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: gemini
    model: gemini-2.5-pro
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        model_warnings = [w for w in plan.validation_warnings if "may not be valid" in w]
        assert len(model_warnings) == 0

    def test_reasoning_effort_on_gemini_task_generates_warning(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: gemini
    reasoning_effort: high
    prompt: "Do something"
""")
        plan = load_plan(plan_file)
        assert any(
            "Gemini CLI does not currently support reasoning_effort" in w
            for w in plan.validation_warnings
        )

    def test_defaults_gemini_reasoning_effort_generates_warning(self, tmp_path: Path) -> None:
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
            "defaults.gemini.reasoning_effort" in w
            for w in plan.validation_warnings
        )

    def test_gemini_without_prompt_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    engine: gemini
""")
        with pytest.raises(PlanValidationError, match="no prompt source"):
            load_plan(plan_file)


# ---------------------------------------------------------------------------
# 7. Environment allowlist
# ---------------------------------------------------------------------------

class TestGeminiEnvAllowlist:
    def test_gemini_api_key_in_allowlist(self) -> None:
        assert "GEMINI_API_KEY" in _ENV_ALLOWLIST

    def test_google_api_key_in_allowlist(self) -> None:
        assert "GOOGLE_API_KEY" in _ENV_ALLOWLIST

    def test_google_application_credentials_in_allowlist(self) -> None:
        assert "GOOGLE_APPLICATION_CREDENTIALS" in _ENV_ALLOWLIST
