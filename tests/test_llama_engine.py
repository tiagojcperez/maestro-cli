from __future__ import annotations

import sys
from pathlib import Path

import pytest

from maestro_cli.loader import load_plan
from maestro_cli.models import (
    EngineDefaults,
    EngineName,
    LLAMA_MODEL_ALIASES,
    LLAMA_MODELS,
    PlanDefaults,
    PlanSpec,
    TaskSpec,
)
from maestro_cli.runners import (
    _ENV_ALLOWLIST,
    _inject_tool_restriction,
    _resolve_llama_model,
    build_command,
)
from maestro_cli.routing import _MODEL_TIERS, resolve_auto_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


def _make_plan(llama_defaults: EngineDefaults | None = None) -> PlanSpec:
    defaults = PlanDefaults(
        llama=llama_defaults if llama_defaults is not None else EngineDefaults(),
    )
    return PlanSpec(version=1, name="test-plan", defaults=defaults, tasks=[])


# ===========================================================================
# TestLlamaModelAliases
# ===========================================================================


class TestLlamaModelAliases:
    """LLAMA_MODEL_ALIASES resolution via _resolve_llama_model."""

    def test_resolve_known_alias_llama3(self) -> None:
        assert _resolve_llama_model("llama3") == "llama-3-8b"

    def test_resolve_known_alias_llama3_1(self) -> None:
        assert _resolve_llama_model("llama3.1") == "llama-3.1-8b"

    def test_resolve_known_alias_llama3_2(self) -> None:
        assert _resolve_llama_model("llama3.2") == "llama-3.2-3b"

    def test_resolve_known_alias_codellama(self) -> None:
        assert _resolve_llama_model("codellama") == "codellama-13b"

    def test_resolve_known_alias_phi3(self) -> None:
        assert _resolve_llama_model("phi3") == "phi-3-mini"

    def test_resolve_known_alias_mistral(self) -> None:
        assert _resolve_llama_model("mistral") == "mistral-7b"

    def test_resolve_known_alias_qwen25_coder(self) -> None:
        assert _resolve_llama_model("qwen2.5-coder") == "qwen2.5-coder-7b"

    def test_all_aliases_resolve(self) -> None:
        """Every key in LLAMA_MODEL_ALIASES must resolve via _resolve_llama_model."""
        for alias, expected in LLAMA_MODEL_ALIASES.items():
            assert _resolve_llama_model(alias) == expected

    def test_unknown_model_passes_through(self) -> None:
        assert _resolve_llama_model("custom-gguf-model") == "custom-gguf-model"

    def test_unknown_model_with_tag_passes_through(self) -> None:
        assert _resolve_llama_model("my-finetune:Q4_K_M") == "my-finetune:Q4_K_M"

    def test_none_returns_none(self) -> None:
        assert _resolve_llama_model(None) is None

    def test_llama_models_constant(self) -> None:
        assert "llama3" in LLAMA_MODELS
        assert "codellama" in LLAMA_MODELS
        assert "mistral" in LLAMA_MODELS
        assert "phi3" in LLAMA_MODELS


# ===========================================================================
# TestBuildLlamaCommand
# ===========================================================================


class TestBuildLlamaCommand:
    """Command construction for the llama engine via build_command()."""

    def test_basic_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        monkeypatch.delenv("LLAMA_MODEL_DIR", raising=False)
        plan = _make_plan()
        task = TaskSpec(id="t", engine="llama", model="llama3", prompt="Say hello")
        cmd, shell = build_command(plan, task, Path("/tmp"))
        assert isinstance(cmd, list)
        assert cmd[0] == "llama-cli"
        assert "-m" in cmd
        model_idx = cmd.index("-m")
        assert cmd[model_idx + 1] == "llama-3-8b"
        assert "-p" in cmd
        prompt_idx = cmd.index("-p")
        assert cmd[prompt_idx + 1] == "Say hello"
        assert "--no-display-prompt" in cmd
        assert shell is False

    def test_default_model_when_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When no model is set at task or defaults, -m is omitted."""
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        monkeypatch.delenv("LLAMA_MODEL_DIR", raising=False)
        plan = _make_plan()
        task = TaskSpec(id="t", engine="llama", prompt="Say hello")
        cmd, shell = build_command(plan, task, Path("/tmp"))
        assert cmd[0] == "llama-cli"
        # No model → -m should not be present
        assert "-m" not in cmd
        assert "-p" in cmd
        assert "--no-display-prompt" in cmd
        assert shell is False

    def test_defaults_model_used(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        monkeypatch.delenv("LLAMA_MODEL_DIR", raising=False)
        plan = _make_plan(EngineDefaults(model="mistral"))
        task = TaskSpec(id="t", engine="llama", prompt="Hello")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        model_idx = cmd.index("-m")
        assert cmd[model_idx + 1] == "mistral-7b"

    def test_task_model_overrides_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        monkeypatch.delenv("LLAMA_MODEL_DIR", raising=False)
        plan = _make_plan(EngineDefaults(model="mistral"))
        task = TaskSpec(id="t", engine="llama", model="codellama", prompt="Code")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        model_idx = cmd.index("-m")
        assert cmd[model_idx + 1] == "codellama-13b"

    def test_llama_model_dir_prepends_to_relative(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        monkeypatch.setenv("LLAMA_MODEL_DIR", "/models/gguf")
        plan = _make_plan()
        task = TaskSpec(id="t", engine="llama", model="llama3", prompt="Hi")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        model_idx = cmd.index("-m")
        model_val = cmd[model_idx + 1]
        # Alias resolves first (llama3 -> llama-3-8b), then LLAMA_MODEL_DIR is prepended
        assert "llama-3-8b" in model_val
        assert model_val == str(Path("/models/gguf") / "llama-3-8b")

    def test_absolute_model_path_bypasses_model_dir(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        # Use an OS-appropriate absolute path so Path.is_absolute() is True on
        # the running platform (a "C:\..." path is NOT absolute on POSIX).
        if sys.platform == "win32":
            model_dir = "C:\\models\\gguf"
            abs_path = "C:\\opt\\custom\\my-model.gguf"
        else:
            model_dir = "/models/gguf"
            abs_path = "/opt/custom/my-model.gguf"
        monkeypatch.setenv("LLAMA_MODEL_DIR", model_dir)
        # Use an absolute path as the model name (raw, not an alias)
        plan = _make_plan()
        task = TaskSpec(id="t", engine="llama", model=abs_path, prompt="Hi")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        model_idx = cmd.index("-m")
        # Absolute path should NOT have LLAMA_MODEL_DIR prepended
        assert cmd[model_idx + 1] == abs_path

    def test_task_args_appended(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        monkeypatch.delenv("LLAMA_MODEL_DIR", raising=False)
        plan = _make_plan()
        task = TaskSpec(
            id="t",
            engine="llama",
            model="llama3",
            prompt="Hello",
            args=["--ctx-size", "4096", "--threads", "8"],
        )
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "--ctx-size" in cmd
        assert "4096" in cmd
        assert "--threads" in cmd
        assert "8" in cmd

    def test_no_display_prompt_always_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        monkeypatch.delenv("LLAMA_MODEL_DIR", raising=False)
        plan = _make_plan()
        task = TaskSpec(id="t", engine="llama", model="phi3", prompt="Test")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        assert "--no-display-prompt" in cmd

    def test_shell_is_always_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        monkeypatch.delenv("LLAMA_MODEL_DIR", raising=False)
        plan = _make_plan()
        task = TaskSpec(id="t", engine="llama", model="llama3", prompt="Hi")
        _, shell = build_command(plan, task, Path("/tmp"))
        assert shell is False

    def test_empty_llama_model_dir_no_prepend(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When LLAMA_MODEL_DIR is empty string, no prepending occurs."""
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        monkeypatch.setenv("LLAMA_MODEL_DIR", "")
        plan = _make_plan()
        task = TaskSpec(id="t", engine="llama", model="llama3", prompt="Hi")
        cmd, _ = build_command(plan, task, Path("/tmp"))
        model_idx = cmd.index("-m")
        # Should just be the resolved alias, no path prefix
        assert cmd[model_idx + 1] == "llama-3-8b"


# ===========================================================================
# TestLlamaExecutionProfile
# ===========================================================================


class TestLlamaExecutionProfile:
    """Execution profiles pass through unchanged for llama (local engine)."""

    @pytest.mark.parametrize("profile", ["plan", "safe", "yolo"])
    def test_profile_does_not_alter_command(
        self, monkeypatch: pytest.MonkeyPatch, profile: str
    ) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        monkeypatch.delenv("LLAMA_MODEL_DIR", raising=False)
        plan = _make_plan()
        task = TaskSpec(id="t", engine="llama", model="llama3", prompt="Hello")
        cmd, shell = build_command(plan, task, Path("/tmp"), execution_profile=profile)
        assert cmd[0] == "llama-cli"
        assert "-m" in cmd
        assert "--no-display-prompt" in cmd
        assert shell is False


# ===========================================================================
# TestLlamaZeroCost
# ===========================================================================


class TestLlamaZeroCost:
    """Llama engine uses local execution — zero API cost."""

    def test_no_pricing_table(self) -> None:
        """Llama engine plugin has no load_pricing_table (zero cost)."""
        from maestro_cli.runners import _get_registered_engine_plugin

        plugin = _get_registered_engine_plugin("llama")
        assert plugin is not None
        # No pricing loader → zero cost
        assert plugin.load_pricing_table is None

    def test_doctor_probe_executable(self) -> None:
        """Llama engine plugin uses llama-cli as its doctor probe."""
        from maestro_cli.runners import _get_registered_engine_plugin

        plugin = _get_registered_engine_plugin("llama")
        assert plugin is not None
        assert plugin.doctor_probe is not None
        assert plugin.doctor_probe.executable == "llama-cli"


# ===========================================================================
# TestLlamaLoaderParsing
# ===========================================================================


class TestLlamaLoaderParsing:
    """YAML parsing and validation for llama engine tasks."""

    def test_valid_llama_plan_loads(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: llama-plan
tasks:
  - id: ask
    engine: llama
    model: llama3
    prompt: "What is 2+2?"
""")
        plan = load_plan(plan_file)
        task = plan.tasks[0]
        assert task.engine == "llama"
        assert task.model == "llama3"

    def test_llama_defaults_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: llama-defaults
defaults:
  llama:
    model: codellama
    args: ["--ctx-size", "8192"]
tasks:
  - id: code
    engine: llama
    prompt: "Write a function"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.llama.model == "codellama"
        assert plan.defaults.llama.args == ["--ctx-size", "8192"]

    def test_defaults_model_inherited_by_task(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: llama-inherit
defaults:
  llama:
    model: mistral
tasks:
  - id: ask
    engine: llama
    prompt: "Hello"
""")
        plan = load_plan(plan_file)
        # Task has no model, should inherit from defaults at command build time
        assert plan.tasks[0].model is None
        assert plan.defaults.llama.model == "mistral"

    def test_engine_name_includes_llama(self) -> None:
        assert "llama" in EngineName.__args__

    def test_llama_with_no_defaults_uses_empty(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: llama-no-defaults
tasks:
  - id: ask
    engine: llama
    prompt: "Hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.llama.model is None
        assert plan.defaults.llama.args == []

    def test_reasoning_effort_warning(self, tmp_path: Path) -> None:
        """Llama does not support reasoning_effort — loader emits a warning."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: llama-effort
tasks:
  - id: ask
    engine: llama
    model: llama3
    reasoning_effort: high
    prompt: "Think hard"
""")
        plan = load_plan(plan_file)
        warnings = plan.validation_warnings
        effort_warnings = [
            w for w in warnings
            if "reasoning_effort" in w and "Llama" in w
        ]
        assert len(effort_warnings) >= 1

    def test_llama_defaults_empty_dict_ok(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: llama-empty-defaults
defaults:
  llama: {}
tasks:
  - id: ask
    engine: llama
    prompt: "Hello"
""")
        plan = load_plan(plan_file)
        assert plan.defaults.llama.model is None


# ===========================================================================
# TestLlamaRouting
# ===========================================================================


class TestLlamaRouting:
    """Auto-routing tier table and resolution for llama engine."""

    def test_model_tiers_includes_llama(self) -> None:
        assert "llama" in _MODEL_TIERS

    def test_tier_low(self) -> None:
        assert _MODEL_TIERS["llama"]["low"] == "llama-3.2-3b"

    def test_tier_medium(self) -> None:
        assert _MODEL_TIERS["llama"]["medium"] == "llama-3-8b"

    def test_tier_high(self) -> None:
        assert _MODEL_TIERS["llama"]["high"] == "codellama-13b"

    def test_auto_routing_resolves(self) -> None:
        plan = _make_plan()
        task = TaskSpec(id="t", engine="llama", model="auto", prompt="Hello")
        resolved = resolve_auto_model(task, plan, "llama")
        assert resolved in {"llama-3.2-3b", "llama-3-8b", "codellama-13b"}

    def test_auto_routing_security_tag_high_tier(self) -> None:
        plan = _make_plan()
        task = TaskSpec(
            id="t", engine="llama", model="auto",
            prompt="Audit security", tags=["security"],
        )
        resolved = resolve_auto_model(task, plan, "llama")
        assert resolved == "codellama-13b"

    def test_auto_routing_trivial_tag_low_tier(self) -> None:
        plan = _make_plan()
        task = TaskSpec(
            id="t", engine="llama", model="auto",
            prompt="Fix typo", tags=["trivial"],
        )
        resolved = resolve_auto_model(task, plan, "llama")
        assert resolved == "llama-3.2-3b"


# ===========================================================================
# TestLlamaChat
# ===========================================================================


class TestLlamaChat:
    """Llama is a valid chat engine with model aliases available."""

    def test_llama_in_valid_chat_engines(self) -> None:
        from maestro_cli.chat import _VALID_ENGINES

        assert "llama" in _VALID_ENGINES

    def test_llama_aliases_in_chat_model_map(self) -> None:
        from maestro_cli.chat import _ENGINE_ALIASES

        assert "llama" in _ENGINE_ALIASES
        assert _ENGINE_ALIASES["llama"] is LLAMA_MODEL_ALIASES


# ===========================================================================
# TestLlamaToolRestriction
# ===========================================================================


class TestLlamaToolRestriction:
    """allowed_tools on llama injects system prompt restriction."""

    def test_tool_restriction_injected_for_llama(self) -> None:
        task = TaskSpec(
            id="t", engine="llama", prompt="Do something",
            allowed_tools=["Read", "Grep"],
        )
        result = _inject_tool_restriction("Do something", task)
        assert result.startswith("IMPORTANT: You are restricted")
        assert "Read, Grep" in result
        assert "Do something" in result

    def test_no_restriction_when_allowed_tools_none(self) -> None:
        task = TaskSpec(id="t", engine="llama", prompt="Do something")
        result = _inject_tool_restriction("Do something", task)
        assert result == "Do something"

    def test_allowed_tools_warning_in_loader(self, tmp_path: Path) -> None:
        """Loader emits W27 advisory for llama allowed_tools."""
        plan_file = _write_plan(tmp_path, """\
version: 1
name: llama-tools
tasks:
  - id: t1
    engine: llama
    model: llama3
    prompt: "Do something"
    allowed_tools: [Read]
""")
        plan = load_plan(plan_file)
        w27_warnings = [w for w in plan.validation_warnings if "W27" in w and "Llama" in w]
        assert len(w27_warnings) >= 1
        assert "advisory" in w27_warnings[0].lower() or "no tool restriction" in w27_warnings[0].lower()


# ===========================================================================
# TestLlamaEnvironment
# ===========================================================================


class TestLlamaEnvironment:
    """Environment allowlist includes LLAMA_MODEL_DIR."""

    def test_env_allowlist_has_llama_model_dir(self) -> None:
        assert "LLAMA_MODEL_DIR" in _ENV_ALLOWLIST


# ===========================================================================
# TestLlamaEnginePlugin
# ===========================================================================


class TestLlamaEnginePlugin:
    """Verify the llama engine plugin is properly registered."""

    def test_plugin_registered(self) -> None:
        from maestro_cli.runners import _get_registered_engine_plugin

        plugin = _get_registered_engine_plugin("llama")
        assert plugin is not None
        assert plugin.name == "llama"

    def test_plugin_model_aliases(self) -> None:
        from maestro_cli.runners import _get_registered_engine_plugin

        plugin = _get_registered_engine_plugin("llama")
        assert plugin is not None
        assert plugin.model_aliases == LLAMA_MODEL_ALIASES

    def test_plugin_resolve_model(self) -> None:
        from maestro_cli.runners import _get_registered_engine_plugin

        plugin = _get_registered_engine_plugin("llama")
        assert plugin is not None
        # resolve_model uses the model_aliases dict
        assert plugin.resolve_model("llama3") == "llama-3-8b"
        assert plugin.resolve_model("unknown-model") == "unknown-model"

    def test_plugin_get_default_model(self) -> None:
        from maestro_cli.runners import _get_registered_engine_plugin

        plugin = _get_registered_engine_plugin("llama")
        assert plugin is not None
        plan = _make_plan(EngineDefaults(model="codellama"))
        assert plugin.get_default_model(plan) == "codellama"

    def test_plugin_get_default_model_none(self) -> None:
        from maestro_cli.runners import _get_registered_engine_plugin

        plugin = _get_registered_engine_plugin("llama")
        assert plugin is not None
        plan = _make_plan()
        assert plugin.get_default_model(plan) is None
