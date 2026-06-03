from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.loader import load_plan
from maestro_cli.models import (
    EngineDefaults,
    EngineName,
    OLLAMA_MODELS,
    PlanDefaults,
    PlanSpec,
    TaskSpec,
)
from maestro_cli.runners import (
    _ENV_ALLOWLIST,
    _resolve_ollama_model,
    build_command,
)


class TestOllama:
    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def test_resolve_ollama_model_known(self) -> None:
        assert _resolve_ollama_model("llama3") == "llama3"

    def test_resolve_ollama_model_unknown(self) -> None:
        assert _resolve_ollama_model("custom-model:7b") == "custom-model:7b"

    def test_resolve_ollama_model_none(self) -> None:
        assert _resolve_ollama_model(None) is None

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------

    def _make_plan(self, ollama_defaults: EngineDefaults | None = None) -> PlanSpec:
        defaults = PlanDefaults(
            ollama=ollama_defaults if ollama_defaults is not None else EngineDefaults(),
        )
        return PlanSpec(version=1, name="test-plan", defaults=defaults, tasks=[])

    def test_build_command_ollama_basic(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="ollama", model="llama3", prompt="Say hello")
        cmd, shell = build_command(plan, task, Path("/tmp"))
        assert cmd == ["ollama", "run", "llama3", "Say hello"]
        assert shell is False

    def test_build_command_ollama_default_model(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="ollama", prompt="Say hello")
        cmd, shell = build_command(plan, task, Path("/tmp"))
        assert cmd[0] == "ollama"
        assert cmd[1] == "run"
        assert cmd[2] == "llama3"
        assert shell is False

    def test_execution_profile_ollama(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("maestro_cli.runners._resolve_executable", lambda x: [x])
        plan = self._make_plan()
        task = TaskSpec(id="t", engine="ollama", model="mistral", prompt="Hello")
        for profile in ("plan", "safe", "yolo"):
            cmd, shell = build_command(plan, task, Path("/tmp"), execution_profile=profile)
            assert cmd == ["ollama", "run", "mistral", "Hello"]
            assert shell is False

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------

    def test_env_allowlist_has_ollama_host(self) -> None:
        assert "OLLAMA_HOST" in _ENV_ALLOWLIST

    # ------------------------------------------------------------------
    # Loader / validation
    # ------------------------------------------------------------------

    def test_loader_ollama_engine_valid(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: ollama-plan
tasks:
  - id: ask
    engine: ollama
    model: llama3
    prompt: "What is 2+2?"
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        task = plan.tasks[0]
        assert task.engine == "ollama"
        assert task.model == "llama3"

    def test_loader_ollama_defaults(self, tmp_path: Path) -> None:
        content = """\
version: 1
name: ollama-defaults-plan
defaults:
  ollama:
    model: mistral
    args: ["--verbose"]
tasks:
  - id: ask
    engine: ollama
    prompt: "Hello"
"""
        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(content, encoding="utf-8")
        plan = load_plan(plan_file)
        assert plan.defaults.ollama.model == "mistral"
        assert plan.defaults.ollama.args == ["--verbose"]

    # ------------------------------------------------------------------
    # Constants
    # ------------------------------------------------------------------

    def test_ollama_models_constant(self) -> None:
        assert "llama3" in OLLAMA_MODELS
        assert "mistral" in OLLAMA_MODELS
        assert "codellama" in OLLAMA_MODELS

    def test_engine_name_includes_ollama(self) -> None:
        assert "ollama" in EngineName.__args__
