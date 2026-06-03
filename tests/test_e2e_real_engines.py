"""Opt-in E2E coverage for real engine CLIs.

These tests are intentionally excluded from normal offline `pytest tests/` runs.
Enable them with `MAESTRO_RUN_REAL_ENGINE_TESTS=1` plus per-engine model env vars.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.real_engine

_CODEX_MODEL_ENV = "MAESTRO_E2E_CODEX_MODEL"
_OLLAMA_MODEL_ENV = "MAESTRO_E2E_OLLAMA_MODEL"
_SENTINEL_PREFIX = "MAESTRO_E2E"


def _require_env(name: str, help_text: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"set {name} to {help_text}")
    return value


def _require_executable(name: str) -> None:
    if shutil.which(name) is None:
        pytest.skip(f"required executable not found on PATH: {name}")


def _ollama_model_ready(model: str) -> None:
    try:
        result = subprocess.run(
            ["ollama", "list"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except OSError:
        pytest.skip("ollama CLI is not runnable")
    except subprocess.TimeoutExpired:
        pytest.skip("ollama list timed out")

    if result.returncode != 0:
        pytest.skip("ollama is not ready for local model runs")

    available_models = {
        line.split()[0]
        for line in result.stdout.splitlines()
        if line.strip() and not line.lower().startswith("name ")
    }
    if model not in available_models:
        pytest.skip(f"ollama model is not available locally: {model}")


def _assert_successful_probe(
    manifest: dict[str, Any],
    task_result: dict[str, Any],
    run_dir: Path,
    sentinel: str,
) -> None:
    assert manifest["success"] is True
    assert task_result["status"] == "success"
    assert sentinel in task_result["stdout_tail"]
    assert run_dir.joinpath("run_manifest.json").exists()
    assert Path(task_result["log_path"]).exists()
    assert Path(task_result["result_path"]).exists()


def test_codex_real_engine_round_trip(
    real_engine_plan_runner: Callable[[str, str, str | None], tuple[dict[str, Any], dict[str, Any], Path]],
) -> None:
    _require_executable("codex")
    model = _require_env(_CODEX_MODEL_ENV, "a Codex model alias such as 5-mini")
    sentinel = f"{_SENTINEL_PREFIX}_CODEX_OK"

    manifest, task_result, run_dir = real_engine_plan_runner(
        "codex",
        f"Reply with exactly {sentinel} and nothing else.",
        model,
    )

    _assert_successful_probe(manifest, task_result, run_dir, sentinel)


def test_ollama_real_engine_round_trip(
    real_engine_plan_runner: Callable[[str, str, str | None], tuple[dict[str, Any], dict[str, Any], Path]],
) -> None:
    _require_executable("ollama")
    model = _require_env(_OLLAMA_MODEL_ENV, "a locally pulled Ollama model such as llama3")
    _ollama_model_ready(model)
    sentinel = f"{_SENTINEL_PREFIX}_OLLAMA_OK"

    manifest, task_result, run_dir = real_engine_plan_runner(
        "ollama",
        f"Reply with exactly {sentinel} and nothing else.",
        model,
    )

    _assert_successful_probe(manifest, task_result, run_dir, sentinel)
