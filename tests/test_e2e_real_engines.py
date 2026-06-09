"""Opt-in E2E coverage for real engine CLIs.

These tests are intentionally excluded from normal offline `pytest tests/` runs.
Enable them with `MAESTRO_RUN_REAL_ENGINE_TESTS=1` plus per-engine model env vars.
"""

from __future__ import annotations

import json
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


def _ollama_preflight() -> str:
    """Shared skip-guards + model resolution for the free Ollama E2E lane."""
    _require_executable("ollama")
    model = _require_env(
        _OLLAMA_MODEL_ENV, "a locally pulled Ollama model such as llama3.2:1b"
    )
    _ollama_model_ready(model)
    return model


def test_ollama_context_passing_e2e(
    real_engine_yaml_runner: Callable[[str], tuple[int, dict[str, Any], Path]],
) -> None:
    """Real-engine E2E: upstream output flows into a downstream task via
    ``context_from`` and the multi-task run completes against a live engine."""
    model = _ollama_preflight()
    body = (
        "version: 1\n"
        "name: ollama-context-e2e\n"
        "max_parallel: 1\n"
        "fail_fast: true\n"
        "tasks:\n"
        "  - id: produce\n"
        "    engine: ollama\n"
        f"    model: {json.dumps(model)}\n"
        '    prompt: "Reply with one short word."\n'
        "  - id: consume\n"
        "    engine: ollama\n"
        f"    model: {json.dumps(model)}\n"
        "    depends_on: [produce]\n"
        "    context_from: [produce]\n"
        '    prompt: "Given the previous output in your context, reply with one short sentence."\n'
    )

    rc, manifest, _run_dir = real_engine_yaml_runner(body)

    assert rc == 0
    assert manifest["success"] is True
    assert manifest["task_results"]["produce"]["status"] == "success"
    # The downstream task only succeeds if the context pipeline assembled and
    # injected the real upstream output without breaking the live run.
    assert manifest["task_results"]["consume"]["status"] == "success"


def test_ollama_quality_gates_e2e(
    real_engine_yaml_runner: Callable[[str], tuple[int, dict[str, Any], Path]],
) -> None:
    """Real-engine E2E: the deterministic quality gates (``verify_command`` and
    ``guard_command``) run end-to-end after a live engine call."""
    model = _ollama_preflight()
    body = (
        "version: 1\n"
        "name: ollama-quality-e2e\n"
        "fail_fast: true\n"
        "tasks:\n"
        "  - id: gen\n"
        "    engine: ollama\n"
        f"    model: {json.dumps(model)}\n"
        '    prompt: "Reply with one short sentence."\n'
        '    verify_command: "true"\n'
        '    guard_command: "cat"\n'
    )

    rc, manifest, _run_dir = real_engine_yaml_runner(body)

    assert rc == 0
    # A success status means the real engine call passed BOTH the verify_command
    # and the (stdin-piped) guard_command gates.
    assert manifest["task_results"]["gen"]["status"] == "success"


def test_ollama_budget_and_cost_e2e(
    real_engine_yaml_runner: Callable[[str], tuple[int, dict[str, Any], Path]],
) -> None:
    """Real-engine E2E: cost/budget tracking runs against a live engine. The
    local engine is zero-cost, so the budget is tracked but never exceeded."""
    model = _ollama_preflight()
    body = (
        "version: 1\n"
        "name: ollama-budget-e2e\n"
        "max_cost_usd: 10.0\n"
        "tasks:\n"
        "  - id: gen\n"
        "    engine: ollama\n"
        f"    model: {json.dumps(model)}\n"
        '    prompt: "Write a short two-line note."\n'
    )

    rc, manifest, _run_dir = real_engine_yaml_runner(body)

    assert rc == 0
    task_result = manifest["task_results"]["gen"]
    assert task_result["status"] == "success"
    # Local execution is zero-cost: cost is tracked and the budget is not tripped.
    assert task_result.get("cost_usd") in (None, 0, 0.0)
    assert manifest.get("total_cost_usd") in (None, 0, 0.0)
