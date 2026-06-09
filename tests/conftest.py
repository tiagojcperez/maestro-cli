from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest


REAL_ENGINE_MARKER = "real_engine"
REAL_ENGINE_OPT_IN_ENV = "MAESTRO_RUN_REAL_ENGINE_TESTS"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


# The real os.name, captured before any test can fake it. Several tests set
# os.name = "nt" (often via monkeypatch on a module's ``os`` reference, which is
# the shared os module) to drive Windows-only code paths on Linux.
_REAL_OS_NAME = os.name


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):  # type: ignore[no-untyped-def]
    """Restore the real ``os.name`` before pytest renders a test report.

    When a test that faked ``os.name = "nt"`` fails, the fake is still active
    while pytest formats the failure. pytest's formatter calls
    ``Path(os.getcwd())``, which raises ``NotImplementedError`` trying to build a
    ``WindowsPath`` on POSIX -- turning a normal failure into a session-killing
    ``INTERNALERROR``. Resetting here keeps failures legible on every platform.
    """
    if os.name != _REAL_OS_NAME:
        os.name = _REAL_OS_NAME
    yield


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _env_truthy(REAL_ENGINE_OPT_IN_ENV):
        return

    skip_real_engine = pytest.mark.skip(
        reason=f"set {REAL_ENGINE_OPT_IN_ENV}=1 to enable opt-in real-engine tests",
    )
    for item in items:
        if REAL_ENGINE_MARKER in item.keywords:
            item.add_marker(skip_real_engine)


@pytest.fixture
def minimal_plan_yaml(tmp_path: Path) -> Path:
    """Create a minimal valid plan YAML for testing."""
    content = """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
"""
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


@pytest.fixture
def context_plan_yaml(tmp_path: Path) -> Path:
    """Create a plan YAML with context_from for testing."""
    content = """\
version: 1
name: context-test
tasks:
  - id: a
    command: "echo a-output"
  - id: b
    depends_on: [a]
    context_from: [a]
    command: "echo b-output"
  - id: c
    depends_on: [a, b]
    context_from: ["*"]
    command: "echo c-output"
"""
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


@pytest.fixture
def sample_brief_yaml(tmp_path: Path) -> Path:
    """Create a sample brief YAML for scaffold testing."""
    content = """\
name: test-feature
goal: "Add a new feature"
workspace_root: "C:/test/project"
branch_name: feature/test
max_parallel: 3
tasks:
  - id: db-migration
    description: "Create migration"
    task_type: implementation
    prompt_hint: "Create tables..."
  - id: api-endpoint
    description: "Add REST endpoint"
    task_type: implementation
    depends_on: [db-migration]
    prompt_hint: "Add GET /api/thing endpoint"
  - id: security-check
    description: "Audit authentication"
    task_type: security-audit
    depends_on: [api-endpoint]
"""
    brief_file = tmp_path / "brief.yaml"
    brief_file.write_text(content, encoding="utf-8")
    return brief_file


@pytest.fixture
def real_engine_plan_runner(
    tmp_path: Path,
) -> Callable[[str, str, str | None], tuple[dict[str, Any], dict[str, Any], Path]]:
    """Run a one-task Maestro plan and return manifest, task result, and run dir."""

    def _run(engine: str, prompt: str, model: str | None = None) -> tuple[dict[str, Any], dict[str, Any], Path]:
        from maestro_cli.cli import main

        plan_path = tmp_path / f"{engine}-e2e-plan.yaml"
        run_root = tmp_path / "runs"
        workspace_root = tmp_path / "workspace"
        workspace_root.mkdir(exist_ok=True)

        lines = [
            "version: 1",
            f"name: {engine}-real-engine-e2e",
            f"workspace_root: {json.dumps(str(workspace_root))}",
            "max_parallel: 1",
            "fail_fast: true",
            "tasks:",
            "  - id: probe",
            f"    engine: {engine}",
        ]
        if model is not None:
            lines.append(f"    model: {json.dumps(model)}")
        lines.append(f"    prompt: {json.dumps(prompt)}")
        plan_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        rc = main([
            "run",
            str(plan_path),
            "--run-dir",
            str(run_root),
            "--mask-secrets",
            "--quiet",
        ])
        assert rc == 0

        manifests = sorted(run_root.glob("*/run_manifest.json"))
        assert len(manifests) == 1
        manifest = json.loads(manifests[0].read_text(encoding="utf-8"))
        return manifest, manifest["task_results"]["probe"], manifests[0].parent

    return _run


@pytest.fixture
def real_engine_yaml_runner(
    tmp_path: Path,
) -> Callable[[str], tuple[int, dict[str, Any], Path]]:
    """Run a full multi-task Maestro plan YAML against a real engine.

    Unlike ``real_engine_plan_runner`` (one task), this drives an arbitrary plan
    body so E2E tests can exercise context passing, quality gates, and budgets
    end-to-end. Returns ``(exit_code, manifest, run_dir)``.
    """

    def _run(plan_body: str) -> tuple[int, dict[str, Any], Path]:
        from maestro_cli.cli import main

        plan_path = tmp_path / "e2e-plan.yaml"
        run_root = tmp_path / "runs"
        plan_path.write_text(plan_body, encoding="utf-8")

        rc = main([
            "run",
            str(plan_path),
            "--run-dir",
            str(run_root),
            "--quiet",
        ])

        manifests = sorted(run_root.glob("*/run_manifest.json"))
        assert manifests, "no run manifest was produced"
        manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
        return rc, manifest, manifests[-1].parent

    return _run
