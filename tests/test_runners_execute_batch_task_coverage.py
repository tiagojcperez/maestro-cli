from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import maestro_cli.runners as runners
from maestro_cli.models import (
    BatchSpec,
    PlanDefaults,
    PlanSpec,
    TaskSpec,
    TokenUsage,
)
from maestro_cli.runners import _CostAndTokens, _execute_batch_task


def _make_plan(tmp_path: Path, *, secrets: list[str] | str | None = None) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="batch-cov",
        defaults=PlanDefaults(),
        tasks=[],
        workspace_root=str(tmp_path),
        secrets=secrets if secrets is not None else [],
    )


def _make_task(
    *,
    items: list[str],
    max_per_call: int = 5,
    guard_command: str | list[str] | None = None,
    allow_failure: bool = False,
    env: dict[str, str] | None = None,
) -> TaskSpec:
    return TaskSpec(
        id="batch-cov-task",
        engine="claude",
        prompt="review files",
        batch=BatchSpec(
            items=items,
            template="Process {{ batch.item }}",
            max_per_call=max_per_call,
        ),
        guard_command=guard_command,
        allow_failure=allow_failure,
        env=env or {},
    )


def _stub_proc(returncode: int, stdout: str) -> Any:
    return type("R", (), {"returncode": returncode, "stdout": stdout, "stderr": ""})()


# ---------------------------------------------------------------------------
# Lines 6916-6926: build_command raises -> failed result returned early.
# ---------------------------------------------------------------------------


def test_build_command_failure_returns_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _boom(*a: Any, **kw: Any) -> Any:
        raise RuntimeError("cannot build command")

    monkeypatch.setattr(runners, "build_command", _boom)

    plan = _make_plan(tmp_path)
    task = _make_task(items=["a", "b"], max_per_call=1)

    result = _execute_batch_task(
        plan,
        task,
        tmp_path,
        dry_run=False,
        execution_profile="plan",
        upstream_results={},
        context_synthesis="",
        workspace_brief="",
        event_callback=None,
        extra_template_vars=None,
    )

    assert result.status == "failed"
    assert result.exit_code == 1
    assert "Command build failed for chunk 1" in result.message
    assert "cannot build command" in result.message
    # The result.json should have been written by the early return path.
    assert (tmp_path / "batch-cov-task.result.json").exists()


# ---------------------------------------------------------------------------
# Lines 6953-6955: generic (non-timeout) Exception from subprocess.run.
# ---------------------------------------------------------------------------


def test_subprocess_generic_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _raise_generic(*a: Any, **kw: Any) -> Any:
        raise OSError("subprocess exploded")

    monkeypatch.setattr(runners.subprocess, "run", _raise_generic)

    plan = _make_plan(tmp_path)
    task = _make_task(items=["a"], max_per_call=1)

    result = _execute_batch_task(
        plan,
        task,
        tmp_path,
        dry_run=False,
        execution_profile="plan",
        upstream_results={},
        context_synthesis="",
        workspace_brief="",
        event_callback=None,
        extra_template_vars=None,
    )

    # last_exit_code set to 1 in the generic-except branch -> failed status.
    assert result.status == "failed"
    assert result.exit_code == 1
    log_text = (tmp_path / "batch-cov-task.log").read_text(encoding="utf-8")
    assert "subprocess exploded" in log_text


# ---------------------------------------------------------------------------
# Line 6959: secret masking branch is taken when secret_values is non-empty.
# ---------------------------------------------------------------------------


def test_secret_masking_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = "SUPERSECRETVALUE"
    monkeypatch.setenv("MY_TOKEN", secret)

    # Engine output that leaks the secret value verbatim.
    monkeypatch.setattr(
        runners.subprocess,
        "run",
        lambda *a, **kw: _stub_proc(0, f"### Item 1: a\nleaked={secret}\n"),
    )

    plan = _make_plan(tmp_path, secrets=["MY_TOKEN"])
    task = _make_task(items=["a"], max_per_call=1)

    result = _execute_batch_task(
        plan,
        task,
        tmp_path,
        dry_run=False,
        execution_profile="plan",
        upstream_results={},
        context_synthesis="",
        workspace_brief="",
        event_callback=None,
        extra_template_vars=None,
    )

    assert result.status == "success"
    log_text = (tmp_path / "batch-cov-task.log").read_text(encoding="utf-8")
    # The raw secret value must NOT appear in the written log -> masking ran.
    assert secret not in log_text
    assert secret not in (result.stdout_tail or "")


# ---------------------------------------------------------------------------
# Lines 6992 & 6994: cost_usd and token_usage extracted from the log are
# non-None and assigned into the totals.
# ---------------------------------------------------------------------------


def test_cost_and_tokens_extracted_from_log(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runners.subprocess,
        "run",
        lambda *a, **kw: _stub_proc(0, "### Item 1: a\nok\n"),
    )

    # total_tokens is a computed property (input + cached + output).
    usage = TokenUsage(input_tokens=100, output_tokens=50)
    monkeypatch.setattr(
        runners,
        "_extract_cost_and_tokens_from_log",
        lambda *a, **kw: _CostAndTokens(cost_usd=0.42, token_usage=usage),
    )

    plan = _make_plan(tmp_path)
    task = _make_task(items=["a"], max_per_call=1)

    result = _execute_batch_task(
        plan,
        task,
        tmp_path,
        dry_run=False,
        execution_profile="plan",
        upstream_results={},
        context_synthesis="",
        workspace_brief="",
        event_callback=None,
        extra_template_vars=None,
    )

    assert result.status == "success"
    assert result.cost_usd == 0.42
    assert result.token_usage is not None
    assert result.token_usage.total_tokens == 150


# ---------------------------------------------------------------------------
# Lines 7013-7015: guard_command fails -> status set to failed with message.
# ---------------------------------------------------------------------------


def test_guard_command_failure_marks_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runners.subprocess,
        "run",
        lambda *a, **kw: _stub_proc(0, "### Item 1: a\nok\n"),
    )
    monkeypatch.setattr(
        runners,
        "_run_guard_command",
        lambda *a, **kw: (False, "guard says no" + "x" * 400),
    )

    plan = _make_plan(tmp_path)
    task = _make_task(items=["a"], max_per_call=1, guard_command=["check.sh"])

    result = _execute_batch_task(
        plan,
        task,
        tmp_path,
        dry_run=False,
        execution_profile="plan",
        upstream_results={},
        context_synthesis="",
        workspace_brief="",
        event_callback=None,
        extra_template_vars=None,
    )

    assert result.status == "failed"
    assert "guard_command failed" in result.message
    # Guard output is truncated to 300 chars in the message.
    assert "guard says no" in result.message


# ---------------------------------------------------------------------------
# Line 7018: failed status + allow_failure -> soft_failed.
# Driven via a guard_command failure on an allow_failure task.
# ---------------------------------------------------------------------------


def test_guard_failure_with_allow_failure_soft_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        runners.subprocess,
        "run",
        lambda *a, **kw: _stub_proc(0, "### Item 1: a\nok\n"),
    )
    monkeypatch.setattr(
        runners,
        "_run_guard_command",
        lambda *a, **kw: (False, "nope"),
    )

    plan = _make_plan(tmp_path)
    task = _make_task(
        items=["a"],
        max_per_call=1,
        guard_command=["check.sh"],
        allow_failure=True,
    )

    result = _execute_batch_task(
        plan,
        task,
        tmp_path,
        dry_run=False,
        execution_profile="plan",
        upstream_results={},
        context_synthesis="",
        workspace_brief="",
        event_callback=None,
        extra_template_vars=None,
    )

    assert result.status == "soft_failed"
