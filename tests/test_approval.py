from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from maestro_cli.cli import main
from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import (
    PlanDefaults,
    PlanRunResult,
    PlanSpec,
    TaskResult,
    TaskSpec,
)
from maestro_cli.scheduler import _request_approval, run_plan


# ---------------------------------------------------------------------------
# YAML helpers
# ---------------------------------------------------------------------------

_APPROVAL_PLAN_YAML = """\
version: 1
name: approval-plan
tasks:
  - id: gate
    requires_approval: true
    approval_message: "Deploy to production?"
    command: "echo deployed"
"""

_APPROVAL_NO_MESSAGE_YAML = """\
version: 1
name: approval-plan
tasks:
  - id: gate
    requires_approval: true
    command: "echo deployed"
"""

_APPROVAL_MESSAGE_WITHOUT_FLAG_YAML = """\
version: 1
name: bad-plan
tasks:
  - id: gate
    approval_message: "This has no requires_approval"
    command: "echo fail"
"""

_PLAIN_PLAN_YAML = """\
version: 1
name: plain-plan
tasks:
  - id: t1
    command: "echo hello"
"""

_APPROVAL_PLAN_DRY_RUN_YAML = """\
version: 1
name: approval-dry-plan
tasks:
  - id: guarded
    requires_approval: true
    approval_message: "Proceed?"
    command: "echo ok"
"""


def _write_yaml(tmp_path: Path, content: str, name: str = "plan.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


def _make_mock_execute(tmp_path: Path) -> tuple[MagicMock, list[str]]:
    """Return a mock execute_task and a call log."""
    call_log: list[str] = []

    def mock_execute(
        plan,
        task,
        run_path,
        dry_run=False,
        execution_profile="plan",
        upstream_results=None,
        context_synthesis="",
        workspace_brief="",
        **kwargs,
    ) -> TaskResult:
        call_log.append(task.id)
        now = datetime.now(UTC)
        result = TaskResult(
            task_id=task.id,
            status="dry_run" if dry_run else "success",
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=0.01,
            command=f"echo {task.id}",
            log_path=run_path / f"{task.id}.log",
            result_path=run_path / f"{task.id}.result.json",
            message="ok",
        )
        result.log_path.write_text(f"status={result.status}\n", encoding="utf-8")
        result.result_path.write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        return result

    return MagicMock(side_effect=mock_execute), call_log


# ===========================================================================
# Loader tests
# ===========================================================================


class TestApprovalLoader:
    def test_requires_approval_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_yaml(tmp_path, _APPROVAL_PLAN_YAML)
        plan = load_plan(plan_file)
        task = plan.tasks[0]
        assert task.requires_approval is True

    def test_approval_message_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_yaml(tmp_path, _APPROVAL_PLAN_YAML)
        plan = load_plan(plan_file)
        task = plan.tasks[0]
        assert task.approval_message == "Deploy to production?"

    def test_approval_default_false(self, tmp_path: Path) -> None:
        plan_file = _write_yaml(tmp_path, _PLAIN_PLAN_YAML)
        plan = load_plan(plan_file)
        task = plan.tasks[0]
        assert task.requires_approval is False
        assert task.approval_message is None

    def test_approval_message_without_requires_raises_E029(
        self, tmp_path: Path
    ) -> None:
        plan_file = _write_yaml(tmp_path, _APPROVAL_MESSAGE_WITHOUT_FLAG_YAML)
        with pytest.raises(PlanValidationError, match=r"\[E029\]"):
            load_plan(plan_file)

    def test_approval_message_without_requires_error_mentions_task(
        self, tmp_path: Path
    ) -> None:
        plan_file = _write_yaml(tmp_path, _APPROVAL_MESSAGE_WITHOUT_FLAG_YAML)
        with pytest.raises(PlanValidationError, match="gate"):
            load_plan(plan_file)


# ===========================================================================
# _request_approval unit tests
# ===========================================================================


class TestRequestApproval:
    def test_non_interactive_returns_false(self) -> None:
        result = _request_approval("my-task", None, interactive=False)
        assert result is False

    def test_non_interactive_with_message_returns_false(self) -> None:
        result = _request_approval("my-task", "Are you sure?", interactive=False)
        assert result is False

    def test_approval_yes_returns_true(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda: "y")
        result = _request_approval("my-task", None, interactive=True)
        assert result is True

    def test_approval_yes_uppercase_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda: "Y")
        result = _request_approval("my-task", None, interactive=True)
        assert result is True

    def test_approval_yes_full_word_returns_true(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda: "yes")
        result = _request_approval("my-task", None, interactive=True)
        assert result is True

    def test_approval_no_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda: "n")
        result = _request_approval("my-task", None, interactive=True)
        assert result is False

    def test_approval_empty_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda: "")
        result = _request_approval("my-task", None, interactive=True)
        assert result is False

    def test_approval_eof_returns_false(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def raise_eof() -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        result = _request_approval("my-task", None, interactive=True)
        assert result is False

    def test_approval_uses_custom_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda: "n")
        _request_approval("my-task", "Deploy to prod?", interactive=True)
        captured = capsys.readouterr()
        assert "Deploy to prod?" in captured.out

    def test_approval_uses_default_message_when_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr("builtins.input", lambda: "n")
        _request_approval("my-task", None, interactive=True)
        captured = capsys.readouterr()
        assert "my-task" in captured.out


# ===========================================================================
# Scheduler integration tests
# ===========================================================================


class TestApprovalSchedulerIntegration:
    def _make_plan(self, tmp_path: Path, requires_approval: bool = True) -> PlanSpec:
        task = TaskSpec(
            id="gate",
            description="approval gate",
            command="echo ok",
            requires_approval=requires_approval,
            approval_message="Proceed with deployment?",
        )
        return PlanSpec(
            version=1,
            name="test-plan",
            tasks=[task],
            defaults=PlanDefaults(),
            source_path=tmp_path / "plan.yaml",
        )

    def test_approval_denied_skips_task(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = self._make_plan(tmp_path)
        mock_execute, call_log = _make_mock_execute(tmp_path)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: False))

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert "gate" not in call_log
        assert result.task_results["gate"].status == "skipped"

    def test_auto_approve_bypasses_prompt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = self._make_plan(tmp_path)
        mock_execute, call_log = _make_mock_execute(tmp_path)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        # stdin is non-interactive, but auto_approve should still proceed
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: False))

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"), auto_approve=True)

        assert "gate" in call_log
        assert result.task_results["gate"].status == "success"

    def test_approval_interactive_yes_runs_task(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = self._make_plan(tmp_path)
        mock_execute, call_log = _make_mock_execute(tmp_path)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: True))
        monkeypatch.setattr("builtins.input", lambda: "y")

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert "gate" in call_log
        assert result.task_results["gate"].status == "success"

    def test_dry_run_skips_approval_gate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """In dry-run mode, approval gate is not triggered; task gets dry_run status."""
        plan = self._make_plan(tmp_path)
        mock_execute, call_log = _make_mock_execute(tmp_path)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        # Even if non-interactive, dry_run should not invoke approval logic
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: False))

        run_plan(plan, dry_run=True, run_dir_override=str(tmp_path / "runs"))

        # Task should have been handed to execute_task (dry_run path) without approval prompt
        assert "gate" in call_log

    def test_task_without_approval_always_runs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = self._make_plan(tmp_path, requires_approval=False)
        mock_execute, call_log = _make_mock_execute(tmp_path)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("sys.stdin", MagicMock(isatty=lambda: False))

        run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert "gate" in call_log


# ===========================================================================
# CLI tests
# ===========================================================================


class TestApprovalCLI:
    def _make_mock_result(self, success: bool = True) -> MagicMock:
        mock_result = MagicMock(spec=PlanRunResult)
        mock_result.success = success
        mock_result.task_results = {}
        mock_result.total_cost_usd = None
        mock_result.total_tokens = None
        mock_result.duration_sec = 0.0
        mock_result.budget_exceeded = False
        return mock_result

    def test_auto_approve_flag_parsed(self) -> None:
        from maestro_cli.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml", "--auto-approve"])
        assert args.auto_approve is True

    def test_auto_approve_flag_default_false(self) -> None:
        from maestro_cli.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["run", "plan.yaml"])
        assert args.auto_approve is False

    def test_auto_approve_passed_to_run_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan_file = _write_yaml(tmp_path, _PLAIN_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file), "--auto-approve"])

        _, kwargs = mock_run.call_args
        assert kwargs["auto_approve"] is True

    def test_auto_approve_not_passed_without_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan_file = _write_yaml(tmp_path, _PLAIN_PLAN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file)])

        _, kwargs = mock_run.call_args
        assert kwargs.get("auto_approve") is False

    def test_dry_run_mentions_approval_in_checklist(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        plan_file = _write_yaml(tmp_path, _APPROVAL_PLAN_DRY_RUN_YAML)
        mock_run = MagicMock(return_value=self._make_mock_result(success=True))
        mock_run.return_value.task_results = {}
        monkeypatch.setattr("maestro_cli.cli.run_plan", mock_run)

        main(["run", str(plan_file), "--dry-run"])

        captured = capsys.readouterr()
        assert "approval" in captured.out.lower()


# ===========================================================================
# Serialization tests
# ===========================================================================


class TestApprovalSerialization:
    def test_approval_fields_in_to_dict(self) -> None:
        task = TaskSpec(
            id="gate",
            description="approval gate",
            command="echo ok",
            requires_approval=True,
            approval_message="Ready to deploy?",
        )
        d = task.to_dict()
        assert d["requires_approval"] is True
        assert d["approval_message"] == "Ready to deploy?"

    def test_approval_defaults_in_to_dict(self) -> None:
        task = TaskSpec(
            id="plain",
            description="plain task",
            command="echo ok",
        )
        d = task.to_dict()
        assert d["requires_approval"] is False
        assert d["approval_message"] is None
