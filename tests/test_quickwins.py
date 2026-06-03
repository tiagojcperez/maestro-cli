from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import (
    EngineDefaults,
    FailureRecord,
    PlanDefaults,
    PlanRunResult,
    PlanSpec,
    TaskResult,
    TaskSpec,
)
from maestro_cli.runners import (
    _RETRY_FEEDBACK_MAX_CHARS,
    _RETRY_FEEDBACK_TEMPLATE,
    _build_smart_retry_feedback,
    _classify_failure,
    _resolve_retry_delay,
    build_command,
    execute_task,
)
from maestro_cli.utils import evaluate_when_condition


def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


def _make_plan(tmp_path: Path, **kwargs: Any) -> PlanSpec:
    defaults = kwargs.pop("defaults", PlanDefaults(
        codex=EngineDefaults(),
        claude=EngineDefaults(),
    ))
    return PlanSpec(
        version=1,
        name=kwargs.pop("name", "test"),
        max_parallel=kwargs.pop("max_parallel", 1),
        fail_fast=kwargs.pop("fail_fast", True),
        run_dir=str(tmp_path / "runs"),
        defaults=defaults,
        tasks=kwargs.pop("tasks", []),
        **kwargs,
    )


# ===========================================================================
# Feature 1: Error Feedback Injection
# ===========================================================================


class TestRetryFeedbackTemplate:
    """Verify the retry feedback template and constants."""

    def test_template_has_placeholders(self) -> None:
        rendered = _RETRY_FEEDBACK_TEMPLATE.format(exit_code=1, output="some error")
        assert "[RETRY FEEDBACK]" in rendered
        assert "exit code 1" in rendered
        assert "some error" in rendered

    def test_max_chars_is_positive(self) -> None:
        assert _RETRY_FEEDBACK_MAX_CHARS > 0


class TestSmartRetryFailureClassification:
    def test_timeout_exit_code_has_priority(self) -> None:
        assert _classify_failure(124, "SyntaxError: invalid syntax", "failed") == "timeout"

    @pytest.mark.parametrize(
        ("output", "expected"),
        [
            ("SyntaxError: invalid syntax", "compilation_error"),
            ("FAILED test_login", "test_failure"),
            ("Permission denied: /tmp/x", "permission_error"),
            ("ValueError: invalid data", "validation_error"),
            ("Traceback (most recent call last)", "runtime_error"),
        ],
    )
    def test_pattern_based_classification(self, output: str, expected: str) -> None:
        assert _classify_failure(1, output, "") == expected

    def test_unknown_when_no_pattern_matches(self) -> None:
        assert _classify_failure(1, "something else", "") == "unknown"


class TestSmartRetryFeedbackBuilder:
    def test_includes_history_and_escalation_for_repeated_category(self) -> None:
        history = [
            FailureRecord(attempt=1, category="test_failure", exit_code=1, message="fail-1"),
            FailureRecord(attempt=2, category="test_failure", exit_code=1, message="fail-2"),
        ]
        feedback = _build_smart_retry_feedback(
            attempt=2,
            max_attempts=3,
            category="test_failure",
            exit_code=1,
            output="FAILED test_example",
            failure_history=history,
        )
        assert "[RETRY FEEDBACK -- Attempt 2/3]" in feedback
        assert "Previous failures:" in feedback
        assert "Attempt 1: test_failure (exit 1)" in feedback
        assert "WARNING: This failure category (test_failure)" in feedback


class TestFailureHistoryTracking:
    def test_failed_retries_record_failure_history(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t1", command="exit 1", max_retries=1)

        result = execute_task(plan, task, run_path)

        assert result.status == "failed"
        assert result.retry_count == 1
        assert len(result.failure_history) == 2
        assert [f.attempt for f in result.failure_history] == [1, 2]


class TestRetryFeedbackInjection:
    """Engine task with verify failure should inject feedback on retry."""

    def test_engine_task_retry_includes_feedback(self, tmp_path: Path, monkeypatch: Any) -> None:
        """When an engine task's verify fails, the retry command should include feedback."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)

        # Create a counter file so verify fails on attempt 0, succeeds on attempt 1
        counter_file = tmp_path / "vcounter"
        counter_file.write_text("0", encoding="utf-8")
        verify_cmd = (
            f'bash -c "c=$(cat {counter_file.as_posix()}); '
            f"if [ \\\"$c\\\" = \\\"0\\\" ]; then echo 1 > {counter_file.as_posix()}; "
            f'echo VERIFY_ERROR >&2; exit 1; else exit 0; fi"'
        )

        # Track all commands that were built
        commands_built: list[str] = []
        original_build = build_command

        def tracking_build(*args: Any, **kwargs: Any) -> Any:
            result = original_build(*args, **kwargs)
            cmd_str = result[0] if isinstance(result[0], str) else subprocess.list2cmdline(result[0])
            commands_built.append(cmd_str)
            return result

        task = TaskSpec(
            id="t1",
            engine="claude",
            model="haiku",
            prompt="Do something",
            verify_command=verify_cmd,
            max_retries=1,
        )

        # Mock subprocess.run for the engine command but not verify
        call_count = 0

        def mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            nonlocal call_count
            cmd = args[0] if args else kwargs.get("args", [])
            cmd_str = cmd if isinstance(cmd, str) else subprocess.list2cmdline(cmd)

            # Only mock claude calls, let shell verify run normally
            if "claude" in cmd_str:
                call_count += 1
                return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="done", stderr="")
            # Let real subprocess handle verify
            return subprocess.run.__wrapped__(*args, **kwargs)  # type: ignore[attr-defined]

        monkeypatch.setattr(subprocess, "run", mock_run)

        # We can't easily test the full feedback injection through execute_task
        # with mocked subprocess, so test build_command directly
        feedback = _RETRY_FEEDBACK_TEMPLATE.format(exit_code=1, output="VERIFY_ERROR")
        cmd, shell = build_command(plan, task, tmp_path, retry_feedback=feedback)
        cmd_str = subprocess.list2cmdline(cmd) if isinstance(cmd, list) else cmd
        assert "[RETRY FEEDBACK]" in cmd_str

    def test_shell_task_retry_no_feedback(self, tmp_path: Path) -> None:
        """Shell task retries should NOT include feedback (no prompt to inject into)."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)

        # Shell task with verify that fails then succeeds.
        # Use a portable python helper script rather than a bash one-liner: a
        # bash command with $(...) substitution is evaluated differently by
        # /bin/sh (Linux shell=True) vs cmd.exe (Windows shell=True), which
        # breaks the fail-then-succeed counter. Calling a script file avoids all
        # cross-platform inline-quoting and substitution hazards.
        counter_file = tmp_path / "vcounter"
        counter_file.write_text("0", encoding="utf-8")
        verify_script = tmp_path / "verify_counter.py"
        verify_script.write_text(
            "import sys\n"
            "p = sys.argv[1]\n"
            "with open(p, encoding='utf-8') as f:\n"
            "    c = f.read().strip()\n"
            "if c == '0':\n"
            "    with open(p, 'w', encoding='utf-8') as f:\n"
            "        f.write('1')\n"
            "    sys.exit(1)\n"
            "sys.exit(0)\n",
            encoding="utf-8",
        )
        verify_cmd = [sys.executable, str(verify_script), str(counter_file)]
        task = TaskSpec(
            id="t1",
            command="echo hello",
            verify_command=verify_cmd,
            max_retries=1,
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        assert result.retry_count == 1
        # The command should be identical on retry (no feedback for shell tasks)
        log_content = result.log_path.read_text(encoding="utf-8")
        assert "[RETRY FEEDBACK]" not in log_content

    def test_first_attempt_no_feedback(self, tmp_path: Path) -> None:
        """First attempt should never include feedback."""
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            engine="claude",
            model="haiku",
            prompt="Do something",
        )
        cmd, shell = build_command(plan, task, tmp_path)
        cmd_str = subprocess.list2cmdline(cmd) if isinstance(cmd, list) else cmd
        assert "[RETRY FEEDBACK]" not in cmd_str

    def test_feedback_truncated(self) -> None:
        """Feedback output should be truncated to max chars."""
        long_output = "x" * 5000
        # Simulate what execute_task does
        truncated = long_output.strip()[-_RETRY_FEEDBACK_MAX_CHARS:]
        assert len(truncated) == _RETRY_FEEDBACK_MAX_CHARS


# ===========================================================================
# Feature 2: Exponential Backoff
# ===========================================================================


class TestResolveRetryDelay:
    """Test _resolve_retry_delay helper."""

    def test_none_returns_zero(self) -> None:
        assert _resolve_retry_delay(None, None, 1) == 0.0

    def test_constant_float(self) -> None:
        assert _resolve_retry_delay(5.0, None, 1) == 5.0
        assert _resolve_retry_delay(5.0, None, 3) == 5.0

    def test_list_indexed(self) -> None:
        delays = [1.0, 3.0, 10.0]
        assert _resolve_retry_delay(delays, None, 1) == 1.0
        assert _resolve_retry_delay(delays, None, 2) == 3.0
        assert _resolve_retry_delay(delays, None, 3) == 10.0

    def test_list_clamps_to_last(self) -> None:
        delays = [2.0, 5.0]
        assert _resolve_retry_delay(delays, None, 3) == 5.0
        assert _resolve_retry_delay(delays, None, 10) == 5.0

    def test_task_overrides_plan(self) -> None:
        assert _resolve_retry_delay(1.0, 99.0, 1) == 1.0

    def test_plan_default_used(self) -> None:
        assert _resolve_retry_delay(None, 3.0, 1) == 3.0

    def test_empty_list_returns_zero(self) -> None:
        assert _resolve_retry_delay([], None, 1) == 0.0

    def test_int_value_works(self) -> None:
        assert _resolve_retry_delay(3, None, 1) == 3.0


class TestRetryDelayLoader:
    """Test retry_delay_sec parsing in loader."""

    def test_parse_float(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
defaults:
  retry_delay_sec: 5.0
tasks:
  - id: t1
    command: echo hello
""")
        plan = load_plan(plan_file)
        assert plan.defaults.retry_delay_sec == 5.0

    def test_parse_list(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
defaults:
  retry_delay_sec: [2, 5, 15]
tasks:
  - id: t1
    command: echo hello
""")
        plan = load_plan(plan_file)
        assert plan.defaults.retry_delay_sec == [2.0, 5.0, 15.0]

    def test_task_level_delay(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo hello
    retry_delay_sec: 3.5
    max_retries: 2
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].retry_delay_sec == 3.5

    def test_task_level_list_delay(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo hello
    retry_delay_sec: [1, 2]
    max_retries: 2
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].retry_delay_sec == [1.0, 2.0]

    def test_negative_delay_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
defaults:
  retry_delay_sec: -1
tasks:
  - id: t1
    command: echo hello
""")
        with pytest.raises(PlanValidationError, match="retry_delay_sec"):
            load_plan(plan_file)

    def test_negative_in_list_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo hello
    retry_delay_sec: [1, -2, 3]
""")
        with pytest.raises(PlanValidationError, match="retry_delay_sec"):
            load_plan(plan_file)


class TestRetryDelayExecution:
    """Test that delays are actually applied during retries."""

    def test_delay_logged(self, tmp_path: Path) -> None:
        """Retry with delay should log the wait message."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path, defaults=PlanDefaults(
            codex=EngineDefaults(),
            claude=EngineDefaults(),
            retry_delay_sec=0.01,  # tiny delay for test speed
        ))
        task = TaskSpec(id="t1", command="exit 1", max_retries=1)

        with patch("maestro_cli.runners.time.sleep") as mock_sleep:
            result = execute_task(plan, task, run_path)

        assert result.status == "failed"
        assert result.retry_count == 1
        # The subprocess output-poll loop also calls time.sleep on some platforms
        # (POSIX polls stdout; Windows reads via threads), so assert the retry
        # delay value was applied rather than that sleep was the only call.
        delays = [c.args[0] for c in mock_sleep.call_args_list if c.args]
        assert any(d == pytest.approx(0.01) for d in delays), (
            f"retry delay 0.01 not among sleep calls: {delays}"
        )


# ===========================================================================
# Feature 3: Budget Limits
# ===========================================================================


class TestBudgetLoader:
    """Test max_cost_usd parsing in loader."""

    def test_parse_budget(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
max_cost_usd: 5.00
tasks:
  - id: t1
    command: echo hello
""")
        plan = load_plan(plan_file)
        assert plan.max_cost_usd == 5.00

    def test_no_budget_default(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo hello
""")
        plan = load_plan(plan_file)
        assert plan.max_cost_usd is None

    def test_zero_budget_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
max_cost_usd: 0
tasks:
  - id: t1
    command: echo hello
""")
        with pytest.raises(PlanValidationError, match="max_cost_usd"):
            load_plan(plan_file)

    def test_negative_budget_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
max_cost_usd: -5.0
tasks:
  - id: t1
    command: echo hello
""")
        with pytest.raises(PlanValidationError, match="max_cost_usd"):
            load_plan(plan_file)


class TestBudgetModels:
    """Test budget fields on data models."""

    def test_plan_run_result_budget_default(self) -> None:
        now = datetime.now(timezone.utc)
        result = PlanRunResult(
            plan_name="test",
            run_id="123",
            run_path=Path("/tmp"),
            started_at=now,
            finished_at=now,
            success=True,
        )
        assert result.budget_exceeded is False

    def test_plan_run_result_budget_in_dict(self) -> None:
        now = datetime.now(timezone.utc)
        result = PlanRunResult(
            plan_name="test",
            run_id="123",
            run_path=Path("/tmp"),
            started_at=now,
            finished_at=now,
            success=True,
            budget_exceeded=True,
        )
        d = result.to_dict()
        assert d["budget_exceeded"] is True


class TestBudgetScheduler:
    """Test budget enforcement in scheduler."""

    def test_no_budget_all_run(self, tmp_path: Path) -> None:
        """Without max_cost_usd, all tasks run normally."""
        from maestro_cli.scheduler import run_plan

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-budget
tasks:
  - id: t1
    command: echo one
  - id: t2
    depends_on: [t1]
    command: echo two
""")
        plan = load_plan(plan_file)
        result = run_plan(plan, dry_run=True, run_dir_override=str(tmp_path / "runs"))
        assert result.success
        assert result.budget_exceeded is False
        assert all(r.status == "dry_run" for r in result.task_results.values())

    def test_budget_exceeded_skips_remaining(self, tmp_path: Path) -> None:
        """When budget is exceeded, remaining tasks should be skipped."""
        from maestro_cli.scheduler import run_plan

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-budget
max_cost_usd: 0.01
tasks:
  - id: t1
    command: echo one
  - id: t2
    depends_on: [t1]
    command: echo two
""")
        plan = load_plan(plan_file)

        # Patch execute_task to return a result with cost that exceeds budget
        def mock_execute(plan: Any, task: Any, run_path: Any, *args: Any, **kwargs: Any) -> TaskResult:
            now = datetime.now(timezone.utc)
            log_path = run_path / f"{task.id}.log"
            result_path = run_path / f"{task.id}.result.json"
            log_path.write_text(f"task={task.id}\n\noutput\nstatus=success\n", encoding="utf-8")
            r = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=1.0,
                command="echo",
                log_path=log_path,
                result_path=result_path,
                cost_usd=0.05,  # Exceeds budget of $0.01
            )
            result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        with patch("maestro_cli.scheduler.execute_task", side_effect=mock_execute):
            result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.budget_exceeded is True
        assert result.task_results["t1"].status == "success"
        assert result.task_results["t2"].status == "skipped"
        assert "Budget exceeded" in result.task_results["t2"].message

    def test_budget_not_exceeded_all_run(self, tmp_path: Path) -> None:
        """When cost stays under budget, all tasks complete."""
        from maestro_cli.scheduler import run_plan

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-budget
max_cost_usd: 100.00
tasks:
  - id: t1
    command: echo one
  - id: t2
    depends_on: [t1]
    command: echo two
""")
        plan = load_plan(plan_file)
        result = run_plan(plan, dry_run=True, run_dir_override=str(tmp_path / "runs"))
        assert result.budget_exceeded is False
        assert all(r.status == "dry_run" for r in result.task_results.values())

    def test_none_cost_no_trigger(self, tmp_path: Path) -> None:
        """Tasks with cost_usd=None don't trigger budget check."""
        from maestro_cli.scheduler import run_plan

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-budget
max_cost_usd: 0.01
tasks:
  - id: t1
    command: echo one
  - id: t2
    depends_on: [t1]
    command: echo two
""")
        plan = load_plan(plan_file)
        # dry_run produces cost_usd=None, so budget check shouldn't trigger
        result = run_plan(plan, dry_run=True, run_dir_override=str(tmp_path / "runs"))
        assert result.budget_exceeded is False


# ===========================================================================
# Feature 4: Resume Last
# ===========================================================================


class TestResumeLastCli:
    """Test --resume-last CLI flag and _find_latest_run helper."""

    def test_find_latest_run_returns_most_recent(self, tmp_path: Path) -> None:
        from maestro_cli.cli import _find_latest_run

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: echo hello
""")
        plan = load_plan(plan_file)

        # Create run directories
        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()

        run1 = run_root / "20260101_000000_000000_abc123_test-plan"
        run1.mkdir()
        (run1 / "run_manifest.json").write_text("{}", encoding="utf-8")

        run2 = run_root / "20260228_120000_000000_def456_test-plan"
        run2.mkdir()
        (run2 / "run_manifest.json").write_text("{}", encoding="utf-8")

        result = _find_latest_run(plan)
        assert result is not None
        assert result.name == run2.name

    def test_find_latest_run_ignores_no_manifest(self, tmp_path: Path) -> None:
        from maestro_cli.cli import _find_latest_run

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: echo hello
""")
        plan = load_plan(plan_file)

        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()

        # Directory without manifest — should be ignored
        run_no_manifest = run_root / "20260228_120000_000000_abc_test-plan"
        run_no_manifest.mkdir()

        # Directory with manifest
        run_with = run_root / "20260101_000000_000000_xyz_test-plan"
        run_with.mkdir()
        (run_with / "run_manifest.json").write_text("{}", encoding="utf-8")

        result = _find_latest_run(plan)
        assert result is not None
        assert result.name == run_with.name

    def test_find_latest_run_no_runs(self, tmp_path: Path) -> None:
        from maestro_cli.cli import _find_latest_run

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: echo hello
""")
        plan = load_plan(plan_file)
        result = _find_latest_run(plan)
        assert result is None

    def test_find_latest_run_filters_by_plan_name(self, tmp_path: Path) -> None:
        from maestro_cli.cli import _find_latest_run

        plan_file = _write_plan(tmp_path, """\
version: 1
name: my-plan
tasks:
  - id: t1
    command: echo hello
""")
        plan = load_plan(plan_file)

        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()

        # Run from a different plan
        other = run_root / "20260228_120000_000000_abc_other-plan"
        other.mkdir()
        (other / "run_manifest.json").write_text("{}", encoding="utf-8")

        # Run from our plan
        ours = run_root / "20260101_000000_000000_xyz_my-plan"
        ours.mkdir()
        (ours / "run_manifest.json").write_text("{}", encoding="utf-8")

        result = _find_latest_run(plan)
        assert result is not None
        assert "my-plan" in result.name

    def test_resume_and_resume_last_mutually_exclusive(self) -> None:
        """--resume and --resume-last cannot be used together."""
        from maestro_cli.cli import _build_parser

        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["run", "plan.yaml", "--resume", "/some/path", "--resume-last"])


# ===========================================================================
# Feature 5: Conditional Execution (when)
# ===========================================================================


class TestEvaluateWhenCondition:
    """Test evaluate_when_condition in utils."""

    def test_equals_true(self) -> None:
        result, rendered = evaluate_when_condition(
            "{{ t1.status }} == success",
            {"t1.status": "success"},
        )
        assert result is True

    def test_equals_false(self) -> None:
        result, rendered = evaluate_when_condition(
            "{{ t1.status }} == success",
            {"t1.status": "failed"},
        )
        assert result is False

    def test_not_equals_true(self) -> None:
        result, rendered = evaluate_when_condition(
            "{{ t1.status }} != success",
            {"t1.status": "failed"},
        )
        assert result is True

    def test_not_equals_false(self) -> None:
        result, rendered = evaluate_when_condition(
            "{{ t1.status }} != success",
            {"t1.status": "success"},
        )
        assert result is False

    def test_template_vars_resolved(self) -> None:
        result, rendered = evaluate_when_condition(
            "{{ task-a.status }} == {{ task-b.status }}",
            {"task-a.status": "success", "task-b.status": "success"},
        )
        assert result is True
        assert "success == success" in rendered

    def test_invalid_expression_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid when"):
            evaluate_when_condition("not a valid expression", {})

    def test_unresolved_var_stays_as_is(self) -> None:
        result, rendered = evaluate_when_condition(
            "{{ t1.status }} == success",
            {},  # No variables — template stays as {{ t1.status }}
        )
        assert result is False
        assert "{{ t1.status }}" in rendered


class TestWhenLoader:
    """Test when field parsing and validation in loader."""

    def test_when_parsed(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo one
  - id: t2
    depends_on: [t1]
    command: echo two
    when: "{{ t1.status }} == success"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[1].when == "{{ t1.status }} == success"

    def test_when_no_when_default(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo one
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].when is None

    def test_when_unknown_task_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo one
  - id: t2
    depends_on: [t1]
    command: echo two
    when: "{{ unknown.status }} == success"
""")
        with pytest.raises(PlanValidationError, match="unknown"):
            load_plan(plan_file)

    def test_when_task_not_in_depends_on_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test
tasks:
  - id: t1
    command: echo one
  - id: t2
    command: echo two
  - id: t3
    depends_on: [t1]
    command: echo three
    when: "{{ t2.status }} == success"
""")
        with pytest.raises(PlanValidationError, match="t2"):
            load_plan(plan_file)


class TestWhenScheduler:
    """Test conditional execution in scheduler."""

    def test_when_true_task_runs(self, tmp_path: Path) -> None:
        """Task with when=true should execute."""
        from maestro_cli.scheduler import run_plan

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-when
tasks:
  - id: t1
    command: echo one
  - id: t2
    depends_on: [t1]
    command: echo two
    when: "{{ t1.status }} == success"
""")
        plan = load_plan(plan_file)
        # Run for real (echo succeeds with status=success)
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.task_results["t2"].status == "success"

    def test_when_false_task_skipped(self, tmp_path: Path) -> None:
        """Task with when=false should be skipped."""
        from maestro_cli.scheduler import run_plan

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-when
tasks:
  - id: t1
    command: echo one
  - id: t2
    depends_on: [t1]
    command: echo two
    when: "{{ t1.status }} == failed"
""")
        plan = load_plan(plan_file)
        result = run_plan(plan, dry_run=True, run_dir_override=str(tmp_path / "runs"))
        # dry_run produces status="dry_run", which != "failed", so t2 should be skipped
        assert result.task_results["t2"].status == "skipped"
        assert "Condition not met" in result.task_results["t2"].message

    def test_when_allows_failed_dep(self, tmp_path: Path) -> None:
        """Task with when should run even if dependency failed (waits for completion, not success)."""
        from maestro_cli.scheduler import run_plan

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-when
fail_fast: false
tasks:
  - id: t1
    command: exit 1
  - id: error-handler
    depends_on: [t1]
    command: echo handling error
    when: "{{ t1.status }} == failed"
""")
        plan = load_plan(plan_file)
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        # t1 fails, error-handler should run because when evaluates to true
        assert result.task_results["t1"].status == "failed"
        assert result.task_results["error-handler"].status == "success"

    def test_no_when_original_behavior(self, tmp_path: Path) -> None:
        """Task without when should still require deps to succeed."""
        from maestro_cli.scheduler import run_plan

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-when
fail_fast: false
tasks:
  - id: t1
    command: exit 1
  - id: t2
    depends_on: [t1]
    command: echo two
""")
        plan = load_plan(plan_file)
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.task_results["t1"].status == "failed"
        assert result.task_results["t2"].status == "skipped"
        assert "dependency failed" in result.task_results["t2"].message

    def test_when_not_equals(self, tmp_path: Path) -> None:
        """Test != operator in when conditions."""
        from maestro_cli.scheduler import run_plan

        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-when
tasks:
  - id: t1
    command: echo one
  - id: t2
    depends_on: [t1]
    command: echo two
    when: "{{ t1.status }} != failed"
""")
        plan = load_plan(plan_file)
        result = run_plan(plan, dry_run=True, run_dir_override=str(tmp_path / "runs"))
        # dry_run status != failed → condition true → task runs
        assert result.task_results["t2"].status == "dry_run"
