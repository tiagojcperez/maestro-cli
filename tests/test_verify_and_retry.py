from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import (
    MAX_RETRIES_LIMIT,
    EngineDefaults,
    PlanDefaults,
    PlanSpec,
    TaskResult,
    TaskSpec,
)
from maestro_cli.runners import execute_task


def _write_plan(tmp_path: Path, content: str) -> Path:
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(content, encoding="utf-8")
    return plan_file


def _make_plan(tmp_path: Path) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="test",
        workspace_root=str(tmp_path),
        max_parallel=1,
        fail_fast=True,
        run_dir=str(tmp_path / "runs"),
        defaults=PlanDefaults(
            codex=EngineDefaults(),
            claude=EngineDefaults(),
        ),
        tasks=[],
    )


# ===========================================================================
# verify_command execution tests
# ===========================================================================


class TestVerifyCommandExecution:
    """Test that verify_command runs after main command and affects status."""

    def test_verify_success_keeps_status(self, tmp_path: Path, monkeypatch: Any) -> None:
        """Main succeeds + verify succeeds = success."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            command="echo hello",
            verify_command="echo ok",
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        log_content = result.log_path.read_text(encoding="utf-8")
        assert "[verify_command]" in log_content

    def test_verify_failure_overrides_success(self, tmp_path: Path) -> None:
        """Main succeeds + verify fails = failed."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            command="echo hello",
            verify_command="exit 1",
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert "verify_command failed" in result.message

    def test_verify_skipped_on_main_failure(self, tmp_path: Path) -> None:
        """Main fails = verify_command does NOT run."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            command="exit 1",
            verify_command="echo this-should-not-run",
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        log_content = result.log_path.read_text(encoding="utf-8")
        assert "[verify_command]" not in log_content
        assert "this-should-not-run" not in log_content

    def test_verify_runs_on_soft_failed(self, tmp_path: Path) -> None:
        """Main fails + allow_failure = soft_failed, verify still runs."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            command="exit 1",
            allow_failure=True,
            verify_command="echo verify-ran",
        )

        result = execute_task(plan, task, run_path)
        # soft_failed + verify succeeds = soft_failed
        assert result.status == "soft_failed"
        log_content = result.log_path.read_text(encoding="utf-8")
        assert "[verify_command]" in log_content

    def test_no_verify_command_no_section(self, tmp_path: Path) -> None:
        """No verify_command = no [verify_command] section in log."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t1", command="echo hello")

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        log_content = result.log_path.read_text(encoding="utf-8")
        assert "[verify_command]" not in log_content


class TestTaskAssertionsExecution:
    def test_assert_success_keeps_status(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            command=[
                sys.executable,
                "-c",
                "from pathlib import Path; Path('artifact.txt').write_text('hello', encoding='utf-8')",
            ],
            assertions=[
                {"type": "file_contains", "path": "artifact.txt", "pattern": "hello"},
            ],
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        log_content = result.log_path.read_text(encoding="utf-8")
        assert "[assert]" in log_content

    def test_assert_failure_overrides_success(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            command=[
                sys.executable,
                "-c",
                "from pathlib import Path; Path('artifact.txt').write_text('hello', encoding='utf-8')",
            ],
            assertions=[
                {"type": "file_contains", "path": "artifact.txt", "pattern": "goodbye"},
            ],
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert "failed" in result.message.lower()

    def test_assert_failure_triggers_retry(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        counter_file = tmp_path / "counter.txt"
        counter_file.write_text("0", encoding="utf-8")
        command = [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; "
                f"counter = Path(r'{counter_file}'); "
                "value = counter.read_text(encoding='utf-8').strip(); "
                "artifact = Path('artifact.txt'); "
                "artifact.write_text('hello' if value == '1' else 'nope', encoding='utf-8'); "
                "counter.write_text('1', encoding='utf-8')"
            ),
        ]
        task = TaskSpec(
            id="t1",
            command=command,
            assertions=[
                {"type": "file_contains", "path": "artifact.txt", "pattern": "hello"},
            ],
            max_retries=1,
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        assert result.retry_count == 1


class TestTypedContractProduction:
    def test_file_inventory_contract_is_normalized(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="inventory",
            command=[
                sys.executable,
                "-c",
                "print('src/app.py\\nsrc/repo.py\\nsrc/app.py')",
            ],
            contract_type="file-inventory",
        )

        result = execute_task(plan, task, run_path)

        assert result.status == "success"
        assert result.produced_contract is not None
        assert result.produced_contract.contract_type == "file-inventory"
        assert result.produced_contract.metadata["file_count"] == 2
        assert "src/app.py" in result.produced_contract.body


# ===========================================================================
# max_retries validation tests
# ===========================================================================


class TestMaxRetriesValidation:
    def test_valid_max_retries_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    max_retries: 2
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].max_retries == 2

    def test_max_retries_zero_default(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].max_retries == 0

    def test_max_retries_negative_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    max_retries: -1
""")
        with pytest.raises(PlanValidationError, match="max_retries must be 0"):
            load_plan(plan_file)

    def test_max_retries_above_limit_raises(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    max_retries: 4
""")
        with pytest.raises(PlanValidationError, match="max_retries must be 0"):
            load_plan(plan_file)

    def test_max_retries_boundary_passes(self, tmp_path: Path) -> None:
        plan_file = _write_plan(tmp_path, f"""\
version: 1
name: test-plan
tasks:
  - id: t1
    command: "echo hello"
    max_retries: {MAX_RETRIES_LIMIT}
""")
        plan = load_plan(plan_file)
        assert plan.tasks[0].max_retries == MAX_RETRIES_LIMIT


# ===========================================================================
# max_retries execution tests
# ===========================================================================


class TestMaxRetriesExecution:
    """Test retry loop behaviour in execute_task."""

    def test_no_retry_on_success(self, tmp_path: Path) -> None:
        """Successful task = no retries, retry_count stays 0."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t1", command="echo hello", max_retries=2)

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        assert result.retry_count == 0

    def test_retry_on_failure_then_success(self, tmp_path: Path) -> None:
        """Task fails first, succeeds on retry."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)

        # Use a file counter to make first attempt fail, second succeed.
        # Portable command: read counter, if 0 -> write 1 and exit 1; if 1 -> exit 0.
        counter_file = tmp_path / "counter"
        counter_file.write_text("0", encoding="utf-8")
        cmd = [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                f"counter = Path(r'{counter_file}'); "
                "value = counter.read_text(encoding='utf-8').strip(); "
                "counter.write_text('1', encoding='utf-8'); "
                "sys.exit(1 if value == '0' else 0)"
            ),
        ]
        task = TaskSpec(id="t1", command=cmd, max_retries=2)

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        assert result.retry_count == 1

    def test_exhausts_retries_then_fails(self, tmp_path: Path) -> None:
        """Task fails all attempts."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t1", command="exit 1", max_retries=2)

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert result.retry_count == 2
        log_content = result.log_path.read_text(encoding="utf-8")
        assert "[retry 1/2]" in log_content
        assert "[retry 2/2]" in log_content
        assert result.handoff_report is not None
        assert "failed after" in result.handoff_report.summary
        assert "[handoff_report]" in log_content

    def test_verify_failure_triggers_retry(self, tmp_path: Path) -> None:
        """Main succeeds + verify fails = retry the whole thing."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)

        # Counter: verify fails on first attempt, succeeds on second.
        # Portable command: read counter, if 0 -> write 1 and exit 1; if 1 -> exit 0.
        counter_file = tmp_path / "vcounter"
        counter_file.write_text("0", encoding="utf-8")
        verify_cmd = [
            sys.executable,
            "-c",
            (
                "import sys; from pathlib import Path; "
                f"counter = Path(r'{counter_file}'); "
                "value = counter.read_text(encoding='utf-8').strip(); "
                "counter.write_text('1', encoding='utf-8'); "
                "sys.exit(1 if value == '0' else 0)"
            ),
        ]
        task = TaskSpec(
            id="t1",
            command="echo hello",
            verify_command=verify_cmd,
            max_retries=1,
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        assert result.retry_count == 1

    def test_pre_command_failure_no_retry(self, tmp_path: Path) -> None:
        """Pre-command failure = no retry (retries only apply to main+verify)."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            command="echo hello",
            pre_command="exit 1",
            max_retries=2,
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert result.retry_count == 0
        assert "pre_command failed" in result.message

    def test_engine_failure_triggers_single_fallback_retry(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            engine="codex",
            model="gpt-5-mini",
            fallback_engine="claude",
            fallback_model="opus",
            prompt="fix it",
            max_retries=1,
        )

        build_calls: list[tuple[str | None, str | None, str | None]] = []
        stream_results = [(1, "429 Too Many Requests", ""), (0, "ok", "")]
        events: list[tuple[str, dict[str, object]]] = []

        def _fake_build_command(
            _plan: PlanSpec,
            _task: TaskSpec,
            _workdir: Path,
            **kwargs: Any,
        ) -> tuple[list[str], bool]:
            engine_override = kwargs.get("engine_override")
            model_override = kwargs.get("model_override")
            retry_feedback = kwargs.get("retry_feedback")
            build_calls.append((engine_override, model_override, retry_feedback))
            engine = engine_override or _task.engine or "unknown"
            model = model_override or _task.model or "default"
            return ([engine, model], False)

        class _DummyProc:
            pass

        def _fake_stream_process(*args: Any, **kwargs: Any) -> tuple[int, str, str]:
            del args, kwargs
            return stream_results.pop(0)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners.subprocess.Popen", lambda *a, **kw: _DummyProc())
        monkeypatch.setattr("maestro_cli.runners._stream_process", _fake_stream_process)

        result = execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda name, payload: events.append((name, payload)),
        )

        assert result.status == "success"
        assert result.retry_count == 1
        assert build_calls[0][:2] == (None, None)
        assert build_calls[1][:2] == ("claude", "opus")
        assert len(build_calls) == 2
        fallback_events = [payload for name, payload in events if name == "engine_fallback"]
        assert len(fallback_events) == 1
        assert fallback_events[0]["from_engine"] == "codex"
        assert fallback_events[0]["to_engine"] == "claude"
        assert fallback_events[0]["reason"] == "test_failure"

    def test_verify_failure_does_not_trigger_engine_fallback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            engine="codex",
            fallback_engine="claude",
            fallback_model="opus",
            prompt="fix it",
            verify_command="exit 1",
            max_retries=1,
        )

        build_calls: list[tuple[str | None, str | None]] = []
        events: list[tuple[str, dict[str, object]]] = []

        def _fake_build_command(
            _plan: PlanSpec,
            _task: TaskSpec,
            _workdir: Path,
            **kwargs: Any,
        ) -> tuple[list[str], bool]:
            build_calls.append((kwargs.get("engine_override"), kwargs.get("model_override")))
            engine = kwargs.get("engine_override") or _task.engine or "unknown"
            return ([engine], False)

        class _DummyProc:
            pass

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners.subprocess.Popen", lambda *a, **kw: _DummyProc())
        monkeypatch.setattr(
            "maestro_cli.runners._stream_process",
            lambda *a, **kw: (0, "ok", ""),
        )

        execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda name, payload: events.append((name, payload)),
        )

        assert not any(call[0] == "claude" for call in build_calls)
        assert not any(name == "engine_fallback" for name, _payload in events)


# ===========================================================================
# TaskResult model tests
# ===========================================================================


class TestTaskResultRetryCount:
    def test_default_zero(self) -> None:
        result = TaskResult(
            task_id="t1",
            status="success",
            exit_code=0,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            duration_sec=1.0,
            command="echo",
            log_path=Path("/tmp/t.log"),
            result_path=Path("/tmp/t.json"),
        )
        assert result.retry_count == 0

    def test_to_dict_includes_retry_count(self) -> None:
        result = TaskResult(
            task_id="t1",
            status="success",
            exit_code=0,
            started_at=datetime.now(timezone.utc),
            finished_at=datetime.now(timezone.utc),
            duration_sec=1.0,
            command="echo",
            log_path=Path("/tmp/t.log"),
            result_path=Path("/tmp/t.json"),
            retry_count=2,
        )
        d = result.to_dict()
        assert d["retry_count"] == 2

    def test_max_retries_default_on_taskspec(self) -> None:
        task = TaskSpec(id="t1")
        assert task.max_retries == 0

    def test_verify_command_default_on_taskspec(self) -> None:
        task = TaskSpec(id="t1")
        assert task.verify_command is None


class TestVerifyOutputInMessage:
    """P0: verify/guard output included in failure messages."""

    def test_verify_output_in_failure_message(self, tmp_path: Path) -> None:
        """Verify output snippet appears in TaskResult.message."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            command="echo hello",
            verify_command=[sys.executable, "-c", "import sys; print('assert page-break in html failed'); sys.exit(1)"],
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert "verify output:" in result.message
        assert "page-break" in result.message

    def test_verify_empty_output_no_hint(self, tmp_path: Path) -> None:
        """Empty verify output → no hint appended."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            command="echo hello",
            verify_command="exit 1",
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert "verify output:" not in result.message

    def test_verify_long_output_truncated(self, tmp_path: Path) -> None:
        """Verify output > 300 chars is truncated with ... prefix."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        long_msg = "x" * 400
        task = TaskSpec(
            id="t1",
            command="echo hello",
            verify_command=[sys.executable, "-c", f"import sys; print('{long_msg}'); sys.exit(1)"],
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert "verify output:" in result.message
        assert "..." in result.message
        # Should not exceed 300 chars of actual verify text in the hint
        hint_start = result.message.index("verify output: ")
        hint_text = result.message[hint_start + len("verify output: "):]
        # Remove trailing )
        hint_text = hint_text.rstrip(")")
        assert len(hint_text) <= 305  # 300 + "..."


class TestVerifyFailureEvent:
    """P3: verify_failure event emitted on verify failure."""

    def test_verify_failure_emits_event(self, tmp_path: Path) -> None:
        """verify_failure event emitted with task_id, exit_code, output_snippet."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            command="echo hello",
            verify_command=[sys.executable, "-c", "import sys; print('validation failed'); sys.exit(1)"],
        )

        events: list[tuple[str, dict]] = []
        result = execute_task(
            plan, task, run_path,
            event_callback=lambda name, payload: events.append((name, payload)),
        )
        assert result.status == "failed"
        verify_events = [(n, p) for n, p in events if n == "verify_failure"]
        assert len(verify_events) == 1
        _, payload = verify_events[0]
        assert payload["task_id"] == "t1"
        assert payload["exit_code"] == 1
        assert "validation failed" in payload["output_snippet"]

    def test_verify_success_no_event(self, tmp_path: Path) -> None:
        """No verify_failure event when verify passes."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t1",
            command="echo hello",
            verify_command="echo ok",
        )

        events: list[tuple[str, dict]] = []
        result = execute_task(
            plan, task, run_path,
            event_callback=lambda name, payload: events.append((name, payload)),
        )
        assert result.status == "success"
        verify_events = [n for n, _ in events if n == "verify_failure"]
        assert len(verify_events) == 0
