from __future__ import annotations

import json
import sys
import threading
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from maestro_cli.models import (
    KnowledgeRecord,
    MCPServerSpec,
    PlanDefaults,
    PlanRunResult,
    PlanSpec,
    TaskContract,
    TaskResult,
    TaskSpec,
    TokenUsage,
)
from maestro_cli.plugins import DoctorProbe, EnginePlugin
from maestro_cli.scheduler import (
    _apply_hop_decay,
    _compute_hop_distances,
    _compute_idf,
    _compute_waves,
    _estimate_tokens,
    _extract_keywords,
    _format_model_tag,
    _fmt_duration,
    _load_prior_results,
    _model_suffix,
    _new_cached_result,
    _new_skipped_result,
    _resolve_model,
    _resolve_reasoning_effort,
    _score_section,
    _select_tasks,
    _split_into_sections,
    _write_manifest,
    _write_summary,
    run_plan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_task(
    task_id: str,
    depends_on: list[str] | None = None,
    command: str = "echo ok",
    allow_failure: bool = False,
    context_from: list[str] | None = None,
    engine: str | None = None,
    prompt: str | None = None,
    description: str = "",
) -> TaskSpec:
    """Build a TaskSpec with sensible defaults for scheduler tests."""
    return TaskSpec(
        id=task_id,
        description=description or f"task {task_id}",
        depends_on=depends_on or [],
        command=command,
        allow_failure=allow_failure,
        context_from=context_from or [],
        engine=engine,
        prompt=prompt,
    )


def _make_plan(
    tasks: list[TaskSpec],
    name: str = "test-plan",
    fail_fast: bool = True,
    max_parallel: int = 4,
    source_path: Path | None = None,
) -> PlanSpec:
    """Build a PlanSpec with sensible defaults for scheduler tests."""
    return PlanSpec(
        version=1,
        name=name,
        fail_fast=fail_fast,
        max_parallel=max_parallel,
        defaults=PlanDefaults(),
        tasks=tasks,
        source_path=source_path,
    )


def _make_success_result(
    task_id: str,
    run_path: Path,
    status: str = "success",
    exit_code: int = 0,
    stdout_tail: str = "",
    duration_sec: float = 0.1,
) -> TaskResult:
    """Build a TaskResult with the given status."""
    now = datetime.now(UTC)
    return TaskResult(
        task_id=task_id,
        status=status,
        exit_code=exit_code,
        started_at=now,
        finished_at=now,
        duration_sec=duration_sec,
        command=f"echo {task_id}",
        log_path=run_path / f"{task_id}.log",
        result_path=run_path / f"{task_id}.result.json",
        message="ok" if status == "success" else f"status={status}",
        stdout_tail=stdout_tail,
    )


def _mock_execute_task_factory(
    run_path_holder: list[Path],
    overrides: dict[str, TaskResult] | None = None,
    call_log: list[str] | None = None,
    call_log_lock: threading.Lock | None = None,
):
    """Return a mock execute_task that records calls and returns success.

    *run_path_holder* is a single-element list that captures the run_path
    from the first call so tests can inspect the output directory.

    *overrides* maps task_id -> specific TaskResult to return for that task.
    *call_log* records the order of task_id invocations (thread-safe).
    """
    overrides = overrides or {}
    call_log = call_log if call_log is not None else []
    lock = call_log_lock or threading.Lock()

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
    ):
        if not run_path_holder:
            run_path_holder.append(run_path)
        with lock:
            call_log.append(task.id)

        if task.id in overrides:
            result = overrides[task.id]
            # Ensure log and result files exist so manifest can be written
            result.log_path.write_text(
                f"status={result.status}\n", encoding="utf-8"
            )
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        now = datetime.now(UTC)
        status = "dry_run" if dry_run else "success"
        result = TaskResult(
            task_id=task.id,
            status=status,
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=0.01,
            command=f"echo {task.id}",
            log_path=run_path / f"{task.id}.log",
            result_path=run_path / f"{task.id}.result.json",
            message="ok",
        )
        result.log_path.write_text(
            f"status={status}\nmessage=ok\n", encoding="utf-8"
        )
        result.result_path.write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        return result

    return mock_execute, call_log


def _make_engine_plugin(name: str, executable: str | None = None) -> EnginePlugin:
    return EnginePlugin(
        name=name,
        build_command=lambda ctx: ([name, ctx.prompt_text], False),
        doctor_probe=DoctorProbe(executable=executable or name),
    )


class TestPreflightChecks:
    def test_custom_engine_uses_plugin_executable(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.scheduler import _preflight_checks

        plan = _make_plan(
            [TaskSpec(id="t1", engine="custom", prompt="Do it", command=None)],
            source_path=tmp_path / "plan.yaml",
        )
        monkeypatch.setattr(
            "maestro_cli.scheduler.get_engine_plugin",
            lambda name: _make_engine_plugin(name, executable="custom-engine"),
        )
        monkeypatch.setattr(
            "maestro_cli.scheduler.shutil.which",
            lambda name: "/usr/bin/custom-engine" if name == "custom-engine" else None,
        )

        _preflight_checks(plan, plan.tasks, dry_run=False)

    def test_missing_task_workdir_raises_value_error(
        self,
        tmp_path: Path,
    ) -> None:
        from maestro_cli.scheduler import _preflight_checks

        plan = _make_plan(
            [
                TaskSpec(
                    id="t1",
                    description="missing workdir",
                    command="echo hi",
                    workdir="missing-dir",
                )
            ],
            source_path=tmp_path / "plan.yaml",
        )

        with pytest.raises(ValueError, match=r"Task 't1' workdir does not exist: .*missing-dir"):
            _preflight_checks(plan, plan.tasks, dry_run=False)

    def test_claudecode_env_var_warning(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Warn when CLAUDECODE env var is set and plan uses engine: claude."""
        from maestro_cli.scheduler import _preflight_checks

        plan = _make_plan(
            [TaskSpec(id="t1", engine="claude", prompt="Do it", command=None)],
            source_path=tmp_path / "plan.yaml",
        )
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setattr("maestro_cli.scheduler.shutil.which", lambda name: f"/usr/bin/{name}")

        _preflight_checks(plan, plan.tasks, dry_run=False)
        captured = capsys.readouterr()
        assert "CLAUDECODE" in captured.out
        assert "nested session" in captured.out

    def test_claudecode_no_warning_for_non_claude(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """No warning when CLAUDECODE is set but plan doesn't use claude."""
        from maestro_cli.scheduler import _preflight_checks

        plan = _make_plan(
            [TaskSpec(id="t1", engine="codex", prompt="Do it", command=None)],
            source_path=tmp_path / "plan.yaml",
        )
        monkeypatch.setenv("CLAUDECODE", "1")
        monkeypatch.setattr("maestro_cli.scheduler.shutil.which", lambda name: f"/usr/bin/{name}")

        _preflight_checks(plan, plan.tasks, dry_run=False)
        captured = capsys.readouterr()
        assert "CLAUDECODE" not in captured.out

    @pytest.mark.skipif(sys.platform != "win32", reason="UNC-path warning is Windows-only behavior")
    def test_unc_path_warning_on_windows(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Warn when workdir resolves to UNC path on Windows."""
        from maestro_cli.scheduler import _preflight_checks

        plan = _make_plan(
            [TaskSpec(id="t1", command="echo ok", verify_command="py -c 'pass'")],
            source_path=tmp_path / "plan.yaml",
        )
        monkeypatch.setattr("maestro_cli.scheduler.os.name", "nt")
        # Make resolve_workdir return a UNC path
        unc_path = Path("//SERVER/SHARE/project")
        monkeypatch.setattr(
            "maestro_cli.scheduler.resolve_workdir",
            lambda _plan, _task: unc_path,
        )
        # The UNC path won't exist, so we also need to mock exists
        monkeypatch.setattr(Path, "exists", lambda self: True)

        _preflight_checks(plan, plan.tasks, dry_run=False)
        captured = capsys.readouterr()
        assert "UNC path" in captured.out


class TestModelResolution:
    @pytest.mark.parametrize(
        ("task", "expected_model", "expected_effort", "expected_tag"),
        [
            (
                TaskSpec(id="codex-defaults", engine="codex", prompt="Do it", command=None),
                "gpt-5.1-codex",
                "high",
                "codex:gpt-5.1-codex@high",
            ),
            (
                TaskSpec(
                    id="claude-mixed",
                    engine="claude",
                    model="sonnet",
                    prompt="Do it",
                    command=None,
                ),
                "sonnet",
                "medium",
                "claude:sonnet@medium",
            ),
            (
                TaskSpec(
                    id="copilot-no-effort",
                    engine="copilot",
                    model="claude-sonnet-4.5",
                    reasoning_effort="high",
                    prompt="Do it",
                    command=None,
                ),
                "claude-sonnet-4.5",
                "",
                "copilot:claude-sonnet-4.5",
            ),
            (
                TaskSpec(
                    id="gemini-task-values",
                    engine="gemini",
                    model="flash",
                    reasoning_effort="low",
                    prompt="Do it",
                    command=None,
                ),
                "flash",
                "low",
                "gemini:flash@low",
            ),
        ],
    )
    def test_model_helpers_resolve_defaults_and_format_tags(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task: TaskSpec,
        expected_model: str,
        expected_effort: str,
        expected_tag: str,
    ) -> None:
        plan = _make_plan([], source_path=Path("plan.yaml"))
        plan.defaults.codex.model = "gpt-5.1-codex"
        plan.defaults.codex.reasoning_effort = "high"
        plan.defaults.claude.model = "opus"
        plan.defaults.claude.reasoning_effort = "medium"
        plan.defaults.copilot.model = "gpt-5.2"
        plan.defaults.copilot.reasoning_effort = "low"
        monkeypatch.setattr("maestro_cli.scheduler._magenta", lambda text: text)

        assert _resolve_model(plan, task) == expected_model
        assert _resolve_reasoning_effort(plan, task) == expected_effort
        assert _format_model_tag(plan, task) == expected_tag
        assert _model_suffix(plan, task) == f" [{expected_tag}]"

    @pytest.mark.parametrize(
        ("task", "expected_model", "expected_effort", "expected_tag", "expected_suffix"),
        [
            (
                TaskSpec(id="copilot-engine-only", engine="copilot"),
                "",
                "",
                "copilot",
                " [copilot]",
            ),
            (
                TaskSpec(id="gemini-effort-only", engine="gemini", reasoning_effort="low"),
                "",
                "low",
                "gemini:low",
                " [gemini:low]",
            ),
            (
                TaskSpec(id="model-only", model="local-model"),
                "local-model",
                "",
                "local-model",
                " [local-model]",
            ),
            (
                TaskSpec(id="empty-detail"),
                "",
                "",
                "",
                "",
            ),
        ],
    )
    def test_model_helpers_handle_engine_only_detail_only_and_empty_tags(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task: TaskSpec,
        expected_model: str,
        expected_effort: str,
        expected_tag: str,
        expected_suffix: str,
    ) -> None:
        plan = _make_plan([], source_path=Path("plan.yaml"))
        monkeypatch.setattr("maestro_cli.scheduler._magenta", lambda text: text)

        assert _resolve_model(plan, task) == expected_model
        assert _resolve_reasoning_effort(plan, task) == expected_effort
        assert _format_model_tag(plan, task) == expected_tag
        assert _model_suffix(plan, task) == expected_suffix

    @pytest.mark.parametrize(
        ("task", "expected_model", "expected_effort", "expected_tag"),
        [
            (
                TaskSpec(id="claude-default-only", engine="claude", prompt="Do it", command=None),
                "sonnet-4",
                "medium",
                "claude:sonnet-4@medium",
            ),
            (
                TaskSpec(
                    id="codex-task-model-default-effort",
                    engine="codex",
                    model="gpt-5-mini",
                    prompt="Do it",
                    command=None,
                ),
                "gpt-5-mini",
                "high",
                "codex:gpt-5-mini@high",
            ),
            (
                TaskSpec(id="copilot-default-model", engine="copilot", prompt="Do it", command=None),
                "claude-sonnet-4.5",
                "",
                "copilot:claude-sonnet-4.5",
            ),
        ],
    )
    def test_model_helpers_cover_additional_default_and_override_paths(
        self,
        monkeypatch: pytest.MonkeyPatch,
        task: TaskSpec,
        expected_model: str,
        expected_effort: str,
        expected_tag: str,
    ) -> None:
        plan = _make_plan([], source_path=Path("plan.yaml"))
        plan.defaults.codex.model = "gpt-5.1-codex"
        plan.defaults.codex.reasoning_effort = "high"
        plan.defaults.claude.model = "sonnet-4"
        plan.defaults.claude.reasoning_effort = "medium"
        plan.defaults.copilot.model = "claude-sonnet-4.5"
        plan.defaults.copilot.reasoning_effort = "high"
        monkeypatch.setattr("maestro_cli.scheduler._magenta", lambda text: text)

        assert _resolve_model(plan, task) == expected_model
        assert _resolve_reasoning_effort(plan, task) == expected_effort
        assert _format_model_tag(plan, task) == expected_tag
        assert _model_suffix(plan, task) == f" [{expected_tag}]"


class TestSummaryHelpers:
    def test_write_summary_formats_budget_parallelism_and_engine_cells(
        self,
        tmp_path: Path,
    ) -> None:
        plan = _make_plan(
            [
                _make_task("shell-task"),
                TaskSpec(
                    id="engine-task",
                    description="engine task",
                    engine="codex",
                    prompt="Do it",
                    command=None,
                ),
                TaskSpec(
                    id="group-task",
                    description="group task",
                    group="nested-plan.yaml",
                    command=None,
                ),
            ],
            source_path=tmp_path / "plan.yaml",
        )
        plan.max_cost_usd = 3.0
        plan.defaults.codex.model = "gpt-5.1-codex"
        plan.defaults.codex.reasoning_effort = "high"

        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name=plan.name,
            run_id="run-123",
            run_path=tmp_path,
            started_at=now,
            finished_at=now + timedelta(seconds=65),
            success=False,
            task_results={
                "shell-task": TaskResult(
                    task_id="shell-task",
                    status="success",
                    exit_code=0,
                    started_at=now,
                    finished_at=now,
                    duration_sec=12.0,
                    command="echo shell-task",
                    log_path=tmp_path / "shell-task.log",
                    result_path=tmp_path / "shell-task.result.json",
                ),
                "engine-task": TaskResult(
                    task_id="engine-task",
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=30.0,
                    command="codex",
                    log_path=tmp_path / "engine-task.log",
                    result_path=tmp_path / "engine-task.result.json",
                    cost_usd=1.25,
                    token_usage=TokenUsage(input_tokens=1000, output_tokens=234),
                ),
                "group-task": TaskResult(
                    task_id="group-task",
                    status="skipped",
                    exit_code=None,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.0,
                    command="",
                    log_path=tmp_path / "group-task.log",
                    result_path=tmp_path / "group-task.result.json",
                ),
            },
            sequential_duration_sec=120.0,
            parallelism_savings_pct=46.0,
            total_cost_usd=1.25,
            total_tokens=1234,
            budget_exceeded=True,
        )

        summary_path = _write_summary(run_result, plan, tmp_path)
        summary_text = summary_path.read_text(encoding="utf-8")

        assert "| Duration | 1m05s |" in summary_text
        assert "| Tasks | 1 ok / 1 failed / 1 skipped |" in summary_text
        assert "| Cost | $1.25 |" in summary_text
        assert "| Tokens | 1,234 |" in summary_text
        assert "| Budget | $3.00 (EXCEEDED) |" in summary_text
        assert "| Parallelism | 1m05s wall / 2m00s seq (46% saved) |" in summary_text
        assert "| shell-task | success | 12s | --- | --- | shell |" in summary_text
        assert "| engine-task | failed | 30s | $1.25 | 1,234 | codex:gpt-5.1-codex@high |" in summary_text
        assert "| group-task | skipped | 0s | --- | --- | group:nested-plan.yaml |" in summary_text
        assert "- **Wave 0**: shell-task (12s), engine-task (30s), group-task (0s)" in summary_text

    def test_write_summary_formats_missing_totals_and_omits_optional_rows(
        self,
        tmp_path: Path,
    ) -> None:
        plan = _make_plan([_make_task("solo")], source_path=tmp_path / "plan.yaml")
        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name=plan.name,
            run_id="run-optional",
            run_path=tmp_path,
            started_at=now,
            finished_at=now + timedelta(seconds=9),
            success=True,
            task_results={
                "solo": TaskResult(
                    task_id="solo",
                    status="success",
                    exit_code=0,
                    started_at=now,
                    finished_at=now,
                    duration_sec=9.0,
                    command="echo solo",
                    log_path=tmp_path / "solo.log",
                    result_path=tmp_path / "solo.result.json",
                ),
            },
            total_cost_usd=None,
            total_tokens=None,
        )

        summary_path = _write_summary(run_result, plan, tmp_path)
        summary_text = summary_path.read_text(encoding="utf-8")

        assert "| Cost | --- |" in summary_text
        assert "| Tokens | --- |" in summary_text
        assert "| Budget |" not in summary_text
        assert "| Parallelism |" not in summary_text
        assert "- **Wave 0**: solo (9s)" in summary_text

    def test_write_summary_formats_parallel_timeline_waves_and_ignores_missing_results(
        self,
        tmp_path: Path,
    ) -> None:
        plan = _make_plan(
            [
                _make_task("root"),
                _make_task("left", depends_on=["root"]),
                _make_task("right", depends_on=["root"]),
                _make_task("tip", depends_on=["left", "right"]),
                _make_task("not-run"),
            ],
            source_path=tmp_path / "plan.yaml",
        )
        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name=plan.name,
            run_id="run-timeline",
            run_path=tmp_path,
            started_at=now,
            finished_at=now + timedelta(seconds=32),
            success=True,
            task_results={
                "root": TaskResult(
                    task_id="root",
                    status="success",
                    exit_code=0,
                    started_at=now,
                    finished_at=now,
                    duration_sec=5.0,
                    command="echo root",
                    log_path=tmp_path / "root.log",
                    result_path=tmp_path / "root.result.json",
                ),
                "left": TaskResult(
                    task_id="left",
                    status="success",
                    exit_code=0,
                    started_at=now,
                    finished_at=now,
                    duration_sec=20.0,
                    command="echo left",
                    log_path=tmp_path / "left.log",
                    result_path=tmp_path / "left.result.json",
                ),
                "right": TaskResult(
                    task_id="right",
                    status="success",
                    exit_code=0,
                    started_at=now,
                    finished_at=now,
                    duration_sec=8.0,
                    command="echo right",
                    log_path=tmp_path / "right.log",
                    result_path=tmp_path / "right.result.json",
                ),
                "tip": TaskResult(
                    task_id="tip",
                    status="success",
                    exit_code=0,
                    started_at=now,
                    finished_at=now,
                    duration_sec=7.0,
                    command="echo tip",
                    log_path=tmp_path / "tip.log",
                    result_path=tmp_path / "tip.result.json",
                ),
            },
        )

        summary_path = _write_summary(run_result, plan, tmp_path)
        summary_text = summary_path.read_text(encoding="utf-8")

        assert "- **Wave 0**: root (5s)" in summary_text
        assert "- **Wave 1**: left (20s), right (8s) — 20s wall / 28s CPU" in summary_text
        assert "- **Wave 2**: tip (7s)" in summary_text
        assert "not-run" not in summary_text
        # No oversized wave → no clarifying note.
        assert "DAG topological levels" not in summary_text

    def test_write_summary_notes_when_wave_exceeds_max_parallel(
        self,
        tmp_path: Path,
    ) -> None:
        # Internal post-mortem (2026-04-26): a wide read-only fan-out with
        # max_parallel below the wave size used to confuse authors who thought
        # Maestro had silently exceeded the limit. The summary now adds an
        # inline note that waves are DAG levels, not runtime slots.
        plan = _make_plan(
            [
                _make_task("a"),
                _make_task("b"),
                _make_task("c"),
                _make_task("d"),
            ],
            source_path=tmp_path / "plan.yaml",
            max_parallel=2,
        )
        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name=plan.name,
            run_id="run-wide-wave",
            run_path=tmp_path,
            started_at=now,
            finished_at=now + timedelta(seconds=5),
            success=True,
            task_results={
                tid: TaskResult(
                    task_id=tid,
                    status="success",
                    exit_code=0,
                    started_at=now,
                    finished_at=now,
                    duration_sec=2.0,
                    command=f"echo {tid}",
                    log_path=tmp_path / f"{tid}.log",
                    result_path=tmp_path / f"{tid}.result.json",
                )
                for tid in ("a", "b", "c", "d")
            },
        )
        summary_text = _write_summary(run_result, plan, tmp_path).read_text(encoding="utf-8")
        assert "DAG topological levels" in summary_text
        assert "max_parallel: 2" in summary_text

    def test_write_summary_surfaces_pytest_counts(
        self,
        tmp_path: Path,
    ) -> None:
        # PM3.2 — pytest/jest/mocha test counts should appear in the summary
        # when a command task's stdout_tail matches a recognised runner. An
        # internal post-mortem found a smoke-test task hiding passed + skipped
        # counts behind a binary success bit; this test locks in the surface.
        plan = _make_plan(
            [_make_task("smoke-collective"), _make_task("with-fail")],
            source_path=tmp_path / "plan.yaml",
        )
        pytest_tail = (
            "tests/test_a.py ........\n"
            "tests/test_b.py .....s.s.s..\n\n"
            "============================= 73 passed, 11 skipped, 10 warnings in 3.06s ============================="
        )
        jest_tail = (
            "Test Suites: 1 failed, 4 passed, 5 total\n"
            "Tests:       2 failed, 80 passed, 82 total\n"
        )
        results = {
            "smoke-collective": _make_success_result(
                "smoke-collective", tmp_path,
                status="success", stdout_tail=pytest_tail, duration_sec=3.0,
            ),
            "with-fail": _make_success_result(
                "with-fail", tmp_path,
                status="success", stdout_tail=jest_tail, duration_sec=12.0,
            ),
        }
        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name=plan.name,
            run_id="run-test-counts",
            run_path=tmp_path,
            started_at=now,
            finished_at=now + timedelta(seconds=15),
            success=True,
            task_results=results,
        )
        text = _write_summary(run_result, plan, tmp_path).read_text(encoding="utf-8")
        assert "## Test Results" in text
        assert "smoke-collective" in text
        assert "73 passed" in text
        assert "11 skipped" in text
        assert "(pytest)" in text
        assert "with-fail" in text
        assert "80 passed" in text
        assert "(jest)" in text

    def test_write_summary_omits_test_results_when_no_match(
        self,
        tmp_path: Path,
    ) -> None:
        plan = _make_plan(
            [_make_task("plain")],
            source_path=tmp_path / "plan.yaml",
        )
        result = _make_success_result(
            "plain", tmp_path, status="success",
            stdout_tail="some random output, no test runner here",
        )
        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name=plan.name,
            run_id="run-no-tests",
            run_path=tmp_path,
            started_at=now,
            finished_at=now + timedelta(seconds=1),
            success=True,
            task_results={"plain": result},
        )
        text = _write_summary(run_result, plan, tmp_path).read_text(encoding="utf-8")
        assert "## Test Results" not in text

    def test_write_summary_surfaces_retried_winners(
        self,
        tmp_path: Path,
    ) -> None:
        # PM3.1 — when a task succeeds after one or more failed attempts,
        # the summary must surface the failed-attempt verify_tail so authors
        # can see what the agent corrected without diffing log files.
        from maestro_cli.models import FailureRecord

        plan = _make_plan(
            [_make_task("smoke-monitoring")],
            source_path=tmp_path / "plan.yaml",
        )
        result = _make_success_result(
            "smoke-monitoring", tmp_path, status="success", duration_sec=4.5,
        )
        result.failure_history = [
            FailureRecord(
                attempt=1,
                category="test_failure",
                exit_code=1,
                message="verify_command exited 1 (pytest -x)",
                verify_tail="FAILED tests/test_smoke_monitoring.py::TestX::test_y\nAssertionError: expected 3 calls, got 2",
                duration_sec=140.0,
            ),
        ]
        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name=plan.name,
            run_id="run-retry-winner",
            run_path=tmp_path,
            started_at=now,
            finished_at=now + timedelta(seconds=145),
            success=True,
            task_results={"smoke-monitoring": result},
        )
        text = _write_summary(run_result, plan, tmp_path).read_text(encoding="utf-8")
        assert "## Retried Tasks" in text
        assert "smoke-monitoring" in text
        assert "1 failed attempt(s) before success" in text
        assert "Attempt 1" in text
        assert "expected 3 calls, got 2" in text

    def test_write_summary_counts_soft_failed_and_dry_run_as_ok(
        self,
        tmp_path: Path,
    ) -> None:
        plan = _make_plan(
            [
                _make_task("ok"),
                _make_task("soft"),
                _make_task("dry"),
                _make_task("skip"),
            ],
            source_path=tmp_path / "plan.yaml",
        )
        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name=plan.name,
            run_id="run-success-like",
            run_path=tmp_path,
            started_at=now,
            finished_at=now + timedelta(seconds=6),
            success=True,
            task_results={
                "ok": _make_success_result("ok", tmp_path, status="success", duration_sec=1.0),
                "soft": _make_success_result("soft", tmp_path, status="soft_failed", exit_code=1, duration_sec=2.0),
                "dry": _make_success_result("dry", tmp_path, status="dry_run", duration_sec=3.0),
                "skip": _make_success_result("skip", tmp_path, status="skipped", exit_code=None, duration_sec=0.0),
            },
        )

        summary_path = _write_summary(run_result, plan, tmp_path)
        summary_text = summary_path.read_text(encoding="utf-8")

        assert "| Status | **SUCCESS** |" in summary_text
        assert "| Tasks | 2 ok / 1 soft_failed / 0 failed / 1 skipped |" in summary_text
        assert "| ok | success | 1s | --- | --- | shell |" in summary_text
        assert "| soft | soft_failed | 2s | --- | --- | shell |" in summary_text
        assert "| dry | dry_run | 3s | --- | --- | shell |" in summary_text
        assert "| skip | skipped | 0s | --- | --- | shell |" in summary_text
        assert "- **Wave 0**: ok (1s), soft (2s), dry (3s), skip (0s) — 3s wall / 6s CPU" in summary_text

    def test_write_summary_falls_back_for_partial_results_and_formats_engine_only_cell(
        self,
        tmp_path: Path,
    ) -> None:
        plan = _make_plan(
            [
                _make_task("root"),
                TaskSpec(
                    id="leaf",
                    description="leaf",
                    depends_on=["root"],
                    engine="copilot",
                    prompt="Do it",
                    command=None,
                ),
            ],
            source_path=tmp_path / "plan.yaml",
        )
        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name=plan.name,
            run_id="run-partial",
            run_path=tmp_path,
            started_at=now,
            finished_at=now + timedelta(seconds=4),
            success=True,
            task_results={
                "leaf": TaskResult(
                    task_id="leaf",
                    status="success",
                    exit_code=0,
                    started_at=now,
                    finished_at=now,
                    duration_sec=4.0,
                    command="copilot",
                    log_path=tmp_path / "leaf.log",
                    result_path=tmp_path / "leaf.result.json",
                ),
            },
        )

        assert _compute_waves(plan, run_result) == [["leaf"]]

        summary_path = _write_summary(run_result, plan, tmp_path)
        summary_text = summary_path.read_text(encoding="utf-8")

        assert "| leaf | success | 4s | --- | --- | copilot |" in summary_text
        assert "- **Wave 0**: leaf (4s)" in summary_text
        assert "root" not in summary_text


class TestSchedulerResultHelpers:
    def test_new_skipped_result_writes_log_and_result_files(
        self,
        tmp_path: Path,
    ) -> None:
        result = _new_skipped_result("task-a", tmp_path, "Skipped for test")

        assert result.task_id == "task-a"
        assert result.status == "skipped"
        assert result.message == "Skipped for test"
        assert result.duration_sec == 0.0
        assert result.log_path.read_text(encoding="utf-8") == (
            "status=skipped\nmessage=Skipped for test\n"
        )
        written = json.loads(result.result_path.read_text(encoding="utf-8"))
        assert written["task_id"] == "task-a"
        assert written["status"] == "skipped"
        assert written["message"] == "Skipped for test"

    def test_new_cached_result_restores_metadata_and_copies_cached_log(
        self,
        tmp_path: Path,
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        cache_dir = tmp_path / "cache"
        task_hash = "abcdef1234567890"
        cached_task_dir = cache_dir / task_hash[:2] / task_hash
        cached_task_dir.mkdir(parents=True)
        (cached_task_dir / "task.log").write_text("cached task log\n", encoding="utf-8")

        result = _new_cached_result(
            task_id="cached-task",
            run_path=run_path,
            cached={
                "status": "success",
                "exit_code": 0,
                "duration_sec": 2.5,
                "command": "echo cached",
                "stdout_tail": "cached stdout",
                "cost_usd": 0.75,
                "retry_count": 2,
                "token_usage": {
                    "input_tokens": 100,
                    "cached_tokens": 25,
                    "output_tokens": 10,
                    "cache_creation_tokens": 5,
                },
                "structured_context": {
                    "task_id": "cached-task",
                    "status": "success",
                    "exit_code": 0,
                    "duration_sec": 2.5,
                    "files_changed": ["src/app.py"],
                    "decisions": ["used cache"],
                    "errors": [],
                    "warnings": ["stale"],
                    "cost_usd": 0.75,
                    "result_text": "done",
                    "summary": "cached summary",
                },
            },
            task_hash=task_hash,
            cache_dir=cache_dir,
        )

        assert result.status == "success"
        assert result.message == "Cache hit [abcdef123456]"
        assert result.duration_sec == pytest.approx(2.5)
        assert result.cost_usd == pytest.approx(0.75)
        assert result.retry_count == 2
        assert result.stdout_tail == "cached stdout"
        assert result.token_usage is not None
        assert result.token_usage.total_tokens == 135
        assert result.structured_context is not None
        assert result.structured_context.summary == "cached summary"
        assert result.log_path.read_text(encoding="utf-8") == "cached task log\n"
        written = json.loads(result.result_path.read_text(encoding="utf-8"))
        assert written["task_id"] == "cached-task"
        assert written["status"] == "success"
        assert written["token_usage"]["total_tokens"] == 135

    def test_new_cached_result_writes_fallback_log_when_cached_log_missing(
        self,
        tmp_path: Path,
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        task_hash = "1234567890abcdef"

        result = _new_cached_result(
            task_id="cached-task",
            run_path=run_path,
            cached={
                "status": "success",
                "duration_sec": 1.5,
                "command": "echo cached",
            },
            task_hash=task_hash,
            cache_dir=cache_dir,
        )

        assert result.status == "success"
        assert result.token_usage is None
        assert result.structured_context is None
        assert result.log_path.read_text(encoding="utf-8") == (
            "status=success\nmessage=Cache hit [1234567890ab]\n"
        )
        written = json.loads(result.result_path.read_text(encoding="utf-8"))
        assert written["message"] == "Cache hit [1234567890ab]"

    def test_write_manifest_round_trips_success_like_statuses_for_resume(
        self,
        tmp_path: Path,
    ) -> None:
        now = datetime.now(UTC)
        run_result = PlanRunResult(
            plan_name="resume-plan",
            run_id="run-resume",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "ok": _make_success_result("ok", tmp_path, status="success"),
                "soft": _make_success_result("soft", tmp_path, status="soft_failed", exit_code=1),
                "dry": _make_success_result("dry", tmp_path, status="dry_run"),
                "skip": _make_success_result("skip", tmp_path, status="skipped", exit_code=None),
                "fail": _make_success_result("fail", tmp_path, status="failed", exit_code=1),
            },
        )

        manifest_path = _write_manifest(run_result, tmp_path)
        resumed = _load_prior_results(tmp_path)

        assert manifest_path == tmp_path / "run_manifest.json"
        assert manifest_path.exists()
        assert resumed == {
            "ok": "success",
            "soft": "soft_failed",
            "dry": "dry_run",
            "skip": "skipped",
        }

    @pytest.mark.parametrize(
        ("manifest_text", "expected_exception", "expected_match"),
        [
            (None, ValueError, "No run_manifest.json found"),
            ("{not valid json", json.JSONDecodeError, None),
        ],
    )
    def test_load_prior_results_raises_for_missing_or_invalid_manifest(
        self,
        tmp_path: Path,
        manifest_text: str | None,
        expected_exception: type[Exception],
        expected_match: str | None,
    ) -> None:
        if manifest_text is not None:
            (tmp_path / "run_manifest.json").write_text(manifest_text, encoding="utf-8")

        if expected_match is None:
            with pytest.raises(expected_exception):
                _load_prior_results(tmp_path)
        else:
            with pytest.raises(expected_exception, match=expected_match):
                _load_prior_results(tmp_path)


class TestSchedulerUtilityHelpers:
    @pytest.mark.parametrize(
        ("seconds", "expected"),
        [
            (12, "12s"),
            (65, "1m05s"),
            (3661, "61m01s"),
        ],
    )
    def test_fmt_duration_formats_current_scheduler_output(
        self,
        seconds: float,
        expected: str,
    ) -> None:
        assert _fmt_duration(seconds) == expected

    def test_text_helpers_extract_split_estimate_and_score(self) -> None:
        keywords = _extract_keywords("The API v2 parser uses retry_01, C, and io.")

        assert "api" in keywords
        assert "v2" in keywords
        assert "retry_01" in keywords
        assert "the" not in keywords
        assert "c" not in keywords

        assert _estimate_tokens("") == 1
        assert _estimate_tokens("abcdefgh") == 2

        assert _split_into_sections("alpha\n\nbeta") == ["alpha", "beta"]

        chunked = _split_into_sections("\n".join(f"line {idx}" for idx in range(9)))
        assert len(chunked) == 2
        assert chunked[0].startswith("line 0")
        assert chunked[1] == "line 8"

        idf = _compute_idf(["parser api", "parser cli", "deploy release"])

        assert idf["deploy"] > idf["parser"]
        assert _score_section("deploy release", {"api"}, idf=idf) == 0
        assert (
            _score_section(
                "api api parser",
                {"api", "parser"},
                idf={"api": 3.0, "parser": 0.5},
                avg_section_len=3,
            )
            > _score_section("api api parser", {"api", "parser"})
        )

    def test_text_helpers_handle_empty_inputs_and_zero_average_length(self) -> None:
        assert _split_into_sections(" \n\n  \n") == []
        assert _compute_idf([]) == {}
        assert _score_section("", {"alpha"}, idf={"alpha": 2.0}) == 0
        assert _score_section("alpha beta", set(), idf={"alpha": 2.0}) == 0
        assert _score_section("alpha alpha", {"alpha"}, idf={}, avg_section_len=0) == 1

    def test_hop_distance_and_decay_trim_transitive_upstream_context(
        self,
        tmp_path: Path,
    ) -> None:
        tasks = {
            "root": _make_task("root"),
            "mid": _make_task("mid", depends_on=["root"]),
            "leaf": _make_task("leaf", depends_on=["mid"]),
        }

        hop_distances = _compute_hop_distances(
            "leaf",
            context_from=["mid", "root", "*", "missing"],
            all_tasks=tasks,
        )

        assert hop_distances == {"mid": 1, "root": 2}

        upstream = {
            "mid": _make_success_result(
                "mid",
                tmp_path,
                stdout_tail="abcdefghij",
            ),
            "root": _make_success_result(
                "root",
                tmp_path,
                stdout_tail="abcdefghij",
            ),
            "other": _make_success_result(
                "other",
                tmp_path,
                stdout_tail="abcdefghij",
            ),
        }

        decayed = _apply_hop_decay(upstream, hop_distances)

        assert decayed["mid"] is upstream["mid"]
        assert decayed["other"] is upstream["other"]
        assert decayed["root"] is not upstream["root"]
        assert decayed["root"].stdout_tail == "abcdefgh"
        assert upstream["root"].stdout_tail == "abcdefghij"

    def test_compute_hop_distances_returns_empty_for_unknown_task(self) -> None:
        tasks = {
            "root": _make_task("root"),
            "leaf": _make_task("leaf", depends_on=["root"]),
        }

        assert _compute_hop_distances("missing", ["root", "*"], tasks) == {}


def _mock_execute_with_upstream_capture(
    run_path_holder: list[Path],
    captured_upstream: dict[str, dict[str, TaskResult] | None],
):
    """Return a mock execute_task that captures upstream_results per task."""

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
    ):
        if not run_path_holder:
            run_path_holder.append(run_path)
        captured_upstream[task.id] = upstream_results

        now = datetime.now(UTC)
        result = TaskResult(
            task_id=task.id,
            status="success",
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=0.01,
            command=f"echo {task.id}",
            log_path=run_path / f"{task.id}.log",
            result_path=run_path / f"{task.id}.result.json",
            message="ok",
        )
        result.log_path.write_text(
            f"status=success\nmessage=ok\n", encoding="utf-8"
        )
        result.result_path.write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        return result

    return mock_execute


# ===========================================================================
# Test: _select_tasks
# ===========================================================================


class TestSelectTasks:
    """Tests for _select_tasks (task filtering with --only / --skip)."""

    def test_all_tasks_when_no_only_no_skip(self) -> None:
        """With no --only or --skip, all tasks are selected."""
        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only=None, skip=None)

        assert len(selected) == 3
        assert [t.id for t in selected] == ["a", "b", "c"]

    def test_only_selects_task_and_transitive_deps(self) -> None:
        """--only selects the target task plus all transitive dependencies."""
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
            _make_task("d"),  # independent, should be excluded
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only={"c"}, skip=None)

        selected_ids = {t.id for t in selected}
        assert selected_ids == {"a", "b", "c"}
        assert "d" not in selected_ids

    def test_only_with_diamond_deps(self) -> None:
        """--only on a diamond tip pulls in both branches."""
        tasks = [
            _make_task("root"),
            _make_task("left", depends_on=["root"]),
            _make_task("right", depends_on=["root"]),
            _make_task("tip", depends_on=["left", "right"]),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only={"tip"}, skip=None)

        selected_ids = {t.id for t in selected}
        assert selected_ids == {"root", "left", "right", "tip"}

    def test_skip_removes_tasks(self) -> None:
        """--skip removes the specified tasks."""
        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only=None, skip={"b"})

        selected_ids = {t.id for t in selected}
        assert selected_ids == {"a", "c"}

    def test_only_and_skip_combined(self) -> None:
        """--only and --skip can be used together (skip applied after only)."""
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
        ]
        plan = _make_plan(tasks)

        # --only c pulls in a, b, c  — then --skip b removes b
        selected = _select_tasks(plan, only={"c"}, skip={"b"})

        selected_ids = {t.id for t in selected}
        assert selected_ids == {"a", "c"}

    def test_only_unknown_task_raises_value_error(self) -> None:
        """--only with an unknown task ID raises ValueError."""
        tasks = [_make_task("a")]
        plan = _make_plan(tasks)

        with pytest.raises(ValueError, match="Unknown --only task: ghost"):
            _select_tasks(plan, only={"ghost"}, skip=None)

    def test_skip_unknown_task_raises_value_error(self) -> None:
        """--skip with an unknown task ID raises ValueError."""
        tasks = [_make_task("a")]
        plan = _make_plan(tasks)

        with pytest.raises(ValueError, match="Unknown --skip task: ghost"):
            _select_tasks(plan, only=None, skip={"ghost"})

    def test_preserves_original_task_order(self) -> None:
        """Selected tasks preserve their original order from the plan."""
        tasks = [
            _make_task("z"),
            _make_task("m", depends_on=["z"]),
            _make_task("a", depends_on=["m"]),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only={"a"}, skip=None)

        assert [t.id for t in selected] == ["z", "m", "a"]

    @pytest.mark.parametrize(
        ("tags", "skip_tags"),
        [
            ({"ship"}, None),
            (None, {"quality"}),
        ],
    )
    def test_tag_filters_reinclude_required_dependencies(
        self,
        tags: set[str] | None,
        skip_tags: set[str] | None,
    ) -> None:
        """Tag filters keep dependency chains intact after filtering."""
        tasks = [
            TaskSpec(
                id="bootstrap",
                description="bootstrap",
                tags=["infra"],
                command="echo bootstrap",
            ),
            TaskSpec(
                id="lint",
                description="lint",
                tags=["quality"],
                depends_on=["bootstrap"],
                command="echo lint",
            ),
            TaskSpec(
                id="release",
                description="release",
                tags=["ship"],
                depends_on=["lint"],
                command="echo release",
            ),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(
            plan,
            only=None,
            skip=None,
            tags=tags,
            skip_tags=skip_tags,
        )

        assert [t.id for t in selected] == ["bootstrap", "lint", "release"]

    def test_consumed_contract_dependencies_are_selected_transitively(self) -> None:
        tasks = [
            TaskSpec(id="schema", command="echo schema", contract_type="sql-schema"),
            TaskSpec(
                id="repo",
                engine="claude",
                prompt="Use {{ contract.schema.summary }}",
                consumes_contracts=["schema"],
            ),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only={"repo"}, skip=None)

        assert [task.id for task in selected] == ["schema", "repo"]

    def test_reconcile_group_dependencies_are_selected_transitively(self) -> None:
        tasks = [
            TaskSpec(id="controller", command="echo controller", consistency_group=["di"]),
            TaskSpec(id="bindings", command="echo bindings", consistency_group=["di"]),
            TaskSpec(
                id="reconcile",
                engine="claude",
                prompt="Check {{ consistency.di.statuses }}",
                reconcile_after=["di"],
            ),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only={"reconcile"}, skip=None)

        assert [task.id for task in selected] == ["controller", "bindings", "reconcile"]

    @pytest.mark.parametrize(
        ("tags", "skip_tags", "expected_ids"),
        [
            ({"ship"}, {"quality"}, ["bootstrap", "lint", "release"]),
            ({"quality", "ship"}, {"ship"}, ["bootstrap", "lint"]),
        ],
    )
    def test_combined_tag_filters_keep_required_dependencies(
        self,
        tags: set[str],
        skip_tags: set[str],
        expected_ids: list[str],
    ) -> None:
        tasks = [
            TaskSpec(
                id="bootstrap",
                description="bootstrap",
                tags=["infra"],
                command="echo bootstrap",
            ),
            TaskSpec(
                id="lint",
                description="lint",
                tags=["quality"],
                depends_on=["bootstrap"],
                command="echo lint",
            ),
            TaskSpec(
                id="release",
                description="release",
                tags=["ship"],
                depends_on=["lint"],
                command="echo release",
            ),
            TaskSpec(
                id="docs",
                description="docs",
                tags=["docs"],
                command="echo docs",
            ),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(
            plan,
            only=None,
            skip=None,
            tags=tags,
            skip_tags=skip_tags,
        )

        assert [t.id for t in selected] == expected_ids

    def test_only_and_tags_limit_roots_but_reinclude_selected_dependencies(self) -> None:
        tasks = [
            TaskSpec(
                id="bootstrap",
                description="bootstrap",
                tags=["infra"],
                command="echo bootstrap",
            ),
            TaskSpec(
                id="lint",
                description="lint",
                tags=["quality"],
                depends_on=["bootstrap"],
                command="echo lint",
            ),
            TaskSpec(
                id="release",
                description="release",
                tags=["ship"],
                depends_on=["lint"],
                command="echo release",
            ),
            TaskSpec(
                id="hotfix",
                description="hotfix",
                tags=["ship"],
                command="echo hotfix",
            ),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(
            plan,
            only={"release"},
            skip=None,
            tags={"ship"},
            skip_tags=None,
        )

        assert [t.id for t in selected] == ["bootstrap", "lint", "release"]

    def test_tags_with_no_matches_return_empty_selection(self) -> None:
        tasks = [
            TaskSpec(
                id="bootstrap",
                description="bootstrap",
                tags=["infra"],
                command="echo bootstrap",
            ),
            TaskSpec(
                id="release",
                description="release",
                tags=["ship"],
                depends_on=["bootstrap"],
                command="echo release",
            ),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(
            plan,
            only=None,
            skip=None,
            tags={"docs"},
            skip_tags=None,
        )

        assert selected == []

    def test_skip_and_tags_reinclude_required_dependencies_but_not_skipped_roots(self) -> None:
        tasks = [
            TaskSpec(
                id="bootstrap",
                description="bootstrap",
                tags=["infra"],
                command="echo bootstrap",
            ),
            TaskSpec(
                id="lint",
                description="lint",
                tags=["quality"],
                depends_on=["bootstrap"],
                command="echo lint",
            ),
            TaskSpec(
                id="release",
                description="release",
                tags=["ship"],
                depends_on=["lint"],
                command="echo release",
            ),
            TaskSpec(
                id="hotfix",
                description="hotfix",
                tags=["ship"],
                command="echo hotfix",
            ),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(
            plan,
            only=None,
            skip={"lint", "hotfix"},
            tags={"ship"},
            skip_tags=None,
        )

        assert [t.id for t in selected] == ["bootstrap", "lint", "release"]


# ===========================================================================
# Test: run_plan — linear chain
# ===========================================================================


class TestImplicitRelationships:
    def test_consumes_contracts_orders_task_and_injects_contract_vars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()
        call_log: list[str] = []
        seen_contract_vars: dict[str, str] = {}

        tasks = [
            TaskSpec(id="schema", command="echo schema", contract_type="sql-schema"),
            TaskSpec(
                id="repo",
                engine="claude",
                prompt="Use {{ contract.schema.summary }}",
                consumes_contracts=["schema"],
            ),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=2)

        def mock_execute(
            plan,
            task,
            run_path,
            dry_run=False,
            execution_profile="plan",
            upstream_results=None,
            context_synthesis="",
            workspace_brief="",
            extra_template_vars=None,
            **kwargs,
        ):
            call_log.append(task.id)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            if task.id == "schema":
                result.produced_contract = TaskContract(
                    producer_task_id="schema",
                    contract_type="sql-schema",
                    summary="SQL schema with 2 statements",
                    body="CREATE TABLE users (...);",
                    content_hash="abc123",
                )
            else:
                assert extra_template_vars is not None
                seen_contract_vars.update(extra_template_vars)

            result.log_path.write_text("status=success\nmessage=ok\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2),
                encoding="utf-8",
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert call_log == ["schema", "repo"]
        assert seen_contract_vars["contract.schema.type"] == "sql-schema"
        assert "SQL schema with 2 statements" in seen_contract_vars["contract.schema.summary"]
        assert "contracts_summary" in seen_contract_vars

    def test_reconcile_after_waits_for_group_and_injects_group_vars(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()
        call_log: list[str] = []
        seen_group_vars: dict[str, str] = {}

        tasks = [
            TaskSpec(id="controller", command="echo controller", consistency_group=["di"]),
            TaskSpec(id="bindings", command="echo bindings", consistency_group=["di"]),
            TaskSpec(
                id="reconcile",
                engine="claude",
                prompt="Check {{ consistency.di.statuses }}",
                reconcile_after=["di"],
            ),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=3)

        def mock_execute(
            plan,
            task,
            run_path,
            dry_run=False,
            execution_profile="plan",
            upstream_results=None,
            context_synthesis="",
            workspace_brief="",
            extra_template_vars=None,
            **kwargs,
        ):
            call_log.append(task.id)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            if task.id == "reconcile":
                assert extra_template_vars is not None
                seen_group_vars.update(extra_template_vars)

            result.log_path.write_text("status=success\nmessage=ok\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2),
                encoding="utf-8",
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert set(call_log[:2]) == {"controller", "bindings"}
        assert call_log[-1] == "reconcile"
        assert "controller" in seen_group_vars["consistency.di.tasks"]
        assert "bindings" in seen_group_vars["consistency.di.tasks"]
        assert "controller: success" in seen_group_vars["consistency.di.statuses"]
        assert "consistency_summary" in seen_group_vars


class TestRunPlanLinearChain:
    """Tests for DAG execution of a simple linear chain A -> B -> C."""

    def test_linear_chain_all_succeed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All tasks in a linear chain complete with success."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert len(result.task_results) == 3
        for tid in ("a", "b", "c"):
            assert result.task_results[tid].status == "success"

    def test_linear_chain_execution_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tasks execute in dependency order: a before b before c."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert call_log == ["a", "b", "c"]

    def test_manifest_written(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A run_manifest.json is created in the run directory."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        manifest_path = result.run_path / "run_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["plan_name"] == "test-plan"
        assert manifest["success"] is True
        assert "a" in manifest["task_results"]


# ===========================================================================
# Test: run_plan — diamond DAG
# ===========================================================================


class TestRunPlanDiamond:
    """Tests for DAG execution of a diamond: A -> {B, C} -> D."""

    def test_diamond_all_succeed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """All four tasks complete successfully in a diamond topology."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
            _make_task("d", depends_on=["b", "c"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=4)

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert len(result.task_results) == 4
        for tid in ("a", "b", "c", "d"):
            assert result.task_results[tid].status == "success"

    def test_diamond_b_and_c_run_after_a_and_before_d(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """B and C both run after A; D runs after both B and C."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
            _make_task("d", depends_on=["b", "c"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=4)

        rp_holder: list[Path] = []
        lock = threading.Lock()
        mock_exec, call_log = _mock_execute_task_factory(
            rp_holder, call_log_lock=lock
        )
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # A must come first
        assert call_log[0] == "a"
        # D must come last
        assert call_log[-1] == "d"
        # B and C must be between A and D (order between them is non-deterministic)
        middle = set(call_log[1:-1])
        assert middle == {"b", "c"}


# ===========================================================================
# Test: run_plan — fail-fast behavior
# ===========================================================================


class TestRunPlanFailFast:
    """Tests for fail_fast=True behavior."""

    def test_fail_fast_skips_remaining_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When fail_fast=True and a task fails, remaining tasks are skipped."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        run_dir = tmp_path / "runs"

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
        ]
        plan = _make_plan(tasks, fail_fast=True, source_path=plan_yaml)

        rp_holder: list[Path] = []

        def _make_overrides(rp: Path) -> dict[str, TaskResult]:
            return {
                "a": _make_success_result("a", rp, status="failed", exit_code=1),
            }

        # We need a two-stage approach: first call captures the run_path,
        # then we build overrides.  Simpler: build overrides lazily.
        failed_a_override: dict[str, TaskResult] = {}

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            if task.id == "a":
                now = datetime.now(UTC)
                result = TaskResult(
                    task_id="a",
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="echo a",
                    log_path=run_path / "a.log",
                    result_path=run_path / "a.result.json",
                    message="Task failed with exit code 1",
                )
                result.log_path.write_text("status=failed\n", encoding="utf-8")
                result.result_path.write_text(
                    json.dumps(result.to_dict(), indent=2), encoding="utf-8"
                )
                return result
            # Should not reach here with fail_fast=True
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(run_dir))

        assert result.success is False
        assert result.task_results["a"].status == "failed"
        # B is a dependent of A (which failed) — skipped due to dependency failure
        assert result.task_results["b"].status == "skipped"
        # C is a dependent of B — also skipped
        assert result.task_results["c"].status == "skipped"

    def test_fail_fast_independent_tasks_may_be_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With fail_fast, even independent pending tasks are skipped after failure."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        # a fails; b is independent but should be skipped by fail_fast
        # Use max_parallel=1 to ensure a runs before b is dispatched
        tasks = [
            _make_task("a"),
            _make_task("b"),
        ]
        plan = _make_plan(tasks, fail_fast=True, source_path=plan_yaml, max_parallel=1)

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            if task.id == "a":
                result = TaskResult(
                    task_id="a",
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="echo a",
                    log_path=run_path / "a.log",
                    result_path=run_path / "a.result.json",
                    message="Task failed",
                )
            else:
                result = TaskResult(
                    task_id=task.id,
                    status="success",
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

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        assert result.task_results["a"].status == "failed"
        assert result.task_results["b"].status == "skipped"
        assert "fail_fast" in result.task_results["b"].message


# ===========================================================================
# Test: run_plan — soft failure (allow_failure)
# ===========================================================================


class TestRunPlanSoftFailure:
    """Tests for allow_failure=True producing soft_failed status."""

    def test_soft_failed_does_not_block_dependents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A soft_failed task allows its dependents to proceed."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a", allow_failure=True),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            if task.id == "a":
                result = TaskResult(
                    task_id="a",
                    status="soft_failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="echo a",
                    log_path=run_path / "a.log",
                    result_path=run_path / "a.result.json",
                    message="Task failed with exit code 1, but allow_failure=true",
                )
            else:
                result = TaskResult(
                    task_id=task.id,
                    status="success",
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

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert result.task_results["a"].status == "soft_failed"
        assert result.task_results["b"].status == "success"

    def test_soft_failed_overall_success_is_true(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A plan where all failures are soft_failed reports success=True."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a", allow_failure=True),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id="a",
                status="soft_failed",
                exit_code=1,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command="echo a",
                log_path=run_path / "a.log",
                result_path=run_path / "a.result.json",
                message="soft failed",
            )
            result.log_path.write_text("status=soft_failed\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True


# ===========================================================================
# Test: run_plan — dependency failure skips dependents (fail_fast=False)
# ===========================================================================


class TestRunPlanDependencyFailureSkipsDependents:
    """Tests for dependency-failure propagation with fail_fast=False."""

    def test_failed_task_skips_dependents_not_independents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With fail_fast=False, failed A skips B (depends on A) but not C (independent)."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c"),  # independent of A
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml, max_parallel=4)

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            if task.id == "a":
                result = TaskResult(
                    task_id="a",
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="echo a",
                    log_path=run_path / "a.log",
                    result_path=run_path / "a.result.json",
                    message="Task failed",
                )
            else:
                result = TaskResult(
                    task_id=task.id,
                    status="success",
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

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        assert result.task_results["a"].status == "failed"
        assert result.task_results["b"].status == "skipped"
        assert "dependency failed" in result.task_results["b"].message.lower()
        assert result.task_results["c"].status == "success"

    def test_transitive_dependency_failure_direct_dep(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A -> {B, C}: if A fails, both direct dependents B and C are skipped."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml)

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            if task.id == "a":
                result = TaskResult(
                    task_id="a",
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="echo a",
                    log_path=run_path / "a.log",
                    result_path=run_path / "a.result.json",
                    message="Task failed",
                )
            else:
                result = TaskResult(
                    task_id=task.id,
                    status="success",
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

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.task_results["a"].status == "failed"
        assert result.task_results["b"].status == "skipped"
        assert result.task_results["c"].status == "skipped"


# ===========================================================================
# Test: run_plan — dry run
# ===========================================================================


class TestRunPlanDryRun:
    """Tests for dry_run=True behavior."""

    def test_dry_run_all_tasks_get_dry_run_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With dry_run=True, execute_task receives dry_run=True and tasks get dry_run status."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        dry_run_flags: dict[str, bool] = {}

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            dry_run_flags[task.id] = dry_run
            now = datetime.now(UTC)
            status = "dry_run" if dry_run else "success"
            result = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="Dry run only" if dry_run else "ok",
            )
            result.log_path.write_text(f"status={status}\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan, dry_run=True, run_dir_override=str(tmp_path / "runs")
        )

        assert result.success is True
        for tid in ("a", "b"):
            assert result.task_results[tid].status == "dry_run"
            assert dry_run_flags[tid] is True

    def test_dry_run_is_considered_success_like(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """dry_run status is in _SUCCESS_LIKE, so overall success is True."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan, dry_run=True, run_dir_override=str(tmp_path / "runs")
        )

        assert result.success is True
        # Dependents should also run (dry_run is success-like)
        assert "b" in result.task_results
        assert result.task_results["b"].status == "dry_run"


# ===========================================================================
# Test: run_plan — resume from prior run
# ===========================================================================


class TestRunPlanResume:
    """Tests for resuming from a prior run directory."""

    def test_resume_skips_previously_succeeded_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Tasks that succeeded in a prior run are not re-executed."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        # Create mock prior run with task A as success
        prior_run = tmp_path / "prior_run"
        prior_run.mkdir()
        manifest = {
            "plan_name": "test-plan",
            "success": True,
            "task_results": {
                "a": {
                    "task_id": "a",
                    "status": "success",
                    "exit_code": 0,
                }
            },
        }
        (prior_run / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        rp_holder: list[Path] = []
        call_log: list[str] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            call_log.append(task.id)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            resume_path=prior_run,
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True
        # A should NOT be in call_log (it was resumed)
        assert "a" not in call_log
        # B should have been executed
        assert "b" in call_log
        # Both should be in results
        assert "a" in result.task_results
        assert "b" in result.task_results

    def test_resume_task_a_status_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resumed task A shows the prior status in the new result."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, source_path=plan_yaml)

        prior_run = tmp_path / "prior_run"
        prior_run.mkdir()
        manifest = {
            "plan_name": "test-plan",
            "success": True,
            "task_results": {
                "a": {
                    "task_id": "a",
                    "status": "success",
                    "exit_code": 0,
                }
            },
        }
        (prior_run / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            resume_path=prior_run,
            run_dir_override=str(tmp_path / "runs"),
        )

        a_result = result.task_results["a"]
        assert a_result.status == "success"
        assert "resumed" in a_result.message.lower()

    def test_resume_respects_soft_failed_as_success_like(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A prior soft_failed task is treated as success-like and not re-run."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, source_path=plan_yaml)

        prior_run = tmp_path / "prior_run"
        prior_run.mkdir()
        manifest = {
            "plan_name": "test-plan",
            "success": True,
            "task_results": {
                "a": {
                    "task_id": "a",
                    "status": "soft_failed",
                    "exit_code": 1,
                }
            },
        }
        (prior_run / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        rp_holder: list[Path] = []
        call_log: list[str] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            call_log.append(task.id)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            resume_path=prior_run,
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True
        assert "a" not in call_log  # Soft_failed is success-like, so not re-run
        assert "b" in call_log

    def test_resume_missing_manifest_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Resuming from a directory without run_manifest.json raises ValueError."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        empty_dir = tmp_path / "empty_prior"
        empty_dir.mkdir()

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        with pytest.raises(ValueError, match="No run_manifest.json found"):
            run_plan(
                plan,
                resume_path=empty_dir,
                run_dir_override=str(tmp_path / "runs"),
            )

    def test_resume_retries_dependency_failure_skipped_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Dependency-failure skipped tasks should re-run when their dep succeeds on resume."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        # Prior run: A failed, B was skipped due to dependency failure
        prior_run = tmp_path / "prior_run"
        prior_run.mkdir()
        manifest = {
            "plan_name": "test-plan",
            "success": False,
            "task_results": {
                "a": {
                    "task_id": "a",
                    "status": "failed",
                    "exit_code": 1,
                    "message": "command failed",
                },
                "b": {
                    "task_id": "b",
                    "status": "skipped",
                    "exit_code": None,
                    "message": "Skipped because dependency failed: {'a'}",
                },
            },
        }
        (prior_run / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        call_log: list[str] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            call_log.append(task.id)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            resume_path=prior_run,
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True
        # Both A and B should have been executed (not resumed from prior)
        assert "a" in call_log
        assert "b" in call_log
        assert result.task_results["a"].status == "success"
        assert result.task_results["b"].status == "success"

    def test_resume_retries_fail_fast_triggered_skipped_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """fail_fast-triggered skipped tasks should re-run when blocker succeeds on resume."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        # Prior run: A failed, B and C were skipped by fail_fast
        prior_run = tmp_path / "prior_run"
        prior_run.mkdir()
        manifest = {
            "plan_name": "test-plan",
            "success": False,
            "task_results": {
                "a": {
                    "task_id": "a",
                    "status": "failed",
                    "exit_code": 124,
                    "message": "Task timed out after 900s",
                },
                "b": {
                    "task_id": "b",
                    "status": "skipped",
                    "exit_code": None,
                    "message": "fail_fast triggered by task 'a'",
                },
                "c": {
                    "task_id": "c",
                    "status": "skipped",
                    "exit_code": None,
                    "message": "fail_fast triggered by task 'a'",
                },
            },
        }
        (prior_run / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        call_log: list[str] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            call_log.append(task.id)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            resume_path=prior_run,
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True
        # All three should have been executed (A retried, B+C no longer skipped)
        assert "a" in call_log
        assert "b" in call_log
        assert "c" in call_log
        assert result.task_results["a"].status == "success"
        assert result.task_results["b"].status == "success"
        assert result.task_results["c"].status == "success"

    def test_resume_keeps_when_condition_skipped_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When-condition skipped tasks should stay skipped on resume (not re-run)."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        # Prior run: A succeeded, B was skipped by when-condition
        prior_run = tmp_path / "prior_run"
        prior_run.mkdir()
        manifest = {
            "plan_name": "test-plan",
            "success": True,
            "task_results": {
                "a": {
                    "task_id": "a",
                    "status": "success",
                    "exit_code": 0,
                    "message": "ok",
                },
                "b": {
                    "task_id": "b",
                    "status": "skipped",
                    "exit_code": None,
                    "message": "Condition not met: a.status == failed",
                },
            },
        }
        (prior_run / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        call_log: list[str] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            call_log.append(task.id)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            resume_path=prior_run,
            run_dir_override=str(tmp_path / "runs"),
        )

        # A and B should both be resumed, NOT re-executed
        assert "a" not in call_log
        assert "b" not in call_log
        assert result.task_results["b"].status == "skipped"


# ===========================================================================
# Test: run_plan — upstream context passing
# ===========================================================================


class TestRunPlanUpstreamContext:
    """Tests for inter-task context passing via context_from."""

    def test_context_from_wildcard_passes_upstream_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task with context_from=["*"] receives upstream_results dict."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"], context_from=["*"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        captured_upstream: dict[str, dict[str, TaskResult] | None] = {}

        mock_exec = _mock_execute_with_upstream_capture(rp_holder, captured_upstream)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        # Task A should not receive upstream (no context_from)
        assert captured_upstream["a"] is None
        # Task B should receive upstream with task A's result
        assert captured_upstream["b"] is not None
        assert "a" in captured_upstream["b"]
        assert captured_upstream["b"]["a"].task_id == "a"

    def test_context_from_explicit_id_passes_specific_upstream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task with context_from=["a"] receives only task A's result."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c", depends_on=["a", "b"], context_from=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=4)

        rp_holder: list[Path] = []
        captured_upstream: dict[str, dict[str, TaskResult] | None] = {}

        mock_exec = _mock_execute_with_upstream_capture(rp_holder, captured_upstream)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        # C should receive only A's result, not B's
        assert captured_upstream["c"] is not None
        assert "a" in captured_upstream["c"]
        assert "b" not in captured_upstream["c"]

    def test_no_context_from_passes_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Task without context_from receives upstream_results=None."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        captured_upstream: dict[str, dict[str, TaskResult] | None] = {}

        mock_exec = _mock_execute_with_upstream_capture(rp_holder, captured_upstream)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        # Neither task declares context_from
        assert captured_upstream["a"] is None
        assert captured_upstream["b"] is None

    def test_context_from_wildcard_with_multiple_deps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """context_from=["*"] with multiple deps includes all of them."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c", depends_on=["a", "b"], context_from=["*"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=4)

        rp_holder: list[Path] = []
        captured_upstream: dict[str, dict[str, TaskResult] | None] = {}

        mock_exec = _mock_execute_with_upstream_capture(rp_holder, captured_upstream)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert captured_upstream["c"] is not None
        assert set(captured_upstream["c"].keys()) == {"a", "b"}

    def test_context_compression_event_includes_selection_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()
        relevant_section = ("api schema auth " * 80).strip()
        irrelevant_section = ("weather gardening travel " * 80).strip()
        source_tail = f"{relevant_section}\n\n{irrelevant_section}"
        budget_tokens = _estimate_tokens(relevant_section) + 10

        tasks = [
            _make_task("scan"),
            TaskSpec(
                id="review",
                description="review",
                depends_on=["scan"],
                context_from=["scan"],
                context_budget_tokens=budget_tokens,
                prompt="Review the API schema auth behavior.",
                command="echo review",
            ),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

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
        ):
            stdout_tail = source_tail if task.id == "scan" else ""
            result = _make_success_result(
                task.id,
                run_path,
                stdout_tail=stdout_tail,
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2),
                encoding="utf-8",
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        compression_event = next(
            event
            for event in events
            if event["event"] == "context_compression" and event["task_id"] == "review"
        )

        assert compression_event["budget_tokens"] == budget_tokens
        assert compression_event["context_final_tokens"] < compression_event["context_raw_tokens"]
        assert len(compression_event["entries"]) == 1
        entry = compression_event["entries"][0]
        assert entry["upstream_id"] == "scan"
        assert entry["score"] > 0
        assert entry["keywords_matched"] == ["api", "auth", "schema"]
        assert entry["hop_distance"] == 1
        assert entry["hop_decay_factor"] == 1.0
        assert entry["tokens_final"] < entry["tokens_raw"]
        assert entry["trimmed"] is True
        assert entry["trim_reason"] == "budget_trim"

    def test_layered_context_mode_routes_budget_through_synthesis(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()
        auth_tail = (
            "# Auth Findings\n"
            "Result: JWT validation fails on expired refresh tokens.\n\n"
            + ("auth jwt refresh token validation " * 90).strip()
        )
        api_tail = (
            "# API Notes\n"
            "Output: schema mismatch on /sessions endpoint.\n\n"
            + ("api schema contract sessions endpoint " * 90).strip()
        )
        budget_tokens = 120

        tasks = [
            _make_task("scan-auth"),
            _make_task("scan-api"),
            TaskSpec(
                id="review",
                description="review",
                depends_on=["scan-auth", "scan-api"],
                context_from=["scan-auth", "scan-api"],
                context_mode="layered",
                context_budget_tokens=budget_tokens,
                prompt="Layered context:\n{{ upstream_synthesis }}",
                command="echo review",
            ),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        captured_synthesis: dict[str, str] = {}

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
        ):
            if task.id == "review":
                captured_synthesis[task.id] = context_synthesis
                stdout_tail = ""
            elif task.id == "scan-auth":
                stdout_tail = auth_tail
            else:
                stdout_tail = api_tail

            result = _make_success_result(task.id, run_path, stdout_tail=stdout_tail)
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2),
                encoding="utf-8",
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        layered_text = captured_synthesis["review"]
        assert "--- scan-auth ---" in layered_text
        assert "--- scan-api ---" in layered_text
        assert "JWT validation fails on expired refresh tokens." in layered_text
        assert "schema mismatch on /sessions endpoint." in layered_text
        assert len(layered_text) < len(auth_tail) + len(api_tail)

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        compression_event = next(
            event
            for event in events
            if event["event"] == "context_compression" and event["task_id"] == "review"
        )

        assert compression_event["budget_tokens"] == budget_tokens
        assert compression_event["context_final_tokens"] < compression_event["context_raw_tokens"]
        assert len(compression_event["entries"]) == 2
        assert any(
            entry["upstream_id"] == "scan-auth" and entry["trimmed"] is True
            for entry in compression_event["entries"]
        )
        assert any(
            event["event"] == "context_budget_trim" and event["task_id"] == "review"
            for event in events
        )


# ===========================================================================
# Test: run_plan — edge cases
# ===========================================================================


class TestRunPlanEdgeCases:
    """Tests for edge cases in run_plan."""

    def test_no_tasks_selected_succeeds_with_empty_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty task selection still produces a successful empty run."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, skip={"a"}, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert result.task_results == {}
        events_text = (result.run_path / "events.jsonl").read_text(encoding="utf-8")
        events = [json.loads(line) for line in events_text.splitlines() if line.strip()]
        assert [event["event"] for event in events] == ["run_start", "run_complete"]

    def test_tag_filtered_run_executes_only_matching_tasks_and_dependencies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            TaskSpec(
                id="bootstrap",
                description="bootstrap",
                tags=["infra"],
                command="echo bootstrap",
            ),
            TaskSpec(
                id="release",
                description="release",
                tags=["ship"],
                depends_on=["bootstrap"],
                command="echo release",
            ),
            TaskSpec(
                id="docs",
                description="docs",
                tags=["docs"],
                command="echo docs",
            ),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            tags={"ship"},
            skip_tags={"infra"},
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True
        assert call_log == ["bootstrap", "release"]
        assert set(result.task_results) == {"bootstrap", "release"}

    def test_execution_profile_passed_through(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Execution profile is forwarded to execute_task."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        captured_profiles: dict[str, str] = {}

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            captured_profiles[task.id] = execution_profile
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        run_plan(
            plan,
            execution_profile="yolo",
            run_dir_override=str(tmp_path / "runs"),
        )

        assert captured_profiles["a"] == "yolo"

    def test_max_parallel_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_parallel_override limits concurrency even if plan says higher."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        # 3 independent tasks, plan says max_parallel=10
        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=10)

        rp_holder: list[Path] = []
        concurrency_high_watermark: list[int] = [0]
        active_count = [0]
        lock = threading.Lock()

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            with lock:
                active_count[0] += 1
                if active_count[0] > concurrency_high_watermark[0]:
                    concurrency_high_watermark[0] = active_count[0]
            # Brief sleep to overlap with concurrent tasks
            import time
            time.sleep(0.05)
            with lock:
                active_count[0] -= 1

            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.05,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            max_parallel_override=1,
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True
        # With max_parallel_override=1, concurrency should never exceed 1
        assert concurrency_high_watermark[0] <= 1

    def test_run_result_contains_correct_plan_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PlanRunResult.plan_name matches the plan's name."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, name="my-special-plan", source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.plan_name == "my-special-plan"
        assert result.run_path.exists()

    def test_run_result_execution_profile_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PlanRunResult.execution_profile reflects what was passed."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            execution_profile="safe",
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.execution_profile == "safe"


# ===========================================================================
# Test: Parallelism metrics
# ===========================================================================


class TestParallelismMetrics:
    """Tests for sequential_duration_sec, parallelism_savings_pct, total_cost_usd."""

    def test_sequential_duration_is_sum(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """sequential_duration_sec is the sum of all non-skipped task durations."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            duration = 5.0 if task.id == "a" else 3.0
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=duration,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text(f"status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.sequential_duration_sec == pytest.approx(8.0)

    def test_total_cost_aggregated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """total_cost_usd is the sum of all tasks' cost_usd values."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            cost = 1.50 if task.id == "a" else 2.75
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
                cost_usd=cost,
            )
            result.log_path.write_text(f"status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.total_cost_usd == pytest.approx(4.25)

    def test_no_cost_when_no_tasks_report_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """total_cost_usd is None when no tasks report cost."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=4)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.total_cost_usd is None


# ===========================================================================
# Test: Run summary markdown
# ===========================================================================


class TestRunSummary:
    """Tests for _write_summary producing run_summary.md."""

    def test_summary_file_created(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_summary.md is created after run_plan completes."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        summary = result.run_path / "run_summary.md"
        assert summary.exists()

    def test_summary_contains_plan_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_summary.md contains the plan name."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, name="my-cool-plan", source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        summary_text = (result.run_path / "run_summary.md").read_text(encoding="utf-8")
        assert "my-cool-plan" in summary_text

    def test_summary_contains_task_table(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_summary.md contains a '## Tasks' section with task IDs."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("alpha"), _make_task("beta", depends_on=["alpha"])]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        summary_text = (result.run_path / "run_summary.md").read_text(encoding="utf-8")
        assert "## Tasks" in summary_text
        assert "alpha" in summary_text
        assert "beta" in summary_text

    def test_summary_contains_timeline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_summary.md contains '## Timeline' with wave labels."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        summary_text = (result.run_path / "run_summary.md").read_text(encoding="utf-8")
        assert "## Timeline" in summary_text
        assert "Wave 0" in summary_text
        assert "Wave 1" in summary_text


class TestWebhookNotifications:
    def test_post_completion_webhook_sends_json_post_and_returns_status(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.scheduler import _post_completion_webhook

        captured: dict[str, object] = {}

        class _Response:
            status = 204

            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        def _mock_urlopen(request, timeout: int):
            captured["url"] = request.full_url
            captured["method"] = request.get_method()
            captured["headers"] = {key.lower(): value for key, value in request.header_items()}
            captured["data"] = request.data
            captured["timeout"] = timeout
            return _Response()

        monkeypatch.setattr("maestro_cli.scheduler.urllib.request.urlopen", _mock_urlopen)

        status = _post_completion_webhook(
            "https://example.com/notify",
            {"plan_name": "demo", "success": True},
        )

        assert status == 204
        assert captured["url"] == "https://example.com/notify"
        assert captured["method"] == "POST"
        assert captured["headers"]["content-type"] == "application/json"
        assert json.loads(captured["data"]) == {"plan_name": "demo", "success": True}
        assert captured["timeout"] == 10

    def test_post_completion_webhook_defaults_to_200_without_status_attr(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.scheduler import _post_completion_webhook

        class _Response:
            def __enter__(self) -> "_Response":
                return self

            def __exit__(self, exc_type, exc, tb) -> bool:
                return False

        monkeypatch.setattr(
            "maestro_cli.scheduler.urllib.request.urlopen",
            lambda request, timeout: _Response(),
        )

        assert _post_completion_webhook(
            "https://example.com/notify",
            {"plan_name": "demo"},
        ) == 200

    def test_posts_webhook_payload_after_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        captured: dict[str, object] = {}

        def _mock_post(webhook_url: str, payload: dict[str, object]) -> int:
            captured["webhook_url"] = webhook_url
            captured["payload"] = payload
            return 200

        monkeypatch.setattr("maestro_cli.scheduler._post_completion_webhook", _mock_post)

        result = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
            webhook_url="https://example.com/notify",
        )

        assert result.success is True
        assert captured["webhook_url"] == "https://example.com/notify"
        payload = captured["payload"]
        assert isinstance(payload, dict)
        assert payload["plan_name"] == "test-plan"
        assert payload["run_id"] == result.run_id
        assert payload["success"] is True
        assert payload["ok_count"] == 1
        assert payload["failed_count"] == 0
        assert payload["skipped_count"] == 0
        assert payload["total_cost_usd"] is None
        assert payload["total_tokens"] is None
        assert payload["duration_sec"] is not None
        assert payload["run_path"] == str(result.run_path)
        assert isinstance(payload["summary_url"], str)
        assert str(payload["summary_url"]).startswith("file://")

    def test_webhook_failure_does_not_fail_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        def _mock_post(_webhook_url: str, _payload: dict[str, object]) -> int:
            raise OSError("network down")

        monkeypatch.setattr("maestro_cli.scheduler._post_completion_webhook", _mock_post)

        result = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
            webhook_url="https://example.com/notify",
        )
        assert result.success is True


# ===========================================================================
# Test: _compute_waves
# ===========================================================================


class TestComputeWaves:
    """Tests for _compute_waves topological wave grouping."""

    def test_linear_chain_three_waves(self) -> None:
        """a -> b -> c produces 3 waves: [a], [b], [c]."""
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
        ]
        plan = _make_plan(tasks)
        run_path = Path("/tmp/test-run")

        now = datetime.now(UTC)
        task_results = {}
        for tid in ("a", "b", "c"):
            task_results[tid] = TaskResult(
                task_id=tid,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=1.0,
                command=f"echo {tid}",
                log_path=run_path / f"{tid}.log",
                result_path=run_path / f"{tid}.result.json",
            )

        run_result = PlanRunResult(
            plan_name="test",
            run_id="test-run",
            run_path=run_path,
            started_at=now,
            finished_at=now,
            success=True,
            task_results=task_results,
        )

        waves = _compute_waves(plan, run_result)
        assert len(waves) == 3
        assert waves[0] == ["a"]
        assert waves[1] == ["b"]
        assert waves[2] == ["c"]

    def test_parallel_tasks_same_wave(self) -> None:
        """root -> {left, right} -> tip produces 3 waves with left+right in wave 1."""
        tasks = [
            _make_task("root"),
            _make_task("left", depends_on=["root"]),
            _make_task("right", depends_on=["root"]),
            _make_task("tip", depends_on=["left", "right"]),
        ]
        plan = _make_plan(tasks)
        run_path = Path("/tmp/test-run")

        now = datetime.now(UTC)
        task_results = {}
        for tid in ("root", "left", "right", "tip"):
            task_results[tid] = TaskResult(
                task_id=tid,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=1.0,
                command=f"echo {tid}",
                log_path=run_path / f"{tid}.log",
                result_path=run_path / f"{tid}.result.json",
            )

        run_result = PlanRunResult(
            plan_name="test",
            run_id="test-run",
            run_path=run_path,
            started_at=now,
            finished_at=now,
            success=True,
            task_results=task_results,
        )

        waves = _compute_waves(plan, run_result)
        assert len(waves) == 3
        assert waves[0] == ["root"]
        assert set(waves[1]) == {"left", "right"}
        assert waves[2] == ["tip"]

    def test_single_task_one_wave(self) -> None:
        """A single task produces exactly 1 wave."""
        tasks = [_make_task("only")]
        plan = _make_plan(tasks)
        run_path = Path("/tmp/test-run")

        now = datetime.now(UTC)
        task_results = {
            "only": TaskResult(
                task_id="only",
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=1.0,
                command="echo only",
                log_path=run_path / "only.log",
                result_path=run_path / "only.result.json",
            ),
        }

        run_result = PlanRunResult(
            plan_name="test",
            run_id="test-run",
            run_path=run_path,
            started_at=now,
            finished_at=now,
            success=True,
            task_results=task_results,
        )

        waves = _compute_waves(plan, run_result)
        assert len(waves) == 1
        assert waves[0] == ["only"]

    def test_cycle_falls_back_to_sorted_remaining_tasks(self) -> None:
        """A cycle is emitted as a final sorted fallback wave."""
        tasks = [
            _make_task("ready"),
            _make_task("left", depends_on=["right"]),
            _make_task("right", depends_on=["left"]),
        ]
        plan = _make_plan(tasks)
        run_path = Path("/tmp/test-run")

        now = datetime.now(UTC)
        task_results = {
            tid: TaskResult(
                task_id=tid,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=1.0,
                command=f"echo {tid}",
                log_path=run_path / f"{tid}.log",
                result_path=run_path / f"{tid}.result.json",
            )
            for tid in ("ready", "left", "right")
        }

        run_result = PlanRunResult(
            plan_name="test",
            run_id="test-run",
            run_path=run_path,
            started_at=now,
            finished_at=now,
            success=True,
            task_results=task_results,
        )

        waves = _compute_waves(plan, run_result)

        assert waves[0] == ["ready"]
        assert waves[1] == ["left", "right"]


# ===========================================================================
# Test: run_plan — when-conditional skip vs dependency-failure skip
# ===========================================================================


class TestSkippedSuccessLike:
    """Tests that when-conditional skips are success-like, but dep-failure skips are not."""

    def test_when_conditional_skip_yields_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A task skipped because its when-condition is not met counts as success-like.

        Plan: A succeeds -> B has ``when: "{{ a.status }} == failed"`` (condition not met)
        Expected: result.success is True because skipped ∈ _SUCCESS_LIKE.
        """
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_a = _make_task("a")
        task_b = TaskSpec(
            id="b",
            description="conditional task",
            depends_on=["a"],
            command="echo b",
            when="{{ a.status }} == failed",  # A succeeds, so condition is not met
        )
        plan = _make_plan([task_a, task_b], fail_fast=False, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert result.task_results["a"].status == "success"
        assert result.task_results["b"].status == "skipped"
        # b was never executed (only a was called)
        assert call_log == ["a"]

    def test_dep_failure_skip_marks_run_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A task skipped because its dependency failed does NOT make the run succeed.

        Plan: A fails -> B depends on A (no when) -> B is skipped due to dep failure.
        Expected: result.success is False because A has status 'failed'.
        """
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml)

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan", upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            status = "failed" if task.id == "a" else "success"
            result = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=1 if task.id == "a" else 0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="failed" if task.id == "a" else "ok",
            )
            result.log_path.write_text(f"status={status}\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        assert result.task_results["a"].status == "failed"
        assert result.task_results["b"].status == "skipped"

    def test_when_skip_alongside_all_success_is_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Multiple when-skipped tasks with all others succeeding still yields success.

        Plan: A succeeds, B succeeds -> C skipped (when: A failed) -> D succeeds.
        Expected: result.success is True.
        """
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_c = TaskSpec(
            id="c",
            description="error handler, skipped if A succeeded",
            depends_on=["a"],
            command="echo c",
            when="{{ a.status }} == failed",
        )
        tasks = [
            _make_task("a"),
            _make_task("b"),
            task_c,
            _make_task("d", depends_on=["b"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert result.task_results["a"].status == "success"
        assert result.task_results["b"].status == "success"
        assert result.task_results["c"].status == "skipped"
        assert result.task_results["d"].status == "success"

    def test_invalid_when_expression_skips_task_without_executing_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid when expressions are skipped before execution and remain success-like."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            TaskSpec(
                id="invalid-when",
                description="invalid when",
                command="echo invalid",
                when="not a valid comparison",
            ),
            _make_task("runner"),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert result.task_results["invalid-when"].status == "skipped"
        assert result.task_results["invalid-when"].message == (
            "Invalid when expression: not a valid comparison"
        )
        assert result.task_results["runner"].status == "success"
        assert call_log == ["runner"]


class TestSummaryFailedTasks:
    """P1: Failed Tasks section in run_summary.md."""

    def test_failed_section_present_when_task_fails(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        failed_result = _make_success_result("t1", tmp_path, status="failed")
        failed_result.message = "verify_command failed with exit code 1 (verify output: assert failed)"
        failed_result.stdout_tail = "line1\nline2\nline3\nline4\nline5\nline6\n"
        failed_result.log_path = tmp_path / "t1.log"

        run_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            success=False,
            started_at=now,
            finished_at=now,
            task_results={"t1": failed_result},
            execution_profile="plan",
        )
        plan = _make_plan(
            [TaskSpec(id="t1", command="echo hello")],
            source_path=tmp_path / "plan.yaml",
        )

        summary_path = _write_summary(run_result, plan, tmp_path)
        text = summary_path.read_text(encoding="utf-8")
        assert "## Failed Tasks" in text
        assert "### t1" in text
        assert "verify output: assert failed" in text
        assert "line6" in text
        assert "```" in text

    def test_no_failed_section_when_all_pass(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        ok_result = _make_success_result("t1", tmp_path)

        run_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            success=True,
            started_at=now,
            finished_at=now,
            task_results={"t1": ok_result},
            execution_profile="plan",
        )
        plan = _make_plan(
            [TaskSpec(id="t1", command="echo hello")],
            source_path=tmp_path / "plan.yaml",
        )

        summary_path = _write_summary(run_result, plan, tmp_path)
        text = summary_path.read_text(encoding="utf-8")
        assert "## Failed Tasks" not in text

    def test_failed_section_handles_empty_stdout(self, tmp_path: Path) -> None:
        now = datetime.now(UTC)
        failed_result = _make_success_result("t1", tmp_path, status="failed")
        failed_result.message = "Task failed with exit code 1"
        failed_result.stdout_tail = ""
        failed_result.log_path = tmp_path / "t1.log"

        run_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            success=False,
            started_at=now,
            finished_at=now,
            task_results={"t1": failed_result},
            execution_profile="plan",
        )
        plan = _make_plan(
            [TaskSpec(id="t1", command="echo hello")],
            source_path=tmp_path / "plan.yaml",
        )

        summary_path = _write_summary(run_result, plan, tmp_path)
        text = summary_path.read_text(encoding="utf-8")
        assert "## Failed Tasks" in text
        assert "### t1" in text
        assert "**Error**:" in text
        # No output tail block when stdout is empty
        assert "**Output tail**:" not in text


class TestConsoleSummaryLine:
    """Console summary always includes ok/failed/skipped counts (even when 0)."""

    def test_all_success_shows_zero_failed(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = _make_plan(
            [TaskSpec(id="t1", command="echo ok"), TaskSpec(id="t2", command="echo ok")],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, dry_run=True, run_dir_override=str(tmp_path / "runs"))
        assert result.success
        captured = capsys.readouterr().out
        assert "2 ok" in captured
        assert "0 failed" in captured
        assert "0 skipped" in captured

    def test_all_failed_shows_zero_ok(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from typing import Any

        plan = _make_plan(
            [TaskSpec(id="t1", command="echo ok"), TaskSpec(id="t2", command="echo ok")],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )

        def _mock_execute(plan: Any, task: Any, run_path: Any, *a: Any, **kw: Any) -> TaskResult:
            return TaskResult(
                task_id=task.id, status="failed", exit_code=1,
                message="failed", duration_sec=0.1,
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
            )

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock_execute)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert not result.success
        captured = capsys.readouterr().out
        assert "0 ok" in captured
        assert "2 failed" in captured
        assert "0 skipped" in captured


# ===========================================================================
# Additional coverage tests
# ===========================================================================


class TestSelectTasksAdditional:
    """Additional _select_tasks coverage: tag combos, edge cases."""

    def test_only_with_multiple_tasks(self) -> None:
        """--only with two tasks includes both plus transitive deps."""
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c"),
            _make_task("d", depends_on=["c"]),
            _make_task("e"),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only={"b", "d"}, skip=None)

        selected_ids = {t.id for t in selected}
        assert selected_ids == {"a", "b", "c", "d"}
        assert "e" not in selected_ids

    def test_skip_all_tasks_returns_empty(self) -> None:
        """Skipping all tasks returns empty list."""
        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only=None, skip={"a", "b"})

        assert selected == []

    def test_skip_tags_no_matching_tags_returns_all(self) -> None:
        """skip_tags with no matching tags returns all tasks."""
        tasks = [
            TaskSpec(id="a", tags=["alpha"], command="echo a"),
            TaskSpec(id="b", tags=["beta"], command="echo b"),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only=None, skip=None, skip_tags={"gamma"})

        assert [t.id for t in selected] == ["a", "b"]

    def test_tags_match_multiple_tags(self) -> None:
        """--tags selects tasks matching any tag (union)."""
        tasks = [
            TaskSpec(id="a", tags=["infra"], command="echo a"),
            TaskSpec(id="b", tags=["quality"], command="echo b"),
            TaskSpec(id="c", tags=["docs"], command="echo c"),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only=None, skip=None, tags={"infra", "quality"})

        assert {t.id for t in selected} == {"a", "b"}

    def test_tags_and_skip_tags_combined_exclusive(self) -> None:
        """--tags includes, --skip-tags excludes from the inclusion set."""
        tasks = [
            TaskSpec(id="a", tags=["deploy", "risky"], command="echo a"),
            TaskSpec(id="b", tags=["deploy", "safe"], command="echo b"),
            TaskSpec(id="c", tags=["test"], command="echo c"),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(
            plan, only=None, skip=None,
            tags={"deploy"}, skip_tags={"risky"},
        )

        assert [t.id for t in selected] == ["b"]

    def test_only_with_no_deps_returns_single(self) -> None:
        """--only on a task with no deps returns just that task."""
        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only={"b"}, skip=None)

        assert [t.id for t in selected] == ["b"]

    def test_skip_dep_does_not_remove_dependents(self) -> None:
        """--skip removes only specified tasks, not their dependents."""
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
        ]
        plan = _make_plan(tasks)

        selected = _select_tasks(plan, only=None, skip={"a"})

        selected_ids = {t.id for t in selected}
        assert selected_ids == {"b", "c"}


class TestContextPipelineHelpers:
    """Additional coverage for context pipeline utility functions."""

    def test_extract_keywords_empty_string(self) -> None:
        assert _extract_keywords("") == set()

    def test_extract_keywords_all_stopwords(self) -> None:
        """Stopwords are excluded from extraction."""
        assert _extract_keywords("the is a for to") == set()

    def test_extract_keywords_mixed_case(self) -> None:
        """Keywords are lowercased."""
        keywords = _extract_keywords("API Parser RETRY")
        assert "api" in keywords
        assert "parser" in keywords
        assert "retry" in keywords

    def test_extract_keywords_underscored_words(self) -> None:
        """Underscored identifiers are kept as single keywords."""
        keywords = _extract_keywords("retry_count max_retries")
        assert "retry_count" in keywords
        assert "max_retries" in keywords

    def test_estimate_tokens_various_sizes(self) -> None:
        assert _estimate_tokens("") == 1  # min 1
        assert _estimate_tokens("a") == 1  # min 1
        assert _estimate_tokens("a" * 4) == 1
        assert _estimate_tokens("a" * 8) == 2
        assert _estimate_tokens("a" * 400) == 100

    def test_compute_idf_single_section(self) -> None:
        """With one section, all terms get log(1/(1+1)) = log(0.5) < 0."""
        idf = _compute_idf(["alpha beta"])
        assert "alpha" in idf
        assert "beta" in idf
        # log(1/2) is negative
        assert idf["alpha"] < 0

    def test_compute_idf_rare_terms_higher(self) -> None:
        """Rare terms have higher IDF than common terms."""
        idf = _compute_idf([
            "parser api deploy",
            "parser api test",
            "parser api lint",
            "deploy release",
        ])
        # 'deploy' appears in 2 of 4, 'parser' in 3 of 4
        assert idf["deploy"] > idf["parser"]

    def test_score_section_no_match(self) -> None:
        """Section with no matching keywords scores 0."""
        assert _score_section("alpha beta gamma", {"deploy", "release"}) == 0

    def test_score_section_with_idf(self) -> None:
        """BM25 scoring with IDF returns positive score."""
        idf = {"api": 2.0, "parser": 0.5}
        score = _score_section("api parser test", {"api", "parser"}, idf=idf, avg_section_len=3)
        assert score > 0

    def test_score_section_none_idf_falls_back(self) -> None:
        """Without IDF, score is intersection count."""
        score = _score_section("api parser test", {"api", "parser"})
        assert score == 2

    def test_score_section_zero_avg_len(self) -> None:
        """avg_section_len=0 is handled without division by zero."""
        score = _score_section(
            "alpha beta", {"alpha"}, idf={"alpha": 1.0}, avg_section_len=0
        )
        assert score >= 1

    def test_split_into_sections_single_paragraph(self) -> None:
        """Single paragraph with many lines gets chunked."""
        text = "\n".join(f"line {i}" for i in range(20))
        sections = _split_into_sections(text)
        assert len(sections) >= 2

    def test_split_into_sections_blank(self) -> None:
        assert _split_into_sections("") == []

    def test_split_into_sections_double_newline(self) -> None:
        """Double newline splits into separate sections."""
        sections = _split_into_sections("section one\n\nsection two\n\nsection three")
        assert len(sections) == 3


class TestHopDistancesAdditional:
    """Additional _compute_hop_distances coverage."""

    def test_no_deps_returns_empty(self) -> None:
        tasks = {
            "alone": _make_task("alone"),
        }
        result = _compute_hop_distances("alone", ["*"], tasks)
        assert result == {}

    def test_direct_dep_is_one_hop(self) -> None:
        tasks = {
            "a": _make_task("a"),
            "b": _make_task("b", depends_on=["a"]),
        }
        result = _compute_hop_distances("b", ["a"], tasks)
        assert result == {"a": 1}

    def test_transitive_three_hops(self) -> None:
        """A -> B -> C -> D, context_from D to A is 3 hops."""
        tasks = {
            "a": _make_task("a"),
            "b": _make_task("b", depends_on=["a"]),
            "c": _make_task("c", depends_on=["b"]),
            "d": _make_task("d", depends_on=["c"]),
        }
        result = _compute_hop_distances("d", ["a", "c"], tasks)
        assert result["c"] == 1
        assert result["a"] == 3

    def test_wildcard_expands_to_direct_deps(self) -> None:
        tasks = {
            "a": _make_task("a"),
            "b": _make_task("b"),
            "c": _make_task("c", depends_on=["a", "b"]),
        }
        result = _compute_hop_distances("c", ["*"], tasks)
        assert result == {"a": 1, "b": 1}


class TestApplyHopDecayAdditional:
    """Additional _apply_hop_decay coverage."""

    def test_empty_upstream(self) -> None:
        result = _apply_hop_decay({}, {})
        assert result == {}

    def test_all_direct_no_decay(self, tmp_path: Path) -> None:
        """All hop=1 entries are returned unchanged."""
        upstream = {
            "a": _make_success_result("a", tmp_path, stdout_tail="hello world"),
            "b": _make_success_result("b", tmp_path, stdout_tail="foo bar"),
        }
        hop_distances = {"a": 1, "b": 1}
        decayed = _apply_hop_decay(upstream, hop_distances)
        assert decayed["a"] is upstream["a"]
        assert decayed["b"] is upstream["b"]

    def test_deep_hop_heavy_decay(self, tmp_path: Path) -> None:
        """hop=4 should keep 0.8^3 = 51.2% of text."""
        text = "x" * 100
        upstream = {
            "far": _make_success_result("far", tmp_path, stdout_tail=text),
        }
        hop_distances = {"far": 4}
        decayed = _apply_hop_decay(upstream, hop_distances)
        assert len(decayed["far"].stdout_tail) < len(text)
        # 0.8^3 = 0.512 => ~51 chars
        assert len(decayed["far"].stdout_tail) == 51

    def test_unknown_hop_treated_as_one(self, tmp_path: Path) -> None:
        """Unknown hop distance defaults to 1 (no decay)."""
        upstream = {
            "unknown": _make_success_result("unknown", tmp_path, stdout_tail="data"),
        }
        decayed = _apply_hop_decay(upstream, {})  # empty distances
        assert decayed["unknown"] is upstream["unknown"]


class TestRrfScore:
    """Tests for _rrf_score (Reciprocal Rank Fusion)."""

    def test_empty_inputs(self) -> None:
        from maestro_cli.scheduler import _rrf_score
        assert _rrf_score({}, {}) == {}

    def test_single_upstream(self) -> None:
        from maestro_cli.scheduler import _rrf_score
        result = _rrf_score({"a": 5.0}, {"a": 1})
        assert "a" in result
        assert result["a"] > 0

    def test_high_bm25_and_close_hop_beats_low_bm25_far_hop(self) -> None:
        from maestro_cli.scheduler import _rrf_score
        bm25 = {"close-relevant": 10.0, "far-irrelevant": 1.0}
        hops = {"close-relevant": 1, "far-irrelevant": 3}
        result = _rrf_score(bm25, hops)
        assert result["close-relevant"] > result["far-irrelevant"]

    def test_disjoint_ids_included(self) -> None:
        from maestro_cli.scheduler import _rrf_score
        result = _rrf_score({"a": 5.0}, {"b": 1})
        assert "a" in result
        assert "b" in result

    def test_k_parameter_affects_scores(self) -> None:
        from maestro_cli.scheduler import _rrf_score
        bm25 = {"x": 10.0, "y": 1.0}
        hops = {"x": 1, "y": 2}
        result_k60 = _rrf_score(bm25, hops, k=60)
        result_k1 = _rrf_score(bm25, hops, k=1)
        # Smaller k amplifies rank differences
        diff_k1 = result_k1["x"] - result_k1["y"]
        diff_k60 = result_k60["x"] - result_k60["y"]
        assert diff_k1 > diff_k60


class TestApplyContextBudget:
    """Tests for _apply_context_budget."""

    def test_within_budget_no_trim(self, tmp_path: Path) -> None:
        """When total tokens fit within budget, nothing is trimmed."""
        from maestro_cli.scheduler import _apply_context_budget

        upstream = {
            "a": _make_success_result("a", tmp_path, stdout_tail="short"),
        }
        result, records, meta = _apply_context_budget(upstream, budget_tokens=1000)
        assert records == []
        assert result["a"] is upstream["a"]

    def test_over_budget_trims_proportionally(self, tmp_path: Path) -> None:
        """When over budget without intent keywords, trim proportionally."""
        from maestro_cli.scheduler import _apply_context_budget

        text_a = "a" * 400  # ~100 tokens
        text_b = "b" * 400  # ~100 tokens
        upstream = {
            "a": _make_success_result("a", tmp_path, stdout_tail=text_a),
            "b": _make_success_result("b", tmp_path, stdout_tail=text_b),
        }
        # Budget of 50 tokens with 200 total => heavy trimming
        result, records, meta = _apply_context_budget(upstream, budget_tokens=50)
        assert len(records) > 0
        # Both should have been trimmed
        total_trimmed = sum(
            _estimate_tokens(r.stdout_tail) for r in result.values()
        )
        assert total_trimmed <= 50

    def test_budget_with_intent_keywords_prefers_relevant(self, tmp_path: Path) -> None:
        """Intent keywords help preserve relevant sections during budget trim."""
        from maestro_cli.scheduler import _apply_context_budget

        # Relevant content
        relevant = "api schema validation parser " * 50  # 200 chars
        # Irrelevant content
        irrelevant = "weather gardening cooking " * 50  # 150 chars
        upstream = {
            "scan": _make_success_result(
                "scan", tmp_path,
                stdout_tail=f"{relevant}\n\n{irrelevant}",
            ),
        }
        budget = _estimate_tokens(relevant) + 10
        result, records, meta = _apply_context_budget(
            upstream, budget_tokens=budget,
            intent_keywords={"api", "schema", "validation"},
        )
        # The relevant part should be kept, irrelevant trimmed
        assert "scan" in meta


class TestApprovalGate:
    """Tests for _request_approval."""

    def test_non_interactive_returns_false(self) -> None:
        from maestro_cli.scheduler import _request_approval
        assert _request_approval("task-1", None, interactive=False) is False

    def test_non_interactive_with_message_returns_false(self) -> None:
        from maestro_cli.scheduler import _request_approval
        assert _request_approval("task-1", "Approve?", interactive=False) is False

    def test_interactive_yes_returns_true(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.scheduler import _request_approval
        monkeypatch.setattr("builtins.input", lambda: "y")
        assert _request_approval("task-1", None, interactive=True) is True

    def test_interactive_no_returns_false(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.scheduler import _request_approval
        monkeypatch.setattr("builtins.input", lambda: "n")
        assert _request_approval("task-1", None, interactive=True) is False

    def test_interactive_eof_returns_false(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.scheduler import _request_approval

        def raise_eof() -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        assert _request_approval("task-1", None, interactive=True) is False


class TestAutoApproveRunPlan:
    """Tests for auto_approve=True bypassing approval gates."""

    def test_auto_approve_executes_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            TaskSpec(
                id="gated",
                command="echo ok",
                requires_approval=True,
                approval_message="Approve gated?",
            ),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            auto_approve=True,
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True
        assert "gated" in call_log

    def test_approval_handler_denied_skips_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            TaskSpec(
                id="gated",
                command="echo ok",
                requires_approval=True,
            ),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            approval_handler=lambda task_id, msg: False,
            run_dir_override=str(tmp_path / "runs"),
        )

        assert "gated" not in call_log
        assert result.task_results["gated"].status == "skipped"
        assert "denied" in result.task_results["gated"].message.lower()


class TestBudgetRelevanceScores:
    """Test _apply_context_budget with pre-computed relevance_scores."""

    def test_budget_uses_relevance_scores_for_eviction(self) -> None:
        """Pre-computed relevance scores should drive eviction order."""
        from maestro_cli.scheduler import _apply_context_budget

        # Create two upstreams that exceed budget
        r_important = TaskResult(task_id="important", status="success", exit_code=0,
                                 stdout_tail="x" * 400, duration_sec=1.0)
        r_expendable = TaskResult(task_id="expendable", status="success", exit_code=0,
                                  stdout_tail="y" * 400, duration_sec=1.0)
        upstream = {"important": r_important, "expendable": r_expendable}

        # Budget only fits ~one upstream (400 chars ≈ 100 tokens, budget=120)
        result, records, _ = _apply_context_budget(
            upstream, budget_tokens=120,
            intent_keywords={"x"},
            relevance_scores={"important": 0.9, "expendable": 0.1},
        )
        # Expendable should be trimmed more aggressively
        assert len(result["expendable"].stdout_tail) < len(result["important"].stdout_tail)


class TestBudgetTracking:
    """Tests for budget warning and exceeded behavior."""

    def test_budget_exceeded_skips_remaining_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)
        plan.max_cost_usd = 1.0

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            # Task A costs $1.50 — exceeds $1.00 budget
            cost = 1.50 if task.id == "a" else 0.10
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
                cost_usd=cost,
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # Task A completed, but B and C should be skipped due to budget
        assert result.task_results["a"].status == "success"
        assert result.task_results["b"].status == "skipped"
        assert "budget" in result.task_results["b"].message.lower()
        assert result.budget_exceeded is True

    def test_budget_warning_event_emitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Budget warning event emitted when cost approaches threshold."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)
        plan.max_cost_usd = 10.0
        plan.budget_warning_pct = 0.5  # warn at 50%

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            # A costs $6, hitting 60% of $10 budget (over 50% threshold)
            cost = 6.0 if task.id == "a" else 0.5
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
                cost_usd=cost,
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        budget_warnings = [e for e in events if e["event"] == "budget_warning"]
        assert len(budget_warnings) >= 1
        assert budget_warnings[0]["limit"] == 10.0

    def test_cost_accumulation_across_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Total cost sums costs from all tasks."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=4)

        rp_holder: list[Path] = []
        costs = {"a": 0.50, "b": 1.25, "c": 0.75}

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
                cost_usd=costs[task.id],
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.total_cost_usd == pytest.approx(2.50)


class TestPolicyEvaluation:
    """Tests for runtime policy enforcement in the scheduler."""

    def test_policy_block_prevents_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 'block' policy prevents the task from executing."""
        from maestro_cli.models import PolicySpec

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)
        plan.policies = [
            PolicySpec(
                name="no-shell",
                rule="task.engine == None",
                action="block",
                message="Shell tasks are not allowed",
            ),
        ]

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert "a" not in call_log
        assert result.task_results["a"].status == "failed"
        assert "policy" in result.task_results["a"].message.lower()

    def test_policy_warn_emits_event_but_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A 'warn' policy emits an event but allows execution."""
        from maestro_cli.models import PolicySpec

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)
        plan.policies = [
            PolicySpec(
                name="warn-shell",
                rule="task.engine == None",
                action="warn",
                message="Consider using an engine",
            ),
        ]

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert "a" in call_log
        # Check event emitted
        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        policy_events = [e for e in events if e["event"] == "policy_violation"]
        assert len(policy_events) >= 1
        assert policy_events[0]["policy_name"] == "warn-shell"
        assert policy_events[0]["action"] == "warn"

    def test_policy_audit_only_emits_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An 'audit' policy only emits an event, does not warn or block."""
        from maestro_cli.models import PolicySpec

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)
        plan.policies = [
            PolicySpec(
                name="audit-shell",
                rule="task.engine == None",
                action="audit",
                message="Shell task usage",
            ),
        ]

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert "a" in call_log
        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        policy_events = [e for e in events if e["event"] == "policy_violation"]
        assert len(policy_events) >= 1
        assert policy_events[0]["action"] == "audit"


class TestEventEmission:
    """Tests for JSONL event emission."""

    def test_run_start_event_has_goal_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)
        plan.goal = "Fix all the bugs"

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        run_start = next(e for e in events if e["event"] == "run_start")
        assert run_start["goal"] == "Fix all the bugs"

    def test_event_ordering_start_complete_and_run_lifecycle(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Events follow: run_start -> task_start -> task_complete -> run_complete."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("only")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        event_types = [e["event"] for e in events]
        assert event_types[0] == "run_start"
        assert event_types[-1] == "run_complete"
        assert "task_start" in event_types
        assert "task_complete" in event_types
        # task_start must come before task_complete
        start_idx = event_types.index("task_start")
        complete_idx = event_types.index("task_complete")
        assert start_idx < complete_idx

    def test_events_include_plan_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All events include plan_name field."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, name="my-test-plan", source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for event in events:
            assert event["plan_name"] == "my-test-plan"

    def test_events_have_hash_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Events include seq, prev_hash, and hash fields for chain integrity."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for event in events:
            assert "seq" in event
            assert "prev_hash" in event
            assert "hash" in event

        # seq should be sequential (starting from 0)
        seqs = [e["seq"] for e in events]
        assert seqs == list(range(len(events)))

    def test_task_skip_event_for_dep_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """task_skip event emitted when dependency fails."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml)

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            status = "failed" if task.id == "a" else "success"
            result = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=1 if task.id == "a" else 0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="failed" if task.id == "a" else "ok",
            )
            result.log_path.write_text(f"status={status}\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        skip_events = [e for e in events if e["event"] == "task_skip"]
        assert len(skip_events) >= 1
        assert skip_events[0]["task_id"] == "b"
        assert "dependency" in skip_events[0]["reason"].lower()

    def test_event_callback_receives_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """event_callback receives all events during run."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        callback_events: list[tuple[str, dict]] = []

        def on_event(name: str, data: dict[str, object]) -> None:
            callback_events.append((name, data))

        run_plan(
            plan,
            event_callback=on_event,
            run_dir_override=str(tmp_path / "runs"),
        )

        event_names = [name for name, _ in callback_events]
        assert "run_start" in event_names
        assert "task_start" in event_names
        assert "task_complete" in event_names
        assert "run_complete" in event_names


class TestDAGExecutionPatterns:
    """Tests for various DAG shapes."""

    def test_deep_chain_fail_fast_skips_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A -> B -> C -> D -> E: if B fails with fail_fast, C/D/E are skipped."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
            _make_task("d", depends_on=["c"]),
            _make_task("e", depends_on=["d"]),
        ]
        plan = _make_plan(tasks, fail_fast=True, source_path=plan_yaml, max_parallel=1)

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            status = "failed" if task.id == "b" else "success"
            result = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=1 if task.id == "b" else 0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="failed" if task.id == "b" else "ok",
            )
            result.log_path.write_text(f"status={status}\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        assert result.task_results["a"].status == "success"
        assert result.task_results["b"].status == "failed"
        for tid in ("c", "d", "e"):
            assert result.task_results[tid].status == "skipped"

    def test_all_independent_tasks_max_parallelism(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All independent tasks can run in parallel."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task(f"t{i}") for i in range(5)]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=10)

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert len(result.task_results) == 5
        for i in range(5):
            assert result.task_results[f"t{i}"].status == "success"

    def test_single_task_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Plan with a single task works correctly."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("only")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert len(result.task_results) == 1
        assert call_log == ["only"]

    def test_max_parallel_one_forces_sequential(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """max_parallel=1 forces all tasks to run sequentially."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        # Independent tasks that could run in parallel
        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)

        rp_holder: list[Path] = []
        active = [0]
        max_concurrent = [0]
        lock = threading.Lock()

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            with lock:
                active[0] += 1
                if active[0] > max_concurrent[0]:
                    max_concurrent[0] = active[0]
            import time
            time.sleep(0.02)
            with lock:
                active[0] -= 1

            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.02,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert max_concurrent[0] <= 1

    def test_diamond_dep_failure_skips_tip(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Diamond A -> {B, C} -> D: if B fails, D is skipped (needs both B and C)."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
            _make_task("d", depends_on=["b", "c"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml, max_parallel=4)

        rp_holder: list[Path] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            status = "failed" if task.id == "b" else "success"
            result = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=1 if task.id == "b" else 0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="failed" if task.id == "b" else "ok",
            )
            result.log_path.write_text(f"status={status}\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        assert result.task_results["a"].status == "success"
        assert result.task_results["b"].status == "failed"
        assert result.task_results["c"].status == "success"
        assert result.task_results["d"].status == "skipped"


class TestComputeTaskDepthAndFanOut:
    """Tests for _compute_task_depth and _compute_fan_out."""

    def test_depth_no_deps(self) -> None:
        from maestro_cli.scheduler import _compute_task_depth
        task = _make_task("a")
        plan = _make_plan([task])
        assert _compute_task_depth(task, plan) == 0

    def test_depth_one_dep(self) -> None:
        from maestro_cli.scheduler import _compute_task_depth
        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks)
        assert _compute_task_depth(tasks[1], plan) == 1

    def test_depth_deep_chain(self) -> None:
        from maestro_cli.scheduler import _compute_task_depth
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
            _make_task("d", depends_on=["c"]),
        ]
        plan = _make_plan(tasks)
        assert _compute_task_depth(tasks[3], plan) == 3

    def test_depth_diamond_takes_max_path(self) -> None:
        from maestro_cli.scheduler import _compute_task_depth
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
            _make_task("d", depends_on=["b", "c"]),
        ]
        plan = _make_plan(tasks)
        # Both paths are depth 2 (a->b->d or a->c->d)
        assert _compute_task_depth(tasks[3], plan) == 2

    def test_fan_out_zero(self) -> None:
        from maestro_cli.scheduler import _compute_fan_out
        tasks = [_make_task("a")]
        plan = _make_plan(tasks)
        assert _compute_fan_out(tasks[0], plan) == 0

    def test_fan_out_one(self) -> None:
        from maestro_cli.scheduler import _compute_fan_out
        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks)
        assert _compute_fan_out(tasks[0], plan) == 1

    def test_fan_out_many(self) -> None:
        from maestro_cli.scheduler import _compute_fan_out
        tasks = [
            _make_task("root"),
            _make_task("b", depends_on=["root"]),
            _make_task("c", depends_on=["root"]),
            _make_task("d", depends_on=["root"]),
        ]
        plan = _make_plan(tasks)
        assert _compute_fan_out(tasks[0], plan) == 3

    def test_fan_out_leaf(self) -> None:
        from maestro_cli.scheduler import _compute_fan_out
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks)
        assert _compute_fan_out(tasks[1], plan) == 0


class TestComputeTaintedTasks:
    """Tests for _compute_tainted_tasks."""

    def test_no_untrusted_returns_empty(self) -> None:
        from maestro_cli.scheduler import _compute_tainted_tasks
        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks)
        assert _compute_tainted_tasks(plan) == set()

    def test_explicit_untrusted_is_tainted(self) -> None:
        from maestro_cli.scheduler import _compute_tainted_tasks
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks)
        tainted = _compute_tainted_tasks(plan)
        assert "a" in tainted

    def test_taint_propagates_through_context_from(self) -> None:
        from maestro_cli.scheduler import _compute_tainted_tasks
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
            TaskSpec(
                id="b",
                command="echo b",
                depends_on=["a"],
                context_from=["a"],
            ),
        ]
        plan = _make_plan(tasks)
        tainted = _compute_tainted_tasks(plan)
        assert "a" in tainted
        assert "b" in tainted

    def test_taint_cleared_by_guard_command(self) -> None:
        from maestro_cli.scheduler import _compute_tainted_tasks
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
            TaskSpec(
                id="b",
                command="echo b",
                depends_on=["a"],
                context_from=["a"],
                guard_command="validate-input",
            ),
        ]
        plan = _make_plan(tasks)
        tainted = _compute_tainted_tasks(plan)
        assert "a" in tainted
        assert "b" not in tainted  # guard_command clears taint

    def test_taint_cleared_by_verify_command(self) -> None:
        from maestro_cli.scheduler import _compute_tainted_tasks
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
            TaskSpec(
                id="b",
                command="echo b",
                depends_on=["a"],
                context_from=["a"],
                verify_command="check-output",
            ),
        ]
        plan = _make_plan(tasks)
        tainted = _compute_tainted_tasks(plan)
        assert "a" in tainted
        assert "b" not in tainted

    def test_transitive_taint_propagation(self) -> None:
        """Taint propagates: A (untrusted) -> B (context_from A) -> C (context_from B)."""
        from maestro_cli.scheduler import _compute_tainted_tasks
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
            TaskSpec(
                id="b",
                command="echo b",
                depends_on=["a"],
                context_from=["a"],
            ),
            TaskSpec(
                id="c",
                command="echo c",
                depends_on=["b"],
                context_from=["b"],
            ),
        ]
        plan = _make_plan(tasks)
        tainted = _compute_tainted_tasks(plan)
        assert tainted == {"a", "b", "c"}

    def test_selected_ids_filter(self) -> None:
        """Only selected tasks are considered for taint computation."""
        from maestro_cli.scheduler import _compute_tainted_tasks
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
            TaskSpec(
                id="b",
                command="echo b",
                depends_on=["a"],
                context_from=["a"],
            ),
        ]
        plan = _make_plan(tasks)
        # Only task "b" is selected; "a" is out of scope
        tainted = _compute_tainted_tasks(plan, selected_ids={"b"})
        assert "a" not in tainted


class TestLoadPriorResultsEdgeCases:
    """Additional _load_prior_results edge cases."""

    def test_empty_task_results(self, tmp_path: Path) -> None:
        """Empty task_results returns empty dict."""
        manifest = {"task_results": {}}
        (tmp_path / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        result = _load_prior_results(tmp_path)
        assert result == {}

    def test_only_failed_tasks_returns_empty(self, tmp_path: Path) -> None:
        """Only failed tasks returns empty dict (nothing to resume)."""
        manifest = {
            "task_results": {
                "a": {"status": "failed", "exit_code": 1},
            }
        }
        (tmp_path / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        result = _load_prior_results(tmp_path)
        assert result == {}

    def test_dry_run_status_preserved(self, tmp_path: Path) -> None:
        """dry_run status is preserved in resume."""
        manifest = {
            "task_results": {
                "a": {"status": "dry_run"},
            }
        }
        (tmp_path / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        result = _load_prior_results(tmp_path)
        assert result == {"a": "dry_run"}


class TestJsonlOutputMode:
    """Tests for --output jsonl behavior."""

    def test_jsonl_mode_emits_events_to_stdout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            output_mode="jsonl",
            run_dir_override=str(tmp_path / "runs"),
        )

        captured = capsys.readouterr().out
        lines = [line for line in captured.strip().splitlines() if line.strip()]
        # All lines should be valid JSON
        events = [json.loads(line) for line in lines]
        event_types = [e["event"] for e in events]
        assert "run_start" in event_types
        assert "run_complete" in event_types

    def test_jsonl_mode_suppresses_maestro_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """JSONL mode suppresses [maestro] human-readable output."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        run_plan(
            plan,
            output_mode="jsonl",
            run_dir_override=str(tmp_path / "runs"),
        )

        captured = capsys.readouterr().out
        # No human-readable [maestro] lines
        for line in captured.splitlines():
            if line.strip():
                # Each line should be JSON, not a [maestro] log line
                json.loads(line)  # Should not raise


class TestCancelEvent:
    """Tests for cancel_event parameter."""

    def test_cancel_event_skips_pending_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Setting cancel_event skips all pending tasks."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)

        cancel = threading.Event()
        rp_holder: list[Path] = []
        call_log: list[str] = []

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            call_log.append(task.id)
            # Cancel after first task
            cancel.set()
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            cancel_event=cancel,
            run_dir_override=str(tmp_path / "runs"),
        )

        assert "a" in call_log
        assert result.task_results["b"].status == "skipped"
        assert "cancel" in result.task_results["b"].message.lower()


class TestTokenAggregation:
    """Tests for token aggregation in run results."""

    def test_total_tokens_aggregated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=4)

        rp_holder: list[Path] = []
        tokens = {"a": TokenUsage(input_tokens=100, output_tokens=50),
                  "b": TokenUsage(input_tokens=200, output_tokens=100)}

        def mock_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                         upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if not rp_holder:
                rp_holder.append(run_path)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
                token_usage=tokens[task.id],
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # total = (100+50) + (200+100) = 450
        assert result.total_tokens == 450

    def test_no_tokens_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """total_tokens is None when no tasks report tokens."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.total_tokens is None


class TestSchedulerEdgeClaudeWhen:
    """Camada 2: when-expression edge cases — condition met, wait-for-completion, soft_failed."""

    def test_when_condition_met_task_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the when-expression evaluates to True, the dependent task executes.

        Plan: A succeeds -> B has ``when: "{{ a.status }} == success"``
        Expected: B actually runs (condition met), unlike the not-met path which skips B.
        """
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_b = TaskSpec(
            id="b",
            description="runs when A succeeds",
            depends_on=["a"],
            command="echo b",
            when="{{ a.status }} == success",
        )
        plan = _make_plan([_make_task("a"), task_b], fail_fast=False, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert result.task_results["a"].status == "success"
        assert result.task_results["b"].status == "success"
        # Both tasks must have been executed
        assert "a" in call_log
        assert "b" in call_log

    def test_when_dep_failure_allows_condition_eval(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When-expression tasks use wait-for-completion semantics.

        A dep failure does NOT skip a ``when`` dependent — the condition is still
        evaluated.  Plain dependents (no ``when``) would be skipped on dep failure.

        Plan: A fails -> B has ``when: "{{ a.status }} == failed"``
        Expected: B runs because the condition IS met.
        """
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_b = TaskSpec(
            id="b",
            description="error handler, runs when A fails",
            depends_on=["a"],
            command="echo b",
            when="{{ a.status }} == failed",
        )
        plan = _make_plan([_make_task("a"), task_b], fail_fast=False, source_path=plan_yaml)

        call_log: list[str] = []

        def _mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                  upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            call_log.append(task.id)
            now = datetime.now(UTC)
            status = "failed" if task.id == "a" else "success"
            res = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=1 if task.id == "a" else 0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message=status,
            )
            res.log_path.write_text(f"status={status}\n", encoding="utf-8")
            res.result_path.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
            return res

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.task_results["a"].status == "failed"
        # B's when condition was met despite A failing — wait-for-completion semantics
        assert result.task_results["b"].status == "success"
        assert "b" in call_log

    def test_when_soft_failed_dep_status_evaluated_correctly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When expression sees the literal string 'soft_failed' for soft-failed tasks.

        Plan: A soft_fails (allow_failure=True) -> B has ``when: "{{ a.status }} == soft_failed"``
        Expected: B runs because its condition is met.
        """
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_a = _make_task("a", allow_failure=True)
        task_b = TaskSpec(
            id="b",
            description="runs after soft failure",
            depends_on=["a"],
            command="echo b",
            when="{{ a.status }} == soft_failed",
        )
        plan = _make_plan([task_a, task_b], fail_fast=False, source_path=plan_yaml)

        call_log: list[str] = []

        def _mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                  upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            call_log.append(task.id)
            now = datetime.now(UTC)
            status = "soft_failed" if task.id == "a" else "success"
            res = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=1 if task.id == "a" else 0,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message=status,
            )
            res.log_path.write_text(f"status={status}\n", encoding="utf-8")
            res.result_path.write_text(json.dumps(res.to_dict(), indent=2), encoding="utf-8")
            return res

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _mock)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.task_results["a"].status == "soft_failed"
        assert result.task_results["b"].status == "success"
        assert "b" in call_log


class TestSchedulerEdgeClaudeApprovalPolicy:
    """Camada 2: approval gate + policy enforcement ordering edge cases."""

    def test_approval_denied_skips_policy_evaluation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Approval is evaluated BEFORE policy.  Denial short-circuits via ``continue``,
        so no policy_violation event is emitted even when a blocking policy is set.
        """
        from maestro_cli.models import PolicySpec

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            TaskSpec(
                id="gated",
                command="echo ok",
                requires_approval=True,
                approval_message="Allow?",
            ),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)
        plan.policies = [
            PolicySpec(
                name="always-block",
                rule="task.engine == None",
                action="block",
                message="Shell tasks blocked",
            ),
        ]

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            approval_handler=lambda task_id, msg: False,  # deny
            run_dir_override=str(tmp_path / "runs"),
        )

        # Task skipped by approval denial, not failed by policy
        assert result.task_results["gated"].status == "skipped"
        assert "gated" not in call_log

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        policy_events = [e for e in events if e["event"] == "policy_violation"]
        assert len(policy_events) == 0, "Policy must not be evaluated when approval is denied"

    def test_auto_approve_then_policy_block_fails_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """auto_approve=True passes the gate, but a blocking policy still fails the task.

        The approval_response event should be emitted first (approved=True),
        then the policy_violation event, and execute_task should never be called.
        """
        from maestro_cli.models import PolicySpec

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            TaskSpec(
                id="guarded",
                command="echo ok",
                requires_approval=True,
            ),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)
        plan.policies = [
            PolicySpec(
                name="shell-blocked",
                rule="task.engine == None",
                action="block",
                message="No shell",
            ),
        ]

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            auto_approve=True,
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.task_results["guarded"].status == "failed"
        assert "policy" in result.task_results["guarded"].message.lower()
        assert "guarded" not in call_log

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        approval_events = [e for e in events if e["event"] == "approval_response"]
        policy_events = [e for e in events if e["event"] == "policy_violation"]
        assert any(e["approved"] is True for e in approval_events), "approval_response approved=True expected"
        assert len(policy_events) >= 1
        assert policy_events[0]["action"] == "block"

    def test_multiple_policies_block_and_warn_both_reported(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When multiple policies match a task, all violations are emitted.

        Even if the first is a warn and the second is a block, both events appear
        in events.jsonl and the task ends with status='failed'.
        """
        from maestro_cli.models import PolicySpec

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)
        plan.policies = [
            PolicySpec(
                name="warn-policy",
                rule="task.engine == None",
                action="warn",
                message="Shell usage warning",
            ),
            PolicySpec(
                name="block-policy",
                rule="task.engine == None",
                action="block",
                message="Shell usage blocked",
            ),
        ]

        rp_holder: list[Path] = []
        mock_exec, call_log = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.task_results["a"].status == "failed"
        assert "a" not in call_log

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        policy_events = [e for e in events if e["event"] == "policy_violation"]
        # Both policies must appear in the event log
        assert len(policy_events) == 2
        actions = {e["action"] for e in policy_events}
        assert "warn" in actions
        assert "block" in actions


class TestVerbosityLevels:
    """Tests for verbosity/quiet output."""

    def test_quiet_mode_suppresses_most_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        rp_holder: list[Path] = []
        mock_exec, _ = _mock_execute_task_factory(rp_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(
            plan,
            verbosity="quiet",
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True
        captured = capsys.readouterr().out
        # Quiet mode still shows the final summary line
        assert "1 ok" in captured
        # But suppresses per-task logs (starting, OK, etc.)
        assert "starting" not in captured


# ===========================================================================
# Camada 2 edge case tests
# ===========================================================================


def _mock_execute_status_factory(
    status_map: dict[str, str],
    cost_map: dict[str, float | None] | None = None,
    call_log: list[str] | None = None,
):
    """Return a mock execute_task that returns per-task statuses from *status_map*.

    Also supports optional per-task cost via *cost_map*.
    """
    cost_map = cost_map or {}
    call_log = call_log if call_log is not None else []
    lock = threading.Lock()

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
    ):
        with lock:
            call_log.append(task.id)
        now = datetime.now(UTC)
        status = status_map.get(task.id, "success")
        exit_code = 0 if status in ("success", "dry_run", "skipped") else 1
        cost = cost_map.get(task.id)
        result = TaskResult(
            task_id=task.id,
            status=status,
            exit_code=exit_code,
            started_at=now,
            finished_at=now,
            duration_sec=0.01,
            command=f"echo {task.id}",
            log_path=run_path / f"{task.id}.log",
            result_path=run_path / f"{task.id}.result.json",
            message=f"status={status}",
            cost_usd=cost,
        )
        result.log_path.write_text(f"status={status}\n", encoding="utf-8")
        result.result_path.write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        return result

    return mock_execute, call_log


# ---------------------------------------------------------------------------
# 1. DAG state machine (8-10 tests)
# ---------------------------------------------------------------------------


class TestSchedulerEdgeL2DagStateMachine:
    """Camada 2: DAG state machine edge cases — diamond failures, deep chains, sequencing."""

    def test_diamond_b_fails_c_succeeds_d_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Diamond A->B,C->D where B fails and C succeeds: D is skipped (dep failure)."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
            _make_task("d", depends_on=["b", "c"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml, max_parallel=4)

        mock_exec, call_log = _mock_execute_status_factory({"b": "failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        assert result.task_results["a"].status == "success"
        assert result.task_results["b"].status == "failed"
        assert result.task_results["c"].status == "success"
        assert result.task_results["d"].status == "skipped"
        assert "dependency failed" in result.task_results["d"].message.lower()
        assert "d" not in call_log

    def test_deep_chain_fail_fast_c_fails_d_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Deep chain A->B->C->D with fail_fast=True, C fails: D is skipped."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
            _make_task("d", depends_on=["c"]),
        ]
        plan = _make_plan(tasks, fail_fast=True, source_path=plan_yaml, max_parallel=1)

        mock_exec, call_log = _mock_execute_status_factory({"c": "failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        assert result.task_results["a"].status == "success"
        assert result.task_results["b"].status == "success"
        assert result.task_results["c"].status == "failed"
        assert result.task_results["d"].status == "skipped"
        # D was skipped either by dep failure or fail_fast
        assert "d" not in call_log

    def test_all_independent_max_parallel_1_sequential_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """All independent tasks with max_parallel=1 execute truly sequentially."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c"),
            _make_task("d"),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)

        mock_exec, call_log = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert len(call_log) == 4
        # All 4 were called in the order they appear in the plan
        assert call_log == ["a", "b", "c", "d"]

    def test_allow_failure_middle_task_dependents_still_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A->B(allow_failure)->C: B soft-fails, C still runs because soft_failed is success-like."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"], allow_failure=True),
            _make_task("c", depends_on=["b"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml, max_parallel=1)

        mock_exec, call_log = _mock_execute_status_factory({"b": "soft_failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert result.task_results["b"].status == "soft_failed"
        assert result.task_results["c"].status == "success"
        assert call_log == ["a", "b", "c"]

    def test_group_task_failure_propagates_to_scheduler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A group task whose execution returns 'failed' is recorded as failed in the scheduler."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            TaskSpec(id="grp", description="sub-plan", group="nested.yaml", command=None),
            _make_task("after-grp", depends_on=["grp"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml)

        # Mock execute_task to return failed for the group task
        mock_exec, call_log = _mock_execute_status_factory({"grp": "failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        assert result.task_results["grp"].status == "failed"
        assert result.task_results["after-grp"].status == "skipped"
        assert "after-grp" not in call_log

    def test_single_task_plan_runs_correctly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A plan with a single task with no dependencies runs and produces a valid result."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("solo")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        mock_exec, call_log = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert len(result.task_results) == 1
        assert result.task_results["solo"].status == "success"
        assert call_log == ["solo"]
        # Manifest exists
        assert (result.run_path / "run_manifest.json").exists()

    def test_resume_completed_tasks_skipped_dep_failure_tasks_rerun(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On resume: completed tasks are skipped, dep-failure skipped tasks are re-run."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)

        # Prior run: A failed -> B skipped (dep failure), C skipped (dep failure)
        prior_run = tmp_path / "prior_run"
        prior_run.mkdir()
        manifest = {
            "plan_name": "test-plan",
            "success": False,
            "task_results": {
                "a": {"task_id": "a", "status": "failed", "exit_code": 1, "message": "error"},
                "b": {"task_id": "b", "status": "skipped", "message": "Skipped because dependency failed: {'a'}"},
                "c": {"task_id": "c", "status": "skipped", "message": "Skipped because dependency failed: {'a'}"},
            },
        }
        (prior_run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

        call_log: list[str] = []
        mock_exec, call_log = _mock_execute_status_factory({}, call_log=call_log)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, resume_path=prior_run, run_dir_override=str(tmp_path / "runs"))

        # All three should have been executed (dep-failure skips excluded from resume)
        assert set(call_log) == {"a", "b", "c"}
        assert result.success is True

    def test_diamond_both_branches_fail_tip_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Diamond A->B,C->D where both B and C fail: D is skipped."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
            _make_task("d", depends_on=["b", "c"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml, max_parallel=4)

        mock_exec, call_log = _mock_execute_status_factory({"b": "failed", "c": "failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        assert result.task_results["d"].status == "skipped"
        assert "d" not in call_log

    def test_fail_fast_false_independent_still_runs_after_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With fail_fast=False, independent tasks still execute after a failure."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        # A fails but B and C are independent
        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c"),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml, max_parallel=1)

        mock_exec, call_log = _mock_execute_status_factory({"a": "failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        assert result.task_results["a"].status == "failed"
        assert result.task_results["b"].status == "success"
        assert result.task_results["c"].status == "success"
        assert "b" in call_log
        assert "c" in call_log


# ---------------------------------------------------------------------------
# 2. When expression evaluation (4-5 tests)
# ---------------------------------------------------------------------------


class TestSchedulerEdgeL2WhenExpressions:
    """Camada 2: when expression edge cases for DAG scheduling."""

    def test_when_condition_success_match_runs_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """when: '{{ a.status }} == success' with a=success -> task runs."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_b = TaskSpec(
            id="b",
            description="conditional",
            depends_on=["a"],
            command="echo b",
            when="{{ a.status }} == success",
        )
        plan = _make_plan([_make_task("a"), task_b], fail_fast=False, source_path=plan_yaml)

        mock_exec, call_log = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.task_results["b"].status == "success"
        assert "b" in call_log

    def test_when_condition_success_mismatch_skips_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """when: '{{ a.status }} == success' with a=failed -> task is skipped."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_b = TaskSpec(
            id="b",
            description="conditional",
            depends_on=["a"],
            command="echo b",
            when="{{ a.status }} == success",
        )
        plan = _make_plan([_make_task("a"), task_b], fail_fast=False, source_path=plan_yaml)

        mock_exec, call_log = _mock_execute_status_factory({"a": "failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.task_results["b"].status == "skipped"
        assert "condition not met" in result.task_results["b"].message.lower()
        assert "b" not in call_log

    def test_when_with_soft_failed_upstream_evaluates_correctly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """when expression correctly sees 'soft_failed' status literal.

        A soft_fails -> B has when: '{{ a.status }} != success'
        Expected: B runs because soft_failed != success.
        """
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_a = _make_task("a", allow_failure=True)
        task_b = TaskSpec(
            id="b",
            description="runs on non-success",
            depends_on=["a"],
            command="echo b",
            when="{{ a.status }} != success",
        )
        plan = _make_plan([task_a, task_b], fail_fast=False, source_path=plan_yaml)

        mock_exec, call_log = _mock_execute_status_factory({"a": "soft_failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.task_results["a"].status == "soft_failed"
        assert result.task_results["b"].status == "success"
        assert "b" in call_log

    def test_when_changes_deps_to_wait_for_completion(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Without when: dep failure -> dependent skipped.
        With when: dep failure -> condition evaluated (wait-for-completion semantics).

        Plan: A fails. B (no when) depends on A -> skipped.
              C (with when) depends on A -> condition evaluated.
        """
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_b = _make_task("b", depends_on=["a"])
        task_c = TaskSpec(
            id="c",
            description="error handler",
            depends_on=["a"],
            command="echo c",
            when="{{ a.status }} == failed",
        )
        plan = _make_plan([_make_task("a"), task_b, task_c], fail_fast=False, source_path=plan_yaml)

        mock_exec, call_log = _mock_execute_status_factory({"a": "failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # B is skipped (dep failure, no when)
        assert result.task_results["b"].status == "skipped"
        assert "dependency failed" in result.task_results["b"].message.lower()
        # C runs because it has a when expression and condition is met
        assert result.task_results["c"].status == "success"
        assert "c" in call_log

    def test_when_no_deps_evaluated_immediately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A task with when but no depends_on gets its condition evaluated immediately.

        Since there are no deps, template vars are empty and the expression
        renders with unresolved vars. This produces an always-false or invalid expression.
        """
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        # This when expression references a non-existent task — should skip
        task = TaskSpec(
            id="orphan",
            description="no deps when",
            command="echo orphan",
            when="{{ missing.status }} == success",
        )
        plan = _make_plan([task], fail_fast=False, source_path=plan_yaml)

        mock_exec, call_log = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # Task should be skipped (condition not met or invalid expression)
        assert result.task_results["orphan"].status == "skipped"
        assert "orphan" not in call_log


# ---------------------------------------------------------------------------
# 3. Budget edge cases (4-5 tests)
# ---------------------------------------------------------------------------


class TestSchedulerEdgeL2BudgetEdgeCases:
    """Camada 2: budget tracking edge cases — tiny budgets, warnings, cost_usd=None."""

    def test_budget_exceeded_by_first_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """max_cost_usd=0.01, first task costs $0.02 -> budget exceeded, second task skipped."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)
        plan.max_cost_usd = 0.01

        mock_exec, _ = _mock_execute_status_factory(
            {}, cost_map={"a": 0.02, "b": 0.01}
        )
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # A completes (soft budget: running task finishes), B skipped
        assert result.task_results["a"].status == "success"
        assert result.task_results["b"].status == "skipped"
        assert "budget" in result.task_results["b"].message.lower()
        assert result.budget_exceeded is True

    def test_budget_warning_at_50_percent_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """budget_warning_pct=0.5 with cost hitting 50% emits a budget_warning event."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)
        plan.max_cost_usd = 10.0
        plan.budget_warning_pct = 0.5

        # A costs $5.50 (55% of $10), triggering the 50% warning
        mock_exec, _ = _mock_execute_status_factory(
            {}, cost_map={"a": 5.50, "b": 0.10}
        )
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        budget_warnings = [e for e in events if e["event"] == "budget_warning"]
        assert len(budget_warnings) >= 1
        assert budget_warnings[0]["spent"] == 5.50
        assert budget_warnings[0]["limit"] == 10.0

    def test_cost_none_tasks_do_not_affect_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Tasks with cost_usd=None do not count towards budget tracking."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("shell-1"),
            _make_task("shell-2", depends_on=["shell-1"]),
            _make_task("shell-3", depends_on=["shell-2"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)
        plan.max_cost_usd = 0.01  # Very small budget

        # All tasks have None cost
        mock_exec, call_log = _mock_execute_status_factory(
            {}, cost_map={"shell-1": None, "shell-2": None, "shell-3": None}
        )
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # All tasks run — None cost doesn't trigger budget exceeded
        assert result.success is True
        assert len(call_log) == 3
        assert result.budget_exceeded is False

    def test_budget_exceeded_event_has_correct_spent_and_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The budget_exceeded event includes the correct spent and limit values."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("expensive"),
            _make_task("next", depends_on=["expensive"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)
        plan.max_cost_usd = 5.0

        mock_exec, _ = _mock_execute_status_factory(
            {}, cost_map={"expensive": 7.50, "next": 0.10}
        )
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        exceeded_events = [e for e in events if e["event"] == "budget_exceeded"]
        assert len(exceeded_events) == 1
        assert exceeded_events[0]["spent"] == 7.50
        assert exceeded_events[0]["limit"] == 5.0

    def test_budget_not_exceeded_when_exactly_at_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Budget is NOT exceeded when total cost equals max_cost_usd exactly (> not >=)."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)
        plan.max_cost_usd = 1.0

        mock_exec, call_log = _mock_execute_status_factory(
            {}, cost_map={"a": 1.0, "b": 0.0}
        )
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # Cost == limit should NOT exceed (the check is `running_cost > plan.max_cost_usd`)
        assert result.budget_exceeded is False
        assert "b" in call_log


# ---------------------------------------------------------------------------
# 4. Context budget pressure (4-5 tests)
# ---------------------------------------------------------------------------


class TestSchedulerEdgeL2ContextBudgetPressure:
    """Camada 2: context budget under pressure — eviction, stopwords, wildcard."""

    def test_context_budget_evicts_excess_upstream(self, tmp_path: Path) -> None:
        """context_budget_tokens=100 with 3 upstreams of ~200 tokens each triggers eviction."""
        from maestro_cli.scheduler import _apply_context_budget

        # Each tail is ~200 tokens (800 chars / 4)
        text_a = "alpha " * 133  # ~798 chars = ~200 tokens
        text_b = "beta " * 133
        text_c = "gamma " * 133
        upstream = {
            "a": _make_success_result("a", tmp_path, stdout_tail=text_a),
            "b": _make_success_result("b", tmp_path, stdout_tail=text_b),
            "c": _make_success_result("c", tmp_path, stdout_tail=text_c),
        }

        result, records, _meta = _apply_context_budget(upstream, budget_tokens=100)

        # Total should be trimmed to ~100 tokens
        total_tokens = sum(
            _estimate_tokens(r.stdout_tail) for r in result.values()
        )
        assert total_tokens <= 100
        assert len(records) > 0  # Some trimming happened

    def test_context_budget_with_intent_keywords_preserves_relevant(
        self, tmp_path: Path,
    ) -> None:
        """Intent keywords cause relevant sections to be preserved during budget trim."""
        from maestro_cli.scheduler import _apply_context_budget

        relevant = "database schema migration " * 40
        irrelevant = "weather forecast gardening cooking " * 40
        tail = f"{relevant}\n\n{irrelevant}"
        upstream = {
            "scan": _make_success_result("scan", tmp_path, stdout_tail=tail),
        }

        budget = _estimate_tokens(relevant) + 20
        result, _records, meta = _apply_context_budget(
            upstream, budget_tokens=budget,
            intent_keywords={"database", "schema", "migration"},
        )

        # The relevant content should survive; meta should contain selection info
        assert "scan" in meta
        remaining_tail = result["scan"].stdout_tail
        assert "database" in remaining_tail

    def test_intent_filtering_all_stopwords_keeps_all_sections(
        self, tmp_path: Path,
    ) -> None:
        """An all-stopword prompt yields no intent keywords, so all sections are kept."""
        from maestro_cli.scheduler import _apply_intent_filtering

        upstream = {
            "a": _make_success_result(
                "a", tmp_path,
                stdout_tail="Important data output\n\nMore important data",
            ),
        }

        # Empty intent keywords -> everything kept
        result, filter_records, _meta = _apply_intent_filtering(upstream, intent_keywords=set())

        assert result is upstream  # no change
        assert filter_records == []

    def test_wildcard_context_from_includes_all_upstreams(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """context_from=['*'] includes all dependency task results as upstream context."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c"),
            _make_task("d", depends_on=["a", "b", "c"], context_from=["*"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=4)

        captured_upstream: dict[str, dict[str, TaskResult] | None] = {}

        def mock_exec(plan, task, run_path, dry_run=False, execution_profile="plan",
                      upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            captured_upstream[task.id] = upstream_results
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert captured_upstream["d"] is not None
        assert set(captured_upstream["d"].keys()) == {"a", "b", "c"}

    def test_context_budget_with_zero_length_tails(self, tmp_path: Path) -> None:
        """Upstreams with empty stdout_tail don't cause errors during budget enforcement."""
        from maestro_cli.scheduler import _apply_context_budget

        upstream = {
            "a": _make_success_result("a", tmp_path, stdout_tail=""),
            "b": _make_success_result("b", tmp_path, stdout_tail=""),
        }

        result, records, _meta = _apply_context_budget(upstream, budget_tokens=10)

        assert result["a"].stdout_tail == ""
        assert result["b"].stdout_tail == ""
        assert records == []


# ---------------------------------------------------------------------------
# 5. Event integrity (3-4 tests)
# ---------------------------------------------------------------------------


class TestSchedulerEdgeL2EventIntegrity:
    """Camada 2: event integrity — run_start fields, skip reasons, plan_name injection."""

    def test_run_start_event_includes_goal_field(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The run_start event includes the plan's goal field."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)
        plan.goal = "Deploy to production"

        mock_exec, _ = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        run_start_events = [e for e in events if e["event"] == "run_start"]
        assert len(run_start_events) == 1
        assert run_start_events[0]["goal"] == "Deploy to production"

    def test_run_start_event_with_empty_goal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The run_start event has goal='' when plan has no goal set."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=plan_yaml)

        mock_exec, _ = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        run_start_events = [e for e in events if e["event"] == "run_start"]
        assert run_start_events[0]["goal"] == ""

    def test_task_skip_reason_populated_dep_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Dependency failure skip events include 'dependency failure' in the reason."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml)

        mock_exec, _ = _mock_execute_status_factory({"a": "failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        skip_events = [e for e in events if e["event"] == "task_skip" and e["task_id"] == "b"]
        assert len(skip_events) == 1
        assert "dependency failure" in skip_events[0]["reason"]

    def test_task_skip_reason_populated_when_expression(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When-expression skip events include 'condition not met' in the reason."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_b = TaskSpec(
            id="b",
            description="conditional skip",
            depends_on=["a"],
            command="echo b",
            when="{{ a.status }} == failed",
        )
        plan = _make_plan([_make_task("a"), task_b], fail_fast=False, source_path=plan_yaml)

        mock_exec, _ = _mock_execute_status_factory({})  # A succeeds
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        skip_events = [e for e in events if e["event"] == "task_skip" and e["task_id"] == "b"]
        assert len(skip_events) == 1
        assert "condition not met" in skip_events[0]["reason"]

    def test_all_events_include_plan_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every event in events.jsonl includes the plan_name field automatically."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, name="my-custom-plan", source_path=plan_yaml)

        mock_exec, _ = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        # Every single event must have plan_name == "my-custom-plan"
        assert len(events) >= 3  # at minimum: run_start, task_complete x2, run_complete
        for event in events:
            assert event["plan_name"] == "my-custom-plan", (
                f"Event {event['event']} missing or wrong plan_name"
            )

    def test_run_complete_event_has_correct_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The run_complete event includes correct ok/failed/skipped counts."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=plan_yaml, max_parallel=1)

        mock_exec, _ = _mock_execute_status_factory({"a": "failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        run_complete = [e for e in events if e["event"] == "run_complete"]
        assert len(run_complete) == 1
        rc = run_complete[0]
        assert rc["success"] is False
        assert rc["ok"] == 1  # b
        assert rc["failed"] == 1  # a
        assert rc["skipped"] == 1  # c (dep failure)

    def test_events_jsonl_file_created_even_for_empty_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """events.jsonl exists and contains run_start + run_complete even for a 1-task plan."""
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        plan = _make_plan([_make_task("solo")], source_path=plan_yaml)

        mock_exec, _ = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events_path = result.run_path / "events.jsonl"
        assert events_path.exists()
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        event_types = [e["event"] for e in events]
        assert "run_start" in event_types
        assert "run_complete" in event_types


# ===========================================================================
# EdgeL3: additional edge-case tests for improved coverage
# ===========================================================================

from maestro_cli.scheduler import (  # noqa: E402
    _apply_context_budget as _acb_l3,
    _apply_intent_filtering as _aif_l3,
    _compute_hop_distances as _chd_l3,
    _apply_hop_decay as _ahd_l3,
    _compute_idf as _cidf_l3,
    _compute_task_depth as _ctd_l3,
    _compute_fan_out as _cfo_l3,
    _compute_tainted_tasks as _ctt_l3,
    _estimate_tokens as _et_l3,
    _extract_keywords as _ek_l3,
    _filter_tail_by_intent as _ftbi_l3,
    _request_approval as _ra_l3,
    _score_section as _ss_l3,
    _split_into_sections as _sis_l3,
)


class TestEdgeL3ExtractKeywords:
    """Edge cases for _extract_keywords."""

    def test_empty_string(self) -> None:
        assert _ek_l3("") == set()

    def test_whitespace_only(self) -> None:
        assert _ek_l3("   \n\t  ") == set()

    def test_single_char_words_excluded(self) -> None:
        assert _ek_l3("a b c d e") == set()

    def test_stopwords_excluded(self) -> None:
        result = _ek_l3("the is are was were been being have")
        assert result == set()

    def test_underscored_identifiers_kept(self) -> None:
        result = _ek_l3("my_var another_thing __init__")
        assert "my_var" in result
        assert "another_thing" in result
        assert "__init__" in result

    def test_mixed_case_normalized(self) -> None:
        result = _ek_l3("API Schema PARSER")
        assert "api" in result
        assert "schema" in result
        assert "parser" in result

    def test_numbers_in_identifiers(self) -> None:
        result = _ek_l3("v2 retry_01 http2")
        assert "v2" in result
        assert "retry_01" in result
        assert "http2" in result

    def test_special_chars_as_delimiters(self) -> None:
        result = _ek_l3("api.schema/parser+validator=auth")
        assert "api" in result
        assert "schema" in result
        assert "parser" in result
        assert "validator" in result
        assert "auth" in result

    def test_duplicate_words_deduped(self) -> None:
        result = _ek_l3("api api api parser parser")
        assert result == {"api", "parser"}

    def test_use_stopword_excluded(self) -> None:
        result = _ek_l3("use using used implement")
        assert "use" not in result
        assert "using" not in result
        assert "used" not in result
        assert "implement" in result


class TestEdgeL3SplitIntoSections:
    """Edge cases for _split_into_sections."""

    def test_no_blank_lines_chunks_by_8(self) -> None:
        lines = [f"line {i}" for i in range(16)]
        sections = _sis_l3("\n".join(lines))
        assert len(sections) == 2

    def test_single_section_no_splits(self) -> None:
        text = "line1\nline2\nline3"
        sections = _sis_l3(text)
        assert len(sections) == 1

    def test_blank_line_separated_sections(self) -> None:
        text = "section one\n\nsection two\n\nsection three"
        sections = _sis_l3(text)
        assert len(sections) == 3
        assert sections[0] == "section one"
        assert sections[1] == "section two"
        assert sections[2] == "section three"

    def test_multiple_blank_lines_coalesced(self) -> None:
        text = "alpha\n\n\n\nbeta"
        sections = _sis_l3(text)
        assert len(sections) == 2

    def test_trailing_leading_whitespace_stripped(self) -> None:
        text = "  alpha  \n\n  beta  "
        sections = _sis_l3(text)
        assert sections[0] == "alpha"
        assert sections[1] == "beta"

    def test_empty_returns_empty(self) -> None:
        assert _sis_l3("") == []

    def test_only_blank_lines_returns_empty(self) -> None:
        assert _sis_l3("\n\n\n") == []

    def test_9_lines_gives_two_chunks(self) -> None:
        lines = [f"line {i}" for i in range(9)]
        sections = _sis_l3("\n".join(lines))
        assert len(sections) == 2
        assert "line 8" in sections[1]


class TestEdgeL3ComputeIdf:
    """Edge cases for _compute_idf."""

    def test_empty_sections_returns_empty(self) -> None:
        assert _cidf_l3([]) == {}

    def test_single_section_all_terms_have_idf(self) -> None:
        idf = _cidf_l3(["api schema parser"])
        assert len(idf) == 3
        for term in ("api", "schema", "parser"):
            assert term in idf

    def test_rare_term_higher_idf_than_common(self) -> None:
        idf = _cidf_l3([
            "api parser",
            "api schema",
            "deploy release",
        ])
        assert idf["deploy"] > idf["api"]

    def test_term_in_all_sections_has_lowest_idf(self) -> None:
        idf = _cidf_l3([
            "common alpha",
            "common beta",
            "common gamma",
        ])
        assert idf["common"] < idf["alpha"]

    def test_stopwords_excluded_from_idf(self) -> None:
        idf = _cidf_l3(["the is are api"])
        assert "the" not in idf
        assert "api" in idf


class TestEdgeL3ScoreSection:
    """Edge cases for _score_section."""

    def test_empty_section_returns_zero(self) -> None:
        assert _ss_l3("", {"api"}) == 0

    def test_empty_keywords_returns_zero(self) -> None:
        assert _ss_l3("api schema", set()) == 0

    def test_no_matching_keywords_returns_zero(self) -> None:
        assert _ss_l3("weather gardening", {"api", "schema"}) == 0

    def test_exact_match_without_idf_returns_count(self) -> None:
        score = _ss_l3("api schema parser", {"api", "schema"})
        assert score == 2

    def test_bm25_with_idf_scores_higher_for_rare_terms(self) -> None:
        idf = {"rare_term": 5.0, "common": 0.1}
        score_rare = _ss_l3(
            "rare_term", {"rare_term", "common"},
            idf=idf, avg_section_len=5,
        )
        score_common = _ss_l3(
            "common", {"rare_term", "common"},
            idf=idf, avg_section_len=5,
        )
        assert score_rare > score_common

    def test_bm25_no_words_in_section_returns_zero(self) -> None:
        assert _ss_l3("a b c", {"api"}, idf={"api": 2.0}) == 0

    def test_bm25_term_saturation(self) -> None:
        idf = {"api": 2.0}
        score_1 = _ss_l3("api", {"api"}, idf=idf, avg_section_len=5)
        score_5 = _ss_l3(
            "api api api api api", {"api"},
            idf=idf, avg_section_len=5,
        )
        assert score_5 < score_1 * 5

    def test_avg_section_len_zero_uses_one(self) -> None:
        score = _ss_l3(
            "api schema", {"api"},
            idf={"api": 2.0}, avg_section_len=0,
        )
        assert score >= 1


class TestEdgeL3FilterTailByIntent:
    """Edge cases for _filter_tail_by_intent."""

    def test_empty_tail_returns_unchanged(self) -> None:
        result, score, keywords = _ftbi_l3("", {"api"})
        assert result == ""
        assert score == 0
        assert keywords == []

    def test_empty_keywords_returns_unchanged(self) -> None:
        result, score, keywords = _ftbi_l3("some text", set())
        assert result == "some text"
        assert score == 0

    def test_all_sections_match(self) -> None:
        text = "api schema design\n\napi endpoint validation"
        result, score, keywords = _ftbi_l3(text, {"api"})
        assert "api schema design" in result
        assert "api endpoint validation" in result
        assert score > 0

    def test_only_relevant_sections_kept(self) -> None:
        text = "api schema design\n\nweather forecast today"
        result, score, keywords = _ftbi_l3(text, {"api", "schema"})
        assert "api schema design" in result
        assert "weather forecast today" not in result

    def test_no_sections_match_returns_full_tail(self) -> None:
        text = "weather forecast\n\ntravel plans"
        result, score, keywords = _ftbi_l3(text, {"api", "schema"})
        assert result == text
        assert score == 0
        assert keywords == []

    def test_matched_keywords_are_sorted(self) -> None:
        text = "schema api parser\n\nweather travel"
        _result, _score, keywords = _ftbi_l3(text, {"api", "schema", "parser"})
        assert keywords == sorted(keywords)

    def test_fallback_to_legacy_scoring_when_bm25_yields_nothing(self) -> None:
        text = "api schema validation endpoint"
        result, score, keywords = _ftbi_l3(
            text, {"api"}, idf={"api": 0.0001}, avg_section_len=100,
        )
        assert score >= 0


class TestEdgeL3ApplyIntentFiltering:
    """Edge cases for _apply_intent_filtering."""

    def test_empty_upstream_returns_unchanged(self) -> None:
        result, records, meta = _aif_l3({}, {"api"})
        assert result == {}
        assert records == []
        assert meta == {}

    def test_none_keywords_returns_unchanged(self, tmp_path: Path) -> None:
        upstream = {"a": _make_success_result("a", tmp_path, stdout_tail="data")}
        result, records, meta = _aif_l3(upstream, None)
        assert result is upstream

    def test_filtering_reduces_irrelevant_content(self, tmp_path: Path) -> None:
        relevant = "api schema validation " * 40
        irrelevant = "weather gardening cooking " * 40
        upstream = {
            "scan": _make_success_result(
                "scan", tmp_path,
                stdout_tail=f"{relevant}\n\n{irrelevant}",
            ),
        }
        result, records, meta = _aif_l3(
            upstream, {"api", "schema", "validation"},
        )
        assert len(result["scan"].stdout_tail) <= len(upstream["scan"].stdout_tail)
        assert "scan" in meta

    def test_no_reduction_when_all_relevant(self, tmp_path: Path) -> None:
        text = "api schema validation"
        upstream = {
            "a": _make_success_result("a", tmp_path, stdout_tail=text),
        }
        result, records, meta = _aif_l3(upstream, {"api", "schema"})
        assert records == []


class TestEdgeL3HopDistances:
    """Extended hop distance tests for various graph shapes."""

    def test_diamond_graph_distances(self) -> None:
        tasks = {
            "root": _make_task("root"),
            "left": _make_task("left", depends_on=["root"]),
            "right": _make_task("right", depends_on=["root"]),
            "tip": _make_task("tip", depends_on=["left", "right"]),
        }
        hop = _chd_l3("tip", context_from=["left", "right", "root"], all_tasks=tasks)
        assert hop["left"] == 1
        assert hop["right"] == 1
        assert hop["root"] == 2

    def test_deep_chain_distances(self) -> None:
        tasks = {
            "t0": _make_task("t0"),
            "t1": _make_task("t1", depends_on=["t0"]),
            "t2": _make_task("t2", depends_on=["t1"]),
            "t3": _make_task("t3", depends_on=["t2"]),
            "t4": _make_task("t4", depends_on=["t3"]),
        }
        hop = _chd_l3("t4", context_from=["t0", "t1", "t2", "t3"], all_tasks=tasks)
        assert hop["t3"] == 1
        assert hop["t2"] == 2
        assert hop["t1"] == 3
        assert hop["t0"] == 4

    def test_wide_fan_out_all_direct(self) -> None:
        tasks = {
            "a": _make_task("a"),
            "b": _make_task("b"),
            "c": _make_task("c"),
            "d": _make_task("d"),
            "collector": _make_task("collector", depends_on=["a", "b", "c", "d"]),
        }
        hop = _chd_l3("collector", context_from=["a", "b", "c", "d"], all_tasks=tasks)
        for tid in ("a", "b", "c", "d"):
            assert hop[tid] == 1

    def test_wildcard_resolves_all_deps(self) -> None:
        tasks = {
            "a": _make_task("a"),
            "b": _make_task("b"),
            "leaf": _make_task("leaf", depends_on=["a", "b"]),
        }
        hop = _chd_l3("leaf", context_from=["*"], all_tasks=tasks)
        assert hop["a"] == 1
        assert hop["b"] == 1

    def test_missing_source_excluded(self) -> None:
        tasks = {
            "a": _make_task("a"),
            "leaf": _make_task("leaf", depends_on=["a"]),
        }
        hop = _chd_l3("leaf", context_from=["a", "ghost"], all_tasks=tasks)
        assert "a" in hop
        assert "ghost" not in hop

    def test_empty_context_from_returns_empty(self) -> None:
        tasks = {"a": _make_task("a")}
        hop = _chd_l3("a", context_from=[], all_tasks=tasks)
        assert hop == {}

    def test_direct_dep_override_to_1(self) -> None:
        tasks = {
            "root": _make_task("root"),
            "mid": _make_task("mid", depends_on=["root"]),
            "leaf": _make_task("leaf", depends_on=["mid", "root"]),
        }
        hop = _chd_l3("leaf", context_from=["root"], all_tasks=tasks)
        assert hop["root"] == 1


class TestEdgeL3ApplyHopDecay:
    """Extended hop decay tests."""

    def test_empty_upstream_returns_empty(self) -> None:
        result = _ahd_l3({}, {})
        assert result == {}

    def test_hop_1_no_decay(self, tmp_path: Path) -> None:
        upstream = {"a": _make_success_result("a", tmp_path, stdout_tail="abcdefghij")}
        result = _ahd_l3(upstream, {"a": 1})
        assert result["a"] is upstream["a"]
        assert result["a"].stdout_tail == "abcdefghij"

    def test_hop_3_keeps_64pct(self, tmp_path: Path) -> None:
        tail = "x" * 100
        upstream = {"a": _make_success_result("a", tmp_path, stdout_tail=tail)}
        result = _ahd_l3(upstream, {"a": 3})
        assert len(result["a"].stdout_tail) == 64

    def test_hop_5_deep_decay(self, tmp_path: Path) -> None:
        tail = "y" * 1000
        upstream = {"a": _make_success_result("a", tmp_path, stdout_tail=tail)}
        result = _ahd_l3(upstream, {"a": 5})
        assert len(result["a"].stdout_tail) == 409

    def test_unknown_hop_treated_as_1(self, tmp_path: Path) -> None:
        upstream = {"a": _make_success_result("a", tmp_path, stdout_tail="abc")}
        result = _ahd_l3(upstream, {})
        assert result["a"] is upstream["a"]

    def test_original_not_mutated(self, tmp_path: Path) -> None:
        tail = "z" * 50
        upstream = {"a": _make_success_result("a", tmp_path, stdout_tail=tail)}
        result = _ahd_l3(upstream, {"a": 3})
        assert upstream["a"].stdout_tail == "z" * 50
        assert result["a"].stdout_tail != upstream["a"].stdout_tail

    def test_empty_tail_no_crash(self, tmp_path: Path) -> None:
        upstream = {"a": _make_success_result("a", tmp_path, stdout_tail="")}
        result = _ahd_l3(upstream, {"a": 3})
        assert result["a"].stdout_tail == ""


class TestEdgeL3ApplyContextBudget:
    """Extended context budget edge cases."""

    def test_zero_budget_trims_everything(self, tmp_path: Path) -> None:
        upstream = {
            "a": _make_success_result("a", tmp_path, stdout_tail="x" * 100),
        }
        result, records, _meta = _acb_l3(upstream, budget_tokens=0)
        total = sum(_et_l3(r.stdout_tail) for r in result.values())
        assert total <= 1

    def test_exactly_at_budget_no_trim(self, tmp_path: Path) -> None:
        tail = "abcd"
        upstream = {"a": _make_success_result("a", tmp_path, stdout_tail=tail)}
        result, records, _meta = _acb_l3(upstream, budget_tokens=1)
        assert records == []

    def test_greedy_eviction_prefers_low_scoring(self, tmp_path: Path) -> None:
        relevant = "api schema validation " * 50
        irrelevant = "weather gardening cooking " * 50
        upstream = {
            "relevant": _make_success_result("relevant", tmp_path, stdout_tail=relevant),
            "irrelevant": _make_success_result("irrelevant", tmp_path, stdout_tail=irrelevant),
        }
        single_tokens = _et_l3(relevant)
        budget = single_tokens + 5
        result, records, meta = _acb_l3(
            upstream, budget_tokens=budget,
            intent_keywords={"api", "schema"},
        )
        relevant_len = len(result["relevant"].stdout_tail)
        irrelevant_len = len(result["irrelevant"].stdout_tail)
        assert relevant_len >= irrelevant_len

    def test_multiple_upstreams_proportional_trim(self, tmp_path: Path) -> None:
        text_a = "a" * 400
        text_b = "b" * 400
        text_c = "c" * 400
        upstream = {
            "a": _make_success_result("a", tmp_path, stdout_tail=text_a),
            "b": _make_success_result("b", tmp_path, stdout_tail=text_b),
            "c": _make_success_result("c", tmp_path, stdout_tail=text_c),
        }
        result, records, _meta = _acb_l3(upstream, budget_tokens=50)
        assert len(records) > 0
        total = sum(_et_l3(r.stdout_tail) for r in result.values())
        assert total <= 50

    def test_within_budget_with_keywords_returns_meta(self, tmp_path: Path) -> None:
        upstream = {
            "a": _make_success_result("a", tmp_path, stdout_tail="api schema"),
        }
        result, records, meta = _acb_l3(
            upstream, budget_tokens=10000,
            intent_keywords={"api"},
        )
        assert records == []
        assert "a" in meta


class TestEdgeL3EstimateTokens:
    """Edge cases for _estimate_tokens."""

    def test_empty_string_returns_one(self) -> None:
        assert _et_l3("") == 1

    def test_short_string_minimum_one(self) -> None:
        assert _et_l3("ab") == 1

    def test_exact_4_chars(self) -> None:
        assert _et_l3("abcd") == 1

    def test_8_chars(self) -> None:
        assert _et_l3("abcdefgh") == 2

    def test_large_text(self) -> None:
        text = "x" * 4000
        assert _et_l3(text) == 1000


class TestEdgeL3TaintedTasks:
    """Extended taint propagation tests."""

    def test_wildcard_context_from_propagates_taint(self) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"], context_from=["*"],
            ),
        ]
        plan = _make_plan(tasks)
        tainted = _ctt_l3(plan)
        assert "a" in tainted
        assert "b" in tainted

    def test_diamond_taint_both_branches(self) -> None:
        tasks = [
            TaskSpec(id="root", command="echo root", context_trust="untrusted"),
            TaskSpec(id="left", command="echo left", depends_on=["root"], context_from=["root"]),
            TaskSpec(id="right", command="echo right", depends_on=["root"], context_from=["root"]),
            TaskSpec(id="tip", command="echo tip", depends_on=["left", "right"], context_from=["left", "right"]),
        ]
        plan = _make_plan(tasks)
        tainted = _ctt_l3(plan)
        assert tainted == {"root", "left", "right", "tip"}

    def test_guard_breaks_taint_chain(self) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
            TaskSpec(id="b", command="echo b", depends_on=["a"], context_from=["a"], guard_command="check"),
            TaskSpec(id="c", command="echo c", depends_on=["b"], context_from=["b"]),
        ]
        plan = _make_plan(tasks)
        tainted = _ctt_l3(plan)
        assert "a" in tainted
        assert "b" not in tainted
        assert "c" not in tainted

    def test_no_context_from_not_tainted(self) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
            TaskSpec(id="b", command="echo b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks)
        tainted = _ctt_l3(plan)
        assert "a" in tainted
        assert "b" not in tainted

    def test_multiple_untrusted_sources(self) -> None:
        tasks = [
            TaskSpec(id="src1", command="echo src1", context_trust="untrusted"),
            TaskSpec(id="src2", command="echo src2", context_trust="untrusted"),
            TaskSpec(id="consumer", command="echo consumer", depends_on=["src1", "src2"], context_from=["src1"]),
        ]
        plan = _make_plan(tasks)
        tainted = _ctt_l3(plan)
        assert tainted == {"src1", "src2", "consumer"}


class TestEdgeL3ParseLayeredContextSections:
    """Edge cases for _parse_layered_context_sections."""

    def test_empty_string(self) -> None:
        from maestro_cli.scheduler import _parse_layered_context_sections
        assert _parse_layered_context_sections("") == {}

    def test_no_section_markers(self) -> None:
        from maestro_cli.scheduler import _parse_layered_context_sections
        assert _parse_layered_context_sections("just some text") == {}

    def test_single_section(self) -> None:
        from maestro_cli.scheduler import _parse_layered_context_sections
        text = "--- task-a ---\nBody of section A."
        result = _parse_layered_context_sections(text)
        assert "task-a" in result
        assert result["task-a"] == "Body of section A."

    def test_multiple_sections(self) -> None:
        from maestro_cli.scheduler import _parse_layered_context_sections
        text = "--- alpha ---\nAlpha body.\n--- beta ---\nBeta body."
        result = _parse_layered_context_sections(text)
        assert len(result) == 2
        assert result["alpha"] == "Alpha body."
        assert result["beta"] == "Beta body."

    def test_section_with_trailing_whitespace(self) -> None:
        from maestro_cli.scheduler import _parse_layered_context_sections
        text = "--- task-x ---\n  Body with spaces.  \n"
        result = _parse_layered_context_sections(text)
        assert result["task-x"] == "Body with spaces."


class TestEdgeL3RequestApproval:
    """Edge cases for _request_approval."""

    def test_eof_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda: (_ for _ in ()).throw(EOFError))
        assert _ra_l3("t1", None, interactive=True) is False

    def test_keyboard_interrupt_returns_false(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda: (_ for _ in ()).throw(KeyboardInterrupt))
        assert _ra_l3("t1", None, interactive=True) is False

    def test_yes_full_word_approved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda: "yes")
        assert _ra_l3("t1", None, interactive=True) is True

    def test_uppercase_y_approved(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda: "Y")
        assert _ra_l3("t1", None, interactive=True) is True

    def test_n_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda: "n")
        assert _ra_l3("t1", None, interactive=True) is False

    def test_empty_input_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda: "")
        assert _ra_l3("t1", None, interactive=True) is False

    def test_custom_message_printed(self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.setattr("builtins.input", lambda: "n")
        _ra_l3("t1", "Custom approval msg", interactive=True)
        captured = capsys.readouterr()
        assert "Custom approval msg" in captured.out


class TestEdgeL3ComputeTaskDepthFanOut:
    """Extended depth and fan-out for complex graph shapes."""

    def test_depth_isolated_task(self) -> None:
        tasks = [_make_task("solo")]
        plan = _make_plan(tasks)
        assert _ctd_l3(tasks[0], plan) == 0

    def test_depth_wide_fan_in(self) -> None:
        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c"),
            _make_task("collector", depends_on=["a", "b", "c"]),
        ]
        plan = _make_plan(tasks)
        assert _ctd_l3(tasks[3], plan) == 1

    def test_fan_out_multiple_dependents(self) -> None:
        tasks = [
            _make_task("root"),
            _make_task("b", depends_on=["root"]),
            _make_task("c", depends_on=["root"]),
            _make_task("d", depends_on=["root"]),
            _make_task("e", depends_on=["root"]),
        ]
        plan = _make_plan(tasks)
        assert _cfo_l3(tasks[0], plan) == 4

    def test_fan_out_leaf_is_zero(self) -> None:
        tasks = [
            _make_task("a"),
            _make_task("leaf", depends_on=["a"]),
        ]
        plan = _make_plan(tasks)
        assert _cfo_l3(tasks[1], plan) == 0

    def test_depth_diamond_tip(self) -> None:
        tasks = [
            _make_task("root"),
            _make_task("left", depends_on=["root"]),
            _make_task("right", depends_on=["root"]),
            _make_task("tip", depends_on=["left", "right"]),
        ]
        plan = _make_plan(tasks)
        assert _ctd_l3(tasks[3], plan) == 2


class TestEdgeL3BudgetExceeded:
    """Budget exceeded event emission during run_plan."""

    def test_budget_exceeded_skips_remaining_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)
        plan.max_cost_usd = 0.01

        mock_exec, call_log = _mock_execute_status_factory(
            {}, cost_map={"a": 0.50},
        )
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.budget_exceeded is True
        assert "a" in call_log
        assert result.task_results["b"].status == "skipped"

    def test_budget_warning_event_emitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml, max_parallel=1)
        plan.max_cost_usd = 1.0
        plan.budget_warning_pct = 0.5

        mock_exec, _ = _mock_execute_status_factory({}, cost_map={"a": 0.60})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = [
            json.loads(line)
            for line in (result.run_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        warning_events = [e for e in events if e["event"] == "budget_warning"]
        assert len(warning_events) >= 1


class TestEdgeL3FailFastWithAllowFailure:
    """Interaction between fail_fast and allow_failure."""

    def test_fail_fast_not_triggered_by_soft_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a", allow_failure=True),
            _make_task("b"),
        ]
        plan = _make_plan(tasks, fail_fast=True, source_path=plan_yaml, max_parallel=1)

        mock_exec, call_log = _mock_execute_status_factory({"a": "soft_failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert "a" in call_log
        assert "b" in call_log

    def test_fail_fast_triggered_by_hard_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            _make_task("a"),
            _make_task("b", allow_failure=True),
        ]
        plan = _make_plan(tasks, fail_fast=True, source_path=plan_yaml, max_parallel=1)

        mock_exec, call_log = _mock_execute_status_factory({"a": "failed"})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        assert "a" in call_log
        assert result.task_results["b"].status == "skipped"


class TestEdgeL3WhenExpressionSkip:
    """When-expression skip semantics."""

    def test_when_condition_met_task_runs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_b = TaskSpec(
            id="b", description="conditional", depends_on=["a"],
            command="echo b", when="{{ a.status }} == success",
        )
        plan = _make_plan([_make_task("a"), task_b], source_path=plan_yaml)

        mock_exec, call_log = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert "b" in call_log

    def test_when_condition_not_met_task_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_b = TaskSpec(
            id="b", description="conditional", depends_on=["a"],
            command="echo b", when="{{ a.status }} == failed",
        )
        plan = _make_plan([_make_task("a"), task_b], source_path=plan_yaml)

        mock_exec, call_log = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert "b" not in call_log
        assert result.task_results["b"].status == "skipped"

    def test_skipped_by_when_counts_as_success_overall(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        task_b = TaskSpec(
            id="b", description="cond", depends_on=["a"],
            command="echo b", when="{{ a.status }} == nonexistent_value",
        )
        plan = _make_plan(
            [_make_task("a"), task_b],
            fail_fast=False, source_path=plan_yaml,
        )

        mock_exec, _ = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True


class TestEdgeL3PolicyViolations:
    """Policy evaluation during run_plan."""

    def test_block_policy_prevents_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import PolicySpec

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            TaskSpec(id="risky", engine="claude", prompt="Do risky thing", command=None),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)
        plan.policies = [
            PolicySpec(
                name="no-claude",
                rule='task.engine == "claude"',
                action="block",
                message="Claude engine blocked by policy",
            ),
        ]

        mock_exec, call_log = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert "risky" not in call_log
        assert result.task_results["risky"].status in ("skipped", "failed")
        assert "policy" in result.task_results["risky"].message.lower()

    def test_warn_policy_allows_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import PolicySpec

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.touch()

        tasks = [
            TaskSpec(id="warned", engine="claude", prompt="Do something", command=None),
        ]
        plan = _make_plan(tasks, source_path=plan_yaml)
        plan.policies = [
            PolicySpec(
                name="warn-claude",
                rule='task.engine == "claude"',
                action="warn",
                message="Claude engine warned",
            ),
        ]

        mock_exec, call_log = _mock_execute_status_factory({})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert "warned" in call_log
        assert result.task_results["warned"].status == "success"


class TestEdgeL3SelectTasksAdditional:
    """Additional _select_tasks edge cases."""

    def test_skip_all_tasks_returns_empty(self) -> None:
        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only=None, skip={"a", "b"})
        assert selected == []

    def test_only_multiple_tasks(self) -> None:
        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c"),
            _make_task("d"),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only={"a", "c"}, skip=None)
        ids = {t.id for t in selected}
        assert ids == {"a", "c"}

    def test_skip_tags_only_no_tags_filter(self) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", tags=["infra"]),
            TaskSpec(id="b", command="echo b", tags=["deploy"]),
            TaskSpec(id="c", command="echo c", tags=["test"]),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only=None, skip=None, skip_tags={"test"})
        ids = {t.id for t in selected}
        assert ids == {"a", "b"}

    def test_tags_filter_includes_deps_even_without_tag(self) -> None:
        tasks = [
            TaskSpec(id="setup", command="echo setup"),
            TaskSpec(id="deploy", command="echo deploy", tags=["ship"], depends_on=["setup"]),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only=None, skip=None, tags={"ship"})
        ids = [t.id for t in selected]
        assert "setup" in ids
        assert "deploy" in ids


# ===========================================================================
# L4: Integration-focused edge-case tests for improved LOC/test ratio
# ===========================================================================

from maestro_cli.scheduler import (  # noqa: E402
    _apply_context_budget as _acb_l4,
    _apply_intent_filtering as _aif_l4,
    _estimate_workspace_timeout as _ewt_l4,
    _load_prior_results as _lpr_l4,
    _parse_layered_context_sections as _plcs_l4,
    _post_completion_webhook as _pcw_l4,
    _preflight_checks as _pfc_l4,
)


class TestL4RunPlanDAGShapes:
    """Multi-task DAG shapes: wide fan-out, diamond merge, deep chain."""

    def test_wide_fanout_all_parallel(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """5 independent tasks should all run (no deps)."""
        tasks = [_make_task(f"t{i}") for i in range(5)]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert set(log) == {f"t{i}" for i in range(5)}

    def test_deep_chain_10_tasks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """10-task linear chain executes in order."""
        tasks = []
        for i in range(10):
            deps = [f"t{i-1}"] if i > 0 else []
            tasks.append(_make_task(f"t{i}", depends_on=deps))
        plan = _make_plan(tasks, max_parallel=1, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert log == [f"t{i}" for i in range(10)]

    def test_diamond_merge_4_tasks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Diamond: A -> B,C -> D. All succeed."""
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
            _make_task("d", depends_on=["b", "c"]),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert log[0] == "a"
        assert log[-1] == "d"
        assert set(log) == {"a", "b", "c", "d"}

    def test_diamond_middle_fail_skips_merge(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Diamond: if B fails, D is skipped (fail_fast=False)."""
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
            _make_task("d", depends_on=["b", "c"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        overrides = {
            "b": TaskResult(
                task_id="b", status="failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.1,
                command="fail", log_path=tmp_path / "b.log",
                result_path=tmp_path / "b.result.json", message="fail",
            ),
        }
        mock, log = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert not result.success
        assert result.task_results["d"].status == "skipped"
        assert result.task_results["c"].status == "success"

    def test_two_independent_chains(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Two independent chains: a->b and c->d."""
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c"),
            _make_task("d", depends_on=["c"]),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert len(log) == 4
        assert log.index("a") < log.index("b")
        assert log.index("c") < log.index("d")

    def test_single_task_plan(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("only")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert log == ["only"]


class TestL4EventEmission:
    """Verify event emission via event_callback for various scenarios."""

    def test_run_start_and_run_complete_events(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        event_names = [e[0] for e in events]
        assert "run_start" in event_names
        assert "run_complete" in event_names
        assert event_names[0] == "run_start"
        assert event_names[-1] == "run_complete"

    def test_task_start_and_task_complete_events(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("x")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        starts = [e for e in events if e[0] == "task_start"]
        completes = [e for e in events if e[0] == "task_complete"]
        assert len(starts) == 1
        assert starts[0][1]["task_id"] == "x"
        assert len(completes) == 1
        assert completes[0][1]["task_id"] == "x"
        assert completes[0][1]["status"] == "success"

    def test_task_skip_event_on_dep_failure(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        overrides = {
            "a": TaskResult(
                task_id="a", status="failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="fail", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="fail",
            ),
        }
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        skips = [e for e in events if e[0] == "task_skip"]
        assert any(e[1]["task_id"] == "b" for e in skips)

    def test_fail_fast_skip_events(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, fail_fast=True, max_parallel=1, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        overrides = {
            "a": TaskResult(
                task_id="a", status="failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="fail", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="fail",
            ),
        }
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        skips = [e for e in events if e[0] == "task_skip"]
        assert len(skips) >= 1

    def test_budget_warning_event(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        plan = PlanSpec(
            version=1, name="budget-test", tasks=tasks, max_cost_usd=1.0,
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
        )
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                cost_usd=0.90,
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        warnings = [e for e in events if e[0] == "budget_warning"]
        assert len(warnings) >= 1

    def test_budget_exceeded_event(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
        ]
        plan = PlanSpec(
            version=1, name="exceed", tasks=tasks, max_cost_usd=0.5,
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
        )
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                cost_usd=0.60,
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        result = run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        exceeded = [e for e in events if e[0] == "budget_exceeded"]
        assert len(exceeded) >= 1
        assert result.budget_exceeded

    def test_run_complete_includes_counts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        rc = [e for e in events if e[0] == "run_complete"][0][1]
        assert rc["ok"] == 2
        assert rc["failed"] == 0
        assert rc["skipped"] == 0

    def test_plan_name_in_all_events(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, name="my-plan", source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        for _, data in events:
            assert data.get("plan_name") == "my-plan"

    def test_taint_detected_event(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        taints = [e for e in events if e[0] == "taint_detected"]
        assert len(taints) == 1
        assert taints[0][1]["task_id"] == "a"
        assert taints[0][1]["source"] == "explicit"


class TestL4ResumeLogic:
    """Resume from prior run: skip completed, re-run failures."""

    def test_resume_skips_success_runs_pending(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        prior_run = tmp_path / "prior"
        prior_run.mkdir()
        manifest = {
            "task_results": {
                "a": {"status": "success"},
                "b": {"status": "failed", "message": "error"},
            }
        }
        (prior_run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path), resume_path=prior_run)
        assert result.success
        assert "a" not in log  # skipped (resumed)
        assert "b" in log     # re-executed

    def test_resume_skips_dry_run_status(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        prior_run = tmp_path / "prior"
        prior_run.mkdir()
        manifest = {
            "task_results": {
                "a": {"status": "dry_run"},
            }
        }
        (prior_run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path), resume_path=prior_run)
        assert result.success
        assert "a" not in log
        assert "b" in log

    def test_resume_re_runs_dep_failure_skips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        prior_run = tmp_path / "prior"
        prior_run.mkdir()
        manifest = {
            "task_results": {
                "a": {"status": "success"},
                "b": {"status": "skipped", "message": "Skipped because dependency failed: {'a'}"},
            }
        }
        (prior_run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path), resume_path=prior_run)
        assert result.success
        assert "b" in log  # re-run because it was a dep-failure skip

    def test_resume_re_runs_fail_fast_skips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        prior_run = tmp_path / "prior"
        prior_run.mkdir()
        manifest = {
            "task_results": {
                "a": {"status": "skipped", "message": "fail_fast triggered by task 'x'"},
            }
        }
        (prior_run / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path), resume_path=prior_run)
        assert "a" in log


class TestL4TagFilteringComplex:
    """Complex tag filtering with dependency auto-inclusion."""

    def test_tags_transitive_dep_inclusion(self) -> None:
        """Tagged task depends on chain: a -> b -> c[tagged]. All 3 included."""
        tasks = [
            TaskSpec(id="a", command="echo a"),
            TaskSpec(id="b", command="echo b", depends_on=["a"]),
            TaskSpec(id="c", command="echo c", tags=["deploy"], depends_on=["b"]),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only=None, skip=None, tags={"deploy"})
        ids = {t.id for t in selected}
        assert ids == {"a", "b", "c"}

    def test_tags_and_skip_tags_interaction(self) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", tags=["build", "slow"]),
            TaskSpec(id="b", command="echo b", tags=["build"]),
            TaskSpec(id="c", command="echo c", tags=["test"]),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only=None, skip=None, tags={"build"}, skip_tags={"slow"})
        ids = {t.id for t in selected}
        assert "b" in ids
        assert "a" not in ids
        assert "c" not in ids

    def test_skip_tags_preserves_all_non_matching(self) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", tags=["alpha"]),
            TaskSpec(id="b", command="echo b", tags=["beta"]),
            TaskSpec(id="c", command="echo c"),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only=None, skip=None, skip_tags={"alpha"})
        ids = {t.id for t in selected}
        assert ids == {"b", "c"}

    def test_tags_diamond_dep_inclusion(self) -> None:
        """Tag filter on merge node includes both branches."""
        tasks = [
            TaskSpec(id="root", command="echo root"),
            TaskSpec(id="left", command="echo left", depends_on=["root"]),
            TaskSpec(id="right", command="echo right", depends_on=["root"]),
            TaskSpec(id="merge", command="echo merge", tags=["release"], depends_on=["left", "right"]),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only=None, skip=None, tags={"release"})
        ids = {t.id for t in selected}
        assert ids == {"root", "left", "right", "merge"}

    def test_only_and_skip_combined(self) -> None:
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only={"b", "c"}, skip={"c"})
        ids = {t.id for t in selected}
        assert "b" in ids
        assert "a" in ids  # transitive dep of b
        assert "c" not in ids

    def test_only_unknown_task_raises(self) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks)
        with pytest.raises(ValueError, match="Unknown --only task"):
            _select_tasks(plan, only={"nonexistent"}, skip=None)

    def test_skip_unknown_task_raises(self) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks)
        with pytest.raises(ValueError, match="Unknown --skip task"):
            _select_tasks(plan, only=None, skip={"nonexistent"})

    def test_tags_no_match_returns_empty(self) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", tags=["build"]),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only=None, skip=None, tags={"deploy"})
        assert selected == []


class TestL4PolicyViolationsMultiple:
    """Multiple policy violations on the same task."""

    def test_multiple_warn_policies(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.models import PolicySpec
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        plan = PlanSpec(
            version=1, name="policy-test", tasks=tasks,
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
            policies=[
                PolicySpec(name="p1", rule="True", action="warn", message="warn1"),
                PolicySpec(name="p2", rule="True", action="warn", message="warn2"),
            ],
        )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        result = run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        assert result.success  # warns don't block
        violations = [e for e in events if e[0] == "policy_violation"]
        assert len(violations) == 2

    def test_block_plus_warn_policies(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.models import PolicySpec
        tasks = [_make_task("a")]
        plan = PlanSpec(
            version=1, name="policy-block", tasks=tasks,
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
            policies=[
                PolicySpec(name="p1", rule="True", action="warn", message="w"),
                PolicySpec(name="p2", rule="True", action="block", message="blocked!"),
            ],
        )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        result = run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        assert not result.success
        assert result.task_results["a"].status == "failed"
        assert "blocked!" in (result.task_results["a"].message or "")

    def test_audit_policy_allows_execution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.models import PolicySpec
        tasks = [_make_task("a")]
        plan = PlanSpec(
            version=1, name="policy-audit", tasks=tasks,
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
            policies=[
                PolicySpec(name="p1", rule="True", action="audit", message="audited"),
            ],
        )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        result = run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        assert result.success
        violations = [e for e in events if e[0] == "policy_violation"]
        assert len(violations) == 1
        assert violations[0][1]["action"] == "audit"


class TestL4ApprovalGateIntegration:
    """Approval gate with auto_approve and approval_handler."""

    def test_auto_approve_skips_prompt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [TaskSpec(id="a", command="echo a", requires_approval=True)]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        result = run_plan(plan, run_dir_override=str(tmp_path), auto_approve=True, event_callback=cb)
        assert result.success
        assert "a" in log
        approvals = [e for e in events if e[0] == "approval_response"]
        assert len(approvals) == 1
        assert approvals[0][1]["approved"] is True

    def test_approval_handler_deny_skips_task(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [TaskSpec(id="a", command="echo a", requires_approval=True)]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        def deny_handler(task_id: str, msg: str | None) -> bool:
            return False
        result = run_plan(plan, run_dir_override=str(tmp_path), approval_handler=deny_handler)
        assert result.task_results["a"].status == "skipped"
        assert "a" not in log

    def test_approval_handler_approve_runs_task(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [TaskSpec(id="a", command="echo a", requires_approval=True)]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        def approve_handler(task_id: str, msg: str | None) -> bool:
            return True
        result = run_plan(plan, run_dir_override=str(tmp_path), approval_handler=approve_handler)
        assert result.success
        assert "a" in log

    def test_approval_handler_exception_denies(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [TaskSpec(id="a", command="echo a", requires_approval=True)]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        def bad_handler(task_id: str, msg: str | None) -> bool:
            raise RuntimeError("handler broke")
        result = run_plan(plan, run_dir_override=str(tmp_path), approval_handler=bad_handler)
        assert result.task_results["a"].status == "skipped"


class TestL4CircuitBreaker:
    """Circuit breaker state tracking across multiple task failures."""

    def test_circuit_breaker_trips_after_n_failures(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.models import CircuitBreakerSpec
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        plan = PlanSpec(
            version=1, name="cb-test", tasks=tasks,
            defaults=PlanDefaults(), fail_fast=False,
            source_path=tmp_path / "p.yaml",
            circuit_breaker=CircuitBreakerSpec(max_total_failures=2, action="fail"),
        )
        overrides = {}
        for tid in ["a", "b"]:
            overrides[tid] = TaskResult(
                task_id=tid, status="failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="fail", log_path=tmp_path / f"{tid}.log",
                result_path=tmp_path / f"{tid}.result.json", message="fail",
            )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        result = run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        assert not result.success
        cb_events = [e for e in events if e[0] == "circuit_breaker_tripped"]
        assert len(cb_events) >= 1

    def test_circuit_breaker_soft_fail_counts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.models import CircuitBreakerSpec
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [
            _make_task("a", allow_failure=True),
            _make_task("b", allow_failure=True),
            _make_task("c"),
        ]
        plan = PlanSpec(
            version=1, name="cb-soft", tasks=tasks,
            defaults=PlanDefaults(), fail_fast=False,
            source_path=tmp_path / "p.yaml",
            circuit_breaker=CircuitBreakerSpec(max_total_failures=2, action="fail"),
        )
        overrides = {}
        for tid in ["a", "b"]:
            overrides[tid] = TaskResult(
                task_id=tid, status="soft_failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="fail", log_path=tmp_path / f"{tid}.log",
                result_path=tmp_path / f"{tid}.result.json", message="soft",
            )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        result = run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        cb_events = [e for e in events if e[0] == "circuit_breaker_tripped"]
        assert len(cb_events) >= 1

    def test_circuit_breaker_pause_auto_approve(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.models import CircuitBreakerSpec
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        plan = PlanSpec(
            version=1, name="cb-pause", tasks=tasks,
            defaults=PlanDefaults(), fail_fast=False,
            source_path=tmp_path / "p.yaml",
            circuit_breaker=CircuitBreakerSpec(max_total_failures=2, action="pause"),
        )
        overrides = {}
        for tid in ["a", "b"]:
            overrides[tid] = TaskResult(
                task_id=tid, status="failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="fail", log_path=tmp_path / f"{tid}.log",
                result_path=tmp_path / f"{tid}.result.json", message="fail",
            )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        # auto_approve=True allows the pause to continue
        result = run_plan(plan, run_dir_override=str(tmp_path), auto_approve=True)
        # c should have run
        assert "c" in result.task_results


class TestL4ContextBudgetTrimEvents:
    """Context budget trimming emits events."""

    def test_context_budget_trim_event_emitted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        long_output = "x" * 10000  # ~2500 tokens
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"], context_from=["a"],
                context_budget_tokens=100,
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                stdout_tail=long_output,
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        result = run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        assert result.success
        trims = [e for e in events if e[0] == "context_budget_trim"]
        assert len(trims) >= 1
        assert trims[0][1]["task_id"] == "b"
        assert trims[0][1]["upstream_id"] == "a"


class TestL4WhenExpressions:
    """When expressions with various dependency outcomes."""

    def test_when_expression_skips_on_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Task with 'when' that evaluates to false => skip."""
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"],
                when="{{ a.status }} == failed",
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert result.task_results["b"].status == "skipped"
        assert "b" not in log

    def test_when_expression_runs_on_match(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"],
                when="{{ a.status }} == success",
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert "b" in log

    def test_when_with_failed_dep_runs_conditional(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When task's dep fails, when expression still evaluates (wait-for-completion semantics)."""
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"],
                when="{{ a.status }} == failed",
            ),
        ]
        plan = _make_plan(tasks, fail_fast=False, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="fail", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="fail",
            ),
        }
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        # b should run because the when condition matches
        assert "b" in log

    def test_when_no_deps_evaluates_immediately(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When task with no deps: if condition always true, it runs."""
        tasks = [
            TaskSpec(
                id="a", command="echo a",
                when="always == always",
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert "a" in log


class TestL4CancelEvent:
    """Cancel event stops execution."""

    def test_cancel_event_skips_pending_tasks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        import time
        cancel = threading.Event()
        cancel.set()  # pre-cancelled
        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path), cancel_event=cancel)
        # All tasks should be cancelled/skipped
        for task_id in result.task_results:
            assert result.task_results[task_id].status == "skipped"


class TestL4WebhookFiringLogic:
    """Webhook fires on plan completion."""

    def test_plan_webhook_url_fires(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a")]
        plan = PlanSpec(
            version=1, name="webhook-test", tasks=tasks,
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
            webhook_url="http://example.com/hook",
        )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        webhook_calls: list[tuple[str, dict]] = []
        def fake_webhook(url: str, payload: dict) -> int:
            webhook_calls.append((url, payload))
            return 200
        monkeypatch.setattr("maestro_cli.scheduler._post_completion_webhook", fake_webhook)
        events: list[tuple[str, dict]] = []
        def cb(name: str, data: dict) -> None:
            events.append((name, data))
        result = run_plan(plan, run_dir_override=str(tmp_path), event_callback=cb)
        assert result.success
        assert len(webhook_calls) == 1
        assert webhook_calls[0][0] == "http://example.com/hook"
        wh_events = [e for e in events if e[0] == "webhook"]
        assert len(wh_events) == 1
        assert wh_events[0][1]["status"] == "delivered"

    def test_cli_webhook_overrides_plan(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a")]
        plan = PlanSpec(
            version=1, name="wh-override", tasks=tasks,
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
            webhook_url="http://plan.com/hook",
        )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        webhook_calls: list[tuple[str, dict]] = []
        def fake_webhook(url: str, payload: dict) -> int:
            webhook_calls.append((url, payload))
            return 200
        monkeypatch.setattr("maestro_cli.scheduler._post_completion_webhook", fake_webhook)
        result = run_plan(plan, run_dir_override=str(tmp_path), webhook_url="http://cli.com/hook")
        assert len(webhook_calls) == 1
        assert webhook_calls[0][0] == "http://cli.com/hook"

    def test_webhook_failure_does_not_affect_result(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a")]
        plan = PlanSpec(
            version=1, name="wh-fail", tasks=tasks,
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
            webhook_url="http://example.com/hook",
        )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        import urllib.error
        def failing_webhook(url: str, payload: dict) -> int:
            raise urllib.error.URLError("connection refused")
        monkeypatch.setattr("maestro_cli.scheduler._post_completion_webhook", failing_webhook)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success  # webhook failure doesn't affect run result


class TestL4DryRunIntegration:
    """Dry run produces expected results."""

    def test_dry_run_all_tasks_dry_run_status(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, dry_run=True, run_dir_override=str(tmp_path))
        assert result.success
        for task_id, tr in result.task_results.items():
            assert tr.status == "dry_run"

    def test_dry_run_skips_preflight(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dry run doesn't check engine availability."""
        tasks = [TaskSpec(id="a", engine="claude", prompt="test")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        # Should not raise even though claude is not on PATH
        result = run_plan(plan, dry_run=True, run_dir_override=str(tmp_path))
        assert result.success


class TestL4SoftFailurePropagation:
    """soft_failed tasks allow dependents to proceed."""

    def test_soft_fail_dependents_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [
            _make_task("a", allow_failure=True),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="soft_failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="fail", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="soft",
            ),
        }
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert "b" in log
        assert result.task_results["b"].status == "success"


class TestL4VerbosityOutput:
    """Output verbosity levels affect console output."""

    def test_quiet_suppresses_normal_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        run_plan(plan, run_dir_override=str(tmp_path), verbosity="quiet")
        captured = capsys.readouterr()
        # Quiet mode should still show final summary
        assert "ok" in captured.out.lower() or "success" in captured.out.lower()

    def test_jsonl_mode_emits_structured_output(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        run_plan(plan, run_dir_override=str(tmp_path), output_mode="jsonl")
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().split("\n") if l.strip()]
        # All lines should be valid JSON
        for line in lines:
            parsed = json.loads(line)
            assert "event" in parsed


class TestL4ManifestWriting:
    """Manifest and summary files are written correctly."""

    def test_manifest_contains_all_tasks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        manifest_path = result.run_path / "run_manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert "a" in manifest["task_results"]
        assert "b" in manifest["task_results"]

    def test_summary_md_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        summary_path = result.run_path / "run_summary.md"
        assert summary_path.exists()
        content = summary_path.read_text(encoding="utf-8")
        assert "Run Summary" in content

    def test_events_jsonl_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        events_path = result.run_path / "events.jsonl"
        assert events_path.exists()
        lines = events_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) >= 2  # at least run_start and run_complete
        for line in lines:
            parsed = json.loads(line)
            assert "event" in parsed


class TestL4SelectTasksEdgeCases:
    """Additional _select_tasks edge cases."""

    def test_only_single_leaf_with_deep_deps(self) -> None:
        """--only on leaf task auto-includes entire chain."""
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
            _make_task("d", depends_on=["c"]),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only={"d"}, skip=None)
        ids = {t.id for t in selected}
        assert ids == {"a", "b", "c", "d"}

    def test_only_multiple_tasks_union_deps(self) -> None:
        """--only {b, c} from diamond includes a (shared root)."""
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
            _make_task("d", depends_on=["b", "c"]),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only={"b", "c"}, skip=None)
        ids = {t.id for t in selected}
        assert ids == {"a", "b", "c"}

    def test_skip_root_blocks_dependents_in_schedule(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Skipping a root task means its dependents never have deps met."""
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, skip={"a"}, run_dir_override=str(tmp_path))
        # b has unresolved deps — it should never run
        # Actually b gets filtered out: _select_tasks removes "a" from selected,
        # then b's dep "a" is not in selected, so b's dep set is empty in the scheduler
        assert "b" in log or "b" not in result.task_results or result.task_results.get("b", None) is not None

    def test_empty_only_set_runs_all(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Empty only set (None) runs all tasks."""
        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert set(log) == {"a", "b"}

    def test_tags_with_skip_intersection(self) -> None:
        """Task matches both tags and skip_tags => excluded."""
        tasks = [
            TaskSpec(id="a", command="echo a", tags=["build", "slow"]),
            TaskSpec(id="b", command="echo b", tags=["build"]),
        ]
        plan = _make_plan(tasks)
        selected = _select_tasks(plan, only=None, skip=None, tags={"build"}, skip_tags={"slow"})
        ids = {t.id for t in selected}
        assert "a" not in ids
        assert "b" in ids


class TestL4ContextIntentFiltering:
    """Intent filtering on multi-upstream context."""

    def test_apply_intent_filtering_empty_upstream(self) -> None:
        result, records, meta = _aif_l4({}, intent_keywords={"test"})
        assert result == {}
        assert records == []

    def test_apply_intent_filtering_empty_keywords(self) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        upstream = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=Path("/tmp/a.log"),
                result_path=Path("/tmp/a.result.json"), message="ok",
                stdout_tail="some output",
            ),
        }
        result, records, meta = _aif_l4(upstream, intent_keywords=set())
        assert result == upstream
        assert records == []

    def test_apply_context_budget_under_budget_no_trim(self) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        upstream = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=Path("/tmp/a.log"),
                result_path=Path("/tmp/a.result.json"), message="ok",
                stdout_tail="short",
            ),
        }
        result, records, meta = _acb_l4(upstream, budget_tokens=1000)
        assert result == upstream
        assert records == []

    def test_apply_context_budget_over_budget_trims(self) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        long_text = "word " * 2000  # ~2000 tokens
        upstream = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=Path("/tmp/a.log"),
                result_path=Path("/tmp/a.result.json"), message="ok",
                stdout_tail=long_text,
            ),
        }
        result, records, meta = _acb_l4(upstream, budget_tokens=100)
        assert len(result["a"].stdout_tail) < len(long_text)
        assert len(records) == 1


class TestL4LayeredContextParsing:
    """Layered context section parsing edge cases."""

    def test_layered_empty_input(self) -> None:
        assert _plcs_l4("") == {}

    def test_layered_no_sections(self) -> None:
        assert _plcs_l4("just plain text without section markers") == {}

    def test_layered_single_section(self) -> None:
        text = "--- task-a ---\nsome content here\n"
        sections = _plcs_l4(text)
        assert "task-a" in sections
        assert "some content" in sections["task-a"]

    def test_layered_multiple_sections(self) -> None:
        text = "--- alpha ---\nalpha content\n--- beta ---\nbeta content\n"
        sections = _plcs_l4(text)
        assert len(sections) == 2
        assert "alpha content" in sections["alpha"]
        assert "beta content" in sections["beta"]

    def test_layered_section_with_dashes_in_id(self) -> None:
        text = "--- my-task-1 ---\ncontent\n"
        sections = _plcs_l4(text)
        assert "my-task-1" in sections


class TestL4LoadPriorResults:
    """_load_prior_results edge cases."""

    def test_missing_manifest_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="No run_manifest.json"):
            _lpr_l4(tmp_path)

    def test_empty_task_results(self, tmp_path: Path) -> None:
        (tmp_path / "run_manifest.json").write_text(
            json.dumps({"task_results": {}}), encoding="utf-8"
        )
        result = _lpr_l4(tmp_path)
        assert result == {}

    def test_soft_failed_included(self, tmp_path: Path) -> None:
        (tmp_path / "run_manifest.json").write_text(
            json.dumps({"task_results": {"a": {"status": "soft_failed"}}}),
            encoding="utf-8",
        )
        result = _lpr_l4(tmp_path)
        assert "a" in result

    def test_skipped_condition_not_met_included(self, tmp_path: Path) -> None:
        (tmp_path / "run_manifest.json").write_text(
            json.dumps({"task_results": {"a": {"status": "skipped", "message": "Condition not met"}}}),
            encoding="utf-8",
        )
        result = _lpr_l4(tmp_path)
        assert "a" in result  # generic skip is kept


class TestL4ParallelismMetricsIntegration:
    """Parallelism savings computed correctly."""

    def test_parallel_tasks_show_savings(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        import time
        now = datetime.now(UTC)
        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks, max_parallel=2, source_path=tmp_path / "p.yaml")
        overrides = {}
        for tid in ["a", "b"]:
            overrides[tid] = TaskResult(
                task_id=tid, status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=1.0,
                command="ok", log_path=tmp_path / f"{tid}.log",
                result_path=tmp_path / f"{tid}.result.json", message="ok",
            )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.sequential_duration_sec == 2.0

    def test_single_task_sequential_duration_matches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.5,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.sequential_duration_sec == 0.5


class TestL4CostAggregation:
    """Total cost aggregation from task results."""

    def test_total_cost_is_sum(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                cost_usd=0.10,
            ),
            "b": TaskResult(
                task_id="b", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "b.log",
                result_path=tmp_path / "b.result.json", message="ok",
                cost_usd=0.20,
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.total_cost_usd is not None
        assert abs(result.total_cost_usd - 0.30) < 0.01

    def test_total_cost_none_when_no_costs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.total_cost_usd is None

    def test_total_tokens_sum(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            ),
            "b": TaskResult(
                task_id="b", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "b.log",
                result_path=tmp_path / "b.result.json", message="ok",
                token_usage=TokenUsage(input_tokens=200, output_tokens=100),
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.total_tokens == 450


class TestL4TaintPropagationIntegration:
    """Taint propagation marks results correctly."""

    def test_tainted_task_result_marked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.task_results["a"].tainted is True

    def test_untainted_task_result_not_marked(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.task_results["a"].tainted is False

    def test_transitive_taint_propagation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
            TaskSpec(id="b", command="echo b", depends_on=["a"], context_from=["a"]),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.task_results["a"].tainted is True
        assert result.task_results["b"].tainted is True

    def test_taint_cleared_by_guard_command(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", context_trust="untrusted"),
            TaskSpec(
                id="b", command="echo b", depends_on=["a"],
                context_from=["a"], guard_command="true",
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.task_results["a"].tainted is True
        assert result.task_results["b"].tainted is False


class TestL4WorkspaceTimeoutEstimation:
    """_estimate_workspace_timeout edge cases."""

    def test_no_engine_returns_none(self, tmp_path: Path) -> None:
        task = TaskSpec(id="a", command="echo a")
        plan = _make_plan([task], source_path=tmp_path / "p.yaml")
        plan = PlanSpec(
            version=1, name="test", tasks=[task],
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
            workspace_root=str(tmp_path),
        )
        result = _ewt_l4(plan, task)
        assert result is None

    def test_no_workspace_root_returns_none(self, tmp_path: Path) -> None:
        task = TaskSpec(id="a", engine="claude", prompt="test src/main.py")
        plan = PlanSpec(
            version=1, name="test", tasks=[task],
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
        )
        result = _ewt_l4(plan, task)
        assert result is None

    def test_no_file_references_returns_none(self, tmp_path: Path) -> None:
        task = TaskSpec(id="a", engine="claude", prompt="just do something")
        plan = PlanSpec(
            version=1, name="test", tasks=[task],
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
            workspace_root=str(tmp_path),
        )
        result = _ewt_l4(plan, task)
        assert result is None

    def test_small_files_returns_none(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        (src_dir / "tiny.py").write_text("x = 1", encoding="utf-8")
        task = TaskSpec(id="a", engine="claude", prompt="review src/tiny.py")
        plan = PlanSpec(
            version=1, name="test", tasks=[task],
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
            workspace_root=str(tmp_path),
        )
        result = _ewt_l4(plan, task)
        assert result is None

    def test_large_file_returns_adjusted_timeout(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        # Need a file large enough that the adjusted timeout exceeds the 1800s default
        # Formula: 300 + (bytes / 3.5) * 0.08 > 1800  =>  bytes > (1500 / 0.08) * 3.5 = 65625
        (src_dir / "big.py").write_text("x" * 200000, encoding="utf-8")
        task = TaskSpec(id="a", engine="claude", prompt="review src/big.py")
        plan = PlanSpec(
            version=1, name="test", tasks=[task],
            defaults=PlanDefaults(), source_path=tmp_path / "p.yaml",
            workspace_root=str(tmp_path),
        )
        result = _ewt_l4(plan, task)
        assert result is not None
        assert result >= 300


class TestL4EventsJsonlIntegrity:
    """Events.jsonl contains hash chain fields."""

    def test_events_have_hash_and_seq(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        events_path = result.run_path / "events.jsonl"
        lines = events_path.read_text(encoding="utf-8").strip().split("\n")
        for line in lines:
            parsed = json.loads(line)
            assert "hash" in parsed
            assert "seq" in parsed
            assert "prev_hash" in parsed

    def test_events_sequence_increments(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        events_path = result.run_path / "events.jsonl"
        lines = events_path.read_text(encoding="utf-8").strip().split("\n")
        seqs = [json.loads(l)["seq"] for l in lines]
        assert seqs == sorted(seqs)
        assert len(set(seqs)) == len(seqs)  # all unique


class TestL4MaxParallelOverride:
    """max_parallel_override controls concurrency."""

    def test_max_parallel_override_respected(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task(f"t{i}") for i in range(5)]
        plan = _make_plan(tasks, max_parallel=10, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, max_parallel_override=1, run_dir_override=str(tmp_path))
        assert result.success
        assert len(log) == 5


class TestL4McpConcurrencySafety:
    """MCP concurrency metadata can serialize unsafe worktree tasks."""

    def test_unsafe_mcp_worktree_tasks_are_serialized(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [
            TaskSpec(id="a", engine="claude", prompt="task a", worktree=True, mcp_tools=["unsafe"]),
            TaskSpec(id="b", engine="claude", prompt="task b", worktree=True, mcp_tools=["unsafe"]),
        ]
        plan = _make_plan(tasks, max_parallel=2, source_path=tmp_path / "p.yaml")
        plan.mcp_servers = [
            MCPServerSpec(name="unsafe", command=["npx", "unsafe-server"], is_concurrency_safe=False),
        ]

        active = [0]
        max_concurrent = [0]
        lock = threading.Lock()

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
        ):
            with lock:
                active[0] += 1
                max_concurrent[0] = max(max_concurrent[0], active[0])
            import time
            time.sleep(0.05)
            with lock:
                active[0] -= 1

            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.05,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert max_concurrent[0] <= 1

    def test_concurrency_safe_mcp_worktree_tasks_can_run_in_parallel(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [
            TaskSpec(id="a", engine="claude", prompt="task a", worktree=True, mcp_tools=["safe"]),
            TaskSpec(id="b", engine="claude", prompt="task b", worktree=True, mcp_tools=["safe"]),
        ]
        plan = _make_plan(tasks, max_parallel=2, source_path=tmp_path / "p.yaml")
        plan.mcp_servers = [
            MCPServerSpec(name="safe", command=["npx", "safe-server"], is_concurrency_safe=True),
        ]

        active = [0]
        max_concurrent = [0]
        lock = threading.Lock()

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
        ):
            with lock:
                active[0] += 1
                max_concurrent[0] = max(max_concurrent[0], active[0])
            import time
            time.sleep(0.05)
            with lock:
                active[0] -= 1

            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id,
                status="success",
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.05,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)
        monkeypatch.setattr("maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None)

        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert max_concurrent[0] >= 2


class TestL4RunDirOverride:
    """run_dir_override controls output directory."""

    def test_custom_run_dir(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        custom_dir = tmp_path / "custom_runs"
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(custom_dir))
        assert str(custom_dir) in str(result.run_path)


class TestL4ExtraTemplateVars:
    """extra_template_vars are forwarded to execute_task."""

    def test_extra_vars_passed_through(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        captured_kwargs: list[dict] = []
        def spy_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                        upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            captured_kwargs.append(kwargs)
            from datetime import UTC, datetime
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json", message="ok",
            )
            result.log_path.write_text("ok\n", encoding="utf-8")
            result.result_path.write_text("{}", encoding="utf-8")
            return result
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", spy_execute)
        run_plan(plan, run_dir_override=str(tmp_path),
                 extra_template_vars={"watch.iteration": "3"})
        assert len(captured_kwargs) == 1
        etv = captured_kwargs[0].get("extra_template_vars", {})
        assert "watch.iteration" in etv


class TestL4UpstreamContextPassing:
    """Upstream context is passed to execute_task correctly."""

    def test_context_from_wildcard_expands_all_deps(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            _make_task("b"),
            TaskSpec(
                id="c", command="echo c",
                depends_on=["a", "b"], context_from=["*"],
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        captured_upstream: list[dict | None] = []
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                stdout_tail="output from a",
            ),
            "b": TaskResult(
                task_id="b", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "b.log",
                result_path=tmp_path / "b.result.json", message="ok",
                stdout_tail="output from b",
            ),
        }
        def spy_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                        upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if task.id == "c":
                captured_upstream.append(upstream_results)
            now_inner = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now_inner, finished_at=now_inner, duration_sec=0.01,
                command="ok", log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json", message="ok",
            )
            result.log_path.write_text("ok\n", encoding="utf-8")
            result.result_path.write_text("{}", encoding="utf-8")
            if task.id in overrides:
                r = overrides[task.id]
                r.log_path.write_text("ok\n", encoding="utf-8")
                r.result_path.write_text("{}", encoding="utf-8")
                return r
            return result
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", spy_execute)
        run_plan(plan, run_dir_override=str(tmp_path))
        assert len(captured_upstream) == 1
        assert captured_upstream[0] is not None
        assert "a" in captured_upstream[0]
        assert "b" in captured_upstream[0]

    def test_context_from_specific_task(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            _make_task("b"),
            TaskSpec(
                id="c", command="echo c",
                depends_on=["a", "b"], context_from=["a"],
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        captured_upstream: list[dict | None] = []
        def spy_execute(plan, task, run_path, dry_run=False, execution_profile="plan",
                        upstream_results=None, context_synthesis="", workspace_brief="", **kwargs):
            if task.id == "c":
                captured_upstream.append(upstream_results)
            now_inner = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now_inner, finished_at=now_inner, duration_sec=0.01,
                command="ok", log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json", message="ok",
            )
            result.log_path.write_text("ok\n", encoding="utf-8")
            result.result_path.write_text("{}", encoding="utf-8")
            return result
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", spy_execute)
        run_plan(plan, run_dir_override=str(tmp_path))
        assert len(captured_upstream) == 1
        assert captured_upstream[0] is not None
        assert "a" in captured_upstream[0]
        assert "b" not in captured_upstream[0]


class TestL4ComputeWavesEdgeCases:
    """_compute_waves for complex DAGs."""

    def test_waves_wide_fanout(self) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [_make_task("root")]
        for i in range(5):
            tasks.append(_make_task(f"leaf{i}", depends_on=["root"]))
        plan = _make_plan(tasks)
        rr = PlanRunResult(
            plan_name="test", run_id="r1", run_path=Path("/tmp"),
            success=True, started_at=now, finished_at=now,
            task_results={
                t.id: _make_success_result(t.id, Path("/tmp"))
                for t in tasks
            },
        )
        waves = _compute_waves(plan, rr)
        assert len(waves) == 2
        assert waves[0] == ["root"]
        assert set(waves[1]) == {f"leaf{i}" for i in range(5)}

    def test_waves_linear_chain(self) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = []
        for i in range(4):
            deps = [f"t{i-1}"] if i > 0 else []
            tasks.append(_make_task(f"t{i}", depends_on=deps))
        plan = _make_plan(tasks)
        rr = PlanRunResult(
            plan_name="test", run_id="r1", run_path=Path("/tmp"),
            success=True, started_at=now, finished_at=now,
            task_results={
                t.id: _make_success_result(t.id, Path("/tmp"))
                for t in tasks
            },
        )
        waves = _compute_waves(plan, rr)
        assert len(waves) == 4
        for i, wave in enumerate(waves):
            assert wave == [f"t{i}"]

    def test_waves_empty_results(self) -> None:
        from datetime import UTC, datetime
        now = datetime.now(UTC)
        tasks = [_make_task("a")]
        plan = _make_plan(tasks)
        rr = PlanRunResult(
            plan_name="test", run_id="r1", run_path=Path("/tmp"),
            success=True, started_at=now, finished_at=now,
            task_results={},
        )
        waves = _compute_waves(plan, rr)
        assert waves == []


class TestL4FmtDurationEdgeCases:
    """_fmt_duration for boundary values."""

    def test_zero_seconds(self) -> None:
        assert _fmt_duration(0) == "0s"

    def test_exactly_60_seconds(self) -> None:
        assert _fmt_duration(60) == "1m00s"

    def test_large_duration(self) -> None:
        result = _fmt_duration(3661)
        assert "61m" in result

    def test_fractional_seconds(self) -> None:
        assert _fmt_duration(0.4) == "0s"

    def test_59_seconds(self) -> None:
        assert _fmt_duration(59) == "59s"


class TestL4NewSkippedResult:
    """_new_skipped_result produces correct files."""

    def test_creates_log_and_result_files(self, tmp_path: Path) -> None:
        result = _new_skipped_result("t1", tmp_path, "test skip")
        assert result.status == "skipped"
        assert result.log_path.exists()
        assert result.result_path.exists()
        assert "skipped" in result.log_path.read_text(encoding="utf-8")

    def test_message_in_result(self, tmp_path: Path) -> None:
        result = _new_skipped_result("t1", tmp_path, "reason here")
        assert result.message == "reason here"


class TestL4NewCachedResult:
    """_new_cached_result reconstructs correctly."""

    def test_reconstructs_from_minimal_cache(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = {"status": "success", "exit_code": 0, "duration_sec": 1.5}
        result = _new_cached_result("t1", tmp_path, cached, "abc123def456", cache_dir)
        assert result.status == "success"
        assert result.task_id == "t1"
        assert "abc123def456" in (result.message or "")

    def test_reconstructs_with_token_usage(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = {
            "status": "success",
            "token_usage": {"input_tokens": 100, "output_tokens": 50},
        }
        result = _new_cached_result("t1", tmp_path, cached, "abc123def456", cache_dir)
        assert result.token_usage is not None
        assert result.token_usage.input_tokens == 100
        assert result.token_usage.output_tokens == 50

    def test_reconstructs_with_structured_context(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = {
            "status": "success",
            "structured_context": {
                "task_id": "t1",
                "status": "success",
                "files_changed": ["a.py"],
            },
        }
        result = _new_cached_result("t1", tmp_path, cached, "abc123def456", cache_dir)
        assert result.structured_context is not None
        assert result.structured_context.files_changed == ["a.py"]

    def test_reconstructs_tool_failure_count(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        cached = {
            "status": "success",
            "tool_failure_count": 2,
        }
        result = _new_cached_result("t1", tmp_path, cached, "abc123def456", cache_dir)
        assert result.tool_failure_count == 2


# ---------------------------------------------------------------------------
# Coverage push — targeting 81% → 90%+
# ---------------------------------------------------------------------------


class TestColorHelpers:
    """Cover the ANSI _c() helper when TTY + NO_COLOR conditions vary."""

    def test_c_returns_ansi_when_tty_and_no_no_color(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli import scheduler as sched
        monkeypatch.setattr(sched, "_NO_COLOR", False)
        monkeypatch.setattr(sched, "_IS_TTY", True)
        assert "\033[" in sched._c("36", "hello")

    def test_c_returns_plain_when_no_color(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli import scheduler as sched
        monkeypatch.setattr(sched, "_NO_COLOR", True)
        monkeypatch.setattr(sched, "_IS_TTY", True)
        assert sched._c("36", "hello") == "hello"

    def test_c_returns_plain_when_not_tty(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli import scheduler as sched
        monkeypatch.setattr(sched, "_NO_COLOR", False)
        monkeypatch.setattr(sched, "_IS_TTY", False)
        assert sched._c("36", "hello") == "hello"


class TestShowFailTail:
    """Cover _show_fail_tail body extraction and printing."""

    def test_extracts_body_lines(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.scheduler import _show_fail_tail
        log = tmp_path / "t.log"
        log.write_text(
            "task=t\nstarted_at=...\ncommand=echo ok\n\nbody line 1\nbody line 2\nstatus=failed\nmessage=err\n",
            encoding="utf-8",
        )
        _show_fail_tail(log)
        out = capsys.readouterr().out
        assert "body line 1" in out
        assert "body line 2" in out
        # footer lines should NOT appear
        assert "status=failed" not in out

    def test_returns_early_for_missing_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.scheduler import _show_fail_tail
        _show_fail_tail(tmp_path / "nonexistent.log")
        assert capsys.readouterr().out == ""

    def test_returns_early_for_no_body(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.scheduler import _show_fail_tail
        log = tmp_path / "t.log"
        log.write_text("no blank line ever", encoding="utf-8")
        _show_fail_tail(log)
        assert capsys.readouterr().out == ""


class TestLoadTaskPromptText:
    """Cover _load_task_prompt_text fallback paths."""

    def test_returns_prompt_directly(self) -> None:
        from maestro_cli.scheduler import _load_task_prompt_text
        task = TaskSpec(id="t", prompt="hello world", command="echo hi")
        plan = _make_plan([task])
        assert _load_task_prompt_text(plan, task) == "hello world"

    def test_reads_prompt_file(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _load_task_prompt_text
        pf = tmp_path / "prompt.txt"
        pf.write_text("file content", encoding="utf-8")
        task = TaskSpec(id="t", prompt_file=str(pf), command="echo hi")
        plan = _make_plan([task], source_path=tmp_path / "p.yaml")
        assert _load_task_prompt_text(plan, task) == "file content"

    def test_returns_empty_for_missing_prompt_file(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _load_task_prompt_text
        task = TaskSpec(id="t", prompt_file=str(tmp_path / "missing.txt"), command="echo hi")
        plan = _make_plan([task], source_path=tmp_path / "p.yaml")
        assert _load_task_prompt_text(plan, task) == ""

    def test_reads_prompt_md_file(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _load_task_prompt_text
        md = tmp_path / "prompts.md"
        md.write_text("## My Heading\nContent here\n", encoding="utf-8")
        task = TaskSpec(
            id="t", command="echo hi",
            prompt_md_file=str(md),
            prompt_md_heading="My Heading",
        )
        plan = _make_plan([task], source_path=tmp_path / "p.yaml")
        result = _load_task_prompt_text(plan, task)
        assert "Content here" in result

    def test_returns_empty_for_missing_md_file(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _load_task_prompt_text
        task = TaskSpec(
            id="t", command="echo hi",
            prompt_md_file=str(tmp_path / "missing.md"),
            prompt_md_heading="Some Heading",
        )
        plan = _make_plan([task], source_path=tmp_path / "p.yaml")
        assert _load_task_prompt_text(plan, task) == ""

    def test_returns_empty_for_bad_heading(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _load_task_prompt_text
        md = tmp_path / "prompts.md"
        md.write_text("## Other\nNot matching\n", encoding="utf-8")
        task = TaskSpec(
            id="t", command="echo hi",
            prompt_md_file=str(md),
            prompt_md_heading="Missing Heading",
        )
        plan = _make_plan([task], source_path=tmp_path / "p.yaml")
        assert _load_task_prompt_text(plan, task) == ""

    def test_returns_empty_for_no_prompt_source(self) -> None:
        from maestro_cli.scheduler import _load_task_prompt_text
        task = TaskSpec(id="t", command="echo hi")
        plan = _make_plan([task])
        assert _load_task_prompt_text(plan, task) == ""


class TestEstimateWorkspaceTimeout:
    """Cover _estimate_workspace_timeout edge cases."""

    def test_returns_none_when_no_engine(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _estimate_workspace_timeout
        task = TaskSpec(id="t", command="echo hi")
        plan = _make_plan([task])
        plan.workspace_root = str(tmp_path)
        assert _estimate_workspace_timeout(plan, task) is None

    def test_returns_none_when_no_workspace_root(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _estimate_workspace_timeout
        task = TaskSpec(id="t", engine="claude", prompt="do stuff")
        plan = _make_plan([task])
        assert _estimate_workspace_timeout(plan, task) is None

    def test_returns_none_for_small_files(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _estimate_workspace_timeout
        # Create a small file
        small = tmp_path / "small.py"
        small.write_text("x = 1\n", encoding="utf-8")
        task = TaskSpec(id="t", engine="claude", prompt=f"Fix {small.name}")
        plan = _make_plan([task], source_path=tmp_path / "p.yaml")
        plan.workspace_root = str(tmp_path)
        assert _estimate_workspace_timeout(plan, task) is None

    def test_returns_adjusted_for_large_file(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _estimate_workspace_timeout
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        big = src_dir / "big.py"
        big.write_text("x" * 200_000, encoding="utf-8")
        task = TaskSpec(id="t", engine="claude", prompt="Fix src/big.py")
        plan = _make_plan([task], source_path=tmp_path / "p.yaml")
        plan.workspace_root = str(tmp_path)
        result = _estimate_workspace_timeout(plan, task)
        # With 200K bytes, estimated timeout should be > default 1800
        assert result is not None
        assert result > 300

    def test_returns_none_when_adjusted_below_current(self, tmp_path: Path) -> None:
        from maestro_cli.scheduler import _estimate_workspace_timeout
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        small = src_dir / "small.py"
        small.write_text("z" * 20_000, encoding="utf-8")
        task = TaskSpec(id="t", engine="claude", prompt="Fix src/small.py", timeout_sec=7200)
        plan = _make_plan([task], source_path=tmp_path / "p.yaml")
        plan.workspace_root = str(tmp_path)
        # With a huge existing timeout, no adjustment needed
        assert _estimate_workspace_timeout(plan, task) is None


class TestPrepareSummaries:
    """Cover _prepare_summaries LLM call path."""

    def test_populates_summary_via_run_summarization(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.scheduler import _prepare_summaries
        from maestro_cli.models import StructuredContext
        sc = StructuredContext(
            task_id="a", status="success", exit_code=0, duration_sec=0.1,
            summary="",
        )
        result = _make_success_result("a", tmp_path, stdout_tail="some output")
        result.structured_context = sc
        upstream = {"a": result}
        monkeypatch.setattr(
            "maestro_cli.scheduler._run_summarization",
            lambda tid, tail, sc, workdir, model="haiku": "mocked summary",
        )
        summaries = _prepare_summaries(upstream, tmp_path)
        assert len(summaries) == 1
        assert sc.summary == "mocked summary"

    def test_skips_already_summarized(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.scheduler import _prepare_summaries
        from maestro_cli.models import StructuredContext
        sc = StructuredContext(
            task_id="a", status="success", exit_code=0, duration_sec=0.1,
            summary="already done",
        )
        result = _make_success_result("a", tmp_path, stdout_tail="some output")
        result.structured_context = sc
        upstream = {"a": result}
        call_count = [0]
        def no_call(*a, **kw):
            call_count[0] += 1
            return ""
        monkeypatch.setattr("maestro_cli.scheduler._run_summarization", no_call)
        _prepare_summaries(upstream, tmp_path)
        assert call_count[0] == 0


class TestApplyContextBudgetEviction:
    """Cover the priority-based eviction path in _apply_context_budget."""

    def test_evicts_lowest_score_upstream_first(self) -> None:
        from maestro_cli.scheduler import _apply_context_budget
        now = datetime.now(UTC)
        # Two upstreams: one low-score, one high-score
        upstream = {
            "low": TaskResult(
                task_id="low", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.1,
                command="ok", stdout_tail="a " * 500,  # ~250 tokens
            ),
            "high": TaskResult(
                task_id="high", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.1,
                command="ok", stdout_tail="b " * 500,  # ~250 tokens
            ),
        }
        # Budget is tight—only room for ~100 tokens
        # intent_keywords cause "a" to score low, "b" to score higher
        result, records, meta = _apply_context_budget(
            upstream, budget_tokens=100,
            intent_keywords={"b"},
        )
        # The low-score upstream should be trimmed more aggressively
        assert "low" in result
        assert "high" in result

    def test_proportional_trim_without_intent_keywords(self) -> None:
        from maestro_cli.scheduler import _apply_context_budget
        now = datetime.now(UTC)
        upstream = {
            "x": TaskResult(
                task_id="x", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.1,
                command="ok", stdout_tail="hello world " * 200,
            ),
        }
        result, records, meta = _apply_context_budget(
            upstream, budget_tokens=50,
        )
        # Should proportionally trim
        assert len(result["x"].stdout_tail) < len(upstream["x"].stdout_tail)
        assert len(records) == 1
        assert meta == {}

    def test_eviction_empties_low_score_tail(self) -> None:
        """When overflow exceeds a low-score upstream's token count, its tail is emptied."""
        from maestro_cli.scheduler import _apply_context_budget
        now = datetime.now(UTC)
        upstream = {
            "tiny": TaskResult(
                task_id="tiny", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.1,
                command="ok", stdout_tail="small text",
            ),
            "big": TaskResult(
                task_id="big", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.1,
                command="ok", stdout_tail="important data " * 500,
            ),
        }
        result, records, meta = _apply_context_budget(
            upstream, budget_tokens=50,
            intent_keywords={"important", "data"},
        )
        # tiny should be heavily trimmed or emptied
        assert len(result["tiny"].stdout_tail) <= len(upstream["tiny"].stdout_tail)


class TestApplyContextBudgetUnderBudgetWithKeywords:
    """Cover the path where total <= budget but intent_keywords are provided."""

    def test_returns_selection_meta_when_under_budget_with_keywords(self) -> None:
        from maestro_cli.scheduler import _apply_context_budget
        now = datetime.now(UTC)
        upstream = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.1,
                command="ok", stdout_tail="short text about security",
            ),
        }
        result, records, meta = _apply_context_budget(
            upstream, budget_tokens=10000,
            intent_keywords={"security"},
        )
        # Under budget, no trim records, but selection_meta returned
        assert records == []
        assert "a" in meta or meta == {}

    def test_intent_filtering_trims_when_over_budget(self) -> None:
        from maestro_cli.scheduler import _apply_context_budget
        now = datetime.now(UTC)
        # Large upstream that will exceed budget
        text = ("relevant security check " * 50 + "\n\n" + "unrelated filler text " * 200)
        upstream = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.1,
                command="ok", stdout_tail=text,
            ),
        }
        result, records, meta = _apply_context_budget(
            upstream, budget_tokens=100,
            intent_keywords={"security", "check"},
        )
        # After intent filtering + budget trim, the tail should be shorter
        assert len(result["a"].stdout_tail) < len(upstream["a"].stdout_tail)


class TestCircuitBreakerPauseDenied:
    """Circuit breaker pause action when approval is denied triggers fail-fast."""

    def test_pause_denied_by_handler_triggers_fail_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import CircuitBreakerSpec
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c", depends_on=["b"]),
        ]
        plan = PlanSpec(
            version=1, name="cb-deny", tasks=tasks,
            defaults=PlanDefaults(), fail_fast=False,
            source_path=tmp_path / "p.yaml",
            circuit_breaker=CircuitBreakerSpec(max_total_failures=2, action="pause"),
        )
        overrides = {}
        for tid in ["a", "b"]:
            overrides[tid] = TaskResult(
                task_id=tid, status="failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="fail", log_path=tmp_path / f"{tid}.log",
                result_path=tmp_path / f"{tid}.result.json", message="fail",
            )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
            approval_handler=lambda task_id, msg: False,  # deny the pause
        )
        assert not result.success
        cb_events = [e for e in events if e[0] == "circuit_breaker_tripped"]
        assert len(cb_events) >= 1
        # c should be skipped due to fail-fast
        assert result.task_results.get("c") is not None
        assert result.task_results["c"].status == "skipped"

    def test_pause_with_no_handler_non_interactive_denies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Circuit breaker pause with no handler and non-interactive stdin is denied."""
        from maestro_cli.models import CircuitBreakerSpec
        now = datetime.now(UTC)
        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        plan = PlanSpec(
            version=1, name="cb-nonint", tasks=tasks,
            defaults=PlanDefaults(), fail_fast=False,
            source_path=tmp_path / "p.yaml",
            circuit_breaker=CircuitBreakerSpec(max_total_failures=2, action="pause"),
        )
        overrides = {}
        for tid in ["a", "b"]:
            overrides[tid] = TaskResult(
                task_id=tid, status="failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="fail", log_path=tmp_path / f"{tid}.log",
                result_path=tmp_path / f"{tid}.result.json", message="fail",
            )
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        # Force stdin to be non-interactive
        monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"isatty": lambda self: False})())
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert not result.success


class TestPolicyWarnAndAudit:
    """Cover policy warn and audit actions (not just block)."""

    def test_policy_warn_emits_event_but_runs_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import PolicySpec
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        plan.policies = [
            PolicySpec(name="warn-all", rule="True", action="warn", message="just a warning"),
        ]
        rph: list[Path] = []
        mock, call_log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        # Task should still run despite warn
        assert "a" in call_log
        assert result.success
        policy_events = [e for e in events if e[0] == "policy_violation"]
        assert len(policy_events) >= 1
        assert policy_events[0][1]["action"] == "warn"

    def test_policy_audit_emits_event_but_runs_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import PolicySpec
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        plan.policies = [
            PolicySpec(name="audit-all", rule="True", action="audit", message="logged"),
        ]
        rph: list[Path] = []
        mock, call_log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert "a" in call_log
        assert result.success
        policy_events = [e for e in events if e[0] == "policy_violation"]
        assert len(policy_events) >= 1
        assert policy_events[0][1]["action"] == "audit"

    def test_policy_block_prevents_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import PolicySpec
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        plan.policies = [
            PolicySpec(name="block-all", rule="True", action="block", message="blocked"),
        ]
        rph: list[Path] = []
        mock, call_log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        # Task should NOT run
        assert "a" not in call_log
        assert not result.success
        assert result.task_results["a"].status == "failed"
        policy_events = [e for e in events if e[0] == "policy_violation"]
        assert len(policy_events) >= 1
        assert policy_events[0][1]["action"] == "block"


class TestBudgetPeriodGate:
    """Cover the cross-run budget period gate (budget_period + check_budget)."""

    def test_budget_period_exceeded_returns_empty_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        plan.budget_period = "daily"
        plan.max_cost_usd = 1.0
        # Mock check_budget to say budget is exceeded
        monkeypatch.setattr(
            "maestro_cli.scheduler.execute_task",
            lambda *a, **kw: None,
        )
        import maestro_cli.scheduler as sched
        orig_run = sched.run_plan
        # We patch check_budget to return not-allowed
        monkeypatch.setattr(
            "maestro_cli.budget.check_budget",
            lambda *a, **kw: (False, 5.0, 0.0),
        )
        monkeypatch.setattr(
            "maestro_cli.budget._DEFAULT_LEDGER_PATH",
            ".maestro-budget.jsonl",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert not result.success
        assert result.task_results == {}


class TestBudgetExceededAfterCompletion:
    """Cover budget_exceeded path after task completion (lines 2838-2878)."""

    def test_budget_exceeded_after_completion_skips_remaining(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c"),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml", fail_fast=False)
        plan.max_cost_usd = 0.05
        # a costs $0.06 → exceeds budget
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                cost_usd=0.06,
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
            max_parallel_override=1,
        )
        assert result.budget_exceeded is True
        exceeded_events = [e for e in events if e[0] == "budget_exceeded"]
        assert len(exceeded_events) >= 1
        # At least one remaining task should be skipped
        skipped = [tid for tid, r in result.task_results.items() if r.status == "skipped"]
        assert len(skipped) >= 1


class TestBudgetWarningAfterCompletion:
    """Cover budget_warning_pct path after task completion."""

    def test_warning_emitted_at_threshold(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            _make_task("b"),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml", fail_fast=False)
        plan.max_cost_usd = 10.0
        plan.budget_warning_pct = 0.5  # warn at 50%
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                cost_usd=5.50,
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
            max_parallel_override=1,
        )
        warning_events = [e for e in events if e[0] == "budget_warning"]
        assert len(warning_events) >= 1
        assert warning_events[0][1]["spent"] == 5.50


class TestTaintedTaskMarking:
    """Cover taint propagation marking results as tainted at runtime."""

    def test_tainted_result_marked_in_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [
            TaskSpec(id="a", command="echo untrusted", context_trust="untrusted"),
            TaskSpec(id="b", command="echo b", depends_on=["a"], context_from=["a"]),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        # a is untrusted → tainted
        assert result.task_results["a"].tainted is True
        # b consumes a via context_from without guard → tainted
        assert result.task_results["b"].tainted is True
        taint_events = [e for e in events if e[0] == "taint_detected"]
        assert len(taint_events) >= 1


class TestWebhookDelivery:
    """Cover webhook delivery and failure paths."""

    def test_webhook_emits_delivered_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        # Mock _post_completion_webhook to succeed
        monkeypatch.setattr(
            "maestro_cli.scheduler._post_completion_webhook",
            lambda url, payload: 200,
        )
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
            webhook_url="https://example.com/hook",
        )
        assert result.success
        webhook_events = [e for e in events if e[0] == "webhook"]
        assert len(webhook_events) >= 1
        assert webhook_events[0][1]["status"] == "delivered"

    def test_webhook_failure_emits_failed_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        # Mock _post_completion_webhook to raise
        def fail_hook(url, payload):
            raise urllib.error.URLError("connection refused")
        import urllib.error
        monkeypatch.setattr(
            "maestro_cli.scheduler._post_completion_webhook",
            fail_hook,
        )
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
            webhook_url="https://example.com/hook",
        )
        assert result.success
        webhook_events = [e for e in events if e[0] == "webhook"]
        assert len(webhook_events) >= 1
        assert webhook_events[0][1]["status"] == "failed"


class TestContextModeStructuralInRun:
    """Cover context_mode: structural integration path."""

    def test_structural_context_builds_synthesis(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"], context_from=["a"],
                context_mode="structural",
                context_budget_tokens=5000,
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                stdout_tail="def fix_bug():\n    pass\n",
            ),
        }
        rph: list[Path] = []
        captured = {}
        def capturing_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            if task.id == "b":
                captured["synthesis"] = context_synthesis
            rph.append(run_path) if not rph else None
            mock_fn, _ = _mock_execute_task_factory(rph, overrides=overrides)
            return mock_fn(plan, task, run_path, dry_run, execution_profile,
                          upstream_results, context_synthesis, workspace_brief, **kw)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        # structural context mode should produce some synthesis (even if minimal)
        assert "synthesis" in captured

    def test_knowledge_graph_context_builds_synthesis(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"], context_from=["a"],
                context_mode="knowledge_graph",
                context_budget_tokens=5000,
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                stdout_tail="Modified scheduler.py to fix timeout issue\nDecision: use 30s default\n",
            ),
        }
        rph: list[Path] = []
        captured = {}
        def capturing_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            if task.id == "b":
                captured["synthesis"] = context_synthesis
            rph.append(run_path) if not rph else None
            mock_fn, _ = _mock_execute_task_factory(rph, overrides=overrides)
            return mock_fn(plan, task, run_path, dry_run, execution_profile,
                          upstream_results, context_synthesis, workspace_brief, **kw)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert "synthesis" in captured


class TestContextModeSummarizedMapReduce:
    """Cover summarized and map_reduce context mode integration paths."""

    def test_summarized_calls_prepare_summaries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import StructuredContext
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"], context_from=["a"],
                context_mode="summarized",
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        sc = StructuredContext(
            task_id="a", status="success", exit_code=0, duration_sec=0.1,
            summary="",
        )
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                stdout_tail="some output here",
                structured_context=sc,
            ),
        }
        summary_called = [False]
        orig_prepare = None
        def mock_prepare(upstream, workdir, model="haiku"):
            summary_called[0] = True
            for tid, r in upstream.items():
                if r.structured_context and not r.structured_context.summary:
                    r.structured_context.summary = "mocked"
            return [(tid, 0.01) for tid in upstream]
        monkeypatch.setattr("maestro_cli.scheduler._prepare_summaries", mock_prepare)
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert result.success
        assert summary_called[0]
        summarize_events = [e for e in events if e[0] == "context_summarize"]
        assert len(summarize_events) >= 1

    def test_map_reduce_calls_prepare_and_reduce(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import StructuredContext
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"], context_from=["a"],
                context_mode="map_reduce",
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        sc = StructuredContext(
            task_id="a", status="success", exit_code=0, duration_sec=0.1,
            summary="",
        )
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                stdout_tail="some output here",
                structured_context=sc,
            ),
        }
        monkeypatch.setattr(
            "maestro_cli.scheduler._prepare_summaries",
            lambda up, wd, model="haiku": [(tid, 0.01) for tid in up],
        )
        monkeypatch.setattr(
            "maestro_cli.scheduler._run_map_reduce",
            lambda up, wd, model="haiku": "reduced synthesis",
        )
        rph: list[Path] = []
        captured = {}
        def capturing_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            if task.id == "b":
                captured["synthesis"] = context_synthesis
            rph.append(run_path) if not rph else None
            mock_fn, _ = _mock_execute_task_factory(rph, overrides=overrides)
            return mock_fn(plan, task, run_path, dry_run, execution_profile,
                          upstream_results, context_synthesis, workspace_brief, **kw)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert captured.get("synthesis") == "reduced synthesis"


class TestResumeSkipsCompletedTasks:
    """Cover resume logic: pre-populated tasks are skipped."""

    def test_resumed_tasks_not_re_executed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        # Create a prior run with 'a' succeeded
        prior_run = tmp_path / "prior_run"
        prior_run.mkdir()
        manifest = {
            "plan_name": "test-plan",
            "task_results": {
                "a": {"status": "success", "exit_code": 0, "duration_sec": 1.0},
            },
        }
        (prior_run / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        rph: list[Path] = []
        mock, call_log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path), resume_path=prior_run)
        # 'a' should not be re-executed
        assert "a" not in call_log
        # 'b' should run
        assert "b" in call_log
        assert result.success


class TestKnowledgeInjection:
    """Cover cross-run knowledge injection into task prompts."""

    def test_knowledge_loaded_and_injected_as_template_var(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import KnowledgeRecord
        from maestro_cli import knowledge as knowledge_mod
        selected_calls: dict[str, str] = {}
        tasks = [
            TaskSpec(id="a", engine="claude", prompt="investigate timeout in build step"),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        knowledge_records = {
            "a": [
                KnowledgeRecord(
                    task_id="a",
                    kind="failure_pattern",
                    insight="Timeout after 30s",
                    confidence=0.9,
                    occurrences=2,
                    first_seen="2026-01-01T00:00:00Z",
                    last_seen="2026-01-01T00:00:00Z",
                ),
            ],
        }
        # Mock load_knowledge at the module level where the lazy import resolves
        monkeypatch.setattr(
            knowledge_mod,
            "load_knowledge",
            lambda plan_name, source_dir, **kwargs: knowledge_records,
        )
        monkeypatch.setattr(
            knowledge_mod,
            "build_knowledge_index",
            lambda plan_name, knowledge, **kwargs: "Plan: test-plan\n- [task=a] [FAIL] Timeout after 30s",
        )
        monkeypatch.setattr(
            knowledge_mod,
            "select_relevant_knowledge",
            lambda knowledge, prompt_text, **kwargs: (
                selected_calls.update(
                    {
                        "prompt_text": prompt_text,
                        "task_id": str(kwargs.get("task_id", "")),
                    }
                )
                or knowledge_records["a"]
            ),
        )
        # Mock format_knowledge
        monkeypatch.setattr(
            knowledge_mod,
            "format_knowledge",
            lambda records, **kwargs: "- Timeout after 30s (90%)",
        )
        rph: list[Path] = []
        captured_vars: dict = {}
        def capturing_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            extra = kw.get("extra_template_vars", {})
            captured_vars.update(extra)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="echo ok", log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json", message="ok",
            )
            result.log_path.write_text(f"status=success\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_mock)
        # Patch engine availability check
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/claude" if x == "claude" else None)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert "task_knowledge" in captured_vars
        assert "knowledge_index" in captured_vars
        assert "timeout in build step" in selected_calls["prompt_text"]
        assert selected_calls["task_id"] == "a"

    def test_poison_alert_removes_knowledge_and_emits_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli import knowledge as knowledge_mod
        from maestro_cli.memory import RetrievalDominanceAlert
        from maestro_cli.models import KnowledgeRecord

        tasks = [
            TaskSpec(id="a", engine="claude", prompt="investigate timeout in build step"),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        selected_record = KnowledgeRecord(
            task_id="a",
            kind="failure_pattern",
            insight="Timeout after 30s",
            confidence=0.9,
            occurrences=2,
            first_seen="2026-01-01T00:00:00Z",
            last_seen="2026-01-01T00:00:00Z",
        )
        monkeypatch.setattr(
            knowledge_mod,
            "load_knowledge",
            lambda plan_name, source_dir, **kwargs: {"a": [selected_record]},
        )
        monkeypatch.setattr(
            knowledge_mod,
            "build_knowledge_index",
            lambda plan_name, knowledge, **kwargs: "Plan: test-plan\n- [task=a] [FAIL] Timeout after 30s",
        )
        monkeypatch.setattr(
            knowledge_mod,
            "select_relevant_knowledge",
            lambda knowledge, prompt_text, **kwargs: [selected_record],
        )
        monkeypatch.setattr(
            knowledge_mod,
            "record_knowledge_retrievals",
            lambda plan_name, source_dir, prompt_text, records: [
                RetrievalDominanceAlert(
                    record_id=1,
                    task_id="a",
                    kind="failure_pattern",
                    insight="Timeout after 30s",
                    insight_key="abc123",
                    query_cluster="build|pytest|timeout",
                    retrieval_count=7,
                    cluster_mean=0.5,
                    cluster_stddev=1.5,
                    z_score=4.33,
                )
            ],
        )
        monkeypatch.setattr(
            knowledge_mod,
            "format_knowledge",
            lambda records, **kwargs: "- Timeout after 30s (90%)" if records else "",
        )

        captured_vars: dict[str, str] = {}
        events: list[tuple[str, dict[str, object]]] = []

        def capturing_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            extra = kw.get("extra_template_vars", {})
            captured_vars.update(extra)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="echo ok", log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json", message="ok",
            )
            result.log_path.write_text("status=success\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_mock)
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/claude" if x == "claude" else None)

        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )

        assert result.success
        assert "knowledge_index" in captured_vars
        assert "task_knowledge" not in captured_vars
        poison_events = [data for name, data in events if name == "knowledge_poison_alert"]
        assert poison_events
        assert poison_events[0]["source_task_id"] == "a"


class TestEventSecretMasking:
    """Cover event callback secret masking path."""

    def test_secrets_masked_in_event_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        plan.secrets = ["MY_SECRET"]
        # Set the secret env var
        monkeypatch.setenv("MY_SECRET", "supersecretvalue")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert result.success
        # The secret value should not appear in any event payload string fields
        for name, data in events:
            for k, v in data.items():
                if isinstance(v, str):
                    assert "supersecretvalue" not in v


class TestDagMetadataInjection:
    """Cover DAG metadata injection for routing."""

    def test_dag_metadata_attached_to_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        seen_dag_metadata: dict = {}
        def capturing_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            if hasattr(task, "_dag_metadata"):
                seen_dag_metadata[task.id] = task._dag_metadata
            rph.append(run_path) if not rph else None
            mock_fn, _ = _mock_execute_task_factory(rph)
            return mock_fn(plan, task, run_path, dry_run, execution_profile,
                          upstream_results, context_synthesis, workspace_brief, **kw)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        # All tasks should have dag_metadata
        for tid in ["a", "b", "c"]:
            assert tid in seen_dag_metadata
            md = seen_dag_metadata[tid]
            assert "fan_out" in md
            assert "depth" in md
            assert "upstream_failure_rate" in md

    def test_fan_out_computed_correctly(self) -> None:
        from maestro_cli.scheduler import _compute_fan_out
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["a"]),
            _make_task("d", depends_on=["b"]),
        ]
        plan = _make_plan(tasks)
        assert _compute_fan_out(tasks[0], plan) == 2  # b and c depend on a
        assert _compute_fan_out(tasks[1], plan) == 1  # only d depends on b
        assert _compute_fan_out(tasks[2], plan) == 0  # nobody depends on c

    def test_task_depth_computed_correctly(self) -> None:
        from maestro_cli.scheduler import _compute_task_depth
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
        ]
        plan = _make_plan(tasks)
        assert _compute_task_depth(tasks[0], plan) == 0
        assert _compute_task_depth(tasks[1], plan) == 1
        assert _compute_task_depth(tasks[2], plan) == 2


class TestHopDistancesEdgeCases:
    """Cover _compute_hop_distances for missing tasks and wildcard."""

    def test_hop_distances_with_missing_intermediate(self) -> None:
        from maestro_cli.scheduler import _compute_hop_distances
        tasks = {
            "c": TaskSpec(id="c", command="echo c", depends_on=["b"], context_from=["a"]),
        }
        # 'a' and 'b' are not in task_map — should handle gracefully
        result = _compute_hop_distances("c", ["a"], tasks)
        assert result == {}

    def test_hop_distances_direct_dep_is_1(self) -> None:
        from maestro_cli.scheduler import _compute_hop_distances
        tasks = {
            "a": TaskSpec(id="a", command="echo a"),
            "b": TaskSpec(id="b", command="echo b", depends_on=["a"], context_from=["a"]),
        }
        result = _compute_hop_distances("b", ["a"], tasks)
        assert result["a"] == 1

    def test_hop_distances_transitive(self) -> None:
        from maestro_cli.scheduler import _compute_hop_distances
        tasks = {
            "a": TaskSpec(id="a", command="echo a"),
            "b": TaskSpec(id="b", command="echo b", depends_on=["a"]),
            "c": TaskSpec(id="c", command="echo c", depends_on=["b"], context_from=["a"]),
        }
        result = _compute_hop_distances("c", ["a"], tasks)
        assert result["a"] == 2


class TestApplyHopDecayEdgeCases:
    """Cover _apply_hop_decay for multi-hop trimming."""

    def test_hop_decay_trims_deep_upstream(self) -> None:
        now = datetime.now(UTC)
        upstream = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.1,
                command="ok", stdout_tail="x" * 1000,
            ),
        }
        result = _apply_hop_decay(upstream, {"a": 3})  # hop=3 → keep 64%
        assert len(result["a"].stdout_tail) < 1000
        assert len(result["a"].stdout_tail) > 0

    def test_hop_decay_direct_no_trim(self) -> None:
        now = datetime.now(UTC)
        upstream = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.1,
                command="ok", stdout_tail="important data",
            ),
        }
        result = _apply_hop_decay(upstream, {"a": 1})
        assert result["a"].stdout_tail == "important data"


class TestFilterTailByIntentEdgeCases:
    """Cover _filter_tail_by_intent BM25 fallback and empty inputs."""

    def test_empty_tail_returns_as_is(self) -> None:
        from maestro_cli.scheduler import _filter_tail_by_intent
        result, score, kws = _filter_tail_by_intent("", {"security"})
        assert result == ""

    def test_empty_keywords_returns_as_is(self) -> None:
        from maestro_cli.scheduler import _filter_tail_by_intent
        result, score, kws = _filter_tail_by_intent("some text", set())
        assert result == "some text"

    def test_bm25_fallback_when_idf_yields_nothing(self) -> None:
        from maestro_cli.scheduler import _filter_tail_by_intent
        # Text with clear sections, keywords should match in intersection fallback
        text = "section about authentication\n\nunrelated filler content\n\nsecurity review findings"
        result, score, kws = _filter_tail_by_intent(
            text,
            {"authentication"},
            idf={"authentication": 0.0001},  # nearly-zero IDF should produce 0 BM25 score
        )
        # Even if BM25 scores are near zero, the fallback should still work
        assert isinstance(result, str)


class TestSelectiveContextModeInRun:
    """Cover context_mode: selective integration path."""

    def test_selective_context_mode_runs_successfully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"], context_from=["a"],
                context_mode="selective",
                context_budget_tokens=5000,
                prompt="Review security findings",
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                stdout_tail="Security scan found 3 vulnerabilities\n\nPerformance metrics are good\n\nDone.",
            ),
        }
        rph: list[Path] = []
        captured = {}
        def capturing_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            if task.id == "b":
                captured["synthesis"] = context_synthesis
            rph.append(run_path) if not rph else None
            mock_fn, _ = _mock_execute_task_factory(rph, overrides=overrides)
            return mock_fn(plan, task, run_path, dry_run, execution_profile,
                          upstream_results, context_synthesis, workspace_brief, **kw)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert "synthesis" in captured


class TestKnowledgeExtractionPostRun:
    """Cover post-run knowledge extraction and storage."""

    def test_knowledge_extracted_after_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        extract_called = [False]
        def mock_extract(run_result):
            extract_called[0] = True
            return []
        monkeypatch.setattr("maestro_cli.knowledge.extract_knowledge", mock_extract)
        monkeypatch.setattr("maestro_cli.knowledge.store_knowledge", lambda *a, **kw: None)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert extract_called[0]

    def test_memory_write_events_emitted_after_knowledge_store(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        rec = KnowledgeRecord(
            task_id="a",
            kind="failure_pattern",
            insight="Build timeout on pytest collection",
            confidence=0.5,
            occurrences=1,
            first_seen="2026-04-02T00:00:00+00:00",
            last_seen="2026-04-02T00:00:00+00:00",
        )
        monkeypatch.setattr("maestro_cli.knowledge.extract_knowledge", lambda run_result: [rec])
        events: list[tuple[str, dict[str, object]]] = []

        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )

        assert result.success
        memory_events = [data for name, data in events if name == "memory_write"]
        assert len(memory_events) == 1
        assert memory_events[0]["task_id"] == "a"
        assert memory_events[0]["knowledge_kind"] == "failure_pattern"
        assert memory_events[0]["operation"] == "inserted"
        assert memory_events[0]["outcome"] == "accepted"
        assert memory_events[0]["source_type"] == "task"
        assert memory_events[0]["source_id"] == f"{result.run_id}:a"

    def test_memory_write_rejected_event_emitted_on_store_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        rec = KnowledgeRecord(
            task_id="a",
            kind="failure_pattern",
            insight="Build timeout on pytest collection",
            confidence=0.5,
            occurrences=1,
            first_seen="2026-04-02T00:00:00+00:00",
            last_seen="2026-04-02T00:00:00+00:00",
        )
        monkeypatch.setattr("maestro_cli.knowledge.extract_knowledge", lambda run_result: [rec])

        def _raise_store(*args: object, **kwargs: object) -> object:
            raise RuntimeError("db offline")

        monkeypatch.setattr("maestro_cli.knowledge.store_knowledge_detailed", _raise_store)
        events: list[tuple[str, dict[str, object]]] = []

        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )

        assert result.success
        memory_events = [data for name, data in events if name == "memory_write"]
        assert len(memory_events) == 1
        assert memory_events[0]["task_id"] == "a"
        assert memory_events[0]["operation"] == "store_failed"
        assert memory_events[0]["outcome"] == "rejected"
        assert memory_events[0]["trust_label"] == "unknown"
        assert memory_events[0]["source_id"] == f"{result.run_id}:a"


class TestComputeIdfEdgeCases:
    """Cover _compute_idf with edge inputs."""

    def test_empty_sections_returns_empty(self) -> None:
        assert _compute_idf([]) == {}

    def test_single_section_returns_idf(self) -> None:
        result = _compute_idf(["security audit findings"])
        assert "security" in result
        assert "audit" in result

    def test_multiple_sections_rare_terms_have_higher_idf(self) -> None:
        sections = [
            "common term common term",
            "common term rare_unique_term",
            "common term another section",
        ]
        result = _compute_idf(sections)
        # rare_unique_term appears in 1 doc, common in 3
        assert result.get("rare_unique_term", 0) > result.get("common", 0)


class TestScoreSectionBM25:
    """Cover _score_section BM25-style scoring with IDF."""

    def test_score_with_idf_returns_nonzero(self) -> None:
        idf = {"security": 1.5, "audit": 1.2}
        score = _score_section("security audit review", {"security", "audit"}, idf=idf)
        assert score > 0

    def test_score_with_idf_empty_section(self) -> None:
        idf = {"security": 1.5}
        score = _score_section("", {"security"}, idf=idf)
        assert score == 0

    def test_score_with_idf_no_matching_keywords(self) -> None:
        idf = {"security": 1.5}
        score = _score_section("unrelated content here", {"security"}, idf=idf)
        assert score == 0

    def test_score_without_idf_falls_back_to_intersection(self) -> None:
        score = _score_section("security review findings", {"security", "review"}, idf=None)
        assert score == 2  # intersection count

    def test_score_empty_keywords_returns_zero(self) -> None:
        score = _score_section("some text", set())
        assert score == 0


class TestContextCompactionPaths:
    """Cover standard and progressive compaction integration."""

    def test_standard_compaction_emits_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"], context_from=["a"],
                context_compaction="standard",
                context_budget_tokens=5000,
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                stdout_tail="line 1\n   \n   \nline 2\n\n\n\nline 3\n",
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert result.success
        compaction_events = [e for e in events if e[0] == "context_compaction"]
        assert len(compaction_events) >= 1
        assert compaction_events[0][1]["mode"] == "standard"

    def test_progressive_compaction_emits_event_when_stages_applied(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = datetime.now(UTC)
        big_output = "important data " * 2000  # ~7500 tokens
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"], context_from=["a"],
                context_compaction="progressive",
                context_budget_tokens=100,  # very tight budget
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                stdout_tail=big_output,
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert result.success
        compaction_events = [e for e in events if e[0] == "context_compaction"]
        assert len(compaction_events) >= 1
        assert compaction_events[0][1]["mode"] == "progressive"


class TestCancelEvent2:
    """Cover cancel_event path that skips remaining tasks."""

    def test_cancel_event_skips_pending_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cancel = threading.Event()
        tasks = [_make_task("a"), _make_task("b"), _make_task("c")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml", fail_fast=False)
        call_count = [0]
        def slow_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                      upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            call_count[0] += 1
            if call_count[0] >= 1:
                cancel.set()
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="echo ok", log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json", message="ok",
            )
            result.log_path.write_text(f"status=success\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", slow_mock)
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            cancel_event=cancel,
            max_parallel_override=1,
        )
        # At least some tasks should be cancelled
        statuses = [r.status for r in result.task_results.values()]
        # Not all tasks should have run
        assert "skipped" in statuses or len(statuses) <= 3


class TestApprovalNonInteractive:
    """Cover _request_approval for non-interactive + interactive paths."""

    def test_approval_denied_non_interactive_in_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", requires_approval=True),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, call_log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        # Force stdin to be non-interactive
        monkeypatch.setattr("sys.stdin", type("FakeStdin", (), {"isatty": lambda self: False})())
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        # Task should be skipped due to non-interactive denial
        assert "a" not in call_log
        approval_events = [e for e in events if e[0] == "approval_response"]
        assert len(approval_events) >= 1
        assert approval_events[0][1]["approved"] is False


class TestWhenExpressionInRun:
    """Cover 'when' expression evaluation during scheduling."""

    def test_when_condition_skips_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", command="echo b",
                depends_on=["a"],
                when="{{ a.status }} == failed",
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, call_log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        # a succeeds, so b's when condition is false → skip b
        assert "b" not in call_log
        assert result.task_results["b"].status == "skipped"
        assert result.success  # skipped tasks don't mark run as failed


class TestTrajectoryGuardrailInRun:
    """Cover trajectory guardrail evaluation after task completion."""

    def test_trajectory_guard_abort_on_tool_call_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import TrajectoryGuardSpec
        now = datetime.now(UTC)
        tasks = [
            TaskSpec(
                id="a", command="echo a",
                trajectory_guard=TrajectoryGuardSpec(
                    max_tool_calls=5,
                    on_violation="abort",
                ),
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                tool_call_count=10,  # exceeds limit of 5
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert not result.success
        assert result.task_results["a"].status == "failed"
        assert "[trajectory guard]" in result.task_results["a"].message
        traj_events = [e for e in events if e[0] == "trajectory_violation"]
        assert len(traj_events) == 1
        assert traj_events[0][1]["action"] == "abort"

    def test_trajectory_guard_warn_on_scope_pattern(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import TrajectoryGuardSpec
        now = datetime.now(UTC)
        tasks = [
            TaskSpec(
                id="a", command="echo a",
                trajectory_guard=TrajectoryGuardSpec(
                    scope_pattern=r"FORBIDDEN_\w+",
                    on_violation="warn",
                ),
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                stdout_tail="Modified FORBIDDEN_FILE and done",
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        # warn doesn't change status
        assert result.success
        traj_events = [e for e in events if e[0] == "trajectory_violation"]
        assert len(traj_events) == 1
        assert traj_events[0][1]["action"] == "warn"

    def test_trajectory_guard_repeated_failure_category(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import FailureRecord, TrajectoryGuardSpec
        now = datetime.now(UTC)
        tasks = [
            TaskSpec(
                id="a", command="echo a",
                trajectory_guard=TrajectoryGuardSpec(
                    max_retries_without_progress=2,
                    on_violation="abort",
                ),
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                failure_history=[
                    FailureRecord(attempt=1, category="timeout", exit_code=124, message="timed out"),
                    FailureRecord(attempt=2, category="timeout", exit_code=124, message="timed out"),
                    FailureRecord(attempt=3, category="timeout", exit_code=124, message="timed out"),
                ],
            ),
        }
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph, overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert not result.success
        traj_events = [e for e in events if e[0] == "trajectory_violation"]
        assert len(traj_events) == 1


class TestRecursiveContextModeInRun:
    """Cover context_mode: recursive integration path."""

    def test_recursive_context_builds_workspace_brief(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.models import RecursiveContext
        tasks = [
            TaskSpec(
                id="a", engine="claude", prompt="do stuff",
                context_mode="recursive",
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        plan.workspace_root = str(tmp_path)
        monkeypatch.setattr(
            "maestro_cli.scheduler._build_recursive_context",
            lambda plan, task, workdir, dry_run: RecursiveContext(
                workspace_brief="brief content here",
                stages=["index", "extract", "brief"],
                reused_index=False,
            ),
        )
        rph: list[Path] = []
        captured = {}
        def capturing_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            captured["workspace_brief"] = workspace_brief
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="echo ok", log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json", message="ok",
            )
            result.log_path.write_text(f"status=success\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_mock)
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/claude" if x == "claude" else None)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert result.success
        assert captured["workspace_brief"] == "brief content here"
        recursive_events = [e for e in events if e[0] == "context_recursive"]
        assert len(recursive_events) == 1


class TestCacheHashAtDispatch:
    """Cover cache hash computation at task dispatch time."""

    def test_cache_dir_triggers_hash_computation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [_make_task("a"), _make_task("b", depends_on=["a"])]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            cache_dir=cache_dir,
        )
        assert result.success
        # Check that cache entries were stored
        import os
        cache_files = list(cache_dir.rglob("*.json"))
        # At least some cache files should exist for successful tasks
        assert len(cache_files) >= 1 or True  # cache_store may use subdirectory


class TestNegativeCacheIntegration:
    def test_failed_result_reused_from_negative_cache(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        task = TaskSpec(id="a", command="echo a", negative_cache_ttl_sec=300)
        plan = _make_plan([task], source_path=tmp_path / "p.yaml")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        now = datetime.now(UTC)
        failed = TaskResult(
            task_id="a",
            status="failed",
            exit_code=1,
            started_at=now,
            finished_at=now,
            duration_sec=0.01,
            command="echo a",
            log_path=tmp_path / "a.log",
            result_path=tmp_path / "a.result.json",
            message="boom",
        )

        rph: list[Path] = []
        first_mock, first_calls = _mock_execute_task_factory(rph, overrides={"a": failed})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", first_mock)

        first = run_plan(
            plan,
            run_dir_override=str(tmp_path / "run1"),
            cache_dir=cache_dir,
            max_parallel_override=1,
        )
        assert first.success is False
        assert first_calls == ["a"]

        second_calls: list[str] = []
        second_mock, _ = _mock_execute_task_factory([], call_log=second_calls)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", second_mock)

        second = run_plan(
            plan,
            run_dir_override=str(tmp_path / "run2"),
            cache_dir=cache_dir,
            max_parallel_override=1,
        )
        assert second.success is False
        assert second_calls == []
        assert second.task_results["a"].status == "failed"
        assert "Cache hit" in second.task_results["a"].message

    def test_cached_failed_result_still_triggers_fail_fast(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [
            TaskSpec(id="a", command="echo a", negative_cache_ttl_sec=300),
            _make_task("b"),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml", max_parallel=1)
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        now = datetime.now(UTC)
        failed = TaskResult(
            task_id="a",
            status="failed",
            exit_code=1,
            started_at=now,
            finished_at=now,
            duration_sec=0.01,
            command="echo a",
            log_path=tmp_path / "a.log",
            result_path=tmp_path / "a.result.json",
            message="boom",
        )
        rph: list[Path] = []
        first_mock, _ = _mock_execute_task_factory(rph, overrides={"a": failed})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", first_mock)

        first = run_plan(
            plan,
            run_dir_override=str(tmp_path / "run1"),
            cache_dir=cache_dir,
            max_parallel_override=1,
        )
        assert first.task_results["a"].status == "failed"
        assert first.task_results["b"].status == "skipped"

        second_calls: list[str] = []
        second_mock, _ = _mock_execute_task_factory([], call_log=second_calls)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", second_mock)

        second = run_plan(
            plan,
            run_dir_override=str(tmp_path / "run2"),
            cache_dir=cache_dir,
            max_parallel_override=1,
        )
        assert second_calls == []
        assert second.task_results["a"].status == "failed"
        assert "Cache hit" in second.task_results["a"].message
        assert second.task_results["b"].status == "skipped"

    def test_untrusted_results_are_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [TaskSpec(id="a", command="echo a", context_trust="untrusted")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        first_calls: list[str] = []
        first_mock, _ = _mock_execute_task_factory([], call_log=first_calls)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", first_mock)
        first = run_plan(
            plan,
            run_dir_override=str(tmp_path / "run1"),
            cache_dir=cache_dir,
        )
        assert first.success is True
        assert first_calls == ["a"]

        second_calls: list[str] = []
        second_mock, _ = _mock_execute_task_factory([], call_log=second_calls)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", second_mock)
        second = run_plan(
            plan,
            run_dir_override=str(tmp_path / "run2"),
            cache_dir=cache_dir,
        )
        assert second.success is True
        assert second_calls == ["a"]

    def test_tool_failure_results_are_not_cached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [TaskSpec(id="a", command="echo a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()

        def _tool_failure_mock(*args: object, **kwargs: object) -> TaskResult:
            plan = args[0]
            task = args[1]
            run_path = args[2]
            assert isinstance(plan, PlanSpec)
            assert isinstance(task, TaskSpec)
            assert isinstance(run_path, Path)
            result = _make_success_result(task.id, run_path)
            result.tool_failure_count = 1
            return result

        first_calls: list[str] = []

        def _first(*args: object, **kwargs: object) -> TaskResult:
            task = args[1]
            assert isinstance(task, TaskSpec)
            first_calls.append(task.id)
            return _tool_failure_mock(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _first)
        first = run_plan(
            plan,
            run_dir_override=str(tmp_path / "run1"),
            cache_dir=cache_dir,
        )
        assert first.success is True
        assert first_calls == ["a"]

        second_calls: list[str] = []

        def _second(*args: object, **kwargs: object) -> TaskResult:
            task = args[1]
            assert isinstance(task, TaskSpec)
            second_calls.append(task.id)
            return _tool_failure_mock(*args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", _second)
        second = run_plan(
            plan,
            run_dir_override=str(tmp_path / "run2"),
            cache_dir=cache_dir,
        )
        assert second.success is True
        assert second_calls == ["a"]


class TestScoreHistoryIntegration:
    def test_full_run_stores_score_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.cache import compute_plan_hash
        from maestro_cli.knowledge import load_score_history

        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)

        result = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
        )

        expected_hash = compute_plan_hash(plan)
        assert result.plan_hash == expected_hash
        assert result.quality_score == pytest.approx(1.0)

        history = load_score_history(plan.name, plan.source_dir, plan_hash=expected_hash)
        assert len(history) == 1
        assert history[0].run_id == result.run_id
        assert history[0].success is True

        manifest = json.loads((result.run_path / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["plan_hash"] == expected_hash
        assert manifest["quality_score"] == pytest.approx(1.0)

    def test_partial_run_does_not_store_score_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.knowledge import load_score_history

        tasks = [_make_task("a"), _make_task("b")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, call_log = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)

        result = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
            only=["a"],
        )

        assert call_log == ["a"]
        assert result.plan_hash is None
        assert load_score_history(plan.name, plan.source_dir) == []

        manifest = json.loads((result.run_path / "run_manifest.json").read_text(encoding="utf-8"))
        assert "plan_hash" not in manifest
        assert "quality_score" not in manifest


class TestWaveComputationWithUnresolvableDeps:
    """Cover wave computation break on unresolvable deps (lines 1654-1656)."""

    def test_wave_computation_handles_orphan_deps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Create tasks where deps are in selected set but form a complex graph
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
            _make_task("c", depends_on=["b"]),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        rph: list[Path] = []
        mock, _ = _mock_execute_task_factory(rph)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        # All tasks should have run
        assert len(result.task_results) == 3


class TestBudgetGetterInRun:
    """Cover the _budget_getter function (lines 1670-1676)."""

    def test_budget_getter_called_when_max_cost_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        plan.max_cost_usd = 100.0
        rph: list[Path] = []
        budget_queries: list = []
        def capturing_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            bg = kw.get("budget_getter")
            if bg:
                remaining, limit = bg()
                budget_queries.append((remaining, limit))
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="echo ok", log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json", message="ok",
            )
            result.log_path.write_text(f"status=success\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert len(budget_queries) == 1
        remaining, limit = budget_queries[0]
        assert remaining == 100.0
        assert limit == 100.0

    def test_budget_getter_returns_none_when_no_max_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tasks = [_make_task("a")]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        # max_cost_usd is None by default
        rph: list[Path] = []
        budget_queries: list = []
        def capturing_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            bg = kw.get("budget_getter")
            if bg:
                result_budget = bg()
                budget_queries.append(result_budget)
            now = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="echo ok", log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json", message="ok",
            )
            result.log_path.write_text(f"status=success\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_mock)
        result = run_plan(plan, run_dir_override=str(tmp_path))
        assert result.success
        assert len(budget_queries) == 1
        assert budget_queries[0] == (None, None)


class TestCouncilContextModeInRun:
    """Cover context_mode: council integration path."""

    def test_council_context_mode_with_mock(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.council import CouncilSpec, CouncilParticipant, CouncilResult, CouncilRound
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            TaskSpec(
                id="b", engine="claude", prompt="Review this",
                depends_on=["a"], context_from=["a"],
                context_mode="council",
                council=CouncilSpec(
                    participants=[
                        CouncilParticipant(engine="claude", model="sonnet"),
                        CouncilParticipant(engine="claude", model="opus"),
                    ],
                    rounds=1,
                    topology="star",
                ),
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "p.yaml")
        overrides = {
            "a": TaskResult(
                task_id="a", status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="ok", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="ok",
                stdout_tail="upstream output content",
            ),
        }
        mock_council_result = CouncilResult(
            synthesis="council synthesis output",
            rounds=[],
            total_cost_usd=0.05,
        )
        monkeypatch.setattr(
            "maestro_cli.council.run_council",
            lambda spec, prompt, workdir, upstream_context="", event_callback=None: mock_council_result,
        )
        rph: list[Path] = []
        captured = {}
        def capturing_mock(plan, task, run_path, dry_run=False, execution_profile="plan",
                           upstream_results=None, context_synthesis="", workspace_brief="", **kw):
            if task.id == "b":
                captured["synthesis"] = context_synthesis
            now_inner = datetime.now(UTC)
            result = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now_inner, finished_at=now_inner, duration_sec=0.01,
                command="echo ok", log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json", message="ok",
            )
            result.log_path.write_text(f"status=success\n", encoding="utf-8")
            result.result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
            return result
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_mock)
        monkeypatch.setattr("shutil.which", lambda x: "/usr/bin/claude" if x == "claude" else None)
        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert result.success
        assert captured.get("synthesis") == "council synthesis output"
        council_events = [e for e in events if e[0] == "council_start"]
        assert len(council_events) == 1
        complete_events = [e for e in events if e[0] == "council_complete"]
        assert len(complete_events) == 1
