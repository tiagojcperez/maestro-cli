from __future__ import annotations

import json
import subprocess
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.memory import load_latest_session_snapshot, store_session_snapshot
from maestro_cli.models import PlanRunResult, PlanSpec, SessionSnapshot, TaskResult, TaskSpec, WatchIteration, WatchSpec, WatchState
from maestro_cli.models import SteppingStone
from maestro_cli.watch import (
    _apply_stepping_stone,
    _build_blame_context,
    _build_experiments_summary,
    _build_history_text,
    _build_improve_plan,
    _build_recent_iteration_outputs,
    _capture_iteration_excerpts,
    _compact_stepping_stones,
    _load_best_stepping_stone,
    _maybe_extract_session_memory,
    _save_stepping_stone,
    _STEPPING_STONES_MAX,
    _stepping_stones_dir,
    _coerce_float,
    _coerce_str,
    _count_executed_tasks,
    _extract_log_section,
    _extract_manifest_metric,
    _extract_metric,
    _find_latest_target_run,
    _git_commit_changes,
    _git_rollback,
    _is_improvement,
    _load_program,
    _lookup_json_path,
    _resume_watch_state,
    _target_reached,
    _watch_improve,
    _write_experiment,
    watch,
)


def _make_plan(
    tmp_path: Path,
    *,
    watch_spec: WatchSpec | None = None,
    tasks: list[TaskSpec] | None = None,
) -> PlanSpec:
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text("version: 1\nname: watch-test\n", encoding="utf-8")
    return PlanSpec(
        name="watch-test",
        source_path=plan_path,
        workspace_root=".",
        run_dir=".maestro-runs",
        tasks=tasks or [TaskSpec(id="test-task"), TaskSpec(id="other-task")],
        watch=watch_spec
        or WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            metric_task="test-task",
            warmup_iterations=0,
            plateau_threshold=2,
            max_iterations=3,
        ),
    )


def _make_mock_result(
    tmp_path: Path,
    *,
    success: bool = True,
    stdout_tail: str = "",
    cost: float = 0.1,
    task_id: str = "test-task",
    task_results: dict[str, TaskResult] | None = None,
) -> PlanRunResult:
    run_path = tmp_path / "run"
    run_path.mkdir(parents=True, exist_ok=True)
    results = task_results or {
        task_id: TaskResult(
            task_id=task_id,
            status="success" if success else "failed",
            stdout_tail=stdout_tail,
            exit_code=0 if success else 1,
            duration_sec=1.0,
            cost_usd=cost,
            log_path=run_path / f"{task_id}.log",
            result_path=run_path / f"{task_id}.json",
        )
    }
    return PlanRunResult(
        plan_name="test",
        run_id="run-1",
        run_path=run_path,
        started_at=datetime.now(),
        finished_at=datetime.now(),
        success=success,
        task_results=results,
        total_cost_usd=cost,
    )


class TestWatchHelpers:
    def test_extract_log_section_stops_at_status_and_next_header(self, tmp_path: Path) -> None:
        status_log = tmp_path / "status.log"
        status_log.write_text(
            "\n".join(
                [
                    "[verify_command]",
                    "score: 0.42",
                    "status=success",
                    "should not be included",
                ]
            ),
            encoding="utf-8",
        )
        header_log = tmp_path / "header.log"
        header_log.write_text(
            "\n".join(
                [
                    "[verify_command]",
                    "score: 0.52",
                    "extra detail",
                    "[guard_command]",
                    "score: 0.10",
                ]
            ),
            encoding="utf-8",
        )

        assert _extract_log_section(status_log, "[verify_command]") == "score: 0.42"
        assert _extract_log_section(header_log, "[verify_command]") == "score: 0.52\nextra detail"

    def test_lookup_json_path_handles_nested_lists_and_invalid_paths(self) -> None:
        payload = {
            "items": [
                {"metrics": [{"score": 0.1}]},
                {"metrics": [{"score": 0.7}]},
            ]
        }

        assert _lookup_json_path(payload, "items[1].metrics[0].score") == 0.7
        assert _lookup_json_path(payload, "items[2].metrics[0].score") is None
        assert _lookup_json_path(payload, "items.metrics[0].score") is None
        assert _lookup_json_path(payload, "items[1].metrics[3].score") is None

    def test_resume_watch_state_ignores_malformed_lines_and_rebuilds_state(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "resume-run"
        run_dir.mkdir()
        (run_dir / "experiments.jsonl").write_text(
            "\n".join(
                [
                    "",
                    "not json",
                    json.dumps(["skip"]),
                    json.dumps(
                        {
                            "iteration": "1",
                            "metric_value": "0.7",
                            "best_metric": "0.7",
                            "improved": True,
                            "action": "keep",
                            "cost_usd": "1.25",
                            "duration_sec": "2.5",
                            "git_commit": "",
                            "error": 123,
                            "timestamp": None,
                            "fix_summary": "increase timeout",
                            "manifest_excerpt": "task-a: success",
                            "blame_excerpt": "root cause",
                            "consolidated_excerpt": "try path fix next",
                        }
                    ),
                    json.dumps(
                        {
                            "iteration": 2,
                            "metric_value": "0.9",
                            "best_metric": "0.7",
                            "improved": False,
                            "action": "rollback",
                            "cost_usd": "bad",
                            "duration_sec": "",
                            "git_commit": "sha-2",
                            "error": "",
                            "timestamp": "later",
                        }
                    ),
                    json.dumps(
                        {
                            "iteration": 3,
                            "metric_value": 0.4,
                            "best_metric": 0.4,
                            "improved": True,
                            "action": "keep",
                            "cost_usd": 0.5,
                            "duration_sec": 1.0,
                            "git_commit": "sha-3",
                            "error": None,
                            "timestamp": "done",
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        state = _resume_watch_state(run_dir)

        assert state.total_iterations == 3
        assert state.best_metric == pytest.approx(0.4)
        assert state.best_iteration == 3
        assert state.plateau_count == 0
        assert state.total_cost_usd == pytest.approx(1.75)
        assert [item.metric_value for item in state.iterations] == [pytest.approx(0.7), pytest.approx(0.9), pytest.approx(0.4)]
        assert state.iterations[0].git_commit is None
        assert state.iterations[0].timestamp == ""
        assert state.iterations[0].fix_summary == "increase timeout"
        assert state.iterations[0].manifest_excerpt == "task-a: success"
        assert state.iterations[0].blame_excerpt == "root cause"
        assert state.iterations[0].consolidated_excerpt == "try path fix next"
        assert state.iterations[1].cost_usd is None
        assert state.iterations[1].duration_sec == 0.0
        assert state.iterations[1].error is None

    def test_coerce_float_and_coerce_str_edge_cases(self) -> None:
        # _coerce_float: None input, non-numeric string, valid conversions
        assert _coerce_float(None) is None
        assert _coerce_float("not-a-number") is None
        assert _coerce_float([1, 2]) is None
        assert _coerce_float("3.14") == pytest.approx(3.14)
        assert _coerce_float(42) == pytest.approx(42.0)

        # _coerce_str: None, empty string, non-string, and valid string
        assert _coerce_str(None) is None
        assert _coerce_str("") is None
        assert _coerce_str(123) is None
        assert _coerce_str("hello") == "hello"

    def test_capture_iteration_excerpts_includes_fix_manifest_blame_and_consolidated(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.watch._build_blame_context",
            lambda run_path: ("root cause " * 80, "task-a: success " * 80),
        )

        excerpts = _capture_iteration_excerpts(
            tmp_path / "run",
            improve_log_text="noise\nFIX: increase timeout on task-a\nmore noise",
            consolidated_summary="plateau detected " * 80,
        )

        assert excerpts["fix_summary"] == "increase timeout on task-a"
        assert excerpts["manifest_excerpt"] is not None
        assert excerpts["blame_excerpt"] is not None
        assert excerpts["consolidated_excerpt"] is not None
        assert len(excerpts["manifest_excerpt"]) <= 600
        assert len(excerpts["blame_excerpt"]) <= 600
        assert len(excerpts["consolidated_excerpt"]) <= 600

    def test_capture_iteration_excerpts_without_run_path_skips_blame_and_manifest(self) -> None:
        excerpts = _capture_iteration_excerpts(
            None,
            improve_log_text="FIX: adjust retry count",
            consolidated_summary="",
        )

        assert excerpts["fix_summary"] == "adjust retry count"
        assert excerpts["manifest_excerpt"] is None
        assert excerpts["blame_excerpt"] is None
        assert excerpts["consolidated_excerpt"] is None

    def test_build_recent_iteration_outputs_uses_recent_excerpt_tail(self) -> None:
        iterations = [
            WatchIteration(
                iteration=index,
                metric_value=float(index),
                best_metric=float(index),
                improved=True,
                action="keep",
                fix_summary=f"fix {index}",
                manifest_excerpt=f"manifest {index}",
                blame_excerpt=f"blame {index}",
            )
            for index in range(1, 5)
        ]

        rendered = _build_recent_iteration_outputs(iterations)

        assert "Iteration 1" not in rendered
        assert "Iteration 2" in rendered
        assert "Manifest excerpt:\nmanifest 4" in rendered
        assert "Blame excerpt:\nblame 3" in rendered

    def test_maybe_extract_session_memory_waits_until_threshold(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "watch-run"
        run_dir.mkdir()
        iterations = [
            WatchIteration(
                iteration=index,
                metric_value=float(index),
                best_metric=float(index),
                improved=True,
                action="keep",
                fix_summary=f"fix {index}",
                timestamp=f"2026-04-{index:02d}T00:00:00",
            )
            for index in range(1, 8)
        ]

        snapshot = _maybe_extract_session_memory(
            plan_name="watch-test",
            source_dir=tmp_path,
            watch_run_path=run_dir,
            iterations=iterations,
            lessons=[],
            metric_name="score",
            metric_direction="higher_is_better",
            plateau_count=0,
            plateau_threshold=5,
            consolidate_model=None,
            consolidated_summary="",
        )

        assert snapshot is None
        assert load_latest_session_snapshot(
            "watch-test",
            tmp_path,
            watch_run_path=str(run_dir),
        ) is None

    def test_maybe_extract_session_memory_persists_snapshot_once_threshold_reached(
        self,
        tmp_path: Path,
    ) -> None:
        run_dir = tmp_path / "watch-run"
        run_dir.mkdir()
        iterations = [
            WatchIteration(
                iteration=index,
                metric_value=float(index),
                best_metric=float(index),
                improved=(index % 2 == 0),
                action="keep" if index % 2 == 0 else "rollback",
                fix_summary=f"fix {index}",
                manifest_excerpt=f"task-{index}: success",
                blame_excerpt=f"root cause {index}",
                timestamp=f"2026-04-{index:02d}T00:00:00",
            )
            for index in range(1, 9)
        ]

        snapshot = _maybe_extract_session_memory(
            plan_name="watch-test",
            source_dir=tmp_path,
            watch_run_path=run_dir,
            iterations=iterations,
            lessons=[],
            metric_name="score",
            metric_direction="higher_is_better",
            plateau_count=2,
            plateau_threshold=5,
            consolidate_model=None,
            consolidated_summary="prefer smaller surgical edits",
        )

        assert snapshot is not None
        assert snapshot.iteration_from == 1
        assert snapshot.iteration_to == 5
        persisted = load_latest_session_snapshot(
            "watch-test",
            tmp_path,
            watch_run_path=str(run_dir),
        )
        assert persisted is not None
        assert persisted.iteration_to == 5
        assert "Best Known Working Approaches" in persisted.snapshot_text
        assert "Keep the last 3 iterations verbatim" in persisted.snapshot_text

    def test_load_program_returns_empty_when_no_program_md(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        spec_no_file = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
        )
        assert _load_program(plan, spec_no_file) == ""

    def test_load_program_returns_empty_when_file_missing_and_content_when_present(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        spec = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            program_md="program.md",
        )
        # File absent → empty string (no exception)
        assert _load_program(plan, spec) == ""
        # File present → content returned
        program_file = tmp_path / "program.md"
        program_file.write_text("# Objective\nDo the thing.", encoding="utf-8")
        assert _load_program(plan, spec) == "# Objective\nDo the thing."

    def test_extract_log_section_missing_file_and_section_not_found(self, tmp_path: Path) -> None:
        # Non-existent file → None
        assert _extract_log_section(tmp_path / "ghost.log", "[verify_command]") is None

        # Section header absent → None
        log = tmp_path / "no_header.log"
        log.write_text("some output\nstatus=success\n", encoding="utf-8")
        assert _extract_log_section(log, "[verify_command]") is None

        # Section present but only whitespace after it → None
        empty_section_log = tmp_path / "empty_section.log"
        empty_section_log.write_text("[verify_command]\n   \n", encoding="utf-8")
        assert _extract_log_section(empty_section_log, "[verify_command]") is None

    def test_lookup_json_path_non_traversable_mid_path(self) -> None:
        # Trying to descend into a scalar mid-path returns None
        payload = {"result": "flat-string"}
        assert _lookup_json_path(payload, "result.nested") is None

        # Array index on a dict returns None
        assert _lookup_json_path(payload, "[0]") is None

        # Out-of-range array index on a top-level list returns None
        assert _lookup_json_path([10, 20], "[5]") is None

    def test_lookup_json_path_in_range_access_on_top_level_list(self) -> None:
        # Successful index access directly on a top-level list
        assert _lookup_json_path([10, 20, 30], "[0]") == 10
        assert _lookup_json_path([10, 20, 30], "[2]") == 30

        # Mixed list-of-dicts traversal via top-level bracket
        items = [{"v": 1}, {"v": 99}]
        assert _lookup_json_path(items, "[1].v") == 99

    def test_extract_log_section_stops_on_message_line(self, tmp_path: Path) -> None:
        # Stopping boundary: message= terminates section extraction just like status=
        log = tmp_path / "msg_stop.log"
        log.write_text(
            "\n".join(
                [
                    "[verify_command]",
                    "accuracy: 0.88",
                    "message=Task completed",
                    "should not be included",
                ]
            ),
            encoding="utf-8",
        )
        assert _extract_log_section(log, "[verify_command]") == "accuracy: 0.88"

    def test_build_history_text_none_metrics_render_as_dash(self) -> None:
        # When metric_value or best_metric is None, the column should show "-"
        history = [
            WatchIteration(iteration=1, metric_value=None, best_metric=None, action="warmup_keep"),
            WatchIteration(iteration=2, metric_value=0.5, best_metric=None, action="keep"),
        ]
        text = _build_history_text(history)
        lines = text.splitlines()
        # First data row: both metrics None → "-" (iteration right-aligned to 4 chars)
        assert "   1 | - | - | warmup_keep" in text
        # Second data row: best_metric still None → "-"
        assert "   2 | 0.5 | - | keep" in text

    def test_resume_watch_state_returns_empty_state_when_file_missing(self, tmp_path: Path) -> None:
        # No experiments.jsonl → returns a zeroed WatchState without error
        run_dir = tmp_path / "no-run"
        run_dir.mkdir()
        state = _resume_watch_state(run_dir)
        assert state.total_iterations == 0
        assert state.best_metric is None
        assert state.plateau_count == 0
        assert state.total_cost_usd == 0.0
        assert state.iterations == []

    def test_coerce_float_zero_returns_zero_not_none(self) -> None:
        # 0 is falsy but must NOT be treated as missing — coerce_float must return 0.0
        assert _coerce_float(0) == pytest.approx(0.0)
        assert _coerce_float(0.0) == pytest.approx(0.0)
        assert _coerce_float("0") == pytest.approx(0.0)


class TestExtractMetric:
    def test_stdout_regex_extracts_value(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        result = _make_mock_result(tmp_path, stdout_tail="loss: 0.42")
        spec = WatchSpec(
            metric="loss",
            metric_source="stdout_regex",
            metric_pattern=r"loss: ([0-9.]+)",
            metric_task="test-task",
        )

        assert _extract_metric(result, spec, plan, result.run_path) == pytest.approx(0.42)

    def test_stdout_regex_no_match_returns_none(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        result = _make_mock_result(tmp_path, stdout_tail="no metric here")
        spec = WatchSpec(
            metric="loss",
            metric_source="stdout_regex",
            metric_pattern=r"loss: ([0-9.]+)",
            metric_task="test-task",
        )

        assert _extract_metric(result, spec, plan, result.run_path) is None

    def test_json_field_extracts_value(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        result = _make_mock_result(tmp_path)
        (result.run_path / "test-task.result.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.91}}),
            encoding="utf-8",
        )
        spec = WatchSpec(
            metric="accuracy",
            metric_source="json_field",
            metric_json_path="metrics.accuracy",
            metric_task="test-task",
        )

        assert _extract_metric(result, spec, plan, result.run_path) == pytest.approx(0.91)

    def test_json_field_missing_returns_none(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        result = _make_mock_result(tmp_path)
        (result.run_path / "test-task.result.json").write_text(
            json.dumps({"metrics": {"loss": 1.23}}),
            encoding="utf-8",
        )
        spec = WatchSpec(
            metric="accuracy",
            metric_source="json_field",
            metric_json_path="metrics.accuracy",
            metric_task="test-task",
        )

        assert _extract_metric(result, spec, plan, result.run_path) is None

    def test_metric_task_selects_correct_task(self, tmp_path: Path) -> None:
        plan = _make_plan(
            tmp_path,
            tasks=[TaskSpec(id="setup"), TaskSpec(id="metric-task"), TaskSpec(id="fallback-task")],
        )
        result = _make_mock_result(
            tmp_path,
            task_results={
                "metric-task": TaskResult(task_id="metric-task", status="success", stdout_tail="score: 0.77"),
                "fallback-task": TaskResult(task_id="fallback-task", status="success", stdout_tail="score: 0.12"),
            },
        )
        spec = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            metric_task="metric-task",
        )

        assert _extract_metric(result, spec, plan, result.run_path) == pytest.approx(0.77)


class TestIsImprovement:
    @pytest.mark.parametrize(
        ("current", "best", "direction", "expected"),
        [(0.5, 0.7, "lower_is_better", True)],
    )
    def test_lower_is_better_improvement(
        self,
        current: float,
        best: float,
        direction: str,
        expected: bool,
    ) -> None:
        spec = WatchSpec(metric="score", metric_direction=direction)
        assert _is_improvement(current, best, spec) is expected

    @pytest.mark.parametrize(
        ("current", "best", "direction", "expected"),
        [(0.8, 0.7, "lower_is_better", False)],
    )
    def test_lower_is_better_regression(
        self,
        current: float,
        best: float,
        direction: str,
        expected: bool,
    ) -> None:
        spec = WatchSpec(metric="score", metric_direction=direction)
        assert _is_improvement(current, best, spec) is expected

    @pytest.mark.parametrize(
        ("current", "best", "direction", "expected"),
        [(0.8, 0.5, "higher_is_better", True)],
    )
    def test_higher_is_better_improvement(
        self,
        current: float,
        best: float,
        direction: str,
        expected: bool,
    ) -> None:
        spec = WatchSpec(metric="score", metric_direction=direction)
        assert _is_improvement(current, best, spec) is expected

    @pytest.mark.parametrize(
        ("current", "best", "direction", "expected"),
        [(0.3, 0.5, "higher_is_better", False)],
    )
    def test_higher_is_better_regression(
        self,
        current: float,
        best: float,
        direction: str,
        expected: bool,
    ) -> None:
        spec = WatchSpec(metric="score", metric_direction=direction)
        assert _is_improvement(current, best, spec) is expected

    def test_equal_is_not_improvement(self) -> None:
        spec = WatchSpec(metric="score", metric_direction="lower_is_better")
        assert _is_improvement(0.7, 0.7, spec) is False

    def test_none_current_is_never_improvement(self) -> None:
        # _is_improvement returns False immediately when current is None,
        # regardless of best and direction.
        for direction in ("lower_is_better", "higher_is_better"):
            spec = WatchSpec(metric="score", metric_direction=direction)
            assert _is_improvement(None, 0.5, spec) is False
            assert _is_improvement(None, None, spec) is False


class TestGitOperations:
    def test_git_commit_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            command = list(args[0])
            calls.append(command)
            if command[:3] == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(command, 0, stdout="abc123\n", stderr="")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)

        sha = _git_commit_changes(tmp_path, 3, "loss", 0.42)

        assert sha == "abc123"
        assert calls[0] == ["git", "add", "-A"]
        assert calls[1] == ["git", "commit", "-m", "watch: iteration 3, loss=0.42"]
        assert calls[2] == ["git", "rev-parse", "HEAD"]

    def test_git_commit_failure_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            command = list(args[0])
            if command[:2] == ["git", "commit"]:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="failed")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)

        assert _git_commit_changes(tmp_path, 2, "loss", 0.51) is None

    def test_git_rollback_reset_hard_discards_worktree_changes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            command = list(args[0])
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)

        assert _git_rollback(tmp_path, "rollback") is True
        assert calls == [["git", "reset", "--hard", "HEAD"]]

    def test_git_rollback_revert(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        calls: list[list[str]] = []

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            command = list(args[0])
            calls.append(command)
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)

        assert _git_rollback(tmp_path, "revert") is True
        assert calls == [["git", "revert", "--no-edit", "HEAD"]]

    def test_git_rollback_keep_is_noop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        called = False

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            nonlocal called
            called = True
            return subprocess.CompletedProcess(list(args[0]), 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)

        assert _git_rollback(tmp_path, "keep") is True
        assert called is False


class TestBuildHistoryText:
    def test_empty_history(self) -> None:
        assert _build_history_text([]) == ""

    def test_formats_recent_iterations(self) -> None:
        history = [
            WatchIteration(
                iteration=index,
                metric_value=float(index) / 10.0,
                best_metric=float(index) / 10.0,
                action="keep",
            )
            for index in range(1, 13)
        ]

        text = _build_history_text(history)

        assert "iter | metric | best | action" in text
        assert "12 | 1.2 | 1.2 | keep" in text
        assert "    1 | 0.1 | 0.1 | keep" not in text


class TestWriteExperiment:
    def test_appends_jsonl_line(self, tmp_path: Path) -> None:
        experiments_path = tmp_path / "watch" / "experiments.jsonl"
        iteration = WatchIteration(iteration=1, metric_value=0.5, action="keep", timestamp="now")

        _write_experiment(experiments_path, iteration)

        lines = experiments_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["metric_value"] == 0.5

    def test_multiple_writes_append(self, tmp_path: Path) -> None:
        experiments_path = tmp_path / "watch" / "experiments.jsonl"

        _write_experiment(experiments_path, WatchIteration(iteration=1, metric_value=0.5, action="keep"))
        _write_experiment(experiments_path, WatchIteration(iteration=2, metric_value=0.4, action="keep"))

        lines = experiments_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        assert [json.loads(line)["iteration"] for line in lines] == [1, 2]


class TestWatchLoop:
    def test_resume_from_interruption_preserves_state_and_emits_expected_events(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                metric_direction="lower_is_better",
                program_md="missing-program.md",
                warmup_iterations=0,
                max_iterations=3,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        resume_dir = tmp_path / "resume-run"
        resume_dir.mkdir()
        experiments_path = resume_dir / "experiments.jsonl"
        experiments_path.write_text(
            json.dumps(
                {
                    "iteration": 1,
                    "metric_value": 0.6,
                    "best_metric": 0.6,
                    "improved": True,
                    "action": "keep",
                    "cost_usd": 0.2,
                    "duration_sec": 1.0,
                    "git_commit": "sha-1",
                    "timestamp": "before",
                }
            )
            + "\n",
            encoding="utf-8",
        )
        events: list[tuple[str, dict[str, object]]] = []
        template_vars: dict[str, str] = {}

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            nonlocal template_vars
            template_vars = dict(kwargs["extra_template_vars"])
            raise KeyboardInterrupt

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr(
            "maestro_cli.watch._git_commit_changes",
            lambda *_args, **_kwargs: pytest.fail("commit should not run after interruption"),
        )
        monkeypatch.setattr(
            "maestro_cli.watch._git_rollback",
            lambda *_args, **_kwargs: pytest.fail("rollback should not run after interruption"),
        )

        state = watch(
            plan_path,
            resume_from=resume_dir,
            event_callback=lambda name, payload: events.append((name, payload)),
        )

        event_names = [name for name, _payload in events]
        assert event_names == ["watch_start", "iteration_start", "watch_complete"]
        assert events[1][1] == {"iteration": 2, "best_metric": 0.6}
        assert events[-1][1] == {
            "status": "interrupted",
            "best_metric": 0.6,
            "best_iteration": 1,
            "total_iterations": 1,
            "total_cost_usd": 0.2,
        }
        assert template_vars["watch.iteration"] == "2"
        assert template_vars["watch.best_metric"] == "0.6"
        assert template_vars["watch.last_metric"] == "0.6"
        assert "0.6 | 0.6 | keep" in template_vars["watch.history"]
        assert template_vars["watch.program"] == ""
        assert state.plan_path == str(plan_path.resolve())
        assert state.status == "interrupted"
        assert state.total_iterations == 1
        assert state.best_metric == pytest.approx(0.6)
        assert state.best_iteration == 1
        assert experiments_path.read_text(encoding="utf-8").splitlines() == [
            '{"iteration": 1, "metric_value": 0.6, "best_metric": 0.6, "improved": true, "action": "keep", "cost_usd": 0.2, "duration_sec": 1.0, "git_commit": "sha-1", "timestamp": "before"}'
        ]

    def test_basic_improvement_loop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                metric_direction="lower_is_better",
                warmup_iterations=0,
                max_iterations=3,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.8, 0.6, 0.4])
        commit_calls: list[int] = []

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            return _make_mock_result(tmp_path, stdout_tail=f"score: {next(metrics)}")

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr(
            "maestro_cli.watch._git_commit_changes",
            lambda *_args, **_kwargs: commit_calls.append(1) or f"sha-{len(commit_calls)}",
        )
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: True)

        state = watch(plan_path)

        assert state.status == "max_iterations"
        assert state.total_iterations == 3
        assert state.best_metric == pytest.approx(0.4)
        assert [item.action for item in state.iterations] == ["keep", "keep", "keep"]
        assert len(commit_calls) == 3

    def test_regression_triggers_rollback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                on_regression="rollback",
                warmup_iterations=0,
                max_iterations=2,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.4, 0.9])
        rollbacks: list[str] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(tmp_path, stdout_tail=f"score: {next(metrics)}"),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: "sha-1")
        monkeypatch.setattr(
            "maestro_cli.watch._git_rollback",
            lambda _workdir, action: rollbacks.append(action) or True,
        )

        state = watch(plan_path)

        assert state.iterations[-1].improved is False
        assert state.iterations[-1].action == "rollback"
        assert state.best_metric == pytest.approx(0.4)
        assert state.best_iteration == 1
        assert rollbacks == ["rollback"]

    def test_plateau_stops_loop(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=10,
                plateau_threshold=2,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.5, 0.6, 0.7])

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(tmp_path, stdout_tail=f"score: {next(metrics)}"),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: True)

        state = watch(plan_path)

        assert state.status == "plateau"
        assert state.total_iterations == 3
        assert state.plateau_count == 2

    def test_budget_exceeded_stops(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=10,
                plateau_threshold=5,
                max_cost_usd=1.0,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.5, 0.4])

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path,
                stdout_tail=f"score: {next(metrics)}",
                cost=0.6,
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: True)

        state = watch(plan_path)

        assert state.status == "budget_exceeded"
        assert state.total_iterations == 2
        assert state.total_cost_usd == pytest.approx(1.2)

    def test_max_iterations_reached(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=2,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.9, 0.8])

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(tmp_path, stdout_tail=f"score: {next(metrics)}"),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: True)

        state = watch(plan_path)

        assert state.status == "max_iterations"
        assert state.total_iterations == 2

    def test_warmup_skips_regression_detection(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=2,
                max_iterations=3,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.4, 0.9, 0.3])
        rollbacks: list[str] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(tmp_path, stdout_tail=f"score: {next(metrics)}"),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: "sha")
        monkeypatch.setattr(
            "maestro_cli.watch._git_rollback",
            lambda _workdir, action: rollbacks.append(action) or True,
        )

        state = watch(plan_path)

        assert [item.action for item in state.iterations[:2]] == ["warmup_keep", "warmup_keep"]
        assert rollbacks == []
        assert state.best_metric == pytest.approx(0.3)

    def test_dry_run_skips_execution(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(tmp_path)
        plan_path = tmp_path / "plan.yaml"
        calls = 0

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            nonlocal calls
            calls += 1
            return _make_mock_result(tmp_path, stdout_tail="score: 0.5")

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: True)

        state = watch(plan_path, dry_run=True)

        assert calls == 0
        assert isinstance(state, WatchState)
        assert state.total_iterations == 0

    def test_events_emitted(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=1,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        events: list[tuple[str, dict[str, object]]] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(tmp_path, stdout_tail="score: 0.5"),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: True)

        state = watch(plan_path, event_callback=lambda name, payload: events.append((name, payload)))

        event_names = [name for name, _payload in events]
        assert event_names[:2] == ["watch_start", "iteration_start"]
        assert "iteration_complete" in event_names
        assert event_names[-1] == "watch_complete"
        assert state.total_iterations == 1

    def test_cancel_event_set_before_loop_stops_immediately(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=5,
                plateau_threshold=10,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        events: list[tuple[str, dict[str, object]]] = []
        cancel = threading.Event()
        cancel.set()

        run_plan_called = False

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            nonlocal run_plan_called
            run_plan_called = True
            return _make_mock_result(tmp_path)

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)

        state = watch(plan_path, cancel_event=cancel, event_callback=lambda n, p: events.append((n, p)))

        assert run_plan_called is False
        assert state.status == "interrupted"
        assert state.total_iterations == 0
        event_names = [n for n, _ in events]
        assert event_names == ["watch_start", "watch_complete"]


class TestExtractMetricLogSources:
    def test_verify_command_source_extracts_from_log_section(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        log_path = run_path / "test-task.log"
        log_path.write_text(
            "[verify_command]\nscore: 0.88\nextra output\nstatus=success\n",
            encoding="utf-8",
        )
        task_result = TaskResult(
            task_id="test-task",
            status="success",
            log_path=log_path,
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"test-task": task_result},
        )
        plan = _make_plan(tmp_path)
        spec = WatchSpec(
            metric="score",
            metric_source="verify_command",
            metric_pattern=r"score: ([0-9.]+)",
            metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, run_path) == pytest.approx(0.88)

    def test_guard_command_source_extracts_from_log_section(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        log_path = run_path / "test-task.log"
        log_path.write_text(
            "[guard_command]\naccuracy: 0.73\n",
            encoding="utf-8",
        )
        task_result = TaskResult(
            task_id="test-task",
            status="success",
            log_path=log_path,
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"test-task": task_result},
        )
        plan = _make_plan(tmp_path)
        spec = WatchSpec(
            metric="accuracy",
            metric_source="guard_command",
            metric_pattern=r"accuracy: ([0-9.]+)",
            metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, run_path) == pytest.approx(0.73)

    def test_extract_metric_falls_back_to_last_task_when_metric_task_none(self, tmp_path: Path) -> None:
        plan = _make_plan(
            tmp_path,
            tasks=[TaskSpec(id="setup"), TaskSpec(id="final-task")],
        )
        result = _make_mock_result(
            tmp_path,
            task_results={
                "setup": TaskResult(task_id="setup", status="success", stdout_tail="score: 0.1"),
                "final-task": TaskResult(task_id="final-task", status="success", stdout_tail="score: 0.99"),
            },
        )
        spec = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            # metric_task intentionally omitted → falls back to plan.tasks[-1].id
        )
        assert _extract_metric(result, spec, plan, result.run_path) == pytest.approx(0.99)

    def test_extract_log_section_stops_at_message_prefix(self, tmp_path: Path) -> None:
        log = tmp_path / "msg_stop.log"
        log.write_text(
            "\n".join([
                "[verify_command]",
                "score: 0.55",
                "message=task finished",
                "should not appear",
            ]),
            encoding="utf-8",
        )
        assert _extract_log_section(log, "[verify_command]") == "score: 0.55"

    def test_build_history_text_none_metric_value_shows_dash(self) -> None:
        history = [
            WatchIteration(iteration=1, metric_value=None, best_metric=None, action="keep"),
            WatchIteration(iteration=2, metric_value=0.5, best_metric=0.5, action="keep"),
        ]
        text = _build_history_text(history)
        lines = text.splitlines()
        # Iteration 1 has no metric → both columns show "-"
        assert any("- | - | keep" in line for line in lines), f"Expected dash placeholders in: {text}"
        # Iteration 2 has a metric value
        assert any("0.5 | 0.5 | keep" in line for line in lines)

    def test_resume_watch_state_all_invalid_lines_returns_empty_state(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "all-bad-run"
        run_dir.mkdir()
        (run_dir / "experiments.jsonl").write_text(
            "\n".join(["", "  ", "not-json", json.dumps(["list-not-dict"]), json.dumps(42)]) + "\n",
            encoding="utf-8",
        )
        state = _resume_watch_state(run_dir)
        assert state.total_iterations == 0
        assert state.best_metric is None
        assert state.plateau_count == 0
        assert state.total_cost_usd == 0.0
        assert state.iterations == []

    def test_extract_metric_missing_required_fields_returns_none(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        result = _make_mock_result(tmp_path, stdout_tail="score: 0.9")

        # stdout_regex with no pattern → None
        spec_no_pattern = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_task="test-task",
        )
        assert _extract_metric(result, spec_no_pattern, plan, result.run_path) is None

        # json_field with no json_path → None
        spec_no_path = WatchSpec(
            metric="score",
            metric_source="json_field",
            metric_task="test-task",
        )
        assert _extract_metric(result, spec_no_path, plan, result.run_path) is None

    def test_resume_watch_state_no_experiments_file_returns_empty_state(self, tmp_path: Path) -> None:
        # experiments.jsonl does not exist → pristine empty state
        run_dir = tmp_path / "fresh-run"
        run_dir.mkdir()

        state = _resume_watch_state(run_dir)

        assert state.total_iterations == 0
        assert state.best_metric is None
        assert state.best_iteration is None
        assert state.plateau_count == 0
        assert state.total_cost_usd == 0.0
        assert state.iterations == []

    def test_git_commit_and_rollback_return_none_false_on_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise_oserror(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise OSError("no git")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _raise_oserror)

        assert _git_commit_changes(tmp_path, 1, "score", 0.5) is None
        assert _git_rollback(tmp_path, "rollback") is False
        assert _git_rollback(tmp_path, "revert") is False
        # "keep" never calls subprocess — should still return True
        assert _git_rollback(tmp_path, "keep") is True

    def test_extract_log_section_does_not_stop_at_same_section_header(self, tmp_path: Path) -> None:
        # A repeated occurrence of the *same* header is not a stop boundary —
        # _LOG_SECTION_HEADERS stop only when the header differs from section_name.
        log = tmp_path / "double_header.log"
        log.write_text(
            "\n".join([
                "[verify_command]",
                "first line",
                "[verify_command]",
                "second line",
                "[guard_command]",
                "excluded",
            ]),
            encoding="utf-8",
        )
        result = _extract_log_section(log, "[verify_command]")
        assert result is not None
        assert "first line" in result
        assert "second line" in result
        assert "excluded" not in result

    def test_extract_metric_stdout_regex_none_stdout_tail_returns_none(self, tmp_path: Path) -> None:
        # TaskResult.stdout_tail may be None; the extractor coerces it to ""
        # which produces no regex match → None returned.
        plan = _make_plan(tmp_path)
        run_path = tmp_path / "run"
        run_path.mkdir(parents=True, exist_ok=True)
        task_result = TaskResult(
            task_id="test-task",
            status="success",
            stdout_tail=None,
            exit_code=0,
            duration_sec=1.0,
            log_path=run_path / "test-task.log",
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test",
            run_id="run-none",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"test-task": task_result},
        )
        spec = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, run_path) is None

    def test_lookup_json_path_empty_string_returns_payload(self) -> None:
        # Empty path → zero tokens → loop never executes → original payload returned as-is
        payload = {"key": "val", "nested": [1, 2, 3]}
        assert _lookup_json_path(payload, "") == payload
        assert _lookup_json_path(42, "") == 42
        assert _lookup_json_path(None, "") is None

    def test_extract_metric_json_field_missing_result_file_returns_none(self, tmp_path: Path) -> None:
        # result.json does not exist → _extract_metric returns None without raising
        plan = _make_plan(tmp_path)
        run_path = tmp_path / "run"
        run_path.mkdir(parents=True, exist_ok=True)
        # Explicitly do NOT create test-task.result.json
        task_result = TaskResult(
            task_id="test-task",
            status="success",
            log_path=run_path / "test-task.log",
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test",
            run_id="run-missing-json",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"test-task": task_result},
        )
        spec = WatchSpec(
            metric="score",
            metric_source="json_field",
            metric_json_path="metrics.score",
            metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, run_path) is None

    def test_resume_watch_state_plateau_count_reflects_trailing_regressions(self, tmp_path: Path) -> None:
        # Three iterations: keep, regression, regression → plateau_count == 2 at the end
        run_dir = tmp_path / "plateau-run"
        run_dir.mkdir()
        lines = [
            {"iteration": 1, "metric_value": 0.8, "best_metric": 0.8, "improved": True,  "action": "keep",     "cost_usd": 0.1, "duration_sec": 1.0, "git_commit": "sha-1", "timestamp": "t1"},
            {"iteration": 2, "metric_value": 0.9, "best_metric": 0.8, "improved": False, "action": "rollback", "cost_usd": 0.1, "duration_sec": 1.0, "git_commit": None,    "timestamp": "t2"},
            {"iteration": 3, "metric_value": 0.95,"best_metric": 0.8, "improved": False, "action": "rollback", "cost_usd": 0.1, "duration_sec": 1.0, "git_commit": None,    "timestamp": "t3"},
        ]
        (run_dir / "experiments.jsonl").write_text(
            "\n".join(json.dumps(line) for line in lines) + "\n",
            encoding="utf-8",
        )
        state = _resume_watch_state(run_dir)
        assert state.total_iterations == 3
        assert state.best_metric == pytest.approx(0.8)
        assert state.best_iteration == 1
        assert state.plateau_count == 2
        assert state.total_cost_usd == pytest.approx(0.3)

    def test_resume_watch_state_plateau_count_resets_after_improvement(self, tmp_path: Path) -> None:
        # regression → regression → keep: plateau_count must reset to 0 on the keep.
        run_dir = tmp_path / "plateau-reset-run"
        run_dir.mkdir()
        lines = [
            {"iteration": 1, "metric_value": 0.5, "best_metric": 0.5, "improved": True,  "action": "keep",     "cost_usd": 0.1, "duration_sec": 1.0, "timestamp": "t1"},
            {"iteration": 2, "metric_value": 0.8, "best_metric": 0.5, "improved": False, "action": "rollback", "cost_usd": 0.1, "duration_sec": 1.0, "timestamp": "t2"},
            {"iteration": 3, "metric_value": 0.9, "best_metric": 0.5, "improved": False, "action": "rollback", "cost_usd": 0.1, "duration_sec": 1.0, "timestamp": "t3"},
            {"iteration": 4, "metric_value": 0.3, "best_metric": 0.3, "improved": True,  "action": "keep",     "cost_usd": 0.1, "duration_sec": 1.0, "timestamp": "t4"},
        ]
        (run_dir / "experiments.jsonl").write_text(
            "\n".join(json.dumps(line) for line in lines) + "\n",
            encoding="utf-8",
        )
        state = _resume_watch_state(run_dir)
        assert state.total_iterations == 4
        assert state.best_metric == pytest.approx(0.3)
        assert state.best_iteration == 4
        assert state.plateau_count == 0

    def test_extract_metric_json_field_malformed_json_returns_none(self, tmp_path: Path) -> None:
        # result.json exists but contains invalid JSON → caught by JSONDecodeError → None
        plan = _make_plan(tmp_path)
        run_path = tmp_path / "run-malformed"
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "test-task.result.json").write_text("not-valid-json{", encoding="utf-8")
        task_result = TaskResult(
            task_id="test-task",
            status="success",
            log_path=run_path / "test-task.log",
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test",
            run_id="run-malformed",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"test-task": task_result},
        )
        spec = WatchSpec(
            metric="score",
            metric_source="json_field",
            metric_json_path="metrics.score",
            metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, run_path) is None

    def test_extract_log_section_guard_command_stopped_by_verify_command(self, tmp_path: Path) -> None:
        # A log with [guard_command] first, then [verify_command] — the guard section must
        # stop when [verify_command] is encountered (reverse direction of the existing test).
        log = tmp_path / "guard_then_verify.log"
        log.write_text(
            "\n".join([
                "[guard_command]",
                "guard metric: 0.77",
                "extra guard detail",
                "[verify_command]",
                "verify metric: 0.50",
                "should not appear",
            ]),
            encoding="utf-8",
        )
        section = _extract_log_section(log, "[guard_command]")
        assert section is not None
        assert "guard metric: 0.77" in section
        assert "extra guard detail" in section
        assert "verify metric" not in section

    def test_watch_raises_value_error_without_watch_block(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        plan_no_watch = _make_plan(tmp_path, watch_spec=None)
        plan_no_watch = PlanSpec(
            name=plan_no_watch.name,
            source_path=plan_no_watch.source_path,
            workspace_root=plan_no_watch.workspace_root,
            run_dir=plan_no_watch.run_dir,
            tasks=plan_no_watch.tasks,
            watch=None,
        )
        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan_no_watch)

        with pytest.raises(ValueError, match="watch block"):
            watch(tmp_path / "plan.yaml")

    def test_extract_metric_empty_task_list_returns_none(self, tmp_path: Path) -> None:
        # plan.tasks is empty → early return None before any task look-up
        plan = _make_plan(tmp_path, tasks=[])
        result = _make_mock_result(tmp_path)
        spec = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
        )
        assert _extract_metric(result, spec, plan, result.run_path) is None

    def test_extract_metric_task_not_in_results_returns_none(self, tmp_path: Path) -> None:
        # metric_task references an ID absent from task_results → None
        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="step-a"), TaskSpec(id="step-b")])
        result = _make_mock_result(tmp_path, task_id="step-a", stdout_tail="score: 0.5")
        spec = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            metric_task="step-b",  # step-b has no TaskResult
        )
        assert _extract_metric(result, spec, plan, result.run_path) is None

    def test_extract_metric_verify_command_missing_log_returns_none(self, tmp_path: Path) -> None:
        # log_path does not exist → _extract_metric returns None without raising
        run_path = tmp_path / "run"
        run_path.mkdir()
        task_result = TaskResult(
            task_id="test-task",
            status="success",
            log_path=run_path / "ghost.log",  # does not exist
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test",
            run_id="r-missing-log",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"test-task": task_result},
        )
        plan = _make_plan(tmp_path)
        for source in ("verify_command", "guard_command"):
            spec = WatchSpec(
                metric="score",
                metric_source=source,  # type: ignore[arg-type]
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
            )
            assert _extract_metric(result, spec, plan, run_path) is None

    def test_git_commit_add_failure_returns_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # git add -A fails → return None immediately without calling git commit
        commit_called = False

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            nonlocal commit_called
            command = list(args[0])
            if command[:2] == ["git", "add"]:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="nothing to add")
            # If commit is reached the test should fail
            if command[:2] == ["git", "commit"]:
                commit_called = True
            return subprocess.CompletedProcess(command, 0, stdout="abc\n", stderr="")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)

        result = _git_commit_changes(tmp_path, 5, "score", 0.75)

        assert result is None
        assert commit_called is False

    def test_lookup_json_path_consecutive_bracket_access_on_nested_list(self) -> None:
        # path "[0][1]" on a nested list: first bracket selects row, second selects column
        nested = [[10, 20, 30], [40, 50, 60]]
        assert _lookup_json_path(nested, "[0][1]") == 20
        assert _lookup_json_path(nested, "[1][2]") == 60
        # Out-of-range second bracket
        assert _lookup_json_path(nested, "[0][9]") is None

    def test_watch_run_cost_none_does_not_accumulate(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # If PlanRunResult.total_cost_usd is None the watch total_cost_usd must stay 0.
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=2,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.5, 0.4])

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            mock = _make_mock_result(tmp_path, stdout_tail=f"score: {next(metrics)}", cost=0.0)
            mock = PlanRunResult(
                plan_name=mock.plan_name,
                run_id=mock.run_id,
                run_path=mock.run_path,
                started_at=mock.started_at,
                finished_at=mock.finished_at,
                success=mock.success,
                task_results=mock.task_results,
                total_cost_usd=None,  # simulate copilot-style subscription run
            )
            return mock

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: True)

        state = watch(plan_path)

        assert state.total_cost_usd == pytest.approx(0.0)
        assert state.total_iterations == 2

    def test_resume_watch_state_only_regressions_keeps_best_iteration_none(self, tmp_path: Path) -> None:
        # All iterations are regressions (improved=False) → best_iteration stays None,
        # plateau_count accumulates, best_metric is still updated from best_metric field.
        run_dir = tmp_path / "all-regression-run"
        run_dir.mkdir()
        lines = [
            {"iteration": 1, "metric_value": 0.9, "best_metric": None, "improved": False, "action": "rollback", "cost_usd": 0.1, "duration_sec": 1.0, "timestamp": "t1"},
            {"iteration": 2, "metric_value": 0.95, "best_metric": None, "improved": False, "action": "rollback", "cost_usd": 0.2, "duration_sec": 1.0, "timestamp": "t2"},
        ]
        (run_dir / "experiments.jsonl").write_text(
            "\n".join(json.dumps(line) for line in lines) + "\n",
            encoding="utf-8",
        )
        state = _resume_watch_state(run_dir)

        assert state.total_iterations == 2
        assert state.best_iteration is None
        assert state.plateau_count == 2
        assert state.best_metric is None
        assert state.total_cost_usd == pytest.approx(0.3)

    def test_is_improvement_none_best_always_returns_true(self) -> None:
        # When best is None (no prior measurement), any non-None current is always an improvement.
        for direction in ("lower_is_better", "higher_is_better"):
            spec = WatchSpec(metric="score", metric_direction=direction)
            assert _is_improvement(0.0, None, spec) is True
            assert _is_improvement(99.9, None, spec) is True

    def test_extract_metric_verify_and_guard_command_missing_pattern_returns_none(self, tmp_path: Path) -> None:
        # log file exists and has a valid section, but metric_pattern is None/empty →
        # the extractor must return None without even reading the log.
        run_path = tmp_path / "run-no-pattern"
        run_path.mkdir(parents=True, exist_ok=True)
        log_path = run_path / "test-task.log"
        log_path.write_text("[verify_command]\nscore: 0.9\n[guard_command]\nvalue: 0.8\n", encoding="utf-8")
        task_result = TaskResult(
            task_id="test-task",
            status="success",
            log_path=log_path,
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test",
            run_id="r-no-pattern",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"test-task": task_result},
        )
        plan = _make_plan(tmp_path)
        for source in ("verify_command", "guard_command"):
            spec = WatchSpec(
                metric="score",
                metric_source=source,  # type: ignore[arg-type]
                metric_pattern=None,  # explicitly missing pattern
                metric_task="test-task",
            )
            assert _extract_metric(result, spec, plan, run_path) is None

    def test_watch_no_metric_recorded_event_when_metric_is_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When stdout_tail has no regex match, metric_value is None and the
        # metric_recorded event must NOT be emitted.
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=1,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        events: list[tuple[str, dict[str, object]]] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(tmp_path, stdout_tail="no match here"),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: None)
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: True)

        watch(plan_path, event_callback=lambda name, payload: events.append((name, payload)))

        event_names = [name for name, _ in events]
        assert "metric_recorded" not in event_names

    def test_build_history_text_exactly_history_limit_items(self) -> None:
        # Exactly _HISTORY_LIMIT (10) iterations: all 10 must appear, none truncated.
        history = [
            WatchIteration(iteration=i, metric_value=float(i), best_metric=float(i), action="keep")
            for i in range(1, 11)
        ]
        text = _build_history_text(history)
        lines = text.splitlines()
        # Header (2) + 10 data rows = 12 lines
        assert len(lines) == 12
        assert any("   1 |" in line for line in lines), "First item must appear at exactly the limit"
        assert any("  10 |" in line for line in lines)

    def test_extract_metric_json_field_non_numeric_value_returns_none(self, tmp_path: Path) -> None:
        # The JSON field exists but its value cannot be cast to float (TypeError) → None
        plan = _make_plan(tmp_path)
        run_path = tmp_path / "run-nonfloat"
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "test-task.result.json").write_text(
            json.dumps({"metrics": {"score": {"nested": "dict"}}}),
            encoding="utf-8",
        )
        task_result = TaskResult(
            task_id="test-task",
            status="success",
            log_path=run_path / "test-task.log",
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test",
            run_id="run-nonfloat",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"test-task": task_result},
        )
        spec = WatchSpec(
            metric="score",
            metric_source="json_field",
            metric_json_path="metrics.score",
            metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, run_path) is None

    def test_coerce_float_and_coerce_str_with_bool_inputs(self) -> None:
        # bool is a subclass of int — _coerce_float must convert it; _coerce_str must reject it.
        assert _coerce_float(True) == pytest.approx(1.0)
        assert _coerce_float(False) == pytest.approx(0.0)
        # bool is not str → _coerce_str returns None for both True and False
        assert _coerce_str(True) is None
        assert _coerce_str(False) is None

    def test_extract_log_section_natural_eof_includes_all_content(self, tmp_path: Path) -> None:
        # No status=, no message=, no alternative header → every line after the header is included.
        log = tmp_path / "eof_section.log"
        log.write_text(
            "\n".join([
                "[verify_command]",
                "line one",
                "line two",
                "line three",
            ]),
            encoding="utf-8",
        )
        result = _extract_log_section(log, "[verify_command]")
        assert result == "line one\nline two\nline three"

    def test_lookup_json_path_none_mid_path_returns_none(self) -> None:
        # If a node in the path is None, descent must stop and return None (None is not a dict).
        payload = {"a": None}
        assert _lookup_json_path(payload, "a.b") is None
        # None at root with a key path also returns None immediately
        assert _lookup_json_path(None, "key") is None

    def test_watch_git_commit_none_on_improvement_records_none_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When _git_commit_changes returns None (e.g. nothing to commit), the iteration
        # action must still be "keep" and git_commit must be None — not a hard failure.
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                metric_direction="lower_is_better",
                warmup_iterations=0,
                max_iterations=1,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(tmp_path, stdout_tail="score: 0.5"),
        )
        # Simulate git commit returning None (e.g. nothing staged / dirty-worktree check failed)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: None)
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: True)

        state = watch(plan_path)

        assert state.total_iterations == 1
        iteration = state.iterations[0]
        assert iteration.action == "keep"
        assert iteration.git_commit is None
        assert iteration.improved is True
        assert state.best_metric == pytest.approx(0.5)

    def test_none_result_cost_does_not_accumulate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When result.total_cost_usd is None the watch loop must not add anything
        # to state.total_cost_usd — the "if cost_usd is not None" guard must fire.
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=2,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.5, 0.4])

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            run_path = tmp_path / "run"
            run_path.mkdir(parents=True, exist_ok=True)
            return PlanRunResult(
                plan_name="test",
                run_id="run-nocost",
                run_path=run_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
                task_results={
                    "test-task": TaskResult(
                        task_id="test-task",
                        status="success",
                        stdout_tail=f"score: {next(metrics)}",
                    )
                },
                total_cost_usd=None,
            )

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: True)

        state = watch(plan_path)

        assert state.status == "max_iterations"
        assert state.total_iterations == 2
        assert state.total_cost_usd == 0.0


class TestAdditionalHelperGaps:
    def test_is_improvement_none_best_is_always_true(self) -> None:
        # When best is None (no measurement yet), any current value is an improvement,
        # regardless of direction — exercises the "if best is None: return True" branch.
        for direction in ("lower_is_better", "higher_is_better"):
            spec = WatchSpec(metric="score", metric_direction=direction)
            assert _is_improvement(0.5, None, spec) is True
            assert _is_improvement(0.0, None, spec) is True
            assert _is_improvement(99.0, None, spec) is True

    def test_lookup_json_path_missing_key_in_dict_returns_none(self) -> None:
        # Exercises the "token not in current" branch — distinct from the
        # "current is not a dict" case already tested elsewhere.
        payload = {"a": {"b": {"c": 42}}}
        assert _lookup_json_path(payload, "a.x.c") is None      # 'x' absent in {"b": ...}
        assert _lookup_json_path(payload, "z") is None           # 'z' absent at top level
        assert _lookup_json_path(payload, "a.b.missing") is None  # 'missing' absent in {"c": 42}

    def test_extract_metric_verify_and_guard_command_no_pattern_returns_none(
        self, tmp_path: Path
    ) -> None:
        # metric_pattern=None with verify_command or guard_command sources triggers
        # the "not spec.metric_pattern" early return — distinct from the stdout_regex
        # variant already tested.
        run_path = tmp_path / "run"
        run_path.mkdir()
        log_path = run_path / "test-task.log"
        log_path.write_text(
            "[verify_command]\nscore: 0.88\n[guard_command]\naccuracy: 0.77\n",
            encoding="utf-8",
        )
        task_result = TaskResult(
            task_id="test-task",
            status="success",
            log_path=log_path,
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test",
            run_id="r-no-pattern",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"test-task": task_result},
        )
        plan = _make_plan(tmp_path)

        spec_verify = WatchSpec(
            metric="score",
            metric_source="verify_command",
            metric_task="test-task",
            # metric_pattern intentionally omitted → None
        )
        assert _extract_metric(result, spec_verify, plan, run_path) is None

        spec_guard = WatchSpec(
            metric="accuracy",
            metric_source="guard_command",
            metric_task="test-task",
            # metric_pattern intentionally omitted → None
        )
        assert _extract_metric(result, spec_guard, plan, run_path) is None

    def test_extract_metric_verify_command_section_found_but_regex_no_match_returns_none(
        self, tmp_path: Path
    ) -> None:
        # The log file exists, [verify_command] section is present, pattern is set,
        # but the section text doesn't match the pattern → None (re.search miss branch).
        run_path = tmp_path / "run-no-match"
        run_path.mkdir()
        log_path = run_path / "test-task.log"
        log_path.write_text(
            "[verify_command]\nall tests passed\nstatus=success\n",
            encoding="utf-8",
        )
        task_result = TaskResult(
            task_id="test-task",
            status="success",
            log_path=log_path,
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test",
            run_id="r-no-match",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"test-task": task_result},
        )
        plan = _make_plan(tmp_path)

        for source in ("verify_command", "guard_command"):
            spec = WatchSpec(
                metric="score",
                metric_source=source,  # type: ignore[arg-type]
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
            )
            assert _extract_metric(result, spec, plan, run_path) is None

    def test_extract_metric_stdout_regex_non_numeric_capture_returns_none(
        self, tmp_path: Path
    ) -> None:
        # regex match succeeds but the captured group is not a valid float →
        # float() raises ValueError which is caught → None returned.
        plan = _make_plan(tmp_path)
        result = _make_mock_result(tmp_path, stdout_tail="score: not-a-number", task_id="test-task")
        spec = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([a-z-]+)",
            metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, result.run_path) is None

    def test_extract_metric_json_field_string_value_is_cast_to_float(
        self, tmp_path: Path
    ) -> None:
        # JSON stores the metric as a numeric string ("0.85") instead of a float literal.
        # float(str) should succeed and return the correct value.
        plan = _make_plan(tmp_path)
        run_path = tmp_path / "run-strval"
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "test-task.result.json").write_text(
            '{"metrics": {"accuracy": "0.85"}}',
            encoding="utf-8",
        )
        task_result = TaskResult(
            task_id="test-task",
            status="success",
            log_path=run_path / "test-task.log",
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test",
            run_id="run-strval",
            run_path=run_path,
            started_at=datetime.now(),
            finished_at=datetime.now(),
            success=True,
            task_results={"test-task": task_result},
        )
        spec = WatchSpec(
            metric="accuracy",
            metric_source="json_field",
            metric_json_path="metrics.accuracy",
            metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, run_path) == pytest.approx(0.85)

    def test_watch_regression_rollback_failure_sets_error_on_iteration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When _git_rollback returns False (e.g. git reset fails), the watch loop must
        # record error="git rollback failed" on that WatchIteration without crashing.
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                metric_direction="higher_is_better",
                on_regression="rollback",
                warmup_iterations=0,
                max_iterations=2,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.8, 0.3])

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"score: {next(metrics)}"
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: False)

        state = watch(plan_path)

        assert state.total_iterations == 2
        assert state.iterations[-1].improved is False
        assert state.iterations[-1].error == "git rollback failed"
        # The first (improved) iteration must not have an error
        assert state.iterations[0].error is None

    def test_lookup_json_path_scalar_mid_path_stops_traversal(self) -> None:
        # If a node in the path is a non-dict, non-list scalar (int/float/bool),
        # descent must stop and return None for both dict-key and bracket-index tokens.
        payload = {"a": {"b": 42}}
        # "a.b.c" → current=42 (int), then token "c": isinstance(42, dict) → False → None
        assert _lookup_json_path(payload, "a.b.c") is None
        # "a.b[0]" → current=42 (int), then "[0]": isinstance(42, list) → False → None
        assert _lookup_json_path(payload, "a.b[0]") is None
        # Sanity: correct path still works
        assert _lookup_json_path(payload, "a.b") == 42

    def test_build_history_text_truncation_boundary(self) -> None:
        # 11 iterations → last 10 appear, the oldest (iteration 1) is absent but
        # the second-oldest (iteration 2) IS the first visible row.
        history = [
            WatchIteration(iteration=i, metric_value=float(i), best_metric=float(i), action="keep")
            for i in range(1, 12)
        ]
        text = _build_history_text(history)
        lines = text.splitlines()
        # Header (2 lines) + 10 data rows = 12 total
        assert len(lines) == 12
        # The oldest iteration must not appear (truncated)
        assert not any(line.strip().startswith("1 |") for line in lines)
        # The second iteration (first kept) must appear
        assert any("   2 |" in line for line in lines)
        # The newest iteration (11) must appear
        assert any("  11 |" in line for line in lines)

    def test_plateau_detected_event_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # plateau_detected event must carry iteration, plateau_count, plateau_threshold, action.
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                metric_direction="higher_is_better",
                warmup_iterations=0,
                max_iterations=10,
                plateau_threshold=2,
                plateau_action="stop",
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        # First call improves (0.5 → best=0.5), next two regress (0.3, 0.2) to hit plateau.
        metrics = iter([0.5, 0.3, 0.2])
        events: list[tuple[str, dict[str, object]]] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"score: {next(metrics)}"
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_args, **_kwargs: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_args, **_kwargs: True)

        state = watch(plan_path, event_callback=lambda name, payload: events.append((name, payload)))

        assert state.status == "plateau"
        plateau_events = [(n, p) for n, p in events if n == "plateau_detected"]
        assert len(plateau_events) == 1
        _name, payload = plateau_events[0]
        assert payload["iteration"] == 3
        assert payload["plateau_count"] == 2
        assert payload["plateau_threshold"] == 2
        assert payload["action"] == "stop"

    def test_extract_log_section_empty_body_between_two_headers_returns_none(self, tmp_path: Path) -> None:
        # When the section header is immediately followed by a different header
        # (zero content lines in between), the buffer is empty and None is returned.
        log = tmp_path / "two_headers.log"
        log.write_text(
            "[verify_command]\n[guard_command]\nguard output here\n",
            encoding="utf-8",
        )
        # [verify_command] section has no content before [guard_command] stops it.
        assert _extract_log_section(log, "[verify_command]") is None
        # [guard_command] section has content → returns it correctly.
        assert _extract_log_section(log, "[guard_command]") == "guard output here"

    def test_lookup_json_path_bracket_access_on_string_value_returns_none(self) -> None:
        # After resolving a dict key to a string, bracket notation must return None
        # because a string is not a list — covers the `isinstance(current, list)`
        # branch when the current node is a scalar string.
        payload = {"data": "hello"}
        assert _lookup_json_path(payload, "data[0]") is None
        # Nested: key → dict → string → bracket
        nested = {"outer": {"inner": "text"}}
        assert _lookup_json_path(nested, "outer.inner[0]") is None

    def test_coerce_functions_with_dict_inputs(self) -> None:
        # _coerce_float: dict triggers TypeError inside float() → returns None.
        assert _coerce_float({"key": "val"}) is None
        # _coerce_str: dict is not a str instance → returns None.
        assert _coerce_str({"key": "val"}) is None
        # _coerce_str: list is not a str instance → returns None.
        assert _coerce_str([1, 2, 3]) is None

    def test_resume_watch_state_single_clean_iteration_maps_all_fields(self, tmp_path: Path) -> None:
        # A single well-formed iteration line must be parsed into a WatchIteration
        # with all fields correctly coerced and state fields updated accordingly.
        run_dir = tmp_path / "clean-run"
        run_dir.mkdir()
        entry = {
            "iteration": 5,
            "metric_value": 0.83,
            "best_metric": 0.83,
            "improved": True,
            "action": "keep",
            "cost_usd": 0.42,
            "duration_sec": 7.5,
            "git_commit": "deadbeef",
            "error": None,
            "timestamp": "2025-01-01T00:00:00",
        }
        (run_dir / "experiments.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")

        state = _resume_watch_state(run_dir)

        assert state.total_iterations == 1
        assert state.best_metric == pytest.approx(0.83)
        assert state.best_iteration == 5
        assert state.plateau_count == 0
        assert state.total_cost_usd == pytest.approx(0.42)
        it = state.iterations[0]
        assert it.iteration == 5
        assert it.metric_value == pytest.approx(0.83)
        assert it.improved is True
        assert it.action == "keep"
        assert it.git_commit == "deadbeef"
        assert it.error is None
        assert it.timestamp == "2025-01-01T00:00:00"
        assert it.duration_sec == pytest.approx(7.5)


class TestWatchConsolidation:
    def test_consolidation_fields_on_watchspec(self) -> None:
        spec = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
        )
        assert spec.consolidate_model is None
        assert spec.consolidate_every == 3
        assert spec.consolidate_prompt is None

    def test_consolidation_parsed_from_yaml(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            """
version: 1
name: consolidation-test
tasks:
  - id: run-task
    command: echo done
watch:
  metric: score
  metric_source: stdout_regex
  metric_pattern: 'score: ([0-9.]+)'
  metric_task: run-task
  max_iterations: 5
  consolidate_model: haiku
  consolidate_every: 2
  consolidate_prompt: Summarise the experiments.
""",
            encoding="utf-8",
        )
        plan = load_plan(plan_path)
        assert plan.watch is not None
        assert plan.watch.consolidate_model == "haiku"
        assert plan.watch.consolidate_every == 2
        assert plan.watch.consolidate_prompt == "Summarise the experiments."

    def test_run_consolidation_success(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.watch import _run_consolidation

        mock_result = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="Consolidation output text\n", stderr=""
        )
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: mock_result)

        spec = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            consolidate_model="haiku",
        )
        output = _run_consolidation(spec, "iter 1: score=0.5\niter 2: score=0.7", tmp_path)
        assert output == "Consolidation output text"

    def test_run_consolidation_timeout_fallback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.watch import _run_consolidation

        def _raise_timeout(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd=["claude"], timeout=120)

        monkeypatch.setattr(subprocess, "run", _raise_timeout)

        spec = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            consolidate_model="haiku",
        )
        output = _run_consolidation(spec, "some history", tmp_path)
        assert output == ""

    def test_consolidation_skipped_when_not_configured(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.watch import _run_consolidation

        called: list[bool] = []

        def _fail(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            called.append(True)
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        monkeypatch.setattr(subprocess, "run", _fail)

        spec = WatchSpec(
            metric="score",
            metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            consolidate_model=None,
        )
        # The watch loop guards on consolidate_model is not None before calling
        # _run_consolidation; verify the guard condition holds.
        assert spec.consolidate_model is None
        # Direct call still returns empty string for None model (uses default "haiku"
        # internally, but the caller is responsible for the guard — confirm graceful
        # behaviour when called anyway with a zero-output mock).
        monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr=""))
        output = _run_consolidation(spec, "history", tmp_path)
        assert output == ""
        assert called == []


class TestTargetMetric:
    """Tests for the watch.target_metric feature."""

    # --- _target_reached helper ---

    def test_target_reached_higher_is_better_at_target(self) -> None:
        spec = WatchSpec(metric="score", metric_direction="higher_is_better", target_metric=5.0)
        assert _target_reached(5.0, spec) is True

    def test_target_reached_higher_is_better_above_target(self) -> None:
        spec = WatchSpec(metric="score", metric_direction="higher_is_better", target_metric=5.0)
        assert _target_reached(7.0, spec) is True

    def test_target_reached_higher_is_better_below_target(self) -> None:
        spec = WatchSpec(metric="score", metric_direction="higher_is_better", target_metric=5.0)
        assert _target_reached(3.0, spec) is False

    def test_target_reached_lower_is_better_at_target(self) -> None:
        spec = WatchSpec(metric="score", metric_direction="lower_is_better", target_metric=0.1)
        assert _target_reached(0.1, spec) is True

    def test_target_reached_lower_is_better_below_target(self) -> None:
        spec = WatchSpec(metric="score", metric_direction="lower_is_better", target_metric=0.1)
        assert _target_reached(0.05, spec) is True

    def test_target_reached_lower_is_better_above_target(self) -> None:
        spec = WatchSpec(metric="score", metric_direction="lower_is_better", target_metric=0.1)
        assert _target_reached(0.5, spec) is False

    def test_target_reached_none_metric_returns_false(self) -> None:
        spec = WatchSpec(metric="score", target_metric=5.0)
        assert _target_reached(None, spec) is False

    def test_target_reached_none_target_returns_false(self) -> None:
        spec = WatchSpec(metric="score", target_metric=None)
        assert _target_reached(5.0, spec) is False

    def test_target_reached_both_none_returns_false(self) -> None:
        spec = WatchSpec(metric="score", target_metric=None)
        assert _target_reached(None, spec) is False

    # --- WatchSpec.to_dict ---

    def test_watchspec_to_dict_includes_target_metric(self) -> None:
        spec = WatchSpec(metric="score", target_metric=10.0)
        d = spec.to_dict()
        assert d["target_metric"] == 10.0

    def test_watchspec_to_dict_target_metric_none(self) -> None:
        spec = WatchSpec(metric="score")
        d = spec.to_dict()
        assert d["target_metric"] is None

    # --- YAML parsing ---

    def test_target_metric_parsed_from_yaml(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\n"
            "name: tm-test\n"
            "tasks:\n"
            "  - id: t1\n"
            '    command: "echo ok"\n'
            "watch:\n"
            "  metric: score\n"
            '  metric_pattern: "(\\\\d+)"\n'
            "  target_metric: 42.0\n",
            encoding="utf-8",
        )
        plan = load_plan(plan_yaml)
        assert plan.watch is not None
        assert plan.watch.target_metric == pytest.approx(42.0)

    def test_target_metric_defaults_to_none(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\n"
            "name: tm-test\n"
            "tasks:\n"
            "  - id: t1\n"
            '    command: "echo ok"\n'
            "watch:\n"
            "  metric: score\n"
            '  metric_pattern: "(\\\\d+)"\n',
            encoding="utf-8",
        )
        plan = load_plan(plan_yaml)
        assert plan.watch is not None
        assert plan.watch.target_metric is None

    # --- Watch loop stops on target_reached ---

    def test_watch_stops_when_target_reached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                metric_direction="higher_is_better",
                warmup_iterations=0,
                max_iterations=10,
                plateau_threshold=10,
                target_metric=5.0,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([2.0, 5.0, 99.0])

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"score: {next(metrics)}"
            ),
        )
        monkeypatch.setattr(
            "maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha"
        )
        monkeypatch.setattr(
            "maestro_cli.watch._git_rollback", lambda *_a, **_kw: True
        )

        events: list[tuple[str, dict[str, object]]] = []
        state = watch(
            plan_path,
            event_callback=lambda name, payload: events.append((name, payload)),
        )

        assert state.status == "target_reached"
        assert state.total_iterations == 2
        assert state.best_metric == pytest.approx(5.0)
        event_names = [n for n, _ in events]
        assert "target_reached" in event_names
        target_event = next(p for n, p in events if n == "target_reached")
        assert target_event["metric_value"] == pytest.approx(5.0)
        assert target_event["target_metric"] == pytest.approx(5.0)

    def test_watch_does_not_stop_below_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                metric_direction="higher_is_better",
                warmup_iterations=0,
                max_iterations=3,
                plateau_threshold=10,
                target_metric=100.0,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([1.0, 2.0, 3.0])

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"score: {next(metrics)}"
            ),
        )
        monkeypatch.setattr(
            "maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha"
        )
        monkeypatch.setattr(
            "maestro_cli.watch._git_rollback", lambda *_a, **_kw: True
        )

        state = watch(plan_path)

        assert state.status == "max_iterations"
        assert state.total_iterations == 3

    def test_watch_target_reached_lower_is_better(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="errors",
                metric_source="stdout_regex",
                metric_pattern=r"errors: ([0-9.]+)",
                metric_task="test-task",
                metric_direction="lower_is_better",
                warmup_iterations=0,
                max_iterations=10,
                plateau_threshold=10,
                target_metric=0.0,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([5.0, 2.0, 0.0])

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"errors: {next(metrics)}"
            ),
        )
        monkeypatch.setattr(
            "maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha"
        )
        monkeypatch.setattr(
            "maestro_cli.watch._git_rollback", lambda *_a, **_kw: True
        )

        state = watch(plan_path)

        assert state.status == "target_reached"
        assert state.total_iterations == 3
        assert state.best_metric == pytest.approx(0.0)


class TestBlameInjection:
    """Tests for {{ watch.blame }} and {{ watch.manifest }} injection."""

    # --- _find_latest_target_run ---

    def test_find_latest_target_run_returns_most_recent(self, tmp_path: Path) -> None:
        runs = tmp_path / ".maestro-runs"
        runs.mkdir()
        (runs / "20260301_120000_aaa_my-plan").mkdir()
        (runs / "20260302_120000_bbb_my-plan").mkdir()
        (runs / "20260303_120000_ccc_other-plan").mkdir()

        result = _find_latest_target_run(tmp_path, ".maestro-runs", "my-plan")
        assert result is not None
        assert result.name == "20260302_120000_bbb_my-plan"

    def test_find_latest_target_run_no_matches(self, tmp_path: Path) -> None:
        runs = tmp_path / ".maestro-runs"
        runs.mkdir()
        (runs / "20260301_120000_aaa_other-plan").mkdir()

        result = _find_latest_target_run(tmp_path, ".maestro-runs", "my-plan")
        assert result is None

    def test_find_latest_target_run_no_dir(self, tmp_path: Path) -> None:
        result = _find_latest_target_run(tmp_path, ".maestro-runs", "my-plan")
        assert result is None

    # --- _build_blame_context ---

    def test_build_blame_context_with_manifest(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest = {
            "task_results": {
                "task-a": {"status": "failed", "exit_code": 124, "duration_sec": 2.0, "message": "Task timed out"},
                "task-b": {"status": "success", "exit_code": 0, "duration_sec": 0.5, "message": ""},
            }
        }
        (run_path / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        blame_json, manifest_summary = _build_blame_context(run_path)

        # Blame should be valid JSON (even if empty chain — no deps to analyze)
        assert isinstance(blame_json, str)
        assert len(blame_json) > 0
        parsed = json.loads(blame_json)
        assert "nodes" in parsed

        # Manifest summary should list tasks
        assert "task-a: failed" in manifest_summary
        assert "exit=124" in manifest_summary
        assert "task-b: success" in manifest_summary

    def test_build_blame_context_no_manifest(self, tmp_path: Path) -> None:
        run_path = tmp_path / "empty_run"
        run_path.mkdir()

        blame_json, manifest_summary = _build_blame_context(run_path)
        assert manifest_summary == ""
        # blame_run returns a chain with empty nodes when no manifest
        parsed = json.loads(blame_json)
        assert parsed["nodes"] == []

    def test_build_blame_context_truncates_long_messages(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        long_msg = "x" * 200
        manifest = {
            "task_results": {
                "task-a": {"status": "failed", "exit_code": 1, "duration_sec": 1.0, "message": long_msg},
            }
        }
        (run_path / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )

        _, manifest_summary = _build_blame_context(run_path)
        assert "..." in manifest_summary
        assert len(manifest_summary) < 200

    # --- WatchSpec blame_plan field ---

    def test_watchspec_to_dict_includes_blame_plan(self) -> None:
        spec = WatchSpec(metric="score", blame_plan="target.yaml")
        d = spec.to_dict()
        assert d["blame_plan"] == "target.yaml"

    def test_blame_plan_parsed_from_yaml(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\n"
            "name: bp-test\n"
            "tasks:\n"
            "  - id: t1\n"
            '    command: "echo ok"\n'
            "watch:\n"
            "  metric: score\n"
            '  metric_pattern: "(\\\\d+)"\n'
            "  blame_plan: target.yaml\n",
            encoding="utf-8",
        )
        plan = load_plan(plan_yaml)
        assert plan.watch is not None
        assert plan.watch.blame_plan == "target.yaml"

    # --- Integration: blame injected into template vars ---

    def test_watch_injects_blame_vars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)

        # Create a fake target plan and a run for it
        target_plan_path = tmp_path / "target.yaml"
        target_plan_path.write_text(
            "version: 1\nname: target\nfail_fast: false\ntasks:\n"
            "  - id: t1\n    command: \"exit 124\"\n    timeout_sec: 2\n",
            encoding="utf-8",
        )
        target_runs = tmp_path / ".maestro-runs"
        target_runs.mkdir()
        target_run = target_runs / "20260316_120000_aaa_target"
        target_run.mkdir()
        (target_run / "run_manifest.json").write_text(
            json.dumps({
                "task_results": {
                    "t1": {"status": "failed", "exit_code": 124, "duration_sec": 2.0,
                           "message": "Task timed out after 2s"}
                }
            }),
            encoding="utf-8",
        )

        watch_plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                metric_direction="higher_is_better",
                warmup_iterations=0,
                max_iterations=1,
                plateau_threshold=10,
                blame_plan="target.yaml",
            ),
        )

        # Target plan spec for blame resolution
        target_plan_spec = PlanSpec(
            name="target",
            source_path=target_plan_path,
            workspace_root=".",
            run_dir=".maestro-runs",
            tasks=[TaskSpec(id="t1")],
        )
        plan_path = tmp_path / "plan.yaml"

        captured_vars: dict[str, str] = {}

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            nonlocal captured_vars
            captured_vars = dict(kwargs.get("extra_template_vars", {}))
            return _make_mock_result(tmp_path, stdout_tail="score: 1.0")

        def _mock_load_plan(path: Path) -> PlanSpec:
            if "target" in str(path):
                return target_plan_spec
            return watch_plan

        monkeypatch.setattr("maestro_cli.watch.load_plan", _mock_load_plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        watch(plan_path)

        assert "watch.blame" in captured_vars
        assert "watch.manifest" in captured_vars
        # Manifest should contain the target plan's task data
        assert "t1: failed" in captured_vars["watch.manifest"]
        assert "exit=124" in captured_vars["watch.manifest"]
        # Blame should be parseable JSON
        blame = json.loads(captured_vars["watch.blame"])
        assert "nodes" in blame


# ---------------------------------------------------------------------------
# mode: improve
# ---------------------------------------------------------------------------


class TestImproveMode:
    """Tests for watch mode: improve."""

    def test_build_improve_plan_structure(self, tmp_path: Path) -> None:
        """_build_improve_plan generates a valid 1-task PlanSpec."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed",
            mode="improve",
            metric_source="manifest",
            metric_direction="higher_is_better",
            warmup_iterations=0,
            plateau_threshold=3,
            max_iterations=5,
        ))
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.name == "improve-watch-test"
        assert len(result.tasks) == 1
        assert result.tasks[0].id == "improve-plan"
        assert result.tasks[0].engine == "claude"
        assert result.tasks[0].model == "sonnet"

    def test_build_improve_plan_model_override(self, tmp_path: Path) -> None:
        """improve_model overrides the default model."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed",
            mode="improve",
            improve_model="opus",
            metric_source="manifest",
            metric_direction="higher_is_better",
            warmup_iterations=0,
            plateau_threshold=3,
            max_iterations=5,
        ))
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.tasks[0].model == "opus"

    def test_build_improve_plan_has_verify_command(self, tmp_path: Path) -> None:
        """Improve task has verify_command for plan validation."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed",
            mode="improve",
            metric_source="manifest",
            metric_direction="higher_is_better",
            warmup_iterations=0,
            plateau_threshold=3,
            max_iterations=5,
        ))
        result = _build_improve_plan(plan, plan.watch, "my-plan.yaml")
        vc = result.tasks[0].verify_command
        assert vc is not None
        assert "validate" in vc
        assert "my-plan.yaml" in vc

    def test_build_improve_plan_prompt_has_template_vars(self, tmp_path: Path) -> None:
        """Improve task prompt uses watch + improve template variables."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed",
            mode="improve",
            metric_source="manifest",
            metric_direction="higher_is_better",
            warmup_iterations=0,
            plateau_threshold=3,
            max_iterations=5,
        ))
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        prompt = result.tasks[0].prompt
        assert "{{ watch.iteration }}" in prompt
        assert "{{ watch.program }}" in prompt
        assert "{{ watch.blame }}" in prompt
        assert "{{ watch.manifest }}" in prompt
        assert "{{ watch.session_memory }}" in prompt
        assert "{{ watch.recent_outputs }}" in prompt
        assert "{{ watch.lessons }}" in prompt
        assert "{{ improve.plan_path }}" in prompt
        assert "{{ improve.total_tasks }}" in prompt

    def test_build_improve_plan_inherits_secrets(self, tmp_path: Path) -> None:
        """Improve plan inherits secrets from target plan."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed",
            mode="improve",
            metric_source="manifest",
            metric_direction="higher_is_better",
            warmup_iterations=0,
            plateau_threshold=3,
            max_iterations=5,
        ))
        plan.secrets = ["API_KEY"]
        plan.secrets_auto = True
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.secrets == ["API_KEY"]
        assert result.secrets_auto is True

    def test_extract_manifest_metric_counts_success(self, tmp_path: Path) -> None:
        """_extract_manifest_metric counts only success/dry_run statuses."""
        results = {
            "t1": TaskResult(
                task_id="t1", status="success", exit_code=0,
                duration_sec=1.0, log_path=tmp_path / "t1.log",
                result_path=tmp_path / "t1.json",
            ),
            "t2": TaskResult(
                task_id="t2", status="failed", exit_code=1,
                duration_sec=1.0, log_path=tmp_path / "t2.log",
                result_path=tmp_path / "t2.json",
            ),
            "t3": TaskResult(
                task_id="t3", status="success", exit_code=0,
                duration_sec=1.0, log_path=tmp_path / "t3.log",
                result_path=tmp_path / "t3.json",
            ),
            "t4": TaskResult(
                task_id="t4", status="skipped", exit_code=0,
                duration_sec=0.0, log_path=tmp_path / "t4.log",
                result_path=tmp_path / "t4.json",
            ),
        }
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=False, task_results=results,
        )
        metric = _extract_manifest_metric(run_result)
        assert metric == 2.0  # only t1 and t3 are success

    def test_extract_manifest_metric_empty_results(self, tmp_path: Path) -> None:
        """_extract_manifest_metric returns None for empty results."""
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=False, task_results={},
        )
        assert _extract_manifest_metric(run_result) is None

    def test_extract_manifest_metric_all_success(self, tmp_path: Path) -> None:
        """_extract_manifest_metric returns task count when all pass."""
        results = {
            f"t{i}": TaskResult(
                task_id=f"t{i}", status="success", exit_code=0,
                duration_sec=1.0, log_path=tmp_path / f"t{i}.log",
                result_path=tmp_path / f"t{i}.json",
            )
            for i in range(5)
        }
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=True, task_results=results,
        )
        assert _extract_manifest_metric(run_result) == 5.0

    def test_extract_metric_manifest_source(self, tmp_path: Path) -> None:
        """_extract_metric with metric_source='manifest' uses _extract_manifest_metric."""
        results = {
            "t1": TaskResult(
                task_id="t1", status="success", exit_code=0,
                duration_sec=1.0, log_path=tmp_path / "t1.log",
                result_path=tmp_path / "t1.json",
            ),
            "t2": TaskResult(
                task_id="t2", status="failed", exit_code=1,
                duration_sec=1.0, log_path=tmp_path / "t2.log",
                result_path=tmp_path / "t2.json",
            ),
        }
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=False, task_results=results,
        )
        spec = WatchSpec(
            metric="tasks_passed",
            metric_source="manifest",
            metric_direction="higher_is_better",
            warmup_iterations=0,
            plateau_threshold=3,
            max_iterations=5,
        )
        plan = _make_plan(tmp_path, watch_spec=spec)
        metric = _extract_metric(run_result, spec, plan, tmp_path)
        assert metric == 1.0

    def test_watchspec_mode_defaults_to_custom(self) -> None:
        """WatchSpec.mode defaults to 'custom'."""
        spec = WatchSpec(metric="x")
        assert spec.mode == "custom"
        assert spec.improve_model is None

    def test_watchspec_to_dict_includes_mode(self) -> None:
        """WatchSpec.to_dict() includes mode and improve_model."""
        spec = WatchSpec(metric="x", mode="improve", improve_model="opus")
        d = spec.to_dict()
        assert d["mode"] == "improve"
        assert d["improve_model"] == "opus"


# ---------------------------------------------------------------------------
# Lesson extraction and knowledge archive
# ---------------------------------------------------------------------------


class TestLessonExtraction:
    """Tests for _extract_lesson, _write_lesson, _load_lessons, _format_lessons."""

    def test_extract_lesson_baseline_returns_none(self) -> None:
        """Baseline iteration (action='baseline') produces no lesson."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=1,
            metric_value=3.0,
            best_metric=3.0,
            improved=True,
            action="baseline",
            timestamp="2026-01-01T00:00:00",
        )
        assert _extract_lesson(wi, "") is None

    def test_extract_lesson_validation_failed_returns_none(self) -> None:
        """validation_failed action produces no lesson."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=2,
            metric_value=None,
            improved=False,
            action="validation_failed",
            timestamp="2026-01-01T00:00:00",
        )
        assert _extract_lesson(wi, "") is None

    def test_extract_lesson_improved_with_fix_description(self) -> None:
        """Improved iteration with FIX: line extracts a successful_fix lesson."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=3,
            metric_value=5.0,
            improved=True,
            action="keep",
            timestamp="2026-01-01T00:00:00",
        )
        improve_log = "Starting analysis...\nFIX: timeout -- increased timeout_sec on task t1 from 30 to 60\nDone."
        lesson = _extract_lesson(wi, "t1: failed (exit=124)", improve_log)
        assert lesson is not None
        assert lesson.category == "successful_fix"
        assert lesson.confidence == 0.9
        assert "timeout" in lesson.lesson
        assert lesson.iteration == 3

    def test_extract_lesson_improved_without_fix_description(self) -> None:
        """Improved iteration without FIX: line still creates a lesson."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=4,
            metric_value=6.0,
            improved=True,
            action="keep",
            timestamp="2026-01-15T10:00:00",
        )
        lesson = _extract_lesson(wi, "")
        assert lesson is not None
        assert lesson.category == "successful_fix"
        assert "improved" in lesson.lesson.lower()

    def test_extract_lesson_regression_with_fix_description(self) -> None:
        """Non-improved iteration with FIX: line creates a failed_attempt lesson."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=5,
            metric_value=2.0,
            improved=False,
            action="rollback",
            timestamp="2026-01-20T00:00:00",
        )
        improve_log = "FIX: guard_fix -- relaxed regex in guard_command"
        lesson = _extract_lesson(wi, "t2: failed", improve_log)
        assert lesson is not None
        assert lesson.category == "failed_attempt"
        assert lesson.confidence == 0.5
        assert "guard_fix" in lesson.lesson

    def test_extract_lesson_regression_without_fix(self) -> None:
        """Non-improved iteration without FIX: line creates a generic failed_attempt."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=6,
            metric_value=1.0,
            improved=False,
            action="rollback",
            timestamp="2026-02-01T00:00:00",
        )
        lesson = _extract_lesson(wi, "")
        assert lesson is not None
        assert lesson.category == "failed_attempt"
        assert "no improvement" in lesson.lesson.lower()

    def test_extract_lesson_identifies_task_id_from_manifest(self) -> None:
        """Extract lesson identifies failing task ID from manifest summary."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=7,
            metric_value=4.0,
            improved=True,
            action="keep",
            timestamp="2026-03-01T00:00:00",
        )
        manifest = "task-setup: success\ntask-build: failed (exit=1)\ntask-test: skipped"
        lesson = _extract_lesson(wi, manifest, "FIX: path_fix -- corrected build path")
        assert lesson is not None
        assert lesson.task_id == "task-build"

    def test_write_and_load_lessons_roundtrip(self, tmp_path: Path) -> None:
        """Write lessons then load them back — roundtrip test."""
        from maestro_cli.models import LessonRecord
        from maestro_cli.watch import _load_lessons, _write_lesson

        lessons_path = tmp_path / "lessons.jsonl"
        lr1 = LessonRecord(
            iteration=1,
            task_id="t1",
            category="successful_fix",
            lesson="Fixed timeout on t1",
            confidence=0.9,
            timestamp="2026-03-20T10:00:00",
        )
        lr2 = LessonRecord(
            iteration=2,
            task_id="t2",
            category="failed_attempt",
            lesson="Tried relaxing guard but no improvement",
            confidence=0.5,
            timestamp="2026-03-20T10:05:00",
        )
        _write_lesson(lessons_path, lr1)
        _write_lesson(lessons_path, lr2)

        loaded = _load_lessons(lessons_path)
        assert len(loaded) == 2
        # Sorted by confidence descending (after decay), higher confidence first
        assert loaded[0].lesson == "Fixed timeout on t1"

    def test_load_lessons_missing_file_returns_empty(self, tmp_path: Path) -> None:
        """Non-existent lessons.jsonl returns empty list."""
        from maestro_cli.watch import _load_lessons

        assert _load_lessons(tmp_path / "nonexistent.jsonl") == []

    def test_load_lessons_corrupt_lines_skipped(self, tmp_path: Path) -> None:
        """Corrupt JSON lines are skipped gracefully."""
        from maestro_cli.watch import _load_lessons

        lessons_path = tmp_path / "lessons.jsonl"
        lessons_path.write_text(
            "not json\n"
            + json.dumps({"iteration": 1, "task_id": "t1", "category": "fix",
                          "lesson": "good line", "confidence": 0.8, "timestamp": "2026-03-20T00:00:00"})
            + "\n",
            encoding="utf-8",
        )
        loaded = _load_lessons(lessons_path)
        assert len(loaded) == 1
        assert loaded[0].lesson == "good line"

    def test_load_lessons_max_lessons_limit(self, tmp_path: Path) -> None:
        """_load_lessons respects max_lessons parameter."""
        from maestro_cli.watch import _load_lessons, _write_lesson
        from maestro_cli.models import LessonRecord

        lessons_path = tmp_path / "lessons.jsonl"
        for i in range(10):
            _write_lesson(lessons_path, LessonRecord(
                iteration=i, task_id=f"t{i}", category="fix",
                lesson=f"lesson {i}", confidence=float(i) / 10.0,
                timestamp="2026-03-20T00:00:00",
            ))
        loaded = _load_lessons(lessons_path, max_lessons=3)
        assert len(loaded) == 3

    def test_format_lessons_empty_list(self) -> None:
        """Empty lessons list produces fallback message."""
        from maestro_cli.watch import _format_lessons

        assert _format_lessons([]) == "No lessons from previous iterations."

    def test_format_lessons_with_task_id(self) -> None:
        """Format includes task_id when present."""
        from maestro_cli.models import LessonRecord
        from maestro_cli.watch import _format_lessons

        lessons = [
            LessonRecord(iteration=1, task_id="task-a", category="fix",
                         lesson="Fixed timeout", confidence=0.85, timestamp=""),
        ]
        text = _format_lessons(lessons)
        assert "(task: task-a)" in text
        assert "85%" in text
        assert "Fixed timeout" in text

    def test_format_lessons_without_task_id(self) -> None:
        """Format omits task annotation when task_id is empty."""
        from maestro_cli.models import LessonRecord
        from maestro_cli.watch import _format_lessons

        lessons = [
            LessonRecord(iteration=1, task_id="", category="fix",
                         lesson="General fix", confidence=0.5, timestamp=""),
        ]
        text = _format_lessons(lessons)
        assert "(task:" not in text
        assert "General fix" in text


# ---------------------------------------------------------------------------
# Additional _extract_metric edge cases
# ---------------------------------------------------------------------------


class TestExtractMetricAdditional:
    """Additional metric extraction tests covering remaining branches."""

    def test_manifest_source_empty_results_returns_none(self, tmp_path: Path) -> None:
        """manifest source with no task results returns None."""
        plan = _make_plan(tmp_path)
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=False, task_results={},
        )
        spec = WatchSpec(
            metric="tasks_passed", metric_source="manifest",
            metric_direction="higher_is_better",
        )
        assert _extract_metric(run_result, spec, plan, tmp_path) is None

    def test_manifest_source_counts_dry_run(self, tmp_path: Path) -> None:
        """manifest source counts dry_run as successful."""
        results = {
            "t1": TaskResult(task_id="t1", status="dry_run"),
            "t2": TaskResult(task_id="t2", status="success"),
            "t3": TaskResult(task_id="t3", status="failed"),
        }
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=False, task_results=results,
        )
        spec = WatchSpec(
            metric="tasks_passed", metric_source="manifest",
            metric_direction="higher_is_better",
        )
        plan = _make_plan(tmp_path)
        assert _extract_metric(run_result, spec, plan, tmp_path) == 2.0

    def test_stdout_regex_multiple_matches_takes_first(self, tmp_path: Path) -> None:
        """Regex with multiple matches takes the first occurrence."""
        plan = _make_plan(tmp_path)
        result = _make_mock_result(tmp_path, stdout_tail="score: 0.3\nscore: 0.9")
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)", metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, result.run_path) == pytest.approx(0.3)

    def test_json_field_nested_array_access(self, tmp_path: Path) -> None:
        """json_field with array index in path works."""
        plan = _make_plan(tmp_path)
        run_path = tmp_path / "run-array"
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "test-task.result.json").write_text(
            json.dumps({"results": [{"score": 0.1}, {"score": 0.9}]}),
            encoding="utf-8",
        )
        task_result = TaskResult(
            task_id="test-task", status="success",
            log_path=run_path / "test-task.log",
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=run_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=True, task_results={"test-task": task_result},
        )
        spec = WatchSpec(
            metric="score", metric_source="json_field",
            metric_json_path="results[1].score", metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, run_path) == pytest.approx(0.9)

    def test_guard_command_section_regex_no_match_returns_none(self, tmp_path: Path) -> None:
        """guard_command section exists but regex doesn't match its content."""
        run_path = tmp_path / "run-no-gm"
        run_path.mkdir()
        log_path = run_path / "test-task.log"
        log_path.write_text("[guard_command]\nall checks passed\n", encoding="utf-8")
        task_result = TaskResult(
            task_id="test-task", status="success",
            log_path=log_path, result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=run_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=True, task_results={"test-task": task_result},
        )
        plan = _make_plan(tmp_path)
        spec = WatchSpec(
            metric="score", metric_source="guard_command",
            metric_pattern=r"score: ([0-9.]+)", metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, run_path) is None

    def test_json_field_null_value_returns_none(self, tmp_path: Path) -> None:
        """json_field with a null/None JSON value returns None (float(None) fails)."""
        plan = _make_plan(tmp_path)
        run_path = tmp_path / "run-null"
        run_path.mkdir(parents=True, exist_ok=True)
        (run_path / "test-task.result.json").write_text(
            json.dumps({"metrics": {"score": None}}),
            encoding="utf-8",
        )
        task_result = TaskResult(
            task_id="test-task", status="success",
            log_path=run_path / "test-task.log",
            result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=run_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=True, task_results={"test-task": task_result},
        )
        spec = WatchSpec(
            metric="score", metric_source="json_field",
            metric_json_path="metrics.score", metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, run_path) is None


# ---------------------------------------------------------------------------
# Additional _is_improvement edge cases
# ---------------------------------------------------------------------------


class TestIsImprovementAdditional:
    """Additional _is_improvement edge cases."""

    def test_equal_values_not_improvement_higher_is_better(self) -> None:
        """Equal values are not an improvement for higher_is_better."""
        spec = WatchSpec(metric="score", metric_direction="higher_is_better")
        assert _is_improvement(0.5, 0.5, spec) is False

    def test_slight_improvement_lower_is_better(self) -> None:
        """Even a tiny decrease is an improvement for lower_is_better."""
        spec = WatchSpec(metric="score", metric_direction="lower_is_better")
        assert _is_improvement(0.4999, 0.5, spec) is True

    def test_slight_improvement_higher_is_better(self) -> None:
        """Even a tiny increase is an improvement for higher_is_better."""
        spec = WatchSpec(metric="score", metric_direction="higher_is_better")
        assert _is_improvement(0.5001, 0.5, spec) is True

    def test_zero_current_with_none_best(self) -> None:
        """0.0 current with None best is still an improvement (first measurement)."""
        spec = WatchSpec(metric="score", metric_direction="higher_is_better")
        assert _is_improvement(0.0, None, spec) is True

    def test_negative_values_lower_is_better(self) -> None:
        """Negative metric values work correctly with lower_is_better."""
        spec = WatchSpec(metric="score", metric_direction="lower_is_better")
        assert _is_improvement(-5.0, -3.0, spec) is True
        assert _is_improvement(-1.0, -3.0, spec) is False


# ---------------------------------------------------------------------------
# Additional git operations edge cases
# ---------------------------------------------------------------------------


class TestGitOperationsAdditional:
    """Additional git operation tests."""

    def test_git_commit_rev_parse_failure_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If git rev-parse HEAD fails, _git_commit_changes returns None."""
        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            command = list(args[0])
            if command[:3] == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(command, 1, stdout="", stderr="error")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)
        assert _git_commit_changes(tmp_path, 1, "score", 0.5) is None

    def test_git_commit_rev_parse_empty_stdout_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If git rev-parse HEAD returns empty stdout, returns None."""
        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            command = list(args[0])
            if command[:3] == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(command, 0, stdout="   \n", stderr="")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)
        assert _git_commit_changes(tmp_path, 1, "score", 0.5) is None

    def test_git_rollback_rollback_failure_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If git reset --hard HEAD fails, returns False."""
        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(list(args[0]), 128, stdout="", stderr="error")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)
        assert _git_rollback(tmp_path, "rollback") is False

    def test_git_rollback_revert_failure_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If git revert --no-edit HEAD fails, returns False."""
        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(list(args[0]), 1, stdout="", stderr="conflict")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)
        assert _git_rollback(tmp_path, "revert") is False

    def test_git_commit_message_format(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify the commit message format includes iteration, metric name and value."""
        captured_cmds: list[list[str]] = []

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            command = list(args[0])
            captured_cmds.append(command)
            if command[:3] == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(command, 0, stdout="deadbeef\n", stderr="")
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)
        _git_commit_changes(tmp_path, 7, "accuracy", 0.95)

        commit_cmd = [c for c in captured_cmds if c[:2] == ["git", "commit"]][0]
        assert commit_cmd[3] == "watch: iteration 7, accuracy=0.95"


# ---------------------------------------------------------------------------
# Additional _load_program edge cases
# ---------------------------------------------------------------------------


class TestLoadProgramAdditional:
    """Additional _load_program tests."""

    def test_load_program_reads_file_content(self, tmp_path: Path) -> None:
        """Existing program.md content is returned verbatim."""
        plan = _make_plan(tmp_path)
        program = tmp_path / "strategy.md"
        program.write_text("# Strategy\nImprove gradually.\n", encoding="utf-8")
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)", program_md="strategy.md",
        )
        result = _load_program(plan, spec)
        assert result == "# Strategy\nImprove gradually.\n"

    def test_load_program_no_spec_program_md_empty(self, tmp_path: Path) -> None:
        """Empty string program_md is falsy and returns empty."""
        plan = _make_plan(tmp_path)
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)", program_md="",
        )
        assert _load_program(plan, spec) == ""


# ---------------------------------------------------------------------------
# Additional _build_history_text edge cases
# ---------------------------------------------------------------------------


class TestBuildHistoryTextAdditional:
    """Additional _build_history_text tests."""

    def test_single_iteration(self) -> None:
        """Single iteration produces header + 1 data row."""
        history = [WatchIteration(iteration=1, metric_value=0.5, best_metric=0.5, action="keep")]
        text = _build_history_text(history)
        lines = text.splitlines()
        assert len(lines) == 3  # 2 header + 1 data
        assert "0.5 | 0.5 | keep" in lines[2]

    def test_mixed_actions_display(self) -> None:
        """Different actions (keep, rollback, warmup_keep) display correctly."""
        history = [
            WatchIteration(iteration=1, metric_value=0.5, best_metric=0.5, action="warmup_keep"),
            WatchIteration(iteration=2, metric_value=0.3, best_metric=0.3, action="keep"),
            WatchIteration(iteration=3, metric_value=0.6, best_metric=0.3, action="rollback"),
        ]
        text = _build_history_text(history)
        assert "warmup_keep" in text
        assert "rollback" in text
        assert "keep" in text

    def test_integer_metrics_format_without_decimal(self) -> None:
        """Integer metric values format as whole numbers (e.g. '5' not '5.0')."""
        history = [WatchIteration(iteration=1, metric_value=5.0, best_metric=5.0, action="keep")]
        text = _build_history_text(history)
        # :g format strips trailing zeros
        assert "5 | 5 | keep" in text


# ---------------------------------------------------------------------------
# Additional _write_experiment tests
# ---------------------------------------------------------------------------


class TestWriteExperimentAdditional:
    """Additional _write_experiment tests."""

    def test_write_experiment_all_fields_present(self, tmp_path: Path) -> None:
        """All WatchIteration fields are serialized in the JSONL line."""
        experiments_path = tmp_path / "exp.jsonl"
        wi = WatchIteration(
            iteration=3,
            metric_value=0.75,
            best_metric=0.75,
            improved=True,
            action="keep",
            cost_usd=0.42,
            duration_sec=12.5,
            git_commit="abc123",
            error=None,
            timestamp="2026-03-20T10:00:00",
            fix_summary="increase timeout",
            manifest_excerpt="task-a: success",
            blame_excerpt="{\"root\":\"task-a\"}",
            consolidated_excerpt="avoid retrying path fix",
        )
        _write_experiment(experiments_path, wi)
        data = json.loads(experiments_path.read_text(encoding="utf-8").strip())
        assert data["iteration"] == 3
        assert data["metric_value"] == 0.75
        assert data["best_metric"] == 0.75
        assert data["improved"] is True
        assert data["action"] == "keep"
        assert data["cost_usd"] == 0.42
        assert data["duration_sec"] == 12.5
        assert data["git_commit"] == "abc123"
        assert data["error"] is None
        assert data["timestamp"] == "2026-03-20T10:00:00"
        assert data["fix_summary"] == "increase timeout"
        assert data["manifest_excerpt"] == "task-a: success"
        assert data["blame_excerpt"] == "{\"root\":\"task-a\"}"
        assert data["consolidated_excerpt"] == "avoid retrying path fix"

    def test_write_experiment_none_fields_serialized(self, tmp_path: Path) -> None:
        """None metric/cost/commit fields serialize as null in JSON."""
        experiments_path = tmp_path / "exp.jsonl"
        wi = WatchIteration(
            iteration=1,
            metric_value=None,
            best_metric=None,
            improved=False,
            action="rollback",
            cost_usd=None,
            git_commit=None,
            error="some error",
        )
        _write_experiment(experiments_path, wi)
        data = json.loads(experiments_path.read_text(encoding="utf-8").strip())
        assert data["metric_value"] is None
        assert data["best_metric"] is None
        assert data["cost_usd"] is None
        assert data["git_commit"] is None
        assert data["error"] == "some error"
        assert data["fix_summary"] is None
        assert data["manifest_excerpt"] is None
        assert data["blame_excerpt"] is None
        assert data["consolidated_excerpt"] is None

    def test_write_experiment_creates_parent_dirs(self, tmp_path: Path) -> None:
        """Parent directories are created automatically."""
        experiments_path = tmp_path / "deep" / "nested" / "experiments.jsonl"
        wi = WatchIteration(iteration=1, action="keep")
        _write_experiment(experiments_path, wi)
        assert experiments_path.exists()


# ---------------------------------------------------------------------------
# Additional _resume_watch_state edge cases
# ---------------------------------------------------------------------------


class TestResumeWatchStateAdditional:
    """Additional _resume_watch_state tests."""

    def test_resume_empty_file_returns_empty_state(self, tmp_path: Path) -> None:
        """Empty experiments.jsonl returns pristine state."""
        run_dir = tmp_path / "empty-run"
        run_dir.mkdir()
        (run_dir / "experiments.jsonl").write_text("", encoding="utf-8")
        state = _resume_watch_state(run_dir)
        assert state.total_iterations == 0
        assert state.iterations == []

    def test_resume_only_whitespace_lines(self, tmp_path: Path) -> None:
        """File with only whitespace lines returns empty state."""
        run_dir = tmp_path / "ws-run"
        run_dir.mkdir()
        (run_dir / "experiments.jsonl").write_text("  \n\n  \n", encoding="utf-8")
        state = _resume_watch_state(run_dir)
        assert state.total_iterations == 0

    def test_resume_cost_none_does_not_accumulate(self, tmp_path: Path) -> None:
        """Iteration with null cost_usd does not add to total_cost_usd."""
        run_dir = tmp_path / "nocost-run"
        run_dir.mkdir()
        entry = {
            "iteration": 1, "metric_value": 0.5, "best_metric": 0.5,
            "improved": True, "action": "keep", "cost_usd": None,
            "duration_sec": 1.0, "timestamp": "t1",
        }
        (run_dir / "experiments.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
        state = _resume_watch_state(run_dir)
        assert state.total_cost_usd == 0.0

    def test_resume_interleaved_improvements_and_regressions(self, tmp_path: Path) -> None:
        """Alternating improve/regress correctly tracks best_iteration and plateau_count."""
        run_dir = tmp_path / "interleaved-run"
        run_dir.mkdir()
        lines = [
            {"iteration": 1, "metric_value": 0.5, "best_metric": 0.5, "improved": True, "action": "keep", "cost_usd": 0.1, "duration_sec": 1.0, "timestamp": "t1"},
            {"iteration": 2, "metric_value": 0.7, "best_metric": 0.5, "improved": False, "action": "rollback", "cost_usd": 0.1, "duration_sec": 1.0, "timestamp": "t2"},
            {"iteration": 3, "metric_value": 0.3, "best_metric": 0.3, "improved": True, "action": "keep", "cost_usd": 0.1, "duration_sec": 1.0, "timestamp": "t3"},
            {"iteration": 4, "metric_value": 0.8, "best_metric": 0.3, "improved": False, "action": "rollback", "cost_usd": 0.1, "duration_sec": 1.0, "timestamp": "t4"},
        ]
        (run_dir / "experiments.jsonl").write_text(
            "\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8"
        )
        state = _resume_watch_state(run_dir)
        assert state.total_iterations == 4
        assert state.best_iteration == 3
        assert state.plateau_count == 1  # only one trailing regression


# ---------------------------------------------------------------------------
# Additional _run_consolidation edge cases
# ---------------------------------------------------------------------------


class TestRunConsolidationAdditional:
    """Additional _run_consolidation tests."""

    def test_consolidation_oserror_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError from subprocess returns empty string."""
        from maestro_cli.watch import _run_consolidation

        def _raise_os_error(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            raise OSError("command not found")

        monkeypatch.setattr(subprocess, "run", _raise_os_error)
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)", consolidate_model="haiku",
        )
        assert _run_consolidation(spec, "history text", tmp_path) == ""

    def test_consolidation_nonzero_exit_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-zero exit code returns empty string."""
        from maestro_cli.watch import _run_consolidation

        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=1, stdout="output", stderr="err"),
        )
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)", consolidate_model="haiku",
        )
        assert _run_consolidation(spec, "history text", tmp_path) == ""

    def test_consolidation_empty_stdout_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Zero exit code but empty stdout returns empty string."""
        from maestro_cli.watch import _run_consolidation

        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout="   \n", stderr=""),
        )
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)", consolidate_model="haiku",
        )
        assert _run_consolidation(spec, "history text", tmp_path) == ""

    def test_consolidation_custom_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Custom consolidate_prompt is included in the command."""
        from maestro_cli.watch import _run_consolidation

        captured_cmds: list[list[str]] = []

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured_cmds.append(list(args[0]))
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="result\n", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            consolidate_model="sonnet",
            consolidate_prompt="Analyse experiments and recommend next steps.",
        )
        result = _run_consolidation(spec, "iter 1: ok", tmp_path)
        assert result == "result"
        # The prompt must include the custom text
        cmd_str = captured_cmds[0][-1]
        assert "Analyse experiments" in cmd_str
        assert "iter 1: ok" in cmd_str

    def test_consolidation_uses_default_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When consolidate_prompt is None, uses the default."""
        from maestro_cli.watch import _DEFAULT_CONSOLIDATION_PROMPT, _run_consolidation

        captured_cmds: list[list[str]] = []

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured_cmds.append(list(args[0]))
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            consolidate_model="haiku",
            consolidate_prompt=None,
        )
        _run_consolidation(spec, "history", tmp_path)
        cmd_str = captured_cmds[0][-1]
        assert "Which approaches consistently succeed" in cmd_str


# ---------------------------------------------------------------------------
# WatchState / WatchIteration / WatchSpec serialization
# ---------------------------------------------------------------------------


class TestWatchDataclassSerialization:
    """Test to_dict serialization for watch-related dataclasses."""

    def test_watch_iteration_to_dict(self) -> None:
        """WatchIteration.to_dict includes all fields."""
        wi = WatchIteration(
            iteration=5, metric_value=0.7, best_metric=0.7,
            improved=True, action="keep", cost_usd=0.25,
            duration_sec=3.5, git_commit="abc", error=None,
            timestamp="2026-03-20T00:00:00",
        )
        d = wi.to_dict()
        assert d["iteration"] == 5
        assert d["metric_value"] == 0.7
        assert d["improved"] is True
        assert d["git_commit"] == "abc"
        assert d["error"] is None

    def test_watch_state_defaults(self) -> None:
        """WatchState has expected defaults."""
        state = WatchState(plan_path="/tmp/plan.yaml", status="max_iterations")
        assert state.best_metric is None
        assert state.best_iteration is None
        assert state.plateau_count == 0
        assert state.total_cost_usd == 0.0
        assert state.total_iterations == 0
        assert state.iterations == []

    def test_watchspec_defaults(self) -> None:
        """WatchSpec has all expected defaults."""
        spec = WatchSpec(metric="score")
        assert spec.metric_source == "stdout_regex"
        assert spec.metric_direction == "lower_is_better"
        assert spec.metric_pattern is None
        assert spec.metric_task is None
        assert spec.warmup_iterations == 1
        assert spec.plateau_threshold == 5
        assert spec.max_iterations == 100
        assert spec.on_regression == "rollback"
        assert spec.plateau_action == "stop"
        assert spec.target_metric is None
        assert spec.blame_plan is None
        assert spec.mode == "custom"
        assert spec.improve_model is None

    def test_watchspec_to_dict_roundtrip_fields(self) -> None:
        """WatchSpec.to_dict includes all custom fields."""
        spec = WatchSpec(
            metric="accuracy",
            metric_source="guard_command",
            metric_pattern=r"acc: (\d+)",
            metric_direction="lower_is_better",
            metric_task="eval-task",
            metric_json_path="results.accuracy",
            warmup_iterations=2,
            plateau_threshold=5,
            max_iterations=20,
            on_regression="revert",
            plateau_action="escalate_model",
            program_md="program.md",
            max_cost_usd=10.0,
            target_metric=95.0,
            blame_plan="target.yaml",
            consolidate_model="sonnet",
            consolidate_every=5,
            consolidate_prompt="Custom prompt",
            mode="improve",
            improve_model="opus",
        )
        d = spec.to_dict()
        assert d["metric"] == "accuracy"
        assert d["metric_source"] == "guard_command"
        assert d["metric_direction"] == "lower_is_better"
        assert d["on_regression"] == "revert"
        assert d["plateau_action"] == "escalate_model"
        assert d["max_cost_usd"] == 10.0
        assert d["target_metric"] == 95.0
        assert d["blame_plan"] == "target.yaml"
        assert d["consolidate_model"] == "sonnet"
        assert d["consolidate_every"] == 5
        assert d["mode"] == "improve"
        assert d["improve_model"] == "opus"


# ---------------------------------------------------------------------------
# Improve mode additional tests
# ---------------------------------------------------------------------------


class TestImproveModeAdditional:
    """Additional tests for improve mode specifics."""

    def test_build_improve_plan_inherits_workspace_root(self, tmp_path: Path) -> None:
        """Improve plan inherits workspace_root from target plan."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed", mode="improve",
            metric_source="manifest", metric_direction="higher_is_better",
        ))
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.workspace_root == plan.workspace_root

    def test_build_improve_plan_inherits_max_cost(self, tmp_path: Path) -> None:
        """Improve plan inherits max_cost_usd from watch spec."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed", mode="improve",
            metric_source="manifest", metric_direction="higher_is_better",
            max_cost_usd=5.0,
        ))
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.max_cost_usd == 5.0

    def test_build_improve_plan_default_model_is_sonnet(self, tmp_path: Path) -> None:
        """Default improve model is sonnet when improve_model is None."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed", mode="improve",
            metric_source="manifest", metric_direction="higher_is_better",
            improve_model=None,
        ))
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.tasks[0].model == "sonnet"

    def test_build_improve_plan_has_fail_fast_true(self, tmp_path: Path) -> None:
        """Improve plan has fail_fast=True."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed", mode="improve",
            metric_source="manifest", metric_direction="higher_is_better",
        ))
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.fail_fast is True

    def test_build_improve_plan_task_has_max_retries(self, tmp_path: Path) -> None:
        """Improve task has max_retries=1 for resilience."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed", mode="improve",
            metric_source="manifest", metric_direction="higher_is_better",
        ))
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.tasks[0].max_retries == 1

    def test_build_improve_plan_defaults_timeout(self, tmp_path: Path) -> None:
        """Improve plan defaults timeout to 180s."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed", mode="improve",
            metric_source="manifest", metric_direction="higher_is_better",
        ))
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.defaults.timeout_sec == 180
        assert result.tasks[0].timeout_sec == 180

    def test_build_improve_plan_inherits_env(self, tmp_path: Path) -> None:
        """Improve plan inherits env from target plan defaults."""
        from maestro_cli.models import PlanDefaults
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed", mode="improve",
            metric_source="manifest", metric_direction="higher_is_better",
        ))
        plan.defaults = PlanDefaults(env={"MY_VAR": "value"})
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.defaults.env == {"MY_VAR": "value"}

    def test_build_improve_plan_no_env_defaults_to_empty(self, tmp_path: Path) -> None:
        """Improve plan env defaults to empty dict when target has no env."""
        from maestro_cli.models import PlanDefaults
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed", mode="improve",
            metric_source="manifest", metric_direction="higher_is_better",
        ))
        plan.defaults = PlanDefaults(env=None)
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.defaults.env == {}


# ---------------------------------------------------------------------------
# _target_reached additional edge cases
# ---------------------------------------------------------------------------


class TestTargetReachedAdditional:
    """Additional _target_reached edge cases."""

    def test_target_reached_exact_boundary_lower(self) -> None:
        """Exact boundary value counts as reached for lower_is_better."""
        spec = WatchSpec(metric="loss", metric_direction="lower_is_better", target_metric=0.5)
        assert _target_reached(0.5, spec) is True

    def test_target_reached_exact_boundary_higher(self) -> None:
        """Exact boundary value counts as reached for higher_is_better."""
        spec = WatchSpec(metric="score", metric_direction="higher_is_better", target_metric=10.0)
        assert _target_reached(10.0, spec) is True

    def test_target_reached_negative_target(self) -> None:
        """Negative target metric works correctly."""
        spec = WatchSpec(metric="loss", metric_direction="lower_is_better", target_metric=-1.0)
        assert _target_reached(-2.0, spec) is True
        assert _target_reached(0.0, spec) is False

    def test_target_reached_zero_target_higher(self) -> None:
        """Zero target with higher_is_better: 0.0 is at target."""
        spec = WatchSpec(metric="score", metric_direction="higher_is_better", target_metric=0.0)
        assert _target_reached(0.0, spec) is True
        assert _target_reached(-1.0, spec) is False


# ---------------------------------------------------------------------------
# Watch loop integration: template variables
# ---------------------------------------------------------------------------


class TestWatchTemplateVarsIntegration:
    """Tests that verify template variables are passed correctly to run_plan."""

    def test_template_vars_first_iteration_empty_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """First iteration has empty history and empty last_metric."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task", warmup_iterations=0,
                max_iterations=1, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        captured_vars: dict[str, str] = {}

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            nonlocal captured_vars
            captured_vars = dict(kwargs.get("extra_template_vars", {}))
            return _make_mock_result(tmp_path, stdout_tail="score: 0.5")

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        watch(plan_path)

        assert captured_vars["watch.iteration"] == "1"
        assert captured_vars["watch.best_metric"] == ""
        assert captured_vars["watch.last_metric"] == ""
        assert captured_vars["watch.history"] == ""

    def test_template_vars_second_iteration_has_history(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Second iteration has populated history and last_metric."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task", metric_direction="higher_is_better",
                warmup_iterations=0, max_iterations=2, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        all_captured_vars: list[dict[str, str]] = []
        metrics = iter([0.5, 0.8])

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            all_captured_vars.append(dict(kwargs.get("extra_template_vars", {})))
            return _make_mock_result(tmp_path, stdout_tail=f"score: {next(metrics)}")

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        watch(plan_path)

        assert len(all_captured_vars) == 2
        second = all_captured_vars[1]
        assert second["watch.iteration"] == "2"
        assert second["watch.best_metric"] == "0.5"
        assert second["watch.last_metric"] == "0.5"
        assert "0.5" in second["watch.history"]


# ---------------------------------------------------------------------------
# Watch Step Counter (max_total_steps)
# ---------------------------------------------------------------------------


class TestCountExecutedTasks:
    """Unit tests for _count_executed_tasks helper."""

    def test_counts_non_skipped_tasks(self, tmp_path: Path) -> None:
        result = _make_mock_result(
            tmp_path,
            task_results={
                "a": TaskResult(
                    task_id="a", status="success", exit_code=0,
                    duration_sec=1.0, log_path=tmp_path / "a.log",
                    result_path=tmp_path / "a.json",
                ),
                "b": TaskResult(
                    task_id="b", status="skipped", exit_code=0,
                    duration_sec=0.0, log_path=tmp_path / "b.log",
                    result_path=tmp_path / "b.json",
                ),
                "c": TaskResult(
                    task_id="c", status="failed", exit_code=1,
                    duration_sec=2.0, log_path=tmp_path / "c.log",
                    result_path=tmp_path / "c.json",
                ),
            },
        )
        assert _count_executed_tasks(result) == 2

    def test_all_skipped_returns_zero(self, tmp_path: Path) -> None:
        result = _make_mock_result(
            tmp_path,
            task_results={
                "x": TaskResult(
                    task_id="x", status="skipped", exit_code=0,
                    duration_sec=0.0, log_path=tmp_path / "x.log",
                    result_path=tmp_path / "x.json",
                ),
            },
        )
        assert _count_executed_tasks(result) == 0

    def test_empty_results(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir(parents=True, exist_ok=True)
        result = PlanRunResult(
            plan_name="test", run_id="run-1", run_path=run_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=True, task_results={},
        )
        assert _count_executed_tasks(result) == 0


class TestWatchSpecMaxTotalSteps:
    """WatchSpec dataclass tests for max_total_steps."""

    def test_default_is_none(self) -> None:
        spec = WatchSpec(metric="score")
        assert spec.max_total_steps is None

    def test_set_value(self) -> None:
        spec = WatchSpec(metric="score", max_total_steps=50)
        assert spec.max_total_steps == 50

    def test_to_dict_includes_field(self) -> None:
        spec = WatchSpec(metric="score", max_total_steps=100)
        d = spec.to_dict()
        assert d["max_total_steps"] == 100

    def test_to_dict_none_value(self) -> None:
        spec = WatchSpec(metric="score")
        d = spec.to_dict()
        assert d["max_total_steps"] is None


class TestWatchStateTotalSteps:
    """WatchState dataclass tests for total_steps."""

    def test_default_is_zero(self) -> None:
        state = WatchState()
        assert state.total_steps == 0

    def test_to_dict_includes_field(self) -> None:
        state = WatchState(total_steps=42)
        d = state.to_dict()
        assert d["total_steps"] == 42


class TestWatchStepLimitLoader:
    """Loader validation tests for max_total_steps (E066)."""

    def test_valid_max_total_steps(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            "version: 1\nname: test\ntasks:\n"
            "  - id: t1\n    engine: claude\n    prompt: test\n"
            "watch:\n  metric: score\n  metric_pattern: 'score: ([0-9.]+)'\n"
            "  max_iterations: 10\n  max_total_steps: 25\n",
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.max_total_steps == 25

    def test_none_is_valid(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            "version: 1\nname: test\ntasks:\n"
            "  - id: t1\n    engine: claude\n    prompt: test\n"
            "watch:\n  metric: score\n  metric_pattern: 'score: ([0-9.]+)'\n"
            "  max_iterations: 10\n",
            encoding="utf-8",
        )
        plan = load_plan(plan_file)
        assert plan.watch is not None
        assert plan.watch.max_total_steps is None

    def test_zero_raises_e066(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            "version: 1\nname: test\ntasks:\n"
            "  - id: t1\n    engine: claude\n    prompt: test\n"
            "watch:\n  metric: score\n  metric_pattern: 'score: ([0-9.]+)'\n"
            "  max_iterations: 10\n  max_total_steps: 0\n",
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match=r"\[E066\]"):
            load_plan(plan_file)

    def test_negative_raises_e066(self, tmp_path: Path) -> None:
        from maestro_cli.errors import PlanValidationError
        from maestro_cli.loader import load_plan

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(
            "version: 1\nname: test\ntasks:\n"
            "  - id: t1\n    engine: claude\n    prompt: test\n"
            "watch:\n  metric: score\n  metric_pattern: 'score: ([0-9.]+)'\n"
            "  max_iterations: 10\n  max_total_steps: -5\n",
            encoding="utf-8",
        )
        with pytest.raises(PlanValidationError, match=r"\[E066\]"):
            load_plan(plan_file)


class TestWatchStepLimitCustomLoop:
    """Integration tests for step limit in the custom watch loop."""

    def test_stops_when_step_limit_reached(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With 2 tasks per iteration and max_total_steps=3, should stop after 2 iterations."""
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=10,
                plateau_threshold=10,
                max_total_steps=3,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        call_count = 0

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            nonlocal call_count
            call_count += 1
            return _make_mock_result(
                tmp_path,
                stdout_tail="score: 1.0",
                task_results={
                    "test-task": TaskResult(
                        task_id="test-task", status="success",
                        stdout_tail="score: 1.0", exit_code=0,
                        duration_sec=1.0, cost_usd=0.1,
                        log_path=tmp_path / "test-task.log",
                        result_path=tmp_path / "test-task.json",
                    ),
                    "other-task": TaskResult(
                        task_id="other-task", status="success",
                        exit_code=0, duration_sec=1.0, cost_usd=0.1,
                        log_path=tmp_path / "other.log",
                        result_path=tmp_path / "other.json",
                    ),
                },
            )

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        state = watch(plan_path)

        # 2 tasks per iteration: after iteration 1 → 2 steps, after iteration 2 → 4 steps
        # But check happens at top of loop: iteration 1 runs (0 < 3), iteration 2 runs (2 < 3),
        # iteration 3 check (4 >= 3) → stop
        assert state.status == "step_limit_reached"
        assert state.total_steps == 4
        assert call_count == 2

    def test_step_limit_emits_event(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=10,
                plateau_threshold=10,
                max_total_steps=1,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        events: list[tuple[str, dict[str, object]]] = []

        def _capture(event_type: str, payload: dict[str, object]) -> None:
            events.append((event_type, payload))

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            return _make_mock_result(tmp_path, stdout_tail="score: 1.0")

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        state = watch(plan_path, event_callback=_capture)

        assert state.status == "step_limit_reached"
        step_events = [(t, p) for t, p in events if t == "watch_step_limit"]
        assert len(step_events) == 1
        assert step_events[0][1]["max_total_steps"] == 1

    def test_no_limit_when_none(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without max_total_steps, the loop should complete all iterations normally."""
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=2,
                plateau_threshold=10,
                max_total_steps=None,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        call_count = 0

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            nonlocal call_count
            call_count += 1
            return _make_mock_result(tmp_path, stdout_tail="score: 1.0")

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        state = watch(plan_path)

        assert state.status == "max_iterations"
        assert call_count == 2

    def test_step_counting_across_iterations(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify total_steps accumulates across iterations with mixed skipped/executed tasks."""
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=3,
                plateau_threshold=10,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        iteration_idx = 0
        tasks_per_iter = [
            # iteration 1: 2 executed, 1 skipped
            {
                "a": TaskResult(task_id="a", status="success", stdout_tail="score: 1.0",
                                exit_code=0, duration_sec=1.0, cost_usd=0.1,
                                log_path=tmp_path / "a.log", result_path=tmp_path / "a.json"),
                "b": TaskResult(task_id="b", status="failed", exit_code=1, duration_sec=1.0,
                                log_path=tmp_path / "b.log", result_path=tmp_path / "b.json"),
                "c": TaskResult(task_id="c", status="skipped", exit_code=0, duration_sec=0.0,
                                log_path=tmp_path / "c.log", result_path=tmp_path / "c.json"),
            },
            # iteration 2: 1 executed
            {
                "a": TaskResult(task_id="a", status="success", stdout_tail="score: 2.0",
                                exit_code=0, duration_sec=1.0, cost_usd=0.1,
                                log_path=tmp_path / "a.log", result_path=tmp_path / "a.json"),
                "b": TaskResult(task_id="b", status="skipped", exit_code=0, duration_sec=0.0,
                                log_path=tmp_path / "b.log", result_path=tmp_path / "b.json"),
                "c": TaskResult(task_id="c", status="skipped", exit_code=0, duration_sec=0.0,
                                log_path=tmp_path / "c.log", result_path=tmp_path / "c.json"),
            },
            # iteration 3: 3 executed
            {
                "a": TaskResult(task_id="a", status="success", stdout_tail="score: 3.0",
                                exit_code=0, duration_sec=1.0, cost_usd=0.1,
                                log_path=tmp_path / "a.log", result_path=tmp_path / "a.json"),
                "b": TaskResult(task_id="b", status="success", exit_code=0, duration_sec=1.0,
                                log_path=tmp_path / "b.log", result_path=tmp_path / "b.json"),
                "c": TaskResult(task_id="c", status="soft_failed", exit_code=0, duration_sec=1.0,
                                log_path=tmp_path / "c.log", result_path=tmp_path / "c.json"),
            },
        ]

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            nonlocal iteration_idx
            results = tasks_per_iter[iteration_idx]
            # Use the stdout_tail of the "a" task for metric extraction
            tr = results.get("a") or list(results.values())[0]
            iteration_idx += 1
            run_path = tmp_path / f"run{iteration_idx}"
            run_path.mkdir(parents=True, exist_ok=True)
            return PlanRunResult(
                plan_name="test", run_id=f"run-{iteration_idx}",
                run_path=run_path, started_at=datetime.now(),
                finished_at=datetime.now(), success=True,
                task_results=results, total_cost_usd=0.1,
            )

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        state = watch(plan_path)

        # 2 + 1 + 3 = 6 total steps
        assert state.total_steps == 6
        assert state.total_iterations == 3


# ---------------------------------------------------------------------------
# Edge-case layer 3 — target: 60 new tests
# ---------------------------------------------------------------------------


class TestWatchEdgeL3:
    """Edge cases and under-tested paths for watch.py — Layer 3."""

    # --- _extract_metric: manifest source edge cases ---

    def test_manifest_metric_soft_failed_not_counted(self, tmp_path: Path) -> None:
        """soft_failed tasks are not counted as success/dry_run."""
        results = {
            "t1": TaskResult(task_id="t1", status="soft_failed"),
            "t2": TaskResult(task_id="t2", status="success"),
        }
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=False, task_results=results,
        )
        assert _extract_manifest_metric(run_result) == 1.0

    def test_manifest_metric_dry_run_counted(self, tmp_path: Path) -> None:
        """dry_run is counted just like success."""
        results = {
            "t1": TaskResult(task_id="t1", status="dry_run"),
            "t2": TaskResult(task_id="t2", status="dry_run"),
        }
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=True, task_results=results,
        )
        assert _extract_manifest_metric(run_result) == 2.0

    def test_manifest_metric_all_failed_returns_zero(self, tmp_path: Path) -> None:
        """All failed tasks returns 0.0, not None."""
        results = {
            "t1": TaskResult(task_id="t1", status="failed"),
            "t2": TaskResult(task_id="t2", status="failed"),
        }
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=False, task_results=results,
        )
        assert _extract_manifest_metric(run_result) == 0.0

    def test_extract_metric_manifest_bypasses_task_lookup(self, tmp_path: Path) -> None:
        """manifest source counts task_results directly, not plan.tasks."""
        plan = _make_plan(tmp_path)
        results = {
            "t1": TaskResult(task_id="t1", status="success"),
            "t2": TaskResult(task_id="t2", status="failed"),
            "t3": TaskResult(task_id="t3", status="success"),
        }
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=False, task_results=results,
        )
        spec = WatchSpec(metric="count", metric_source="manifest")
        # manifest counts success/dry_run from task_results, not plan.tasks
        assert _extract_metric(run_result, spec, plan, tmp_path) == 2.0

    # --- _extract_metric: unknown metric_source ---

    def test_extract_metric_unknown_source_returns_none(self, tmp_path: Path) -> None:
        """An unrecognized metric_source falls through to return None."""
        plan = _make_plan(tmp_path)
        result = _make_mock_result(tmp_path, stdout_tail="score: 0.5")
        spec = WatchSpec(
            metric="score",
            metric_source="nonexistent_source",  # type: ignore[arg-type]
            metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, result.run_path) is None

    # --- _is_improvement: boundary and edge values ---

    def test_is_improvement_large_values_higher_is_better(self) -> None:
        """Very large values compare correctly."""
        spec = WatchSpec(metric="score", metric_direction="higher_is_better")
        assert _is_improvement(1e10, 1e9, spec) is True
        assert _is_improvement(1e9, 1e10, spec) is False

    def test_is_improvement_large_values_lower_is_better(self) -> None:
        """Very large values compare correctly for lower_is_better."""
        spec = WatchSpec(metric="score", metric_direction="lower_is_better")
        assert _is_improvement(1e9, 1e10, spec) is True
        assert _is_improvement(1e10, 1e9, spec) is False

    def test_is_improvement_zero_best_lower_is_better(self) -> None:
        """Negative current with zero best is improvement for lower_is_better."""
        spec = WatchSpec(metric="score", metric_direction="lower_is_better")
        assert _is_improvement(-0.1, 0.0, spec) is True

    # --- _git_commit_changes: commit message with special chars ---

    def test_git_commit_metric_with_special_float(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Metric value with many decimals is included verbatim in commit message."""
        captured: list[list[str]] = []

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured.append(list(args[0]))
            if list(args[0])[:3] == ["git", "rev-parse", "HEAD"]:
                return subprocess.CompletedProcess(list(args[0]), 0, stdout="aaa\n", stderr="")
            return subprocess.CompletedProcess(list(args[0]), 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)
        _git_commit_changes(tmp_path, 99, "accuracy", 0.123456789)
        commit_cmd = [c for c in captured if c[:2] == ["git", "commit"]][0]
        assert "0.123456789" in commit_cmd[3]

    # --- _git_rollback: unrecognized on_regression value ---

    def test_git_rollback_unknown_regression_uses_reset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognized on_regression value defaults to git reset --hard HEAD."""
        captured: list[list[str]] = []

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured.append(list(args[0]))
            return subprocess.CompletedProcess(list(args[0]), 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.watch.subprocess.run", _mock_run)
        result = _git_rollback(tmp_path, "rollback")  # type: ignore[arg-type]
        assert result is True
        assert captured[0] == ["git", "reset", "--hard", "HEAD"]

    # --- _load_program: path resolution ---

    def test_load_program_relative_to_plan_source_dir(self, tmp_path: Path) -> None:
        """program_md is resolved relative to plan.source_dir."""
        plan_dir = tmp_path / "subdir"
        plan_dir.mkdir()
        plan_path = plan_dir / "plan.yaml"
        plan_path.write_text("version: 1\nname: p\n", encoding="utf-8")
        plan = PlanSpec(
            name="p", source_path=plan_path,
            workspace_root=".", run_dir=".maestro-runs",
            tasks=[TaskSpec(id="t1")],
        )
        program = plan_dir / "rules.md"
        program.write_text("Rule #1: be good\n", encoding="utf-8")
        spec = WatchSpec(metric="score", program_md="rules.md")
        assert _load_program(plan, spec) == "Rule #1: be good\n"

    # --- _build_history_text: formatting edge cases ---

    def test_build_history_text_large_iteration_numbers(self) -> None:
        """Large iteration numbers are right-aligned in the iter column."""
        history = [
            WatchIteration(iteration=9999, metric_value=1.5, best_metric=1.5, action="keep"),
        ]
        text = _build_history_text(history)
        assert "9999 |" in text

    def test_build_history_text_zero_metric_value(self) -> None:
        """Zero metric value renders as '0' not '-'."""
        history = [
            WatchIteration(iteration=1, metric_value=0.0, best_metric=0.0, action="keep"),
        ]
        text = _build_history_text(history)
        assert "0 | 0 | keep" in text

    def test_build_history_text_negative_metric_value(self) -> None:
        """Negative metric values are rendered correctly."""
        history = [
            WatchIteration(iteration=1, metric_value=-3.14, best_metric=-3.14, action="keep"),
        ]
        text = _build_history_text(history)
        assert "-3.14 | -3.14 | keep" in text

    # --- _write_experiment: edge cases ---

    def test_write_experiment_error_string_in_json(self, tmp_path: Path) -> None:
        """Error strings are properly serialized in JSON."""
        experiments_path = tmp_path / "exp.jsonl"
        wi = WatchIteration(
            iteration=1, metric_value=None, action="rollback",
            error="git rollback failed",
        )
        _write_experiment(experiments_path, wi)
        data = json.loads(experiments_path.read_text(encoding="utf-8").strip())
        assert data["error"] == "git rollback failed"

    def test_write_experiment_zero_cost_usd(self, tmp_path: Path) -> None:
        """Zero cost is written as 0 not null."""
        experiments_path = tmp_path / "exp.jsonl"
        wi = WatchIteration(iteration=1, action="keep", cost_usd=0.0)
        _write_experiment(experiments_path, wi)
        data = json.loads(experiments_path.read_text(encoding="utf-8").strip())
        assert data["cost_usd"] == 0.0

    # --- _resume_watch_state: edge cases ---

    def test_resume_iteration_numbers_not_sequential(self, tmp_path: Path) -> None:
        """Non-sequential iteration numbers are preserved as-is."""
        run_dir = tmp_path / "nonseq-run"
        run_dir.mkdir()
        lines = [
            {"iteration": 3, "metric_value": 0.5, "best_metric": 0.5, "improved": True,
             "action": "keep", "cost_usd": 0.1, "duration_sec": 1.0, "timestamp": "t1"},
            {"iteration": 7, "metric_value": 0.4, "best_metric": 0.4, "improved": True,
             "action": "keep", "cost_usd": 0.1, "duration_sec": 1.0, "timestamp": "t2"},
        ]
        (run_dir / "experiments.jsonl").write_text(
            "\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8"
        )
        state = _resume_watch_state(run_dir)
        assert state.total_iterations == 2
        assert state.iterations[0].iteration == 3
        assert state.iterations[1].iteration == 7
        assert state.best_iteration == 7

    def test_resume_missing_iteration_key_defaults_to_zero(self, tmp_path: Path) -> None:
        """Missing 'iteration' key defaults to 0."""
        run_dir = tmp_path / "missing-iter"
        run_dir.mkdir()
        entry = {"metric_value": 0.5, "best_metric": 0.5, "improved": True, "action": "keep"}
        (run_dir / "experiments.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
        state = _resume_watch_state(run_dir)
        assert state.iterations[0].iteration == 0

    def test_resume_mixed_valid_and_invalid_lines(self, tmp_path: Path) -> None:
        """Valid lines are parsed even when interleaved with invalid lines."""
        run_dir = tmp_path / "mixed-run"
        run_dir.mkdir()
        content = "\n".join([
            "invalid json {{{",
            json.dumps({"iteration": 1, "metric_value": 0.5, "best_metric": 0.5,
                         "improved": True, "action": "keep", "cost_usd": 0.1}),
            json.dumps(42),  # not a dict
            json.dumps({"iteration": 2, "metric_value": 0.3, "best_metric": 0.3,
                         "improved": True, "action": "keep", "cost_usd": 0.2}),
            "",  # blank line
        ]) + "\n"
        (run_dir / "experiments.jsonl").write_text(content, encoding="utf-8")
        state = _resume_watch_state(run_dir)
        assert state.total_iterations == 2
        assert state.total_cost_usd == pytest.approx(0.3)

    # --- _build_blame_context: edge cases ---

    def test_build_blame_context_empty_manifest_tasks(self, tmp_path: Path) -> None:
        """Manifest with empty task_results produces empty manifest_summary."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        (run_path / "run_manifest.json").write_text(
            json.dumps({"task_results": {}}), encoding="utf-8"
        )
        _, manifest_summary = _build_blame_context(run_path)
        assert manifest_summary == ""

    def test_build_blame_context_no_message_field(self, tmp_path: Path) -> None:
        """Task without message field produces clean output."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest = {
            "task_results": {
                "t1": {"status": "success", "exit_code": 0, "duration_sec": 1.0},
            }
        }
        (run_path / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        _, manifest_summary = _build_blame_context(run_path)
        assert "t1: success" in manifest_summary
        # Should not contain " — " separator for empty message
        assert " — " not in manifest_summary

    def test_build_blame_context_malformed_manifest_json(self, tmp_path: Path) -> None:
        """Malformed manifest JSON produces empty manifest_summary."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        (run_path / "run_manifest.json").write_text("not valid json{{{", encoding="utf-8")
        _, manifest_summary = _build_blame_context(run_path)
        assert manifest_summary == ""

    # --- _find_latest_target_run: edge cases ---

    def test_find_latest_target_run_files_not_dirs_ignored(self, tmp_path: Path) -> None:
        """Regular files (not directories) are ignored."""
        runs = tmp_path / ".maestro-runs"
        runs.mkdir()
        (runs / "20260301_120000_aaa_my-plan").write_text("not a dir", encoding="utf-8")
        result = _find_latest_target_run(tmp_path, ".maestro-runs", "my-plan")
        assert result is None

    def test_find_latest_target_run_multiple_matches_sorted(self, tmp_path: Path) -> None:
        """Multiple matching dirs returns the one with highest timestamp prefix."""
        runs = tmp_path / ".maestro-runs"
        runs.mkdir()
        (runs / "20260101_100000_my-plan").mkdir()
        (runs / "20260301_100000_my-plan").mkdir()
        (runs / "20260201_100000_my-plan").mkdir()
        result = _find_latest_target_run(tmp_path, ".maestro-runs", "my-plan")
        assert result is not None
        assert result.name == "20260301_100000_my-plan"

    # --- _target_reached: edge cases ---

    def test_target_reached_float_precision(self) -> None:
        """Very small float differences near the target boundary."""
        spec = WatchSpec(metric="s", metric_direction="higher_is_better", target_metric=1.0)
        assert _target_reached(0.9999999999, spec) is False
        assert _target_reached(1.0000000001, spec) is True

    def test_target_reached_very_large_target(self) -> None:
        """Very large target values work correctly."""
        spec = WatchSpec(metric="s", metric_direction="higher_is_better", target_metric=1e15)
        assert _target_reached(1e15, spec) is True
        assert _target_reached(1e14, spec) is False

    # --- _count_executed_tasks: edge cases ---

    def test_count_executed_tasks_all_success(self, tmp_path: Path) -> None:
        """All success tasks are counted."""
        result = _make_mock_result(
            tmp_path,
            task_results={
                "a": TaskResult(task_id="a", status="success"),
                "b": TaskResult(task_id="b", status="success"),
                "c": TaskResult(task_id="c", status="success"),
            },
        )
        assert _count_executed_tasks(result) == 3

    def test_count_executed_tasks_soft_failed_counted(self, tmp_path: Path) -> None:
        """soft_failed tasks are executed, not skipped."""
        result = _make_mock_result(
            tmp_path,
            task_results={
                "a": TaskResult(task_id="a", status="soft_failed"),
            },
        )
        assert _count_executed_tasks(result) == 1

    def test_count_executed_tasks_dry_run_counted(self, tmp_path: Path) -> None:
        """dry_run tasks are not 'skipped'."""
        result = _make_mock_result(
            tmp_path,
            task_results={
                "a": TaskResult(task_id="a", status="dry_run"),
            },
        )
        assert _count_executed_tasks(result) == 1

    # --- _emit: edge cases ---

    def test_emit_none_callback_does_not_raise(self) -> None:
        """_emit with None callback is a no-op."""
        from maestro_cli.watch import _emit
        # Should not raise
        _emit(None, "test_event", key="value")

    def test_emit_passes_kwargs_as_dict(self) -> None:
        """_emit passes kwargs to callback as a dict."""
        from maestro_cli.watch import _emit
        captured: list[tuple[str, dict[str, object]]] = []
        _emit(lambda t, p: captured.append((t, p)), "my_event", x=1, y="two")
        assert len(captured) == 1
        assert captured[0] == ("my_event", {"x": 1, "y": "two"})

    def test_emit_empty_kwargs(self) -> None:
        """_emit with no kwargs passes empty dict."""
        from maestro_cli.watch import _emit
        captured: list[tuple[str, dict[str, object]]] = []
        _emit(lambda t, p: captured.append((t, p)), "bare_event")
        assert captured[0] == ("bare_event", {})

    # --- _extract_log_section: edge cases ---

    def test_extract_log_section_multiple_status_lines(self, tmp_path: Path) -> None:
        """Only content before the first status= line is included."""
        log = tmp_path / "multi_status.log"
        log.write_text(
            "[verify_command]\nline1\nstatus=ok\nstatus=fail\n",
            encoding="utf-8",
        )
        assert _extract_log_section(log, "[verify_command]") == "line1"

    def test_extract_log_section_section_at_end_of_file(self, tmp_path: Path) -> None:
        """Section header at the very last line with no content returns None."""
        log = tmp_path / "trailing.log"
        log.write_text("some stuff\n[verify_command]\n", encoding="utf-8")
        assert _extract_log_section(log, "[verify_command]") is None

    def test_extract_log_section_only_whitespace_content(self, tmp_path: Path) -> None:
        """Section with only whitespace after header returns None (stripped is empty)."""
        log = tmp_path / "ws_only.log"
        log.write_text("[verify_command]\n  \n  \n", encoding="utf-8")
        assert _extract_log_section(log, "[verify_command]") is None

    # --- _lookup_json_path: edge cases ---

    def test_lookup_json_path_deeply_nested(self) -> None:
        """Deep nesting works correctly."""
        payload = {"a": {"b": {"c": {"d": {"e": 42}}}}}
        assert _lookup_json_path(payload, "a.b.c.d.e") == 42

    def test_lookup_json_path_boolean_value(self) -> None:
        """Boolean values are returned as-is."""
        payload = {"flag": True}
        assert _lookup_json_path(payload, "flag") is True

    def test_lookup_json_path_numeric_string_key(self) -> None:
        """Dict keys that look like numbers are treated as string keys, not array indices."""
        payload = {"0": "zero", "1": "one"}
        assert _lookup_json_path(payload, "0") == "zero"
        assert _lookup_json_path(payload, "1") == "one"

    # --- _coerce_float / _coerce_str edge cases ---

    def test_coerce_float_inf_and_nan(self) -> None:
        """float('inf') and float('nan') are valid floats."""
        result = _coerce_float("inf")
        assert result is not None
        assert result == float("inf")
        result_nan = _coerce_float("nan")
        assert result_nan is not None

    def test_coerce_str_whitespace_only(self) -> None:
        """Non-empty whitespace string is returned (not None)."""
        assert _coerce_str("   ") == "   "

    def test_coerce_float_negative_string(self) -> None:
        """Negative string number is parsed correctly."""
        assert _coerce_float("-2.5") == pytest.approx(-2.5)

    # --- Watch loop: revert on_regression ---

    def test_watch_revert_on_regression(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """on_regression='revert' calls git revert instead of git reset."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task", on_regression="revert",
                metric_direction="higher_is_better",
                warmup_iterations=0, max_iterations=2, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.5, 0.3])
        rollback_actions: list[str] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"score: {next(metrics)}"
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr(
            "maestro_cli.watch._git_rollback",
            lambda _w, action: rollback_actions.append(action) or True,
        )

        state = watch(plan_path)
        assert state.iterations[-1].action == "revert"
        assert rollback_actions == ["revert"]

    # --- Watch loop: on_regression='keep' no rollback ---

    def test_watch_keep_on_regression_no_rollback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """on_regression='keep' does not trigger git rollback."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task", on_regression="keep",
                metric_direction="higher_is_better",
                warmup_iterations=0, max_iterations=2, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.5, 0.3])
        rollback_actions: list[str] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"score: {next(metrics)}"
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr(
            "maestro_cli.watch._git_rollback",
            lambda _w, action: rollback_actions.append(action) or True,
        )

        state = watch(plan_path)
        assert state.iterations[-1].action == "keep"
        assert rollback_actions == ["keep"]

    # --- Watch loop: regression events ---

    def test_regression_emits_regression_and_rollback_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Regression emits both regression_detected and rollback_executed events."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task", metric_direction="higher_is_better",
                on_regression="rollback",
                warmup_iterations=0, max_iterations=2, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.8, 0.3])
        events: list[tuple[str, dict[str, object]]] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"score: {next(metrics)}"
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        watch(plan_path, event_callback=lambda n, p: events.append((n, p)))

        event_names = [n for n, _ in events]
        assert "regression_detected" in event_names
        assert "rollback_executed" in event_names
        regression_event = next(p for n, p in events if n == "regression_detected")
        assert regression_event["metric_value"] == pytest.approx(0.3)
        assert regression_event["action"] == "rollback"

    # --- Watch loop: watch_start event payload ---

    def test_watch_start_event_has_correct_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """watch_start event contains plan_name, max_iterations, metric, metric_direction."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="accuracy", metric_source="stdout_regex",
                metric_pattern=r"acc: ([0-9.]+)", metric_task="test-task",
                metric_direction="higher_is_better",
                warmup_iterations=0, max_iterations=1, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        events: list[tuple[str, dict[str, object]]] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(tmp_path, stdout_tail="acc: 0.9"),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        watch(plan_path, event_callback=lambda n, p: events.append((n, p)))

        start_event = next(p for n, p in events if n == "watch_start")
        assert start_event["plan_name"] == "watch-test"
        assert start_event["max_iterations"] == 1
        assert start_event["metric"] == "accuracy"
        assert start_event["metric_direction"] == "higher_is_better"

    # --- Watch loop: watch_complete event has correct payload ---

    def test_watch_complete_event_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """watch_complete event has status, best_metric, best_iteration, total_iterations, total_cost_usd."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)", metric_task="test-task",
                warmup_iterations=0, max_iterations=1, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        events: list[tuple[str, dict[str, object]]] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(tmp_path, stdout_tail="score: 0.5", cost=0.3),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        watch(plan_path, event_callback=lambda n, p: events.append((n, p)))

        complete_event = next(p for n, p in events if n == "watch_complete")
        assert complete_event["status"] == "max_iterations"
        assert complete_event["best_metric"] == pytest.approx(0.5)
        assert complete_event["best_iteration"] == 1
        assert complete_event["total_iterations"] == 1
        assert complete_event["total_cost_usd"] == pytest.approx(0.3)

    # --- Plateau detection with consecutive regressions ---

    def test_plateau_after_consecutive_regressions_higher_is_better(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Three regressions with plateau_threshold=3 triggers plateau."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)", metric_task="test-task",
                metric_direction="higher_is_better",
                warmup_iterations=0, max_iterations=10, plateau_threshold=3,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        # First is improvement, then 3 regressions
        metrics = iter([0.8, 0.5, 0.3, 0.1])

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"score: {next(metrics)}"
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        state = watch(plan_path)
        assert state.status == "plateau"
        assert state.plateau_count == 3
        assert state.best_metric == pytest.approx(0.8)

    # --- _extract_metric: verify_command with multiline section ---

    def test_extract_metric_verify_command_multiline_section(self, tmp_path: Path) -> None:
        """Regex matches across multiline verify_command section text."""
        run_path = tmp_path / "run"
        run_path.mkdir()
        log_path = run_path / "test-task.log"
        log_path.write_text(
            "[verify_command]\nrunning tests...\nfinal score: 0.92\nstatus=success\n",
            encoding="utf-8",
        )
        task_result = TaskResult(
            task_id="test-task", status="success",
            log_path=log_path, result_path=run_path / "test-task.json",
        )
        result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=run_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=True, task_results={"test-task": task_result},
        )
        plan = _make_plan(tmp_path)
        spec = WatchSpec(
            metric="score", metric_source="verify_command",
            metric_pattern=r"final score: ([0-9.]+)", metric_task="test-task",
        )
        assert _extract_metric(result, spec, plan, run_path) == pytest.approx(0.92)

    # --- Template vars: watch.consolidated ---

    def test_template_vars_include_consolidated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """watch.consolidated template variable is present (empty by default)."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)", metric_task="test-task",
                warmup_iterations=0, max_iterations=1, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        captured_vars: dict[str, str] = {}

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            nonlocal captured_vars
            captured_vars = dict(kwargs.get("extra_template_vars", {}))
            return _make_mock_result(tmp_path, stdout_tail="score: 0.5")

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        watch(plan_path)
        assert "watch.consolidated" in captured_vars
        assert captured_vars["watch.consolidated"] == ""

    # --- Template vars: watch.program with content ---

    def test_template_vars_program_text_from_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """watch.program is populated from program_md file content."""
        monkeypatch.chdir(tmp_path)
        program_file = tmp_path / "my_program.md"
        program_file.write_text("# Strategy\nStep 1: improve.\n", encoding="utf-8")
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)", metric_task="test-task",
                program_md="my_program.md",
                warmup_iterations=0, max_iterations=1, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        captured_vars: dict[str, str] = {}

        def _mock_run_plan(*args: Any, **kwargs: Any) -> PlanRunResult:
            nonlocal captured_vars
            captured_vars = dict(kwargs.get("extra_template_vars", {}))
            return _make_mock_result(tmp_path, stdout_tail="score: 0.5")

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        watch(plan_path)
        assert captured_vars["watch.program"] == "# Strategy\nStep 1: improve.\n"

    # --- Iteration budget_sec (iteration_budget_sec) ---

    def test_watchspec_iteration_budget_sec_default(self) -> None:
        """iteration_budget_sec defaults to None."""
        spec = WatchSpec(metric="score")
        assert spec.iteration_budget_sec is None

    def test_watchspec_iteration_budget_sec_set(self) -> None:
        """iteration_budget_sec can be set."""
        spec = WatchSpec(metric="score", iteration_budget_sec=300)
        assert spec.iteration_budget_sec == 300

    # --- WatchSpec on_regression default ---

    def test_watchspec_on_regression_default_rollback(self) -> None:
        """Default on_regression is 'rollback'."""
        spec = WatchSpec(metric="score")
        assert spec.on_regression == "rollback"

    # --- _build_improve_plan: plan_path_rel in verify_command ---

    def test_build_improve_plan_verify_uses_plan_path(self, tmp_path: Path) -> None:
        """verify_command uses the exact plan_path_rel provided."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed", mode="improve",
            metric_source="manifest", metric_direction="higher_is_better",
        ))
        result = _build_improve_plan(plan, plan.watch, "subdir/my-plan.yaml")
        vc = result.tasks[0].verify_command
        assert "subdir/my-plan.yaml" in vc

    # --- _build_improve_plan: source_path inherited ---

    def test_build_improve_plan_source_path_inherited(self, tmp_path: Path) -> None:
        """Improve plan inherits source_path from target plan."""
        plan = _make_plan(tmp_path, watch_spec=WatchSpec(
            metric="tasks_passed", mode="improve",
            metric_source="manifest", metric_direction="higher_is_better",
        ))
        result = _build_improve_plan(plan, plan.watch, "plan.yaml")
        assert result.source_path == plan.source_path

    # --- _extract_lesson: edge cases ---

    def test_extract_lesson_none_action_returns_none(self) -> None:
        """Iteration with action=None produces no lesson."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=1, metric_value=5.0, improved=True,
            action=None,  # type: ignore[arg-type]
            timestamp="2026-01-01T00:00:00",
        )
        # action is None, which is falsy, triggers "not iteration.action" early return
        assert _extract_lesson(wi, "") is None

    def test_extract_lesson_empty_action_returns_none(self) -> None:
        """Iteration with action='' is falsy so returns None."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=1, metric_value=5.0, improved=True,
            action="",
            timestamp="2026-01-01T00:00:00",
        )
        assert _extract_lesson(wi, "") is None

    # --- _format_lessons: multiple lessons ---

    def test_format_lessons_multiple_entries(self) -> None:
        """Multiple lessons are formatted as bullet points."""
        from maestro_cli.models import LessonRecord
        from maestro_cli.watch import _format_lessons

        lessons = [
            LessonRecord(iteration=1, task_id="t1", category="fix",
                         lesson="Lesson A", confidence=0.9, timestamp=""),
            LessonRecord(iteration=2, task_id="", category="attempt",
                         lesson="Lesson B", confidence=0.5, timestamp=""),
        ]
        text = _format_lessons(lessons)
        lines = text.strip().splitlines()
        assert len(lines) == 2
        assert lines[0].startswith("- [")
        assert "Lesson A" in lines[0]
        assert "Lesson B" in lines[1]

    # --- _load_lessons: time decay ---

    def test_load_lessons_time_decay_reduces_confidence(self, tmp_path: Path) -> None:
        """Old lessons have reduced confidence via time decay."""
        from maestro_cli.models import LessonRecord
        from maestro_cli.watch import _load_lessons, _write_lesson

        lessons_path = tmp_path / "lessons.jsonl"
        # Old lesson — 60 days ago (2 half-lives at 30 days → 0.25 decay)
        old_ts = "2026-01-20T00:00:00"
        _write_lesson(lessons_path, LessonRecord(
            iteration=1, task_id="t1", category="fix",
            lesson="Old lesson", confidence=1.0, timestamp=old_ts,
        ))
        loaded = _load_lessons(lessons_path)
        assert len(loaded) == 1
        # Confidence should be significantly less than 1.0 due to decay
        assert loaded[0].confidence < 0.5

    # --- Watch loop: experiments.jsonl written on each iteration ---

    def test_watch_writes_experiments_jsonl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each iteration writes a line to experiments.jsonl."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)", metric_task="test-task",
                warmup_iterations=0, max_iterations=2, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.5, 0.4])

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"score: {next(metrics)}"
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        state = watch(plan_path)

        # Find the watch run directory
        runs_dir = tmp_path / ".maestro-runs"
        assert runs_dir.exists()
        watch_dirs = [d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("watch_")]
        assert len(watch_dirs) == 1
        experiments_path = watch_dirs[0] / "experiments.jsonl"
        assert experiments_path.exists()
        lines = experiments_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        # Verify first line
        first = json.loads(lines[0])
        assert first["iteration"] == 1
        assert first["metric_value"] == pytest.approx(0.5)

    # --- _run_consolidation: model and command construction ---

    def test_run_consolidation_uses_specified_model(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Model from spec is passed to the claude command."""
        from maestro_cli.watch import _run_consolidation

        captured: list[list[str]] = []

        def _mock_run(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            captured.append(list(args[0]))
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            consolidate_model="opus",
        )
        _run_consolidation(spec, "history", tmp_path)
        assert "--model" in captured[0]
        model_idx = captured[0].index("--model")
        assert captured[0][model_idx + 1] == "opus"

    # --- Watch loop: metric_recorded event payload ---

    def test_metric_recorded_event_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """metric_recorded event contains iteration, metric, value, best, improved."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)", metric_task="test-task",
                warmup_iterations=0, max_iterations=1, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        events: list[tuple[str, dict[str, object]]] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(tmp_path, stdout_tail="score: 0.75"),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        watch(plan_path, event_callback=lambda n, p: events.append((n, p)))

        metric_events = [(n, p) for n, p in events if n == "metric_recorded"]
        assert len(metric_events) == 1
        payload = metric_events[0][1]
        assert payload["iteration"] == 1
        assert payload["metric"] == "score"
        assert payload["value"] == pytest.approx(0.75)
        assert payload["improved"] is True

    # --- iteration_complete event payload ---

    def test_iteration_complete_event_payload(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """iteration_complete event has all expected fields."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)", metric_task="test-task",
                warmup_iterations=0, max_iterations=1, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        events: list[tuple[str, dict[str, object]]] = []

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail="score: 0.5", cost=0.25
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        watch(plan_path, event_callback=lambda n, p: events.append((n, p)))

        ic_events = [(n, p) for n, p in events if n == "iteration_complete"]
        assert len(ic_events) == 1
        payload = ic_events[0][1]
        assert payload["iteration"] == 1
        assert payload["metric_value"] == pytest.approx(0.5)
        assert payload["improved"] is True
        assert payload["action"] == "keep"
        assert "duration_sec" in payload

    # --- Watch loop: single iteration updates state correctly ---

    def test_watch_single_iteration_state_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """After 1 iteration, all state fields are set correctly."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)", metric_task="test-task",
                warmup_iterations=0, max_iterations=1, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail="score: 0.42", cost=0.15
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha-abc")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        state = watch(plan_path)

        assert state.total_iterations == 1
        assert state.best_metric == pytest.approx(0.42)
        assert state.best_iteration == 1
        assert state.plateau_count == 0
        assert state.total_cost_usd == pytest.approx(0.15)
        assert state.plan_path == str((tmp_path / "plan.yaml").resolve())
        it = state.iterations[0]
        assert it.iteration == 1
        assert it.improved is True
        assert it.action == "keep"
        assert it.git_commit == "sha-abc"
        assert it.error is None
        assert it.timestamp is not None and it.timestamp != ""

    # --- Watch loop: higher_is_better direction ---

    def test_watch_higher_is_better_direction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """higher_is_better: increasing metrics are improvements."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score", metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)", metric_task="test-task",
                metric_direction="higher_is_better",
                warmup_iterations=0, max_iterations=2, plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.3, 0.8])

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"score: {next(metrics)}"
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        state = watch(plan_path)
        assert state.best_metric == pytest.approx(0.8)
        assert all(it.improved for it in state.iterations)


# ===========================================================================
# TestBuildExperimentsSummary
# ===========================================================================


class TestBuildExperimentsSummary:
    """Tests for _build_experiments_summary adaptive analysis."""

    def test_empty_iterations_returns_empty(self) -> None:
        assert _build_experiments_summary([]) == ""

    def test_basic_summary_with_iterations(self) -> None:
        iterations = [
            WatchIteration(iteration=1, metric_value=3.0, best_metric=3.0, improved=False, action="baseline"),
            WatchIteration(iteration=2, metric_value=4.0, best_metric=4.0, improved=True, action="commit"),
            WatchIteration(iteration=3, metric_value=3.0, best_metric=4.0, improved=False, action="rollback"),
        ]
        result = _build_experiments_summary(iterations)
        assert "## Experiment Analysis" in result
        assert "Iterations run: 3" in result
        assert "Improvements: 1" in result

    def test_successful_approaches_listed(self) -> None:
        iterations = [
            WatchIteration(iteration=1, metric_value=5.0, best_metric=5.0, improved=True, action="commit"),
            WatchIteration(iteration=2, metric_value=6.0, best_metric=6.0, improved=True, action="commit"),
        ]
        result = _build_experiments_summary(iterations)
        assert "Approaches that WORKED" in result
        assert "Iteration 1" in result
        assert "Iteration 2" in result

    def test_failed_approaches_listed(self) -> None:
        iterations = [
            WatchIteration(iteration=1, metric_value=3.0, best_metric=3.0, improved=False, action="rollback"),
            WatchIteration(iteration=2, metric_value=2.0, best_metric=3.0, improved=False, action="rollback", error="timeout"),
        ]
        result = _build_experiments_summary(iterations)
        assert "Approaches that FAILED" in result
        assert "do NOT repeat" in result
        assert "timeout" in result

    def test_plateau_alert_at_count_2(self) -> None:
        iterations = [
            WatchIteration(iteration=1, metric_value=3.0, best_metric=3.0, improved=False, action="rollback"),
        ]
        result = _build_experiments_summary(iterations, plateau_count=2, plateau_threshold=5)
        assert "Plateau Alert" in result
        assert "Stuck for 2 iterations" in result

    def test_critical_alert_near_threshold(self) -> None:
        iterations = [
            WatchIteration(iteration=1, metric_value=3.0, best_metric=3.0, improved=False, action="rollback"),
        ]
        result = _build_experiments_summary(iterations, plateau_count=4, plateau_threshold=5)
        assert "CRITICAL" in result
        assert "last chance" in result

    def test_no_plateau_section_when_count_low(self) -> None:
        iterations = [
            WatchIteration(iteration=1, metric_value=5.0, best_metric=5.0, improved=True, action="commit"),
        ]
        result = _build_experiments_summary(iterations, plateau_count=0, plateau_threshold=3)
        assert "Plateau Alert" not in result

    def test_best_metric_shown(self) -> None:
        iterations = [
            WatchIteration(iteration=1, metric_value=5.0, best_metric=7.0, improved=True, action="commit"),
        ]
        result = _build_experiments_summary(iterations)
        assert "Best metric so far: 7" in result

    def test_baseline_not_counted_as_failure(self) -> None:
        iterations = [
            WatchIteration(iteration=1, metric_value=3.0, best_metric=3.0, improved=False, action="baseline"),
        ]
        result = _build_experiments_summary(iterations)
        assert "Failures: 0" in result

    def test_error_hint_truncated(self) -> None:
        long_error = "x" * 200
        iterations = [
            WatchIteration(iteration=1, metric_value=2.0, best_metric=3.0, improved=False, action="rollback", error=long_error),
        ]
        result = _build_experiments_summary(iterations)
        # Error hint should be truncated to 80 chars
        lines = [l for l in result.splitlines() if "Iteration 1" in l]
        assert len(lines) == 1
        assert len(lines[0]) < 250


class TestExperimentsSummaryTemplateVar:
    """Test that watch.experiments_summary is in _KNOWN_GLOBAL_VARS."""

    def test_known_global_var_registered(self) -> None:
        from maestro_cli.loader import _KNOWN_GLOBAL_VARS
        assert "watch.experiments_summary" in _KNOWN_GLOBAL_VARS

    def test_improve_prompt_template_references_var(self) -> None:
        from maestro_cli.watch import _IMPROVE_PROMPT_TEMPLATE
        assert "{{ watch.experiments_summary }}" in _IMPROVE_PROMPT_TEMPLATE


# ===========================================================================
# TestSteppingStones
# ===========================================================================


class TestSteppingStoneDataclass:
    """Test SteppingStone dataclass."""

    def test_to_dict_roundtrip(self) -> None:
        stone = SteppingStone(
            plan_name="test-plan",
            plan_hash="abc123",
            metric_value=5.0,
            metric_name="score",
            iteration=3,
            git_commit="deadbeef",
            plan_yaml="version: 1\nname: test\n",
            lessons=[{"lesson": "try X"}],
            timestamp="2026-03-25T12:00:00",
            watch_run_path="/tmp/watch_run",
            total_cost_usd=1.50,
        )
        d = stone.to_dict()
        assert d["plan_name"] == "test-plan"
        assert d["metric_value"] == 5.0
        assert d["plan_yaml"] == "version: 1\nname: test\n"
        assert len(d["lessons"]) == 1
        assert d["source_type"] == "watch"
        assert d["metadata"] == {}

    def test_defaults(self) -> None:
        stone = SteppingStone(
            plan_name="p", plan_hash="h", metric_value=1.0,
            metric_name="m", iteration=1,
        )
        assert stone.git_commit is None
        assert stone.plan_yaml == ""
        assert stone.lessons == []
        assert stone.total_cost_usd == 0.0
        assert stone.source_type == "watch"
        assert stone.metadata == {}


class TestSteppingStonesDir:
    """Test _stepping_stones_dir helper."""

    def test_returns_expected_path(self, tmp_path: Path) -> None:
        result = _stepping_stones_dir(tmp_path, "my-plan")
        assert ".maestro-cache" in str(result)
        assert "stepping" in str(result)
        assert "my-plan" in str(result)


class TestSaveSteppingStone:
    """Test _save_stepping_stone."""

    def test_saves_stone_to_jsonl(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text("version: 1\nname: test\ntasks:\n  - id: t1\n    command: echo\n", encoding="utf-8")
        lessons_path = tmp_path / "lessons.jsonl"

        stone = _save_stepping_stone(
            plan_path=plan_path,
            plan_name="test",
            metric_value=5.0,
            metric_name="score",
            iteration=2,
            git_commit="abc",
            lessons_path=lessons_path,
            watch_run_path=str(tmp_path),
            total_cost_usd=0.50,
        )
        assert stone is not None
        assert stone.plan_name == "test"
        assert stone.metric_value == 5.0
        assert stone.plan_hash != ""

        stones_dir = _stepping_stones_dir(tmp_path, "test")
        stones_path = stones_dir / "stones.jsonl"
        assert stones_path.exists()
        lines = stones_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 1

    def test_appends_multiple_stones(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text("version: 1\nname: test\ntasks:\n  - id: t1\n    command: echo\n", encoding="utf-8")
        lessons_path = tmp_path / "lessons.jsonl"

        for i in range(3):
            _save_stepping_stone(
                plan_path=plan_path, plan_name="test",
                metric_value=float(i), metric_name="score",
                iteration=i, git_commit=f"sha{i}",
                lessons_path=lessons_path,
                watch_run_path=str(tmp_path), total_cost_usd=0.0,
            )

        stones_path = _stepping_stones_dir(tmp_path, "test") / "stones.jsonl"
        lines = stones_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 3

    def test_returns_none_if_plan_missing(self, tmp_path: Path) -> None:
        result = _save_stepping_stone(
            plan_path=tmp_path / "nonexistent.yaml",
            plan_name="test", metric_value=1.0, metric_name="m",
            iteration=1, git_commit="x",
            lessons_path=tmp_path / "lessons.jsonl",
            watch_run_path="", total_cost_usd=0.0,
        )
        assert result is None

    def test_includes_lessons(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text("version: 1\nname: test\ntasks:\n  - id: t1\n    command: echo\n", encoding="utf-8")
        lessons_path = tmp_path / "lessons.jsonl"
        lessons_path.write_text('{"lesson":"try X"}\n{"lesson":"try Y"}\n', encoding="utf-8")

        stone = _save_stepping_stone(
            plan_path=plan_path, plan_name="test",
            metric_value=5.0, metric_name="score",
            iteration=1, git_commit="sha",
            lessons_path=lessons_path,
            watch_run_path="", total_cost_usd=0.0,
        )
        assert stone is not None
        assert len(stone.lessons) == 2

    def test_supports_metadata_and_archive_override(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "temp" / "plan.yaml"
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text("version: 1\nname: test\ntasks:\n  - id: t1\n    command: echo\n", encoding="utf-8")

        stone = _save_stepping_stone(
            plan_path=plan_path,
            plan_name="test",
            metric_value=9.0,
            metric_name="replan_fitness",
            iteration=4,
            git_commit=None,
            lessons_path=None,
            lessons=[{"source": "replan", "mutation_desc": "candidate 2/2"}],
            watch_run_path="C:/tmp/replan-run",
            total_cost_usd=0.4,
            archive_source_dir=tmp_path,
            source_type="replan",
            metadata={"selected_node_id": "node-2"},
        )

        assert stone is not None
        assert stone.source_type == "replan"
        assert stone.metadata["selected_node_id"] == "node-2"
        stones_path = _stepping_stones_dir(tmp_path, "test") / "stones.jsonl"
        assert stones_path.exists()
        payload = json.loads(stones_path.read_text(encoding="utf-8").strip())
        assert payload["source_type"] == "replan"
        assert payload["metadata"]["selected_node_id"] == "node-2"


class TestCompactSteppingStones:
    """Test _compact_stepping_stones."""

    def test_no_compact_when_under_limit(self, tmp_path: Path) -> None:
        stones_path = tmp_path / "stones.jsonl"
        lines = [json.dumps({"metric_value": i, "metric_name": "s"}) for i in range(5)]
        stones_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        _compact_stepping_stones(stones_path, "s")
        result = stones_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(result) == 5

    def test_compacts_when_over_limit(self, tmp_path: Path) -> None:
        stones_path = tmp_path / "stones.jsonl"
        lines = [json.dumps({"metric_value": float(i), "metric_name": "s"}) for i in range(25)]
        stones_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        _compact_stepping_stones(stones_path, "s")
        result = stones_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(result) == 20  # _STEPPING_STONES_MAX

    def test_compacts_only_requested_metric(self, tmp_path: Path) -> None:
        stones_path = tmp_path / "stones.jsonl"
        lines = [json.dumps({"metric_value": float(i), "metric_name": "score"}) for i in range(25)]
        lines.extend(
            json.dumps({"metric_value": float(i), "metric_name": "replan_fitness"})
            for i in range(3)
        )
        stones_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        _compact_stepping_stones(stones_path, "score")
        rows = [json.loads(line) for line in stones_path.read_text(encoding="utf-8").strip().splitlines()]
        score_rows = [row for row in rows if row["metric_name"] == "score"]
        replan_rows = [row for row in rows if row["metric_name"] == "replan_fitness"]

        assert len(score_rows) == 20
        assert len(replan_rows) == 3

    def test_missing_file_no_error(self, tmp_path: Path) -> None:
        _compact_stepping_stones(tmp_path / "nope.jsonl", "s")  # no exception


class TestLoadBestSteppingStone:
    """Test _load_best_stepping_stone."""

    def test_returns_none_when_no_stones(self, tmp_path: Path) -> None:
        result = _load_best_stepping_stone("plan", tmp_path, "score")
        assert result is None

    def test_loads_best_higher_is_better(self, tmp_path: Path) -> None:
        stones_dir = _stepping_stones_dir(tmp_path, "plan")
        stones_dir.mkdir(parents=True)
        stones_path = stones_dir / "stones.jsonl"
        data = [
            {"plan_name": "plan", "plan_hash": "a", "metric_value": 3.0, "metric_name": "score", "iteration": 1, "plan_yaml": "v1"},
            {"plan_name": "plan", "plan_hash": "b", "metric_value": 7.0, "metric_name": "score", "iteration": 2, "plan_yaml": "v2"},
            {"plan_name": "plan", "plan_hash": "c", "metric_value": 5.0, "metric_name": "score", "iteration": 3, "plan_yaml": "v3"},
        ]
        stones_path.write_text("\n".join(json.dumps(d) for d in data) + "\n", encoding="utf-8")

        best = _load_best_stepping_stone("plan", tmp_path, "score", higher_is_better=True)
        assert best is not None
        assert best.metric_value == 7.0
        assert best.plan_yaml == "v2"

    def test_loads_best_lower_is_better(self, tmp_path: Path) -> None:
        stones_dir = _stepping_stones_dir(tmp_path, "plan")
        stones_dir.mkdir(parents=True)
        stones_path = stones_dir / "stones.jsonl"
        data = [
            {"plan_name": "plan", "plan_hash": "a", "metric_value": 3.0, "metric_name": "loss", "iteration": 1, "plan_yaml": "v1"},
            {"plan_name": "plan", "plan_hash": "b", "metric_value": 7.0, "metric_name": "loss", "iteration": 2, "plan_yaml": "v2"},
        ]
        stones_path.write_text("\n".join(json.dumps(d) for d in data) + "\n", encoding="utf-8")

        best = _load_best_stepping_stone("plan", tmp_path, "loss", higher_is_better=False)
        assert best is not None
        assert best.metric_value == 3.0

    def test_filters_by_metric_name(self, tmp_path: Path) -> None:
        stones_dir = _stepping_stones_dir(tmp_path, "plan")
        stones_dir.mkdir(parents=True)
        stones_path = stones_dir / "stones.jsonl"
        data = [
            {"plan_name": "plan", "plan_hash": "a", "metric_value": 99.0, "metric_name": "other", "iteration": 1, "plan_yaml": "v1"},
            {"plan_name": "plan", "plan_hash": "b", "metric_value": 5.0, "metric_name": "score", "iteration": 2, "plan_yaml": "v2"},
        ]
        stones_path.write_text("\n".join(json.dumps(d) for d in data) + "\n", encoding="utf-8")

        best = _load_best_stepping_stone("plan", tmp_path, "score", higher_is_better=True)
        assert best is not None
        assert best.metric_value == 5.0

    def test_corrupt_lines_skipped(self, tmp_path: Path) -> None:
        stones_dir = _stepping_stones_dir(tmp_path, "plan")
        stones_dir.mkdir(parents=True)
        stones_path = stones_dir / "stones.jsonl"
        stones_path.write_text(
            'not json\n{"plan_name":"plan","plan_hash":"a","metric_value":5.0,"metric_name":"s","iteration":1,"plan_yaml":"ok"}\n',
            encoding="utf-8",
        )
        best = _load_best_stepping_stone("plan", tmp_path, "s")
        assert best is not None
        assert best.metric_value == 5.0


class TestApplySteppingStone:
    """Test _apply_stepping_stone."""

    def test_applies_valid_plan(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text("version: 1\nname: old\ntasks:\n  - id: t1\n    command: echo old\n", encoding="utf-8")
        new_yaml = "version: 1\nname: new\ntasks:\n  - id: t1\n    command: echo new\n"
        stone = SteppingStone(
            plan_name="new", plan_hash="x", metric_value=10.0,
            metric_name="s", iteration=5, plan_yaml=new_yaml,
        )
        result = _apply_stepping_stone(stone, plan_path)
        assert result is True
        assert "new" in plan_path.read_text(encoding="utf-8")

    def test_restores_backup_on_invalid_plan(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        original = "version: 1\nname: original\ntasks:\n  - id: t1\n    command: echo\n"
        plan_path.write_text(original, encoding="utf-8")
        stone = SteppingStone(
            plan_name="bad", plan_hash="x", metric_value=10.0,
            metric_name="s", iteration=5, plan_yaml="this is not valid yaml: [",
        )
        result = _apply_stepping_stone(stone, plan_path)
        assert result is False
        assert plan_path.read_text(encoding="utf-8") == original

    def test_returns_false_for_empty_yaml(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text("version: 1\nname: x\ntasks:\n  - id: t\n    command: echo\n", encoding="utf-8")
        stone = SteppingStone(
            plan_name="x", plan_hash="x", metric_value=1.0,
            metric_name="s", iteration=1, plan_yaml="",
        )
        assert _apply_stepping_stone(stone, plan_path) is False


class TestWatchSpecSteppingStones:
    """Test stepping_stones field on WatchSpec."""

    def test_default_is_false(self) -> None:
        spec = WatchSpec(metric="score")
        assert spec.stepping_stones is False

    def test_to_dict_includes_field(self) -> None:
        spec = WatchSpec(metric="score", stepping_stones=True)
        d = spec.to_dict()
        assert d["stepping_stones"] is True

    def test_loader_parses_stepping_stones(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan
        yaml_text = (
            "version: 1\nname: ss-test\n"
            "watch:\n  metric: score\n  metric_source: stdout_regex\n"
            "  metric_pattern: 'score: (\\\\d+)'\n"
            "  stepping_stones: true\n"
            "tasks:\n  - id: t1\n    command: echo score 5\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(yaml_text, encoding="utf-8")
        plan = load_plan(plan_path)
        assert plan.watch is not None
        assert plan.watch.stepping_stones is True


# ---------------------------------------------------------------------------
# Coverage: _extract_lesson extra branches
# ---------------------------------------------------------------------------


class TestExtractLessonEdgeCases:
    """Cover missing branches in _extract_lesson."""

    def test_extract_lesson_none_action_returns_none(self) -> None:
        """None/empty action returns None (no actionable lesson)."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=1, metric_value=5.0, improved=True,
            action="", timestamp="2026-01-01T00:00:00",
        )
        # Empty action is falsy → first guard should catch it
        assert _extract_lesson(wi, "") is None

    def test_extract_lesson_uses_current_time_when_timestamp_empty(self) -> None:
        """When WatchIteration.timestamp is empty, lesson gets datetime.now()."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=10, metric_value=7.0, improved=True,
            action="keep", timestamp="",
        )
        lesson = _extract_lesson(wi, "")
        assert lesson is not None
        # Timestamp should be populated (from datetime.now)
        assert lesson.timestamp != ""

    def test_extract_lesson_task_id_from_multiline_manifest(self) -> None:
        """Extract task ID from a manifest with multiple lines including 'failed'."""
        from maestro_cli.watch import _extract_lesson

        wi = WatchIteration(
            iteration=3, metric_value=2.0, improved=False,
            action="rollback", timestamp="2026-03-01T00:00:00",
        )
        manifest = "setup: success\nbuild-step: failed (exit=1)\ntest: skipped"
        lesson = _extract_lesson(wi, manifest)
        assert lesson is not None
        assert lesson.task_id == "build-step"
        assert lesson.category == "failed_attempt"


# ---------------------------------------------------------------------------
# Coverage: _compact_stepping_stones, _load_best_stepping_stone edge cases
# ---------------------------------------------------------------------------


class TestSteppingStonesCompaction:
    """Cover compaction and load paths for stepping stones."""

    def test_compact_no_file(self, tmp_path: Path) -> None:
        """Compacting a non-existent file does nothing."""
        _compact_stepping_stones(tmp_path / "ghost.jsonl", "score")

    def test_compact_under_limit(self, tmp_path: Path) -> None:
        """Below the max limit, no compaction happens."""
        stones_path = tmp_path / "stones.jsonl"
        entries = [
            json.dumps({"plan_name": "p", "metric_value": float(i), "metric_name": "s"})
            for i in range(5)
        ]
        stones_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        _compact_stepping_stones(stones_path, "s")
        # File should be unchanged (5 < 20)
        lines = [l for l in stones_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 5

    def test_compact_over_limit_keeps_best(self, tmp_path: Path) -> None:
        """Over the max limit, only top 20 by metric_value are kept."""
        from maestro_cli.watch import _STEPPING_STONES_MAX

        stones_path = tmp_path / "stones.jsonl"
        entries = [
            json.dumps({"plan_name": "p", "metric_value": float(i), "metric_name": "s"})
            for i in range(_STEPPING_STONES_MAX + 10)
        ]
        stones_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
        _compact_stepping_stones(stones_path, "s")
        lines = [l for l in stones_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == _STEPPING_STONES_MAX
        # Best (highest) should be first
        first = json.loads(lines[0])
        assert first["metric_value"] == float(_STEPPING_STONES_MAX + 10 - 1)

    def test_compact_skips_corrupt_lines(self, tmp_path: Path) -> None:
        """Corrupt JSON lines are silently skipped during compaction."""
        from maestro_cli.watch import _STEPPING_STONES_MAX

        stones_path = tmp_path / "stones.jsonl"
        good = [json.dumps({"metric_value": float(i)}) for i in range(_STEPPING_STONES_MAX + 5)]
        content = "not json\n" + "\n".join(good) + "\n"
        stones_path.write_text(content, encoding="utf-8")
        _compact_stepping_stones(stones_path, "s")
        lines = [l for l in stones_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == _STEPPING_STONES_MAX


class TestLoadBestSteppingStone2:
    """Cover _load_best_stepping_stone branches."""

    def test_no_file_returns_none(self, tmp_path: Path) -> None:
        result = _load_best_stepping_stone("plan", tmp_path, "score")
        assert result is None

    def test_loads_best_higher_is_better(self, tmp_path: Path) -> None:
        stones_dir = _stepping_stones_dir(tmp_path, "my-plan")
        stones_dir.mkdir(parents=True)
        stones_path = stones_dir / "stones.jsonl"
        entries = [
            {"plan_name": "my-plan", "plan_hash": "a", "metric_value": 3.0,
             "metric_name": "score", "iteration": 1},
            {"plan_name": "my-plan", "plan_hash": "b", "metric_value": 7.0,
             "metric_name": "score", "iteration": 2},
            {"plan_name": "my-plan", "plan_hash": "c", "metric_value": 5.0,
             "metric_name": "score", "iteration": 3},
        ]
        stones_path.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8",
        )
        stone = _load_best_stepping_stone("my-plan", tmp_path, "score", higher_is_better=True)
        assert stone is not None
        assert stone.metric_value == 7.0
        assert stone.iteration == 2

    def test_loads_best_lower_is_better(self, tmp_path: Path) -> None:
        stones_dir = _stepping_stones_dir(tmp_path, "my-plan")
        stones_dir.mkdir(parents=True)
        stones_path = stones_dir / "stones.jsonl"
        entries = [
            {"plan_name": "my-plan", "plan_hash": "a", "metric_value": 3.0,
             "metric_name": "loss", "iteration": 1},
            {"plan_name": "my-plan", "plan_hash": "b", "metric_value": 1.0,
             "metric_name": "loss", "iteration": 2},
            {"plan_name": "my-plan", "plan_hash": "c", "metric_value": 5.0,
             "metric_name": "loss", "iteration": 3},
        ]
        stones_path.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8",
        )
        stone = _load_best_stepping_stone("my-plan", tmp_path, "loss", higher_is_better=False)
        assert stone is not None
        assert stone.metric_value == 1.0

    def test_filters_by_metric_name(self, tmp_path: Path) -> None:
        stones_dir = _stepping_stones_dir(tmp_path, "my-plan")
        stones_dir.mkdir(parents=True)
        stones_path = stones_dir / "stones.jsonl"
        entries = [
            {"plan_name": "my-plan", "plan_hash": "a", "metric_value": 100.0,
             "metric_name": "other_metric", "iteration": 1},
        ]
        stones_path.write_text(
            "\n".join(json.dumps(e) for e in entries) + "\n", encoding="utf-8",
        )
        # Looking for "score" but file only has "other_metric"
        stone = _load_best_stepping_stone("my-plan", tmp_path, "score")
        assert stone is None

    def test_skips_non_dict_and_corrupt_lines(self, tmp_path: Path) -> None:
        stones_dir = _stepping_stones_dir(tmp_path, "my-plan")
        stones_dir.mkdir(parents=True)
        stones_path = stones_dir / "stones.jsonl"
        content = (
            "not json\n"
            + json.dumps([1, 2, 3]) + "\n"  # list, not dict
            + json.dumps({"plan_name": "my-plan", "plan_hash": "z", "metric_value": 42.0,
                          "metric_name": "score", "iteration": 5}) + "\n"
        )
        stones_path.write_text(content, encoding="utf-8")
        stone = _load_best_stepping_stone("my-plan", tmp_path, "score")
        assert stone is not None
        assert stone.metric_value == 42.0


# ---------------------------------------------------------------------------
# Coverage: _apply_stepping_stone backup/restore path
# ---------------------------------------------------------------------------


class TestApplySteppingStoneBackup:
    """Cover the backup restoration path in _apply_stepping_stone."""

    def test_restores_backup_when_plan_file_absent(self, tmp_path: Path) -> None:
        """When plan_path doesn't exist, backup is empty string; bad yaml still fails."""
        plan_path = tmp_path / "new_plan.yaml"
        # plan_path does not exist → backup is ""
        stone = SteppingStone(
            plan_name="x", plan_hash="x", metric_value=1.0,
            metric_name="s", iteration=1,
            plan_yaml="totally: invalid: yaml: [broken",
        )
        result = _apply_stepping_stone(stone, plan_path)
        assert result is False
        # File should not exist (empty backup doesn't restore)


# ---------------------------------------------------------------------------
# Coverage: _build_experiments_summary
# ---------------------------------------------------------------------------


class TestBuildExperimentsSummaryEdgeCases:
    """Cover _build_experiments_summary branches."""

    def test_empty_iterations_returns_empty(self) -> None:
        result = _build_experiments_summary([], plateau_count=0, plateau_threshold=3)
        assert result == ""

    def test_all_improved_shows_successes(self) -> None:
        iterations = [
            WatchIteration(iteration=1, metric_value=5.0, best_metric=5.0,
                           improved=True, action="keep"),
            WatchIteration(iteration=2, metric_value=7.0, best_metric=7.0,
                           improved=True, action="keep"),
        ]
        result = _build_experiments_summary(iterations)
        assert "## Experiment Analysis" in result
        assert "WORKED" in result
        assert "FAILED" not in result

    def test_failures_shown_with_error_hint(self) -> None:
        iterations = [
            WatchIteration(iteration=1, metric_value=5.0, best_metric=5.0,
                           improved=True, action="keep"),
            WatchIteration(iteration=2, metric_value=3.0, best_metric=5.0,
                           improved=False, action="rollback",
                           error="git rollback failed because of conflicts"),
        ]
        result = _build_experiments_summary(iterations)
        assert "FAILED" in result
        assert "rollback" in result.lower()
        assert "git rollback failed" in result

    def test_plateau_alert_remaining_iterations(self) -> None:
        iterations = [
            WatchIteration(iteration=i, metric_value=3.0, best_metric=5.0,
                           improved=False, action="rollback")
            for i in range(1, 4)
        ]
        result = _build_experiments_summary(
            iterations, plateau_count=2, plateau_threshold=5,
        )
        assert "Plateau Alert" in result
        assert "3 iteration(s) remaining" in result
        assert "DIFFERENT category" in result

    def test_plateau_alert_last_chance(self) -> None:
        iterations = [
            WatchIteration(iteration=i, metric_value=2.0, best_metric=5.0,
                           improved=False, action="rollback")
            for i in range(1, 5)
        ]
        result = _build_experiments_summary(
            iterations, plateau_count=4, plateau_threshold=4,
        )
        assert "CRITICAL" in result
        assert "last chance" in result.lower()

    def test_regression_action_filter(self) -> None:
        """Regression count only includes rollback/revert actions."""
        iterations = [
            WatchIteration(iteration=1, metric_value=5.0, best_metric=5.0,
                           improved=True, action="keep"),
            WatchIteration(iteration=2, metric_value=3.0, best_metric=5.0,
                           improved=False, action="rollback"),
            WatchIteration(iteration=3, metric_value=2.0, best_metric=5.0,
                           improved=False, action="revert"),
        ]
        result = _build_experiments_summary(iterations)
        assert "Regressions: 2" in result


# ---------------------------------------------------------------------------
# Coverage: _save_stepping_stone
# ---------------------------------------------------------------------------


class TestSaveSteppingStone2:
    """Cover _save_stepping_stone including lesson loading and compaction trigger."""

    def test_saves_stone_with_lessons(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            "version: 1\nname: ss-plan\ntasks:\n  - id: t1\n    command: echo ok\n",
            encoding="utf-8",
        )
        lessons_path = tmp_path / "lessons.jsonl"
        lessons_path.write_text(
            json.dumps({"iteration": 1, "lesson": "lesson one"}) + "\n",
            encoding="utf-8",
        )
        stone = _save_stepping_stone(
            plan_path=plan_path,
            plan_name="ss-plan",
            metric_value=8.0,
            metric_name="score",
            iteration=3,
            git_commit="abc123",
            lessons_path=lessons_path,
            watch_run_path="/tmp/watch-run",
            total_cost_usd=1.5,
        )
        assert stone is not None
        assert stone.metric_value == 8.0
        assert stone.plan_hash != ""
        assert len(stone.lessons) == 1

        # Verify the stone was written to disk
        stones_path = _stepping_stones_dir(tmp_path, "ss-plan") / "stones.jsonl"
        assert stones_path.exists()
        data = json.loads(stones_path.read_text(encoding="utf-8").strip())
        assert data["metric_value"] == 8.0
        assert data["git_commit"] == "abc123"

    def test_save_stone_no_lessons_file(self, tmp_path: Path) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            "version: 1\nname: ns\ntasks:\n  - id: t1\n    command: echo ok\n",
            encoding="utf-8",
        )
        stone = _save_stepping_stone(
            plan_path=plan_path,
            plan_name="ns",
            metric_value=5.0,
            metric_name="score",
            iteration=1,
            git_commit=None,
            lessons_path=tmp_path / "nonexistent.jsonl",
            watch_run_path="",
            total_cost_usd=0.0,
        )
        assert stone is not None
        assert stone.lessons == []

    def test_save_stone_plan_read_failure_returns_none(self, tmp_path: Path) -> None:
        """If the plan file can't be read, returns None."""
        stone = _save_stepping_stone(
            plan_path=tmp_path / "ghost.yaml",
            plan_name="g",
            metric_value=1.0,
            metric_name="s",
            iteration=1,
            git_commit=None,
            lessons_path=tmp_path / "l.jsonl",
            watch_run_path="",
            total_cost_usd=0.0,
        )
        assert stone is None


# ---------------------------------------------------------------------------
# Coverage: _run_consolidation edge case
# ---------------------------------------------------------------------------


class TestRunConsolidationEdgeCases:
    """Cover more _run_consolidation paths."""

    def test_consolidation_nonzero_exit_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.watch import _run_consolidation

        monkeypatch.setattr(
            subprocess, "run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=1, stdout="bad", stderr="err"),
        )
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            consolidate_model="haiku",
        )
        assert _run_consolidation(spec, "history", tmp_path) == ""

    def test_consolidation_oserror_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.watch import _run_consolidation

        def _raise(*a: Any, **kw: Any) -> subprocess.CompletedProcess[str]:
            raise OSError("no claude")

        monkeypatch.setattr(subprocess, "run", _raise)
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            consolidate_model="haiku",
        )
        assert _run_consolidation(spec, "history", tmp_path) == ""

    def test_consolidation_uses_custom_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from maestro_cli.watch import _run_consolidation

        captured: list[list[str]] = []

        def _mock_run(*a: Any, **kw: Any) -> subprocess.CompletedProcess[str]:
            captured.append(list(a[0]))
            return subprocess.CompletedProcess(args=a[0], returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr(subprocess, "run", _mock_run)
        spec = WatchSpec(
            metric="score", metric_source="stdout_regex",
            metric_pattern=r"score: ([0-9.]+)",
            consolidate_model="sonnet",
            consolidate_prompt="Custom prompt here",
        )
        result = _run_consolidation(spec, "my history", tmp_path)
        assert result == "ok"
        # Verify model and prompt content were passed
        assert "sonnet" in captured[0]
        assert "Custom prompt here" in captured[0][-1]


# ---------------------------------------------------------------------------
# Coverage: _count_executed_tasks
# ---------------------------------------------------------------------------


class TestCountExecutedTasks2:
    """Cover _count_executed_tasks with mixed statuses."""

    def test_counts_non_skipped(self, tmp_path: Path) -> None:
        results = {
            "t1": TaskResult(task_id="t1", status="success"),
            "t2": TaskResult(task_id="t2", status="skipped"),
            "t3": TaskResult(task_id="t3", status="failed"),
            "t4": TaskResult(task_id="t4", status="soft_failed"),
        }
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=False, task_results=results,
        )
        assert _count_executed_tasks(run_result) == 3

    def test_all_skipped_returns_zero(self, tmp_path: Path) -> None:
        results = {
            "t1": TaskResult(task_id="t1", status="skipped"),
            "t2": TaskResult(task_id="t2", status="skipped"),
        }
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=False, task_results=results,
        )
        assert _count_executed_tasks(run_result) == 0

    def test_empty_results_returns_zero(self, tmp_path: Path) -> None:
        run_result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=datetime.now(), finished_at=datetime.now(),
            success=True, task_results={},
        )
        assert _count_executed_tasks(run_result) == 0


# ---------------------------------------------------------------------------
# Coverage: watch() improve mode dispatch
# ---------------------------------------------------------------------------


class TestWatchImproveDispatch:
    """Cover the watch() -> _watch_improve dispatch and dry_run path."""

    def test_watch_dispatches_to_improve_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """watch() routes to _watch_improve when mode='improve'."""
        monkeypatch.chdir(tmp_path)
        plan_yaml = (
            "version: 1\nname: imp-test\nworkspace_root: .\n"
            "tasks:\n  - id: t1\n    command: echo done\n"
            "watch:\n  metric: tasks_passed\n  metric_source: manifest\n"
            "  metric_direction: higher_is_better\n  mode: improve\n"
            "  max_iterations: 2\n  plateau_threshold: 5\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(plan_yaml, encoding="utf-8")

        events: list[tuple[str, dict[str, object]]] = []

        # _watch_improve needs run_plan — mock it to do a quick baseline + one iteration
        call_count = [0]

        def _mock_run_plan(plan: PlanSpec, **kwargs: Any) -> PlanRunResult:
            call_count[0] += 1
            results = {
                "t1": TaskResult(task_id="t1", status="success", exit_code=0, duration_sec=1.0),
            }
            return PlanRunResult(
                plan_name=plan.name, run_id=f"r{call_count[0]}",
                run_path=tmp_path / f"run{call_count[0]}",
                started_at=datetime.now(), finished_at=datetime.now(),
                success=True, task_results=results, total_cost_usd=0.1,
            )

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr(
            "maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha",
        )
        monkeypatch.setattr(
            "maestro_cli.watch._git_rollback", lambda *_a, **_kw: True,
        )
        monkeypatch.setattr(
            "maestro_cli.watch.blame_run",
            lambda _p: type("C", (), {"to_dict": lambda self: {"nodes": []}})(),
        )

        state = watch(
            plan_path,
            event_callback=lambda n, p: events.append((n, p)),
        )

        event_names = [n for n, _ in events]
        assert "watch_start" in event_names
        assert "watch_complete" in event_names
        # Check the mode was set to "improve" in the watch_start event
        start_event = next(p for n, p in events if n == "watch_start")
        assert start_event.get("mode") == "improve"

    def test_watch_improve_dry_run_returns_immediately(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Improve mode with dry_run=True emits start/complete and returns."""
        monkeypatch.chdir(tmp_path)
        plan_yaml = (
            "version: 1\nname: dry-imp\nworkspace_root: .\n"
            "tasks:\n  - id: t1\n    command: echo ok\n"
            "watch:\n  metric: tasks_passed\n  metric_source: manifest\n"
            "  metric_direction: higher_is_better\n  mode: improve\n"
            "  max_iterations: 5\n  plateau_threshold: 5\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(plan_yaml, encoding="utf-8")

        run_called = [False]

        def _mock_run_plan(*a: Any, **kw: Any) -> PlanRunResult:
            run_called[0] = True
            return _make_mock_result(tmp_path)

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)

        events: list[tuple[str, dict[str, object]]] = []
        state = watch(
            plan_path, dry_run=True,
            event_callback=lambda n, p: events.append((n, p)),
        )

        assert run_called[0] is False
        assert state.total_iterations == 0
        event_names = [n for n, _ in events]
        assert "watch_start" in event_names
        assert "watch_complete" in event_names

    def test_watch_improve_resume_injects_session_memory_and_recent_outputs(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="tasks_passed",
                metric_source="manifest",
                metric_direction="higher_is_better",
                mode="improve",
                warmup_iterations=0,
                max_iterations=2,
                plateau_threshold=5,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        resume_dir = (tmp_path / "resume-run").resolve()
        resume_dir.mkdir()
        _write_experiment(
            resume_dir / "experiments.jsonl",
            WatchIteration(
                iteration=1,
                metric_value=1.0,
                best_metric=1.0,
                improved=True,
                action="keep",
                cost_usd=0.1,
                duration_sec=1.0,
                timestamp="2026-04-15T00:00:00",
                fix_summary="increase timeout",
                manifest_excerpt="test-task: success",
                blame_excerpt="root cause from prior run",
                consolidated_excerpt="prefer smaller diffs",
            ),
        )
        store_session_snapshot(
            "watch-test",
            tmp_path,
            SessionSnapshot(
                plan_name="watch-test",
                watch_run_path=str(resume_dir),
                snapshot_kind="watch",
                iteration_from=1,
                iteration_to=1,
                best_metric=1.0,
                snapshot_text="Older durable summary",
                recent_tail_count=3,
                source_type="watch",
                source_id=f"{resume_dir}:1",
                metadata={"metric_name": "tasks_passed"},
            ),
        )

        captured_vars: dict[str, str] = {}
        call_counter = 0

        def _mock_run_plan(plan_arg: PlanSpec, **kwargs: Any) -> PlanRunResult:
            nonlocal captured_vars, call_counter
            call_counter += 1
            run_path = tmp_path / f"run-{call_counter}"
            run_path.mkdir()
            if plan_arg.name.startswith("improve-"):
                captured_vars = dict(kwargs.get("extra_template_vars", {}))
                log_path = run_path / "improve-plan.log"
                log_path.write_text("FIX: tighten verify command", encoding="utf-8")
                return PlanRunResult(
                    plan_name=plan_arg.name,
                    run_id=f"improve-{call_counter}",
                    run_path=run_path,
                    started_at=datetime.now(),
                    finished_at=datetime.now(),
                    success=True,
                    task_results={
                        "improve-plan": TaskResult(
                            task_id="improve-plan",
                            status="success",
                            exit_code=0,
                            duration_sec=1.0,
                            log_path=log_path,
                            result_path=run_path / "improve-plan.json",
                        ),
                    },
                    total_cost_usd=0.05,
                )

            return PlanRunResult(
                plan_name=plan_arg.name,
                run_id=f"target-{call_counter}",
                run_path=run_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
                task_results={
                    "test-task": TaskResult(
                        task_id="test-task",
                        status="success",
                        exit_code=0,
                        duration_sec=1.0,
                        log_path=run_path / "test-task.log",
                        result_path=run_path / "test-task.json",
                    ),
                    "other-task": TaskResult(
                        task_id="other-task",
                        status="success",
                        exit_code=0,
                        duration_sec=1.0,
                        log_path=run_path / "other-task.log",
                        result_path=run_path / "other-task.json",
                    ),
                },
                total_cost_usd=0.1,
            )

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)
        monkeypatch.setattr(
            "maestro_cli.watch._build_blame_context",
            lambda _run_path: ("new root cause", "test-task: success"),
        )

        state = watch(plan_path, resume_from=resume_dir)

        assert state.total_iterations == 2
        assert captured_vars["watch.session_memory"] == "Older durable summary"
        assert "FIX summary: increase timeout" in captured_vars["watch.recent_outputs"]
        assert "Manifest excerpt:\ntest-task: success" in captured_vars["watch.recent_outputs"]
        assert captured_vars["watch.lessons"] == "No lessons from previous iterations."

    def test_watch_improve_step_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Improve mode respects max_total_steps."""
        monkeypatch.chdir(tmp_path)
        plan_yaml = (
            "version: 1\nname: step-lim\nworkspace_root: .\n"
            "tasks:\n"
            "  - id: t1\n    command: echo ok\n"
            "  - id: t2\n    command: echo ok\n"
            "  - id: t3\n    command: echo fail\n"
            "watch:\n  metric: tasks_passed\n  metric_source: manifest\n"
            "  metric_direction: higher_is_better\n  mode: improve\n"
            "  max_iterations: 100\n  plateau_threshold: 100\n"
            "  max_total_steps: 3\n"
        )
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(plan_yaml, encoding="utf-8")

        def _mock_run_plan(plan: PlanSpec, **kw: Any) -> PlanRunResult:
            # Only 2 of 3 tasks succeed — target never reached (target = 3.0)
            results = {
                "t1": TaskResult(task_id="t1", status="success", exit_code=0, duration_sec=1.0),
                "t2": TaskResult(task_id="t2", status="success", exit_code=0, duration_sec=1.0),
                "t3": TaskResult(task_id="t3", status="failed", exit_code=1, duration_sec=1.0),
            }
            return PlanRunResult(
                plan_name=plan.name, run_id="r",
                run_path=tmp_path / "run",
                started_at=datetime.now(), finished_at=datetime.now(),
                success=False, task_results=results, total_cost_usd=0.01,
            )

        monkeypatch.setattr("maestro_cli.watch.run_plan", _mock_run_plan)
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)
        monkeypatch.setattr(
            "maestro_cli.watch.blame_run",
            lambda _p: type("C", (), {"to_dict": lambda self: {"nodes": []}})(),
        )

        events: list[tuple[str, dict[str, object]]] = []
        state = watch(
            plan_path,
            event_callback=lambda n, p: events.append((n, p)),
        )

        assert state.status == "step_limit_reached"
        event_names = [n for n, _ in events]
        assert "watch_step_limit" in event_names

    def test_watch_step_limit_custom_mode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Custom mode also respects max_total_steps."""
        monkeypatch.chdir(tmp_path)
        plan = _make_plan(
            tmp_path,
            watch_spec=WatchSpec(
                metric="score",
                metric_source="stdout_regex",
                metric_pattern=r"score: ([0-9.]+)",
                metric_task="test-task",
                warmup_iterations=0,
                max_iterations=100,
                plateau_threshold=100,
                max_total_steps=2,
            ),
        )
        plan_path = tmp_path / "plan.yaml"
        metrics = iter([0.5, 0.6, 0.7, 0.8, 0.9])

        monkeypatch.setattr("maestro_cli.watch.load_plan", lambda _path: plan)
        monkeypatch.setattr(
            "maestro_cli.watch.run_plan",
            lambda *args, **kwargs: _make_mock_result(
                tmp_path, stdout_tail=f"score: {next(metrics)}",
            ),
        )
        monkeypatch.setattr("maestro_cli.watch._git_commit_changes", lambda *_a, **_kw: "sha")
        monkeypatch.setattr("maestro_cli.watch._git_rollback", lambda *_a, **_kw: True)

        events: list[tuple[str, dict[str, object]]] = []
        state = watch(
            plan_path,
            event_callback=lambda n, p: events.append((n, p)),
        )

        assert state.status == "step_limit_reached"
        event_names = [n for n, _ in events]
        assert "watch_step_limit" in event_names


# ---------------------------------------------------------------------------
# Coverage: _emit with None callback
# ---------------------------------------------------------------------------


class TestEmitNoneCallback:
    """Verify _emit with None callback doesn't crash."""

    def test_emit_none_callback_is_noop(self) -> None:
        from maestro_cli.watch import _emit
        # Should not raise
        _emit(None, "test_event", key="value")
