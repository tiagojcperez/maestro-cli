from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.eventsource import replay_events, verify_chain
from maestro_cli.models import PlanDefaults, PlanSpec, TaskResult, TaskSpec, TokenUsage
from maestro_cli.scheduler import run_plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task(
    task_id: str,
    depends_on: list[str] | None = None,
    command: str = "echo ok",
    allow_failure: bool = False,
) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        description=f"task {task_id}",
        depends_on=depends_on or [],
        command=command,
        allow_failure=allow_failure,
    )


def _make_plan(
    tasks: list[TaskSpec],
    name: str = "integration-test-plan",
    fail_fast: bool = False,
    max_parallel: int = 4,
    max_cost_usd: float | None = None,
    source_path: Path | None = None,
) -> PlanSpec:
    plan = PlanSpec(
        version=1,
        name=name,
        fail_fast=fail_fast,
        max_parallel=max_parallel,
        defaults=PlanDefaults(),
        tasks=tasks,
        source_path=source_path,
    )
    plan.max_cost_usd = max_cost_usd
    return plan


def _make_mock_execute(
    run_path_holder: list[Path],
    overrides: dict[str, TaskResult] | None = None,
    call_log: list[str] | None = None,
) -> Any:
    """Return (mock_fn, call_log). mock_fn records calls and returns success by default."""
    overrides = overrides or {}
    if call_log is None:
        call_log = []
    lock = threading.Lock()

    def mock_execute(
        plan: Any,
        task: Any,
        run_path: Path,
        dry_run: bool = False,
        execution_profile: str = "plan",
        upstream_results: Any = None,
        context_synthesis: str = "",
        workspace_brief: str = "",
        **kwargs: Any,
    ) -> TaskResult:
        if not run_path_holder:
            run_path_holder.append(run_path)
        with lock:
            call_log.append(task.id)

        if task.id in overrides:
            result = overrides[task.id]
            result.log_path.write_text(f"status={result.status}\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        now = datetime.now(UTC)
        status: str = "dry_run" if dry_run else "success"
        result = TaskResult(
            task_id=task.id,
            status=status,  # type: ignore[arg-type]
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_sec=0.01,
            command=f"echo {task.id}",
            log_path=run_path / f"{task.id}.log",
            result_path=run_path / f"{task.id}.result.json",
            message="ok",
        )
        result.log_path.write_text(f"status={status}\n", encoding="utf-8")
        result.result_path.write_text(
            json.dumps(result.to_dict(), indent=2), encoding="utf-8"
        )
        return result

    return mock_execute, call_log


def _parse_jsonl(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            events.append(json.loads(line))
    return events


# ---------------------------------------------------------------------------
# Test 1: Manifest schema — required fields present and correctly typed
# ---------------------------------------------------------------------------


class TestManifestSchema:
    def test_manifest_required_top_level_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_manifest.json contains required top-level fields with correct types."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("t1"), _make_task("t2", depends_on=["t1"])],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        manifest_path = result.run_path / "run_manifest.json"
        assert manifest_path.exists(), "run_manifest.json must exist after run"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # Top-level required fields
        for field_name in ("plan_name", "run_id", "run_path", "started_at", "finished_at", "success", "task_results"):
            assert field_name in manifest, f"Missing field '{field_name}' in manifest"

        # Type checks
        assert isinstance(manifest["plan_name"], str)
        assert isinstance(manifest["run_id"], str)
        assert isinstance(manifest["run_path"], str)
        assert isinstance(manifest["started_at"], str)
        assert isinstance(manifest["finished_at"], str)
        assert isinstance(manifest["success"], bool)
        assert isinstance(manifest["task_results"], dict)

        # Timestamps must be ISO 8601
        datetime.fromisoformat(manifest["started_at"])
        datetime.fromisoformat(manifest["finished_at"])

    def test_manifest_per_task_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each task entry in task_results has required per-task fields."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("task-a"), _make_task("task-b")],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "task-a" in manifest["task_results"]
        assert "task-b" in manifest["task_results"]

        for task_id in ("task-a", "task-b"):
            tr = manifest["task_results"][task_id]
            for req in ("status", "exit_code", "duration_sec", "message", "cost_usd", "token_usage"):
                assert req in tr, f"Missing per-task field '{req}' for '{task_id}'"
            assert isinstance(tr["status"], str)
            assert isinstance(tr["duration_sec"], float)


# ---------------------------------------------------------------------------
# Test 2: Event ordering — run_start first, run_complete last,
#          task_start < task_complete for each task
# ---------------------------------------------------------------------------


class TestEventOrdering:
    def test_task_start_before_task_complete_for_each_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For every task, task_start appears before task_complete in events.jsonl."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("alpha"),
                _make_task("beta", depends_on=["alpha"]),
                _make_task("gamma", depends_on=["alpha"]),
            ],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        event_names = [e["event"] for e in events]

        # run_start must be first
        assert event_names[0] == "run_start", f"Expected run_start first, got {event_names[0]}"
        # run_complete must be last
        assert event_names[-1] == "run_complete", f"Expected run_complete last, got {event_names[-1]}"

        # For each task, task_start index < task_complete index
        task_ids = ["alpha", "beta", "gamma"]
        for tid in task_ids:
            starts = [i for i, e in enumerate(events) if e.get("event") == "task_start" and e.get("task_id") == tid]
            completes = [i for i, e in enumerate(events) if e.get("event") == "task_complete" and e.get("task_id") == tid]
            assert starts, f"No task_start event found for '{tid}'"
            assert completes, f"No task_complete event found for '{tid}'"
            assert starts[0] < completes[0], (
                f"task_start for '{tid}' (idx {starts[0]}) must precede "
                f"task_complete (idx {completes[0]})"
            )


# ---------------------------------------------------------------------------
# Test 3: Hash chain integrity — verify_chain() returns "valid"
# ---------------------------------------------------------------------------


class TestHashChainIntegrity:
    def test_verify_chain_passes_on_produced_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """verify_chain() reports 'valid' for the events.jsonl written by run_plan()."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("x1"),
                _make_task("x2", depends_on=["x1"]),
                _make_task("x3", depends_on=["x1"]),
                _make_task("x4", depends_on=["x2", "x3"]),
            ],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events_path = result.run_path / "events.jsonl"
        assert events_path.exists()

        records = replay_events(events_path)
        status = verify_chain(records)
        assert status == "valid", (
            f"Expected hash chain to be 'valid', got '{status}'. "
            f"Events file: {events_path}"
        )

    def test_verify_chain_detects_tampering(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """verify_chain() returns 'tampered' when events.jsonl is modified after the run."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("s1"), _make_task("s2", depends_on=["s1"])],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events_path = result.run_path / "events.jsonl"
        original = events_path.read_text(encoding="utf-8")

        # Tamper: modify a character in the middle of the file
        mid = len(original) // 2
        tampered = original[:mid] + ("X" if original[mid] != "X" else "Y") + original[mid + 1:]
        events_path.write_text(tampered, encoding="utf-8")

        records = replay_events(events_path)
        status = verify_chain(records)
        assert status == "tampered", (
            f"Expected 'tampered' after modifying events.jsonl, got '{status}'"
        )


# ---------------------------------------------------------------------------
# Test 4: Failed Tasks section appears in run_summary.md when a task fails
# ---------------------------------------------------------------------------


class TestSummaryFailedSection:
    def test_failed_tasks_section_present_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_summary.md includes a '## Failed Tasks' section when a task fails."""
        run_path_holder: list[Path] = []

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            if not run_path_holder:
                run_path_holder.append(run_path)
            now = datetime.now(UTC)
            if task.id == "fail-me":
                r = TaskResult(
                    task_id=task.id,
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="false",
                    log_path=run_path / f"{task.id}.log",
                    result_path=run_path / f"{task.id}.result.json",
                    message="Intentional failure for test",
                )
            else:
                r = TaskResult(
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
            r.log_path.write_text(f"status={r.status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [_make_task("ok-task"), _make_task("fail-me")],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        summary_path = result.run_path / "run_summary.md"
        assert summary_path.exists(), "run_summary.md must be created after run"
        summary = summary_path.read_text(encoding="utf-8")

        assert "## Failed Tasks" in summary, (
            "Expected '## Failed Tasks' section in summary when a task fails"
        )
        assert "fail-me" in summary, "Failed task ID must appear in the Failed Tasks section"
        assert "Intentional failure for test" in summary, (
            "Failure message must appear in the Failed Tasks section"
        )

    def test_no_failed_tasks_section_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_summary.md does NOT include a '## Failed Tasks' section on a clean run."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("p1"), _make_task("p2", depends_on=["p1"])],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")
        assert "## Failed Tasks" not in summary, (
            "Should not have '## Failed Tasks' section when all tasks succeed"
        )


# ---------------------------------------------------------------------------
# Test 5: Diamond DAG — A → B, A → C, B + C → D execute in correct order
# ---------------------------------------------------------------------------


class TestDiamondDAGOrder:
    def test_diamond_dag_execution_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In a diamond DAG (A→B, A→C, B+C→D), A finishes before B/C start,
        and both B and C finish before D starts."""
        call_log: list[str] = []
        call_times: dict[str, int] = {}
        lock = threading.Lock()
        counter = [0]

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                counter[0] += 1
                call_times[task.id] = counter[0]
                call_log.append(task.id)

            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("node-a"),
                _make_task("node-b", depends_on=["node-a"]),
                _make_task("node-c", depends_on=["node-a"]),
                _make_task("node-d", depends_on=["node-b", "node-c"]),
            ],
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success, "Diamond DAG run should succeed"
        assert set(call_log) == {"node-a", "node-b", "node-c", "node-d"}, (
            "All 4 tasks must have been executed"
        )

        # A must be called before B and C
        assert call_times["node-a"] < call_times["node-b"], "A must start before B"
        assert call_times["node-a"] < call_times["node-c"], "A must start before C"
        # B and C must both be called before D
        assert call_times["node-b"] < call_times["node-d"], "B must start before D"
        assert call_times["node-c"] < call_times["node-d"], "C must start before D"

        # Also verify events reflect the same ordering
        events = _parse_jsonl(result.run_path / "events.jsonl")
        complete_order = [
            e["task_id"]
            for e in events
            if e.get("event") == "task_complete"
        ]
        assert complete_order.index("node-a") < complete_order.index("node-d"), (
            "node-a must complete before node-d in events"
        )
        assert complete_order.index("node-b") < complete_order.index("node-d"), (
            "node-b must complete before node-d in events"
        )
        assert complete_order.index("node-c") < complete_order.index("node-d"), (
            "node-c must complete before node-d in events"
        )


# ---------------------------------------------------------------------------
# Test 6: Budget exhaustion — max_cost_usd causes tasks to be skipped
# ---------------------------------------------------------------------------


class TestBudgetExhaustion:
    def test_budget_exhaustion_skips_tasks_and_sets_flag(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When max_cost_usd is exceeded after first task, remaining tasks are skipped
        and PlanRunResult.budget_exceeded is True."""

        def mock_costly_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
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
                cost_usd=5.00,  # each task costs $5
                token_usage=TokenUsage(input_tokens=1000, output_tokens=1000),
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_costly_execute)

        # 3 independent tasks but budget only allows 1 ($5 limit, each costs $5)
        plan = _make_plan(
            [_make_task("cost-1"), _make_task("cost-2"), _make_task("cost-3")],
            fail_fast=False,
            max_parallel=1,  # serial so budget check fires between tasks
            max_cost_usd=4.99,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.budget_exceeded is True, "budget_exceeded must be True"

        # At least one task should have been skipped
        statuses = {tid: tr.status for tid, tr in result.task_results.items()}
        skipped = [tid for tid, s in statuses.items() if s == "skipped"]
        assert skipped, (
            f"Expected at least one skipped task after budget exhaustion, "
            f"got statuses: {statuses}"
        )

        # Manifest must also reflect budget_exceeded
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["budget_exceeded"] is True, (
            "budget_exceeded must be True in run_manifest.json"
        )


# ---------------------------------------------------------------------------
# Test 7: fail_fast — first failure skips all remaining tasks
# ---------------------------------------------------------------------------


class TestFailFast:
    def test_fail_fast_skips_remaining_tasks_on_first_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With fail_fast=True, a single task failure causes all other pending
        (non-dependent) tasks to be skipped."""

        executed_tasks: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed_tasks.append(task.id)
            now = datetime.now(UTC)
            if task.id == "ff-fail":
                r = TaskResult(
                    task_id=task.id,
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="false",
                    log_path=run_path / f"{task.id}.log",
                    result_path=run_path / f"{task.id}.result.json",
                    message="Failure that triggers fail_fast",
                )
            else:
                r = TaskResult(
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
            r.log_path.write_text(f"status={r.status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        # 4 independent tasks; ff-fail will fail; the others should be skipped
        plan = _make_plan(
            [
                _make_task("ff-fail"),
                _make_task("ff-ok-1"),
                _make_task("ff-ok-2"),
                _make_task("ff-ok-3"),
            ],
            fail_fast=True,
            max_parallel=1,  # serial so fail_fast fires before subsequent tasks start
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False, "Plan should fail when fail_fast task fails"

        statuses = {tid: tr.status for tid, tr in result.task_results.items()}
        assert statuses.get("ff-fail") == "failed", "ff-fail must have status 'failed'"

        # With fail_fast and serial execution, not all tasks should have been executed
        skipped = [tid for tid, s in statuses.items() if s == "skipped"]
        assert skipped, (
            f"Expected skipped tasks after fail_fast trigger, "
            f"got statuses: {statuses}"
        )
        # Skipped tasks should not have been passed to execute_task
        for tid in skipped:
            assert tid not in executed_tasks, (
                f"Skipped task '{tid}' should not have been passed to execute_task"
            )


# ---------------------------------------------------------------------------
# Test 8: allow_failure — soft_failed task does not block dependents
# ---------------------------------------------------------------------------


class TestAllowFailure:
    def test_soft_failed_does_not_block_dependents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A task with allow_failure=True that fails becomes soft_failed,
        and its downstream dependents still execute."""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            # Return soft_failed (as execute_task does when allow_failure=True and task fails)
            if task.id == "soft-fail-task":
                r = TaskResult(
                    task_id=task.id,
                    status="soft_failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="false",
                    log_path=run_path / f"{task.id}.log",
                    result_path=run_path / f"{task.id}.result.json",
                    message="Allowed failure",
                )
            else:
                r = TaskResult(
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
            r.log_path.write_text(f"status={r.status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("pre-task"),
                _make_task("soft-fail-task", depends_on=["pre-task"], allow_failure=True),
                _make_task("post-task", depends_on=["soft-fail-task"]),
            ],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # A soft_failed task does not count as a plan-level failure
        assert result.success is True, "Plan should succeed when only soft_failed task is present"

        # The dependent task must have been called — soft_failed doesn't block dependents
        assert "post-task" in executed, (
            "post-task must execute even though its dependency was soft_failed"
        )

        # soft-fail-task must remain soft_failed in the final results
        assert result.task_results["soft-fail-task"].status == "soft_failed", (
            "soft_failed task must be recorded as soft_failed in PlanRunResult"
        )


# ---------------------------------------------------------------------------
# Test 9: Per-task result files — *.result.json and *.log exist with schema
# ---------------------------------------------------------------------------


class TestPerTaskResultFiles:
    def test_result_json_schema_per_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each executed task produces a <task-id>.result.json with required fields,
        and a <task-id>.log transcript file."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("res-a"), _make_task("res-b", depends_on=["res-a"])],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        for task_id in ("res-a", "res-b"):
            # result.json must exist with required fields
            result_file = result.run_path / f"{task_id}.result.json"
            assert result_file.exists(), f"{task_id}.result.json must exist"
            data = json.loads(result_file.read_text(encoding="utf-8"))
            for req in ("task_id", "status", "exit_code", "duration_sec"):
                assert req in data, f"Missing '{req}' in {task_id}.result.json"
            assert data["task_id"] == task_id, (
                f"task_id in {task_id}.result.json must match the task"
            )
            assert isinstance(data["exit_code"], int)
            assert isinstance(data["duration_sec"], float)

            # log file must also exist
            log_file = result.run_path / f"{task_id}.log"
            assert log_file.exists(), f"{task_id}.log must exist after execution"


# ---------------------------------------------------------------------------
# Test 10: Event hash chain fields — seq, hash, prev_hash present and monotonic
# ---------------------------------------------------------------------------


class TestEventHashChainFields:
    def test_events_have_seq_hash_prev_hash_and_monotonic_seq(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every event has seq, hash, and prev_hash fields; seq is monotonically
        increasing across the chain (tamper-evident event sourcing contract)."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("hc-1"), _make_task("hc-2", depends_on=["hc-1"]), _make_task("hc-3", depends_on=["hc-1"])],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        assert events, "events.jsonl must be non-empty"

        seqs: list[int] = []
        for i, evt in enumerate(events):
            assert "seq" in evt, f"Event {i} ({evt.get('event')!r}) missing 'seq'"
            assert "hash" in evt, f"Event {i} ({evt.get('event')!r}) missing 'hash'"
            assert "prev_hash" in evt, f"Event {i} ({evt.get('event')!r}) missing 'prev_hash'"
            assert isinstance(evt["seq"], int), f"Event {i} 'seq' must be int"
            assert isinstance(evt["hash"], str) and evt["hash"], (
                f"Event {i} 'hash' must be a non-empty string"
            )
            seqs.append(evt["seq"])

        # seq must be strictly increasing
        for i in range(1, len(seqs)):
            assert seqs[i] > seqs[i - 1], (
                f"seq not monotonically increasing at position {i}: {seqs[i-1]} → {seqs[i]}"
            )


# ---------------------------------------------------------------------------
# Test 11: plan_name auto-injection — every event carries the plan name
# ---------------------------------------------------------------------------


class TestPlanNameInjection:
    def test_plan_name_present_in_all_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every event in events.jsonl has a 'plan_name' field that matches
        the plan's name (auto-injected by _emit())."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("pn-1"), _make_task("pn-2", depends_on=["pn-1"])],
            name="my-injection-test-plan",
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        assert events, "events.jsonl must be non-empty"

        for i, evt in enumerate(events):
            assert "plan_name" in evt, (
                f"Event {i} ({evt.get('event')!r}) is missing 'plan_name'"
            )
            assert evt["plan_name"] == "my-injection-test-plan", (
                f"Event {i} has wrong plan_name: {evt['plan_name']!r}"
            )


# ---------------------------------------------------------------------------
# Test 12: Summary content — task IDs appear in the task table
# ---------------------------------------------------------------------------


class TestSummaryContent:
    def test_summary_contains_task_ids_and_plan_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_summary.md contains the plan name and each task ID in the task table."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("summary-a"), _make_task("summary-b", depends_on=["summary-a"])],
            name="content-check-plan",
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        summary_path = result.run_path / "run_summary.md"
        assert summary_path.exists(), "run_summary.md must be written after run"
        summary = summary_path.read_text(encoding="utf-8")

        # Plan name must appear somewhere in the summary
        assert "content-check-plan" in summary, (
            "Plan name must appear in run_summary.md"
        )
        # Both task IDs must appear (task table)
        assert "summary-a" in summary, "summary-a must appear in run_summary.md"
        assert "summary-b" in summary, "summary-b must appear in run_summary.md"
        # Summary must have markdown table markers
        assert "|" in summary, "run_summary.md must contain a markdown table"


# ---------------------------------------------------------------------------
# Test 13: PlanRunResult.total_cost_usd matches sum of per-task costs
# ---------------------------------------------------------------------------


class TestTotalCostAggregation:
    def test_total_cost_usd_equals_sum_of_task_costs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PlanRunResult.total_cost_usd equals the sum of all per-task cost_usd values,
        and the same value appears in run_manifest.json under 'total_cost_usd'."""

        per_task_cost = 1.25

        def mock_costly_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
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
                cost_usd=per_task_cost,
                token_usage=TokenUsage(input_tokens=100, output_tokens=50),
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_costly_execute)

        plan = _make_plan(
            [
                _make_task("cost-a"),
                _make_task("cost-b"),
                _make_task("cost-c"),
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        expected_total = per_task_cost * 3
        assert result.total_cost_usd is not None, "total_cost_usd must be set"
        assert abs(result.total_cost_usd - expected_total) < 0.001, (
            f"Expected total_cost_usd={expected_total:.4f}, "
            f"got {result.total_cost_usd:.4f}"
        )

        # manifest must echo the same value
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "total_cost_usd" in manifest, "total_cost_usd must be in manifest"
        assert abs(manifest["total_cost_usd"] - expected_total) < 0.001, (
            f"Manifest total_cost_usd={manifest['total_cost_usd']:.4f} "
            f"does not match expected {expected_total:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 14: run_manifest.json round-trips through PlanRunResult.to_dict()
# ---------------------------------------------------------------------------


class TestManifestRoundTrip:
    def test_manifest_matches_plan_run_result_to_dict(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_manifest.json is the exact JSON serialisation of PlanRunResult.to_dict().
        Top-level keys in the manifest must match to_dict() output keys exactly."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("rt-1"), _make_task("rt-2", depends_on=["rt-1"])],
            name="round-trip-plan",
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        expected_dict = result.to_dict()

        # Every key produced by to_dict() must be present in the manifest
        for key in expected_dict:
            assert key in manifest, (
                f"Key '{key}' from PlanRunResult.to_dict() is missing in run_manifest.json"
            )

        # Core value checks: plan_name, run_id, success must match Python object
        assert manifest["plan_name"] == result.plan_name
        assert manifest["run_id"] == result.run_id
        assert manifest["success"] == result.success
        assert set(manifest["task_results"].keys()) == set(result.task_results.keys()), (
            "task_results keys in manifest must match PlanRunResult.task_results keys"
        )


# ---------------------------------------------------------------------------
# Test 15: task_complete event status matches manifest task status
# ---------------------------------------------------------------------------


class TestEventStatusMatchesManifest:
    def test_task_complete_event_status_matches_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For each task, the 'status' field in the task_complete event must equal
        the 'status' field in the corresponding entry in run_manifest.json task_results.
        This cross-module contract ensures events and manifest never diverge."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("ev-a"),
                _make_task("ev-b", depends_on=["ev-a"]),
                _make_task("ev-c", depends_on=["ev-a"]),
            ],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )

        complete_events = {
            e["task_id"]: e["status"]
            for e in events
            if e.get("event") == "task_complete" and "task_id" in e
        }

        assert complete_events, "Expected at least one task_complete event"

        for task_id, event_status in complete_events.items():
            assert task_id in manifest["task_results"], (
                f"task_id '{task_id}' from task_complete event not found in manifest"
            )
            manifest_status = manifest["task_results"][task_id]["status"]
            assert event_status == manifest_status, (
                f"Status mismatch for task '{task_id}': "
                f"task_complete event has '{event_status}', "
                f"manifest has '{manifest_status}'"
            )


# ---------------------------------------------------------------------------
# Test 16: replay_run_state() reconstructs completed tasks and total cost
# ---------------------------------------------------------------------------


class TestReplayRunState:
    def test_replay_run_state_matches_plan_run_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """replay_run_state() reconstructed from events.jsonl must agree with
        PlanRunResult on which tasks completed and total cost.

        This validates the cross-module contract between the event log
        (produced by scheduler.py via eventsource.py) and the in-memory result
        (PlanRunResult).  If events and result diverge, replay-based resume and
        audit tooling will silently produce wrong state."""
        from maestro_cli.eventsource import replay_events, replay_run_state

        per_task_cost = 0.75

        def mock_execute_with_cost(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
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
                cost_usd=per_task_cost,
                token_usage=TokenUsage(input_tokens=50, output_tokens=50),
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute_with_cost)

        plan = _make_plan(
            [
                _make_task("rr-x"),
                _make_task("rr-y", depends_on=["rr-x"]),
                _make_task("rr-z", depends_on=["rr-x"]),
            ],
            fail_fast=False,
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events_path = result.run_path / "events.jsonl"
        records = replay_events(events_path)
        state = replay_run_state(records)

        # All 3 tasks must appear in completed_tasks
        expected_task_ids = {"rr-x", "rr-y", "rr-z"}
        assert state["completed_tasks"] == expected_task_ids, (
            f"replay_run_state completed_tasks={state['completed_tasks']!r} "
            f"does not match expected {expected_task_ids!r}"
        )

        # All tasks must have 'success' status in replayed state
        for tid in expected_task_ids:
            assert state["tasks"].get(tid) == "success", (
                f"Replayed status for '{tid}' is {state['tasks'].get(tid)!r}, expected 'success'"
            )

        # Total cost from events must equal sum of per-task costs
        expected_cost = per_task_cost * 3
        assert abs(state["total_cost_usd"] - expected_cost) < 0.001, (
            f"replay_run_state total_cost_usd={state['total_cost_usd']:.4f} "
            f"does not match expected {expected_cost:.4f}"
        )


# ---------------------------------------------------------------------------
# Test 17: run_start event carries goal field when plan sets a goal
# ---------------------------------------------------------------------------


class TestRunStartGoalField:
    def test_run_start_event_carries_goal_when_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the plan has a non-empty 'goal' field, the run_start event must
        include a 'goal' key whose value matches the plan goal.

        This cross-module contract ensures the CLAUDE.md-documented behaviour
        ('run_start event includes goal field') is enforced by the scheduler."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("goal-t1"), _make_task("goal-t2", depends_on=["goal-t1"])],
            name="goal-test-plan",
            source_path=tmp_path / "plan.yaml",
        )
        plan.goal = "Reach 95% test coverage on module X"

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        run_start_events = [e for e in events if e.get("event") == "run_start"]
        assert run_start_events, "run_start event must be present in events.jsonl"

        run_start = run_start_events[0]
        assert "goal" in run_start, (
            "run_start event must include a 'goal' field when plan.goal is set"
        )
        assert run_start["goal"] == "Reach 95% test coverage on module X", (
            f"run_start 'goal' field={run_start['goal']!r} does not match plan.goal"
        )

    def test_run_start_event_goal_empty_when_not_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When plan.goal is not set, the run_start event 'goal' field must be
        an empty string (the PlanSpec default), never absent."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("ng-t1")],
            name="no-goal-plan",
            source_path=tmp_path / "plan.yaml",
        )
        # goal defaults to "" — do not set it

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        run_start = next(e for e in events if e.get("event") == "run_start")

        # 'goal' key must exist; its value must be the empty string
        assert "goal" in run_start, (
            "run_start event must always carry a 'goal' field (even when empty)"
        )
        assert run_start["goal"] == "", (
            f"run_start 'goal' should be '' when plan has no goal, "
            f"got {run_start['goal']!r}"
        )


# ---------------------------------------------------------------------------
# Test 18: Token aggregation — total_tokens matches sum of per-task tokens
# ---------------------------------------------------------------------------


class TestTokenAggregation:
    def test_total_tokens_equals_sum_of_task_token_counts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PlanRunResult.total_tokens equals the sum of all per-task token counts,
        and the manifest carries the same value under 'total_tokens'."""

        per_task_input = 100
        per_task_output = 50
        # TokenUsage.total_tokens = input_tokens + cached_tokens + output_tokens
        # cached_tokens defaults to 0, so each task's total = 150
        expected_per_task_tokens = per_task_input + per_task_output

        def mock_token_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
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
                cost_usd=0.10,
                token_usage=TokenUsage(
                    input_tokens=per_task_input,
                    output_tokens=per_task_output,
                ),
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_token_execute)

        plan = _make_plan(
            [_make_task("tok-a"), _make_task("tok-b"), _make_task("tok-c")],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert result.total_tokens is not None, "total_tokens must be set after a run with token data"

        expected_total = expected_per_task_tokens * 3
        assert result.total_tokens == expected_total, (
            f"Expected total_tokens={expected_total}, got {result.total_tokens}"
        )

        # manifest must carry the same value
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "total_tokens" in manifest, "total_tokens must be present in run_manifest.json"
        assert manifest["total_tokens"] == expected_total, (
            f"Manifest total_tokens={manifest['total_tokens']} != expected {expected_total}"
        )


# ---------------------------------------------------------------------------
# Test 19: Dependency-skipped tasks — dependents of failed tasks get status=skipped
# ---------------------------------------------------------------------------


class TestDependencySkippedTasks:
    def test_dependent_of_failed_task_is_skipped_in_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When task A fails (no allow_failure), any task that depends on A
        must appear in the manifest with status='skipped' and must NOT have
        been passed to execute_task."""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            if task.id == "dep-fail":
                r = TaskResult(
                    task_id=task.id,
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="false",
                    log_path=run_path / f"{task.id}.log",
                    result_path=run_path / f"{task.id}.result.json",
                    message="upstream failure",
                )
            else:
                r = TaskResult(
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
            r.log_path.write_text(f"status={r.status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("dep-ok"),
                _make_task("dep-fail"),
                _make_task("dep-child", depends_on=["dep-fail"]),
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False, "Plan should fail when a task fails"

        statuses = {tid: tr.status for tid, tr in result.task_results.items()}
        assert statuses.get("dep-fail") == "failed", "dep-fail must be 'failed'"
        assert statuses.get("dep-child") == "skipped", (
            f"dep-child must be 'skipped' because its dependency failed, "
            f"got {statuses.get('dep-child')!r}"
        )

        # dep-child must NOT have been executed
        assert "dep-child" not in executed, (
            "dep-child must not be passed to execute_task when its dependency failed"
        )

        # manifest must also reflect skipped status
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "dep-child" in manifest["task_results"], (
            "dep-child must appear in run_manifest.json task_results even when skipped"
        )
        assert manifest["task_results"]["dep-child"]["status"] == "skipped", (
            f"dep-child manifest status={manifest['task_results']['dep-child']['status']!r}, "
            f"expected 'skipped'"
        )


# ---------------------------------------------------------------------------
# Test 20: task_skip events emitted for dependency-skipped tasks
# ---------------------------------------------------------------------------


class TestTaskSkipEvents:
    def test_task_skip_events_emitted_for_dependency_failed_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """task_skip events are emitted in events.jsonl for tasks that are
        skipped because a dependency failed, and each event carries
        'task_id' and 'reason' fields."""

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            if task.id == "skip-root-fail":
                r = TaskResult(
                    task_id=task.id,
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="false",
                    log_path=run_path / f"{task.id}.log",
                    result_path=run_path / f"{task.id}.result.json",
                    message="root failure",
                )
            else:
                r = TaskResult(
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
            r.log_path.write_text(f"status={r.status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("skip-root-fail"),
                _make_task("skip-child-1", depends_on=["skip-root-fail"]),
                _make_task("skip-child-2", depends_on=["skip-root-fail"]),
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        skip_events = [e for e in events if e.get("event") == "task_skip"]

        assert skip_events, "At least one task_skip event must be emitted when a dependency fails"

        skip_task_ids = {e.get("task_id") for e in skip_events}
        for child in ("skip-child-1", "skip-child-2"):
            assert child in skip_task_ids, (
                f"Expected task_skip event for '{child}', "
                f"found skip events for: {skip_task_ids}"
            )

        # Each skip event must have task_id and reason
        for evt in skip_events:
            assert "task_id" in evt, f"task_skip event missing 'task_id': {evt}"
            assert "reason" in evt, f"task_skip event missing 'reason': {evt}"
            assert evt["reason"], "task_skip 'reason' must be a non-empty string"


# ---------------------------------------------------------------------------
# Test 21: Dry run mode — all tasks get status 'dry_run' in manifest
# ---------------------------------------------------------------------------


class TestDryRunMode:
    def test_dry_run_marks_all_tasks_as_dry_run_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When run_plan() is called with dry_run=True, every task receives
        status='dry_run' in the manifest, and the plan run is reported as
        successful (dry_run counts as ok)."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("dr-a"),
                _make_task("dr-b", depends_on=["dr-a"]),
                _make_task("dr-c", depends_on=["dr-a"]),
            ],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, dry_run=True, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True, "Dry run should report success"

        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        for task_id in ("dr-a", "dr-b", "dr-c"):
            assert task_id in manifest["task_results"], (
                f"Task '{task_id}' must appear in manifest even in dry_run mode"
            )
            assert manifest["task_results"][task_id]["status"] == "dry_run", (
                f"Task '{task_id}' must have status 'dry_run' in dry_run mode, "
                f"got {manifest['task_results'][task_id]['status']!r}"
            )


# ---------------------------------------------------------------------------
# Test 22: run_complete event — success field matches PlanRunResult.success
# ---------------------------------------------------------------------------


class TestRunCompleteEvent:
    def test_run_complete_success_field_matches_plan_result_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 'success' field in the run_complete event must match
        PlanRunResult.success for a successful plan run."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("rc-ok-1"), _make_task("rc-ok-2", depends_on=["rc-ok-1"])],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        complete_events = [e for e in events if e.get("event") == "run_complete"]
        assert complete_events, "run_complete event must be present"

        run_complete = complete_events[0]
        assert "success" in run_complete, "run_complete event must have a 'success' field"
        assert run_complete["success"] == result.success, (
            f"run_complete 'success'={run_complete['success']!r} "
            f"does not match PlanRunResult.success={result.success!r}"
        )

    def test_run_complete_success_false_on_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The run_complete event 'success' field is False when a task fails."""

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
                task_id=task.id,
                status="failed",
                exit_code=1,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command="false",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="failure",
            )
            r.log_path.write_text("status=failed\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [_make_task("rc-fail-1")],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        run_complete = next(e for e in events if e.get("event") == "run_complete")

        assert run_complete["success"] is False, (
            "run_complete 'success' must be False when a task fails"
        )
        assert result.success is False
        assert run_complete["success"] == result.success


# ---------------------------------------------------------------------------
# Test 23: Resume flow — prior succeeded tasks are not re-executed
# ---------------------------------------------------------------------------


class TestResumeFlow:
    def test_resume_skips_prior_succeeded_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run 1: task 'step-a' succeeds, 'step-b' fails.
        Run 2: resume from run 1 — 'step-a' must be skipped (not passed to
        execute_task), and 'step-b' must be re-executed.

        This validates the _load_prior_results + resume pre-population path in
        scheduler.run_plan() and ensures the resume contract documented in
        CLAUDE.md ('succeeded tasks are skipped') is enforced end-to-end."""
        executed_run1: list[str] = []
        executed_run2: list[str] = []
        lock = threading.Lock()

        # --- Run 1: step-a succeeds, step-b fails ---
        def mock_run1(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed_run1.append(task.id)
            now = datetime.now(UTC)
            status = "failed" if task.id == "step-b" else "success"
            exit_code = 1 if task.id == "step-b" else 0
            r = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=exit_code,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok" if status == "success" else "failed intentionally",
            )
            r.log_path.write_text(f"status={status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_run1)

        plan = _make_plan(
            [_make_task("step-a"), _make_task("step-b")],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        run1 = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert "step-a" in executed_run1, "step-a must execute in run 1"
        assert "step-b" in executed_run1, "step-b must execute in run 1"
        assert run1.task_results["step-a"].status == "success"
        assert run1.task_results["step-b"].status == "failed"

        # --- Run 2: resume from run 1 ---
        def mock_run2(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed_run2.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_run2)

        run2 = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs2"),
            resume_path=run1.run_path,
        )

        # step-a succeeded in run 1 — must NOT be re-executed in run 2
        assert "step-a" not in executed_run2, (
            "step-a already succeeded in run 1 and must not be re-executed on resume"
        )
        # step-b failed in run 1 — must be re-executed in run 2
        assert "step-b" in executed_run2, (
            "step-b failed in run 1 and must be re-executed on resume"
        )

        # Resumed run must succeed (both tasks end up succeeded)
        assert run2.success is True, "Resumed run should succeed when step-b succeeds"

        # Manifest from run 2 must contain both task IDs
        manifest2 = json.loads(
            (run2.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "step-a" in manifest2["task_results"], "step-a must appear in run 2 manifest"
        assert "step-b" in manifest2["task_results"], "step-b must appear in run 2 manifest"


# ---------------------------------------------------------------------------
# Test 24: Budget warning event — budget_warning emitted before budget_exceeded
# ---------------------------------------------------------------------------


class TestBudgetWarningEvent:
    def test_budget_warning_emitted_before_budget_exceeded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When running costs approach max_cost_usd, a 'budget_warning' event
        must appear in events.jsonl, and it must precede any 'budget_exceeded'
        event (or run_complete if budget is exhausted).

        Uses budget_warning_pct=0.5 with 3 tasks each costing $2 and
        max_cost_usd=$3.00: after the first $2 task, running_cost ($2) >=
        $3 * 0.5 ($1.50) so a warning fires; the second task pushes over
        the limit triggering budget_exceeded."""

        def mock_costly_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
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
                cost_usd=2.00,
                token_usage=TokenUsage(input_tokens=100, output_tokens=100),
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_costly_execute)

        plan = _make_plan(
            [_make_task("bw-1"), _make_task("bw-2"), _make_task("bw-3")],
            fail_fast=False,
            max_parallel=1,
            max_cost_usd=3.00,
            source_path=tmp_path / "plan.yaml",
        )
        plan.budget_warning_pct = 0.5

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        event_names = [e["event"] for e in events]

        assert "budget_warning" in event_names, (
            "budget_warning event must be emitted when running cost exceeds "
            "max_cost_usd * budget_warning_pct"
        )

        warning_idx = event_names.index("budget_warning")

        # budget_warning must precede run_complete
        assert warning_idx < event_names.index("run_complete"), (
            "budget_warning must appear before run_complete"
        )

        # budget_warning must precede budget_exceeded (if it exists)
        if "budget_exceeded" in event_names:
            exceeded_idx = event_names.index("budget_exceeded")
            assert warning_idx < exceeded_idx, (
                "budget_warning must appear before budget_exceeded"
            )


# ---------------------------------------------------------------------------
# Test 25: run_complete skipped counter — accurate when tasks are dep-skipped
# ---------------------------------------------------------------------------


class TestRunCompleteSkippedCounter:
    def test_run_complete_skipped_count_is_accurate(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When task 'blocker' fails and two dependents are skipped, the
        run_complete event's 'skipped' field must equal 2 (the number of
        tasks that were dependency-skipped).

        This cross-module contract ensures scheduler.py correctly aggregates
        skipped counts into the run_complete event payload."""

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            status = "failed" if task.id == "blocker" else "success"
            exit_code = 1 if task.id == "blocker" else 0
            r = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=exit_code,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="failure" if status == "failed" else "ok",
            )
            r.log_path.write_text(f"status={status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("independent"),
                _make_task("blocker"),
                _make_task("dep-1", depends_on=["blocker"]),
                _make_task("dep-2", depends_on=["blocker"]),
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        run_complete_events = [e for e in events if e.get("event") == "run_complete"]
        assert run_complete_events, "run_complete event must be present"

        run_complete = run_complete_events[0]
        assert "skipped" in run_complete, "run_complete event must have a 'skipped' field"

        skipped_count = run_complete["skipped"]
        assert skipped_count == 2, (
            f"Expected skipped=2 in run_complete (dep-1 and dep-2 are dep-skipped), "
            f"got skipped={skipped_count}"
        )

        # Manifest task_results must also show dep-1 and dep-2 as skipped
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["task_results"]["dep-1"]["status"] == "skipped"
        assert manifest["task_results"]["dep-2"]["status"] == "skipped"


# ---------------------------------------------------------------------------
# Test 26: Serial chain ordering — max_parallel=1 enforces strict A→B→C order
# ---------------------------------------------------------------------------


class TestSerialChainOrdering:
    def test_serial_chain_executes_in_strict_dependency_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With max_parallel=1 and a three-task chain (chain-a → chain-b → chain-c),
        the tasks must start in strict topological order: chain-a before chain-b,
        chain-b before chain-c, as evidenced by task_start events in events.jsonl.

        This validates that the DAG scheduler respects dependency ordering even
        with a fully serial execution profile."""
        holder: list[Path] = []
        mock_fn, call_log = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("chain-a"),
                _make_task("chain-b", depends_on=["chain-a"]),
                _make_task("chain-c", depends_on=["chain-b"]),
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True, "Serial chain should complete successfully"
        assert set(call_log) == {"chain-a", "chain-b", "chain-c"}, (
            "All 3 tasks must be executed"
        )

        # Execution order captured by _make_mock_execute call_log must be A→B→C
        assert call_log.index("chain-a") < call_log.index("chain-b"), (
            "chain-a must be called before chain-b"
        )
        assert call_log.index("chain-b") < call_log.index("chain-c"), (
            "chain-b must be called before chain-c"
        )

        # Confirm via events: task_start ordering must also be A→B→C
        events = _parse_jsonl(result.run_path / "events.jsonl")
        starts = [
            e["task_id"]
            for e in events
            if e.get("event") == "task_start"
        ]
        assert starts.index("chain-a") < starts.index("chain-b"), (
            "task_start for chain-a must precede task_start for chain-b in events"
        )
        assert starts.index("chain-b") < starts.index("chain-c"), (
            "task_start for chain-b must precede task_start for chain-c in events"
        )


# ---------------------------------------------------------------------------
# Test 27: Manifest task entry count — matches number of plan tasks
# ---------------------------------------------------------------------------


class TestManifestTaskCount:
    def test_manifest_task_results_count_matches_plan_task_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The number of entries in run_manifest.json 'task_results' must equal
        the number of tasks in the plan, including tasks that were dependency-
        skipped due to a failure.

        This ensures the scheduler writes a manifest entry for every task it
        was asked to run, regardless of outcome."""

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            status = "failed" if task.id == "cnt-fail" else "success"
            exit_code = 1 if task.id == "cnt-fail" else 0
            r = TaskResult(
                task_id=task.id,
                status=status,
                exit_code=exit_code,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok" if status == "success" else "failure",
            )
            r.log_path.write_text(f"status={status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        tasks = [
            _make_task("cnt-ok-1"),
            _make_task("cnt-ok-2"),
            _make_task("cnt-fail"),
            _make_task("cnt-dep-1", depends_on=["cnt-fail"]),
            _make_task("cnt-dep-2", depends_on=["cnt-fail"]),
        ]
        plan = _make_plan(
            tasks,
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )

        expected_count = len(tasks)
        actual_count = len(manifest["task_results"])
        assert actual_count == expected_count, (
            f"Expected {expected_count} entries in manifest task_results "
            f"(one per plan task), got {actual_count}. "
            f"Keys present: {sorted(manifest['task_results'].keys())}"
        )

        # All task IDs must be present
        for task in tasks:
            assert task.id in manifest["task_results"], (
                f"Task '{task.id}' is missing from manifest task_results"
            )


# ---------------------------------------------------------------------------
# Test 28: run_summary.md cost line — "$X.XX" appears when tasks have costs
# ---------------------------------------------------------------------------


class TestSummaryCostLine:
    def test_summary_cost_line_shows_formatted_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When tasks report cost_usd, run_summary.md must include a Cost row
        with a '$X.XX' formatted value matching the total cost of all tasks.

        This validates the _write_summary() cost_str rendering path:
        '| Cost | $X.XX |' in the header table."""

        per_task_cost = 1.50

        def mock_costly_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
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
                cost_usd=per_task_cost,
                token_usage=TokenUsage(input_tokens=200, output_tokens=100),
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_costly_execute)

        plan = _make_plan(
            [_make_task("cost-line-a"), _make_task("cost-line-b")],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        summary_path = result.run_path / "run_summary.md"
        assert summary_path.exists(), "run_summary.md must exist"
        summary = summary_path.read_text(encoding="utf-8")

        expected_cost = per_task_cost * 2
        expected_cost_str = f"${expected_cost:.2f}"

        assert expected_cost_str in summary, (
            f"Expected cost string '{expected_cost_str}' in run_summary.md, "
            f"but it was not found. Summary excerpt:\n{summary[:500]}"
        )
        # The Cost row must be a markdown table row
        cost_row_found = any(
            "| Cost |" in line and expected_cost_str in line
            for line in summary.splitlines()
        )
        assert cost_row_found, (
            f"Expected '| Cost | {expected_cost_str}' row in summary table"
        )

    def test_summary_cost_line_shows_dashes_when_no_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no tasks report cost_usd, the Cost row must show '---' (not '$0.00').

        This validates the _write_summary() fallback: cost_str = '---' when
        total_cost_usd is None."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("nocost-a"), _make_task("nocost-b")],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")

        cost_row_found = any(
            "| Cost |" in line and "---" in line
            for line in summary.splitlines()
        )
        assert cost_row_found, (
            "Expected '| Cost | --- |' row when no tasks report cost_usd"
        )


# ---------------------------------------------------------------------------
# Test 29: run_id uniqueness — two sequential runs produce distinct run_ids
# ---------------------------------------------------------------------------


class TestRunIdUniqueness:
    def test_two_sequential_runs_produce_different_run_ids(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each call to run_plan() must produce a unique run_id.

        Deterministic uniqueness is required because run_id is the primary
        identifier for audit, diff, and report tooling.  A collision would
        cause tooling to silently overwrite previous run artifacts."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("uid-t1")],
            source_path=tmp_path / "plan.yaml",
        )

        result_a = run_plan(plan, run_dir_override=str(tmp_path / "runs_a"))
        result_b = run_plan(plan, run_dir_override=str(tmp_path / "runs_b"))

        assert result_a.run_id != result_b.run_id, (
            f"Expected unique run_ids for successive runs, "
            f"but both returned '{result_a.run_id}'"
        )

        # Both manifests must also carry distinct run_ids
        manifest_a = json.loads(
            (result_a.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        manifest_b = json.loads(
            (result_b.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest_a["run_id"] != manifest_b["run_id"], (
            "run_manifest.json run_id must be unique across runs"
        )


# ---------------------------------------------------------------------------
# Test 30: task_complete cost_usd cross-module — event matches manifest
# ---------------------------------------------------------------------------


class TestTaskCompleteCostMatchesManifest:
    def test_task_complete_cost_usd_matches_manifest_per_task_cost(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For each executed task, the 'cost_usd' field in the task_complete
        event must equal the 'cost_usd' value recorded in run_manifest.json
        for that task.

        This cross-module contract ensures that the cost data written to
        events.jsonl (by scheduler.py via _emit()) and the cost data
        serialised into run_manifest.json (via PlanRunResult.to_dict()) are
        sourced from the same TaskResult.cost_usd value and never diverge."""

        per_task_cost = 0.42

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
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
                cost_usd=per_task_cost,
                token_usage=TokenUsage(input_tokens=50, output_tokens=30),
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("xm-a"),
                _make_task("xm-b", depends_on=["xm-a"]),
                _make_task("xm-c", depends_on=["xm-a"]),
            ],
            fail_fast=False,
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )

        complete_events = {
            e["task_id"]: e
            for e in events
            if e.get("event") == "task_complete" and "task_id" in e and "cost_usd" in e
        }

        assert complete_events, (
            "At least one task_complete event with 'cost_usd' must be present"
        )

        for task_id, evt in complete_events.items():
            assert task_id in manifest["task_results"], (
                f"task '{task_id}' from task_complete event not in manifest"
            )
            manifest_cost = manifest["task_results"][task_id].get("cost_usd")
            event_cost = evt["cost_usd"]

            # Both should be equal to per_task_cost within floating-point tolerance
            assert manifest_cost is not None, (
                f"manifest cost_usd for '{task_id}' must not be None when task reports cost"
            )
            assert abs(event_cost - manifest_cost) < 1e-9, (
                f"task_complete event cost_usd ({event_cost}) does not match "
                f"manifest cost_usd ({manifest_cost}) for task '{task_id}'"
            )


# ---------------------------------------------------------------------------
# Test 31: Parallel execution completeness — all independent tasks execute
# ---------------------------------------------------------------------------


class TestParallelExecutionCompleteness:
    def test_all_independent_tasks_execute_with_max_parallel_4(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With max_parallel=4 and 5 fully independent tasks, every task must
        execute exactly once and the final result must be successful.

        This validates that the ThreadPoolExecutor-based DAG scheduler does not
        deadlock, starve, or silently drop tasks when all tasks are runnable
        from wave 0 and the pool has sufficient capacity."""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        task_ids = ["par-1", "par-2", "par-3", "par-4", "par-5"]
        plan = _make_plan(
            [_make_task(tid) for tid in task_ids],
            fail_fast=False,
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True, "All-independent-tasks run must succeed"

        # Every task must have been executed exactly once
        assert sorted(executed) == sorted(task_ids), (
            f"Expected all tasks {task_ids} to execute exactly once, "
            f"got: {sorted(executed)}"
        )

        # All tasks must appear as 'success' in the manifest
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        for tid in task_ids:
            assert tid in manifest["task_results"], (
                f"Task '{tid}' must appear in manifest"
            )
            assert manifest["task_results"][tid]["status"] == "success", (
                f"Task '{tid}' must have status 'success', "
                f"got {manifest['task_results'][tid]['status']!r}"
            )

        # events.jsonl must contain a task_complete event for each task
        events = _parse_jsonl(result.run_path / "events.jsonl")
        completed_in_events = {
            e["task_id"]
            for e in events
            if e.get("event") == "task_complete"
        }
        assert completed_in_events == set(task_ids), (
            f"events.jsonl task_complete events {completed_in_events} "
            f"do not match expected task IDs {set(task_ids)}"
        )


# ---------------------------------------------------------------------------
# Test 32: --only filter — only specified task (+ its transitive deps) runs
# ---------------------------------------------------------------------------


class TestOnlyFilter:
    def test_only_runs_target_and_deps_not_others(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_plan() with only={'leaf'} must execute only 'leaf' and its
        transitive dependencies.  Sibling tasks that are not in that subgraph
        must be absent from PlanRunResult.task_results (they were never selected).

        Plan layout:
          root → middle → leaf
                         (sibling — independent of leaf)

        With only={'leaf'}, the scheduler must run root, middle, leaf;
        'sibling' must NOT appear in results at all."""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("only-root"),
                _make_task("only-middle", depends_on=["only-root"]),
                _make_task("only-leaf", depends_on=["only-middle"]),
                _make_task("only-sibling"),  # independent — not in leaf's dep chain
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            only={"only-leaf"},
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True, "only-filtered run must succeed"

        # only-sibling was not in the subgraph — must NOT appear in results
        assert "only-sibling" not in result.task_results, (
            "only-sibling is outside the only={'only-leaf'} subgraph "
            "and must not appear in task_results"
        )
        # only-sibling must never have been executed
        assert "only-sibling" not in executed, (
            "only-sibling must not be passed to execute_task when not in the only set"
        )

        # All three tasks in the dep chain must have run and succeeded
        for tid in ("only-root", "only-middle", "only-leaf"):
            assert tid in result.task_results, (
                f"'{tid}' is a transitive dependency of 'only-leaf' and must appear in results"
            )
            assert result.task_results[tid].status == "success", (
                f"'{tid}' must have status 'success'"
            )


# ---------------------------------------------------------------------------
# Test 33: --skip filter — excluded task is absent from results
# ---------------------------------------------------------------------------


class TestSkipFilter:
    def test_skip_excludes_task_from_execution_and_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_plan() with skip={'skip-me'} must exclude 'skip-me' entirely:
        it must not be passed to execute_task and must not appear in
        PlanRunResult.task_results.

        The sibling task 'skip-ok' (which has no dependency on 'skip-me')
        must still execute normally."""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("skip-me"),
                _make_task("skip-ok"),
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            skip={"skip-me"},
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True, "Skipping one task must not make the run fail"

        # skip-me must have been excluded from execution and from results
        assert "skip-me" not in executed, (
            "skip-me must not be passed to execute_task when it is in the skip set"
        )
        assert "skip-me" not in result.task_results, (
            "skip-me must not appear in task_results when it is in the skip set"
        )

        # skip-ok must have executed normally
        assert "skip-ok" in executed, "skip-ok must be executed (not in skip set)"
        assert result.task_results["skip-ok"].status == "success", (
            "skip-ok must have status 'success'"
        )


# ---------------------------------------------------------------------------
# Test 34: Tag filtering — only tasks with matching tag (+ deps) run
# ---------------------------------------------------------------------------


class TestTagFilter:
    def test_tags_filter_runs_only_matching_tasks_and_their_deps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_plan() with tags={'deploy'} must run only tasks tagged 'deploy'
        plus any transitive dependencies those tasks require.

        Tasks with no matching tag that are not dependencies must be excluded
        from execution and from PlanRunResult.task_results.

        Plan layout:
          tagged-dep (no tags) → tagged-leaf (tags=['deploy'])
          untagged (tags=['qa'])  — independent, no deploy tag

        With tags={'deploy'}, only tagged-dep and tagged-leaf must run."""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        from maestro_cli.models import TaskSpec as _TaskSpec

        def _tagged_task(task_id: str, tags: list[str], depends_on: list[str] | None = None) -> _TaskSpec:
            return _TaskSpec(
                id=task_id,
                description=f"task {task_id}",
                tags=tags,
                depends_on=depends_on or [],
                command="echo ok",
            )

        plan = _make_plan(
            [
                _tagged_task("tagged-dep", tags=[]),  # dependency of deploy task, no tag
                _tagged_task("tagged-leaf", tags=["deploy"], depends_on=["tagged-dep"]),
                _tagged_task("untagged", tags=["qa"]),  # irrelevant tag
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            tags={"deploy"},
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True, "Tag-filtered run must succeed"

        # untagged (qa tag only) must be excluded
        assert "untagged" not in executed, (
            "'untagged' has tag 'qa' not 'deploy' and must not be executed"
        )
        assert "untagged" not in result.task_results, (
            "'untagged' must not appear in task_results when filtered by tags={'deploy'}"
        )

        # tagged-dep is a transitive dependency of tagged-leaf — must have run
        assert "tagged-dep" in executed, (
            "'tagged-dep' is a transitive dependency of the 'deploy'-tagged task and must run"
        )
        assert "tagged-leaf" in executed, (
            "'tagged-leaf' has tag 'deploy' and must be executed"
        )
        assert result.task_results["tagged-leaf"].status == "success"


# ---------------------------------------------------------------------------
# Test 35: Summary Tokens row — correct count when tasks report token usage
# ---------------------------------------------------------------------------


class TestSummaryTokensRow:
    def test_summary_tokens_row_shows_correct_total(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When tasks report token_usage, run_summary.md must include a
        '| Tokens |' row with the aggregated total formatted with thousands
        separators (e.g. '| Tokens | 450 |' for 3 tasks × 150 tokens each).

        This validates the _write_summary() tokens_str rendering path."""

        per_task_input = 100
        per_task_output = 50
        expected_per_task = per_task_input + per_task_output  # 150

        def mock_token_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
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
                cost_usd=0.01,
                token_usage=TokenUsage(
                    input_tokens=per_task_input,
                    output_tokens=per_task_output,
                ),
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_token_execute)

        task_count = 3
        plan = _make_plan(
            [_make_task(f"tok-row-{i}") for i in range(task_count)],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        assert result.total_tokens == expected_per_task * task_count, (
            f"Expected total_tokens={expected_per_task * task_count}, "
            f"got {result.total_tokens}"
        )

        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")

        # A Tokens row must be present in the summary header table
        tokens_row_found = any(
            "| Tokens |" in line for line in summary.splitlines()
        )
        assert tokens_row_found, (
            "Expected '| Tokens |' row in run_summary.md when tasks report token usage"
        )

        # The total token count must appear in the summary
        expected_tokens_str = f"{expected_per_task * task_count:,}"
        assert expected_tokens_str in summary, (
            f"Expected formatted token count '{expected_tokens_str}' in run_summary.md, "
            f"but it was not found.\nSummary:\n{summary[:600]}"
        )

    def test_summary_tokens_row_shows_dashes_when_no_tokens(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no tasks report token_usage, the Tokens row must show '---'
        (matching the '---' fallback in _write_summary() for None total_tokens)."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("notok-a"), _make_task("notok-b")],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")

        tokens_row_dashes = any(
            "| Tokens |" in line and "---" in line
            for line in summary.splitlines()
        )
        assert tokens_row_dashes, (
            "Expected '| Tokens | --- |' row when no tasks report token usage"
        )


# ---------------------------------------------------------------------------
# Test 36: Summary Timeline section — always present with wave entries
# ---------------------------------------------------------------------------


class TestSummaryTimelineSection:
    def test_summary_has_timeline_section_with_wave_entries(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_summary.md must always contain a '## Timeline' section with at
        least one '**Wave N**' entry listing executed tasks.

        This validates _write_summary() → _compute_waves() integration and
        ensures the timeline section is written for every successful run."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("tl-a"),
                _make_task("tl-b", depends_on=["tl-a"]),
                _make_task("tl-c", depends_on=["tl-a"]),
                _make_task("tl-d", depends_on=["tl-b", "tl-c"]),
            ],
            fail_fast=False,
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True

        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")

        # Timeline section header must be present
        assert "## Timeline" in summary, (
            "run_summary.md must contain '## Timeline' section"
        )

        # At least one Wave entry must exist
        wave_lines = [line for line in summary.splitlines() if "**Wave" in line]
        assert wave_lines, (
            "run_summary.md Timeline section must contain at least one '**Wave N**' entry"
        )

        # All four task IDs must appear somewhere in the summary (Tasks table + Timeline)
        for tid in ("tl-a", "tl-b", "tl-c", "tl-d"):
            assert tid in summary, (
                f"Task '{tid}' must appear in run_summary.md"
            )

    def test_summary_has_tasks_section_header(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_summary.md must contain the '## Tasks' section header and a
        markdown table with the correct column headers.

        This validates the task table rendering in _write_summary()."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("tasks-hdr-a"), _make_task("tasks-hdr-b", depends_on=["tasks-hdr-a"])],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")

        assert "## Tasks" in summary, "run_summary.md must contain '## Tasks' section"

        # The task table header row must be present
        assert "| Task | Status | Duration | Cost | Tokens | Engine |" in summary, (
            "run_summary.md must contain the task table column headers"
        )


# ---------------------------------------------------------------------------
# Test 37: run_complete soft_failed counter — accurate when tasks are soft_failed
# ---------------------------------------------------------------------------


class TestRunCompleteSoftFailedCounter:
    def test_run_complete_soft_failed_count_and_plan_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When one task is soft_failed and two succeed, the run_complete event
        must have soft_failed=1, ok=2, and the plan must be reported as
        successful (soft_failed does not count as a plan-level failure).

        This validates the soft_failed counter path in run_complete and the
        _SUCCESS_LIKE semantics documented in CLAUDE.md."""

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            if task.id == "sf-soft":
                status: str = "soft_failed"
                exit_code = 1
                msg = "Intentional soft failure"
            else:
                status = "success"
                exit_code = 0
                msg = "ok"
            r = TaskResult(
                task_id=task.id,
                status=status,  # type: ignore[arg-type]
                exit_code=exit_code,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message=msg,
            )
            r.log_path.write_text(f"status={status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [_make_task("sf-ok-1"), _make_task("sf-soft"), _make_task("sf-ok-2")],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # Plan success must be True (soft_failed is in _SUCCESS_LIKE)
        assert result.success is True, (
            "Plan with only soft_failed tasks must still report success=True"
        )

        events = _parse_jsonl(result.run_path / "events.jsonl")
        run_complete = next(e for e in events if e.get("event") == "run_complete")

        assert "soft_failed" in run_complete, (
            "run_complete event must carry a 'soft_failed' field"
        )
        assert run_complete["soft_failed"] == 1, (
            f"Expected soft_failed=1 in run_complete, got {run_complete['soft_failed']}"
        )
        assert run_complete["ok"] == 2, (
            f"Expected ok=2 in run_complete (sf-ok-1 and sf-ok-2), got {run_complete['ok']}"
        )
        assert run_complete["failed"] == 0, (
            "Expected failed=0 in run_complete (no hard failures)"
        )
        assert run_complete["success"] is True, (
            "run_complete event success must be True when only soft_failed tasks are present"
        )


# ---------------------------------------------------------------------------
# Test 38: Resume message — resumed task manifest entry has "Resumed from prior run"
# ---------------------------------------------------------------------------


class TestResumeManifestMessage:
    def test_resumed_task_manifest_message_starts_with_resumed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a task that succeeded in run 1 is resumed in run 2, its entry
        in the run 2 manifest must have a message starting with
        'Resumed from prior run', and its status must equal the prior status.

        This validates the pre-population path in run_plan() that writes
        synthetic TaskResult entries for resumed tasks."""

        def mock_run1(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_run1)

        plan = _make_plan(
            [_make_task("resume-msg-a"), _make_task("resume-msg-b")],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        run1 = run_plan(plan, run_dir_override=str(tmp_path / "runs1"))
        assert run1.success is True
        assert run1.task_results["resume-msg-a"].status == "success"

        # Run 2: resume — resume-msg-a must be pre-populated, resume-msg-b re-executed
        executed_run2: list[str] = []

        def mock_run2(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            executed_run2.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_run2)

        run2 = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs2"),
            resume_path=run1.run_path,
        )

        # resume-msg-a was pre-populated — must NOT have been executed
        assert "resume-msg-a" not in executed_run2, (
            "resume-msg-a already succeeded in run 1 and must not be re-executed"
        )

        manifest2 = json.loads(
            (run2.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )

        # resume-msg-a must be in run2 manifest
        assert "resume-msg-a" in manifest2["task_results"], (
            "Resumed task 'resume-msg-a' must appear in run 2 manifest"
        )
        resumed_entry = manifest2["task_results"]["resume-msg-a"]

        # Message must start with "Resumed from prior run"
        assert resumed_entry["message"].startswith("Resumed from prior run"), (
            f"Resumed task message={resumed_entry['message']!r} "
            "must start with 'Resumed from prior run'"
        )
        # Status must be "success" (carried from prior run)
        assert resumed_entry["status"] == "success", (
            f"Resumed task status={resumed_entry['status']!r}, expected 'success'"
        )


# ---------------------------------------------------------------------------
# Test 39: budget_exceeded=False when plan stays within budget
# ---------------------------------------------------------------------------


class TestBudgetNotExceeded:
    def test_budget_not_exceeded_when_within_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When all tasks complete within max_cost_usd, PlanRunResult.budget_exceeded
        must be False and run_manifest.json must also carry budget_exceeded=False.

        This is the inverse of TestBudgetExhaustion and validates the default
        (no-budget-violation) path through the scheduler's budget check."""

        per_task_cost = 0.10

        def mock_cheap_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
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
                cost_usd=per_task_cost,
                token_usage=TokenUsage(input_tokens=10, output_tokens=10),
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_cheap_execute)

        # 3 tasks × $0.10 = $0.30 total; budget is $5.00 — should never be exceeded
        plan = _make_plan(
            [_make_task("budget-safe-1"), _make_task("budget-safe-2"), _make_task("budget-safe-3")],
            fail_fast=False,
            max_parallel=1,
            max_cost_usd=5.00,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True, "Plan should succeed when within budget"
        assert result.budget_exceeded is False, (
            "budget_exceeded must be False when all tasks complete within max_cost_usd"
        )

        # All tasks must have run (budget did not cut them short)
        for tid in ("budget-safe-1", "budget-safe-2", "budget-safe-3"):
            assert tid in result.task_results, f"Task '{tid}' must be in results"
            assert result.task_results[tid].status == "success", (
                f"Task '{tid}' must be 'success'"
            )

        # Manifest must also have budget_exceeded=False
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "budget_exceeded" in manifest, (
            "budget_exceeded must be present in run_manifest.json"
        )
        assert manifest["budget_exceeded"] is False, (
            "run_manifest.json budget_exceeded must be False when within budget"
        )

        # Summary must show Budget row as "OK" (not "EXCEEDED")
        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")
        budget_ok_found = any(
            "| Budget |" in line and "OK" in line
            for line in summary.splitlines()
        )
        assert budget_ok_found, (
            "run_summary.md must show '| Budget | ... (OK) |' when budget is not exceeded"
        )


# ---------------------------------------------------------------------------
# Test 40: budget_exceeded event — emitted with spent and limit fields
# ---------------------------------------------------------------------------


class TestBudgetExceededEvent:
    def test_budget_exceeded_event_has_spent_and_limit_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When max_cost_usd is exceeded, a 'budget_exceeded' event must appear
        in events.jsonl carrying 'spent' (actual running cost) and 'limit'
        (the plan's max_cost_usd) fields.

        This validates the _emit('budget_exceeded', ...) call in scheduler.py
        and ensures the budget exhaustion contract is observable via events."""

        def mock_costly_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
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
                cost_usd=3.00,
                token_usage=TokenUsage(input_tokens=300, output_tokens=300),
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_costly_execute)

        # First task costs $3.00, limit is $2.00 → budget exceeded after first task
        plan = _make_plan(
            [_make_task("bex-1"), _make_task("bex-2"), _make_task("bex-3")],
            fail_fast=False,
            max_parallel=1,
            max_cost_usd=2.00,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.budget_exceeded is True, "budget_exceeded must be True"

        events = _parse_jsonl(result.run_path / "events.jsonl")
        budget_events = [e for e in events if e.get("event") == "budget_exceeded"]

        assert budget_events, (
            "At least one 'budget_exceeded' event must be emitted in events.jsonl "
            "when max_cost_usd is exceeded"
        )

        bev = budget_events[0]
        assert "spent" in bev, "budget_exceeded event must have a 'spent' field"
        assert "limit" in bev, "budget_exceeded event must have a 'limit' field"

        # 'spent' must be a positive number exceeding the limit
        assert isinstance(bev["spent"], (int, float)), (
            f"'spent' must be numeric, got {type(bev['spent'])}"
        )
        assert bev["spent"] > bev["limit"], (
            f"'spent' ({bev['spent']}) must be greater than 'limit' ({bev['limit']}) "
            "when budget is exceeded"
        )
        assert abs(bev["limit"] - 2.00) < 0.001, (
            f"'limit' must be the plan's max_cost_usd=2.00, got {bev['limit']}"
        )

        # The budget_exceeded event must appear before run_complete
        event_names = [e["event"] for e in events]
        budget_idx = event_names.index("budget_exceeded")
        complete_idx = event_names.index("run_complete")
        assert budget_idx < complete_idx, (
            "budget_exceeded event must appear before run_complete"
        )


# ---------------------------------------------------------------------------
# Test 41: Summary task table row shows soft_failed status for soft_failed tasks
# ---------------------------------------------------------------------------


class TestSummarySoftFailedTaskRow:
    def test_summary_task_table_shows_soft_failed_status(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a task is soft_failed, the run_summary.md task table row for
        that task must show 'soft_failed' as the status, and the Tasks header
        row ('| Tasks | ... |') must include 'soft_failed' in the count summary.

        This validates that _write_summary() correctly renders soft_failed
        tasks in both the task table and the header status line."""

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            if task.id == "soft-row-task":
                status: str = "soft_failed"
                exit_code = 1
                msg = "Soft failure for summary test"
            else:
                status = "success"
                exit_code = 0
                msg = "ok"
            r = TaskResult(
                task_id=task.id,
                status=status,  # type: ignore[arg-type]
                exit_code=exit_code,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message=msg,
            )
            r.log_path.write_text(f"status={status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [_make_task("soft-row-ok"), _make_task("soft-row-task")],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True, "Plan with soft_failed must still report success=True"

        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")

        # The task table row for soft-row-task must show 'soft_failed' as status
        soft_row_found = any(
            "soft-row-task" in line and "soft_failed" in line
            for line in summary.splitlines()
        )
        assert soft_row_found, (
            "run_summary.md task table must contain a row with 'soft-row-task' "
            "and 'soft_failed' status"
        )

        # The header '| Tasks | ... |' line must include 'soft_failed' count
        tasks_header_with_soft = any(
            "| Tasks |" in line and "soft_failed" in line
            for line in summary.splitlines()
        )
        assert tasks_header_with_soft, (
            "run_summary.md header '| Tasks |' row must include 'soft_failed' "
            "count when at least one task is soft_failed"
        )


# ---------------------------------------------------------------------------
# Test 39: Wave numbering — tasks in the second wave have a higher wave number
#           than tasks in the first wave, as recorded in task_start events
# ---------------------------------------------------------------------------


class TestWaveNumbering:
    def test_task_start_wave_increases_with_dependency_depth(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """task_start events carry a 'wave' field.  For a simple A → B chain,
        A must start in wave 0 and B must start in a higher (later) wave.

        This validates the wave-dispatch contract in scheduler.run_plan():
        the scheduler emits task_start with the wave index so that tooling
        (live renderer, TUI, audit) can distinguish independent from dependent
        tasks."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("wave-a"),                            # wave 0
                _make_task("wave-b", depends_on=["wave-a"]),    # wave 1
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True

        events = _parse_jsonl(result.run_path / "events.jsonl")
        starts = {
            e["task_id"]: e["wave"]
            for e in events
            if e.get("event") == "task_start" and "task_id" in e and "wave" in e
        }

        assert "wave-a" in starts, "task_start event for wave-a must carry a 'wave' field"
        assert "wave-b" in starts, "task_start event for wave-b must carry a 'wave' field"

        assert isinstance(starts["wave-a"], int), "'wave' field must be an integer"
        assert isinstance(starts["wave-b"], int), "'wave' field must be an integer"

        assert starts["wave-b"] > starts["wave-a"], (
            f"wave-b (wave={starts['wave-b']}) must be in a later wave "
            f"than wave-a (wave={starts['wave-a']}), because wave-b depends on wave-a"
        )


# ---------------------------------------------------------------------------
# Test 40: fail_fast=False — independent tasks all run despite one failure
# ---------------------------------------------------------------------------


class TestFailFastFalseMultipleFailures:
    def test_fail_fast_false_runs_independent_tasks_despite_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With fail_fast=False, a failing task must NOT stop independent tasks
        from executing.  Two independent tasks ('ind-ok-1' and 'ind-fail') run
        concurrently with 'ind-ok-2'; the failure of 'ind-fail' must not
        prevent 'ind-ok-2' from completing.

        This is the inverse of TestFailFast and validates the default plan
        behavior that individual task failures are isolated."""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            if task.id == "ind-fail":
                r = TaskResult(
                    task_id=task.id,
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="false",
                    log_path=run_path / f"{task.id}.log",
                    result_path=run_path / f"{task.id}.result.json",
                    message="Intentional failure",
                )
            else:
                r = TaskResult(
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
            r.log_path.write_text(f"status={r.status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("ind-ok-1"),
                _make_task("ind-fail"),
                _make_task("ind-ok-2"),
            ],
            fail_fast=False,
            max_parallel=1,  # serial so ordering is deterministic
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # All three tasks must have been passed to execute_task
        assert set(executed) == {"ind-ok-1", "ind-fail", "ind-ok-2"}, (
            f"With fail_fast=False all independent tasks must execute; "
            f"got executed={executed!r}"
        )

        # Overall run must be failed (one task failed)
        assert result.success is False, "Plan must report failure when a task fails"

        statuses = {tid: tr.status for tid, tr in result.task_results.items()}
        assert statuses["ind-ok-1"] == "success"
        assert statuses["ind-fail"] == "failed"
        assert statuses["ind-ok-2"] == "success"

        # run_complete must agree: success=False, ok=2, failed=1
        events = _parse_jsonl(result.run_path / "events.jsonl")
        run_complete = next(e for e in events if e.get("event") == "run_complete")
        assert run_complete["success"] is False
        assert run_complete["ok"] == 2, (
            f"run_complete 'ok' must be 2, got {run_complete['ok']}"
        )
        assert run_complete["failed"] == 1, (
            f"run_complete 'failed' must be 1, got {run_complete['failed']}"
        )


# ---------------------------------------------------------------------------
# Test 41: Multiple failures in summary — all failed task IDs appear in
#           the '## Failed Tasks' section of run_summary.md
# ---------------------------------------------------------------------------


class TestSummaryMultipleFailedTasks:
    def test_all_failed_task_ids_appear_in_failed_tasks_section(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When multiple tasks fail, every failed task's ID must appear in
        the '## Failed Tasks' section of run_summary.md.

        This extends TestSummaryFailedSection (which tests a single failure)
        to validate that _write_summary() iterates over all failed tasks and
        does not truncate the section after the first entry."""

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            if task.id in ("mf-fail-1", "mf-fail-2"):
                r = TaskResult(
                    task_id=task.id,
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="false",
                    log_path=run_path / f"{task.id}.log",
                    result_path=run_path / f"{task.id}.result.json",
                    message=f"Deliberate failure in {task.id}",
                )
            else:
                r = TaskResult(
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
            r.log_path.write_text(f"status={r.status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("mf-ok"),
                _make_task("mf-fail-1"),
                _make_task("mf-fail-2"),
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False

        summary_path = result.run_path / "run_summary.md"
        assert summary_path.exists(), "run_summary.md must exist after the run"
        summary = summary_path.read_text(encoding="utf-8")

        assert "## Failed Tasks" in summary, (
            "run_summary.md must have a '## Failed Tasks' section when tasks fail"
        )
        assert "mf-fail-1" in summary, (
            "Failed task 'mf-fail-1' must appear in run_summary.md"
        )
        assert "mf-fail-2" in summary, (
            "Failed task 'mf-fail-2' must appear in run_summary.md"
        )
        # Sanity: successful task should NOT appear in the failed section.
        # Check that 'mf-ok' is in the summary but not in the failed section
        # by checking that any line containing 'mf-ok' does not also contain
        # 'failed' in a way that would indicate it is listed as a failure.
        failed_section_start = summary.find("## Failed Tasks")
        failed_section = summary[failed_section_start:]
        assert "mf-ok" not in failed_section, (
            "Successful task 'mf-ok' must not appear in the Failed Tasks section"
        )


# ---------------------------------------------------------------------------
# Test 42: run_path directory — the artifacts directory is a real subdirectory
#           of run_dir_override and contains the canonical artifact files
# ---------------------------------------------------------------------------


class TestRunPathDirectoryStructure:
    def test_run_path_is_a_subdirectory_of_run_dir_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PlanRunResult.run_path must be a subdirectory of the run_dir_override
        argument.  The directory must contain the three canonical artifacts:
        run_manifest.json, run_summary.md, and events.jsonl.

        This validates the file-system contract that users and tooling
        (maestro diff, maestro report, maestro verify) rely on: the run
        directory is a named subdirectory under the specified root, not the
        root itself, and always contains these three files after a run."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        run_dir = tmp_path / "my-runs"

        plan = _make_plan(
            [_make_task("struct-a"), _make_task("struct-b", depends_on=["struct-a"])],
            name="structure-test-plan",
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(run_dir))

        # run_path must exist as a directory
        assert result.run_path.is_dir(), (
            f"PlanRunResult.run_path must be an existing directory, "
            f"got: {result.run_path}"
        )

        # run_path must be a child of run_dir (not run_dir itself)
        assert result.run_path.parent == run_dir, (
            f"run_path.parent must equal the run_dir_override path. "
            f"Expected parent: {run_dir}, got: {result.run_path.parent}"
        )

        # run_path name must include the plan name
        assert "structure-test-plan" in result.run_path.name, (
            f"run_path directory name must contain the plan name. "
            f"Got: {result.run_path.name!r}"
        )

        # Three canonical artifacts must be present
        for artifact in ("run_manifest.json", "run_summary.md", "events.jsonl"):
            artifact_path = result.run_path / artifact
            assert artifact_path.exists(), (
                f"Canonical artifact '{artifact}' must exist in run_path "
                f"({result.run_path})"
            )
            assert artifact_path.stat().st_size > 0, (
                f"Canonical artifact '{artifact}' must not be empty"
            )


# ---------------------------------------------------------------------------
# Test 43: task_start wave field is consistent across task_start and
#           task_complete events for the same task
# ---------------------------------------------------------------------------


class TestTaskEventWaveConsistency:
    def test_task_start_and_task_complete_carry_consistent_wave(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """For every executed task, the 'wave' field emitted in the task_start
        event must equal the 'wave' field (if present) emitted in the
        corresponding task_complete event.

        This validates that scheduler.py sources the wave number from the same
        place for both emissions, preventing observability inconsistencies where
        the event feed and manifest tooling would see different wave values for
        the same task."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("wc-root"),
                _make_task("wc-child-1", depends_on=["wc-root"]),
                _make_task("wc-child-2", depends_on=["wc-root"]),
                _make_task("wc-leaf", depends_on=["wc-child-1", "wc-child-2"]),
            ],
            fail_fast=False,
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True, "Diamond DAG must succeed"

        events = _parse_jsonl(result.run_path / "events.jsonl")

        start_waves: dict[str, int] = {
            e["task_id"]: e["wave"]
            for e in events
            if e.get("event") == "task_start" and "task_id" in e and "wave" in e
        }
        complete_waves: dict[str, int] = {
            e["task_id"]: e["wave"]
            for e in events
            if e.get("event") == "task_complete" and "task_id" in e and "wave" in e
        }

        # Every task must have a task_start wave recorded
        for tid in ("wc-root", "wc-child-1", "wc-child-2", "wc-leaf"):
            assert tid in start_waves, (
                f"task_start event for '{tid}' must carry a 'wave' field"
            )

        # Where task_complete also carries a wave, it must match task_start
        for tid, start_wave in start_waves.items():
            if tid in complete_waves:
                assert complete_waves[tid] == start_wave, (
                    f"task_complete 'wave' for '{tid}' ({complete_waves[tid]}) "
                    f"differs from task_start 'wave' ({start_wave})"
                )


# ---------------------------------------------------------------------------
# Test 44: task_complete event carries 'status' field matching manifest
# ---------------------------------------------------------------------------


class TestTaskCompleteEventCarriesStatus:
    def test_task_complete_status_matches_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every task_complete event must carry a 'status' field, and that
        status must match the corresponding entry in run_manifest.json."""
        from datetime import UTC, datetime

        holder: list[Path] = []
        fail_result: TaskResult | None = None

        def make_fail_result(task: Any, run_path: Path) -> TaskResult:
            now = datetime.now(UTC)
            r = TaskResult(
                task_id=task.id,
                status="failed",
                exit_code=1,
                started_at=now,
                finished_at=now,
                duration_sec=0.05,
                command="false",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="intentional failure",
            )
            r.log_path.write_text("failed\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        mock_fn, _ = _make_mock_execute(
            holder,
            overrides={"tc-fail": None},  # placeholder; handled below
        )

        def mixed_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            if not holder:
                holder.append(run_path)
            if task.id == "tc-fail":
                return make_fail_result(task, run_path)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("ok\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mixed_execute)

        plan = _make_plan(
            [
                _make_task("tc-ok-1"),
                _make_task("tc-fail", allow_failure=True),
                _make_task("tc-ok-2", depends_on=["tc-ok-1"]),
            ],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        manifest = json.loads((result.run_path / "run_manifest.json").read_text(encoding="utf-8"))
        events = _parse_jsonl(result.run_path / "events.jsonl")

        complete_events = {
            e["task_id"]: e
            for e in events
            if e.get("event") == "task_complete" and "task_id" in e
        }

        for tid, ev in complete_events.items():
            assert "status" in ev, (
                f"task_complete event for '{tid}' must carry a 'status' field"
            )
            manifest_status = manifest["task_results"][tid]["status"]
            assert ev["status"] == manifest_status, (
                f"task_complete 'status' for '{tid}' ({ev['status']!r}) "
                f"must match manifest status ({manifest_status!r})"
            )


# ---------------------------------------------------------------------------
# Test 45: manifest task_results entries always have a 'cost_usd' key
# ---------------------------------------------------------------------------


class TestManifestTaskCostUsdKey:
    def test_every_task_result_has_cost_usd_key(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every entry in run_manifest.json['task_results'] must contain a
        'cost_usd' key, even if its value is null/None.

        This validates the manifest schema contract so downstream tools can
        always access the field without a KeyError."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("cost-a"),
                _make_task("cost-b", depends_on=["cost-a"]),
                _make_task("cost-c", depends_on=["cost-a"]),
            ],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        manifest = json.loads((result.run_path / "run_manifest.json").read_text(encoding="utf-8"))
        task_results = manifest["task_results"]

        assert len(task_results) == 3, (
            f"Expected 3 task entries in manifest, got {len(task_results)}"
        )
        for tid, entry in task_results.items():
            assert "cost_usd" in entry, (
                f"task_results['{tid}'] must contain 'cost_usd' key "
                f"(got keys: {list(entry.keys())})"
            )


# ---------------------------------------------------------------------------
# Test 46: serial chain event interleaving — task_complete(A) before task_start(B)
# ---------------------------------------------------------------------------


class TestSerialChainEventInterleaving:
    def test_task_complete_precedes_next_task_start_in_serial_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In a strict serial chain A→B→C with max_parallel=1, the events.jsonl
        must interleave correctly: task_complete(A) appears before task_start(B),
        and task_complete(B) appears before task_start(C).

        This verifies that the scheduler does not dispatch a task before its
        dependency has emitted its completion event."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("chain-a"),
                _make_task("chain-b", depends_on=["chain-a"]),
                _make_task("chain-c", depends_on=["chain-b"]),
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True, "Serial chain must succeed"

        events = _parse_jsonl(result.run_path / "events.jsonl")

        def index_of(event_type: str, task_id: str) -> int:
            for i, e in enumerate(events):
                if e.get("event") == event_type and e.get("task_id") == task_id:
                    return i
            return -1

        complete_a = index_of("task_complete", "chain-a")
        start_b = index_of("task_start", "chain-b")
        complete_b = index_of("task_complete", "chain-b")
        start_c = index_of("task_start", "chain-c")

        for idx, label in [
            (complete_a, "task_complete(chain-a)"),
            (start_b, "task_start(chain-b)"),
            (complete_b, "task_complete(chain-b)"),
            (start_c, "task_start(chain-c)"),
        ]:
            assert idx >= 0, f"Event '{label}' not found in events.jsonl"

        assert complete_a < start_b, (
            f"task_complete(chain-a) (idx={complete_a}) must precede "
            f"task_start(chain-b) (idx={start_b})"
        )
        assert complete_b < start_c, (
            f"task_complete(chain-b) (idx={complete_b}) must precede "
            f"task_start(chain-c) (idx={start_c})"
        )


# ---------------------------------------------------------------------------
# Test 47: run_complete 'ok' count matches count of successful tasks in manifest
# ---------------------------------------------------------------------------


class TestRunCompleteOkCountMatchesManifest:
    def test_run_complete_ok_matches_manifest_success_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 'ok' field in the run_complete event must equal the number of
        tasks with status='success' in run_manifest.json['task_results'].

        This validates the cross-module contract between scheduler's event
        emission and its manifest writing."""
        from datetime import UTC, datetime

        def mixed_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            # Second task fails (allow_failure so plan continues)
            status = "failed" if task.id == "ok-fail" else "success"
            exit_code = 1 if status == "failed" else 0
            r = TaskResult(
                task_id=task.id,
                status=status,  # type: ignore[arg-type]
                exit_code=exit_code,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok" if status == "success" else "fail",
            )
            r.log_path.write_text(f"status={status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mixed_execute)

        plan = _make_plan(
            [
                _make_task("ok-a"),
                _make_task("ok-fail", allow_failure=True),
                _make_task("ok-b", depends_on=["ok-a"]),
                _make_task("ok-c", depends_on=["ok-a"]),
            ],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        manifest = json.loads((result.run_path / "run_manifest.json").read_text(encoding="utf-8"))
        events = _parse_jsonl(result.run_path / "events.jsonl")

        manifest_ok_count = sum(
            1
            for v in manifest["task_results"].values()
            if v.get("status") == "success"
        )

        run_complete = next(
            (e for e in events if e.get("event") == "run_complete"), None
        )
        assert run_complete is not None, "run_complete event must be present"
        assert "ok" in run_complete, "run_complete event must carry an 'ok' field"

        assert run_complete["ok"] == manifest_ok_count, (
            f"run_complete 'ok' ({run_complete['ok']}) must equal "
            f"count of success tasks in manifest ({manifest_ok_count})"
        )


# ---------------------------------------------------------------------------
# Test 48: per-task result.json 'task_id' field matches the directory filename
# ---------------------------------------------------------------------------


class TestPerTaskResultJsonTaskId:
    def test_result_json_task_id_matches_filename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Each <task-id>.result.json file must contain a 'task_id' field whose
        value matches the stem of the filename.

        This is the canonical way downstream tools (diff, report, suggest)
        locate per-task results without re-parsing the manifest."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        task_ids = ["rj-alpha", "rj-beta", "rj-gamma"]
        plan = _make_plan(
            [_make_task(tid) for tid in task_ids],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        for tid in task_ids:
            result_file = result.run_path / f"{tid}.result.json"
            assert result_file.exists(), f"{tid}.result.json must exist in run_path"
            data = json.loads(result_file.read_text(encoding="utf-8"))
            assert "task_id" in data, (
                f"{tid}.result.json must contain 'task_id' field"
            )
            assert data["task_id"] == tid, (
                f"{tid}.result.json 'task_id' field ({data['task_id']!r}) "
                f"must match the filename stem ({tid!r})"
            )


# ---------------------------------------------------------------------------
# Test 49: PlanRunResult timing — started_at and finished_at are populated,
#           finished_at >= started_at, and both appear in the manifest
# ---------------------------------------------------------------------------


class TestPlanRunResultTiming:
    def test_started_at_and_finished_at_are_populated_and_ordered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PlanRunResult.started_at and finished_at must both be non-None
        datetimes and finished_at must be >= started_at.  Both values must
        also be written to run_manifest.json as ISO 8601 strings that parse
        back to equivalent datetimes (within 1-second tolerance).

        This validates that scheduler.run_plan() records wall-clock timing
        correctly and that _write_manifest() faithfully serialises both fields."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("timing-a"), _make_task("timing-b", depends_on=["timing-a"])],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # started_at and finished_at must be real datetime objects
        from datetime import datetime as _dt

        assert isinstance(result.started_at, _dt), (
            "PlanRunResult.started_at must be a datetime object"
        )
        assert isinstance(result.finished_at, _dt), (
            "PlanRunResult.finished_at must be a datetime object"
        )

        # finished_at must be >= started_at (no negative durations)
        assert result.finished_at >= result.started_at, (
            f"finished_at ({result.finished_at.isoformat()}) must be >= "
            f"started_at ({result.started_at.isoformat()})"
        )

        # Manifest must carry matching ISO 8601 timestamps
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "started_at" in manifest, "run_manifest.json must have 'started_at'"
        assert "finished_at" in manifest, "run_manifest.json must have 'finished_at'"

        manifest_started = _dt.fromisoformat(manifest["started_at"])
        manifest_finished = _dt.fromisoformat(manifest["finished_at"])

        # Allow up to 1 second of floating-point drift
        assert abs((manifest_started - result.started_at).total_seconds()) < 1.0, (
            f"Manifest started_at ({manifest['started_at']!r}) differs from "
            f"PlanRunResult.started_at ({result.started_at.isoformat()!r})"
        )
        assert abs((manifest_finished - result.finished_at).total_seconds()) < 1.0, (
            f"Manifest finished_at ({manifest['finished_at']!r}) differs from "
            f"PlanRunResult.finished_at ({result.finished_at.isoformat()!r})"
        )
        assert manifest_finished >= manifest_started, (
            "Manifest finished_at must be >= started_at"
        )


# ---------------------------------------------------------------------------
# Test 50: execution_profile propagation — the profile passed to run_plan()
#           is recorded in PlanRunResult and in run_manifest.json
# ---------------------------------------------------------------------------


class TestExecutionProfilePropagation:
    def test_execution_profile_recorded_in_result_and_manifest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When run_plan() is called with execution_profile='safe', both
        PlanRunResult.execution_profile and the 'execution_profile' key in
        run_manifest.json must equal 'safe'.

        This validates that the profile is threaded from the CLI call through
        the scheduler into the manifest so audit and diff tooling can identify
        which safety posture was in effect during a run."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("prof-a"), _make_task("prof-b", depends_on=["prof-a"])],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            execution_profile="safe",
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.execution_profile == "safe", (
            f"PlanRunResult.execution_profile must be 'safe', "
            f"got {result.execution_profile!r}"
        )

        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "execution_profile" in manifest, (
            "run_manifest.json must carry 'execution_profile'"
        )
        assert manifest["execution_profile"] == "safe", (
            f"Manifest 'execution_profile' must be 'safe', "
            f"got {manifest['execution_profile']!r}"
        )

    def test_default_execution_profile_is_plan(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no execution_profile is passed to run_plan(), the default
        must be 'plan' in both PlanRunResult and run_manifest.json."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("prof-default")],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.execution_profile == "plan", (
            f"Default execution_profile must be 'plan', got {result.execution_profile!r}"
        )
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest.get("execution_profile") == "plan", (
            f"Manifest execution_profile must default to 'plan', "
            f"got {manifest.get('execution_profile')!r}"
        )


# ---------------------------------------------------------------------------
# Test 51: Fan-in DAG — task with two independent upstream deps only executes
#           after BOTH upstreams have completed
# ---------------------------------------------------------------------------


class TestFanInDAG:
    def test_fan_in_task_runs_only_after_all_upstreams_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A task D that depends on both B and C (fan-in) must not start until
        both B and C have completed.  Using max_parallel=2, B and C can run
        concurrently, so the test verifies that D's call_time counter is
        strictly greater than both B's and C's call_time counters.

        Plan layout:
          A → B ─┐
                  ├→ D
          A → C ─┘

        All start from A; B and C run in parallel; D waits for both."""
        call_order: list[str] = []
        call_counter: dict[str, int] = {}
        counter = [0]
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                counter[0] += 1
                call_counter[task.id] = counter[0]
                call_order.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("fi-a"),
                _make_task("fi-b", depends_on=["fi-a"]),
                _make_task("fi-c", depends_on=["fi-a"]),
                _make_task("fi-d", depends_on=["fi-b", "fi-c"]),
            ],
            fail_fast=False,
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True, "Fan-in DAG run must succeed"
        assert set(call_order) == {"fi-a", "fi-b", "fi-c", "fi-d"}, (
            "All 4 tasks must be executed"
        )

        # D must start after both B and C have started
        assert call_counter["fi-d"] > call_counter["fi-b"], (
            "fi-d must be dispatched after fi-b"
        )
        assert call_counter["fi-d"] > call_counter["fi-c"], (
            "fi-d must be dispatched after fi-c"
        )

        # Verify via events: task_complete for B and C must appear before
        # task_start for D
        events = _parse_jsonl(result.run_path / "events.jsonl")

        def idx_of(evt: str, tid: str) -> int:
            for i, e in enumerate(events):
                if e.get("event") == evt and e.get("task_id") == tid:
                    return i
            return -1

        complete_b = idx_of("task_complete", "fi-b")
        complete_c = idx_of("task_complete", "fi-c")
        start_d = idx_of("task_start", "fi-d")

        assert complete_b >= 0, "task_complete(fi-b) must be present in events"
        assert complete_c >= 0, "task_complete(fi-c) must be present in events"
        assert start_d >= 0, "task_start(fi-d) must be present in events"

        assert complete_b < start_d, (
            f"task_complete(fi-b) (idx={complete_b}) must precede "
            f"task_start(fi-d) (idx={start_d})"
        )
        assert complete_c < start_d, (
            f"task_complete(fi-c) (idx={complete_c}) must precede "
            f"task_start(fi-d) (idx={start_d})"
        )


# ---------------------------------------------------------------------------
# Test 52: run_summary.md Status row — shows SUCCESS on clean run,
#           FAILED when at least one task fails
# ---------------------------------------------------------------------------


class TestSummaryStatusRow:
    def test_summary_status_row_shows_success_on_clean_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_summary.md must contain a '| Status | **SUCCESS** |' row
        when every task in the plan succeeds.

        This validates the _write_summary() status_label rendering path."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("stat-ok-a"), _make_task("stat-ok-b", depends_on=["stat-ok-a"])],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True
        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")

        success_row_found = any(
            "| Status |" in line and "SUCCESS" in line
            for line in summary.splitlines()
        )
        assert success_row_found, (
            "run_summary.md must contain '| Status | **SUCCESS** |' row "
            "when all tasks succeed"
        )

    def test_summary_status_row_shows_failed_when_task_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_summary.md must contain a '| Status | **FAILED** |' row
        when at least one task fails.

        This validates the _write_summary() status_label='FAILED' path."""

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            status: str = "failed" if task.id == "stat-fail" else "success"
            exit_code = 1 if status == "failed" else 0
            r = TaskResult(
                task_id=task.id,
                status=status,  # type: ignore[arg-type]
                exit_code=exit_code,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="failure" if status == "failed" else "ok",
            )
            r.log_path.write_text(f"status={status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [_make_task("stat-ok"), _make_task("stat-fail")],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False
        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")

        failed_row_found = any(
            "| Status |" in line and "FAILED" in line
            for line in summary.splitlines()
        )
        assert failed_row_found, (
            "run_summary.md must contain '| Status | **FAILED** |' row "
            "when at least one task fails"
        )


# ---------------------------------------------------------------------------
# Test 53: run_path directory naming — name follows <timestamp>_<plan-name>
#           pattern and is a direct child of run_dir_override
# ---------------------------------------------------------------------------


class TestRunPathDirectoryNaming:
    def test_run_path_name_matches_timestamp_plan_name_pattern(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """PlanRunResult.run_path directory name must follow the
        '<timestamp>_<plan-name>' naming convention: the name must contain
        an underscore separating a digit-prefixed timestamp segment from the
        plan name, and must be a direct child of run_dir_override.

        This validates the run directory naming contract that downstream tools
        (maestro diff, maestro report, maestro verify, maestro blame) use to
        discover and sort run directories by timestamp."""
        import re

        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan_name = "naming-test-plan"
        plan = _make_plan(
            [_make_task("name-t1"), _make_task("name-t2", depends_on=["name-t1"])],
            name=plan_name,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        run_dir_name = result.run_path.name

        # Name must contain the plan name
        assert plan_name in run_dir_name, (
            f"run_path directory name {run_dir_name!r} must contain plan name {plan_name!r}"
        )

        # Name must have an underscore (separating timestamp from plan name)
        assert "_" in run_dir_name, (
            f"run_path directory name {run_dir_name!r} must contain '_' "
            "separating timestamp from plan name"
        )

        # The prefix before the plan name must start with digits (timestamp)
        # Pattern: one or more digits somewhere before an underscore
        timestamp_prefix = run_dir_name.split(plan_name)[0]
        assert re.search(r"\d", timestamp_prefix), (
            f"run_path directory name {run_dir_name!r} must have a digit-based "
            f"timestamp prefix before the plan name; got prefix: {timestamp_prefix!r}"
        )

        # run_path must be a direct child of the run_dir_override directory
        run_dir = tmp_path / "runs"
        assert result.run_path.parent == run_dir, (
            f"run_path.parent must equal run_dir_override ({run_dir}), "
            f"got {result.run_path.parent}"
        )

        # run_path must exist as an actual directory on disk
        assert result.run_path.is_dir(), (
            f"run_path {result.run_path} must be an existing directory"
        )


# ---------------------------------------------------------------------------
# Test 54: Manifest schema — run_path is an absolute path matching PlanRunResult
# ---------------------------------------------------------------------------


class TestManifestRunPath:
    def test_manifest_run_path_is_absolute_and_matches_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_manifest.json 'run_path' must be an absolute path string that
        matches str(PlanRunResult.run_path).

        This validates that the manifest serialises the run_path with full
        filesystem location information, which downstream tools (maestro diff,
        maestro report, maestro blame) rely on to locate artifacts without
        needing to know the working directory at the time of the run."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("mrp-t1"), _make_task("mrp-t2", depends_on=["mrp-t1"])],
            name="manifest-run-path-plan",
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        manifest_path = result.run_path / "run_manifest.json"
        assert manifest_path.exists(), "run_manifest.json must exist after run"

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        # run_path must be a string in the manifest
        assert "run_path" in manifest, "manifest must contain 'run_path'"
        assert isinstance(manifest["run_path"], str), (
            f"manifest 'run_path' must be a str, got {type(manifest['run_path'])}"
        )

        # run_path in manifest must be absolute
        manifest_run_path = Path(manifest["run_path"])
        assert manifest_run_path.is_absolute(), (
            f"manifest 'run_path' must be an absolute path, got {manifest['run_path']!r}"
        )

        # run_path in manifest must match PlanRunResult.run_path
        assert manifest_run_path == result.run_path, (
            f"manifest run_path {manifest_run_path!r} does not match "
            f"PlanRunResult.run_path {result.run_path!r}"
        )


# ---------------------------------------------------------------------------
# Test 55: Event ordering — all task_complete events are bracketed by
#           run_start and run_complete
# ---------------------------------------------------------------------------


class TestTaskCompleteEventsBracketed:
    def test_all_task_complete_events_between_run_start_and_run_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every task_complete event must appear after run_start and before
        run_complete in events.jsonl.

        This validates that the scheduler never emits task lifecycle events
        outside the bookend run_start/run_complete envelope — a property that
        event replay and audit tools depend on."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("brk-a"),
                _make_task("brk-b", depends_on=["brk-a"]),
                _make_task("brk-c", depends_on=["brk-a"]),
                _make_task("brk-d", depends_on=["brk-b", "brk-c"]),
            ],
            fail_fast=False,
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        event_names = [e["event"] for e in events]

        assert "run_start" in event_names, "run_start must be present"
        assert "run_complete" in event_names, "run_complete must be present"

        run_start_idx = event_names.index("run_start")
        run_complete_idx = len(event_names) - 1 - event_names[::-1].index("run_complete")

        # Every task_complete must appear strictly inside the bookend range
        for i, evt in enumerate(events):
            if evt.get("event") == "task_complete":
                assert i > run_start_idx, (
                    f"task_complete for '{evt.get('task_id')}' at index {i} "
                    f"appears before run_start at index {run_start_idx}"
                )
                assert i < run_complete_idx, (
                    f"task_complete for '{evt.get('task_id')}' at index {i} "
                    f"appears after run_complete at index {run_complete_idx}"
                )

        # All 4 task IDs must appear in task_complete events
        completed_ids = {
            e["task_id"]
            for e in events
            if e.get("event") == "task_complete" and "task_id" in e
        }
        assert completed_ids == {"brk-a", "brk-b", "brk-c", "brk-d"}, (
            f"Expected task_complete events for all 4 tasks, got {completed_ids}"
        )


# ---------------------------------------------------------------------------
# Test 56: Hash chain — appending a spurious line invalidates verify_chain()
# ---------------------------------------------------------------------------


class TestHashChainSpoofedAppend:
    def test_appending_spurious_event_invalidates_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Appending an arbitrary JSON line to events.jsonl after the run must
        cause verify_chain() to return 'tampered' or 'incomplete' — never 'valid'.

        The hash chain contract requires that any post-hoc mutation is detectable,
        including additions as well as modifications.  This test confirms the
        forward-only hash chain catches spoofed appended events."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("hca-1"), _make_task("hca-2", depends_on=["hca-1"])],
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events_path = result.run_path / "events.jsonl"
        assert events_path.exists(), "events.jsonl must exist"

        # Confirm the original chain is valid
        original_records = replay_events(events_path)
        original_status = verify_chain(original_records)
        assert original_status == "valid", (
            f"Pre-tamper chain must be 'valid', got '{original_status}'"
        )

        # Append a spurious event line (well-formed JSON but not chained)
        original_text = events_path.read_text(encoding="utf-8")
        spurious_line = json.dumps({
            "event": "run_start",
            "ts": "2099-01-01T00:00:00+00:00",
            "plan": "injected",
            "plan_name": "injected",
            "seq": 9999,
            "hash": "0" * 64,
            "prev_hash": "0" * 64,
        })
        events_path.write_text(
            original_text + spurious_line + "\n", encoding="utf-8"
        )

        tampered_records = replay_events(events_path)
        tampered_status = verify_chain(tampered_records)
        assert tampered_status != "valid", (
            f"verify_chain() must return 'tampered' or 'incomplete' after "
            f"appending a spurious event, but returned '{tampered_status}'"
        )


# ---------------------------------------------------------------------------
# Test 57: Diamond DAG — task_start(D) appears after task_complete(B) and
#           task_complete(C) in events.jsonl
# ---------------------------------------------------------------------------


class TestDiamondDAGEventOrdering:
    def test_diamond_dag_node_d_task_start_after_b_and_c_task_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """In a diamond DAG (A→B, A→C, B+C→D), the task_start event for D
        must appear in events.jsonl strictly after the task_complete events
        for both B and C.

        This validates the event-level ordering contract of the DAG scheduler:
        D may only start after its two direct dependencies have both completed,
        and that ordering must be reflected in the event log (not just inferred
        from call_log timestamps)."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("dmd-a"),
                _make_task("dmd-b", depends_on=["dmd-a"]),
                _make_task("dmd-c", depends_on=["dmd-a"]),
                _make_task("dmd-d", depends_on=["dmd-b", "dmd-c"]),
            ],
            fail_fast=False,
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True, "Diamond DAG must complete successfully"

        events = _parse_jsonl(result.run_path / "events.jsonl")

        # Find the index of task_complete for B and C
        def _complete_idx(task_id: str) -> int:
            for i, e in enumerate(events):
                if e.get("event") == "task_complete" and e.get("task_id") == task_id:
                    return i
            raise AssertionError(f"No task_complete event found for '{task_id}'")

        # Find the index of task_start for D
        def _start_idx(task_id: str) -> int:
            for i, e in enumerate(events):
                if e.get("event") == "task_start" and e.get("task_id") == task_id:
                    return i
            raise AssertionError(f"No task_start event found for '{task_id}'")

        b_complete = _complete_idx("dmd-b")
        c_complete = _complete_idx("dmd-c")
        d_start = _start_idx("dmd-d")

        assert b_complete < d_start, (
            f"task_complete(dmd-b) at index {b_complete} must precede "
            f"task_start(dmd-d) at index {d_start} in events.jsonl"
        )
        assert c_complete < d_start, (
            f"task_complete(dmd-c) at index {c_complete} must precede "
            f"task_start(dmd-d) at index {d_start} in events.jsonl"
        )


# ---------------------------------------------------------------------------
# Test 58: fail_fast — run_complete event carries success=False and
#           PlanRunResult.success is False when fail_fast triggers
# ---------------------------------------------------------------------------


class TestFailFastRunCompleteConsistency:
    def test_fail_fast_run_complete_event_and_result_both_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With fail_fast=True, when a task fails the run_complete event must
        carry success=False, and PlanRunResult.success must also be False.

        This cross-module contract ensures that the scheduler's fail_fast
        logic propagates the failure into both the event log (consumed by
        streaming watchers) and the in-memory result (consumed by callers).
        Both signals must agree — a divergence would silently lie to either
        the event stream or the caller."""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            if task.id == "ffrc-fail":
                r = TaskResult(
                    task_id=task.id,
                    status="failed",
                    exit_code=1,
                    started_at=now,
                    finished_at=now,
                    duration_sec=0.01,
                    command="false",
                    log_path=run_path / f"{task.id}.log",
                    result_path=run_path / f"{task.id}.result.json",
                    message="intentional failure for fail_fast test",
                )
            else:
                r = TaskResult(
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
            r.log_path.write_text(f"status={r.status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("ffrc-fail"),
                _make_task("ffrc-ok-1"),
                _make_task("ffrc-ok-2"),
            ],
            fail_fast=True,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # PlanRunResult.success must be False
        assert result.success is False, (
            "PlanRunResult.success must be False when fail_fast triggers"
        )

        # The failed task must appear in task_results
        assert "ffrc-fail" in result.task_results, (
            "Failed task 'ffrc-fail' must appear in PlanRunResult.task_results"
        )
        assert result.task_results["ffrc-fail"].status == "failed", (
            "ffrc-fail status must be 'failed'"
        )

        # At least one task must have been skipped (fail_fast effect)
        statuses = {tid: tr.status for tid, tr in result.task_results.items()}
        skipped = [tid for tid, s in statuses.items() if s == "skipped"]
        assert skipped, (
            f"fail_fast must cause at least one task to be skipped, "
            f"got statuses: {statuses}"
        )

        # run_complete event must also report success=False
        events = _parse_jsonl(result.run_path / "events.jsonl")
        run_complete_events = [e for e in events if e.get("event") == "run_complete"]
        assert run_complete_events, "run_complete event must be present in events.jsonl"

        run_complete = run_complete_events[0]
        assert "success" in run_complete, (
            "run_complete event must have a 'success' field"
        )
        assert run_complete["success"] is False, (
            f"run_complete event 'success' must be False when fail_fast triggers, "
            f"got {run_complete['success']!r}"
        )

        # Both signals must agree
        assert run_complete["success"] == result.success, (
            f"run_complete event success={run_complete['success']!r} must match "
            f"PlanRunResult.success={result.success!r}"
        )


# ---------------------------------------------------------------------------
# Test 59: skip_tags filter — tasks with matching tag are excluded from run
# ---------------------------------------------------------------------------


class TestSkipTagsFilter:
    def test_skip_tags_excludes_matching_tasks_from_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_plan() with skip_tags={'ci-only'} must exclude every task that
        carries the 'ci-only' tag from execution and from PlanRunResult.task_results.
        Tasks without that tag must still execute normally.

        Plan layout (all independent, max_parallel=1):
          tagged-ci   (tags=['ci-only'])    — should be excluded
          tagged-qa   (tags=['qa'])          — should run
          untagged    (no tags)              — should run

        With skip_tags={'ci-only'}, tagged-ci must be absent from executed list
        and absent from task_results; tagged-qa and untagged must succeed."""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        from maestro_cli.models import TaskSpec as _TaskSpec

        def _tagged(task_id: str, tags: list[str]) -> _TaskSpec:
            return _TaskSpec(
                id=task_id,
                description=f"task {task_id}",
                tags=tags,
                depends_on=[],
                command="echo ok",
            )

        plan = _make_plan(
            [
                _tagged("tagged-ci", ["ci-only"]),
                _tagged("tagged-qa", ["qa"]),
                _tagged("untagged", []),
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            skip_tags={"ci-only"},
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True, "skip_tags run must succeed"

        # 'tagged-ci' must be excluded entirely from execution and task_results
        assert "tagged-ci" not in executed, (
            "tagged-ci has tag 'ci-only' and must not be passed to execute_task "
            "when skip_tags={'ci-only'}"
        )
        assert "tagged-ci" not in result.task_results, (
            "tagged-ci must not appear in PlanRunResult.task_results when skip_tags={'ci-only'}"
        )

        # 'tagged-qa' and 'untagged' must have executed normally
        assert "tagged-qa" in executed, "tagged-qa (tag='qa') must execute when skip_tags={'ci-only'}"
        assert "untagged" in executed, "untagged must execute when skip_tags={'ci-only'}"
        assert result.task_results["tagged-qa"].status == "success"
        assert result.task_results["untagged"].status == "success"

        # Manifest must not contain tagged-ci
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "tagged-ci" not in manifest["task_results"], (
            "tagged-ci must not appear in run_manifest.json when excluded by skip_tags"
        )


# ---------------------------------------------------------------------------
# Test 60: when condition met — task executes when expression evaluates true
# ---------------------------------------------------------------------------


class TestWhenConditionMet:
    def test_when_condition_true_task_executes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A task with ``when: '{{ dep.status }} == success'`` must execute
        when its dependency completed successfully.

        Plan layout (serial, max_parallel=1):
          dep-task  (command task, will succeed)
          when-task (depends_on=['dep-task'], when='{{ dep-task.status }} == success')

        Expected: dep-task runs and succeeds → when condition is true →
        when-task executes → both tasks end up as 'success' in the manifest.

        This validates the _evaluate_ready() path that evaluates when expressions
        against actual upstream results before dispatching a task."""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        from maestro_cli.models import TaskSpec as _TaskSpec

        dep_task = _TaskSpec(
            id="dep-task",
            description="dependency that succeeds",
            depends_on=[],
            command="echo ok",
        )
        when_task = _TaskSpec(
            id="when-task",
            description="conditional task that should run",
            depends_on=["dep-task"],
            command="echo conditional",
            when="{{ dep-task.status }} == success",
        )

        plan = _make_plan(
            [dep_task, when_task],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True, "Plan must succeed when when-condition is met"

        assert "dep-task" in executed, "dep-task must have been executed"
        assert "when-task" in executed, (
            "when-task must execute because '{{ dep-task.status }} == success' is true "
            "after dep-task succeeds"
        )

        assert result.task_results["dep-task"].status == "success"
        assert result.task_results["when-task"].status == "success"

        # Verify no task_skip event was emitted for when-task
        events = _parse_jsonl(result.run_path / "events.jsonl")
        skip_events_for_when = [
            e for e in events
            if e.get("event") == "task_skip" and e.get("task_id") == "when-task"
        ]
        assert not skip_events_for_when, (
            "when-task must not have a task_skip event when its when condition is met"
        )


# ---------------------------------------------------------------------------
# Test 61: when condition not met — task is skipped and task_skip event emitted
# ---------------------------------------------------------------------------


class TestWhenConditionNotMet:
    def test_when_condition_false_task_skipped_with_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A task with ``when: '{{ dep.status }} == failed'`` must be skipped
        (not executed) when its dependency actually succeeded, and a task_skip
        event must be emitted for it.  The overall plan should still succeed
        because a when-skipped task is not a failure.

        Plan layout (serial, max_parallel=1):
          dep-ok     (command task, will succeed)
          cond-skip  (depends_on=['dep-ok'],
                      when='{{ dep-ok.status }} == failed')  ← condition is false

        Expected: dep-ok succeeds → when condition is false → cond-skip is skipped →
        task_skip event emitted → plan succeeds → manifest shows cond-skip as 'skipped'.

        This validates the _evaluate_ready() false-branch path in scheduler.py
        and the documented CLAUDE.md contract: 'skipped' tasks from when-expressions
        do not mark the run as failed."""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        from maestro_cli.models import TaskSpec as _TaskSpec

        dep_task = _TaskSpec(
            id="dep-ok",
            description="dependency that succeeds",
            depends_on=[],
            command="echo ok",
        )
        cond_task = _TaskSpec(
            id="cond-skip",
            description="conditional task whose condition is never met",
            depends_on=["dep-ok"],
            command="echo should-not-run",
            # This condition can never be true because dep-ok always succeeds
            when="{{ dep-ok.status }} == failed",
        )

        plan = _make_plan(
            [dep_task, cond_task],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # Plan must still succeed — when-skipped tasks don't cause failure
        assert result.success is True, (
            "Plan must succeed when a task is skipped due to an unmet when condition"
        )

        # cond-skip must NOT have been passed to execute_task
        assert "cond-skip" not in executed, (
            "cond-skip must not be passed to execute_task when its when condition is false"
        )

        # cond-skip must appear in task_results with status 'skipped'
        assert "cond-skip" in result.task_results, (
            "cond-skip must appear in PlanRunResult.task_results even when skipped"
        )
        assert result.task_results["cond-skip"].status == "skipped", (
            f"cond-skip status must be 'skipped', "
            f"got {result.task_results['cond-skip'].status!r}"
        )

        # A task_skip event must have been emitted for cond-skip
        events = _parse_jsonl(result.run_path / "events.jsonl")
        skip_events = [
            e for e in events
            if e.get("event") == "task_skip" and e.get("task_id") == "cond-skip"
        ]
        assert skip_events, (
            "A task_skip event must be emitted for 'cond-skip' when its when condition is false"
        )
        assert skip_events[0].get("reason"), (
            "task_skip event for 'cond-skip' must have a non-empty 'reason' field"
        )

        # Manifest must show cond-skip as skipped
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["task_results"]["cond-skip"]["status"] == "skipped", (
            "run_manifest.json must record cond-skip as 'skipped' when when-condition is false"
        )


# ---------------------------------------------------------------------------
# Test 62: max_parallel_override — overrides plan's max_parallel, reflected
#           in the run_start event's max_parallel field
# ---------------------------------------------------------------------------


class TestMaxParallelOverride:
    def test_max_parallel_override_reflected_in_run_start_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When run_plan() is called with max_parallel_override=2 and the plan
        has max_parallel=1, the run_start event must carry max_parallel=2
        (the override value, not the plan default).

        This validates the override wiring in scheduler.py:
            max_parallel = max_parallel_override or plan.max_parallel
        and that the resolved value is included in the run_start event payload."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("mp-a"), _make_task("mp-b"), _make_task("mp-c")],
            fail_fast=False,
            max_parallel=1,  # plan says serial
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            max_parallel_override=3,  # caller overrides to 3
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True, "run with max_parallel_override must succeed"

        # All tasks must have been executed regardless of parallelism setting
        assert set(result.task_results.keys()) == {"mp-a", "mp-b", "mp-c"}, (
            "All 3 tasks must appear in task_results"
        )

        # The run_start event must carry the overridden max_parallel value
        events = _parse_jsonl(result.run_path / "events.jsonl")
        run_start_events = [e for e in events if e.get("event") == "run_start"]
        assert run_start_events, "run_start event must be present"

        run_start = run_start_events[0]
        assert "max_parallel" in run_start, (
            "run_start event must have a 'max_parallel' field"
        )
        assert run_start["max_parallel"] == 3, (
            f"run_start 'max_parallel' must equal the override value 3, "
            f"got {run_start['max_parallel']!r} "
            f"(plan.max_parallel=1 must be overridden)"
        )


# ---------------------------------------------------------------------------
# Test 63: event_callback — caller-provided callback receives all events
# ---------------------------------------------------------------------------


class TestEventCallback:
    def test_event_callback_receives_all_events_during_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When run_plan() is called with a non-None event_callback, the callback
        must be invoked for every event emitted during the run (run_start,
        task_start, task_complete, run_complete at minimum).

        The callback receives (event_name: str, payload: dict) pairs.
        Each payload must carry the same 'event' field as the name argument,
        and every payload must carry a 'plan_name' field matching the plan's name.

        This validates the _emit() → event_callback integration path in scheduler.py
        and ensures external consumers (e.g. TUI, Live renderer) receive a complete
        and consistent event stream."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        received_events: list[tuple[str, dict[str, Any]]] = []
        callback_lock = threading.Lock()

        def my_callback(event_name: str, payload: dict[str, Any]) -> None:
            with callback_lock:
                received_events.append((event_name, payload))

        plan = _make_plan(
            [
                _make_task("cb-a"),
                _make_task("cb-b", depends_on=["cb-a"]),
            ],
            name="callback-test-plan",
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            event_callback=my_callback,
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True, "callback-enabled run must succeed"
        assert received_events, "event_callback must have been invoked at least once"

        # Extract the event names received by the callback
        callback_event_names = [name for name, _ in received_events]

        # At minimum, run_start, task_start (×2), task_complete (×2), run_complete
        # must all have been delivered to the callback
        assert "run_start" in callback_event_names, (
            "event_callback must receive run_start"
        )
        assert "task_start" in callback_event_names, (
            "event_callback must receive task_start"
        )
        assert "task_complete" in callback_event_names, (
            "event_callback must receive task_complete"
        )
        assert "run_complete" in callback_event_names, (
            "event_callback must receive run_complete"
        )

        # Every payload must carry 'event' and 'plan_name'
        for event_name, payload in received_events:
            assert "event" in payload, (
                f"Callback payload for '{event_name}' is missing 'event' field"
            )
            assert payload["event"] == event_name, (
                f"Callback payload 'event'={payload['event']!r} must match "
                f"the event_name argument {event_name!r}"
            )
            assert "plan_name" in payload, (
                f"Callback payload for '{event_name}' is missing 'plan_name' field"
            )
            assert payload["plan_name"] == "callback-test-plan", (
                f"Callback payload 'plan_name'={payload['plan_name']!r} must equal "
                f"the plan name 'callback-test-plan'"
            )

        # The callback must have been called at least once per task (task_start + task_complete)
        task_starts = [p for n, p in received_events if n == "task_start"]
        task_completes = [p for n, p in received_events if n == "task_complete"]
        assert len(task_starts) == 2, (
            f"Expected 2 task_start events (one per task), got {len(task_starts)}"
        )
        assert len(task_completes) >= 2, (
            f"Expected at least 2 task_complete events, got {len(task_completes)}"
        )


# ---------------------------------------------------------------------------
# Test 64: output_mode="jsonl" — all canonical artifacts are still written
# ---------------------------------------------------------------------------


class TestJsonlOutputModeArtifacts:
    def test_jsonl_output_mode_still_writes_all_canonical_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When run_plan() is called with output_mode='jsonl', all three canonical
        artifacts (run_manifest.json, run_summary.md, events.jsonl) must still be
        written to the run directory.

        CLAUDE.md documents: '--output jsonl suppresses all [maestro] text output'
        but this must not suppress artifact creation.  Downstream tools that rely
        on these files (maestro diff, maestro report, maestro verify) must work
        identically regardless of whether the run was executed in jsonl mode."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("jl-a"), _make_task("jl-b", depends_on=["jl-a"])],
            name="jsonl-mode-plan",
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            output_mode="jsonl",
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True, "jsonl-mode run must succeed"

        # All three canonical artifacts must be present and non-empty
        for artifact in ("run_manifest.json", "run_summary.md", "events.jsonl"):
            artifact_path = result.run_path / artifact
            assert artifact_path.exists(), (
                f"Canonical artifact '{artifact}' must exist even when output_mode='jsonl'"
            )
            assert artifact_path.stat().st_size > 0, (
                f"Canonical artifact '{artifact}' must be non-empty in jsonl mode"
            )

        # Manifest must be valid JSON with expected fields
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["plan_name"] == "jsonl-mode-plan"
        assert manifest["success"] is True
        assert "jl-a" in manifest["task_results"]
        assert "jl-b" in manifest["task_results"]

        # events.jsonl must have valid JSON lines with run_start and run_complete
        events = _parse_jsonl(result.run_path / "events.jsonl")
        event_names = [e.get("event") for e in events]
        assert "run_start" in event_names, (
            "events.jsonl must contain run_start even in jsonl output mode"
        )
        assert "run_complete" in event_names, (
            "events.jsonl must contain run_complete even in jsonl output mode"
        )


# ---------------------------------------------------------------------------
# Test 65: verbosity="quiet" — canonical artifacts are written on quiet run
# ---------------------------------------------------------------------------


class TestQuietVerbosityArtifacts:
    def test_quiet_verbosity_still_writes_canonical_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When run_plan() is called with verbosity='quiet', all canonical
        artifacts (run_manifest.json, run_summary.md, events.jsonl) must be
        written and each task must still appear in the manifest with its correct
        status.

        Verbosity controls console output, not artifact creation.  A quiet run
        must be functionally identical to a normal run from the perspective of
        any tooling that reads the output directory."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("qt-a"),
                _make_task("qt-b", depends_on=["qt-a"]),
                _make_task("qt-c", depends_on=["qt-a"]),
            ],
            name="quiet-verbosity-plan",
            fail_fast=False,
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            verbosity="quiet",
            run_dir_override=str(tmp_path / "runs"),
        )

        assert result.success is True, "quiet run must succeed"

        # All three canonical artifacts must exist
        for artifact in ("run_manifest.json", "run_summary.md", "events.jsonl"):
            assert (result.run_path / artifact).exists(), (
                f"'{artifact}' must be written on a quiet run"
            )

        # Manifest must contain all three task IDs with success status
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        for tid in ("qt-a", "qt-b", "qt-c"):
            assert tid in manifest["task_results"], (
                f"Task '{tid}' must appear in manifest on a quiet run"
            )
            assert manifest["task_results"][tid]["status"] == "success", (
                f"Task '{tid}' must have status 'success' on a quiet run"
            )

        # events.jsonl must be parseable and contain the right bookend events
        events = _parse_jsonl(result.run_path / "events.jsonl")
        assert events[0]["event"] == "run_start", (
            "First event must be run_start on a quiet run"
        )
        assert events[-1]["event"] == "run_complete", (
            "Last event must be run_complete on a quiet run"
        )


# ---------------------------------------------------------------------------
# Test 66: cancel_event — cancelling mid-run aborts pending tasks and
#           still writes run_manifest.json and events.jsonl
# ---------------------------------------------------------------------------


class TestCancelEvent:
    def test_cancel_event_aborts_pending_tasks_and_writes_artifacts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When run_plan() receives a cancel_event that is set before the run
        begins, the scheduler must abort quickly without executing any tasks,
        and must still write run_manifest.json and events.jsonl to the run
        directory so that the interrupted run is observable.

        This validates the cancel_event early-abort path: setting the event
        before run_plan() is called simulates an immediate cancellation request
        (e.g. from Ctrl-C or TUI exit) and the scheduler must handle it
        gracefully rather than hanging or raising an unhandled exception."""
        holder: list[Path] = []
        mock_fn, call_log = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        cancel_ev = threading.Event()
        cancel_ev.set()  # cancel before the run starts

        plan = _make_plan(
            [
                _make_task("cancel-a"),
                _make_task("cancel-b", depends_on=["cancel-a"]),
                _make_task("cancel-c", depends_on=["cancel-a"]),
            ],
            fail_fast=False,
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        # Must not raise; scheduler should detect the cancelled event and return
        result = run_plan(
            plan,
            cancel_event=cancel_ev,
            run_dir_override=str(tmp_path / "runs"),
        )

        # The run must return a PlanRunResult (not raise)
        assert result is not None, "run_plan must return a result even when cancelled"

        # run_manifest.json must be written so the run is discoverable
        manifest_path = result.run_path / "run_manifest.json"
        assert manifest_path.exists(), (
            "run_manifest.json must be written even when the run is cancelled"
        )

        # events.jsonl must be written (at minimum a run_start or run_complete)
        events_path = result.run_path / "events.jsonl"
        assert events_path.exists(), (
            "events.jsonl must be written even when the run is cancelled"
        )

        # Cancelled tasks must not have been executed by execute_task
        # (all tasks skipped/not-started due to immediate cancellation)
        for tid in ("cancel-b", "cancel-c"):
            if tid in result.task_results:
                assert result.task_results[tid].status in ("skipped", "failed"), (
                    f"Cancelled task '{tid}' must be skipped or failed, "
                    f"got {result.task_results[tid].status!r}"
                )


# ---------------------------------------------------------------------------
# Test 67: manifest total_tokens key — always present, even when None
# ---------------------------------------------------------------------------


class TestManifestTotalTokensKeyAlwaysPresent:
    def test_total_tokens_key_present_in_manifest_when_no_token_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_manifest.json must always contain a 'total_tokens' key, even when
        no tasks report token usage (value must be null/None rather than absent).

        This validates the manifest schema contract so downstream tools
        (maestro diff, maestro suggest, cost_backfill) can always access
        manifest['total_tokens'] without a KeyError, regardless of whether the
        run used engine tasks with token tracking."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [_make_task("tt-a"), _make_task("tt-b", depends_on=["tt-a"])],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True

        # PlanRunResult.total_tokens must be None when no tasks report tokens
        assert result.total_tokens is None, (
            f"total_tokens must be None when no tasks report token_usage, "
            f"got {result.total_tokens!r}"
        )

        # run_manifest.json must contain 'total_tokens' key (even if value is null)
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "total_tokens" in manifest, (
            "run_manifest.json must always contain the 'total_tokens' key "
            "(value may be null, but the key must be present for schema consistency)"
        )

        # Value must be null (None serialised as JSON null)
        assert manifest["total_tokens"] is None, (
            f"manifest 'total_tokens' must be null when no tasks report tokens, "
            f"got {manifest['total_tokens']!r}"
        )

    def test_total_tokens_key_present_in_manifest_with_token_data(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When tasks DO report token_usage, 'total_tokens' in run_manifest.json
        must be a positive integer equal to the summed token counts."""

        def mock_token_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            from datetime import UTC, datetime as _dt
            now = _dt.now(UTC)
            r = TaskResult(
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
                cost_usd=0.01,
                token_usage=TokenUsage(input_tokens=80, output_tokens=40),
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_token_execute)

        plan = _make_plan(
            [_make_task("tt2-a"), _make_task("tt2-b")],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # 2 tasks × (80 + 40) = 240 tokens
        expected = 240
        assert result.total_tokens == expected, (
            f"total_tokens must be {expected}, got {result.total_tokens!r}"
        )

        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "total_tokens" in manifest, (
            "run_manifest.json must contain 'total_tokens' when tasks report token usage"
        )
        assert manifest["total_tokens"] == expected, (
            f"manifest 'total_tokens' must be {expected}, got {manifest['total_tokens']!r}"
        )


# ---------------------------------------------------------------------------
# Test 68: replay_run_state with skipped tasks — skipped status flows through
# ---------------------------------------------------------------------------


class TestReplayRunStateWithSkippedTasks:
    def test_replay_run_state_includes_skipped_tasks_from_dep_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """replay_run_state() must include dependency-skipped tasks in its
        'completed_tasks' set and record their status as 'skipped'.

        When task 'rs-fail' fails (no allow_failure), its child 'rs-child' is
        skipped and a task_skip event is emitted.  replay_run_state() must pick
        up that event and reflect the correct skipped state.

        This validates the cross-module contract between scheduler.py (which
        emits task_skip events) and eventsource.replay_run_state() (which must
        consume those events to reconstruct run state for resume and audit)."""
        from maestro_cli.eventsource import replay_events, replay_run_state

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            status: str = "failed" if task.id == "rs-fail" else "success"
            exit_code = 1 if status == "failed" else 0
            r = TaskResult(
                task_id=task.id,
                status=status,  # type: ignore[arg-type]
                exit_code=exit_code,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="failure" if status == "failed" else "ok",
            )
            r.log_path.write_text(f"status={status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        plan = _make_plan(
            [
                _make_task("rs-ok"),
                _make_task("rs-fail"),
                _make_task("rs-child", depends_on=["rs-fail"]),
            ],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is False, "Plan must fail when rs-fail fails"

        # rs-child must be skipped in PlanRunResult
        assert result.task_results["rs-child"].status == "skipped", (
            "rs-child must be 'skipped' due to dep failure in PlanRunResult"
        )

        # Now replay from events.jsonl
        events_path = result.run_path / "events.jsonl"
        records = replay_events(events_path)
        state = replay_run_state(records)

        # rs-ok must be 'success' in replayed state
        assert state["tasks"].get("rs-ok") == "success", (
            f"Replayed status for 'rs-ok' must be 'success', "
            f"got {state['tasks'].get('rs-ok')!r}"
        )

        # rs-fail must be 'failed' in replayed state
        assert state["tasks"].get("rs-fail") == "failed", (
            f"Replayed status for 'rs-fail' must be 'failed', "
            f"got {state['tasks'].get('rs-fail')!r}"
        )

        # rs-child must be 'skipped' in replayed state and in completed_tasks
        assert state["tasks"].get("rs-child") == "skipped", (
            f"Replayed status for 'rs-child' must be 'skipped' "
            f"(from task_skip event), got {state['tasks'].get('rs-child')!r}"
        )
        assert "rs-child" in state["completed_tasks"], (
            "rs-child must be in completed_tasks after replay because "
            "task_skip adds tasks to the completed set"
        )


# ---------------------------------------------------------------------------
# Test 69: Resume re-runs dep-skipped tasks when their dependency succeeds
# ---------------------------------------------------------------------------


class TestResumeDepSkippedReExecution:
    def test_resume_reruns_dep_skipped_task_when_dependency_now_succeeds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run 1: task 'root' fails → 'child' is dep-skipped.
        Run 2: resume — 'root' succeeds → 'child' must be re-executed (not skipped).

        This validates the cross-module contract between _load_prior_results()
        (which excludes dep-skipped tasks from the "succeeded" set) and
        run_plan() resume logic (which re-executes non-succeeded tasks).  The
        dep-skipped message pattern 'Skipped because dependency failed:' must be
        detected correctly for the child task to be eligible for re-execution.

        Artifacts checked:
        - run_manifest.json from run 2: child must have status 'success'
        - events.jsonl from run 2: must contain task_start for 'child'
        - PlanRunResult from run 2: success=True"""
        lock = threading.Lock()
        run1_tasks: list[str] = []
        run2_tasks: list[str] = []

        # --- Run 1: root fails, child dep-skipped ---
        def mock_run1(
            plan: Any, task: Any, run_path: Path,
            dry_run: bool = False, execution_profile: str = "plan",
            upstream_results: Any = None, context_synthesis: str = "",
            workspace_brief: str = "", **kwargs: Any,
        ) -> TaskResult:
            with lock:
                run1_tasks.append(task.id)
            now = datetime.now(UTC)
            status = "failed" if task.id == "root" else "success"
            r = TaskResult(
                task_id=task.id, status=status,  # type: ignore[arg-type]
                exit_code=1 if status == "failed" else 0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="root failed" if status == "failed" else "ok",
            )
            r.log_path.write_text(f"status={status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_run1)
        plan = _make_plan(
            [_make_task("root"), _make_task("child", depends_on=["root"])],
            fail_fast=False, max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        run1 = run_plan(plan, run_dir_override=str(tmp_path / "runs1"))

        assert run1.success is False
        assert run1.task_results["child"].status == "skipped"

        # --- Run 2: resume — root now succeeds ---
        def mock_run2(
            plan: Any, task: Any, run_path: Path,
            dry_run: bool = False, execution_profile: str = "plan",
            upstream_results: Any = None, context_synthesis: str = "",
            workspace_brief: str = "", **kwargs: Any,
        ) -> TaskResult:
            with lock:
                run2_tasks.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
                task_id=task.id, status="success",
                exit_code=0, started_at=now, finished_at=now,
                duration_sec=0.01, command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok",
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_run2)
        run2 = run_plan(
            plan, run_dir_override=str(tmp_path / "runs2"),
            resume_path=run1.run_path,
        )

        # Both root and child must have been re-executed (dep-skipped excluded from resume)
        assert "root" in run2_tasks, (
            "root must be re-executed on resume (it failed in run 1)"
        )
        assert "child" in run2_tasks, (
            "child must be re-executed on resume — it was dep-skipped in run 1 "
            "and _load_prior_results excludes dep-skipped tasks"
        )

        # Cross-artifact verification: manifest shows both as success
        manifest = json.loads(
            (run2.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["task_results"]["root"]["status"] == "success"
        assert manifest["task_results"]["child"]["status"] == "success"
        assert manifest["success"] is True

        # events.jsonl must contain task_start for child (proving it ran)
        events = _parse_jsonl(run2.run_path / "events.jsonl")
        child_starts = [e for e in events if e["event"] == "task_start" and e.get("task_id") == "child"]
        assert len(child_starts) == 1, (
            "events.jsonl must contain exactly one task_start for 'child' on resume"
        )


# ---------------------------------------------------------------------------
# Test 70: Event sequence numbers are monotonically increasing
# ---------------------------------------------------------------------------


class TestEventSequenceMonotonicity:
    def test_event_seq_values_are_strictly_increasing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every event in events.jsonl must have a 'seq' field with values that
        are strictly monotonically increasing (seq[i] < seq[i+1]).

        This validates the cross-module contract between scheduler.py (which
        calls _emit() for run-level events) and eventsource.py (which manages
        the hash chain with sequence numbers).  A non-monotonic sequence would
        indicate a concurrency bug in event emission or a broken ChainState.

        Artifacts checked:
        - events.jsonl: all events have 'seq', values strictly increase
        - Hash chain must be valid (verify_chain confirms seq ordering)"""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        # Diamond DAG: A → B, A → C, B+C → D (maximizes concurrency)
        plan = _make_plan(
            [
                _make_task("seq-a"),
                _make_task("seq-b", depends_on=["seq-a"]),
                _make_task("seq-c", depends_on=["seq-a"]),
                _make_task("seq-d", depends_on=["seq-b", "seq-c"]),
            ],
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        assert len(events) >= 2, "Must have at least run_start and run_complete"

        # Every event must have a 'seq' field
        for i, evt in enumerate(events):
            assert "seq" in evt, (
                f"Event at index {i} ({evt.get('event', '?')}) missing 'seq' field"
            )

        # seq values must be strictly increasing
        seq_values = [evt["seq"] for evt in events]
        for i in range(1, len(seq_values)):
            assert seq_values[i] > seq_values[i - 1], (
                f"seq values must be strictly increasing: "
                f"seq[{i-1}]={seq_values[i-1]} >= seq[{i}]={seq_values[i]}"
            )

        # Hash chain must also be valid (secondary cross-module check)
        records = replay_events(result.run_path / "events.jsonl")
        status = verify_chain(records)
        assert status == "valid", (
            f"Hash chain must be valid on a concurrent diamond DAG run, got '{status}'"
        )


# ---------------------------------------------------------------------------
# Test 71: Mixed-status run — manifest, events, summary all consistent
# ---------------------------------------------------------------------------


class TestMixedStatusRunConsistency:
    def test_mixed_status_run_artifacts_are_mutually_consistent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Run with success, failed, soft_failed, and dep-skipped tasks.
        Verify that all three artifacts (manifest, events, summary) agree
        on each task's status.

        This is the canonical cross-module integration test: it validates that
        scheduler.py task dispatch → runners mock → result collection →
        manifest serialization → event emission → summary generation all
        produce mutually consistent artifacts.

        Artifacts checked:
        - run_manifest.json: correct per-task statuses
        - events.jsonl: task_complete events carry matching statuses
        - run_summary.md: mentions each task with correct status text
        - Hash chain: valid even with mixed statuses"""

        def mock_mixed(
            plan: Any, task: Any, run_path: Path,
            dry_run: bool = False, execution_profile: str = "plan",
            upstream_results: Any = None, context_synthesis: str = "",
            workspace_brief: str = "", **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            status_map: dict[str, str] = {
                "mx-ok": "success",
                "mx-fail": "failed",
                "mx-soft": "soft_failed",
            }
            status = status_map.get(task.id, "success")
            exit_code = 1 if status in ("failed", "soft_failed") else 0
            r = TaskResult(
                task_id=task.id, status=status,  # type: ignore[arg-type]
                exit_code=exit_code, started_at=now, finished_at=now,
                duration_sec=0.01, command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message=f"status is {status}",
            )
            r.log_path.write_text(f"status={status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_mixed)

        plan = _make_plan(
            [
                _make_task("mx-ok"),
                _make_task("mx-fail"),
                _make_task("mx-soft", allow_failure=True),
                _make_task("mx-child", depends_on=["mx-fail"]),  # will be dep-skipped
            ],
            fail_fast=False, max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # --- Manifest checks ---
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["task_results"]["mx-ok"]["status"] == "success"
        assert manifest["task_results"]["mx-fail"]["status"] == "failed"
        assert manifest["task_results"]["mx-soft"]["status"] == "soft_failed"
        assert manifest["task_results"]["mx-child"]["status"] == "skipped"
        assert manifest["success"] is False, (
            "Plan must fail when mx-fail fails without allow_failure"
        )

        # --- Events checks: task_complete statuses must match manifest ---
        events = _parse_jsonl(result.run_path / "events.jsonl")
        completes = {
            e["task_id"]: e["status"]
            for e in events if e["event"] == "task_complete"
        }
        for tid in ("mx-ok", "mx-fail", "mx-soft"):
            assert completes.get(tid) == manifest["task_results"][tid]["status"], (
                f"task_complete event status for '{tid}' must match manifest: "
                f"event={completes.get(tid)!r}, manifest={manifest['task_results'][tid]['status']!r}"
            )

        # --- Summary checks ---
        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")
        for tid in ("mx-ok", "mx-fail", "mx-soft", "mx-child"):
            assert tid in summary, (
                f"Task '{tid}' must appear in run_summary.md"
            )

        # --- Hash chain must be valid even with mixed statuses ---
        records = replay_events(result.run_path / "events.jsonl")
        assert verify_chain(records) == "valid", (
            "Hash chain must be valid on a mixed-status run"
        )


# ---------------------------------------------------------------------------
# Test 72: Summary contains all task IDs from manifest (cross-artifact)
# ---------------------------------------------------------------------------


class TestSummaryContainsAllManifestTasks:
    def test_every_manifest_task_appears_in_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every task_id listed in run_manifest.json must appear as text in
        run_summary.md.  This validates the cross-artifact contract between
        _write_manifest() and _write_summary() — both must enumerate the
        exact same set of tasks.

        Artifacts checked:
        - run_manifest.json: extract all task_ids
        - run_summary.md: verify each task_id appears as a substring"""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("inv-alpha"),
                _make_task("inv-beta", depends_on=["inv-alpha"]),
                _make_task("inv-gamma", depends_on=["inv-alpha"]),
                _make_task("inv-delta", depends_on=["inv-beta", "inv-gamma"]),
                _make_task("inv-epsilon"),
            ],
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.success is True

        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")

        manifest_tasks = set(manifest["task_results"].keys())
        assert len(manifest_tasks) == 5, (
            f"Manifest must contain all 5 tasks, got {len(manifest_tasks)}"
        )

        for tid in manifest_tasks:
            assert tid in summary, (
                f"Task '{tid}' found in manifest but missing from run_summary.md — "
                f"_write_manifest() and _write_summary() must enumerate the same tasks"
            )


# ---------------------------------------------------------------------------
# Test 73: Cost propagation end-to-end across all artifacts
# ---------------------------------------------------------------------------


class TestCostPropagationEndToEnd:
    def test_task_costs_aggregate_through_manifest_and_summary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Verify that per-task cost_usd flows correctly from execute_task
        return values → PlanRunResult.total_cost_usd → manifest total_cost_usd
        → summary text.  This is a cross-module integration test spanning
        runners (mock) → scheduler → manifest serialization → summary writing.

        Uses 3 tasks with known costs: $1.50, $2.50, $0.00 (shell-like).
        Expected total: $4.00.

        Artifacts checked:
        - PlanRunResult.total_cost_usd == 4.00
        - manifest total_cost_usd == 4.00
        - Each task's cost_usd in manifest matches mock
        - Summary contains '$4.00' cost string"""
        cost_table = {"cp-engine": 1.50, "cp-review": 2.50, "cp-shell": None}

        def mock_cost_execute(
            plan: Any, task: Any, run_path: Path,
            dry_run: bool = False, execution_profile: str = "plan",
            upstream_results: Any = None, context_synthesis: str = "",
            workspace_brief: str = "", **kwargs: Any,
        ) -> TaskResult:
            now = datetime.now(UTC)
            cost = cost_table.get(task.id)
            token_usage = TokenUsage(input_tokens=100, output_tokens=50) if cost else None
            r = TaskResult(
                task_id=task.id, status="success", exit_code=0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="ok", cost_usd=cost, token_usage=token_usage,
            )
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_cost_execute)

        plan = _make_plan(
            [
                _make_task("cp-engine"),
                _make_task("cp-review", depends_on=["cp-engine"]),
                _make_task("cp-shell"),
            ],
            max_parallel=2,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # PlanRunResult must aggregate costs
        assert result.total_cost_usd == pytest.approx(4.00, abs=0.01), (
            f"total_cost_usd must be ~$4.00, got {result.total_cost_usd!r}"
        )

        # Manifest must reflect the same total
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert manifest["total_cost_usd"] == pytest.approx(4.00, abs=0.01), (
            f"manifest total_cost_usd must be ~$4.00, got {manifest['total_cost_usd']!r}"
        )

        # Per-task costs in manifest must match mocked values
        assert manifest["task_results"]["cp-engine"]["cost_usd"] == pytest.approx(1.50)
        assert manifest["task_results"]["cp-review"]["cost_usd"] == pytest.approx(2.50)
        assert manifest["task_results"]["cp-shell"]["cost_usd"] is None

        # Summary must contain the total cost
        summary = (result.run_path / "run_summary.md").read_text(encoding="utf-8")
        assert "$4.00" in summary, (
            "run_summary.md must contain '$4.00' reflecting the aggregated cost"
        )


# ---------------------------------------------------------------------------
# Test 74: Manifest success field consistency with events and PlanRunResult
# ---------------------------------------------------------------------------


class TestManifestSuccessConsistency:
    def test_manifest_success_matches_result_and_run_complete_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The 'success' field must be identical across three sources:
        PlanRunResult.success, manifest['success'], and run_complete event['success'].

        Tests both the success=True and success=False paths to ensure the
        cross-module contract is upheld in both directions.

        Artifacts checked:
        - PlanRunResult.success
        - run_manifest.json success field
        - run_complete event success field
        All three must agree."""
        holder: list[Path] = []

        # --- Case 1: all tasks succeed → success=True ---
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan_ok = _make_plan(
            [_make_task("sc-a"), _make_task("sc-b")],
            source_path=tmp_path / "plan.yaml",
        )
        result_ok = run_plan(plan_ok, run_dir_override=str(tmp_path / "runs-ok"))

        manifest_ok = json.loads(
            (result_ok.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        events_ok = _parse_jsonl(result_ok.run_path / "events.jsonl")
        rc_ok = next(e for e in events_ok if e["event"] == "run_complete")

        assert result_ok.success is True
        assert manifest_ok["success"] is True
        assert rc_ok["success"] is True, (
            "run_complete event 'success' must be True when all tasks succeed"
        )

        # --- Case 2: one task fails → success=False ---
        holder.clear()

        def mock_fail(
            plan: Any, task: Any, run_path: Path,
            dry_run: bool = False, execution_profile: str = "plan",
            upstream_results: Any = None, context_synthesis: str = "",
            workspace_brief: str = "", **kwargs: Any,
        ) -> TaskResult:
            if not holder:
                holder.append(run_path)
            now = datetime.now(UTC)
            status = "failed" if task.id == "sc-fail" else "success"
            r = TaskResult(
                task_id=task.id, status=status,  # type: ignore[arg-type]
                exit_code=1 if status == "failed" else 0,
                started_at=now, finished_at=now, duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message="fail" if status == "failed" else "ok",
            )
            r.log_path.write_text(f"status={status}\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fail)
        plan_fail = _make_plan(
            [_make_task("sc-ok"), _make_task("sc-fail")],
            fail_fast=False,
            source_path=tmp_path / "plan.yaml",
        )
        result_fail = run_plan(plan_fail, run_dir_override=str(tmp_path / "runs-fail"))

        manifest_fail = json.loads(
            (result_fail.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        events_fail = _parse_jsonl(result_fail.run_path / "events.jsonl")
        rc_fail = next(e for e in events_fail if e["event"] == "run_complete")

        assert result_fail.success is False
        assert manifest_fail["success"] is False
        assert rc_fail["success"] is False, (
            "run_complete event 'success' must be False when a task fails"
        )


# ---------------------------------------------------------------------------
# Test 75: sequential_duration_sec and parallelism_savings_pct in manifest
# ---------------------------------------------------------------------------


class TestManifestParallelismFields:
    def test_manifest_contains_sequential_duration_and_parallelism_savings(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """run_manifest.json must contain 'sequential_duration_sec' and
        'parallelism_savings_pct' keys (from PlanRunResult.to_dict()), both
        as non-negative floats.

        These fields are emitted by the scheduler to allow callers to estimate
        how much wall-clock time parallelism saved vs a hypothetical serial run.
        Downstream tools (maestro suggest, maestro diff) may read them.

        This validates that _write_manifest() faithfully serialises all fields
        from PlanRunResult.to_dict() including the parallelism stats."""
        holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        plan = _make_plan(
            [
                _make_task("ps-a"),
                _make_task("ps-b", depends_on=["ps-a"]),
                _make_task("ps-c", depends_on=["ps-a"]),
                _make_task("ps-d", depends_on=["ps-b", "ps-c"]),
            ],
            fail_fast=False,
            max_parallel=4,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True

        # PlanRunResult must have both fields as non-negative numbers
        assert isinstance(result.sequential_duration_sec, float), (
            "PlanRunResult.sequential_duration_sec must be a float"
        )
        assert result.sequential_duration_sec >= 0.0, (
            f"sequential_duration_sec must be >= 0.0, got {result.sequential_duration_sec}"
        )
        assert isinstance(result.parallelism_savings_pct, float), (
            "PlanRunResult.parallelism_savings_pct must be a float"
        )
        assert result.parallelism_savings_pct >= 0.0, (
            f"parallelism_savings_pct must be >= 0.0, got {result.parallelism_savings_pct}"
        )

        # Manifest must contain both keys with matching values
        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert "sequential_duration_sec" in manifest, (
            "run_manifest.json must contain 'sequential_duration_sec'"
        )
        assert "parallelism_savings_pct" in manifest, (
            "run_manifest.json must contain 'parallelism_savings_pct'"
        )

        assert isinstance(manifest["sequential_duration_sec"], (int, float)), (
            "manifest 'sequential_duration_sec' must be numeric"
        )
        assert isinstance(manifest["parallelism_savings_pct"], (int, float)), (
            "manifest 'parallelism_savings_pct' must be numeric"
        )
        assert manifest["sequential_duration_sec"] >= 0.0, (
            "manifest 'sequential_duration_sec' must be non-negative"
        )
        assert manifest["parallelism_savings_pct"] >= 0.0, (
            "manifest 'parallelism_savings_pct' must be non-negative"
        )

        # Values must match the PlanRunResult object
        assert abs(manifest["sequential_duration_sec"] - result.sequential_duration_sec) < 0.001, (
            "manifest sequential_duration_sec must match PlanRunResult value"
        )
        assert abs(manifest["parallelism_savings_pct"] - result.parallelism_savings_pct) < 0.001, (
            "manifest parallelism_savings_pct must match PlanRunResult value"
        )


# ---------------------------------------------------------------------------
# Test 76: extra_template_vars parameter does not break execution
# ---------------------------------------------------------------------------


class TestExtraTemplateVars:
    def test_extra_template_vars_passed_to_run_plan_does_not_break_execution(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When run_plan() is called with extra_template_vars={'watch.iteration': '5',
        'watch.best_metric': '0.95'}, the run must complete successfully and all
        canonical artifacts must be written correctly.

        extra_template_vars is used by 'maestro watch' to inject iteration-specific
        context into engine task prompts.  A regression in the parameter threading
        from run_plan() → execute_task() → _load_prompt() / build_command() would
        silently break watch loop execution.

        This test validates that the parameter is accepted, forwarded without error,
        and does not corrupt any artifact output."""
        received_kwargs: list[dict[str, Any]] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                received_kwargs.append(dict(kwargs))
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        extra: dict[str, str] = {
            "watch.iteration": "5",
            "watch.best_metric": "0.95",
            "watch.history": "iter 1: 0.80\niter 2: 0.90",
        }

        plan = _make_plan(
            [_make_task("etv-a"), _make_task("etv-b", depends_on=["etv-a"])],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            extra_template_vars=extra,
            run_dir_override=str(tmp_path / "runs"),
        )

        # Run must complete successfully
        assert result.success is True, (
            "Run with extra_template_vars must succeed"
        )

        # Both tasks must appear in task_results with 'success'
        assert set(result.task_results.keys()) == {"etv-a", "etv-b"}, (
            "Both tasks must appear in task_results"
        )
        for tid in ("etv-a", "etv-b"):
            assert result.task_results[tid].status == "success"

        # All three canonical artifacts must be present
        for artifact in ("run_manifest.json", "run_summary.md", "events.jsonl"):
            assert (result.run_path / artifact).exists(), (
                f"'{artifact}' must exist after run with extra_template_vars"
            )

        # Hash chain must still be valid
        records = replay_events(result.run_path / "events.jsonl")
        status = verify_chain(records)
        assert status == "valid", (
            f"Hash chain must be valid even when extra_template_vars is passed, "
            f"got '{status}'"
        )


# ---------------------------------------------------------------------------
# Test 77: auto_approve=True bypasses requires_approval gate and emits events
# ---------------------------------------------------------------------------


class TestAutoApprove:
    def test_auto_approve_bypasses_requires_approval_and_task_executes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When run_plan() is called with auto_approve=True, tasks with
        requires_approval=True must execute normally (not be skipped), and
        both 'approval_required' and 'approval_response' events must be
        emitted in events.jsonl.

        This validates the auto_approve fast-path in scheduler.py:
            if auto_approve:
                _emit('approval_response', task_id=task_id, approved=True)
                # ... continue to execute the task
        """
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        from maestro_cli.models import TaskSpec as _TaskSpec

        gated_task = _TaskSpec(
            id="gated",
            description="task requiring approval",
            depends_on=[],
            command="echo gated",
            requires_approval=True,
            approval_message="Please approve this task",
        )
        free_task = _TaskSpec(
            id="free",
            description="task without approval gate",
            depends_on=[],
            command="echo free",
        )

        plan = _make_plan(
            [gated_task, free_task],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            auto_approve=True,
            run_dir_override=str(tmp_path / "runs"),
        )

        # Plan must succeed — auto_approve bypasses the gate
        assert result.success is True, (
            "Plan with auto_approve=True must succeed even when tasks require approval"
        )

        # The gated task must have been executed (not skipped)
        assert "gated" in executed, (
            "Task with requires_approval=True must execute when auto_approve=True"
        )
        assert result.task_results["gated"].status == "success", (
            "gated task must be 'success' when auto_approve=True"
        )

        # events.jsonl must contain approval_required and approval_response for gated task
        events = _parse_jsonl(result.run_path / "events.jsonl")
        approval_required_events = [
            e for e in events
            if e.get("event") == "approval_required" and e.get("task_id") == "gated"
        ]
        approval_response_events = [
            e for e in events
            if e.get("event") == "approval_response" and e.get("task_id") == "gated"
        ]

        assert approval_required_events, (
            "approval_required event must be emitted for 'gated' task with auto_approve=True"
        )
        assert approval_response_events, (
            "approval_response event must be emitted for 'gated' task with auto_approve=True"
        )

        # The approval_response must show approved=True
        response = approval_response_events[0]
        assert "approved" in response, (
            "approval_response event must carry an 'approved' field"
        )
        assert response["approved"] is True, (
            f"approval_response 'approved' must be True with auto_approve=True, "
            f"got {response['approved']!r}"
        )


# ---------------------------------------------------------------------------
# Test 78: approval_handler deny — task skipped and approval_response emitted
# ---------------------------------------------------------------------------


class TestApprovalHandlerDeny:
    def test_approval_handler_deny_skips_task_and_emits_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When run_plan() is called with an approval_handler that returns False,
        a task with requires_approval=True must be skipped (not executed), and
        an 'approval_response' event with approved=False must appear in events.jsonl.
        The overall plan must fail because the skipped task counts as a failed
        dependency for its dependents.

        This validates the approval_handler=False path in scheduler.py:
            approved = approval_handler(task_id, task.approval_message)
            if not approved:
                result = _new_skipped_result(task_id, run_path, 'Approval denied by handler')

        Artifacts checked:
        - 'gated-deny' must NOT be in executed list
        - 'gated-deny' must have status 'skipped' in task_results
        - approval_response event has approved=False"""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        from maestro_cli.models import TaskSpec as _TaskSpec

        gated_task = _TaskSpec(
            id="gated-deny",
            description="task that approval_handler will deny",
            depends_on=[],
            command="echo should-not-run",
            requires_approval=True,
            approval_message="Deny me",
        )

        plan = _make_plan(
            [gated_task, _make_task("other-task")],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )
        result = run_plan(
            plan,
            approval_handler=lambda task_id, msg: False,  # always deny
            run_dir_override=str(tmp_path / "runs"),
        )

        # gated-deny must NOT have been passed to execute_task
        assert "gated-deny" not in executed, (
            "gated-deny must not be executed when approval_handler returns False"
        )

        # gated-deny must appear in task_results as 'skipped'
        assert "gated-deny" in result.task_results, (
            "gated-deny must appear in task_results even when denied"
        )
        assert result.task_results["gated-deny"].status == "skipped", (
            f"gated-deny must be 'skipped' when approval_handler denies it, "
            f"got {result.task_results['gated-deny'].status!r}"
        )

        # approval_response event must carry approved=False
        events = _parse_jsonl(result.run_path / "events.jsonl")
        response_events = [
            e for e in events
            if e.get("event") == "approval_response" and e.get("task_id") == "gated-deny"
        ]
        assert response_events, (
            "approval_response event must be emitted for 'gated-deny' when handler denies"
        )
        assert response_events[0].get("approved") is False, (
            f"approval_response 'approved' must be False when handler returns False, "
            f"got {response_events[0].get('approved')!r}"
        )

        # other-task must still have executed (fail_fast=False, gated-deny is skipped not failed)
        assert "other-task" in executed, (
            "other-task must still execute even when an unrelated task is denied"
        )


# ---------------------------------------------------------------------------
# Test 79: approval_handler approve — task executes and approval_response emitted
# ---------------------------------------------------------------------------


class TestApprovalHandlerApprove:
    def test_approval_handler_approve_executes_task_and_emits_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When run_plan() is called with an approval_handler that returns True,
        a task with requires_approval=True must execute normally (not be skipped),
        and an 'approval_response' event with approved=True must appear in events.jsonl.
        The overall plan must succeed.

        This validates the approval_handler=True path in scheduler.py and
        complements TestApprovalHandlerDeny by exercising the approval branch.

        Artifacts checked:
        - 'gated-approve' must be in executed list
        - 'gated-approve' must have status 'success' in task_results
        - approval_required event emitted before approval_response
        - approval_response event has approved=True
        - Overall plan success=True"""
        executed: list[str] = []
        lock = threading.Lock()

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            with lock:
                executed.append(task.id)
            now = datetime.now(UTC)
            r = TaskResult(
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
            r.log_path.write_text("status=success\n", encoding="utf-8")
            r.result_path.write_text(json.dumps(r.to_dict(), indent=2), encoding="utf-8")
            return r

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        from maestro_cli.models import TaskSpec as _TaskSpec

        gated_task = _TaskSpec(
            id="gated-approve",
            description="task that approval_handler will approve",
            depends_on=[],
            command="echo approved",
            requires_approval=True,
            approval_message="Please approve — handler will say yes",
        )

        plan = _make_plan(
            [gated_task],
            fail_fast=False,
            max_parallel=1,
            source_path=tmp_path / "plan.yaml",
        )

        # Track what the handler was called with
        handler_calls: list[tuple[str, str | None]] = []

        def approving_handler(task_id: str, msg: str | None) -> bool:
            handler_calls.append((task_id, msg))
            return True  # always approve

        result = run_plan(
            plan,
            approval_handler=approving_handler,
            run_dir_override=str(tmp_path / "runs"),
        )

        # Plan must succeed
        assert result.success is True, (
            "Plan must succeed when approval_handler approves the task"
        )

        # gated-approve must have been executed
        assert "gated-approve" in executed, (
            "gated-approve must be passed to execute_task when approval_handler returns True"
        )
        assert result.task_results["gated-approve"].status == "success", (
            f"gated-approve must have status 'success' when approved, "
            f"got {result.task_results['gated-approve'].status!r}"
        )

        # Handler must have been called with the correct task_id and message
        assert handler_calls, "approval_handler must have been called"
        called_task_id, called_msg = handler_calls[0]
        assert called_task_id == "gated-approve", (
            f"approval_handler was called with task_id={called_task_id!r}, "
            f"expected 'gated-approve'"
        )
        assert called_msg == "Please approve — handler will say yes", (
            f"approval_handler was called with msg={called_msg!r}, "
            f"expected the task's approval_message"
        )

        # events.jsonl must contain approval_required then approval_response (approved=True)
        events = _parse_jsonl(result.run_path / "events.jsonl")
        event_names = [e["event"] for e in events]

        assert "approval_required" in event_names, (
            "approval_required event must be emitted when task has requires_approval=True"
        )
        assert "approval_response" in event_names, (
            "approval_response event must be emitted after approval_handler is called"
        )

        # approval_required must precede approval_response
        req_idx = event_names.index("approval_required")
        resp_idx = event_names.index("approval_response")
        assert req_idx < resp_idx, (
            f"approval_required (idx={req_idx}) must appear before "
            f"approval_response (idx={resp_idx})"
        )

        # approval_response must carry approved=True
        resp_event = next(
            e for e in events if e.get("event") == "approval_response"
        )
        assert resp_event.get("approved") is True, (
            f"approval_response 'approved' must be True when handler returns True, "
            f"got {resp_event.get('approved')!r}"
        )


# ---------------------------------------------------------------------------
# Test 83: Policy block violation — task is failed, event emitted
# ---------------------------------------------------------------------------


class TestPolicyBlockViolation:
    def test_policy_block_prevents_execution_and_emits_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a plan has a blocking policy that matches a task, that task
        must NOT be passed to execute_task. Instead it must appear as 'failed'
        in the manifest with a message referencing the policy name, and a
        'policy_violation' event must be emitted with action='block'.

        This validates the policy enforcement path in scheduler.py:
            if plan.policies:
                violations = evaluate_policies(...)
                blocked = [v for v in violations if v.action == "block"]
                if blocked:
                    result = TaskResult(status="failed", ...)
        """
        from maestro_cli.models import PolicySpec

        executed: list[str] = []
        run_path_holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(run_path_holder, call_log=executed)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        # Policy: block any task whose task.id == "blocked-task"
        block_policy = PolicySpec(
            name="no-blocked-task",
            rule='task.id == "blocked-task"',
            action="block",
            message="This task is blocked by policy",
        )

        tasks = [
            _make_task("allowed-task"),
            _make_task("blocked-task"),
        ]
        plan = _make_plan(tasks, fail_fast=False, max_parallel=1)
        plan.policies = [block_policy]
        plan.source_path = tmp_path / "plan.yaml"

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # blocked-task must NOT have been passed to execute_task
        assert "blocked-task" not in executed, (
            "Task 'blocked-task' must not be passed to execute_task when "
            "a blocking policy matches"
        )

        # allowed-task must have been executed
        assert "allowed-task" in executed, (
            "Task 'allowed-task' must still execute when it does not match "
            "the blocking policy"
        )

        # blocked-task must be 'failed' in the manifest
        assert "blocked-task" in result.task_results, (
            "blocked-task must appear in task_results even when blocked by policy"
        )
        assert result.task_results["blocked-task"].status == "failed", (
            f"blocked-task must have status 'failed', got "
            f"{result.task_results['blocked-task'].status!r}"
        )
        assert "no-blocked-task" in result.task_results["blocked-task"].message, (
            "blocked-task failure message must reference the policy name "
            f"'no-blocked-task', got {result.task_results['blocked-task'].message!r}"
        )

        # events.jsonl must contain a policy_violation event
        events = _parse_jsonl(result.run_path / "events.jsonl")
        policy_events = [
            e for e in events
            if e.get("event") == "policy_violation"
            and e.get("task_id") == "blocked-task"
        ]
        assert policy_events, (
            "A 'policy_violation' event must be emitted for 'blocked-task'"
        )
        assert policy_events[0]["action"] == "block", (
            f"policy_violation action must be 'block', got {policy_events[0]['action']!r}"
        )
        assert policy_events[0]["policy_name"] == "no-blocked-task", (
            f"policy_violation policy_name must be 'no-blocked-task', "
            f"got {policy_events[0]['policy_name']!r}"
        )


# ---------------------------------------------------------------------------
# Test 84: Policy warn violation — task executes, event emitted
# ---------------------------------------------------------------------------


class TestPolicyWarnContinues:
    def test_policy_warn_emits_event_but_task_executes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When a plan has a warn-level policy that matches a task, the task
        must still be passed to execute_task and succeed. A 'policy_violation'
        event with action='warn' must appear in events.jsonl.

        This validates the warn path in scheduler.py:
            for v in violations:
                _emit("policy_violation", ...)
                if v.action == "warn": print warning
            if blocked: ... (no block here since action=warn)
        """
        from maestro_cli.models import PolicySpec

        executed: list[str] = []
        run_path_holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(run_path_holder, call_log=executed)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        warn_policy = PolicySpec(
            name="warn-on-notimeout",
            rule="task.timeout_sec == None",
            action="warn",
            message="Task has no timeout set",
        )

        tasks = [_make_task("warn-task")]
        plan = _make_plan(tasks, fail_fast=False, max_parallel=1)
        plan.policies = [warn_policy]
        plan.source_path = tmp_path / "plan.yaml"

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        # Plan must succeed — warn does not block
        assert result.success is True, (
            "Plan must succeed when only warn-level policies are violated"
        )

        # Task must have been executed
        assert "warn-task" in executed, (
            "Task must be passed to execute_task when policy action is 'warn'"
        )
        assert result.task_results["warn-task"].status == "success", (
            f"warn-task must be 'success', got "
            f"{result.task_results['warn-task'].status!r}"
        )

        # events.jsonl must contain a policy_violation with action=warn
        events = _parse_jsonl(result.run_path / "events.jsonl")
        warn_events = [
            e for e in events
            if e.get("event") == "policy_violation"
            and e.get("task_id") == "warn-task"
        ]
        assert warn_events, (
            "A 'policy_violation' event must be emitted for warn-level policy"
        )
        assert warn_events[0]["action"] == "warn", (
            f"policy_violation action must be 'warn', got {warn_events[0]['action']!r}"
        )


# ---------------------------------------------------------------------------
# Test 85: Manifest timing fields — started_at <= finished_at
# ---------------------------------------------------------------------------


class TestManifestTimingFieldsValid:
    def test_manifest_started_at_before_finished_at_for_all_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every task in run_manifest.json must have started_at <= finished_at
        (when both are present), and the plan-level started_at must be <=
        finished_at. This is a cross-module invariant: scheduler.py records
        wall-clock times, and the manifest serialization must preserve this.

        Also validates that ISO 8601 timestamps are parseable.
        """
        from datetime import datetime as dt

        run_path_holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(run_path_holder)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        tasks = [
            _make_task("first"),
            _make_task("second", depends_on=["first"]),
            _make_task("third", depends_on=["second"]),
        ]
        plan = _make_plan(tasks, max_parallel=1)
        plan.source_path = tmp_path / "plan.yaml"

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.success is True

        manifest = json.loads(
            (result.run_path / "run_manifest.json").read_text(encoding="utf-8")
        )

        # Plan-level timing
        plan_started = manifest.get("started_at")
        plan_finished = manifest.get("finished_at")
        assert plan_started is not None, "manifest must have 'started_at'"
        assert plan_finished is not None, "manifest must have 'finished_at'"

        # Parse ISO timestamps (support both Z and +00:00 suffixes)
        def _parse_ts(s: str) -> dt:
            s = s.replace("Z", "+00:00")
            return dt.fromisoformat(s)

        ps = _parse_ts(plan_started)
        pf = _parse_ts(plan_finished)
        assert pf >= ps, (
            f"Plan finished_at ({plan_finished}) must be >= started_at ({plan_started})"
        )

        # Per-task timing
        for task_id, task_data in manifest.get("task_results", {}).items():
            ts = task_data.get("started_at")
            tf = task_data.get("finished_at")
            if ts is not None and tf is not None:
                t_start = _parse_ts(ts)
                t_finish = _parse_ts(tf)
                assert t_finish >= t_start, (
                    f"Task '{task_id}' finished_at ({tf}) must be >= "
                    f"started_at ({ts})"
                )


# ---------------------------------------------------------------------------
# Test 86: Two independent chains both complete without interference
# ---------------------------------------------------------------------------


class TestIndependentChainsComplete:
    def test_two_independent_chains_both_complete(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two independent chains (A→B and C→D) must both complete fully.
        All four tasks must appear in the manifest with status 'success',
        and the event ordering must show task_start before task_complete
        for each task. The chains must not interfere with each other.
        """
        executed: list[str] = []
        run_path_holder: list[Path] = []
        mock_fn, _ = _make_mock_execute(run_path_holder, call_log=executed)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_fn)

        tasks = [
            _make_task("chain1-a"),
            _make_task("chain1-b", depends_on=["chain1-a"]),
            _make_task("chain2-c"),
            _make_task("chain2-d", depends_on=["chain2-c"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, max_parallel=4)
        plan.source_path = tmp_path / "plan.yaml"

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        assert result.success is True, "Plan with two independent chains must succeed"

        # All four tasks must be in the results
        for tid in ["chain1-a", "chain1-b", "chain2-c", "chain2-d"]:
            assert tid in result.task_results, (
                f"Task '{tid}' must appear in task_results"
            )
            assert result.task_results[tid].status == "success", (
                f"Task '{tid}' must be 'success', got "
                f"{result.task_results[tid].status!r}"
            )

        # All four must have been executed
        assert set(executed) == {"chain1-a", "chain1-b", "chain2-c", "chain2-d"}, (
            f"All four tasks must be executed, got {sorted(executed)}"
        )

        # Event ordering: task_start before task_complete per task
        events = _parse_jsonl(result.run_path / "events.jsonl")
        for tid in ["chain1-a", "chain1-b", "chain2-c", "chain2-d"]:
            starts = [
                i for i, e in enumerate(events)
                if e.get("event") == "task_start" and e.get("task_id") == tid
            ]
            completes = [
                i for i, e in enumerate(events)
                if e.get("event") == "task_complete" and e.get("task_id") == tid
            ]
            assert starts, f"task_start event missing for '{tid}'"
            assert completes, f"task_complete event missing for '{tid}'"
            assert starts[0] < completes[0], (
                f"task_start for '{tid}' (idx={starts[0]}) must come before "
                f"task_complete (idx={completes[0]})"
            )

        # Chain ordering: chain1-a completes before chain1-b starts
        a_complete = next(
            i for i, e in enumerate(events)
            if e.get("event") == "task_complete" and e.get("task_id") == "chain1-a"
        )
        b_start = next(
            i for i, e in enumerate(events)
            if e.get("event") == "task_start" and e.get("task_id") == "chain1-b"
        )
        assert a_complete < b_start, (
            f"chain1-a task_complete (idx={a_complete}) must precede "
            f"chain1-b task_start (idx={b_start})"
        )

        # Chain ordering: chain2-c completes before chain2-d starts
        c_complete = next(
            i for i, e in enumerate(events)
            if e.get("event") == "task_complete" and e.get("task_id") == "chain2-c"
        )
        d_start = next(
            i for i, e in enumerate(events)
            if e.get("event") == "task_start" and e.get("task_id") == "chain2-d"
        )
        assert c_complete < d_start, (
            f"chain2-c task_complete (idx={c_complete}) must precede "
            f"chain2-d task_start (idx={d_start})"
        )


# ---------------------------------------------------------------------------
# Test 87: run_complete event counters sum to total task count
# ---------------------------------------------------------------------------


class TestRunCompleteCountersSum:
    def test_run_complete_ok_failed_soft_skipped_sum_to_total(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The run_complete event emits ok, failed, soft_failed, and skipped
        counters. Their sum must equal the total number of tasks in the plan.
        This tests a mixed scenario: one success, one soft_failed, one hard
        failure with a dependent that gets skipped.
        """
        run_path_holder: list[Path] = []

        def mock_execute(
            plan: Any,
            task: Any,
            run_path: Path,
            dry_run: bool = False,
            execution_profile: str = "plan",
            upstream_results: Any = None,
            context_synthesis: str = "",
            workspace_brief: str = "",
            **kwargs: Any,
        ) -> TaskResult:
            if not run_path_holder:
                run_path_holder.append(run_path)
            now = datetime.now(UTC)

            if task.id == "ok-task":
                status, code, msg = "success", 0, "ok"
            elif task.id == "soft-task":
                status, code, msg = "soft_failed", 1, "soft fail"
            elif task.id == "hard-fail":
                status, code, msg = "failed", 1, "hard fail"
            else:
                status, code, msg = "success", 0, "ok"

            result = TaskResult(
                task_id=task.id,
                status=status,  # type: ignore[arg-type]
                exit_code=code,
                started_at=now,
                finished_at=now,
                duration_sec=0.01,
                command=f"echo {task.id}",
                log_path=run_path / f"{task.id}.log",
                result_path=run_path / f"{task.id}.result.json",
                message=msg,
            )
            result.log_path.write_text(f"status={status}\n", encoding="utf-8")
            result.result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_execute)

        tasks = [
            _make_task("ok-task"),
            _make_task("soft-task", allow_failure=True),
            _make_task("hard-fail"),
            _make_task("dep-of-hard", depends_on=["hard-fail"]),
        ]
        plan = _make_plan(tasks, fail_fast=False, max_parallel=1)
        plan.source_path = tmp_path / "plan.yaml"

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))

        events = _parse_jsonl(result.run_path / "events.jsonl")
        run_complete = next(
            e for e in events if e.get("event") == "run_complete"
        )

        ok = run_complete.get("ok", 0)
        failed = run_complete.get("failed", 0)
        soft_failed = run_complete.get("soft_failed", 0)
        skipped = run_complete.get("skipped", 0)
        total = ok + failed + soft_failed + skipped

        assert total == len(tasks), (
            f"run_complete counters must sum to {len(tasks)}: "
            f"ok={ok} + failed={failed} + soft_failed={soft_failed} + "
            f"skipped={skipped} = {total}"
        )

        # Verify individual counts match expected scenario
        assert ok >= 1, f"Expected at least 1 ok task, got {ok}"
        assert failed >= 1, f"Expected at least 1 failed task, got {failed}"
        assert soft_failed >= 1, f"Expected at least 1 soft_failed task, got {soft_failed}"
        assert skipped >= 1, f"Expected at least 1 skipped task, got {skipped}"
