from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

import pytest

from maestro_cli.models import (
    CircuitBreakerSpec,
    JudgeResult,
    JudgeSpec,
    MCPServerSpec,
    PlanDefaults,
    PlanSpec,
    TaskResult,
    TaskSpec,
)
from maestro_cli.scheduler import run_plan


# ---------------------------------------------------------------------------
# Local self-contained helpers (mirroring tests/test_scheduler.py conventions)
# ---------------------------------------------------------------------------

def _make_task(
    task_id: str,
    depends_on: list[str] | None = None,
    command: str | None = "echo ok",
    *,
    engine: str | None = None,
    prompt: str | None = None,
    allow_failure: bool = False,
    context_from: list[str] | None = None,
    **extra: object,
) -> TaskSpec:
    return TaskSpec(
        id=task_id,
        description=f"task {task_id}",
        depends_on=depends_on or [],
        command=command,
        engine=engine,
        prompt=prompt,
        allow_failure=allow_failure,
        context_from=context_from or [],
        **extra,  # type: ignore[arg-type]
    )


def _make_plan(
    tasks: list[TaskSpec],
    *,
    name: str = "test-plan",
    fail_fast: bool = True,
    max_parallel: int = 4,
    source_path: Path | None = None,
    **extra: object,
) -> PlanSpec:
    return PlanSpec(
        version=1,
        name=name,
        fail_fast=fail_fast,
        max_parallel=max_parallel,
        defaults=PlanDefaults(),
        tasks=tasks,
        source_path=source_path,
        **extra,  # type: ignore[arg-type]
    )


def _write_result_files(result: TaskResult) -> None:
    result.log_path.write_text(
        f"status={result.status}\nmessage={result.message}\n", encoding="utf-8"
    )
    result.result_path.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )


def _success(task_id: str, run_path: Path, **fields: object) -> TaskResult:
    now = datetime.now(UTC)
    result = TaskResult(
        task_id=task_id,
        status="success",
        exit_code=0,
        started_at=now,
        finished_at=now,
        duration_sec=0.01,
        command=f"echo {task_id}",
        log_path=run_path / f"{task_id}.log",
        result_path=run_path / f"{task_id}.result.json",
        message="ok",
        **fields,  # type: ignore[arg-type]
    )
    return result


def _make_exec(
    *,
    call_log: list[str] | None = None,
    overrides: dict[str, TaskResult] | None = None,
    capture_tasks: dict[str, TaskSpec] | None = None,
):
    """Build a mock execute_task that records calls and writes artefacts."""
    log = call_log if call_log is not None else []
    over = overrides or {}
    cap = capture_tasks if capture_tasks is not None else {}
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
            log.append(task.id)
            cap[task.id] = task
        if task.id in over:
            result = over[task.id]
            # Re-home log/result paths onto the active run_path.
            result.log_path = run_path / f"{task.id}.log"
            result.result_path = run_path / f"{task.id}.result.json"
            _write_result_files(result)
            return result
        result = _success(task.id, run_path)
        _write_result_files(result)
        return result

    return mock_execute, log


def _no_preflight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "maestro_cli.scheduler._preflight_checks", lambda *a, **kw: None
    )


# ===========================================================================
# 1795 — run directory cannot be resolved
# ===========================================================================

class TestRunRootResolution:
    def test_unresolvable_run_root_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = _make_plan([_make_task("a")], source_path=tmp_path / "plan.yaml")
        monkeypatch.setattr(
            "maestro_cli.scheduler.resolve_path", lambda *a, **kw: None
        )
        with pytest.raises(ValueError, match="Unable to resolve run directory"):
            run_plan(plan, run_dir_override=str(tmp_path / "runs"))


# ===========================================================================
# 1812-1813 / 1827-1828 — graceful degradation when history / knowledge load fails
# ===========================================================================

class TestStartupGracefulDegradation:
    def test_history_and_knowledge_load_exceptions_are_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = _make_plan([_make_task("a")], source_path=tmp_path / "plan.yaml")

        def _boom(*a: object, **kw: object):
            raise RuntimeError("history blew up")

        # These imports happen inside run_plan via `from .routing import ...`
        # and `from .knowledge import ...`, so patch the source modules.
        monkeypatch.setattr("maestro_cli.routing.load_task_histories", _boom)
        monkeypatch.setattr("maestro_cli.knowledge.load_knowledge", _boom)

        mock_exec, log = _make_exec()
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.success is True
        assert log == ["a"]


# ===========================================================================
# 1858-1859 — cross-run budget gate ALLOWED + verbose budget line
# ===========================================================================

class TestBudgetGateAllowed:
    def test_budget_allowed_logs_remaining(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = _make_plan(
            [_make_task("a")],
            source_path=tmp_path / "plan.yaml",
            budget_period="daily",
            max_cost_usd=10.0,
        )
        # Local import: `from .budget import check_budget, _DEFAULT_LEDGER_PATH`
        monkeypatch.setattr(
            "maestro_cli.budget.check_budget",
            lambda *a, **kw: (True, 2.5, 7.5),
        )
        mock_exec, log = _make_exec()
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        result = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
            verbosity="normal",
        )
        assert result.success is True
        out = capsys.readouterr().out
        assert "budget" in out
        assert "remaining" in out

    def test_budget_exceeded_returns_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = _make_plan(
            [_make_task("a")],
            source_path=tmp_path / "plan.yaml",
            budget_period="daily",
            max_cost_usd=10.0,
        )
        monkeypatch.setattr(
            "maestro_cli.budget.check_budget",
            lambda *a, **kw: (False, 12.0, 0.0),
        )
        mock_exec, log = _make_exec()
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.success is False
        # Budget gate short-circuits before any task runs.
        assert log == []


# ===========================================================================
# 1894 / 1899-1901 — DAG wave precomputation: deferred dep + cycle fallback
# ===========================================================================

class TestWavePrecomputation:
    def test_deferred_dependency_wave_assignment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Order tasks so a dependent appears before its dependency: this forces
        # the "deps not yet assigned" continue branch in the wave loop.
        tasks = [
            _make_task("c", depends_on=["b"]),
            _make_task("b", depends_on=["a"]),
            _make_task("a"),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "plan.yaml", max_parallel=1)
        events: list[tuple[str, dict]] = []
        mock_exec, log = _make_exec()
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        def cb(name: str, data: dict) -> None:
            events.append((name, data))

        result = run_plan(
            plan, run_dir_override=str(tmp_path / "runs"), event_callback=cb
        )
        assert result.success is True
        # Wave metadata reflects the linear chain.
        starts = {e[1]["task_id"]: e[1]["wave"] for e in events if e[0] == "task_start"}
        assert starts["a"] == 0
        assert starts["b"] == 1
        assert starts["c"] == 2

    def test_cyclic_plan_falls_back_to_wave_zero(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # run_plan does not re-validate cycles; a self-referential cycle exercises
        # the "no progress -> assign wave 0" fallback. The tasks will be skipped
        # because their dependencies never complete, so the run is not blocked.
        tasks = [
            _make_task("x", depends_on=["y"]),
            _make_task("y", depends_on=["x"]),
        ]
        plan = _make_plan(
            tasks, source_path=tmp_path / "plan.yaml", fail_fast=False
        )
        mock_exec, log = _make_exec()
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        # Neither task can ever become ready (mutual dependency); nothing executes.
        # The wave precomputation still terminates via the no-progress fallback.
        assert log == []
        assert result.run_id


# ===========================================================================
# 2021 — _handle_dependents skips a dependent already removed from pending
# ===========================================================================

class TestHandleDependentsAlreadyGone:
    def test_dependent_skipped_then_other_dep_completes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = datetime.now(UTC)
        # c depends on [a, b]. With max_parallel=1, 'a' runs first and FAILS,
        # which skips 'c' (removed from pending). Then 'b' succeeds and its
        # _handle_dependents iterates 'c' -> not in pending -> continue.
        tasks = [
            _make_task("a"),
            _make_task("b"),
            _make_task("c", depends_on=["a", "b"]),
        ]
        plan = _make_plan(
            tasks, source_path=tmp_path / "plan.yaml",
            fail_fast=False, max_parallel=1,
        )
        overrides = {
            "a": TaskResult(
                task_id="a", status="failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="fail", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="boom",
            ),
        }
        mock_exec, log = _make_exec(overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.task_results["a"].status == "failed"
        assert result.task_results["b"].status == "success"
        assert result.task_results["c"].status == "skipped"
        # 'c' was never dispatched.
        assert "c" not in log


# ===========================================================================
# 2141-2142 — circuit breaker pause + approval_handler raises -> denied
# ===========================================================================

class TestCircuitBreakerPause:
    def test_pause_action_handler_exception_treated_as_denial(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = datetime.now(UTC)
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(
            tasks, source_path=tmp_path / "plan.yaml",
            fail_fast=False, max_parallel=1,
            circuit_breaker=CircuitBreakerSpec(max_total_failures=1, action="pause"),
        )
        overrides = {
            "a": TaskResult(
                task_id="a", status="failed", exit_code=1,
                started_at=now, finished_at=now, duration_sec=0.01,
                command="fail", log_path=tmp_path / "a.log",
                result_path=tmp_path / "a.result.json", message="boom",
            ),
        }
        mock_exec, log = _make_exec(overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        def raising_handler(_kind: str, _msg: str | None) -> bool:
            raise RuntimeError("approval boom")

        events: list[tuple[str, dict]] = []

        result = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
            approval_handler=raising_handler,
            event_callback=lambda n, d: events.append((n, d)),
        )
        # Breaker tripped (pause denied) -> fail_fast -> b skipped.
        assert result.task_results["a"].status == "failed"
        assert result.task_results["b"].status == "skipped"
        assert any(e[0] == "circuit_breaker_tripped" for e in events)


# ===========================================================================
# 2222 / 2234 — MCP exclusive worktree slot dispatch arbitration
# ===========================================================================

class TestMcpExclusiveSlot:
    def test_mcp_exclusive_tasks_serialize_dispatch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two MCP-exclusive worktree tasks plus a normal task, all dependency-free.
        # With max_parallel=2 the second MCP task is blocked while the first runs,
        # exercising the requeue + "continue" arbitration branch (2234).
        server = MCPServerSpec(name="db", command=["db"], is_concurrency_safe=False)
        tasks = [
            _make_task(
                "m1", command=None, engine="claude", prompt="one",
                worktree=True, mcp_tools=["db"],
            ),
            _make_task(
                "m2", command=None, engine="claude", prompt="two",
                worktree=True, mcp_tools=["db"],
            ),
            _make_task("n1"),
        ]
        plan = _make_plan(
            tasks, source_path=tmp_path / "plan.yaml",
            max_parallel=2, fail_fast=False,
        )
        plan.mcp_servers = [server]

        # Make execute_task slow enough that the first MCP task stays "running"
        # while the scheduler arbitrates the rest of the ready queue.
        barrier = threading.Event()

        def slow_exec(
            plan, task, run_path, dry_run=False, execution_profile="plan",
            upstream_results=None, context_synthesis="", workspace_brief="",
            **kwargs,
        ):
            if task.id == "m1":
                # Hold the slot briefly so m2 is observed as blocked.
                barrier.wait(timeout=2.0)
            result = _success(task.id, run_path)
            _write_result_files(result)
            return result

        # Release the barrier shortly after dispatch begins.
        def _release() -> None:
            barrier.set()

        timer = threading.Timer(0.25, _release)
        timer.start()
        try:
            monkeypatch.setattr("maestro_cli.scheduler.execute_task", slow_exec)
            _no_preflight(monkeypatch)
            result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        finally:
            timer.cancel()
            barrier.set()

        assert result.success is True
        assert {"m1", "m2", "n1"} <= set(result.task_results)


# ===========================================================================
# 2366 / 2370-2372 — cache-miss hash computed when lookup left it unset
# ===========================================================================

class TestCacheMissHashCompute:
    def test_hash_recomputed_on_cache_miss(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On the lookup pass the hash compute fails (returns None), so the task
        # hash is not stored; the dedicated cache-miss block then recomputes it.
        calls: dict[str, int] = {}

        def staged_hash(task, plan, upstream_hashes):
            n = calls.get(task.id, 0)
            calls[task.id] = n + 1
            if n == 0:
                return None  # lookup pass: leave task_hashes unset
            return f"hash-{task.id}"  # miss pass: succeeds

        monkeypatch.setattr(
            "maestro_cli.scheduler._compute_task_hash_safe", staged_hash
        )
        # cache_lookup is only consulted when the lookup hash is non-None; here it
        # never is, so the run is a clean miss. Guard against accidental hits.
        monkeypatch.setattr(
            "maestro_cli.scheduler.cache_lookup", lambda *a, **kw: None
        )
        stored: list[str] = []
        monkeypatch.setattr(
            "maestro_cli.scheduler.cache_store",
            lambda cache_dir, h, result, task=None: stored.append(h),
        )

        plan = _make_plan([_make_task("a")], source_path=tmp_path / "plan.yaml")
        mock_exec, log = _make_exec()
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        cache_dir = tmp_path / "cache"
        cache_dir.mkdir()
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
            cache_dir=cache_dir,
        )
        assert result.success is True
        # Hash was computed twice (lookup miss + dedicated miss block) and the
        # recomputed value drove the post-run cache_store.
        assert calls["a"] >= 2
        assert "hash-a" in stored


# ===========================================================================
# 2402 — hop decay applied when a context_from upstream is 2+ hops away
# ===========================================================================

class TestHopDecay:
    def test_transitive_context_triggers_hop_decay(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # c pulls context from 'a' which is two hops away (a -> b -> c). Because
        # 'c' only directly depends on 'b', the hop distance to 'a' is 2 and the
        # hop-decay branch fires.
        tasks = [
            _make_task("a", command="echo a"),
            _make_task("b", depends_on=["a"], command="echo b"),
            _make_task(
                "c",
                depends_on=["b"],
                engine="claude",
                prompt="use upstream",
                command=None,
                context_from=["a"],
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "plan.yaml", max_parallel=1)
        now = datetime.now(UTC)
        overrides = {
            "a": _success("a", tmp_path, stdout_tail="alpha output line"),
        }
        mock_exec, log = _make_exec(overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.success is True
        assert log[-1] == "c"


# ===========================================================================
# 2426 / 2436-2441 — context_allowlist filtering + output_redact rewriting
# ===========================================================================

class TestContextPrivacy:
    def test_allowlist_and_redact_applied_to_upstream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen_upstream: dict[str, TaskResult] = {}

        producer = _make_task(
            "prod",
            command="echo prod",
            output_redact=[r"secret-\w+"],
        )
        consumer = _make_task(
            "cons",
            depends_on=["prod"],
            engine="claude",
            prompt="read it",
            command=None,
            context_from=["prod"],
            context_allowlist=["stdout_tail", "status"],
        )
        plan = _make_plan(
            [producer, consumer], source_path=tmp_path / "plan.yaml", max_parallel=1
        )
        overrides = {
            "prod": _success(
                "prod", tmp_path,
                stdout_tail="value=secret-abc123 and other text",
            ),
        }

        def capturing_exec(
            plan, task, run_path, dry_run=False, execution_profile="plan",
            upstream_results=None, context_synthesis="", workspace_brief="",
            **kwargs,
        ):
            if task.id == "cons" and upstream_results:
                seen_upstream.update(upstream_results)
            if task.id in overrides:
                result = overrides[task.id]
                result.log_path = run_path / f"{task.id}.log"
                result.result_path = run_path / f"{task.id}.result.json"
                _write_result_files(result)
                return result
            result = _success(task.id, run_path)
            _write_result_files(result)
            return result

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_exec)
        _no_preflight(monkeypatch)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.success is True
        assert "prod" in seen_upstream
        # output_redact stripped the secret before it reached the consumer.
        assert "secret-abc123" not in seen_upstream["prod"].stdout_tail
        assert "REDACTED" in seen_upstream["prod"].stdout_tail


# ===========================================================================
# 2643 — council context: upstream stdout_tail appended to council prompt
# ===========================================================================

class TestCouncilUpstream:
    def test_council_receives_upstream_context(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli.council import (
            CouncilParticipant,
            CouncilResult,
            CouncilSpec,
        )

        council = CouncilSpec(
            participants=[CouncilParticipant(engine="claude", model="haiku")],
            rounds=1,
            topology="star",
        )
        tasks = [
            _make_task("up", command="echo up"),
            _make_task(
                "panel",
                depends_on=["up"],
                engine="claude",
                prompt="deliberate",
                command=None,
                context_from=["up"],
                context_mode="council",
                council=council,
            ),
        ]
        plan = _make_plan(tasks, source_path=tmp_path / "plan.yaml", max_parallel=1)
        overrides = {
            "up": _success("up", tmp_path, stdout_tail="important upstream finding"),
        }

        captured_upstream: dict[str, str] = {}

        def fake_run_council(spec, prompt, workdir, upstream_context="", **kw):
            captured_upstream["ctx"] = upstream_context
            return CouncilResult(rounds=[], synthesis="agreed", total_cost_usd=0.0)

        monkeypatch.setattr("maestro_cli.council.run_council", fake_run_council)
        mock_exec, log = _make_exec(overrides=overrides)
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.success is True
        assert "important upstream finding" in captured_upstream["ctx"]
        assert "--- up ---" in captured_upstream["ctx"]


# ===========================================================================
# 2816-2820 — standard context compaction shortens upstream output
# ===========================================================================

class TestStandardCompaction:
    def test_standard_compaction_collapses_duplicate_lines(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen_upstream: dict[str, TaskResult] = {}

        # Repeated identical [maestro] lines are collapsed by _compact_context,
        # guaranteeing the compacted text is shorter than the original.
        noisy = "".join(["[maestro] doing work\n"] * 12)
        producer = _make_task("prod", command="echo prod")
        consumer = _make_task(
            "cons",
            depends_on=["prod"],
            engine="claude",
            prompt="summarize",
            command=None,
            context_from=["prod"],
            context_compaction="standard",
        )
        plan = _make_plan(
            [producer, consumer], source_path=tmp_path / "plan.yaml", max_parallel=1
        )
        overrides = {"prod": _success("prod", tmp_path, stdout_tail=noisy)}

        def capturing_exec(
            plan, task, run_path, dry_run=False, execution_profile="plan",
            upstream_results=None, context_synthesis="", workspace_brief="",
            **kwargs,
        ):
            if task.id == "cons" and upstream_results:
                seen_upstream.update(upstream_results)
            if task.id in overrides:
                result = overrides[task.id]
                result.log_path = run_path / f"{task.id}.log"
                result.result_path = run_path / f"{task.id}.result.json"
                _write_result_files(result)
                return result
            result = _success(task.id, run_path)
            _write_result_files(result)
            return result

        events: list[tuple[str, dict]] = []
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", capturing_exec)
        _no_preflight(monkeypatch)

        result = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert result.success is True
        assert "prod" in seen_upstream
        assert len(seen_upstream["prod"].stdout_tail) < len(noisy)
        compaction_events = [e for e in events if e[0] == "context_compaction"]
        assert any(e[1].get("mode") == "standard" for e in compaction_events)


# ===========================================================================
# 3038-3047 — workspace-aware timeout adjustment for large referenced files
# ===========================================================================

class TestWorkspaceTimeoutAdjustment:
    def test_large_file_bumps_timeout(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws = tmp_path / "ws"
        (ws / "src").mkdir(parents=True)
        big = ws / "src" / "huge.py"
        big.write_text("# pad\n" + ("x" * 40_000), encoding="utf-8")

        task = _make_task(
            "edit",
            command=None,
            engine="claude",
            prompt="please edit src/huge.py carefully",
            timeout_sec=10,  # low enough that adjusted > current
        )
        plan = _make_plan(
            [task], source_path=tmp_path / "plan.yaml",
            workspace_root=str(ws),
        )
        events: list[tuple[str, dict]] = []
        mock_exec, log = _make_exec()
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        result = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert result.success is True
        adj = [e for e in events if e[0] == "timeout_adjusted"]
        assert len(adj) == 1
        assert adj[0][1]["task_id"] == "edit"
        assert adj[0][1]["adjusted_timeout_sec"] > 10


# ===========================================================================
# 3136 / 3144 — judge_result and checkpoint supplementary events
# ===========================================================================

class TestSupplementaryEvents:
    def test_judge_result_and_checkpoint_events_emitted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = datetime.now(UTC)
        judged = TaskResult(
            task_id="t", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=0.02,
            command="echo t", log_path=tmp_path / "t.log",
            result_path=tmp_path / "t.result.json", message="ok",
            judge_result=JudgeResult(verdict="pass", overall_score=0.91),
            checkpoint_count=3,
        )
        task = _make_task(
            "t", command=None, engine="claude", prompt="do",
            judge=JudgeSpec(criteria=["is good"], pass_threshold=0.5, on_fail="warn"),
        )
        plan = _make_plan([task], source_path=tmp_path / "plan.yaml")
        mock_exec, log = _make_exec(overrides={"t": judged})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert result.success is True
        judge_events = [e for e in events if e[0] == "judge_result"]
        ckpt_events = [e for e in events if e[0] == "task_checkpoint"]
        assert len(judge_events) == 1
        assert judge_events[0][1]["verdict"] == "pass"
        assert judge_events[0][1]["on_fail"] == "warn"
        assert len(ckpt_events) == 1
        assert ckpt_events[0][1]["count"] == 3


# ===========================================================================
# 3247-3261 — KeyboardInterrupt during the dispatch loop
# ===========================================================================

class TestKeyboardInterrupt:
    def test_ctrl_c_skips_pending_and_kills_active(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        tasks = [
            _make_task("a"),
            _make_task("b", depends_on=["a"]),
        ]
        plan = _make_plan(
            tasks, source_path=tmp_path / "plan.yaml", max_parallel=1
        )

        killed: list[bool] = []
        monkeypatch.setattr(
            "maestro_cli.scheduler.kill_all_active",
            lambda: killed.append(True),
        )

        def interrupting_exec(
            plan, task, run_path, dry_run=False, execution_profile="plan",
            upstream_results=None, context_synthesis="", workspace_brief="",
            **kwargs,
        ):
            raise KeyboardInterrupt

        monkeypatch.setattr("maestro_cli.scheduler.execute_task", interrupting_exec)
        _no_preflight(monkeypatch)

        events: list[tuple[str, dict]] = []
        result = run_plan(
            plan,
            run_dir_override=str(tmp_path / "runs"),
            event_callback=lambda n, d: events.append((n, d)),
        )
        assert killed == [True]
        # Pending tasks were marked skipped/interrupted.
        skip_events = [
            e for e in events
            if e[0] == "task_skip" and "interrupt" in str(e[1].get("reason", ""))
        ]
        assert skip_events
        # The interrupted task 'a' produced no result; 'b' is recorded as skipped.
        assert result.task_results["b"].status == "skipped"


# ===========================================================================
# 3382-3383 — summary_url as_uri() raises ValueError on a relative path
# ===========================================================================

class TestWebhookSummaryUri:
    def test_relative_summary_path_yields_null_summary_url(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = _make_plan(
            [_make_task("a")],
            source_path=tmp_path / "plan.yaml",
            webhook_url="https://example.com/hook",
        )
        mock_exec, log = _make_exec()
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        captured: dict[str, object] = {}

        def fake_post(url: str, payload: dict[str, object]) -> int:
            captured["payload"] = payload
            return 200

        monkeypatch.setattr(
            "maestro_cli.scheduler._post_completion_webhook", fake_post
        )

        # Force Path.resolve() to return a relative path so as_uri() raises.
        real_resolve = Path.resolve

        def fake_resolve(self, *a, **kw):
            if self.name == "run_summary.md":
                return Path("relative/run_summary.md")
            return real_resolve(self, *a, **kw)

        monkeypatch.setattr(Path, "resolve", fake_resolve)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.success is True
        payload = captured["payload"]
        assert isinstance(payload, dict)
        assert payload["summary_url"] is None


# ===========================================================================
# 3435-3437 — record cost into cross-run budget ledger after the run
# ===========================================================================

class TestBudgetLedgerRecording:
    def test_total_cost_recorded_to_ledger(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        now = datetime.now(UTC)
        costed = TaskResult(
            task_id="a", status="success", exit_code=0,
            started_at=now, finished_at=now, duration_sec=0.01,
            command="echo a", log_path=tmp_path / "a.log",
            result_path=tmp_path / "a.result.json", message="ok",
            cost_usd=0.42,
        )
        plan = _make_plan(
            [_make_task("a", command=None, engine="claude", prompt="x")],
            source_path=tmp_path / "plan.yaml",
            budget_period="daily",
            max_cost_usd=100.0,
        )
        # Allow the pre-run budget gate to pass.
        monkeypatch.setattr(
            "maestro_cli.budget.check_budget",
            lambda *a, **kw: (True, 0.0, 100.0),
        )
        recorded: list[tuple[object, ...]] = []
        monkeypatch.setattr(
            "maestro_cli.budget.record_cost",
            lambda *a, **kw: recorded.append(a),
        )
        mock_exec, log = _make_exec(overrides={"a": costed})
        monkeypatch.setattr("maestro_cli.scheduler.execute_task", mock_exec)
        _no_preflight(monkeypatch)

        result = run_plan(plan, run_dir_override=str(tmp_path / "runs"))
        assert result.success is True
        assert len(recorded) == 1
        # record_cost(ledger_path, plan_name, run_id, total_cost_usd)
        assert recorded[0][1] == "test-plan"
        assert recorded[0][3] == pytest.approx(0.42)
