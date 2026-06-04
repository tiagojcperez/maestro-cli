from __future__ import annotations

"""Coverage tests for the post-execution tail of ``execute_task`` in runners.py.

These exercise the result-decoration branches that run *after* the main
subprocess loop: signal-data attachment, output-envelope / scope verification,
worktree merge + dual verification, contract normalization, dynamic-group
decomposition (Phase 2), auto-routed-model attachment, and phantom-workspace
commit/cleanup.

The engine/subprocess boundary is always mocked: ``build_command`` returns a
fixed argv, ``subprocess.Popen`` returns a dummy proc, and ``_stream_process``
returns a crafted ``(returncode, stdout_tail, stderr_tail)`` tuple so the task
reaches the tail with a known status and output.
"""

from pathlib import Path
from typing import Any

import pytest

from maestro_cli.models import (
    DualVerificationResult,
    EngineDefaults,
    MergeOverlap,
    MergeReview,
    PlanDefaults,
    PlanSpec,
    StructuredContext,
    TaskSpec,
    WorktreeMergeResult,
)
from maestro_cli import runners
from maestro_cli.runners import execute_task


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_plan(tmp_path: Path, *, workspace_root: str | None = None) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="cov",
        max_parallel=1,
        fail_fast=False,
        run_dir=str(tmp_path / "runs"),
        workspace_root=workspace_root,
        defaults=PlanDefaults(
            codex=EngineDefaults(),
            claude=EngineDefaults(),
            gemini=EngineDefaults(),
        ),
        tasks=[],
    )


class _DummyProc:
    pass


def _wire_engine_success(
    monkeypatch: pytest.MonkeyPatch,
    *,
    returncode: int = 0,
    stdout_tail: str = "ok",
    stderr_tail: str = "",
) -> None:
    """Mock the engine boundary so ``execute_task`` reaches the tail.

    The command-build, process-spawn, and stream-drain calls are all replaced
    with deterministic stand-ins.
    """

    def _fake_build_command(
        _plan: PlanSpec,
        _task: TaskSpec,
        _workdir: Path,
        **_kwargs: Any,
    ) -> tuple[list[str], bool]:
        return (["gemini", "-m", "flash", "-p", "go"], False)

    monkeypatch.setattr(runners, "build_command", _fake_build_command)
    monkeypatch.setattr(
        runners.subprocess, "Popen", lambda *a, **kw: _DummyProc()
    )
    monkeypatch.setattr(
        runners,
        "_stream_process",
        lambda *a, **kw: (returncode, stdout_tail, stderr_tail),
    )


# ---------------------------------------------------------------------------
# Signal data attachment (8001-8003)
# ---------------------------------------------------------------------------


class TestSignalDataAttachment:
    def test_signals_attached_to_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t", engine="gemini", model="flash", prompt="go", signals=True)

        _wire_engine_success(monkeypatch)
        result = execute_task(plan, task, run_path)

        # signal_handler was created (signals enabled) -> result fields populated
        assert result.status == "success"
        assert result.signals_received == []
        assert result.last_progress_pct is None


# ---------------------------------------------------------------------------
# Structured-context extraction failure (8011-8012)
# ---------------------------------------------------------------------------


class TestStructuredContextException:
    def test_extract_raises_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t", engine="gemini", model="flash", prompt="go")

        _wire_engine_success(monkeypatch)

        def _boom(*_a: Any, **_kw: Any) -> StructuredContext:
            raise RuntimeError("extract failed")

        monkeypatch.setattr(runners, "extract_structured_context", _boom)

        result = execute_task(plan, task, run_path)

        # exception swallowed -> structured_context stays None, status unchanged
        assert result.status == "success"
        assert result.structured_context is None


# ---------------------------------------------------------------------------
# Output envelope + scope verification (8016-8034)
# ---------------------------------------------------------------------------


class TestOutputEnvelope:
    def _ctx(self, files: list[str]) -> StructuredContext:
        return StructuredContext(
            task_id="t",
            status="success",
            exit_code=0,
            duration_sec=0.1,
            files_changed=files,
        )

    def test_scope_violation_emits_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="gemini",
            model="flash",
            prompt="go",
            output_scope=["src/*.py"],
        )

        _wire_engine_success(monkeypatch)
        monkeypatch.setattr(
            runners,
            "extract_structured_context",
            lambda *a, **kw: self._ctx(["secrets/leak.txt"]),
        )

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda n, p: events.append((n, p)),
        )

        assert result.output_envelope is not None
        assert result.output_envelope.scope_verified is False
        names = [n for n, _ in events]
        assert "scope_violation" in names
        payload = next(p for n, p in events if n == "scope_violation")
        assert "secrets/leak.txt" in payload["violations"]

    def test_scope_clean_no_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="gemini",
            model="flash",
            prompt="go",
            output_scope=["src/*.py"],
        )

        _wire_engine_success(monkeypatch)
        monkeypatch.setattr(
            runners,
            "extract_structured_context",
            lambda *a, **kw: self._ctx(["src/main.py"]),
        )

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda n, p: events.append((n, p)),
        )

        assert result.output_envelope is not None
        assert result.output_envelope.scope_verified is True
        assert "scope_violation" not in [n for n, _ in events]

    def test_envelope_build_raises_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="gemini",
            model="flash",
            prompt="go",
            output_scope=["src/*.py"],
        )

        _wire_engine_success(monkeypatch)
        monkeypatch.setattr(
            runners,
            "extract_structured_context",
            lambda *a, **kw: self._ctx(["src/main.py"]),
        )

        import maestro_cli.eventsource as eventsource

        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("envelope build failed")

        monkeypatch.setattr(eventsource, "build_output_envelope", _boom)

        result = execute_task(plan, task, run_path)

        # exception swallowed -> envelope stays None, status unaffected
        assert result.status == "success"
        assert result.output_envelope is None


# ---------------------------------------------------------------------------
# Worktree merge + dual verification (8036-8103)
# ---------------------------------------------------------------------------


class TestWorktreeMerge:
    def _wire_worktree(
        self,
        monkeypatch: pytest.MonkeyPatch,
        ws_root: Path,
        merge_result: WorktreeMergeResult,
    ) -> dict[str, Any]:
        import maestro_cli.worktree as worktree

        captured: dict[str, Any] = {}

        wt = ws_root / ".maestro-worktrees" / "t"
        wt.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(worktree, "get_base_branch", lambda *_a, **_kw: "main")
        monkeypatch.setattr(worktree, "create_worktree", lambda *_a, **_kw: wt)
        monkeypatch.setattr(
            worktree, "merge_worktree", lambda *_a, **_kw: merge_result
        )

        def _cleanup(*_a: Any, **_kw: Any) -> None:
            captured["cleaned"] = True

        monkeypatch.setattr(worktree, "cleanup_worktree", _cleanup)
        return captured

    def test_merge_success_with_review_and_verification(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws_root = tmp_path / "ws"
        ws_root.mkdir()
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path, workspace_root=str(ws_root))
        plan.source_path = tmp_path / "plan.yaml"
        task = TaskSpec(id="t", engine="gemini", model="flash", prompt="go", worktree=True)

        review = MergeReview(
            verdict="clean",
            overlapping_files=[MergeOverlap(file="a.py", merged_by=["x"])],
        )
        merge_result = WorktreeMergeResult(
            status="merged",
            files_changed=["a.py"],
            review=review,
        )
        captured = self._wire_worktree(monkeypatch, ws_root, merge_result)
        _wire_engine_success(monkeypatch)

        # Dual verification: report a gap so the print branch runs
        import maestro_cli.worktree as worktree

        monkeypatch.setattr(
            worktree,
            "verify_worktree_output",
            lambda *_a, **_kw: DualVerificationResult(
                verified=False,
                files_in_diff=["a.py"],
                files_claimed=["b.py"],
                unclaimed_files=["a.py"],
                phantom_files=["b.py"],
                overlap_ratio=0.0,
            ),
        )

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda n, p: events.append((n, p)),
        )

        names = [n for n, _ in events]
        assert "worktree_merge" in names
        assert "worktree_verification" in names
        assert "worktree_cleanup" in names
        assert captured.get("cleaned") is True
        assert result.worktree_merge is merge_result
        assert result.worktree_merge.verification is not None
        merge_payload = next(p for n, p in events if n == "worktree_merge")
        assert merge_payload["review_verdict"] == "clean"
        assert merge_payload["overlapping_files"] == ["a.py"]

    def test_merge_verification_raises_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws_root = tmp_path / "ws"
        ws_root.mkdir()
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path, workspace_root=str(ws_root))
        plan.source_path = tmp_path / "plan.yaml"
        task = TaskSpec(id="t", engine="gemini", model="flash", prompt="go", worktree=True)

        merge_result = WorktreeMergeResult(status="merged", files_changed=["a.py"])
        self._wire_worktree(monkeypatch, ws_root, merge_result)
        _wire_engine_success(monkeypatch)

        import maestro_cli.worktree as worktree

        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("verify failed")

        monkeypatch.setattr(worktree, "verify_worktree_output", _boom)

        result = execute_task(plan, task, run_path)

        # verification error swallowed; merge still recorded, status stays success
        assert result.status == "success"
        assert result.worktree_merge is merge_result
        assert result.worktree_merge.verification is None

    def test_merge_conflict_marks_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws_root = tmp_path / "ws"
        ws_root.mkdir()
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path, workspace_root=str(ws_root))
        plan.source_path = tmp_path / "plan.yaml"
        task = TaskSpec(id="t", engine="gemini", model="flash", prompt="go", worktree=True)

        review = MergeReview(
            verdict="conflict",
            overlapping_files=[MergeOverlap(file="a.py", merged_by=["other"])],
            resolution_suggestion="rebase first",
        )
        merge_result = WorktreeMergeResult(
            status="conflict",
            conflict_files=["a.py"],
            review=review,
        )
        self._wire_worktree(monkeypatch, ws_root, merge_result)
        _wire_engine_success(monkeypatch)

        result = execute_task(plan, task, run_path)

        assert result.status == "failed"
        assert "Worktree merge conflict" in (result.message or "")
        assert "rebase first" in (result.message or "")
        assert "other" in (result.message or "")

    def test_cleanup_raises_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws_root = tmp_path / "ws"
        ws_root.mkdir()
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path, workspace_root=str(ws_root))
        plan.source_path = tmp_path / "plan.yaml"
        task = TaskSpec(id="t", engine="gemini", model="flash", prompt="go", worktree=True)

        # status not in success/soft_failed avoids merge; cleanup still runs
        merge_result = WorktreeMergeResult(status="merged")
        self._wire_worktree(monkeypatch, ws_root, merge_result)

        import maestro_cli.worktree as worktree

        def _boom(*_a: Any, **_kw: Any) -> None:
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(worktree, "cleanup_worktree", _boom)
        # task fails so merge block is skipped but cleanup still attempted
        _wire_engine_success(monkeypatch, returncode=2, stdout_tail="boom")

        result = execute_task(plan, task, run_path)

        # cleanup error swallowed -> result still produced
        assert result.status == "failed"


# ---------------------------------------------------------------------------
# Contract normalization failure (8112-8113)
# ---------------------------------------------------------------------------


class TestContractNormalizationException:
    def test_normalize_raises_sets_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="gemini",
            model="flash",
            prompt="go",
            contract_type="file-inventory",
        )

        _wire_engine_success(monkeypatch)

        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("normalize failed")

        monkeypatch.setattr(runners, "normalize_task_contract", _boom)

        result = execute_task(plan, task, run_path)

        assert result.status == "success"
        assert result.produced_contract is None


# ---------------------------------------------------------------------------
# Dynamic-group Phase 2 decomposition (8132-8166)
# ---------------------------------------------------------------------------


class TestDynamicGroupPhase2:
    def _wire_output_schema_success(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # stdout is valid JSON matching a trivial schema -> structured_output set
        _wire_engine_success(
            monkeypatch, stdout_tail='{"items": ["a", "b"]}'
        )

    def test_subplan_runs_and_emits_events(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        plan.source_path = tmp_path / "plan.yaml"
        task = TaskSpec(
            id="t",
            engine="gemini",
            model="flash",
            prompt="go",
            dynamic_group=True,
            output_schema={"type": "object"},
        )

        self._wire_output_schema_success(monkeypatch)

        import maestro_cli.dynamic as dynamic

        sub_plan = _make_plan(tmp_path)
        sub_plan.name = "sub"
        sub_plan.tasks = [
            TaskSpec(id="s1", engine="gemini", model="flash", prompt="x"),
        ]

        class _SubResult:
            success = True
            task_results = [object(), object()]
            total_cost_usd = 1.25

        sub_result = _SubResult()
        merged_marker: dict[str, Any] = {}

        def _merge(phase1: Any, sub: Any, _task: Any) -> Any:
            merged_marker["called"] = True
            phase1.message = "merged"
            return phase1

        monkeypatch.setattr(dynamic, "write_raw_output", lambda *a, **kw: None)
        monkeypatch.setattr(
            dynamic, "build_plan_from_output", lambda *a, **kw: sub_plan
        )
        monkeypatch.setattr(
            dynamic, "run_dynamic_subplan", lambda *a, **kw: sub_result
        )
        monkeypatch.setattr(dynamic, "merge_dynamic_result", _merge)

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda n, p: events.append((n, p)),
        )

        names = [n for n, _ in events]
        assert "dynamic_subplan_start" in names
        assert "dynamic_subplan_complete" in names
        assert merged_marker.get("called") is True
        start = next(p for n, p in events if n == "dynamic_subplan_start")
        assert start["sub_plan_name"] == "sub"
        assert start["sub_task_count"] == 1
        complete = next(p for n, p in events if n == "dynamic_subplan_complete")
        assert complete["success"] is True
        assert complete["sub_task_count"] == 2
        assert complete["total_cost_usd"] == 1.25
        assert result.message == "merged"

    def test_subplan_build_returns_none_fails(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        plan.source_path = tmp_path / "plan.yaml"
        task = TaskSpec(
            id="t",
            engine="gemini",
            model="flash",
            prompt="go",
            dynamic_group=True,
            output_schema={"type": "object"},
        )

        self._wire_output_schema_success(monkeypatch)

        import maestro_cli.dynamic as dynamic

        monkeypatch.setattr(dynamic, "write_raw_output", lambda *a, **kw: None)
        monkeypatch.setattr(
            dynamic, "build_plan_from_output", lambda *a, **kw: None
        )

        result = execute_task(plan, task, run_path)

        assert result.status == "failed"
        assert "could not be built" in (result.message or "")


# ---------------------------------------------------------------------------
# Auto-routed model attachment (8168-8169)
# ---------------------------------------------------------------------------


class TestAutoRoutedModel:
    def test_auto_model_attached(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t", engine="gemini", model="auto", prompt="go")

        _wire_engine_success(monkeypatch)

        import maestro_cli.routing as routing

        monkeypatch.setattr(
            routing, "resolve_auto_model", lambda *a, **kw: "flash"
        )

        result = execute_task(plan, task, run_path)

        assert result.status == "success"
        assert result.auto_routed_model == "flash"


# ---------------------------------------------------------------------------
# Phantom workspace commit/cleanup (8172-8181)
# ---------------------------------------------------------------------------


class TestPhantomWorkspace:
    def test_phantom_commit_emits_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws_root = tmp_path / "ws"
        ws_root.mkdir()
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path, workspace_root=str(ws_root))
        task = TaskSpec(
            id="t",
            engine="gemini",
            model="flash",
            prompt="go",
            phantom_workspace=True,
        )

        _wire_engine_success(monkeypatch)

        # Drop a file into the phantom dir AFTER it is created, so the commit
        # step finds something to copy. Wrap the real setup helper.
        real_setup = runners._setup_phantom_workspace
        created: dict[str, Path] = {}

        def _setup(run_p: Path, task_id: str) -> Path:
            d = real_setup(run_p, task_id)
            (d / "generated.txt").write_text("hello", encoding="utf-8")
            created["dir"] = d
            return d

        monkeypatch.setattr(runners, "_setup_phantom_workspace", _setup)

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda n, p: events.append((n, p)),
        )

        assert result.status == "success"
        names = [n for n, _ in events]
        assert "phantom_commit" in names
        payload = next(p for n, p in events if n == "phantom_commit")
        assert "generated.txt" in payload["files_committed"]
        # committed file copied into workspace root, phantom dir cleaned up
        assert (ws_root / "generated.txt").exists()
        assert not created["dir"].exists()

    def test_phantom_no_files_no_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws_root = tmp_path / "ws"
        ws_root.mkdir()
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path, workspace_root=str(ws_root))
        task = TaskSpec(
            id="t",
            engine="gemini",
            model="flash",
            prompt="go",
            phantom_workspace=True,
        )

        _wire_engine_success(monkeypatch)

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda n, p: events.append((n, p)),
        )

        # empty phantom dir -> no commit event but cleanup still ran
        assert result.status == "success"
        assert "phantom_commit" not in [n for n, _ in events]
