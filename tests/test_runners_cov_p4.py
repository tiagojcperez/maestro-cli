from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from maestro_cli.models import (
    EngineDefaults,
    JudgeSpec,
    PlanDefaults,
    PlanSpec,
    TaskSpec,
    WorkspaceExtraction,
    WorktreeMergeResult,
)
from maestro_cli.workspace_index import FileEntry, WorkspaceIndex
from maestro_cli.runners import (
    _evaluate_typed_assertion,
    _execute_group_task,
    _extract_json_from_text,
    _generate_eval_steps,
    _parse_judge_response,
    _run_reflection_evaluation,
    _run_workspace_brief,
    execute_task,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_plan(tmp_path: Path, *, workspace_root: str | None = None) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="test",
        max_parallel=1,
        fail_fast=False,
        run_dir=str(tmp_path / "runs"),
        workspace_root=workspace_root,
        defaults=PlanDefaults(codex=EngineDefaults(), claude=EngineDefaults()),
        tasks=[],
    )


class _FakeProc:
    """Stand-in for subprocess.run / Popen results."""

    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _patch_success_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    *,
    build_command: Any = None,
) -> None:
    """Mock the engine execution boundary so execute_task succeeds quickly."""

    def _fake_build_command(
        _plan: PlanSpec,
        _task: TaskSpec,
        _workdir: Path,
        **kwargs: Any,
    ) -> tuple[list[str], bool]:
        return (["claude", "--print", "do something"], False)

    monkeypatch.setattr(
        "maestro_cli.runners.build_command",
        build_command or _fake_build_command,
    )
    monkeypatch.setattr(
        "maestro_cli.runners.subprocess.Popen",
        lambda *a, **kw: _FakeProc(),
    )
    monkeypatch.setattr(
        "maestro_cli.runners._stream_process",
        lambda *a, **kw: (0, "ok", ""),
    )


# ---------------------------------------------------------------------------
# _generate_eval_steps — blank-line skip (line ~5005)
# ---------------------------------------------------------------------------


class TestGenerateEvalStepsBlankLine:
    def test_blank_lines_are_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # stdout has a blank line between numbered steps -> blank-line `continue`.
        stdout = "1. First step\n\n   \n2. Second step\n"

        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable",
            lambda _name: ["claude"],
        )
        monkeypatch.setattr(
            "maestro_cli.runners.subprocess.run",
            lambda *a, **kw: _FakeProc(returncode=0, stdout=stdout),
        )

        steps = _generate_eval_steps("some criteria", workdir=tmp_path)
        assert steps == ["First step", "Second step"]


# ---------------------------------------------------------------------------
# _extract_json_from_text — markdown block with invalid JSON (lines ~5278-5279)
# ---------------------------------------------------------------------------


class TestExtractJsonMarkdownBlockInvalid:
    def test_markdown_block_with_invalid_json_returns_none(self) -> None:
        # Fenced block matches the regex but content is not valid JSON.
        text = "```json\n{bad json here}\n```"
        assert _extract_json_from_text(text) is None

    def test_markdown_block_invalid_then_valid_balanced_block(self) -> None:
        # Block content is invalid, but there's a valid balanced block later.
        text = "```\n{nope}\n```\nand here: {\"k\": 1}"
        # The fenced block fails (except: pass); balanced fallback finds {nope}
        # first which also fails. So overall None for this crafted input.
        assert _extract_json_from_text(text) is None


# ---------------------------------------------------------------------------
# _evaluate_typed_assertion — is-json invalid token + None values
# (lines ~5418-5419, 5437, 5474)
# ---------------------------------------------------------------------------


class TestTypedAssertionEdges:
    def test_is_json_invalid_token_continues(self) -> None:
        # Contains a `{` that does not decode to valid JSON anywhere.
        result = _evaluate_typed_assertion(
            {"type": "is-json"},
            "prose with a stray { brace but no json",
            None,
            0.0,
        )
        assert result is not None
        assert result.passed is False
        assert "does not contain valid JSON" in result.reasoning

    def test_is_json_recovers_after_invalid_prefix(self) -> None:
        # First `{` fails raw_decode (continue), later valid JSON object found.
        result = _evaluate_typed_assertion(
            {"type": "is-json"},
            "{oops not json}\nthen {\"ok\": true}",
            None,
            0.0,
        )
        assert result is not None
        assert result.passed is True

    def test_cost_under_value_none(self) -> None:
        result = _evaluate_typed_assertion(
            {"type": "cost_under"},
            "output",
            0.5,
            0.0,
        )
        assert result is not None
        assert result.passed is False
        assert "value must be numeric" in result.reasoning

    def test_duration_under_value_none(self) -> None:
        result = _evaluate_typed_assertion(
            {"type": "duration_under"},
            "output",
            None,
            1.0,
        )
        assert result is not None
        assert result.passed is False
        assert "value must be numeric" in result.reasoning


# ---------------------------------------------------------------------------
# _run_workspace_brief — relevant file missing from index (line ~6490)
# ---------------------------------------------------------------------------


class TestWorkspaceBriefMissingFile:
    def test_relevant_file_not_in_index_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        present = FileEntry(
            path="src/known.py",
            size_bytes=10,
            mtime_ns=0,
            sha256="x",
            language="python",
            first_lines=["print('hi')"],
        )
        index = WorkspaceIndex(files=[present])
        extraction = WorkspaceExtraction(
            relevant_files=["src/known.py", "src/missing.py"],
        )

        monkeypatch.setattr(
            "maestro_cli.runners._resolve_executable",
            lambda _name: ["claude"],
        )
        monkeypatch.setattr(
            "maestro_cli.runners.subprocess.run",
            lambda *a, **kw: _FakeProc(returncode=0, stdout="a focused brief"),
        )

        brief = _run_workspace_brief(index, extraction, "do work", tmp_path)
        assert brief.brief_text == "a focused brief"
        # Both relevant files are referenced even though one was skipped in preview.
        assert "src/missing.py" in brief.files_referenced


# ---------------------------------------------------------------------------
# execute_task — deliberation gate skip (lines ~7102-7139)
# ---------------------------------------------------------------------------


class TestExecuteTaskDeliberationSkip:
    def test_deliberation_gate_skips_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="answerable from context",
            deliberation=True,
            deliberation_threshold=0.5,
        )

        # Mock the gate to return (gate_pass=False, score) -> skip branch.
        monkeypatch.setattr(
            "maestro_cli.runners._run_deliberation_gate",
            lambda *a, **kw: (False, 0.1),
        )

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda name, payload: events.append((name, payload)),
        )

        assert result.status == "skipped"
        assert "deliberation" in result.message
        # deliberation_skip event emitted
        skip_events = [p for n, p in events if n == "deliberation_skip"]
        assert len(skip_events) == 1
        assert skip_events[0]["task_id"] == "t"
        # log file written
        assert (run_path / "t.log").exists()

    def test_deliberation_gate_skip_without_callback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Exercise the skip path with no event_callback (event branch not taken).
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t2",
            engine="claude",
            model="haiku",
            prompt="answerable",
            deliberation=True,
            deliberation_threshold=0.5,
        )
        monkeypatch.setattr(
            "maestro_cli.runners._run_deliberation_gate",
            lambda *a, **kw: (False, 0.2),
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "skipped"


# ---------------------------------------------------------------------------
# execute_task — model: auto routing (lines ~7149-7169)
# ---------------------------------------------------------------------------


class TestExecuteTaskAutoRouting:
    def test_auto_model_resolution_emits_event(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="auto-task",
            engine="claude",
            model="auto",
            prompt="do something",
        )

        def _fake_resolve(
            _task: TaskSpec,
            _plan: PlanSpec,
            _engine: str,
            *,
            routing_strategy: Any = None,
            dag_metadata: Any = None,
            evidence: dict[str, object] | None = None,
        ) -> str:
            if evidence is not None:
                evidence["complexity_score"] = 0.42
                evidence["historical_runs"] = 3
            return "sonnet"

        monkeypatch.setattr(
            "maestro_cli.routing.resolve_auto_model", _fake_resolve
        )
        _patch_success_pipeline(monkeypatch)

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda name, payload: events.append((name, payload)),
        )

        assert result.status == "success"
        routed = [p for n, p in events if n == "model_routed"]
        assert len(routed) == 1
        assert routed[0]["resolved"] == "sonnet"
        assert routed[0]["requested"] == "auto"
        assert routed[0]["complexity_score"] == 0.42


# ---------------------------------------------------------------------------
# execute_task — worktree create success + failure (lines ~7172-7194)
# ---------------------------------------------------------------------------


class TestExecuteTaskWorktree:
    def test_worktree_created_and_used(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws_root = tmp_path / "ws"
        ws_root.mkdir()
        worktree_dir = tmp_path / "wt"
        worktree_dir.mkdir()  # must exist (later workdir.exists() check)

        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path, workspace_root=str(ws_root))
        task = TaskSpec(
            id="wt-task",
            engine="claude",
            model="haiku",
            prompt="do something",
            worktree=True,
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.get_base_branch", lambda _root: "main"
        )
        monkeypatch.setattr(
            "maestro_cli.worktree.create_worktree",
            lambda _root, _tid, _branch: worktree_dir,
        )
        # Avoid real merge/cleanup work after success.
        monkeypatch.setattr(
            "maestro_cli.worktree.merge_worktree",
            lambda *a, **kw: WorktreeMergeResult(status="empty"),
        )
        monkeypatch.setattr(
            "maestro_cli.worktree.cleanup_worktree",
            lambda *a, **kw: None,
        )
        monkeypatch.setattr(
            "maestro_cli.worktree.verify_worktree_output",
            lambda *a, **kw: None,
        )
        _patch_success_pipeline(monkeypatch)

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda name, payload: events.append((name, payload)),
        )

        assert result.status == "success"
        create_events = [p for n, p in events if n == "worktree_create"]
        assert len(create_events) == 1
        assert create_events[0]["task_id"] == "wt-task"

    def test_worktree_creation_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ws_root = tmp_path / "ws"
        ws_root.mkdir()

        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path, workspace_root=str(ws_root))
        task = TaskSpec(
            id="wt-fail",
            engine="claude",
            model="haiku",
            prompt="do something",
            worktree=True,
        )

        monkeypatch.setattr(
            "maestro_cli.worktree.get_base_branch", lambda _root: "main"
        )

        def _boom(*_a: Any, **_kw: Any) -> Path:
            raise RuntimeError("worktree explosion")

        monkeypatch.setattr("maestro_cli.worktree.create_worktree", _boom)

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert "Failed to create worktree" in result.message


# ---------------------------------------------------------------------------
# execute_task — build_command exception (lines ~7210-7226)
# ---------------------------------------------------------------------------


class TestExecuteTaskBuildCommandError:
    def test_build_command_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="bc-fail",
            engine="claude",
            model="haiku",
            prompt="do something",
        )

        def _boom(*_a: Any, **_kw: Any) -> tuple[list[str], bool]:
            raise ValueError("cannot build command")

        monkeypatch.setattr("maestro_cli.runners.build_command", _boom)

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert result.exit_code == 1
        assert "cannot build command" in result.message


# ---------------------------------------------------------------------------
# execute_task — mid-task signals env setup (lines ~7281-7282)
# ---------------------------------------------------------------------------


class TestExecuteTaskSignals:
    def test_signals_enabled_runs_to_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="sig-task",
            engine="claude",
            model="haiku",
            prompt="do something",
            signals=True,
        )

        _patch_success_pipeline(monkeypatch)

        result = execute_task(plan, task, run_path)
        assert result.status == "success"


# ---------------------------------------------------------------------------
# _parse_judge_response — criterion score coercion failure (lines ~5054-5055)
# ---------------------------------------------------------------------------


class TestParseJudgeResponseBadScore:
    def test_non_numeric_score_skips_criterion(self) -> None:
        # score is a non-numeric value -> float() raises ValueError -> continue.
        text = (
            '{"criteria": ['
            '{"criterion": "good", "passed": true, "score": "not-a-number"},'
            '{"criterion": "ok", "passed": true, "score": 0.9}'
            '], "overall_score": 0.9, "reasoning": "fine"}'
        )
        result = _parse_judge_response(text)
        # The bad criterion is dropped; the valid one is kept.
        assert len(result.criterion_scores) == 1
        assert result.criterion_scores[0].criterion == "ok"


# ---------------------------------------------------------------------------
# _run_reflection_evaluation — empty criteria_text fallback (line ~5652)
# ---------------------------------------------------------------------------


class TestReflectionEmptyCriteria:
    def test_only_typed_assertions_yield_default_criteria_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # All criteria are typed assertions -> criteria_text becomes empty ->
        # falls into the default "(evaluate overall quality...)" branch.
        judge = JudgeSpec(
            criteria=[{"type": "contains", "value": "x"}],
            pass_threshold=0.7,
            method="reflection",
        )

        def _raise(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("no claude here")

        # Make all subprocess calls raise so the function returns quickly
        # while still passing through the criteria_text fallback line.
        monkeypatch.setattr("maestro_cli.runners.subprocess.run", _raise)

        result = _run_reflection_evaluation(
            "t", judge, "some output", tmp_path
        )
        # Phase 1 fails -> direct fallback also fails -> error verdict.
        assert result.verdict == "error"


# ---------------------------------------------------------------------------
# _execute_group_task — sub-plan execution exception (lines ~6715-6716)
# ---------------------------------------------------------------------------


class TestGroupSubPlanExecutionError:
    def test_sub_plan_run_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sub_plan_path = tmp_path / "sub_plan.yaml"
        sub_plan_path.write_text(
            "version: 1\nname: sub\ntasks:\n"
            "  - id: sub-task-1\n    command: echo hello\n",
            encoding="utf-8",
        )
        plan = PlanSpec(
            version=1,
            name="parent",
            defaults=PlanDefaults(),
            tasks=[],
            workspace_root=str(tmp_path),
            source_path=sub_plan_path,
        )
        task = TaskSpec(id="group-1", group=str(sub_plan_path))

        def _boom(*_a: Any, **_kw: Any) -> Any:
            raise RuntimeError("sub-plan blew up")

        # run_plan is imported inside the function from .scheduler
        monkeypatch.setattr("maestro_cli.scheduler.run_plan", _boom)

        result = _execute_group_task(
            plan, task, tmp_path, dry_run=False, execution_profile="plan"
        )
        assert result.status == "failed"
        assert "sub-plan execution error" in result.message


# ---------------------------------------------------------------------------
# execute_task — deliberation_skip callback that raises (lines ~7137-7138)
# ---------------------------------------------------------------------------


class TestDeliberationSkipCallbackRaises:
    def test_callback_exception_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t-cb",
            engine="claude",
            model="haiku",
            prompt="answerable",
            deliberation=True,
            deliberation_threshold=0.5,
        )
        monkeypatch.setattr(
            "maestro_cli.runners._run_deliberation_gate",
            lambda *a, **kw: (False, 0.1),
        )

        def _bad_callback(name: str, payload: dict[str, object]) -> None:
            if name == "deliberation_skip":
                raise RuntimeError("callback failure")

        # Should not propagate — the except Exception: pass swallows it.
        result = execute_task(
            plan, task, run_path, event_callback=_bad_callback
        )
        assert result.status == "skipped"
