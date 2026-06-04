from __future__ import annotations

"""Coverage tests for the ``execute_task`` retry/judge/fallback loop in runners.py.

These tests drive the real ``execute_task`` function through specific branches of
its main attempt loop.  All external boundaries (``subprocess.Popen``,
``_stream_process``, ``build_command``, judge/guard/verify helpers) are mocked;
``execute_task`` itself is never mocked.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.models import (
    CriterionScore,
    EngineDefaults,
    JudgeResult,
    JudgeSpec,
    PlanDefaults,
    PlanSpec,
    TaskSpec,
)
from maestro_cli.runners import execute_task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(tmp_path: Path) -> PlanSpec:
    return PlanSpec(
        version=1,
        name="test",
        max_parallel=1,
        fail_fast=False,
        run_dir=str(tmp_path / "runs"),
        defaults=PlanDefaults(codex=EngineDefaults(), claude=EngineDefaults()),
        tasks=[],
    )


class _DummyProc:
    """Stand-in for a ``subprocess.Popen`` object — never actually used."""


def _patch_popen(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "maestro_cli.runners.subprocess.Popen",
        lambda *a, **kw: _DummyProc(),
    )


def _patch_build(
    monkeypatch: pytest.MonkeyPatch,
    *,
    raise_on_call: int | None = None,
) -> list[dict[str, Any]]:
    """Patch ``build_command``; record kwargs per call.

    When *raise_on_call* is set (0-based), that call raises to exercise the
    rebuild-failure fallback path.
    """
    calls: list[dict[str, Any]] = []

    def _fake_build(
        _plan: PlanSpec,
        _task: TaskSpec,
        _workdir: Path,
        **kwargs: Any,
    ) -> tuple[list[str], bool]:
        idx = len(calls)
        calls.append(dict(kwargs))
        if raise_on_call is not None and idx == raise_on_call:
            raise RuntimeError("synthetic build failure")
        engine = kwargs.get("engine_override") or _task.engine or "claude"
        return ([str(engine), "--print", "prompt"], False)

    monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build)
    return calls


def _patch_stream(
    monkeypatch: pytest.MonkeyPatch,
    results: list[tuple[int, str, str]],
    *,
    feed_lines: list[str] | None = None,
) -> None:
    """Patch ``_stream_process`` to pop pre-canned results.

    If *feed_lines* is supplied, the line_callback (if any) is invoked with each
    line before returning the result (exercises ``_on_line``).
    """

    def _fake_stream(*args: Any, **kwargs: Any) -> tuple[int, str, str]:
        cb = kwargs.get("line_callback")
        if feed_lines and cb is not None:
            for line in feed_lines:
                cb(line)
        return results.pop(0)

    monkeypatch.setattr("maestro_cli.runners._stream_process", _fake_stream)


# ---------------------------------------------------------------------------
# task_retry callback exception (7438-7439) + rebuild failure (7458-7459)
# ---------------------------------------------------------------------------


class TestRetryCallbackAndRebuild:
    def test_task_retry_callback_raises_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            verify_command="true",
            max_retries=1,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        # First attempt fails (exit 1), second succeeds.
        _patch_stream(monkeypatch, [(1, "boom", ""), (0, "ok", "")])
        # verify_command runs only on success/soft_fail; make it pass.
        monkeypatch.setattr(
            "maestro_cli.runners._run_pre_command",
            lambda *a, **kw: (True, 0, "verified"),
        )

        def _cb(name: str, payload: dict[str, object]) -> None:
            if name == "task_retry":
                raise ValueError("callback explode")

        result = execute_task(plan, task, run_path, event_callback=_cb)
        # The raised callback must not crash execute_task.
        assert result.status == "success"
        assert result.retry_count == 1

    def test_rebuild_command_failure_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            verify_command="true",
            max_retries=1,
        )
        # First build (call 0) succeeds; the retry rebuild (call 1) raises.
        _patch_build(monkeypatch, raise_on_call=1)
        _patch_popen(monkeypatch)
        # Attempt 0 fails verify -> sets verify_feedback so rebuild is attempted.
        verify_outcomes = [(False, 1, "verify failed"), (True, 0, "ok")]
        monkeypatch.setattr(
            "maestro_cli.runners._run_pre_command",
            lambda *a, **kw: verify_outcomes.pop(0),
        )
        _patch_stream(monkeypatch, [(0, "ok1", ""), (0, "ok2", "")])

        result = execute_task(plan, task, run_path)
        # Rebuild raised but was swallowed; second attempt then passed verify.
        assert result.status == "success"
        assert result.retry_count == 1


# ---------------------------------------------------------------------------
# signal handler init + _on_line signal handling (7490-7491, 7504-7510)
# ---------------------------------------------------------------------------


class TestSignalHandling:
    def test_signal_line_routed_to_handler(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            signals=True,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)

        captured: list[str] = []

        # A valid progress signal line — routed into the handler, not emitted
        # as task_output.
        sig = "[MAESTRO_SIGNAL] " + json.dumps({"type": "progress", "pct": 42})
        _patch_stream(
            monkeypatch,
            [(0, "ok", "")],
            feed_lines=[sig, "regular output line"],
        )

        def _cb(name: str, payload: dict[str, object]) -> None:
            if name == "task_output":
                captured.append(str(payload.get("line")))

        result = execute_task(plan, task, run_path, event_callback=_cb)
        assert result.status == "success"
        # Signal line must NOT have been emitted as task_output.
        assert all("MAESTRO_SIGNAL" not in c for c in captured)
        assert "regular output line" in captured

    def test_signal_handler_handle_exception_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            signals=True,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)

        # Force _SignalHandler.handle to raise so the except path runs.
        def _boom(self: Any, data: dict[str, Any]) -> None:
            raise RuntimeError("handler boom")

        monkeypatch.setattr("maestro_cli.runners._SignalHandler.handle", _boom)

        sig = "[MAESTRO_SIGNAL] " + json.dumps({"type": "progress", "pct": 10})
        _patch_stream(monkeypatch, [(0, "ok", "")], feed_lines=[sig])

        result = execute_task(plan, task, run_path)
        # The handler raised; execute_task swallowed it and still completed.
        assert result.status == "success"


# ---------------------------------------------------------------------------
# claude tool_use parsing (7524-7541) + result-text extraction (7604)
# ---------------------------------------------------------------------------


class TestClaudeStreamParsing:
    def test_tool_use_event_emits_task_tool_call(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t", engine="claude", model="haiku", prompt="do it")
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)

        assistant_line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {
                            "type": "tool_use",
                            "name": "Read",
                            "input": {"file_path": "/tmp/x"},
                        },
                        {"type": "text", "text": "hello"},
                    ]
                },
            }
        )
        _patch_stream(
            monkeypatch, [(0, "ok", "")], feed_lines=[assistant_line]
        )

        events: list[tuple[str, dict[str, object]]] = []
        result = execute_task(
            plan,
            task,
            run_path,
            event_callback=lambda n, p: events.append((n, p)),
        )
        assert result.status == "success"
        tool_calls = [p for n, p in events if n == "task_tool_call"]
        assert len(tool_calls) == 1
        assert tool_calls[0]["tool"] == "Read"
        assert "/tmp/x" in str(tool_calls[0]["input_preview"])

    def test_tool_call_callback_exception_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t", engine="claude", model="haiku", prompt="do it")
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)

        assistant_line = json.dumps(
            {
                "type": "assistant",
                "message": {
                    "content": [
                        {"type": "tool_use", "name": "Bash", "input": "ls"}
                    ]
                },
            }
        )
        _patch_stream(
            monkeypatch, [(0, "ok", "")], feed_lines=[assistant_line]
        )

        def _cb(name: str, payload: dict[str, object]) -> None:
            if name == "task_tool_call":
                raise RuntimeError("tool call cb boom")

        # Should not crash despite the callback raising on task_tool_call.
        result = execute_task(plan, task, run_path, event_callback=_cb)
        assert result.status == "success"
        assert result.tool_call_count == 1

    def test_claude_result_text_replaces_tail(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t", engine="claude", model="haiku", prompt="do it")
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)

        result_event = json.dumps(
            {
                "type": "result",
                "result": "HUMAN READABLE ANSWER",
                "is_error": False,
            }
        )
        # tail_output contains the JSON event so result-text extraction kicks in.
        _patch_stream(monkeypatch, [(0, result_event, "")])

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        # The raw JSON line should have been replaced by the readable result.
        assert result.stdout_tail == "HUMAN READABLE ANSWER"


# ---------------------------------------------------------------------------
# stderr hint truncation (7567)
# ---------------------------------------------------------------------------


class TestStderrHint:
    def test_long_stderr_truncated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(id="t", engine="gemini", model="flash", prompt="do it")
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)

        long_stderr = "E" * 500
        # returncode != 0, tail_output nearly empty (<20), stderr present + long.
        _patch_stream(monkeypatch, [(1, "x", long_stderr)])

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert "stderr:" in result.message
        assert result.message.rstrip(")").endswith("...")


# ---------------------------------------------------------------------------
# timeout + allow_failure -> soft_failed (7572-7573)
# ---------------------------------------------------------------------------


class TestTimeoutSoftFailure:
    def test_timeout_with_allow_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            allow_failure=True,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(monkeypatch, [(124, "", "")])

        result = execute_task(plan, task, run_path)
        assert result.status == "soft_failed"
        assert "timed out" in result.message
        assert "allow_failure=true" in result.message


# ---------------------------------------------------------------------------
# verify_failure callback exception (7638-7639)
# ---------------------------------------------------------------------------


class TestVerifyFailureCallback:
    def test_verify_failure_callback_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            verify_command="false",
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(monkeypatch, [(0, "ok", "")])
        monkeypatch.setattr(
            "maestro_cli.runners._run_pre_command",
            lambda *a, **kw: (False, 2, "verify boom"),
        )

        def _cb(name: str, payload: dict[str, object]) -> None:
            if name == "verify_failure":
                raise RuntimeError("verify cb boom")

        result = execute_task(plan, task, run_path, event_callback=_cb)
        assert result.status == "failed"
        assert "verify_command failed" in result.message


# ---------------------------------------------------------------------------
# guard output truncation (7659)
# ---------------------------------------------------------------------------


class TestGuardTruncation:
    def test_long_guard_output_truncated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            guard_command="false",
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(monkeypatch, [(0, "ok", "")])
        long_guard = "G" * 500
        monkeypatch.setattr(
            "maestro_cli.runners._run_guard_command",
            lambda *a, **kw: (False, long_guard),
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert "guard output:" in result.message
        assert "..." in result.message


# ---------------------------------------------------------------------------
# honeypot triggered callback exception (7702-7703)
# ---------------------------------------------------------------------------


class TestHoneypotCallback:
    def test_honeypot_callback_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            honeypot=True,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        # Output contains the honeypot marker -> triggers the access check.
        tainted = "agent leaked trap-00000000 oops"
        _patch_stream(monkeypatch, [(0, tainted, "")])

        def _cb(name: str, payload: dict[str, object]) -> None:
            if name == "honeypot_triggered":
                raise RuntimeError("honeypot cb boom")

        result = execute_task(plan, task, run_path, event_callback=_cb)
        assert result.status == "failed"
        assert "honeypot triggered" in result.message


# ---------------------------------------------------------------------------
# judge cost extraction + judge_start callback exception (7710, 7713, 7725-7726)
# ---------------------------------------------------------------------------


class TestJudgeStartCallback:
    def test_judge_start_callback_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            judge=JudgeSpec(criteria=["clear"], on_fail="warn"),
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(monkeypatch, [(0, "answer", "")])
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_quorum",
            lambda **kw: JudgeResult(
                verdict="pass", overall_score=0.95, reasoning="good"
            ),
        )

        def _cb(name: str, payload: dict[str, object]) -> None:
            if name == "judge_start":
                raise RuntimeError("judge_start cb boom")

        result = execute_task(plan, task, run_path, event_callback=_cb)
        # judge passed, on_fail=warn -> task still success.
        assert result.status == "success"
        assert result.judge_result is not None
        assert result.judge_result.verdict == "pass"


# ---------------------------------------------------------------------------
# comparative evaluation + judge fail retry storing previous
# (7740, 7750-7751, 7757, 7775, 7779-7780)
# ---------------------------------------------------------------------------


class TestComparativeJudge:
    def test_judge_retry_then_comparative(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            judge=JudgeSpec(criteria=["clear"], on_fail="retry"),
            max_retries=1,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        # Both attempts run the command successfully (exit 0); the judge decides.
        _patch_stream(monkeypatch, [(0, "first answer", ""), (0, "second answer", "")])

        # First judge: fail (sets previous_*). Second judge: pass.
        judge_results = [
            JudgeResult(
                verdict="fail",
                overall_score=0.2,
                reasoning="bad",
                criterion_scores=[
                    CriterionScore(
                        criterion="clear",
                        passed=False,
                        score=0.2,
                        reasoning="unclear",
                    )
                ],
            ),
            JudgeResult(verdict="pass", overall_score=0.9, reasoning="now good"),
        ]
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_quorum",
            lambda **kw: judge_results.pop(0),
        )

        comparative_calls: list[dict[str, Any]] = []

        def _fake_comparative(**kwargs: Any) -> JudgeResult:
            comparative_calls.append(kwargs)
            return JudgeResult(
                verdict="pass",
                overall_score=0.9,
                reasoning="improved over previous",
                previous_score=kwargs.get("previous_score"),
            )

        monkeypatch.setattr(
            "maestro_cli.runners._run_comparative_evaluation",
            _fake_comparative,
        )

        result = execute_task(plan, task, run_path)
        # Second attempt's judge passed -> overall success.
        assert result.status == "success"
        assert result.retry_count == 1
        # Comparative evaluation must have run on the retry, comparing against
        # the previous failed attempt's stored output/score.
        assert len(comparative_calls) == 1
        assert comparative_calls[0]["previous_score"] == pytest.approx(0.2)
        assert comparative_calls[0]["previous_output"] == "first answer"

    def test_judge_fail_with_comparative_feedback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            judge=JudgeSpec(criteria=["clear"], on_fail="retry"),
            max_retries=2,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(
            monkeypatch,
            [(0, "a1", ""), (0, "a2", ""), (0, "a3", "")],
        )
        # All three judge attempts fail so the comparative-feedback append path
        # (7774-7777) and the on_fail==retry store path (7778-7780) both run on
        # the retry attempts.
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_quorum",
            lambda **kw: JudgeResult(
                verdict="fail", overall_score=0.1, reasoning="still bad"
            ),
        )
        monkeypatch.setattr(
            "maestro_cli.runners._run_comparative_evaluation",
            lambda **kw: JudgeResult(
                verdict="fail",
                overall_score=0.1,
                reasoning="no better",
                previous_score=kw.get("previous_score"),
            ),
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "failed"
        assert result.judge_result is not None
        assert result.judge_result.verdict == "fail"
        # Exhausted all attempts.
        assert result.retry_count == 2


# ---------------------------------------------------------------------------
# fallback engine: claude reasoning effort handling + pop, callback exception
# (7829-7831, 7840-7841)
# ---------------------------------------------------------------------------


class TestFallbackReasoning:
    def test_fallback_to_claude_sets_reasoning_effort(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="codex",
            model="gpt-5-mini",
            reasoning_effort="high",
            fallback_engine="claude",
            fallback_model="sonnet",
            prompt="do it",
            max_retries=1,
        )
        # First attempt: engine-not-found (127) -> triggers fallback to claude.
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(monkeypatch, [(127, "command not found", ""), (0, "ok", "")])

        captured_env: dict[str, str] = {}

        # Capture the env passed to the fallback Popen to confirm the reasoning
        # effort var was set.
        def _spy_popen(*a: Any, **kw: Any) -> _DummyProc:
            env = kw.get("env") or {}
            captured_env.clear()
            captured_env.update(env)
            return _DummyProc()

        monkeypatch.setattr("maestro_cli.runners.subprocess.Popen", _spy_popen)

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        # After fallback to claude with reasoning_effort=high, env carries it.
        assert captured_env.get("CLAUDE_CODE_EFFORT_LEVEL") == "high"

    def test_fallback_pops_effort_for_non_claude(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        # Start on claude (sets CLAUDE_CODE_EFFORT_LEVEL), fall back to gemini
        # which is non-claude -> the env var must be popped .
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            reasoning_effort="high",
            fallback_engine="gemini",
            fallback_model="flash",
            prompt="do it",
            max_retries=1,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(monkeypatch, [(127, "command not found", ""), (0, "ok", "")])

        captured_envs: list[dict[str, str]] = []

        def _spy_popen(*a: Any, **kw: Any) -> _DummyProc:
            captured_envs.append(dict(kw.get("env") or {}))
            return _DummyProc()

        monkeypatch.setattr("maestro_cli.runners.subprocess.Popen", _spy_popen)

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        # Second Popen (fallback to gemini) must NOT carry the claude effort var.
        assert len(captured_envs) == 2
        assert "CLAUDE_CODE_EFFORT_LEVEL" not in captured_envs[1]

    def test_engine_fallback_callback_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="codex",
            model="gpt-5-mini",
            fallback_engine="claude",
            fallback_model="sonnet",
            prompt="do it",
            max_retries=1,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        _patch_stream(monkeypatch, [(127, "command not found", ""), (0, "ok", "")])

        def _cb(name: str, payload: dict[str, object]) -> None:
            if name == "engine_fallback":
                raise RuntimeError("fallback cb boom")

        result = execute_task(plan, task, run_path, event_callback=_cb)
        # Callback raised but fallback continued and the retry succeeded.
        assert result.status == "success"


# ---------------------------------------------------------------------------
# context compression block (7870-7871, 7874, 7877, 7880, 7884, 7888)
# ---------------------------------------------------------------------------


class TestContextCompression:
    def test_compress_before_triggers_compression(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            compress_before=True,
            max_retries=1,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        # First fails (generic), second succeeds. compress_before forces the
        # compression block to run after the first failure.
        _patch_stream(monkeypatch, [(1, "generic boom", ""), (0, "ok", "")])

        compress_calls: list[int] = []
        orig_text = "maestro_cli.runners._compress_context_for_retry"

        def _spy(text: str, level: int) -> str:
            compress_calls.append(level)
            return text

        monkeypatch.setattr(orig_text, _spy)
        monkeypatch.setattr(
            "maestro_cli.runners._compress_upstream_context_for_retry",
            lambda ur, level: ur,
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        # Compression ran at level 1 (retry context synthesis + workspace brief).
        assert compress_calls and compress_calls[0] == 1

    def test_context_exceeded_triggers_compression(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            max_retries=1,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)

        # The standard "Task failed with exit code N" message always classifies
        # as test_failure (the word "failed" matches first).  To reach the
        # context_exceeded classification we route the failure through the
        # ``except Exception`` path, whose message is "Execution error: ..." and
        # carries a context-window phrase that classifies cleanly.
        call_count = {"n": 0}

        def _fake_stream(*args: Any, **kwargs: Any) -> tuple[int, str, str]:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("token limit reached")
            return (0, "ok", "")

        monkeypatch.setattr("maestro_cli.runners._stream_process", _fake_stream)

        levels: list[int] = []
        monkeypatch.setattr(
            "maestro_cli.runners._compress_context_for_retry",
            lambda text, level: (levels.append(level) or text),
        )
        monkeypatch.setattr(
            "maestro_cli.runners._compress_upstream_context_for_retry",
            lambda ur, level: ur,
        )

        result = execute_task(plan, task, run_path)
        assert result.status == "success"
        assert levels and levels[0] == 1


# ---------------------------------------------------------------------------
# task_escalation callback exception (7922-7923)
# ---------------------------------------------------------------------------


class TestEscalationCallback:
    def test_escalation_callback_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        plan = _make_plan(tmp_path)
        task = TaskSpec(
            id="t",
            engine="claude",
            model="haiku",
            prompt="do it",
            escalation=["haiku", "sonnet"],
            max_retries=1,
        )
        _patch_build(monkeypatch)
        _patch_popen(monkeypatch)
        # First fails, second (escalated) succeeds.
        _patch_stream(monkeypatch, [(1, "boom", ""), (0, "ok", "")])

        def _cb(name: str, payload: dict[str, object]) -> None:
            if name == "task_escalation":
                raise RuntimeError("escalation cb boom")

        result = execute_task(plan, task, run_path, event_callback=_cb)
        # Callback raised but escalation continued and the retry succeeded.
        assert result.status == "success"
        assert result.retry_count == 1
