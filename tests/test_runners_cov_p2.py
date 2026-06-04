"""Targeted line-coverage tests for selected helpers in ``runners.py``.

Each test drives the real function with crafted inputs and mocks only the
external boundaries (os attribute lookups, ``shutil.which``, ``Path.exists``,
``resolve_path``).  No subprocess / engine / network call is ever made.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest

from maestro_cli import runners
from maestro_cli.errors import TaskExecutionError
from maestro_cli.models import PlanSpec, TaskResult, TaskSpec
from maestro_cli.runners import (
    _apply_execution_profile,
    _claude_json_is_success,
    _classify_failure,
    _coerce_exit_code,
    _filter_context_fields,
    _find_git_bash,
    _load_prompt,
    _normalize_model_for_pricing,
    _resolve_prompt_path,
    _SignalHandler,
    _structured_tool_failure_count,
    _structured_tool_payload_failed,
    resolve_workdir,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(**overrides: Any) -> TaskResult:
    base: dict[str, Any] = {
        "task_id": "t1",
        "status": "success",
        "exit_code": 0,
        "duration_sec": 1.5,
        "cost_usd": 0.25,
        "stdout_tail": "hello",
    }
    base.update(overrides)
    return TaskResult(**base)


class _OsShim:
    """Proxy to the real ``os`` module but reporting ``name = 'nt'``.

    Replacing ``runners.os`` with this avoids mutating the real
    ``os.name`` (which pathlib reads on some platforms), keeping the test
    portable across operating systems.
    """

    name = "nt"

    def __getattr__(self, key: str) -> Any:
        return getattr(os, key)


# ---------------------------------------------------------------------------
# _filter_context_fields — full allowlist -> no overrides -> passthrough
# ---------------------------------------------------------------------------


class TestFilterContextFields:
    def test_full_allowlist_returns_same_object(self) -> None:
        # When the allowlist contains every filterable field, ``overrides``
        # stays empty and the original object is returned unchanged.
        result = _make_result()
        allowlist = [
            "stdout_tail",
            "exit_code",
            "status",
            "duration_sec",
            "cost_usd",
            "extra_field",
        ]
        out = _filter_context_fields(result, allowlist)
        assert out is result
        assert out.stdout_tail == "hello"
        assert out.cost_usd == 0.25

    def test_partial_allowlist_zeroes_missing_fields(self) -> None:
        # Sanity check the contrasting path returns a *new* replaced object.
        result = _make_result()
        out = _filter_context_fields(result, ["status"])
        assert out is not result
        assert out.stdout_tail == ""
        assert out.exit_code == 0
        assert out.cost_usd is None

    def test_empty_allowlist_short_circuits(self) -> None:
        result = _make_result()
        assert _filter_context_fields(result, []) is result


# ---------------------------------------------------------------------------
# _classify_failure — exit code branches
# ---------------------------------------------------------------------------


class TestClassifyFailure:
    def test_exit_9009_is_dependency_missing(self) -> None:
        assert _classify_failure(9009, "", "") == "dependency_missing"

    def test_exit_3_without_patterns_is_runtime_error(self) -> None:
        # No regex pattern matches, exit code 3 -> runtime_error.
        assert _classify_failure(3, "plain output", "plain message") == "runtime_error"

    def test_exit_124_is_timeout(self) -> None:
        assert _classify_failure(124, "", "") == "timeout"

    def test_unknown_when_nothing_matches(self) -> None:
        assert _classify_failure(1, "ordinary output", "ordinary") == "unknown"


# ---------------------------------------------------------------------------
# _claude_json_is_success — JSON edge branches
# ---------------------------------------------------------------------------


class TestClaudeJsonIsSuccess:
    def test_is_error_true_returns_false(self) -> None:
        assert _claude_json_is_success('{"is_error": true}') is False

    def test_invalid_last_line_continues_to_valid_earlier_line(self) -> None:
        # The reversed scan hits an unparseable ``{``-line first (continue),
        # then a valid result object -> True.
        out = '{"result": "ok"}\n{not valid json'
        assert _claude_json_is_success(out) is True

    def test_no_iserror_no_result_returns_false(self) -> None:
        # First valid object has neither is_error nor result -> return False.
        assert _claude_json_is_success('{"foo": 1}') is False

    def test_no_json_object_returns_false(self) -> None:
        assert _claude_json_is_success("not json at all") is False

    def test_is_error_false_returns_true(self) -> None:
        assert _claude_json_is_success('{"is_error": false}') is True


# ---------------------------------------------------------------------------
# _coerce_exit_code — all input-type branches
# ---------------------------------------------------------------------------


class TestCoerceExitCode:
    def test_none_returns_none(self) -> None:
        assert _coerce_exit_code(None) is None

    def test_bool_returns_none(self) -> None:
        # bool is an int subclass but is treated as non-numeric here.
        assert _coerce_exit_code(True) is None
        assert _coerce_exit_code(False) is None

    def test_int_passthrough(self) -> None:
        assert _coerce_exit_code(7) == 7

    def test_float_truncates_to_int(self) -> None:
        assert _coerce_exit_code(2.9) == 2

    def test_str_numeric_parsed(self) -> None:
        assert _coerce_exit_code("  3 ") == 3

    def test_str_empty_returns_none(self) -> None:
        assert _coerce_exit_code("   ") is None

    def test_str_non_numeric_returns_none(self) -> None:
        assert _coerce_exit_code("abc") is None

    def test_unsupported_type_returns_none(self) -> None:
        assert _coerce_exit_code([1, 2]) is None


# ---------------------------------------------------------------------------
# _structured_tool_payload_failed — every failure signal
# ---------------------------------------------------------------------------


class TestStructuredToolPayloadFailed:
    def test_is_error_true(self) -> None:
        assert _structured_tool_payload_failed({"is_error": True}) is True

    def test_success_false(self) -> None:
        assert _structured_tool_payload_failed({"success": False}) is True

    def test_ok_false(self) -> None:
        assert _structured_tool_payload_failed({"ok": False}) is True

    def test_status_in_failure_set(self) -> None:
        assert _structured_tool_payload_failed({"status": "error"}) is True
        assert _structured_tool_payload_failed({"status": "TIMED_OUT"}) is True

    def test_nonzero_exit_code(self) -> None:
        assert _structured_tool_payload_failed({"exit_code": 1}) is True
        assert _structured_tool_payload_failed({"returncode": "2"}) is True

    def test_error_string(self) -> None:
        assert _structured_tool_payload_failed({"error": "boom"}) is True

    def test_error_dict(self) -> None:
        assert _structured_tool_payload_failed({"error": {"code": 5}}) is True

    def test_clean_payload_returns_false(self) -> None:
        assert _structured_tool_payload_failed({"status": "ok", "exit_code": 0}) is False

    def test_empty_error_string_not_failure(self) -> None:
        # Empty error string and empty error dict are not failures.
        assert _structured_tool_payload_failed({"error": ""}) is False
        assert _structured_tool_payload_failed({"error": {}}) is False


# ---------------------------------------------------------------------------
# _structured_tool_failure_count — tool_result event + content-list branches
# ---------------------------------------------------------------------------


class TestStructuredToolFailureCount:
    def test_tool_result_event_failure(self) -> None:
        # event_type == "tool_result" and the event itself reports failure.
        event = {"type": "tool_result", "is_error": True}
        assert _structured_tool_failure_count(event) == 1

    def test_tool_result_event_clean(self) -> None:
        event = {"type": "tool_result", "status": "ok"}
        assert _structured_tool_failure_count(event) == 0

    def test_content_list_with_failing_tool_result(self) -> None:
        # Top-level "content" list containing a failing tool_result part.
        event = {"content": [{"type": "tool_result", "status": "error"}]}
        assert _structured_tool_failure_count(event) == 1

    def test_content_list_part_not_dict_ignored(self) -> None:
        event = {"content": ["raw string part", {"type": "tool_result", "ok": False}]}
        assert _structured_tool_failure_count(event) == 1

    def test_content_list_clean_parts(self) -> None:
        event = {"content": [{"type": "text", "text": "hi"}]}
        assert _structured_tool_failure_count(event) == 0


# ---------------------------------------------------------------------------
# _SignalHandler — _emit and individual signal handlers
# ---------------------------------------------------------------------------


class TestSignalHandlerEmit:
    def test_emit_swallows_callback_exception(self) -> None:
        def _boom(_event: str, _payload: dict[str, object]) -> None:
            raise RuntimeError("callback failure")

        handler = _SignalHandler(
            task_id="t1",
            workdir=Path("."),
            event_callback=_boom,
        )
        # Should not raise — the exception is swallowed.
        handler._emit("anything", {"k": "v"})

    def test_emit_without_callback_is_noop(self) -> None:
        handler = _SignalHandler(task_id="t1", workdir=Path("."))
        handler._emit("anything", {})  # no callback -> nothing happens


class TestSignalHandlerArtifact:
    def test_empty_path_returns_early(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        handler = _SignalHandler(
            task_id="t1",
            workdir=Path("."),
            event_callback=lambda e, p: events.append((e, p)),
        )
        handler._handle_artifact({"path": ""})
        assert handler.artifacts == []
        assert events == []

    def test_valid_relative_path_recorded(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        handler = _SignalHandler(
            task_id="t1",
            workdir=Path("."),
            event_callback=lambda e, p: events.append((e, p)),
        )
        handler._handle_artifact({"path": "out/report.txt", "label": "Report"})
        assert handler.artifacts == [{"path": "out/report.txt", "label": "Report"}]
        assert events and events[0][0] == "task_artifact"


class TestSignalHandlerTimeoutExtend:
    def test_non_numeric_additional_returns_early(self) -> None:
        handler = _SignalHandler(
            task_id="t1", workdir=Path("."), deadline_ref=[100.0]
        )
        handler._handle_timeout_extend({"additional_sec": "lots"})
        assert handler._deadline_ref == [100.0]

    def test_zero_additional_returns_early(self) -> None:
        handler = _SignalHandler(
            task_id="t1", workdir=Path("."), deadline_ref=[100.0]
        )
        handler._handle_timeout_extend({"additional_sec": 0})
        assert handler._deadline_ref == [100.0]

    def test_positive_additional_extends_deadline(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        handler = _SignalHandler(
            task_id="t1",
            workdir=Path("."),
            deadline_ref=[100.0],
            event_callback=lambda e, p: events.append((e, p)),
        )
        handler._handle_timeout_extend({"additional_sec": 30, "reason": "slow"})
        assert handler._deadline_ref[0] == 130.0
        assert events and events[0][0] == "timeout_extended"


class TestSignalHandlerBudgetQuery:
    def test_budget_getter_exception_swallowed(self) -> None:
        def _boom() -> tuple[float | None, float | None]:
            raise RuntimeError("budget unavailable")

        events: list[tuple[str, dict[str, object]]] = []
        handler = _SignalHandler(
            task_id="t1",
            workdir=Path("."),
            budget_getter=_boom,
            event_callback=lambda e, p: events.append((e, p)),
        )
        handler._handle_budget_query({})
        # Event still emitted with None values despite the getter blowing up.
        assert events and events[0][0] == "budget_query"
        assert events[0][1]["remaining_usd"] is None
        assert events[0][1]["limit_usd"] is None


class TestSignalHandlerCheckpoint:
    def test_non_dict_data_defaults_to_empty(self) -> None:
        events: list[tuple[str, dict[str, object]]] = []
        handler = _SignalHandler(
            task_id="t1",
            workdir=Path("."),
            event_callback=lambda e, p: events.append((e, p)),
        )
        handler._handle_checkpoint({"name": "cp1", "data": "not-a-dict"})
        assert events and events[0][0] == "task_checkpoint_signal"
        assert events[0][1]["data"] == {}


# ---------------------------------------------------------------------------
# _find_git_bash — PATH fallback branch (Windows-only logic)
# ---------------------------------------------------------------------------


class TestFindGitBash:
    def test_path_fallback_git_bash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force the "nt" branch via an os shim (no real os.name mutation),
        # make the search paths absent, and have shutil.which return a
        # git-flavoured bash path.
        monkeypatch.setattr(runners, "os", _OsShim())
        monkeypatch.setattr(runners.Path, "exists", lambda self: False)
        monkeypatch.setattr(
            runners.shutil,
            "which",
            lambda _name: "C:\\custom\\Git\\bin\\bash.exe",
        )
        assert _find_git_bash() == "C:\\custom\\Git\\bin\\bash.exe"

    def test_path_fallback_non_git_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runners, "os", _OsShim())
        monkeypatch.setattr(runners.Path, "exists", lambda self: False)
        monkeypatch.setattr(
            runners.shutil, "which", lambda _name: "C:\\msys\\bin\\bash.exe"
        )
        assert _find_git_bash() is None

    def test_path_fallback_which_none_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runners, "os", _OsShim())
        monkeypatch.setattr(runners.Path, "exists", lambda self: False)
        monkeypatch.setattr(runners.shutil, "which", lambda _name: None)
        assert _find_git_bash() is None

    def test_non_nt_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class _PosixShim(_OsShim):
            name = "posix"

        monkeypatch.setattr(runners, "os", _PosixShim())
        assert _find_git_bash() is None


# ---------------------------------------------------------------------------
# _apply_execution_profile — trailing fall-through (unexpected profile)
# ---------------------------------------------------------------------------


class TestApplyExecutionProfileFallthrough:
    @pytest.mark.parametrize(
        "engine",
        ["codex", "claude", "gemini", "copilot", "qwen"],
    )
    def test_unexpected_profile_returns_args_copy(self, engine: str) -> None:
        # An execution_profile outside {plan, safe, yolo} falls through to the
        # trailing ``return out`` for each engine branch.
        args = ["--flag", "value"]
        out = _apply_execution_profile(engine, args, "bogus")  # type: ignore[arg-type]
        assert out == args

    def test_plan_profile_returns_args_unchanged(self) -> None:
        args = ["--x"]
        assert _apply_execution_profile("codex", args, "plan") is args

    def test_local_engine_returns_args_unchanged(self) -> None:
        args = ["--x"]
        assert _apply_execution_profile("ollama", args, "safe") is args
        assert _apply_execution_profile("llama", args, "yolo") is args

    def test_unknown_engine_returns_args(self) -> None:
        args = ["--x"]
        assert _apply_execution_profile("mystery", args, "safe") is args  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# resolve_workdir — unresolvable workdir raises
# ---------------------------------------------------------------------------


class TestResolveWorkdir:
    def test_unresolvable_workdir_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        plan = PlanSpec(name="p")
        task = TaskSpec(id="t1", workdir="some/dir")
        # Force resolve_path to return None for a truthy workdir.
        monkeypatch.setattr(runners, "resolve_path", lambda *_a, **_k: None)
        with pytest.raises(TaskExecutionError, match="unable to resolve workdir"):
            resolve_workdir(plan, task)

    def test_workspace_root_used_when_no_task_workdir(self, tmp_path: Path) -> None:
        plan = PlanSpec(name="p", workspace_root=str(tmp_path))
        task = TaskSpec(id="t1")
        out = resolve_workdir(plan, task)
        assert out == tmp_path.resolve()

    def test_defaults_to_cwd(self) -> None:
        plan = PlanSpec(name="p")
        task = TaskSpec(id="t1")
        assert resolve_workdir(plan, task) == Path.cwd()


# ---------------------------------------------------------------------------
# _resolve_prompt_path — empty path returns None
# ---------------------------------------------------------------------------


class TestResolvePromptPath:
    def test_empty_relative_path_returns_none(self) -> None:
        plan = PlanSpec(name="p")
        assert _resolve_prompt_path(plan, "") is None

    def test_absolute_path_returned_directly(self, tmp_path: Path) -> None:
        plan = PlanSpec(name="p")
        abs_path = tmp_path / "x.md"
        out = _resolve_prompt_path(plan, str(abs_path))
        assert out == abs_path


# ---------------------------------------------------------------------------
# _load_prompt — no prompt source raises
# ---------------------------------------------------------------------------


class TestLoadPromptNoSource:
    def test_no_prompt_source_raises(self) -> None:
        plan = PlanSpec(name="p")
        task = TaskSpec(id="t1", engine="claude")
        with pytest.raises(TaskExecutionError, match="no prompt source"):
            _load_prompt(plan, task)

    def test_inline_prompt_loads(self) -> None:
        plan = PlanSpec(name="p")
        task = TaskSpec(id="t1", engine="claude", prompt="hello {{ task_id }}")
        out = _load_prompt(plan, task)
        assert "hello t1" in out


# ---------------------------------------------------------------------------
# _normalize_model_for_pricing — None passthrough
# ---------------------------------------------------------------------------


class TestNormalizeModelForPricing:
    def test_none_model_returns_none(self) -> None:
        assert _normalize_model_for_pricing(None) is None

    def test_alias_normalized(self) -> None:
        # "5.4" alias resolves to gpt-5.4-codex (already canonical for pricing).
        out = _normalize_model_for_pricing("5.4")
        assert isinstance(out, str)

    def test_pricing_alias_applied(self) -> None:
        # A raw model label that maps via _CODEX_PRICING_MODEL_ALIASES.
        out = _normalize_model_for_pricing("gpt-5.4")
        assert out == "gpt-5.4-codex"
