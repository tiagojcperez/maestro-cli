"""Tests for AG-UI protocol adapter (ag_ui.py)."""
from __future__ import annotations

import json

import pytest

from maestro_cli.ag_ui import (
    CUSTOM,
    RUN_FINISHED,
    RUN_STARTED,
    STATE_DELTA,
    STATE_SNAPSHOT,
    STEP_FINISHED,
    STEP_STARTED,
    TEXT_MESSAGE_CONTENT,
    TEXT_MESSAGE_END,
    TEXT_MESSAGE_START,
    TOOL_CALL_END,
    TOOL_CALL_START,
    AgUiRunState,
    format_sse,
    translate_event,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state(task_ids: list[str] | None = None) -> AgUiRunState:
    return AgUiRunState(
        run_id="run-1",
        thread_id="thread-1",
        task_ids=task_ids or ["a", "b", "c"],
        task_statuses={t: "pending" for t in (task_ids or ["a", "b", "c"])},
    )


def _types(events: list[dict]) -> list[str]:
    return [e["type"] for e in events]


# ---------------------------------------------------------------------------
# Lifecycle events
# ---------------------------------------------------------------------------


class TestLifecycleEvents:
    def test_run_start_suppressed(self) -> None:
        """run_start is suppressed (generator emits RUN_STARTED)."""
        state = _state()
        evts = translate_event("run_start", {"tasks": 3}, state)
        assert evts == []
        # But state is reset
        assert state.total_cost_usd == 0.0

    def test_run_complete_suppressed(self) -> None:
        """run_complete is suppressed (generator emits RUN_FINISHED)."""
        state = _state()
        evts = translate_event("run_complete", {"success": True}, state)
        assert evts == []


# ---------------------------------------------------------------------------
# Task lifecycle
# ---------------------------------------------------------------------------


class TestTaskEvents:
    def test_task_start_produces_step_and_message(self) -> None:
        state = _state()
        evts = translate_event("task_start", {"task_id": "a"}, state)
        types = _types(evts)
        assert STEP_STARTED in types
        assert TEXT_MESSAGE_START in types
        assert STATE_DELTA in types
        assert state.task_statuses["a"] == "running"

    def test_task_complete_success(self) -> None:
        state = _state()
        # Start first to open message
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_complete", {
            "task_id": "a", "status": "success",
            "cost_usd": 0.05, "duration_sec": 12.3, "total_tokens": 1500,
        }, state)
        types = _types(evts)
        assert TEXT_MESSAGE_END in types
        assert STEP_FINISHED in types
        assert STATE_DELTA in types
        assert state.task_statuses["a"] == "success"
        assert state.completed_count == 1
        assert state.total_cost_usd == pytest.approx(0.05)
        assert state.total_tokens == 1500

    def test_task_complete_failed(self) -> None:
        state = _state()
        translate_event("task_start", {"task_id": "b"}, state)
        evts = translate_event("task_complete", {
            "task_id": "b", "status": "failed",
        }, state)
        assert state.task_statuses["b"] == "failed"

    def test_task_skip(self) -> None:
        state = _state()
        evts = translate_event("task_skip", {
            "task_id": "c", "reason": "dependency failed",
        }, state)
        types = _types(evts)
        assert STEP_FINISHED in types
        assert CUSTOM in types
        custom = [e for e in evts if e["type"] == CUSTOM][0]
        assert custom["name"] == "task_skip"
        assert state.task_statuses["c"] == "skipped"


# ---------------------------------------------------------------------------
# Task output
# ---------------------------------------------------------------------------


class TestTaskOutput:
    def test_output_with_active_message(self) -> None:
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_output", {
            "task_id": "a", "line": "Hello world",
        }, state)
        assert len(evts) == 1
        assert evts[0]["type"] == TEXT_MESSAGE_CONTENT
        assert "Hello world" in evts[0]["delta"]

    def test_output_without_active_message_ignored(self) -> None:
        state = _state()
        evts = translate_event("task_output", {
            "task_id": "a", "line": "orphan",
        }, state)
        assert evts == []


# ---------------------------------------------------------------------------
# Tool calls
# ---------------------------------------------------------------------------


class TestToolCalls:
    def test_tool_call_produces_start_and_end(self) -> None:
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_tool_call", {
            "task_id": "a", "tool": "Edit", "input_preview": "file.py",
        }, state)
        types = _types(evts)
        assert TOOL_CALL_START in types
        assert TOOL_CALL_END in types
        start = [e for e in evts if e["type"] == TOOL_CALL_START][0]
        assert start["toolCallName"] == "Edit"


# ---------------------------------------------------------------------------
# Suppressed events
# ---------------------------------------------------------------------------


class TestSuppressedEvents:
    @pytest.mark.parametrize("event_name", [
        "context_compression", "context_budget_trim", "context_summarize",
        "context_recursive", "worktree_cleanup", "deliberation_skip",
        "task_checkpoint", "webhook",
    ])
    def test_suppressed_events_produce_nothing(self, event_name: str) -> None:
        state = _state()
        evts = translate_event(event_name, {}, state)
        assert evts == []


# ---------------------------------------------------------------------------
# Custom (domain) events
# ---------------------------------------------------------------------------


class TestCustomEvents:
    @pytest.mark.parametrize("event_name", [
        "task_retry", "task_escalation", "engine_fallback",
        "judge_start", "judge_result",
        "approval_required", "approval_response",
        "budget_warning", "budget_exceeded",
        "policy_violation", "taint_detected",
        "circuit_breaker_tripped", "model_routed",
        "verify_failure",
    ])
    def test_domain_events_become_custom(self, event_name: str) -> None:
        state = _state()
        evts = translate_event(event_name, {"task_id": "a", "detail": "x"}, state)
        assert len(evts) == 1
        assert evts[0]["type"] == CUSTOM
        assert evts[0]["name"] == event_name


# ---------------------------------------------------------------------------
# Watch events
# ---------------------------------------------------------------------------


class TestWatchEvents:
    def test_watch_start(self) -> None:
        state = _state()
        evts = translate_event("watch_start", {}, state)
        assert _types(evts) == [RUN_STARTED]

    def test_watch_complete(self) -> None:
        state = _state()
        evts = translate_event("watch_complete", {"status": "converged"}, state)
        assert _types(evts) == [RUN_FINISHED]

    def test_iteration_start(self) -> None:
        state = _state()
        evts = translate_event("iteration_start", {"iteration": 3}, state)
        assert evts[0]["type"] == STEP_STARTED
        assert evts[0]["stepName"] == "iteration_3"

    def test_iteration_complete(self) -> None:
        state = _state()
        evts = translate_event("iteration_complete", {"iteration": 3}, state)
        assert evts[0]["type"] == STEP_FINISHED
        assert evts[0]["stepName"] == "iteration_3"


# ---------------------------------------------------------------------------
# Dynamic sub-plan events
# ---------------------------------------------------------------------------


class TestDynamicEvents:
    def test_dynamic_subplan_start(self) -> None:
        state = _state()
        evts = translate_event("dynamic_subplan_start", {"task_id": "a"}, state)
        assert evts[0]["type"] == STEP_STARTED
        assert evts[0]["stepName"] == "a/dynamic"

    def test_dynamic_subplan_complete(self) -> None:
        state = _state()
        evts = translate_event("dynamic_subplan_complete", {"task_id": "a"}, state)
        assert evts[0]["type"] == STEP_FINISHED


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestStateManagement:
    def test_initial_snapshot(self) -> None:
        state = _state(["x", "y"])
        snap = state.to_snapshot()
        assert snap["progress"] == 0
        assert snap["tasks"]["x"]["status"] == "pending"
        assert snap["tasks"]["y"]["status"] == "pending"

    def test_progress_updates(self) -> None:
        state = _state(["a", "b", "c", "d"])
        assert state.progress_pct == 0
        state.completed_count = 2
        assert state.progress_pct == 50
        state.completed_count = 4
        assert state.progress_pct == 100

    def test_state_delta_has_patches(self) -> None:
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_complete", {
            "task_id": "a", "status": "success", "cost_usd": 0.1,
        }, state)
        deltas = [e for e in evts if e["type"] == STATE_DELTA]
        assert len(deltas) == 1
        patches = deltas[0]["delta"]
        ops = {p["path"] for p in patches}
        assert "/tasks/a/status" in ops
        assert "/progress" in ops
        assert "/totalCostUsd" in ops


# ---------------------------------------------------------------------------
# SSE wire format
# ---------------------------------------------------------------------------


class TestSseFormat:
    def test_format_sse_produces_data_line(self) -> None:
        line = format_sse({"type": "RUN_STARTED", "runId": "r1"})
        assert line.startswith("data: ")
        assert line.endswith("\n\n")
        payload = json.loads(line[len("data: "):-2])
        assert payload["type"] == "RUN_STARTED"
        assert payload["runId"] == "r1"

    def test_format_sse_no_event_prefix(self) -> None:
        line = format_sse({"type": "CUSTOM", "name": "test"})
        # AG-UI uses bare data: lines (no event: prefix)
        assert not line.startswith("event:")


# ---------------------------------------------------------------------------
# Fallback: unknown events
# ---------------------------------------------------------------------------


class TestFallback:
    def test_unknown_event_becomes_custom(self) -> None:
        state = _state()
        evts = translate_event("some_future_event", {"foo": "bar"}, state)
        assert len(evts) == 1
        assert evts[0]["type"] == CUSTOM
        assert evts[0]["name"] == "some_future_event"
        assert evts[0]["value"]["foo"] == "bar"


# ---------------------------------------------------------------------------
# AgUiRunState edge cases
# ---------------------------------------------------------------------------


class TestAgUiRunStateDefaults:
    """Verify default field values and basic state operations."""

    def test_default_values(self) -> None:
        state = AgUiRunState(run_id="r", thread_id="t")
        assert state.total_cost_usd == 0.0
        assert state.total_tokens == 0
        assert state.completed_count == 0
        assert state.task_ids == []
        assert state.task_statuses == {}
        assert state.task_costs == {}
        assert state.task_durations == {}

    def test_progress_zero_tasks(self) -> None:
        """progress_pct returns 0 when task_ids is empty (no division by zero)."""
        state = AgUiRunState(run_id="r", thread_id="t", task_ids=[])
        assert state.progress_pct == 0

    def test_progress_one_task_complete(self) -> None:
        state = AgUiRunState(run_id="r", thread_id="t", task_ids=["a"])
        state.completed_count = 1
        assert state.progress_pct == 100

    def test_progress_partial(self) -> None:
        state = AgUiRunState(run_id="r", thread_id="t", task_ids=["a", "b", "c"])
        state.completed_count = 1
        assert state.progress_pct == 33  # int(1/3 * 100)

    def test_progress_all_complete(self) -> None:
        state = _state(["a", "b", "c", "d", "e"])
        state.completed_count = 5
        assert state.progress_pct == 100


class TestSnapshotFormat:
    """Verify to_snapshot() dict shape."""

    def test_snapshot_keys(self) -> None:
        state = _state(["x"])
        snap = state.to_snapshot()
        assert set(snap.keys()) == {"runId", "progress", "tasks", "totalCostUsd", "totalTokens"}

    def test_snapshot_run_id(self) -> None:
        state = _state(["x"])
        snap = state.to_snapshot()
        assert snap["runId"] == "run-1"

    def test_snapshot_task_pending_has_status_only(self) -> None:
        """Pending tasks should only have status (no costUsd or durationSec)."""
        state = _state(["x"])
        snap = state.to_snapshot()
        assert snap["tasks"]["x"] == {"status": "pending"}

    def test_snapshot_includes_cost_and_duration(self) -> None:
        state = _state(["x"])
        state.task_statuses["x"] = "success"
        state.task_costs["x"] = 0.42
        state.task_durations["x"] = 7.5
        snap = state.to_snapshot()
        assert snap["tasks"]["x"]["costUsd"] == 0.42
        assert snap["tasks"]["x"]["durationSec"] == 7.5

    def test_snapshot_cost_none_excluded(self) -> None:
        """Cost of None should NOT appear in the task snapshot."""
        state = _state(["x"])
        state.task_costs["x"] = None
        snap = state.to_snapshot()
        assert "costUsd" not in snap["tasks"]["x"]

    def test_snapshot_duration_none_excluded(self) -> None:
        state = _state(["x"])
        state.task_durations["x"] = None
        snap = state.to_snapshot()
        assert "durationSec" not in snap["tasks"]["x"]

    def test_snapshot_total_tokens(self) -> None:
        state = _state(["x"])
        state.total_tokens = 5000
        snap = state.to_snapshot()
        assert snap["totalTokens"] == 5000

    def test_snapshot_total_cost(self) -> None:
        state = _state(["x"])
        state.total_cost_usd = 1.23
        snap = state.to_snapshot()
        assert snap["totalCostUsd"] == pytest.approx(1.23)


class TestMessageIdTracking:
    """Verify message ID creation, retrieval, and closing."""

    def test_next_message_id_format(self) -> None:
        state = _state()
        mid = state.next_message_id("a")
        assert mid == "run-1:a:msg"

    def test_get_message_id_returns_active(self) -> None:
        state = _state()
        state.next_message_id("a")
        assert state.get_message_id("a") == "run-1:a:msg"

    def test_get_message_id_returns_none_when_absent(self) -> None:
        state = _state()
        assert state.get_message_id("nonexistent") is None

    def test_close_message_returns_and_removes(self) -> None:
        state = _state()
        state.next_message_id("a")
        mid = state.close_message("a")
        assert mid == "run-1:a:msg"
        assert state.get_message_id("a") is None

    def test_close_message_returns_none_when_absent(self) -> None:
        state = _state()
        assert state.close_message("a") is None

    def test_tool_id_increments(self) -> None:
        state = _state()
        t1 = state.next_tool_id("a")
        t2 = state.next_tool_id("a")
        t3 = state.next_tool_id("b")
        assert t1 == "run-1:a:tool:1"
        assert t2 == "run-1:a:tool:2"
        assert t3 == "run-1:b:tool:3"


# ---------------------------------------------------------------------------
# State delta specifics
# ---------------------------------------------------------------------------


class TestStateDelta:
    """Verify JSON Patch operations in STATE_DELTA events."""

    def test_delta_after_task_start_has_running_status(self) -> None:
        state = _state()
        evts = translate_event("task_start", {"task_id": "a"}, state)
        deltas = [e for e in evts if e["type"] == STATE_DELTA]
        assert len(deltas) == 1
        patches = deltas[0]["delta"]
        status_patch = [p for p in patches if p["path"] == "/tasks/a/status"]
        assert len(status_patch) == 1
        assert status_patch[0]["value"] == "running"

    def test_delta_after_task_complete_has_final_status(self) -> None:
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_complete", {
            "task_id": "a", "status": "success",
            "cost_usd": 0.5, "duration_sec": 10.0,
        }, state)
        deltas = [e for e in evts if e["type"] == STATE_DELTA]
        assert len(deltas) == 1
        patches = deltas[0]["delta"]
        paths = {p["path"]: p["value"] for p in patches}
        assert paths["/tasks/a/status"] == "success"
        assert paths["/progress"] == 33  # 1/3 complete
        assert paths["/totalCostUsd"] == pytest.approx(0.5)
        assert paths["/tasks/a/costUsd"] == pytest.approx(0.5)
        assert paths["/tasks/a/durationSec"] == pytest.approx(10.0)

    def test_delta_no_cost_patch_when_cost_none(self) -> None:
        """If a task has no cost, the delta should NOT have a costUsd patch."""
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_complete", {
            "task_id": "a", "status": "success",
        }, state)
        deltas = [e for e in evts if e["type"] == STATE_DELTA]
        patches = deltas[0]["delta"]
        cost_patches = [p for p in patches if "costUsd" in p["path"]]
        # totalCostUsd is always present, but task-level costUsd should be absent
        task_cost = [p for p in patches if p["path"] == "/tasks/a/costUsd"]
        assert task_cost == []

    def test_delta_no_duration_patch_when_duration_none(self) -> None:
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_complete", {
            "task_id": "a", "status": "success",
        }, state)
        deltas = [e for e in evts if e["type"] == STATE_DELTA]
        patches = deltas[0]["delta"]
        dur_patches = [p for p in patches if p["path"] == "/tasks/a/durationSec"]
        assert dur_patches == []

    def test_delta_after_task_skip(self) -> None:
        state = _state()
        evts = translate_event("task_skip", {"task_id": "b", "reason": "dep"}, state)
        deltas = [e for e in evts if e["type"] == STATE_DELTA]
        assert len(deltas) == 1
        patches = deltas[0]["delta"]
        status_patch = [p for p in patches if p["path"] == "/tasks/b/status"]
        assert status_patch[0]["value"] == "skipped"


# ---------------------------------------------------------------------------
# Task complete — additional status variants
# ---------------------------------------------------------------------------


class TestTaskCompleteVariants:
    def test_soft_failed_status(self) -> None:
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_complete", {
            "task_id": "a", "status": "soft_failed", "cost_usd": 0.01,
        }, state)
        assert state.task_statuses["a"] == "soft_failed"
        assert state.completed_count == 1
        assert state.total_cost_usd == pytest.approx(0.01)

    def test_skipped_via_task_complete(self) -> None:
        """task_complete with status=skipped still updates progress."""
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        translate_event("task_complete", {
            "task_id": "a", "status": "skipped",
        }, state)
        assert state.task_statuses["a"] == "skipped"
        assert state.completed_count == 1

    def test_complete_without_start_no_message_end(self) -> None:
        """If task_complete fires without prior task_start, no TEXT_MESSAGE_END."""
        state = _state()
        evts = translate_event("task_complete", {
            "task_id": "a", "status": "success",
        }, state)
        types = _types(evts)
        assert TEXT_MESSAGE_END not in types
        assert STEP_FINISHED in types

    def test_cost_accumulation_multiple_tasks(self) -> None:
        state = _state()
        for tid in ["a", "b", "c"]:
            translate_event("task_start", {"task_id": tid}, state)
            translate_event("task_complete", {
                "task_id": tid, "status": "success", "cost_usd": 0.10,
                "total_tokens": 100,
            }, state)
        assert state.total_cost_usd == pytest.approx(0.30)
        assert state.total_tokens == 300
        assert state.completed_count == 3
        assert state.progress_pct == 100

    def test_cost_not_accumulated_when_not_numeric(self) -> None:
        """Non-numeric cost should be ignored (no crash)."""
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        translate_event("task_complete", {
            "task_id": "a", "status": "success", "cost_usd": "unknown",
        }, state)
        assert state.total_cost_usd == 0.0

    def test_tokens_not_accumulated_when_not_int(self) -> None:
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        translate_event("task_complete", {
            "task_id": "a", "status": "success", "total_tokens": "lots",
        }, state)
        assert state.total_tokens == 0

    def test_integer_cost_accepted(self) -> None:
        """Integer cost (not just float) should be accumulated."""
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        translate_event("task_complete", {
            "task_id": "a", "status": "success", "cost_usd": 1,
        }, state)
        assert state.total_cost_usd == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Task output edge cases
# ---------------------------------------------------------------------------


class TestTaskOutputEdgeCases:
    def test_output_empty_line_ignored(self) -> None:
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_output", {
            "task_id": "a", "line": "",
        }, state)
        assert evts == []

    def test_output_uses_output_field_fallback(self) -> None:
        """When 'line' is absent, 'output' field is used."""
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_output", {
            "task_id": "a", "output": "fallback text",
        }, state)
        assert len(evts) == 1
        assert "fallback text" in evts[0]["delta"]

    def test_output_no_line_no_output_ignored(self) -> None:
        """Missing both 'line' and 'output' produces nothing."""
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_output", {"task_id": "a"}, state)
        assert evts == []


# ---------------------------------------------------------------------------
# Tool call edge cases
# ---------------------------------------------------------------------------


class TestToolCallEdgeCases:
    def test_tool_call_missing_tool_name_defaults_unknown(self) -> None:
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_tool_call", {"task_id": "a"}, state)
        start = [e for e in evts if e["type"] == TOOL_CALL_START][0]
        assert start["toolCallName"] == "unknown"

    def test_tool_call_parent_message_id(self) -> None:
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_tool_call", {
            "task_id": "a", "tool": "Read",
        }, state)
        start = [e for e in evts if e["type"] == TOOL_CALL_START][0]
        assert start["parentMessageId"] == "run-1:a:msg"

    def test_tool_call_without_active_message(self) -> None:
        """Tool call still works even without an active message."""
        state = _state()
        evts = translate_event("task_tool_call", {
            "task_id": "a", "tool": "Bash",
        }, state)
        start = [e for e in evts if e["type"] == TOOL_CALL_START][0]
        assert start["parentMessageId"] is None


# ---------------------------------------------------------------------------
# Signal events
# ---------------------------------------------------------------------------


class TestSignalEvents:
    def test_progress_signal_produces_state_delta(self) -> None:
        state = _state()
        evts = translate_event("task_signal", {
            "task_id": "a", "signal_type": "progress", "pct": 42, "step": "linting",
        }, state)
        assert len(evts) == 1
        assert evts[0]["type"] == STATE_DELTA
        patches = evts[0]["delta"]
        assert len(patches) == 1
        assert patches[0]["path"] == "/tasks/a/progress"
        assert patches[0]["value"] == 42

    def test_non_progress_signal_becomes_custom(self) -> None:
        state = _state()
        evts = translate_event("task_signal", {
            "task_id": "a", "signal_type": "metric", "value": 0.95,
        }, state)
        assert len(evts) == 1
        assert evts[0]["type"] == CUSTOM
        assert evts[0]["name"] == "signal_metric"

    def test_signal_filters_event_and_plan_name(self) -> None:
        """event and plan_name keys should be stripped from signal value."""
        state = _state()
        evts = translate_event("task_signal", {
            "task_id": "a", "signal_type": "log",
            "event": "task_signal", "plan_name": "my-plan",
            "message": "hello",
        }, state)
        value = evts[0]["value"]
        assert "event" not in value
        assert "plan_name" not in value
        assert value["message"] == "hello"

    def test_signal_missing_type_defaults_empty(self) -> None:
        state = _state()
        evts = translate_event("task_signal", {"task_id": "a"}, state)
        assert evts[0]["type"] == CUSTOM
        assert evts[0]["name"] == "signal_"


# ---------------------------------------------------------------------------
# Watch event edge cases
# ---------------------------------------------------------------------------


class TestWatchEventEdgeCases:
    def test_watch_start_thread_and_run_ids(self) -> None:
        state = _state()
        evts = translate_event("watch_start", {}, state)
        assert evts[0]["threadId"] == "thread-1"
        assert evts[0]["runId"] == "run-1"

    def test_watch_complete_result_field(self) -> None:
        state = _state()
        evts = translate_event("watch_complete", {"status": "plateau"}, state)
        assert evts[0]["result"]["status"] == "plateau"

    def test_watch_complete_missing_status(self) -> None:
        state = _state()
        evts = translate_event("watch_complete", {}, state)
        assert evts[0]["result"]["status"] == ""

    def test_iteration_default_number(self) -> None:
        """If iteration is missing, defaults to 0."""
        state = _state()
        evts = translate_event("iteration_start", {}, state)
        assert evts[0]["stepName"] == "iteration_0"
        evts2 = translate_event("iteration_complete", {}, state)
        assert evts2[0]["stepName"] == "iteration_0"


# ---------------------------------------------------------------------------
# Domain custom events — payload filtering
# ---------------------------------------------------------------------------


class TestCustomEventPayloads:
    def test_event_and_plan_name_stripped_from_custom(self) -> None:
        state = _state()
        evts = translate_event("task_retry", {
            "task_id": "a", "attempt": 2, "max_retries": 3,
            "event": "task_retry", "plan_name": "my-plan",
        }, state)
        value = evts[0]["value"]
        assert "event" not in value
        assert "plan_name" not in value
        assert value["task_id"] == "a"
        assert value["attempt"] == 2

    def test_fallback_unknown_also_strips(self) -> None:
        """Completely unknown events also strip event/plan_name."""
        state = _state()
        evts = translate_event("totally_new_event", {
            "foo": 1, "event": "totally_new_event", "plan_name": "p",
        }, state)
        value = evts[0]["value"]
        assert "event" not in value
        assert "plan_name" not in value
        assert value["foo"] == 1

    @pytest.mark.parametrize("event_name", [
        "batch_chunk_complete", "worktree_create", "worktree_merge",
        "metric_recorded", "regression_detected", "rollback_executed",
        "plateau_detected", "target_reached", "timeout_adjusted",
    ])
    def test_remaining_custom_events_produce_custom(self, event_name: str) -> None:
        """All _CUSTOM_EVENTS members that were not in the original parametrize."""
        state = _state()
        evts = translate_event(event_name, {"task_id": "x"}, state)
        assert len(evts) == 1
        assert evts[0]["type"] == CUSTOM
        assert evts[0]["name"] == event_name


# ---------------------------------------------------------------------------
# Translate event — missing and None fields
# ---------------------------------------------------------------------------


class TestTranslateEventGraceful:
    def test_empty_payload(self) -> None:
        """translate_event with empty dict should not crash."""
        state = _state()
        # task_start with no task_id — tid becomes ""
        evts = translate_event("task_start", {}, state)
        assert len(evts) == 3
        assert state.task_statuses[""] == "running"

    def test_task_complete_none_values(self) -> None:
        """None cost/duration/tokens should not crash."""
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_complete", {
            "task_id": "a", "status": None,
            "cost_usd": None, "duration_sec": None, "total_tokens": None,
        }, state)
        assert state.task_statuses["a"] == "None"
        assert state.total_cost_usd == 0.0
        assert state.total_tokens == 0

    def test_task_start_missing_task_id_uses_empty_string(self) -> None:
        state = _state()
        evts = translate_event("task_start", {"other": "data"}, state)
        step = [e for e in evts if e["type"] == STEP_STARTED][0]
        assert step["stepName"] == ""

    def test_task_skip_missing_reason(self) -> None:
        state = _state()
        evts = translate_event("task_skip", {"task_id": "a"}, state)
        custom = [e for e in evts if e["type"] == CUSTOM][0]
        assert custom["value"]["reason"] == ""

    def test_duplicate_task_status_updates_idempotent(self) -> None:
        """Multiple task_start for same task should not crash."""
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        translate_event("task_start", {"task_id": "a"}, state)
        assert state.task_statuses["a"] == "running"
        # Two active messages — the second overwrites
        assert state.get_message_id("a") == "run-1:a:msg"


# ---------------------------------------------------------------------------
# Multiple events from one translate call — ordering
# ---------------------------------------------------------------------------


class TestEventOrdering:
    def test_task_start_order(self) -> None:
        """task_start: STEP_STARTED, TEXT_MESSAGE_START, STATE_DELTA."""
        state = _state()
        evts = translate_event("task_start", {"task_id": "a"}, state)
        types = _types(evts)
        assert types == [STEP_STARTED, TEXT_MESSAGE_START, STATE_DELTA]

    def test_task_complete_order_with_message(self) -> None:
        """task_complete: TEXT_MESSAGE_END, STEP_FINISHED, STATE_DELTA."""
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_complete", {
            "task_id": "a", "status": "success",
        }, state)
        types = _types(evts)
        assert types == [TEXT_MESSAGE_END, STEP_FINISHED, STATE_DELTA]

    def test_task_complete_order_without_message(self) -> None:
        """task_complete without prior start: STEP_FINISHED, STATE_DELTA only."""
        state = _state()
        evts = translate_event("task_complete", {
            "task_id": "a", "status": "failed",
        }, state)
        types = _types(evts)
        assert types == [STEP_FINISHED, STATE_DELTA]

    def test_task_skip_order(self) -> None:
        """task_skip: STEP_FINISHED, CUSTOM, STATE_DELTA."""
        state = _state()
        evts = translate_event("task_skip", {"task_id": "a"}, state)
        types = _types(evts)
        assert types == [STEP_FINISHED, CUSTOM, STATE_DELTA]

    def test_tool_call_order(self) -> None:
        """task_tool_call: TOOL_CALL_START, TOOL_CALL_END."""
        state = _state()
        translate_event("task_start", {"task_id": "a"}, state)
        evts = translate_event("task_tool_call", {
            "task_id": "a", "tool": "X",
        }, state)
        types = _types(evts)
        assert types == [TOOL_CALL_START, TOOL_CALL_END]


# ---------------------------------------------------------------------------
# format_sse edge cases
# ---------------------------------------------------------------------------


class TestSseFormatEdgeCases:
    def test_empty_dict(self) -> None:
        line = format_sse({})
        assert line == "data: {}\n\n"

    def test_special_characters_in_data(self) -> None:
        """Newlines and unicode in values should be JSON-escaped, not literal."""
        line = format_sse({"msg": "line1\nline2", "emoji": "\u2603"})
        # The data line itself should be a single line (no literal newlines in JSON)
        data_part = line[len("data: "):-2]
        parsed = json.loads(data_part)
        assert parsed["msg"] == "line1\nline2"
        assert parsed["emoji"] == "\u2603"
        # Verify it's valid SSE (only one "data:" prefix)
        assert line.count("data:") == 1

    def test_large_payload_not_truncated(self) -> None:
        big_value = "x" * 100_000
        line = format_sse({"big": big_value})
        parsed = json.loads(line[len("data: "):-2])
        assert len(parsed["big"]) == 100_000

    def test_non_serializable_uses_default_str(self) -> None:
        """Non-JSON-serializable types should use str() via default=str."""
        from pathlib import Path
        line = format_sse({"path": Path("/foo/bar")})
        parsed = json.loads(line[len("data: "):-2])
        # Path.__str__() produces the path string
        assert "foo" in parsed["path"]

    def test_nested_dict(self) -> None:
        line = format_sse({"a": {"b": {"c": [1, 2, 3]}}})
        parsed = json.loads(line[len("data: "):-2])
        assert parsed["a"]["b"]["c"] == [1, 2, 3]


# ---------------------------------------------------------------------------
# run_start resets state
# ---------------------------------------------------------------------------


class TestRunStartReset:
    def test_run_start_resets_cost_and_tokens(self) -> None:
        state = _state()
        state.total_cost_usd = 99.9
        state.total_tokens = 50000
        translate_event("run_start", {}, state)
        assert state.total_cost_usd == 0.0
        assert state.total_tokens == 0

    def test_run_start_does_not_reset_completed_count(self) -> None:
        """run_start only resets cost/tokens, not completed_count or statuses."""
        state = _state()
        state.completed_count = 2
        state.task_statuses["a"] = "success"
        translate_event("run_start", {}, state)
        # completed_count is NOT reset by run_start
        assert state.completed_count == 2


# ---------------------------------------------------------------------------
# Dynamic sub-plan edge cases
# ---------------------------------------------------------------------------


class TestDynamicSubplanEdgeCases:
    def test_dynamic_subplan_step_name_format(self) -> None:
        state = _state()
        evts = translate_event("dynamic_subplan_start", {"task_id": "gen-1"}, state)
        assert evts[0]["stepName"] == "gen-1/dynamic"
        evts2 = translate_event("dynamic_subplan_complete", {"task_id": "gen-1"}, state)
        assert evts2[0]["stepName"] == "gen-1/dynamic"


# ---------------------------------------------------------------------------
# Timestamp presence
# ---------------------------------------------------------------------------


class TestTimestamps:
    def test_all_events_have_timestamp(self) -> None:
        """Every event dict from translate_event should have a 'timestamp' key."""
        state = _state()
        # Run through several event types
        for event_name, payload in [
            ("task_start", {"task_id": "a"}),
            ("task_output", {"task_id": "a", "line": "hi"}),
            ("task_tool_call", {"task_id": "a", "tool": "X"}),
            ("task_complete", {"task_id": "a", "status": "success"}),
            ("task_skip", {"task_id": "b", "reason": "dep"}),
            ("watch_start", {}),
            ("iteration_start", {"iteration": 1}),
            ("dynamic_subplan_start", {"task_id": "c"}),
            ("budget_warning", {"task_id": "a"}),
            ("some_unknown", {"data": True}),
        ]:
            evts = translate_event(event_name, payload, state)
            for evt in evts:
                assert "timestamp" in evt, f"Missing timestamp in {event_name} event: {evt}"
                assert isinstance(evt["timestamp"], int)
