"""AG-UI protocol adapter for Maestro CLI.

Translates Maestro's 48+ internal event types into the AG-UI event stream
format.  This module works with plain dicts (no external deps) so that the
translation logic can be tested and used without installing ``ag-ui-protocol``.

The FastAPI SSE endpoint (``web/routes_agui.py``) imports this module and
serialises the dicts into ``data: {json}\\n\\n`` lines per the AG-UI spec.

See https://docs.ag-ui.com and ``docs/PROTOCOL-ROADMAP.md`` for context.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# AG-UI event type constants (subset used by the Maestro adapter)
# ---------------------------------------------------------------------------

RUN_STARTED = "RUN_STARTED"
RUN_FINISHED = "RUN_FINISHED"
RUN_ERROR = "RUN_ERROR"
STEP_STARTED = "STEP_STARTED"
STEP_FINISHED = "STEP_FINISHED"
TEXT_MESSAGE_START = "TEXT_MESSAGE_START"
TEXT_MESSAGE_CONTENT = "TEXT_MESSAGE_CONTENT"
TEXT_MESSAGE_END = "TEXT_MESSAGE_END"
TOOL_CALL_START = "TOOL_CALL_START"
TOOL_CALL_END = "TOOL_CALL_END"
STATE_SNAPSHOT = "STATE_SNAPSHOT"
STATE_DELTA = "STATE_DELTA"
CUSTOM = "CUSTOM"

# Internal events that are suppressed (too low-level for UI consumers)
_SUPPRESSED_EVENTS: frozenset[str] = frozenset({
    "context_compression",
    "context_budget_trim",
    "context_summarize",
    "context_recursive",
    "worktree_cleanup",
    "deliberation_skip",
    "task_checkpoint",
    "webhook",
})


# ---------------------------------------------------------------------------
# Run state tracker
# ---------------------------------------------------------------------------

@dataclass
class AgUiRunState:
    """Tracks cumulative run state for AG-UI state events."""

    run_id: str
    thread_id: str
    task_ids: list[str] = field(default_factory=list)
    task_statuses: dict[str, str] = field(default_factory=dict)
    task_costs: dict[str, float | None] = field(default_factory=dict)
    task_durations: dict[str, float | None] = field(default_factory=dict)
    completed_count: int = 0
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    _active_messages: dict[str, str] = field(default_factory=dict)  # task_id -> message_id
    _tool_seq: int = 0

    @property
    def progress_pct(self) -> int:
        if not self.task_ids:
            return 0
        return int(self.completed_count / len(self.task_ids) * 100)

    def to_snapshot(self) -> dict[str, Any]:
        """Full state snapshot for AG-UI STATE_SNAPSHOT."""
        tasks: dict[str, dict[str, Any]] = {}
        for tid in self.task_ids:
            t: dict[str, Any] = {"status": self.task_statuses.get(tid, "pending")}
            if self.task_costs.get(tid) is not None:
                t["costUsd"] = self.task_costs[tid]
            if self.task_durations.get(tid) is not None:
                t["durationSec"] = self.task_durations[tid]
            tasks[tid] = t
        return {
            "runId": self.run_id,
            "progress": self.progress_pct,
            "tasks": tasks,
            "totalCostUsd": self.total_cost_usd,
            "totalTokens": self.total_tokens,
        }

    def next_message_id(self, task_id: str) -> str:
        mid = f"{self.run_id}:{task_id}:msg"
        self._active_messages[task_id] = mid
        return mid

    def get_message_id(self, task_id: str) -> str | None:
        return self._active_messages.get(task_id)

    def close_message(self, task_id: str) -> str | None:
        return self._active_messages.pop(task_id, None)

    def next_tool_id(self, task_id: str) -> str:
        self._tool_seq += 1
        return f"{self.run_id}:{task_id}:tool:{self._tool_seq}"


# ---------------------------------------------------------------------------
# Event translation
# ---------------------------------------------------------------------------

def _ts() -> int:
    """Current timestamp in milliseconds (AG-UI convention)."""
    import time
    return int(time.time() * 1000)


def _ev(event_type: str, **kwargs: Any) -> dict[str, Any]:
    """Build an AG-UI event dict."""
    d: dict[str, Any] = {"type": event_type, "timestamp": _ts()}
    d.update(kwargs)
    return d


def translate_event(
    event_name: str,
    payload: dict[str, object],
    state: AgUiRunState,
) -> list[dict[str, Any]]:
    """Translate a Maestro event into zero or more AG-UI event dicts.

    One Maestro event may produce multiple AG-UI events (e.g. ``task_start``
    produces ``STEP_STARTED`` + ``TEXT_MESSAGE_START``), or zero for
    suppressed internal events.
    """
    if event_name in _SUPPRESSED_EVENTS:
        return []

    tid = str(payload.get("task_id", ""))

    # -- Lifecycle -----------------------------------------------------------

    if event_name == "run_start":
        # Suppressed — the SSE generator emits RUN_STARTED + STATE_SNAPSHOT
        # before the event loop starts.  We still reset state here.
        state.total_cost_usd = 0.0
        state.total_tokens = 0
        return []

    if event_name == "run_complete":
        # Suppressed — the SSE generator emits the final STATE_SNAPSHOT +
        # RUN_FINISHED after the queue sentinel.
        return []

    # -- Task lifecycle ------------------------------------------------------

    if event_name == "task_start":
        state.task_statuses[tid] = "running"
        mid: str | None = state.next_message_id(tid)
        return [
            _ev(STEP_STARTED, stepName=tid),
            _ev(TEXT_MESSAGE_START, messageId=mid, role="assistant",
                name=tid),
            _state_delta(state, tid, "running"),
        ]

    if event_name == "task_complete":
        status = str(payload.get("status", "success"))
        cost = payload.get("cost_usd")
        duration = payload.get("duration_sec")
        state.task_statuses[tid] = status
        state.completed_count += 1
        if isinstance(cost, (int, float)):
            state.task_costs[tid] = float(cost)
            state.total_cost_usd += float(cost)
        tokens = payload.get("total_tokens")
        if isinstance(tokens, int):
            state.total_tokens += tokens
        if isinstance(duration, (int, float)):
            state.task_durations[tid] = float(duration)
        events: list[dict[str, Any]] = []
        # Close text message if open
        mid = state.close_message(tid)
        if mid:
            events.append(_ev(TEXT_MESSAGE_END, messageId=mid))
        events.append(_ev(STEP_FINISHED, stepName=tid))
        events.append(_state_delta(state, tid, status))
        return events

    if event_name == "task_skip":
        state.task_statuses[tid] = "skipped"
        state.completed_count += 1
        return [
            _ev(STEP_FINISHED, stepName=tid),
            _ev(CUSTOM, name="task_skip", value={"taskId": tid,
                "reason": str(payload.get("reason", ""))}),
            _state_delta(state, tid, "skipped"),
        ]

    # -- Task output ---------------------------------------------------------

    if event_name == "task_output":
        mid = state.get_message_id(tid)
        line = str(payload.get("line", payload.get("output", "")))
        if mid and line:
            return [_ev(TEXT_MESSAGE_CONTENT, messageId=mid, delta=line + "\n")]
        return []

    # -- Tool calls ----------------------------------------------------------

    if event_name == "task_tool_call":
        tool_name = str(payload.get("tool", "unknown"))
        tool_id = state.next_tool_id(tid)
        mid = state.get_message_id(tid)
        return [
            _ev(TOOL_CALL_START, toolCallId=tool_id, toolCallName=tool_name,
                parentMessageId=mid),
            _ev(TOOL_CALL_END, toolCallId=tool_id),
        ]

    # -- Signals (mid-task) --------------------------------------------------

    if event_name == "task_signal":
        signal_type = str(payload.get("signal_type", ""))
        if signal_type == "progress":
            pct = payload.get("pct")
            step = payload.get("step")
            return [_ev(STATE_DELTA, delta=[
                {"op": "replace", "path": f"/tasks/{tid}/progress",
                 "value": pct},
            ])]
        return [_ev(CUSTOM, name=f"signal_{signal_type}",
                    value={k: v for k, v in payload.items()
                           if k not in ("event", "plan_name")})]

    # -- Watch events --------------------------------------------------------

    if event_name == "watch_start":
        return [_ev(RUN_STARTED, threadId=state.thread_id,
                    runId=state.run_id)]

    if event_name == "watch_complete":
        return [_ev(RUN_FINISHED, threadId=state.thread_id,
                    runId=state.run_id,
                    result={"status": str(payload.get("status", ""))})]

    if event_name == "iteration_start":
        iteration = payload.get("iteration", 0)
        step = f"iteration_{iteration}"
        return [_ev(STEP_STARTED, stepName=step)]

    if event_name == "iteration_complete":
        iteration = payload.get("iteration", 0)
        step = f"iteration_{iteration}"
        return [_ev(STEP_FINISHED, stepName=step)]

    # -- Dynamic sub-plans ---------------------------------------------------

    if event_name == "dynamic_subplan_start":
        return [_ev(STEP_STARTED, stepName=f"{tid}/dynamic")]

    if event_name == "dynamic_subplan_complete":
        return [_ev(STEP_FINISHED, stepName=f"{tid}/dynamic")]

    # -- Domain events → CUSTOM ----------------------------------------------

    _CUSTOM_EVENTS: frozenset[str] = frozenset({
        "task_retry", "task_escalation", "engine_fallback",
        "judge_start", "judge_result",
        "approval_required", "approval_response",
        "budget_warning", "budget_exceeded",
        "policy_violation", "taint_detected",
        "circuit_breaker_tripped", "model_routed",
        "verify_failure", "batch_chunk_complete",
        "worktree_create", "worktree_merge",
        "metric_recorded", "regression_detected",
        "rollback_executed", "plateau_detected", "target_reached",
        "timeout_adjusted",
    })

    if event_name in _CUSTOM_EVENTS:
        value = {k: v for k, v in payload.items()
                 if k not in ("event", "plan_name")}
        return [_ev(CUSTOM, name=event_name, value=value)]

    # -- Fallback: emit as CUSTOM with raw payload ---------------------------
    return [_ev(CUSTOM, name=event_name,
                value={k: v for k, v in payload.items()
                       if k not in ("event", "plan_name")})]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_delta(
    state: AgUiRunState, task_id: str, status: str,
) -> dict[str, Any]:
    """Build a STATE_DELTA event with progress + task status update."""
    patches: list[dict[str, Any]] = [
        {"op": "replace", "path": f"/tasks/{task_id}/status", "value": status},
        {"op": "replace", "path": "/progress", "value": state.progress_pct},
        {"op": "replace", "path": "/totalCostUsd", "value": state.total_cost_usd},
    ]
    cost = state.task_costs.get(task_id)
    if cost is not None:
        patches.append({"op": "replace", "path": f"/tasks/{task_id}/costUsd",
                         "value": cost})
    dur = state.task_durations.get(task_id)
    if dur is not None:
        patches.append({"op": "replace", "path": f"/tasks/{task_id}/durationSec",
                         "value": dur})
    return _ev(STATE_DELTA, delta=patches)


def format_sse(event_dict: dict[str, Any]) -> str:
    """Serialise an AG-UI event dict to the SSE wire format.

    AG-UI uses bare ``data:`` lines (no ``event:`` prefix) because the
    event type is inside the JSON payload's ``type`` field.
    """
    import json
    return f"data: {json.dumps(event_dict, default=str)}\n\n"
