"""AG-UI protocol SSE endpoint for Maestro CLI.

Implements the AG-UI wire protocol: HTTP POST with SSE response.
The client sends a JSON body with run parameters; the server streams
AG-UI events as ``data: {json}\\n\\n`` lines.

Requires the ``[agui]`` optional extra (``pip install maestro-cli[web,agui]``).
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..ag_ui import (
    AgUiRunState,
    RUN_ERROR,
    RUN_FINISHED,
    RUN_STARTED,
    format_sse,
    translate_event,
)
from ..loader import load_plan
from ..errors import PlanValidationError
from ..scheduler import run_plan

router = APIRouter()

_STREAM_TIMEOUT = 1800  # 30 min max

# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------


class AgUiRunRequest(BaseModel):
    """AG-UI ``RunAgentInput`` adapted for Maestro."""

    thread_id: str = Field(alias="threadId", default_factory=lambda: str(uuid.uuid4()))
    run_id: str = Field(alias="runId", default_factory=lambda: str(uuid.uuid4()))
    state: dict[str, Any] | None = None
    messages: list[dict[str, Any]] = []
    tools: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    forwarded_props: dict[str, Any] = Field(default_factory=dict, alias="forwardedProps")

    model_config = {"populate_by_name": True}


class ApprovalResponse(BaseModel):
    task_id: str = Field(alias="taskId")
    approved: bool = True

    model_config = {"populate_by_name": True}


# ---------------------------------------------------------------------------
# Approval registry (per-run pending approvals)
# ---------------------------------------------------------------------------

_approval_lock = threading.Lock()
_pending_approvals: dict[str, dict[str, tuple[threading.Event, list[bool]]]] = {}


def _register_run(run_id: str) -> None:
    with _approval_lock:
        _pending_approvals[run_id] = {}


def _cleanup_run(run_id: str) -> None:
    with _approval_lock:
        _pending_approvals.pop(run_id, None)


def _make_approval_handler(
    run_id: str,
) -> Any:
    """Create an approval_handler callable for run_plan()."""
    def _handler(task_id: str, message: str | None = None) -> bool:
        ev = threading.Event()
        result: list[bool] = [False]
        with _approval_lock:
            registry = _pending_approvals.get(run_id)
            if registry is None:
                return False
            registry[task_id] = (ev, result)
        # Block until client responds or timeout
        approved = ev.wait(timeout=300)
        with _approval_lock:
            registry = _pending_approvals.get(run_id, {})
            registry.pop(task_id, None)
        return result[0] if approved else False
    return _handler


# ---------------------------------------------------------------------------
# SSE endpoint
# ---------------------------------------------------------------------------


@router.post("/agui/runs")
async def agui_run(req: AgUiRunRequest) -> StreamingResponse:
    """AG-UI protocol endpoint: accept a run request, stream events."""
    props = req.forwarded_props
    plan_path_str = props.get("plan_path") or props.get("planPath")
    yaml_content = props.get("yaml_content") or props.get("yamlContent")

    if not plan_path_str and not yaml_content:
        raise HTTPException(
            status_code=400,
            detail="Provide 'planPath' or 'yamlContent' in forwardedProps",
        )

    # Resolve plan
    plan_path: Path
    _tmp_file: Path | None = None
    if yaml_content:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False,
        )
        tmp.write(str(yaml_content))
        tmp.close()
        plan_path = Path(tmp.name)
        _tmp_file = plan_path
    else:
        plan_path = Path(str(plan_path_str))

    try:
        plan = load_plan(plan_path)
    except PlanValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    task_ids = [t.id for t in plan.tasks]
    dry_run = bool(props.get("dry_run", props.get("dryRun", False)))
    execution_profile = str(props.get("execution_profile",
                                       props.get("executionProfile", "plan")))
    max_parallel = props.get("max_parallel", props.get("maxParallel"))
    auto_approve = bool(props.get("auto_approve", props.get("autoApprove", False)))
    only_raw = props.get("only")
    skip_raw = props.get("skip")
    only = set(only_raw) if isinstance(only_raw, list) else None
    skip = set(skip_raw) if isinstance(skip_raw, list) else None

    # Build state tracker
    state = AgUiRunState(
        run_id=req.run_id,
        thread_id=req.thread_id,
        task_ids=task_ids,
    )
    for tid in task_ids:
        state.task_statuses[tid] = "pending"

    # Event bridge: sync callback → async queue
    loop = asyncio.get_event_loop()
    event_queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue(maxsize=10000)

    def _event_callback(event_name: str, payload: dict[str, object]) -> None:
        data: dict[str, object] = {"_event_name": event_name}
        data.update(payload)
        try:
            loop.call_soon_threadsafe(event_queue.put_nowait, data)
        except (RuntimeError, asyncio.QueueFull):
            pass  # loop closed or queue full — drop non-critical events

    # Approval handler
    run_id = req.run_id
    _register_run(run_id)
    approval_handler = None if auto_approve else _make_approval_handler(run_id)

    # Background execution
    error_holder: list[str] = []

    def _run() -> None:
        try:
            run_plan(
                plan,
                dry_run=dry_run,
                execution_profile=execution_profile,  # type: ignore[arg-type]
                max_parallel_override=int(max_parallel) if max_parallel else None,
                only=only,
                skip=skip,
                auto_approve=auto_approve,
                event_callback=_event_callback,
                approval_handler=approval_handler,
            )
        except Exception as exc:
            error_holder.append(str(exc))
        finally:
            try:
                loop.call_soon_threadsafe(event_queue.put_nowait, None)
            except RuntimeError:
                pass
            _cleanup_run(run_id)
            if _tmp_file and _tmp_file.exists():
                _tmp_file.unlink(missing_ok=True)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    return StreamingResponse(
        _agui_event_generator(event_queue, state, error_holder),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def _agui_event_generator(
    queue: asyncio.Queue[dict[str, object] | None],
    state: AgUiRunState,
    error_holder: list[str],
) -> AsyncIterator[str]:
    """Consume Maestro events from the queue and yield AG-UI SSE lines."""
    import time

    # Emit RUN_STARTED + initial STATE_SNAPSHOT
    yield format_sse({"type": RUN_STARTED, "threadId": state.thread_id,
                       "runId": state.run_id, "timestamp": int(time.time() * 1000)})
    yield format_sse({"type": "STATE_SNAPSHOT", "snapshot": state.to_snapshot(),
                       "timestamp": int(time.time() * 1000)})

    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=_STREAM_TIMEOUT)
        except asyncio.TimeoutError:
            yield format_sse({"type": RUN_ERROR,
                              "message": "Stream timed out after 30 minutes",
                              "timestamp": int(time.time() * 1000)})
            return

        if item is None:
            # Run complete or error
            break

        event_name = str(item.pop("_event_name", ""))
        payload = dict(item)
        ag_events = translate_event(event_name, payload, state)
        for ev in ag_events:
            yield format_sse(ev)

    # Emit final STATE_SNAPSHOT + RUN_FINISHED (or RUN_ERROR)
    ts = int(time.time() * 1000)
    if error_holder:
        yield format_sse({"type": RUN_ERROR, "message": error_holder[0],
                           "timestamp": ts})
    else:
        yield format_sse({"type": "STATE_SNAPSHOT", "snapshot": state.to_snapshot(),
                           "timestamp": ts})
        yield format_sse({"type": RUN_FINISHED, "threadId": state.thread_id,
                           "runId": state.run_id,
                           "result": {"success": True,
                                       "totalCostUsd": state.total_cost_usd,
                                       "progress": state.progress_pct},
                           "timestamp": ts})


# ---------------------------------------------------------------------------
# Approval companion endpoint
# ---------------------------------------------------------------------------


@router.post("/agui/runs/{run_id}/approve")
async def approve_task(run_id: str, body: ApprovalResponse) -> dict[str, bool]:
    """Resolve a pending approval for an AG-UI run."""
    with _approval_lock:
        registry = _pending_approvals.get(run_id)
        if registry is None:
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
        pending = registry.get(body.task_id)
        if pending is None:
            raise HTTPException(
                status_code=404,
                detail=f"No pending approval for task '{body.task_id}'",
            )
        ev, result = pending
        result[0] = body.approved
        ev.set()
    return {"approved": body.approved}
