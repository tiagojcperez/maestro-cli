"""Server-Sent Events for real-time run progress.

Polls the run directory for new .result.json files and streams
task completion events to connected clients.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .state import get_run

router = APIRouter()

_POLL_INTERVAL = 0.5  # seconds
_TIMEOUT = 1800  # 30 minutes max


async def _event_stream(run_path: Path, task_ids: list[str]) -> AsyncIterator[str]:
    """Yield SSE events by polling the run directory for new results."""
    seen: set[str] = set()
    elapsed = 0.0

    # Send initial event with task list
    yield _format_sse("run_started", {"tasks": task_ids})

    while elapsed < _TIMEOUT:
        # Check for new individual task results
        for tid in task_ids:
            if tid in seen:
                continue
            result_file = run_path / f"{tid}.result.json"
            if result_file.exists():
                try:
                    data = json.loads(result_file.read_text(encoding="utf-8"))
                    seen.add(tid)
                    yield _format_sse("task_complete", data)
                except (json.JSONDecodeError, OSError):
                    pass

        # Check for run completion (manifest written at the end)
        manifest = run_path / "run_manifest.json"
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                summary = {
                    "success": data.get("success"),
                    "plan_name": data.get("plan_name"),
                    "started_at": data.get("started_at"),
                    "finished_at": data.get("finished_at"),
                    "execution_profile": data.get("execution_profile"),
                    "task_count": len(data.get("task_results", {})),
                }
                # Send any remaining task results not yet seen
                for tid, tr in data.get("task_results", {}).items():
                    if tid not in seen:
                        seen.add(tid)
                        yield _format_sse("task_complete", tr)

                yield _format_sse("run_complete", summary)
                return
            except (json.JSONDecodeError, OSError):
                pass

        await asyncio.sleep(_POLL_INTERVAL)
        elapsed += _POLL_INTERVAL

    yield _format_sse("timeout", {"message": "SSE stream timed out after 30 minutes"})


def _format_sse(event: str, data: object) -> str:
    """Format a Server-Sent Event message."""
    json_str = json.dumps(data, default=str)
    return f"event: {event}\ndata: {json_str}\n\n"


@router.get("/runs/{run_id}/events")
async def run_events(run_id: str) -> StreamingResponse:
    rs = get_run(run_id)
    if rs is None:
        raise HTTPException(status_code=404, detail=f"Active run '{run_id}' not found")

    run_path = rs.run_path
    task_ids = rs.task_ids

    return StreamingResponse(
        _event_stream(run_path, task_ids),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
