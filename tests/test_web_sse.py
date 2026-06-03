from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from maestro_cli.utils import now_utc
from maestro_cli.web.routes_sse import _format_sse, router as sse_router
from maestro_cli.web.state import RunState, _active_runs, _lock, register_run


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_sse_events(raw: str) -> list[dict[str, object]]:
    """Parse raw SSE text into a list of {event, data} dicts."""
    events: list[dict[str, object]] = []
    current_event: str | None = None
    current_data: str | None = None

    for line in raw.splitlines():
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            current_data = line[len("data: "):]
        elif line == "" and current_event is not None and current_data is not None:
            events.append({
                "event": current_event,
                "data": json.loads(current_data),
            })
            current_event = None
            current_data = None

    return events


def _make_run_state(
    run_id: str,
    plan_name: str,
    task_ids: list[str],
    run_path: Path,
) -> RunState:
    """Create a RunState with a dummy thread that is already finished."""
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    return RunState(
        run_id=run_id,
        plan_name=plan_name,
        task_ids=task_ids,
        run_path=run_path,
        started_at=now_utc(),
        thread=t,
    )


def _write_result_json(run_dir: Path, task_id: str, status: str = "success") -> None:
    """Write a minimal .result.json file for a task."""
    data = {
        "task_id": task_id,
        "status": status,
        "exit_code": 0,
        "duration": 1.23,
    }
    (run_dir / f"{task_id}.result.json").write_text(
        json.dumps(data), encoding="utf-8",
    )


def _write_manifest(
    run_dir: Path,
    plan_name: str = "test-plan",
    *,
    success: bool = True,
    task_results: dict[str, object] | None = None,
) -> None:
    """Write a run_manifest.json to the run directory."""
    data = {
        "plan_name": plan_name,
        "success": success,
        "started_at": "2026-02-26T10:00:00+00:00",
        "finished_at": "2026-02-26T10:01:00+00:00",
        "task_results": task_results or {},
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(data), encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def app() -> FastAPI:
    """Create a minimal FastAPI app with the SSE router mounted."""
    app = FastAPI()
    app.include_router(sse_router, prefix="/api")
    return app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def _cleanup_active_runs() -> None:
    """Ensure _active_runs is empty before and after every test."""
    with _lock:
        _active_runs.clear()
    yield  # type: ignore[misc]
    with _lock:
        _active_runs.clear()


# ---------------------------------------------------------------------------
# Tests: _format_sse helper
# ---------------------------------------------------------------------------

class TestFormatSse:
    def test_format_sse_produces_valid_event_block(self) -> None:
        result = _format_sse("task_complete", {"task_id": "t1", "status": "success"})
        assert result.startswith("event: task_complete\n")
        assert "data: " in result
        assert result.endswith("\n\n")

    def test_format_sse_data_is_valid_json(self) -> None:
        result = _format_sse("run_started", {"tasks": ["a", "b"]})
        lines = result.strip().split("\n")
        data_line = [ln for ln in lines if ln.startswith("data: ")][0]
        payload = json.loads(data_line[len("data: "):])
        assert payload == {"tasks": ["a", "b"]}

    def test_format_sse_handles_non_serialisable_types(self) -> None:
        """datetime and Path objects should be serialised via default=str."""
        result = _format_sse("info", {"path": Path("/tmp/test"), "ts": now_utc()})
        lines = result.strip().split("\n")
        data_line = [ln for ln in lines if ln.startswith("data: ")][0]
        payload = json.loads(data_line[len("data: "):])
        assert isinstance(payload["path"], str)
        assert isinstance(payload["ts"], str)


# ---------------------------------------------------------------------------
# Tests: SSE endpoint
# ---------------------------------------------------------------------------

class TestEventsEndpoint:
    def test_events_returns_404_for_unknown_run(self, client: TestClient) -> None:
        resp = client.get("/api/runs/nonexistent-id/events")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_events_sends_run_started(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """First event from the stream should be run_started with the task list."""
        run_dir = tmp_path / "run1"
        run_dir.mkdir()
        # Write manifest immediately so the stream terminates quickly
        _write_manifest(run_dir)

        rs = _make_run_state("run-1", "plan-a", ["t1", "t2"], run_dir)
        register_run(rs)

        with client.stream("GET", "/api/runs/run-1/events") as resp:
            assert resp.status_code == 200
            text = resp.read().decode("utf-8")

        events = _parse_sse_events(text)
        assert len(events) >= 1
        first = events[0]
        assert first["event"] == "run_started"
        assert first["data"] == {"tasks": ["t1", "t2"]}

    def test_events_streams_task_complete(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """When a .result.json exists, a task_complete event is emitted."""
        run_dir = tmp_path / "run2"
        run_dir.mkdir()
        _write_result_json(run_dir, "t1", status="success")
        # Write manifest so stream stops after one poll cycle
        _write_manifest(run_dir, task_results={})

        rs = _make_run_state("run-2", "plan-b", ["t1"], run_dir)
        register_run(rs)

        with client.stream("GET", "/api/runs/run-2/events") as resp:
            text = resp.read().decode("utf-8")

        events = _parse_sse_events(text)
        task_events = [e for e in events if e["event"] == "task_complete"]
        assert len(task_events) >= 1
        assert task_events[0]["data"]["task_id"] == "t1"
        assert task_events[0]["data"]["status"] == "success"

    def test_events_streams_run_complete(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """When run_manifest.json exists, a run_complete event is emitted and stream closes."""
        run_dir = tmp_path / "run3"
        run_dir.mkdir()
        _write_manifest(run_dir, plan_name="my-plan", success=True)

        rs = _make_run_state("run-3", "my-plan", [], run_dir)
        register_run(rs)

        with client.stream("GET", "/api/runs/run-3/events") as resp:
            text = resp.read().decode("utf-8")

        events = _parse_sse_events(text)
        run_complete = [e for e in events if e["event"] == "run_complete"]
        assert len(run_complete) == 1
        payload = run_complete[0]["data"]
        assert payload["success"] is True
        assert payload["plan_name"] == "my-plan"
        assert "started_at" in payload
        assert "finished_at" in payload
        assert "task_count" in payload

    def test_events_run_complete_is_last_event(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """The stream must terminate after run_complete (no further events)."""
        run_dir = tmp_path / "run4"
        run_dir.mkdir()
        _write_manifest(run_dir)

        rs = _make_run_state("run-4", "plan-x", ["t1"], run_dir)
        register_run(rs)

        with client.stream("GET", "/api/runs/run-4/events") as resp:
            text = resp.read().decode("utf-8")

        events = _parse_sse_events(text)
        assert events[-1]["event"] == "run_complete"

    def test_events_manifest_catches_unseen_task_results(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """Task results embedded in the manifest are emitted if not yet seen."""
        run_dir = tmp_path / "run5"
        run_dir.mkdir()
        # Do NOT write individual .result.json files — only the manifest
        task_results = {
            "t1": {"task_id": "t1", "status": "success", "exit_code": 0},
            "t2": {"task_id": "t2", "status": "failed", "exit_code": 1},
        }
        _write_manifest(run_dir, task_results=task_results)

        rs = _make_run_state("run-5", "plan-y", ["t1", "t2"], run_dir)
        register_run(rs)

        with client.stream("GET", "/api/runs/run-5/events") as resp:
            text = resp.read().decode("utf-8")

        events = _parse_sse_events(text)
        task_events = [e for e in events if e["event"] == "task_complete"]
        task_ids_received = {e["data"]["task_id"] for e in task_events}
        assert task_ids_received == {"t1", "t2"}

    def test_events_does_not_duplicate_already_seen_tasks(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """If a task .result.json is polled AND it appears in the manifest,
        the event should only be sent once."""
        run_dir = tmp_path / "run6"
        run_dir.mkdir()
        _write_result_json(run_dir, "t1")
        task_results = {
            "t1": {"task_id": "t1", "status": "success", "exit_code": 0},
        }
        _write_manifest(run_dir, task_results=task_results)

        rs = _make_run_state("run-6", "plan-z", ["t1"], run_dir)
        register_run(rs)

        with client.stream("GET", "/api/runs/run-6/events") as resp:
            text = resp.read().decode("utf-8")

        events = _parse_sse_events(text)
        task_events = [e for e in events if e["event"] == "task_complete"]
        assert len(task_events) == 1

    def test_events_detects_new_results(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """Results written after the stream starts should be detected on the
        next poll cycle."""
        run_dir = tmp_path / "run7"
        run_dir.mkdir()

        rs = _make_run_state("run-7", "plan-w", ["t1", "t2"], run_dir)
        register_run(rs)

        def _write_delayed() -> None:
            time.sleep(0.7)  # slightly more than one poll interval
            _write_result_json(run_dir, "t1")
            time.sleep(0.7)
            _write_result_json(run_dir, "t2")
            time.sleep(0.7)
            _write_manifest(run_dir)

        writer = threading.Thread(target=_write_delayed)
        writer.start()

        with client.stream("GET", "/api/runs/run-7/events") as resp:
            text = resp.read().decode("utf-8")

        writer.join(timeout=10)

        events = _parse_sse_events(text)
        event_types = [e["event"] for e in events]
        assert event_types[0] == "run_started"
        assert "task_complete" in event_types
        assert event_types[-1] == "run_complete"

        task_events = [e for e in events if e["event"] == "task_complete"]
        task_ids_received = {e["data"]["task_id"] for e in task_events}
        assert task_ids_received == {"t1", "t2"}

    def test_events_response_headers(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """Response must have correct SSE headers."""
        run_dir = tmp_path / "run8"
        run_dir.mkdir()
        _write_manifest(run_dir)

        rs = _make_run_state("run-8", "plan-h", [], run_dir)
        register_run(rs)

        with client.stream("GET", "/api/runs/run-8/events") as resp:
            assert resp.headers["content-type"].startswith("text/event-stream")
            assert resp.headers.get("cache-control") == "no-cache"
            _ = resp.read()

    def test_events_malformed_result_json_skipped(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """A .result.json with invalid JSON should be silently skipped."""
        run_dir = tmp_path / "run9"
        run_dir.mkdir()
        (run_dir / "t1.result.json").write_text(
            "NOT VALID JSON {{", encoding="utf-8",
        )
        _write_manifest(run_dir)

        rs = _make_run_state("run-9", "plan-m", ["t1"], run_dir)
        register_run(rs)

        with client.stream("GET", "/api/runs/run-9/events") as resp:
            text = resp.read().decode("utf-8")

        events = _parse_sse_events(text)
        # The malformed t1 should not appear as task_complete from the file poll;
        # it may appear from the manifest's task_results (empty in this case).
        event_types = [e["event"] for e in events]
        assert "run_started" in event_types
        assert "run_complete" in event_types

    def test_events_empty_task_list(
        self, client: TestClient, tmp_path: Path,
    ) -> None:
        """A run with no tasks should still emit run_started then run_complete."""
        run_dir = tmp_path / "run10"
        run_dir.mkdir()
        _write_manifest(run_dir)

        rs = _make_run_state("run-10", "plan-e", [], run_dir)
        register_run(rs)

        with client.stream("GET", "/api/runs/run-10/events") as resp:
            text = resp.read().decode("utf-8")

        events = _parse_sse_events(text)
        assert len(events) == 2
        assert events[0]["event"] == "run_started"
        assert events[0]["data"] == {"tasks": []}
        assert events[1]["event"] == "run_complete"
