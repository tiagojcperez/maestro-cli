"""Coverage tests for small web helpers in state.py, app.py, and routes_sse.py.

Targets specific uncovered branches:
- state.RunState.to_summary duration when finished_at is set (via result)
- state._normalize_project_roots: OSError on resolve, duplicate dedup, empty input
- state.shutdown_active_runs: join alive thread + clear
- app._lifespan shutdown hook + agui ImportError fallback in create_app
- routes_sse._event_stream: corrupt manifest skip + timeout yield
"""
from __future__ import annotations

import json
import sys
import threading
import time
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from starlette.testclient import TestClient

from maestro_cli.models import PlanRunResult
from maestro_cli.utils import now_utc
from maestro_cli.web import create_app
from maestro_cli.web import state as state_mod
from maestro_cli.web.routes_sse import router as sse_router
from maestro_cli.web.state import (
    RunState,
    _active_runs,
    _lock,
    _normalize_project_roots,
    register_run,
    shutdown_active_runs,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _cleanup_active_runs() -> None:
    """Ensure _active_runs is empty before and after every test."""
    with _lock:
        _active_runs.clear()
    yield  # type: ignore[misc]
    with _lock:
        _active_runs.clear()


def _finished_run_state(
    run_id: str,
    task_ids: list[str],
    run_path: Path,
    *,
    result: PlanRunResult | None = None,
    error: str | None = None,
) -> RunState:
    """Build a RunState whose thread has already finished."""
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    return RunState(
        run_id=run_id,
        plan_name="plan",
        task_ids=task_ids,
        run_path=run_path,
        started_at=now_utc(),
        thread=t,
        result=result,
        error=error,
    )


def _make_result(run_path: Path) -> PlanRunResult:
    started = now_utc()
    return PlanRunResult(
        plan_name="plan",
        run_id="rid",
        run_path=run_path,
        started_at=started,
        finished_at=started + timedelta(seconds=5),
        success=True,
        execution_profile="safe",
        total_cost_usd=0.42,
    )


def _parse_sse_events(raw: str) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    current_event: str | None = None
    current_data: str | None = None
    for line in raw.splitlines():
        if line.startswith("event: "):
            current_event = line[len("event: "):]
        elif line.startswith("data: "):
            current_data = line[len("data: "):]
        elif line == "" and current_event is not None and current_data is not None:
            events.append({"event": current_event, "data": json.loads(current_data)})
            current_event = None
            current_data = None
    return events


# ---------------------------------------------------------------------------
# state.RunState.to_summary — duration_sec when finished_at is set
# ---------------------------------------------------------------------------

class TestToSummaryDuration:
    def test_duration_from_result_finished_at(self, tmp_path: Path) -> None:
        """When a result with finished_at is present, duration is computed from it."""
        result = _make_result(tmp_path)
        rs = _finished_run_state("r1", ["t1"], tmp_path, result=result)
        summary = rs.to_summary()

        assert summary["active"] is False
        # finished_at branch (state.py) drives a non-None positive duration
        assert summary["duration_sec"] is not None
        assert summary["duration_sec"] >= 0.0
        assert summary["finished_at"] is not None
        # profile and cost come from the result
        assert summary["execution_profile"] == "safe"
        assert summary["total_cost_usd"] == 0.42
        assert summary["success"] is True

    def test_duration_clamped_non_negative(self, tmp_path: Path) -> None:
        """finished_at earlier than started_at still yields a clamped 0.0 duration."""
        result = _make_result(tmp_path)
        rs = _finished_run_state("r2", ["t1"], tmp_path, result=result)
        # Force started_at AFTER the result.finished_at so the delta is negative.
        rs.started_at = result.finished_at + timedelta(seconds=100)
        summary = rs.to_summary()

        assert summary["duration_sec"] == 0.0

    def test_finished_no_result_uses_now(self, tmp_path: Path) -> None:
        """A finished run with no result still reports a finished_at and duration."""
        rs = _finished_run_state("r3", ["t1"], tmp_path, result=None, error="boom")
        summary = rs.to_summary()

        assert summary["active"] is False
        assert summary["finished_at"] is not None
        assert summary["duration_sec"] is not None
        assert summary["duration_sec"] >= 0.0
        # falls back to the RunState.execution_profile default
        assert summary["execution_profile"] == "plan"
        assert summary["error"] == "boom"
        assert summary["success"] is None


# ---------------------------------------------------------------------------
# state._normalize_project_roots — OSError, dedup, empty
# ---------------------------------------------------------------------------

class TestNormalizeProjectRoots:
    def test_resolve_oserror_falls_back_to_root(self) -> None:
        """If Path.resolve() raises OSError, the original root is used as-is."""
        bad = MagicMock(spec=Path)
        bad.resolve.side_effect = OSError("cannot resolve")

        result = _normalize_project_roots([bad])

        # The unresolved mock root itself is kept (state.py)
        assert result == [bad]
        bad.resolve.assert_called_once()

    def test_duplicate_roots_are_deduplicated(self, tmp_path: Path) -> None:
        """Two roots that resolve to the same path collapse to one."""
        same = tmp_path / "proj"
        same.mkdir()
        # Pass the same logical path twice (one with a redundant '.' segment)
        result = _normalize_project_roots([same, same / "." ])

        assert len(result) == 1
        assert result[0] == same.resolve()

    def test_empty_input_returns_cwd(self) -> None:
        """An empty list yields the resolved current directory."""
        result = _normalize_project_roots([])

        assert result == [Path(".").resolve()]


# ---------------------------------------------------------------------------
# state.shutdown_active_runs — join alive thread, then clear
# ---------------------------------------------------------------------------

class TestShutdownActiveRuns:
    def test_joins_alive_thread_and_clears(self) -> None:
        """An in-flight run is joined; the registry is cleared afterwards."""
        stop = threading.Event()

        def _worker() -> None:
            stop.wait(timeout=2)

        alive = threading.Thread(target=_worker)
        alive.start()

        rs_alive = RunState(
            run_id="alive",
            plan_name="plan",
            task_ids=[],
            run_path=Path("."),
            started_at=now_utc(),
            thread=alive,
        )
        register_run(rs_alive)

        # Also register an already-finished run to exercise the non-join path.
        done = threading.Thread(target=lambda: None)
        done.start()
        done.join()
        rs_done = _finished_run_state("done", [], Path("."))
        register_run(rs_done)

        # Release the alive worker shortly so join() returns within timeout.
        releaser = threading.Thread(target=lambda: (time.sleep(0.1), stop.set()))
        releaser.start()

        shutdown_active_runs()
        releaser.join(timeout=5)

        with _lock:
            assert _active_runs == {}
        assert not alive.is_alive()

    def test_shutdown_with_no_runs_is_noop(self) -> None:
        """Calling shutdown with an empty registry simply clears (idempotent)."""
        shutdown_active_runs()
        with _lock:
            assert _active_runs == {}


# ---------------------------------------------------------------------------
# app._lifespan + create_app agui ImportError fallback
# ---------------------------------------------------------------------------

class TestCreateAppLifespanAndAgui:
    def test_lifespan_shutdown_invoked_via_testclient(self, tmp_path: Path) -> None:
        """Entering/exiting the TestClient context runs the lifespan shutdown hook."""
        app = create_app(project_root=tmp_path)
        # Register a finished run so shutdown_active_runs has something to clear.
        rs = _finished_run_state("life", [], tmp_path)
        register_run(rs)

        # Using TestClient as a context manager drives startup + shutdown,
        # exercising the `yield` then shutdown_active_runs() in _lifespan.
        with TestClient(app) as client:
            resp = client.get("/", follow_redirects=False)
            assert resp.status_code in (302, 307)

        # After the context exits, the lifespan shutdown cleared active runs.
        with _lock:
            assert _active_runs == {}

    def test_create_app_handles_agui_import_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If routes_agui cannot be imported, create_app swallows ImportError."""
        # Setting the module to None in sys.modules makes `import ...` raise
        # ImportError, driving the except branch in create_app.
        monkeypatch.setitem(sys.modules, "maestro_cli.web.routes_agui", None)

        app = create_app(project_root=tmp_path)

        assert isinstance(app, FastAPI)
        # Core routers are still mounted despite the agui import failing.
        with TestClient(app) as client:
            resp = client.get("/", follow_redirects=False)
            assert resp.status_code in (302, 307)


# ---------------------------------------------------------------------------
# routes_sse._event_stream — corrupt manifest skip + timeout
# ---------------------------------------------------------------------------

@pytest.fixture
def sse_client() -> TestClient:
    app = FastAPI()
    app.include_router(sse_router, prefix="/api")
    return TestClient(app)


def _register_finished_sse_run(run_id: str, task_ids: list[str], run_path: Path) -> None:
    t = threading.Thread(target=lambda: None)
    t.start()
    t.join()
    rs = RunState(
        run_id=run_id,
        plan_name="plan",
        task_ids=task_ids,
        run_path=run_path,
        started_at=now_utc(),
        thread=t,
    )
    register_run(rs)


class TestEventStreamBranches:
    def test_corrupt_manifest_is_skipped_then_recovers(
        self, sse_client: TestClient, tmp_path: Path,
    ) -> None:
        """A corrupt run_manifest.json is skipped (except branch) and the stream
        recovers once a valid manifest is written."""
        run_dir = tmp_path / "run_corrupt"
        run_dir.mkdir()
        manifest = run_dir / "run_manifest.json"
        # Initial manifest is invalid JSON -> drives except (JSONDecodeError) branch.
        manifest.write_text("{ this is not valid json", encoding="utf-8")

        _register_finished_sse_run("sse-corrupt", ["t1"], run_dir)

        def _fix_manifest() -> None:
            # After at least one poll cycle, replace with valid JSON so the
            # stream terminates with run_complete.
            time.sleep(0.8)
            manifest.write_text(
                json.dumps(
                    {
                        "plan_name": "plan",
                        "success": True,
                        "started_at": "2026-02-26T10:00:00+00:00",
                        "finished_at": "2026-02-26T10:01:00+00:00",
                        "task_results": {},
                    }
                ),
                encoding="utf-8",
            )

        writer = threading.Thread(target=_fix_manifest)
        writer.start()

        with sse_client.stream("GET", "/api/runs/sse-corrupt/events") as resp:
            text = resp.read().decode("utf-8")
        writer.join(timeout=10)

        events = _parse_sse_events(text)
        event_types = [e["event"] for e in events]
        assert "run_started" in event_types
        # The corrupt manifest did not abort the stream; it recovered.
        assert event_types[-1] == "run_complete"

    def test_stream_times_out(
        self,
        sse_client: TestClient,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With the timeout set to 0, the loop is skipped and a timeout event
        is yielded."""
        run_dir = tmp_path / "run_timeout"
        run_dir.mkdir()
        # No manifest -> would otherwise poll forever; force immediate timeout.
        monkeypatch.setattr("maestro_cli.web.routes_sse._TIMEOUT", 0.0)

        _register_finished_sse_run("sse-timeout", ["t1"], run_dir)

        with sse_client.stream("GET", "/api/runs/sse-timeout/events") as resp:
            text = resp.read().decode("utf-8")

        events = _parse_sse_events(text)
        event_types = [e["event"] for e in events]
        assert event_types[0] == "run_started"
        assert event_types[-1] == "timeout"
        timeout_evt = events[-1]
        assert "timed out" in str(timeout_evt["data"]).lower()
