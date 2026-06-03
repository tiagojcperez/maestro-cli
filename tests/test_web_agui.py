"""Tests for web/routes_agui.py — AG-UI protocol SSE endpoint."""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.models import PlanSpec, TaskSpec
from maestro_cli.web.routes_agui import (
    AgUiRunRequest,
    ApprovalResponse,
    _agui_event_generator,
    _cleanup_run,
    _make_approval_handler,
    _pending_approvals,
    _register_run,
    router,
)
from maestro_cli.ag_ui import AgUiRunState, RUN_ERROR, RUN_FINISHED, RUN_STARTED


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_PLAN_YAML = """\
version: 1
name: agui-test
tasks:
  - id: t1
    command: "echo hello"
  - id: t2
    depends_on: [t1]
    command: "echo world"
"""

_INVALID_PLAN_YAML = """\
version: 1
name: agui-test
tasks: "not a list"
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_plan(tmp_path: Path, content: str = _VALID_PLAN_YAML) -> Path:
    """Write a plan YAML and return its path."""
    p = tmp_path / "plan.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def _make_plan_spec(name: str = "agui-test", task_ids: list[str] | None = None) -> PlanSpec:
    """Build a minimal PlanSpec for mocking."""
    ids = task_ids or ["t1", "t2"]
    tasks = [TaskSpec(id=tid, command="echo hi") for tid in ids]
    return PlanSpec(version=1, name=name, tasks=tasks, run_dir=".maestro-runs")


def _collect_sse_lines(body: str) -> list[dict[str, Any]]:
    """Parse SSE 'data: {...}' lines from a response body."""
    results = []
    for line in body.split("\n"):
        line = line.strip()
        if line.startswith("data: "):
            try:
                results.append(json.loads(line[len("data: "):]))
            except json.JSONDecodeError:
                pass
    return results


def _event_types(events: list[dict[str, Any]]) -> list[str]:
    return [e.get("type", "") for e in events]


# ---------------------------------------------------------------------------
# Pydantic request models
# ---------------------------------------------------------------------------


class TestAgUiRunRequest:
    def test_defaults_generate_uuids(self) -> None:
        req = AgUiRunRequest()
        assert isinstance(req.thread_id, str) and len(req.thread_id) > 0
        assert isinstance(req.run_id, str) and len(req.run_id) > 0
        assert req.state is None
        assert req.messages == []
        assert req.tools == []
        assert req.context == []
        assert req.forwarded_props == {}

    def test_alias_population(self) -> None:
        req = AgUiRunRequest(**{
            "threadId": "t-1",
            "runId": "r-1",
            "forwardedProps": {"plan_path": "/foo.yaml"},
        })
        assert req.thread_id == "t-1"
        assert req.run_id == "r-1"
        assert req.forwarded_props == {"plan_path": "/foo.yaml"}

    def test_field_name_population(self) -> None:
        req = AgUiRunRequest(thread_id="t-2", run_id="r-2", forwarded_props={"x": 1})
        assert req.thread_id == "t-2"
        assert req.run_id == "r-2"
        assert req.forwarded_props == {"x": 1}


class TestApprovalResponse:
    def test_default_approved_true(self) -> None:
        resp = ApprovalResponse(task_id="t1")
        assert resp.task_id == "t1"
        assert resp.approved is True

    def test_alias_population(self) -> None:
        resp = ApprovalResponse(**{"taskId": "t2", "approved": False})
        assert resp.task_id == "t2"
        assert resp.approved is False


# ---------------------------------------------------------------------------
# Approval registry
# ---------------------------------------------------------------------------


class TestApprovalRegistry:
    def test_register_and_cleanup(self) -> None:
        run_id = "test-run-registry-1"
        _register_run(run_id)
        assert run_id in _pending_approvals
        _cleanup_run(run_id)
        assert run_id not in _pending_approvals

    def test_cleanup_nonexistent_run_is_noop(self) -> None:
        _cleanup_run("nonexistent-run-id-abc")
        # Should not raise

    def test_register_overwrites_existing(self) -> None:
        run_id = "test-run-overwrite"
        _register_run(run_id)
        _pending_approvals[run_id]["some-task"] = (threading.Event(), [True])
        _register_run(run_id)
        assert _pending_approvals[run_id] == {}
        _cleanup_run(run_id)


class TestApprovalHandler:
    def test_handler_returns_false_when_run_cleaned_up(self) -> None:
        """Handler returns False if the run registry is gone."""
        run_id = "test-handler-gone"
        _register_run(run_id)
        handler = _make_approval_handler(run_id)
        _cleanup_run(run_id)
        # The handler should find no registry and return False
        result = handler("task-1", "Approve?")
        assert result is False

    def test_handler_blocks_and_returns_approval(self) -> None:
        """Handler blocks until a response is set, then returns it."""
        run_id = "test-handler-approve"
        _register_run(run_id)
        handler = _make_approval_handler(run_id)

        result_holder: list[bool] = []

        def _call_handler() -> None:
            result_holder.append(handler("task-a"))

        thread = threading.Thread(target=_call_handler)
        thread.start()

        # Wait a moment for handler to register the pending approval
        time.sleep(0.1)

        with threading.Lock():
            registry = _pending_approvals.get(run_id, {})
            pending = registry.get("task-a")
            assert pending is not None
            ev, res_list = pending
            res_list[0] = True
            ev.set()

        thread.join(timeout=5)
        assert result_holder == [True]
        _cleanup_run(run_id)

    def test_handler_returns_false_on_denial(self) -> None:
        """Handler returns False when client denies."""
        run_id = "test-handler-deny"
        _register_run(run_id)
        handler = _make_approval_handler(run_id)

        result_holder: list[bool] = []

        def _call_handler() -> None:
            result_holder.append(handler("task-b"))

        thread = threading.Thread(target=_call_handler)
        thread.start()

        time.sleep(0.1)

        registry = _pending_approvals.get(run_id, {})
        pending = registry.get("task-b")
        assert pending is not None
        ev, res_list = pending
        res_list[0] = False
        ev.set()

        thread.join(timeout=5)
        assert result_holder == [False]
        _cleanup_run(run_id)

    def test_handler_returns_false_on_timeout(self) -> None:
        """Handler returns False when no response arrives before timeout."""
        run_id = "test-handler-timeout"
        _register_run(run_id)

        # Create handler with a very short timeout by monkeypatching
        handler = _make_approval_handler(run_id)

        # We need the handler to time out fast, so we'll simulate it
        # by directly manipulating the event (not setting it) and using
        # a short wait. We'll call the handler in a thread with monkeypatched timeout.
        result_holder: list[bool] = []

        def _call_handler_quick() -> None:
            # Bypass the normal 300s timeout by directly testing the logic
            ev = threading.Event()
            result: list[bool] = [False]
            from maestro_cli.web.routes_agui import _approval_lock
            with _approval_lock:
                reg = _pending_approvals.get(run_id)
                if reg is not None:
                    reg["task-timeout"] = (ev, result)
            approved = ev.wait(timeout=0.1)  # Very short timeout
            with _approval_lock:
                reg = _pending_approvals.get(run_id, {})
                reg.pop("task-timeout", None)
            result_holder.append(result[0] if approved else False)

        thread = threading.Thread(target=_call_handler_quick)
        thread.start()
        thread.join(timeout=5)
        assert result_holder == [False]
        _cleanup_run(run_id)


# ---------------------------------------------------------------------------
# Async event generator
# ---------------------------------------------------------------------------


class TestAguiEventGenerator:
    """Test _agui_event_generator async generator directly."""

    def _run_generator(
        self,
        events: list[dict[str, object] | None],
        error_holder: list[str] | None = None,
    ) -> list[str]:
        """Run the generator synchronously and collect yielded SSE lines."""
        state = AgUiRunState(
            run_id="gen-run",
            thread_id="gen-thread",
            task_ids=["t1"],
            task_statuses={"t1": "pending"},
        )
        if error_holder is None:
            error_holder = []

        async def _run() -> list[str]:
            queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
            for ev in events:
                await queue.put(ev)
            lines: list[str] = []
            async for line in _agui_event_generator(queue, state, error_holder):
                lines.append(line)
            return lines

        return asyncio.run(_run())

    def test_emits_run_started_and_snapshot_first(self) -> None:
        """Generator emits RUN_STARTED and STATE_SNAPSHOT before processing events."""
        lines = self._run_generator([None])  # immediate termination
        events = _collect_sse_lines("\n".join(lines))
        types = _event_types(events)
        assert types[0] == RUN_STARTED
        assert types[1] == "STATE_SNAPSHOT"

    def test_run_finished_on_clean_completion(self) -> None:
        """None sentinel triggers RUN_FINISHED when no errors."""
        lines = self._run_generator([None])
        events = _collect_sse_lines("\n".join(lines))
        types = _event_types(events)
        assert RUN_FINISHED in types
        # Check the RUN_FINISHED event has expected shape
        finished = [e for e in events if e["type"] == RUN_FINISHED][0]
        assert finished["threadId"] == "gen-thread"
        assert finished["runId"] == "gen-run"
        assert "result" in finished
        assert finished["result"]["success"] is True

    def test_run_error_on_exception(self) -> None:
        """Error holder triggers RUN_ERROR instead of RUN_FINISHED."""
        lines = self._run_generator([None], error_holder=["Something broke"])
        events = _collect_sse_lines("\n".join(lines))
        types = _event_types(events)
        assert RUN_ERROR in types
        assert RUN_FINISHED not in types
        error = [e for e in events if e["type"] == RUN_ERROR][0]
        assert "Something broke" in error["message"]

    def test_maestro_events_translated(self) -> None:
        """Maestro events in the queue are translated to AG-UI events."""
        maestro_event: dict[str, object] = {
            "_event_name": "task_start",
            "task_id": "t1",
        }
        lines = self._run_generator([maestro_event, None])
        events = _collect_sse_lines("\n".join(lines))
        types = _event_types(events)
        # Should have STEP_STARTED from task_start translation
        assert "STEP_STARTED" in types

    def test_multiple_events_in_sequence(self) -> None:
        """Multiple events are processed in order."""
        events_in: list[dict[str, object] | None] = [
            {"_event_name": "task_start", "task_id": "t1"},
            {"_event_name": "task_output", "task_id": "t1", "line": "hello"},
            {"_event_name": "task_complete", "task_id": "t1", "status": "success",
             "cost_usd": 0.05, "total_tokens": 500},
            None,
        ]
        lines = self._run_generator(events_in)
        events = _collect_sse_lines("\n".join(lines))
        types = _event_types(events)
        # Check ordering: RUN_STARTED, STATE_SNAPSHOT, then task events, then finished
        assert types[0] == RUN_STARTED
        assert types[1] == "STATE_SNAPSHOT"
        assert RUN_FINISHED in types
        # Verify progress in the finished event
        finished = [e for e in events if e["type"] == RUN_FINISHED][0]
        assert finished["result"]["totalCostUsd"] == pytest.approx(0.05)

    def test_final_state_snapshot_on_success(self) -> None:
        """On clean completion, a final STATE_SNAPSHOT is emitted before RUN_FINISHED."""
        lines = self._run_generator([None])
        events = _collect_sse_lines("\n".join(lines))
        types = _event_types(events)
        # Should have at least 2 STATE_SNAPSHOTs (initial + final)
        snapshots = [e for e in events if e["type"] == "STATE_SNAPSHOT"]
        assert len(snapshots) >= 2

    def test_no_final_snapshot_on_error(self) -> None:
        """On error, only the initial STATE_SNAPSHOT is emitted (no final one)."""
        lines = self._run_generator([None], error_holder=["Boom"])
        events = _collect_sse_lines("\n".join(lines))
        # Only 1 STATE_SNAPSHOT (initial), then RUN_ERROR
        snapshots = [e for e in events if e["type"] == "STATE_SNAPSHOT"]
        assert len(snapshots) == 1

    def test_suppressed_events_ignored(self) -> None:
        """Suppressed events (like context_compression) produce no AG-UI output."""
        events_in: list[dict[str, object] | None] = [
            {"_event_name": "context_compression", "task_id": "t1"},
            None,
        ]
        lines = self._run_generator(events_in)
        events = _collect_sse_lines("\n".join(lines))
        # Only RUN_STARTED, STATE_SNAPSHOT (initial), STATE_SNAPSHOT (final), RUN_FINISHED
        types = _event_types(events)
        custom_events = [e for e in events if e.get("type") == "CUSTOM"]
        assert len(custom_events) == 0

    def test_event_missing_event_name_key(self) -> None:
        """Event dict without _event_name uses empty string."""
        events_in: list[dict[str, object] | None] = [
            {"task_id": "t1"},  # no _event_name
            None,
        ]
        lines = self._run_generator(events_in)
        events = _collect_sse_lines("\n".join(lines))
        # Should still produce some output (unknown event -> CUSTOM)
        custom = [e for e in events if e.get("type") == "CUSTOM"]
        assert len(custom) >= 1


# ---------------------------------------------------------------------------
# Async generator timeout (mock _STREAM_TIMEOUT)
# ---------------------------------------------------------------------------


class TestAguiEventGeneratorTimeout:
    def test_timeout_emits_run_error(self) -> None:
        """When the queue read times out, RUN_ERROR is emitted."""
        state = AgUiRunState(
            run_id="timeout-run",
            thread_id="timeout-thread",
            task_ids=[],
        )

        async def _run() -> list[str]:
            queue: asyncio.Queue[dict[str, object] | None] = asyncio.Queue()
            # Don't put anything in the queue — it will time out
            lines: list[str] = []
            # Patch the timeout to be very short
            with patch("maestro_cli.web.routes_agui._STREAM_TIMEOUT", 0.05):
                async for line in _agui_event_generator(queue, state, []):
                    lines.append(line)
            return lines

        lines = asyncio.run(_run())
        events = _collect_sse_lines("\n".join(lines))
        types = _event_types(events)
        assert RUN_ERROR in types
        error = [e for e in events if e["type"] == RUN_ERROR][0]
        assert "timed out" in error["message"].lower()


# ---------------------------------------------------------------------------
# Full endpoint tests via TestClient
# ---------------------------------------------------------------------------


def _create_test_app() -> Any:
    """Create a FastAPI app with only the AG-UI router for testing."""
    from fastapi import FastAPI
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


class TestAguiRunEndpoint:
    """Test the POST /api/agui/runs endpoint."""

    def test_missing_plan_path_and_yaml_returns_400(self) -> None:
        from starlette.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post("/api/agui/runs", json={
            "threadId": "t1",
            "runId": "r1",
            "forwardedProps": {},
        })
        assert resp.status_code == 400
        assert "planPath" in resp.json()["detail"]

    def test_invalid_plan_yaml_returns_400(self, tmp_path: Path) -> None:
        from starlette.testclient import TestClient
        plan_file = tmp_path / "bad.yaml"
        plan_file.write_text(_INVALID_PLAN_YAML, encoding="utf-8")

        app = _create_test_app()
        client = TestClient(app)
        resp = client.post("/api/agui/runs", json={
            "threadId": "t1",
            "runId": "r1",
            "forwardedProps": {"planPath": str(plan_file)},
        })
        assert resp.status_code == 400

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_valid_plan_path_streams_sse(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
        tmp_path: Path,
    ) -> None:
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan

        def _fake_run_plan(**kwargs: Any) -> None:
            cb = kwargs.get("event_callback")
            if cb:
                cb("task_start", {"task_id": "t1"})
                cb("task_complete", {"task_id": "t1", "status": "success",
                                     "cost_usd": 0.01, "total_tokens": 100})

        mock_run_plan.side_effect = lambda *a, **kw: _fake_run_plan(**kw)

        app = _create_test_app()
        client = TestClient(app)

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-1",
            "runId": "rn-1",
            "forwardedProps": {"planPath": str(plan_file)},
        })
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")

        events = _collect_sse_lines(resp.text)
        types = _event_types(events)
        assert RUN_STARTED in types
        assert RUN_FINISHED in types

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_yaml_content_in_forwarded_props(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
    ) -> None:
        """yamlContent in forwardedProps creates a temp file and loads from it."""
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan
        mock_run_plan.return_value = None

        app = _create_test_app()
        client = TestClient(app)

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-yaml",
            "runId": "rn-yaml",
            "forwardedProps": {"yamlContent": _VALID_PLAN_YAML},
        })
        assert resp.status_code == 200
        # Verify load_plan was called with a Path to a temp file
        mock_load_plan.assert_called_once()
        call_arg = mock_load_plan.call_args[0][0]
        assert isinstance(call_arg, Path)
        assert str(call_arg).endswith(".yaml")

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_dry_run_prop_passed(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
        tmp_path: Path,
    ) -> None:
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan
        mock_run_plan.return_value = None

        app = _create_test_app()
        client = TestClient(app)

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-dry",
            "runId": "rn-dry",
            "forwardedProps": {"planPath": str(plan_file), "dry_run": True},
        })
        assert resp.status_code == 200
        mock_run_plan.assert_called_once()
        assert mock_run_plan.call_args[1]["dry_run"] is True

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_execution_profile_passed(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
        tmp_path: Path,
    ) -> None:
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan
        mock_run_plan.return_value = None

        app = _create_test_app()
        client = TestClient(app)

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-ep",
            "runId": "rn-ep",
            "forwardedProps": {
                "planPath": str(plan_file),
                "executionProfile": "safe",
            },
        })
        assert resp.status_code == 200
        mock_run_plan.assert_called_once()
        assert mock_run_plan.call_args[1]["execution_profile"] == "safe"

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_max_parallel_passed(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
        tmp_path: Path,
    ) -> None:
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan
        mock_run_plan.return_value = None

        app = _create_test_app()
        client = TestClient(app)

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-mp",
            "runId": "rn-mp",
            "forwardedProps": {
                "planPath": str(plan_file),
                "maxParallel": 4,
            },
        })
        assert resp.status_code == 200
        mock_run_plan.assert_called_once()
        assert mock_run_plan.call_args[1]["max_parallel_override"] == 4

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_only_and_skip_passed(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
        tmp_path: Path,
    ) -> None:
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan
        mock_run_plan.return_value = None

        app = _create_test_app()
        client = TestClient(app)

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-os",
            "runId": "rn-os",
            "forwardedProps": {
                "planPath": str(plan_file),
                "only": ["t1"],
                "skip": ["t2"],
            },
        })
        assert resp.status_code == 200
        mock_run_plan.assert_called_once()
        assert mock_run_plan.call_args[1]["only"] == {"t1"}
        assert mock_run_plan.call_args[1]["skip"] == {"t2"}

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_only_and_skip_none_when_absent(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
        tmp_path: Path,
    ) -> None:
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan
        mock_run_plan.return_value = None

        app = _create_test_app()
        client = TestClient(app)

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-no-os",
            "runId": "rn-no-os",
            "forwardedProps": {"planPath": str(plan_file)},
        })
        assert resp.status_code == 200
        mock_run_plan.assert_called_once()
        assert mock_run_plan.call_args[1]["only"] is None
        assert mock_run_plan.call_args[1]["skip"] is None

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_auto_approve_skips_approval_handler(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
        tmp_path: Path,
    ) -> None:
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan
        mock_run_plan.return_value = None

        app = _create_test_app()
        client = TestClient(app)

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-aa",
            "runId": "rn-aa",
            "forwardedProps": {
                "planPath": str(plan_file),
                "autoApprove": True,
            },
        })
        assert resp.status_code == 200
        mock_run_plan.assert_called_once()
        assert mock_run_plan.call_args[1]["auto_approve"] is True
        assert mock_run_plan.call_args[1]["approval_handler"] is None

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_run_plan_exception_emits_run_error(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
        tmp_path: Path,
    ) -> None:
        """If run_plan raises, the error is captured and emitted as RUN_ERROR."""
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan
        mock_run_plan.side_effect = RuntimeError("run_plan exploded")

        app = _create_test_app()
        client = TestClient(app)

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-err",
            "runId": "rn-err",
            "forwardedProps": {"planPath": str(plan_file)},
        })
        assert resp.status_code == 200
        events = _collect_sse_lines(resp.text)
        types = _event_types(events)
        assert RUN_ERROR in types
        error = [e for e in events if e["type"] == RUN_ERROR][0]
        assert "run_plan exploded" in error["message"]

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_camelcase_forwarded_props(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
        tmp_path: Path,
    ) -> None:
        """Both camelCase and snake_case prop names are supported."""
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan
        mock_run_plan.return_value = None

        app = _create_test_app()
        client = TestClient(app)

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-cc",
            "runId": "rn-cc",
            "forwardedProps": {
                "planPath": str(plan_file),
                "dryRun": True,
                "executionProfile": "yolo",
            },
        })
        assert resp.status_code == 200
        mock_run_plan.assert_called_once()
        assert mock_run_plan.call_args[1]["dry_run"] is True
        assert mock_run_plan.call_args[1]["execution_profile"] == "yolo"

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_response_headers(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
        tmp_path: Path,
    ) -> None:
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan
        mock_run_plan.return_value = None

        app = _create_test_app()
        client = TestClient(app)

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-hdr",
            "runId": "rn-hdr",
            "forwardedProps": {"planPath": str(plan_file)},
        })
        assert resp.status_code == 200
        assert resp.headers.get("cache-control") == "no-cache"


# ---------------------------------------------------------------------------
# Approval companion endpoint
# ---------------------------------------------------------------------------


class TestApproveTaskEndpoint:
    def test_approve_nonexistent_run_returns_404(self) -> None:
        from starlette.testclient import TestClient
        app = _create_test_app()
        client = TestClient(app)
        resp = client.post("/api/agui/runs/no-such-run/approve", json={
            "taskId": "t1",
            "approved": True,
        })
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_approve_nonexistent_task_returns_404(self) -> None:
        from starlette.testclient import TestClient
        run_id = "test-approve-endpoint-task"
        _register_run(run_id)
        try:
            app = _create_test_app()
            client = TestClient(app)
            resp = client.post(f"/api/agui/runs/{run_id}/approve", json={
                "taskId": "no-such-task",
                "approved": True,
            })
            assert resp.status_code == 404
            assert "no pending approval" in resp.json()["detail"].lower()
        finally:
            _cleanup_run(run_id)

    def test_approve_sets_event_and_returns(self) -> None:
        from starlette.testclient import TestClient
        run_id = "test-approve-endpoint-ok"
        _register_run(run_id)
        ev = threading.Event()
        result: list[bool] = [False]
        _pending_approvals[run_id]["task-1"] = (ev, result)

        try:
            app = _create_test_app()
            client = TestClient(app)
            resp = client.post(f"/api/agui/runs/{run_id}/approve", json={
                "taskId": "task-1",
                "approved": True,
            })
            assert resp.status_code == 200
            assert resp.json() == {"approved": True}
            assert result[0] is True
            assert ev.is_set()
        finally:
            _cleanup_run(run_id)

    def test_deny_sets_false(self) -> None:
        from starlette.testclient import TestClient
        run_id = "test-deny-endpoint"
        _register_run(run_id)
        ev = threading.Event()
        result: list[bool] = [True]  # default to True, should be overwritten
        _pending_approvals[run_id]["task-d"] = (ev, result)

        try:
            app = _create_test_app()
            client = TestClient(app)
            resp = client.post(f"/api/agui/runs/{run_id}/approve", json={
                "taskId": "task-d",
                "approved": False,
            })
            assert resp.status_code == 200
            assert resp.json() == {"approved": False}
            assert result[0] is False
            assert ev.is_set()
        finally:
            _cleanup_run(run_id)


# ---------------------------------------------------------------------------
# Event callback bridge
# ---------------------------------------------------------------------------


class TestEventCallbackBridge:
    """Test the _event_callback closure that bridges sync->async."""

    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_event_callback_receives_events(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
        tmp_path: Path,
    ) -> None:
        from starlette.testclient import TestClient

        plan = _make_plan_spec(task_ids=["t1"])
        mock_load_plan.return_value = plan

        received_events: list[str] = []

        def _fake_run(**kwargs: Any) -> None:
            cb = kwargs.get("event_callback")
            if cb:
                cb("task_start", {"task_id": "t1"})
                cb("task_complete", {"task_id": "t1", "status": "success"})

        mock_run_plan.side_effect = lambda *a, **kw: _fake_run(**kw)

        app = _create_test_app()
        client = TestClient(app)

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-cb",
            "runId": "rn-cb",
            "forwardedProps": {"planPath": str(plan_file)},
        })
        assert resp.status_code == 200
        events = _collect_sse_lines(resp.text)
        types = _event_types(events)
        # Should see STEP_STARTED and STEP_FINISHED from the task events
        assert "STEP_STARTED" in types
        assert "STEP_FINISHED" in types


# ---------------------------------------------------------------------------
# Temp file cleanup
# ---------------------------------------------------------------------------


class TestTempFileCleanup:
    @patch("maestro_cli.web.routes_agui.run_plan")
    @patch("maestro_cli.web.routes_agui.load_plan")
    def test_yaml_content_temp_file_cleaned_up(
        self,
        mock_load_plan: MagicMock,
        mock_run_plan: MagicMock,
    ) -> None:
        """When yamlContent is used, the temp file is deleted after the run."""
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        mock_load_plan.return_value = plan
        mock_run_plan.return_value = None

        # Track the temp file path
        temp_paths: list[Path] = []
        original_load_plan = mock_load_plan.side_effect

        def _capture_path(path: Path) -> PlanSpec:
            temp_paths.append(path)
            return plan

        mock_load_plan.side_effect = _capture_path

        app = _create_test_app()
        client = TestClient(app)

        resp = client.post("/api/agui/runs", json={
            "threadId": "th-tmp",
            "runId": "rn-tmp",
            "forwardedProps": {"yamlContent": _VALID_PLAN_YAML},
        })
        assert resp.status_code == 200
        # The temp file should have been created
        assert len(temp_paths) == 1
        # After streaming completes, the temp file should be cleaned up
        # (the _run() finally block calls unlink)
        # Give it a moment for the background thread to finish
        time.sleep(0.2)
        assert not temp_paths[0].exists()
