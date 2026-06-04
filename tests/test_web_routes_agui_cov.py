"""Coverage tests for web/routes_agui.py exception-handling branches.

Targets the two defensive ``except`` blocks inside ``agui_run``:

* ``_event_callback``: ``except (RuntimeError, asyncio.QueueFull)`` — fired when
  the bridge from the sync engine callback to the async queue cannot enqueue
  (event loop closed or queue full).
* ``_run`` finally block: ``except RuntimeError`` — fired when the terminating
  ``None`` sentinel cannot be scheduled because the loop is gone.

Both branches are driven by replacing the captured event loop with a fake whose
``call_soon_threadsafe`` raises, so the closures take their error path. The
engine (``run_plan``) and plan loader (``load_plan``) are always mocked.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from maestro_cli.models import PlanSpec, TaskSpec
from maestro_cli.web.routes_agui import router


# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_VALID_PLAN_YAML = """\
version: 1
name: agui-cov-test
tasks:
  - id: t1
    command: "echo hello"
"""


def _make_plan_spec(task_ids: list[str] | None = None) -> PlanSpec:
    ids = task_ids or ["t1"]
    tasks = [TaskSpec(id=tid, command="echo hi") for tid in ids]
    return PlanSpec(version=1, name="agui-cov-test", tasks=tasks, run_dir=".maestro-runs")


def _create_test_app() -> Any:
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api")
    return app


class _RaisingLoop:
    """Stand-in event loop whose ``call_soon_threadsafe`` raises.

    Used to force the bridge closures in ``agui_run`` down their except paths.
    Only the attribute the production code touches (``call_soon_threadsafe``)
    is implemented; the real running loop still drives the async generator.

    The callback path schedules a payload dict; the finally block schedules the
    ``None`` sentinel. We branch on that so the finally block can always raise a
    ``RuntimeError`` (the only type its ``except`` catches) while the callback
    path raises whichever exception the test wants to exercise — keeping the
    background thread free of uncaught exceptions.
    """

    def __init__(self, callback_exc: BaseException) -> None:
        self._callback_exc = callback_exc
        self.calls = 0
        self.sentinel_calls = 0
        self.callback_calls = 0

    def call_soon_threadsafe(self, callback: Any, *args: Any, **kwargs: Any) -> None:
        self.calls += 1
        # The finally block schedules ``put_nowait`` with the ``None`` sentinel.
        if args and args[0] is None:
            self.sentinel_calls += 1
            raise RuntimeError("loop closed (sentinel)")
        self.callback_calls += 1
        raise self._callback_exc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestEventCallbackEnqueueFailure:
    """Drive the ``except (RuntimeError, asyncio.QueueFull)`` bridge branch."""

    @pytest.fixture(autouse=True)
    def _confine_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.get_project_root", lambda: tmp_path,
        )

    def _run_with_raising_loop(
        self, exc: BaseException, tmp_path: Path
    ) -> _RaisingLoop:
        """Run the endpoint with a loop whose scheduling always raises ``exc``.

        ``run_plan`` is mocked to invoke the engine callback (so the callback's
        except path is exercised) and the loop is replaced with one that raises
        on every ``call_soon_threadsafe`` (so the finally-block except path is
        exercised too). ``_STREAM_TIMEOUT`` is shortened because the terminating
        sentinel can never be enqueued, so the generator must time out.
        """
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        raising_loop = _RaisingLoop(exc)

        def _fake_run(**kwargs: Any) -> None:
            cb = kwargs.get("event_callback")
            if cb:
                # Each callback invocation hits the bridge's except path.
                cb("task_start", {"task_id": "t1"})
                cb("task_complete", {"task_id": "t1", "status": "success"})

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        with patch("maestro_cli.web.routes_agui.load_plan", return_value=plan), \
                patch(
                    "maestro_cli.web.routes_agui.run_plan",
                    side_effect=lambda *a, **kw: _fake_run(**kw),
                ), \
                patch(
                    "maestro_cli.web.routes_agui.asyncio.get_event_loop",
                    return_value=raising_loop,
                ), \
                patch("maestro_cli.web.routes_agui._STREAM_TIMEOUT", 0.2):
            app = _create_test_app()
            client = TestClient(app)
            resp = client.post(
                "/api/agui/runs",
                json={
                    "threadId": "th-cov",
                    "runId": "rn-cov",
                    "forwardedProps": {"planPath": str(plan_file)},
                },
            )
            assert resp.status_code == 200
            # Force the streaming body (and thus the generator) to be consumed.
            _ = resp.text
        return raising_loop

    def test_runtime_error_on_enqueue_is_swallowed(self, tmp_path: Path) -> None:
        """RuntimeError from call_soon_threadsafe is caught (loop-closed case).

        Covers both the callback bridge except and the finally-block except,
        since the same raising loop is used for the sentinel scheduling.
        """
        raising_loop = self._run_with_raising_loop(RuntimeError("loop closed"), tmp_path)
        # Callback bridge fired (>=1) and the finally sentinel scheduling fired.
        assert raising_loop.callback_calls >= 1
        assert raising_loop.sentinel_calls >= 1

    def test_queue_full_on_enqueue_is_swallowed(self, tmp_path: Path) -> None:
        """QueueFull from the enqueue path is caught by the tuple except."""
        raising_loop = self._run_with_raising_loop(asyncio.QueueFull(), tmp_path)
        assert raising_loop.callback_calls >= 1
        assert raising_loop.sentinel_calls >= 1


class TestRunFinallySentinelFailure:
    """Isolate the finally-block ``except RuntimeError`` for the sentinel put."""

    @pytest.fixture(autouse=True)
    def _confine_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            "maestro_cli.web.routes_api.get_project_root", lambda: tmp_path,
        )

    def test_finally_runtime_error_swallowed_when_run_plan_raises(
        self, tmp_path: Path
    ) -> None:
        """run_plan raises AND the sentinel scheduling raises RuntimeError.

        The endpoint must still return 200; the background thread's finally
        block swallows the scheduling RuntimeError and proceeds to cleanup.
        """
        from starlette.testclient import TestClient

        plan = _make_plan_spec()
        raising_loop = _RaisingLoop(RuntimeError("loop gone"))

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(_VALID_PLAN_YAML, encoding="utf-8")

        with patch("maestro_cli.web.routes_agui.load_plan", return_value=plan), \
                patch(
                    "maestro_cli.web.routes_agui.run_plan",
                    side_effect=RuntimeError("engine boom"),
                ), \
                patch(
                    "maestro_cli.web.routes_agui.asyncio.get_event_loop",
                    return_value=raising_loop,
                ), \
                patch("maestro_cli.web.routes_agui._STREAM_TIMEOUT", 0.2):
            app = _create_test_app()
            client = TestClient(app)
            resp = client.post(
                "/api/agui/runs",
                json={
                    "threadId": "th-fin",
                    "runId": "rn-fin",
                    "forwardedProps": {"planPath": str(plan_file)},
                },
            )
            assert resp.status_code == 200
            _ = resp.text

        # The finally block attempted to schedule the sentinel at least once;
        # the engine error means the callback path is never reached here.
        assert raising_loop.sentinel_calls >= 1
