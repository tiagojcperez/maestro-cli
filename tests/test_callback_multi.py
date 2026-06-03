from __future__ import annotations

import threading
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from maestro_cli.models import (
    PlanDefaults,
    PlanRunResult,
    PlanSpec,
    TaskResult,
    TaskSpec,
)
from maestro_cli.multi import run_multi_plan
from maestro_cli.replan import replan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_PLAN_YAML = """\
version: 1
name: {name}
tasks:
  - id: t1
    command: echo ok
"""


def _write_plan(tmp_path: Path, name: str = "test-plan") -> Path:
    plan_file = tmp_path / f"{name}.yaml"
    plan_file.write_text(_MINIMAL_PLAN_YAML.format(name=name), encoding="utf-8")
    return plan_file


def _make_result(plan_name: str, *, success: bool = True) -> PlanRunResult:
    now = datetime.now(UTC)
    return PlanRunResult(
        plan_name=plan_name,
        run_id=f"run-{plan_name}",
        run_path=Path("/tmp/fake-runs"),
        started_at=now,
        finished_at=now,
        success=success,
        task_results={
            "t1": TaskResult(
                task_id="t1",
                status="success" if success else "failed",
                exit_code=0 if success else 1,
            )
        },
    )


def _make_run_plan_with_events(
    plan_name: str,
    *,
    success: bool = True,
) -> Callable[..., PlanRunResult]:
    """Return a mock run_plan that fires synthetic events through event_callback."""

    def _mock_run_plan(
        plan: Any,
        *,
        event_callback: Callable[[str, dict[str, object]], None] | None = None,
        **_kwargs: Any,
    ) -> PlanRunResult:
        result = _make_result(plan.name, success=success)
        if event_callback is not None:
            run_id = f"run-{plan.name}"
            event_callback("run_start", {"event": "run_start", "run_id": run_id, "plan": plan.name, "tasks": 1})
            event_callback("task_start", {"event": "task_start", "run_id": run_id, "task_id": "t1", "plan": plan.name})
            event_callback("task_complete", {"event": "task_complete", "run_id": run_id, "task_id": "t1", "plan": plan.name})
            event_callback("run_complete", {"event": "run_complete", "run_id": run_id, "plan": plan.name, "success": success})
        return result

    return _mock_run_plan


# ---------------------------------------------------------------------------
# 1. Sequential multi-plan callback
# ---------------------------------------------------------------------------


def test_multi_sequential_callback(tmp_path: Path) -> None:
    """Callback receives events from both plans in sequential order."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plan_a = _write_plan(tmp_path / "a", name="plan-a")
    plan_b = _write_plan(tmp_path / "b", name="plan-b")

    events: list[dict[str, Any]] = []

    plan_mock_a = _make_run_plan_with_events("plan-a")
    plan_mock_b = _make_run_plan_with_events("plan-b")
    call_counter = {"n": 0}

    def combined_run_plan(plan: Any, **kwargs: Any) -> PlanRunResult:
        call_counter["n"] += 1
        if plan.name == "plan-a":
            return plan_mock_a(plan, **kwargs)
        return plan_mock_b(plan, **kwargs)

    def callback(event_name: str, payload: dict[str, object]) -> None:
        events.append(dict(payload))

    with patch("maestro_cli.multi.run_plan", side_effect=combined_run_plan):
        result = run_multi_plan(
            [str(plan_a), str(plan_b)],
            parallel=False,
            event_callback=callback,
        )

    assert result.success is True

    # Both plans fired events
    plan_names_seen = {str(e.get("plan")) for e in events if "plan" in e}
    assert "plan-a" in plan_names_seen
    assert "plan-b" in plan_names_seen

    # Each plan has run_start + run_complete
    for plan_name in ("plan-a", "plan-b"):
        plan_events = [e["event"] for e in events if e.get("plan") == plan_name]
        assert "run_start" in plan_events, f"no run_start for {plan_name}"
        assert "run_complete" in plan_events, f"no run_complete for {plan_name}"

    # Plan A events precede Plan B events (sequential)
    a_indices = [i for i, e in enumerate(events) if e.get("plan") == "plan-a"]
    b_indices = [i for i, e in enumerate(events) if e.get("plan") == "plan-b"]
    assert a_indices, "no plan-a events"
    assert b_indices, "no plan-b events"
    assert max(a_indices) < min(b_indices), (
        "plan-a events should all precede plan-b events in sequential mode"
    )


# ---------------------------------------------------------------------------
# 2. Parallel multi-plan callback
# ---------------------------------------------------------------------------


def test_multi_parallel_callback(tmp_path: Path) -> None:
    """Callback receives events from both plans even when interleaved."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plan_a = _write_plan(tmp_path / "a", name="plan-a")
    plan_b = _write_plan(tmp_path / "b", name="plan-b")

    events: list[dict[str, Any]] = []
    events_lock = threading.Lock()

    def combined_run_plan(plan: Any, **kwargs: Any) -> PlanRunResult:
        mock = _make_run_plan_with_events(plan.name)
        return mock(plan, **kwargs)

    def callback(event_name: str, payload: dict[str, object]) -> None:
        with events_lock:
            events.append(dict(payload))

    with patch("maestro_cli.multi.run_plan", side_effect=combined_run_plan):
        result = run_multi_plan(
            [str(plan_a), str(plan_b)],
            parallel=True,
            event_callback=callback,
        )

    assert result.success is True

    plan_names_seen = {str(e.get("plan")) for e in events if "plan" in e}
    assert "plan-a" in plan_names_seen
    assert "plan-b" in plan_names_seen

    for plan_name in ("plan-a", "plan-b"):
        plan_events = [e["event"] for e in events if e.get("plan") == plan_name]
        assert "run_start" in plan_events
        assert "run_complete" in plan_events


# ---------------------------------------------------------------------------
# 3. event_callback=None works normally
# ---------------------------------------------------------------------------


def test_multi_callback_none(tmp_path: Path) -> None:
    """run_multi_plan with event_callback=None completes without error."""
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plan_a = _write_plan(tmp_path / "a", name="plan-a")
    plan_b = _write_plan(tmp_path / "b", name="plan-b")

    def combined_run_plan(plan: Any, **kwargs: Any) -> PlanRunResult:
        # event_callback should be None — verify it is not called
        cb = kwargs.get("event_callback")
        assert cb is None, "event_callback should be None"
        return _make_result(plan.name)

    with patch("maestro_cli.multi.run_plan", side_effect=combined_run_plan):
        result = run_multi_plan(
            [str(plan_a), str(plan_b)],
            parallel=False,
            event_callback=None,
        )

    assert result.success is True


# ---------------------------------------------------------------------------
# 4. replan() forwards event_callback
# ---------------------------------------------------------------------------


def test_replan_callback(tmp_path: Path) -> None:
    """replan() passes event_callback through to run_plan on each attempt."""
    plan_file = _write_plan(tmp_path, name="replan-test")

    events: list[dict[str, Any]] = []

    def callback(event_name: str, payload: dict[str, object]) -> None:
        events.append(dict(payload))

    mock_run = _make_run_plan_with_events("replan-test", success=True)

    with patch("maestro_cli.replan.run_plan", side_effect=mock_run):
        state = replan(
            plan_file,
            max_attempts=3,
            auto_approve=True,
            event_callback=callback,
        )

    assert state.final_success is True
    assert state.status == "success"

    # Callback received events from the run attempt
    event_names = [e["event"] for e in events]
    assert "run_start" in event_names
    assert "run_complete" in event_names


# ---------------------------------------------------------------------------
# 5. run_start payload — plan name availability audit
# ---------------------------------------------------------------------------


def test_multi_callback_identifies_plan(tmp_path: Path) -> None:
    """run_start payload must include the plan name for TUI multi-plan display.

    This test documents the expected contract: each run_start event fired
    through event_callback must include a ``plan`` key so the TUI can label
    concurrent progress rows.  If this assertion fails, document the gap.
    """
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    plan_a = _write_plan(tmp_path / "a", name="plan-alpha")
    plan_b = _write_plan(tmp_path / "b", name="plan-beta")

    run_start_payloads: list[dict[str, Any]] = []

    def combined_run_plan(plan: Any, **kwargs: Any) -> PlanRunResult:
        mock = _make_run_plan_with_events(plan.name)
        return mock(plan, **kwargs)

    def callback(event_name: str, payload: dict[str, object]) -> None:
        if event_name == "run_start":
            run_start_payloads.append(dict(payload))

    with patch("maestro_cli.multi.run_plan", side_effect=combined_run_plan):
        run_multi_plan(
            [str(plan_a), str(plan_b)],
            parallel=False,
            event_callback=callback,
        )

    assert len(run_start_payloads) == 2, (
        f"Expected 2 run_start events (one per plan), got {len(run_start_payloads)}"
    )

    for payload in run_start_payloads:
        # GAP NOTE: if this assertion fails, the run_start payload has no
        # plan identifier — the TUI will not be able to label rows per-plan.
        assert "plan" in payload, (
            "# GAP: run_start payload has no plan_name — TUI needs this for "
            "multi-plan display. "
            f"Actual keys: {list(payload.keys())}"
        )

    plan_names_in_events = {str(p["plan"]) for p in run_start_payloads}
    assert "plan-alpha" in plan_names_in_events
    assert "plan-beta" in plan_names_in_events
