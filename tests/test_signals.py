"""Tests for T2.2 — Mid-task signals (agent-to-scheduler communication)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.models import (
    SIGNAL_TYPES,
    PlanDefaults,
    PlanSpec,
    SignalType,
    TaskResult,
    TaskSignal,
    TaskSpec,
)
from maestro_cli.runners import _SignalHandler, _parse_signal_line


# ---------------------------------------------------------------------------
# _parse_signal_line
# ---------------------------------------------------------------------------

class TestParseSignalLine:
    def test_valid_progress_signal(self) -> None:
        line = '[MAESTRO_SIGNAL] {"type": "progress", "pct": 50}'
        result = _parse_signal_line(line)
        assert result is not None
        assert result["type"] == "progress"
        assert result["pct"] == 50

    def test_valid_metric_signal(self) -> None:
        line = '[MAESTRO_SIGNAL] {"type": "metric", "name": "latency", "value": 42.5}'
        result = _parse_signal_line(line)
        assert result is not None
        assert result["type"] == "metric"
        assert result["value"] == 42.5

    def test_valid_log_signal(self) -> None:
        line = '[MAESTRO_SIGNAL] {"type": "log", "level": "warn", "message": "deprecated API"}'
        result = _parse_signal_line(line)
        assert result is not None
        assert result["type"] == "log"

    def test_valid_artifact_signal(self) -> None:
        line = '[MAESTRO_SIGNAL] {"type": "artifact", "path": "coverage.html", "label": "Coverage"}'
        result = _parse_signal_line(line)
        assert result is not None
        assert result["type"] == "artifact"

    def test_valid_timeout_extend_signal(self) -> None:
        line = '[MAESTRO_SIGNAL] {"type": "timeout_extend", "additional_sec": 300}'
        result = _parse_signal_line(line)
        assert result is not None
        assert result["additional_sec"] == 300

    def test_valid_budget_query_signal(self) -> None:
        line = '[MAESTRO_SIGNAL] {"type": "budget_query"}'
        result = _parse_signal_line(line)
        assert result is not None
        assert result["type"] == "budget_query"

    def test_valid_checkpoint_signal(self) -> None:
        line = '[MAESTRO_SIGNAL] {"type": "checkpoint", "name": "phase-1", "data": {"files": 3}}'
        result = _parse_signal_line(line)
        assert result is not None
        assert result["name"] == "phase-1"

    def test_not_a_signal(self) -> None:
        assert _parse_signal_line("regular stdout line") is None
        assert _parse_signal_line('{"type": "result"}') is None
        assert _parse_signal_line("[maestro] starting task") is None

    def test_invalid_json(self) -> None:
        assert _parse_signal_line("[MAESTRO_SIGNAL] not-json") is None

    def test_unknown_signal_type(self) -> None:
        line = '[MAESTRO_SIGNAL] {"type": "unknown_signal"}'
        assert _parse_signal_line(line) is None

    def test_missing_type_field(self) -> None:
        line = '[MAESTRO_SIGNAL] {"pct": 50}'
        assert _parse_signal_line(line) is None

    def test_oversized_line_rejected(self) -> None:
        payload = {"type": "log", "message": "x" * 5000}
        line = f"[MAESTRO_SIGNAL] {json.dumps(payload)}"
        assert _parse_signal_line(line) is None

    def test_non_dict_json(self) -> None:
        line = '[MAESTRO_SIGNAL] [1, 2, 3]'
        assert _parse_signal_line(line) is None


# ---------------------------------------------------------------------------
# _SignalHandler
# ---------------------------------------------------------------------------

class TestSignalHandler:
    def _make_handler(
        self,
        deadline_ref: list[float] | None = None,
        budget_getter: Any = None,
    ) -> tuple[_SignalHandler, list[tuple[str, dict[str, object]]]]:
        events: list[tuple[str, dict[str, object]]] = []

        def _cb(event: str, data: dict[str, object]) -> None:
            events.append((event, dict(data)))

        handler = _SignalHandler(
            task_id="test-task",
            workdir=Path("."),
            event_callback=_cb,
            budget_getter=budget_getter,
            deadline_ref=deadline_ref,
        )
        return handler, events

    def test_progress_emits_event(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "progress", "pct": 75, "step": "running tests"})
        assert len(events) == 1
        assert events[0][0] == "task_progress"
        assert events[0][1]["pct"] == 75
        assert events[0][1]["step"] == "running tests"
        assert handler.last_progress_pct == 75

    def test_progress_clamps_pct(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "progress", "pct": 150})
        assert events[0][1]["pct"] == 100
        handler.handle({"type": "progress", "pct": -10})
        assert events[1][1]["pct"] == 0

    def test_progress_rejects_non_numeric(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "progress", "pct": "fifty"})
        assert len(events) == 0

    def test_metric_emits_event(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "metric", "name": "latency", "value": 42.5})
        assert len(events) == 1
        assert events[0][0] == "task_metric"
        assert events[0][1]["name"] == "latency"
        assert events[0][1]["value"] == 42.5

    def test_metric_rejects_bad_types(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "metric", "name": 123, "value": 1.0})
        assert len(events) == 0
        handler.handle({"type": "metric", "name": "x", "value": "not-a-number"})
        assert len(events) == 0

    def test_log_emits_event(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "log", "level": "error", "message": "something broke"})
        assert events[0][0] == "task_signal_log"
        assert events[0][1]["level"] == "error"

    def test_log_normalizes_unknown_level(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "log", "level": "critical", "message": "test"})
        assert events[0][1]["level"] == "info"

    def test_log_rejects_empty_message(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "log", "level": "info", "message": ""})
        assert len(events) == 0

    def test_artifact_emits_event(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "artifact", "path": "reports/coverage.html", "label": "Coverage"})
        assert events[0][0] == "task_artifact"
        assert handler.artifacts == [{"path": "reports/coverage.html", "label": "Coverage"}]

    def test_artifact_rejects_absolute_path(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "artifact", "path": "/etc/passwd"})
        assert len(events) == 0
        assert len(handler.artifacts) == 0

    def test_artifact_rejects_parent_traversal(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "artifact", "path": "../../../etc/passwd"})
        assert len(events) == 0

    def test_timeout_extend(self) -> None:
        deadline_ref = [time.monotonic() + 100]
        old_deadline = deadline_ref[0]
        handler, events = self._make_handler(deadline_ref=deadline_ref)
        handler.handle({"type": "timeout_extend", "additional_sec": 60, "reason": "large codebase"})
        assert events[0][0] == "timeout_extended"
        assert deadline_ref[0] > old_deadline

    def test_timeout_extend_capped(self) -> None:
        deadline_ref = [time.monotonic() + 100]
        handler, events = self._make_handler(deadline_ref=deadline_ref)
        handler.handle({"type": "timeout_extend", "additional_sec": 99999})
        # Should be capped at _TIMEOUT_EXTEND_MAX (1800)
        assert events[0][1]["additional_sec"] == 1800

    def test_timeout_extend_no_deadline_ref(self) -> None:
        handler, events = self._make_handler(deadline_ref=None)
        handler.handle({"type": "timeout_extend", "additional_sec": 60})
        assert len(events) == 0

    def test_budget_query_with_getter(self) -> None:
        def _getter() -> tuple[float | None, float | None]:
            return 7.50, 10.00

        handler, events = self._make_handler(budget_getter=_getter)
        handler.handle({"type": "budget_query"})
        assert events[0][0] == "budget_query"
        assert events[0][1]["remaining_usd"] == 7.50
        assert events[0][1]["limit_usd"] == 10.00

    def test_budget_query_without_getter(self) -> None:
        handler, events = self._make_handler(budget_getter=None)
        handler.handle({"type": "budget_query"})
        assert events[0][1]["remaining_usd"] is None

    def test_checkpoint_emits_event(self) -> None:
        handler, events = self._make_handler()
        handler.handle({"type": "checkpoint", "name": "phase-1", "data": {"files": 3}})
        assert events[0][0] == "task_checkpoint_signal"
        assert events[0][1]["name"] == "phase-1"

    def test_rate_limiting(self) -> None:
        handler, events = self._make_handler()
        for i in range(15):
            handler.handle({"type": "progress", "pct": i})
        assert len(events) == 10  # capped at _SIGNAL_MAX_PER_SEC

    def test_total_signal_cap(self) -> None:
        handler, events = self._make_handler()
        # Fill to max
        for i in range(1001):
            handler._rate_window.clear()  # bypass rate limit
            handler.handle({"type": "log", "level": "info", "message": f"msg-{i}"})
        assert len(handler.signals) == 1000  # capped at _SIGNAL_MAX_TOTAL


# ---------------------------------------------------------------------------
# SIGNAL_TYPES constant
# ---------------------------------------------------------------------------

class TestSignalTypes:
    def test_all_types_present(self) -> None:
        expected = {"progress", "metric", "log", "artifact", "timeout_extend", "budget_query", "checkpoint", "compress"}
        assert SIGNAL_TYPES == expected


# ---------------------------------------------------------------------------
# TaskSignal dataclass
# ---------------------------------------------------------------------------

class TestTaskSignal:
    def test_to_dict(self) -> None:
        sig = TaskSignal(
            signal_type="progress",
            timestamp="2026-03-19T12:00:00Z",
            payload={"pct": 50},
        )
        d = sig.to_dict()
        assert d["signal_type"] == "progress"
        assert d["timestamp"] == "2026-03-19T12:00:00Z"
        assert d["payload"]["pct"] == 50


# ---------------------------------------------------------------------------
# TaskResult signal fields
# ---------------------------------------------------------------------------

class TestTaskResultSignals:
    def test_signals_in_to_dict(self) -> None:
        from datetime import datetime, timezone

        sig = TaskSignal(signal_type="progress", timestamp="2026-03-19T12:00:00Z", payload={"pct": 50})
        now = datetime.now(tz=timezone.utc)
        result = TaskResult(
            task_id="t1",
            status="success",
            started_at=now,
            finished_at=now,
            duration_sec=1.0,
            command="echo",
            log_path=Path("/tmp/log"),
            result_path=Path("/tmp/result"),
            signals_received=[sig],
            artifacts=[{"path": "out.html", "label": "Report"}],
            last_progress_pct=50,
        )
        d = result.to_dict()
        assert "signals_received" in d
        assert len(d["signals_received"]) == 1
        assert d["artifacts"] == [{"path": "out.html", "label": "Report"}]
        assert d["last_progress_pct"] == 50

    def test_empty_signals_not_in_dict(self) -> None:
        from datetime import datetime, timezone

        now = datetime.now(tz=timezone.utc)
        result = TaskResult(
            task_id="t1",
            status="success",
            started_at=now,
            finished_at=now,
            duration_sec=1.0,
            command="echo",
            log_path=Path("/tmp/log"),
            result_path=Path("/tmp/result"),
        )
        d = result.to_dict()
        assert "signals_received" not in d
        assert "artifacts" not in d
        assert "last_progress_pct" not in d


# ---------------------------------------------------------------------------
# Loader: signals field parsed
# ---------------------------------------------------------------------------

class TestLoaderSignals:
    def test_signals_parsed_on_task(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: test\ntasks:\n"
            "  - id: t1\n    engine: claude\n    signals: true\n    prompt: hello\n"
            "  - id: t2\n    engine: claude\n    prompt: world\n",
            encoding="utf-8",
        )
        plan = load_plan(str(plan_yaml))
        assert plan.tasks[0].signals is True
        assert plan.tasks[1].signals is False

    def test_signals_parsed_on_defaults(self, tmp_path: Path) -> None:
        from maestro_cli.loader import load_plan

        plan_yaml = tmp_path / "plan.yaml"
        plan_yaml.write_text(
            "version: 1\nname: test\ndefaults:\n  signals: true\n"
            "tasks:\n  - id: t1\n    engine: claude\n    prompt: hello\n",
            encoding="utf-8",
        )
        plan = load_plan(str(plan_yaml))
        assert plan.defaults.signals is True


# ---------------------------------------------------------------------------
# TaskSpec.signals in to_dict
# ---------------------------------------------------------------------------

class TestTaskSpecSignals:
    def test_signals_in_to_dict(self) -> None:
        task = TaskSpec(id="t1", signals=True)
        d = task.to_dict()
        assert d["signals"] is True

    def test_signals_default_false(self) -> None:
        task = TaskSpec(id="t1")
        assert task.signals is False


# ---------------------------------------------------------------------------
# Agent-Triggered Context Compression — Signal tests
# ---------------------------------------------------------------------------

class TestCompressSignalType:
    """Tests for the 'compress' signal type added to SIGNAL_TYPES."""

    def test_compress_in_signal_types(self) -> None:
        """'compress' must be present in SIGNAL_TYPES."""
        assert "compress" in SIGNAL_TYPES

    def test_signal_types_includes_all_eight(self) -> None:
        """SIGNAL_TYPES should now contain 8 types (7 original + compress)."""
        expected = {
            "progress", "metric", "log", "artifact",
            "timeout_extend", "budget_query", "checkpoint",
            "compress",
        }
        assert SIGNAL_TYPES == expected


class TestParseSignalLineCompress:
    """Tests for _parse_signal_line recognizing the compress signal type."""

    def test_compress_signal_parsed(self) -> None:
        line = '[MAESTRO_SIGNAL] {"type": "compress", "reason": "context too large"}'
        result = _parse_signal_line(line)
        assert result is not None
        assert result["type"] == "compress"
        assert result["reason"] == "context too large"

    def test_compress_signal_without_reason(self) -> None:
        line = '[MAESTRO_SIGNAL] {"type": "compress"}'
        result = _parse_signal_line(line)
        assert result is not None
        assert result["type"] == "compress"


class TestSignalHandlerCompress:
    """Tests for _SignalHandler compress handling."""

    def _make_handler(
        self,
    ) -> tuple[_SignalHandler, list[tuple[str, dict[str, object]]]]:
        events: list[tuple[str, dict[str, object]]] = []

        def _cb(event: str, data: dict[str, object]) -> None:
            events.append((event, dict(data)))

        handler = _SignalHandler(
            task_id="test-task",
            workdir=Path("."),
            event_callback=_cb,
        )
        return handler, events

    def test_compress_requested_starts_false(self) -> None:
        """compress_requested must default to False on a new handler."""
        handler, _ = self._make_handler()
        assert handler.compress_requested is False

    def test_handle_compress_sets_flag(self) -> None:
        """Handling a compress signal sets compress_requested to True."""
        handler, _ = self._make_handler()
        handler.handle({"type": "compress"})
        assert handler.compress_requested is True

    def test_handle_compress_emits_event(self) -> None:
        """Handling a compress signal emits context_compress_requested event."""
        handler, events = self._make_handler()
        handler.handle({"type": "compress", "reason": "tokens exceeded"})
        assert len(events) == 1
        assert events[0][0] == "context_compress_requested"
        assert events[0][1]["task_id"] == "test-task"
        assert events[0][1]["reason"] == "tokens exceeded"

    def test_handle_compress_without_reason(self) -> None:
        """Compress signal without reason field should emit event with empty reason."""
        handler, events = self._make_handler()
        handler.handle({"type": "compress"})
        assert len(events) == 1
        assert events[0][1]["reason"] == ""
