"""Tests for OTLP exporter (otel.py)."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from maestro_cli.otel import (
    _HAS_OTEL,
    _parse_iso,
    _load_manifest,
    _load_events,
    _safe_str,
    _safe_float,
    build_span_data,
    format_otel_json,
    export_run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_run(
    tmp_path: Path,
    plan_name: str = "test",
    success: bool = True,
    tasks: dict[str, dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
) -> Path:
    run_dir = tmp_path / f"20260321_{plan_name}"
    run_dir.mkdir(parents=True)

    if tasks is None:
        tasks = {
            "task-a": {
                "task_id": "task-a",
                "status": "success",
                "exit_code": 0,
                "duration_sec": 5.0,
                "cost_usd": 0.05,
                "started_at": "2026-03-21T10:00:00+00:00",
                "finished_at": "2026-03-21T10:00:05+00:00",
                "engine": "claude",
                "model": "sonnet",
                "token_usage": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "total_tokens": 1500,
                },
            }
        }

    manifest = {
        "plan_name": plan_name,
        "run_id": f"20260321_{plan_name}",
        "success": success,
        "execution_profile": "plan",
        "started_at": "2026-03-21T10:00:00+00:00",
        "finished_at": "2026-03-21T10:01:00+00:00",
        "total_cost_usd": sum(
            t.get("cost_usd", 0) or 0 for t in tasks.values()
        ),
        "total_tokens": sum(
            (t.get("token_usage") or {}).get("total_tokens", 0)
            for t in tasks.values()
        ),
        "task_results": tasks,
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )

    if events is None:
        events = [
            {"event": "run_start", "plan_name": plan_name},
            {"event": "task_start", "task_id": "task-a"},
            {"event": "task_complete", "task_id": "task-a", "status": "success"},
            {"event": "run_complete", "success": True},
        ]
    (run_dir / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )

    return run_dir


# ---------------------------------------------------------------------------
# TestParseIso
# ---------------------------------------------------------------------------

class TestParseIso:
    def test_valid_iso(self) -> None:
        dt = _parse_iso("2026-03-21T10:00:00+00:00")
        assert dt is not None
        assert dt.year == 2026

    def test_naive_iso(self) -> None:
        dt = _parse_iso("2026-03-21T10:00:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc

    def test_none(self) -> None:
        assert _parse_iso(None) is None

    def test_empty(self) -> None:
        assert _parse_iso("") is None

    def test_invalid(self) -> None:
        assert _parse_iso("not-a-date") is None

    def test_non_string(self) -> None:
        assert _parse_iso(12345) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# TestSafeHelpers
# ---------------------------------------------------------------------------

class TestSafeHelpers:
    def test_safe_str_none(self) -> None:
        assert _safe_str(None) == ""

    def test_safe_str_value(self) -> None:
        assert _safe_str("hello") == "hello"

    def test_safe_str_int(self) -> None:
        assert _safe_str(42) == "42"

    def test_safe_float_none(self) -> None:
        assert _safe_float(None) == 0.0

    def test_safe_float_value(self) -> None:
        assert _safe_float(3.14) == pytest.approx(3.14)

    def test_safe_float_string(self) -> None:
        assert _safe_float("not-a-number") == 0.0

    def test_safe_float_int(self) -> None:
        assert _safe_float(42) == 42.0


# ---------------------------------------------------------------------------
# TestLoadManifest
# ---------------------------------------------------------------------------

class TestLoadManifest:
    def test_valid_manifest(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        manifest = _load_manifest(run_dir)
        assert manifest is not None
        assert manifest["plan_name"] == "test"

    def test_no_manifest(self, tmp_path: Path) -> None:
        assert _load_manifest(tmp_path) is None

    def test_corrupt_manifest(self, tmp_path: Path) -> None:
        (tmp_path / "run_manifest.json").write_text("NOT JSON", encoding="utf-8")
        assert _load_manifest(tmp_path) is None


# ---------------------------------------------------------------------------
# TestLoadEvents
# ---------------------------------------------------------------------------

class TestLoadEvents:
    def test_valid_events(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        events = _load_events(run_dir)
        assert len(events) == 4

    def test_no_events_file(self, tmp_path: Path) -> None:
        assert _load_events(tmp_path) == []

    def test_corrupt_lines_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "events.jsonl").write_text(
            'NOT JSON\n{"event":"ok"}\n', encoding="utf-8",
        )
        events = _load_events(tmp_path)
        assert len(events) == 1

    def test_empty_lines_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "events.jsonl").write_text(
            '\n{"event":"ok"}\n\n', encoding="utf-8",
        )
        events = _load_events(tmp_path)
        assert len(events) == 1


# ---------------------------------------------------------------------------
# TestBuildSpanData
# ---------------------------------------------------------------------------

class TestBuildSpanData:
    def test_basic_run(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        data = build_span_data(run_dir)
        assert data is not None
        assert "root_span" in data
        assert "task_spans" in data
        assert len(data["task_spans"]) == 1

    def test_root_span_attributes(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        data = build_span_data(run_dir)
        root = data["root_span"]
        assert root["name"] == "maestro:test"
        assert root["attributes"]["maestro.plan.name"] == "test"
        assert root["attributes"]["maestro.run.success"] is True
        assert root["status"] == "OK"

    def test_task_span_attributes(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        data = build_span_data(run_dir)
        task = data["task_spans"][0]
        assert task["name"] == "task:task-a"
        attrs = task["attributes"]
        assert attrs["maestro.task.id"] == "task-a"
        assert attrs["maestro.task.status"] == "success"
        assert attrs["maestro.task.engine"] == "claude"
        assert attrs["maestro.task.model"] == "sonnet"
        assert attrs["maestro.task.cost_usd"] == pytest.approx(0.05)
        assert attrs["maestro.task.tokens.total_tokens"] == 1500
        assert task["status"] == "OK"

    def test_failed_run(self, tmp_path: Path) -> None:
        tasks = {
            "fail-task": {
                "task_id": "fail-task",
                "status": "failed",
                "exit_code": 1,
                "duration_sec": 10.0,
            }
        }
        run_dir = _write_run(tmp_path, success=False, tasks=tasks)
        data = build_span_data(run_dir)
        assert data["root_span"]["status"] == "ERROR"
        assert data["task_spans"][0]["status"] == "ERROR"

    def test_multiple_tasks(self, tmp_path: Path) -> None:
        tasks = {
            "a": {"task_id": "a", "status": "success", "exit_code": 0, "duration_sec": 1.0},
            "b": {"task_id": "b", "status": "failed", "exit_code": 1, "duration_sec": 2.0},
            "c": {"task_id": "c", "status": "skipped", "exit_code": None, "duration_sec": 0.0},
        }
        run_dir = _write_run(tmp_path, plan_name="multi", tasks=tasks)
        data = build_span_data(run_dir)
        assert len(data["task_spans"]) == 3
        statuses = {s["name"]: s["status"] for s in data["task_spans"]}
        assert statuses["task:a"] == "OK"
        assert statuses["task:b"] == "ERROR"
        assert statuses["task:c"] == "ERROR"  # skipped = not OK

    def test_no_manifest(self, tmp_path: Path) -> None:
        assert build_span_data(tmp_path) is None

    def test_dry_run_status(self, tmp_path: Path) -> None:
        tasks = {"a": {"task_id": "a", "status": "dry_run", "exit_code": 0, "duration_sec": 0.0}}
        run_dir = _write_run(tmp_path, tasks=tasks)
        data = build_span_data(run_dir)
        assert data["task_spans"][0]["status"] == "OK"

    def test_task_with_judge(self, tmp_path: Path) -> None:
        tasks = {
            "a": {
                "task_id": "a", "status": "success", "exit_code": 0,
                "duration_sec": 5.0,
                "judge_result": {"verdict": "pass", "overall_score": 0.95},
            }
        }
        run_dir = _write_run(tmp_path, tasks=tasks)
        data = build_span_data(run_dir)
        attrs = data["task_spans"][0]["attributes"]
        assert attrs["maestro.task.judge.verdict"] == "pass"
        assert attrs["maestro.task.judge.score"] == pytest.approx(0.95)

    def test_task_with_retries(self, tmp_path: Path) -> None:
        tasks = {
            "a": {
                "task_id": "a", "status": "success", "exit_code": 0,
                "duration_sec": 10.0, "retry_count": 2,
            }
        }
        run_dir = _write_run(tmp_path, tasks=tasks)
        data = build_span_data(run_dir)
        assert data["task_spans"][0]["attributes"]["maestro.task.retry_count"] == 2

    def test_task_events_attached(self, tmp_path: Path) -> None:
        events = [
            {"event": "task_retry", "task_id": "task-a", "attempt": 2},
            {"event": "verify_failure", "task_id": "task-a", "message": "check failed"},
        ]
        run_dir = _write_run(tmp_path, events=events)
        data = build_span_data(run_dir)
        span_events = data["task_spans"][0]["events"]
        assert len(span_events) == 2
        assert span_events[0]["name"] == "task_retry"
        assert span_events[1]["name"] == "verify_failure"

    def test_non_dict_task_results_skipped(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        manifest["task_results"]["bad"] = "not a dict"
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        data = build_span_data(run_dir)
        assert len(data["task_spans"]) == 1  # only task-a

    def test_task_without_engine(self, tmp_path: Path) -> None:
        tasks = {"cmd": {"task_id": "cmd", "status": "success", "exit_code": 0, "duration_sec": 1.0}}
        run_dir = _write_run(tmp_path, tasks=tasks)
        data = build_span_data(run_dir)
        attrs = data["task_spans"][0]["attributes"]
        assert "maestro.task.engine" not in attrs

    def test_auto_routed_model(self, tmp_path: Path) -> None:
        tasks = {
            "a": {
                "task_id": "a", "status": "success", "exit_code": 0,
                "duration_sec": 5.0, "auto_routed_model": "opus",
            }
        }
        run_dir = _write_run(tmp_path, tasks=tasks)
        data = build_span_data(run_dir)
        assert data["task_spans"][0]["attributes"]["maestro.task.model"] == "opus"

    def test_engine_tasks_include_gen_ai_attributes(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        data = build_span_data(run_dir)
        attrs = data["task_spans"][0]["attributes"]
        assert attrs["gen_ai.system"] == "claude"
        assert attrs["gen_ai.model.id"] == "sonnet"
        assert attrs["gen_ai.usage.prompt_tokens"] == 1000
        assert attrs["gen_ai.usage.completion_tokens"] == 500


# ---------------------------------------------------------------------------
# TestFormatOtelJson
# ---------------------------------------------------------------------------

class TestFormatOtelJson:
    def test_valid_json(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        data = build_span_data(run_dir)
        output = format_otel_json(data)
        parsed = json.loads(output)
        assert "root_span" in parsed
        assert "task_spans" in parsed

    def test_indent(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        data = build_span_data(run_dir)
        output = format_otel_json(data)
        assert "  " in output  # indented


# ---------------------------------------------------------------------------
# TestExportRun
# ---------------------------------------------------------------------------

class TestExportRun:
    def test_json_output(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        result = export_run(run_dir, json_output=True)
        assert result["success"] is True
        assert result["span_count"] == 2  # root + 1 task
        assert result["format"] == "json"
        assert "json" in result

    def test_no_manifest(self, tmp_path: Path) -> None:
        result = export_run(tmp_path)
        assert result["success"] is False
        assert "error" in result

    def test_fallback_to_json_without_sdk(self, tmp_path: Path) -> None:
        run_dir = _write_run(tmp_path)
        # Without SDK installed, should fallback to JSON
        result = export_run(run_dir)
        if not _HAS_OTEL:
            assert result["format"] == "json"

    def test_multiple_tasks_span_count(self, tmp_path: Path) -> None:
        tasks = {
            "a": {"task_id": "a", "status": "success", "exit_code": 0, "duration_sec": 1.0},
            "b": {"task_id": "b", "status": "success", "exit_code": 0, "duration_sec": 2.0},
            "c": {"task_id": "c", "status": "success", "exit_code": 0, "duration_sec": 3.0},
        }
        run_dir = _write_run(tmp_path, tasks=tasks)
        result = export_run(run_dir, json_output=True)
        assert result["span_count"] == 4  # root + 3 tasks

    def test_include_content_flows_into_json_export(self, tmp_path: Path) -> None:
        tasks = {
            "a": {
                "task_id": "a",
                "status": "success",
                "exit_code": 0,
                "duration_sec": 1.0,
                "engine": "claude",
                "command": "prompt text",
                "stdout_tail": "output text",
            },
        }
        run_dir = _write_run(tmp_path, tasks=tasks)
        result = export_run(run_dir, json_output=True, include_content=True, mask_content=True)
        assert result["success"] is True
        payload = json.loads(result["json"])
        attrs = payload["task_spans"][0]["attributes"]
        assert attrs["maestro.task.input"] == "[masked input]"
        assert attrs["maestro.task.output"] == "[masked output]"


# ---------------------------------------------------------------------------
# TestLoadEventsOSError
# ---------------------------------------------------------------------------

class TestLoadEventsOSError:
    """Cover the OSError path in _load_events."""

    def test_oserror_reading_events_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When read_text raises OSError, _load_events returns []."""
        events_path = tmp_path / "events.jsonl"
        events_path.write_text('{"event":"ok"}\n', encoding="utf-8")

        original_read_text = events_path.__class__.read_text

        def _raise_on_events(self: Any, *args: Any, **kwargs: Any) -> str:
            if self.name == "events.jsonl":
                raise OSError("disk failure")
            return original_read_text(self, *args, **kwargs)

        monkeypatch.setattr(events_path.__class__, "read_text", _raise_on_events)

        result = _load_events(tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# TestBuildSpanDataEdgeCases
# ---------------------------------------------------------------------------

class TestBuildSpanDataEdgeCases:
    """Additional build_span_data edge cases for coverage."""

    def test_non_dict_task_results_at_top_level(self, tmp_path: Path) -> None:
        """When task_results is not a dict (e.g., a list), it's treated as empty."""
        run_dir = tmp_path / "bad_results"
        run_dir.mkdir()
        manifest = {
            "plan_name": "test",
            "run_id": "r1",
            "success": True,
            "started_at": "2026-03-21T10:00:00+00:00",
            "finished_at": "2026-03-21T10:01:00+00:00",
            "task_results": ["not", "a", "dict"],  # Invalid type
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        data = build_span_data(run_dir)
        assert data is not None
        assert data["task_spans"] == []

    def test_task_with_cached_tokens(self, tmp_path: Path) -> None:
        """Token usage includes cached_tokens when present."""
        tasks = {
            "a": {
                "task_id": "a",
                "status": "success",
                "exit_code": 0,
                "duration_sec": 5.0,
                "token_usage": {
                    "input_tokens": 1000,
                    "output_tokens": 500,
                    "cached_tokens": 200,
                    "total_tokens": 1700,
                },
            }
        }
        run_dir = _write_run(tmp_path, tasks=tasks)
        data = build_span_data(run_dir)
        attrs = data["task_spans"][0]["attributes"]
        assert attrs["maestro.task.tokens.cached_tokens"] == 200
        assert attrs["maestro.task.tokens.input_tokens"] == 1000

    def test_task_without_cost(self, tmp_path: Path) -> None:
        """Task without cost_usd omits the cost attribute."""
        tasks = {
            "a": {
                "task_id": "a",
                "status": "success",
                "exit_code": 0,
                "duration_sec": 1.0,
                # No cost_usd field
            }
        }
        run_dir = _write_run(tmp_path, tasks=tasks)
        data = build_span_data(run_dir)
        attrs = data["task_spans"][0]["attributes"]
        assert "maestro.task.cost_usd" not in attrs

    def test_task_cost_none_omits_attribute(self, tmp_path: Path) -> None:
        """Task with cost_usd=None omits the cost attribute."""
        tasks = {
            "a": {
                "task_id": "a",
                "status": "success",
                "exit_code": 0,
                "duration_sec": 1.0,
                "cost_usd": None,
            }
        }
        run_dir = _write_run(tmp_path, tasks=tasks)
        data = build_span_data(run_dir)
        attrs = data["task_spans"][0]["attributes"]
        assert "maestro.task.cost_usd" not in attrs

    def test_events_filtered_to_relevant_types(self, tmp_path: Path) -> None:
        """Only relevant event types are attached as span events."""
        events = [
            {"event": "task_start", "task_id": "task-a"},  # Not in the filter list
            {"event": "task_retry", "task_id": "task-a", "attempt": 1},  # Included
            {"event": "judge_result", "task_id": "task-a", "verdict": "pass"},  # Included
            {"event": "run_complete", "success": True},  # No task_id
        ]
        run_dir = _write_run(tmp_path, events=events)
        data = build_span_data(run_dir)
        span_events = data["task_spans"][0]["events"]
        assert len(span_events) == 2
        names = [e["name"] for e in span_events]
        assert "task_retry" in names
        assert "judge_result" in names
        assert "task_start" not in names

    def test_missing_timestamps(self, tmp_path: Path) -> None:
        """Tasks and root with missing timestamps use None."""
        tasks = {
            "a": {
                "task_id": "a",
                "status": "success",
                "exit_code": 0,
                "duration_sec": 1.0,
                # No started_at or finished_at
            }
        }
        run_dir = tmp_path / "no_ts"
        run_dir.mkdir()
        manifest = {
            "plan_name": "test",
            "run_id": "r1",
            "success": True,
            # No started_at / finished_at
            "task_results": tasks,
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        data = build_span_data(run_dir)
        assert data is not None
        assert data["root_span"]["start_time"] is None
        assert data["root_span"]["end_time"] is None
        assert data["task_spans"][0]["start_time"] is None

    def test_include_content_captures_input_output_and_structured_fields(
        self, tmp_path: Path,
    ) -> None:
        tasks = {
            "a": {
                "task_id": "a",
                "status": "success",
                "exit_code": 0,
                "duration_sec": 1.0,
                "engine": "claude",
                "command": 'claude --print "implement feature"',
                "stdout_tail": "done",
                "structured_context": {
                    "result_text": "full result",
                    "summary": "short summary",
                },
            }
        }
        run_dir = _write_run(tmp_path, tasks=tasks)
        data = build_span_data(run_dir, include_content=True)
        attrs = data["task_spans"][0]["attributes"]
        assert attrs["maestro.task.input"] == 'claude --print "implement feature"'
        assert attrs["maestro.task.output"] == "done"
        assert attrs["maestro.task.summary"] == "short summary"
        assert attrs["maestro.task.result_text"] == "full result"

    def test_mask_content_redacts_captured_fields(self, tmp_path: Path) -> None:
        tasks = {
            "a": {
                "task_id": "a",
                "status": "success",
                "exit_code": 0,
                "duration_sec": 1.0,
                "engine": "claude",
                "command": "sensitive prompt",
                "stdout_tail": "sensitive output",
                "structured_context": {
                    "result_text": "sensitive result",
                    "summary": "sensitive summary",
                },
            }
        }
        run_dir = _write_run(tmp_path, tasks=tasks)
        data = build_span_data(run_dir, include_content=True, mask_content=True)
        attrs = data["task_spans"][0]["attributes"]
        assert attrs["maestro.task.input"] == "[masked input]"
        assert attrs["maestro.task.output"] == "[masked output]"
        assert attrs["maestro.task.summary"] == "[masked summary]"
        assert attrs["maestro.task.result_text"] == "[masked result]"

    def test_knowledge_poison_alert_event_attached(self, tmp_path: Path) -> None:
        events = [
            {
                "event": "knowledge_poison_alert",
                "task_id": "task-a",
                "source_task_id": "seed-task",
                "action": "quarantine",
            },
        ]
        run_dir = _write_run(tmp_path, events=events)
        data = build_span_data(run_dir)
        span_events = data["task_spans"][0]["events"]
        assert len(span_events) == 1
        assert span_events[0]["name"] == "knowledge_poison_alert"
        assert span_events[0]["attributes"]["source_task_id"] == "seed-task"

    def test_memory_write_event_attached(self, tmp_path: Path) -> None:
        events = [
            {
                "event": "memory_write",
                "task_id": "task-a",
                "knowledge_kind": "failure_pattern",
                "operation": "inserted",
                "outcome": "quarantined",
                "source_id": "run-1:task-a",
            },
        ]
        run_dir = _write_run(tmp_path, events=events)
        data = build_span_data(run_dir)
        span_events = data["task_spans"][0]["events"]
        assert len(span_events) == 1
        assert span_events[0]["name"] == "memory_write"
        assert span_events[0]["attributes"]["outcome"] == "quarantined"
        assert span_events[0]["attributes"]["source_id"] == "run-1:task-a"


# ---------------------------------------------------------------------------
# TestExportToOtlp (mocked SDK)
# ---------------------------------------------------------------------------

class TestExportToOtlpMocked:
    """Test export_to_otlp by mocking the OpenTelemetry SDK."""

    def test_returns_false_when_sdk_not_available(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When _HAS_OTEL is False, export_to_otlp returns False."""
        from maestro_cli import otel

        monkeypatch.setattr(otel, "_HAS_OTEL", False)
        result = otel.export_to_otlp({"root_span": {}, "task_spans": []})
        assert result is False

    def _setup_mocked_sdk(self, monkeypatch: pytest.MonkeyPatch) -> tuple:
        """Set up mocked SDK on the otel module, returning (otel_module, provider_mock)."""
        from maestro_cli import otel

        monkeypatch.setattr(otel, "_HAS_OTEL", True)

        mock_provider_instance = MagicMock()
        mock_tracer = MagicMock()
        mock_provider_instance.get_tracer.return_value = mock_tracer

        mock_root_span = MagicMock()
        mock_task_span = MagicMock()

        call_count = [0]

        def _mock_start_span(name, attributes=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return mock_root_span
            return mock_task_span

        mock_tracer.start_as_current_span = _mock_start_span
        mock_root_span.__enter__ = MagicMock(return_value=mock_root_span)
        mock_root_span.__exit__ = MagicMock(return_value=False)
        mock_task_span.__enter__ = MagicMock(return_value=mock_task_span)
        mock_task_span.__exit__ = MagicMock(return_value=False)

        mock_trace = MagicMock()

        # Set attributes on the module (they may not exist when SDK not installed)
        monkeypatch.setattr(otel, "Resource", MagicMock(), raising=False)
        monkeypatch.setattr(otel, "TracerProvider", MagicMock(return_value=mock_provider_instance), raising=False)
        monkeypatch.setattr(otel, "ConsoleSpanExporter", MagicMock(), raising=False)
        monkeypatch.setattr(otel, "BatchSpanProcessor", MagicMock(), raising=False)
        monkeypatch.setattr(otel, "trace", mock_trace, raising=False)

        return otel, mock_provider_instance

    def test_export_with_mocked_sdk_console(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """With mocked SDK, export_to_otlp with no endpoint returns True."""
        otel, mock_provider = self._setup_mocked_sdk(monkeypatch)

        span_data = {
            "root_span": {
                "name": "maestro:test",
                "attributes": {"maestro.plan.name": "test"},
                "status": "OK",
            },
            "task_spans": [
                {
                    "name": "task:a",
                    "attributes": {"maestro.task.id": "a"},
                    "events": [{"name": "task_retry", "attributes": {"attempt": "1"}}],
                    "status": "ERROR",
                },
            ],
        }

        result = otel.export_to_otlp(span_data, endpoint=None)
        assert result is True
        mock_provider.force_flush.assert_called_once()
        mock_provider.shutdown.assert_called_once()

    def test_export_with_endpoint_both_imports_fail(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When both gRPC and HTTP exporter imports fail, returns False."""
        from maestro_cli import otel

        monkeypatch.setattr(otel, "_HAS_OTEL", True)
        monkeypatch.setattr(otel, "Resource", MagicMock(), raising=False)
        monkeypatch.setattr(otel, "TracerProvider", MagicMock(), raising=False)
        monkeypatch.setattr(otel, "trace", MagicMock(), raising=False)

        original_import = __import__
        grpc_mod = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
        http_mod = "opentelemetry.exporter.otlp.proto.http.trace_exporter"

        def _fail_imports(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in (grpc_mod, http_mod):
                raise ImportError(f"no {name}")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _fail_imports)

        span_data = {
            "root_span": {"name": "test", "attributes": {}, "status": "OK"},
            "task_spans": [],
        }
        result = otel.export_to_otlp(span_data, endpoint="http://localhost:4317")
        assert result is False


# ---------------------------------------------------------------------------
# TestExportRunWithOTLP
# ---------------------------------------------------------------------------

class TestExportRunWithOTLP:
    """Test export_run with OTLP path (mocked)."""

    def test_export_run_otlp_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When _HAS_OTEL is True and json_output is False, OTLP path is used."""
        from maestro_cli import otel
        from unittest.mock import MagicMock

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr(otel, "_HAS_OTEL", True)

        # Mock export_to_otlp to return True
        monkeypatch.setattr(otel, "export_to_otlp", lambda sd, endpoint=None: True)

        result = otel.export_run(run_dir, endpoint="http://localhost:4317")
        assert result["success"] is True
        assert result["format"] == "otlp"
        assert result["endpoint"] == "http://localhost:4317"
        assert result["span_count"] == 2  # root + 1 task

    def test_export_run_otlp_path_no_endpoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OTLP with no endpoint uses 'console' label."""
        from maestro_cli import otel

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr(otel, "_HAS_OTEL", True)
        monkeypatch.setattr(otel, "export_to_otlp", lambda sd, endpoint=None: True)

        result = otel.export_run(run_dir)
        assert result["success"] is True
        assert result["endpoint"] == "console"

    def test_export_run_otlp_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When export_to_otlp returns False, export_run reports failure."""
        from maestro_cli import otel

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr(otel, "_HAS_OTEL", True)
        monkeypatch.setattr(otel, "export_to_otlp", lambda sd, endpoint=None: False)

        result = otel.export_run(run_dir, endpoint="http://bad:4317")
        assert result["success"] is False
        assert result["format"] == "otlp"

    def test_export_run_json_output_forced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """json_output=True forces JSON even when SDK is available."""
        from maestro_cli import otel

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr(otel, "_HAS_OTEL", True)

        result = otel.export_run(run_dir, json_output=True)
        assert result["format"] == "json"
        assert result["success"] is True
        assert "json" in result


# ---------------------------------------------------------------------------
# TestFormatOtelJsonAdditional
# ---------------------------------------------------------------------------

class TestFormatOtelJsonAdditional:
    """Additional format_otel_json tests."""

    def test_handles_datetime_via_default_str(self) -> None:
        """Non-serializable objects are handled by default=str."""
        from maestro_cli.otel import format_otel_json

        data = {
            "root_span": {"start_time": datetime(2026, 3, 21, 10, 0, 0)},
            "task_spans": [],
        }
        output = format_otel_json(data)
        parsed = json.loads(output)
        assert "2026" in parsed["root_span"]["start_time"]

    def test_empty_span_data(self) -> None:
        """Empty span data produces valid JSON."""
        from maestro_cli.otel import format_otel_json

        data = {"root_span": {}, "task_spans": []}
        output = format_otel_json(data)
        parsed = json.loads(output)
        assert parsed["task_spans"] == []


# ---------------------------------------------------------------------------
# TestSafeHelpersAdditional
# ---------------------------------------------------------------------------

class TestSafeHelpersAdditional:
    """Additional safe helper tests."""

    def test_safe_float_string_numeric(self) -> None:
        assert _safe_float("3.14") == pytest.approx(3.14)

    def test_safe_float_list_returns_zero(self) -> None:
        assert _safe_float([1, 2]) == 0.0

    def test_safe_str_bool(self) -> None:
        assert _safe_str(True) == "True"

    def test_safe_str_float(self) -> None:
        assert _safe_str(3.14) == "3.14"
