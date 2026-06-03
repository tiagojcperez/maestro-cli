"""OTLP (OpenTelemetry) exporter for Maestro CLI runs.

Converts Maestro run data (manifest + events) into OpenTelemetry spans
for export to Jaeger, Grafana Tempo, or any OTLP-compatible backend.

Install: ``pip install maestro-cli[otel]``
Usage:   ``maestro export-otel <run-path> [--endpoint URL]``

Each plan run becomes a root span.  Each task becomes a child span with
attributes for engine, model, cost, tokens, status, and exit code.
Events (retries, judge results, budget warnings) are attached as span
events.

When the SDK is not installed, ``export_run()`` returns structured dicts
that can be serialised to JSON for offline ingestion.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# SDK availability check
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SERVICE_NAME = "maestro-cli"
_SPAN_KIND_INTERNAL = "INTERNAL"
_CONTENT_PREVIEW_LIMIT = 4000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_iso(ts_str: str | None) -> datetime | None:
    """Parse ISO timestamp string to datetime."""
    if not ts_str or not isinstance(ts_str, str):
        return None
    try:
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (ValueError, TypeError):
        return None


def _load_manifest(run_path: Path) -> dict[str, Any] | None:
    """Load run_manifest.json from a run directory."""
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        result: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
        return result
    except (json.JSONDecodeError, OSError):
        return None


def _load_events(run_path: Path) -> list[dict[str, Any]]:
    """Load events.jsonl from a run directory."""
    events_path = run_path / "events.jsonl"
    if not events_path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    except OSError:
        pass
    return events


def _safe_str(val: Any) -> str:
    """Convert value to string, handling None."""
    if val is None:
        return ""
    return str(val)


def _safe_float(val: Any) -> float:
    """Convert value to float, handling None and non-numeric."""
    if val is None:
        return 0.0
    try:
        return float(val)
    except (ValueError, TypeError):
        return 0.0


def _clip_content(text: str, limit: int = _CONTENT_PREVIEW_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def _capture_content(text: Any, *, label: str, mask_content: bool) -> str | None:
    if text is None:
        return None
    rendered = str(text)
    if not rendered.strip():
        return None
    if mask_content:
        return f"[masked {label}]"
    return _clip_content(rendered)


def _gen_ai_system(engine: str) -> str:
    return engine.strip().lower()


# ---------------------------------------------------------------------------
# Build span data (SDK-independent)
# ---------------------------------------------------------------------------

def build_span_data(
    run_path: Path,
    *,
    include_content: bool = False,
    mask_content: bool = False,
) -> dict[str, Any] | None:
    """Build structured span data from a completed run.

    Returns a dict with ``root_span`` and ``task_spans`` keys,
    ready for OTLP export or JSON serialisation.

    Returns ``None`` if the manifest cannot be loaded.
    """
    manifest = _load_manifest(run_path)
    if manifest is None:
        return None

    events = _load_events(run_path)

    plan_name = manifest.get("plan_name", "unknown")
    run_id = manifest.get("run_id", run_path.name)
    success = manifest.get("success", False)
    profile = manifest.get("execution_profile", "plan")
    started_at = _parse_iso(manifest.get("started_at"))
    finished_at = _parse_iso(manifest.get("finished_at"))

    # Task events indexed by task_id
    task_events: dict[str, list[dict[str, Any]]] = {}
    for evt in events:
        tid = evt.get("task_id")
        if tid:
            task_events.setdefault(tid, []).append(evt)

    # Build task spans
    task_spans: list[dict[str, Any]] = []
    task_results = manifest.get("task_results", {})
    if not isinstance(task_results, dict):
        task_results = {}

    for task_id, result in task_results.items():
        if not isinstance(result, dict):
            continue

        task_started = _parse_iso(result.get("started_at"))
        task_finished = _parse_iso(result.get("finished_at"))

        # Attributes
        attrs: dict[str, Any] = {
            "maestro.task.id": task_id,
            "maestro.task.status": _safe_str(result.get("status")),
            "maestro.task.exit_code": result.get("exit_code"),
            "maestro.task.duration_sec": _safe_float(result.get("duration_sec")),
        }

        # Engine/model (may not be present for command tasks)
        engine = result.get("engine")
        if engine:
            attrs["maestro.task.engine"] = str(engine)
            attrs["gen_ai.system"] = _gen_ai_system(str(engine))
        model = result.get("model") or result.get("auto_routed_model")
        if model:
            attrs["maestro.task.model"] = str(model)
            attrs["gen_ai.model.id"] = str(model)

        # Cost/tokens
        cost = result.get("cost_usd")
        if cost is not None:
            attrs["maestro.task.cost_usd"] = _safe_float(cost)

        token_usage = result.get("token_usage")
        if isinstance(token_usage, dict):
            for key in ("input_tokens", "output_tokens", "cached_tokens", "total_tokens"):
                val = token_usage.get(key)
                if val is not None:
                    attrs[f"maestro.task.tokens.{key}"] = int(val)
            input_tokens = token_usage.get("input_tokens")
            if input_tokens is not None:
                attrs["gen_ai.usage.prompt_tokens"] = int(input_tokens)
            output_tokens = token_usage.get("output_tokens")
            if output_tokens is not None:
                attrs["gen_ai.usage.completion_tokens"] = int(output_tokens)

        if include_content:
            captured_command = _capture_content(
                result.get("command"),
                label="input",
                mask_content=mask_content,
            )
            if captured_command is not None:
                attrs["maestro.task.input"] = captured_command

            captured_output = _capture_content(
                result.get("stdout_tail"),
                label="output",
                mask_content=mask_content,
            )
            if captured_output is not None:
                attrs["maestro.task.output"] = captured_output

            structured_context = result.get("structured_context")
            if isinstance(structured_context, dict):
                captured_summary = _capture_content(
                    structured_context.get("summary"),
                    label="summary",
                    mask_content=mask_content,
                )
                if captured_summary is not None:
                    attrs["maestro.task.summary"] = captured_summary
                captured_result_text = _capture_content(
                    structured_context.get("result_text"),
                    label="result",
                    mask_content=mask_content,
                )
                if captured_result_text is not None:
                    attrs["maestro.task.result_text"] = captured_result_text

        # Retry info
        retry_count = result.get("retry_count", 0)
        if retry_count:
            attrs["maestro.task.retry_count"] = int(retry_count)

        # Judge result
        judge = result.get("judge_result")
        if isinstance(judge, dict):
            attrs["maestro.task.judge.verdict"] = _safe_str(judge.get("verdict"))
            score = judge.get("overall_score")
            if score is not None:
                attrs["maestro.task.judge.score"] = _safe_float(score)

        # Span events from task-level events.jsonl entries
        span_events: list[dict[str, Any]] = []
        for evt in task_events.get(task_id, []):
            evt_type = evt.get("event") or evt.get("event_type", "")
            if evt_type in (
                "task_retry", "task_escalation", "engine_fallback",
                "verify_failure", "judge_result", "task_output",
                "context_compress_requested", "taint_detected",
                "knowledge_poison_alert", "task_progress",
                "task_metric", "task_artifact", "timeout_extended",
                "task_tool_call", "worktree_merge", "worktree_verification",
                "memory_write",
            ):
                span_events.append({
                    "name": evt_type,
                    "timestamp": evt.get("timestamp"),
                    "attributes": {
                        k: str(v) for k, v in evt.items()
                        if k not in ("event", "event_type", "timestamp", "plan_name")
                    },
                })

        task_spans.append({
            "name": f"task:{task_id}",
            "start_time": task_started.isoformat() if task_started else None,
            "end_time": task_finished.isoformat() if task_finished else None,
            "attributes": {k: v for k, v in attrs.items() if v is not None},
            "events": span_events,
            "status": "OK" if result.get("status") in ("success", "dry_run") else "ERROR",
        })

    # Root span
    root_span = {
        "name": f"maestro:{plan_name}",
        "start_time": started_at.isoformat() if started_at else None,
        "end_time": finished_at.isoformat() if finished_at else None,
        "attributes": {
            "maestro.plan.name": plan_name,
            "maestro.run.id": run_id,
            "maestro.run.success": success,
            "maestro.run.profile": profile,
            "maestro.run.task_count": len(task_results),
            "maestro.run.total_cost_usd": _safe_float(manifest.get("total_cost_usd")),
            "maestro.run.total_tokens": manifest.get("total_tokens"),
            "service.name": _SERVICE_NAME,
        },
        "status": "OK" if success else "ERROR",
    }

    return {
        "root_span": root_span,
        "task_spans": task_spans,
        "run_path": str(run_path),
    }


# ---------------------------------------------------------------------------
# Export to OTLP (requires SDK)
# ---------------------------------------------------------------------------

def export_to_otlp(
    span_data: dict[str, Any],
    endpoint: str | None = None,
) -> bool:
    """Export span data to an OTLP endpoint.

    Falls back to console export when no endpoint is provided.
    Returns ``True`` on success, ``False`` on failure.
    """
    if not _HAS_OTEL:
        return False

    resource = Resource.create({"service.name": _SERVICE_NAME})
    provider = TracerProvider(resource=resource)

    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
                OTLPSpanExporter,
            )
            exporter: SpanExporter = OTLPSpanExporter(endpoint=endpoint)
        except ImportError:
            try:
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                    OTLPSpanExporter as HTTPExporter,
                )
                exporter = HTTPExporter(endpoint=endpoint)
            except ImportError:
                return False
    else:
        exporter = ConsoleSpanExporter()

    provider.add_span_processor(BatchSpanProcessor(exporter))
    tracer = provider.get_tracer("maestro-cli")

    root = span_data.get("root_span", {})
    task_spans = span_data.get("task_spans", [])

    # Create root span
    with tracer.start_as_current_span(
        root.get("name", "maestro:unknown"),
        attributes={k: v for k, v in root.get("attributes", {}).items() if v is not None},
    ) as root_span:
        # Create task child spans
        for ts in task_spans:
            with tracer.start_as_current_span(
                ts.get("name", "task:unknown"),
                attributes={k: v for k, v in ts.get("attributes", {}).items() if v is not None},
            ) as task_span:
                for evt in ts.get("events", []):
                    task_span.add_event(
                        evt.get("name", "event"),
                        attributes=evt.get("attributes", {}),
                    )
                if ts.get("status") == "ERROR":
                    task_span.set_status(trace.StatusCode.ERROR)

        if root.get("status") == "ERROR":
            root_span.set_status(trace.StatusCode.ERROR)

    provider.force_flush()
    provider.shutdown()
    return True


# ---------------------------------------------------------------------------
# Format as JSON (SDK-independent)
# ---------------------------------------------------------------------------

def format_otel_json(span_data: dict[str, Any]) -> str:
    """Format span data as JSON for file export or debugging."""
    return json.dumps(span_data, indent=2, default=str, ensure_ascii=False)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def export_run(
    run_path: Path,
    endpoint: str | None = None,
    json_output: bool = False,
    include_content: bool = False,
    mask_content: bool = False,
) -> dict[str, Any]:
    """Export a completed run as OTLP spans.

    Args:
        run_path: Path to the run directory
        endpoint: OTLP endpoint URL (None = console/JSON)
        json_output: Return JSON instead of sending to endpoint

    Returns:
        Dict with ``success``, ``span_count``, and optionally ``json``.
    """
    span_data = build_span_data(
        run_path,
        include_content=include_content,
        mask_content=mask_content,
    )
    if span_data is None:
        return {"success": False, "error": f"Cannot load manifest from {run_path}"}

    task_count = len(span_data.get("task_spans", []))

    if json_output or not _HAS_OTEL:
        return {
            "success": True,
            "span_count": task_count + 1,  # tasks + root
            "format": "json",
            "json": format_otel_json(span_data),
        }

    ok = export_to_otlp(span_data, endpoint=endpoint)
    return {
        "success": ok,
        "span_count": task_count + 1,
        "format": "otlp",
        "endpoint": endpoint or "console",
    }
