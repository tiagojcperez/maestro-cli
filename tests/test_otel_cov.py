"""Coverage-focused tests for otel.py uncovered lines.

Targets specific branches not exercised by tests/test_otel.py:
- module-level ImportError guard (_HAS_OTEL = False)
- _clip_content truncation branch
- _capture_content None and empty/whitespace branches
- export_to_otlp gRPC exporter path
- export_to_otlp HTTP exporter fallback path
- export_to_otlp root span ERROR status branch

External boundaries (OpenTelemetry SDK, exporters) are mocked — no network.
"""
from __future__ import annotations

import builtins
import importlib
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from maestro_cli import otel
from maestro_cli.otel import (
    _CONTENT_PREVIEW_LIMIT,
    _capture_content,
    _clip_content,
)


# ---------------------------------------------------------------------------
# _clip_content truncation branch
# ---------------------------------------------------------------------------

class TestClipContent:
    def test_short_text_unchanged(self) -> None:
        assert _clip_content("short", limit=100) == "short"

    def test_text_at_limit_unchanged(self) -> None:
        text = "x" * 10
        assert _clip_content(text, limit=10) == text

    def test_text_over_limit_truncated(self) -> None:
        text = "x" * 50
        result = _clip_content(text, limit=10)
        assert result.startswith("x" * 10)
        assert "truncated" in result
        assert len(result) < len(text) + 30

    def test_default_limit_truncation(self) -> None:
        text = "y" * (_CONTENT_PREVIEW_LIMIT + 100)
        result = _clip_content(text)
        assert "truncated" in result
        assert result.startswith("y" * _CONTENT_PREVIEW_LIMIT)


# ---------------------------------------------------------------------------
# _capture_content None and empty branches
# ---------------------------------------------------------------------------

class TestCaptureContent:
    def test_none_returns_none(self) -> None:
        assert _capture_content(None, label="input", mask_content=False) is None

    def test_empty_string_returns_none(self) -> None:
        assert _capture_content("", label="input", mask_content=False) is None

    def test_whitespace_only_returns_none(self) -> None:
        assert _capture_content("   \n\t  ", label="output", mask_content=False) is None

    def test_masked_value(self) -> None:
        result = _capture_content("secret", label="input", mask_content=True)
        assert result is not None
        assert "masked" in result
        assert "input" in result

    def test_plain_value_passthrough(self) -> None:
        result = _capture_content("hello world", label="output", mask_content=False)
        assert result == "hello world"


# ---------------------------------------------------------------------------
# export_to_otlp endpoint exporter paths + root ERROR status
# ---------------------------------------------------------------------------

def _mock_sdk(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Mock the OpenTelemetry SDK objects used by export_to_otlp.

    Returns the mock TracerProvider *instance* for assertions.
    """
    monkeypatch.setattr(otel, "_HAS_OTEL", True)

    provider_instance = MagicMock()
    tracer = MagicMock()
    provider_instance.get_tracer.return_value = tracer

    root_span = MagicMock()
    task_span = MagicMock()
    root_span.__enter__ = MagicMock(return_value=root_span)
    root_span.__exit__ = MagicMock(return_value=False)
    task_span.__enter__ = MagicMock(return_value=task_span)
    task_span.__exit__ = MagicMock(return_value=False)

    calls = [0]

    def _start_span(name: str, attributes: Any = None) -> MagicMock:
        calls[0] += 1
        return root_span if calls[0] == 1 else task_span

    tracer.start_as_current_span = _start_span

    monkeypatch.setattr(otel, "Resource", MagicMock(), raising=False)
    monkeypatch.setattr(
        otel, "TracerProvider", MagicMock(return_value=provider_instance), raising=False,
    )
    monkeypatch.setattr(otel, "ConsoleSpanExporter", MagicMock(), raising=False)
    monkeypatch.setattr(otel, "BatchSpanProcessor", MagicMock(), raising=False)
    monkeypatch.setattr(otel, "trace", MagicMock(), raising=False)
    return provider_instance


class TestExportToOtlpExporterPaths:
    def test_grpc_exporter_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With an endpoint and a working gRPC exporter import, the gRPC
        exporter is constructed and export succeeds."""
        provider = _mock_sdk(monkeypatch)

        grpc_mod = importlib.import_module(
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
        )
        # Replace the gRPC exporter constructor so no channel/network is set up.
        monkeypatch.setattr(grpc_mod, "OTLPSpanExporter", MagicMock(), raising=True)

        span_data = {
            "root_span": {
                "name": "maestro:test",
                "attributes": {"maestro.plan.name": "test"},
                "status": "OK",
            },
            "task_spans": [],
        }
        result = otel.export_to_otlp(span_data, endpoint="http://localhost:4317")
        assert result is True
        provider.force_flush.assert_called_once()
        provider.shutdown.assert_called_once()
        # gRPC constructor was used with the endpoint
        grpc_mod.OTLPSpanExporter.assert_called_once_with(endpoint="http://localhost:4317")

    def test_http_exporter_fallback_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the gRPC exporter import fails but the HTTP exporter import
        succeeds, the HTTP exporter is used and export succeeds."""
        provider = _mock_sdk(monkeypatch)

        http_mod = importlib.import_module(
            "opentelemetry.exporter.otlp.proto.http.trace_exporter",
        )
        monkeypatch.setattr(http_mod, "OTLPSpanExporter", MagicMock(), raising=True)

        grpc_name = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
        original_import = builtins.__import__

        def _fail_grpc(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == grpc_name:
                raise ImportError("no grpc")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail_grpc)

        span_data = {
            "root_span": {
                "name": "maestro:test",
                "attributes": {"maestro.plan.name": "test"},
                "status": "OK",
            },
            "task_spans": [],
        }
        result = otel.export_to_otlp(span_data, endpoint="http://localhost:4318")
        assert result is True
        provider.force_flush.assert_called_once()
        http_mod.OTLPSpanExporter.assert_called_once_with(endpoint="http://localhost:4318")

    def test_root_span_error_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When the root span status is ERROR, set_status is invoked on the
        root span (console path, no endpoint)."""
        provider = _mock_sdk(monkeypatch)

        # Capture the root span object the tracer hands back.
        tracer = provider.get_tracer.return_value
        produced: list[MagicMock] = []
        original_start = tracer.start_as_current_span

        def _spy_start(name: str, attributes: Any = None) -> MagicMock:
            span = original_start(name, attributes=attributes)
            produced.append(span)
            return span

        tracer.start_as_current_span = _spy_start

        span_data = {
            "root_span": {
                "name": "maestro:test",
                "attributes": {"maestro.plan.name": "test"},
                "status": "ERROR",
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
        # First produced span is the root; its set_status must have been called.
        root_span = produced[0]
        assert root_span.set_status.called
        provider.shutdown.assert_called_once()


# ---------------------------------------------------------------------------
# Module-level ImportError guard (_HAS_OTEL = False)
# ---------------------------------------------------------------------------

class TestImportGuard:
    def test_missing_sdk_sets_has_otel_false(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Reimporting otel with the OpenTelemetry import forced to fail
        drives the module-level ``except ImportError`` branch."""
        original_import = builtins.__import__

        def _fail_otel(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "opentelemetry" or name.startswith("opentelemetry."):
                raise ImportError(f"blocked {name}")
            return original_import(name, *args, **kwargs)

        # Drop any cached opentelemetry submodules + the otel module so the
        # reload genuinely re-runs the top-level import block.
        saved = {
            k: v for k, v in list(sys.modules.items())
            if k == "opentelemetry" or k.startswith("opentelemetry.")
        }
        saved["maestro_cli.otel"] = sys.modules.get("maestro_cli.otel")
        for k in list(sys.modules):
            if k == "opentelemetry" or k.startswith("opentelemetry."):
                del sys.modules[k]

        monkeypatch.setattr(builtins, "__import__", _fail_otel)
        try:
            reloaded = importlib.reload(sys.modules["maestro_cli.otel"])
            assert reloaded._HAS_OTEL is False
            # export_to_otlp short-circuits to False when the SDK is absent.
            assert reloaded.export_to_otlp({"root_span": {}, "task_spans": []}) is False
        finally:
            # Restore real import + a clean, SDK-backed otel module so other
            # tests in the session see the normal state.
            monkeypatch.setattr(builtins, "__import__", original_import)
            for k, v in saved.items():
                if v is not None:
                    sys.modules[k] = v
            importlib.reload(sys.modules["maestro_cli.otel"])
