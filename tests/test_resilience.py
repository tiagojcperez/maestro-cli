"""Tests for v0.8.0 resilience features: new failure categories, handoff reports,
conciseness hints, context compression for retry, and compression metrics."""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from maestro_cli.models import (
    FailureCategory,
    FailureRecord,
    HandoffReport,
    TaskResult,
    TaskSpec,
)
from maestro_cli.runners import (
    _CONCISENESS_HINT,
    _build_smart_retry_feedback,
    _classify_failure,
    _compress_context_for_retry,
    _compress_upstream_context_for_retry,
    _generate_handoff_report,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(stdout_tail: str = "", **kwargs: object) -> TaskResult:
    now = datetime.now(tz=timezone.utc)
    return TaskResult(
        task_id=str(kwargs.get("task_id", "t1")),
        status="success",
        started_at=now,
        finished_at=now,
        duration_sec=0.1,
        command="echo ok",
        stdout_tail=stdout_tail,
    )


# ===========================================================================
# New failure categories: context_exceeded, rate_limited
# ===========================================================================


class TestContextExceededCategory:
    def test_context_window_exceeded(self) -> None:
        assert _classify_failure(None, "context window exceeded max length", "") == "context_exceeded"

    def test_maximum_context_length(self) -> None:
        assert _classify_failure(None, "maximum context length reached", "") == "context_exceeded"

    def test_token_limit_exceeded(self) -> None:
        assert _classify_failure(None, "token limit exceeded", "") == "context_exceeded"

    def test_input_too_long(self) -> None:
        assert _classify_failure(None, "input too long for model", "") == "context_exceeded"

    def test_context_length_exceeded(self) -> None:
        assert _classify_failure(None, "context length exceeded", "") == "context_exceeded"

    def test_token_limit_in_message(self) -> None:
        assert _classify_failure(None, "", "token limit exceeded") == "context_exceeded"

    def test_case_insensitive(self) -> None:
        assert _classify_failure(None, "CONTEXT WINDOW EXCEEDED", "") == "context_exceeded"

    def test_exit_124_still_timeout(self) -> None:
        # Even with context error text, exit 124 → timeout takes priority
        assert _classify_failure(124, "context window exceeded", "") == "timeout"


class TestRateLimitedCategory:
    def test_rate_limit_exceeded(self) -> None:
        assert _classify_failure(None, "rate limit exceeded", "") == "rate_limited"

    def test_too_many_requests(self) -> None:
        assert _classify_failure(None, "too many requests", "") == "rate_limited"

    def test_http_429(self) -> None:
        assert _classify_failure(None, "HTTP 429 error", "") == "rate_limited"

    def test_quota_exceeded(self) -> None:
        assert _classify_failure(None, "quota exceeded", "") == "rate_limited"

    def test_throttled(self) -> None:
        assert _classify_failure(None, "request throttled by API", "") == "rate_limited"

    def test_retry_after_header(self) -> None:
        assert _classify_failure(None, "retry after 30 seconds", "") == "rate_limited"

    def test_overloaded(self) -> None:
        assert _classify_failure(None, "service overloaded please wait", "") == "rate_limited"

    def test_case_insensitive(self) -> None:
        assert _classify_failure(None, "RATE LIMIT EXCEEDED", "") == "rate_limited"

    def test_existing_categories_preserved(self) -> None:
        """New patterns should not break existing category detection."""
        assert _classify_failure(124, "", "") == "timeout"
        assert _classify_failure(None, "FAILED test_auth.py", "") == "test_failure"
        assert _classify_failure(None, "Permission denied", "") == "permission_error"
        assert _classify_failure(None, "SyntaxError: invalid", "") == "compilation_error"
        assert _classify_failure(None, "ValueError: bad", "") == "validation_error"
        assert _classify_failure(None, "Traceback:", "") == "runtime_error"


# ===========================================================================
# Conciseness hint
# ===========================================================================


class TestConcisenessHint:
    def test_hint_is_defined(self) -> None:
        assert _CONCISENESS_HINT
        assert len(_CONCISENESS_HINT) > 20

    def test_hint_contains_concise_keyword(self) -> None:
        lower = _CONCISENESS_HINT.lower()
        assert "concis" in lower or "brief" in lower or "budget" in lower

    def test_injected_for_context_exceeded(self) -> None:
        fb = _build_smart_retry_feedback(
            1, 2, "context_exceeded", 1, "error", [],
        )
        assert "CONTEXT BUDGET" in fb or "CONCIS" in fb.upper()

    def test_not_injected_for_other_categories(self) -> None:
        fb = _build_smart_retry_feedback(
            1, 2, "test_failure", 1, "error", [],
        )
        assert "CONTEXT BUDGET" not in fb


# ===========================================================================
# Handoff report
# ===========================================================================


class TestHandoffReport:
    def test_report_dataclass_fields(self) -> None:
        report = HandoffReport(
            failure_category="context_exceeded",
            partial_output="some output",
            summary="task failed",
        )
        assert report.failure_category == "context_exceeded"
        assert report.partial_output == "some output"
        assert report.summary == "task failed"

    def test_report_defaults(self) -> None:
        report = HandoffReport()
        assert report.failure_category == "runtime_error"
        assert report.partial_output == ""
        assert report.summary == ""

    def test_generate_returns_handoff_report(self) -> None:
        task = TaskSpec(id="test-task", description="test", engine="claude", max_retries=2)
        history = [FailureRecord(attempt=1, category="context_exceeded", exit_code=1, message="too long")]
        report = _generate_handoff_report(
            task=task, max_attempts=3, message="context exceeded",
            output="partial", failure_history=history,
        )
        assert isinstance(report, HandoffReport)
        assert "test-task" in report.summary
        assert "context_exceeded" in report.summary

    def test_generate_empty_history(self) -> None:
        task = TaskSpec(id="t", description="t", engine="claude")
        report = _generate_handoff_report(
            task=task, max_attempts=1, message="failed",
            output="", failure_history=[],
        )
        assert report is not None
        assert report.failure_category == "unknown"

    def test_generate_includes_partial_output(self) -> None:
        task = TaskSpec(id="t", description="t", engine="claude")
        report = _generate_handoff_report(
            task=task, max_attempts=2, message="error",
            output="this is the partial output from the engine",
            failure_history=[FailureRecord(attempt=1, category="timeout", exit_code=124, message="timed out")],
        )
        assert "partial output" in report.partial_output

    def test_generate_uses_message_when_output_empty(self) -> None:
        task = TaskSpec(id="t", description="t", engine="claude")
        report = _generate_handoff_report(
            task=task, max_attempts=1, message="the error message here",
            output="", failure_history=[],
        )
        assert "error message" in report.partial_output

    def test_generate_with_compression_count(self) -> None:
        task = TaskSpec(id="t", description="t", engine="claude")
        report = _generate_handoff_report(
            task=task, max_attempts=2, message="error",
            output="output", failure_history=[],
            context_compression_count=3,
        )
        assert "compression" in report.summary.lower()

    def test_task_result_has_handoff_field(self) -> None:
        r = _make_result()
        assert hasattr(r, "handoff_report")
        assert r.handoff_report is None


# ===========================================================================
# Context compression for retry
# ===========================================================================


class TestCompressUpstreamContext:
    def test_empty_upstream_returns_as_is(self) -> None:
        result = _compress_upstream_context_for_retry({}, 1)
        assert result == {}

    def test_compresses_stdout_tail(self) -> None:
        upstream = {"t1": _make_result("x" * 5000)}
        compressed = _compress_upstream_context_for_retry(upstream, 1)
        assert len(compressed["t1"].stdout_tail) < 5000

    def test_higher_level_compresses_more(self) -> None:
        upstream = {"t1": _make_result("x" * 10000)}
        c1 = _compress_upstream_context_for_retry(upstream, 1)
        c2 = _compress_upstream_context_for_retry(upstream, 2)
        assert len(c2["t1"].stdout_tail) < len(c1["t1"].stdout_tail)


# ===========================================================================
# Compression metrics in TaskResult
# ===========================================================================


class TestCompressionMetrics:
    def test_default_values(self) -> None:
        r = _make_result()
        assert r.context_raw_tokens == 0
        assert r.context_final_tokens == 0
        assert r.context_compression_ratio == 0.0

    def test_can_set_values(self) -> None:
        r = _make_result()
        r.context_raw_tokens = 1000
        r.context_final_tokens = 600
        r.context_compression_ratio = 0.4
        assert r.context_raw_tokens == 1000
        assert r.context_final_tokens == 600
        assert r.context_compression_ratio == 0.4

    def test_to_dict_includes_compression(self) -> None:
        r = _make_result()
        r.context_raw_tokens = 500
        r.context_final_tokens = 300
        r.context_compression_ratio = 0.4
        d = r.to_dict()
        assert "context_raw_tokens" in d
        assert "context_final_tokens" in d
        assert "context_compression_ratio" in d
        assert d["context_raw_tokens"] == 500
        assert d["context_compression_ratio"] == 0.4


# ===========================================================================
# Timestamp fix
# ===========================================================================


class TestLocalTimestamp:
    def test_returns_valid_time_format(self) -> None:
        from maestro_cli.scheduler import _local_timestamp
        ts = _local_timestamp()
        parts = ts.split(":")
        assert len(parts) == 3
        hour, minute, second = int(parts[0]), int(parts[1]), int(parts[2])
        assert 0 <= hour <= 23
        assert 0 <= minute <= 59
        assert 0 <= second <= 59

    def test_matches_system_time(self) -> None:
        from datetime import datetime as dt
        from maestro_cli.scheduler import _local_timestamp
        ts = _local_timestamp()
        now = dt.now().strftime("%H:%M")
        # At least hour:minute should match
        assert ts[:5] == now
