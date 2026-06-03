from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from maestro_cli.models import FailureRecord, TaskResult
from maestro_cli.runners import (
    _CONCISENESS_HINT,
    _RETRY_FEEDBACK_MAX_CHARS,
    _compress_context_for_retry,
    _build_smart_retry_feedback,
    _classify_failure,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_task_result(tmp_path: Path, **kwargs: object) -> TaskResult:
    now = datetime.now(tz=timezone.utc)
    failure_history = kwargs.get("failure_history", [])
    assert isinstance(failure_history, list)
    return TaskResult(
        task_id=str(kwargs.get("task_id", "t1")),
        status="failed",  # type: ignore[arg-type]
        exit_code=1,
        started_at=now,
        finished_at=now,
        duration_sec=0.1,
        command="echo test",
        log_path=tmp_path / "t1.log",
        result_path=tmp_path / "t1.result.json",
        failure_history=list(failure_history),
    )


def _make_record(attempt: int, category: str, exit_code: int | None = 1) -> FailureRecord:
    return FailureRecord(
        attempt=attempt,
        category=category,  # type: ignore[arg-type]
        exit_code=exit_code,
        message=f"failed at attempt {attempt}",
    )


# ===========================================================================
# TestClassifyFailure
# ===========================================================================


class TestClassifyFailure:
    def test_exit_code_124_returns_timeout(self) -> None:
        assert _classify_failure(124, "", "") == "timeout"

    def test_timeout_keyword_in_output(self) -> None:
        assert _classify_failure(1, "Operation timed out", "") == "timeout"

    def test_deadline_exceeded_returns_timeout(self) -> None:
        assert _classify_failure(1, "deadline exceeded for request", "") == "timeout"

    def test_syntax_error_returns_compilation(self) -> None:
        assert _classify_failure(1, "SyntaxError: unexpected token at line 5", "") == "compilation_error"

    def test_indent_error_returns_compilation(self) -> None:
        assert _classify_failure(1, "IndentationError: unexpected indent", "") == "compilation_error"

    def test_pytest_failure_returns_test_failure(self) -> None:
        # pytest in output should be detected as test_failure
        assert _classify_failure(1, "pytest: 3 failed, 2 passed", "") == "test_failure"

    def test_assertion_error_returns_test_failure(self) -> None:
        assert _classify_failure(1, "AssertionError: expected True but got False", "") == "test_failure"

    def test_failed_keyword_returns_test_failure(self) -> None:
        assert _classify_failure(1, "FAILED test_auth_module", "") == "test_failure"

    def test_permission_denied_returns_permission(self) -> None:
        assert _classify_failure(1, "Permission denied: /etc/passwd", "") == "permission_error"

    def test_eacces_returns_permission(self) -> None:
        assert _classify_failure(1, "EACCES: permission denied, open '/etc/hosts'", "") == "permission_error"

    def test_type_error_returns_validation(self) -> None:
        assert _classify_failure(1, "TypeError: unsupported operand type(s)", "") == "validation_error"

    def test_value_error_returns_validation(self) -> None:
        assert _classify_failure(1, "ValueError: invalid literal for int()", "") == "validation_error"

    def test_traceback_returns_runtime(self) -> None:
        assert _classify_failure(1, "Traceback (most recent call last):", "") == "runtime_error"

    def test_exception_keyword_returns_runtime(self) -> None:
        assert _classify_failure(1, "RuntimeError: maximum recursion depth exceeded", "") == "runtime_error"

    def test_unknown_output_returns_unknown(self) -> None:
        assert _classify_failure(1, "something totally unrecognized xyzzy", "") == "unknown"

    def test_empty_output_returns_unknown(self) -> None:
        assert _classify_failure(1, "", "") == "unknown"

    def test_exit_code_124_overrides_output_pattern(self) -> None:
        # Even if output contains "test failure", exit 124 → timeout wins
        assert _classify_failure(124, "FAILED test_example", "test failed") == "timeout"

    def test_exit_code_124_overrides_syntax_error_output(self) -> None:
        assert _classify_failure(124, "SyntaxError: invalid syntax", "") == "timeout"

    def test_message_field_also_searched(self) -> None:
        # Pattern in message (third param) should also trigger classification
        assert _classify_failure(1, "", "SyntaxError in generated code") == "compilation_error"

    def test_none_exit_code_falls_through_to_patterns(self) -> None:
        assert _classify_failure(None, "ValueError: something wrong", "") == "validation_error"

    def test_combined_output_and_message_searched(self) -> None:
        # Pattern in output OR message should trigger; output takes priority order
        assert _classify_failure(1, "Permission denied", "also has stuff") == "permission_error"

    def test_pattern_search_is_case_insensitive(self) -> None:
        assert _classify_failure(1, "syntax error detected", "") == "compilation_error"
        assert _classify_failure(1, "PERMISSION DENIED", "") == "permission_error"


# ===========================================================================
# TestBuildSmartRetryFeedback
# ===========================================================================


class TestBuildSmartRetryFeedback:
    def test_includes_category(self) -> None:
        record = _make_record(1, "test_failure")
        feedback = _build_smart_retry_feedback(1, 3, "test_failure", 1, "test output", [record])
        assert "test_failure" in feedback

    def test_includes_attempt_counter(self) -> None:
        record = _make_record(1, "compilation_error")
        feedback = _build_smart_retry_feedback(2, 3, "compilation_error", 1, "error output", [record])
        assert "2/3" in feedback

    def test_includes_error_output(self) -> None:
        record = _make_record(1, "runtime_error")
        feedback = _build_smart_retry_feedback(1, 2, "runtime_error", 1, "my error output here", [record])
        assert "my error output here" in feedback

    def test_includes_exit_code(self) -> None:
        record = _make_record(1, "permission_error", exit_code=13)
        feedback = _build_smart_retry_feedback(1, 2, "permission_error", 13, "denied", [record])
        assert "13" in feedback

    def test_truncates_long_output(self) -> None:
        record = _make_record(1, "unknown")
        long_output = "x" * (_RETRY_FEEDBACK_MAX_CHARS * 3)
        feedback = _build_smart_retry_feedback(1, 2, "unknown", 1, long_output, [record])
        # The output section should be truncated to at most _RETRY_FEEDBACK_MAX_CHARS
        # (the overall feedback string will be slightly longer due to template overhead)
        assert len(feedback) < len(long_output)

    def test_no_history_section_for_first_failure(self) -> None:
        # Single record in history → no "Previous failures:" section
        record = _make_record(1, "test_failure")
        feedback = _build_smart_retry_feedback(1, 3, "test_failure", 1, "error", [record])
        assert "Previous failures" not in feedback

    def test_history_section_for_multiple_failures(self) -> None:
        history = [
            _make_record(1, "compilation_error"),
            _make_record(2, "compilation_error"),
        ]
        feedback = _build_smart_retry_feedback(2, 3, "compilation_error", 1, "error", history)
        assert "Previous failures" in feedback
        assert "Attempt 1: compilation_error" in feedback

    def test_escalation_hint_on_repeated_category(self) -> None:
        history = [
            _make_record(1, "test_failure"),
            _make_record(2, "test_failure"),
        ]
        feedback = _build_smart_retry_feedback(2, 3, "test_failure", 1, "error", history)
        assert "WARNING" in feedback
        assert "test_failure" in feedback

    def test_no_escalation_hint_on_different_categories(self) -> None:
        history = [
            _make_record(1, "compilation_error"),
            _make_record(2, "test_failure"),
        ]
        # test_failure appears only once → no escalation
        feedback = _build_smart_retry_feedback(2, 3, "test_failure", 1, "error", history)
        assert "WARNING" not in feedback

    def test_template_has_retry_feedback_marker(self) -> None:
        record = _make_record(1, "unknown")
        feedback = _build_smart_retry_feedback(1, 2, "unknown", 1, "some output", [record])
        assert "RETRY FEEDBACK" in feedback

    def test_context_exceeded_includes_conciseness_hint(self) -> None:
        record = _make_record(1, "context_exceeded")
        feedback = _build_smart_retry_feedback(
            1, 2, "context_exceeded", 1, "context window exceeded", [record],
        )
        assert "IMPORTANT: CONTEXT BUDGET" in feedback
        assert _CONCISENESS_HINT.strip() in feedback


class TestContextCompression:
    def test_compression_reduces_text_length(self) -> None:
        text = "x" * 10_000
        compressed = _compress_context_for_retry(text, 1)
        assert len(compressed) < len(text)

    def test_higher_compression_level_reduces_more(self) -> None:
        text = "x" * 10_000
        first = _compress_context_for_retry(text, 1)
        second = _compress_context_for_retry(text, 2)
        assert len(second) < len(first)


# ===========================================================================
# TestFailureRecord
# ===========================================================================


class TestFailureRecord:
    def test_to_dict_all_fields(self) -> None:
        record = FailureRecord(
            attempt=2,
            category="timeout",
            exit_code=124,
            message="command timed out",
        )
        d = record.to_dict()
        assert d["attempt"] == 2
        assert d["category"] == "timeout"
        assert d["exit_code"] == 124
        assert d["message"] == "command timed out"

    def test_to_dict_with_none_exit_code(self) -> None:
        record = FailureRecord(attempt=1, category="unknown", exit_code=None, message="")
        d = record.to_dict()
        assert d["exit_code"] is None

    def test_failure_history_in_task_result(self, tmp_path: Path) -> None:
        records = [
            FailureRecord(attempt=1, category="compilation_error", exit_code=1, message="fail 1"),
            FailureRecord(attempt=2, category="test_failure", exit_code=1, message="fail 2"),
        ]
        result = _make_task_result(tmp_path, failure_history=records)
        assert len(result.failure_history) == 2
        assert result.failure_history[0].category == "compilation_error"
        assert result.failure_history[1].category == "test_failure"

    def test_task_result_to_dict_includes_failure_history(self, tmp_path: Path) -> None:
        records = [
            FailureRecord(attempt=1, category="runtime_error", exit_code=1, message="error"),
        ]
        result = _make_task_result(tmp_path, failure_history=records)
        d = result.to_dict()
        assert "failure_history" in d
        assert len(d["failure_history"]) == 1
        assert d["failure_history"][0]["category"] == "runtime_error"
        assert d["failure_history"][0]["attempt"] == 1

    def test_task_result_failure_history_empty_by_default(self, tmp_path: Path) -> None:
        result = _make_task_result(tmp_path)
        assert result.failure_history == []
        d = result.to_dict()
        assert d["failure_history"] == []

    def test_failure_record_attempt_ordering(self) -> None:
        records = [
            _make_record(1, "compilation_error"),
            _make_record(2, "test_failure"),
            _make_record(3, "unknown"),
        ]
        assert [r.attempt for r in records] == [1, 2, 3]
        assert [r.category for r in records] == ["compilation_error", "test_failure", "unknown"]
