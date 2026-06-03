from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro_cli.cli import main
from maestro_cli.report import (
    _format_cost,
    _format_tokens,
    build_report_html,
    generate_report,
)


def _sample_manifest() -> dict[str, object]:
    return {
        "plan_name": "report-sample",
        "run_id": "rpt-123",
        "started_at": "2026-03-01T10:00:00+00:00",
        "finished_at": "2026-03-01T10:03:00+00:00",
        "success": False,
        "task_results": {
            "task-a": {
                "status": "success",
                "duration_sec": 60.0,
                "cost_usd": 0.2,
                "token_usage": {"total_tokens": 2000},
                "command": "codex -m gpt-5.3-codex",
                "stdout_tail": "all good",
                "started_at": "2026-03-01T10:00:00+00:00",
                "finished_at": "2026-03-01T10:01:00+00:00",
            },
            "task-b": {
                "status": "failed",
                "duration_sec": 25.0,
                "cost_usd": 0.1,
                "token_usage": {"total_tokens": 500},
                "command": "echo fail",
                "stdout_tail": "traceback",
                "started_at": "2026-03-01T10:01:10+00:00",
                "finished_at": "2026-03-01T10:01:35+00:00",
            },
        },
    }


class TestBuildReportHtml:
    def test_contains_required_sections(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run-123"
        run_path.mkdir()
        html = build_report_html(_sample_manifest(), run_path)

        assert "Task Table" in html
        assert "Execution Timeline" in html
        assert "Cost Breakdown" in html
        assert "Token Breakdown" in html
        assert "Task Details" in html
        assert "stdout_tail" in html
        assert "report-sample" in html
        assert "rpt-123" in html
        assert '<script id="report-data" type="application/json">' in html

    def test_valid_html_structure(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run-abc"
        run_path.mkdir()
        html = build_report_html(_sample_manifest(), run_path)

        assert "<html" in html
        assert "<head>" in html
        assert "<body>" in html
        assert "</html>" in html

    def test_report_data_script_is_valid_json(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run-abc"
        run_path.mkdir()
        html = build_report_html(_sample_manifest(), run_path)

        # Extract the JSON blob from the <script> tag
        start = html.index('<script id="report-data" type="application/json">') + len(
            '<script id="report-data" type="application/json">'
        )
        end = html.index("</script>", start)
        json_text = html[start:end]
        data = json.loads(json_text)

        assert data["plan_name"] == "report-sample"
        assert data["run_id"] == "rpt-123"
        assert isinstance(data["tasks"], list)
        assert len(data["tasks"]) == 2

    def test_null_cost_and_tokens_handled_gracefully(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run-null"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "plan_name": "no-cost-plan",
            "run_id": "nc-001",
            "success": True,
            "task_results": {
                "t1": {
                    "status": "success",
                    "command": "echo hello",
                    # No cost_usd, no token_usage
                },
            },
        }
        html = build_report_html(manifest, run_path)

        assert "no-cost-plan" in html
        assert "—" in html  # Null cost/token formatted as em-dash

    def test_success_status_badge(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run-ok"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "plan_name": "ok-plan",
            "run_id": "ok-001",
            "success": True,
            "task_results": {},
        }
        html = build_report_html(manifest, run_path)
        assert "SUCCESS" in html

    def test_failed_status_badge(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run-fail"
        run_path.mkdir()
        html = build_report_html(_sample_manifest(), run_path)
        assert "FAILED" in html


class TestGenerateReport:
    def test_writes_html_to_default_path(self, tmp_path: Path) -> None:
        run_path = tmp_path / ".maestro-runs" / "rpt-123_report-sample"
        run_path.mkdir(parents=True)
        (run_path / "run_manifest.json").write_text(
            json.dumps(_sample_manifest(), indent=2),
            encoding="utf-8",
        )

        output_path = generate_report(run_path)
        assert output_path == run_path / "report.html"
        assert output_path.exists()

        html = output_path.read_text(encoding="utf-8")
        assert "report-sample" in html
        assert "FAILED" in html
        assert "task-a" in html
        assert "task-b" in html

    def test_custom_output_path(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run-dir"
        run_path.mkdir()
        (run_path / "run_manifest.json").write_text(
            json.dumps(_sample_manifest(), indent=2), encoding="utf-8"
        )
        custom_out = tmp_path / "custom" / "output.html"

        result_path = generate_report(run_path, output_path=custom_out)

        assert result_path == custom_out
        assert custom_out.exists()
        html = custom_out.read_text(encoding="utf-8")
        assert "report-sample" in html

    def test_raises_on_missing_manifest(self, tmp_path: Path) -> None:
        run_path = tmp_path / "empty-run"
        run_path.mkdir()

        with pytest.raises(FileNotFoundError, match="run_manifest.json"):
            generate_report(run_path)

    def test_raises_on_invalid_json(self, tmp_path: Path) -> None:
        run_path = tmp_path / "bad-run"
        run_path.mkdir()
        (run_path / "run_manifest.json").write_text("not json {{", encoding="utf-8")

        with pytest.raises(ValueError, match="invalid JSON"):
            generate_report(run_path)


class TestCliReport:
    def test_report_cli_writes_file(self, tmp_path: Path, capsys: object) -> None:
        run_path = tmp_path / "run-dir"
        run_path.mkdir()
        (run_path / "run_manifest.json").write_text(
            json.dumps(_sample_manifest(), indent=2), encoding="utf-8"
        )

        exit_code = main(["report", str(run_path)])
        assert exit_code == 0
        assert (run_path / "report.html").exists()

    def test_report_cli_custom_output(self, tmp_path: Path, capsys: object) -> None:
        run_path = tmp_path / "run-dir"
        run_path.mkdir()
        (run_path / "run_manifest.json").write_text(
            json.dumps(_sample_manifest(), indent=2), encoding="utf-8"
        )
        out_file = tmp_path / "out.html"

        exit_code = main(["report", str(run_path), "-o", str(out_file)])
        assert exit_code == 0
        assert out_file.exists()

    def test_report_cli_nonexistent_dir(self, tmp_path: Path, capsys: object) -> None:
        missing = tmp_path / "nonexistent"
        exit_code = main(["report", str(missing)])
        assert exit_code == 1
        captured = capsys.readouterr()  # type: ignore[attr-defined]
        assert "error" in captured.out.lower()


class TestFormatHelpers:
    def test_format_cost_none(self) -> None:
        assert _format_cost(None) == "—"

    def test_format_cost_value(self) -> None:
        assert _format_cost(0.5) == "$0.50"

    def test_format_tokens_none(self) -> None:
        assert _format_tokens(None) == "—"

    def test_format_tokens_value(self) -> None:
        assert _format_tokens(1000) == "1,000"


# ---------------------------------------------------------------------------
# Internal helper imports for thorough coverage
# ---------------------------------------------------------------------------

from maestro_cli.report import (
    _coerce_float,
    _coerce_int,
    _clean_cli_token,
    _duration_from_timestamps,
    _format_duration,
    _infer_engine_label,
    _json_for_script,
    _normalize_task,
    _parse_iso,
    _prepare_report_data,
    _status_counts,
    _status_label,
    _total_cost,
    _total_tokens,
    _run_duration,
)


# ---------------------------------------------------------------------------
# _coerce_float / _coerce_int
# ---------------------------------------------------------------------------


class TestCoerceFloat:
    def test_none_returns_none(self) -> None:
        assert _coerce_float(None) is None

    def test_int_returns_float(self) -> None:
        assert _coerce_float(5) == 5.0

    def test_string_number_returns_float(self) -> None:
        assert _coerce_float("3.14") == pytest.approx(3.14)

    def test_non_numeric_returns_none(self) -> None:
        assert _coerce_float("abc") is None

    def test_list_returns_none(self) -> None:
        assert _coerce_float([1, 2]) is None


class TestCoerceInt:
    def test_none_returns_none(self) -> None:
        assert _coerce_int(None) is None

    def test_float_returns_int(self) -> None:
        assert _coerce_int(3.7) == 3

    def test_string_int_returns_int(self) -> None:
        assert _coerce_int("42") == 42

    def test_non_numeric_returns_none(self) -> None:
        assert _coerce_int("xyz") is None


# ---------------------------------------------------------------------------
# _parse_iso / _duration_from_timestamps
# ---------------------------------------------------------------------------


class TestParseIso:
    def test_valid_iso_string(self) -> None:
        result = _parse_iso("2026-03-01T10:00:00+00:00")
        assert result is not None

    def test_z_suffix_normalized(self) -> None:
        result = _parse_iso("2026-03-01T10:00:00Z")
        assert result is not None

    def test_none_returns_none(self) -> None:
        assert _parse_iso(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_iso("") is None

    def test_non_string_returns_none(self) -> None:
        assert _parse_iso(12345) is None

    def test_invalid_string_returns_none(self) -> None:
        assert _parse_iso("not-a-date") is None


class TestDurationFromTimestamps:
    def test_valid_pair(self) -> None:
        result = _duration_from_timestamps(
            "2026-03-01T10:00:00+00:00",
            "2026-03-01T10:01:30+00:00",
        )
        assert result == pytest.approx(90.0)

    def test_same_timestamp_returns_zero(self) -> None:
        ts = "2026-03-01T10:00:00+00:00"
        assert _duration_from_timestamps(ts, ts) == 0.0

    def test_finish_before_start_returns_zero(self) -> None:
        result = _duration_from_timestamps(
            "2026-03-01T10:01:00+00:00",
            "2026-03-01T10:00:00+00:00",
        )
        assert result == 0.0

    def test_missing_start_returns_none(self) -> None:
        assert _duration_from_timestamps(None, "2026-03-01T10:00:00+00:00") is None

    def test_missing_finish_returns_none(self) -> None:
        assert _duration_from_timestamps("2026-03-01T10:00:00+00:00", None) is None


# ---------------------------------------------------------------------------
# _clean_cli_token
# ---------------------------------------------------------------------------


class TestCleanCliToken:
    def test_strips_whitespace(self) -> None:
        assert _clean_cli_token("  sonnet  ") == "sonnet"

    def test_strips_double_quotes(self) -> None:
        assert _clean_cli_token('"sonnet"') == "sonnet"

    def test_strips_single_quotes(self) -> None:
        assert _clean_cli_token("'sonnet'") == "sonnet"

    def test_no_quotes_unchanged(self) -> None:
        assert _clean_cli_token("opus") == "opus"


# ---------------------------------------------------------------------------
# _infer_engine_label
# ---------------------------------------------------------------------------


class TestInferEngineLabel:
    def test_shell_for_empty_command(self) -> None:
        assert _infer_engine_label({}) == "shell"
        assert _infer_engine_label({"command": ""}) == "shell"

    def test_claude_model_flag(self) -> None:
        result = _infer_engine_label({"command": "claude --model sonnet --print"})
        assert result == "claude:sonnet"

    def test_codex_model_flag(self) -> None:
        result = _infer_engine_label({"command": "codex -m gpt-5.4-codex exec"})
        assert result == "codex:gpt-5.4-codex"

    def test_codex_keyword(self) -> None:
        result = _infer_engine_label({"command": "codex exec something"})
        assert result == "codex"

    def test_claude_keyword(self) -> None:
        result = _infer_engine_label({"command": "claude --print prompt"})
        assert result == "claude"

    def test_gemini_keyword(self) -> None:
        result = _infer_engine_label({"command": "gemini run something"})
        assert result == "gemini"

    def test_shell_fallback(self) -> None:
        result = _infer_engine_label({"command": "echo hello"})
        assert result == "shell"

    def test_non_string_command(self) -> None:
        result = _infer_engine_label({"command": 123})
        assert result == "shell"

    def test_claude_model_with_quoted_value(self) -> None:
        result = _infer_engine_label({"command": 'claude --model="opus" --print'})
        assert result == "claude:opus"


# ---------------------------------------------------------------------------
# _normalize_task
# ---------------------------------------------------------------------------


class TestNormalizeTask:
    def test_basic_normalization(self) -> None:
        raw = {
            "status": "success",
            "duration_sec": 10.5,
            "cost_usd": 0.5,
            "token_usage": {"total_tokens": 3000},
            "command": "echo ok",
            "stdout_tail": "done",
            "started_at": "2026-03-01T10:00:00+00:00",
            "finished_at": "2026-03-01T10:00:10+00:00",
            "message": "",
        }
        result = _normalize_task("my-task", raw)
        assert result["task_id"] == "my-task"
        assert result["status"] == "success"
        assert result["duration_sec"] == pytest.approx(10.5)
        assert result["cost_usd"] == pytest.approx(0.5)
        assert result["tokens"] == 3000

    def test_missing_duration_uses_timestamps(self) -> None:
        raw = {
            "status": "success",
            "command": "echo ok",
            "started_at": "2026-03-01T10:00:00+00:00",
            "finished_at": "2026-03-01T10:00:30+00:00",
        }
        result = _normalize_task("t1", raw)
        assert result["duration_sec"] == pytest.approx(30.0)

    def test_missing_all_duration_defaults_zero(self) -> None:
        raw = {"status": "success", "command": "echo ok"}
        result = _normalize_task("t1", raw)
        assert result["duration_sec"] == 0.0

    def test_missing_cost_returns_none(self) -> None:
        raw = {"status": "success", "command": "echo ok"}
        result = _normalize_task("t1", raw)
        assert result["cost_usd"] is None

    def test_missing_token_usage_returns_none(self) -> None:
        raw = {"status": "success", "command": "echo ok"}
        result = _normalize_task("t1", raw)
        assert result["tokens"] is None

    def test_token_usage_from_components(self) -> None:
        raw = {
            "status": "success",
            "command": "echo ok",
            "token_usage": {
                "input_tokens": 100,
                "cached_tokens": 50,
                "output_tokens": 200,
            },
        }
        result = _normalize_task("t1", raw)
        assert result["tokens"] == 350

    def test_token_usage_all_zero_returns_none(self) -> None:
        raw = {
            "status": "success",
            "command": "echo ok",
            "token_usage": {
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
            },
        }
        result = _normalize_task("t1", raw)
        assert result["tokens"] is None

    def test_status_lowercased(self) -> None:
        raw = {"status": "SUCCESS", "command": "echo ok"}
        result = _normalize_task("t1", raw)
        assert result["status"] == "success"

    def test_missing_status_defaults_pending(self) -> None:
        raw = {"command": "echo ok"}
        result = _normalize_task("t1", raw)
        assert result["status"] == "pending"


# ---------------------------------------------------------------------------
# _status_counts
# ---------------------------------------------------------------------------


class TestStatusCounts:
    def test_all_success_like(self) -> None:
        tasks = [
            {"status": "success"},
            {"status": "soft_failed"},
            {"status": "dry_run"},
        ]
        ok, failed, skipped = _status_counts(tasks)
        assert ok == 3
        assert failed == 0
        assert skipped == 0

    def test_mixed_statuses(self) -> None:
        tasks = [
            {"status": "success"},
            {"status": "failed"},
            {"status": "skipped"},
            {"status": "failed"},
        ]
        ok, failed, skipped = _status_counts(tasks)
        assert ok == 1
        assert failed == 2
        assert skipped == 1

    def test_empty_list(self) -> None:
        ok, failed, skipped = _status_counts([])
        assert ok == 0
        assert failed == 0
        assert skipped == 0

    def test_unknown_status_not_counted(self) -> None:
        tasks = [{"status": "pending"}, {"status": "running"}]
        ok, failed, skipped = _status_counts(tasks)
        assert ok == 0
        assert failed == 0
        assert skipped == 0


# ---------------------------------------------------------------------------
# _total_cost / _total_tokens / _run_duration
# ---------------------------------------------------------------------------


class TestTotalCost:
    def test_manifest_value_takes_priority(self) -> None:
        manifest = {"total_cost_usd": 5.0}
        tasks = [{"cost_usd": 1.0}, {"cost_usd": 2.0}]
        assert _total_cost(manifest, tasks) == pytest.approx(5.0)

    def test_sums_task_costs_when_no_manifest(self) -> None:
        manifest: dict[str, object] = {}
        tasks = [{"cost_usd": 1.0}, {"cost_usd": 2.0}]
        assert _total_cost(manifest, tasks) == pytest.approx(3.0)

    def test_none_when_no_costs(self) -> None:
        manifest: dict[str, object] = {}
        tasks = [{"cost_usd": None}]
        assert _total_cost(manifest, tasks) is None

    def test_ignores_none_in_task_costs(self) -> None:
        manifest: dict[str, object] = {}
        tasks = [{"cost_usd": 1.0}, {"cost_usd": None}]
        assert _total_cost(manifest, tasks) == pytest.approx(1.0)


class TestTotalTokens:
    def test_manifest_value_takes_priority(self) -> None:
        manifest = {"total_tokens": 9999}
        tasks = [{"tokens": 100}]
        assert _total_tokens(manifest, tasks) == 9999

    def test_sums_task_tokens_when_no_manifest(self) -> None:
        manifest: dict[str, object] = {}
        tasks = [{"tokens": 100}, {"tokens": 200}]
        assert _total_tokens(manifest, tasks) == 300

    def test_none_when_no_tokens(self) -> None:
        manifest: dict[str, object] = {}
        tasks = [{"tokens": None}]
        assert _total_tokens(manifest, tasks) is None


class TestRunDuration:
    def test_manifest_timestamps_take_priority(self) -> None:
        manifest = {
            "started_at": "2026-03-01T10:00:00+00:00",
            "finished_at": "2026-03-01T10:02:00+00:00",
        }
        tasks = [{"duration_sec": 5.0}]
        assert _run_duration(manifest, tasks) == pytest.approx(120.0)

    def test_sums_task_durations_as_fallback(self) -> None:
        manifest: dict[str, object] = {}
        tasks = [{"duration_sec": 10.0}, {"duration_sec": 20.0}]
        assert _run_duration(manifest, tasks) == pytest.approx(30.0)

    def test_none_when_empty_tasks_and_no_timestamps(self) -> None:
        manifest: dict[str, object] = {}
        assert _run_duration(manifest, []) is None


# ---------------------------------------------------------------------------
# _status_label
# ---------------------------------------------------------------------------


class TestStatusLabel:
    def test_success_from_manifest(self) -> None:
        text, kind = _status_label({"success": True}, [])
        assert text == "SUCCESS"
        assert kind == "success"

    def test_failed_from_manifest(self) -> None:
        text, kind = _status_label({"success": False}, [])
        assert text == "FAILED"
        assert kind == "failed"

    def test_failed_inferred_from_tasks(self) -> None:
        tasks = [{"status": "failed"}]
        text, kind = _status_label({}, tasks)
        assert text == "FAILED"
        assert kind == "failed"

    def test_success_inferred_from_tasks(self) -> None:
        tasks = [{"status": "success"}]
        text, kind = _status_label({}, tasks)
        assert text == "SUCCESS"
        assert kind == "success"

    def test_unknown_when_no_data(self) -> None:
        text, kind = _status_label({}, [])
        assert text == "UNKNOWN"
        assert kind == "pending"


# ---------------------------------------------------------------------------
# _format_duration
# ---------------------------------------------------------------------------


class TestFormatDuration:
    def test_none_returns_dash(self) -> None:
        assert _format_duration(None) == "\u2014"

    def test_sub_second_shows_ms(self) -> None:
        assert _format_duration(0.5) == "500ms"

    def test_seconds_range(self) -> None:
        assert _format_duration(30.5) == "30.5s"

    def test_minutes_range(self) -> None:
        assert _format_duration(90.0) == "1m 30s"

    def test_zero_shows_ms(self) -> None:
        assert _format_duration(0.0) == "0ms"


# ---------------------------------------------------------------------------
# _json_for_script
# ---------------------------------------------------------------------------


class TestJsonForScript:
    def test_escapes_closing_script_tag(self) -> None:
        result = _json_for_script({"value": "</script>"})
        assert "</" not in result
        assert "<\\/" in result

    def test_valid_json_output(self) -> None:
        data = {"name": "test", "count": 42}
        result = _json_for_script(data)
        parsed = json.loads(result)
        assert parsed["name"] == "test"
        assert parsed["count"] == 42


# ---------------------------------------------------------------------------
# _prepare_report_data
# ---------------------------------------------------------------------------


class TestPrepareReportData:
    def test_unknown_plan_name_fallback(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {"task_results": {}}
        data = _prepare_report_data(manifest, run_path)
        assert data["plan_name"] == "(unknown plan)"

    def test_run_id_fallback_to_dirname(self, tmp_path: Path) -> None:
        run_path = tmp_path / "my-custom-run"
        run_path.mkdir()
        manifest: dict[str, object] = {"task_results": {}}
        data = _prepare_report_data(manifest, run_path)
        assert data["run_id"] == "my-custom-run"

    def test_non_dict_task_results_treated_as_empty(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {"task_results": "bad"}
        data = _prepare_report_data(manifest, run_path)
        assert data["tasks"] == []
        assert data["task_count"] == 0

    def test_tasks_sorted_by_started_at(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "task_results": {
                "late-task": {
                    "status": "success",
                    "started_at": "2026-03-01T10:05:00+00:00",
                    "finished_at": "2026-03-01T10:06:00+00:00",
                    "command": "echo late",
                },
                "early-task": {
                    "status": "success",
                    "started_at": "2026-03-01T10:00:00+00:00",
                    "finished_at": "2026-03-01T10:01:00+00:00",
                    "command": "echo early",
                },
            },
        }
        data = _prepare_report_data(manifest, run_path)
        assert data["tasks"][0]["task_id"] == "early-task"
        assert data["tasks"][1]["task_id"] == "late-task"

    def test_all_five_statuses(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "task_results": {
                "ok": {"status": "success", "command": "echo ok"},
                "fail": {"status": "failed", "command": "echo fail"},
                "soft": {"status": "soft_failed", "command": "echo soft"},
                "skip": {"status": "skipped", "command": "echo skip"},
                "dry": {"status": "dry_run", "command": "echo dry"},
            },
        }
        data = _prepare_report_data(manifest, run_path)
        assert data["ok_count"] == 3  # success + soft_failed + dry_run
        assert data["failed_count"] == 1
        assert data["skipped_count"] == 1
        assert data["task_count"] == 5


# ---------------------------------------------------------------------------
# build_report_html — additional coverage
# ---------------------------------------------------------------------------


class TestBuildReportHtmlAdditional:
    def test_empty_task_results(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "plan_name": "empty-plan",
            "run_id": "e-001",
            "success": True,
            "task_results": {},
        }
        html = build_report_html(manifest, run_path)
        assert "empty-plan" in html
        assert "e-001" in html
        # Should still be valid HTML
        assert "<!doctype html>" in html

    def test_all_statuses_in_html(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "plan_name": "multi-status",
            "success": False,
            "task_results": {
                "ok": {"status": "success", "command": "echo ok"},
                "fail": {"status": "failed", "command": "echo fail"},
                "soft": {"status": "soft_failed", "command": "echo soft"},
                "skip": {"status": "skipped", "command": "echo skip"},
                "dry": {"status": "dry_run", "command": "echo dry"},
            },
        }
        html = build_report_html(manifest, run_path)
        # All task IDs should appear in the report data JSON
        for tid in ["ok", "fail", "soft", "skip", "dry"]:
            assert tid in html

    def test_html_escaping_in_plan_name(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "plan_name": '<script>alert("xss")</script>',
            "run_id": "xss-001",
            "success": True,
            "task_results": {},
        }
        html = build_report_html(manifest, run_path)
        # The plan name in the HTML header section should be escaped
        assert "&lt;script&gt;" in html
        # The closing </script> inside JSON data should be escaped to <\/
        # to prevent premature script tag closure
        assert "</script>alert" not in html

    def test_large_run_many_tasks(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        task_results: dict[str, object] = {}
        for i in range(50):
            task_results[f"task-{i:03d}"] = {
                "status": "success" if i % 3 != 0 else "failed",
                "duration_sec": float(i),
                "cost_usd": float(i) * 0.01,
                "token_usage": {"total_tokens": i * 100},
                "command": f"echo task-{i:03d}",
                "stdout_tail": f"output {i}",
            }
        manifest: dict[str, object] = {
            "plan_name": "large-plan",
            "run_id": "large-001",
            "success": False,
            "task_results": task_results,
        }
        html = build_report_html(manifest, run_path)
        # Verify report data JSON contains all tasks
        start = html.index('<script id="report-data" type="application/json">') + len(
            '<script id="report-data" type="application/json">'
        )
        end = html.index("</script>", start)
        data = json.loads(html[start:end])
        assert len(data["tasks"]) == 50

    def test_report_with_cost_and_tokens(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "plan_name": "costed-plan",
            "run_id": "c-001",
            "success": True,
            "total_cost_usd": 1.23,
            "total_tokens": 45000,
            "task_results": {
                "t1": {
                    "status": "success",
                    "cost_usd": 1.23,
                    "token_usage": {"total_tokens": 45000},
                    "command": "claude --model sonnet",
                },
            },
        }
        html = build_report_html(manifest, run_path)
        assert "$1.23" in html
        assert "45,000" in html

    def test_report_without_timestamps(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "plan_name": "no-ts",
            "run_id": "nts-001",
            "success": True,
            "task_results": {
                "t1": {"status": "success", "duration_sec": 5.0, "command": "echo ok"},
            },
        }
        html = build_report_html(manifest, run_path)
        assert "no-ts" in html
        # Duration should still show from task duration_sec
        assert "5.0s" in html

    def test_task_with_message_field(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "plan_name": "msg-plan",
            "success": False,
            "task_results": {
                "t1": {
                    "status": "failed",
                    "command": "echo fail",
                    "message": "Something went wrong",
                },
            },
        }
        html = build_report_html(manifest, run_path)
        start = html.index('<script id="report-data" type="application/json">') + len(
            '<script id="report-data" type="application/json">'
        )
        end = html.index("</script>", start)
        data = json.loads(html[start:end])
        assert data["tasks"][0]["message"] == "Something went wrong"

    def test_non_dict_task_entry_skipped(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "plan_name": "bad-task",
            "success": True,
            "task_results": {
                "good": {"status": "success", "command": "echo ok"},
                "bad": "not-a-dict",
            },
        }
        html = build_report_html(manifest, run_path)
        start = html.index('<script id="report-data" type="application/json">') + len(
            '<script id="report-data" type="application/json">'
        )
        end = html.index("</script>", start)
        data = json.loads(html[start:end])
        assert len(data["tasks"]) == 1
        assert data["tasks"][0]["task_id"] == "good"


# ---------------------------------------------------------------------------
# generate_report — additional coverage
# ---------------------------------------------------------------------------


class TestGenerateReportAdditional:
    def test_run_path_does_not_exist(self, tmp_path: Path) -> None:
        run_path = tmp_path / "nonexistent-dir"
        with pytest.raises(FileNotFoundError):
            generate_report(run_path)

    def test_output_path_parent_created(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        (run_path / "run_manifest.json").write_text(
            json.dumps(_sample_manifest()), encoding="utf-8"
        )
        deep_output = tmp_path / "a" / "b" / "c" / "report.html"
        result = generate_report(run_path, output_path=deep_output)
        assert result == deep_output
        assert deep_output.exists()

    def test_report_html_is_valid_utf8(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest = {
            "plan_name": "unicode-plan-\u00e9\u00e0\u00fc",
            "success": True,
            "task_results": {
                "t1": {"status": "success", "command": "echo \u00e9"},
            },
        }
        (run_path / "run_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
        )
        result = generate_report(run_path)
        content = result.read_text(encoding="utf-8")
        assert "unicode-plan" in content

    def test_report_contains_doctype(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        (run_path / "run_manifest.json").write_text(
            json.dumps(_sample_manifest()), encoding="utf-8"
        )
        result = generate_report(run_path)
        content = result.read_text(encoding="utf-8")
        assert content.startswith("<!doctype html>")

    def test_report_with_only_skipped_tasks(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "plan_name": "all-skipped",
            "run_id": "sk-001",
            "success": True,
            "task_results": {
                "t1": {"status": "skipped", "command": "echo skip1"},
                "t2": {"status": "skipped", "command": "echo skip2"},
            },
        }
        (run_path / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        result = generate_report(run_path)
        content = result.read_text(encoding="utf-8")
        assert "0 ok / 0 failed / 2 skipped" in content

    def test_report_with_zero_cost_tasks(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run"
        run_path.mkdir()
        manifest: dict[str, object] = {
            "plan_name": "zero-cost",
            "success": True,
            "task_results": {
                "t1": {"status": "success", "cost_usd": 0.0, "command": "echo ok"},
            },
        }
        (run_path / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        result = generate_report(run_path)
        content = result.read_text(encoding="utf-8")
        assert "$0.00" in content

