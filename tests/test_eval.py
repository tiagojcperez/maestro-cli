from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.errors import PlanValidationError
from maestro_cli.eval import (
    DimensionResult,
    EvalResult,
    EvalSuiteResult,
    _build_judge_spec,
    _coerce_float,
    _extract_duration_sec,
    _is_glob_pattern,
    _matches_patterns,
    _missing_requested_task_ids,
    _parse_iso,
    _resolve_tasks,
    format_eval,
    format_eval_json,
    load_eval_spec,
    run_eval,
)
from maestro_cli.models import JudgeResult


def _write_eval(tmp_path: Path, text: str) -> Path:
    eval_path = tmp_path / "eval.yaml"
    eval_path.write_text(text, encoding="utf-8")
    return eval_path


class TestLoadEvalSpec:
    def test_loads_and_merges_overrides(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: quality-suite
tasks: ["task-a", "task-b"]
exclude: ["task-c"]
judge:
  model: sonnet
  criteria:
    - "must be correct"
  pass_threshold: 0.7
overrides:
  task-b:
    pass_threshold: 0.9
""",
        )

        spec = load_eval_spec(eval_path)

        assert spec["name"] == "quality-suite"
        assert spec["tasks"] == ["task-a", "task-b"]
        assert spec["exclude"] == ["task-c"]
        assert spec["judge"].model == "sonnet"
        assert spec["judge"].pass_threshold == pytest.approx(0.7)
        assert spec["overrides"]["task-b"].pass_threshold == pytest.approx(0.9)
        assert spec["overrides"]["task-b"].criteria == ["must be correct"]

    def test_defaults_tasks_and_timeout(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: defaults
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )

        spec = load_eval_spec(eval_path)

        assert spec["tasks"] == ["*"]
        assert spec["timeout_sec"] == 45
        assert any("timeout_sec was not provided" in warning for warning in spec["validation_warnings"])

    def test_rejects_missing_judge(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: broken
tasks: ["*"]
""",
        )

        with pytest.raises(PlanValidationError, match="judge must be an object"):
            load_eval_spec(eval_path)

    def test_rejects_nonexistent_file(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.yaml"
        with pytest.raises(PlanValidationError, match="Eval file not found"):
            load_eval_spec(missing)

    def test_rejects_invalid_yaml(self, tmp_path: Path) -> None:
        eval_path = tmp_path / "bad.yaml"
        eval_path.write_text(":\nfoo: [unclosed", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="Invalid YAML"):
            load_eval_spec(eval_path)

    def test_rejects_non_dict_root(self, tmp_path: Path) -> None:
        eval_path = tmp_path / "list.yaml"
        eval_path.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(PlanValidationError, match="Eval root must be an object"):
            load_eval_spec(eval_path)

    def test_rejects_empty_name(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: ""
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        with pytest.raises(PlanValidationError, match="Eval name must be a non-empty string"):
            load_eval_spec(eval_path)

    def test_rejects_negative_timeout(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: -5
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        with pytest.raises(PlanValidationError, match="timeout_sec must be"):
            load_eval_spec(eval_path)

    def test_rejects_non_int_timeout(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: "fast"
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        with pytest.raises(PlanValidationError, match="timeout_sec must be an integer"):
            load_eval_spec(eval_path)

    def test_rejects_empty_override_key(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
judge:
  model: sonnet
  criteria:
    - "ok"
overrides:
  "": {pass_threshold: 0.8}
""",
        )
        with pytest.raises(PlanValidationError, match="overrides keys must be non-empty"):
            load_eval_spec(eval_path)

    def test_rejects_non_dict_override_value(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
judge:
  model: sonnet
  criteria:
    - "ok"
overrides:
  task-a: "not-a-dict"
""",
        )
        with pytest.raises(PlanValidationError, match="overrides.task-a must be an object"):
            load_eval_spec(eval_path)

    def test_warns_unknown_model(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
judge:
  model: unknown-model-xyz
  criteria:
    - "ok"
""",
        )
        spec = load_eval_spec(eval_path)
        assert any("unknown-model-xyz" in w for w in spec["validation_warnings"])

    def test_warns_g_eval_with_haiku(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
judge:
  model: haiku
  method: g_eval
  criteria:
    - "ok"
""",
        )
        spec = load_eval_spec(eval_path)
        assert any("g_eval" in w and "haiku" in w for w in spec["validation_warnings"])


class TestResolveTasks:
    def test_filters_with_glob_and_exclude(self) -> None:
        selected = _resolve_tasks(
            {"tasks": ["api-*", "worker"], "exclude": ["*-old", "worker"]},
            ["api-a", "api-old", "worker", "misc"],
        )
        assert selected == ["api-a"]

    def test_wildcard_selects_all_tasks(self) -> None:
        selected = _resolve_tasks({"tasks": ["*"]}, ["task-a", "task-b", "task-c"])
        assert selected == ["task-a", "task-b", "task-c"]

    def test_empty_include_defaults_to_wildcard(self) -> None:
        # Empty tasks list defaults to ["*"]
        selected = _resolve_tasks({}, ["task-a", "task-b"])
        assert selected == ["task-a", "task-b"]

    def test_task_matching_both_include_and_exclude_is_excluded(self) -> None:
        selected = _resolve_tasks(
            {"tasks": ["*"], "exclude": ["task-b"]},
            ["task-a", "task-b", "task-c"],
        )
        assert "task-b" not in selected
        assert selected == ["task-a", "task-c"]

    def test_no_tasks_match_patterns(self) -> None:
        selected = _resolve_tasks({"tasks": ["xyz-*"]}, ["task-a", "task-b"])
        assert selected == []


class TestMissingRequestedTaskIds:
    def test_plain_pattern_not_in_available_returned(self) -> None:
        missing = _missing_requested_task_ids(
            {"tasks": ["task-a", "task-missing"]},
            ["task-a"],
        )
        assert missing == ["task-missing"]

    def test_glob_pattern_not_matching_not_returned(self) -> None:
        # Globs are skipped — only plain task IDs are checked
        missing = _missing_requested_task_ids(
            {"tasks": ["task-*", "exact-miss"]},
            ["task-a"],
        )
        assert missing == ["exact-miss"]

    def test_all_found_returns_empty(self) -> None:
        missing = _missing_requested_task_ids(
            {"tasks": ["task-a", "task-b"]},
            ["task-a", "task-b", "task-c"],
        )
        assert missing == []

    def test_default_wildcard_spec_returns_empty(self) -> None:
        # No tasks key → defaults to ["*"] which is a glob → nothing counted as missing
        missing = _missing_requested_task_ids({}, ["task-a", "task-b"])
        assert missing == []


class TestIsGlobPattern:
    @pytest.mark.parametrize("pattern", ["task-*", "task-?", "task-[ab]", "*"])
    def test_glob_chars_detected(self, pattern: str) -> None:
        assert _is_glob_pattern(pattern) is True

    @pytest.mark.parametrize("pattern", ["task-a", "my-task", "build"])
    def test_plain_strings_not_glob(self, pattern: str) -> None:
        assert _is_glob_pattern(pattern) is False


class TestRunEval:
    def test_runs_judge_for_selected_tasks(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: run-suite
tasks: ["task-a", "task-b"]
judge:
  model: sonnet
  criteria:
    - "must be correct"
  pass_threshold: 0.7
overrides:
  task-b:
    pass_threshold: 0.9
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {
            "task_results": {
                "task-a": {"stdout_tail": "alpha", "cost_usd": 1.2, "duration_sec": 2.5},
                "task-b": {"stdout_tail": "beta", "cost_usd": 2.3, "duration_sec": 3.0},
            }
        }

        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)

        calls: list[dict[str, Any]] = []

        def _fake_judge(
            task_id: str,
            judge: Any,
            stdout_tail: str,
            workdir: Path,
            cost_usd: float | None = None,
            duration_sec: float = 0.0,
            timeout_sec: int = 45,
        ) -> JudgeResult:
            calls.append(
                {
                    "task_id": task_id,
                    "pass_threshold": judge.pass_threshold,
                    "stdout_tail": stdout_tail,
                    "workdir": workdir,
                    "cost_usd": cost_usd,
                    "duration_sec": duration_sec,
                    "timeout_sec": timeout_sec,
                }
            )
            score = 0.85 if task_id == "task-b" else 0.95
            verdict = "pass" if score >= judge.pass_threshold else "fail"
            return JudgeResult(verdict=verdict, overall_score=score, reasoning="ok")

        monkeypatch.setattr("maestro_cli.runners._run_judge_evaluation", _fake_judge)

        suite = run_eval(eval_path, run_path)

        assert [result.task_id for result in suite.results] == ["task-a", "task-b"]
        assert suite.passed == 1
        assert suite.failed == 1
        assert suite.errors == 0
        assert suite.overall_pass is False
        assert calls[0]["pass_threshold"] == pytest.approx(0.7)
        assert calls[1]["pass_threshold"] == pytest.approx(0.9)
        assert calls[0]["stdout_tail"] == "alpha"
        assert calls[1]["duration_sec"] == pytest.approx(3.0)

    def test_skips_unparseable_task_entries_and_captures_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: error-suite
tasks: ["task-a", "task-b", "task-missing"]
judge:
  model: sonnet
  criteria:
    - "must be correct"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {
            "task_results": {
                "task-a": {"stdout_tail": "alpha"},
                "task-b": "not-an-object",
            }
        }
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)

        def _raising_judge(*args: Any, **kwargs: Any) -> JudgeResult:
            raise RuntimeError("judge boom")

        monkeypatch.setattr("maestro_cli.runners._run_judge_evaluation", _raising_judge)

        suite = run_eval(eval_path, run_path)

        assert len(suite.results) == 1
        assert suite.results[0].task_id == "task-a"
        assert suite.results[0].judge_result.verdict == "error"
        assert "task-b" in suite.skipped
        assert "task-missing" in suite.skipped
        assert suite.errors == 1


class TestEvalSuiteResultProperties:
    def _make_result(self, verdict: str) -> EvalResult:
        return EvalResult(
            task_id="t",
            judge_result=JudgeResult(verdict=verdict, overall_score=0.5, reasoning=""),
            passed=(verdict == "pass"),
        )

    def test_passed_counts_pass_verdicts(self) -> None:
        suite = EvalSuiteResult(
            name="s", run_path=Path("."),
            results=[self._make_result("pass"), self._make_result("fail"), self._make_result("pass")],
            skipped=[],
        )
        assert suite.passed == 2

    def test_failed_counts_non_pass_non_error(self) -> None:
        suite = EvalSuiteResult(
            name="s", run_path=Path("."),
            results=[self._make_result("fail"), self._make_result("error"), self._make_result("pass")],
            skipped=[],
        )
        assert suite.failed == 1

    def test_errors_counts_error_verdicts(self) -> None:
        suite = EvalSuiteResult(
            name="s", run_path=Path("."),
            results=[self._make_result("error"), self._make_result("error"), self._make_result("pass")],
            skipped=[],
        )
        assert suite.errors == 2

    def test_overall_pass_requires_no_failures_and_no_errors(self) -> None:
        all_pass = EvalSuiteResult(
            name="s", run_path=Path("."),
            results=[self._make_result("pass")],
            skipped=[],
        )
        has_fail = EvalSuiteResult(
            name="s", run_path=Path("."),
            results=[self._make_result("pass"), self._make_result("fail")],
            skipped=[],
        )
        has_error = EvalSuiteResult(
            name="s", run_path=Path("."),
            results=[self._make_result("pass"), self._make_result("error")],
            skipped=[],
        )
        assert all_pass.overall_pass is True
        assert has_fail.overall_pass is False
        assert has_error.overall_pass is False

    def test_overall_pass_for_empty_results(self) -> None:
        suite = EvalSuiteResult(name="s", run_path=Path("."), results=[], skipped=[])
        assert suite.overall_pass is True


class TestHelperFunctions:
    def test_coerce_float_none_returns_none(self) -> None:
        assert _coerce_float(None) is None

    def test_coerce_float_valid_value(self) -> None:
        assert _coerce_float("3.14") == pytest.approx(3.14)
        assert _coerce_float(42) == pytest.approx(42.0)

    def test_coerce_float_invalid_string_returns_none(self) -> None:
        assert _coerce_float("not-a-number") is None

    def test_parse_iso_valid_string(self) -> None:
        dt = _parse_iso("2024-01-15T10:00:00+00:00")
        assert dt is not None
        assert dt.year == 2024

    def test_parse_iso_z_suffix_normalized(self) -> None:
        dt = _parse_iso("2024-01-15T10:00:00Z")
        assert dt is not None
        assert dt.year == 2024

    def test_parse_iso_empty_returns_none(self) -> None:
        assert _parse_iso("") is None

    def test_parse_iso_non_string_returns_none(self) -> None:
        assert _parse_iso(None) is None
        assert _parse_iso(12345) is None

    def test_extract_duration_explicit(self) -> None:
        result = _extract_duration_sec({"duration_sec": 7.5})
        assert result == pytest.approx(7.5)

    def test_extract_duration_from_timestamps(self) -> None:
        result = _extract_duration_sec({
            "started_at": "2024-01-15T10:00:00Z",
            "finished_at": "2024-01-15T10:00:30Z",
        })
        assert result == pytest.approx(30.0)

    def test_extract_duration_missing_timestamps_returns_zero(self) -> None:
        assert _extract_duration_sec({}) == pytest.approx(0.0)


class TestFormatting:
    def test_formatters_render_summary_and_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: fmt-suite
tasks: ["*"]
judge:
  model: sonnet
  criteria:
    - "must be correct"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {"task_results": {"task-a": {"stdout_tail": "alpha"}}}
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_evaluation",
            lambda *args, **kwargs: JudgeResult(verdict="pass", overall_score=0.93, reasoning="ok"),
        )

        suite = run_eval(eval_path, run_path)

        text = format_eval(suite)
        assert "Summary:" in text
        assert "task-a" in text
        assert "0.93" in text

        payload = json.loads(format_eval_json(suite))
        assert payload["name"] == "fmt-suite"
        assert payload["overall_pass"] is True
        assert payload["passed"] == 1

    def test_format_eval_empty_results_shows_message(self, tmp_path: Path) -> None:
        suite = EvalSuiteResult(
            name="empty-suite",
            run_path=tmp_path,
            results=[],
            skipped=["missing-task"],
        )
        text = format_eval(suite)
        assert "No evaluated tasks." in text
        assert "missing-task" in text

    def test_format_eval_json_includes_all_required_fields(self, tmp_path: Path) -> None:
        suite = EvalSuiteResult(
            name="json-suite",
            run_path=tmp_path,
            results=[
                EvalResult(
                    task_id="task-a",
                    judge_result=JudgeResult(verdict="pass", overall_score=0.9, reasoning="ok"),
                    passed=True,
                )
            ],
            skipped=["task-b"],
        )
        payload = json.loads(format_eval_json(suite))
        assert "name" in payload
        assert "run_path" in payload
        assert "results" in payload
        assert "skipped" in payload
        assert "passed" in payload
        assert "failed" in payload
        assert "errors" in payload
        assert "overall_pass" in payload
        assert payload["skipped"] == ["task-b"]
        assert payload["results"][0]["task_id"] == "task-a"

    def test_run_eval_non_dict_task_results_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        monkeypatch.setattr(
            "maestro_cli.eval.load_run_manifest", lambda _: {"task_results": "not-a-dict"}
        )
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_evaluation",
            lambda *a, **kw: JudgeResult(verdict="pass", overall_score=1.0, reasoning=""),
        )

        suite = run_eval(eval_path, run_path)
        assert suite.results == []

    def test_run_eval_stdout_tail_non_string_coerced(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {"task_results": {"task-a": {"stdout_tail": 12345}}}
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)

        captured: list[str] = []

        def _capture_judge(task_id: str, judge: Any, stdout_tail: str, **kwargs: Any) -> JudgeResult:
            captured.append(stdout_tail)
            return JudgeResult(verdict="pass", overall_score=1.0, reasoning="")

        monkeypatch.setattr("maestro_cli.runners._run_judge_evaluation", _capture_judge)

        run_eval(eval_path, run_path)
        assert captured == ["12345"]

    def test_rejects_invalid_preset(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
judge:
  model: sonnet
  preset: nonexistent_preset
  criteria:
    - "ok"
""",
        )
        with pytest.raises(PlanValidationError, match="nonexistent_preset"):
            load_eval_spec(eval_path)

    def test_extract_duration_negative_clamped_to_zero(self) -> None:
        result = _extract_duration_sec({"duration_sec": -5.0})
        assert result == pytest.approx(0.0)

    def test_build_judge_spec_non_dict_raises(self) -> None:
        with pytest.raises(PlanValidationError, match="must be an object"):
            _build_judge_spec("not-a-dict")  # type: ignore[arg-type]

    def test_matches_patterns_with_multiple_patterns(self) -> None:
        assert _matches_patterns("task-a", ["task-a", "task-b"]) is True
        assert _matches_patterns("task-c", ["task-a", "task-b"]) is False
        assert _matches_patterns("api-task", ["api-*"]) is True

    def test_parse_iso_invalid_format_returns_none(self) -> None:
        # Hits the ValueError branch in _parse_iso
        assert _parse_iso("not-a-date") is None
        assert _parse_iso("2024-99-99T99:99:99") is None

    def test_format_eval_skipped_shown_alongside_results(self, tmp_path: Path) -> None:
        suite = EvalSuiteResult(
            name="mixed-suite",
            run_path=tmp_path,
            results=[
                EvalResult(
                    task_id="task-a",
                    judge_result=JudgeResult(verdict="pass", overall_score=0.9, reasoning="ok"),
                    passed=True,
                )
            ],
            skipped=["task-b", "task-c"],
        )
        text = format_eval(suite)
        assert "task-a" in text
        assert "Skipped:" in text
        assert "task-b" in text
        assert "task-c" in text

    def test_load_eval_spec_null_overrides_treated_as_empty(self, tmp_path: Path) -> None:
        # overrides: null should not raise — coerced to {}
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
judge:
  model: sonnet
  criteria:
    - "ok"
overrides: null
""",
        )
        spec = load_eval_spec(eval_path)
        assert spec["overrides"] == {}

    def test_load_eval_spec_empty_tasks_list_defaults_to_wildcard(self, tmp_path: Path) -> None:
        # tasks: [] after normalisation → falls back to ["*"]
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
tasks: []
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        spec = load_eval_spec(eval_path)
        assert spec["tasks"] == ["*"]

    def test_run_eval_manifest_missing_task_results_key_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Manifest dict without 'task_results' key → no tasks evaluated
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: {})
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_evaluation",
            lambda *a, **kw: JudgeResult(verdict="pass", overall_score=1.0, reasoning=""),
        )

        suite = run_eval(eval_path, run_path)
        assert suite.results == []
        assert suite.skipped == []

    def test_rejects_zero_timeout(self, tmp_path: Path) -> None:
        # timeout_sec: 0 fails the >= 1 check
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 0
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        with pytest.raises(PlanValidationError, match="timeout_sec must be"):
            load_eval_spec(eval_path)

    def test_run_eval_non_string_task_id_keys_filtered(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Non-string keys in task_results are skipped (isinstance check in run_eval)
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        # task_results with a non-string key (int 42) and a valid string key
        manifest: dict[str, Any] = {
            "task_results": {
                "task-a": {"stdout_tail": "hello"},
                42: {"stdout_tail": "ignored"},
            }
        }
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_evaluation",
            lambda *a, **kw: JudgeResult(verdict="pass", overall_score=1.0, reasoning=""),
        )

        suite = run_eval(eval_path, run_path)
        assert len(suite.results) == 1
        assert suite.results[0].task_id == "task-a"

    def test_extract_duration_only_one_timestamp_returns_zero(self) -> None:
        # Only started_at present (no finished_at) → 0.0
        result = _extract_duration_sec({"started_at": "2024-01-15T10:00:00Z"})
        assert result == pytest.approx(0.0)
        # Only finished_at present (no started_at) → 0.0
        result2 = _extract_duration_sec({"finished_at": "2024-01-15T10:00:30Z"})
        assert result2 == pytest.approx(0.0)

    def test_extract_duration_reversed_timestamps_clamped_to_zero(self) -> None:
        # finished_at before started_at → negative delta → clamped to 0.0
        result = _extract_duration_sec({
            "started_at": "2024-01-15T10:00:30Z",
            "finished_at": "2024-01-15T10:00:00Z",
        })
        assert result == pytest.approx(0.0)

    def test_format_eval_no_skipped_omits_skipped_section(self, tmp_path: Path) -> None:
        suite = EvalSuiteResult(
            name="full-suite",
            run_path=tmp_path,
            results=[
                EvalResult(
                    task_id="task-a",
                    judge_result=JudgeResult(verdict="pass", overall_score=1.0, reasoning="ok"),
                    passed=True,
                )
            ],
            skipped=[],
        )
        text = format_eval(suite)
        assert "task-a" in text
        assert "Skipped:" not in text

    def test_coerce_float_list_input_returns_none(self) -> None:
        # list input → float([1, 2]) raises TypeError → coerce returns None
        assert _coerce_float([1, 2]) is None  # type: ignore[arg-type]
        assert _coerce_float({}) is None  # type: ignore[arg-type]

    def test_load_eval_spec_explicit_timeout_no_default_warning(self, tmp_path: Path) -> None:
        # When timeout_sec is explicitly provided, the "not provided" warning must be absent
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 60
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        spec = load_eval_spec(eval_path)
        assert spec["timeout_sec"] == 60
        assert not any("timeout_sec was not provided" in w for w in spec["validation_warnings"])

    def test_format_eval_json_result_judge_result_nested_structure(self, tmp_path: Path) -> None:
        suite = EvalSuiteResult(
            name="struct-suite",
            run_path=tmp_path,
            results=[
                EvalResult(
                    task_id="task-x",
                    judge_result=JudgeResult(verdict="fail", overall_score=0.4, reasoning="bad"),
                    passed=False,
                )
            ],
            skipped=[],
        )
        payload = json.loads(format_eval_json(suite))
        result = payload["results"][0]
        assert "judge_result" in result
        assert isinstance(result["judge_result"], dict)
        assert result["judge_result"]["verdict"] == "fail"
        assert result["passed"] is False
        assert payload["failed"] == 1
        assert payload["overall_pass"] is False

    def test_rejects_list_overrides(self, tmp_path: Path) -> None:
        # overrides: [foo] is a list, not a dict → PlanValidationError
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
judge:
  model: sonnet
  criteria:
    - "ok"
overrides:
  - task-a
""",
        )
        with pytest.raises(PlanValidationError, match="overrides must be an object"):
            load_eval_spec(eval_path)

    def test_warns_unknown_model_in_override(self, tmp_path: Path) -> None:
        # Override spec with unknown model triggers a warning at overrides.task-b level
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
judge:
  model: sonnet
  criteria:
    - "ok"
overrides:
  task-b:
    model: totally-unknown-xyz
""",
        )
        spec = load_eval_spec(eval_path)
        assert any("overrides.task-b" in w and "totally-unknown-xyz" in w for w in spec["validation_warnings"])

    def test_extract_duration_invalid_duration_string_falls_through_to_timestamps(self) -> None:
        # duration_sec as non-numeric string → _coerce_float returns None → falls through to timestamps
        result = _extract_duration_sec({
            "duration_sec": "bad",
            "started_at": "2024-01-15T10:00:00Z",
            "finished_at": "2024-01-15T10:00:10Z",
        })
        assert result == pytest.approx(10.0)

    def test_matches_patterns_empty_patterns_returns_false(self) -> None:
        # Empty patterns list → nothing matches → False
        assert _matches_patterns("task-a", []) is False
        assert _matches_patterns("*", []) is False

    def test_eval_suite_result_all_errors_failed_is_zero(self) -> None:
        # When every result is an error, failed == 0 but overall_pass is still False
        suite = EvalSuiteResult(
            name="err-suite",
            run_path=Path("."),
            results=[
                EvalResult(
                    task_id="t1",
                    judge_result=JudgeResult(verdict="error", overall_score=0.0, reasoning=""),
                    passed=False,
                ),
                EvalResult(
                    task_id="t2",
                    judge_result=JudgeResult(verdict="error", overall_score=0.0, reasoning=""),
                    passed=False,
                ),
            ],
            skipped=[],
        )
        assert suite.errors == 2
        assert suite.failed == 0
        assert suite.overall_pass is False

    def test_load_eval_spec_tasks_as_single_string(self, tmp_path: Path) -> None:
        # tasks: "task-a" (scalar) should be accepted and treated as single-element list
        eval_path = _write_eval(
            tmp_path,
            """\
name: scalar-suite
timeout_sec: 30
tasks: "task-a"
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        spec = load_eval_spec(eval_path)
        assert spec["tasks"] == ["task-a"]

    def test_run_eval_exclude_pattern_filters_tasks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: exclude-suite
tasks: ["*"]
exclude: ["task-b"]
judge:
  model: sonnet
  criteria:
    - "must be correct"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {
            "task_results": {
                "task-a": {"stdout_tail": "alpha"},
                "task-b": {"stdout_tail": "beta"},
            }
        }
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_evaluation",
            lambda *a, **kw: JudgeResult(verdict="pass", overall_score=1.0, reasoning=""),
        )

        suite = run_eval(eval_path, run_path)

        evaluated_ids = [r.task_id for r in suite.results]
        assert "task-a" in evaluated_ids
        assert "task-b" not in evaluated_ids

    def test_run_eval_explicit_timeout_forwarded_to_judge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Explicit timeout_sec in eval spec must be forwarded verbatim to _run_judge_evaluation.
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 90
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        monkeypatch.setattr(
            "maestro_cli.eval.load_run_manifest",
            lambda _: {"task_results": {"task-a": {"stdout_tail": "out"}}},
        )

        captured_timeouts: list[int] = []

        def _capture(
            task_id: str,
            judge: Any,
            stdout_tail: str,
            workdir: Path,
            **kwargs: Any,
        ) -> JudgeResult:
            captured_timeouts.append(kwargs.get("timeout_sec", -1))
            return JudgeResult(verdict="pass", overall_score=1.0, reasoning="")

        monkeypatch.setattr("maestro_cli.runners._run_judge_evaluation", _capture)

        run_eval(eval_path, run_path)
        assert captured_timeouts == [90]

    def test_run_eval_uses_judge_quorum_when_configured(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: quorum-suite
timeout_sec: 30
judge:
  model: sonnet
  quorum: 3
  quorum_strategy: majority
  criteria:
    - "ok"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        monkeypatch.setattr(
            "maestro_cli.eval.load_run_manifest",
            lambda _: {"task_results": {"task-a": {"stdout_tail": "out"}}},
        )

        calls: list[int] = []

        def _fake_single(
            task_id: str, judge: Any, stdout_tail: str, workdir: Path, **kwargs: Any
        ) -> JudgeResult:
            calls.append(len(calls) + 1)
            if len(calls) == 1:
                return JudgeResult(verdict="pass", overall_score=0.9, reasoning="one")
            if len(calls) == 2:
                return JudgeResult(verdict="fail", overall_score=0.2, reasoning="two")
            return JudgeResult(verdict="pass", overall_score=0.8, reasoning="three")

        monkeypatch.setattr("maestro_cli.runners._run_judge_evaluation", _fake_single)

        suite = run_eval(eval_path, run_path)

        assert calls == [1, 2, 3]
        assert suite.results[0].judge_result.verdict == "pass"
        assert suite.results[0].judge_result.overall_score == pytest.approx((0.9 + 0.2 + 0.8) / 3.0)

    def test_normalize_patterns_filters_whitespace_only_items(self) -> None:
        # Whitespace-only strings are stripped from normalized pattern lists.
        from maestro_cli.eval import _normalize_patterns

        result = _normalize_patterns(["task-a", "   ", "task-b", ""], "tasks")
        assert result == ["task-a", "task-b"]

    def test_collect_judge_warnings_unknown_preset_adds_warning(self) -> None:
        # _collect_judge_warnings warns for presets unknown to JUDGE_PRESETS.
        # (This path is unreachable via load_eval_spec since _to_judge_spec raises first,
        # but the warning branch is still present and should behave correctly when called directly.)
        from maestro_cli.eval import _collect_judge_warnings
        from maestro_cli.models import JudgeSpec

        judge = JudgeSpec(criteria=["ok"], model="sonnet", preset="totally-unknown-preset-xyz")
        warnings: list[str] = []
        _collect_judge_warnings(judge, "judge", warnings)
        assert any("totally-unknown-preset-xyz" in w for w in warnings)

    def test_warns_g_eval_haiku_in_override(self, tmp_path: Path) -> None:
        # An override that uses method: g_eval + model: haiku should emit a warning
        # at the overrides.<task-id> level.
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
judge:
  model: sonnet
  criteria:
    - "ok"
overrides:
  task-special:
    model: haiku
    method: g_eval
""",
        )
        spec = load_eval_spec(eval_path)
        assert any(
            "g_eval" in w and "haiku" in w and "overrides.task-special" in w
            for w in spec["validation_warnings"]
        )

    def test_missing_requested_task_ids_deduplicates(self) -> None:
        # Same plain task ID listed twice → only one entry in the missing list.
        missing = _missing_requested_task_ids(
            {"tasks": ["ghost", "ghost", "also-missing"]},
            [],
        )
        assert missing.count("ghost") == 1
        assert "also-missing" in missing

    def test_format_eval_json_empty_results(self, tmp_path: Path) -> None:
        # Suite with no results → overall_pass True, all counts zero, skipped list present.
        suite = EvalSuiteResult(
            name="empty",
            run_path=tmp_path,
            results=[],
            skipped=["t1", "t2"],
        )
        payload = json.loads(format_eval_json(suite))
        assert payload["passed"] == 0
        assert payload["failed"] == 0
        assert payload["errors"] == 0
        assert payload["overall_pass"] is True
        assert set(payload["skipped"]) == {"t1", "t2"}

    def test_load_eval_spec_null_tasks_falls_back_to_wildcard(self, tmp_path: Path) -> None:
        # tasks: null → _normalize_patterns returns [] → falls back to ["*"]
        eval_path = _write_eval(
            tmp_path,
            """\
name: null-tasks-suite
timeout_sec: 30
tasks: null
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        spec = load_eval_spec(eval_path)
        assert spec["tasks"] == ["*"]

    def test_format_eval_fail_verdict_shown_as_no_in_table(self, tmp_path: Path) -> None:
        # A failed result should appear with verdict "fail" and "no" in the PASSED column.
        suite = EvalSuiteResult(
            name="fail-suite",
            run_path=tmp_path,
            results=[
                EvalResult(
                    task_id="task-fail",
                    judge_result=JudgeResult(verdict="fail", overall_score=0.3, reasoning="bad"),
                    passed=False,
                )
            ],
            skipped=[],
        )
        text = format_eval(suite)
        assert "task-fail" in text
        assert "fail" in text
        assert "no" in text
        assert "0.30" in text

    def test_run_eval_non_numeric_cost_usd_coerced_to_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # cost_usd: "bad" in manifest → _coerce_float returns None → judge receives None.
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {"task_results": {"task-a": {"stdout_tail": "out", "cost_usd": "bad-value"}}}
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)

        captured_costs: list[float | None] = []

        def _capture(
            task_id: str, judge: Any, stdout_tail: str, workdir: Path, **kwargs: Any
        ) -> JudgeResult:
            captured_costs.append(kwargs.get("cost_usd"))
            return JudgeResult(verdict="pass", overall_score=1.0, reasoning="")

        monkeypatch.setattr("maestro_cli.runners._run_judge_evaluation", _capture)

        run_eval(eval_path, run_path)
        assert captured_costs == [None]


# ---------------------------------------------------------------------------
# Additional load_eval_spec tests — edge cases
# ---------------------------------------------------------------------------

class TestLoadEvalSpecAdditional:
    def test_rejects_directory_as_eval_path(self, tmp_path: Path) -> None:
        with pytest.raises(PlanValidationError, match="Eval file not found"):
            load_eval_spec(tmp_path)

    def test_rejects_missing_name_field(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        with pytest.raises(PlanValidationError, match="Eval name must be a non-empty string"):
            load_eval_spec(eval_path)

    def test_rejects_whitespace_only_name(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: "   "
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        with pytest.raises(PlanValidationError, match="Eval name must be a non-empty string"):
            load_eval_spec(eval_path)

    def test_valid_timeout_one(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 1
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        spec = load_eval_spec(eval_path)
        assert spec["timeout_sec"] == 1

    def test_large_timeout(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 3600
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        spec = load_eval_spec(eval_path)
        assert spec["timeout_sec"] == 3600

    def test_multiple_criteria(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: multi-criteria
timeout_sec: 30
judge:
  model: sonnet
  criteria:
    - "must be correct"
    - "must be concise"
    - "must have tests"
  pass_threshold: 0.8
""",
        )
        spec = load_eval_spec(eval_path)
        assert len(spec["judge"].criteria) == 3

    def test_multiple_overrides(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
judge:
  model: sonnet
  criteria:
    - "ok"
  pass_threshold: 0.5
overrides:
  task-a:
    pass_threshold: 0.9
  task-b:
    pass_threshold: 0.3
  task-c:
    model: opus
""",
        )
        spec = load_eval_spec(eval_path)
        assert len(spec["overrides"]) == 3
        assert spec["overrides"]["task-a"].pass_threshold == pytest.approx(0.9)
        assert spec["overrides"]["task-b"].pass_threshold == pytest.approx(0.3)
        assert spec["overrides"]["task-c"].model == "opus"

    def test_exclude_as_single_string(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
exclude: "setup-*"
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        spec = load_eval_spec(eval_path)
        assert spec["exclude"] == ["setup-*"]

    def test_exclude_null_treated_as_empty(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
exclude: null
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        spec = load_eval_spec(eval_path)
        assert spec["exclude"] == []

    def test_judge_with_method_field(self, tmp_path: Path) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
timeout_sec: 30
judge:
  model: sonnet
  method: g_eval
  criteria:
    - "correctness"
""",
        )
        spec = load_eval_spec(eval_path)
        assert spec["judge"].method == "g_eval"


# ---------------------------------------------------------------------------
# Additional run_eval tests
# ---------------------------------------------------------------------------

class TestRunEvalAdditional:
    def test_run_eval_all_tasks_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: all-pass
judge:
  model: sonnet
  criteria:
    - "ok"
  pass_threshold: 0.5
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {
            "task_results": {
                "t1": {"stdout_tail": "a"},
                "t2": {"stdout_tail": "b"},
                "t3": {"stdout_tail": "c"},
            }
        }
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_evaluation",
            lambda *a, **kw: JudgeResult(verdict="pass", overall_score=0.9, reasoning="good"),
        )

        suite = run_eval(eval_path, run_path)
        assert suite.passed == 3
        assert suite.failed == 0
        assert suite.errors == 0
        assert suite.overall_pass is True

    def test_run_eval_judge_exception_captured_as_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: exception-suite
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {"task_results": {"t1": {"stdout_tail": "output"}}}
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)

        def _raise(*args: Any, **kwargs: Any) -> JudgeResult:
            raise ValueError("something broke")

        monkeypatch.setattr("maestro_cli.runners._run_judge_evaluation", _raise)

        suite = run_eval(eval_path, run_path)
        assert len(suite.results) == 1
        assert suite.results[0].judge_result.verdict == "error"
        assert "something broke" in suite.results[0].judge_result.reasoning
        assert suite.errors == 1

    def test_run_eval_empty_manifest_task_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: empty-manifest
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {"task_results": {}}
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)
        monkeypatch.setattr(
            "maestro_cli.runners._run_judge_evaluation",
            lambda *a, **kw: JudgeResult(verdict="pass", overall_score=1.0, reasoning=""),
        )

        suite = run_eval(eval_path, run_path)
        assert suite.results == []
        assert suite.skipped == []

    def test_run_eval_with_cost_and_duration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: cost-suite
timeout_sec: 30
judge:
  model: sonnet
  criteria:
    - type: cost_under
      value: 2.0
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {
            "task_results": {
                "t1": {"stdout_tail": "ok", "cost_usd": 1.5, "duration_sec": 10.0},
            }
        }
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)

        captured: list[dict[str, Any]] = []

        def _capture(task_id: str, judge: Any, stdout_tail: str, workdir: Path, **kw: Any) -> JudgeResult:
            captured.append({"cost_usd": kw.get("cost_usd"), "duration_sec": kw.get("duration_sec")})
            return JudgeResult(verdict="pass", overall_score=1.0, reasoning="ok")

        monkeypatch.setattr("maestro_cli.runners._run_judge_evaluation", _capture)

        run_eval(eval_path, run_path)
        assert captured[0]["cost_usd"] == pytest.approx(1.5)
        assert captured[0]["duration_sec"] == pytest.approx(10.0)

    def test_run_eval_missing_stdout_tail_defaults_to_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        eval_path = _write_eval(
            tmp_path,
            """\
name: suite
judge:
  model: sonnet
  criteria:
    - "ok"
""",
        )
        run_path = tmp_path / "run"
        run_path.mkdir()

        manifest = {"task_results": {"t1": {}}}
        monkeypatch.setattr("maestro_cli.eval.load_run_manifest", lambda _: manifest)

        captured_tails: list[str] = []

        def _capture(task_id: str, judge: Any, stdout_tail: str, **kw: Any) -> JudgeResult:
            captured_tails.append(stdout_tail)
            return JudgeResult(verdict="pass", overall_score=1.0, reasoning="")

        monkeypatch.setattr("maestro_cli.runners._run_judge_evaluation", _capture)

        run_eval(eval_path, run_path)
        assert captured_tails == [""]


# ---------------------------------------------------------------------------
# Additional EvalSuiteResult tests
# ---------------------------------------------------------------------------

class TestEvalSuiteResultAdditional:
    def _make_result(self, task_id: str, verdict: str, score: float) -> EvalResult:
        return EvalResult(
            task_id=task_id,
            judge_result=JudgeResult(verdict=verdict, overall_score=score, reasoning=""),
            passed=(verdict == "pass"),
        )

    def test_mixed_verdicts(self) -> None:
        suite = EvalSuiteResult(
            name="mixed",
            run_path=Path("."),
            results=[
                self._make_result("t1", "pass", 0.9),
                self._make_result("t2", "fail", 0.3),
                self._make_result("t3", "error", 0.0),
                self._make_result("t4", "pass", 0.8),
            ],
            skipped=["t5"],
        )
        assert suite.passed == 2
        assert suite.failed == 1
        assert suite.errors == 1
        assert suite.overall_pass is False

    def test_single_pass_overall(self) -> None:
        suite = EvalSuiteResult(
            name="single",
            run_path=Path("."),
            results=[self._make_result("t1", "pass", 1.0)],
            skipped=[],
        )
        assert suite.overall_pass is True

    def test_skipped_only(self) -> None:
        suite = EvalSuiteResult(
            name="skipped",
            run_path=Path("."),
            results=[],
            skipped=["t1", "t2"],
        )
        assert suite.passed == 0
        assert suite.failed == 0
        assert suite.overall_pass is True


# ---------------------------------------------------------------------------
# Additional format_eval tests
# ---------------------------------------------------------------------------

class TestFormatEvalAdditional:
    def test_format_eval_table_alignment(self, tmp_path: Path) -> None:
        """Table columns should be properly aligned."""
        suite = EvalSuiteResult(
            name="align-test",
            run_path=tmp_path,
            results=[
                EvalResult(
                    task_id="short",
                    judge_result=JudgeResult(verdict="pass", overall_score=1.0, reasoning="ok"),
                    passed=True,
                ),
                EvalResult(
                    task_id="very-long-task-id-here",
                    judge_result=JudgeResult(verdict="fail", overall_score=0.2, reasoning="bad"),
                    passed=False,
                ),
            ],
            skipped=[],
        )
        text = format_eval(suite)
        lines = text.split("\n")
        # Verify header line and separator exist
        assert any("TASK" in line and "VERDICT" in line for line in lines)
        assert any("---" in line for line in lines)

    def test_format_eval_json_multiple_results(self, tmp_path: Path) -> None:
        suite = EvalSuiteResult(
            name="multi",
            run_path=tmp_path,
            results=[
                EvalResult(
                    task_id="t1",
                    judge_result=JudgeResult(verdict="pass", overall_score=0.95, reasoning="ok"),
                    passed=True,
                ),
                EvalResult(
                    task_id="t2",
                    judge_result=JudgeResult(verdict="fail", overall_score=0.3, reasoning="bad"),
                    passed=False,
                ),
                EvalResult(
                    task_id="t3",
                    judge_result=JudgeResult(verdict="error", overall_score=0.0, reasoning="crash"),
                    passed=False,
                ),
            ],
            skipped=["t4"],
        )
        payload = json.loads(format_eval_json(suite))
        assert payload["passed"] == 1
        assert payload["failed"] == 1
        assert payload["errors"] == 1
        assert payload["overall_pass"] is False
        assert len(payload["results"]) == 3
        assert payload["skipped"] == ["t4"]

    def test_format_eval_summary_line(self, tmp_path: Path) -> None:
        suite = EvalSuiteResult(
            name="summary-test",
            run_path=tmp_path,
            results=[
                EvalResult(
                    task_id="t1",
                    judge_result=JudgeResult(verdict="pass", overall_score=0.9, reasoning="ok"),
                    passed=True,
                ),
            ],
            skipped=[],
        )
        text = format_eval(suite)
        assert "passed=1" in text
        assert "failed=0" in text
        assert "PASS" in text

    def test_format_eval_overall_fail_shown(self, tmp_path: Path) -> None:
        suite = EvalSuiteResult(
            name="fail-test",
            run_path=tmp_path,
            results=[
                EvalResult(
                    task_id="t1",
                    judge_result=JudgeResult(verdict="fail", overall_score=0.2, reasoning="bad"),
                    passed=False,
                ),
            ],
            skipped=[],
        )
        text = format_eval(suite)
        assert "FAIL" in text

    def test_format_eval_json_run_path_as_string(self, tmp_path: Path) -> None:
        suite = EvalSuiteResult(
            name="path-test",
            run_path=tmp_path / "my-run",
            results=[],
            skipped=[],
        )
        payload = json.loads(format_eval_json(suite))
        assert isinstance(payload["run_path"], str)
        assert "my-run" in payload["run_path"]


# ---------------------------------------------------------------------------
# Additional helper function tests
# ---------------------------------------------------------------------------

class TestHelperFunctionsAdditional:
    def test_coerce_float_bool_input(self) -> None:
        # bool is a subclass of int in Python, float(True) == 1.0
        assert _coerce_float(True) == pytest.approx(1.0)
        assert _coerce_float(False) == pytest.approx(0.0)

    def test_coerce_float_negative(self) -> None:
        assert _coerce_float(-3.14) == pytest.approx(-3.14)

    def test_coerce_float_zero(self) -> None:
        assert _coerce_float(0) == pytest.approx(0.0)
        assert _coerce_float("0") == pytest.approx(0.0)

    def test_extract_duration_zero_explicit(self) -> None:
        result = _extract_duration_sec({"duration_sec": 0.0})
        assert result == pytest.approx(0.0)

    def test_parse_iso_with_timezone_offset(self) -> None:
        dt = _parse_iso("2024-06-15T14:30:00+05:00")
        assert dt is not None
        assert dt.hour == 14


# ---------------------------------------------------------------------------
# Multi-Dimensional Eval
# ---------------------------------------------------------------------------

def _make_eval_result(
    task_id: str,
    verdict: str = "pass",
    score: float = 0.85,
    passed: bool | None = None,
) -> EvalResult:
    """Helper to build an EvalResult with sensible defaults."""
    if passed is None:
        passed = verdict == "pass"
    return EvalResult(
        task_id=task_id,
        judge_result=JudgeResult(verdict=verdict, overall_score=score),
        passed=passed,
    )


class TestMultiDimensionalEval:
    """Tests for multi-dimensional eval: DimensionResult, EvalSuiteResult
    dimensions support, load_eval_spec dimensions parsing, and format output."""

    # -- DimensionResult dataclass -----------------------------------------

    def test_dimension_result_defaults(self) -> None:
        dim = DimensionResult(name="correctness", results=[], skipped=[])
        assert dim.name == "correctness"
        assert dim.results == []
        assert dim.skipped == []

    def test_dimension_result_passed_count(self) -> None:
        dim = DimensionResult(
            name="d",
            results=[
                _make_eval_result("t1", "pass"),
                _make_eval_result("t2", "fail", passed=False),
                _make_eval_result("t3", "pass"),
            ],
            skipped=[],
        )
        assert dim.passed == 2

    def test_dimension_result_failed_count(self) -> None:
        dim = DimensionResult(
            name="d",
            results=[
                _make_eval_result("t1", "pass"),
                _make_eval_result("t2", "fail", passed=False),
                _make_eval_result("t3", "fail", passed=False),
            ],
            skipped=[],
        )
        assert dim.failed == 2

    def test_dimension_result_errors_count(self) -> None:
        dim = DimensionResult(
            name="d",
            results=[
                _make_eval_result("t1", "error", score=0.0, passed=False),
                _make_eval_result("t2", "pass"),
            ],
            skipped=[],
        )
        assert dim.errors == 1

    def test_dimension_result_overall_pass_true(self) -> None:
        dim = DimensionResult(
            name="d",
            results=[
                _make_eval_result("t1", "pass"),
                _make_eval_result("t2", "pass"),
            ],
            skipped=[],
        )
        assert dim.overall_pass is True

    def test_dimension_result_overall_pass_false_on_failure(self) -> None:
        dim = DimensionResult(
            name="d",
            results=[
                _make_eval_result("t1", "pass"),
                _make_eval_result("t2", "fail", passed=False),
            ],
            skipped=[],
        )
        assert dim.overall_pass is False

    def test_dimension_result_overall_pass_false_on_error(self) -> None:
        dim = DimensionResult(
            name="d",
            results=[
                _make_eval_result("t1", "error", score=0.0, passed=False),
            ],
            skipped=[],
        )
        assert dim.overall_pass is False

    def test_dimension_result_avg_score(self) -> None:
        dim = DimensionResult(
            name="d",
            results=[
                _make_eval_result("t1", score=0.8),
                _make_eval_result("t2", score=0.6),
            ],
            skipped=[],
        )
        assert dim.avg_score == pytest.approx(0.7)

    def test_dimension_result_avg_score_empty(self) -> None:
        dim = DimensionResult(name="d", results=[], skipped=[])
        assert dim.avg_score == pytest.approx(0.0)

    def test_dimension_result_avg_score_ignores_none(self) -> None:
        dim = DimensionResult(
            name="d",
            results=[
                _make_eval_result("t1", score=0.9),
                EvalResult(
                    task_id="t2",
                    judge_result=JudgeResult(verdict="error", overall_score=None),  # type: ignore[arg-type]
                    passed=False,
                ),
            ],
            skipped=[],
        )
        assert dim.avg_score == pytest.approx(0.9)

    def test_dimension_result_mixed_pass_fail_error(self) -> None:
        dim = DimensionResult(
            name="mix",
            results=[
                _make_eval_result("t1", "pass", score=0.9),
                _make_eval_result("t2", "fail", score=0.3, passed=False),
                _make_eval_result("t3", "error", score=0.0, passed=False),
                _make_eval_result("t4", "pass", score=0.8),
            ],
            skipped=["t5"],
        )
        assert dim.passed == 2
        assert dim.failed == 1
        assert dim.errors == 1
        assert dim.overall_pass is False
        assert dim.avg_score == pytest.approx(0.5)

    def test_dimension_result_empty_results(self) -> None:
        dim = DimensionResult(name="empty", results=[], skipped=["t1"])
        assert dim.passed == 0
        assert dim.failed == 0
        assert dim.errors == 0
        assert dim.overall_pass is True
        assert dim.avg_score == pytest.approx(0.0)

    # -- EvalSuiteResult.dimensions ----------------------------------------

    def test_suite_result_dimensions_default_none(self) -> None:
        suite = EvalSuiteResult(
            name="s", run_path=Path("."), results=[], skipped=[],
        )
        assert suite.dimensions is None

    def test_suite_overall_pass_with_dimensions_all_pass(self) -> None:
        suite = EvalSuiteResult(
            name="s",
            run_path=Path("."),
            results=[_make_eval_result("t1", "fail", passed=False)],
            skipped=[],
            dimensions=[
                DimensionResult(name="a", results=[_make_eval_result("t1")], skipped=[]),
                DimensionResult(name="b", results=[_make_eval_result("t2")], skipped=[]),
            ],
        )
        # When dimensions are present, overall_pass defers to dimensions
        assert suite.overall_pass is True

    def test_suite_overall_pass_with_dimensions_one_fails(self) -> None:
        suite = EvalSuiteResult(
            name="s",
            run_path=Path("."),
            results=[_make_eval_result("t1")],
            skipped=[],
            dimensions=[
                DimensionResult(name="a", results=[_make_eval_result("t1")], skipped=[]),
                DimensionResult(
                    name="b",
                    results=[_make_eval_result("t2", "fail", passed=False)],
                    skipped=[],
                ),
            ],
        )
        assert suite.overall_pass is False

    def test_suite_overall_pass_without_dimensions_backward_compat(self) -> None:
        suite = EvalSuiteResult(
            name="s",
            run_path=Path("."),
            results=[_make_eval_result("t1")],
            skipped=[],
            dimensions=None,
        )
        assert suite.overall_pass is True

    def test_suite_overall_pass_without_dimensions_failure(self) -> None:
        suite = EvalSuiteResult(
            name="s",
            run_path=Path("."),
            results=[_make_eval_result("t1", "fail", passed=False)],
            skipped=[],
            dimensions=None,
        )
        assert suite.overall_pass is False

    # -- load_eval_spec: dimensions parsing --------------------------------

    def test_load_eval_spec_parses_dimensions(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: multi
tasks: ["*"]
judge:
  criteria: ["ok?"]
  pass_threshold: 0.7
dimensions:
  - name: correctness
    tasks: ["impl-*"]
  - name: security
    tasks: ["sec-*"]
""")
        spec = load_eval_spec(p)
        dims = spec["dimensions"]
        assert len(dims) == 2
        assert dims[0]["name"] == "correctness"
        assert dims[0]["tasks"] == ["impl-*"]
        assert dims[1]["name"] == "security"
        assert dims[1]["tasks"] == ["sec-*"]

    def test_load_eval_spec_dimension_inherits_top_level_judge(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: multi
tasks: ["*"]
judge:
  criteria: ["is it good?"]
  pass_threshold: 0.6
dimensions:
  - name: general
""")
        spec = load_eval_spec(p)
        dim_judge = spec["dimensions"][0]["judge"]
        assert dim_judge.pass_threshold == pytest.approx(0.6)
        assert dim_judge is spec["judge"]

    def test_load_eval_spec_dimension_own_judge(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: multi
tasks: ["*"]
judge:
  criteria: ["baseline"]
  pass_threshold: 0.5
dimensions:
  - name: strict
    judge:
      criteria: ["very strict check"]
      pass_threshold: 0.95
""")
        spec = load_eval_spec(p)
        dim_judge = spec["dimensions"][0]["judge"]
        assert dim_judge.pass_threshold == pytest.approx(0.95)
        assert dim_judge is not spec["judge"]

    def test_load_eval_spec_dimension_with_preset(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: multi
tasks: ["*"]
judge:
  criteria: ["fallback"]
  pass_threshold: 0.5
dimensions:
  - name: quality
    judge:
      preset: code_quality
""")
        spec = load_eval_spec(p)
        dim_judge = spec["dimensions"][0]["judge"]
        assert dim_judge.preset == "code_quality"

    def test_load_eval_spec_dimension_task_patterns(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: multi
tasks: ["*"]
judge:
  criteria: ["ok?"]
  pass_threshold: 0.5
dimensions:
  - name: perf
    tasks: ["perf-*", "bench-*"]
""")
        spec = load_eval_spec(p)
        assert spec["dimensions"][0]["tasks"] == ["perf-*", "bench-*"]

    def test_load_eval_spec_dimension_with_exclude(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: multi
tasks: ["*"]
judge:
  criteria: ["ok?"]
  pass_threshold: 0.5
dimensions:
  - name: filtered
    tasks: ["*"]
    exclude: ["slow-*"]
""")
        spec = load_eval_spec(p)
        assert spec["dimensions"][0]["exclude"] == ["slow-*"]

    def test_load_eval_spec_no_dimensions_backward_compat(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: basic
tasks: ["*"]
judge:
  criteria: ["ok?"]
  pass_threshold: 0.5
""")
        spec = load_eval_spec(p)
        assert spec["dimensions"] == []

    def test_load_eval_spec_error_dimensions_not_list(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: bad
tasks: ["*"]
judge:
  criteria: ["ok?"]
  pass_threshold: 0.5
dimensions: "not-a-list"
""")
        with pytest.raises(PlanValidationError, match="dimensions must be a list"):
            load_eval_spec(p)

    def test_load_eval_spec_error_dimension_not_dict(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: bad
tasks: ["*"]
judge:
  criteria: ["ok?"]
  pass_threshold: 0.5
dimensions:
  - "just-a-string"
""")
        with pytest.raises(PlanValidationError, match=r"dimensions\[0\] must be an object"):
            load_eval_spec(p)

    def test_load_eval_spec_error_dimension_missing_name(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: bad
tasks: ["*"]
judge:
  criteria: ["ok?"]
  pass_threshold: 0.5
dimensions:
  - tasks: ["*"]
""")
        with pytest.raises(PlanValidationError, match=r"dimensions\[0\]\.name is required"):
            load_eval_spec(p)

    def test_load_eval_spec_error_dimension_judge_not_dict(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: bad
tasks: ["*"]
judge:
  criteria: ["ok?"]
  pass_threshold: 0.5
dimensions:
  - name: broken
    judge: "not-a-dict"
""")
        with pytest.raises(PlanValidationError, match=r"dimensions\[0\]\.judge must be an object"):
            load_eval_spec(p)

    # -- format_eval with dimensions ---------------------------------------

    def test_format_eval_includes_dimension_breakdown(self) -> None:
        suite = EvalSuiteResult(
            name="suite",
            run_path=Path("/run"),
            results=[_make_eval_result("t1", "pass", 0.9)],
            skipped=[],
            dimensions=[
                DimensionResult(
                    name="correctness",
                    results=[_make_eval_result("t1", "pass", 0.9)],
                    skipped=[],
                ),
                DimensionResult(
                    name="security",
                    results=[_make_eval_result("t1", "fail", 0.3, passed=False)],
                    skipped=[],
                ),
            ],
        )
        text = format_eval(suite)
        assert "Dimensions:" in text
        assert "correctness: PASS" in text
        assert "security: FAIL" in text

    def test_format_eval_no_dimensions_backward_compat(self) -> None:
        suite = EvalSuiteResult(
            name="suite",
            run_path=Path("/run"),
            results=[_make_eval_result("t1")],
            skipped=[],
            dimensions=None,
        )
        text = format_eval(suite)
        assert "Dimensions:" not in text

    def test_format_eval_dimension_stats_in_output(self) -> None:
        suite = EvalSuiteResult(
            name="s",
            run_path=Path("/r"),
            results=[_make_eval_result("t1")],
            skipped=[],
            dimensions=[
                DimensionResult(
                    name="perf",
                    results=[
                        _make_eval_result("t1", "pass", 0.8),
                        _make_eval_result("t2", "fail", 0.2, passed=False),
                    ],
                    skipped=[],
                ),
            ],
        )
        text = format_eval(suite)
        assert "passed=1" in text
        assert "failed=1" in text
        assert "avg_score=0.50" in text

    # -- format_eval_json with dimensions ----------------------------------

    def test_format_eval_json_includes_dimensions(self) -> None:
        suite = EvalSuiteResult(
            name="suite",
            run_path=Path("/run"),
            results=[_make_eval_result("t1")],
            skipped=[],
            dimensions=[
                DimensionResult(
                    name="correctness",
                    results=[_make_eval_result("t1", "pass", 0.85)],
                    skipped=["t2"],
                ),
            ],
        )
        payload = json.loads(format_eval_json(suite))
        assert "dimensions" in payload
        assert len(payload["dimensions"]) == 1
        dim = payload["dimensions"][0]
        assert dim["name"] == "correctness"
        assert dim["passed"] == 1
        assert dim["failed"] == 0
        assert dim["errors"] == 0
        assert dim["overall_pass"] is True
        assert dim["skipped"] == ["t2"]
        assert dim["avg_score"] == pytest.approx(0.85)

    def test_format_eval_json_no_dimensions_omits_key(self) -> None:
        suite = EvalSuiteResult(
            name="suite",
            run_path=Path("/run"),
            results=[_make_eval_result("t1")],
            skipped=[],
            dimensions=None,
        )
        payload = json.loads(format_eval_json(suite))
        assert "dimensions" not in payload

    def test_format_eval_json_dimensions_multiple(self) -> None:
        suite = EvalSuiteResult(
            name="s",
            run_path=Path("."),
            results=[],
            skipped=[],
            dimensions=[
                DimensionResult(
                    name="a",
                    results=[_make_eval_result("t1", "pass", 1.0)],
                    skipped=[],
                ),
                DimensionResult(
                    name="b",
                    results=[_make_eval_result("t2", "error", 0.0, passed=False)],
                    skipped=[],
                ),
            ],
        )
        payload = json.loads(format_eval_json(suite))
        assert len(payload["dimensions"]) == 2
        assert payload["dimensions"][0]["overall_pass"] is True
        assert payload["dimensions"][1]["overall_pass"] is False
        assert payload["overall_pass"] is False

    # -- Multiple dimensions with different judges -------------------------

    def test_load_eval_spec_multiple_dimensions_different_judges(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: multi-judge
tasks: ["*"]
judge:
  criteria: ["baseline"]
  pass_threshold: 0.5
dimensions:
  - name: strict
    judge:
      criteria: ["very strict"]
      pass_threshold: 0.95
  - name: lenient
    judge:
      criteria: ["lenient check"]
      pass_threshold: 0.3
""")
        spec = load_eval_spec(p)
        assert len(spec["dimensions"]) == 2
        assert spec["dimensions"][0]["judge"].pass_threshold == pytest.approx(0.95)
        assert spec["dimensions"][1]["judge"].pass_threshold == pytest.approx(0.3)
        assert spec["dimensions"][0]["judge"] is not spec["dimensions"][1]["judge"]

    # -- Dimension default task patterns -----------------------------------

    def test_load_eval_spec_dimension_default_tasks_wildcard(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: multi
tasks: ["*"]
judge:
  criteria: ["ok?"]
  pass_threshold: 0.5
dimensions:
  - name: all-tasks
""")
        spec = load_eval_spec(p)
        assert spec["dimensions"][0]["tasks"] == ["*"]

    def test_load_eval_spec_dimension_empty_exclude(self, tmp_path: Path) -> None:
        p = _write_eval(tmp_path, """\
name: multi
tasks: ["*"]
judge:
  criteria: ["ok?"]
  pass_threshold: 0.5
dimensions:
  - name: no-exclude
    tasks: ["impl-*"]
""")
        spec = load_eval_spec(p)
        assert spec["dimensions"][0]["exclude"] == []

    # -- Suite overall_pass with empty dimension ---------------------------

    def test_suite_overall_pass_with_empty_dimension(self) -> None:
        suite = EvalSuiteResult(
            name="s",
            run_path=Path("."),
            results=[],
            skipped=[],
            dimensions=[
                DimensionResult(name="empty", results=[], skipped=[]),
            ],
        )
        assert suite.overall_pass is True
