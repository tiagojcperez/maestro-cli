from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.diff import (
    RunDiff,
    TaskDiff,
    compare_manifests,
    compare_runs,
    diff_runs,
    format_diff,
    format_diff_json,
    load_run_manifest,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_manifest(run_dir: Path, data: dict[str, Any]) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    return manifest_path


def _minimal_manifest(
    *,
    run_id: str = "run-001",
    plan_name: str = "my-plan",
    success: bool = True,
    started_at: str = "2025-01-01T00:00:00+00:00",
    finished_at: str = "2025-01-01T00:01:00+00:00",
    task_results: dict[str, Any] | None = None,
    total_cost_usd: float | None = None,
    total_tokens: int | None = None,
) -> dict[str, Any]:
    data: dict[str, Any] = {
        "run_id": run_id,
        "plan_name": plan_name,
        "success": success,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    if task_results is not None:
        data["task_results"] = task_results
    if total_cost_usd is not None:
        data["total_cost_usd"] = total_cost_usd
    if total_tokens is not None:
        data["total_tokens"] = total_tokens
    return data


def _task_result(
    status: str = "success",
    duration_sec: float = 1.0,
    cost_usd: float | None = None,
    tokens: int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "status": status,
        "duration_sec": duration_sec,
    }
    if cost_usd is not None:
        result["cost_usd"] = cost_usd
    if tokens is not None:
        result["token_usage"] = {"total_tokens": tokens}
    return result


# ---------------------------------------------------------------------------
# load_run_manifest
# ---------------------------------------------------------------------------


class TestLoadRunManifest:
    def test_valid_manifest_loaded(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        _write_manifest(run_dir, {"run_id": "r1", "plan_name": "p"})

        manifest = load_run_manifest(run_dir)

        assert manifest["run_id"] == "r1"
        assert manifest["plan_name"] == "p"

    def test_missing_manifest_raises_file_not_found(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "nonexistent"
        run_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="run_manifest.json"):
            load_run_manifest(run_dir)

    def test_invalid_json_raises_value_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text("not json {{{", encoding="utf-8")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_run_manifest(run_dir)

    def test_non_object_json_raises_value_error(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        (run_dir / "run_manifest.json").write_text("[1, 2, 3]", encoding="utf-8")

        with pytest.raises(ValueError, match="expected JSON object"):
            load_run_manifest(run_dir)


# ---------------------------------------------------------------------------
# compare_manifests — basic properties
# ---------------------------------------------------------------------------


class TestCompareManifestsBasic:
    def test_run_ids_extracted(self) -> None:
        a = _minimal_manifest(run_id="aaa")
        b = _minimal_manifest(run_id="bbb")

        diff = compare_manifests(a, b)

        assert diff.run_id_a == "aaa"
        assert diff.run_id_b == "bbb"

    def test_plan_names_extracted(self) -> None:
        a = _minimal_manifest(plan_name="plan-alpha")
        b = _minimal_manifest(plan_name="plan-beta")

        diff = compare_manifests(a, b)

        assert diff.plan_name_a == "plan-alpha"
        assert diff.plan_name_b == "plan-beta"

    def test_success_flags_extracted(self) -> None:
        a = _minimal_manifest(success=True)
        b = _minimal_manifest(success=False)

        diff = compare_manifests(a, b)

        assert diff.success_a is True
        assert diff.success_b is False

    def test_duration_computed_from_timestamps(self) -> None:
        a = _minimal_manifest(
            started_at="2025-01-01T00:00:00+00:00",
            finished_at="2025-01-01T00:01:00+00:00",
        )
        b = _minimal_manifest(
            started_at="2025-01-01T00:00:00+00:00",
            finished_at="2025-01-01T00:02:30+00:00",
        )

        diff = compare_manifests(a, b)

        assert diff.duration_a == pytest.approx(60.0)
        assert diff.duration_b == pytest.approx(150.0)

    def test_total_cost_extracted_from_manifest(self) -> None:
        a = _minimal_manifest(total_cost_usd=1.5)
        b = _minimal_manifest(total_cost_usd=2.0)

        diff = compare_manifests(a, b)

        assert diff.cost_a == pytest.approx(1.5)
        assert diff.cost_b == pytest.approx(2.0)

    def test_cost_none_when_absent(self) -> None:
        a = _minimal_manifest()
        b = _minimal_manifest()

        diff = compare_manifests(a, b)

        assert diff.cost_a is None
        assert diff.cost_b is None

    def test_total_tokens_extracted_from_manifest(self) -> None:
        a = _minimal_manifest(total_tokens=1000)
        b = _minimal_manifest(total_tokens=2000)

        diff = compare_manifests(a, b)

        assert diff.tokens_a == 1000
        assert diff.tokens_b == 2000

    def test_fallback_run_ids_used_when_missing(self) -> None:
        a: dict[str, Any] = {"plan_name": "p", "success": True}
        b: dict[str, Any] = {"plan_name": "p", "success": True}

        diff = compare_manifests(a, b, fallback_run_id_a="fallback-a", fallback_run_id_b="fallback-b")

        assert diff.run_id_a == "fallback-a"
        assert diff.run_id_b == "fallback-b"


# ---------------------------------------------------------------------------
# compare_manifests — task-level diffs
# ---------------------------------------------------------------------------


class TestCompareManifestsTaskDiffs:
    def test_task_present_in_both_runs(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success", 1.0)})
        b = _minimal_manifest(task_results={"t1": _task_result("success", 2.0)})

        diff = compare_manifests(a, b)

        assert len(diff.task_diffs) == 1
        td = diff.task_diffs[0]
        assert td.task_id == "t1"
        assert td.status_a == "success"
        assert td.status_b == "success"
        assert td.duration_a == pytest.approx(1.0)
        assert td.duration_b == pytest.approx(2.0)

    def test_regression_detected(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success")})
        b = _minimal_manifest(task_results={"t1": _task_result("failed")})

        diff = compare_manifests(a, b)

        assert "t1" in diff.regressions
        assert "t1" not in diff.fixes

    def test_fix_detected(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("failed")})
        b = _minimal_manifest(task_results={"t1": _task_result("success")})

        diff = compare_manifests(a, b)

        assert "t1" in diff.fixes
        assert "t1" not in diff.regressions

    def test_added_tasks_detected(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result()})
        b = _minimal_manifest(task_results={"t1": _task_result(), "t2": _task_result()})

        diff = compare_manifests(a, b)

        assert "t2" in diff.added_tasks
        assert "t1" not in diff.added_tasks

    def test_removed_tasks_detected(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result(), "t2": _task_result()})
        b = _minimal_manifest(task_results={"t1": _task_result()})

        diff = compare_manifests(a, b)

        assert "t2" in diff.removed_tasks
        assert "t1" not in diff.removed_tasks

    def test_task_only_in_a_has_none_status_b(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success")})
        b = _minimal_manifest(task_results={})

        diff = compare_manifests(a, b)

        td = next(d for d in diff.task_diffs if d.task_id == "t1")
        assert td.status_a == "success"
        assert td.status_b is None

    def test_task_only_in_b_has_none_status_a(self) -> None:
        a = _minimal_manifest(task_results={})
        b = _minimal_manifest(task_results={"t1": _task_result("success")})

        diff = compare_manifests(a, b)

        td = next(d for d in diff.task_diffs if d.task_id == "t1")
        assert td.status_a is None
        assert td.status_b == "success"

    def test_task_cost_and_tokens_extracted(self) -> None:
        a = _minimal_manifest(
            task_results={"t1": _task_result("success", cost_usd=0.05, tokens=100)}
        )
        b = _minimal_manifest(
            task_results={"t1": _task_result("success", cost_usd=0.10, tokens=200)}
        )

        diff = compare_manifests(a, b)

        td = diff.task_diffs[0]
        assert td.cost_a == pytest.approx(0.05)
        assert td.cost_b == pytest.approx(0.10)
        assert td.tokens_a == 100
        assert td.tokens_b == 200

    def test_empty_both_runs_no_task_diffs(self) -> None:
        a = _minimal_manifest()
        b = _minimal_manifest()

        diff = compare_manifests(a, b)

        assert diff.task_diffs == []
        assert diff.regressions == []
        assert diff.fixes == []
        assert diff.added_tasks == []
        assert diff.removed_tasks == []


# ---------------------------------------------------------------------------
# diff_runs and compare_runs (file-based)
# ---------------------------------------------------------------------------


class TestDiffRunsFileBased:
    def test_diff_runs_loads_manifests_and_returns_run_diff(self, tmp_path: Path) -> None:
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        _write_manifest(run_a, _minimal_manifest(run_id="aaa", success=True))
        _write_manifest(run_b, _minimal_manifest(run_id="bbb", success=False))

        result = diff_runs(run_a, run_b)

        assert isinstance(result, RunDiff)
        assert result.run_id_a == "aaa"
        assert result.run_id_b == "bbb"
        assert result.success_a is True
        assert result.success_b is False

    def test_diff_runs_accepts_string_paths(self, tmp_path: Path) -> None:
        run_a = tmp_path / "run_a"
        run_b = tmp_path / "run_b"
        _write_manifest(run_a, _minimal_manifest())
        _write_manifest(run_b, _minimal_manifest())

        result = diff_runs(str(run_a), str(run_b))

        assert isinstance(result, RunDiff)

    def test_compare_runs_uses_dir_name_as_fallback_run_id(self, tmp_path: Path) -> None:
        run_a = tmp_path / "20250101_plan-a"
        run_b = tmp_path / "20250102_plan-b"
        _write_manifest(run_a, {"plan_name": "p", "success": True})
        _write_manifest(run_b, {"plan_name": "p", "success": True})

        result = compare_runs(run_a, run_b)

        assert result.run_id_a == "20250101_plan-a"
        assert result.run_id_b == "20250102_plan-b"

    def test_diff_runs_raises_on_missing_manifest(self, tmp_path: Path) -> None:
        run_a = tmp_path / "run_a"
        run_a.mkdir()
        run_b = tmp_path / "run_b"
        _write_manifest(run_b, _minimal_manifest())

        with pytest.raises(FileNotFoundError):
            diff_runs(run_a, run_b)


# ---------------------------------------------------------------------------
# format_diff
# ---------------------------------------------------------------------------


class TestFormatDiff:
    def _make_diff(self, **kwargs: object) -> RunDiff:
        defaults: dict[str, object] = {
            "run_id_a": "aaa",
            "run_id_b": "bbb",
            "plan_name_a": "plan",
            "plan_name_b": "plan",
            "success_a": True,
            "success_b": True,
            "duration_a": 10.0,
            "duration_b": 20.0,
            "cost_a": 0.5,
            "cost_b": 1.0,
            "tokens_a": 1000,
            "tokens_b": 2000,
            "task_diffs": [],
            "added_tasks": [],
            "removed_tasks": [],
            "regressions": [],
            "fixes": [],
        }
        defaults.update(kwargs)
        return RunDiff(**defaults)  # type: ignore[arg-type]

    def test_format_diff_contains_run_ids(self) -> None:
        diff = self._make_diff()
        output = format_diff(diff)

        assert "aaa" in output
        assert "bbb" in output

    def test_format_diff_contains_success_status(self) -> None:
        diff = self._make_diff(success_a=True, success_b=False)
        output = format_diff(diff)

        assert "True" in output or "true" in output.lower()

    def test_format_diff_shows_regressions(self) -> None:
        diff = self._make_diff(regressions=["task-a", "task-b"])
        output = format_diff(diff)

        assert "task-a" in output
        assert "task-b" in output
        assert "Regression" in output or "regression" in output.lower()

    def test_format_diff_shows_fixes(self) -> None:
        diff = self._make_diff(fixes=["task-c"])
        output = format_diff(diff)

        assert "task-c" in output
        assert "Fix" in output or "fix" in output.lower()

    def test_format_diff_shows_added_removed(self) -> None:
        diff = self._make_diff(added_tasks=["new-task"], removed_tasks=["old-task"])
        output = format_diff(diff)

        assert "new-task" in output
        assert "old-task" in output

    def test_format_diff_omits_empty_sections(self) -> None:
        diff = self._make_diff()
        output = format_diff(diff)

        assert "Regressions:" not in output
        assert "Fixes:" not in output
        assert "Added tasks:" not in output
        assert "Removed tasks:" not in output

    def test_format_diff_handles_none_cost(self) -> None:
        diff = self._make_diff(cost_a=None, cost_b=None)
        output = format_diff(diff)

        assert "n/a" in output


# ---------------------------------------------------------------------------
# format_diff_json
# ---------------------------------------------------------------------------


class TestFormatDiffJson:
    def test_format_diff_json_is_valid_json(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success")})
        b = _minimal_manifest(task_results={"t1": _task_result("failed")})
        diff = compare_manifests(a, b)

        output = format_diff_json(diff)
        parsed = json.loads(output)

        assert isinstance(parsed, dict)

    def test_format_diff_json_contains_task_diffs(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success")})
        b = _minimal_manifest(task_results={"t1": _task_result("failed")})
        diff = compare_manifests(a, b)

        parsed = json.loads(format_diff_json(diff))

        assert "task_diffs" in parsed
        assert len(parsed["task_diffs"]) == 1
        assert parsed["task_diffs"][0]["task_id"] == "t1"

    def test_format_diff_json_contains_regressions(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success")})
        b = _minimal_manifest(task_results={"t1": _task_result("failed")})
        diff = compare_manifests(a, b)

        parsed = json.loads(format_diff_json(diff))

        assert "t1" in parsed["regressions"]

    def test_format_diff_json_contains_fixes(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("failed")})
        b = _minimal_manifest(task_results={"t1": _task_result("success")})
        diff = compare_manifests(a, b)

        parsed = json.loads(format_diff_json(diff))

        assert "t1" in parsed["fixes"]


# ---------------------------------------------------------------------------
# Internal helper imports for additional coverage
# ---------------------------------------------------------------------------

from maestro_cli.diff import (
    _coerce_float,
    _coerce_int,
    _duration_from_timestamps,
    _format_money_delta,
    _format_number_delta,
    _parse_iso,
    _run_cost,
    _run_duration,
    _run_success,
    _run_tokens,
    _task_cost,
    _task_duration,
    _task_results,
    _task_status,
    _task_tokens,
)


# ---------------------------------------------------------------------------
# _coerce_float / _coerce_int (diff module versions)
# ---------------------------------------------------------------------------


class TestDiffCoerceFloat:
    def test_none(self) -> None:
        assert _coerce_float(None) is None

    def test_valid_float(self) -> None:
        assert _coerce_float(3.14) == pytest.approx(3.14)

    def test_string_number(self) -> None:
        assert _coerce_float("2.5") == pytest.approx(2.5)

    def test_invalid(self) -> None:
        assert _coerce_float("abc") is None

    def test_dict_returns_none(self) -> None:
        assert _coerce_float({"a": 1}) is None


class TestDiffCoerceInt:
    def test_none(self) -> None:
        assert _coerce_int(None) is None

    def test_valid_int(self) -> None:
        assert _coerce_int(42) == 42

    def test_string_int(self) -> None:
        assert _coerce_int("99") == 99

    def test_invalid(self) -> None:
        assert _coerce_int("nope") is None


# ---------------------------------------------------------------------------
# _parse_iso / _duration_from_timestamps (diff module versions)
# ---------------------------------------------------------------------------


class TestDiffParseIso:
    def test_valid(self) -> None:
        assert _parse_iso("2025-01-01T00:00:00+00:00") is not None

    def test_z_suffix(self) -> None:
        assert _parse_iso("2025-01-01T00:00:00Z") is not None

    def test_none_input(self) -> None:
        assert _parse_iso(None) is None

    def test_empty_string(self) -> None:
        assert _parse_iso("") is None

    def test_non_string(self) -> None:
        assert _parse_iso(42) is None

    def test_garbage(self) -> None:
        assert _parse_iso("not-a-date") is None


class TestDiffDurationFromTimestamps:
    def test_valid_pair(self) -> None:
        result = _duration_from_timestamps(
            "2025-01-01T00:00:00+00:00", "2025-01-01T00:01:00+00:00"
        )
        assert result == pytest.approx(60.0)

    def test_reversed_returns_zero(self) -> None:
        result = _duration_from_timestamps(
            "2025-01-01T00:01:00+00:00", "2025-01-01T00:00:00+00:00"
        )
        assert result == 0.0

    def test_missing_start(self) -> None:
        assert _duration_from_timestamps(None, "2025-01-01T00:00:00+00:00") is None

    def test_missing_end(self) -> None:
        assert _duration_from_timestamps("2025-01-01T00:00:00+00:00", None) is None


# ---------------------------------------------------------------------------
# _task_results
# ---------------------------------------------------------------------------


class TestTaskResults:
    def test_valid_extraction(self) -> None:
        manifest: dict[str, Any] = {
            "task_results": {
                "t1": {"status": "success"},
                "t2": {"status": "failed"},
            }
        }
        result = _task_results(manifest)
        assert set(result.keys()) == {"t1", "t2"}

    def test_missing_task_results(self) -> None:
        assert _task_results({}) == {}

    def test_non_dict_task_results(self) -> None:
        assert _task_results({"task_results": [1, 2, 3]}) == {}

    def test_filters_non_dict_entries(self) -> None:
        manifest: dict[str, Any] = {
            "task_results": {
                "good": {"status": "success"},
                "bad": "not-a-dict",
                "also_bad": 42,
            }
        }
        result = _task_results(manifest)
        assert list(result.keys()) == ["good"]


# ---------------------------------------------------------------------------
# _task_status
# ---------------------------------------------------------------------------


class TestTaskStatus:
    def test_valid_statuses(self) -> None:
        for status in ("success", "failed", "soft_failed", "skipped", "dry_run"):
            assert _task_status({"status": status}) == status

    def test_none_task_result(self) -> None:
        assert _task_status(None) is None

    def test_invalid_status(self) -> None:
        assert _task_status({"status": "running"}) is None

    def test_missing_status_field(self) -> None:
        assert _task_status({"duration_sec": 1.0}) is None


# ---------------------------------------------------------------------------
# _task_duration
# ---------------------------------------------------------------------------


class TestTaskDuration:
    def test_duration_sec_field(self) -> None:
        assert _task_duration({"duration_sec": 5.5}) == pytest.approx(5.5)

    def test_falls_back_to_timestamps(self) -> None:
        result = _task_duration({
            "started_at": "2025-01-01T00:00:00+00:00",
            "finished_at": "2025-01-01T00:00:30+00:00",
        })
        assert result == pytest.approx(30.0)

    def test_none_task_result(self) -> None:
        assert _task_duration(None) is None

    def test_no_duration_data(self) -> None:
        assert _task_duration({"status": "success"}) is None


# ---------------------------------------------------------------------------
# _task_cost / _task_tokens
# ---------------------------------------------------------------------------


class TestTaskCost:
    def test_with_cost(self) -> None:
        assert _task_cost({"cost_usd": 0.42}) == pytest.approx(0.42)

    def test_without_cost(self) -> None:
        assert _task_cost({"status": "success"}) is None

    def test_none_task(self) -> None:
        assert _task_cost(None) is None


class TestTaskTokens:
    def test_total_tokens(self) -> None:
        result = _task_tokens({"token_usage": {"total_tokens": 5000}})
        assert result == 5000

    def test_component_tokens(self) -> None:
        result = _task_tokens({
            "token_usage": {
                "input_tokens": 100,
                "cached_tokens": 50,
                "output_tokens": 200,
            }
        })
        assert result == 350

    def test_zero_tokens(self) -> None:
        result = _task_tokens({
            "token_usage": {
                "input_tokens": 0,
                "cached_tokens": 0,
                "output_tokens": 0,
            }
        })
        assert result is None

    def test_no_token_usage(self) -> None:
        assert _task_tokens({"status": "success"}) is None

    def test_none_task(self) -> None:
        assert _task_tokens(None) is None

    def test_non_dict_token_usage(self) -> None:
        assert _task_tokens({"token_usage": "bad"}) is None


# ---------------------------------------------------------------------------
# _run_success
# ---------------------------------------------------------------------------


class TestRunSuccess:
    def test_manifest_bool_takes_priority(self) -> None:
        assert _run_success({"success": True}, {}) is True
        assert _run_success({"success": False}, {}) is False

    def test_inferred_from_tasks_all_success(self) -> None:
        tasks = {"t1": {"status": "success"}, "t2": {"status": "success"}}
        assert _run_success({}, tasks) is True

    def test_inferred_from_tasks_with_failure(self) -> None:
        tasks = {"t1": {"status": "success"}, "t2": {"status": "failed"}}
        assert _run_success({}, tasks) is False

    def test_empty_tasks_returns_false(self) -> None:
        assert _run_success({}, {}) is False


# ---------------------------------------------------------------------------
# _run_duration / _run_cost / _run_tokens
# ---------------------------------------------------------------------------


class TestDiffRunDuration:
    def test_from_timestamps(self) -> None:
        manifest: dict[str, Any] = {
            "started_at": "2025-01-01T00:00:00+00:00",
            "finished_at": "2025-01-01T00:02:00+00:00",
        }
        assert _run_duration(manifest, {}) == pytest.approx(120.0)

    def test_sum_task_durations(self) -> None:
        tasks = {"t1": {"duration_sec": 10.0}, "t2": {"duration_sec": 20.0}}
        assert _run_duration({}, tasks) == pytest.approx(30.0)

    def test_zero_when_no_data(self) -> None:
        assert _run_duration({}, {}) == 0.0

    def test_skips_none_durations(self) -> None:
        tasks: dict[str, dict[str, Any]] = {
            "t1": {"duration_sec": 10.0},
            "t2": {"status": "success"},  # no duration
        }
        assert _run_duration({}, tasks) == pytest.approx(10.0)


class TestDiffRunCost:
    def test_from_manifest(self) -> None:
        manifest: dict[str, Any] = {"total_cost_usd": 5.0}
        assert _run_cost(manifest, {}) == pytest.approx(5.0)

    def test_sum_task_costs(self) -> None:
        tasks = {"t1": {"cost_usd": 1.0}, "t2": {"cost_usd": 2.0}}
        assert _run_cost({}, tasks) == pytest.approx(3.0)

    def test_none_when_no_costs(self) -> None:
        assert _run_cost({}, {"t1": {"status": "success"}}) is None

    def test_partial_costs_summed(self) -> None:
        tasks = {"t1": {"cost_usd": 1.5}, "t2": {"status": "success"}}
        assert _run_cost({}, tasks) == pytest.approx(1.5)


class TestDiffRunTokens:
    def test_from_manifest(self) -> None:
        manifest: dict[str, Any] = {"total_tokens": 8000}
        assert _run_tokens(manifest, {}) == 8000

    def test_sum_task_tokens(self) -> None:
        tasks = {
            "t1": {"token_usage": {"total_tokens": 100}},
            "t2": {"token_usage": {"total_tokens": 200}},
        }
        assert _run_tokens({}, tasks) == 300

    def test_none_when_no_tokens(self) -> None:
        assert _run_tokens({}, {"t1": {"status": "success"}}) is None


# ---------------------------------------------------------------------------
# _format_money_delta / _format_number_delta
# ---------------------------------------------------------------------------


class TestFormatMoneyDelta:
    def test_positive_delta(self) -> None:
        assert _format_money_delta(1.0, 2.0) == "+$1.0000"

    def test_negative_delta(self) -> None:
        assert _format_money_delta(2.0, 1.0) == "$-1.0000"

    def test_zero_delta(self) -> None:
        assert _format_money_delta(1.0, 1.0) == "$0.0000"

    def test_none_a(self) -> None:
        assert _format_money_delta(None, 1.0) == "n/a"

    def test_none_b(self) -> None:
        assert _format_money_delta(1.0, None) == "n/a"

    def test_both_none(self) -> None:
        assert _format_money_delta(None, None) == "n/a"


class TestFormatNumberDelta:
    def test_positive_float_delta(self) -> None:
        result = _format_number_delta(1.0, 3.5)
        assert result == "+2.50"

    def test_negative_float_delta(self) -> None:
        result = _format_number_delta(5.0, 2.0)
        assert result == "-3.00"

    def test_integer_delta(self) -> None:
        result = _format_number_delta(100, 150)
        assert result == "+50"

    def test_zero_delta_int(self) -> None:
        result = _format_number_delta(5, 5)
        assert result == "0"

    def test_none_a(self) -> None:
        assert _format_number_delta(None, 1.0) == "n/a"

    def test_none_b(self) -> None:
        assert _format_number_delta(1.0, None) == "n/a"


# ---------------------------------------------------------------------------
# compare_manifests — additional scenarios
# ---------------------------------------------------------------------------


class TestCompareManifestsAdditional:
    def test_both_success_no_regressions_or_fixes(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success")})
        b = _minimal_manifest(task_results={"t1": _task_result("success")})
        diff = compare_manifests(a, b)
        assert diff.regressions == []
        assert diff.fixes == []

    def test_both_failed_no_regressions_or_fixes(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("failed")})
        b = _minimal_manifest(task_results={"t1": _task_result("failed")})
        diff = compare_manifests(a, b)
        assert diff.regressions == []
        assert diff.fixes == []

    def test_soft_failed_to_failed_not_regression(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("soft_failed")})
        b = _minimal_manifest(task_results={"t1": _task_result("failed")})
        diff = compare_manifests(a, b)
        assert diff.regressions == []

    def test_skipped_to_success_not_fix(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("skipped")})
        b = _minimal_manifest(task_results={"t1": _task_result("success")})
        diff = compare_manifests(a, b)
        assert diff.fixes == []

    def test_multiple_regressions_and_fixes(self) -> None:
        a = _minimal_manifest(task_results={
            "t1": _task_result("success"),
            "t2": _task_result("failed"),
            "t3": _task_result("success"),
        })
        b = _minimal_manifest(task_results={
            "t1": _task_result("failed"),
            "t2": _task_result("success"),
            "t3": _task_result("failed"),
        })
        diff = compare_manifests(a, b)
        assert sorted(diff.regressions) == ["t1", "t3"]
        assert diff.fixes == ["t2"]

    def test_cost_delta_increase(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success", cost_usd=0.10)})
        b = _minimal_manifest(task_results={"t1": _task_result("success", cost_usd=0.50)})
        diff = compare_manifests(a, b)
        td = diff.task_diffs[0]
        assert td.cost_a == pytest.approx(0.10)
        assert td.cost_b == pytest.approx(0.50)

    def test_cost_delta_decrease(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success", cost_usd=0.50)})
        b = _minimal_manifest(task_results={"t1": _task_result("success", cost_usd=0.10)})
        diff = compare_manifests(a, b)
        td = diff.task_diffs[0]
        assert td.cost_b < td.cost_a  # type: ignore[operator]

    def test_cost_none_in_one_side(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success", cost_usd=0.10)})
        b = _minimal_manifest(task_results={"t1": _task_result("success")})
        diff = compare_manifests(a, b)
        td = diff.task_diffs[0]
        assert td.cost_a == pytest.approx(0.10)
        assert td.cost_b is None

    def test_token_delta(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success", tokens=1000)})
        b = _minimal_manifest(task_results={"t1": _task_result("success", tokens=5000)})
        diff = compare_manifests(a, b)
        td = diff.task_diffs[0]
        assert td.tokens_a == 1000
        assert td.tokens_b == 5000

    def test_duration_delta(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success", 10.0)})
        b = _minimal_manifest(task_results={"t1": _task_result("success", 25.0)})
        diff = compare_manifests(a, b)
        td = diff.task_diffs[0]
        assert td.duration_a == pytest.approx(10.0)
        assert td.duration_b == pytest.approx(25.0)

    def test_task_diffs_sorted_by_id(self) -> None:
        a = _minimal_manifest(task_results={
            "z-task": _task_result("success"),
            "a-task": _task_result("success"),
            "m-task": _task_result("success"),
        })
        b = _minimal_manifest(task_results={
            "z-task": _task_result("success"),
            "a-task": _task_result("success"),
            "m-task": _task_result("success"),
        })
        diff = compare_manifests(a, b)
        ids = [td.task_id for td in diff.task_diffs]
        assert ids == sorted(ids)

    def test_single_task_run(self) -> None:
        a = _minimal_manifest(task_results={"only": _task_result("success", 5.0, 0.01, 100)})
        b = _minimal_manifest(task_results={"only": _task_result("failed", 10.0, 0.05, 500)})
        diff = compare_manifests(a, b)
        assert len(diff.task_diffs) == 1
        assert diff.regressions == ["only"]

    def test_success_inferred_when_no_manifest_bool(self) -> None:
        a: dict[str, Any] = {
            "plan_name": "p",
            "started_at": "2025-01-01T00:00:00+00:00",
            "finished_at": "2025-01-01T00:01:00+00:00",
            "task_results": {"t1": _task_result("success")},
        }
        b: dict[str, Any] = {
            "plan_name": "p",
            "started_at": "2025-01-01T00:00:00+00:00",
            "finished_at": "2025-01-01T00:01:00+00:00",
            "task_results": {"t1": _task_result("failed")},
        }
        diff = compare_manifests(a, b)
        assert diff.success_a is True
        assert diff.success_b is False

    def test_empty_plan_name_defaults_to_empty_string(self) -> None:
        a: dict[str, Any] = {"success": True}
        b: dict[str, Any] = {"success": True}
        diff = compare_manifests(a, b)
        assert diff.plan_name_a == ""
        assert diff.plan_name_b == ""

    def test_tokens_from_component_fields(self) -> None:
        a = _minimal_manifest(task_results={
            "t1": {
                "status": "success",
                "duration_sec": 1.0,
                "token_usage": {"input_tokens": 100, "output_tokens": 200},
            }
        })
        b = _minimal_manifest(task_results={
            "t1": {
                "status": "success",
                "duration_sec": 1.0,
                "token_usage": {"input_tokens": 300, "output_tokens": 400},
            }
        })
        diff = compare_manifests(a, b)
        td = diff.task_diffs[0]
        assert td.tokens_a == 300  # 100 + 0 + 200
        assert td.tokens_b == 700  # 300 + 0 + 400


# ---------------------------------------------------------------------------
# diff_runs — additional file-based scenarios
# ---------------------------------------------------------------------------


class TestDiffRunsFileBasedAdditional:
    def test_both_manifests_missing(self, tmp_path: Path) -> None:
        run_a = tmp_path / "run_a"
        run_a.mkdir()
        run_b = tmp_path / "run_b"
        run_b.mkdir()
        with pytest.raises(FileNotFoundError):
            diff_runs(run_a, run_b)

    def test_second_manifest_invalid_json(self, tmp_path: Path) -> None:
        run_a = tmp_path / "run_a"
        _write_manifest(run_a, _minimal_manifest())
        run_b = tmp_path / "run_b"
        run_b.mkdir()
        (run_b / "run_manifest.json").write_text("{{bad}}", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            diff_runs(run_a, run_b)


# ---------------------------------------------------------------------------
# format_diff — additional scenarios
# ---------------------------------------------------------------------------


class TestFormatDiffAdditional:
    def _make_diff(self, **kwargs: object) -> RunDiff:
        defaults: dict[str, object] = {
            "run_id_a": "aaa",
            "run_id_b": "bbb",
            "plan_name_a": "plan",
            "plan_name_b": "plan",
            "success_a": True,
            "success_b": True,
            "duration_a": 10.0,
            "duration_b": 20.0,
            "cost_a": 0.5,
            "cost_b": 1.0,
            "tokens_a": 1000,
            "tokens_b": 2000,
            "task_diffs": [],
            "added_tasks": [],
            "removed_tasks": [],
            "regressions": [],
            "fixes": [],
        }
        defaults.update(kwargs)
        return RunDiff(**defaults)  # type: ignore[arg-type]

    def test_format_diff_contains_plan_names(self) -> None:
        diff = self._make_diff(plan_name_a="alpha", plan_name_b="beta")
        output = format_diff(diff)
        assert "alpha" in output
        assert "beta" in output

    def test_format_diff_shows_duration_delta(self) -> None:
        diff = self._make_diff(duration_a=10.0, duration_b=20.0)
        output = format_diff(diff)
        assert "+10.00" in output

    def test_format_diff_shows_cost_delta(self) -> None:
        diff = self._make_diff(cost_a=0.5, cost_b=1.5)
        output = format_diff(diff)
        assert "+$1.0000" in output

    def test_format_diff_shows_token_delta(self) -> None:
        diff = self._make_diff(tokens_a=1000, tokens_b=3000)
        output = format_diff(diff)
        assert "+2000" in output

    def test_format_diff_none_tokens(self) -> None:
        diff = self._make_diff(tokens_a=None, tokens_b=None)
        output = format_diff(diff)
        assert "n/a" in output

    def test_format_diff_empty_plan_name_shows_dash(self) -> None:
        diff = self._make_diff(plan_name_a="", plan_name_b="")
        output = format_diff(diff)
        assert "plan=-" in output

    def test_format_diff_multiple_added_and_removed(self) -> None:
        diff = self._make_diff(
            added_tasks=["new-1", "new-2"],
            removed_tasks=["old-1", "old-2", "old-3"],
        )
        output = format_diff(diff)
        assert "added=2" in output
        assert "removed=3" in output
        assert "new-1" in output
        assert "old-3" in output


# ---------------------------------------------------------------------------
# format_diff_json — additional scenarios
# ---------------------------------------------------------------------------


class TestFormatDiffJsonAdditional:
    def test_json_contains_all_top_level_keys(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success")})
        b = _minimal_manifest(task_results={"t1": _task_result("success")})
        diff = compare_manifests(a, b)
        parsed = json.loads(format_diff_json(diff))
        for key in (
            "run_id_a", "run_id_b", "plan_name_a", "plan_name_b",
            "success_a", "success_b", "duration_a", "duration_b",
            "cost_a", "cost_b", "tokens_a", "tokens_b",
            "task_diffs", "added_tasks", "removed_tasks",
            "regressions", "fixes",
        ):
            assert key in parsed

    def test_json_task_diff_has_all_fields(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success", 1.0, 0.5, 100)})
        b = _minimal_manifest(task_results={"t1": _task_result("failed", 2.0, 0.8, 200)})
        diff = compare_manifests(a, b)
        parsed = json.loads(format_diff_json(diff))
        td = parsed["task_diffs"][0]
        for key in (
            "task_id", "status_a", "status_b",
            "duration_a", "duration_b",
            "cost_a", "cost_b",
            "tokens_a", "tokens_b",
        ):
            assert key in td

    def test_json_added_and_removed_tasks(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success")})
        b = _minimal_manifest(task_results={"t2": _task_result("success")})
        diff = compare_manifests(a, b)
        parsed = json.loads(format_diff_json(diff))
        assert "t2" in parsed["added_tasks"]
        assert "t1" in parsed["removed_tasks"]

    def test_json_null_values_for_none(self) -> None:
        a = _minimal_manifest(task_results={"t1": _task_result("success")})
        b = _minimal_manifest(task_results={})
        diff = compare_manifests(a, b)
        parsed = json.loads(format_diff_json(diff))
        td = next(d for d in parsed["task_diffs"] if d["task_id"] == "t1")
        assert td["status_b"] is None
        assert td["duration_b"] is None
        assert td["cost_b"] is None
        assert td["tokens_b"] is None


# ---------------------------------------------------------------------------
# RunDiff / TaskDiff dataclass structure
# ---------------------------------------------------------------------------


class TestDataclassStructure:
    def test_task_diff_fields(self) -> None:
        td = TaskDiff(
            task_id="t1",
            status_a="success",
            status_b="failed",
            duration_a=1.0,
            duration_b=2.0,
            cost_a=0.1,
            cost_b=0.2,
            tokens_a=100,
            tokens_b=200,
        )
        assert td.task_id == "t1"
        assert td.status_a == "success"
        assert td.status_b == "failed"

    def test_run_diff_fields(self) -> None:
        rd = RunDiff(
            run_id_a="a",
            run_id_b="b",
            plan_name_a="pa",
            plan_name_b="pb",
            success_a=True,
            success_b=False,
            duration_a=10.0,
            duration_b=20.0,
            cost_a=1.0,
            cost_b=2.0,
            tokens_a=100,
            tokens_b=200,
            task_diffs=[],
            added_tasks=["new"],
            removed_tasks=["old"],
            regressions=["reg"],
            fixes=["fix"],
        )
        assert rd.run_id_a == "a"
        assert rd.added_tasks == ["new"]
        assert rd.regressions == ["reg"]
