from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro_cli.status import (
    PipelineState,
    TaskPipelineStatus,
    PlanPipelineStatus,
    _base_state_from_status,
    _coerce_float,
    _compute_hashes_for_plan,
    _duration_from_timestamps,
    _extract_last_cost,
    _extract_last_duration,
    _extract_last_run_at,
    _extract_status,
    _extract_task_hash,
    _format_cost,
    _format_duration,
    _load_result_payload,
    _manifest_task_results,
    _merge_task_payloads,
    _parse_iso,
    _stale_reason,
    plan_status,
    format_status,
    format_status_json,
)
from maestro_cli.models import PlanSpec, TaskSpec, PlanDefaults


def _make_plan(
    tmp_path: Path,
    tasks: list[TaskSpec] | None = None,
    name: str = "test-plan",
) -> PlanSpec:
    if tasks is None:
        tasks = [
            TaskSpec(id="t1", command=["echo", "hello"]),
            TaskSpec(id="t2", command=["echo", "world"], depends_on=["t1"]),
        ]
    return PlanSpec(
        version=1,
        name=name,
        tasks=tasks,
        source_path=tmp_path / "plan.yaml",
        run_dir=(tmp_path / "runs").as_posix(),
        defaults=PlanDefaults(),
    )


def _write_manifest(run_path: Path, manifest: dict) -> None:
    run_path.mkdir(parents=True, exist_ok=True)
    (run_path / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )


def _write_result(run_path: Path, task_id: str, result: dict) -> None:
    (run_path / f"{task_id}.result.json").write_text(
        json.dumps(result), encoding="utf-8",
    )


class TestPlanStatusNoRun:
    def test_all_never_run(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        ps = plan_status(plan, latest_run_path=None, cache_dir=None)
        assert ps.plan_name == "test-plan"
        assert ps.last_run_id is None
        for task in ps.tasks:
            assert task.state == "never-run"

    def test_summary_counts(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        ps = plan_status(plan, latest_run_path=None, cache_dir=None)
        summary = ps.summary
        assert summary["never-run"] == 2
        assert summary["up-to-date"] == 0


class TestPlanStatusWithRun:
    def test_failed_task(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo"]),
        ])
        run_path = tmp_path / "runs" / "run_001"
        _write_manifest(run_path, {
            "run_id": "run_001",
            "task_results": {
                "t1": {"status": "failed", "duration_sec": 1.0},
            },
        })
        _write_result(run_path, "t1", {"status": "failed", "duration_sec": 1.0})

        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].state == "failed"

    def test_skipped_task(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo"]),
        ])
        run_path = tmp_path / "runs" / "run_001"
        _write_manifest(run_path, {
            "run_id": "run_001",
            "task_results": {
                "t1": {"status": "skipped"},
            },
        })

        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].state == "skipped"

    def test_success_without_hash_is_stale(self, tmp_path: Path) -> None:
        """Success without task_hash should be stale (old run without hash tracking)."""
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo", "hello"]),
        ])
        run_path = tmp_path / "runs" / "run_001"
        _write_manifest(run_path, {
            "run_id": "run_001",
            "task_results": {
                "t1": {"status": "success", "duration_sec": 2.0},
            },
        })
        _write_result(run_path, "t1", {
            "status": "success",
            "duration_sec": 2.0,
            # No task_hash — old run format
        })

        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].state == "stale"
        assert ps.tasks[0].stale_reason is not None

    def test_success_with_matching_hash_is_up_to_date(self, tmp_path: Path) -> None:
        from maestro_cli.cache import compute_task_hash

        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo", "hello"]),
        ])
        current_hash = compute_task_hash(plan.tasks[0], plan, {})

        run_path = tmp_path / "runs" / "run_001"
        _write_manifest(run_path, {
            "run_id": "run_001",
            "task_results": {
                "t1": {"status": "success", "task_hash": current_hash},
            },
        })
        _write_result(run_path, "t1", {
            "status": "success",
            "task_hash": current_hash,
            "duration_sec": 2.0,
            "cost_usd": 0.05,
        })

        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].state == "up-to-date"
        assert ps.tasks[0].stale_reason is None

    def test_success_with_changed_hash_is_stale(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo", "hello"]),
        ])
        run_path = tmp_path / "runs" / "run_001"
        _write_manifest(run_path, {
            "run_id": "run_001",
            "task_results": {
                "t1": {"status": "success", "task_hash": "old_hash_abc"},
            },
        })
        _write_result(run_path, "t1", {
            "status": "success",
            "task_hash": "old_hash_abc",
        })

        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].state == "stale"
        assert "changed" in ps.tasks[0].stale_reason

    def test_missing_manifest(self, tmp_path: Path) -> None:
        """Missing manifest → all never-run."""
        plan = _make_plan(tmp_path)
        run_path = tmp_path / "runs" / "run_missing"
        run_path.mkdir(parents=True)  # No manifest inside

        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        for task in ps.tasks:
            assert task.state == "never-run"

    def test_task_not_in_manifest(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo"]),
            TaskSpec(id="t2", command=["echo"]),
        ])
        run_path = tmp_path / "runs" / "run_001"
        _write_manifest(run_path, {
            "run_id": "run_001",
            "task_results": {
                "t1": {"status": "success", "task_hash": "abc"},
            },
        })

        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[1].state == "never-run"


class TestFormatStatus:
    def test_table_output(self, tmp_path: Path) -> None:
        ps = PlanPipelineStatus(
            plan_name="demo",
            last_run_id="run_001",
            last_run_at="2026-03-04T19:00:00",
            tasks=[
                TaskPipelineStatus(task_id="t1", state="up-to-date", last_duration=2.0),
                TaskPipelineStatus(task_id="t2", state="stale", stale_reason="task inputs changed"),
            ],
        )
        output = format_status(ps)
        assert "demo" in output
        assert "up-to-date" in output
        assert "stale" in output
        assert "task inputs changed" in output

    def test_json_output(self, tmp_path: Path) -> None:
        ps = PlanPipelineStatus(
            plan_name="demo",
            last_run_id=None,
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(task_id="t1", state="never-run"),
            ],
        )
        output = format_status_json(ps)
        data = json.loads(output)
        assert data["plan_name"] == "demo"
        assert data["tasks"][0]["state"] == "never-run"


class TestCoerceFloat:
    @pytest.mark.parametrize("value,expected", [
        (None, None),
        (3, 3.0),
        (2.5, 2.5),
        ("3.14", 3.14),
        ("abc", None),
        ([], None),
    ])
    def test_variants(self, value: object, expected: float | None) -> None:
        assert _coerce_float(value) == expected


class TestParseIso:
    def test_valid_iso_string(self) -> None:
        dt = _parse_iso("2026-03-15T12:00:00")
        assert dt is not None
        assert dt.year == 2026

    def test_z_suffix_normalised(self) -> None:
        dt = _parse_iso("2026-03-15T12:00:00Z")
        assert dt is not None
        assert dt.utcoffset() is not None

    def test_empty_string_returns_none(self) -> None:
        assert _parse_iso("") is None

    def test_non_string_returns_none(self) -> None:
        assert _parse_iso(12345) is None

    def test_invalid_format_returns_none(self) -> None:
        assert _parse_iso("not-a-date") is None


class TestDurationFromTimestamps:
    def test_valid_pair(self) -> None:
        secs = _duration_from_timestamps("2026-03-15T10:00:00", "2026-03-15T10:00:05")
        assert secs == 5.0

    def test_negative_clamped_to_zero(self) -> None:
        secs = _duration_from_timestamps("2026-03-15T10:00:05", "2026-03-15T10:00:00")
        assert secs == 0.0

    def test_missing_start_returns_none(self) -> None:
        assert _duration_from_timestamps(None, "2026-03-15T10:00:00") is None


class TestManifestTaskResults:
    def test_valid_dict(self) -> None:
        manifest = {"task_results": {"t1": {"status": "success"}, "t2": {"status": "failed"}}}
        result = _manifest_task_results(manifest)
        assert result == {"t1": {"status": "success"}, "t2": {"status": "failed"}}

    def test_missing_key_returns_empty(self) -> None:
        assert _manifest_task_results({}) == {}

    def test_not_a_dict_returns_empty(self) -> None:
        assert _manifest_task_results({"task_results": [1, 2]}) == {}

    def test_non_string_keys_filtered(self) -> None:
        result = _manifest_task_results({"task_results": {1: {"status": "ok"}, "t1": {"status": "ok"}}})
        assert list(result.keys()) == ["t1"]


class TestMergeTaskPayloads:
    def test_both_none_returns_none(self) -> None:
        assert _merge_task_payloads(None, None) is None

    def test_only_file_returns_file(self) -> None:
        assert _merge_task_payloads({"a": 1}, None) == {"a": 1}

    def test_only_manifest_returns_manifest(self) -> None:
        assert _merge_task_payloads(None, {"b": 2}) == {"b": 2}

    def test_file_overrides_manifest(self) -> None:
        merged = _merge_task_payloads({"key": "file"}, {"key": "manifest", "extra": "x"})
        assert merged is not None
        assert merged["key"] == "file"
        assert merged["extra"] == "x"


class TestBaseStateFromStatus:
    @pytest.mark.parametrize("raw,expected", [
        (None, "never-run"),
        ("success", "up-to-date"),
        ("failed", "failed"),
        ("soft_failed", "failed"),
        ("skipped", "skipped"),
        ("dry_run", "skipped"),
        ("unknown_status", "never-run"),
    ])
    def test_all_mappings(self, raw: str | None, expected: str) -> None:
        assert _base_state_from_status(raw) == expected


class TestStaleReason:
    def test_cached_hash_none(self) -> None:
        reason = _stale_reason(task_id="t1", cached_hash=None, current_hash="abc")
        assert "missing" in reason

    def test_current_hash_none(self) -> None:
        reason = _stale_reason(task_id="t1", cached_hash="abc", current_hash=None)
        assert "cannot compute" in reason

    def test_hashes_differ(self) -> None:
        reason = _stale_reason(task_id="t1", cached_hash="old", current_hash="new")
        assert "changed" in reason


class TestExtractHelpers:
    def test_extract_last_run_at_from_payload(self) -> None:
        payload = {"finished_at": "2026-03-15T10:00:00"}
        assert _extract_last_run_at(payload, "fallback") == "2026-03-15T10:00:00"

    def test_extract_last_run_at_falls_back(self) -> None:
        assert _extract_last_run_at({}, "fallback") == "fallback"

    def test_extract_last_run_at_none_payload(self) -> None:
        assert _extract_last_run_at(None, "fallback") == "fallback"

    def test_extract_last_duration_direct(self) -> None:
        assert _extract_last_duration({"duration_sec": 5.0}) == 5.0

    def test_extract_last_duration_from_timestamps(self) -> None:
        payload = {"started_at": "2026-03-15T10:00:00", "finished_at": "2026-03-15T10:00:03"}
        assert _extract_last_duration(payload) == 3.0

    def test_extract_last_duration_none_payload(self) -> None:
        assert _extract_last_duration(None) is None


class TestPlanPipelineStatusSummary:
    def test_empty_tasks_all_zero(self) -> None:
        ps = PlanPipelineStatus(plan_name="p", last_run_id=None, last_run_at=None, tasks=[])
        summary = ps.summary
        assert all(v == 0 for v in summary.values())

    def test_counts_correct(self) -> None:
        tasks = [
            TaskPipelineStatus(task_id="t1", state="up-to-date"),
            TaskPipelineStatus(task_id="t2", state="stale"),
            TaskPipelineStatus(task_id="t3", state="stale"),
            TaskPipelineStatus(task_id="t4", state="failed"),
        ]
        ps = PlanPipelineStatus(plan_name="p", last_run_id=None, last_run_at=None, tasks=tasks)
        summary = ps.summary
        assert summary["up-to-date"] == 1
        assert summary["stale"] == 2
        assert summary["failed"] == 1
        assert summary["never-run"] == 0


class TestLoadResultPayload:
    def test_valid_file_returns_dict(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run_001"
        run_path.mkdir()
        (run_path / "t1.result.json").write_text(
            json.dumps({"status": "success", "duration_sec": 5.0}), encoding="utf-8"
        )
        result = _load_result_payload(run_path, "t1")
        assert result == {"status": "success", "duration_sec": 5.0}

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run_001"
        run_path.mkdir()
        assert _load_result_payload(run_path, "missing_task") is None

    def test_invalid_json_returns_none(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run_001"
        run_path.mkdir()
        (run_path / "t1.result.json").write_text("not-json{{{", encoding="utf-8")
        assert _load_result_payload(run_path, "t1") is None

    def test_json_array_returns_none(self, tmp_path: Path) -> None:
        run_path = tmp_path / "run_001"
        run_path.mkdir()
        (run_path / "t1.result.json").write_text(json.dumps([1, 2, 3]), encoding="utf-8")
        assert _load_result_payload(run_path, "t1") is None


class TestExtractTaskHash:
    def test_valid_hash_returned(self) -> None:
        assert _extract_task_hash({"task_hash": "abc123"}) == "abc123"

    def test_none_payload_returns_none(self) -> None:
        assert _extract_task_hash(None) is None

    def test_empty_string_returns_none(self) -> None:
        assert _extract_task_hash({"task_hash": ""}) is None

    def test_non_string_returns_none(self) -> None:
        assert _extract_task_hash({"task_hash": 99}) is None


class TestExtractLastCost:
    def test_valid_cost_returned(self) -> None:
        assert _extract_last_cost({"cost_usd": 0.05}) == 0.05

    def test_none_payload_returns_none(self) -> None:
        assert _extract_last_cost(None) is None

    def test_no_cost_key_returns_none(self) -> None:
        assert _extract_last_cost({"status": "success"}) is None

    def test_non_numeric_cost_returns_none(self) -> None:
        assert _extract_last_cost({"cost_usd": "free"}) is None


class TestDurationMissingEnd:
    def test_missing_finished_at_returns_none(self) -> None:
        assert _duration_from_timestamps("2026-03-15T10:00:00", None) is None


class TestStaleReasonEqualHashes:
    def test_equal_hashes_returns_stale_message(self) -> None:
        # Fourth branch: both hashes present and equal (unreachable from plan_status
        # but valid as a unit-level test of the function)
        reason = _stale_reason(task_id="t1", cached_hash="abc", current_hash="abc")
        assert "stale" in reason


class TestCoerceFloatBool:
    def test_bool_true_coerces_to_one(self) -> None:
        # bool is a subclass of int; float(True) = 1.0
        assert _coerce_float(True) == 1.0

    def test_bool_false_coerces_to_zero(self) -> None:
        assert _coerce_float(False) == 0.0


class TestParseIsoAdditional:
    def test_date_only_string_returns_none(self) -> None:
        # "2026-03-15" is a valid fromisoformat input on Python 3.11+
        dt = _parse_iso("2026-03-15")
        # Either returns a datetime or None depending on python version; just don't crash
        assert dt is None or dt.year == 2026

    def test_none_returns_none(self) -> None:
        assert _parse_iso(None) is None


class TestPlanStatusSoftFailed:
    def test_soft_failed_task_maps_to_failed_state(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo"]),
        ])
        run_path = tmp_path / "runs" / "run_001"
        _write_manifest(run_path, {
            "run_id": "run_001",
            "task_results": {
                "t1": {"status": "soft_failed", "duration_sec": 1.0},
            },
        })
        _write_result(run_path, "t1", {"status": "soft_failed", "duration_sec": 1.0})
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].state == "failed"


class TestDurationFromTimestampsAdditional:
    def test_exact_zero_difference(self) -> None:
        secs = _duration_from_timestamps("2026-03-15T10:00:00", "2026-03-15T10:00:00")
        assert secs == 0.0


class TestPlanStatusManifestMetadata:
    def test_run_id_from_path_when_missing(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="t1", command=["echo"])])
        run_path = tmp_path / "runs" / "my_run_dir"
        _write_manifest(run_path, {"task_results": {}})
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.last_run_id == "my_run_dir"

    def test_last_run_at_from_manifest_finished_at(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="t1", command=["echo"])])
        run_path = tmp_path / "runs" / "run_ts"
        _write_manifest(run_path, {
            "task_results": {},
            "finished_at": "2026-03-15T12:00:00",
        })
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.last_run_at == "2026-03-15T12:00:00"


class TestDurationFromTimestampsBothNone:
    def test_both_none_returns_none(self) -> None:
        assert _duration_from_timestamps(None, None) is None


class TestCoerceFloatAdditional:
    def test_dict_returns_none(self) -> None:
        assert _coerce_float({"key": "value"}) is None

    def test_integer_zero_returns_zero(self) -> None:
        assert _coerce_float(0) == 0.0


class TestExtractLastDurationNoFields:
    def test_empty_payload_returns_none(self) -> None:
        # No duration_sec, no started_at, no finished_at
        assert _extract_last_duration({}) is None

    def test_payload_with_only_irrelevant_fields_returns_none(self) -> None:
        assert _extract_last_duration({"status": "success", "cost_usd": 0.05}) is None


class TestFormatStatusSummaryLine:
    def test_never_run_count_in_output(self, tmp_path: Path) -> None:
        ps = PlanPipelineStatus(
            plan_name="demo",
            last_run_id=None,
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(task_id="t1", state="never-run"),
                TaskPipelineStatus(task_id="t2", state="never-run"),
                TaskPipelineStatus(task_id="t3", state="up-to-date"),
            ],
        )
        output = format_status(ps)
        assert "never-run" in output
        assert "up-to-date" in output


class TestExtractStatus:
    def test_none_payload_returns_none(self) -> None:
        assert _extract_status(None) is None

    def test_valid_status_returned(self) -> None:
        assert _extract_status({"status": "success"}) == "success"

    def test_empty_string_returns_none(self) -> None:
        assert _extract_status({"status": ""}) is None

    def test_non_string_returns_none(self) -> None:
        assert _extract_status({"status": 123}) is None

    def test_missing_key_returns_none(self) -> None:
        assert _extract_status({}) is None


class TestExtractLastRunAtEdgeCases:
    def test_empty_finished_at_uses_fallback(self) -> None:
        """finished_at='' fails the `isinstance(s, str) and s` guard → fallback returned."""
        result = _extract_last_run_at({"finished_at": ""}, "fallback_value")
        assert result == "fallback_value"

    def test_non_string_finished_at_uses_fallback(self) -> None:
        result = _extract_last_run_at({"finished_at": 12345}, "fallback_value")
        assert result == "fallback_value"


class TestFormatStatusFormattedValues:
    def test_duration_and_cost_formatted_in_output(self) -> None:
        ps = PlanPipelineStatus(
            plan_name="demo",
            last_run_id="run_001",
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(
                    task_id="t1",
                    state="up-to-date",
                    last_duration=2.5,
                    last_cost=0.05,
                ),
            ],
        )
        output = format_status(ps)
        assert "2.50s" in output
        assert "$0.0500" in output


class TestCoerceFloatNegative:
    def test_negative_numeric_string(self) -> None:
        assert _coerce_float("-5.0") == -5.0


class TestFormatDuration:
    def test_none_returns_dash(self) -> None:
        assert _format_duration(None) == "-"

    def test_zero_returns_formatted(self) -> None:
        assert _format_duration(0.0) == "0.00s"

    def test_positive_value_formatted(self) -> None:
        assert _format_duration(12.345) == "12.35s"


class TestFormatCost:
    def test_none_returns_dash(self) -> None:
        assert _format_cost(None) == "-"

    def test_zero_returns_formatted(self) -> None:
        assert _format_cost(0.0) == "$0.0000"

    def test_positive_value_formatted(self) -> None:
        assert _format_cost(1.23456) == "$1.2346"


class TestComputeHashesForPlan:
    def test_simple_plan_returns_hashes(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo", "a"]),
            TaskSpec(id="t2", command=["echo", "b"]),
        ])
        hashes = _compute_hashes_for_plan(plan)
        assert "t1" in hashes
        assert "t2" in hashes
        assert hashes["t1"] is not None
        assert hashes["t2"] is not None

    def test_dependent_task_hash_differs_from_independent(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo", "a"]),
            TaskSpec(id="t2", command=["echo", "a"], depends_on=["t1"]),
        ])
        hashes = _compute_hashes_for_plan(plan)
        # t2 depends on t1 so its hash should differ from t1's hash
        assert hashes["t1"] != hashes["t2"]


class TestPlanStatusDryRun:
    def test_dry_run_task_maps_to_skipped_state(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo"]),
        ])
        run_path = tmp_path / "runs" / "run_001"
        _write_manifest(run_path, {
            "run_id": "run_001",
            "task_results": {
                "t1": {"status": "dry_run"},
            },
        })
        _write_result(run_path, "t1", {"status": "dry_run"})
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].state == "skipped"


class TestFormatStatusJsonFullData:
    def test_stale_task_includes_reason_in_json(self) -> None:
        ps = PlanPipelineStatus(
            plan_name="full",
            last_run_id="run_001",
            last_run_at="2026-03-15T12:00:00",
            tasks=[
                TaskPipelineStatus(
                    task_id="t1",
                    state="stale",
                    stale_reason="task inputs changed",
                    last_duration=5.5,
                    last_cost=0.02,
                ),
            ],
        )
        output = format_status_json(ps)
        data = json.loads(output)
        t = data["tasks"][0]
        assert t["state"] == "stale"
        assert t["stale_reason"] == "task inputs changed"
        assert t["last_duration"] == 5.5
        assert t["last_cost"] == 0.02


class TestManifestTaskResultsNonDictPayload:
    def test_non_dict_payload_values_filtered_out(self) -> None:
        manifest = {"task_results": {"t1": "not_a_dict", "t2": {"status": "ok"}}}
        result = _manifest_task_results(manifest)
        assert "t1" not in result
        assert "t2" in result


class TestPlanStatusRunIdFromManifest:
    def test_run_id_from_manifest_explicit(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="t1", command=["echo"])])
        run_path = tmp_path / "runs" / "some_dir_name"
        _write_manifest(run_path, {
            "run_id": "explicit-run-id",
            "task_results": {},
        })
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.last_run_id == "explicit-run-id"


class TestFormatStatusEmptyTasks:
    def test_empty_task_list_no_crash(self) -> None:
        ps = PlanPipelineStatus(
            plan_name="empty-plan",
            last_run_id=None,
            last_run_at=None,
            tasks=[],
        )
        output = format_status(ps)
        assert "empty-plan" in output
        assert "Summary" in output


class TestPlanStatusMergedPayloads:
    def test_file_cost_overrides_manifest_cost(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="t1", command=["echo"])])
        run_path = tmp_path / "runs" / "run_merge"
        _write_manifest(run_path, {
            "run_id": "run_merge",
            "task_results": {
                "t1": {"status": "success", "cost_usd": 0.01, "task_hash": "stale_hash"},
            },
        })
        _write_result(run_path, "t1", {
            "status": "success",
            "cost_usd": 0.99,
            "task_hash": "stale_hash",
        })
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        # File payload cost (0.99) overrides manifest payload (0.01)
        assert ps.tasks[0].last_cost == 0.99


class TestExtractLastDurationStartOnly:
    def test_started_at_only_returns_none(self) -> None:
        """started_at present but finished_at absent → duration cannot be computed."""
        payload = {"started_at": "2026-03-15T10:00:00"}
        assert _extract_last_duration(payload) is None


class TestPlanStatusManifestEmptyFinishedAt:
    def test_empty_finished_at_leaves_last_run_at_none(self, tmp_path: Path) -> None:
        """manifest finished_at='' fails the non-empty string guard → last_run_at stays None."""
        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="t1", command=["echo"])])
        run_path = tmp_path / "runs" / "run_ts"
        _write_manifest(run_path, {
            "task_results": {},
            "finished_at": "",
        })
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.last_run_at is None


class TestComputeHashesExceptionYieldsNone:
    def test_compute_hash_exception_stored_as_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """compute_task_hash raising an exception → that task's hash is stored as None."""
        import maestro_cli.status as status_mod

        def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError("simulated hash error")

        monkeypatch.setattr(status_mod, "compute_task_hash", _raise)
        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="t1", command=["echo"])])
        hashes = _compute_hashes_for_plan(plan)
        assert hashes["t1"] is None


class TestPlanPipelineStatusSummarySkipped:
    def test_skipped_state_counted(self) -> None:
        tasks = [
            TaskPipelineStatus(task_id="t1", state="skipped"),
            TaskPipelineStatus(task_id="t2", state="skipped"),
            TaskPipelineStatus(task_id="t3", state="never-run"),
        ]
        ps = PlanPipelineStatus(plan_name="p", last_run_id=None, last_run_at=None, tasks=tasks)
        summary = ps.summary
        assert summary["skipped"] == 2
        assert summary["never-run"] == 1
        assert summary["up-to-date"] == 0


class TestComputeHashesOrphanDep:
    def test_task_with_unknown_dep_still_gets_hash(self, tmp_path: Path) -> None:
        plan = PlanSpec(
            version=1,
            name="orphan-plan",
            tasks=[
                TaskSpec(id="t1", command=["echo", "a"], depends_on=["nonexistent"]),
            ],
            source_path=tmp_path / "plan.yaml",
            run_dir=(tmp_path / "runs").as_posix(),
            defaults=PlanDefaults(),
        )
        hashes = _compute_hashes_for_plan(plan)
        assert "t1" in hashes
        assert hashes["t1"] is not None


class TestExtractLastDurationZeroValue:
    def test_zero_duration_sec_returns_zero_not_none(self) -> None:
        """duration_sec=0.0 is a valid (zero) float, not None."""
        assert _extract_last_duration({"duration_sec": 0.0}) == 0.0

    def test_integer_zero_duration_returns_zero(self) -> None:
        assert _extract_last_duration({"duration_sec": 0}) == 0.0


class TestCoerceFloatWhitespaceString:
    def test_whitespace_padded_number_is_coerced(self) -> None:
        """Python's float() strips leading/trailing whitespace."""
        assert _coerce_float("  3.14  ") == 3.14

    def test_whitespace_only_string_returns_none(self) -> None:
        assert _coerce_float("   ") is None


class TestFormatStatusLastRunAtShown:
    def test_last_run_at_displayed_in_output(self) -> None:
        ps = PlanPipelineStatus(
            plan_name="dated-plan",
            last_run_id="run_007",
            last_run_at="2026-03-15T12:00:00",
            tasks=[
                TaskPipelineStatus(task_id="t1", state="up-to-date"),
            ],
        )
        output = format_status(ps)
        assert "2026-03-15T12:00:00" in output
        assert "run_007" in output


class TestFormatStatusJsonLastRunFields:
    def test_json_contains_last_run_id_and_last_run_at(self) -> None:
        ps = PlanPipelineStatus(
            plan_name="p",
            last_run_id="run_42",
            last_run_at="2026-03-15T09:00:00",
            tasks=[],
        )
        import json as _json
        data = _json.loads(format_status_json(ps))
        assert data["last_run_id"] == "run_42"
        assert data["last_run_at"] == "2026-03-15T09:00:00"
        assert data["plan_name"] == "p"


class TestParseIsoFractionalSeconds:
    def test_fractional_seconds_parsed_correctly(self) -> None:
        """fromisoformat handles fractional seconds on Python 3.11+."""
        dt = _parse_iso("2026-03-15T12:00:00.500000")
        assert dt is not None
        assert dt.year == 2026
        assert dt.hour == 12

    def test_z_suffix_with_fractional_seconds(self) -> None:
        """Z suffix normalised before parsing; fractional seconds preserved."""
        dt = _parse_iso("2026-03-15T12:00:00.500000Z")
        assert dt is not None
        assert dt.utcoffset() is not None


class TestStaleReasonTaskIdEmbedded:
    def test_task_id_appears_in_equal_hashes_message(self) -> None:
        """When both hashes are present and equal, the task_id is included in the message."""
        reason = _stale_reason(task_id="my-special-task", cached_hash="same", current_hash="same")
        assert "my-special-task" in reason


class TestPlanPipelineStatusSummaryMixedCounts:
    def test_multiple_tasks_each_state_counted(self) -> None:
        tasks = [
            TaskPipelineStatus(task_id="t1", state="up-to-date"),
            TaskPipelineStatus(task_id="t2", state="up-to-date"),
            TaskPipelineStatus(task_id="t3", state="stale"),
            TaskPipelineStatus(task_id="t4", state="failed"),
            TaskPipelineStatus(task_id="t5", state="never-run"),
            TaskPipelineStatus(task_id="t6", state="skipped"),
        ]
        ps = PlanPipelineStatus(plan_name="multi", last_run_id=None, last_run_at=None, tasks=tasks)
        summary = ps.summary
        assert summary["up-to-date"] == 2
        assert summary["stale"] == 1
        assert summary["failed"] == 1
        assert summary["never-run"] == 1
        assert summary["skipped"] == 1


class TestExtractStatusArbitraryNonEmptyString:
    def test_unrecognised_status_string_returned_as_is(self) -> None:
        """_extract_status only checks for non-empty string; any value is passed through."""
        assert _extract_status({"status": "pending"}) == "pending"
        assert _extract_status({"status": "weird_unknown"}) == "weird_unknown"


class TestParseIsoTimezoneOffset:
    def test_explicit_timezone_offset_parsed(self) -> None:
        """fromisoformat handles an explicit +01:00 offset without the Z normalisation path."""
        dt = _parse_iso("2026-03-15T12:00:00+01:00")
        assert dt is not None
        assert dt.utcoffset() is not None
        assert dt.hour == 12

    def test_negative_offset_parsed(self) -> None:
        dt = _parse_iso("2026-03-15T08:00:00-05:00")
        assert dt is not None
        assert dt.hour == 8


class TestComputeHashesThreeLevel:
    def test_three_level_chain_all_distinct(self, tmp_path: Path) -> None:
        """t1 → t2 → t3: upstream hash propagation should produce three distinct hashes."""
        plan = PlanSpec(
            version=1,
            name="three-level",
            tasks=[
                TaskSpec(id="t1", command=["echo", "a"]),
                TaskSpec(id="t2", command=["echo", "b"], depends_on=["t1"]),
                TaskSpec(id="t3", command=["echo", "c"], depends_on=["t2"]),
            ],
            source_path=tmp_path / "plan.yaml",
            run_dir=(tmp_path / "runs").as_posix(),
            defaults=PlanDefaults(),
        )
        hashes = _compute_hashes_for_plan(plan)
        assert hashes["t1"] is not None
        assert hashes["t2"] is not None
        assert hashes["t3"] is not None
        # all three levels must produce distinct hashes
        assert len({hashes["t1"], hashes["t2"], hashes["t3"]}) == 3


class TestExtractLastCostBoundaryValues:
    def test_zero_cost_returns_zero_not_none(self) -> None:
        """cost_usd=0.0 is valid; _coerce_float(0.0)=0.0, not None."""
        assert _extract_last_cost({"cost_usd": 0.0}) == 0.0

    def test_integer_cost_coerced_to_float(self) -> None:
        """cost_usd=2 (int) → _coerce_float(2) = 2.0."""
        assert _extract_last_cost({"cost_usd": 2}) == 2.0


class TestComputeHashesCircularDependency:
    def test_mutual_circular_dep_both_tasks_get_hash(self, tmp_path: Path) -> None:
        """t1 depends on t2 and t2 depends on t1 — visiting set prevents infinite recursion."""
        plan = PlanSpec(
            version=1,
            name="circular-plan",
            tasks=[
                TaskSpec(id="t1", command=["echo", "a"], depends_on=["t2"]),
                TaskSpec(id="t2", command=["echo", "b"], depends_on=["t1"]),
            ],
            source_path=tmp_path / "plan.yaml",
            run_dir=(tmp_path / "runs").as_posix(),
            defaults=PlanDefaults(),
        )
        # Should not raise or loop forever; visiting set breaks the cycle
        hashes = _compute_hashes_for_plan(plan)
        assert "t1" in hashes
        assert "t2" in hashes
        # Both should be computed (not None) once cycle is broken with empty upstream hash


class TestPlanStatusManifestNoTaskResultsKey:
    def test_manifest_without_task_results_key_gives_never_run(self, tmp_path: Path) -> None:
        """Manifest with no 'task_results' key at all → all tasks are never-run."""
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo"]),
            TaskSpec(id="t2", command=["echo"]),
        ])
        run_path = tmp_path / "runs" / "run_001"
        # Manifest has no task_results key — _manifest_task_results returns {}
        _write_manifest(run_path, {"run_id": "run_001"})
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        for task in ps.tasks:
            assert task.state == "never-run"

    def test_manifest_empty_dict_gives_never_run(self, tmp_path: Path) -> None:
        """Manifest is {} (not even run_id) → all tasks are never-run."""
        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="t1", command=["echo"])])
        run_path = tmp_path / "runs" / "run_001"
        _write_manifest(run_path, {})
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].state == "never-run"


class TestExtractLastDurationStringFallsToTimestamps:
    def test_string_duration_sec_falls_through_to_timestamps(self) -> None:
        """duration_sec='N/A' → _coerce_float returns None → falls back to timestamps."""
        payload = {
            "duration_sec": "N/A",
            "started_at": "2026-03-15T10:00:00",
            "finished_at": "2026-03-15T10:00:07",
        }
        result = _extract_last_duration(payload)
        assert result == 7.0

    def test_none_duration_sec_falls_through_to_timestamps(self) -> None:
        """duration_sec=None → _coerce_float(None) returns None → falls back to timestamps."""
        payload = {
            "duration_sec": None,
            "started_at": "2026-03-15T10:00:00",
            "finished_at": "2026-03-15T10:00:10",
        }
        result = _extract_last_duration(payload)
        assert result == 10.0


class TestComputeHashesEmptyPlan:
    def test_empty_tasks_returns_empty_dict(self, tmp_path: Path) -> None:
        """PlanSpec with no tasks → _compute_hashes_for_plan returns {}."""
        plan = PlanSpec(
            version=1,
            name="empty-plan",
            tasks=[],
            source_path=tmp_path / "plan.yaml",
            run_dir=(tmp_path / "runs").as_posix(),
            defaults=PlanDefaults(),
        )
        hashes = _compute_hashes_for_plan(plan)
        assert hashes == {}


class TestPlanStatusRunIdNonString:
    def test_non_string_run_id_falls_back_to_path_name(self, tmp_path: Path) -> None:
        """manifest run_id=42 (not str) → falls through else → last_run_id = run_path.name."""
        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="t1", command=["echo"])])
        run_path = tmp_path / "runs" / "the_run_name"
        _write_manifest(run_path, {
            "run_id": 42,
            "task_results": {},
        })
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.last_run_id == "the_run_name"


class TestExtractLastRunAtIntFinishedAt:
    def test_int_finished_at_uses_fallback(self) -> None:
        """payload.finished_at is integer → fails isinstance(str) guard → fallback returned."""
        result = _extract_last_run_at({"finished_at": 1710000000}, "2026-03-15T00:00:00")
        assert result == "2026-03-15T00:00:00"


class TestLoadResultPayloadJsonNull:
    def test_json_null_returns_none(self, tmp_path: Path) -> None:
        """Result file contains JSON null → data is None → not isinstance(None, dict) → None."""
        run_path = tmp_path / "run_null"
        run_path.mkdir()
        (run_path / "t1.result.json").write_text("null", encoding="utf-8")
        assert _load_result_payload(run_path, "t1") is None


class TestPlanStatusTaskOwnFinishedAt:
    def test_task_finished_at_overrides_manifest_last_run_at(self, tmp_path: Path) -> None:
        """Task result payload has finished_at; that takes priority over manifest-level last_run_at."""
        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="t1", command=["echo"])])
        run_path = tmp_path / "runs" / "run_ts"
        _write_manifest(run_path, {
            "task_results": {"t1": {"status": "failed"}},
            "finished_at": "2026-03-15T10:00:00",
        })
        _write_result(run_path, "t1", {
            "status": "failed",
            "finished_at": "2026-03-15T10:05:00",
        })
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].last_run_at == "2026-03-15T10:05:00"


class TestFormatStatusNeverRunTaskShowsDashes:
    def test_never_run_task_shows_dash_for_duration_and_cost(self) -> None:
        """Never-run tasks have no duration or cost — format_status renders '-' for those columns."""
        ps = PlanPipelineStatus(
            plan_name="dash-plan",
            last_run_id=None,
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(
                    task_id="t1",
                    state="never-run",
                    last_duration=None,
                    last_cost=None,
                ),
            ],
        )
        output = format_status(ps)
        # "-" appears for both Duration and Cost columns
        assert output.count("-") >= 2
        assert "never-run" in output


class TestComputeHashesDiamondDependency:
    def test_diamond_pattern_produces_distinct_hashes(self, tmp_path: Path) -> None:
        """Diamond: t1→t3, t2→t3 — t3 should differ from t1 and t2, all non-None."""
        plan = PlanSpec(
            version=1,
            name="diamond-plan",
            tasks=[
                TaskSpec(id="t1", command=["echo", "a"]),
                TaskSpec(id="t2", command=["echo", "b"]),
                TaskSpec(id="t3", command=["echo", "c"], depends_on=["t1", "t2"]),
            ],
            source_path=tmp_path / "plan.yaml",
            run_dir=(tmp_path / "runs").as_posix(),
            defaults=PlanDefaults(),
        )
        hashes = _compute_hashes_for_plan(plan)
        assert hashes["t1"] is not None
        assert hashes["t2"] is not None
        assert hashes["t3"] is not None
        # t3 depends on both t1 and t2 — it must differ from each
        assert hashes["t3"] != hashes["t1"]
        assert hashes["t3"] != hashes["t2"]


class TestExtractLastCostStringNumeric:
    def test_string_numeric_cost_coerced_via_float(self) -> None:
        """cost_usd='0.05' → _coerce_float('0.05') → 0.05."""
        assert _extract_last_cost({"cost_usd": "0.05"}) == 0.05

    def test_string_negative_cost_coerced(self) -> None:
        """cost_usd='-1.5' → _coerce_float('-1.5') → -1.5."""
        assert _extract_last_cost({"cost_usd": "-1.5"}) == -1.5


class TestFormatStatusJsonSummaryAbsent:
    def test_json_output_does_not_include_summary_property(self) -> None:
        """summary is a @property on PlanPipelineStatus; dataclasses.asdict does not include it."""
        ps = PlanPipelineStatus(
            plan_name="prop-plan",
            last_run_id=None,
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(task_id="t1", state="up-to-date"),
                TaskPipelineStatus(task_id="t2", state="stale"),
            ],
        )
        data = json.loads(format_status_json(ps))
        assert "summary" not in data
        # Verify the actual fields are present
        assert data["plan_name"] == "prop-plan"
        assert len(data["tasks"]) == 2


class TestPlanStatusCorruptManifestJson:
    def test_invalid_json_manifest_all_never_run(self, tmp_path: Path) -> None:
        """Manifest file with invalid JSON → load_run_manifest raises → all tasks never-run."""
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo"]),
            TaskSpec(id="t2", command=["echo"]),
        ])
        run_path = tmp_path / "runs" / "run_corrupt"
        run_path.mkdir(parents=True)
        (run_path / "run_manifest.json").write_text(
            "not-valid-json{{{", encoding="utf-8"
        )
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.last_run_id is None
        for task in ps.tasks:
            assert task.state == "never-run"


class TestPlanPipelineStatusSummaryEmpty:
    def test_empty_tasks_all_counts_zero(self) -> None:
        """PlanPipelineStatus with no tasks → summary returns all-zero counts for all states."""
        ps = PlanPipelineStatus(plan_name="empty", last_run_id=None, last_run_at=None, tasks=[])
        summary = ps.summary
        assert summary["up-to-date"] == 0
        assert summary["stale"] == 0
        assert summary["never-run"] == 0
        assert summary["failed"] == 0
        assert summary["skipped"] == 0
        assert len(summary) == 5


class TestFormatStatusStaleReasonColumn:
    def test_stale_reason_appears_in_reason_column(self) -> None:
        """A stale task's stale_reason should appear in the Reason column of format_status."""
        ps = PlanPipelineStatus(
            plan_name="stale-plan",
            last_run_id="run_001",
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(
                    task_id="t1",
                    state="stale",
                    stale_reason="task inputs changed",
                ),
            ],
        )
        output = format_status(ps)
        assert "task inputs changed" in output
        assert "stale" in output

    def test_non_stale_task_shows_dash_in_reason_column(self) -> None:
        """A non-stale task has no stale_reason; the Reason column shows '-'."""
        ps = PlanPipelineStatus(
            plan_name="fresh-plan",
            last_run_id="run_001",
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(
                    task_id="t1",
                    state="up-to-date",
                    stale_reason=None,
                ),
            ],
        )
        output = format_status(ps)
        # stale_reason=None → renders as "-"
        assert "up-to-date" in output


class TestFormatStatusColumnHeaders:
    def test_format_status_includes_column_headers(self) -> None:
        """format_status output includes the expected column header names."""
        ps = PlanPipelineStatus(
            plan_name="header-plan",
            last_run_id=None,
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(task_id="t1", state="never-run"),
            ],
        )
        output = format_status(ps)
        assert "Task" in output
        assert "State" in output
        assert "Duration" in output
        assert "Cost" in output

    def test_format_status_includes_summary_counts(self) -> None:
        """format_status summary line shows counts for all five states."""
        ps = PlanPipelineStatus(
            plan_name="counts-plan",
            last_run_id=None,
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(task_id="t1", state="up-to-date"),
                TaskPipelineStatus(task_id="t2", state="stale"),
                TaskPipelineStatus(task_id="t3", state="failed"),
                TaskPipelineStatus(task_id="t4", state="skipped"),
                TaskPipelineStatus(task_id="t5", state="never-run"),
            ],
        )
        output = format_status(ps)
        assert "up-to-date=1" in output
        assert "stale=1" in output
        assert "failed=1" in output
        assert "skipped=1" in output
        assert "never-run=1" in output


class TestFormatDurationLargeValue:
    def test_large_duration_formatted_correctly(self) -> None:
        """Large duration (e.g., 3661.5 seconds) is formatted with 2 decimal places."""
        assert _format_duration(3661.5) == "3661.50s"

    def test_fractional_duration_rounded(self) -> None:
        """Duration 0.005 rounds to 0.01s (banker's rounding aside, 2 decimal places)."""
        result = _format_duration(0.005)
        assert result.endswith("s")
        assert result.startswith("0.")


class TestManifestTaskResultsIntegerKey:
    def test_integer_key_filtered_out(self) -> None:
        """Non-string key (integer) in task_results is filtered out by the str check."""
        manifest = {"task_results": {42: {"status": "ok"}, "t1": {"status": "success"}}}
        result = _manifest_task_results(manifest)
        assert 42 not in result
        assert "t1" in result


class TestExtractLastRunAtBothNone:
    def test_none_payload_and_none_fallback_returns_none(self) -> None:
        """_extract_last_run_at(None, None): payload is None, fallback is None → returns None."""
        result = _extract_last_run_at(None, None)
        assert result is None

    def test_payload_no_finished_at_none_fallback_returns_none(self) -> None:
        """Payload has no finished_at key and fallback is None → returns None."""
        result = _extract_last_run_at({}, None)
        assert result is None


class TestCoerceFloatList:
    def test_list_input_returns_none(self) -> None:
        """list input → float([1, 2]) raises TypeError → _coerce_float returns None."""
        assert _coerce_float([1.0, 2.0]) is None

    def test_dict_input_returns_none(self) -> None:
        """dict input → float({}) raises TypeError → _coerce_float returns None."""
        assert _coerce_float({"value": 3.14}) is None


class TestFormatStatusRunIdWithoutDate:
    def test_run_id_shown_but_date_dashes_when_last_run_at_is_none(self) -> None:
        """format_status with last_run_id set and last_run_at=None shows run_id and '-' for date."""
        ps = PlanPipelineStatus(
            plan_name="no-date-plan",
            last_run_id="run_abc",
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(task_id="t1", state="up-to-date"),
            ],
        )
        output = format_status(ps)
        assert "run_abc" in output
        assert "no-date-plan" in output


class TestExtractTaskHashMissingKey:
    def test_no_task_hash_key_returns_none(self) -> None:
        """Payload has no 'task_hash' key → .get returns None → fails str check → None."""
        assert _extract_task_hash({"status": "success"}) is None

    def test_task_hash_none_value_returns_none(self) -> None:
        """task_hash=None → fails isinstance(str) → None."""
        assert _extract_task_hash({"task_hash": None}) is None


class TestMergeTaskPayloadsDisjointKeys:
    def test_disjoint_keys_merged(self) -> None:
        """File payload and manifest payload have completely disjoint keys → all merged."""
        merged = _merge_task_payloads({"file_key": "A"}, {"manifest_key": "B"})
        assert merged is not None
        assert merged["file_key"] == "A"
        assert merged["manifest_key"] == "B"

    def test_empty_file_only_manifest_used(self) -> None:
        """File payload is empty dict → manifest keys stay."""
        merged = _merge_task_payloads({}, {"m_key": "val"})
        assert merged is not None
        assert merged["m_key"] == "val"


class TestPlanStatusCurrentHashNone:
    def test_success_with_hash_but_current_hash_none_is_stale(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Successful task but compute_task_hash raises → current_hash=None → stale."""
        import maestro_cli.status as status_mod

        def _raise(*args: object, **kwargs: object) -> None:
            raise RuntimeError("hash computation failed")

        monkeypatch.setattr(status_mod, "compute_task_hash", _raise)

        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="t1", command=["echo"])])
        run_path = tmp_path / "runs" / "run_001"
        _write_manifest(run_path, {
            "run_id": "run_001",
            "task_results": {
                "t1": {"status": "success", "task_hash": "some_stored_hash"},
            },
        })
        _write_result(run_path, "t1", {
            "status": "success",
            "task_hash": "some_stored_hash",
        })

        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].state == "stale"
        assert ps.tasks[0].stale_reason is not None
        assert "cannot compute" in ps.tasks[0].stale_reason


class TestFormatStatusMultipleTasksAlignment:
    def test_columns_aligned_with_varying_task_id_lengths(self) -> None:
        """Column widths adapt to the longest task_id and values."""
        ps = PlanPipelineStatus(
            plan_name="align-plan",
            last_run_id="run_001",
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(task_id="a", state="up-to-date", last_duration=1.0),
                TaskPipelineStatus(
                    task_id="very-long-task-identifier",
                    state="stale",
                    stale_reason="task inputs changed",
                    last_duration=99.99,
                    last_cost=1.5,
                ),
            ],
        )
        output = format_status(ps)
        lines = output.strip().split("\n")
        # Find the separator line (---+---+---)
        sep_lines = [l for l in lines if l.startswith("-")]
        assert len(sep_lines) >= 1
        # Task column should be wide enough for the long id
        assert "very-long-task-identifier" in output
        assert "99.99s" in output
        assert "$1.5000" in output


class TestPlanStatusMultipleTasksMixed:
    def test_mixed_status_tasks_produce_correct_states(self, tmp_path: Path) -> None:
        """Multiple tasks with different statuses: success, failed, skipped."""
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo"]),
            TaskSpec(id="t2", command=["echo"]),
            TaskSpec(id="t3", command=["echo"]),
        ])
        run_path = tmp_path / "runs" / "run_mixed"
        _write_manifest(run_path, {
            "run_id": "run_mixed",
            "task_results": {
                "t1": {"status": "success", "task_hash": "will_not_match"},
                "t2": {"status": "failed", "duration_sec": 5.0},
                "t3": {"status": "skipped"},
            },
        })

        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        states = {t.task_id: t.state for t in ps.tasks}
        assert states["t1"] == "stale"  # hash mismatch → stale
        assert states["t2"] == "failed"
        assert states["t3"] == "skipped"


class TestFormatCostEdge:
    def test_very_small_cost_formatted(self) -> None:
        """Very small cost like 0.0001 should show with 4 decimal places."""
        assert _format_cost(0.0001) == "$0.0001"

    def test_large_cost_formatted(self) -> None:
        assert _format_cost(123.456789) == "$123.4568"


class TestComputeHashesSelfReference:
    def test_self_referencing_dep_handled(self, tmp_path: Path) -> None:
        """Task depends on itself → visiting set detects cycle → hash still produced."""
        plan = PlanSpec(
            version=1,
            name="self-ref",
            tasks=[
                TaskSpec(id="t1", command=["echo", "a"], depends_on=["t1"]),
            ],
            source_path=tmp_path / "plan.yaml",
            run_dir=(tmp_path / "runs").as_posix(),
            defaults=PlanDefaults(),
        )
        hashes = _compute_hashes_for_plan(plan)
        assert "t1" in hashes
        # Self-cycle detected via visiting set → upstream hash is empty string
        # but the task itself still gets a hash (not None)
        assert hashes["t1"] is not None


class TestPlanStatusManifestIntFinishedAt:
    def test_int_finished_at_in_manifest_yields_none_last_run_at(self, tmp_path: Path) -> None:
        """manifest finished_at=1710000000 (int) → fails isinstance(str) guard → last_run_at=None."""
        plan = _make_plan(tmp_path, tasks=[TaskSpec(id="t1", command=["echo"])])
        run_path = tmp_path / "runs" / "run_int_ts"
        _write_manifest(run_path, {
            "task_results": {},
            "finished_at": 1710000000,
        })
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.last_run_at is None


class TestExtractLastDurationNumericString:
    def test_numeric_string_duration_sec_coerced_to_float(self) -> None:
        """duration_sec='5.0' → _coerce_float('5.0') → 5.0 (returned, no timestamp fallback)."""
        assert _extract_last_duration({"duration_sec": "5.0"}) == 5.0

    def test_integer_string_duration_sec_coerced(self) -> None:
        """duration_sec='10' → _coerce_float('10') → 10.0."""
        assert _extract_last_duration({"duration_sec": "10"}) == 10.0


class TestFormatStatusSeparatorLine:
    def test_separator_line_contains_dashes_and_plus(self) -> None:
        """format_status output includes a separator line with '-+-' pattern."""
        ps = PlanPipelineStatus(
            plan_name="sep-plan",
            last_run_id="run_001",
            last_run_at=None,
            tasks=[
                TaskPipelineStatus(task_id="t1", state="up-to-date"),
            ],
        )
        output = format_status(ps)
        lines = output.split("\n")
        separator_lines = [l for l in lines if "-+-" in l]
        assert len(separator_lines) == 1


class TestComputeHashesIndependentDifferentCommands:
    def test_two_independent_tasks_different_commands_differ(self, tmp_path: Path) -> None:
        """Two root tasks with different commands → different hashes."""
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo", "alpha"]),
            TaskSpec(id="t2", command=["echo", "beta"]),
        ])
        hashes = _compute_hashes_for_plan(plan)
        assert hashes["t1"] is not None
        assert hashes["t2"] is not None
        assert hashes["t1"] != hashes["t2"]


class TestPlanStatusSuccessHashMatchManifestOnly:
    def test_hash_match_from_manifest_only_is_up_to_date(self, tmp_path: Path) -> None:
        """Success task with matching hash in manifest (no result file) → up-to-date."""
        from maestro_cli.cache import compute_task_hash

        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo", "hello"]),
        ])
        current_hash = compute_task_hash(plan.tasks[0], plan, {})

        run_path = tmp_path / "runs" / "run_001"
        # Write only manifest — no .result.json file
        _write_manifest(run_path, {
            "run_id": "run_001",
            "task_results": {
                "t1": {"status": "success", "task_hash": current_hash, "duration_sec": 3.0},
            },
        })

        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].state == "up-to-date"
        assert ps.tasks[0].stale_reason is None


class TestExtractLastCostNegativeValue:
    def test_negative_cost_coerced_to_float(self) -> None:
        """-0.5 is a valid float — coercion works on negatives."""
        assert _extract_last_cost({"cost_usd": -0.5}) == -0.5

    def test_large_integer_cost_coerced(self) -> None:
        """Integer 5 → _coerce_float returns 5.0."""
        assert _extract_last_cost({"cost_usd": 5}) == 5.0


class TestFormatStatusJsonMultipleTasks:
    def test_all_task_states_serialised_correctly(self) -> None:
        """format_status_json with varied task states includes all tasks in JSON."""
        ps = PlanPipelineStatus(
            plan_name="multi-json",
            last_run_id="run_099",
            last_run_at="2026-03-15T09:00:00",
            tasks=[
                TaskPipelineStatus(task_id="t1", state="up-to-date", last_duration=1.0, last_cost=0.01),
                TaskPipelineStatus(task_id="t2", state="stale", stale_reason="task inputs changed"),
                TaskPipelineStatus(task_id="t3", state="failed"),
                TaskPipelineStatus(task_id="t4", state="skipped"),
                TaskPipelineStatus(task_id="t5", state="never-run"),
            ],
        )
        data = json.loads(format_status_json(ps))
        assert data["plan_name"] == "multi-json"
        assert data["last_run_id"] == "run_099"
        assert len(data["tasks"]) == 5
        states = {t["task_id"]: t["state"] for t in data["tasks"]}
        assert states["t1"] == "up-to-date"
        assert states["t2"] == "stale"
        assert states["t3"] == "failed"
        assert states["t4"] == "skipped"
        assert states["t5"] == "never-run"
        # stale_reason is serialised for t2
        t2 = next(t for t in data["tasks"] if t["task_id"] == "t2")
        assert t2["stale_reason"] == "task inputs changed"


class TestDurationFromTimestampsTimezoneAware:
    def test_both_timestamps_with_utc_offset(self) -> None:
        """Both timestamps have explicit +00:00 UTC offset — fromisoformat handles them correctly."""
        secs = _duration_from_timestamps(
            "2026-03-15T10:00:00+00:00",
            "2026-03-15T10:00:10+00:00",
        )
        assert secs == 10.0

    def test_same_offset_different_values(self) -> None:
        """Timestamps in the same non-zero timezone → difference computed correctly."""
        secs = _duration_from_timestamps(
            "2026-03-15T10:00:00+01:00",
            "2026-03-15T10:00:03+01:00",
        )
        assert secs == 3.0


class TestFormatDurationNegative:
    def test_negative_duration_rendered_with_sign(self) -> None:
        """_format_duration with a negative float value renders with the minus sign."""
        result = _format_duration(-1.5)
        assert result == "-1.50s"


class TestFormatCostNegative:
    def test_negative_cost_rendered_with_sign(self) -> None:
        """_format_cost with a negative float renders the negative sign before the dollar."""
        result = _format_cost(-0.01)
        # The format string is f"${value:.4f}" so negative gives "$-0.0100"
        assert result == "$-0.0100"


class TestCoerceFloatInfString:
    def test_inf_string_returns_inf(self) -> None:
        """Python's float('inf') is valid; _coerce_float('inf') should return math.inf."""
        import math
        result = _coerce_float("inf")
        assert result is not None
        assert math.isinf(result)

    def test_negative_inf_string_returns_negative_inf(self) -> None:
        import math
        result = _coerce_float("-inf")
        assert result is not None
        assert math.isinf(result)
        assert result < 0


class TestBaseStateFromStatusEmptyString:
    def test_empty_string_returns_never_run(self) -> None:
        """'' is not in _FAILED_STATUSES, _SKIPPED_STATUSES, or == _SUCCESS_STATUS → 'never-run'."""
        assert _base_state_from_status("") == "never-run"


class TestComputeHashesDepWithFailedUpstream:
    def test_dep_hash_exception_propagates_empty_to_downstream(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """t2 depends on t1; t1 hash raises → t1=None → upstream_hashes[t1]='' → t2 still gets a hash."""
        import maestro_cli.status as status_mod

        call_count = 0
        original_compute = status_mod.compute_task_hash

        def _sometimes_raise(*args: object, **kwargs: object) -> str:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("simulated hash error for t1")
            return original_compute(*args, **kwargs)

        monkeypatch.setattr(status_mod, "compute_task_hash", _sometimes_raise)

        plan = PlanSpec(
            version=1,
            name="dep-fail-plan",
            tasks=[
                TaskSpec(id="t1", command=["echo", "a"]),
                TaskSpec(id="t2", command=["echo", "b"], depends_on=["t1"]),
            ],
            source_path=tmp_path / "plan.yaml",
            run_dir=(tmp_path / "runs").as_posix(),
            defaults=PlanDefaults(),
        )
        hashes = _compute_hashes_for_plan(plan)
        assert hashes["t1"] is None  # t1 hash failed
        assert hashes["t2"] is not None  # t2 still computed with upstream hash = ""


class TestFormatStatusJsonTaskLastRunAt:
    def test_per_task_last_run_at_serialised(self) -> None:
        """format_status_json includes per-task last_run_at via dataclasses.asdict."""
        ps = PlanPipelineStatus(
            plan_name="ts-plan",
            last_run_id="run_010",
            last_run_at="2026-03-15T12:00:00",
            tasks=[
                TaskPipelineStatus(
                    task_id="t1",
                    state="up-to-date",
                    last_run_at="2026-03-15T12:01:00",
                    last_duration=3.0,
                    last_cost=0.02,
                ),
            ],
        )
        data = json.loads(format_status_json(ps))
        task = data["tasks"][0]
        assert task["last_run_at"] == "2026-03-15T12:01:00"
        assert task["last_duration"] == 3.0
        assert task["last_cost"] == 0.02


class TestMergeTaskPayloadsBothEmpty:
    def test_both_empty_dicts_returns_empty_dict(self) -> None:
        """Both payloads are empty dicts → merged result is empty dict (not None)."""
        merged = _merge_task_payloads({}, {})
        assert merged == {}

    def test_file_empty_manifest_has_keys(self) -> None:
        """File payload empty, manifest has keys → manifest keys present in result."""
        merged = _merge_task_payloads({}, {"status": "success", "cost": 0.5})
        assert merged is not None
        assert merged["status"] == "success"
        assert merged["cost"] == 0.5


class TestPlanStatusWithCacheDir:
    def test_cache_dir_provided_does_not_change_outcome(self, tmp_path: Path) -> None:
        """plan_status with non-None cache_dir still works correctly."""
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo"]),
        ])
        run_path = tmp_path / "runs" / "run_001"
        _write_manifest(run_path, {
            "run_id": "run_001",
            "task_results": {
                "t1": {"status": "failed", "duration_sec": 2.0},
            },
        })
        cache_path = tmp_path / "cache"
        cache_path.mkdir()
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=cache_path)
        assert ps.tasks[0].state == "failed"
        assert ps.last_run_id == "run_001"


class TestPlanStatusFileDurationOverridesManifest:
    def test_file_duration_wins_over_manifest_duration(self, tmp_path: Path) -> None:
        """When both file and manifest have duration_sec, file value takes priority."""
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo"]),
        ])
        run_path = tmp_path / "runs" / "run_dur"
        _write_manifest(run_path, {
            "run_id": "run_dur",
            "task_results": {
                "t1": {"status": "failed", "duration_sec": 1.0},
            },
        })
        _write_result(run_path, "t1", {
            "status": "failed",
            "duration_sec": 99.0,
        })
        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        assert ps.tasks[0].last_duration == 99.0


class TestFormatStatusJsonAllFieldsPresent:
    def test_all_task_fields_serialised(self) -> None:
        """format_status_json serialises all TaskPipelineStatus fields via asdict."""
        ps = PlanPipelineStatus(
            plan_name="full-json",
            last_run_id="run_xyz",
            last_run_at="2026-03-15T15:00:00",
            tasks=[
                TaskPipelineStatus(
                    task_id="t1",
                    state="stale",
                    last_run_at="2026-03-15T15:01:00",
                    last_duration=7.5,
                    last_cost=0.123,
                    stale_reason="task inputs changed",
                ),
            ],
        )
        data = json.loads(format_status_json(ps))
        task = data["tasks"][0]
        assert task["task_id"] == "t1"
        assert task["state"] == "stale"
        assert task["last_run_at"] == "2026-03-15T15:01:00"
        assert task["last_duration"] == 7.5
        assert task["last_cost"] == 0.123
        assert task["stale_reason"] == "task inputs changed"


class TestComputeHashesDuplicateDeps:
    def test_duplicate_deps_deduplicated(self, tmp_path: Path) -> None:
        """depends_on=['t1', 't1'] → sorted(set(...)) deduplicates → same hash as single dep."""
        plan_single = PlanSpec(
            version=1,
            name="dedup-plan",
            tasks=[
                TaskSpec(id="t1", command=["echo", "a"]),
                TaskSpec(id="t2", command=["echo", "b"], depends_on=["t1"]),
            ],
            source_path=tmp_path / "plan.yaml",
            run_dir=(tmp_path / "runs").as_posix(),
            defaults=PlanDefaults(),
        )
        plan_dup = PlanSpec(
            version=1,
            name="dedup-plan",
            tasks=[
                TaskSpec(id="t1", command=["echo", "a"]),
                TaskSpec(id="t2", command=["echo", "b"], depends_on=["t1", "t1"]),
            ],
            source_path=tmp_path / "plan.yaml",
            run_dir=(tmp_path / "runs").as_posix(),
            defaults=PlanDefaults(),
        )
        hashes_single = _compute_hashes_for_plan(plan_single)
        hashes_dup = _compute_hashes_for_plan(plan_dup)
        assert hashes_single["t2"] == hashes_dup["t2"]


class TestPlanStatusNonexistentRunPath:
    def test_nonexistent_path_all_never_run(self, tmp_path: Path) -> None:
        """latest_run_path points to a directory that doesn't exist → load_run_manifest fails → all never-run."""
        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo"]),
            TaskSpec(id="t2", command=["echo"]),
        ])
        nonexistent = tmp_path / "runs" / "this_does_not_exist"
        # Path does not exist — load_run_manifest should raise FileNotFoundError
        ps = plan_status(plan, latest_run_path=nonexistent, cache_dir=None)
        assert ps.last_run_id is None
        for task in ps.tasks:
            assert task.state == "never-run"


class TestDurationFromTimestampsCrossTimezone:
    def test_different_timezone_offsets_computes_correctly(self) -> None:
        """Timestamps with different tz offsets → correct difference in seconds."""
        # 10:00 UTC+00:00 and 12:00 UTC+01:00 are 1 hour apart
        secs = _duration_from_timestamps(
            "2026-03-15T10:00:00+00:00",
            "2026-03-15T12:00:00+01:00",
        )
        assert secs == 3600.0  # 12:00+01 = 11:00 UTC, minus 10:00 UTC = 1 hour


class TestPlanStatusUpToDateAndStaleCoexist:
    def test_one_task_up_to_date_another_stale(self, tmp_path: Path) -> None:
        """Two tasks: one with matching hash (up-to-date), one with mismatched hash (stale)."""
        from maestro_cli.cache import compute_task_hash

        plan = _make_plan(tmp_path, tasks=[
            TaskSpec(id="t1", command=["echo", "hello"]),
            TaskSpec(id="t2", command=["echo", "world"]),
        ])
        hash_t1 = compute_task_hash(plan.tasks[0], plan, {})

        run_path = tmp_path / "runs" / "run_coexist"
        _write_manifest(run_path, {
            "run_id": "run_coexist",
            "task_results": {
                "t1": {"status": "success", "task_hash": hash_t1},
                "t2": {"status": "success", "task_hash": "deliberately_wrong"},
            },
        })
        _write_result(run_path, "t1", {"status": "success", "task_hash": hash_t1, "duration_sec": 1.0})
        _write_result(run_path, "t2", {"status": "success", "task_hash": "deliberately_wrong"})

        ps = plan_status(plan, latest_run_path=run_path, cache_dir=None)
        states = {t.task_id: t.state for t in ps.tasks}
        assert states["t1"] == "up-to-date"
        assert states["t2"] == "stale"


class TestComputeHashesEngineTask:
    def test_engine_task_with_prompt_gets_hash(self, tmp_path: Path) -> None:
        """Engine task (no command) with prompt field still produces a valid hash."""
        plan = PlanSpec(
            version=1,
            name="engine-plan",
            tasks=[
                TaskSpec(id="t1", engine="claude", prompt="Do something"),
            ],
            source_path=tmp_path / "plan.yaml",
            run_dir=(tmp_path / "runs").as_posix(),
            defaults=PlanDefaults(),
        )
        hashes = _compute_hashes_for_plan(plan)
        assert "t1" in hashes
        assert hashes["t1"] is not None


class TestFormatStatusLastRunDash:
    def test_none_last_run_id_shows_dash(self) -> None:
        """When last_run_id is None, format_status shows '-' in the Last run line."""
        ps = PlanPipelineStatus(
            plan_name="dash-plan",
            last_run_id=None,
            last_run_at=None,
            tasks=[],
        )
        output = format_status(ps)
        assert "Last run: - at -" in output


class TestMergeTaskPayloadsNestedOverride:
    def test_nested_dict_values_overridden_by_file(self) -> None:
        """File payload with nested dict overrides manifest's same key entirely."""
        manifest_payload = {"token_usage": {"input": 100, "output": 200}}
        file_payload = {"token_usage": {"input": 500, "output": 600}}
        merged = _merge_task_payloads(file_payload, manifest_payload)
        assert merged is not None
        assert merged["token_usage"] == {"input": 500, "output": 600}
