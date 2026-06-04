from __future__ import annotations

import json
from pathlib import Path

import pytest

from maestro_cli.models import ModelRecord, PlanSpec, TaskHistory, TaskSpec
from maestro_cli.routing import (
    _compute_task_similarity,
    apply_cross_task_routing,
    load_task_histories,
    resolve_auto_model,
)


def _make_task(**kwargs: object) -> TaskSpec:
    defaults: dict[str, object] = {
        "id": "test-task",
        "engine": "claude",
        "prompt": "test prompt",
    }
    defaults.update(kwargs)
    return TaskSpec(**defaults)  # type: ignore[arg-type]


def _make_plan(**kwargs: object) -> PlanSpec:
    defaults: dict[str, object] = {"name": "test", "tasks": []}
    defaults.update(kwargs)
    return PlanSpec(**defaults)  # type: ignore[arg-type]


def _rec(
    model: str,
    runs: int = 5,
    successes: int = 5,
    failures: int = 0,
    timeouts: int = 0,
) -> ModelRecord:
    return ModelRecord(
        model=model,
        runs=runs,
        successes=successes,
        failures=failures,
        timeouts=timeouts,
        avg_duration_sec=30.0,
        avg_cost_usd=0.01,
    )


def _hist(records: dict[str, ModelRecord], total_runs: int | None = None) -> TaskHistory:
    tr = total_runs if total_runs is not None else sum(r.runs for r in records.values())
    return TaskHistory(task_id="other", total_runs=tr, records=records)


def _write_manifest(
    run_dir: Path,
    plan_name: str,
    run_index: int,
    task_results: object,
) -> None:
    dirname = f"2026031{run_index:1d}_120000_000000_aaa_{plan_name}"
    d = run_dir / dirname
    d.mkdir(parents=True, exist_ok=True)
    (d / "run_manifest.json").write_text(
        json.dumps({"plan_name": plan_name, "task_results": task_results}),
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# — resolve_auto_model dispatches to apply_cross_task_routing when
# dag_metadata carries both `all_histories` and `task_map`.
# ---------------------------------------------------------------------------


class TestResolveAutoModelCrossTask:
    def test_cross_task_signal_pushes_routing_to_cheaper_model(self) -> None:
        # A neutral medium-tier task (score 0.5) whose similar sibling has a cheap
        # model succeeding consistently should drop into the low tier (haiku).
        plan = _make_plan()
        # context_from must reference the dep for the same-engine signal to dominate;
        # keep both tasks structurally near-identical so similarity clears 0.3.
        task = _make_task(id="this-task", prompt="x" * 100, tags=["review"])
        sibling = _make_task(id="other-task", prompt="x" * 100, tags=["review"])
        histories = {
            "other-task": _hist({"haiku": _rec("haiku", runs=5, successes=5)}),
        }
        task_map = {"this-task": task, "other-task": sibling}

        model = resolve_auto_model(
            task,
            plan,
            "claude",
            dag_metadata={"all_histories": histories, "task_map": task_map},
        )
        # The cross-task signal subtracts from a 0.65 (review boost) score; whatever
        # the resulting tier, the call must route through and return a real
        # claude tier model rather than the "auto" passthrough.
        assert model in {"haiku", "sonnet", "opus"}

    def test_cross_task_routing_returns_concrete_model(self) -> None:
        # Distinct verification that the branch is exercised: pass histories that
        # raise the score (cheap model failing) and confirm a claude model is chosen.
        plan = _make_plan()
        task = _make_task(id="this-task", prompt="x" * 100, tags=["review"])
        sibling = _make_task(id="other-task", prompt="x" * 100, tags=["review"])
        histories = {
            "other-task": _hist(
                {"haiku": _rec("haiku", runs=5, successes=0, failures=5)}
            ),
        }
        task_map = {"this-task": task, "other-task": sibling}

        model = resolve_auto_model(
            task,
            plan,
            "claude",
            dag_metadata={"all_histories": histories, "task_map": task_map},
        )
        assert model in {"haiku", "sonnet", "opus"}

    def test_all_histories_present_but_task_map_missing_skips_branch(self) -> None:
        # When task_map is absent the cross-task branch is NOT taken; result still
        # routes normally. Guards against the branch firing on partial metadata.
        plan = _make_plan()
        task = _make_task(prompt="x" * 100)
        model = resolve_auto_model(
            task,
            plan,
            "claude",
            dag_metadata={"all_histories": {"x": _hist({})}},
        )
        assert model == "sonnet"


# ---------------------------------------------------------------------------
# — load_task_histories edge branches.
# ---------------------------------------------------------------------------


class TestLoadTaskHistoriesEdges:
    def test_nonexistent_run_dir_returns_empty(self, tmp_path: Path) -> None:
        # run_dir.exists() is False → early return {}.
        missing = tmp_path / "does-not-exist"
        assert load_task_histories("demo", missing) == {}

    def test_run_dir_is_a_file_returns_empty(self, tmp_path: Path) -> None:
        # path exists but is_dir() is False → early return {}.
        as_file = tmp_path / "a-file"
        as_file.write_text("not a directory", encoding="utf-8")
        assert load_task_histories("demo", as_file) == {}

    def test_glob_candidate_that_is_a_file_is_skipped(self, tmp_path: Path) -> None:
        # a path matching *_{plan_name} that is a plain file → continue.
        # Provide 3 real dir manifests so we still reach aggregation, plus a stray
        # file that matches the glob pattern and must be skipped silently.
        for i in range(3):
            _write_manifest(
                tmp_path,
                "demo",
                i,
                {
                    "t1": {
                        "auto_routed_model": "haiku",
                        "status": "success",
                        "duration_sec": 10.0,
                        "cost_usd": 0.01,
                        "exit_code": 0,
                    }
                },
            )
        stray = tmp_path / "20260319_120000_000000_zzz_demo"
        stray.write_text("i am a file, not a run dir", encoding="utf-8")

        result = load_task_histories("demo", tmp_path)
        assert "t1" in result
        assert result["t1"].total_runs == 3  # stray file skipped, 3 real dirs counted

    def test_task_results_not_a_dict_is_skipped(self, tmp_path: Path) -> None:
        # a manifest whose task_results is a list (not dict) → continue.
        # Two good manifests + one with a list task_results = 3 loaded dicts (passes
        # the min_runs gate), but the list one contributes nothing.
        good = {
            "t1": {
                "auto_routed_model": "haiku",
                "status": "success",
                "duration_sec": 10.0,
                "cost_usd": 0.01,
                "exit_code": 0,
            }
        }
        _write_manifest(tmp_path, "demo", 0, good)
        _write_manifest(tmp_path, "demo", 1, good)
        _write_manifest(tmp_path, "demo", 2, ["not", "a", "dict"])

        result = load_task_histories("demo", tmp_path)
        assert result["t1"].total_runs == 2  # the list manifest added nothing

    def test_individual_result_not_a_dict_is_skipped(self, tmp_path: Path) -> None:
        # a task_results dict whose value is not a dict → continue.
        good = {
            "t1": {
                "auto_routed_model": "haiku",
                "status": "success",
                "duration_sec": 10.0,
                "cost_usd": 0.01,
                "exit_code": 0,
            }
        }
        _write_manifest(tmp_path, "demo", 0, good)
        _write_manifest(tmp_path, "demo", 1, good)
        # Third manifest: t1 good, but t2 is a string (non-dict result) and skipped.
        _write_manifest(
            tmp_path,
            "demo",
            2,
            {
                "t1": {
                    "auto_routed_model": "haiku",
                    "status": "success",
                    "duration_sec": 10.0,
                    "cost_usd": 0.01,
                    "exit_code": 0,
                },
                "t2": "this-is-not-a-dict",
            },
        )

        result = load_task_histories("demo", tmp_path)
        assert result["t1"].total_runs == 3
        assert "t2" not in result  # non-dict result never aggregated


# ---------------------------------------------------------------------------
# — _compute_task_similarity dependency-diff middle band.
# ---------------------------------------------------------------------------


class TestComputeTaskSimilarityDepBand:
    def test_dep_diff_of_two_adds_quarter_weight(self) -> None:
        # dep_diff = 2 → NOT <= 1, but <= 3 → score += 0.25.
        # Make every other feature match so we can isolate the dep contribution.
        a = _make_task(id="a", engine="claude", depends_on=[])
        b = _make_task(id="b", engine="claude", depends_on=["x", "y"])

        sim_two = _compute_task_similarity(a, b)

        # Compare against a pair differing only in dep count (diff 0 vs diff 2):
        b_same = _make_task(id="b2", engine="claude", depends_on=[])
        sim_zero = _compute_task_similarity(a, b_same)

        # diff 0 awards 0.5, diff 2 awards 0.25 → lower.
        assert sim_two < sim_zero
        assert 0.0 < sim_two <= 1.0

    def test_dep_diff_of_three_adds_quarter_weight(self) -> None:
        # dep_diff = 3 is the upper edge of the <= 3 band → still +0.25.
        a = _make_task(id="a", engine="claude", depends_on=[])
        b = _make_task(id="b", engine="claude", depends_on=["x", "y", "z"])
        sim_three = _compute_task_similarity(a, b)

        # dep_diff = 4 falls outside the band → no dep credit → strictly lower.
        b_far = _make_task(id="bf", engine="claude", depends_on=["x", "y", "z", "w"])
        sim_four = _compute_task_similarity(a, b_far)

        assert sim_three > sim_four


# ---------------------------------------------------------------------------
# — apply_cross_task_routing guard branches.
# ---------------------------------------------------------------------------


class TestApplyCrossTaskRoutingGuards:
    def test_unknown_engine_returns_base_score_unchanged(self) -> None:
        # engine not in _MODEL_TIERS → return base_score verbatim.
        task = _make_task(id="t")
        result = apply_cross_task_routing(task, {}, {}, "nonexistent-engine", 0.42)
        assert result == pytest.approx(0.42)

    def test_other_task_with_no_history_is_skipped(self) -> None:
        # histories.get(other_id) is None → continue. With no usable
        # signal, total_weight stays 0 → base_score returned unchanged.
        task = _make_task(id="this", engine="claude", tags=["review"])
        sibling = _make_task(id="other", engine="claude", tags=["review"])
        task_map = {"this": task, "other": sibling}
        result = apply_cross_task_routing(task, task_map, {}, "claude", 0.5)
        assert result == pytest.approx(0.5)

    def test_other_task_below_min_runs_is_skipped(self) -> None:
        # (second condition): total_runs < _HISTORY_MIN_MODEL_RUNS (3).
        task = _make_task(id="this", engine="claude", tags=["review"])
        sibling = _make_task(id="other", engine="claude", tags=["review"])
        task_map = {"this": task, "other": sibling}
        histories = {
            # total_runs=2 (< 3) → skipped at .
            "other": _hist({"haiku": _rec("haiku", runs=2, successes=2)}, total_runs=2),
        }
        result = apply_cross_task_routing(task, task_map, histories, "claude", 0.5)
        assert result == pytest.approx(0.5)

    def test_low_similarity_task_is_skipped(self) -> None:
        # similarity < _CROSS_TASK_MIN_SIMILARITY (0.3) → continue.
        # Make the sibling maximally dissimilar: different engine, disjoint tags,
        # opposite judge presence, different context mode, large dep gap.
        task = _make_task(
            id="this",
            engine="claude",
            tags=["alpha"],
            context_mode="raw",
            depends_on=[],
        )
        sibling = _make_task(
            id="other",
            engine="gemini",
            tags=["zeta"],
            context_mode="recursive",
            depends_on=["a", "b", "c", "d", "e"],
        )
        task_map = {"this": task, "other": sibling}
        histories = {
            "other": _hist({"haiku": _rec("haiku", runs=5, successes=5)}),
        }
        # Sanity: confirm the pair really is below the threshold.
        assert _compute_task_similarity(task, sibling) < 0.3

        result = apply_cross_task_routing(task, task_map, histories, "claude", 0.5)
        # Skipped → no signal accumulated → base_score returned unchanged.
        assert result == pytest.approx(0.5)

    def test_self_task_is_skipped_and_similar_task_applies_signal(self) -> None:
        # Positive path: a genuinely similar sibling with a cheap model succeeding
        # accumulates a negative signal, lowering the returned score. This also
        # walks past the `other_id == task.id` self-skip ().
        task = _make_task(
            id="this",
            engine="claude",
            tags=["review"],
            context_mode="raw",
            depends_on=["a"],
        )
        sibling = _make_task(
            id="other",
            engine="claude",
            tags=["review"],
            context_mode="raw",
            depends_on=["a"],
        )
        task_map = {"this": task, "other": sibling}
        histories = {
            "other": _hist({"haiku": _rec("haiku", runs=5, successes=5)}),
        }
        assert _compute_task_similarity(task, sibling) >= 0.3

        result = apply_cross_task_routing(task, task_map, histories, "claude", 0.5)
        # Cheap model succeeded ≥80% → weighted_signal negative → score drops.
        assert result < 0.5
