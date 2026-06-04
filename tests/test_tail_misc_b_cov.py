from __future__ import annotations

import random
from pathlib import Path

from maestro_cli import diff as diff_mod
from maestro_cli import report as report_mod
from maestro_cli.mcts import create_workflow_variant, select_variant_from_pool
from maestro_cli.models import (
    KnowledgeWriteOutcome,
    PlanDefaults,
    PlanRunResult,
    PlanSpec,
    TaskSpec,
)
from maestro_cli.suggest import _analyze_task


# ---------------------------------------------------------------------------
# diff.py:72 — _coerce_int returns None for a value that is not None, not int,
# and not (float | str): the final `return None` fall-through.
# ---------------------------------------------------------------------------
class TestDiffCoerceIntFallthrough:
    def test_uncoercible_type_returns_none(self) -> None:
        # A list is not None/int/float/str → falls through to final return None.
        assert diff_mod._coerce_int(["not", "coercible"]) is None

    def test_dict_returns_none(self) -> None:
        assert diff_mod._coerce_int({"k": "v"}) is None

    def test_int_still_works(self) -> None:
        # Guard: ensure normal coercion still behaves.
        assert diff_mod._coerce_int(5) == 5
        assert diff_mod._coerce_int("7") == 7
        assert diff_mod._coerce_int(None) is None


# ---------------------------------------------------------------------------
# report.py:38 — _coerce_int final `return None` for a non-coercible type.
# ---------------------------------------------------------------------------
class TestReportCoerceIntFallthrough:
    def test_uncoercible_type_returns_none(self) -> None:
        assert report_mod._coerce_int(["x"]) is None

    def test_bytes_returns_none(self) -> None:
        assert report_mod._coerce_int(b"123") is None

    def test_int_and_str_still_work(self) -> None:
        assert report_mod._coerce_int(3) == 3
        assert report_mod._coerce_int("9") == 9
        assert report_mod._coerce_int(None) is None


# ---------------------------------------------------------------------------
# models.py:2016 — KnowledgeWriteOutcome.to_dict() body.
# ---------------------------------------------------------------------------
class TestKnowledgeWriteOutcomeToDict:
    def test_to_dict_serializes_all_fields(self) -> None:
        outcome = KnowledgeWriteOutcome(
            task_id="task-1",
            kind="failure_pattern",
            operation="insert",
            outcome="stored",
            trust_label="trusted",
            instructionality_score=0.123456,
            source_type="task_output",
            source_id="src-42",
        )
        result = outcome.to_dict()
        assert result["task_id"] == "task-1"
        assert result["kind"] == "failure_pattern"
        assert result["operation"] == "insert"
        assert result["outcome"] == "stored"
        assert result["trust_label"] == "trusted"
        # instructionality_score is rounded to 3 decimals.
        assert result["instructionality_score"] == round(0.123456, 3)
        assert result["source_type"] == "task_output"
        assert result["source_id"] == "src-42"


# ---------------------------------------------------------------------------
# suggest.py:267 — `continue` in the all_durations collection loop when a run's
# `task_results` is not a dict. Requires task_runs non-empty (so the function
# does not early-return) plus at least one run with non-dict task_results.
# ---------------------------------------------------------------------------
class TestAnalyzeTaskDurationLoopContinue:
    def test_non_dict_task_results_in_duration_loop_is_skipped(self) -> None:
        task_id = "build"
        # One valid run gives us a non-empty task_runs (avoids early return).
        valid_run = {
            "task_results": {
                task_id: {
                    "status": "success",
                    "cost_usd": 0.10,
                    "duration_sec": 5.0,
                    "retry_count": 0,
                }
            },
            "total_cost_usd": 0.10,
        }
        # A second run whose task_results is NOT a dict → hits the `continue`.
        bad_run = {"task_results": ["this", "is", "not", "a", "dict"]}
        # A third run missing task_results entirely → also non-dict (None).
        missing_run: dict[str, object] = {"total_cost_usd": 0.0}

        runs = [valid_run, bad_run, missing_run]
        task_spec_map = {task_id: TaskSpec(id=task_id, command="echo build")}

        suggestions = _analyze_task(task_id, task_spec_map, runs)
        # Function should complete without error; the bad/missing runs are skipped
        # during duration collection. Return type is a list regardless of content.
        assert isinstance(suggestions, list)

    def test_all_runs_valid_does_not_crash(self) -> None:
        # Sanity counterpart: every run has a dict task_results.
        task_id = "build"
        runs = [
            {
                "task_results": {
                    task_id: {
                        "status": "success",
                        "cost_usd": 0.10,
                        "duration_sec": 5.0,
                        "retry_count": 0,
                    }
                },
                "total_cost_usd": 0.10,
            }
        ]
        task_spec_map = {task_id: TaskSpec(id=task_id, command="echo build")}
        assert isinstance(_analyze_task(task_id, task_spec_map, runs), list)


# ---------------------------------------------------------------------------
# mcts.py:253-255 — fall-through tail of select_variant_from_pool under the
# debug_prob policy. These lines are defensive and unreachable in practice:
# once `unexecuted` is empty, candidates partition exactly into invalid+valid
# (a node is either valid or invalid). When `valid` is empty and `invalid` is
# non-empty, 's `(not valid or ...)` is True so it returns at .
# When `valid` is non-empty it returns at . There is no input that
# reaches . We still exercise the surrounding reachable branches
# to lock behaviour.
# ---------------------------------------------------------------------------
def _make_plan(tmp_path: Path, name: str) -> PlanSpec:
    source_path = tmp_path / f"{name}.yaml"
    source_path.write_text(f"version: 1\nname: {name}\n", encoding="utf-8")
    return PlanSpec(
        version=1,
        name=name,
        defaults=PlanDefaults(),
        tasks=[TaskSpec(id="a", command="echo a")],
        source_path=source_path,
    )


def _executed_variant(
    tmp_path: Path, name: str, *, is_valid: bool, score: float
) -> object:
    plan = _make_plan(tmp_path, name)
    result = PlanRunResult(
        plan_name=name,
        run_id=name,
        run_path=tmp_path,
        started_at=None,
        finished_at=None,
        success=is_valid,
    )
    return create_workflow_variant(
        plan,
        run_result=result,
        score=score,
        is_valid=is_valid,
        node_id=name,
    )


class TestSelectVariantFromPoolReachableBranches:
    def test_all_invalid_returns_an_invalid_node(self, tmp_path: Path) -> None:
        # valid empty, invalid non-empty → returns at the line-249/250 branch.
        v1 = _executed_variant(tmp_path, "i1", is_valid=False, score=0.1)
        v2 = _executed_variant(tmp_path, "i2", is_valid=False, score=0.2)
        chosen = select_variant_from_pool(
            [v1, v2],
            selection_policy="debug_prob",
            debug_prob=1.0,
            rng=random.Random(0),
        )
        assert chosen in (v1, v2)

    def test_valid_present_returns_best_valid(self, tmp_path: Path) -> None:
        # invalid present + valid present, debug_prob low so the valid branch wins.
        invalid = _executed_variant(tmp_path, "bad", is_valid=False, score=0.9)
        good_low = _executed_variant(tmp_path, "good_low", is_valid=True, score=0.3)
        good_high = _executed_variant(tmp_path, "good_high", is_valid=True, score=0.7)

        class _AlwaysHigh:
            def random(self) -> float:
                return 1.0  # >= debug_prob → skip the invalid branch

            def choice(self, seq: list) -> object:
                return seq[0]

        chosen = select_variant_from_pool(
            [invalid, good_low, good_high],
            selection_policy="debug_prob",
            debug_prob=0.0,
            rng=_AlwaysHigh(),
        )
        # Highest-score valid node wins ( branch).
        assert chosen is good_high

    def test_empty_pool_returns_none(self) -> None:
        assert select_variant_from_pool([], selection_policy="debug_prob") is None
