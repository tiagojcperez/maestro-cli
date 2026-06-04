from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from maestro_cli import mcts
from maestro_cli.mcts import (
    _parse_timestamp,
    _resolve_tree_path,
    _score_record_value,
    _ucb1_score,
    append_tree_node,
    backpropagate_variant,
    create_workflow_variant,
    iter_variants,
    load_tree_index,
    select_variant_from_pool,
)
from maestro_cli.models import (
    PlanDefaults,
    PlanRunResult,
    PlanSpec,
    ScoreRecord,
    TaskSpec,
)


def _make_plan(tmp_path: Path, name: str = "demo") -> PlanSpec:
    source_path = tmp_path / f"{name}.yaml"
    source_path.write_text(f"version: 1\nname: {name}\n", encoding="utf-8")
    return PlanSpec(
        version=1,
        name=name,
        defaults=PlanDefaults(),
        tasks=[TaskSpec(id="a", command="echo a")],
        source_path=source_path,
    )


def _make_run_result(
    plan_name: str,
    *,
    success: bool,
    run_id: str,
    plan_hash: str | None = None,
) -> PlanRunResult:
    now = datetime.now(timezone.utc)
    return PlanRunResult(
        plan_name=plan_name,
        run_id=run_id,
        run_path=Path("/tmp/fake"),
        started_at=now,
        finished_at=now,
        success=success,
        plan_hash=plan_hash,
    )


def _score_record(
    plan_name: str,
    plan_hash: str,
    *,
    run_id: str,
    success: bool,
    quality_score: float | None = 1.0,
    cost_usd: float = 0.0,
    duration_sec: float = 0.0,
    timestamp: str,
) -> ScoreRecord:
    return ScoreRecord(
        plan_name=plan_name,
        plan_hash=plan_hash,
        run_id=run_id,
        success=success,
        cost_usd=cost_usd,
        quality_score=quality_score,
        duration_sec=duration_sec,
        timestamp=timestamp,
        valid_from=timestamp,
        recorded_at=timestamp,
        source_id=f"{run_id}:score",
        metadata={},
    )


class _StubRandom:
    def __init__(self, random_value: float) -> None:
        self._random_value = random_value

    def random(self) -> float:
        return self._random_value

    def choice(self, values: list[object]) -> object:
        return values[0]


class TestIterVariantsDedup:
    def test_shared_node_visited_once(self, tmp_path: Path) -> None:
        # A diamond: the same grandchild object is referenced by two parents,
        # so it appears on the traversal stack twice. The `seen` guard must
        # skip the duplicate (the `continue` branch).
        plan = _make_plan(tmp_path)
        root = create_workflow_variant(plan, node_id="root")
        left = create_workflow_variant(plan, parent=root, node_id="left")
        right = create_workflow_variant(plan, parent=root, node_id="right")
        shared = create_workflow_variant(plan, parent=left, node_id="shared")
        # Manually wire the shared node under the second parent too.
        right.children.append(shared)

        ordered = iter_variants(root)
        node_ids = [node.node_id for node in ordered]

        assert node_ids.count("shared") == 1
        assert set(node_ids) == {"root", "left", "right", "shared"}


class TestSelectVariantFromPoolValidation:
    def test_invalid_selection_policy_raises(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        with pytest.raises(ValueError, match="selection_policy"):
            select_variant_from_pool([variant], selection_policy="bogus")  # type: ignore[arg-type]

    def test_debug_prob_out_of_range_raises(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        with pytest.raises(ValueError, match="debug_prob"):
            select_variant_from_pool([variant], debug_prob=1.5)

    def test_negative_exploration_constant_raises(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        with pytest.raises(ValueError, match="exploration_constant"):
            select_variant_from_pool([variant], exploration_constant=-1.0)

    def test_empty_candidate_pool_returns_none(self) -> None:
        assert select_variant_from_pool([]) is None

    def test_unexecuted_branch_returns_best_unexecuted(self, tmp_path: Path) -> None:
        # No run_result and no score_record metadata -> the "unexecuted" branch
        # picks the highest-score unexecuted node.
        plan = _make_plan(tmp_path)
        low = create_workflow_variant(plan, node_id="low", score=0.2)
        high = create_workflow_variant(plan, node_id="high", score=0.7)

        selected = select_variant_from_pool([low, high], selection_policy="debug_prob")

        assert selected is high

    def test_debug_prob_returns_invalid_via_first_guard(self, tmp_path: Path) -> None:
        # All candidates executed and invalid (no valid nodes). The first guard
        # `invalid and (not valid or ...)` is satisfied by `not valid`, so it
        # returns a chooser.choice(invalid). chooser.choice returns values[0].
        plan = _make_plan(tmp_path)
        invalid_a = create_workflow_variant(
            plan,
            node_id="bad-a",
            run_result=_make_run_result(plan.name, success=False, run_id="bad-a-run"),
            score=0.1,
            is_valid=False,
        )
        invalid_b = create_workflow_variant(
            plan,
            node_id="bad-b",
            run_result=_make_run_result(plan.name, success=False, run_id="bad-b-run"),
            score=0.2,
            is_valid=False,
        )

        selected = select_variant_from_pool(
            [invalid_a, invalid_b],
            selection_policy="debug_prob",
            debug_prob=0.5,
            rng=_StubRandom(0.9),
        )
        assert selected is invalid_a

    def test_debug_prob_returns_best_valid(self, tmp_path: Path) -> None:
        # Mixed pool, random() above debug_prob and valid present -> the first
        # guard is False, falling through to the `if valid: max(valid, ...)`
        # branch which returns the highest-scoring valid node.
        plan = _make_plan(tmp_path)
        invalid = create_workflow_variant(
            plan,
            node_id="bad",
            run_result=_make_run_result(plan.name, success=False, run_id="bad-run"),
            score=0.1,
            is_valid=False,
        )
        valid_low = create_workflow_variant(
            plan,
            node_id="good-low",
            run_result=_make_run_result(plan.name, success=True, run_id="good-low-run"),
            score=0.4,
            is_valid=True,
        )
        valid_high = create_workflow_variant(
            plan,
            node_id="good-high",
            run_result=_make_run_result(plan.name, success=True, run_id="good-high-run"),
            score=0.8,
            is_valid=True,
        )

        selected = select_variant_from_pool(
            [invalid, valid_low, valid_high],
            selection_policy="debug_prob",
            debug_prob=0.0,
            rng=_StubRandom(0.9),
        )
        assert selected is valid_high


class TestBackpropagateValidation:
    def test_score_discount_out_of_range_raises(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        with pytest.raises(ValueError, match="score_discount"):
            backpropagate_variant(variant, score_discount=0.0)
        with pytest.raises(ValueError, match="score_discount"):
            backpropagate_variant(variant, score_discount=1.5)

    def test_builds_score_record_from_run_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # score_record is None but run_result is present -> build_score_record
        # path. Empty history means records == [built]. No exception, valid score.
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(
            plan,
            node_id="root",
            run_result=_make_run_result(plan.name, success=True, run_id="run-1"),
            is_valid=True,
        )
        monkeypatch.setattr("maestro_cli.mcts.load_score_history", lambda *a, **k: [])
        anchor = datetime.now(timezone.utc)

        score = backpropagate_variant(
            variant,
            history_limit=20,
            horizon_days=3650,
            anchor_time=anchor,
        )

        assert 0.0 <= score <= 1.0
        assert variant.visits == 1
        assert "score_record" in variant.metadata
        assert variant.metadata["history_samples"] == 1

    def test_no_record_and_no_history_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # score_record None AND run_result None AND empty history -> no records.
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        monkeypatch.setattr("maestro_cli.mcts.load_score_history", lambda *a, **k: [])

        with pytest.raises(ValueError, match="Cannot backpropagate"):
            backpropagate_variant(variant)


class TestLoadTreeIndex:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        # tree.jsonl does not exist -> empty list.
        assert load_tree_index(tmp_path / "no-run") == []

    def test_skips_blank_and_malformed_lines(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        tree_path = run_dir / "tree.jsonl"
        tree_path.write_text(
            "\n"  # blank line -> continue
            "   \n"  # whitespace-only -> continue
            "not-json{{{\n"  # JSONDecodeError -> continue
            '"a string not a dict"\n'  # parses but not a dict -> skipped
            '{"node_id": "keep"}\n',  # valid dict row
            encoding="utf-8",
        )

        rows = load_tree_index(run_dir)

        assert rows == [{"node_id": "keep"}]

    def test_oserror_returns_empty(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Drive the `except OSError: return []` branch by making read_text raise.
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        tree_path = run_dir / "tree.jsonl"
        tree_path.write_text('{"node_id": "x"}\n', encoding="utf-8")

        real_read_text = Path.read_text

        def _boom(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == "tree.jsonl":
                raise OSError("simulated read failure")
            return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", _boom)

        assert load_tree_index(run_dir) == []


class TestResolveTreePath:
    def test_path_already_named_tree_file_returned_as_is(self, tmp_path: Path) -> None:
        direct = tmp_path / "tree.jsonl"
        resolved = _resolve_tree_path(direct)
        assert resolved == direct

    def test_directory_path_gets_tree_filename_appended(self, tmp_path: Path) -> None:
        resolved = _resolve_tree_path(tmp_path)
        assert resolved == tmp_path / "tree.jsonl"

    def test_append_then_load_with_direct_tree_path(self, tmp_path: Path) -> None:
        # append_tree_node + load_tree_index both routed through a path already
        # ending in tree.jsonl exercises the early-return branch end to end.
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root")
        tree_file = tmp_path / "tree.jsonl"

        append_tree_node(tree_file, variant)
        rows = load_tree_index(tree_file)

        assert len(rows) == 1
        assert rows[0]["node_id"] == "root"


class TestScoreRecordValue:
    def test_quality_none_uses_success_flag(self) -> None:
        anchor = datetime.now(timezone.utc)
        timestamp = anchor.isoformat()
        # quality_score None + success True -> quality treated as 1.0.
        record_success = _score_record(
            "demo",
            "hash",
            run_id="r1",
            success=True,
            quality_score=None,
            timestamp=timestamp,
        )
        value_success = _score_record_value(record_success, anchor=anchor, horizon_days=0)

        record_fail = _score_record(
            "demo",
            "hash",
            run_id="r2",
            success=False,
            quality_score=None,
            timestamp=timestamp,
        )
        value_fail = _score_record_value(record_fail, anchor=anchor, horizon_days=0)

        # With horizon_days=0 the decay branch is skipped (base_score returned).
        assert value_success > value_fail
        assert value_fail == pytest.approx(0.10 * (1.0 / 1.0) + 0.05 * 1.0)

    def test_horizon_zero_returns_base_without_decay(self) -> None:
        # A very old timestamp; horizon_days <= 0 must skip decay so the value
        # equals the undecayed base score.
        anchor = datetime.now(timezone.utc)
        old_timestamp = "2000-01-01T00:00:00+00:00"
        record = _score_record(
            "demo",
            "hash",
            run_id="old",
            success=True,
            quality_score=1.0,
            timestamp=old_timestamp,
        )
        value = _score_record_value(record, anchor=anchor, horizon_days=0)
        # base = 0.60*1 + 0.25*1 + 0.10*1 + 0.05*1 = 1.0 (cost/duration 0).
        assert value == pytest.approx(1.0)


class TestUcb1Score:
    def test_zero_visits_returns_infinity(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path)
        variant = create_workflow_variant(plan, node_id="root", score=0.5)
        variant.visits = 0
        assert _ucb1_score(variant, exploration_constant=1.4) == float("inf")


class TestParseTimestamp:
    def test_naive_timestamp_gets_utc(self) -> None:
        # No timezone info -> tzinfo assumed UTC.
        parsed = _parse_timestamp("2026-01-01T12:00:00")
        assert parsed.tzinfo is timezone.utc
        assert parsed.hour == 12

    def test_aware_timestamp_converted_to_utc(self) -> None:
        # +02:00 offset -> normalized to UTC (10:00).
        parsed = _parse_timestamp("2026-01-01T12:00:00+02:00")
        assert parsed.tzinfo == timezone.utc
        assert parsed.hour == 10
