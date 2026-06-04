from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from maestro_cli.benchmark import (
    BenchmarkSample,
    _candidate_number,
    _run_samples,
    _scenario_candidate_number,
    _selected_candidate_summary,
)
from maestro_cli.models import PlanSpec, ReplanAttempt, ReplanState


# ---------------------------------------------------------------------------
# _candidate_number — source_path is None branch (returns None)
# ---------------------------------------------------------------------------


class TestCandidateNumber:
    def test_returns_none_when_source_path_is_none(self) -> None:
        plan = PlanSpec(name="no-source", source_path=None)
        assert _candidate_number(plan) is None

    def test_returns_none_when_name_has_no_candidate_marker(self, tmp_path: Path) -> None:
        path = tmp_path / "plain-plan.yaml"
        plan = PlanSpec(name="plain", source_path=path)
        assert _candidate_number(plan) is None

    def test_extracts_number_from_candidate_name(self, tmp_path: Path) -> None:
        path = tmp_path / "candidate-7-foo.yaml"
        plan = PlanSpec(name="cand", source_path=path)
        assert _candidate_number(plan) == 7


# ---------------------------------------------------------------------------
# _scenario_candidate_number — source_path is None branch (returns None)
# ---------------------------------------------------------------------------


class TestScenarioCandidateNumber:
    def test_returns_none_when_source_path_is_none(self, tmp_path: Path) -> None:
        plan = PlanSpec(name="no-source", source_path=None)
        assert _scenario_candidate_number(plan, tmp_path) is None

    def test_returns_none_when_parent_is_scenario_dir(self, tmp_path: Path) -> None:
        path = tmp_path / "candidate-3-x.yaml"
        plan = PlanSpec(name="cand", source_path=path)
        # parent == scenario_dir → the early-return-None branch
        assert _scenario_candidate_number(plan, tmp_path) is None

    def test_returns_candidate_number_for_nested_path(self, tmp_path: Path) -> None:
        nested = tmp_path / "runs"
        nested.mkdir()
        path = nested / "candidate-4-x.yaml"
        plan = PlanSpec(name="cand", source_path=path)
        assert _scenario_candidate_number(plan, tmp_path) == 4

    def test_falls_back_to_one_when_no_candidate_marker(self, tmp_path: Path) -> None:
        nested = tmp_path / "runs"
        nested.mkdir()
        path = nested / "plain.yaml"
        plan = PlanSpec(name="cand", source_path=path)
        # _candidate_number returns None → `or 1` → 1
        assert _scenario_candidate_number(plan, tmp_path) == 1


# ---------------------------------------------------------------------------
# _selected_candidate_summary — defensive AssertionError when the selected
# candidate id is not present among the recorded candidate variants.
# ---------------------------------------------------------------------------


class TestSelectedCandidateSummary:
    def test_returns_matching_candidate(self) -> None:
        attempt = ReplanAttempt(attempt_number=1)
        attempt.selected_candidate_id = "node-2"
        attempt.candidate_variants = [
            {"node_id": "node-1", "run_id": "a"},
            {"node_id": "node-2", "run_id": "b"},
        ]
        state = ReplanState(attempts=[attempt])
        summary = _selected_candidate_summary(state)
        assert summary["run_id"] == "b"

    def test_raises_when_selected_id_missing_from_variants(self) -> None:
        attempt = ReplanAttempt(attempt_number=1)
        attempt.selected_candidate_id = "node-missing"
        attempt.candidate_variants = [
            {"node_id": "node-1", "run_id": "a"},
            {"node_id": "node-2", "run_id": "b"},
        ]
        state = ReplanState(attempts=[attempt])
        with pytest.raises(AssertionError, match="selected candidate summary"):
            _selected_candidate_summary(state)


# ---------------------------------------------------------------------------
# _run_samples — warmup cleanup branch: a warmup sample with a cleanup_path
# must be rmtree'd before the measured iterations begin.
# ---------------------------------------------------------------------------


class TestRunSamplesWarmupCleanup:
    def test_warmup_cleanup_path_is_removed(self, tmp_path: Path) -> None:
        created: list[Path] = []
        phase = {"warmup": True}

        def runner() -> BenchmarkSample:
            if phase["warmup"]:
                # First call is the warmup; return a real dir to be cleaned up.
                phase["warmup"] = False
                target = tmp_path / "warmup-dir"
                target.mkdir(exist_ok=True)
                (target / "f.txt").write_text("data", encoding="utf-8")
                created.append(target)
                return BenchmarkSample(cleanup_path=target)
            return BenchmarkSample()

        result = _run_samples("loader", runner, iterations=1, warmups=1)

        assert result.name == "loader"
        assert result.warmups == 1
        assert len(result.samples_ms) == 1
        # The warmup directory must have been removed by the warmup-cleanup branch.
        assert created and not created[0].exists()

    def test_warmup_with_none_cleanup_path_skips_rmtree(self) -> None:
        # warmup sample without a cleanup_path: the `if cleanup_path is not None`
        # guard is False, so no rmtree — exercised here as the complementary case.
        def runner() -> BenchmarkSample:
            return BenchmarkSample()

        result = _run_samples("cache", runner, iterations=1, warmups=1)
        assert result.warmups == 1
        assert len(result.samples_ms) == 1

    def test_warmup_cleanup_tolerates_missing_dir(self, tmp_path: Path) -> None:
        # shutil.rmtree(..., ignore_errors=True) is used, so a stale path is fine.
        stale = tmp_path / "already-gone"

        def runner() -> BenchmarkSample:
            return BenchmarkSample(cleanup_path=stale)

        # Should not raise even though `stale` never existed.
        result = _run_samples("scheduler", runner, iterations=1, warmups=1)
        assert result.warmups == 1
        assert not stale.exists()
        # Sanity: rmtree on a missing path with ignore_errors is a no-op.
        shutil.rmtree(stale, ignore_errors=True)
