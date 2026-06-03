from __future__ import annotations

import json
import random
from datetime import datetime
from pathlib import Path

import pytest

from maestro_cli.cache import compute_plan_hash
from maestro_cli.eventsource import replay_events, verify_chain
from maestro_cli.knowledge import build_score_record, store_knowledge, store_score_history
from maestro_cli.loader import load_plan
from maestro_cli.models import KnowledgeRecord, PlanRunResult, PlanSpec, ScoreRecord, TaskResult, WorkflowVariant
from maestro_cli.replan import (
    ReplanAttempt,
    ReplanState,
    _build_analysis_prompt,
    _detect_exit_loop,
    _extract_failed_state,
    _parse_corrected_yaml,
    _select_replan_search_variant,
    _show_plan_diff,
)


class TestReplan:
    # ──────────────────────────── _parse_corrected_yaml ──────────────────────

    def test_parse_corrected_yaml_valid(self) -> None:
        response = "Here is the fix:\n```yaml\nversion: 1\nname: test\ntasks: []\n```\n"
        result = _parse_corrected_yaml(response)
        assert result == "version: 1\nname: test\ntasks: []"

    def test_parse_corrected_yaml_no_fence(self) -> None:
        response = "No fences here, just plain text."
        result = _parse_corrected_yaml(response)
        assert result is None

    def test_parse_corrected_yaml_empty_fence(self) -> None:
        # A generic fence with only whitespace — all three patterns match but
        # produce an empty string after strip(), so None is returned.
        response = "```\n   \n```"
        result = _parse_corrected_yaml(response)
        assert result is None

    # ──────────────────────────── _detect_exit_loop ──────────────────────────

    def test_detect_exit_loop_same_failures(self) -> None:
        a1 = ReplanAttempt(attempt_number=1, failed_task_ids=["task-a", "task-b"])
        a2 = ReplanAttempt(attempt_number=2, failed_task_ids=["task-b", "task-a"])
        assert _detect_exit_loop([a1, a2]) is True

    def test_detect_exit_loop_different_failures(self) -> None:
        a1 = ReplanAttempt(attempt_number=1, failed_task_ids=["task-a"])
        a2 = ReplanAttempt(attempt_number=2, failed_task_ids=["task-b"])
        assert _detect_exit_loop([a1, a2]) is False

    def test_detect_exit_loop_single_attempt(self) -> None:
        a1 = ReplanAttempt(attempt_number=1, failed_task_ids=["task-a"])
        assert _detect_exit_loop([a1]) is False

    # ──────────────────────────── _extract_failed_state ──────────────────────

    def test_extract_failed_state(self, tmp_path: Path) -> None:
        now = datetime.now()
        run_result = PlanRunResult(
            plan_name="test-plan",
            run_id="run-001",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "task-ok": TaskResult(
                    task_id="task-ok",
                    status="success",
                    exit_code=0,
                ),
                "task-fail": TaskResult(
                    task_id="task-fail",
                    status="failed",
                    exit_code=1,
                    message="Something broke",
                    stdout_tail="error output",
                ),
                "task-skipped": TaskResult(
                    task_id="task-skipped",
                    status="skipped",
                ),
            },
        )

        state = _extract_failed_state(run_result)

        assert "task-fail" in state["failed_task_ids"]
        assert "task-ok" in state["passed_task_ids"]
        assert "task-skipped" not in state["failed_task_ids"]
        assert "task-skipped" not in state["passed_task_ids"]
        assert state["error_messages"]["task-fail"] == "Something broke"
        assert state["stdout_tails"]["task-fail"] == "error output"

    def test_extract_failed_state_uses_exit_code_when_no_message(self, tmp_path: Path) -> None:
        now = datetime.now()
        run_result = PlanRunResult(
            plan_name="test-plan",
            run_id="run-002",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "task-fail": TaskResult(
                    task_id="task-fail",
                    status="failed",
                    exit_code=42,
                    message="",
                ),
            },
        )
        state = _extract_failed_state(run_result)
        assert state["error_messages"]["task-fail"] == "exit_code=42"

    # ──────────────────────────── _show_plan_diff ────────────────────────────

    def test_show_plan_diff(self, capsys: pytest.CaptureFixture[str]) -> None:
        original = "version: 1\nname: old\n"
        corrected = "version: 1\nname: new\n"
        diff = _show_plan_diff(original, corrected)
        assert "old" in diff
        assert "new" in diff
        assert "---" in diff or "+++" in diff

    def test_show_plan_diff_no_changes(self, capsys: pytest.CaptureFixture[str]) -> None:
        same = "version: 1\nname: x\n"
        diff = _show_plan_diff(same, same)
        assert diff == ""
        captured = capsys.readouterr()
        assert "No plan changes detected" in captured.out

    # ──────────────────────────── _build_analysis_prompt ─────────────────────

    def test_build_analysis_prompt(self) -> None:
        plan_yaml = "version: 1\nname: demo\ntasks: []\n"
        failed_state: dict = {
            "failed_task_ids": ["task-a"],
            "passed_task_ids": [],
            "error_messages": {"task-a": "exit_code=1"},
            "stdout_tails": {},
        }
        prompt = _build_analysis_prompt(plan_yaml, failed_state)
        assert "version: 1" in prompt
        assert "task-a" in prompt
        assert "exit_code=1" in prompt
        assert "```yaml" in prompt
        assert "FAILED EXECUTION STATE" in prompt

    def test_build_analysis_prompt_json_encoded(self) -> None:
        plan_yaml = "version: 1\nname: demo\n"
        failed_state: dict = {
            "failed_task_ids": ["x"],
            "passed_task_ids": ["y"],
            "error_messages": {"x": "boom"},
            "stdout_tails": {},
        }
        prompt = _build_analysis_prompt(plan_yaml, failed_state)
        # The failed state block must be valid JSON embedded in the prompt
        json_start = prompt.index("FAILED EXECUTION STATE:\n") + len("FAILED EXECUTION STATE:\n")
        json_end = prompt.index("\n\nORIGINAL PLAN YAML")
        embedded = prompt[json_start:json_end]
        parsed = json.loads(embedded)
        assert parsed["failed_task_ids"] == ["x"]

    def test_build_analysis_prompt_multi_candidate_instructions(self) -> None:
        prompt = _build_analysis_prompt(
            "version: 1\nname: demo\n",
            {"failed_task_ids": [], "passed_task_ids": [], "error_messages": {}, "stdout_tails": {}},
            candidate_number=2,
            total_candidates=3,
        )
        assert "candidate 2 of 3" in prompt
        assert "materially distinct correction strategy" in prompt

    def test_build_analysis_prompt_includes_knowledge_guidance(self) -> None:
        prompt = _build_analysis_prompt(
            "version: 1\nname: demo\n",
            {"failed_task_ids": ["task-a"], "passed_task_ids": [], "error_messages": {}, "stdout_tails": {}},
            knowledge_guidance="- [80%] [FAIL] [task=task-a] Fails with timeout.",
        )
        assert "HISTORICAL KNOWLEDGE HINTS" in prompt
        assert "Fails with timeout" in prompt

    def test_select_replan_search_variant_tournament_uses_random_pool_when_diversity_disabled(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        plan = PlanSpec(name="demo")
        root = WorkflowVariant(
            node_id="root",
            plan_spec=plan,
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="root-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            is_valid=True,
            visits=5,
        )
        first = WorkflowVariant(
            node_id="first",
            plan_spec=plan,
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="first-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            score=0.3,
            is_valid=True,
            parent=root,
            visits=1,
        )
        second = WorkflowVariant(
            node_id="second",
            plan_spec=plan,
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="second-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            score=0.9,
            is_valid=True,
            parent=root,
            visits=3,
        )
        third = WorkflowVariant(
            node_id="third",
            plan_spec=plan,
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="third-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            score=0.8,
            is_valid=True,
            parent=root,
            visits=2,
        )
        root.children.extend([first, second, third])

        class _StubRandom(random.Random):
            def choice(self, values: list[WorkflowVariant]) -> WorkflowVariant:
                if first in values:
                    return first
                return third

            def random(self) -> float:
                return 0.99

        monkeypatch.setattr("maestro_cli.replan.random.Random", _StubRandom)

        selected = _select_replan_search_variant(
            root,
            selection_policy="debug_prob",
            debug_prob=0.0,
            exploration_constant=1.4,
            population_strategy="tournament",
            tournament_size=2,
            elite_count=0,
            diversity_floor=0.0,
        )

        assert selected is third

    def test_select_replan_search_variant_tournament_filters_near_duplicate_challengers(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        base_plan = PlanSpec(
            name="demo",
            goal="baseline plan",
            tasks=[{"id": "t1", "command": "echo base"}],  # type: ignore[list-item]
        )
        root = WorkflowVariant(
            node_id="root",
            plan_spec=base_plan,
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="root-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            is_valid=True,
            visits=5,
        )
        elite = WorkflowVariant(
            node_id="elite",
            plan_spec=PlanSpec(
                name="demo",
                goal="retry network",
                max_parallel=2,
                tasks=[{"id": "t1", "command": "echo base"}],  # type: ignore[list-item]
            ),
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="elite-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            score=0.95,
            is_valid=True,
            parent=root,
            visits=4,
        )
        near_duplicate = WorkflowVariant(
            node_id="near-duplicate",
            plan_spec=PlanSpec(
                name="demo",
                goal="retry network",
                max_parallel=2,
                tasks=[{"id": "t1", "command": "echo base"}],  # type: ignore[list-item]
            ),
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="near-duplicate-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            score=0.40,
            is_valid=True,
            parent=root,
            visits=1,
        )
        diverse = WorkflowVariant(
            node_id="diverse",
            plan_spec=PlanSpec(
                name="demo",
                goal="switch execution strategy",
                fail_fast=False,
                tasks=[{"id": "t1", "command": "python -m pytest"}],  # type: ignore[list-item]
            ),
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="diverse-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            score=0.50,
            is_valid=True,
            parent=root,
            visits=1,
        )
        root.children.extend([elite, near_duplicate, diverse])

        captured_pool: dict[str, list[str]] = {}

        class _StubRandom(random.Random):
            def choice(self, values: list[WorkflowVariant]) -> WorkflowVariant:
                return values[0]

        monkeypatch.setattr("maestro_cli.replan.random.Random", _StubRandom)
        monkeypatch.setattr(
            "maestro_cli.replan.select_variant_from_pool",
            lambda candidates, **kwargs: captured_pool.setdefault(
                "node_ids",
                [candidate.node_id for candidate in candidates],
            ) and candidates[0],
        )

        selected = _select_replan_search_variant(
            root,
            selection_policy="debug_prob",
            debug_prob=0.0,
            exploration_constant=1.4,
            population_strategy="tournament",
            tournament_size=2,
            elite_count=1,
            diversity_floor=0.25,
        )

        assert selected is elite
        assert captured_pool["node_ids"] == ["elite", "diverse"]

    def test_select_replan_search_variant_tournament_preserves_elite_candidates(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        plan = PlanSpec(name="demo")
        root = WorkflowVariant(
            node_id="root",
            plan_spec=plan,
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="root-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            is_valid=True,
            visits=5,
        )
        elite = WorkflowVariant(
            node_id="elite",
            plan_spec=plan,
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="elite-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            score=0.95,
            is_valid=True,
            parent=root,
            visits=4,
        )
        challenger_a = WorkflowVariant(
            node_id="challenger-a",
            plan_spec=plan,
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="challenger-a-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            score=0.40,
            is_valid=True,
            parent=root,
            visits=1,
        )
        challenger_b = WorkflowVariant(
            node_id="challenger-b",
            plan_spec=PlanSpec(name="demo", goal="different"),
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="challenger-b-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            score=0.50,
            is_valid=True,
            parent=root,
            visits=1,
        )
        root.children.extend([elite, challenger_a, challenger_b])

        class _StubRandom(random.Random):
            def choice(self, values: list[WorkflowVariant]) -> WorkflowVariant:
                return values[0]

            def random(self) -> float:
                return 0.99

        monkeypatch.setattr("maestro_cli.replan.random.Random", _StubRandom)

        selected = _select_replan_search_variant(
            root,
            selection_policy="debug_prob",
            debug_prob=0.0,
            exploration_constant=1.4,
            population_strategy="tournament",
            tournament_size=2,
            elite_count=1,
            diversity_floor=0.25,
        )

        assert selected is elite

    def test_select_replan_search_variant_tournament_falls_back_when_diversity_pool_exhausted(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = PlanSpec(name="demo", goal="baseline")
        root = WorkflowVariant(
            node_id="root",
            plan_spec=plan,
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="root-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            is_valid=True,
            visits=5,
        )
        elite = WorkflowVariant(
            node_id="elite",
            plan_spec=PlanSpec(name="demo", goal="retry network"),
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="elite-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            score=0.95,
            is_valid=True,
            parent=root,
            visits=4,
        )
        near_duplicate = WorkflowVariant(
            node_id="near-duplicate",
            plan_spec=PlanSpec(name="demo", goal="retry network"),
            run_result=PlanRunResult(
                plan_name="demo",
                run_id="near-duplicate-run",
                run_path=tmp_path,
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
            ),
            score=0.60,
            is_valid=True,
            parent=root,
            visits=1,
        )
        root.children.extend([elite, near_duplicate])

        captured_pool: dict[str, list[str]] = {}

        class _StubRandom(random.Random):
            def choice(self, values: list[WorkflowVariant]) -> WorkflowVariant:
                return values[0]

        monkeypatch.setattr("maestro_cli.replan.random.Random", _StubRandom)
        monkeypatch.setattr(
            "maestro_cli.replan.select_variant_from_pool",
            lambda candidates, **kwargs: captured_pool.setdefault(
                "node_ids",
                [candidate.node_id for candidate in candidates],
            ) and candidates[0],
        )

        selected = _select_replan_search_variant(
            root,
            selection_policy="debug_prob",
            debug_prob=0.0,
            exploration_constant=1.4,
            population_strategy="tournament",
            tournament_size=2,
            elite_count=1,
            diversity_floor=0.9,
        )

        assert selected is elite
        assert captured_pool["node_ids"] == ["elite", "near-duplicate"]

    # ──────────────────────────── ReplanState dataclass ──────────────────────

    def test_replan_state_dataclass(self) -> None:
        state = ReplanState(plan_path="/tmp/plan.yaml", max_attempts=3)
        assert state.plan_path == "/tmp/plan.yaml"
        assert state.max_attempts == 3
        assert state.attempts == []
        assert state.status == "max_attempts_exceeded"
        assert state.final_success is False
        assert state.total_cost_usd == 0.0
        assert state.total_tokens == 0
        assert state.analysis_model == "opus"
        assert state.search_tree_path is None

    def test_replan_state_mutable_defaults_independent(self) -> None:
        s1 = ReplanState(plan_path="a.yaml", max_attempts=1)
        s2 = ReplanState(plan_path="b.yaml", max_attempts=1)
        s1.attempts.append(ReplanAttempt(attempt_number=1))
        assert s2.attempts == []

    # ──────────────────────────── ReplanAttempt dataclass ────────────────────

    def test_replan_attempt_dataclass(self) -> None:
        attempt = ReplanAttempt(attempt_number=1)
        assert attempt.attempt_number == 1
        assert attempt.plan_yaml == ""
        assert attempt.run_result is None
        assert attempt.failed_task_ids == []
        assert attempt.analysis_response is None
        assert attempt.analysis_error is None
        assert attempt.corrected_plan_yaml is None
        assert attempt.diff_summary == ""
        assert attempt.approved is False
        assert attempt.error_summary == ""
        assert attempt.candidate_variants == []
        assert attempt.selected_candidate_id is None

    def test_replan_attempt_mutable_defaults_independent(self) -> None:
        a1 = ReplanAttempt(attempt_number=1)
        a2 = ReplanAttempt(attempt_number=2)
        a1.failed_task_ids.append("task-x")
        assert a2.failed_task_ids == []


# ──────────────────────────── Additional Tests ────────────────────────────────

from maestro_cli.replan import _call_analysis_model, replan
import subprocess


class TestParseYamlExtended:
    """Extended tests for _parse_corrected_yaml patterns."""

    def test_parse_yml_fence(self) -> None:
        response = "Fix:\n```yml\nversion: 1\nname: fixed\n```"
        result = _parse_corrected_yaml(response)
        assert result == "version: 1\nname: fixed"

    def test_parse_generic_fence(self) -> None:
        response = "Here:\n```\nversion: 1\nname: plain\n```"
        result = _parse_corrected_yaml(response)
        assert result == "version: 1\nname: plain"

    def test_parse_prefers_yaml_over_generic(self) -> None:
        response = (
            "```yaml\nversion: 1\nname: yaml-fence\n```\n"
            "```\nversion: 1\nname: generic-fence\n```"
        )
        result = _parse_corrected_yaml(response)
        assert result is not None
        assert "yaml-fence" in result

    def test_parse_multiple_yaml_fences_returns_first(self) -> None:
        response = (
            "```yaml\nfirst: yes\n```\n"
            "```yaml\nsecond: yes\n```"
        )
        result = _parse_corrected_yaml(response)
        assert result == "first: yes"

    def test_parse_fence_with_extra_text_around(self) -> None:
        response = (
            "Some preamble text.\n\n"
            "```yaml\nversion: 1\nname: embedded\ntasks: []\n```\n\n"
            "Some closing remarks."
        )
        result = _parse_corrected_yaml(response)
        assert result is not None
        assert "embedded" in result

    def test_parse_case_insensitive_yaml_tag(self) -> None:
        response = "```YAML\nversion: 1\nname: upper\n```"
        result = _parse_corrected_yaml(response)
        assert result == "version: 1\nname: upper"

    def test_parse_fence_with_leading_whitespace(self) -> None:
        response = "```yaml\n  version: 1\n  name: indented\n```"
        result = _parse_corrected_yaml(response)
        assert result == "version: 1\n  name: indented"


class TestExtractFailedStateExtended:
    """Extended tests for _extract_failed_state with various scenarios."""

    def _make_result(
        self,
        tmp_path: Path,
        task_results: dict[str, TaskResult],
    ) -> PlanRunResult:
        now = datetime.now()
        return PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results=task_results,
        )

    def test_no_failures(self, tmp_path: Path) -> None:
        result = self._make_result(tmp_path, {
            "t1": TaskResult(task_id="t1", status="success", exit_code=0),
            "t2": TaskResult(task_id="t2", status="success", exit_code=0),
        })
        state = _extract_failed_state(result)
        assert state["failed_task_ids"] == []
        assert set(state["passed_task_ids"]) == {"t1", "t2"}
        assert state["error_messages"] == {}

    def test_all_failures(self, tmp_path: Path) -> None:
        result = self._make_result(tmp_path, {
            "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="err1"),
            "t2": TaskResult(task_id="t2", status="failed", exit_code=2, message="err2"),
        })
        state = _extract_failed_state(result)
        assert set(state["failed_task_ids"]) == {"t1", "t2"}
        assert state["passed_task_ids"] == []

    def test_soft_failed_counts_as_passed(self, tmp_path: Path) -> None:
        result = self._make_result(tmp_path, {
            "t1": TaskResult(task_id="t1", status="soft_failed", exit_code=1),
        })
        state = _extract_failed_state(result)
        assert state["failed_task_ids"] == []
        assert "t1" in state["passed_task_ids"]

    def test_dry_run_counts_as_passed(self, tmp_path: Path) -> None:
        result = self._make_result(tmp_path, {
            "t1": TaskResult(task_id="t1", status="dry_run"),
        })
        state = _extract_failed_state(result)
        assert "t1" in state["passed_task_ids"]

    def test_skipped_not_in_either_list(self, tmp_path: Path) -> None:
        result = self._make_result(tmp_path, {
            "t1": TaskResult(task_id="t1", status="skipped"),
        })
        state = _extract_failed_state(result)
        assert state["failed_task_ids"] == []
        assert state["passed_task_ids"] == []

    def test_failed_without_stdout_tail(self, tmp_path: Path) -> None:
        result = self._make_result(tmp_path, {
            "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="bad"),
        })
        state = _extract_failed_state(result)
        assert "t1" not in state["stdout_tails"]

    def test_failed_with_none_message(self, tmp_path: Path) -> None:
        result = self._make_result(tmp_path, {
            "t1": TaskResult(task_id="t1", status="failed", exit_code=99, message=None),
        })
        state = _extract_failed_state(result)
        assert state["error_messages"]["t1"] == "exit_code=99"

    def test_mixed_statuses(self, tmp_path: Path) -> None:
        result = self._make_result(tmp_path, {
            "a": TaskResult(task_id="a", status="success", exit_code=0),
            "b": TaskResult(task_id="b", status="failed", exit_code=1, message="broke", stdout_tail="trace"),
            "c": TaskResult(task_id="c", status="soft_failed", exit_code=1),
            "d": TaskResult(task_id="d", status="skipped"),
            "e": TaskResult(task_id="e", status="dry_run"),
        })
        state = _extract_failed_state(result)
        assert state["failed_task_ids"] == ["b"]
        assert set(state["passed_task_ids"]) == {"a", "c", "e"}
        assert state["stdout_tails"]["b"] == "trace"


class TestBuildAnalysisPromptExtended:
    """Extended tests for _build_analysis_prompt."""

    def test_prompt_contains_rules(self) -> None:
        prompt = _build_analysis_prompt("", {"failed_task_ids": [], "passed_task_ids": [], "error_messages": {}, "stdout_tails": {}})
        assert "Keep valid parts unchanged" in prompt
        assert "Address all failed tasks" in prompt
        assert "```yaml" in prompt

    def test_prompt_includes_stdout_tails(self) -> None:
        state = {
            "failed_task_ids": ["x"],
            "passed_task_ids": [],
            "error_messages": {"x": "err"},
            "stdout_tails": {"x": "some trace output"},
        }
        prompt = _build_analysis_prompt("version: 1\n", state)
        assert "some trace output" in prompt

    def test_prompt_plan_yaml_stripped(self) -> None:
        plan_yaml = "version: 1\nname: test\n\n\n"
        prompt = _build_analysis_prompt(plan_yaml, {"failed_task_ids": [], "passed_task_ids": [], "error_messages": {}, "stdout_tails": {}})
        # Plan YAML should be rstripped
        assert "version: 1\nname: test\n```" in prompt


class TestDetectExitLoopExtended:
    """Extended tests for _detect_exit_loop."""

    def test_empty_attempts(self) -> None:
        assert _detect_exit_loop([]) is False

    def test_both_empty_failure_lists(self) -> None:
        a1 = ReplanAttempt(attempt_number=1, failed_task_ids=[])
        a2 = ReplanAttempt(attempt_number=2, failed_task_ids=[])
        # Empty sets match but `bool(previous)` is False
        assert _detect_exit_loop([a1, a2]) is False

    def test_first_empty_second_has_failures(self) -> None:
        a1 = ReplanAttempt(attempt_number=1, failed_task_ids=[])
        a2 = ReplanAttempt(attempt_number=2, failed_task_ids=["t1"])
        assert _detect_exit_loop([a1, a2]) is False

    def test_three_attempts_only_last_two_compared(self) -> None:
        a1 = ReplanAttempt(attempt_number=1, failed_task_ids=["t1"])
        a2 = ReplanAttempt(attempt_number=2, failed_task_ids=["t2"])
        a3 = ReplanAttempt(attempt_number=3, failed_task_ids=["t2"])
        assert _detect_exit_loop([a1, a2, a3]) is True

    def test_superset_does_not_trigger(self) -> None:
        a1 = ReplanAttempt(attempt_number=1, failed_task_ids=["t1"])
        a2 = ReplanAttempt(attempt_number=2, failed_task_ids=["t1", "t2"])
        assert _detect_exit_loop([a1, a2]) is False

    def test_subset_does_not_trigger(self) -> None:
        a1 = ReplanAttempt(attempt_number=1, failed_task_ids=["t1", "t2"])
        a2 = ReplanAttempt(attempt_number=2, failed_task_ids=["t1"])
        assert _detect_exit_loop([a1, a2]) is False


class TestShowPlanDiffExtended:
    """Extended tests for _show_plan_diff."""

    def test_diff_has_unified_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        original = "line1\nline2\nline3\n"
        corrected = "line1\nmodified\nline3\n"
        diff = _show_plan_diff(original, corrected)
        assert "--- original.yaml" in diff
        assert "+++ corrected.yaml" in diff
        assert "-line2" in diff
        assert "+modified" in diff

    def test_diff_prints_each_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        original = "a\n"
        corrected = "b\n"
        _show_plan_diff(original, corrected)
        out = capsys.readouterr().out
        assert "[maestro]" in out

    def test_diff_addition_only(self, capsys: pytest.CaptureFixture[str]) -> None:
        original = "line1\n"
        corrected = "line1\nline2\n"
        diff = _show_plan_diff(original, corrected)
        assert "+line2" in diff


class TestCallAnalysisModel:
    """Tests for _call_analysis_model subprocess interaction."""

    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _mock_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout="corrected plan output",
                stderr="",
            )

        monkeypatch.setattr("maestro_cli.replan.subprocess.run", _mock_run)
        result = _call_analysis_model("prompt text", "opus")
        assert result == "corrected plan output"

    def test_nonzero_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _mock_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["claude"],
                returncode=1,
                stdout="",
                stderr="API error",
            )

        monkeypatch.setattr("maestro_cli.replan.subprocess.run", _mock_run)
        with pytest.raises(RuntimeError, match="Analysis command failed"):
            _call_analysis_model("prompt", "opus")

    def test_empty_stdout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _mock_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            return subprocess.CompletedProcess(
                args=["claude"],
                returncode=0,
                stdout="",
                stderr="",
            )

        monkeypatch.setattr("maestro_cli.replan.subprocess.run", _mock_run)
        with pytest.raises(RuntimeError, match="empty stdout"):
            _call_analysis_model("prompt", "opus")

    def test_timeout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _mock_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=120)

        monkeypatch.setattr("maestro_cli.replan.subprocess.run", _mock_run)
        with pytest.raises(RuntimeError, match="timed out"):
            _call_analysis_model("prompt", "opus")

    def test_os_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _mock_run(*args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
            raise OSError("No such file")

        monkeypatch.setattr("maestro_cli.replan.subprocess.run", _mock_run)
        with pytest.raises(RuntimeError, match="Failed to invoke"):
            _call_analysis_model("prompt", "opus")

    def test_uses_model_param(self, monkeypatch: pytest.MonkeyPatch) -> None:
        captured_args: list[list[str]] = []

        def _mock_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
            captured_args.append(cmd)
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="ok", stderr="")

        monkeypatch.setattr("maestro_cli.replan.subprocess.run", _mock_run)
        _call_analysis_model("my prompt", "sonnet")
        assert "--model" in captured_args[0]
        idx = captured_args[0].index("--model")
        assert captured_args[0][idx + 1] == "sonnet"


class TestReplanMainFunction:
    """Tests for the replan() orchestrator."""

    _VALID_PLAN = "version: 1\nname: test\ntasks:\n  - id: t1\n    command: echo hello\n"

    @staticmethod
    def _knowledge_record(
        *,
        task_id: str = "t1",
        kind: str = "failure_pattern",
        insight: str = "Fails with timeout.",
    ) -> KnowledgeRecord:
        return KnowledgeRecord(
            task_id=task_id,
            kind=kind,
            insight=insight,
            confidence=0.8,
            occurrences=3,
            first_seen="2026-04-01T00:00:00+00:00",
            last_seen="2026-04-02T00:00:00+00:00",
        )

    @staticmethod
    def _score_record(plan_name: str, plan_hash: str, *, run_id: str, success: bool) -> ScoreRecord:
        timestamp = "2026-04-02T00:00:00+00:00"
        return ScoreRecord(
            plan_name=plan_name,
            plan_hash=plan_hash,
            run_id=run_id,
            success=success,
            cost_usd=0.1,
            quality_score=1.0 if success else 0.0,
            duration_sec=1.0,
            timestamp=timestamp,
            valid_from=timestamp,
            recorded_at=timestamp,
            source_id=f"{run_id}:score",
            metadata={},
        )

    def test_replan_success_first_attempt(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        now = datetime.now()
        success_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=True,
            task_results={
                "t1": TaskResult(task_id="t1", status="success", exit_code=0),
            },
            total_cost_usd=0.5,
            total_tokens=1000,
        )

        monkeypatch.setattr("maestro_cli.replan.run_plan", lambda *a, **kw: success_result)

        state = replan(plan_path, max_attempts=3, auto_approve=True)
        assert state.final_success is True
        assert state.status == "success"
        assert len(state.attempts) == 1
        assert state.total_cost_usd == 0.5
        assert state.total_tokens == 1000

    def test_replan_max_attempts_exceeded(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        now = datetime.now()
        fail_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="boom"),
            },
        )

        monkeypatch.setattr("maestro_cli.replan.run_plan", lambda *a, **kw: fail_result)
        # Analysis returns a valid corrected plan
        monkeypatch.setattr(
            "maestro_cli.replan._call_analysis_model",
            lambda prompt, model: f"```yaml\n{self._VALID_PLAN}```",
        )

        state = replan(plan_path, max_attempts=2, auto_approve=True)
        assert state.final_success is False
        assert state.status == "max_attempts_exceeded"
        assert len(state.attempts) == 2

    def test_replan_circuit_breaker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        now = datetime.now()
        fail_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="same error"),
            },
        )

        call_count = [0]

        def _fake_run_plan(*a: object, **kw: object) -> PlanRunResult:
            call_count[0] += 1
            return fail_result

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr(
            "maestro_cli.replan._call_analysis_model",
            lambda prompt, model: f"```yaml\n{self._VALID_PLAN}```",
        )

        state = replan(plan_path, max_attempts=5, auto_approve=True)
        out = capsys.readouterr().out
        assert state.status == "circuit_breaker"
        assert "Circuit breaker" in out

    def test_replan_analysis_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        now = datetime.now()
        fail_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="err"),
            },
        )

        monkeypatch.setattr("maestro_cli.replan.run_plan", lambda *a, **kw: fail_result)
        monkeypatch.setattr(
            "maestro_cli.replan._call_analysis_model",
            lambda prompt, model: (_ for _ in ()).throw(RuntimeError("API down")),
        )

        state = replan(plan_path, max_attempts=1, auto_approve=True)
        assert state.final_success is False
        assert len(state.attempts) == 1
        assert "Analysis failed" in state.attempts[0].error_summary

    def test_replan_prompt_includes_relevant_knowledge_guidance(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")
        store_knowledge(
            "test",
            tmp_path,
            [
                self._knowledge_record(insight="Fails with timeout."),
                self._knowledge_record(kind="model_pattern", insight="Model opus: success."),
            ],
        )

        now = datetime.now()
        fail_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="err"),
            },
        )

        captured: dict[str, str] = {}

        monkeypatch.setattr("maestro_cli.replan.run_plan", lambda *a, **kw: fail_result)

        def _capture_prompt(prompt: str, model: str) -> str:
            captured["prompt"] = prompt
            return f"```yaml\n{self._VALID_PLAN}```"

        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _capture_prompt)

        replan(plan_path, max_attempts=1, auto_approve=True)

        prompt = captured["prompt"]
        assert "HISTORICAL KNOWLEDGE HINTS" in prompt
        assert "Fails with timeout." in prompt
        assert "Model opus: success." in prompt

    def test_replan_no_yaml_in_response(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        now = datetime.now()
        fail_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="err"),
            },
        )

        monkeypatch.setattr("maestro_cli.replan.run_plan", lambda *a, **kw: fail_result)
        monkeypatch.setattr(
            "maestro_cli.replan._call_analysis_model",
            lambda prompt, model: "I think the problem is X but I forgot the code fence.",
        )

        state = replan(plan_path, max_attempts=1, auto_approve=True)
        assert "No YAML code fence" in state.attempts[0].analysis_error

    def test_replan_user_rejects(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        now = datetime.now()
        fail_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="err"),
            },
        )

        monkeypatch.setattr("maestro_cli.replan.run_plan", lambda *a, **kw: fail_result)
        monkeypatch.setattr(
            "maestro_cli.replan._call_analysis_model",
            lambda prompt, model: f"```yaml\n{self._VALID_PLAN}```",
        )
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")

        state = replan(plan_path, max_attempts=3, auto_approve=False)
        assert state.status == "circuit_breaker"
        assert state.attempts[-1].approved is False

    def test_replan_invalid_plan_path(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        missing_path = tmp_path / "nonexistent.yaml"
        state = replan(missing_path, max_attempts=1)
        assert state.final_success is False
        assert len(state.attempts) == 1
        assert "Failed to load plan" in state.attempts[0].error_summary

    def test_replan_corrected_plan_invalid(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        now = datetime.now()
        fail_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="err"),
            },
        )

        monkeypatch.setattr("maestro_cli.replan.run_plan", lambda *a, **kw: fail_result)
        # Return invalid YAML that will fail validation
        monkeypatch.setattr(
            "maestro_cli.replan._call_analysis_model",
            lambda prompt, model: "```yaml\nversion: 99\nname: bad\n```",
        )

        state = replan(plan_path, max_attempts=1, auto_approve=True)
        out = capsys.readouterr().out
        assert "Corrected plan invalid" in out

    def test_replan_blocks_generated_plan_with_new_security_findings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        now = datetime.now()
        fail_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="err"),
            },
        )

        run_calls = {"n": 0}

        def _fake_run_plan(*a: object, **kw: object) -> PlanRunResult:
            run_calls["n"] += 1
            return fail_result

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr(
            "maestro_cli.replan._call_analysis_model",
            lambda prompt, model: (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: rm -rf build\n"
                "```"
            ),
        )

        state = replan(plan_path, max_attempts=1, auto_approve=True)

        assert state.final_success is False
        assert run_calls["n"] == 1
        assert len(state.attempts) == 1
        assert "security gate" in state.attempts[0].error_summary
        assert "SEC008" in state.attempts[0].error_summary

    def test_replan_blocks_generated_plan_when_pass2_rejects(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(
            "version: 1\n"
            "name: test\n"
            "firewall_model: haiku\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n",
            encoding="utf-8",
        )

        now = datetime.now()
        fail_result = PlanRunResult(
            plan_name="test",
            run_id="r1",
            run_path=tmp_path,
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="err"),
            },
        )

        run_calls = {"n": 0}

        def _fake_run_plan(*a: object, **kw: object) -> PlanRunResult:
            run_calls["n"] += 1
            return fail_result

        class _BlockedDecision:
            verdict = "block"
            category = "policy_bypass"
            reason = "attempted override"

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr(
            "maestro_cli.replan._call_analysis_model",
            lambda prompt, model: (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "firewall_model: disabled\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "```"
            ),
        )
        monkeypatch.setattr(
            "maestro_cli.replan._run_firewall_pass2",
            lambda *a, **kw: _BlockedDecision(),
        )

        state = replan(plan_path, max_attempts=1, auto_approve=True)

        assert state.final_success is False
        assert run_calls["n"] == 1
        assert len(state.attempts) == 1
        assert "security gate" in state.attempts[0].error_summary
        assert "block/policy_bypass" in state.attempts[0].error_summary

    def test_replan_cost_accumulation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        now = datetime.now()
        call_count = [0]

        def _fake_run_plan(*a: object, **kw: object) -> PlanRunResult:
            call_count[0] += 1
            if call_count[0] >= 2:
                return PlanRunResult(
                    plan_name="test", run_id="r2", run_path=tmp_path,
                    started_at=now, finished_at=now, success=True,
                    task_results={"t1": TaskResult(task_id="t1", status="success", exit_code=0)},
                    total_cost_usd=0.3, total_tokens=500,
                )
            return PlanRunResult(
                plan_name="test", run_id="r1", run_path=tmp_path,
                started_at=now, finished_at=now, success=False,
                task_results={"t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="err")},
                total_cost_usd=0.2, total_tokens=300,
            )

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr(
            "maestro_cli.replan._call_analysis_model",
            lambda prompt, model: f"```yaml\n{self._VALID_PLAN}```",
        )

        state = replan(plan_path, max_attempts=3, auto_approve=True)
        assert state.final_success is True
        assert state.total_cost_usd == pytest.approx(0.5)
        assert state.total_tokens == 800

    def test_replan_none_cost_tokens(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        now = datetime.now()
        result = PlanRunResult(
            plan_name="test", run_id="r1", run_path=tmp_path,
            started_at=now, finished_at=now, success=True,
            task_results={"t1": TaskResult(task_id="t1", status="success", exit_code=0)},
            total_cost_usd=None, total_tokens=None,
        )

        monkeypatch.setattr("maestro_cli.replan.run_plan", lambda *a, **kw: result)

        state = replan(plan_path, max_attempts=1, auto_approve=True)
        assert state.total_cost_usd == 0.0
        assert state.total_tokens == 0

    def test_replan_search_variants_selects_successful_candidate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        run_counter = {"n": 0}
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-1" in source_name:
                return _result("cand-1", "cand-1", False, "candidate one failed")
            if "candidate-2" in source_name:
                return _result("cand-2", "cand-2", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                return (
                    "```yaml\n"
                    "version: 1\nname: test\nmax_parallel: 2\n"
                    "tasks:\n  - id: t1\n    command: echo hello\n"
                    "```"
                )
            return (
                "```yaml\n"
                "version: 1\nname: test\nmax_parallel: 3\n"
                "tasks:\n  - id: t1\n    command: echo hello\n"
                "```"
            )

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
        )

        assert state.final_success is True
        assert state.status == "success"
        assert len(state.attempts) == 1
        assert run_counter["n"] == 3
        assert state.total_cost_usd == pytest.approx(0.3)
        assert state.total_tokens == 300
        attempt = state.attempts[0]
        assert attempt.run_result is not None
        assert attempt.run_result.success is True
        assert len(attempt.candidate_variants) == 2
        assert attempt.selected_candidate_id is not None
        assert attempt.corrected_plan_yaml is not None
        assert "max_parallel: 3" in attempt.corrected_plan_yaml
        assert state.search_tree_path is not None
        tree_path = Path(state.search_tree_path)
        assert tree_path.is_file()
        assert len(tree_path.read_text(encoding="utf-8").splitlines()) == 3
        stones_path = tmp_path / ".maestro-cache" / "stepping" / "test" / "stones.jsonl"
        assert stones_path.is_file()
        stone = json.loads(stones_path.read_text(encoding="utf-8").strip())
        assert stone["metric_name"] == "replan_fitness"
        assert stone["source_type"] == "replan"
        assert stone["metadata"]["selected_run_id"] == "cand-2"
        assert stone["metadata"]["mutation_desc"] == "candidate 2/2"
        assert stone["metadata"]["fitness_gain"] > 0.0
        assert stone["lessons"][0]["source"] == "replan"

    def test_replan_search_prompt_includes_knowledge_guidance(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")
        store_knowledge(
            "test",
            tmp_path,
            [self._knowledge_record(insight="Fails with timeout.")],
        )

        run_counter = {"n": 0}
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-1" in source_name:
                return _result("cand-1", "cand-1", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        prompts: list[str] = []

        def _fake_analysis(prompt: str, model: str) -> str:
            prompts.append(prompt)
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "max_parallel: 2\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "```"
            )

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        state = replan(
            plan_path,
            max_attempts=2,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
        )

        assert state.final_success is True
        assert prompts
        assert "HISTORICAL KNOWLEDGE HINTS" in prompts[0]
        assert "Fails with timeout." in prompts[0]

    def test_replan_search_prefers_knowledge_aligned_variant_when_scores_tie(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")
        store_knowledge(
            "test",
            tmp_path,
            [
                self._knowledge_record(
                    insight="Fails with timeout. Increase timeout_sec or split the task into smaller parts.",
                ),
            ],
        )

        run_counter = {"n": 0}
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-1" in source_name:
                return _result("cand-1", "cand-1", True)
            if "candidate-2" in source_name:
                return _result("cand-2", "cand-2", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                return (
                    "```yaml\n"
                    "version: 1\n"
                    "name: test\n"
                    "tasks:\n"
                    "  - id: t1\n"
                    "    command: echo hello\n"
                    "```"
                )
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "    timeout_sec: 120\n"
                "```"
            )

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
        )

        assert state.final_success is True
        attempt = state.attempts[0]
        assert attempt.corrected_plan_yaml is not None
        assert "timeout_sec: 120" in attempt.corrected_plan_yaml
        assert attempt.selected_candidate_id is not None
        assert attempt.candidate_variants[0]["knowledge_bonus"] == pytest.approx(0.0)
        assert attempt.candidate_variants[1]["knowledge_bonus"] > 0.0

    def test_replan_search_prefers_more_novel_variant_when_scores_tie(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        run_counter = {"n": 0}
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-1" in source_name:
                return _result("cand-1", "cand-1", True)
            if "candidate-2" in source_name:
                return _result("cand-2", "cand-2", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                return "```yaml\n" + self._VALID_PLAN + "```"
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "max_parallel: 3\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "    timeout_sec: 120\n"
                "```"
            )

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
        )

        assert state.final_success is True
        attempt = state.attempts[0]
        assert attempt.corrected_plan_yaml is not None
        assert "max_parallel: 3" in attempt.corrected_plan_yaml
        assert attempt.candidate_variants[0]["novelty_bonus"] == pytest.approx(0.0)
        assert attempt.candidate_variants[1]["novelty_bonus"] > 0.0

    def test_replan_search_bootstraps_similar_historical_fitness_when_scores_tie(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")
        history_path = tmp_path / "historical-similar.yaml"
        history_path.write_text(
            "version: 1\n"
            "name: test\n"
            "max_parallel: 3\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    max_retries: 2\n"
            "    checkpoint: true\n"
            "    stdout_tail_lines: 30\n"
            "    tags: [stable]\n",
            encoding="utf-8",
        )
        history_plan = load_plan(history_path)
        history_score = build_score_record(
            history_plan,
            PlanRunResult(
                plan_name="test",
                run_id="historical-similar",
                run_path=tmp_path / "history-similar-run",
                started_at=datetime.now(),
                finished_at=datetime.now(),
                success=True,
                task_results={
                    "t1": TaskResult(task_id="t1", status="success", exit_code=0),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            ),
        )

        run_counter = {"n": 0}
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-1" in source_name:
                return _result("cand-1", "cand-1", True)
            if "candidate-2" in source_name:
                return _result("cand-2", "cand-2", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                return (
                    "```yaml\n"
                    "version: 1\n"
                    "name: test\n"
                    "tasks:\n"
                    "  - id: t1\n"
                    "    command: echo hello\n"
                    "    timeout_sec: 30\n"
                    "    allow_failure: true\n"
                    "    tags: [experimental]\n"
                    "```"
                )
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "max_parallel: 3\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "    max_retries: 2\n"
                "    checkpoint: true\n"
                "    tags: [stable]\n"
                "```"
            )

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)
        monkeypatch.setattr(
            "maestro_cli.replan.load_score_history",
            lambda plan_name, source_dir, plan_hash=None, since=None, limit=None: [] if plan_hash else [history_score],
        )
        monkeypatch.setattr(
            "maestro_cli.replan._score_replan_variant_novelty",
            lambda candidate_yaml, baseline_plan_yaml, prior_tree_rows: (0.0, []),
        )

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
        )

        assert state.final_success is True
        attempt = state.attempts[0]
        assert attempt.corrected_plan_yaml is not None
        assert "max_parallel: 3" in attempt.corrected_plan_yaml
        assert attempt.candidate_variants[0]["historical_fitness_bonus"] == pytest.approx(0.0)
        assert attempt.candidate_variants[1]["historical_fitness_bonus"] > 0.0

    def test_replan_search_deduplicates_duplicate_tree_variant_without_resimulation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        run_counter = {"n": 0}
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-1" in source_name:
                return _result("cand-1", "cand-1", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        duplicate_yaml = (
            "```yaml\n"
            "version: 1\n"
            "name: test\n"
            "max_parallel: 2\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "```"
        )
        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            return duplicate_yaml

        seen_events: list[str] = []
        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        state = replan(
            plan_path,
            max_attempts=2,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
            event_callback=lambda event, payload: seen_events.append(event),
        )

        assert state.final_success is True
        assert run_counter["n"] == 2
        attempt = state.attempts[0]
        assert len(attempt.candidate_variants) == 2
        assert attempt.candidate_variants[1]["deduplicated"] is True
        assert attempt.candidate_variants[1]["duplicate_source"] == "search_tree"
        assert attempt.candidate_variants[1]["pruned"] is True
        assert "replan_candidate_deduplicated" in seen_events

    def test_replan_search_reuses_historical_score_for_duplicate_hash(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        candidate_one_yaml = (
            "version: 1\n"
            "name: test\n"
            "max_parallel: 2\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    timeout_sec: 60\n"
        )
        candidate_one_path = tmp_path / "candidate-one.yaml"
        candidate_one_path.write_text(candidate_one_yaml, encoding="utf-8")
        candidate_one_plan = load_plan(candidate_one_path)
        candidate_one_hash = compute_plan_hash(candidate_one_plan)
        store_score_history(
            "test",
            tmp_path,
            self._score_record("test", candidate_one_hash, run_id="unused", success=False),
        )

        run_counter = {"n": 0}
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-2" in source_name:
                return _result("cand-2", "cand-2", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                return f"```yaml\n{candidate_one_yaml}```"
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "max_parallel: 3\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "```"
            )

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        state = replan(
            plan_path,
            max_attempts=2,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
        )

        assert state.final_success is True
        assert run_counter["n"] == 2
        attempt = state.attempts[0]
        duplicate = attempt.candidate_variants[0]
        assert duplicate["deduplicated"] is True
        assert duplicate["duplicate_source"] == "score_history"
        assert duplicate["historical_success"] is False
        assert 0.0 < duplicate["score"] < 0.3

    def test_replan_search_uses_simulation_cache_for_same_family_model_variant(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        prior_plan_yaml = (
            "version: 1\n"
            "name: test\n"
            "tasks:\n"
            "  - id: t1\n"
            "    engine: codex\n"
            "    model: 5.1\n"
            "    prompt: Implement feature\n"
        )
        prior_plan_path = tmp_path / "prior-plan.yaml"
        prior_plan_path.write_text(prior_plan_yaml, encoding="utf-8")
        prior_plan = load_plan(prior_plan_path)

        now = datetime.now()
        prior_run_path = tmp_path / "prior-run"
        prior_result = PlanRunResult(
            plan_name="test",
            run_id="cached-run",
            run_path=prior_run_path,
            started_at=now,
            finished_at=now,
            success=True,
            task_results={"t1": TaskResult(task_id="t1", status="success", exit_code=0)},
            total_cost_usd=0.2,
            total_tokens=321,
        )
        store_score_history("test", tmp_path, build_score_record(prior_plan, prior_result))

        run_counter = {"n": 0}

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            raise AssertionError("candidate simulation should have been served from cache")

        candidate_yaml = (
            "version: 1\n"
            "name: test\n"
            "tasks:\n"
            "  - id: t1\n"
            "    engine: codex\n"
            "    model: 5.4\n"
            "    prompt: Implement feature\n"
        )
        seen_events: list[str] = []
        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr(
            "maestro_cli.replan._call_analysis_model",
            lambda prompt, model: f"```yaml\n{candidate_yaml}```",
        )

        state = replan(
            plan_path,
            max_attempts=2,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
            event_callback=lambda event, payload: seen_events.append(event),
        )

        assert state.final_success is True
        assert run_counter["n"] == 1
        attempt = state.attempts[0]
        cached = attempt.candidate_variants[0]
        assert cached["simulation_cache_hit"] is True
        assert cached["simulation_cache_source_run_id"] == "cached-run"
        assert cached["run_id"] == "cached-run"
        assert 0.9 < cached["score"] < 1.0
        assert "replan_candidate_cache_hit" in seen_events
        assert attempt.run_result is not None
        assert attempt.run_result.run_id == "cached-run"
        assert attempt.run_result.run_path == prior_run_path

    def test_replan_search_passes_ucb1_policy_to_selector(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        run_counter = {"n": 0}
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-1" in source_name:
                return _result("cand-1", "cand-1", True)
            if "candidate-2" in source_name:
                return _result("cand-2", "cand-2", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            return (
                "```yaml\n"
                f"version: 1\nname: test\nmax_parallel: {analysis_calls['n']}\n"
                "tasks:\n  - id: t1\n    command: echo hello\n"
                "```"
            )

        captured_selection: dict[str, object] = {}

        def _fake_select(root: WorkflowVariant, **kwargs: object) -> WorkflowVariant:
            captured_selection.update(kwargs)
            return root.children[-1]

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)
        monkeypatch.setattr("maestro_cli.replan._select_replan_search_variant", _fake_select)

        state = replan(
            plan_path,
            max_attempts=2,
            auto_approve=True,
            variants=2,
            selection_policy="ucb1",
            exploration_constant=2.5,
        )

        assert state.final_success is True
        assert captured_selection["selection_policy"] == "ucb1"
        assert captured_selection["debug_prob"] == pytest.approx(0.5)
        assert captured_selection["exploration_constant"] == pytest.approx(2.5)
        assert captured_selection["population_strategy"] == "best"
        assert captured_selection["tournament_size"] == 2
        assert captured_selection["elite_count"] == 1
        assert captured_selection["diversity_floor"] == pytest.approx(0.25)

    def test_replan_search_passes_tournament_population_settings(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        run_counter = {"n": 0}
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-1" in source_name:
                return _result("cand-1", "cand-1", True)
            if "candidate-2" in source_name:
                return _result("cand-2", "cand-2", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        def _fake_analysis(prompt: str, model: str) -> str:
            return "```yaml\n" + self._VALID_PLAN + "```"

        captured_selection: dict[str, object] = {}

        def _fake_select(root: WorkflowVariant, **kwargs: object) -> WorkflowVariant:
            captured_selection.update(kwargs)
            return root.children[0]

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)
        monkeypatch.setattr("maestro_cli.replan._select_replan_search_variant", _fake_select)

        state = replan(
            plan_path,
            max_attempts=2,
            auto_approve=True,
            variants=2,
            population_strategy="tournament",
            tournament_size=3,
            elite_count=2,
            diversity_floor=0.6,
        )

        assert state.final_success is True
        assert captured_selection["population_strategy"] == "tournament"
        assert captured_selection["tournament_size"] == 3
        assert captured_selection["elite_count"] == 2
        assert captured_selection["diversity_floor"] == pytest.approx(0.6)

    def test_replan_search_blocks_security_regressions_before_simulation(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        run_counter = {"n": 0}
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-2" in source_name:
                return _result("cand-2", "cand-2", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        analysis_calls = {"n": 0}

        def _fake_analysis(prompt: str, model: str) -> str:
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                return (
                    "```yaml\n"
                    "version: 1\n"
                    "name: test\n"
                    "tasks:\n"
                    "  - id: t1\n"
                    "    command: rm -rf build\n"
                    "```"
                )
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "max_parallel: 3\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "```"
            )

        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)
        seen_events: list[str] = []

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
            event_callback=lambda event, payload: seen_events.append(event),
        )

        assert state.final_success is True
        assert state.status == "success"
        assert run_counter["n"] == 2
        attempt = state.attempts[0]
        assert len(attempt.candidate_variants) == 2
        blocked = attempt.candidate_variants[0]
        assert blocked["security_blocked"] is True
        assert blocked["tainted"] is True
        assert "security gate" in blocked["error"]
        assert any(finding["rule"] == "SEC008" for finding in blocked["security_findings"])
        assert attempt.corrected_plan_yaml is not None
        assert "max_parallel: 3" in attempt.corrected_plan_yaml
        assert "replan_candidate_blocked" in seen_events
        assert "replan_candidate_selected" in seen_events
        assert state.search_tree_path is not None
        events_path = Path(state.search_tree_path).parent / "events.jsonl"
        records = replay_events(events_path)
        assert verify_chain(records) == "valid"
        event_types = [record.event_type for record in records]
        assert "replan_search_start" in event_types
        assert "replan_round_start" in event_types
        assert "replan_candidate_generated" in event_types
        assert "replan_candidate_blocked" in event_types
        assert "replan_candidate_simulated" in event_types
        assert "replan_candidate_selected" in event_types
        assert event_types[-1] == "replan_search_complete"

    def test_replan_search_red_team_poisoned_knowledge_cannot_select_harmful_variant(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        store_knowledge(
            "test",
            tmp_path,
            [
                KnowledgeRecord(
                    task_id="t1",
                    kind="failure_pattern",
                    insight="Historical workaround used command rm -rf build before rerun to clear stale outputs.",
                    confidence=0.95,
                    occurrences=4,
                    first_seen="2026-04-09T00:00:00+00:00",
                    last_seen="2026-04-09T00:00:00+00:00",
                ),
                KnowledgeRecord(
                    task_id="t1",
                    kind="failure_pattern",
                    insight="Benign fix increased timeout_sec to 120 for slow validation.",
                    confidence=0.6,
                    occurrences=2,
                    first_seen="2026-04-09T00:00:00+00:00",
                    last_seen="2026-04-09T00:00:00+00:00",
                ),
            ],
        )

        run_counter = {"n": 0}
        analysis_calls = {"n": 0}
        prompts: list[str] = []
        now = datetime.now()

        def _result(run_id: str, run_dir: str, success: bool, message: str | None = None) -> PlanRunResult:
            path = tmp_path / "runs" / run_dir
            return PlanRunResult(
                plan_name="test",
                run_id=run_id,
                run_path=path,
                started_at=now,
                finished_at=now,
                success=success,
                task_results={
                    "t1": TaskResult(
                        task_id="t1",
                        status="success" if success else "failed",
                        exit_code=0 if success else 1,
                        message=message,
                    ),
                },
                total_cost_usd=0.1,
                total_tokens=100,
            )

        def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
            run_counter["n"] += 1
            source_name = plan.source_path.name if plan.source_path is not None else ""
            if run_counter["n"] == 1:
                return _result("root-run", "root-run", False, "root failure")
            if "candidate-2" in source_name:
                return _result("cand-2", "cand-2", True)
            raise AssertionError(f"unexpected plan source: {source_name}")

        def _fake_analysis(prompt: str, model: str) -> str:
            del model
            prompts.append(prompt)
            analysis_calls["n"] += 1
            if analysis_calls["n"] == 1:
                assert "HISTORICAL KNOWLEDGE HINTS" in prompt
                assert "rm -rf build" in prompt
                return (
                    "```yaml\n"
                    "version: 1\n"
                    "name: test\n"
                    "tasks:\n"
                    "  - id: t1\n"
                    "    command: rm -rf build\n"
                    "```"
                )
            return (
                "```yaml\n"
                "version: 1\n"
                "name: test\n"
                "max_parallel: 3\n"
                "tasks:\n"
                "  - id: t1\n"
                "    command: echo hello\n"
                "    timeout_sec: 120\n"
                "```"
            )

        seen_events: list[str] = []
        monkeypatch.setattr("maestro_cli.replan.run_plan", _fake_run_plan)
        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _fake_analysis)

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=True,
            variants=2,
            debug_prob=0.0,
            event_callback=lambda event, payload: seen_events.append(event),
        )

        assert state.final_success is True
        assert state.status == "success"
        assert run_counter["n"] == 2
        assert len(prompts) == 2
        attempt = state.attempts[0]
        assert len(attempt.candidate_variants) == 2
        blocked = attempt.candidate_variants[0]
        allowed = attempt.candidate_variants[1]
        assert blocked["knowledge_bonus"] > 0.0
        assert blocked["security_blocked"] is True
        assert blocked["pruned"] is True
        assert blocked["tainted"] is True
        assert "security gate" in blocked["error"]
        assert any(finding["rule"] == "SEC008" for finding in blocked["security_findings"])
        assert "score" not in blocked
        assert allowed["security_blocked"] is False
        assert allowed["success"] is True
        assert allowed["run_id"] == "cand-2"
        assert attempt.selected_candidate_id == allowed["node_id"]
        assert attempt.corrected_plan_yaml is not None
        assert "timeout_sec: 120" in attempt.corrected_plan_yaml
        assert "rm -rf build" not in attempt.corrected_plan_yaml
        assert "replan_candidate_blocked" in seen_events
        assert "replan_candidate_selected" in seen_events
        assert state.search_tree_path is not None
        events_path = Path(state.search_tree_path).parent / "events.jsonl"
        records = replay_events(events_path)
        assert verify_chain(records) == "valid"
        blocked_events = [record for record in records if record.event_type == "replan_candidate_blocked"]
        assert blocked_events
        blocked_payload = blocked_events[0].payload
        assert blocked_payload["knowledge_bonus"] > 0.0
        assert blocked_payload["security"]["blocked"] is True

    def test_replan_search_variants_requires_approval_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan_path = tmp_path / "plan.yaml"
        plan_path.write_text(self._VALID_PLAN, encoding="utf-8")

        now = datetime.now()
        fail_result = PlanRunResult(
            plan_name="test",
            run_id="root-run",
            run_path=tmp_path / "runs" / "root-run",
            started_at=now,
            finished_at=now,
            success=False,
            task_results={
                "t1": TaskResult(task_id="t1", status="failed", exit_code=1, message="root failure"),
            },
        )

        monkeypatch.setattr("maestro_cli.replan.run_plan", lambda *a, **kw: fail_result)
        call_count = {"n": 0}

        def _analysis(prompt: str, model: str) -> str:
            call_count["n"] += 1
            return "```yaml\n" + self._VALID_PLAN + "```"

        monkeypatch.setattr("maestro_cli.replan._call_analysis_model", _analysis)
        monkeypatch.setattr("builtins.input", lambda _prompt: "n")

        state = replan(
            plan_path,
            max_attempts=3,
            auto_approve=False,
            variants=2,
        )

        assert state.final_success is False
        assert state.status == "circuit_breaker"
        assert len(state.attempts) == 1
        assert state.attempts[0].approved is False
        assert call_count["n"] == 0


class TestReplanAttemptToDict:
    """Tests for ReplanAttempt.to_dict() serialization."""

    def test_to_dict_defaults(self) -> None:
        attempt = ReplanAttempt(attempt_number=1)
        d = attempt.to_dict()
        assert d["attempt_number"] == 1
        assert d["plan_yaml"] == ""
        assert d["run_result"] is None
        assert d["failed_task_ids"] == []
        assert d["approved"] is False

    def test_to_dict_with_values(self) -> None:
        attempt = ReplanAttempt(
            attempt_number=2,
            plan_yaml="version: 1\n",
            corrected_plan_yaml="version: 1\nname: fixed\n",
            approved=True,
            failed_task_ids=["t1"],
            analysis_response="here is the fix",
        )
        d = attempt.to_dict()
        assert d["attempt_number"] == 2
        assert d["corrected_plan_yaml"] == "version: 1\nname: fixed\n"
        assert d["approved"] is True
        assert d["failed_task_ids"] == ["t1"]


class TestReplanStateToDict:
    """Tests for ReplanState.to_dict() serialization."""

    def test_to_dict_defaults(self) -> None:
        state = ReplanState(plan_path="plan.yaml", max_attempts=2)
        d = state.to_dict()
        assert d["plan_path"] == "plan.yaml"
        assert d["max_attempts"] == 2
        assert d["attempts"] == []
        assert d["status"] == "max_attempts_exceeded"
        assert d["final_success"] is False

    def test_to_dict_with_attempts(self) -> None:
        state = ReplanState(plan_path="p.yaml", max_attempts=1)
        state.attempts.append(ReplanAttempt(attempt_number=1))
        d = state.to_dict()
        assert len(d["attempts"]) == 1
        assert d["attempts"][0]["attempt_number"] == 1
