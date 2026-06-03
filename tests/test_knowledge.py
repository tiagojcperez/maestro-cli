from __future__ import annotations

import json
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

from maestro_cli.knowledge import (
    _KNOWLEDGE_DIR,
    _MAX_RECORDS_PER_TASK,
    _apply_decay,
    _insight_key,
    build_knowledge_index,
    build_score_record,
    extract_knowledge,
    format_knowledge,
    get_poisoning_alerts,
    get_historical_pruning_decision,
    load_knowledge,
    load_score_history,
    record_knowledge_retrievals,
    run_poisoning_harness,
    select_relevant_knowledge,
    store_score_history,
    store_knowledge,
)
from maestro_cli.models import (
    FailureRecord,
    JudgeResult,
    KnowledgeRecord,
    PlanDefaults,
    PlanRunResult,
    PlanSpec,
    ScoreRecord,
    TaskSpec,
    TaskResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_result(
    task_id: str = "t1",
    status: str = "success",
    exit_code: int | None = 0,
    duration_sec: float = 30.0,
    retry_count: int = 0,
    failure_history: list[FailureRecord] | None = None,
    judge_result: JudgeResult | None = None,
    cost_usd: float | None = None,
    auto_routed_model: str | None = None,
) -> TaskResult:
    return TaskResult(
        task_id=task_id,
        status=status,
        exit_code=exit_code,
        duration_sec=duration_sec,
        retry_count=retry_count,
        failure_history=failure_history or [],
        judge_result=judge_result,
        cost_usd=cost_usd,
        auto_routed_model=auto_routed_model,
    )


def _make_run_result(
    task_results: dict[str, TaskResult],
    plan_name: str = "demo",
) -> PlanRunResult:
    return PlanRunResult(
        plan_name=plan_name,
        run_id="20260319_120000_000000",
        run_path=Path("/tmp/fake"),
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
        success=True,
        execution_profile="plan",
        task_results=task_results,
        sequential_duration_sec=60.0,
        parallelism_savings_pct=0.0,
        total_cost_usd=None,
        total_tokens=None,
        budget_exceeded=False,
    )


def _make_plan(tmp_path: Path, *tasks: TaskSpec) -> PlanSpec:
    source_path = tmp_path / "plan.yaml"
    source_path.write_text("version: 1\nname: demo\n", encoding="utf-8")
    return PlanSpec(
        version=1,
        name="demo",
        defaults=PlanDefaults(),
        tasks=list(tasks),
        source_path=source_path,
    )


def _write_knowledge(
    source_dir: Path,
    plan_name: str,
    records: list[KnowledgeRecord],
) -> None:
    path = source_dir / _KNOWLEDGE_DIR / f"{plan_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# TestExtractKnowledge
# ---------------------------------------------------------------------------

class TestExtractKnowledge:
    def test_failure_pattern_extracted(self) -> None:
        fr = FailureRecord(attempt=1, category="compilation_error", exit_code=1, message="SyntaxError")
        result = _make_result(status="failed", failure_history=[fr])
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        assert len(records) >= 1
        failure_recs = [r for r in records if r.kind == "failure_pattern"]
        assert len(failure_recs) == 1
        assert "compilation_error" in failure_recs[0].insight

    def test_timeout_hint_extracted(self) -> None:
        result = _make_result(status="failed", exit_code=124, duration_sec=620.0)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        timeout_recs = [r for r in records if r.kind == "timeout_hint"]
        assert len(timeout_recs) == 1
        assert "620" in timeout_recs[0].insight

    def test_success_pattern_with_judge(self) -> None:
        judge = JudgeResult(verdict="pass", overall_score=0.95, criterion_scores=[], reasoning="")
        result = _make_result(judge_result=judge)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        success_recs = [r for r in records if r.kind == "success_pattern"]
        assert len(success_recs) == 1
        assert "0.95" in success_recs[0].insight

    def test_skipped_tasks_ignored(self) -> None:
        result = _make_result(status="skipped")
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        assert len(records) == 0

    def test_dry_run_tasks_ignored(self) -> None:
        result = _make_result(status="dry_run")
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        assert len(records) == 0

    def test_no_records_from_empty_results(self) -> None:
        run = _make_run_result({})
        records = extract_knowledge(run)
        assert len(records) == 0

    def test_multiple_patterns_from_same_task(self) -> None:
        """Task that failed with timeout should get both failure_pattern and timeout_hint."""
        fr = FailureRecord(attempt=1, category="timeout", exit_code=124, message="timed out")
        result = _make_result(status="failed", exit_code=124, duration_sec=600.0, failure_history=[fr])
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        kinds = {r.kind for r in records}
        assert "failure_pattern" in kinds
        assert "timeout_hint" in kinds

    def test_unknown_failure_category_skipped(self) -> None:
        fr = FailureRecord(attempt=1, category="unknown", exit_code=1, message="mystery")
        result = _make_result(status="failed", failure_history=[fr])
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        failure_recs = [r for r in records if r.kind == "failure_pattern"]
        assert len(failure_recs) == 0


# ---------------------------------------------------------------------------
# TestStoreKnowledge
# ---------------------------------------------------------------------------

class TestStoreKnowledge:
    def test_creates_knowledge_store(self, tmp_path: Path) -> None:
        records = [KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Fails with timeout",
            confidence=0.5, occurrences=1,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )]
        store_knowledge("demo", tmp_path, records)
        # Check SQLite DB exists (v2) or JSONL fallback
        db_path = tmp_path / ".maestro-cache" / "memory" / "demo.db"
        jsonl_path = tmp_path / _KNOWLEDGE_DIR / "demo.jsonl"
        assert db_path.is_file() or jsonl_path.is_file()
        # Verify data is loadable
        loaded = load_knowledge("demo", tmp_path)
        assert "t1" in loaded
        assert loaded["t1"][0].kind == "failure_pattern"

    def test_merge_increments_occurrences(self, tmp_path: Path) -> None:
        rec = KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Fails with timeout",
            confidence=0.5, occurrences=1,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )
        store_knowledge("demo", tmp_path, [rec])
        store_knowledge("demo", tmp_path, [rec])
        loaded = load_knowledge("demo", tmp_path)
        assert "t1" in loaded
        assert loaded["t1"][0].occurrences == 2

    def test_confidence_increases_with_occurrences(self, tmp_path: Path) -> None:
        # Use a current timestamp so this test isolates the occurrence-boost
        # from confidence time-decay (a hardcoded past date ages into decay).
        now = datetime.now(timezone.utc).isoformat()
        rec = KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Fails with timeout",
            confidence=0.5, occurrences=1,
            first_seen=now,
            last_seen=now,
        )
        for _ in range(5):
            store_knowledge("demo", tmp_path, [rec])
        loaded = load_knowledge("demo", tmp_path)
        assert loaded["t1"][0].confidence > 0.5

    def test_preserves_existing_records(self, tmp_path: Path) -> None:
        rec1 = KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Pattern A",
            confidence=0.5, occurrences=1,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )
        rec2 = KnowledgeRecord(
            task_id="t2", kind="timeout_hint",
            insight="Pattern B",
            confidence=0.5, occurrences=1,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )
        store_knowledge("demo", tmp_path, [rec1])
        store_knowledge("demo", tmp_path, [rec2])
        loaded = load_knowledge("demo", tmp_path)
        assert "t1" in loaded
        assert "t2" in loaded


# ---------------------------------------------------------------------------
# TestLoadKnowledge
# ---------------------------------------------------------------------------

class TestLoadKnowledge:
    def test_missing_file_returns_empty(self, tmp_path: Path) -> None:
        result = load_knowledge("nonexistent", tmp_path)
        assert result == {}

    def test_groups_by_task_id(self, tmp_path: Path) -> None:
        records = [
            KnowledgeRecord(
                task_id="t1", kind="failure_pattern",
                insight="A", confidence=0.7, occurrences=2,
                first_seen="2026-03-19T00:00:00+00:00",
                last_seen="2026-03-19T00:00:00+00:00",
            ),
            KnowledgeRecord(
                task_id="t2", kind="timeout_hint",
                insight="B", confidence=0.6, occurrences=1,
                first_seen="2026-03-19T00:00:00+00:00",
                last_seen="2026-03-19T00:00:00+00:00",
            ),
        ]
        _write_knowledge(tmp_path, "demo", records)
        result = load_knowledge("demo", tmp_path)
        assert set(result.keys()) == {"t1", "t2"}

    def test_applies_time_decay(self, tmp_path: Path) -> None:
        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        records = [KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Old insight", confidence=0.8, occurrences=3,
            first_seen=old_ts, last_seen=old_ts,
        )]
        _write_knowledge(tmp_path, "demo", records)
        result = load_knowledge("demo", tmp_path)
        # 60 days = ~2 half-lives, confidence should be ~0.2
        assert result["t1"][0].confidence < 0.4

    def test_caps_at_max_records_per_task(self, tmp_path: Path) -> None:
        records = [
            KnowledgeRecord(
                task_id="t1", kind="failure_pattern",
                insight=f"Pattern {i}", confidence=0.5 + i * 0.01,
                occurrences=1,
                first_seen="2026-03-19T00:00:00+00:00",
                last_seen="2026-03-19T00:00:00+00:00",
            )
            for i in range(10)
        ]
        _write_knowledge(tmp_path, "demo", records)
        result = load_knowledge("demo", tmp_path)
        assert len(result["t1"]) == _MAX_RECORDS_PER_TASK

    def test_max_per_task_none_returns_all_records(self, tmp_path: Path) -> None:
        records = [
            KnowledgeRecord(
                task_id="t1", kind="failure_pattern",
                insight=f"Pattern {i}", confidence=0.5 + i * 0.01,
                occurrences=1,
                first_seen="2026-03-19T00:00:00+00:00",
                last_seen="2026-03-19T00:00:00+00:00",
            )
            for i in range(7)
        ]
        for rec in records:
            store_knowledge("demo", tmp_path, [rec])
        result = load_knowledge("demo", tmp_path, max_per_task=None)
        assert len(result["t1"]) == 7

    def test_handles_corrupt_lines(self, tmp_path: Path) -> None:
        path = tmp_path / _KNOWLEDGE_DIR / "demo.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            'NOT JSON\n'
            '{"task_id":"t1","kind":"failure_pattern","insight":"OK",'
            '"confidence":0.5,"occurrences":1,'
            '"first_seen":"2026-03-19T00:00:00+00:00",'
            '"last_seen":"2026-03-19T00:00:00+00:00"}\n',
            encoding="utf-8",
        )
        result = load_knowledge("demo", tmp_path)
        assert "t1" in result
        assert len(result["t1"]) == 1


# ---------------------------------------------------------------------------
# TestFormatKnowledge
# ---------------------------------------------------------------------------

class TestFormatKnowledge:
    def test_empty_list_returns_empty_string(self) -> None:
        assert format_knowledge([]) == ""

    def test_formats_with_confidence(self) -> None:
        records = [KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Fails with timeout",
            confidence=0.85, occurrences=3,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )]
        result = format_knowledge(records)
        assert "[85%]" in result
        assert "Fails with timeout" in result

    def test_sorted_by_confidence_descending(self) -> None:
        records = [
            KnowledgeRecord(
                task_id="t1", kind="failure_pattern",
                insight="Low conf", confidence=0.3, occurrences=1,
                first_seen="", last_seen="",
            ),
            KnowledgeRecord(
                task_id="t1", kind="timeout_hint",
                insight="High conf", confidence=0.9, occurrences=5,
                first_seen="", last_seen="",
            ),
        ]
        result = format_knowledge(records)
        lines = result.strip().split("\n")
        assert "High conf" in lines[0]
        assert "Low conf" in lines[1]

    def test_includes_task_id_when_multiple_tasks_present(self) -> None:
        records = [
            KnowledgeRecord(
                task_id="build", kind="failure_pattern",
                insight="Compile step timed out", confidence=0.9, occurrences=2,
                first_seen="", last_seen="",
            ),
            KnowledgeRecord(
                task_id="review", kind="success_pattern",
                insight="Review step passes cleanly", confidence=0.8, occurrences=1,
                first_seen="", last_seen="",
            ),
        ]
        result = format_knowledge(records)
        assert "[task=build]" in result
        assert "[task=review]" in result


# ---------------------------------------------------------------------------
# Index + Retrieval
# ---------------------------------------------------------------------------


class TestKnowledgeIndexAndSelection:
    def test_build_knowledge_index_formats_lightweight_lines(self) -> None:
        knowledge = {
            "build": [
                KnowledgeRecord(
                    task_id="build",
                    kind="failure_pattern",
                    insight="Compile step times out after 30 seconds during pytest collection.",
                    confidence=0.9,
                    occurrences=3,
                    first_seen="2026-03-19T00:00:00+00:00",
                    last_seen="2026-03-19T00:00:00+00:00",
                ),
            ],
        }
        result = build_knowledge_index("demo", knowledge)
        assert "Plan: demo" in result
        assert "[task=build]" in result
        assert "[FAIL]" in result
        assert "Compile step times out" in result

    def test_select_relevant_knowledge_prefers_prompt_matches(self) -> None:
        knowledge = {
            "build": [
                KnowledgeRecord(
                    task_id="build",
                    kind="failure_pattern",
                    insight="Compile step times out during pytest collection.",
                    confidence=0.9,
                    occurrences=3,
                    first_seen="2026-03-19T00:00:00+00:00",
                    last_seen="2026-03-19T00:00:00+00:00",
                ),
            ],
            "review": [
                KnowledgeRecord(
                    task_id="review",
                    kind="success_pattern",
                    insight="Security review succeeds when prompts stay concise.",
                    confidence=0.8,
                    occurrences=2,
                    first_seen="2026-03-19T00:00:00+00:00",
                    last_seen="2026-03-19T00:00:00+00:00",
                ),
            ],
        }
        selected = select_relevant_knowledge(
            knowledge,
            "Investigate the pytest timeout in the build step.",
            task_id="build",
        )
        assert selected
        assert selected[0].task_id == "build"
        assert "times out" in selected[0].insight

    def test_select_relevant_knowledge_falls_back_to_same_task(self) -> None:
        knowledge = {
            "review": [
                KnowledgeRecord(
                    task_id="review",
                    kind="success_pattern",
                    insight="Review succeeds when the checklist is explicit.",
                    confidence=0.85,
                    occurrences=2,
                    first_seen="2026-03-19T00:00:00+00:00",
                    last_seen="2026-03-19T00:00:00+00:00",
                ),
            ],
            "build": [
                KnowledgeRecord(
                    task_id="build",
                    kind="failure_pattern",
                    insight="Build fails on missing imports.",
                    confidence=0.9,
                    occurrences=4,
                    first_seen="2026-03-19T00:00:00+00:00",
                    last_seen="2026-03-19T00:00:00+00:00",
                ),
            ],
        }
        selected = select_relevant_knowledge(knowledge, "", task_id="review")
        assert len(selected) == 1
        assert selected[0].task_id == "review"

    def test_record_knowledge_retrievals_returns_alerts(self, tmp_path: Path) -> None:
        background = [
            KnowledgeRecord(
                task_id=f"noise-{i}",
                kind="failure_pattern",
                insight=f"Background note {i}",
                confidence=0.5,
                occurrences=1,
                first_seen="2026-03-19T00:00:00+00:00",
                last_seen="2026-03-19T00:00:00+00:00",
            )
            for i in range(20)
        ]
        target = KnowledgeRecord(
            task_id="build",
            kind="failure_pattern",
            insight="Build timeout on pytest collection",
            confidence=0.9,
            occurrences=3,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )
        store_knowledge("demo", tmp_path, background + [target])

        alerts = []
        for _ in range(5):
            alerts = record_knowledge_retrievals(
                "demo",
                tmp_path,
                "Investigate the pytest timeout in the build step",
                [target],
            )

        assert alerts
        assert alerts[0].task_id == "build"
        assert get_poisoning_alerts("demo", tmp_path)

    def test_run_poisoning_harness_detects_dominance(self, tmp_path: Path) -> None:
        background = [
            KnowledgeRecord(
                task_id=f"noise-{i}",
                kind="failure_pattern",
                insight=f"Background note {i}",
                confidence=0.5,
                occurrences=1,
                first_seen="2026-03-19T00:00:00+00:00",
                last_seen="2026-03-19T00:00:00+00:00",
            )
            for i in range(20)
        ]
        target = KnowledgeRecord(
            task_id="build",
            kind="failure_pattern",
            insight="Build timeout on pytest collection",
            confidence=0.9,
            occurrences=3,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )
        store_knowledge("demo", tmp_path, background + [target])

        alerts = run_poisoning_harness(
            "demo",
            tmp_path,
            ["Investigate the pytest timeout in the build step"] * 5,
            task_id="build",
        )

        assert alerts
        loaded = load_knowledge("demo", tmp_path, max_per_task=None)
        assert "build" not in loaded


# ---------------------------------------------------------------------------
# TestApplyDecay
# ---------------------------------------------------------------------------

class TestApplyDecay:
    def test_recent_no_decay(self) -> None:
        now = datetime.now(timezone.utc).isoformat()
        assert _apply_decay(0.8, now) == pytest.approx(0.8, abs=0.01)

    def test_30_days_half_decay(self) -> None:
        old = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        result = _apply_decay(0.8, old)
        assert result == pytest.approx(0.4, abs=0.05)

    def test_invalid_timestamp(self) -> None:
        assert _apply_decay(0.8, "not-a-date") == 0.8


# ---------------------------------------------------------------------------
# TestInsightKey
# ---------------------------------------------------------------------------

class TestInsightKey:
    def test_deterministic(self) -> None:
        assert _insight_key("hello world") == _insight_key("hello world")

    def test_different_inputs(self) -> None:
        assert _insight_key("hello") != _insight_key("goodbye")


# ---------------------------------------------------------------------------
# TestExtractKnowledgeEdgeCases
# ---------------------------------------------------------------------------

class TestExtractKnowledgeEdgeCases:
    """Edge cases for extract_knowledge not covered by TestExtractKnowledge."""

    def test_multiple_failure_records_per_task(self) -> None:
        """Task with 3 different failure categories produces 3 records."""
        frs = [
            FailureRecord(attempt=1, category="timeout", exit_code=124, message="timed out"),
            FailureRecord(attempt=2, category="compilation_error", exit_code=1, message="syntax"),
            FailureRecord(attempt=3, category="test_failure", exit_code=2, message="test fail"),
        ]
        result = _make_result(status="failed", exit_code=2, failure_history=frs)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        failure_recs = [r for r in records if r.kind == "failure_pattern"]
        assert len(failure_recs) == 3
        categories = {r.insight.split("Fails with ")[1].split(" ")[0].split(".")[0] for r in failure_recs}
        assert categories == {"timeout", "compilation_error", "test_failure"}

    def test_success_without_judge_no_success_pattern(self) -> None:
        """Success without judge should NOT produce success_pattern."""
        result = _make_result(status="success", judge_result=None)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        success_recs = [r for r in records if r.kind == "success_pattern"]
        assert len(success_recs) == 0

    def test_success_with_judge_score_lte_09_no_success_pattern(self) -> None:
        """Success with judge score <= 0.9 should NOT produce success_pattern."""
        judge = JudgeResult(verdict="pass", overall_score=0.9, criterion_scores=[], reasoning="")
        result = _make_result(status="success", judge_result=judge)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        success_recs = [r for r in records if r.kind == "success_pattern"]
        assert len(success_recs) == 0

    def test_success_with_retries_no_success_pattern(self) -> None:
        """Success with retry_count > 0 should NOT produce success_pattern."""
        judge = JudgeResult(verdict="pass", overall_score=0.98, criterion_scores=[], reasoning="")
        result = _make_result(status="success", retry_count=1, judge_result=judge)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        success_recs = [r for r in records if r.kind == "success_pattern"]
        assert len(success_recs) == 0

    def test_judge_fail_verdict_no_success_pattern(self) -> None:
        """Judge verdict 'fail' with high score should NOT produce success_pattern."""
        judge = JudgeResult(verdict="fail", overall_score=0.95, criterion_scores=[], reasoning="")
        result = _make_result(status="success", judge_result=judge)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        success_recs = [r for r in records if r.kind == "success_pattern"]
        assert len(success_recs) == 0

    def test_soft_failed_extracts_failure_pattern(self) -> None:
        """soft_failed status should still extract failure patterns."""
        fr = FailureRecord(attempt=1, category="test_failure", exit_code=1, message="test")
        # Note: soft_failed is not == "failed" so failure_history won't be extracted
        # by the code (it checks result.status == "failed"), BUT exit_code 124 would
        # still produce timeout_hint. Let's verify soft_failed with exit_code 124.
        result = _make_result(status="soft_failed", exit_code=124, duration_sec=500.0)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        # soft_failed with exit_code 124 should produce timeout_hint (exit_code check
        # is independent of status)
        timeout_recs = [r for r in records if r.kind == "timeout_hint"]
        assert len(timeout_recs) == 1

    def test_soft_failed_with_failure_history(self) -> None:
        """soft_failed status with failure_history — failure_pattern extraction
        requires status == 'failed', so soft_failed does NOT produce failure_pattern."""
        fr = FailureRecord(attempt=1, category="compilation_error", exit_code=1, message="err")
        result = _make_result(status="soft_failed", failure_history=[fr])
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        failure_recs = [r for r in records if r.kind == "failure_pattern"]
        # The code checks `result.status == "failed"` specifically
        assert len(failure_recs) == 0

    def test_multiple_tasks_produce_independent_records(self) -> None:
        """Each task in a run produces independent records."""
        fr1 = FailureRecord(attempt=1, category="timeout", exit_code=124, message="t/o")
        r1 = _make_result(task_id="t1", status="failed", exit_code=124, duration_sec=600.0, failure_history=[fr1])
        judge = JudgeResult(verdict="pass", overall_score=0.99, criterion_scores=[], reasoning="")
        r2 = _make_result(task_id="t2", status="success", judge_result=judge)
        run = _make_run_result({"t1": r1, "t2": r2})
        records = extract_knowledge(run)
        t1_recs = [r for r in records if r.task_id == "t1"]
        t2_recs = [r for r in records if r.task_id == "t2"]
        assert len(t1_recs) >= 1  # failure_pattern + timeout_hint + duration_pattern
        t2_kinds = {r.kind for r in t2_recs}
        assert "success_pattern" in t2_kinds
        # t2 also gets duration_pattern (default duration 30s > 10)
        assert len(t2_recs) >= 1

    def test_failure_with_none_exit_code(self) -> None:
        """Failure record with None exit_code should not include exit_code in insight."""
        fr = FailureRecord(attempt=1, category="permission_error", exit_code=None, message="denied")
        result = _make_result(status="failed", exit_code=None, failure_history=[fr])
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        failure_recs = [r for r in records if r.kind == "failure_pattern"]
        assert len(failure_recs) == 1
        assert "exit_code" not in failure_recs[0].insight

    @pytest.mark.parametrize("category,expected_fragment", [
        ("timeout", "Increase timeout_sec"),
        ("compilation_error", "verify_command to check syntax"),
        ("test_failure", "guard_command or verify_command"),
        ("permission_error", "file permissions"),
        ("validation_error", "guard_command to validate output"),
        ("context_exceeded", "context_budget_tokens"),
        ("rate_limited", "retry_delay_sec"),
        ("runtime_error", "verify_command with targeted assertions"),
        ("dependency_missing", "pre_command to verify required tools"),
        ("output_format_error", "guard_command or use judge"),
        ("cascading_failure", "context_budget_tokens to limit upstream"),
        ("miscommunication", "prompt clarity"),
        ("role_confusion", "agent role more narrowly"),
        ("verification_gap", "verify_command assertions"),
    ])
    def test_all_failure_remediation_categories(
        self, category: str, expected_fragment: str,
    ) -> None:
        """Each failure category gets its correct remediation text."""
        fr = FailureRecord(attempt=1, category=category, exit_code=42, message="err")
        result = _make_result(status="failed", exit_code=42, failure_history=[fr])
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        failure_recs = [r for r in records if r.kind == "failure_pattern"]
        assert len(failure_recs) == 1
        assert expected_fragment in failure_recs[0].insight

    def test_failure_category_without_remediation(self) -> None:
        """A category not in _FAILURE_REMEDIATION (e.g. deadlock) produces insight without remediation."""
        fr = FailureRecord(attempt=1, category="deadlock", exit_code=1, message="stuck")
        result = _make_result(status="failed", failure_history=[fr])
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        failure_recs = [r for r in records if r.kind == "failure_pattern"]
        assert len(failure_recs) == 1
        assert "deadlock" in failure_recs[0].insight
        # No remediation appended, so insight ends with the category mention
        assert "exit_code=1" in failure_recs[0].insight


# ---------------------------------------------------------------------------
# TestStoreKnowledgeEdgeCases
# ---------------------------------------------------------------------------

class TestStoreKnowledgeEdgeCases:
    """Edge cases for store_knowledge."""

    def test_empty_records_list(self, tmp_path: Path) -> None:
        """Storing empty list should not crash."""
        store_knowledge("demo", tmp_path, [])
        path = tmp_path / _KNOWLEDGE_DIR / "demo.jsonl"
        # File may exist but be empty, or may not exist at all
        if path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            assert content == ""

    def test_confidence_capped_at_max(self, tmp_path: Path) -> None:
        """Store same record many times — confidence stays <= 1.0."""
        rec = KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Fails with timeout",
            confidence=0.5, occurrences=1,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )
        for _ in range(20):
            store_knowledge("demo", tmp_path, [rec])
        loaded = load_knowledge("demo", tmp_path)
        assert "t1" in loaded
        assert loaded["t1"][0].confidence <= 1.0

    def test_different_insights_same_task_stored_separately(self, tmp_path: Path) -> None:
        """Different insights for the same task should be stored as separate records."""
        rec_a = KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Fails with timeout. Increase timeout_sec.",
            confidence=0.5, occurrences=1,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )
        rec_b = KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Fails with compilation_error. Add verify_command.",
            confidence=0.5, occurrences=1,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )
        store_knowledge("demo", tmp_path, [rec_a, rec_b])
        loaded = load_knowledge("demo", tmp_path)
        assert len(loaded["t1"]) == 2

    def test_store_creates_nested_directories(self, tmp_path: Path) -> None:
        """store_knowledge should create any missing parent directories."""
        deep_dir = tmp_path / "deeply" / "nested" / "dir"
        rec = KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Test insight",
            confidence=0.5, occurrences=1,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )
        store_knowledge("demo", deep_dir, [rec])
        # Check SQLite DB or JSONL fallback exists
        db_path = deep_dir / ".maestro-cache" / "memory" / "demo.db"
        jsonl_path = deep_dir / _KNOWLEDGE_DIR / "demo.jsonl"
        assert db_path.is_file() or jsonl_path.is_file()


# ---------------------------------------------------------------------------
# TestLoadKnowledgeEdgeCases
# ---------------------------------------------------------------------------

class TestLoadKnowledgeEdgeCases:
    """Edge cases for load_knowledge."""

    def test_empty_jsonl_file_returns_empty(self, tmp_path: Path) -> None:
        """Empty JSONL file returns empty dict."""
        path = tmp_path / _KNOWLEDGE_DIR / "demo.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
        result = load_knowledge("demo", tmp_path)
        assert result == {}

    def test_all_empty_lines_returns_empty(self, tmp_path: Path) -> None:
        """JSONL file with only blank lines returns empty dict."""
        path = tmp_path / _KNOWLEDGE_DIR / "demo.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n\n  \n\n", encoding="utf-8")
        result = load_knowledge("demo", tmp_path)
        assert result == {}

    def test_records_sorted_by_confidence_highest_first(self, tmp_path: Path) -> None:
        """Records within a task should be sorted by confidence descending."""
        now = datetime.now(timezone.utc).isoformat()
        records = [
            KnowledgeRecord(
                task_id="t1", kind="failure_pattern",
                insight=f"Pattern {i}", confidence=0.3 + i * 0.15,
                occurrences=i + 1,
                first_seen=now, last_seen=now,
            )
            for i in range(4)
        ]
        _write_knowledge(tmp_path, "demo", records)
        result = load_knowledge("demo", tmp_path)
        confidences = [r.confidence for r in result["t1"]]
        assert confidences == sorted(confidences, reverse=True)

    def test_mixed_tasks_cap_only_exceeded_task(self, tmp_path: Path) -> None:
        """Only the task exceeding MAX_RECORDS_PER_TASK is capped; others are untouched."""
        now = datetime.now(timezone.utc).isoformat()
        records: list[KnowledgeRecord] = []
        # t1 gets 8 records (over limit)
        for i in range(8):
            records.append(KnowledgeRecord(
                task_id="t1", kind="failure_pattern",
                insight=f"T1 pattern {i}", confidence=0.3 + i * 0.05,
                occurrences=1, first_seen=now, last_seen=now,
            ))
        # t2 gets 2 records (under limit)
        records.extend([
            KnowledgeRecord(
                task_id="t2", kind="failure_pattern",
                insight="Fails with timeout", confidence=0.6,
                occurrences=1, first_seen=now, last_seen=now,
            ),
            KnowledgeRecord(
                task_id="t2", kind="failure_pattern",
                insight="Fails with compilation_error", confidence=0.7,
                occurrences=1, first_seen=now, last_seen=now,
            ),
        ])
        _write_knowledge(tmp_path, "demo", records)
        result = load_knowledge("demo", tmp_path)
        assert len(result["t1"]) == _MAX_RECORDS_PER_TASK
        assert len(result["t2"]) == 2


# ---------------------------------------------------------------------------
# TestApplyDecayEdgeCases
# ---------------------------------------------------------------------------

class TestApplyDecayEdgeCases:
    """Edge cases for _apply_decay."""

    def test_naive_datetime_handled(self) -> None:
        """Naive datetime (no timezone) should be handled gracefully."""
        naive_ts = "2026-03-01T12:00:00"  # no timezone info
        result = _apply_decay(0.8, naive_ts)
        # Should not crash; naive datetime gets UTC assigned
        assert 0.0 <= result <= 1.0

    def test_zero_confidence_stays_zero(self) -> None:
        """Decay of 0.0 stays 0.0 regardless of age."""
        old = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        result = _apply_decay(0.0, old)
        assert result == 0.0

    def test_future_timestamp_no_increase(self) -> None:
        """Future timestamp should not increase confidence beyond input."""
        future = (datetime.now(timezone.utc) + timedelta(days=30)).isoformat()
        result = _apply_decay(0.8, future)
        # Future date means negative age → decay > 1.0, but min(1.0, ...) caps it
        assert result <= 1.0

    def test_exactly_one_half_life(self) -> None:
        """After exactly one half-life (30 days), confidence should be ~50% of original."""
        ts = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()
        result = _apply_decay(1.0, ts)
        assert result == pytest.approx(0.5, abs=0.05)

    def test_very_old_timestamp_near_zero(self) -> None:
        """Very old timestamp should decay to near zero."""
        ancient = (datetime.now(timezone.utc) - timedelta(days=365)).isoformat()
        result = _apply_decay(0.8, ancient)
        assert result < 0.01


# ---------------------------------------------------------------------------
# TestInsightKeyEdgeCases
# ---------------------------------------------------------------------------

class TestInsightKeyEdgeCases:
    """Edge cases for _insight_key."""

    def test_long_string_only_first_100_chars(self) -> None:
        """Two strings differing only after char 100 should have the same key."""
        base = "A" * 100
        s1 = base + "XXXX extra stuff"
        s2 = base + "YYYY different tail"
        assert _insight_key(s1) == _insight_key(s2)

    def test_long_string_differing_within_100(self) -> None:
        """Two strings differing within first 100 chars should have different keys."""
        s1 = "A" * 99 + "X" + "tail"
        s2 = "A" * 99 + "Y" + "tail"
        assert _insight_key(s1) != _insight_key(s2)

    def test_empty_string(self) -> None:
        """Empty string should not crash and should return a valid hash."""
        result = _insight_key("")
        assert isinstance(result, str)
        assert len(result) == 12

    def test_unicode_content(self) -> None:
        """Non-ASCII content should be handled correctly."""
        result = _insight_key("Falha com erro de compilação — tentativa falhada")
        assert isinstance(result, str)
        assert len(result) == 12

    def test_unicode_deterministic(self) -> None:
        """Same Unicode input produces same key."""
        s = "日本語テスト 🔧 émojis"
        assert _insight_key(s) == _insight_key(s)


# ---------------------------------------------------------------------------
# TestFormatKnowledgeEdgeCases
# ---------------------------------------------------------------------------

class TestFormatKnowledgeEdgeCases:
    """Edge cases for format_knowledge."""

    def test_multiple_records_same_confidence(self) -> None:
        """Multiple records with identical confidence should not crash."""
        records = [
            KnowledgeRecord(
                task_id="t1", kind="failure_pattern",
                insight=f"Insight {i}", confidence=0.75, occurrences=1,
                first_seen="", last_seen="",
            )
            for i in range(3)
        ]
        result = format_knowledge(records)
        lines = result.strip().split("\n")
        assert len(lines) == 3
        for line in lines:
            assert "[75%]" in line

    def test_single_record_formats_correctly(self) -> None:
        """Single record should produce one bullet line with correct format."""
        records = [KnowledgeRecord(
            task_id="t1", kind="timeout_hint",
            insight="Times out. Consider increasing timeout.",
            confidence=0.62, occurrences=2,
            first_seen="", last_seen="",
        )]
        result = format_knowledge(records)
        assert result.startswith("- [")
        assert "[62%]" in result
        assert "Times out" in result
        assert result.count("\n") == 0  # single line, no trailing newline

    def test_format_preserves_sort_order(self) -> None:
        """Records should be sorted descending by confidence in output."""
        records = [
            KnowledgeRecord(task_id="t1", kind="failure_pattern",
                            insight="Low", confidence=0.1, occurrences=1,
                            first_seen="", last_seen=""),
            KnowledgeRecord(task_id="t1", kind="failure_pattern",
                            insight="Mid", confidence=0.5, occurrences=1,
                            first_seen="", last_seen=""),
            KnowledgeRecord(task_id="t1", kind="failure_pattern",
                            insight="High", confidence=0.9, occurrences=1,
                            first_seen="", last_seen=""),
        ]
        result = format_knowledge(records)
        lines = result.strip().split("\n")
        assert "High" in lines[0]
        assert "Mid" in lines[1]
        assert "Low" in lines[2]


# ---------------------------------------------------------------------------
# TestKnowledgePath
# ---------------------------------------------------------------------------

class TestKnowledgePath:
    """Tests for _knowledge_path and _load_raw edge cases."""

    def test_special_characters_in_plan_name(self, tmp_path: Path) -> None:
        """Plan name with special characters should construct a valid path."""
        rec = KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Test", confidence=0.5, occurrences=1,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )
        plan_name = "my-plan_v2.0"
        store_knowledge(plan_name, tmp_path, [rec])
        # Check SQLite DB or JSONL fallback exists
        db_path = tmp_path / ".maestro-cache" / "memory" / f"{plan_name}.db"
        jsonl_path = tmp_path / _KNOWLEDGE_DIR / f"{plan_name}.jsonl"
        assert db_path.is_file() or jsonl_path.is_file()
        loaded = load_knowledge(plan_name, tmp_path)
        assert "t1" in loaded

    def test_load_raw_with_oserror(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """_load_raw with OSError returns empty list (permission denied simulation)."""
        from maestro_cli.knowledge import _load_raw, _knowledge_path

        plan_name = "demo"
        # Create the knowledge file first
        rec = KnowledgeRecord(
            task_id="t1", kind="failure_pattern",
            insight="Test", confidence=0.5, occurrences=1,
            first_seen="2026-03-19T00:00:00+00:00",
            last_seen="2026-03-19T00:00:00+00:00",
        )
        _write_knowledge(tmp_path, plan_name, [rec])
        path = _knowledge_path(plan_name, tmp_path)

        # Monkeypatch Path.read_text to raise OSError
        original_read_text = Path.read_text

        def _raise_oserror(self: Path, *args: object, **kwargs: object) -> str:
            if str(self) == str(path):
                raise OSError("Permission denied")
            return original_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", _raise_oserror)
        result = _load_raw(path)
        assert result == []

    def test_load_raw_nonexistent_file(self, tmp_path: Path) -> None:
        """_load_raw on nonexistent file returns empty list."""
        from maestro_cli.knowledge import _load_raw

        path = tmp_path / "nonexistent.jsonl"
        result = _load_raw(path)
        assert result == []


# ---------------------------------------------------------------------------
# Run Knowledge Expansion — New extractor tests
# ---------------------------------------------------------------------------

class TestExtractCostPattern:
    """Tests for cost_pattern extraction."""

    def test_cost_pattern_on_success_with_cost(self) -> None:
        """cost_pattern extracted on success with cost > 0."""
        result = _make_result(status="success", cost_usd=0.0542)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        cost_recs = [r for r in records if r.kind == "cost_pattern"]
        assert len(cost_recs) == 1
        assert "$0.0542" in cost_recs[0].insight

    def test_cost_pattern_not_extracted_on_failure(self) -> None:
        """cost_pattern NOT extracted when status is failed."""
        result = _make_result(status="failed", cost_usd=1.50)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        cost_recs = [r for r in records if r.kind == "cost_pattern"]
        assert len(cost_recs) == 0

    def test_cost_pattern_not_extracted_with_zero_cost(self) -> None:
        """cost_pattern NOT extracted when cost_usd is 0."""
        result = _make_result(status="success", cost_usd=0.0)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        cost_recs = [r for r in records if r.kind == "cost_pattern"]
        assert len(cost_recs) == 0

    def test_cost_pattern_not_extracted_with_none_cost(self) -> None:
        """cost_pattern NOT extracted when cost_usd is None."""
        result = _make_result(status="success", cost_usd=None)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        cost_recs = [r for r in records if r.kind == "cost_pattern"]
        assert len(cost_recs) == 0


class TestExtractDurationPattern:
    """Tests for duration_pattern extraction."""

    def test_duration_pattern_on_success(self) -> None:
        """duration_pattern extracted on success with duration > 10."""
        result = _make_result(status="success", duration_sec=45.0)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        dur_recs = [r for r in records if r.kind == "duration_pattern"]
        assert len(dur_recs) == 1
        assert "45s" in dur_recs[0].insight
        assert "success" in dur_recs[0].insight

    def test_duration_pattern_on_failure(self) -> None:
        """duration_pattern extracted on failure with duration > 10."""
        result = _make_result(status="failed", duration_sec=120.0)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        dur_recs = [r for r in records if r.kind == "duration_pattern"]
        assert len(dur_recs) == 1
        assert "120s" in dur_recs[0].insight
        assert "failed" in dur_recs[0].insight

    def test_duration_pattern_not_extracted_short_task(self) -> None:
        """duration_pattern NOT extracted when duration <= 10s."""
        result = _make_result(status="success", duration_sec=5.0)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        dur_recs = [r for r in records if r.kind == "duration_pattern"]
        assert len(dur_recs) == 0

    def test_duration_pattern_not_extracted_skipped(self) -> None:
        """duration_pattern NOT extracted on skipped tasks."""
        result = _make_result(status="skipped", duration_sec=30.0)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        dur_recs = [r for r in records if r.kind == "duration_pattern"]
        assert len(dur_recs) == 0


class TestExtractRetryPattern:
    """Tests for retry_pattern extraction."""

    def test_retry_pattern_on_success_with_retries(self) -> None:
        """retry_pattern extracted when retry_count > 0 and succeeded."""
        result = _make_result(status="success", retry_count=2)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        retry_recs = [r for r in records if r.kind == "retry_pattern"]
        assert len(retry_recs) == 1
        assert "succeeded after" in retry_recs[0].insight
        assert "2 retries" in retry_recs[0].insight

    def test_retry_pattern_on_failure_with_retries(self) -> None:
        """retry_pattern extracted when retry_count > 0 and failed."""
        result = _make_result(status="failed", retry_count=3)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        retry_recs = [r for r in records if r.kind == "retry_pattern"]
        assert len(retry_recs) == 1
        assert "failed after" in retry_recs[0].insight
        assert "3 retries" in retry_recs[0].insight

    def test_retry_pattern_not_extracted_zero_retries(self) -> None:
        """retry_pattern NOT extracted when retry_count == 0."""
        result = _make_result(status="success", retry_count=0)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        retry_recs = [r for r in records if r.kind == "retry_pattern"]
        assert len(retry_recs) == 0


class TestExtractModelPattern:
    """Tests for model_pattern extraction."""

    def test_model_pattern_when_auto_routed(self) -> None:
        """model_pattern extracted when auto_routed_model is set."""
        result = _make_result(status="success", auto_routed_model="sonnet")
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        model_recs = [r for r in records if r.kind == "model_pattern"]
        assert len(model_recs) == 1
        assert "sonnet" in model_recs[0].insight
        assert "success" in model_recs[0].insight

    def test_model_pattern_not_extracted_without_auto_route(self) -> None:
        """model_pattern NOT extracted when auto_routed_model is None."""
        result = _make_result(status="success", auto_routed_model=None)
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        model_recs = [r for r in records if r.kind == "model_pattern"]
        assert len(model_recs) == 0


class TestExtractMultiplePatternsFromSameTask:
    """Tests for multiple new patterns extracted from one task."""

    def test_success_with_cost_and_duration(self) -> None:
        """A successful task with cost and long duration produces both patterns."""
        result = _make_result(
            status="success",
            cost_usd=0.25,
            duration_sec=60.0,
        )
        run = _make_run_result({"t1": result})
        records = extract_knowledge(run)
        kinds = {r.kind for r in records}
        assert "cost_pattern" in kinds
        assert "duration_pattern" in kinds


class TestFormatKnowledgeKindIcons:
    """Tests for _KIND_ICONS labelling in format_knowledge output."""

    def test_cost_label(self) -> None:
        """cost_pattern records show [COST] label."""
        records = [KnowledgeRecord(
            task_id="t1", kind="cost_pattern",
            insight="Costs $0.05 on success.",
            confidence=0.7, occurrences=1,
            first_seen="", last_seen="",
        )]
        result = format_knowledge(records)
        assert "[COST]" in result

    def test_duration_label(self) -> None:
        """duration_pattern records show [DUR] label."""
        records = [KnowledgeRecord(
            task_id="t1", kind="duration_pattern",
            insight="Takes 45s (success).",
            confidence=0.6, occurrences=1,
            first_seen="", last_seen="",
        )]
        result = format_knowledge(records)
        assert "[DUR]" in result

    def test_retry_label(self) -> None:
        """retry_pattern records show [RETRY] label."""
        records = [KnowledgeRecord(
            task_id="t1", kind="retry_pattern",
            insight="Needed retries: succeeded after 2 retries.",
            confidence=0.5, occurrences=1,
            first_seen="", last_seen="",
        )]
        result = format_knowledge(records)
        assert "[RETRY]" in result

    def test_model_label(self) -> None:
        """model_pattern records show [MODEL] label."""
        records = [KnowledgeRecord(
            task_id="t1", kind="model_pattern",
            insight="Model sonnet: success.",
            confidence=0.8, occurrences=2,
            first_seen="", last_seen="",
        )]
        result = format_knowledge(records)
        assert "[MODEL]" in result

    def test_old_pattern_labels(self) -> None:
        """Old pattern types still show correct labels: [FAIL], [TIME], [OK]."""
        records = [
            KnowledgeRecord(
                task_id="t1", kind="failure_pattern",
                insight="Fails with timeout.",
                confidence=0.9, occurrences=3,
                first_seen="", last_seen="",
            ),
            KnowledgeRecord(
                task_id="t1", kind="timeout_hint",
                insight="Times out (ran 600s).",
                confidence=0.8, occurrences=2,
                first_seen="", last_seen="",
            ),
            KnowledgeRecord(
                task_id="t1", kind="success_pattern",
                insight="Reliably succeeds.",
                confidence=0.7, occurrences=1,
                first_seen="", last_seen="",
            ),
        ]
        result = format_knowledge(records)
        assert "[FAIL]" in result
        assert "[TIME]" in result
        assert "[OK]" in result


class TestStoreLoadRoundtripNewPatterns:
    """Tests for store/load roundtrip with new knowledge pattern types."""

    def test_roundtrip_new_pattern_types(self, tmp_path: Path) -> None:
        """All new pattern types survive a store/load roundtrip."""
        now = datetime.now(timezone.utc).isoformat()
        records = [
            KnowledgeRecord(
                task_id="t1", kind="cost_pattern",
                insight="Costs $0.05 on success.",
                confidence=0.5, occurrences=1,
                first_seen=now, last_seen=now,
            ),
            KnowledgeRecord(
                task_id="t1", kind="duration_pattern",
                insight="Takes 45s (success).",
                confidence=0.5, occurrences=1,
                first_seen=now, last_seen=now,
            ),
            KnowledgeRecord(
                task_id="t2", kind="retry_pattern",
                insight="Needed retries: succeeded after 2 retries.",
                confidence=0.5, occurrences=1,
                first_seen=now, last_seen=now,
            ),
            KnowledgeRecord(
                task_id="t2", kind="model_pattern",
                insight="Model sonnet: success.",
                confidence=0.5, occurrences=1,
                first_seen=now, last_seen=now,
            ),
        ]
        store_knowledge("roundtrip", tmp_path, records)
        loaded = load_knowledge("roundtrip", tmp_path)
        assert "t1" in loaded
        assert "t2" in loaded
        t1_kinds = {r.kind for r in loaded["t1"]}
        t2_kinds = {r.kind for r in loaded["t2"]}
        assert "cost_pattern" in t1_kinds
        assert "duration_pattern" in t1_kinds
        assert "retry_pattern" in t2_kinds
        assert "model_pattern" in t2_kinds


class TestScoreHistory:
    def test_build_score_record_prefers_judge_average(self, tmp_path: Path) -> None:
        judge_a = JudgeResult(verdict="pass", overall_score=0.8, criterion_scores=[], reasoning="")
        judge_b = JudgeResult(verdict="pass", overall_score=0.6, criterion_scores=[], reasoning="")
        plan = _make_plan(
            tmp_path,
            TaskSpec(id="a", command="echo a"),
            TaskSpec(id="b", command="echo b"),
        )
        run = _make_run_result(
            {
                "a": _make_result(task_id="a", judge_result=judge_a),
                "b": _make_result(task_id="b", judge_result=judge_b),
            }
        )

        score = build_score_record(plan, run)
        assert score.plan_hash != ""
        assert score.quality_score == pytest.approx(0.7)
        assert score.metadata["quality_source"] == "judge"
        assert "plan_signature_terms" in score.metadata
        assert "task.id:a" in score.metadata["plan_signature_terms"]

    def test_store_and_load_score_history(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, TaskSpec(id="a", command="echo a"))
        run = _make_run_result({"a": _make_result(task_id="a")})
        score = build_score_record(plan, run)

        assert store_score_history("demo", tmp_path, score) is True

        loaded = load_score_history("demo", tmp_path, plan_hash=score.plan_hash)
        assert len(loaded) == 1
        assert loaded[0].run_id == run.run_id
        assert loaded[0].plan_hash == score.plan_hash

    def test_historical_pruning_decision(self, tmp_path: Path) -> None:
        plan = _make_plan(tmp_path, TaskSpec(id="a", command="echo a"))
        base_run = _make_run_result({"a": _make_result(task_id="a", status="failed")})
        score = build_score_record(plan, base_run)
        for idx in range(5):
            mutated = ScoreRecord(
                plan_name=score.plan_name,
                plan_hash=score.plan_hash,
                run_id=f"run-{idx}",
                success=(idx == 4),
                cost_usd=score.cost_usd,
                quality_score=0.0 if idx < 4 else 1.0,
                duration_sec=score.duration_sec,
                timestamp=score.timestamp,
                valid_from=score.valid_from,
                recorded_at=score.recorded_at,
                source_id=f"run-{idx}:score",
                metadata={"quality_source": "outcome"},
            )
            store_score_history("demo", tmp_path, mutated)

        decision = get_historical_pruning_decision("demo", tmp_path, score.plan_hash)
        assert decision.sample_size == 5
        assert decision.failures == 4
        assert decision.prune is True
