from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from maestro_cli.knowledge import (
    _CONSOLIDATION_MAX_INSTRUCTIONALITY,
    _CONSOLIDATION_MAX_UNTRUSTED_RATIO,
    _CONSOLIDATION_MIN_CONFIDENCE,
    _CONSOLIDATION_MIN_OCCURRENCES,
    _knowledge_path,
    compact_knowledge,
    consolidate_knowledge,
)
import maestro_cli.memory as memory_mod
from maestro_cli.memory import RecordWithTrust
from maestro_cli.models import KnowledgeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_iso_for_test() -> str:
    """Current UTC timestamp so time-decay is negligible in compaction tests."""
    return datetime.now(timezone.utc).isoformat()


def _record(
    *,
    task_id: str = "build",
    kind: str = "failure_pattern",
    insight: str = "tests fail under load",
    confidence: float = 0.8,
    occurrences: int = 2,
    first_seen: str = "2026-01-01T00:00:00+00:00",
    last_seen: str = "2026-05-01T00:00:00+00:00",
) -> KnowledgeRecord:
    """Build a KnowledgeRecord with sane defaults; override per test."""
    return KnowledgeRecord(
        task_id=task_id,
        kind=kind,
        insight=insight,
        confidence=confidence,
        occurrences=occurrences,
        first_seen=first_seen,
        last_seen=last_seen,
    )


def _write_jsonl(path: Path, records: list[KnowledgeRecord]) -> None:
    """Write KnowledgeRecords to a JSONL file (matches the on-disk format)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wt(
    record: KnowledgeRecord,
    *,
    trust_label: str = "trusted",
    instructionality_score: float = 0.0,
) -> RecordWithTrust:
    return RecordWithTrust(
        record=record,
        trust_label=trust_label,
        instructionality_score=instructionality_score,
    )


# ---------------------------------------------------------------------------
# consolidate_knowledge — fallback path when load_records_detailed raises
# (drives the except-branch that re-loads via load_knowledge and wraps in
#  RecordWithTrust)
# ---------------------------------------------------------------------------


def test_consolidate_falls_back_to_jsonl_when_detailed_loader_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the SQLite-backed detailed loader fails, consolidate reloads from
    JSONL and still produces lessons by wrapping records in RecordWithTrust."""
    plan = "fallbackplan"

    # Force the detailed loader to fail -> exercises the except fallback.
    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("no sqlite backend")

    monkeypatch.setattr(memory_mod, "load_records_detailed", _boom)
    # And force load_knowledge (called in the fallback) to use the JSONL path.
    monkeypatch.setattr(memory_mod, "load_records", _boom)

    # Enough occurrences and high confidence to clear the gates.
    recs = [
        _record(occurrences=_CONSOLIDATION_MIN_OCCURRENCES, confidence=0.9),
    ]
    _write_jsonl(_knowledge_path(plan, tmp_path), recs)

    lessons = consolidate_knowledge(plan, tmp_path)

    assert len(lessons) == 1
    lesson = lessons[0]
    assert lesson.category == "failure_pattern"
    assert lesson.evidence_count >= _CONSOLIDATION_MIN_OCCURRENCES
    # Fallback wrapping defaults trust to "trusted" with zero instructionality.
    assert lesson.source_trust_labels == ["trusted"]
    assert lesson.avg_instructionality == 0.0


def test_consolidate_fallback_returns_empty_when_no_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fallback path with no JSONL records yields an empty list (the
    `if not detailed` short-circuit after wrapping)."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("backend down")

    monkeypatch.setattr(memory_mod, "load_records_detailed", _boom)
    monkeypatch.setattr(memory_mod, "load_records", _boom)

    lessons = consolidate_knowledge("emptyplan", tmp_path)
    assert lessons == []


# ---------------------------------------------------------------------------
# consolidate_knowledge — confidence gate (avg_confidence below threshold)
# ---------------------------------------------------------------------------


def test_consolidate_skips_bucket_below_confidence_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bucket with enough occurrences but low average confidence is skipped
    by the confidence gate (continue), producing no lessons."""
    low_conf = _CONSOLIDATION_MIN_CONFIDENCE - 0.1
    rec = _record(
        confidence=low_conf,
        occurrences=_CONSOLIDATION_MIN_OCCURRENCES + 1,
    )

    monkeypatch.setattr(
        memory_mod,
        "load_records_detailed",
        lambda *_a, **_k: [_wt(rec)],
    )

    lessons = consolidate_knowledge("p", tmp_path)
    assert lessons == []


# ---------------------------------------------------------------------------
# consolidate_knowledge — instructionality safety gate
# ---------------------------------------------------------------------------


def test_consolidate_skips_high_instructionality_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bucket whose average instructionality meets/exceeds the cap is
    rejected by the poisoning safety gate (continue)."""
    rec = _record(confidence=0.9, occurrences=_CONSOLIDATION_MIN_OCCURRENCES + 2)
    high_instr = _CONSOLIDATION_MAX_INSTRUCTIONALITY + 0.2

    monkeypatch.setattr(
        memory_mod,
        "load_records_detailed",
        lambda *_a, **_k: [_wt(rec, instructionality_score=high_instr)],
    )

    lessons = consolidate_knowledge("p", tmp_path)
    assert lessons == []


# ---------------------------------------------------------------------------
# consolidate_knowledge — untrusted-ratio safety gate
# ---------------------------------------------------------------------------


def test_consolidate_skips_predominantly_untrusted_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bucket where the untrusted fraction exceeds the max ratio is rejected
    by the untrusted-evidence safety gate (continue)."""
    # Two records in the same bucket: both untrusted -> ratio 1.0 > 0.5.
    # Same insight => same insight_key => same bucket.
    base_insight = "shared root cause that lands in one bucket"
    rec_a = _record(
        task_id="a",
        insight=base_insight,
        confidence=0.9,
        occurrences=_CONSOLIDATION_MIN_OCCURRENCES,
    )
    rec_b = _record(
        task_id="b",
        insight=base_insight,
        confidence=0.9,
        occurrences=_CONSOLIDATION_MIN_OCCURRENCES,
    )
    # Both untrusted, low instructionality so the instructionality gate passes.
    items = [
        _wt(rec_a, trust_label="untrusted", instructionality_score=0.0),
        _wt(rec_b, trust_label="untrusted", instructionality_score=0.0),
    ]
    assert _CONSOLIDATION_MAX_UNTRUSTED_RATIO < 1.0  # sanity

    monkeypatch.setattr(
        memory_mod,
        "load_records_detailed",
        lambda *_a, **_k: items,
    )

    lessons = consolidate_knowledge("p", tmp_path)
    assert lessons == []


def test_consolidate_keeps_majority_trusted_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bucket where trusted evidence is the majority clears all gates and
    becomes a lesson, with trust labels and instructionality recorded."""
    base_insight = "timeout under heavy parallelism"
    rec_a = _record(
        task_id="a",
        kind="timeout",
        insight=base_insight,
        confidence=0.85,
        occurrences=_CONSOLIDATION_MIN_OCCURRENCES,
    )
    rec_b = _record(
        task_id="b",
        kind="timeout",
        insight=base_insight,
        confidence=0.95,
        occurrences=_CONSOLIDATION_MIN_OCCURRENCES,
    )
    items = [
        _wt(rec_a, trust_label="trusted", instructionality_score=0.1),
        _wt(rec_b, trust_label="trusted", instructionality_score=0.1),
    ]

    monkeypatch.setattr(
        memory_mod,
        "load_records_detailed",
        lambda *_a, **_k: items,
    )

    lessons = consolidate_knowledge("p", tmp_path)
    assert len(lessons) == 1
    lesson = lessons[0]
    assert lesson.category == "timeout"
    # Representative is the highest-confidence record (rec_b's insight).
    assert lesson.lesson == base_insight
    assert sorted(lesson.task_ids) == ["a", "b"]
    assert lesson.source_trust_labels == ["trusted", "trusted"]
    # timeout has a remediation entry.
    assert "timeout" in lesson.recommendation.lower() or lesson.recommendation


# ---------------------------------------------------------------------------
# compact_knowledge — JSONL fallback when SQLite compaction raises
# (drives the except-branch and the dedup/decay/grouping/prune logic)
# ---------------------------------------------------------------------------


def test_compact_fallback_empty_store_returns_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the SQLite compactor fails and no JSONL exists, the fallback
    returns 0 without writing anything."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("compact backend down")

    monkeypatch.setattr(memory_mod, "compact_records", _boom)

    removed = compact_knowledge("nostore", tmp_path)
    assert removed == 0
    assert not _knowledge_path("nostore", tmp_path).is_file()


def test_compact_fallback_dedupes_and_prunes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The JSONL fallback dedupes by (task,kind,insight), keeps top N per task,
    and rewrites the file; it returns the number of records removed."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("compact backend down")

    monkeypatch.setattr(memory_mod, "compact_records", _boom)

    plan = "compactplan"
    path = _knowledge_path(plan, tmp_path)

    # Use a "now"-ish last_seen so time-decay is negligible and confidence
    # comparisons stay stable regardless of the calendar date.
    fresh = _now_iso_for_test()

    # Two duplicates of the same (task, kind, insight) -> dedup to one (the
    # higher-confidence wins).
    dup_lo = _record(
        task_id="t1",
        kind="timeout",
        insight="same insight here",
        confidence=0.4,
        occurrences=2,
        last_seen=fresh,
    )
    dup_hi = _record(
        task_id="t1",
        kind="timeout",
        insight="same insight here",
        confidence=0.7,
        occurrences=5,
        last_seen=fresh,
    )
    # A distinct survivor for a second task.
    other = _record(
        task_id="t2",
        kind="failure_pattern",
        insight="another distinct insight",
        confidence=0.6,
        occurrences=1,
        last_seen=fresh,
    )
    _write_jsonl(path, [dup_lo, dup_hi, other])

    removed = compact_knowledge(plan, tmp_path, max_records_per_task=5)

    # Started with 3 records, dedup collapses the two t1 dups to 1 -> 2 kept.
    assert removed == 1

    # File rewritten with the surviving records.
    surviving = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    survivors_by_task = {r["task_id"]: r for r in surviving}
    assert set(survivors_by_task) == {"t1", "t2"}
    # The higher-confidence duplicate (and its occurrence count) survived. With
    # a fresh last_seen the decayed confidence stays close to the original 0.7
    # and well above the lower duplicate's 0.4.
    assert survivors_by_task["t1"]["confidence"] > 0.5
    assert survivors_by_task["t1"]["occurrences"] == 5


def test_compact_fallback_merges_occurrences_for_weaker_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a later duplicate has lower confidence than the one already kept,
    its occurrence count is merged into the existing record (the else branch
    of the dedup loop), and nothing extra is removed beyond the duplicate."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("no backend")

    monkeypatch.setattr(memory_mod, "compact_records", _boom)

    plan = "mergeoccur"
    path = _knowledge_path(plan, tmp_path)
    fresh = _now_iso_for_test()

    # Higher-confidence record FIRST so it becomes the kept "existing"; the
    # weaker duplicate arrives second and only contributes its occurrence count.
    strong = _record(
        task_id="t1",
        kind="timeout",
        insight="merge me here",
        confidence=0.8,
        occurrences=2,
        last_seen=fresh,
    )
    weak_but_frequent = _record(
        task_id="t1",
        kind="timeout",
        insight="merge me here",
        confidence=0.3,
        occurrences=9,
        last_seen=fresh,
    )
    _write_jsonl(path, [strong, weak_but_frequent])

    removed = compact_knowledge(plan, tmp_path)

    # The two collapse into one record -> exactly one removed.
    assert removed == 1
    surviving = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(surviving) == 1
    # The strong record stayed but absorbed the weaker duplicate's occurrences.
    assert surviving[0]["confidence"] > 0.5
    assert surviving[0]["occurrences"] == 9


def test_compact_fallback_prunes_top_n_per_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With more records than max_records_per_task, only the top N (by
    confidence then occurrences) survive per task."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("no backend")

    monkeypatch.setattr(memory_mod, "compact_records", _boom)

    plan = "prunetop"
    path = _knowledge_path(plan, tmp_path)

    recs = [
        _record(
            task_id="t1",
            kind="timeout",
            insight=f"insight number {i}",
            confidence=0.5 + i * 0.05,
            occurrences=1,
        )
        for i in range(4)
    ]
    _write_jsonl(path, recs)

    removed = compact_knowledge(plan, tmp_path, max_records_per_task=2)

    # 4 distinct records, only top 2 by confidence kept -> 2 removed.
    assert removed == 2
    surviving = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(surviving) == 2
    # The two highest-confidence insights survived (#3 and #2).
    kept_insights = {r["insight"] for r in surviving}
    assert "insight number 3" in kept_insights
    assert "insight number 2" in kept_insights


def test_compact_fallback_drops_records_below_min_confidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Records whose (post-decay) confidence falls below the 0.05 floor are
    removed entirely by the final confidence filter."""

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("no backend")

    monkeypatch.setattr(memory_mod, "compact_records", _boom)

    plan = "lowconf"
    path = _knowledge_path(plan, tmp_path)

    # Already-tiny confidence + a very old last_seen so decay pushes it under
    # the 0.05 floor.
    doomed = _record(
        task_id="t1",
        kind="timeout",
        insight="ancient barely-confident insight",
        confidence=0.06,
        occurrences=1,
        last_seen="2000-01-01T00:00:00+00:00",
    )
    # A healthy survivor so the file is not simply emptied.
    keeper = _record(
        task_id="t2",
        kind="failure_pattern",
        insight="fresh strong insight",
        confidence=0.9,
        occurrences=3,
        last_seen="2026-05-30T00:00:00+00:00",
    )
    _write_jsonl(path, [doomed, keeper])

    removed = compact_knowledge(plan, tmp_path)

    assert removed == 1
    surviving = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(surviving) == 1
    assert surviving[0]["task_id"] == "t2"
