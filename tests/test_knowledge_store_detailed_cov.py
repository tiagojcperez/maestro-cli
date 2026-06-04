from __future__ import annotations

import json
from pathlib import Path

import pytest

import maestro_cli.memory as memory_mod
from maestro_cli.knowledge import (
    _INITIAL_CONFIDENCE,
    _KNOWLEDGE_DIR,
    _insight_key,
    store_knowledge_detailed,
)
from maestro_cli.models import KnowledgeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _force_jsonl_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the SQLite memory backend raise so the JSONL fallback runs.

    ``store_knowledge_detailed`` imports ``store_records_detailed`` from the
    memory module at call time, so re-binding the module attribute redirects
    the lookup performed inside the function.
    """

    def _boom(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("memory backend unavailable")

    monkeypatch.setattr(memory_mod, "store_records_detailed", _boom)


def _make_record(
    *,
    task_id: str = "t1",
    kind: str = "failure_pattern",
    insight: str = "Fails with timeout",
    confidence: float = 0.5,
    occurrences: int = 1,
    last_seen: str = "2026-03-19T00:00:00+00:00",
) -> KnowledgeRecord:
    return KnowledgeRecord(
        task_id=task_id,
        kind=kind,  # type: ignore[arg-type]
        insight=insight,
        confidence=confidence,
        occurrences=occurrences,
        first_seen="2026-03-19T00:00:00+00:00",
        last_seen=last_seen,
    )


def _jsonl_path(tmp_path: Path, plan_name: str = "demo") -> Path:
    return tmp_path / _KNOWLEDGE_DIR / f"{plan_name}.jsonl"


def _read_records(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Empty-records short circuit (within the fallback path)
# ---------------------------------------------------------------------------

class TestEmptyRecordsFallback:
    def test_empty_records_returns_empty_and_writes_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With the SQLite backend disabled, an empty list short-circuits to []."""
        _force_jsonl_fallback(monkeypatch)
        outcomes = store_knowledge_detailed("demo", tmp_path, [])
        assert outcomes == []
        # No JSONL file should be written when there is nothing to store.
        assert not _jsonl_path(tmp_path).exists()


# ---------------------------------------------------------------------------
# Insert path
# ---------------------------------------------------------------------------

class TestInsertFallback:
    def test_single_insert_writes_file_and_returns_inserted_outcome(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_jsonl_fallback(monkeypatch)
        rec = _make_record()
        outcomes = store_knowledge_detailed("demo", tmp_path, [rec])

        assert len(outcomes) == 1
        outcome = outcomes[0]
        assert outcome.operation == "inserted"
        assert "accepted" in outcome.outcome
        assert outcome.task_id == "t1"
        assert outcome.kind == "failure_pattern"

        path = _jsonl_path(tmp_path)
        assert path.is_file()
        stored = _read_records(path)
        assert len(stored) == 1
        assert stored[0]["task_id"] == "t1"
        assert stored[0]["insight"] == "Fails with timeout"

    def test_creates_nested_parent_directories(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The fallback creates missing parent dirs before writing."""
        _force_jsonl_fallback(monkeypatch)
        deep = tmp_path / "a" / "b" / "c"
        outcomes = store_knowledge_detailed("demo", deep, [_make_record()])
        assert len(outcomes) == 1
        assert _jsonl_path(deep).is_file()

    def test_two_distinct_records_both_inserted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_jsonl_fallback(monkeypatch)
        rec_a = _make_record(insight="Pattern A")
        rec_b = _make_record(insight="Pattern B")
        outcomes = store_knowledge_detailed("demo", tmp_path, [rec_a, rec_b])
        assert len(outcomes) == 2
        assert all(o.operation == "inserted" for o in outcomes)
        stored = _read_records(_jsonl_path(tmp_path))
        insights = {r["insight"] for r in stored}
        assert insights == {"Pattern A", "Pattern B"}


# ---------------------------------------------------------------------------
# Merge path (key already present in the existing store)
# ---------------------------------------------------------------------------

class TestMergeFallback:
    def test_merge_increments_occurrences_and_confidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_jsonl_fallback(monkeypatch)
        rec = _make_record()

        first = store_knowledge_detailed("demo", tmp_path, [rec])
        assert first[0].operation == "inserted"

        # Same task_id/kind/insight key triggers the merge branch.
        rec2 = _make_record(last_seen="2026-04-01T00:00:00+00:00")
        second = store_knowledge_detailed("demo", tmp_path, [rec2])
        assert second[0].operation == "merged"

        stored = _read_records(_jsonl_path(tmp_path))
        assert len(stored) == 1
        merged = stored[0]
        assert merged["occurrences"] == 2
        assert merged["last_seen"] == "2026-04-01T00:00:00+00:00"
        # Confidence is recomputed upward from the initial baseline.
        assert merged["confidence"] > _INITIAL_CONFIDENCE

    def test_repeated_merges_clamp_confidence_at_max(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_jsonl_fallback(monkeypatch)
        rec = _make_record()
        for _ in range(15):
            store_knowledge_detailed("demo", tmp_path, [_make_record()])
        store_knowledge_detailed("demo", tmp_path, [rec])

        stored = _read_records(_jsonl_path(tmp_path))
        assert len(stored) == 1
        assert stored[0]["confidence"] <= 1.0

    def test_existing_disk_records_loaded_before_merge(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A pre-existing JSONL entry is indexed and merged into, not duplicated."""
        _force_jsonl_fallback(monkeypatch)
        path = _jsonl_path(tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        rec = _make_record(occurrences=3)
        path.write_text(
            json.dumps(rec.to_dict(), ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        outcomes = store_knowledge_detailed("demo", tmp_path, [_make_record()])
        assert outcomes[0].operation == "merged"
        stored = _read_records(path)
        assert len(stored) == 1
        assert stored[0]["occurrences"] == 4


# ---------------------------------------------------------------------------
# Trust labelling and source_id normalization
# ---------------------------------------------------------------------------

class TestTrustAndSourceId:
    def test_task_source_type_marks_trusted_and_appends_task_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_jsonl_fallback(monkeypatch)
        rec = _make_record(task_id="build")
        outcomes = store_knowledge_detailed(
            "demo",
            tmp_path,
            [rec],
            source_type="task",
            source_id="run-123",
        )
        outcome = outcomes[0]
        assert outcome.trust_label == "trusted"
        assert outcome.source_type == "task"
        # _normalized_source_id appends ":<task_id>" for task-scoped pointers.
        assert outcome.source_id == "run-123:build"
        assert outcome.instructionality_score == 0.0

    def test_non_task_source_type_marks_untrusted_without_suffix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_jsonl_fallback(monkeypatch)
        rec = _make_record(task_id="build")
        outcomes = store_knowledge_detailed(
            "demo",
            tmp_path,
            [rec],
            source_type="external",
            source_id="web-1",
        )
        outcome = outcomes[0]
        assert outcome.trust_label == "untrusted"
        assert outcome.source_type == "external"
        # Non-task source ids are passed through unchanged.
        assert outcome.source_id == "web-1"

    def test_task_source_id_not_double_suffixed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An already-suffixed task source id is left intact."""
        _force_jsonl_fallback(monkeypatch)
        rec = _make_record(task_id="build")
        outcomes = store_knowledge_detailed(
            "demo",
            tmp_path,
            [rec],
            source_type="task",
            source_id="run-123:build",
        )
        assert outcomes[0].source_id == "run-123:build"


# ---------------------------------------------------------------------------
# Insight-key dedup boundary (different insights are distinct keys)
# ---------------------------------------------------------------------------

class TestInsightKeyBoundary:
    def test_distinct_insights_produce_distinct_keys(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _force_jsonl_fallback(monkeypatch)
        rec_a = _make_record(insight="Alpha insight text")
        rec_b = _make_record(insight="Beta insight text")
        # Sanity: the dedup keys actually differ.
        assert _insight_key(rec_a.insight) != _insight_key(rec_b.insight)

        store_knowledge_detailed("demo", tmp_path, [rec_a])
        outcomes = store_knowledge_detailed("demo", tmp_path, [rec_b])
        assert outcomes[0].operation == "inserted"
        stored = _read_records(_jsonl_path(tmp_path))
        assert len(stored) == 2
