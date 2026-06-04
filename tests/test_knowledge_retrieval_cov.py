from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

import maestro_cli.memory as memory_mod
from maestro_cli.knowledge import (
    _KNOWLEDGE_DIR,
    get_poisoning_alerts,
    load_knowledge,
    record_knowledge_retrievals,
    run_poisoning_harness,
    select_relevant_knowledge,
)
from maestro_cli.models import KnowledgeRecord


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _record(
    task_id: str = "t1",
    kind: str = "failure_pattern",
    insight: str = "Build timeout on pytest collection",
    confidence: float = 0.7,
    occurrences: int = 2,
    first_seen: str = "2026-03-19T00:00:00+00:00",
    last_seen: str = "2026-03-19T00:00:00+00:00",
) -> KnowledgeRecord:
    return KnowledgeRecord(
        task_id=task_id,
        kind=kind,
        insight=insight,
        confidence=confidence,
        occurrences=occurrences,
        first_seen=first_seen,
        last_seen=last_seen,
    )


def _write_jsonl(
    source_dir: Path,
    plan_name: str,
    records: list[KnowledgeRecord],
) -> None:
    """Write a JSONL knowledge store the way store_knowledge's fallback does."""
    path = source_dir / _KNOWLEDGE_DIR / f"{plan_name}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _force_memory_load_records_to_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``from .memory import load_records`` succeed but the call raise.

    ``load_knowledge`` wraps both the import and the call in a single
    try/except, so a raising callable forces the JSONL fallback branch.
    """

    def _raise(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("simulated memory backend failure")

    monkeypatch.setattr(memory_mod, "load_records", _raise)


# ---------------------------------------------------------------------------
# load_knowledge — JSONL fallback branch (memory backend unavailable)
# ---------------------------------------------------------------------------

class TestLoadKnowledgeJsonlFallback:
    def test_fallback_reads_jsonl_and_groups_by_task(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the SQLite backend errors, records load from JSONL grouped by task."""
        _force_memory_load_records_to_raise(monkeypatch)
        now = datetime.now(timezone.utc).isoformat()
        records = [
            _record(task_id="t1", insight="A", confidence=0.6, last_seen=now),
            _record(task_id="t2", insight="B", confidence=0.5, last_seen=now),
        ]
        _write_jsonl(tmp_path, "demo", records)

        loaded = load_knowledge("demo", tmp_path)

        assert set(loaded.keys()) == {"t1", "t2"}
        assert loaded["t1"][0].insight == "A"
        assert loaded["t2"][0].insight == "B"

    def test_fallback_sorts_by_confidence_descending(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fallback sorts each task's records by confidence (highest first)."""
        _force_memory_load_records_to_raise(monkeypatch)
        now = datetime.now(timezone.utc).isoformat()
        records = [
            _record(task_id="t1", insight="low", confidence=0.30, last_seen=now),
            _record(task_id="t1", insight="high", confidence=0.95, last_seen=now),
            _record(task_id="t1", insight="mid", confidence=0.60, last_seen=now),
        ]
        _write_jsonl(tmp_path, "demo", records)

        loaded = load_knowledge("demo", tmp_path)

        insights = [r.insight for r in loaded["t1"]]
        assert insights[0] == "high"
        assert insights.index("high") < insights.index("mid") < insights.index("low")

    def test_fallback_applies_max_per_task_truncation(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fallback truncates each task group to max_per_task when set."""
        _force_memory_load_records_to_raise(monkeypatch)
        now = datetime.now(timezone.utc).isoformat()
        records = [
            _record(task_id="t1", insight=f"note-{i}", confidence=0.1 * i, last_seen=now)
            for i in range(1, 8)
        ]
        _write_jsonl(tmp_path, "demo", records)

        loaded = load_knowledge("demo", tmp_path, max_per_task=2)

        assert len(loaded["t1"]) == 2

    def test_fallback_keeps_all_when_max_per_task_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """max_per_task=None keeps every record in the group."""
        _force_memory_load_records_to_raise(monkeypatch)
        now = datetime.now(timezone.utc).isoformat()
        records = [
            _record(task_id="t1", insight=f"note-{i}", confidence=0.1 * i, last_seen=now)
            for i in range(1, 8)
        ]
        _write_jsonl(tmp_path, "demo", records)

        loaded = load_knowledge("demo", tmp_path, max_per_task=None)

        assert len(loaded["t1"]) == 7

    def test_fallback_applies_time_decay_to_confidence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Old last_seen timestamps decay confidence below the stored value."""
        _force_memory_load_records_to_raise(monkeypatch)
        old = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        records = [_record(task_id="t1", insight="stale", confidence=0.9, last_seen=old)]
        _write_jsonl(tmp_path, "demo", records)

        loaded = load_knowledge("demo", tmp_path)

        # Decay over ~4 half-lives should drop confidence well below stored 0.9.
        assert loaded["t1"][0].confidence < 0.9

    def test_fallback_missing_file_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Fallback with no JSONL file present yields an empty mapping."""
        _force_memory_load_records_to_raise(monkeypatch)

        loaded = load_knowledge("does-not-exist", tmp_path)

        assert loaded == {}


# ---------------------------------------------------------------------------
# select_relevant_knowledge — early returns
# ---------------------------------------------------------------------------

class TestSelectRelevantKnowledgeEarlyReturns:
    def test_non_positive_max_records_returns_empty(self) -> None:
        """max_records <= 0 short-circuits to an empty list."""
        knowledge = {"t1": [_record(task_id="t1")]}

        assert select_relevant_knowledge(knowledge, "anything", max_records=0) == []
        assert select_relevant_knowledge(knowledge, "anything", max_records=-3) == []

    def test_empty_knowledge_returns_empty(self) -> None:
        """No records across the whole mapping yields an empty list."""
        assert select_relevant_knowledge({}, "investigate timeout") == []
        assert select_relevant_knowledge({"t1": []}, "investigate timeout") == []


# ---------------------------------------------------------------------------
# record_knowledge_retrievals — empty input + backend error
# ---------------------------------------------------------------------------

class TestRecordKnowledgeRetrievals:
    def test_empty_records_returns_empty(self, tmp_path: Path) -> None:
        """No records to record means no alerts and no backend call."""
        assert record_knowledge_retrievals("demo", tmp_path, "prompt", []) == []

    def test_backend_error_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing memory.record_retrievals is caught and returns no alerts."""

        def _raise(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("backend down")

        monkeypatch.setattr(memory_mod, "record_retrievals", _raise)

        result = record_knowledge_retrievals(
            "demo", tmp_path, "prompt text", [_record()]
        )

        assert result == []


# ---------------------------------------------------------------------------
# get_poisoning_alerts — backend error
# ---------------------------------------------------------------------------

class TestGetPoisoningAlerts:
    def test_backend_error_returns_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failing memory.get_poisoning_alerts is caught and returns []."""

        def _raise(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("backend down")

        monkeypatch.setattr(memory_mod, "get_poisoning_alerts", _raise)

        assert get_poisoning_alerts("demo", tmp_path) == []


# ---------------------------------------------------------------------------
# run_poisoning_harness — skip prompts that select nothing
# ---------------------------------------------------------------------------

class TestRunPoisoningHarness:
    def test_skips_prompts_with_no_selection(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When nothing is selected for a prompt, that iteration is skipped.

        With an empty knowledge store, every prompt selects nothing, so the
        ``continue`` branch runs for each prompt and no alerts are produced.
        ``record_retrievals`` is patched to fail loudly if reached, proving the
        skip happens before any retrieval recording.
        """

        def _fail_if_called(*_args: object, **_kwargs: object) -> object:
            raise AssertionError("record_retrievals should not be reached")

        monkeypatch.setattr(memory_mod, "record_retrievals", _fail_if_called)

        alerts = run_poisoning_harness(
            "empty-plan",
            tmp_path,
            ["one prompt", "another prompt"],
            task_id="build",
        )

        assert alerts == []
