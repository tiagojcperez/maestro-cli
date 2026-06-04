from __future__ import annotations

from pathlib import Path

from maestro_cli.knowledge import (
    _INITIAL_CONFIDENCE,
    _load_raw,
    _normalized_source_id,
)
from maestro_cli.models import KnowledgeRecord


# ---------------------------------------------------------------------------
# _load_raw
# ---------------------------------------------------------------------------


def test_load_raw_missing_file_returns_empty(tmp_path: Path) -> None:
    """A non-existent path short-circuits to an empty list (is_file() False)."""
    result = _load_raw(tmp_path / "does-not-exist.jsonl")
    assert result == []


def test_load_raw_parses_full_record(tmp_path: Path) -> None:
    """A complete JSON line is parsed into a KnowledgeRecord with all fields."""
    path = tmp_path / "plan.jsonl"
    path.write_text(
        '{"task_id": "build", "kind": "failure_pattern", "insight": "flaky test",'
        ' "confidence": 0.75, "occurrences": 3,'
        ' "first_seen": "2026-01-01T00:00:00+00:00",'
        ' "last_seen": "2026-02-01T00:00:00+00:00"}\n',
        encoding="utf-8",
    )
    records = _load_raw(path)
    assert len(records) == 1
    rec = records[0]
    assert isinstance(rec, KnowledgeRecord)
    assert rec.task_id == "build"
    assert rec.kind == "failure_pattern"
    assert rec.insight == "flaky test"
    assert rec.confidence == 0.75
    assert rec.occurrences == 3
    assert rec.first_seen == "2026-01-01T00:00:00+00:00"
    assert rec.last_seen == "2026-02-01T00:00:00+00:00"


def test_load_raw_applies_defaults_for_optional_fields(tmp_path: Path) -> None:
    """Missing confidence/occurrences/first_seen/last_seen fall back to defaults."""
    path = tmp_path / "plan.jsonl"
    path.write_text(
        '{"task_id": "t1", "kind": "success_pattern", "insight": "works"}\n',
        encoding="utf-8",
    )
    records = _load_raw(path)
    assert len(records) == 1
    rec = records[0]
    assert rec.confidence == _INITIAL_CONFIDENCE
    assert rec.occurrences == 1
    assert rec.first_seen == ""
    assert rec.last_seen == ""


def test_load_raw_skips_blank_and_whitespace_lines(tmp_path: Path) -> None:
    """Empty and whitespace-only lines are stripped and skipped."""
    path = tmp_path / "plan.jsonl"
    path.write_text(
        "\n"
        "   \n"
        '{"task_id": "t1", "kind": "timing", "insight": "slow"}\n'
        "\t\n",
        encoding="utf-8",
    )
    records = _load_raw(path)
    assert len(records) == 1
    assert records[0].task_id == "t1"


def test_load_raw_skips_malformed_json_line(tmp_path: Path) -> None:
    """A line that is not valid JSON raises ValueError and is skipped."""
    path = tmp_path / "plan.jsonl"
    path.write_text(
        "this-is-not-json\n"
        '{"task_id": "good", "kind": "timing", "insight": "ok"}\n',
        encoding="utf-8",
    )
    records = _load_raw(path)
    assert len(records) == 1
    assert records[0].task_id == "good"


def test_load_raw_skips_line_missing_required_key(tmp_path: Path) -> None:
    """A JSON object missing a required key (KeyError) is skipped."""
    path = tmp_path / "plan.jsonl"
    # First line lacks "insight" -> KeyError on d["insight"]
    path.write_text(
        '{"task_id": "t1", "kind": "timing"}\n'
        '{"task_id": "t2", "kind": "timing", "insight": "present"}\n',
        encoding="utf-8",
    )
    records = _load_raw(path)
    assert len(records) == 1
    assert records[0].task_id == "t2"


def test_load_raw_skips_line_with_uncoercible_value(tmp_path: Path) -> None:
    """A confidence value that cannot become float raises ValueError and skips."""
    path = tmp_path / "plan.jsonl"
    path.write_text(
        '{"task_id": "t1", "kind": "timing", "insight": "x", "confidence": "not-a-number"}\n'
        '{"task_id": "t2", "kind": "timing", "insight": "y"}\n',
        encoding="utf-8",
    )
    records = _load_raw(path)
    assert len(records) == 1
    assert records[0].task_id == "t2"


def test_load_raw_skips_non_object_json(tmp_path: Path) -> None:
    """A JSON value that is not a dict triggers TypeError on subscript and skips."""
    path = tmp_path / "plan.jsonl"
    path.write_text(
        "[1, 2, 3]\n"
        '{"task_id": "t2", "kind": "timing", "insight": "kept"}\n',
        encoding="utf-8",
    )
    records = _load_raw(path)
    assert len(records) == 1
    assert records[0].task_id == "t2"


def test_load_raw_mixed_valid_and_invalid(tmp_path: Path) -> None:
    """Valid records survive while interspersed bad lines are dropped."""
    path = tmp_path / "plan.jsonl"
    path.write_text(
        '{"task_id": "a", "kind": "timing", "insight": "one"}\n'
        "garbage\n"
        "\n"
        '{"task_id": "b", "kind": "timing", "insight": "two", "occurrences": 4}\n',
        encoding="utf-8",
    )
    records = _load_raw(path)
    assert [r.task_id for r in records] == ["a", "b"]
    assert records[1].occurrences == 4


# ---------------------------------------------------------------------------
# _normalized_source_id
# ---------------------------------------------------------------------------


def test_normalized_source_id_non_task_returns_unchanged() -> None:
    """A source_type other than 'task' is returned verbatim."""
    assert _normalized_source_id("run", "src-123", "build") == "src-123"


def test_normalized_source_id_empty_source_returns_unchanged() -> None:
    """An empty source_id short-circuits regardless of task scope."""
    assert _normalized_source_id("task", "", "build") == ""


def test_normalized_source_id_appends_task_suffix() -> None:
    """A task-scoped source id gets the task id appended as a suffix."""
    assert _normalized_source_id("task", "src-123", "build") == "src-123:build"


def test_normalized_source_id_idempotent_when_suffix_present() -> None:
    """An id already ending in the task suffix is returned unchanged."""
    assert _normalized_source_id("task", "src-123:build", "build") == "src-123:build"
