"""Tests for knowledge archive (lesson extraction, persistence, time-decay)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from maestro_cli.models import LessonRecord, WatchIteration
from maestro_cli.watch import (
    _extract_lesson,
    _format_lessons,
    _load_lessons,
    _write_lesson,
)


# ---------------------------------------------------------------------------
# LessonRecord
# ---------------------------------------------------------------------------

class TestLessonRecord:
    def test_to_dict(self) -> None:
        lr = LessonRecord(
            iteration=3, task_id="fetch-data",
            category="timeout_fix", lesson="Increased timeout to 10s",
            confidence=0.9, timestamp="2026-03-17T12:00:00",
        )
        d = lr.to_dict()
        assert d["iteration"] == 3
        assert d["task_id"] == "fetch-data"
        assert d["category"] == "timeout_fix"
        assert d["confidence"] == 0.9


# ---------------------------------------------------------------------------
# _extract_lesson
# ---------------------------------------------------------------------------

class TestExtractLesson:
    def _make_iteration(self, **kwargs: object) -> WatchIteration:
        defaults = {
            "iteration": 1,
            "metric_value": 3.0,
            "best_metric": 3.0,
            "improved": False,
            "action": "rollback",
            "cost_usd": 0.30,
            "duration_sec": 45.0,
            "timestamp": "2026-03-17T12:00:00",
        }
        defaults.update(kwargs)
        return WatchIteration(**defaults)  # type: ignore[arg-type]

    def test_baseline_returns_none(self) -> None:
        wi = self._make_iteration(action="baseline")
        assert _extract_lesson(wi, "") is None

    def test_validation_failed_returns_none(self) -> None:
        wi = self._make_iteration(action="validation_failed")
        assert _extract_lesson(wi, "") is None

    def test_successful_fix_extracts_lesson(self) -> None:
        wi = self._make_iteration(improved=True, action="keep", metric_value=4.0)
        lesson = _extract_lesson(wi, "fetch-data: failed\nhealth-check: success")
        assert lesson is not None
        assert lesson.category == "successful_fix"
        assert lesson.confidence == 0.9
        assert "4.0" in lesson.lesson

    def test_failed_attempt_extracts_lesson(self) -> None:
        wi = self._make_iteration(improved=False, action="rollback", metric_value=3.0)
        lesson = _extract_lesson(wi, "")
        assert lesson is not None
        assert lesson.category == "failed_attempt"
        assert lesson.confidence == 0.5

    def test_lesson_has_timestamp(self) -> None:
        wi = self._make_iteration(improved=True, action="keep", timestamp="2026-03-17T14:00:00")
        lesson = _extract_lesson(wi, "")
        assert lesson is not None
        assert lesson.timestamp == "2026-03-17T14:00:00"


# ---------------------------------------------------------------------------
# _write_lesson / _load_lessons
# ---------------------------------------------------------------------------

class TestWriteAndLoad:
    def test_write_creates_file(self, tmp_path: Path) -> None:
        path = tmp_path / "lessons.jsonl"
        lr = LessonRecord(iteration=1, task_id="t1", category="fix", lesson="Fixed it")
        _write_lesson(path, lr)
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data["task_id"] == "t1"

    def test_write_appends(self, tmp_path: Path) -> None:
        path = tmp_path / "lessons.jsonl"
        _write_lesson(path, LessonRecord(iteration=1, task_id="t1", category="a", lesson="L1"))
        _write_lesson(path, LessonRecord(iteration=2, task_id="t2", category="b", lesson="L2"))
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2

    def test_load_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "lessons.jsonl"
        assert _load_lessons(path) == []

    def test_load_nonexistent_file(self, tmp_path: Path) -> None:
        path = tmp_path / "nope.jsonl"
        assert _load_lessons(path) == []

    def test_load_returns_lessons_sorted_by_confidence(self, tmp_path: Path) -> None:
        path = tmp_path / "lessons.jsonl"
        now = datetime.now().isoformat()
        lessons = [
            {"iteration": 1, "task_id": "a", "category": "x", "lesson": "L1",
             "confidence": 0.5, "timestamp": now},
            {"iteration": 2, "task_id": "b", "category": "y", "lesson": "L2",
             "confidence": 0.9, "timestamp": now},
        ]
        path.write_text(
            "\n".join(json.dumps(l) for l in lessons) + "\n",
            encoding="utf-8",
        )
        loaded = _load_lessons(path)
        assert len(loaded) == 2
        assert loaded[0].confidence >= loaded[1].confidence  # highest first

    def test_load_respects_max_lessons(self, tmp_path: Path) -> None:
        path = tmp_path / "lessons.jsonl"
        now = datetime.now().isoformat()
        lines = []
        for i in range(25):
            lines.append(json.dumps({
                "iteration": i, "task_id": f"t{i}", "category": "x",
                "lesson": f"L{i}", "confidence": 0.8, "timestamp": now,
            }))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        loaded = _load_lessons(path, max_lessons=10)
        assert len(loaded) == 10

    def test_load_malformed_lines_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "lessons.jsonl"
        path.write_text(
            "not json\n"
            '{"iteration":1,"task_id":"a","category":"x","lesson":"ok","confidence":0.8,"timestamp":"2026-03-17T12:00:00"}\n',
            encoding="utf-8",
        )
        loaded = _load_lessons(path)
        assert len(loaded) == 1
        assert loaded[0].task_id == "a"


# ---------------------------------------------------------------------------
# Time-decay
# ---------------------------------------------------------------------------

class TestTimeDecay:
    def test_recent_lesson_high_confidence(self, tmp_path: Path) -> None:
        path = tmp_path / "lessons.jsonl"
        now = datetime.now()
        path.write_text(json.dumps({
            "iteration": 1, "task_id": "t", "category": "x",
            "lesson": "recent", "confidence": 1.0,
            "timestamp": now.isoformat(),
        }) + "\n", encoding="utf-8")
        loaded = _load_lessons(path)
        assert loaded[0].confidence > 0.9  # almost no decay

    def test_old_lesson_decayed_confidence(self, tmp_path: Path) -> None:
        path = tmp_path / "lessons.jsonl"
        old = datetime.now() - timedelta(days=60)  # 2 half-lives
        path.write_text(json.dumps({
            "iteration": 1, "task_id": "t", "category": "x",
            "lesson": "old", "confidence": 1.0,
            "timestamp": old.isoformat(),
        }) + "\n", encoding="utf-8")
        loaded = _load_lessons(path)
        # After 60 days (2 half-lives), confidence should be ~0.25
        assert loaded[0].confidence < 0.3
        assert loaded[0].confidence > 0.1


# ---------------------------------------------------------------------------
# _format_lessons
# ---------------------------------------------------------------------------

class TestFormatLessons:
    def test_empty_lessons(self) -> None:
        result = _format_lessons([])
        assert "No lessons" in result

    def test_formats_with_task_id(self) -> None:
        lessons = [
            LessonRecord(iteration=1, task_id="fetch-data", category="fix",
                        lesson="Increased timeout", confidence=0.85),
        ]
        result = _format_lessons(lessons)
        assert "fetch-data" in result
        assert "Increased timeout" in result
        assert "85%" in result

    def test_formats_without_task_id(self) -> None:
        lessons = [
            LessonRecord(iteration=1, task_id="", category="fix",
                        lesson="General lesson", confidence=0.50),
        ]
        result = _format_lessons(lessons)
        assert "General lesson" in result
        assert "(task:" not in result  # no task id shown
