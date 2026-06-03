from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from maestro_cli.loader import load_plan
from maestro_cli.runners import _compact_context


class TestCompactContext:
    def test_empty_string(self) -> None:
        assert _compact_context("") == ""

    def test_plain_text_unchanged(self) -> None:
        text = "normal text\nwith two lines\n"
        assert _compact_context(text) == text

    def test_diff_header_stripped(self) -> None:
        text = (
            "diff --git a/foo.py b/foo.py\n"
            "index 123..456 100644\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )

        compacted = _compact_context(text)

        assert "diff --git" not in compacted
        assert "index 123..456 100644" not in compacted
        assert "@@ -1 +1 @@" not in compacted
        assert "-old\n" in compacted
        assert "+new\n" in compacted

    def test_diff_file_path_preserved(self) -> None:
        text = "diff --git a/foo.py b/foo.py\n-old\n+new\n"

        compacted = _compact_context(text)

        assert "--- foo.py\n" in compacted

    def test_traceback_compressed(self) -> None:
        text = (
            "Traceback (most recent call last):\n"
            "  File \"a.py\", line 1, in <module>\n"
            "    one()\n"
            "  File \"b.py\", line 2, in one\n"
            "    two()\n"
            "  File \"c.py\", line 3, in two\n"
            "    three()\n"
            "  File \"d.py\", line 4, in three\n"
            "    boom()\n"
        )

        compacted = _compact_context(text)

        assert 'File "a.py", line 1' in compacted
        assert 'File "d.py", line 4' in compacted
        assert "(2 frames omitted)" in compacted
        assert 'File "b.py", line 2' not in compacted
        assert 'File "c.py", line 3' not in compacted

    def test_short_traceback_unchanged(self) -> None:
        text = (
            "Traceback (most recent call last):\n"
            "  File \"a.py\", line 1, in <module>\n"
            "    one()\n"
            "  File \"b.py\", line 2, in one\n"
            "    boom()\n"
        )

        assert _compact_context(text) == text

    def test_maestro_prefix_dedup(self) -> None:
        text = (
            "[maestro] same line\n"
            "[maestro] same line\n"
            "[maestro] same line\n"
        )

        assert _compact_context(text) == "[maestro] same line\n"

    def test_json_minified(self) -> None:
        text = '{\n  "a": 1,\n  "b": {\n    "c": 2\n  }\n}\n'

        compacted = _compact_context(text)

        assert compacted == '{"a":1,"b":{"c":2}}\n'

    def test_invalid_json_unchanged(self) -> None:
        text = '{\n  "a": 1,\n}\n'

        assert _compact_context(text) == text

    def test_combined_transformers(self) -> None:
        text = (
            "diff --git a/foo.py b/foo.py\n"
            "index 123..456 100644\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
            "Traceback (most recent call last):\n"
            "  File \"a.py\", line 1, in <module>\n"
            "    one()\n"
            "  File \"b.py\", line 2, in one\n"
            "    two()\n"
            "  File \"c.py\", line 3, in two\n"
            "    three()\n"
            "  File \"d.py\", line 4, in three\n"
            "    boom()\n"
            "{\n"
            '  "k": "v",\n'
            '  "n": 1\n'
            "}\n"
        )

        compacted = _compact_context(text)

        assert "diff --git" not in compacted
        assert "--- foo.py\n" in compacted
        assert "-old\n" in compacted
        assert "+new\n" in compacted
        assert "frames omitted" in compacted
        assert '{"k":"v","n":1}' in compacted


class TestCompactLoaderField:
    def _write_plan(self, tmp_path: Path, context_compact: bool | None = None) -> Path:
        plan = {
            "version": 1,
            "name": "compact-test",
            "tasks": [
                {
                    "id": "t1",
                    "command": "echo hello",
                }
            ],
        }
        if context_compact is not None:
            plan["tasks"][0]["context_compact"] = context_compact

        plan_file = tmp_path / "plan.yaml"
        plan_file.write_text(yaml.safe_dump(plan), encoding="utf-8")
        return plan_file

    def test_context_compact_default_false(self, tmp_path: Path) -> None:
        plan_file = self._write_plan(tmp_path)

        plan = load_plan(plan_file)

        assert plan.tasks[0].context_compact is False

    def test_context_compact_true(self, tmp_path: Path) -> None:
        plan_file = self._write_plan(tmp_path, context_compact=True)

        plan = load_plan(plan_file)

        assert plan.tasks[0].context_compact is True

    def test_context_compact_in_to_dict(self, tmp_path: Path) -> None:
        plan_file = self._write_plan(tmp_path, context_compact=True)

        plan = load_plan(plan_file)

        assert plan.tasks[0].to_dict()["context_compact"] is True
