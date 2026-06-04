from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.scaffold import (
    _detect_large_files,
    _load_library,
    _merge_library_into_brief,
)


# ---------------------------------------------------------------------------
# _merge_library_into_brief — non-dict skip branches
# ---------------------------------------------------------------------------


class TestMergeNonDictSkips:
    def test_non_dict_brief_task_is_skipped(self) -> None:
        """A non-dict entry in the user brief tasks is ignored (continue branch)."""
        lib = _load_library("refactor")
        lib_ids = {t["id"] for t in lib["tasks"]}
        # First entry is junk (string), should be silently skipped; the dict
        # override that follows must still take effect.
        brief_tasks = [
            "not-a-dict",  # type: ignore[list-item]
            {"id": "analyse-code", "prompt_hint": "Custom prompt"},
        ]
        merged_tasks, _ = _merge_library_into_brief(lib, brief_tasks, {})
        merged_ids = [t["id"] for t in merged_tasks]
        # The junk entry did not pollute the merged output.
        assert "not-a-dict" not in merged_ids
        # All library task IDs survived.
        for lid in lib_ids:
            assert lid in merged_ids
        analyse = next(t for t in merged_tasks if t["id"] == "analyse-code")
        assert analyse["prompt_hint"] == "Custom prompt"

    def test_non_dict_brief_task_alone_produces_no_extra(self) -> None:
        """Only junk brief tasks -> merged output equals library tasks exactly."""
        lib = _load_library("refactor")
        brief_tasks = [42, None, ["x"]]  # type: ignore[list-item]
        merged_tasks, _ = _merge_library_into_brief(lib, brief_tasks, {})
        merged_ids = [t["id"] for t in merged_tasks]
        lib_ids = [t["id"] for t in lib["tasks"]]
        assert merged_ids == lib_ids

    def test_non_dict_library_task_is_skipped(self) -> None:
        """A non-dict entry inside library['tasks'] is ignored (continue branch)."""
        library = {
            "goal": "synthetic",
            "topology": "linear",
            "tasks": [
                "junk-string",  # non-dict — must be skipped
                {"id": "real-task", "task_type": "shell"},
            ],
        }
        merged_tasks, _ = _merge_library_into_brief(library, [], {})
        merged_ids = [t["id"] for t in merged_tasks]
        # Only the dict library task survived.
        assert merged_ids == ["real-task"]

    def test_non_dict_library_task_among_several(self) -> None:
        """Multiple junk entries in library tasks are all skipped; dicts kept.

        Uses an empty brief so the merge loop over library tasks is the only
        path that touches the non-dict entries.
        """
        library = {
            "tasks": [
                {"id": "first", "task_type": "shell"},
                None,  # non-dict — skipped
                {"id": "second", "task_type": "implementation"},
                [1, 2, 3],  # non-dict — skipped
            ],
        }
        merged_tasks, _ = _merge_library_into_brief(library, [], {})
        merged_ids = [t["id"] for t in merged_tasks]
        assert merged_ids == ["first", "second"]


# ---------------------------------------------------------------------------
# _detect_large_files — OSError on read_text is swallowed
# ---------------------------------------------------------------------------


class TestDetectLargeFilesOSError:
    def test_read_text_oserror_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file that exists and is a regular file but fails to read with
        OSError is silently skipped (except OSError: continue branch)."""
        f = tmp_path / "boom.py"
        # Real file so exists() and is_file() both pass.
        f.write_text("\n".join(f"# line {i}" for i in range(400)), encoding="utf-8")

        real_read_text = Path.read_text

        def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == "boom.py":
                raise OSError("simulated read failure")
            return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", fake_read_text)

        result = _detect_large_files("edit boom.py to add logging", str(tmp_path))
        # The unreadable file was skipped rather than raising.
        assert result == []

    def test_oserror_on_one_file_others_still_detected(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One file raises OSError (skipped); a second large file is still found."""
        bad = tmp_path / "bad.py"
        bad.write_text("\n".join(f"# {i}" for i in range(400)), encoding="utf-8")
        good = tmp_path / "good.py"
        good.write_text("\n".join(f"# {i}" for i in range(400)), encoding="utf-8")

        real_read_text = Path.read_text

        def fake_read_text(self: Path, *args: object, **kwargs: object) -> str:
            if self.name == "bad.py":
                raise OSError("simulated read failure")
            return real_read_text(self, *args, **kwargs)  # type: ignore[arg-type]

        monkeypatch.setattr(Path, "read_text", fake_read_text)

        result = _detect_large_files("edit bad.py and good.py", str(tmp_path))
        assert result == ["good.py"]
