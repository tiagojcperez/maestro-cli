"""Tests for the workspace_index module (v0.7.0 — Recursive Context).

Covers: FileEntry, WorkspaceIndex, build_workspace_index, build_tree_summary,
        load_cached_index, save_index, quick_root_hash, and private helpers.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from maestro_cli.workspace_index import (
    INDEX_SCHEMA_VERSION,
    FileEntry,
    WorkspaceIndex,
    _DEFAULT_EXCLUDES,
    _MAX_FILES,
    _hash_file_head,
    _infer_language,
    _read_first_lines,
    _should_exclude,
    build_tree_summary,
    build_workspace_index,
    default_index_cache_dir,
    load_cached_index,
    quick_root_hash,
    save_index,
)


# ---------------------------------------------------------------------------
# FileEntry
# ---------------------------------------------------------------------------


class TestFileEntry:
    def test_rel_path_property_mirrors_path(self) -> None:
        entry = FileEntry(path="src/main.py", size_bytes=100, mtime_ns=0, sha256="abc")
        assert entry.rel_path == "src/main.py"

    def test_size_property_mirrors_size_bytes(self) -> None:
        entry = FileEntry(path="a.py", size_bytes=512, mtime_ns=0, sha256="abc")
        assert entry.size == 512

    def test_to_dict_contains_all_fields(self) -> None:
        entry = FileEntry(
            path="src/a.py",
            size_bytes=50,
            mtime_ns=123456,
            sha256="deadbeef",
            language="python",
            sha256_head="cafe",
            first_lines=["line1", "line2"],
        )
        d = entry.to_dict()
        assert d["path"] == "src/a.py"
        assert d["size_bytes"] == 50
        assert d["mtime_ns"] == 123456
        assert d["sha256"] == "deadbeef"
        assert d["language"] == "python"
        assert d["sha256_head"] == "cafe"
        assert d["first_lines"] == ["line1", "line2"]

    def test_to_dict_includes_backward_compat_keys(self) -> None:
        entry = FileEntry(path="x.py", size_bytes=10, mtime_ns=0, sha256="aa")
        d = entry.to_dict()
        assert d["rel_path"] == "x.py"
        assert d["size"] == 10

    def test_from_dict_roundtrip(self) -> None:
        entry = FileEntry(
            path="lib/utils.py",
            size_bytes=200,
            mtime_ns=999,
            sha256="ff00",
            language="python",
            sha256_head="ee11",
            first_lines=["import os"],
        )
        restored = FileEntry.from_dict(entry.to_dict())
        assert restored.path == entry.path
        assert restored.size_bytes == entry.size_bytes
        assert restored.mtime_ns == entry.mtime_ns
        assert restored.sha256 == entry.sha256
        assert restored.language == entry.language
        assert restored.sha256_head == entry.sha256_head
        assert restored.first_lines == entry.first_lines

    def test_from_dict_backward_compat_keys(self) -> None:
        raw = {"rel_path": "old/compat.py", "size": 77, "mtime_ns": 1, "sha256": "zz"}
        entry = FileEntry.from_dict(raw)
        assert entry.path == "old/compat.py"
        assert entry.size_bytes == 77

    def test_from_dict_none_language(self) -> None:
        raw = {"path": "x.bin", "size_bytes": 0, "mtime_ns": 0, "sha256": "", "language": None}
        entry = FileEntry.from_dict(raw)
        assert entry.language is None

    def test_from_dict_none_sha256_head(self) -> None:
        raw = {"path": "x.py", "size_bytes": 0, "mtime_ns": 0, "sha256": "", "sha256_head": None}
        entry = FileEntry.from_dict(raw)
        assert entry.sha256_head is None

    def test_from_dict_missing_first_lines_defaults_to_empty(self) -> None:
        raw = {"path": "x.py", "size_bytes": 0, "mtime_ns": 0, "sha256": ""}
        entry = FileEntry.from_dict(raw)
        assert entry.first_lines == []


# ---------------------------------------------------------------------------
# WorkspaceIndex
# ---------------------------------------------------------------------------


class TestWorkspaceIndex:
    def _make_entry(self, path: str = "a.py") -> FileEntry:
        return FileEntry(path=path, size_bytes=10, mtime_ns=0, sha256="aa")

    def test_entries_property_mirrors_files(self) -> None:
        e = self._make_entry()
        idx = WorkspaceIndex(files=[e])
        assert idx.entries is idx.files

    def test_root_hash_prefers_content_id(self) -> None:
        idx = WorkspaceIndex(snapshot_id="snap123", content_id="content456")
        assert idx.root_hash == "content456"

    def test_root_hash_falls_back_to_snapshot_id_when_no_content_id(self) -> None:
        idx = WorkspaceIndex(snapshot_id="snap123", content_id="")
        assert idx.root_hash == "snap123"

    def test_total_bytes_property_mirrors_total_size_bytes(self) -> None:
        idx = WorkspaceIndex(total_size_bytes=1024)
        assert idx.total_bytes == 1024

    def test_to_dict_contains_all_top_level_keys(self) -> None:
        idx = WorkspaceIndex(
            workspace_root="/tmp/proj",
            snapshot_id="s1",
            content_id="c1",
            file_count=1,
            total_size_bytes=42,
        )
        d = idx.to_dict()
        assert d["schema_version"] == INDEX_SCHEMA_VERSION
        assert d["workspace_root"] == "/tmp/proj"
        assert d["snapshot_id"] == "s1"
        assert d["content_id"] == "c1"
        assert d["file_count"] == 1
        assert d["total_size_bytes"] == 42

    def test_to_dict_includes_backward_compat_root_hash(self) -> None:
        idx = WorkspaceIndex(snapshot_id="s", content_id="c")
        d = idx.to_dict()
        assert "root_hash" in d
        assert d["root_hash"] == "c"

    def test_to_dict_files_and_entries_both_present(self) -> None:
        e = self._make_entry()
        idx = WorkspaceIndex(files=[e])
        d = idx.to_dict()
        assert "files" in d
        assert "entries" in d
        assert len(d["files"]) == 1
        assert len(d["entries"]) == 1

    def test_from_dict_roundtrip(self, tmp_path: Path) -> None:
        root = tmp_path / "proj"
        root.mkdir()
        e = FileEntry(path="x.py", size_bytes=5, mtime_ns=1, sha256="ff")
        idx = WorkspaceIndex(
            workspace_root=str(root),
            snapshot_id="s1",
            content_id="c1",
            file_count=1,
            total_size_bytes=5,
            files=[e],
        )
        restored = WorkspaceIndex.from_dict(idx.to_dict())
        assert restored.workspace_root == str(root)
        assert restored.snapshot_id == "s1"
        assert restored.content_id == "c1"
        assert len(restored.files) == 1
        assert restored.files[0].path == "x.py"

    def test_from_dict_accepts_entries_key(self) -> None:
        raw = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "workspace_root": "/tmp",
            "snapshot_id": "",
            "content_id": "",
            "entries": [{"path": "a.py", "size_bytes": 1, "mtime_ns": 0, "sha256": ""}],
        }
        idx = WorkspaceIndex.from_dict(raw)
        assert len(idx.files) == 1

    def test_tree_summary_property_returns_string(self) -> None:
        e = self._make_entry("src/main.py")
        idx = WorkspaceIndex(files=[e])
        summary = idx.tree_summary
        assert "src" in summary or "main.py" in summary


# ---------------------------------------------------------------------------
# _should_exclude
# ---------------------------------------------------------------------------


class TestShouldExclude:
    def test_excludes_dot_git_dir(self) -> None:
        assert _should_exclude(".git") is True

    def test_excludes_pycache(self) -> None:
        assert _should_exclude("__pycache__") is True

    def test_excludes_node_modules(self) -> None:
        assert _should_exclude("node_modules") is True

    def test_excludes_pyc_glob_pattern(self) -> None:
        assert _should_exclude("module.pyc") is True

    def test_excludes_tmp_glob_pattern(self) -> None:
        assert _should_exclude("output.tmp") is True

    def test_does_not_exclude_normal_py_file(self) -> None:
        assert _should_exclude("main.py") is False

    def test_does_not_exclude_normal_dir(self) -> None:
        assert _should_exclude("src") is False

    def test_nested_pycache_excluded(self) -> None:
        assert _should_exclude("src/__pycache__/module.cpython-311.pyc") is True

    def test_custom_excludes_override(self) -> None:
        assert _should_exclude("secret.txt", excludes={"secret.txt"}) is True
        assert _should_exclude("public.txt", excludes={"secret.txt"}) is False

    def test_glob_wildcard_in_custom_excludes(self) -> None:
        assert _should_exclude("build/out.o", excludes={"*.o"}) is True
        assert _should_exclude("build/out.py", excludes={"*.o"}) is False

    def test_empty_excludes_nothing_excluded(self) -> None:
        assert _should_exclude(".git", excludes=[]) is False


# ---------------------------------------------------------------------------
# _infer_language
# ---------------------------------------------------------------------------


_LANGUAGE_CASES = [
    ("main.py", "python"),
    ("app.js", "javascript"),
    ("app.cjs", "javascript"),
    ("app.mjs", "javascript"),
    ("app.jsx", "javascript"),
    ("app.ts", "typescript"),
    ("app.tsx", "typescript"),
    ("data.json", "json"),
    ("README.md", "markdown"),
    ("plan.yml", "yaml"),
    ("plan.yaml", "yaml"),
    ("setup.toml", "toml"),
    ("config.ini", "ini"),
    ("config.cfg", "ini"),
    ("index.html", "html"),
    ("styles.css", "css"),
    ("script.sh", "shell"),
    ("run.ps1", "powershell"),
    ("main.go", "go"),
    ("lib.rs", "rust"),
    ("App.java", "java"),
    ("Utils.kt", "kotlin"),
    ("View.swift", "swift"),
    ("helper.rb", "ruby"),
    ("app.php", "php"),
    ("query.sql", "sql"),
    ("util.c", "c"),
    ("defs.h", "c"),
    ("engine.cpp", "cpp"),
    ("engine.cc", "cpp"),
    ("engine.cxx", "cpp"),
    ("engine.hpp", "cpp"),
    ("data.xml", "xml"),
]


class TestInferLanguage:
    @pytest.mark.parametrize("filename,expected", _LANGUAGE_CASES)
    def test_known_extension(self, filename: str, expected: str) -> None:
        assert _infer_language(filename) == expected

    def test_unknown_extension_returns_none(self) -> None:
        assert _infer_language("binary.exe") is None

    def test_no_extension_returns_none(self) -> None:
        assert _infer_language("Makefile") is None

    def test_path_object_accepted(self) -> None:
        assert _infer_language(Path("src/main.py")) == "python"

    def test_uppercase_extension_normalised(self) -> None:
        # Extension comparison is case-insensitive (.PY → .py)
        assert _infer_language("MAIN.PY") == "python"


# ---------------------------------------------------------------------------
# _hash_file_head
# ---------------------------------------------------------------------------


class TestHashFileHead:
    def test_returns_64_char_hex_string(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_bytes(b"hello world")
        digest = _hash_file_head(f)
        assert len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest)

    def test_deterministic_for_same_content(self, tmp_path: Path) -> None:
        f = tmp_path / "a.txt"
        f.write_bytes(b"same content")
        assert _hash_file_head(f) == _hash_file_head(f)

    def test_differs_for_different_content(self, tmp_path: Path) -> None:
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_bytes(b"content a")
        f2.write_bytes(b"content b")
        assert _hash_file_head(f1) != _hash_file_head(f2)

    def test_matches_manual_sha256_of_first_bytes(self, tmp_path: Path) -> None:
        data = b"x" * 100
        f = tmp_path / "f.bin"
        f.write_bytes(data)
        expected = hashlib.sha256(data).hexdigest()
        assert _hash_file_head(f, max_bytes=len(data)) == expected


# ---------------------------------------------------------------------------
# _read_first_lines
# ---------------------------------------------------------------------------


class TestReadFirstLines:
    def test_reads_up_to_max_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("\n".join(f"line{i}" for i in range(20)), encoding="utf-8")
        lines = _read_first_lines(f, max_lines=8)
        assert len(lines) == 8
        assert lines[0] == "line0"

    def test_reads_fewer_lines_when_file_is_short(self, tmp_path: Path) -> None:
        f = tmp_path / "short.txt"
        f.write_text("a\nb\nc\n", encoding="utf-8")
        lines = _read_first_lines(f, max_lines=10)
        assert lines == ["a", "b", "c"]

    def test_strips_newlines(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("hello\r\nworld\r\n", encoding="utf-8")
        lines = _read_first_lines(f, max_lines=5)
        for line in lines:
            assert "\r" not in line
            assert "\n" not in line

    def test_nonexistent_file_returns_empty_list(self, tmp_path: Path) -> None:
        lines = _read_first_lines(tmp_path / "missing.txt", max_lines=5)
        assert lines == []

    def test_max_lines_zero_returns_empty_list(self, tmp_path: Path) -> None:
        f = tmp_path / "f.txt"
        f.write_text("content", encoding="utf-8")
        assert _read_first_lines(f, max_lines=0) == []

    def test_default_reads_eight_lines(self, tmp_path: Path) -> None:
        f = tmp_path / "big.txt"
        f.write_text("\n".join(str(i) for i in range(100)), encoding="utf-8")
        # Default is _FIRST_LINES_DEFAULT (8)
        lines = _read_first_lines(f)
        assert len(lines) == 8


# ---------------------------------------------------------------------------
# build_tree_summary
# ---------------------------------------------------------------------------


class TestBuildTreeSummary:
    def _entry(self, path: str) -> FileEntry:
        return FileEntry(path=path, size_bytes=1, mtime_ns=0, sha256="x")

    def test_empty_files_returns_empty_string(self) -> None:
        assert build_tree_summary([]) == ""

    def test_flat_files_listed(self) -> None:
        entries = [self._entry("a.py"), self._entry("b.py")]
        summary = build_tree_summary(entries)
        assert "a.py" in summary
        assert "b.py" in summary

    def test_nested_dirs_shown_with_slash(self) -> None:
        entries = [self._entry("src/main.py"), self._entry("src/utils.py")]
        summary = build_tree_summary(entries)
        assert "src/" in summary

    def test_max_lines_respected(self) -> None:
        entries = [self._entry(f"file_{i}.py") for i in range(50)]
        summary = build_tree_summary(entries, max_lines=10)
        lines = summary.splitlines()
        # The last line should be "..." when limit hit
        assert "..." in lines or len(lines) <= 11

    def test_subdirs_indented(self) -> None:
        entries = [self._entry("a/b/c.py")]
        summary = build_tree_summary(entries)
        assert "a/" in summary
        assert "b/" in summary
        assert "c.py" in summary


# ---------------------------------------------------------------------------
# build_workspace_index
# ---------------------------------------------------------------------------


class TestBuildWorkspaceIndex:
    def test_indexes_files_in_workspace(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "hello.py").write_text("print('hi')", encoding="utf-8")
        (root / "readme.md").write_text("# Readme", encoding="utf-8")

        idx = build_workspace_index(root, cache_dir=tmp_path / "cache")
        paths = [e.path for e in idx.files]
        assert "hello.py" in paths
        assert "readme.md" in paths

    def test_excludes_default_dirs(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        pycache = root / "__pycache__"
        pycache.mkdir()
        (pycache / "mod.pyc").write_bytes(b"bytecode")
        (root / "app.py").write_text("pass", encoding="utf-8")

        idx = build_workspace_index(root, cache_dir=tmp_path / "cache")
        paths = [e.path for e in idx.files]
        assert not any("__pycache__" in p for p in paths)
        assert "app.py" in paths

    def test_custom_excludes_respected(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "keep.py").write_text("x", encoding="utf-8")
        (root / "skip.log").write_text("log", encoding="utf-8")

        idx = build_workspace_index(root, excludes=["*.log"], cache_dir=tmp_path / "cache")
        paths = [e.path for e in idx.files]
        assert "keep.py" in paths
        assert not any(p.endswith(".log") for p in paths)

    def test_raises_for_nonexistent_root(self, tmp_path: Path) -> None:
        with pytest.raises(NotADirectoryError):
            build_workspace_index(tmp_path / "no_such_dir")

    def test_raises_for_file_as_root(self, tmp_path: Path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("x", encoding="utf-8")
        with pytest.raises(NotADirectoryError):
            build_workspace_index(f)

    def test_index_has_correct_file_count(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        for i in range(5):
            (root / f"f{i}.py").write_text("pass", encoding="utf-8")

        idx = build_workspace_index(root, cache_dir=tmp_path / "cache")
        assert idx.file_count == len(idx.files) == 5

    def test_reuses_cached_index_when_unchanged(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "a.py").write_text("x", encoding="utf-8")
        cache_dir = tmp_path / "cache"

        idx1 = build_workspace_index(root, cache_dir=cache_dir)
        idx2 = build_workspace_index(root, cache_dir=cache_dir)
        assert idx1.snapshot_id == idx2.snapshot_id
        assert idx1.content_id == idx2.content_id

    def test_force_rebuild_bypasses_cache(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "a.py").write_text("x", encoding="utf-8")
        cache_dir = tmp_path / "cache"

        idx1 = build_workspace_index(root, cache_dir=cache_dir)
        # Force rebuild — same files, should produce same hashes but was rebuilt
        idx2 = build_workspace_index(root, cache_dir=cache_dir, force_rebuild=True)
        assert idx1.content_id == idx2.content_id

    def test_max_files_cap_triggers_truncation(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        for i in range(10):
            (root / f"f{i}.py").write_text("x", encoding="utf-8")

        idx = build_workspace_index(root, max_files=5, cache_dir=tmp_path / "cache")
        assert idx.file_count <= 5
        assert any("file cap" in err for err in idx.errors)

    def test_entries_include_sha256(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "a.py").write_text("hello", encoding="utf-8")

        idx = build_workspace_index(root, cache_dir=tmp_path / "cache")
        assert len(idx.files[0].sha256) == 64

    def test_language_inferred_for_entries(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "app.py").write_text("x", encoding="utf-8")

        idx = build_workspace_index(root, cache_dir=tmp_path / "cache")
        py_entry = next(e for e in idx.files if e.path == "app.py")
        assert py_entry.language == "python"


# ---------------------------------------------------------------------------
# quick_root_hash
# ---------------------------------------------------------------------------


class TestQuickRootHash:
    def test_returns_64_char_hex_string(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "a.py").write_text("x", encoding="utf-8")
        h = quick_root_hash(root)
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_deterministic_for_same_content(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "a.py").write_text("x", encoding="utf-8")
        assert quick_root_hash(root) == quick_root_hash(root)

    def test_changes_when_new_file_added(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        (root / "a.py").write_text("x", encoding="utf-8")
        h1 = quick_root_hash(root)
        (root / "b.py").write_text("y", encoding="utf-8")
        h2 = quick_root_hash(root)
        assert h1 != h2

    def test_empty_workspace_returns_hash(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        h = quick_root_hash(root)
        assert len(h) == 64


# ---------------------------------------------------------------------------
# load_cached_index / save_index roundtrip
# ---------------------------------------------------------------------------


class TestLoadCachedIndexAndSaveIndex:
    def _make_index(self, tmp_path: Path, root: Path) -> WorkspaceIndex:
        return WorkspaceIndex(
            schema_version=INDEX_SCHEMA_VERSION,
            workspace_root=str(root),
            snapshot_id="snap_abc123",
            content_id="cont_def456",
            file_count=1,
            total_size_bytes=10,
            files=[FileEntry(path="a.py", size_bytes=10, mtime_ns=0, sha256="ff")],
        )

    def test_save_then_load_roundtrip(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        cache_dir = tmp_path / "cache"
        idx = self._make_index(tmp_path, root)

        save_index(idx, cache_dir=cache_dir)
        loaded = load_cached_index(root, snapshot_id="snap_abc123", cache_dir=cache_dir)

        assert loaded is not None
        assert loaded.snapshot_id == "snap_abc123"
        assert loaded.content_id == "cont_def456"
        assert len(loaded.files) == 1

    def test_load_without_snapshot_id_uses_latest(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        cache_dir = tmp_path / "cache"
        idx = self._make_index(tmp_path, root)

        save_index(idx, cache_dir=cache_dir)
        loaded = load_cached_index(root, cache_dir=cache_dir)

        assert loaded is not None
        assert loaded.snapshot_id == "snap_abc123"

    def test_load_returns_none_for_missing_cache(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        cache_dir = tmp_path / "empty_cache"
        loaded = load_cached_index(root, cache_dir=cache_dir)
        assert loaded is None

    def test_load_returns_none_for_unknown_snapshot_id(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        cache_dir = tmp_path / "cache"
        idx = self._make_index(tmp_path, root)
        save_index(idx, cache_dir=cache_dir)

        loaded = load_cached_index(root, snapshot_id="nonexistent_hash", cache_dir=cache_dir)
        assert loaded is None

    def test_load_returns_none_for_wrong_workspace_root(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        other = tmp_path / "other"
        other.mkdir()
        cache_dir = tmp_path / "cache"

        idx = self._make_index(tmp_path, root)
        save_index(idx, cache_dir=cache_dir)

        # Load using a different workspace root — should reject
        loaded = load_cached_index(other, snapshot_id="snap_abc123", cache_dir=cache_dir)
        assert loaded is None

    def test_save_creates_latest_json(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        cache_dir = tmp_path / "cache"
        idx = self._make_index(tmp_path, root)
        save_index(idx, cache_dir=cache_dir)

        latest_path = cache_dir / "latest.json"
        assert latest_path.exists()
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        assert latest["snapshot_id"] == "snap_abc123"

    def test_default_cache_dir(self, tmp_path: Path) -> None:
        root = tmp_path / "workspace"
        root.mkdir()
        expected = root / ".maestro-cache" / "index"
        assert default_index_cache_dir(root) == expected
