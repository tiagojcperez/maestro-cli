"""Coverage-focused tests for maestro_cli.workspace_index.

Targets specific uncovered branches: error paths in metadata collection,
hash failures, tree fallbacks, cache version/empty-snapshot rejection, and
the ignore-set/exclude merge branches of the public build helpers.

External boundaries (filesystem stat/hash) are mocked only where a real
failure cannot be reliably provoked on a normal CI filesystem.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from maestro_cli import workspace_index as wi
from maestro_cli.workspace_index import (
    INDEX_SCHEMA_VERSION,
    FileEntry,
    WorkspaceIndex,
    _build_tree,
    _collect_workspace_metadata,
    _DEFAULT_IGNORE_DIRS,
    _DEFAULT_IGNORE_FILES,
    _should_exclude,
    build_tree_summary,
    build_workspace_index,
    get_workspace_index,
    load_cached_workspace_index,
    quick_root_hash,
    save_index,
)


def _entry(path: str) -> FileEntry:
    return FileEntry(path=path, size_bytes=1, mtime_ns=0, sha256="x")


# ---------------------------------------------------------------------------
# WorkspaceIndex.from_dict — tree fallback
# ---------------------------------------------------------------------------


class TestFromDictTreeFallback:
    def test_non_dict_tree_rebuilds_from_files(self) -> None:
        raw = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "workspace_root": "/tmp/proj",
            "snapshot_id": "s",
            "content_id": "c",
            "files": [{"path": "src/a.py", "size_bytes": 1, "mtime_ns": 0, "sha256": "z"}],
            # tree is intentionally not a dict -> triggers _build_tree fallback
            "tree": "not-a-dict",
        }
        idx = WorkspaceIndex.from_dict(raw)
        assert isinstance(idx.tree, dict)
        assert "src" in idx.tree.get("dirs", {})

    def test_non_list_errors_yields_empty_list(self) -> None:
        raw = {
            "schema_version": INDEX_SCHEMA_VERSION,
            "workspace_root": "/tmp",
            "snapshot_id": "s",
            "content_id": "c",
            "files": [],
            "errors": "oops-not-a-list",
        }
        idx = WorkspaceIndex.from_dict(raw)
        assert idx.errors == []


# ---------------------------------------------------------------------------
# _should_exclude — empty token continue (249) + path-part match (257)
# ---------------------------------------------------------------------------


class TestShouldExcludeBranches:
    def test_empty_pattern_token_is_skipped(self) -> None:
        # Whitespace-only / empty pattern tokens are stripped and skipped.
        # With only blank patterns nothing matches.
        assert _should_exclude("main.py", excludes=["", "   "]) is False

    def test_blank_then_real_pattern(self) -> None:
        # Blank token continues, then a real glob still matches.
        assert _should_exclude("out.log", excludes=["", "*.log"]) is True

    def test_token_matches_path_part(self) -> None:
        # Non-glob token equal to an intermediate directory part -> True.
        assert _should_exclude("vendor/lib/x.py", excludes={"vendor"}) is True

    def test_token_part_no_match(self) -> None:
        assert _should_exclude("src/lib/x.py", excludes={"vendor"}) is False


# ---------------------------------------------------------------------------
# _collect_workspace_metadata — dir/file skip branches (288,290,297,299,305)
# ---------------------------------------------------------------------------


class TestCollectMetadataSkips:
    def test_hidden_dir_skipped_when_include_hidden_false(self, tmp_path: Path) -> None:
        (tmp_path / ".hidden").mkdir()
        (tmp_path / ".hidden" / "secret.py").write_text("x", encoding="utf-8")
        (tmp_path / "visible.py").write_text("y", encoding="utf-8")
        files, _ = _collect_workspace_metadata(
            tmp_path,
            include_hidden=False,
            ignore_dirs=set(),
            ignore_files=set(),
            excludes=set(),
            max_files=100,
        )
        rels = [rel for _, rel, _, _ in files]
        assert "visible.py" in rels
        assert not any(".hidden" in r for r in rels)

    def test_excluded_dir_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "build").mkdir()
        (tmp_path / "build" / "out.py").write_text("x", encoding="utf-8")
        (tmp_path / "keep.py").write_text("y", encoding="utf-8")
        files, _ = _collect_workspace_metadata(
            tmp_path,
            include_hidden=True,
            ignore_dirs=set(),
            ignore_files=set(),
            excludes={"build"},
            max_files=100,
        )
        rels = [rel for _, rel, _, _ in files]
        assert "keep.py" in rels
        assert not any("build" in r for r in rels)

    def test_ignored_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "skip.me").write_text("x", encoding="utf-8")
        (tmp_path / "keep.py").write_text("y", encoding="utf-8")
        files, _ = _collect_workspace_metadata(
            tmp_path,
            include_hidden=True,
            ignore_dirs=set(),
            ignore_files={"skip.me"},
            excludes=set(),
            max_files=100,
        )
        rels = [rel for _, rel, _, _ in files]
        assert "keep.py" in rels
        assert "skip.me" not in rels

    def test_hidden_file_skipped_when_include_hidden_false(self, tmp_path: Path) -> None:
        (tmp_path / ".env").write_text("x", encoding="utf-8")
        (tmp_path / "keep.py").write_text("y", encoding="utf-8")
        files, _ = _collect_workspace_metadata(
            tmp_path,
            include_hidden=False,
            ignore_dirs=set(),
            ignore_files=set(),
            excludes=set(),
            max_files=100,
        )
        rels = [rel for _, rel, _, _ in files]
        assert "keep.py" in rels
        assert ".env" not in rels

    def test_excluded_file_skipped(self, tmp_path: Path) -> None:
        (tmp_path / "a.log").write_text("x", encoding="utf-8")
        (tmp_path / "keep.py").write_text("y", encoding="utf-8")
        files, _ = _collect_workspace_metadata(
            tmp_path,
            include_hidden=True,
            ignore_dirs=set(),
            ignore_files=set(),
            excludes={"*.log"},
            max_files=100,
        )
        rels = [rel for _, rel, _, _ in files]
        assert "keep.py" in rels
        assert "a.log" not in rels

    def test_non_regular_file_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        # Force every path to look like neither a symlink nor a regular file
        # so the is_file()/is_symlink() guard continues.
        (tmp_path / "ghost.py").write_text("x", encoding="utf-8")
        monkeypatch.setattr(Path, "is_file", lambda self: False)
        monkeypatch.setattr(Path, "is_symlink", lambda self: False)
        files, errors = _collect_workspace_metadata(
            tmp_path,
            include_hidden=True,
            ignore_dirs=set(),
            ignore_files=set(),
            excludes=set(),
            max_files=100,
        )
        assert files == []
        assert errors == []

    def test_stat_oserror_recorded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "boom.py").write_text("x", encoding="utf-8")
        real_stat = Path.stat

        # is_symlink/is_file internally call stat on some Python versions, so
        # stub them to fixed booleans and only fail the explicit .stat() call.
        monkeypatch.setattr(Path, "is_symlink", lambda self: False)
        monkeypatch.setattr(Path, "is_file", lambda self: True)

        def fake_stat(self: Path, *args: object, **kwargs: object) -> object:
            if self.name == "boom.py":
                raise OSError("stat blew up")
            return real_stat(self, *args, **kwargs)

        monkeypatch.setattr(Path, "stat", fake_stat)
        files, errors = _collect_workspace_metadata(
            tmp_path,
            include_hidden=True,
            ignore_dirs=set(),
            ignore_files=set(),
            excludes=set(),
            max_files=100,
        )
        assert files == []
        assert any("boom.py" in e and "stat blew up" in e for e in errors)

    def test_relative_to_value_error_skips_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "weird.py").write_text("x", encoding="utf-8")
        real_relative_to = Path.relative_to

        def fake_relative_to(self: Path, *other: object, **kwargs: object) -> object:
            if self.name == "weird.py":
                raise ValueError("outside root")
            return real_relative_to(self, *other, **kwargs)

        monkeypatch.setattr(Path, "relative_to", fake_relative_to)
        files, errors = _collect_workspace_metadata(
            tmp_path,
            include_hidden=True,
            ignore_dirs=set(),
            ignore_files=set(),
            excludes=set(),
            max_files=100,
        )
        rels = [rel for _, rel, _, _ in files]
        assert "weird.py" not in rels


# ---------------------------------------------------------------------------
# _build_tree — empty-parts continue (368)
# ---------------------------------------------------------------------------


class TestBuildTreeEmptyParts:
    def test_entry_with_only_slashes_skipped(self) -> None:
        # path that splits to no parts should be skipped, not crash.
        tree = _build_tree([_entry("///"), _entry("real.py")])
        assert "real.py" in tree.get("files", [])

    def test_empty_path_skipped(self) -> None:
        tree = _build_tree([_entry(""), _entry("kept.py")])
        assert tree.get("files") == ["kept.py"]


# ---------------------------------------------------------------------------
# build_tree_summary — max_lines guards (410, 417)
# ---------------------------------------------------------------------------


class TestBuildTreeSummaryMaxLines:
    def test_max_lines_zero_returns_early(self) -> None:
        # _walk returns immediately on the first guard (len(lines) >= max_lines).
        entries = [_entry("a.py"), _entry("b.py")]
        summary = build_tree_summary(entries, max_lines=0)
        # With max_lines=0 nothing gets appended except the trailing "..." sentinel.
        assert summary == "..."

    def test_max_lines_hit_inside_dir_loop(self) -> None:
        # Many top-level dirs so the limit is reached while emitting directory
        # header lines (the guard after appending a dir name).
        entries = [_entry(f"dir_{i}/file.py") for i in range(10)]
        summary = build_tree_summary(entries, max_lines=3)
        lines = summary.splitlines()
        assert "..." in lines
        # Only a handful of lines plus the sentinel.
        assert len(lines) <= 4


# ---------------------------------------------------------------------------
# load_cached_workspace_index — empty snapshot (466) + schema mismatch (474)
# ---------------------------------------------------------------------------


class TestLoadCachedRejections:
    def test_empty_latest_snapshot_returns_none(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        cache_dir = tmp_path / "cache"
        cache_dir.mkdir(parents=True)
        # latest.json with blank snapshot_id -> selected_snapshot empty -> None
        (cache_dir / "latest.json").write_text('{"snapshot_id": "   "}', encoding="utf-8")
        assert load_cached_workspace_index(root, cache_dir=cache_dir) is None

    def test_schema_version_mismatch_returns_none(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        cache_dir = tmp_path / "cache"
        idx = WorkspaceIndex(
            schema_version=INDEX_SCHEMA_VERSION,
            workspace_root=str(root),
            snapshot_id="snap_x",
            content_id="c",
        )
        save_index(idx, cache_dir=cache_dir)
        # Corrupt the stored schema_version so the loaded value mismatches.
        stored = cache_dir / "snap_x.json"
        text = stored.read_text(encoding="utf-8")
        bumped = INDEX_SCHEMA_VERSION + 99
        text = text.replace(
            f'"schema_version": {INDEX_SCHEMA_VERSION}',
            f'"schema_version": {bumped}',
        )
        stored.write_text(text, encoding="utf-8")
        assert (
            load_cached_workspace_index(root, snapshot_id="snap_x", cache_dir=cache_dir)
            is None
        )


# ---------------------------------------------------------------------------
# build_workspace_index — ignore merges (513,517), cached+errors dedup (537),
# file_sha256 OSError (547-549)
# ---------------------------------------------------------------------------


class TestBuildWorkspaceIndexBranches:
    def test_custom_ignore_dirs_and_files_merged(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        (root / "extradir").mkdir()
        (root / "extradir" / "inside.py").write_text("x", encoding="utf-8")
        (root / "drop.txt").write_text("y", encoding="utf-8")
        (root / "keep.py").write_text("z", encoding="utf-8")

        idx = build_workspace_index(
            root,
            cache_dir=tmp_path / "cache",
            ignore_dirs={"extradir"},
            ignore_files={"drop.txt"},
        )
        paths = [e.path for e in idx.files]
        assert "keep.py" in paths
        assert "drop.txt" not in paths
        assert not any("extradir" in p for p in paths)

    def test_file_sha256_oserror_recorded_and_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        (root / "good.py").write_text("ok", encoding="utf-8")
        (root / "bad.py").write_text("nope", encoding="utf-8")

        real_sha = wi.file_sha256

        def fake_sha(path: Path, *args: object, **kwargs: object) -> str:
            if Path(path).name == "bad.py":
                raise OSError("hash failed")
            return real_sha(path, *args, **kwargs)

        monkeypatch.setattr(wi, "file_sha256", fake_sha)
        idx = build_workspace_index(root, cache_dir=tmp_path / "cache", force_rebuild=True)
        paths = [e.path for e in idx.files]
        assert "good.py" in paths
        assert "bad.py" not in paths
        assert any("bad.py" in e and "hash failed" in e for e in idx.errors)

    def test_cached_path_merges_collection_errors(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        (root / "a.py").write_text("x", encoding="utf-8")
        cache_dir = tmp_path / "cache"

        # First build populates the cache (no errors).
        build_workspace_index(root, cache_dir=cache_dir)

        # Second build: same snapshot (cache hit) but force a collection-time
        # error so the cached-branch error-merge path runs.
        sentinel = "synthetic collection error"

        real_collect = wi._collect_workspace_metadata

        def fake_collect(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
            metadata, errors = real_collect(*args, **kwargs)
            return metadata, errors + [sentinel]

        monkeypatch.setattr(wi, "_collect_workspace_metadata", fake_collect)
        idx = build_workspace_index(root, cache_dir=cache_dir)
        assert sentinel in idx.errors
        # Deduplicated even if merged twice.
        assert idx.errors.count(sentinel) == 1


# ---------------------------------------------------------------------------
# quick_root_hash — ignore/exclude merges (593,596,600)
# ---------------------------------------------------------------------------


class TestQuickRootHashMerges:
    def test_ignore_dirs_files_excludes_applied(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        (root / "skipdir").mkdir()
        (root / "skipdir" / "x.py").write_text("a", encoding="utf-8")
        (root / "drop.txt").write_text("b", encoding="utf-8")
        (root / "noise.log").write_text("c", encoding="utf-8")
        (root / "keep.py").write_text("d", encoding="utf-8")

        h_filtered = quick_root_hash(
            root,
            ignore_dirs={"skipdir"},
            ignore_files={"drop.txt"},
            excludes=["*.log"],
        )
        # A hash over only keep.py should differ from one over everything.
        h_all = quick_root_hash(root)
        assert h_filtered != h_all
        assert len(h_filtered) == 64

    def test_blank_excludes_are_dropped(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        (root / "keep.py").write_text("x", encoding="utf-8")
        # Blank/whitespace excludes are filtered out in the generator expression.
        h = quick_root_hash(root, excludes=["", "  "])
        assert len(h) == 64


# ---------------------------------------------------------------------------
# get_workspace_index — delegates to build_workspace_index (636)
# ---------------------------------------------------------------------------


class TestGetWorkspaceIndex:
    def test_returns_index_for_workspace(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        (root / "a.py").write_text("x", encoding="utf-8")
        idx = get_workspace_index(root, cache_dir=tmp_path / "cache")
        assert isinstance(idx, WorkspaceIndex)
        assert any(e.path == "a.py" for e in idx.files)

    def test_uses_cache_on_second_call(self, tmp_path: Path) -> None:
        root = tmp_path / "ws"
        root.mkdir()
        (root / "a.py").write_text("x", encoding="utf-8")
        cache_dir = tmp_path / "cache"
        idx1 = get_workspace_index(root, cache_dir=cache_dir)
        idx2 = get_workspace_index(root, cache_dir=cache_dir)
        assert idx1.snapshot_id == idx2.snapshot_id


# Sanity: default ignore sets are non-empty so merge branches stay meaningful.
def test_default_ignore_sets_present() -> None:
    assert ".git" in _DEFAULT_IGNORE_DIRS
    assert ".DS_Store" in _DEFAULT_IGNORE_FILES
    assert os.name in ("nt", "posix")
