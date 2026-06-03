from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from maestro_cli.cleanup import cleanup_runs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_dirs(
    root: Path,
    names: list[str],
    *,
    age_offset_sec: float = 0.0,
) -> list[Path]:
    """Create run directories under *root*.

    If *age_offset_sec* > 0 each successive directory is backdated by that many
    additional seconds (the first directory is the newest).
    """
    dirs: list[Path] = []
    now = time.time()
    for i, name in enumerate(names):
        d = root / name
        d.mkdir(parents=True, exist_ok=True)
        # Place a dummy file so the dir is non-empty
        (d / "run_manifest.json").write_text("{}", encoding="utf-8")
        mtime = now - i * age_offset_sec
        os.utime(d, (mtime, mtime))
        dirs.append(d)
    return dirs


def _backdate_dir(d: Path, days: float) -> None:
    """Set modification time of *d* to *days* days ago."""
    past = time.time() - days * 86400
    os.utime(d, (past, past))


# ---------------------------------------------------------------------------
# TestCleanupRunsBasic
# ---------------------------------------------------------------------------

class TestCleanupRunsBasic:
    def test_nonexistent_root_returns_empty(self, tmp_path: Path) -> None:
        result = cleanup_runs(tmp_path / "does-not-exist")
        assert result == []

    def test_empty_root_returns_empty(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        root.mkdir()
        result = cleanup_runs(root)
        assert result == []

    def test_fewer_dirs_than_keep(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        _make_run_dirs(root, ["run1", "run2"], age_offset_sec=10)
        result = cleanup_runs(root, keep=5)
        assert result == []
        # Nothing deleted
        assert (root / "run1").exists()
        assert (root / "run2").exists()

    def test_exactly_keep_dirs(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        _make_run_dirs(root, ["r1", "r2", "r3"], age_offset_sec=10)
        result = cleanup_runs(root, keep=3)
        assert result == []


# ---------------------------------------------------------------------------
# TestCleanupRunsDryRun
# ---------------------------------------------------------------------------

class TestCleanupRunsDryRun:
    def test_dry_run_returns_dirs_but_keeps_them(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        _make_run_dirs(root, ["r1", "r2", "r3", "r4", "r5"], age_offset_sec=10)
        result = cleanup_runs(root, keep=2, dry_run=True)
        assert len(result) == 3
        # All directories still exist
        for name in ["r1", "r2", "r3", "r4", "r5"]:
            assert (root / name).exists()

    def test_dry_run_false_actually_deletes(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        dirs = _make_run_dirs(root, ["r1", "r2", "r3", "r4", "r5"], age_offset_sec=10)
        result = cleanup_runs(root, keep=2, dry_run=False)
        assert len(result) == 3
        # The 2 newest survive, the 3 oldest are gone
        remaining = [d.name for d in root.iterdir() if d.is_dir()]
        assert len(remaining) == 2


# ---------------------------------------------------------------------------
# TestCleanupRunsKeepParameter
# ---------------------------------------------------------------------------

class TestCleanupRunsKeep:
    def test_keep_3_from_5(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        _make_run_dirs(root, ["a", "b", "c", "d", "e"], age_offset_sec=10)
        result = cleanup_runs(root, keep=3, dry_run=False)
        assert len(result) == 2
        remaining = sorted(d.name for d in root.iterdir() if d.is_dir())
        assert len(remaining) == 3

    def test_keep_1_with_single_dir(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        _make_run_dirs(root, ["only-one"], age_offset_sec=0)
        result = cleanup_runs(root, keep=1)
        assert result == []
        assert (root / "only-one").exists()

    def test_keep_0_deletes_everything(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        _make_run_dirs(root, ["r1", "r2", "r3"], age_offset_sec=10)
        result = cleanup_runs(root, keep=0, dry_run=False)
        assert len(result) == 3
        remaining = list(root.iterdir())
        assert len(remaining) == 0


# ---------------------------------------------------------------------------
# TestCleanupRunsOlderThan
# ---------------------------------------------------------------------------

class TestCleanupRunsOlderThan:
    def test_older_than_filters_by_age(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        dirs = _make_run_dirs(root, ["new", "mid", "old"], age_offset_sec=1)
        # Backdate "old" to 10 days ago, "mid" to 5 days ago
        _backdate_dir(root / "old", days=10)
        _backdate_dir(root / "mid", days=5)
        # keep=0 so only older_than matters
        result = cleanup_runs(root, keep=0, older_than_days=7, dry_run=False)
        assert len(result) == 1
        assert not (root / "old").exists()
        assert (root / "mid").exists()
        assert (root / "new").exists()

    def test_older_than_none_ignores_age(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        _make_run_dirs(root, ["r1", "r2", "r3"], age_offset_sec=10)
        result = cleanup_runs(root, keep=1, older_than_days=None, dry_run=False)
        # 2 deleted (keep=1, no age filter)
        assert len(result) == 2

    def test_all_recent_nothing_deleted(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        _make_run_dirs(root, ["a", "b", "c", "d"], age_offset_sec=1)
        # All dirs are fresh (created just now), older_than=30 days
        result = cleanup_runs(root, keep=0, older_than_days=30)
        assert result == []

    def test_older_than_combined_with_keep(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        _make_run_dirs(root, ["n1", "n2", "o1", "o2", "o3"], age_offset_sec=1)
        # Backdate the last 3 to 20 days ago
        for name in ("o1", "o2", "o3"):
            _backdate_dir(root / name, days=20)
        # keep=2 protects the 2 newest; older_than=10 only allows deleting dirs >10 days old
        result = cleanup_runs(root, keep=2, older_than_days=10, dry_run=False)
        assert len(result) == 3
        remaining = sorted(d.name for d in root.iterdir() if d.is_dir())
        assert len(remaining) == 2


# ---------------------------------------------------------------------------
# TestCleanupRunsSortOrder
# ---------------------------------------------------------------------------

class TestCleanupRunsSortOrder:
    def test_keeps_newest_by_mtime(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        # Create directories with explicit mtime ordering (oldest first in creation)
        names = ["oldest", "middle", "newest"]
        for i, name in enumerate(names):
            d = root / name
            d.mkdir(parents=True)
            (d / "manifest.json").write_text("{}", encoding="utf-8")
            # oldest = 30 days ago, middle = 15 days ago, newest = now
            _backdate_dir(d, days=30 - i * 15)

        result = cleanup_runs(root, keep=1, dry_run=True)
        # Should mark "oldest" and "middle" for deletion, keep "newest"
        deleted_names = {p.name for p in result}
        assert "newest" not in deleted_names
        assert "oldest" in deleted_names
        assert "middle" in deleted_names


# ---------------------------------------------------------------------------
# TestCleanupRunsEdgeCases
# ---------------------------------------------------------------------------

class TestCleanupRunsEdgeCases:
    def test_files_in_root_ignored(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        root.mkdir()
        (root / "stray_file.txt").write_text("oops", encoding="utf-8")
        _make_run_dirs(root, ["r1", "r2", "r3"], age_offset_sec=10)
        result = cleanup_runs(root, keep=1, dry_run=False)
        # Only dirs are considered; the stray file is not touched
        assert len(result) == 2
        assert (root / "stray_file.txt").exists()

    def test_nested_content_deleted(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        d = _make_run_dirs(root, ["deep"], age_offset_sec=0)[0]
        # Add nested structure
        sub = d / "subdir" / "nested"
        sub.mkdir(parents=True)
        (sub / "data.txt").write_text("nested data", encoding="utf-8")
        result = cleanup_runs(root, keep=0, dry_run=False)
        assert len(result) == 1
        assert not d.exists()

    def test_returns_path_objects(self, tmp_path: Path) -> None:
        root = tmp_path / "runs"
        _make_run_dirs(root, ["r1", "r2"], age_offset_sec=10)
        result = cleanup_runs(root, keep=0, dry_run=True)
        for item in result:
            assert isinstance(item, Path)
