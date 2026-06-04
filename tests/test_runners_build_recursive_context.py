from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.models import (
    PlanSpec,
    TaskSpec,
    WorkspaceBrief,
    WorkspaceExtraction,
)
from maestro_cli.runners import _build_recursive_context
from maestro_cli.workspace_index import WorkspaceIndex


def _make_plan(workspace_root: str | None) -> PlanSpec:
    return PlanSpec(name="recursive-context-plan", workspace_root=workspace_root)


def _make_task() -> TaskSpec:
    return TaskSpec(
        id="implement",
        engine="claude",
        prompt="Implement the feature described in the brief.",
        context_mode="recursive",
    )


def _make_index(snapshot_id: str, workspace_root: str) -> WorkspaceIndex:
    return WorkspaceIndex(
        workspace_root=workspace_root,
        snapshot_id=snapshot_id,
        content_id=snapshot_id,
        file_count=1,
        total_size_bytes=10,
    )


def test_dry_run_skips_pipeline(tmp_path: Path) -> None:
    """Dry-run short-circuits before any index/LLM work (sanity baseline)."""
    plan = _make_plan(str(tmp_path))
    task = _make_task()

    result = _build_recursive_context(plan, task, tmp_path, dry_run=True)

    assert result.stages == []
    assert result.workspace_brief == "[dry-run: workspace brief skipped]"
    assert result.index is None


def test_cached_index_hash_mismatch_rebuilds_and_saves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drives lines 6579-6583: cached index exists but the workspace hash
    changed, so the index must be rebuilt via build_workspace_index and
    persisted via save_index."""
    plan = _make_plan(str(tmp_path))
    task = _make_task()

    stale_index = _make_index("OLD-HASH", str(tmp_path))
    rebuilt_index = _make_index("NEW-HASH", str(tmp_path))

    saved: list[WorkspaceIndex] = []
    build_calls: list[tuple[str, object]] = []

    def fake_load_cached_index(root: str) -> WorkspaceIndex:
        return stale_index

    def fake_quick_root_hash(root: str, excludes: object = None) -> str:
        # Different from stale_index.snapshot_id -> forces the rebuild branch.
        return "NEW-HASH"

    def fake_build_workspace_index(root: str, excludes: object = None) -> WorkspaceIndex:
        build_calls.append((root, excludes))
        return rebuilt_index

    def fake_save_index(index: WorkspaceIndex) -> None:
        saved.append(index)

    def fake_extraction(
        index: WorkspaceIndex, prompt: str, workdir: Path, model: str | None = None
    ) -> WorkspaceExtraction:
        return WorkspaceExtraction(relevant_files=["a.py"], reasoning="ok")

    def fake_brief(
        index: WorkspaceIndex,
        extraction: WorkspaceExtraction,
        prompt: str,
        workdir: Path,
        model: str | None = None,
    ) -> WorkspaceBrief:
        return WorkspaceBrief(brief_text="focused brief", files_referenced=["a.py"])

    monkeypatch.setattr("maestro_cli.runners.load_cached_index", fake_load_cached_index)
    monkeypatch.setattr("maestro_cli.runners.quick_root_hash", fake_quick_root_hash)
    monkeypatch.setattr(
        "maestro_cli.runners.build_workspace_index", fake_build_workspace_index
    )
    monkeypatch.setattr("maestro_cli.runners.save_index", fake_save_index)
    monkeypatch.setattr(
        "maestro_cli.runners._run_workspace_extraction", fake_extraction
    )
    monkeypatch.setattr("maestro_cli.runners._run_workspace_brief", fake_brief)

    result = _build_recursive_context(plan, task, tmp_path, dry_run=False)

    # Rebuild branch executed (6579-6583).
    assert build_calls, "build_workspace_index should have been called on hash mismatch"
    assert saved == [rebuilt_index]
    assert result.reused_index is False
    assert result.index is rebuilt_index
    assert result.stages == ["index", "extract", "brief"]
    assert result.workspace_brief == "focused brief"


def test_extraction_exception_is_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drives lines 6606-6607: _run_workspace_extraction raises, so the
    extraction is replaced with a WorkspaceExtraction carrying the error."""
    plan = _make_plan(str(tmp_path))
    task = _make_task()

    fresh_index = _make_index("HASH-A", str(tmp_path))

    monkeypatch.setattr(
        "maestro_cli.runners.load_cached_index", lambda root: None
    )
    monkeypatch.setattr(
        "maestro_cli.runners.build_workspace_index",
        lambda root, excludes=None: fresh_index,
    )
    monkeypatch.setattr("maestro_cli.runners.save_index", lambda index: None)

    def boom_extraction(
        index: WorkspaceIndex, prompt: str, workdir: Path, model: str | None = None
    ) -> WorkspaceExtraction:
        raise RuntimeError("extraction model unavailable")

    captured_extraction: dict[str, object] = {}

    def fake_brief(
        index: WorkspaceIndex,
        extraction: WorkspaceExtraction,
        prompt: str,
        workdir: Path,
        model: str | None = None,
    ) -> WorkspaceBrief:
        captured_extraction["value"] = extraction
        return WorkspaceBrief(brief_text="brief after extract error")

    monkeypatch.setattr(
        "maestro_cli.runners._run_workspace_extraction", boom_extraction
    )
    monkeypatch.setattr("maestro_cli.runners._run_workspace_brief", fake_brief)

    result = _build_recursive_context(plan, task, tmp_path, dry_run=False)

    assert result.extraction is not None
    assert "[extract error: extraction model unavailable]" == result.extraction.reasoning
    # The error-bearing extraction is forwarded to the brief pass.
    assert captured_extraction["value"] is result.extraction
    assert result.stages == ["index", "extract", "brief"]
    assert result.workspace_brief == "brief after extract error"


def test_brief_exception_is_captured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drives lines 6613-6614: _run_workspace_brief raises, so the brief is
    replaced with a WorkspaceBrief carrying the error message."""
    plan = _make_plan(str(tmp_path))
    task = _make_task()

    fresh_index = _make_index("HASH-B", str(tmp_path))

    monkeypatch.setattr(
        "maestro_cli.runners.load_cached_index", lambda root: None
    )
    monkeypatch.setattr(
        "maestro_cli.runners.build_workspace_index",
        lambda root, excludes=None: fresh_index,
    )
    monkeypatch.setattr("maestro_cli.runners.save_index", lambda index: None)
    monkeypatch.setattr(
        "maestro_cli.runners._run_workspace_extraction",
        lambda index, prompt, workdir, model=None: WorkspaceExtraction(
            relevant_files=["b.py"], reasoning="ok"
        ),
    )

    def boom_brief(
        index: WorkspaceIndex,
        extraction: WorkspaceExtraction,
        prompt: str,
        workdir: Path,
        model: str | None = None,
    ) -> WorkspaceBrief:
        raise ValueError("brief synthesis failed")

    monkeypatch.setattr("maestro_cli.runners._run_workspace_brief", boom_brief)

    result = _build_recursive_context(plan, task, tmp_path, dry_run=False)

    assert result.brief is not None
    assert result.brief.brief_text == "[brief error: brief synthesis failed]"
    # workspace_brief mirrors the failed brief text.
    assert result.workspace_brief == "[brief error: brief synthesis failed]"
    assert result.stages == ["index", "extract", "brief"]


def test_workspace_root_falls_back_to_workdir_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the plan has no workspace_root, the workdir string is used. Also
    exercises the task-level workspace_index_exclude path feeding excludes."""
    plan = _make_plan(None)
    task = _make_task()
    task.workspace_index_exclude = ["*.tmp"]

    fresh_index = _make_index("HASH-C", str(tmp_path))
    seen_roots: list[str] = []
    seen_excludes: list[object] = []

    def fake_build_workspace_index(root: str, excludes: object = None) -> WorkspaceIndex:
        seen_roots.append(root)
        seen_excludes.append(excludes)
        return fresh_index

    monkeypatch.setattr(
        "maestro_cli.runners.load_cached_index", lambda root: None
    )
    monkeypatch.setattr(
        "maestro_cli.runners.build_workspace_index", fake_build_workspace_index
    )
    monkeypatch.setattr("maestro_cli.runners.save_index", lambda index: None)
    monkeypatch.setattr(
        "maestro_cli.runners._run_workspace_extraction",
        lambda index, prompt, workdir, model=None: WorkspaceExtraction(),
    )
    monkeypatch.setattr(
        "maestro_cli.runners._run_workspace_brief",
        lambda index, extraction, prompt, workdir, model=None: WorkspaceBrief(
            brief_text="done"
        ),
    )

    result = _build_recursive_context(plan, task, tmp_path, dry_run=False)

    assert seen_roots == [str(tmp_path)]
    assert seen_excludes == [["*.tmp"]]
    assert result.workspace_brief == "done"
