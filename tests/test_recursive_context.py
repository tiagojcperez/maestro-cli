"""Tests for the v0.7.0 Recursive Context pipeline.

Covers:
- _run_workspace_extraction (pass 2)
- _run_workspace_brief (pass 3)
- _build_recursive_context (full three-pass pipeline)
- Loader validation: context_mode 'recursive' requires workspace_root (E021)
- workspace_index_exclude field parsing (task + plan defaults)
- Template variable {{ workspace_brief }} in _load_prompt
- RecursiveContext / WorkspaceExtraction / WorkspaceBrief dataclasses
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from maestro_cli.errors import E021, PlanValidationError
from maestro_cli.loader import load_plan
from maestro_cli.models import (
    PlanSpec,
    RecursiveContext,
    TaskSpec,
    WorkspaceBrief,
    WorkspaceExtraction,
)
from maestro_cli.runners import (
    _build_recursive_context,
    _load_prompt,
    _run_workspace_brief,
    _run_workspace_extraction,
)
from maestro_cli.workspace_index import FileEntry, WorkspaceIndex


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_plan(tmp_path: Path, content: str, filename: str = "plan.yaml") -> Path:
    p = tmp_path / filename
    p.write_text(content, encoding="utf-8")
    return p


def _make_index(
    workspace_root: str = "/tmp/workspace",
    files: list[FileEntry] | None = None,
) -> WorkspaceIndex:
    if files is None:
        files = [
            FileEntry(
                path="src/main.py",
                size_bytes=100,
                mtime_ns=0,
                sha256="aa",
                language="python",
                first_lines=["import os", "def main(): pass"],
            ),
            FileEntry(
                path="tests/test_main.py",
                size_bytes=50,
                mtime_ns=0,
                sha256="bb",
                language="python",
                first_lines=["import pytest"],
            ),
        ]
    return WorkspaceIndex(
        workspace_root=workspace_root,
        snapshot_id="snap_test",
        content_id="cont_test",
        file_count=len(files),
        total_size_bytes=sum(f.size_bytes for f in files),
        files=files,
    )


def _make_extraction_response(relevant_files: list[str], reasoning: str = "") -> str:
    return json.dumps({"relevant_files": relevant_files, "reasoning": reasoning})


# ---------------------------------------------------------------------------
# WorkspaceExtraction dataclass
# ---------------------------------------------------------------------------


class TestWorkspaceExtractionDataclass:
    def test_default_fields(self) -> None:
        ext = WorkspaceExtraction()
        assert ext.relevant_files == []
        assert ext.snippets == {}
        assert ext.reasoning == ""
        assert ext.token_estimate == 0

    def test_to_dict_contains_all_keys(self) -> None:
        ext = WorkspaceExtraction(
            relevant_files=["a.py"],
            snippets={"a.py": "import os"},
            reasoning="Important file",
            token_estimate=42,
        )
        d = ext.to_dict()
        assert d["relevant_files"] == ["a.py"]
        assert d["snippets"] == {"a.py": "import os"}
        assert d["reasoning"] == "Important file"
        assert d["token_estimate"] == 42


# ---------------------------------------------------------------------------
# WorkspaceBrief dataclass
# ---------------------------------------------------------------------------


class TestWorkspaceBriefDataclass:
    def test_default_fields(self) -> None:
        brief = WorkspaceBrief()
        assert brief.brief_text == ""
        assert brief.token_estimate == 0
        assert brief.files_referenced == []

    def test_to_dict_contains_all_keys(self) -> None:
        brief = WorkspaceBrief(
            brief_text="This repo does X.",
            token_estimate=100,
            files_referenced=["src/main.py"],
        )
        d = brief.to_dict()
        assert d["brief_text"] == "This repo does X."
        assert d["token_estimate"] == 100
        assert d["files_referenced"] == ["src/main.py"]


# ---------------------------------------------------------------------------
# RecursiveContext dataclass
# ---------------------------------------------------------------------------


class TestRecursiveContextDataclass:
    def test_default_fields(self) -> None:
        rc = RecursiveContext()
        assert rc.stages == []
        assert rc.index is None
        assert rc.extraction is None
        assert rc.brief is None
        assert rc.workspace_brief == ""
        assert rc.duration_sec == 0.0
        assert rc.reused_index is False

    def test_to_dict_with_all_fields_populated(self) -> None:
        idx = _make_index()
        ext = WorkspaceExtraction(relevant_files=["a.py"])
        brief = WorkspaceBrief(brief_text="Context here")
        rc = RecursiveContext(
            stages=["index", "extract", "brief"],
            index=idx,
            extraction=ext,
            brief=brief,
            workspace_brief="Context here",
            duration_sec=1.5,
            reused_index=True,
        )
        d = rc.to_dict()
        assert d["stages"] == ["index", "extract", "brief"]
        assert d["workspace_brief"] == "Context here"
        assert d["duration_sec"] == 1.5
        assert d["reused_index"] is True
        assert d["index"] is not None
        assert d["extraction"] is not None
        assert d["brief"] is not None

    def test_to_dict_with_none_subfields(self) -> None:
        rc = RecursiveContext()
        d = rc.to_dict()
        assert d["index"] is None
        assert d["extraction"] is None
        assert d["brief"] is None


# ---------------------------------------------------------------------------
# _run_workspace_extraction (pass 2)
# ---------------------------------------------------------------------------


class TestRunWorkspaceExtraction:
    def test_success_parses_relevant_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        response = _make_extraction_response(
            ["src/main.py", "tests/test_main.py"], "Both files critical"
        )

        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=response, stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        ext = _run_workspace_extraction(index, "Fix the main function", tmp_path)

        assert "src/main.py" in ext.relevant_files
        assert "tests/test_main.py" in ext.relevant_files
        assert ext.reasoning == "Both files critical"

    def test_success_builds_snippets_from_index(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        response = _make_extraction_response(["src/main.py"])

        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=response, stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        ext = _run_workspace_extraction(index, "task prompt", tmp_path)

        assert "src/main.py" in ext.snippets
        assert "import os" in ext.snippets["src/main.py"]

    def test_subprocess_failure_returns_empty_extraction(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="error")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        ext = _run_workspace_extraction(index, "task prompt", tmp_path)

        assert ext.relevant_files == []

    def test_timeout_returns_error_extraction(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def mock_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 60)

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        ext = _run_workspace_extraction(index, "task prompt", tmp_path)

        assert "timed out" in ext.reasoning

    def test_exception_returns_error_extraction(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def mock_run(cmd, **kwargs):
            raise FileNotFoundError("claude not found")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        ext = _run_workspace_extraction(index, "task prompt", tmp_path)

        assert "error" in ext.reasoning.lower()

    def test_non_json_response_returns_empty_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="plain text no json", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        ext = _run_workspace_extraction(index, "task prompt", tmp_path)

        assert ext.relevant_files == []

    def test_uses_haiku_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            captured.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        _run_workspace_extraction(index, "prompt", tmp_path)

        assert captured
        assert "--model" in captured[0]
        idx = captured[0].index("--model")
        assert captured[0][idx + 1] == "haiku"

    def test_token_estimate_computed_from_snippets(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        response = _make_extraction_response(["src/main.py"])

        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=response, stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        ext = _run_workspace_extraction(index, "task prompt", tmp_path)

        # Snippets exist → token_estimate > 0
        if ext.snippets:
            assert ext.token_estimate > 0


# ---------------------------------------------------------------------------
# _run_workspace_brief (pass 3)
# ---------------------------------------------------------------------------


class TestRunWorkspaceBrief:
    def test_success_returns_brief_text(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="Brief content here.\n", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        extraction = WorkspaceExtraction(relevant_files=["src/main.py"])
        brief = _run_workspace_brief(index, extraction, "Fix bugs", tmp_path)

        assert brief.brief_text == "Brief content here."

    def test_no_relevant_files_returns_placeholder(self, tmp_path: Path) -> None:
        index = _make_index()
        extraction = WorkspaceExtraction(relevant_files=[])
        brief = _run_workspace_brief(index, extraction, "task prompt", tmp_path)

        assert "no relevant files" in brief.brief_text

    def test_subprocess_failure_returns_error_brief(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="err")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        extraction = WorkspaceExtraction(relevant_files=["src/main.py"])
        brief = _run_workspace_brief(index, extraction, "prompt", tmp_path)

        assert "failed" in brief.brief_text

    def test_timeout_returns_error_brief(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def mock_run(cmd, **kwargs):
            raise subprocess.TimeoutExpired(cmd, 90)

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        extraction = WorkspaceExtraction(relevant_files=["src/main.py"])
        brief = _run_workspace_brief(index, extraction, "prompt", tmp_path)

        assert "timed out" in brief.brief_text

    def test_exception_returns_error_brief(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def mock_run(cmd, **kwargs):
            raise FileNotFoundError("claude not found")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        extraction = WorkspaceExtraction(relevant_files=["src/main.py"])
        brief = _run_workspace_brief(index, extraction, "prompt", tmp_path)

        assert "error" in brief.brief_text.lower()

    def test_files_referenced_populated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="A brief.\n", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        extraction = WorkspaceExtraction(relevant_files=["src/main.py", "tests/test_main.py"])
        brief = _run_workspace_brief(index, extraction, "prompt", tmp_path)

        assert "src/main.py" in brief.files_referenced
        assert "tests/test_main.py" in brief.files_referenced

    def test_uses_haiku_model(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        captured: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            captured.append(list(cmd))
            return subprocess.CompletedProcess(cmd, 0, stdout="brief\n", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        extraction = WorkspaceExtraction(relevant_files=["src/main.py"])
        _run_workspace_brief(index, extraction, "prompt", tmp_path)

        assert captured
        assert "--model" in captured[0]
        idx = captured[0].index("--model")
        assert captured[0][idx + 1] == "haiku"

    def test_token_estimate_computed_from_brief_length(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        brief_text = "x" * 400  # 400 chars → ~100 tokens

        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout=brief_text + "\n", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)
        index = _make_index()
        extraction = WorkspaceExtraction(relevant_files=["src/main.py"])
        brief = _run_workspace_brief(index, extraction, "prompt", tmp_path)

        assert brief.token_estimate == len(brief_text.strip()) // 4


# ---------------------------------------------------------------------------
# _build_recursive_context (full pipeline)
# ---------------------------------------------------------------------------


class TestBuildRecursiveContext:
    def _make_plan(self, workspace_root: str | None = None) -> PlanSpec:
        task = TaskSpec(id="t1", engine="claude", prompt="Fix bugs", context_mode="recursive")
        return PlanSpec(
            version=1,
            name="test-plan",
            workspace_root=workspace_root,
            tasks=[task],
        )

    def test_dry_run_skips_llm_calls(self, tmp_path: Path) -> None:
        plan = self._make_plan(str(tmp_path))
        task = plan.tasks[0]
        rc = _build_recursive_context(plan, task, tmp_path, dry_run=True)

        assert "dry-run" in rc.workspace_brief
        assert rc.index is None
        assert rc.stages == []

    def test_dry_run_returns_immediately_without_subprocess(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        called = []

        def mock_run(cmd, **kwargs):
            called.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)

        plan = self._make_plan(str(tmp_path))
        task = plan.tasks[0]
        _build_recursive_context(plan, task, tmp_path, dry_run=True)

        assert called == [], "subprocess should not be called in dry-run mode"

    def test_index_failure_returns_gracefully(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Patch names as imported in runners.py (not in workspace_index module)
        monkeypatch.setattr("maestro_cli.runners.load_cached_index", lambda *a, **kw: None)

        def bad_build(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr("maestro_cli.runners.build_workspace_index", bad_build)

        plan = self._make_plan(str(tmp_path))
        task = plan.tasks[0]
        rc = _build_recursive_context(plan, task, tmp_path, dry_run=False)

        assert "workspace index failed" in rc.workspace_brief
        assert "index" in rc.stages

    def test_full_pipeline_sets_all_stages(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "a.py").write_text("x", encoding="utf-8")

        extraction_response = _make_extraction_response(["a.py"], "relevant")
        brief_response = "A focused brief about a.py."
        call_count = [0]

        def mock_run(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return subprocess.CompletedProcess(cmd, 0, stdout=extraction_response, stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout=brief_response, stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)

        plan = self._make_plan(str(tmp_path))
        task = plan.tasks[0]
        rc = _build_recursive_context(plan, task, tmp_path, dry_run=False)

        assert "index" in rc.stages
        assert "extract" in rc.stages
        assert "brief" in rc.stages

    def test_full_pipeline_workspace_brief_populated(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "a.py").write_text("x", encoding="utf-8")

        extraction_response = _make_extraction_response(["a.py"])
        call_count = [0]

        def mock_run(cmd, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return subprocess.CompletedProcess(cmd, 0, stdout=extraction_response, stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="The focused context.\n", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)

        plan = self._make_plan(str(tmp_path))
        task = plan.tasks[0]
        rc = _build_recursive_context(plan, task, tmp_path, dry_run=False)

        assert rc.workspace_brief == "The focused context."

    def test_duration_is_positive(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "a.py").write_text("x", encoding="utf-8")

        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)

        plan = self._make_plan(str(tmp_path))
        task = plan.tasks[0]
        rc = _build_recursive_context(plan, task, tmp_path, dry_run=False)

        assert rc.duration_sec >= 0.0

    def test_reused_index_flag_set_when_cache_valid(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "a.py").write_text("x", encoding="utf-8")

        # Build a real index to capture its snapshot_id
        from maestro_cli.workspace_index import build_workspace_index as _real_build
        real_index = _real_build(tmp_path, cache_dir=tmp_path / ".maestro-cache" / "index")

        # Patch load_cached_index and quick_root_hash so they agree on snapshot_id
        monkeypatch.setattr(
            "maestro_cli.runners.load_cached_index",
            lambda *a, **kw: real_index,
        )
        monkeypatch.setattr(
            "maestro_cli.runners.quick_root_hash",
            lambda *a, **kw: real_index.snapshot_id,
        )

        def mock_run(cmd, **kwargs):
            return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

        monkeypatch.setattr("maestro_cli.runners.subprocess.run", mock_run)

        plan = self._make_plan(str(tmp_path))
        task = plan.tasks[0]
        rc = _build_recursive_context(plan, task, tmp_path, dry_run=False)

        assert rc.reused_index is True


# ---------------------------------------------------------------------------
# Loader validation: context_mode 'recursive' requires workspace_root (E021)
# ---------------------------------------------------------------------------


class TestRecursiveContextLoaderValidation:
    def test_recursive_without_workspace_root_raises_e021(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: echo a
  - id: t2
    command: echo b
    depends_on: [t1]
    context_from: [t1]
    context_mode: recursive
""",
        )
        with pytest.raises(PlanValidationError, match=rf"\[{E021}\]"):
            load_plan(plan_file)

    def test_recursive_with_valid_workspace_root_passes(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        plan_file = _write_plan(
            tmp_path,
            f"""\
version: 1
name: test-plan
workspace_root: {workspace.as_posix()}
tasks:
  - id: t1
    command: echo a
  - id: t2
    command: echo b
    depends_on: [t1]
    context_from: [t1]
    context_mode: recursive
""",
        )
        plan = load_plan(plan_file)
        t2 = next(t for t in plan.tasks if t.id == "t2")
        assert t2.context_mode == "recursive"

    def test_recursive_with_nonexistent_workspace_root_raises_e021(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
workspace_root: /nonexistent/path/does/not/exist
tasks:
  - id: t1
    command: echo a
  - id: t2
    command: echo b
    depends_on: [t1]
    context_from: [t1]
    context_mode: recursive
""",
        )
        with pytest.raises(PlanValidationError, match=rf"\[{E021}\]"):
            load_plan(plan_file)


# ---------------------------------------------------------------------------
# workspace_index_exclude field parsing
# ---------------------------------------------------------------------------


class TestWorkspaceIndexExcludeField:
    def test_task_workspace_index_exclude_parsed(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        plan_file = _write_plan(
            tmp_path,
            f"""\
version: 1
name: test-plan
workspace_root: {workspace.as_posix()}
tasks:
  - id: t1
    command: echo a
  - id: t2
    command: echo b
    depends_on: [t1]
    context_from: [t1]
    context_mode: recursive
    workspace_index_exclude:
      - "*.log"
      - ".venv"
""",
        )
        plan = load_plan(plan_file)
        t2 = next(t for t in plan.tasks if t.id == "t2")
        assert "*.log" in t2.workspace_index_exclude
        assert ".venv" in t2.workspace_index_exclude

    def test_plan_defaults_workspace_index_exclude_parsed(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        plan_file = _write_plan(
            tmp_path,
            f"""\
version: 1
name: test-plan
workspace_root: {workspace.as_posix()}
defaults:
  workspace_index_exclude:
    - "dist/"
    - "build/"
tasks:
  - id: t1
    command: echo a
""",
        )
        plan = load_plan(plan_file)
        assert "dist/" in plan.defaults.workspace_index_exclude
        assert "build/" in plan.defaults.workspace_index_exclude

    def test_workspace_index_exclude_defaults_to_empty_list(self, tmp_path: Path) -> None:
        plan_file = _write_plan(
            tmp_path,
            """\
version: 1
name: test-plan
tasks:
  - id: t1
    command: echo a
""",
        )
        plan = load_plan(plan_file)
        assert plan.tasks[0].workspace_index_exclude == []
        assert plan.defaults.workspace_index_exclude == []

    def test_workspace_index_exclude_accepts_single_string(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        plan_file = _write_plan(
            tmp_path,
            f"""\
version: 1
name: test-plan
workspace_root: {workspace.as_posix()}
tasks:
  - id: t1
    command: echo a
  - id: t2
    command: echo b
    depends_on: [t1]
    context_from: [t1]
    context_mode: recursive
    workspace_index_exclude: "*.pyc"
""",
        )
        plan = load_plan(plan_file)
        t2 = next(t for t in plan.tasks if t.id == "t2")
        assert "*.pyc" in t2.workspace_index_exclude


# ---------------------------------------------------------------------------
# Template variable {{ workspace_brief }} in _load_prompt
# ---------------------------------------------------------------------------


class TestWorkspaceBriefTemplateVariable:
    def _make_plan_spec(self) -> PlanSpec:
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="Context: {{ workspace_brief }}. Do the task.",
            context_mode="recursive",
        )
        return PlanSpec(version=1, name="test-plan", tasks=[task])

    def test_workspace_brief_injected_into_prompt(self) -> None:
        plan = self._make_plan_spec()
        task = plan.tasks[0]
        prompt = _load_prompt(plan, task, workspace_brief="Here is the workspace context.")
        assert "Here is the workspace context." in prompt

    def test_empty_workspace_brief_leaves_variable_unchanged(self) -> None:
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="{{ workspace_brief }}",
        )
        plan = PlanSpec(version=1, name="test-plan", tasks=[task])
        prompt = _load_prompt(plan, task, workspace_brief="")
        # When workspace_brief is empty, the template var is left as-is
        assert "{{ workspace_brief }}" in prompt

    def test_workspace_brief_coexists_with_other_template_vars(self) -> None:
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="Plan: {{ plan_name }}. Brief: {{ workspace_brief }}.",
        )
        plan = PlanSpec(version=1, name="my-plan", tasks=[task])
        prompt = _load_prompt(plan, task, workspace_brief="Brief text.")
        assert "my-plan" in prompt
        assert "Brief text." in prompt

    def test_workspace_brief_not_injected_when_not_in_prompt(self) -> None:
        task = TaskSpec(
            id="t1",
            engine="claude",
            prompt="Just do the thing.",
        )
        plan = PlanSpec(version=1, name="test-plan", tasks=[task])
        prompt = _load_prompt(plan, task, workspace_brief="Some brief content.")
        # Brief is not in template so it won't appear in output
        assert prompt == "Just do the thing."
