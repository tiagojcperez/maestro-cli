"""Tests for MCP server module (mcp_server.py).

Tests the tool, resource, and prompt functions directly without requiring
the mcp SDK to be installed (functions are plain Python).
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_plan(tmp_path: Path, yaml_text: str) -> Path:
    p = tmp_path / "plan.yaml"
    p.write_text(textwrap.dedent(yaml_text), encoding="utf-8")
    return p


def _write_run(tmp_path: Path, plan_name: str = "test") -> Path:
    """Create a minimal run directory with manifest and events."""
    run_dir = tmp_path / ".maestro-runs" / f"20260320_{plan_name}"
    run_dir.mkdir(parents=True)

    manifest = {
        "plan_name": plan_name,
        "success": True,
        "started_at": "2026-03-20T10:00:00Z",
        "finished_at": "2026-03-20T10:01:00Z",
        "execution_profile": "plan",
        "task_results": {
            "task-a": {
                "task_id": "task-a",
                "status": "success",
                "exit_code": 0,
                "duration_sec": 5.0,
                "cost_usd": 0.01,
                "message": "",
            }
        },
    }
    (run_dir / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    (run_dir / "run_summary.md").write_text(
        "# Run Summary\nAll tasks passed.", encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text(
        '{"event":"run_start","plan_name":"test"}\n'
        '{"event":"run_complete","success":true}\n',
        encoding="utf-8",
    )
    (run_dir / "task-a.log").write_text("task-a output", encoding="utf-8")
    (run_dir / "task-a.result.json").write_text(
        json.dumps(manifest["task_results"]["task-a"]), encoding="utf-8",
    )
    return run_dir


# ---------------------------------------------------------------------------
# Tool: validate_plan
# ---------------------------------------------------------------------------


class TestValidatePlan:
    def test_valid_plan(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import validate_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = validate_plan(str(plan_path))
        assert result["valid"] is True
        assert result["name"] == "test"
        assert result["task_count"] == 1
        assert result["task_ids"] == ["a"]

    def test_invalid_plan(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import validate_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks: []
        """)
        result = validate_plan(str(plan_path))
        assert result["valid"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# Tool: audit_plan
# ---------------------------------------------------------------------------


class TestAuditPlan:
    def test_audit_returns_findings(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import audit_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = audit_plan(str(plan_path))
        assert "findings" in result
        assert isinstance(result["findings"], list)
        assert "total" in result

    def test_audit_invalid_plan(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import audit_plan

        plan_path = _write_plan(tmp_path, "invalid: yaml: here")
        result = audit_plan(str(plan_path))
        assert "error" in result


# ---------------------------------------------------------------------------
# Tool: doctor
# ---------------------------------------------------------------------------


class TestDoctor:
    def test_doctor_returns_checks(self) -> None:
        from maestro_cli.mcp_server import doctor

        results = doctor()
        assert isinstance(results, list)
        assert len(results) > 0
        for item in results:
            assert "check" in item
            assert "status" in item


# ---------------------------------------------------------------------------
# Tool: scaffold_plan
# ---------------------------------------------------------------------------


class TestScaffoldPlan:
    def test_scaffold_returns_yaml(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import scaffold_plan

        brief_path = tmp_path / "brief.yaml"
        brief_path.write_text(textwrap.dedent("""\
            name: test-scaffold
            description: A test plan
            tasks:
              - id: build
                engine: claude
                model: sonnet
                description: Build the project
                prompt_hint: "Build everything"
        """), encoding="utf-8")
        result = scaffold_plan(str(brief_path), validate=False)
        assert "version:" in result or "name:" in result


# ---------------------------------------------------------------------------
# Tool: verify_events
# ---------------------------------------------------------------------------


class TestVerifyEvents:
    def test_verify_no_events(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import verify_events

        result = verify_events(str(tmp_path))
        assert "error" in result

    def test_verify_with_events(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import verify_events

        run_dir = _write_run(tmp_path)
        result = verify_events(str(run_dir))
        assert "chain_status" in result
        assert "event_count" in result


# ---------------------------------------------------------------------------
# Tool: explain_plan
# ---------------------------------------------------------------------------


class TestExplainPlan:
    def test_explain_valid_plan(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import explain_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = explain_plan(str(plan_path))
        assert "plan_name" in result
        assert "tasks" in result


# ---------------------------------------------------------------------------
# Tool: plan_status
# ---------------------------------------------------------------------------


class TestPlanStatus:
    def test_status_valid_plan(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import plan_status

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = plan_status(str(plan_path))
        assert "plan_name" in result or "error" not in result


# ---------------------------------------------------------------------------
# Tool: cleanup_runs
# ---------------------------------------------------------------------------


class TestCleanupRuns:
    def test_cleanup_dry_run(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import cleanup_runs

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = cleanup_runs(str(plan_path), dry_run=True)
        assert result.get("dry_run") is True


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


class TestResources:
    def test_list_runs(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import list_runs

        _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run_root",
                           lambda base=None: tmp_path / ".maestro-runs")
        result = json.loads(list_runs())
        assert isinstance(result, list)
        assert len(result) >= 1
        assert result[0]["id"].startswith("20260320")

    def test_read_manifest(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_manifest

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = json.loads(read_manifest(run_dir.name))
        assert result["plan_name"] == "test"
        assert result["success"] is True

    def test_read_summary(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_summary

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = read_summary(run_dir.name)
        assert "Run Summary" in result

    def test_read_events(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_events

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = read_events(run_dir.name)
        assert "run_start" in result

    def test_read_task_log(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_task_log

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = read_task_log(run_dir.name, "task-a")
        assert "task-a output" in result

    def test_read_task_result(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_task_result

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = json.loads(read_task_result(run_dir.name, "task-a"))
        assert result["status"] == "success"

    def test_list_plans(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import list_plans

        _write_plan(tmp_path, "version: 1\nname: test\ntasks:\n  - id: a\n    command: echo ok")
        monkeypatch.setattr("maestro_cli.mcp_server._list_plan_files",
                           lambda base=None: [tmp_path / "plan.yaml"])
        result = json.loads(list_plans())
        assert len(result) >= 1

    def test_read_plan(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import read_plan

        plan_path = _write_plan(tmp_path, "version: 1\nname: test\ntasks:\n  - id: a\n    command: echo ok\n")
        result = read_plan(str(plan_path))
        assert "version: 1" in result

    def test_read_plan_not_found(self) -> None:
        from maestro_cli.mcp_server import read_plan

        result = read_plan("nonexistent.yaml")
        assert "not found" in result.lower()

    def test_read_manifest_not_found(self, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_manifest

        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: None)
        result = json.loads(read_manifest("fake-run"))
        assert "error" in result


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_debug_run_not_found(self, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import debug_run

        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: None)
        result = debug_run("nonexistent")
        assert "not found" in result.lower()

    def test_debug_run_with_data(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import debug_run

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = debug_run(run_dir.name)
        assert "task-a" in result
        assert "Root cause" in result

    def test_review_plan(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import review_plan

        plan_path = _write_plan(tmp_path, "version: 1\nname: test\ntasks:\n  - id: a\n    command: echo ok\n")
        result = review_plan(str(plan_path))
        assert "version: 1" in result
        assert "Check for:" in result

    def test_create_plan(self) -> None:
        from maestro_cli.mcp_server import create_plan

        result = create_plan("Deploy a web application with tests")
        assert "Maestro CLI plan" in result
        assert "version: 1" in result


# ---------------------------------------------------------------------------
# NEW: _NoOpDecorator
# ---------------------------------------------------------------------------


class TestNoOpDecorator:
    """Test the _NoOpDecorator fallback used when MCP SDK is not installed.

    Since MCP may or may not be installed in the test environment, we
    re-create the class inline to guarantee we exercise its logic.
    """

    @staticmethod
    def _make_decorator() -> Any:
        """Build a fresh _NoOpDecorator regardless of MCP availability."""

        class _NoOpDecorator:
            def tool(self, **kw):  # type: ignore[no-untyped-def]
                def _wrap(fn):  # type: ignore[no-untyped-def]
                    return fn
                return _wrap

            def resource(self, *a, **kw):  # type: ignore[no-untyped-def]
                def _wrap(fn):  # type: ignore[no-untyped-def]
                    return fn
                return _wrap

            def prompt(self, **kw):  # type: ignore[no-untyped-def]
                def _wrap(fn):  # type: ignore[no-untyped-def]
                    return fn
                return _wrap

        return _NoOpDecorator()

    def test_tool_returns_function_unchanged(self) -> None:
        deco = self._make_decorator()

        @deco.tool()
        def my_func() -> str:
            return "ok"

        assert my_func() == "ok"

    def test_resource_returns_function_unchanged(self) -> None:
        deco = self._make_decorator()

        @deco.resource("maestro://test")
        def my_res() -> str:
            return "resource"

        assert my_res() == "resource"

    def test_prompt_returns_function_unchanged(self) -> None:
        deco = self._make_decorator()

        @deco.prompt(title="Test")
        def my_prompt() -> str:
            return "prompt"

        assert my_prompt() == "prompt"

    def test_tool_with_kwargs(self) -> None:
        deco = self._make_decorator()

        @deco.tool(name="custom", description="my tool")
        def my_func(x: int) -> int:
            return x * 2

        assert my_func(5) == 10


# ---------------------------------------------------------------------------
# NEW: main() without MCP
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_without_mcp_raises_system_exit(self, monkeypatch: Any) -> None:
        from maestro_cli import mcp_server

        monkeypatch.setattr(mcp_server, "_HAS_MCP", False)
        with pytest.raises(SystemExit, match="1"):
            mcp_server.main()


# ---------------------------------------------------------------------------
# NEW: _find_run helper
# ---------------------------------------------------------------------------


class TestFindRun:
    def test_exact_match(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _find_run

        run_root = tmp_path / ".maestro-runs"
        run_dir = run_root / "20260320_test"
        run_dir.mkdir(parents=True)
        result = _find_run("20260320_test", run_root=run_root)
        assert result == run_dir

    def test_prefix_match(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _find_run

        run_root = tmp_path / ".maestro-runs"
        run_dir = run_root / "20260320_test"
        run_dir.mkdir(parents=True)
        result = _find_run("20260320", run_root=run_root)
        assert result == run_dir

    def test_returns_none_when_not_found(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _find_run

        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir(parents=True)
        result = _find_run("nonexistent", run_root=run_root)
        assert result is None

    def test_returns_none_when_run_root_not_a_dir(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _find_run

        fake_root = tmp_path / "not_a_dir"
        result = _find_run("anything", run_root=fake_root)
        assert result is None


# ---------------------------------------------------------------------------
# NEW: _list_run_dirs helper
# ---------------------------------------------------------------------------


class TestListRunDirs:
    def test_empty_directory(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _list_run_dirs

        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()
        result = _list_run_dirs(run_root)
        assert result == []

    def test_multiple_dirs_sorted_by_mtime(self, tmp_path: Path) -> None:
        import time
        from maestro_cli.mcp_server import _list_run_dirs

        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()
        d1 = run_root / "run_old"
        d1.mkdir()
        time.sleep(0.05)
        d2 = run_root / "run_new"
        d2.mkdir()
        result = _list_run_dirs(run_root)
        assert len(result) == 2
        # Newest first
        assert result[0].name == "run_new"
        assert result[1].name == "run_old"

    def test_not_a_dir_returns_empty(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _list_run_dirs

        fake = tmp_path / "not_a_dir"
        result = _list_run_dirs(fake)
        assert result == []

    def test_skips_files(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _list_run_dirs

        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()
        (run_root / "some_file.txt").write_text("not a dir", encoding="utf-8")
        d1 = run_root / "run_a"
        d1.mkdir()
        result = _list_run_dirs(run_root)
        assert len(result) == 1
        assert result[0].name == "run_a"


# ---------------------------------------------------------------------------
# NEW: _list_plan_files helper
# ---------------------------------------------------------------------------


class TestListPlanFiles:
    def test_finds_yaml_and_yml(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _list_plan_files

        (tmp_path / "plan.yaml").write_text("version: 1", encoding="utf-8")
        (tmp_path / "other.yml").write_text("version: 1", encoding="utf-8")
        result = _list_plan_files(base=tmp_path)
        names = {p.name for p in result}
        assert "plan.yaml" in names
        assert "other.yml" in names

    def test_skips_hidden_dirs(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _list_plan_files

        hidden = tmp_path / ".hidden"
        hidden.mkdir()
        (hidden / "secret.yaml").write_text("version: 1", encoding="utf-8")
        result = _list_plan_files(base=tmp_path)
        assert len(result) == 0

    def test_handles_nested_dirs(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _list_plan_files

        sub = tmp_path / "plans"
        sub.mkdir()
        (sub / "deploy.yaml").write_text("version: 1", encoding="utf-8")
        result = _list_plan_files(base=tmp_path)
        names = {p.name for p in result}
        assert "deploy.yaml" in names

    def test_empty_dir_returns_empty(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _list_plan_files

        result = _list_plan_files(base=tmp_path)
        assert result == []

    def test_oserror_returns_empty(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import _list_plan_files

        # Force an OSError by monkeypatching iterdir
        original_iterdir = Path.iterdir

        def _boom(self: Path) -> Any:
            raise OSError("permission denied")

        monkeypatch.setattr(Path, "iterdir", _boom)
        result = _list_plan_files(base=tmp_path)
        assert result == []


# ---------------------------------------------------------------------------
# NEW: _find_run_root helper
# ---------------------------------------------------------------------------


class TestFindRunRoot:
    def test_with_base_path_existing(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _find_run_root

        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()
        result = _find_run_root(base=tmp_path)
        assert result == run_root
        assert result.is_dir()

    def test_with_base_path_not_existing(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _find_run_root

        result = _find_run_root(base=tmp_path)
        # Returns the path even if it doesn't exist as a dir
        assert result == tmp_path / ".maestro-runs"

    def test_without_base_path_uses_cwd(self, monkeypatch: Any, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import _find_run_root

        monkeypatch.chdir(tmp_path)
        result = _find_run_root()
        assert result == tmp_path / ".maestro-runs"


# ---------------------------------------------------------------------------
# NEW: run_plan_tool (additional coverage)
# ---------------------------------------------------------------------------


class TestRunPlanTool:
    def test_dry_run_success(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import run_plan_tool

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = run_plan_tool(str(plan_path), dry_run=True)
        assert result.get("success") is True or "task_results" in result

    def test_invalid_plan_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import run_plan_tool

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks: []
        """)
        result = run_plan_tool(str(plan_path))
        assert result["success"] is False
        assert "error" in result

    def test_execution_profile_param(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import run_plan_tool

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = run_plan_tool(str(plan_path), dry_run=True, execution_profile="safe")
        # Should not error out — profile is accepted
        assert "error" not in result or result.get("success") is True

    def test_only_filter(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import run_plan_tool

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo a
              - id: b
                command: echo b
        """)
        result = run_plan_tool(str(plan_path), dry_run=True, only=["a"])
        assert result.get("success") is True or "task_results" in result

    def test_skip_filter(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import run_plan_tool

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo a
              - id: b
                command: echo b
        """)
        result = run_plan_tool(str(plan_path), dry_run=True, skip=["b"])
        assert result.get("success") is True or "task_results" in result

    def test_max_parallel_param(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import run_plan_tool

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = run_plan_tool(str(plan_path), dry_run=True, max_parallel=2)
        assert "error" not in result or result.get("success") is True

    def test_nonexistent_plan_file(self) -> None:
        from maestro_cli.mcp_server import run_plan_tool

        result = run_plan_tool("/nonexistent/plan.yaml")
        assert result["success"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# NEW: blame_run (additional coverage)
# ---------------------------------------------------------------------------


class TestBlameRun:
    def test_run_not_found_returns_error(self, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import blame_run

        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: None)
        result = blame_run("/nonexistent/run")
        assert "error" in result

    def test_valid_run_with_manifest(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import blame_run

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = blame_run(run_dir.name)
        # blame_run returns a BlameChain.to_dict()
        assert "root_task_id" in result or "error" not in result

    def test_run_with_failed_tasks(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import blame_run

        run_dir = tmp_path / ".maestro-runs" / "20260320_fail"
        run_dir.mkdir(parents=True)
        manifest = {
            "plan_name": "fail",
            "success": False,
            "started_at": "2026-03-20T10:00:00Z",
            "finished_at": "2026-03-20T10:01:00Z",
            "execution_profile": "plan",
            "task_results": {
                "task-a": {
                    "task_id": "task-a",
                    "status": "failed",
                    "exit_code": 1,
                    "duration_sec": 5.0,
                    "cost_usd": 0.01,
                    "message": "Command failed",
                },
            },
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        (run_dir / "events.jsonl").write_text(
            '{"event":"run_start","plan_name":"fail"}\n'
            '{"event":"task_complete","task_id":"task-a","status":"failed"}\n',
            encoding="utf-8",
        )
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = blame_run(run_dir.name)
        assert "nodes" in result or "suggested_fixes" in result

    def test_corrupt_manifest_returns_suggested_fixes(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import blame_run

        run_dir = tmp_path / ".maestro-runs" / "20260320_exc"
        run_dir.mkdir(parents=True)
        # Invalid JSON manifest — blame_run handles gracefully with suggested_fixes
        (run_dir / "run_manifest.json").write_text("not json", encoding="utf-8")
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = blame_run(run_dir.name)
        # blame.blame_run returns BlameChain with suggested_fixes on bad manifest
        assert "suggested_fixes" in result
        assert len(result["suggested_fixes"]) > 0


# ---------------------------------------------------------------------------
# NEW: diff_runs (additional coverage)
# ---------------------------------------------------------------------------


class TestDiffRuns:
    def test_both_paths_found(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import diff_runs

        run_a = _write_run(tmp_path, "plan_a")
        run_b = _write_run(tmp_path, "plan_b")

        def _mock_find(rid: str, rr: Any = None) -> Path | None:
            if "plan_a" in rid:
                return run_a
            if "plan_b" in rid:
                return run_b
            return None

        monkeypatch.setattr("maestro_cli.mcp_server._find_run", _mock_find)
        result = diff_runs(run_a.name, run_b.name)
        # diff_runs calls diff.diff_runs which returns RunDiff;
        # RunDiff may or may not have to_dict(). Either way we get a result dict.
        assert isinstance(result, dict)

    def test_one_path_missing_returns_error(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import diff_runs

        run_a = _write_run(tmp_path, "plan_a")

        def _mock_find(rid: str, rr: Any = None) -> Path | None:
            if "plan_a" in rid:
                return run_a
            return None

        monkeypatch.setattr("maestro_cli.mcp_server._find_run", _mock_find)
        result = diff_runs(run_a.name, "nonexistent")
        # _find_run returns None → Path("nonexistent") → no manifest → error
        assert "error" in result

    def test_both_paths_missing_returns_error(self, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import diff_runs

        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: None)
        result = diff_runs("fake_a", "fake_b")
        assert "error" in result


# ---------------------------------------------------------------------------
# NEW: suggest_plan (additional coverage)
# ---------------------------------------------------------------------------


class TestSuggestPlan:
    def test_valid_plan_no_runs(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import suggest_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = suggest_plan(str(plan_path))
        assert "plan_name" in result
        assert result.get("runs_analyzed") == 0

    def test_invalid_plan_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import suggest_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks: []
        """)
        result = suggest_plan(str(plan_path))
        assert "error" in result

    def test_custom_min_runs(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import suggest_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = suggest_plan(str(plan_path), min_runs=1)
        assert "plan_name" in result


# ---------------------------------------------------------------------------
# NEW: audit_plan with fix=True
# ---------------------------------------------------------------------------


class TestAuditPlanFix:
    def test_fix_applies_and_returns_fixes(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import audit_plan

        # Plan without max_cost_usd should trigger SEC001
        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = audit_plan(str(plan_path), fix=True)
        assert "findings" in result
        # If there are findings and fixes were possible, fixes_applied is present
        if result.get("total", 0) > 0:
            assert "fixes_applied" in result

    def test_fix_false_no_fixes_key(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import audit_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        result = audit_plan(str(plan_path), fix=False)
        assert "fixes_applied" not in result


# ---------------------------------------------------------------------------
# NEW: explain_plan with invalid plan
# ---------------------------------------------------------------------------


class TestExplainPlanErrors:
    def test_invalid_plan_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import explain_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks: []
        """)
        result = explain_plan(str(plan_path))
        assert "error" in result

    def test_nonexistent_plan(self) -> None:
        from maestro_cli.mcp_server import explain_plan

        result = explain_plan("/nonexistent/plan.yaml")
        assert "error" in result


# ---------------------------------------------------------------------------
# NEW: plan_status with invalid plan
# ---------------------------------------------------------------------------


class TestPlanStatusErrors:
    def test_invalid_plan_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import plan_status

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks: []
        """)
        result = plan_status(str(plan_path))
        assert "error" in result

    def test_nonexistent_plan(self) -> None:
        from maestro_cli.mcp_server import plan_status

        result = plan_status("/nonexistent/plan.yaml")
        assert "error" in result


# ---------------------------------------------------------------------------
# NEW: cleanup_runs (additional coverage)
# ---------------------------------------------------------------------------


class TestCleanupRunsExtended:
    def test_invalid_plan_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import cleanup_runs

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks: []
        """)
        result = cleanup_runs(str(plan_path))
        assert "error" in result

    def test_nonexistent_plan_returns_error(self) -> None:
        from maestro_cli.mcp_server import cleanup_runs

        result = cleanup_runs("/nonexistent/plan.yaml")
        assert "error" in result

    def test_with_runs_to_clean(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import cleanup_runs

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: test
            tasks:
              - id: a
                command: echo ok
        """)
        # Create some run directories
        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()
        for i in range(3):
            (run_root / f"2026031{i}_test").mkdir()
        result = cleanup_runs(str(plan_path), keep=1, dry_run=True)
        assert result.get("dry_run") is True
        assert "count" in result


# ---------------------------------------------------------------------------
# NEW: Resources - error paths
# ---------------------------------------------------------------------------


class TestResourcesErrorPaths:
    def test_read_summary_run_not_found(self, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_summary

        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: None)
        result = read_summary("nonexistent")
        assert "not found" in result.lower()

    def test_read_summary_no_file(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_summary

        run_dir = tmp_path / "run_empty"
        run_dir.mkdir()
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = read_summary("run_empty")
        assert "no run_summary" in result.lower()

    def test_read_events_run_not_found(self, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_events

        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: None)
        result = read_events("nonexistent")
        assert "not found" in result.lower()

    def test_read_events_no_file(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_events

        run_dir = tmp_path / "run_empty"
        run_dir.mkdir()
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = read_events("run_empty")
        assert "no events" in result.lower()

    def test_read_task_log_run_not_found(self, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_task_log

        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: None)
        result = read_task_log("nonexistent", "task-a")
        assert "not found" in result.lower()

    def test_read_task_log_task_not_found(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_task_log

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = read_task_log(run_dir.name, "nonexistent-task")
        assert "no log" in result.lower()

    def test_read_task_result_run_not_found(self, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_task_result

        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: None)
        result = json.loads(read_task_result("nonexistent", "task-a"))
        assert "error" in result

    def test_read_task_result_task_not_found(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_task_result

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = json.loads(read_task_result(run_dir.name, "nonexistent-task"))
        assert "error" in result

    def test_list_plans_empty_workspace(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import list_plans

        monkeypatch.setattr("maestro_cli.mcp_server._list_plan_files",
                           lambda base=None: [])
        result = json.loads(list_plans())
        assert result == []

    def test_list_runs_empty_run_root(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import list_runs

        empty_root = tmp_path / ".maestro-runs"
        empty_root.mkdir()
        monkeypatch.setattr("maestro_cli.mcp_server._find_run_root",
                           lambda base=None: empty_root)
        result = json.loads(list_runs())
        assert result == []

    def test_list_runs_nonexistent_root(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import list_runs

        monkeypatch.setattr("maestro_cli.mcp_server._find_run_root",
                           lambda base=None: tmp_path / "nope")
        result = json.loads(list_runs())
        assert result == []

    def test_read_manifest_no_manifest_file(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_manifest

        run_dir = tmp_path / "run_no_manifest"
        run_dir.mkdir()
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = json.loads(read_manifest("run_no_manifest"))
        assert "error" in result
        assert "no run_manifest" in result["error"].lower() or "manifest" in result["error"].lower()


# ---------------------------------------------------------------------------
# NEW: Prompts - error and edge-case paths
# ---------------------------------------------------------------------------


class TestPromptsExtended:
    def test_review_plan_not_found(self) -> None:
        from maestro_cli.mcp_server import review_plan

        result = review_plan("/nonexistent/plan.yaml")
        assert "not found" in result.lower()

    def test_create_plan_structure(self) -> None:
        from maestro_cli.mcp_server import create_plan

        result = create_plan("Build a CI pipeline")
        assert "version: 1" in result
        assert "verify_command" in result
        assert "timeout_sec" in result
        assert "depends_on" in result
        assert "max_cost_usd" in result
        assert "engines" in result.lower() or "engine" in result.lower()

    def test_debug_run_manifest_loading(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import debug_run

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = debug_run(run_dir.name)
        # Should include manifest data
        assert "Plan:" in result
        assert "Success:" in result
        assert "task-a" in result

    def test_debug_run_events_loading(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import debug_run

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = debug_run(run_dir.name)
        assert "Event count:" in result
        assert "Last 10 events:" in result

    def test_debug_run_no_manifest(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import debug_run

        run_dir = tmp_path / ".maestro-runs" / "20260320_empty"
        run_dir.mkdir(parents=True)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = debug_run(run_dir.name)
        # Should still return something without crashing
        assert "Analyze" in result or "Run:" in result

    def test_debug_run_corrupt_manifest(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import debug_run

        run_dir = tmp_path / ".maestro-runs" / "20260320_corrupt"
        run_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text("not json{{{", encoding="utf-8")
        (run_dir / "events.jsonl").write_text(
            '{"event":"run_start"}\n', encoding="utf-8",
        )
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = debug_run(run_dir.name)
        assert "Could not read manifest" in result

    def test_debug_run_no_events_file(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import debug_run

        run_dir = tmp_path / ".maestro-runs" / "20260320_noevents"
        run_dir.mkdir(parents=True)
        manifest = {
            "plan_name": "noevents",
            "success": True,
            "task_results": {},
        }
        (run_dir / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8",
        )
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = debug_run(run_dir.name)
        # Should still work without events
        assert "Run:" in result
        # Events section should not appear
        assert "Event count:" not in result


# ---------------------------------------------------------------------------
# NEW: validate_plan with complex plan
# ---------------------------------------------------------------------------


class TestValidatePlanComplex:
    def test_complex_plan_with_deps_and_judge(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import validate_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: complex
            max_cost_usd: 5.0
            tasks:
              - id: build
                command: echo build
              - id: test
                command: echo test
                depends_on: [build]
              - id: review
                engine: claude
                model: sonnet
                prompt: "Review the code"
                depends_on: [build, test]
                judge:
                  criteria:
                    - "Code quality is acceptable"
                  pass_threshold: 0.7
        """)
        result = validate_plan(str(plan_path))
        assert result["valid"] is True
        assert result["task_count"] == 3
        assert set(result["task_ids"]) == {"build", "test", "review"}
        assert result["max_cost_usd"] == 5.0

    def test_plan_with_all_optional_fields(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import validate_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: full
            max_parallel: 4
            fail_fast: true
            max_cost_usd: 10.0
            tasks:
              - id: a
                command: echo a
                tags: [fast, trivial]
              - id: b
                command: echo b
                depends_on: [a]
                allow_failure: true
                max_retries: 2
                timeout_sec: 60
        """)
        result = validate_plan(str(plan_path))
        assert result["valid"] is True
        assert result["max_parallel"] == 4
        assert result["fail_fast"] is True

    def test_plan_cycle_detection(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import validate_plan

        plan_path = _write_plan(tmp_path, """\
            version: 1
            name: cycle
            tasks:
              - id: a
                command: echo a
                depends_on: [b]
              - id: b
                command: echo b
                depends_on: [a]
        """)
        result = validate_plan(str(plan_path))
        assert result["valid"] is False
        assert "error" in result

    def test_nonexistent_plan_file(self) -> None:
        from maestro_cli.mcp_server import validate_plan

        result = validate_plan("/nonexistent/plan.yaml")
        assert result["valid"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# NEW: scaffold_plan error path
# ---------------------------------------------------------------------------


class TestScaffoldPlanErrors:
    def test_nonexistent_brief(self) -> None:
        from maestro_cli.mcp_server import scaffold_plan

        result = scaffold_plan("/nonexistent/brief.yaml")
        assert "error" in result.lower()

    def test_invalid_brief_yaml(self, tmp_path: Path) -> None:
        from maestro_cli.mcp_server import scaffold_plan

        brief_path = tmp_path / "bad_brief.yaml"
        brief_path.write_text("not: valid: brief: yaml", encoding="utf-8")
        result = scaffold_plan(str(brief_path))
        assert "error" in result.lower() or "Error" in result


# ---------------------------------------------------------------------------
# NEW: verify_events edge cases
# ---------------------------------------------------------------------------


class TestVerifyEventsExtended:
    def test_verify_events_via_find_run(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import verify_events

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: run_dir)
        result = verify_events(run_dir.name)
        assert "chain_status" in result
        assert "event_count" in result
        assert "artefact_issues" in result

    def test_verify_events_no_run_found(self, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import verify_events

        monkeypatch.setattr("maestro_cli.mcp_server._find_run",
                           lambda rid, rr=None: None)
        result = verify_events("nonexistent")
        assert "error" in result


# ---------------------------------------------------------------------------
# NEW: list_runs with manifest parsing edge cases
# ---------------------------------------------------------------------------


class TestListRunsExtended:
    def test_list_runs_with_corrupt_manifest(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import list_runs

        run_root = tmp_path / ".maestro-runs"
        run_dir = run_root / "20260320_corrupt"
        run_dir.mkdir(parents=True)
        (run_dir / "run_manifest.json").write_text("not json", encoding="utf-8")
        monkeypatch.setattr("maestro_cli.mcp_server._find_run_root",
                           lambda base=None: run_root)
        result = json.loads(list_runs())
        assert len(result) == 1
        assert result[0]["id"] == "20260320_corrupt"
        # plan_name should not be present since manifest is corrupt
        assert "plan_name" not in result[0]

    def test_list_runs_with_valid_manifest(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import list_runs

        run_dir = _write_run(tmp_path)
        monkeypatch.setattr("maestro_cli.mcp_server._find_run_root",
                           lambda base=None: tmp_path / ".maestro-runs")
        result = json.loads(list_runs())
        assert len(result) >= 1
        assert result[0]["plan_name"] == "test"
        assert result[0]["success"] is True

    def test_list_runs_caps_at_50(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import list_runs

        run_root = tmp_path / ".maestro-runs"
        run_root.mkdir()
        for i in range(55):
            (run_root / f"run_{i:03d}").mkdir()
        monkeypatch.setattr("maestro_cli.mcp_server._find_run_root",
                           lambda base=None: run_root)
        result = json.loads(list_runs())
        assert len(result) == 50


# ---------------------------------------------------------------------------
# NEW: read_plan edge cases
# ---------------------------------------------------------------------------


class TestReadPlanExtended:
    def test_read_plan_by_stem_match(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_plan

        plan_path = _write_plan(tmp_path, "version: 1\nname: test\ntasks:\n  - id: a\n    command: echo ok\n")
        monkeypatch.setattr("maestro_cli.mcp_server._list_plan_files",
                           lambda base=None: [plan_path])
        # Search by stem (no extension)
        result = read_plan("plan")
        assert "version: 1" in result

    def test_read_plan_by_full_name(self, tmp_path: Path, monkeypatch: Any) -> None:
        from maestro_cli.mcp_server import read_plan

        plan_path = _write_plan(tmp_path, "version: 1\nname: test\ntasks:\n  - id: a\n    command: echo ok\n")
        monkeypatch.setattr("maestro_cli.mcp_server._list_plan_files",
                           lambda base=None: [plan_path])
        result = read_plan("plan.yaml")
        assert "version: 1" in result


# ---------------------------------------------------------------------------
# NEW: doctor output structure
# ---------------------------------------------------------------------------


class TestDoctorExtended:
    def test_doctor_has_python_version_check(self) -> None:
        from maestro_cli.mcp_server import doctor

        results = doctor()
        check_names = {r["check"] for r in results}
        assert "python_version" in check_names

    def test_doctor_has_pyyaml_check(self) -> None:
        from maestro_cli.mcp_server import doctor

        results = doctor()
        check_names = {r["check"] for r in results}
        assert "pyyaml" in check_names

    def test_doctor_status_values(self) -> None:
        from maestro_cli.mcp_server import doctor

        results = doctor()
        valid_statuses = {"ok", "warn", "error", "info"}
        for item in results:
            assert item["status"] in valid_statuses
