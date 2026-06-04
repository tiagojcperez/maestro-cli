"""Coverage tests for mcp_server.py error/edge branches.

Targets the lazily-imported error paths of the MCP tool functions and the
``main()`` entry point. Engine/LLM/network/git are never invoked: the MCP
tools wrap pure-Python helpers, and the underlying module functions are
monkeypatched to drive ``except`` branches deterministically.

Each MCP tool resolves its real implementation via a late ``from .X import Y``,
so patching ``maestro_cli.X.Y`` is picked up at call time.
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


_VALID_PLAN = """\
    version: 1
    name: test
    tasks:
      - id: a
        command: echo ok
"""


def _write_run_dir(tmp_path: Path, name: str = "20260320_test") -> Path:
    rd = tmp_path / ".maestro-runs" / name
    rd.mkdir(parents=True)
    return rd


# ---------------------------------------------------------------------------
# _list_plan_files inner OSError branch (103-104)
# ---------------------------------------------------------------------------


class TestListPlanFilesInnerOSError:
    def test_subdir_iterdir_raises_oserror_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli import mcp_server

        # Top-level: one yaml file plus one non-dot subdirectory.
        (tmp_path / "top.yaml").write_text("x: 1", encoding="utf-8")
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "child.yaml").write_text("y: 1", encoding="utf-8")

        real_iterdir = Path.iterdir

        def fake_iterdir(self: Path) -> Any:
            if self == subdir:
                raise OSError("permission denied on subdir")
            return real_iterdir(self)

        monkeypatch.setattr(Path, "iterdir", fake_iterdir)

        plans = mcp_server._list_plan_files(tmp_path)
        # Top-level yaml still collected; the failing subdir was skipped silently.
        names = [p.name for p in plans]
        assert "top.yaml" in names
        assert "child.yaml" not in names


# ---------------------------------------------------------------------------
# run_plan_tool except branch (169-170)
# ---------------------------------------------------------------------------


class TestRunPlanToolError:
    def test_runtime_exception_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli import mcp_server
        import maestro_cli.scheduler as scheduler

        plan_path = _write_plan(tmp_path, _VALID_PLAN)

        def boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError("scheduler exploded")

        monkeypatch.setattr(scheduler, "run_plan", boom)

        result = mcp_server.run_plan_tool(str(plan_path), dry_run=True)
        assert result["success"] is False
        assert "scheduler exploded" in result["error"]

    def test_invalid_plan_returns_validation_error(self, tmp_path: Path) -> None:
        from maestro_cli import mcp_server

        plan_path = _write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            tasks: []
            """,
        )
        result = mcp_server.run_plan_tool(str(plan_path))
        assert result["success"] is False
        assert "error" in result


# ---------------------------------------------------------------------------
# blame_run except branch (218-219)
# ---------------------------------------------------------------------------


class TestBlameRunError:
    def test_blame_exception_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli import mcp_server
        import maestro_cli.blame as blame

        rd = _write_run_dir(tmp_path)

        def boom(run_path: Path) -> Any:
            raise ValueError("blame failed")

        monkeypatch.setattr(blame, "blame_run", boom)

        result = mcp_server.blame_run(str(rd))
        assert "blame failed" in result["error"]

    def test_missing_run_dir_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli import mcp_server

        result = mcp_server.blame_run(str(tmp_path / "does-not-exist"))
        assert "not found" in result["error"]


# ---------------------------------------------------------------------------
# explain_plan except branch (258-259)
# ---------------------------------------------------------------------------


class TestExplainPlanError:
    def test_explain_exception_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli import mcp_server
        import maestro_cli.explain as explain

        plan_path = _write_plan(tmp_path, _VALID_PLAN)

        def boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError("explain broke")

        monkeypatch.setattr(explain, "explain_plan", boom)

        result = mcp_server.explain_plan(str(plan_path))
        assert "explain broke" in result["error"]

    def test_invalid_plan_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli import mcp_server

        plan_path = _write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            tasks: []
            """,
        )
        result = mcp_server.explain_plan(str(plan_path))
        assert "error" in result


# ---------------------------------------------------------------------------
# plan_status except branch (279-280)
# ---------------------------------------------------------------------------


class TestPlanStatusError:
    def test_status_exception_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli import mcp_server
        import maestro_cli.status as status

        plan_path = _write_plan(tmp_path, _VALID_PLAN)

        def boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError("status broke")

        monkeypatch.setattr(status, "plan_status", boom)

        result = mcp_server.plan_status(str(plan_path))
        assert "status broke" in result["error"]

    def test_invalid_plan_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli import mcp_server

        plan_path = _write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            tasks: []
            """,
        )
        result = mcp_server.plan_status(str(plan_path))
        assert "error" in result


# ---------------------------------------------------------------------------
# suggest_plan except branch (300-301)
# ---------------------------------------------------------------------------


class TestSuggestPlanError:
    def test_suggest_exception_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli import mcp_server
        import maestro_cli.suggest as suggest

        plan_path = _write_plan(tmp_path, _VALID_PLAN)

        def boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError("suggest broke")

        monkeypatch.setattr(suggest, "suggest_plan", boom)

        result = mcp_server.suggest_plan(str(plan_path))
        assert "suggest broke" in result["error"]

    def test_invalid_plan_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli import mcp_server

        plan_path = _write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            tasks: []
            """,
        )
        result = mcp_server.suggest_plan(str(plan_path))
        assert "error" in result


# ---------------------------------------------------------------------------
# scaffold_plan validate branch (331-342)
# ---------------------------------------------------------------------------

_BRIEF = """\
    name: demo
    goal: build something
    tasks:
      - id: a
        description: do a thing
"""


class TestScaffoldPlanValidateBranch:
    def test_validate_true_valid_yaml_returns_yaml(self, tmp_path: Path) -> None:
        from maestro_cli import mcp_server

        brief_path = tmp_path / "brief.yaml"
        brief_path.write_text(textwrap.dedent(_BRIEF), encoding="utf-8")

        out = mcp_server.scaffold_plan(str(brief_path), validate=True)
        # A successfully validated scaffold returns the YAML directly.
        assert "version: 1" in out
        assert not out.startswith("Error:")
        assert "validation errors" not in out

    def test_validate_true_invalid_yaml_returns_error_prefix(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli import mcp_server
        import maestro_cli.scaffold as scaffold

        brief_path = tmp_path / "brief.yaml"
        brief_path.write_text(textwrap.dedent(_BRIEF), encoding="utf-8")

        # Force the scaffolded YAML to be structurally invalid so the
        # validate branch's load_plan raises PlanValidationError.
        def bad_scaffold(brief: Any) -> str:
            return "version: 1\nname: bad\ntasks: []\n"

        monkeypatch.setattr(scaffold, "scaffold_plan", bad_scaffold)

        out = mcp_server.scaffold_plan(str(brief_path), validate=True)
        assert "validation errors" in out

    def test_load_brief_failure_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli import mcp_server

        # Nonexistent brief path -> load_brief raises -> outer except.
        out = mcp_server.scaffold_plan(str(tmp_path / "nope.yaml"), validate=False)
        assert out.startswith("Error:")


# ---------------------------------------------------------------------------
# verify_events except branch (371-372)
# ---------------------------------------------------------------------------


class TestVerifyEventsError:
    def test_replay_exception_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli import mcp_server
        import maestro_cli.eventsource as eventsource

        rd = _write_run_dir(tmp_path)
        (rd / "events.jsonl").write_text(
            '{"event":"run_start"}\n', encoding="utf-8"
        )

        def boom(events_path: Path) -> Any:
            raise RuntimeError("replay broke")

        monkeypatch.setattr(eventsource, "replay_events", boom)

        result = mcp_server.verify_events(str(rd))
        assert "replay broke" in result["error"]

    def test_missing_events_file_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli import mcp_server

        rd = _write_run_dir(tmp_path)
        result = mcp_server.verify_events(str(rd))
        assert "No events.jsonl" in result["error"]


# ---------------------------------------------------------------------------
# cleanup_runs except branch (407-408)
# ---------------------------------------------------------------------------


class TestCleanupRunsError:
    def test_cleanup_exception_returns_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli import mcp_server
        import maestro_cli.cleanup as cleanup

        plan_path = _write_plan(tmp_path, _VALID_PLAN)

        def boom(*a: Any, **k: Any) -> Any:
            raise RuntimeError("cleanup broke")

        monkeypatch.setattr(cleanup, "cleanup_runs", boom)

        result = mcp_server.cleanup_runs(str(plan_path))
        assert "cleanup broke" in result["error"]

    def test_invalid_plan_returns_error(self, tmp_path: Path) -> None:
        from maestro_cli import mcp_server

        plan_path = _write_plan(
            tmp_path,
            """\
            version: 1
            name: test
            tasks: []
            """,
        )
        result = mcp_server.cleanup_runs(str(plan_path))
        assert "error" in result

    def test_cleanup_dry_run_success(self, tmp_path: Path) -> None:
        from maestro_cli import mcp_server

        plan_path = _write_plan(tmp_path, _VALID_PLAN)
        result = mcp_server.cleanup_runs(str(plan_path), dry_run=True)
        assert result["dry_run"] is True
        assert "affected" in result
        assert "count" in result


# ---------------------------------------------------------------------------
# debug_run events OSError branch (565-566)
# ---------------------------------------------------------------------------


class TestDebugRunEventsOSError:
    def test_events_read_oserror_is_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli import mcp_server

        # debug_run resolves runs via _find_run_root()/Path.cwd(); operate
        # inside tmp_path so the run dir is discoverable by bare name.
        monkeypatch.chdir(tmp_path)
        rd = _write_run_dir(tmp_path)
        # Manifest present and readable.
        manifest = {
            "plan_name": "test",
            "success": False,
            "execution_profile": "plan",
            "task_results": {
                "a": {"status": "failed", "message": "boom"},
            },
        }
        (rd / "run_manifest.json").write_text(
            json.dumps(manifest), encoding="utf-8"
        )
        events_file = rd / "events.jsonl"
        events_file.write_text('{"event":"x"}\n', encoding="utf-8")

        real_read_text = Path.read_text

        def fake_read_text(self: Path, *a: Any, **k: Any) -> str:
            if self == events_file:
                raise OSError("cannot read events")
            return real_read_text(self, *a, **k)

        monkeypatch.setattr(Path, "read_text", fake_read_text)

        # _find_run resolves the run dir; events read raises OSError and is
        # swallowed, so the prompt is still assembled without the events section.
        out = mcp_server.debug_run(rd.name)
        assert "Analyze this failed Maestro run" in out
        assert "Root cause of the failure" in out
        # Event count line never appended because read failed.
        assert "Event count" not in out


# ---------------------------------------------------------------------------
# main() entry point (mcp.run with mcp installed)
# ---------------------------------------------------------------------------


class TestMain:
    def test_main_runs_stdio_transport(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from maestro_cli import mcp_server

        # mcp IS installed in CI, so _HAS_MCP is True and main() reaches
        # mcp.run(transport="stdio"). Patch the run method so no server starts.
        called: dict[str, Any] = {}

        def fake_run(*a: Any, **k: Any) -> None:
            called["transport"] = k.get("transport")

        monkeypatch.setattr(mcp_server.mcp, "run", fake_run)
        # Defensive: ensure the guard is taken as installed.
        monkeypatch.setattr(mcp_server, "_HAS_MCP", True)

        mcp_server.main()
        assert called["transport"] == "stdio"

    def test_main_without_mcp_exits(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from maestro_cli import mcp_server

        # Simulate the SDK-not-installed guard.
        monkeypatch.setattr(mcp_server, "_HAS_MCP", False)
        with pytest.raises(SystemExit):
            mcp_server.main()
        out = capsys.readouterr().out
        assert "MCP SDK not installed" in out
