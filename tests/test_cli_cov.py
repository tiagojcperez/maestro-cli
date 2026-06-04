from __future__ import annotations

import json
import threading
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from maestro_cli.cli import main
from maestro_cli.models import PlanSuggestions


# ---------------------------------------------------------------------------
# Helpers / module-level YAML constants
# ---------------------------------------------------------------------------

def _write(tmp_path: Path, content: str, name: str = "plan.yaml") -> Path:
    p = tmp_path / name
    p.write_text(content, encoding="utf-8")
    return p


# A topologically dense plan: 5 engine tasks, cross dependencies, judges,
# retries, and multiple distinct engines. This pushes the complexity density
# score above the 0.30 threshold so the "factors:" line in _cmd_validate prints.
_DENSE_PLAN_YAML = """\
version: 1
name: dense-plan
max_cost_usd: 10.0
tasks:
  - id: a
    engine: claude
    model: haiku
    prompt: a
    max_retries: 2
    judge:
      criteria: ["good"]
      pass_threshold: 0.5
  - id: b
    depends_on: [a]
    engine: gemini
    model: flash
    prompt: b
    max_retries: 2
    judge:
      criteria: ["good"]
      pass_threshold: 0.5
  - id: c
    depends_on: [a, b]
    engine: codex
    prompt: c
    max_retries: 2
    judge:
      criteria: ["good"]
      pass_threshold: 0.5
  - id: d
    depends_on: [b, c]
    engine: qwen
    prompt: d
    max_retries: 2
    judge:
      criteria: ["good"]
      pass_threshold: 0.5
  - id: e
    depends_on: [c, d]
    engine: claude
    model: haiku
    prompt: e
    max_retries: 2
    judge:
      criteria: ["good"]
      pass_threshold: 0.5
"""

# A simple, clean plan that passes audit without errors.
_CLEAN_PLAN_YAML = """\
version: 1
name: clean-plan
max_cost_usd: 10.0
tasks:
  - id: t1
    command: "echo hello"
"""


# ===========================================================================
# _cmd_validate — complexity "factors:" branch (score >= 0.30)
# ===========================================================================

class TestValidateComplexityFactors:
    def test_validate_prints_factors_for_dense_plan(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = _write(tmp_path, _DENSE_PLAN_YAML)
        rc = main(["validate", str(plan_path)])
        out = capsys.readouterr().out
        assert rc == 0
        # complexity line + the factors detail line only printed when >= 0.30
        assert "complexity:" in out
        assert "factors:" in out
        assert "S_complex=" in out

    def test_validate_omits_factors_for_trivial_plan(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = _write(tmp_path, _CLEAN_PLAN_YAML)
        rc = main(["validate", str(plan_path)])
        out = capsys.readouterr().out
        assert rc == 0
        # A single trivial command task scores well below 0.30.
        assert "factors:" not in out


# ===========================================================================
# _cmd_check — --with-suggest branches (text + json + failure)
# ===========================================================================

class TestCheckWithSuggest:
    def test_check_with_suggest_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = _write(tmp_path, _CLEAN_PLAN_YAML)
        fake = PlanSuggestions(plan_name="clean-plan", runs_analyzed=5)
        monkeypatch.setattr(
            "maestro_cli.suggest.suggest_plan",
            MagicMock(return_value=fake),
        )
        rc = main(["check", str(plan_path), "--with-suggest"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "== Suggestions ==" in out

    def test_check_with_suggest_run_dir_override(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # Exercises the run_dir override branch (args.run_dir is truthy).
        plan_path = _write(tmp_path, _CLEAN_PLAN_YAML)
        custom_dir = tmp_path / "custom-runs"
        custom_dir.mkdir()
        captured: dict[str, object] = {}

        def _fake_suggest(plan: object, run_dir: object, min_runs: int = 3) -> PlanSuggestions:
            captured["run_dir"] = run_dir
            return PlanSuggestions(plan_name="clean-plan", runs_analyzed=2)

        monkeypatch.setattr("maestro_cli.suggest.suggest_plan", _fake_suggest)
        rc = main([
            "check", str(plan_path), "--with-suggest",
            "--run-dir", str(custom_dir),
        ])
        out = capsys.readouterr().out
        assert rc == 0
        assert "== Suggestions ==" in out
        assert Path(str(captured["run_dir"])) == custom_dir

    def test_check_with_suggest_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan_path = _write(tmp_path, _CLEAN_PLAN_YAML)
        fake = PlanSuggestions(plan_name="clean-plan", runs_analyzed=7)
        monkeypatch.setattr(
            "maestro_cli.suggest.suggest_plan",
            MagicMock(return_value=fake),
        )
        rc = main(["check", str(plan_path), "--with-suggest", "--json"])
        out = capsys.readouterr().out
        report = json.loads(out)
        assert rc == 0
        assert "suggestions" in report
        assert report["suggestions"]["plan_name"] == "clean-plan"

    def test_check_with_suggest_failure_is_swallowed_text(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # suggest_plan raising must not break check; the except branch prints
        # a "suggest skipped" note in text mode.
        plan_path = _write(tmp_path, _CLEAN_PLAN_YAML)
        monkeypatch.setattr(
            "maestro_cli.suggest.suggest_plan",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        rc = main(["check", str(plan_path), "--with-suggest"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "suggest skipped" in out
        # suggestions stayed None → no Suggestions section
        assert "== Suggestions ==" not in out

    def test_check_with_suggest_failure_silent_in_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        # In JSON mode the except branch does NOT print the skip note.
        plan_path = _write(tmp_path, _CLEAN_PLAN_YAML)
        monkeypatch.setattr(
            "maestro_cli.suggest.suggest_plan",
            MagicMock(side_effect=RuntimeError("boom")),
        )
        rc = main(["check", str(plan_path), "--with-suggest", "--json"])
        out = capsys.readouterr().out
        report = json.loads(out)
        assert rc == 0
        assert "suggest skipped" not in out
        # suggestions failed → key absent from JSON report
        assert "suggestions" not in report


# ===========================================================================
# _cmd_skill — --dir extension + search --json
# ===========================================================================

class TestSkillDirAndJson:
    def test_skill_dir_extends_search_dirs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        extra = tmp_path / "extra-skills"
        extra.mkdir()
        captured: dict[str, object] = {}

        def _fake_discover(dirs: list[Path]) -> list[object]:
            captured["dirs"] = list(dirs)
            return []

        monkeypatch.setattr(
            "maestro_cli.skill_registry.discover_skills", _fake_discover,
        )
        monkeypatch.setattr(
            "maestro_cli.skill_registry.format_skills",
            MagicMock(return_value="no skills"),
        )
        rc = main(["skill", "--dir", str(extra)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "no skills" in out
        dirs = captured["dirs"]
        assert isinstance(dirs, list)
        # cwd/.claude/skills plus the extra dir we passed
        assert any(Path(d) == extra for d in dirs)

    def test_skill_search_json(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.skill_registry.discover_skills",
            MagicMock(return_value=["skill1"]),
        )
        monkeypatch.setattr(
            "maestro_cli.skill_registry.search_skills",
            MagicMock(return_value=["skill1"]),
        )
        monkeypatch.setattr(
            "maestro_cli.skill_registry.format_skills_json",
            MagicMock(return_value='[{"name":"skill1"}]'),
        )
        rc = main(["skill", "search", "-q", "test", "--json"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "skill1" in out


# ===========================================================================
# _cmd_ui — browser-opening thread + single-project-root print
# ===========================================================================

class TestCmdUiBrowser:
    def test_ui_opens_browser_and_single_root(
        self, monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        import time
        import webbrowser

        opened: dict[str, object] = {}

        # Run the daemon thread target synchronously so its body is covered.
        class _SyncThread:
            def __init__(self, *, target: object, daemon: bool = False) -> None:
                self._target = target
                self.daemon = daemon

            def start(self) -> None:
                assert callable(self._target)
                self._target()

        monkeypatch.setattr(threading, "Thread", _SyncThread)
        monkeypatch.setattr(time, "sleep", lambda *_a, **_k: None)
        monkeypatch.setattr(
            webbrowser, "open", lambda url: opened.__setitem__("url", url),
        )

        captured: dict[str, object] = {}

        def _fake_create_app(*, project_root: object = None, project_roots: object = None) -> object:
            captured["project_roots"] = project_roots
            return object()

        def _fake_uvicorn_run(app: object, **kwargs: object) -> None:
            captured["ran"] = True

        monkeypatch.setattr("maestro_cli.web.create_app", _fake_create_app)
        monkeypatch.setattr("uvicorn.run", _fake_uvicorn_run)

        # No --project-root → deduped_roots == [cwd] → single-root print branch.
        rc = main(["ui", "--host", "127.0.0.1", "--port", "9999"])
        out = capsys.readouterr().out
        assert rc == 0
        # browser thread ran and opened the expected URL
        assert opened.get("url") == "http://127.0.0.1:9999"
        # single-root print branch
        assert "project root:" in out
        assert "project roots:" not in out
        # roots list has exactly one entry (cwd)
        roots = captured["project_roots"]
        assert isinstance(roots, list)
        assert len(roots) == 1
        assert captured.get("ran") is True
