from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.doctor import (
    CheckResult,
    _check_cache_dir,
    _check_engine,
    _check_git,
    _check_knowledge_store,
    _check_plans_in_cwd,
    _check_plugin_discovery,
    _check_prior_runs,
    _check_pyyaml,
    _check_python_version,
    _check_run_dir_writable,
    _check_skill_registry,
    _check_web_deps,
    _engine_check_results,
    _plugin_warning_names,
    run_doctor,
)
from maestro_cli.plugins import DoctorProbe, EnginePlugin, PluginResolutionError


# ===========================================================================
# Python version check
# ===========================================================================


class TestCheckPythonVersion:
    def test_check_python_version_ok(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(sys, "version_info", (3, 11, 0))
        result = _check_python_version()
        assert result[0] == "python_version"
        assert result[2] == "ok"

    def test_check_python_version_too_old(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(sys, "version_info", (3, 10, 0))
        result = _check_python_version()
        assert result[0] == "python_version"
        assert result[2] == "fail"
        assert "3.11" in result[1]

    def test_check_python_version_exactly_311(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(sys, "version_info", (3, 11, 0))
        result = _check_python_version()
        assert result[2] == "ok"

    def test_check_python_version_310_fail_detail(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(sys, "version_info", (3, 10, 5))
        result = _check_python_version()
        assert "need" in result[1]


# ===========================================================================
# Engine availability check
# ===========================================================================


class TestCheckEngine:
    def test_check_engine_found(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        result = _check_engine("claude")
        assert result[0] == "engine_claude"
        assert result[2] == "ok"
        assert "/usr/bin/claude" in result[1]

    def test_check_engine_not_found(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda name: None)
        result = _check_engine("codex")
        assert result[0] == "engine_codex"
        assert result[2] == "warn"
        assert "not found" in result[1]

    def test_check_engine_gemini_found(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/local/bin/{name}",
        )
        result = _check_engine("gemini")
        assert result[0] == "engine_gemini"
        assert result[2] == "ok"
        assert "gemini" in result[1]

    def test_check_engine_result_is_tuple_of_three(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda name: None)
        result = _check_engine("codex")
        assert isinstance(result, tuple)
        assert len(result) == 3


# ===========================================================================
# Git availability check
# ===========================================================================


class TestCheckGit:
    def test_check_git_available(self, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: "/usr/bin/git" if name == "git" else None,
        )
        result = _check_git()
        assert result[0] == "git"
        assert result[2] == "ok"
        assert "found" in result[1]

    def test_check_git_not_found(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda name: None)
        result = _check_git()
        assert result[0] == "git"
        assert result[2] == "warn"
        assert "not found" in result[1]


# ===========================================================================
# run_doctor integration
# ===========================================================================


class TestRunDoctor:
    @staticmethod
    def _plugin(name: str) -> EnginePlugin:
        executable = "qwen-code" if name == "qwen" else ("custom-engine" if name == "custom" else name)
        return EnginePlugin(
            name=name,
            build_command=lambda ctx: ([name], False),
            doctor_probe=DoctorProbe(executable=executable),
        )

    def test_run_doctor_returns_list(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        results = run_doctor(run_dir=str(tmp_path / "runs"), json_output=False)
        assert isinstance(results, list)
        assert len(results) > 0

    def test_run_doctor_returns_list_of_tuples(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        results = run_doctor(run_dir=str(tmp_path / "runs"), json_output=False)
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 3
            name, detail, status = item
            assert isinstance(name, str)
            assert isinstance(detail, str)
            assert status in ("ok", "warn", "fail", "info")

    def test_run_doctor_json_output(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        run_doctor(run_dir=str(tmp_path / "runs"), json_output=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert isinstance(data, list)
        assert len(data) > 0
        for item in data:
            assert "check" in item
            assert "detail" in item
            assert "status" in item
            assert item["status"] in ("ok", "warn", "fail", "info")

    def test_run_doctor_text_output(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        run_doctor(run_dir=str(tmp_path / "runs"), json_output=False)
        captured = capsys.readouterr()
        assert "[maestro] doctor:" in captured.out
        assert "checks" in captured.out

    def test_run_doctor_text_has_check_names(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        run_doctor(run_dir=str(tmp_path / "runs"), json_output=False)
        captured = capsys.readouterr()
        assert "python_version" in captured.out

    def test_run_doctor_json_has_engine_checks(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda name: None)
        run_doctor(run_dir=str(tmp_path / "runs"), json_output=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        check_names = [item["check"] for item in data]
        assert "engine_codex" in check_names
        assert "engine_claude" in check_names
        assert "engine_gemini" in check_names
        assert "git" in check_names

    def test_run_doctor_reports_no_custom_plugin_discovery(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        builtin_names = ["codex", "claude", "gemini", "copilot", "qwen", "ollama"]
        monkeypatch.setattr("maestro_cli.doctor.supported_engine_names", lambda: builtin_names)
        monkeypatch.setattr("maestro_cli.doctor.get_engine_plugin", self._plugin)
        monkeypatch.setattr("maestro_cli.doctor.discover_engine_plugins", lambda: {})
        monkeypatch.setattr("maestro_cli.doctor.plugin_discovery_errors", lambda: {})
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )

        run_doctor(run_dir=str(tmp_path / "runs"), json_output=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        discovery = next(item for item in data if item["check"] == "engine_plugins")

        assert discovery["status"] == "ok"
        assert "no custom engine plugins discovered" in discovery["detail"]
        assert "maestro_cli.engines" in discovery["detail"]

    def test_run_doctor_includes_custom_plugin_probe(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.supported_engine_names",
            lambda: ["codex", "claude", "gemini", "copilot", "qwen", "ollama", "custom"],
        )
        monkeypatch.setattr("maestro_cli.doctor.get_engine_plugin", self._plugin)
        monkeypatch.setattr(
            "maestro_cli.doctor.discover_engine_plugins",
            lambda: {"custom": self._plugin("custom")},
        )
        monkeypatch.setattr("maestro_cli.doctor.plugin_discovery_errors", lambda: {})
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )

        run_doctor(run_dir=str(tmp_path / "runs"), json_output=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        check_names = [item["check"] for item in data]
        discovery = next(item for item in data if item["check"] == "engine_plugins")

        assert "engine_custom" in check_names
        assert discovery["status"] == "ok"
        assert "custom" in discovery["detail"]

    def test_run_doctor_text_output_surfaces_plugin_discovery(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.supported_engine_names",
            lambda: ["codex", "claude", "gemini", "copilot", "qwen", "ollama", "custom"],
        )
        monkeypatch.setattr("maestro_cli.doctor.get_engine_plugin", self._plugin)
        monkeypatch.setattr(
            "maestro_cli.doctor.discover_engine_plugins",
            lambda: {"custom": self._plugin("custom")},
        )
        monkeypatch.setattr("maestro_cli.doctor.plugin_discovery_errors", lambda: {})
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )

        run_doctor(run_dir=str(tmp_path / "runs"), json_output=False)
        captured = capsys.readouterr()

        assert "engine_plugins" in captured.out
        assert "custom engine plugin(s)" in captured.out
        assert "custom" in captured.out

    def test_run_doctor_surfaces_plugin_warning_names(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        builtin_names = ["codex", "claude", "gemini", "copilot", "qwen", "ollama"]
        monkeypatch.setattr("maestro_cli.doctor.supported_engine_names", lambda: builtin_names)
        monkeypatch.setattr("maestro_cli.doctor.get_engine_plugin", self._plugin)
        monkeypatch.setattr("maestro_cli.doctor.discover_engine_plugins", lambda: {})
        monkeypatch.setattr(
            "maestro_cli.doctor.plugin_discovery_errors",
            lambda: {"custom": "bad metadata", "other": "duplicate"},
        )
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )

        run_doctor(run_dir=str(tmp_path / "runs"), json_output=True)
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        discovery = next(item for item in data if item["check"] == "engine_plugins")

        assert discovery["status"] == "warn"
        assert "plugin warning(s): custom, other" in discovery["detail"]


# ===========================================================================
# CheckResult string-key access
# ===========================================================================


class TestCheckResult:
    def test_string_key_access(self) -> None:
        r = CheckResult("my_check", "all good", "ok")
        assert r["check"] == "my_check"
        assert r["detail"] == "all good"
        assert r["status"] == "ok"

    def test_int_key_access(self) -> None:
        r = CheckResult("my_check", "detail here", "warn")
        assert r[0] == "my_check"
        assert r[1] == "detail here"
        assert r[2] == "warn"

    def test_slice_access(self) -> None:
        r = CheckResult("a", "b", "ok")
        assert r[0:2] == ("a", "b")


# ===========================================================================
# _check_pyyaml
# ===========================================================================


class TestCheckPyYaml:
    def test_pyyaml_installed(self) -> None:
        result = _check_pyyaml()
        assert result[0] == "pyyaml"
        assert result[2] == "ok"

    def test_pyyaml_not_installed(self, monkeypatch: Any) -> None:
        import builtins
        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "yaml":
                raise ImportError("no yaml")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = _check_pyyaml()
        assert result[0] == "pyyaml"
        assert result[2] == "fail"
        assert "not installed" in result[1]


# ===========================================================================
# _check_engine edge cases
# ===========================================================================


class TestCheckEngineEdgeCases:
    def test_custom_check_name(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda name: "/usr/bin/qwen-code")
        result = _check_engine("qwen", executable="qwen-code", check_name="engine_qwen")
        assert result[0] == "engine_qwen"
        assert result[2] == "ok"

    def test_install_hint_shown(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda name: None)
        result = _check_engine("custom", install_hint="pip install custom-engine")
        assert "pip install custom-engine" in result[1]
        assert result[2] == "warn"

    def test_no_install_hint(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda name: None)
        result = _check_engine("foo")
        assert "not found" in result[1]
        assert result[2] == "warn"


# ===========================================================================
# _check_run_dir_writable
# ===========================================================================


class TestCheckRunDirWritable:
    def test_writable_dir(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs"
        result = _check_run_dir_writable(str(run_dir))
        assert result[0] == "run_dir_writable"
        assert result[2] == "ok"
        assert "writable" in result[1]

    def test_unwritable_dir(self, monkeypatch: Any) -> None:
        """Simulate an unwritable directory via OSError."""
        import maestro_cli.doctor as doc
        original_mkdir = Path.mkdir

        def failing_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
            if ".maestro_doctor_probe" not in str(self):
                raise OSError("Permission denied")

        monkeypatch.setattr(Path, "mkdir", failing_mkdir)
        result = _check_run_dir_writable("/nonexistent/readonly/path")
        assert result[0] == "run_dir_writable"
        assert result[2] == "fail"
        assert "cannot write" in result[1]


# ===========================================================================
# _check_web_deps
# ===========================================================================


class TestCheckWebDeps:
    def test_web_deps_installed(self) -> None:
        # Both fastapi and uvicorn are likely installed in the dev env
        # but this test is valid either way
        result = _check_web_deps()
        assert result[0] == "web_deps"
        assert result[2] in ("ok", "warn")

    def test_web_deps_missing(self, monkeypatch: Any) -> None:
        import builtins
        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in ("fastapi", "uvicorn"):
                raise ImportError(f"no {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        result = _check_web_deps()
        assert result[0] == "web_deps"
        assert result[2] == "warn"
        assert "fastapi" in result[1]
        assert "uvicorn" in result[1]


# ===========================================================================
# _check_plugin_discovery edge cases
# ===========================================================================


class TestCheckPluginDiscovery:
    def test_entry_points_error(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.doctor.discover_engine_plugins", lambda: {})
        monkeypatch.setattr(
            "maestro_cli.doctor.plugin_discovery_errors",
            lambda: {"__entry_points__": "importlib.metadata unavailable"},
        )
        result = _check_plugin_discovery()
        assert result[0] == "engine_plugins"
        assert result[2] == "warn"
        assert "importlib.metadata" in result[1]

    def test_no_plugins_no_errors(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.doctor.discover_engine_plugins", lambda: {})
        monkeypatch.setattr("maestro_cli.doctor.plugin_discovery_errors", lambda: {})
        result = _check_plugin_discovery()
        assert result[2] == "ok"
        assert "no custom engine plugins" in result[1]


# ===========================================================================
# _plugin_warning_names
# ===========================================================================


class TestPluginWarningNames:
    def test_excludes_entry_points(self) -> None:
        errors = {"__entry_points__": "err", "foo": "bad", "bar": "bad"}
        names = _plugin_warning_names(errors)
        assert "__entry_points__" not in names
        assert names == ["bar", "foo"]

    def test_empty_errors(self) -> None:
        assert _plugin_warning_names({}) == []


# ===========================================================================
# _engine_check_results edge cases
# ===========================================================================


class TestEngineCheckResults:
    @staticmethod
    def _plugin(name: str) -> EnginePlugin:
        return EnginePlugin(
            name=name,
            build_command=lambda ctx: ([name], False),
            doctor_probe=DoctorProbe(executable=name),
        )

    @staticmethod
    def _plugin_no_probe(name: str) -> EnginePlugin:
        return EnginePlugin(
            name=name,
            build_command=lambda ctx: ([name], False),
            doctor_probe=None,
        )

    def test_plugin_resolution_error_handled(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.doctor.discover_engine_plugins", lambda: {})
        monkeypatch.setattr("maestro_cli.doctor.plugin_discovery_errors", lambda: {})
        monkeypatch.setattr("maestro_cli.doctor.supported_engine_names", lambda: ["codex"])

        def failing_get(name: str) -> EnginePlugin:
            raise PluginResolutionError(f"cannot load {name}")

        monkeypatch.setattr("maestro_cli.doctor.get_engine_plugin", failing_get)
        monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda name: None)

        results = _engine_check_results()
        # Should have a warn for the resolution error
        warn_results = [r for r in results if r[2] == "warn" and "codex" in r[1]]
        assert len(warn_results) >= 1

    def test_plugin_without_doctor_probe_skipped(self, monkeypatch: Any) -> None:
        monkeypatch.setattr("maestro_cli.doctor.discover_engine_plugins", lambda: {})
        monkeypatch.setattr("maestro_cli.doctor.plugin_discovery_errors", lambda: {})
        monkeypatch.setattr("maestro_cli.doctor.supported_engine_names", lambda: ["noprobe"])
        monkeypatch.setattr("maestro_cli.doctor.get_engine_plugin", self._plugin_no_probe)
        monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda name: None)

        results = _engine_check_results()
        engine_results = [r for r in results if "engine_noprobe" in r[0]]
        # No engine check result since probe is None
        assert len(engine_results) == 0

    def test_optional_deps_checked(self, monkeypatch: Any) -> None:
        """_engine_check_results includes optional dependency checks."""
        monkeypatch.setattr("maestro_cli.doctor.discover_engine_plugins", lambda: {})
        monkeypatch.setattr("maestro_cli.doctor.plugin_discovery_errors", lambda: {})
        monkeypatch.setattr("maestro_cli.doctor.supported_engine_names", lambda: [])
        monkeypatch.setattr("maestro_cli.doctor.shutil.which", lambda name: None)

        results = _engine_check_results()
        check_names = [r[0] for r in results]
        # Should include optional dependency checks
        assert "tui_dependency" in check_names
        assert "live_dependency" in check_names
        assert "mcp_protocol" in check_names
        assert "otel_protocol" in check_names


# ===========================================================================
# _check_cache_dir
# ===========================================================================


class TestCheckCacheDir:
    def test_writable_cache_dir(self, monkeypatch: Any, tmp_path: Path) -> None:
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        result = _check_cache_dir()
        assert result[0] == "cache_dir"
        assert result[2] == "ok"
        assert "writable" in result[1]

    def test_unwritable_cache_dir(self, monkeypatch: Any) -> None:
        original_mkdir = Path.mkdir

        def failing_mkdir(self: Path, *args: Any, **kwargs: Any) -> None:
            if ".maestro-cache" in str(self):
                raise OSError("Permission denied")
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", failing_mkdir)
        result = _check_cache_dir()
        assert result[0] == "cache_dir"
        assert result[2] == "warn"
        assert "cannot write" in result[1]


# ===========================================================================
# _check_knowledge_store
# ===========================================================================


class TestCheckKnowledgeStore:
    def test_no_knowledge_dir(self, monkeypatch: Any, tmp_path: Path) -> None:
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        result = _check_knowledge_store()
        assert result[0] == "knowledge_store"
        assert result[2] == "info"
        assert "no knowledge store" in result[1]

    def test_knowledge_dir_with_files(self, monkeypatch: Any, tmp_path: Path) -> None:
        knowledge_dir = tmp_path / ".maestro-cache" / "knowledge"
        knowledge_dir.mkdir(parents=True)
        (knowledge_dir / "plan1.jsonl").write_text("{}\n", encoding="utf-8")
        (knowledge_dir / "plan2.jsonl").write_text("{}\n", encoding="utf-8")

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        result = _check_knowledge_store()
        assert result[0] == "knowledge_store"
        assert result[2] == "ok"
        assert "2 knowledge file(s)" in result[1]

    def test_knowledge_dir_empty(self, monkeypatch: Any, tmp_path: Path) -> None:
        knowledge_dir = tmp_path / ".maestro-cache" / "knowledge"
        knowledge_dir.mkdir(parents=True)

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        result = _check_knowledge_store()
        assert result[0] == "knowledge_store"
        assert result[2] == "ok"
        assert "0 knowledge file(s)" in result[1]


# ===========================================================================
# _check_skill_registry
# ===========================================================================


class TestCheckSkillRegistry:
    def test_no_skills_dir(self, monkeypatch: Any, tmp_path: Path) -> None:
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        result = _check_skill_registry()
        assert result[0] == "skill_registry"
        assert result[2] == "info"

    def test_skills_dir_with_skills(self, monkeypatch: Any, tmp_path: Path) -> None:
        skills_dir = tmp_path / ".claude" / "skills"
        skill1 = skills_dir / "my-skill"
        skill1.mkdir(parents=True)
        (skill1 / "SKILL.md").write_text("# Skill", encoding="utf-8")

        skill2 = skills_dir / "other-skill"
        skill2.mkdir(parents=True)
        (skill2 / "SKILL.md").write_text("# Skill2", encoding="utf-8")

        # Dir without SKILL.md should not count
        (skills_dir / "not-a-skill").mkdir()

        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        result = _check_skill_registry()
        assert result[0] == "skill_registry"
        assert result[2] == "ok"
        assert "2 skill(s)" in result[1]


# ===========================================================================
# _check_plans_in_cwd
# ===========================================================================


class TestCheckPlansInCwd:
    def test_no_plans(self, monkeypatch: Any, tmp_path: Path) -> None:
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        result = _check_plans_in_cwd()
        assert result[0] == "plans_in_cwd"
        assert result[2] == "info"
        assert "no Maestro plans" in result[1]

    def test_yaml_plan_found(self, monkeypatch: Any, tmp_path: Path) -> None:
        plan = tmp_path / "myplan.yaml"
        plan.write_text(
            "version: 1\nname: test\ntasks:\n  - id: t1\n    command: echo ok\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        result = _check_plans_in_cwd()
        assert result[0] == "plans_in_cwd"
        assert result[2] == "ok"
        assert "1 plan(s)" in result[1]

    def test_yml_plan_found(self, monkeypatch: Any, tmp_path: Path) -> None:
        plan = tmp_path / "myplan.yml"
        plan.write_text(
            "version: 1\nname: test\ntasks:\n  - id: t1\n    command: echo ok\n",
            encoding="utf-8",
        )
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        result = _check_plans_in_cwd()
        assert result[2] == "ok"

    def test_non_plan_yaml_ignored(self, monkeypatch: Any, tmp_path: Path) -> None:
        """YAML without version: 1 and tasks is not counted."""
        (tmp_path / "config.yaml").write_text("key: value\n", encoding="utf-8")
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        result = _check_plans_in_cwd()
        assert result[2] == "info"

    def test_invalid_yaml_ignored(self, monkeypatch: Any, tmp_path: Path) -> None:
        """Invalid YAML files don't crash the check."""
        (tmp_path / "bad.yaml").write_text("{{invalid yaml", encoding="utf-8")
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        result = _check_plans_in_cwd()
        assert result[2] == "info"


# ===========================================================================
# _check_prior_runs
# ===========================================================================


class TestCheckPriorRuns:
    def test_no_runs_dir(self, tmp_path: Path) -> None:
        result = _check_prior_runs(str(tmp_path / "nonexistent"))
        assert result[0] == "prior_runs"
        assert result[2] == "info"
        assert "no runs directory" in result[1]

    def test_runs_dir_empty(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs"
        run_dir.mkdir()
        result = _check_prior_runs(str(run_dir))
        assert result[2] == "info"
        assert "no completed runs" in result[1]

    def test_runs_dir_with_completed_runs(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "runs"
        run1 = run_dir / "20260101_plan-a"
        run1.mkdir(parents=True)
        (run1 / "run_manifest.json").write_text("{}", encoding="utf-8")

        run2 = run_dir / "20260102_plan-b"
        run2.mkdir(parents=True)
        (run2 / "run_manifest.json").write_text("{}", encoding="utf-8")

        result = _check_prior_runs(str(run_dir))
        assert result[0] == "prior_runs"
        assert result[2] == "ok"
        assert "2 completed run(s)" in result[1]

    def test_runs_dir_with_incomplete_run(self, tmp_path: Path) -> None:
        """Dirs without run_manifest.json are not counted."""
        run_dir = tmp_path / "runs"
        run1 = run_dir / "20260101_plan-a"
        run1.mkdir(parents=True)
        # No manifest file

        result = _check_prior_runs(str(run_dir))
        assert result[2] == "info"
        assert "no completed runs" in result[1]


# ===========================================================================
# run_doctor full mode
# ===========================================================================


class TestRunDoctorFull:
    @staticmethod
    def _plugin(name: str) -> EnginePlugin:
        return EnginePlugin(
            name=name,
            build_command=lambda ctx: ([name], False),
            doctor_probe=DoctorProbe(executable=name),
        )

    def test_full_mode_includes_extra_checks(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        results = run_doctor(
            run_dir=str(tmp_path / "runs"),
            json_output=True,
            full=True,
        )
        check_names = [r[0] for r in results]
        assert "cache_dir" in check_names
        assert "knowledge_store" in check_names
        assert "skill_registry" in check_names
        assert "plans_in_cwd" in check_names
        assert "prior_runs" in check_names

    def test_non_full_mode_excludes_extra_checks(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        results = run_doctor(
            run_dir=str(tmp_path / "runs"),
            json_output=True,
            full=False,
        )
        check_names = [r[0] for r in results]
        assert "cache_dir" not in check_names
        assert "knowledge_store" not in check_names

    def test_full_json_output_structure(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        monkeypatch.setattr(Path, "cwd", lambda: tmp_path)
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        run_doctor(
            run_dir=str(tmp_path / "runs"),
            json_output=True,
            full=True,
        )
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        for item in data:
            assert "check" in item
            assert "detail" in item
            assert "status" in item

    def test_text_output_summary_line(
        self, tmp_path: Path, monkeypatch: Any, capsys: Any
    ) -> None:
        monkeypatch.setattr(
            "maestro_cli.doctor.shutil.which",
            lambda name: f"/usr/bin/{name}",
        )
        run_doctor(run_dir=str(tmp_path / "runs"), json_output=False)
        captured = capsys.readouterr()
        assert "[maestro] doctor:" in captured.out
        assert "failed" in captured.out
        assert "warning(s)" in captured.out
