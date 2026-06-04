from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from maestro_cli.runners import _resolve_executable, _resolve_windows_bash


# ===========================================================================
# Coverage: _resolve_executable (runners.py L3484-3527)
#
# These tests drive the Windows-specific branches by forcing
# ``runners.os.name == "nt"`` and patching ``shutil.which`` / ``Path``
# regardless of the host OS, so they execute deterministically on any
# platform.
# ===========================================================================


class TestResolveExecutableNonWindows:
    def test_non_windows_returns_executable_verbatim(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # On non-Windows, the function short-circuits at L3498-3499 and never
        # touches shutil.which. We assert the verbatim passthrough.
        monkeypatch.setattr("maestro_cli.runners.os.name", "posix")

        def _boom(_name: str) -> str | None:  # pragma: no cover - must not run
            raise AssertionError("shutil.which should not be called on non-nt")

        monkeypatch.setattr("shutil.which", _boom)

        assert _resolve_executable("codex") == ["codex"]


class TestResolveExecutableWhichNone:
    def test_which_returns_none_falls_back_to_executable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drives L3502-3503: shutil.which() returns None -> return [executable].
        monkeypatch.setattr("maestro_cli.runners.os.name", "nt")
        monkeypatch.setattr("shutil.which", lambda _name: None)

        assert _resolve_executable("codex") == ["codex"]


class TestResolveExecutableNativeExe:
    def test_native_exe_returns_executable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drives L3505-3506: resolved is a real .exe (not .cmd/.bat) ->
        # return [executable]. This is the common case for `claude`.
        monkeypatch.setattr("maestro_cli.runners.os.name", "nt")
        monkeypatch.setattr(
            "shutil.which", lambda _name: "C:\\tools\\claude\\claude.exe"
        )

        assert _resolve_executable("claude") == ["claude"]


class TestResolveExecutableCmdParsedToNode:
    def test_cmd_wrapper_parsed_to_node_script(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Happy-path for the .cmd parser (L3508-3522): the npm wrapper
        # references %dp0%\...\script.js, the script exists, and a local
        # node.exe is found -> returns [local_node, script_abs].
        monkeypatch.setattr("maestro_cli.runners.os.name", "nt")
        monkeypatch.setattr(
            "shutil.which", lambda _name: "C:\\npm\\codex.cmd"
        )

        def _read_text(self: Any, *a: Any, **kw: Any) -> str:
            return '@echo off\nnode "%dp0%\\node_modules\\codex\\cli.js" %*'

        # node.exe in the wrapper dir, the .js script, and the .cmd all "exist".
        def _exists(self: Any) -> bool:
            text = str(self)
            return (
                text.endswith("node.exe")
                or text.endswith(".js")
                or text.endswith(".cmd")
            )

        monkeypatch.setattr("maestro_cli.runners.Path.read_text", _read_text)
        monkeypatch.setattr("maestro_cli.runners.Path.exists", _exists)

        result = _resolve_executable("codex")
        assert isinstance(result, list)
        assert len(result) == 2
        # Local node.exe was found, so the first element should reference it.
        assert result[0].endswith("node.exe")
        assert result[1].endswith("cli.js")


class TestResolveExecutableCmdFallback:
    def test_cmd_read_text_raises_uses_cmd_c_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drives the except path (L3523-3524) AND the final fallback (L3527):
        # read_text raises -> caught -> return ["cmd", "/c", executable].
        monkeypatch.setattr("maestro_cli.runners.os.name", "nt")
        monkeypatch.setattr(
            "shutil.which", lambda _name: "C:\\npm\\codex.cmd"
        )

        def _read_text_boom(self: Any, *a: Any, **kw: Any) -> str:
            raise OSError("cannot read wrapper")

        monkeypatch.setattr(
            "maestro_cli.runners.Path.read_text", _read_text_boom
        )

        assert _resolve_executable("codex") == ["cmd", "/c", "codex"]

    def test_cmd_no_js_match_uses_cmd_c_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drives L3526-3527: the wrapper has no "%dp0%\...js" pattern so the
        # regex does not match and we fall through to the cmd /c fallback.
        monkeypatch.setattr("maestro_cli.runners.os.name", "nt")
        monkeypatch.setattr(
            "shutil.which", lambda _name: "C:\\npm\\tool.bat"
        )

        def _read_text(self: Any, *a: Any, **kw: Any) -> str:
            return "@echo off\r\nrem nothing useful here\r\n"

        monkeypatch.setattr("maestro_cli.runners.Path.read_text", _read_text)

        assert _resolve_executable("tool") == ["cmd", "/c", "tool"]

    def test_cmd_js_match_but_script_missing_uses_cmd_c_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Regex matches but script_abs.exists() is False (L3519 False branch),
        # so we skip the node return and hit the cmd /c fallback at L3527.
        monkeypatch.setattr("maestro_cli.runners.os.name", "nt")
        monkeypatch.setattr(
            "shutil.which", lambda _name: "C:\\npm\\codex.cmd"
        )

        def _read_text(self: Any, *a: Any, **kw: Any) -> str:
            return 'node "%dp0%\\node_modules\\codex\\cli.js" %*'

        monkeypatch.setattr("maestro_cli.runners.Path.read_text", _read_text)
        # Nothing exists -> script_abs.exists() is False.
        monkeypatch.setattr(
            "maestro_cli.runners.Path.exists", lambda self: False
        )

        assert _resolve_executable("codex") == ["cmd", "/c", "codex"]


# ===========================================================================
# Coverage: _resolve_windows_bash (runners.py L3530-3552)
# ===========================================================================


class TestResolveWindowsBashNonWindows:
    def test_non_windows_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drives L3532-3533: os.name != "nt" -> return None.
        monkeypatch.setattr("maestro_cli.runners.os.name", "posix")
        assert _resolve_windows_bash() is None


class TestResolveWindowsBashGitBash:
    def test_which_returns_non_system32_bash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # When shutil.which("bash") returns a real Git Bash path (not a WSL
        # launcher in System32), it is returned directly (L3535-3540).
        monkeypatch.setattr("maestro_cli.runners.os.name", "nt")
        git_bash = "C:\\Program Files\\Git\\bin\\bash.exe"
        monkeypatch.setattr("shutil.which", lambda _name: git_bash)
        assert _resolve_windows_bash() == git_bash

    def test_system32_wsl_launcher_falls_through_to_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drives L3539 (system32 detection skips the early return) and
        # L3542-3550: the candidate loop returns the first existing Git Bash.
        monkeypatch.setattr("maestro_cli.runners.os.name", "nt")
        monkeypatch.setattr(
            "shutil.which", lambda _name: "C:\\Windows\\System32\\bash.exe"
        )

        chosen = str(Path("C:/Program Files/Git/bin/bash.exe"))

        def _exists(self: Any) -> bool:
            return str(self) == chosen

        monkeypatch.setattr("maestro_cli.runners.Path.exists", _exists)

        assert _resolve_windows_bash() == chosen

    def test_no_bash_on_path_falls_through_to_candidate(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # shutil.which returns None (so resolved is falsy), candidate loop runs
        # and a later candidate (usr/bin) is the one that exists (L3542-3550).
        monkeypatch.setattr("maestro_cli.runners.os.name", "nt")
        monkeypatch.setattr("shutil.which", lambda _name: None)

        chosen = str(Path("C:/Program Files/Git/usr/bin/bash.exe"))

        def _exists(self: Any) -> bool:
            return str(self) == chosen

        monkeypatch.setattr("maestro_cli.runners.Path.exists", _exists)

        assert _resolve_windows_bash() == chosen

    def test_no_candidate_exists_returns_resolved(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Drives L3552: which returned a System32 launcher, no candidates
        # exist, so the function returns `resolved` (the WSL launcher) as a
        # last resort rather than None.
        monkeypatch.setattr("maestro_cli.runners.os.name", "nt")
        wsl = "C:\\Windows\\System32\\bash.exe"
        monkeypatch.setattr("shutil.which", lambda _name: wsl)
        monkeypatch.setattr(
            "maestro_cli.runners.Path.exists", lambda self: False
        )

        assert _resolve_windows_bash() == wsl

    def test_which_none_and_no_candidate_returns_none(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # which returned None (resolved is None) and no candidate exists, so
        # the final `return resolved` (L3552) yields None.
        monkeypatch.setattr("maestro_cli.runners.os.name", "nt")
        monkeypatch.setattr("shutil.which", lambda _name: None)
        monkeypatch.setattr(
            "maestro_cli.runners.Path.exists", lambda self: False
        )

        assert _resolve_windows_bash() is None
