from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from maestro_cli.shell import _COMMANDS, _setup_readline


def _install_fake_readline(
    monkeypatch: pytest.MonkeyPatch,
) -> list[object]:
    """Install a fake `readline` module that captures the completer closure.

    Returns a list that will hold the completer function passed to
    `set_completer` once `_setup_readline()` runs.
    """
    captured_completer: list[object] = []

    fake_readline = types.ModuleType("readline")
    fake_readline.set_completer = lambda fn: captured_completer.append(fn)  # type: ignore[attr-defined]
    fake_readline.parse_and_bind = lambda _s: None  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "readline", fake_readline)
    return captured_completer


class TestCompletionIterdirOSError:
    """Drive the `except OSError: pass` branch inside the completion closure.

    When `Path.cwd().iterdir()` raises OSError the completer must swallow it and
    still return command-prefix matches.
    """

    def test_completer_handles_iterdir_oserror(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured_completer = _install_fake_readline(monkeypatch)

        _setup_readline()
        assert len(captured_completer) == 1
        completer = captured_completer[0]

        class _FakeCwd:
            def iterdir(self) -> object:
                raise OSError("simulated filesystem failure")

        # Patch the Path that shell.py references so cwd().iterdir() raises.
        monkeypatch.setattr(
            "maestro_cli.shell.Path.cwd",
            staticmethod(lambda: _FakeCwd()),
        )

        # Command matches still work because the OSError is caught and ignored.
        assert completer("/h", 0) == "/help"  # type: ignore[operator]
        assert completer("/h", 1) is None  # type: ignore[operator]

    def test_completer_oserror_does_not_break_command_iteration(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        captured_completer = _install_fake_readline(monkeypatch)

        _setup_readline()
        completer = captured_completer[0]

        class _FailingDir:
            def iterdir(self) -> object:
                raise OSError("permission denied")

        monkeypatch.setattr(
            "maestro_cli.shell.Path.cwd",
            staticmethod(lambda: _FailingDir()),
        )

        # "/" prefix matches every command even though file listing failed.
        all_matches = []
        for i in range(len(_COMMANDS) + 5):
            match = completer("/", i)  # type: ignore[operator]
            if match is None:
                break
            all_matches.append(match)
        assert all_matches == _COMMANDS

    def test_completer_yaml_listing_when_no_error(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Sanity check the non-error path still lists YAML files in cwd."""
        captured_completer = _install_fake_readline(monkeypatch)

        (tmp_path / "alpha.yaml").write_text("x", encoding="utf-8")
        (tmp_path / "beta.yml").write_text("x", encoding="utf-8")
        (tmp_path / "skip.txt").write_text("x", encoding="utf-8")
        monkeypatch.chdir(tmp_path)

        _setup_readline()
        completer = captured_completer[0]

        assert completer("al", 0) == "alpha.yaml"  # type: ignore[operator]
        assert completer("al", 1) is None  # type: ignore[operator]
