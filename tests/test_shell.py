from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.shell import _COMMANDS, _cmd_run_in_shell, _dispatch_command, run_shell, ShellState


class TestShellState:
    def test_default_state(self) -> None:
        state = ShellState()
        assert state.active_plan is None
        assert state.last_run_dir is None
        assert state.history == []

    def test_with_plan(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        state = ShellState(active_plan=plan)
        assert state.active_plan == plan


class TestCommands:
    def test_quit_returns_false(self) -> None:
        state = ShellState()
        assert _dispatch_command("/quit", state) is False

    def test_help_returns_true(self) -> None:
        state = ShellState()
        assert _dispatch_command("/help", state) is True

    def test_unknown_command_returns_true(self) -> None:
        state = ShellState()
        assert _dispatch_command("/foobar", state) is True

    def test_plan_command_sets_active_plan(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        plan = tmp_path / "plan.yaml"
        plan.write_text("tasks: []\n", encoding="utf-8")

        result = _dispatch_command(f"/plan {plan}", state)
        out = capsys.readouterr().out

        assert result is True
        assert state.active_plan == plan
        assert "Active plan set" in out

    def test_plan_command_invalid_path(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        missing = tmp_path / "missing.yaml"

        result = _dispatch_command(f"/plan {missing}", state)
        out = capsys.readouterr().out

        assert result is True
        assert state.active_plan is None
        assert "Plan not found" in out

    def test_validate_without_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        result = _dispatch_command("/validate", state)
        out = capsys.readouterr().out

        assert result is True
        assert "No active plan" in out

    def test_run_without_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        result = _dispatch_command("/run", state)
        out = capsys.readouterr().out

        assert result is True
        assert "No active plan" in out

    def test_suggest_without_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        result = _dispatch_command("/suggest", state)
        out = capsys.readouterr().out

        assert result is True
        assert "No active plan" in out

    def test_run_with_plan_updates_last_run_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("tasks: []\n", encoding="utf-8")
        expected_run = tmp_path / ".maestro-runs" / "latest_run"
        state = ShellState(active_plan=plan)

        monkeypatch.setattr(
            "maestro_cli.shell._cmd_run_in_shell",
            lambda _state, dry_run=False: expected_run,
        )

        result = _dispatch_command("/run", state)

        assert result is True
        assert state.last_run_dir == expected_run

    def test_last_prints_stored_run_dir(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        run_dir = tmp_path / ".maestro-runs" / "latest_run"
        state = ShellState(last_run_dir=run_dir)

        result = _dispatch_command("/last", state)
        out = capsys.readouterr().out

        assert result is True
        assert str(run_dir) in out


class TestShellRunIntegration:
    def test_cmd_run_in_shell_passes_cli_namespace(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("tasks: []\n", encoding="utf-8")
        state = ShellState(active_plan=plan)
        expected_run = tmp_path / ".maestro-runs" / "latest_run"
        captured: dict[str, object] = {}

        def _fake_cmd_run(args) -> int:
            captured.update(vars(args))
            return 0

        monkeypatch.setattr("maestro_cli.cli._cmd_run", _fake_cmd_run)
        monkeypatch.setattr("maestro_cli.loader.load_plan", lambda _path: object())
        monkeypatch.setattr(
            "maestro_cli.cli._find_latest_run",
            lambda _plan, run_dir=None: expected_run,
        )

        run_path = _cmd_run_in_shell(state, dry_run=True)

        assert captured["plan"] == [str(plan)]
        assert captured["parallel"] is False
        assert captured["cache_dir"] is None
        assert captured["execution_profile"] == "plan"
        assert captured["dry_run"] is True
        assert run_path == expected_run


class TestCommandRegistry:
    def test_all_commands_start_with_slash(self) -> None:
        assert all(command.startswith("/") for command in _COMMANDS)

    def test_expected_commands_present(self) -> None:
        expected = {"/run", "/quit", "/help", "/suggest", "/plan"}
        assert expected.issubset(set(_COMMANDS))


# ──────────────────────────── Additional Tests ────────────────────────────────

from maestro_cli.shell import (
    _print_help,
    _cmd_validate_in_shell,
    _cmd_suggest_in_shell,
    _cmd_status_in_shell,
    _cmd_explain_in_shell,
    _setup_readline,
)


class TestDispatchCommandExtended:
    """Extended coverage for _dispatch_command across all slash commands."""

    def test_quit_adds_to_history(self) -> None:
        state = ShellState()
        _dispatch_command("/quit", state)
        assert "/quit" in state.history

    def test_help_with_specific_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        _dispatch_command("/help /run", state)
        out = capsys.readouterr().out
        assert "/run" in out
        assert "--dry-run" in out

    def test_help_with_unknown_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        _dispatch_command("/help /nonexistent", state)
        out = capsys.readouterr().out
        assert "Unknown command for help" in out

    def test_plan_command_no_args(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        result = _dispatch_command("/plan", state)
        out = capsys.readouterr().out
        assert result is True
        assert "Usage:" in out

    def test_last_no_runs(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        result = _dispatch_command("/last", state)
        out = capsys.readouterr().out
        assert result is True
        assert "No runs yet" in out

    def test_unknown_command_message(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        result = _dispatch_command("/banana", state)
        out = capsys.readouterr().out
        assert result is True
        assert "Unknown command: /banana" in out
        assert "/help" in out

    def test_run_unknown_option(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState(active_plan=Path("dummy.yaml"))
        result = _dispatch_command("/run --unknown-flag", state)
        out = capsys.readouterr().out
        assert result is True
        assert "Unknown option for /run" in out

    def test_run_dry_run_flag(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("tasks: []\n", encoding="utf-8")
        state = ShellState(active_plan=plan)
        captured_dry_run: list[bool] = []

        def _fake_run(_state: ShellState, dry_run: bool = False) -> Path:
            captured_dry_run.append(dry_run)
            return tmp_path / "run-dir"

        monkeypatch.setattr("maestro_cli.shell._cmd_run_in_shell", _fake_run)

        _dispatch_command("/run --dry-run", state)
        assert captured_dry_run == [True]

    def test_run_returns_none_does_not_set_last_run_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("tasks: []\n", encoding="utf-8")
        state = ShellState(active_plan=plan)

        monkeypatch.setattr(
            "maestro_cli.shell._cmd_run_in_shell",
            lambda _state, dry_run=False: None,
        )

        _dispatch_command("/run", state)
        assert state.last_run_dir is None

    def test_validate_dispatches(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("version: 1\nname: t\ntasks:\n  - id: a\n    command: echo\n", encoding="utf-8")
        state = ShellState(active_plan=plan)
        called: list[str] = []

        monkeypatch.setattr(
            "maestro_cli.shell._cmd_validate_in_shell",
            lambda s: called.append("validate"),
        )

        result = _dispatch_command("/validate", state)
        assert result is True
        assert called == ["validate"]

    def test_suggest_dispatches(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = ShellState(active_plan=Path("plan.yaml"))
        called: list[str] = []

        monkeypatch.setattr(
            "maestro_cli.shell._cmd_suggest_in_shell",
            lambda s: called.append("suggest"),
        )

        result = _dispatch_command("/suggest", state)
        assert result is True
        assert called == ["suggest"]

    def test_status_dispatches(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = ShellState(active_plan=Path("plan.yaml"))
        called: list[str] = []

        monkeypatch.setattr(
            "maestro_cli.shell._cmd_status_in_shell",
            lambda s: called.append("status"),
        )

        result = _dispatch_command("/status", state)
        assert result is True
        assert called == ["status"]

    def test_explain_dispatches(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        state = ShellState(active_plan=Path("plan.yaml"))
        called: list[str] = []

        monkeypatch.setattr(
            "maestro_cli.shell._cmd_explain_in_shell",
            lambda s: called.append("explain"),
        )

        result = _dispatch_command("/explain", state)
        assert result is True
        assert called == ["explain"]

    def test_history_accumulates(self) -> None:
        state = ShellState()
        _dispatch_command("/help", state)
        _dispatch_command("/last", state)
        _dispatch_command("/help /run", state)
        assert state.history == ["/help", "/last", "/help /run"]

    def test_plan_sets_expanduser_path(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("tasks: []\n", encoding="utf-8")
        state = ShellState()
        # Pass the absolute path to avoid home-relative expansion issues
        _dispatch_command(f"/plan {plan}", state)
        assert state.active_plan == plan


class TestPrintHelp:
    """Tests for _print_help output."""

    def test_print_help_all_commands(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_help()
        out = capsys.readouterr().out
        assert "Available commands:" in out
        for cmd in _COMMANDS:
            assert cmd in out

    def test_print_help_specific_known_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_help("/plan")
        out = capsys.readouterr().out
        assert "/plan" in out
        assert "<path>" in out

    def test_print_help_specific_unknown_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_help("/doesnotexist")
        out = capsys.readouterr().out
        assert "Unknown command for help" in out
        assert "/doesnotexist" in out

    def test_print_help_quit(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_help("/quit")
        out = capsys.readouterr().out
        assert "/quit" in out
        assert "Exit shell" in out

    def test_print_help_validate(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_help("/validate")
        out = capsys.readouterr().out
        assert "/validate" in out
        assert "Validate" in out


class TestShellCommandHelpers:
    """Tests for _cmd_*_in_shell helper functions without plan."""

    def test_validate_in_shell_no_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        _cmd_validate_in_shell(state)
        out = capsys.readouterr().out
        assert "No active plan" in out

    def test_suggest_in_shell_no_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        _cmd_suggest_in_shell(state)
        out = capsys.readouterr().out
        assert "No active plan" in out

    def test_status_in_shell_no_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        _cmd_status_in_shell(state)
        out = capsys.readouterr().out
        assert "No active plan" in out

    def test_explain_in_shell_no_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        _cmd_explain_in_shell(state)
        out = capsys.readouterr().out
        assert "No active plan" in out

    def test_cmd_run_in_shell_no_plan(self, capsys: pytest.CaptureFixture[str]) -> None:
        state = ShellState()
        result = _cmd_run_in_shell(state)
        out = capsys.readouterr().out
        assert result is None
        assert "No active plan" in out

    def test_cmd_run_in_shell_nonzero_rc(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("version: 1\nname: t\ntasks:\n  - id: a\n    command: echo\n", encoding="utf-8")
        state = ShellState(active_plan=plan)

        monkeypatch.setattr("maestro_cli.cli._cmd_run", lambda args: 1)

        result = _cmd_run_in_shell(state)
        assert result is None


class TestShellStateExtended:
    """Extended ShellState dataclass tests."""

    def test_state_last_run_dir(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "run"
        state = ShellState(last_run_dir=run_dir)
        assert state.last_run_dir == run_dir
        assert state.active_plan is None

    def test_state_history_mutable_default_independent(self) -> None:
        s1 = ShellState()
        s2 = ShellState()
        s1.history.append("cmd1")
        assert s2.history == []

    def test_state_with_all_fields(self, tmp_path: Path) -> None:
        plan = tmp_path / "plan.yaml"
        run_dir = tmp_path / "run"
        state = ShellState(active_plan=plan, last_run_dir=run_dir, history=["a", "b"])
        assert state.active_plan == plan
        assert state.last_run_dir == run_dir
        assert state.history == ["a", "b"]


class TestRunShell:
    """Tests for the run_shell() main loop."""

    def test_run_shell_eof_exit(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr("maestro_cli.shell._setup_readline", lambda: None)
        monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError))

        rc = run_shell()
        assert rc == 0

    def test_run_shell_keyboard_interrupt(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr("maestro_cli.shell._setup_readline", lambda: None)
        monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(KeyboardInterrupt))

        rc = run_shell()
        assert rc == 0

    def test_run_shell_quit_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inputs = iter(["/quit"])
        monkeypatch.setattr("maestro_cli.shell._setup_readline", lambda: None)
        monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

        rc = run_shell()
        assert rc == 0

    def test_run_shell_empty_lines_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inputs = iter(["", "   ", "/quit"])
        monkeypatch.setattr("maestro_cli.shell._setup_readline", lambda: None)
        monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

        rc = run_shell()
        assert rc == 0

    def test_run_shell_with_initial_plan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text("tasks: []\n", encoding="utf-8")
        inputs = iter(["/quit"])
        monkeypatch.setattr("maestro_cli.shell._setup_readline", lambda: None)
        monkeypatch.setattr("builtins.input", lambda _prompt: next(inputs))

        rc = run_shell(plan_path=plan)
        out = capsys.readouterr().out
        assert rc == 0
        assert "Active plan:" in out

    def test_run_shell_prints_welcome(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        monkeypatch.setattr("maestro_cli.shell._setup_readline", lambda: None)
        monkeypatch.setattr("builtins.input", lambda _prompt: (_ for _ in ()).throw(EOFError))

        run_shell()
        out = capsys.readouterr().out
        assert "Interactive shell" in out
        assert "/help" in out


class TestSetupReadline:
    """Tests for _setup_readline with mocked readline."""

    def test_setup_readline_no_readline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When neither readline nor pyreadline3 is available, no error."""
        import importlib

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def _mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name in ("readline", "pyreadline3"):
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr("builtins.__import__", _mock_import)
        # Should not raise
        _setup_readline()


class TestTabCompletion:
    """Test the completion logic embedded in _setup_readline."""

    def test_complete_function_matches_commands(self) -> None:
        """Verify the _complete closure matches slash commands."""
        # We replicate the completion logic to test it directly
        from maestro_cli.shell import _COMMANDS

        def _complete(text: str, state: int) -> str | None:
            matches = [cmd for cmd in _COMMANDS if cmd.startswith(text)]
            return matches[state] if state < len(matches) else None

        assert _complete("/h", 0) == "/help"
        assert _complete("/h", 1) is None
        assert _complete("/q", 0) == "/quit"
        assert _complete("/r", 0) == "/run"
        assert _complete("/", 0) == _COMMANDS[0]
        assert _complete("nonexistent", 0) is None

    def test_complete_function_state_iteration(self) -> None:
        """Verify that incrementing state walks through matches."""
        from maestro_cli.shell import _COMMANDS

        def _complete(text: str, state: int) -> str | None:
            matches = [cmd for cmd in _COMMANDS if cmd.startswith(text)]
            return matches[state] if state < len(matches) else None

        # "/" matches all commands
        all_matches = []
        for i in range(20):
            m = _complete("/", i)
            if m is None:
                break
            all_matches.append(m)
        assert all_matches == _COMMANDS


# ──────────────────────── Coverage Expansion Tests ──────────────────────────


class TestValidateInShellWithPlan:
    """Cover _cmd_validate_in_shell when active_plan IS set (lines 163-165)."""

    def test_validate_calls_cmd_validate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "version: 1\nname: t\ntasks:\n  - id: a\n    command: echo\n",
            encoding="utf-8",
        )
        state = ShellState(active_plan=plan)
        called_with: list[str] = []

        monkeypatch.setattr(
            "maestro_cli.cli._cmd_validate",
            lambda path: (called_with.append(path), 0)[1],
        )

        _cmd_validate_in_shell(state)
        assert called_with == [str(plan)]


class TestSuggestInShellWithPlan:
    """Cover _cmd_suggest_in_shell when active_plan IS set (lines 210-217)."""

    def test_suggest_with_plan_calls_suggest_plan(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "version: 1\nname: t\ntasks:\n  - id: a\n    command: echo\n",
            encoding="utf-8",
        )
        state = ShellState(active_plan=plan)
        from types import SimpleNamespace

        fake_plan = SimpleNamespace(run_dir=".maestro-runs")
        fake_result = SimpleNamespace(suggestions=[])

        # The imports are local to _cmd_suggest_in_shell, so monkeypatch the
        # modules that will be imported by it.
        monkeypatch.setattr("maestro_cli.loader.load_plan", lambda _p: fake_plan)
        monkeypatch.setattr("maestro_cli.suggest.suggest_plan", lambda p, rd: fake_result)
        monkeypatch.setattr("maestro_cli.suggest.format_suggestions", lambda r: "no suggestions")

        _cmd_suggest_in_shell(state)
        out = capsys.readouterr().out
        assert "no suggestions" in out


class TestStatusInShellWithPlan:
    """Cover _cmd_status_in_shell when active_plan IS set (lines 224-233)."""

    def test_status_with_plan_calls_cmd_status(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "version: 1\nname: t\ntasks:\n  - id: a\n    command: echo\n",
            encoding="utf-8",
        )
        state = ShellState(active_plan=plan)
        called: list[object] = []

        monkeypatch.setattr(
            "maestro_cli.cli._cmd_status",
            lambda args: (called.append(args), 0)[1],
        )

        _cmd_status_in_shell(state)
        assert len(called) == 1
        ns = called[0]
        assert ns.plan == str(plan)
        assert ns.json is False


class TestExplainInShellWithPlan:
    """Cover _cmd_explain_in_shell when active_plan IS set (lines 240-248)."""

    def test_explain_with_plan_calls_cmd_explain(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        plan = tmp_path / "plan.yaml"
        plan.write_text(
            "version: 1\nname: t\ntasks:\n  - id: a\n    command: echo\n",
            encoding="utf-8",
        )
        state = ShellState(active_plan=plan)
        called: list[object] = []

        monkeypatch.setattr(
            "maestro_cli.cli._cmd_explain",
            lambda args: (called.append(args), 0)[1],
        )

        _cmd_explain_in_shell(state)
        assert len(called) == 1
        ns = called[0]
        assert ns.plan == str(plan)
        assert ns.json is False


class TestSetupReadlineWithModule:
    """Cover the readline success path + completion closure (lines 43-65)."""

    def test_setup_readline_with_mock_readline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """When readline is importable, set_completer and parse_and_bind are called."""
        import types

        captured_completer: list[object] = []
        captured_bind: list[str] = []

        fake_readline = types.ModuleType("readline")
        fake_readline.set_completer = lambda fn: captured_completer.append(fn)  # type: ignore[attr-defined]
        fake_readline.parse_and_bind = lambda s: captured_bind.append(s)  # type: ignore[attr-defined]

        import sys as _sys
        monkeypatch.setitem(_sys.modules, "readline", fake_readline)

        # Re-import to pick up the mock
        from maestro_cli.shell import _setup_readline
        _setup_readline()

        assert len(captured_completer) == 1
        assert captured_bind == ["tab: complete"]

        # Exercise the completion function
        completer = captured_completer[0]
        # Matching "/h" should find "/help"
        assert completer("/h", 0) == "/help"
        assert completer("/h", 1) is None
        # "/" should match all commands
        assert completer("/", 0) == _COMMANDS[0]
        # No match
        assert completer("zzz", 0) is None

    def test_completion_function_matches_yaml_files(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Completion function finds .yaml files in cwd."""
        import types
        import sys as _sys

        captured_completer: list[object] = []

        fake_readline = types.ModuleType("readline")
        fake_readline.set_completer = lambda fn: captured_completer.append(fn)  # type: ignore[attr-defined]
        fake_readline.parse_and_bind = lambda s: None  # type: ignore[attr-defined]

        monkeypatch.setitem(_sys.modules, "readline", fake_readline)

        # Create YAML files in tmp_path
        (tmp_path / "myplan.yaml").write_text("x", encoding="utf-8")
        (tmp_path / "other.yml").write_text("x", encoding="utf-8")
        (tmp_path / "notmatch.txt").write_text("x", encoding="utf-8")

        monkeypatch.chdir(tmp_path)

        from maestro_cli.shell import _setup_readline
        _setup_readline()

        completer = captured_completer[0]
        # "my" should match "myplan.yaml"
        assert completer("my", 0) == "myplan.yaml"
        assert completer("my", 1) is None

    def test_setup_readline_none_module_returns_early(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When readline import raises and pyreadline3 also fails, returns without error."""
        import builtins

        _original = builtins.__import__

        def _fail_import(name: str, *a: object, **kw: object) -> object:
            if name in ("readline", "pyreadline3"):
                raise ImportError(f"no {name}")
            return _original(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _fail_import)
        _setup_readline()  # Should not raise


class TestSetupReadlinePyreadline3Fallback:
    """Cover the pyreadline3 fallback branch (lines 38-39)."""

    def test_pyreadline3_fallback(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import builtins
        import types
        import sys as _sys

        captured_completer: list[object] = []
        captured_bind: list[str] = []

        fake_readline = types.ModuleType("readline")
        fake_readline.set_completer = lambda fn: captured_completer.append(fn)  # type: ignore[attr-defined]
        fake_readline.parse_and_bind = lambda s: captured_bind.append(s)  # type: ignore[attr-defined]

        fake_pyreadline3 = types.ModuleType("pyreadline3")

        call_count = 0
        _original = builtins.__import__

        def _selective_import(name: str, *a: object, **kw: object) -> object:
            nonlocal call_count
            if name == "readline":
                call_count += 1
                if call_count == 1:
                    # First import of readline fails
                    raise ImportError("no readline")
                # Second import succeeds (after pyreadline3 installed it)
                return fake_readline
            if name == "pyreadline3":
                _sys.modules["readline"] = fake_readline
                return fake_pyreadline3
            return _original(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", _selective_import)
        # Remove readline from sys.modules so it reimports
        monkeypatch.delitem(_sys.modules, "readline", raising=False)

        _setup_readline()

        assert len(captured_completer) == 1
        assert captured_bind == ["tab: complete"]
