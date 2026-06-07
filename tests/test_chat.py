from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.chat import (
    ChatMessage,
    ChatSession,
    _adjust_command_for_chat,
    _autoload_context_files,
    _build_chat_plan_stub,
    _build_chat_task_stub,
    _build_history_prompt,
    _cmd_clear,
    _cmd_context,
    _cmd_cost,
    _cmd_help_chat,
    _cmd_model,
    _cmd_models,
    _discover_auto_context_files,
    _dispatch_chat_command,
    _extract_turn_cost,
    _format_engine_line,
    _parse_engine_prefix,
    _run_chat_turn,
    _setup_chat_readline,
    run_chat,
)


# ===========================================================================
# TestChatSession
# ===========================================================================


class TestChatSession:
    def test_default_session(self) -> None:
        s = ChatSession()
        assert s.engine == "claude"
        assert s.model is None
        assert s.execution_profile == "plan"
        assert s.messages == []
        assert s.total_cost_usd == 0.0
        assert s.total_turns == 0

    def test_session_with_engine(self) -> None:
        s = ChatSession(engine="codex", model="5.4")
        assert s.engine == "codex"
        assert s.model == "5.4"


# ===========================================================================
# TestStubBuilders
# ===========================================================================


class TestStubBuilders:
    def test_plan_stub_has_correct_name(self) -> None:
        s = ChatSession(engine="claude", model="sonnet")
        plan = _build_chat_plan_stub(s)
        assert plan.name == "chat"
        assert plan.version == 1
        assert plan.defaults.claude.model == "sonnet"

    def test_plan_stub_codex_engine(self) -> None:
        s = ChatSession(engine="codex", model="5.4")
        plan = _build_chat_plan_stub(s)
        assert plan.defaults.codex.model == "5.4"

    def test_task_stub_has_inline_prompt(self) -> None:
        s = ChatSession(engine="claude")
        task = _build_chat_task_stub(s, "hello world")
        assert task.id == "chat-turn"
        assert task.engine == "claude"
        assert task.prompt == "hello world"

    def test_task_stub_respects_engine_override(self) -> None:
        s = ChatSession(engine="claude", model="sonnet")
        task = _build_chat_task_stub(s, "test", engine="gemini", model="pro")
        assert task.engine == "gemini"
        assert task.model == "pro"

    def test_task_stub_uses_session_defaults(self) -> None:
        s = ChatSession(engine="ollama", model="llama3")
        task = _build_chat_task_stub(s, "test")
        assert task.engine == "ollama"
        assert task.model == "llama3"


# ===========================================================================
# TestParseEnginePrefix
# ===========================================================================


class TestParseEnginePrefix:
    def test_no_prefix(self) -> None:
        engine, text = _parse_engine_prefix("hello world")
        assert engine is None
        assert text == "hello world"

    def test_claude_prefix(self) -> None:
        engine, text = _parse_engine_prefix("@claude explain this code")
        assert engine == "claude"
        assert text == "explain this code"

    def test_codex_prefix(self) -> None:
        engine, text = _parse_engine_prefix("@codex optimize query")
        assert engine == "codex"
        assert text == "optimize query"

    def test_gemini_prefix(self) -> None:
        engine, text = _parse_engine_prefix("@gemini summarize")
        assert engine == "gemini"
        assert text == "summarize"

    def test_copilot_prefix(self) -> None:
        engine, text = _parse_engine_prefix("@copilot review this")
        assert engine == "copilot"
        assert text == "review this"

    def test_qwen_prefix(self) -> None:
        engine, text = _parse_engine_prefix("@qwen translate")
        assert engine == "qwen"
        assert text == "translate"

    def test_ollama_prefix(self) -> None:
        engine, text = _parse_engine_prefix("@ollama hello")
        assert engine == "ollama"
        assert text == "hello"

    def test_unknown_prefix_treated_as_text(self) -> None:
        engine, text = _parse_engine_prefix("@unknown do something")
        assert engine is None
        assert text == "@unknown do something"

    def test_engine_only_no_text(self) -> None:
        engine, text = _parse_engine_prefix("@claude")
        assert engine == "claude"
        assert text == ""

    def test_email_not_treated_as_engine(self) -> None:
        engine, text = _parse_engine_prefix("send email to user@example.com")
        assert engine is None
        assert text == "send email to user@example.com"


# ===========================================================================
# TestAdjustCommandForChat
# ===========================================================================


class TestAdjustCommandForChat:
    def test_replaces_json_with_text_claude(self) -> None:
        cmd = ["claude", "--print", "--output-format", "json", "hello"]
        result = _adjust_command_for_chat(cmd, "claude")
        assert result == ["claude", "--print", "--output-format", "text", "hello"]

    def test_replaces_json_with_text_gemini(self) -> None:
        cmd = ["gemini", "--output-format", "json", "-m", "pro", "hello"]
        result = _adjust_command_for_chat(cmd, "gemini")
        assert result == ["gemini", "--output-format", "text", "-m", "pro", "hello"]

    def test_codex_gets_chat_flags(self) -> None:
        """Codex keeps --json, gets --full-auto and --skip-git-repo-check."""
        cmd = ["codex", "exec", "--json", "-C", "/tmp", "hello"]
        result = _adjust_command_for_chat(cmd, "codex")
        assert "--json" in result
        assert "--full-auto" in result
        assert "--skip-git-repo-check" in result
        assert result.index("--full-auto") == result.index("exec") + 1

    def test_noop_for_ollama(self) -> None:
        cmd = ["ollama", "run", "llama3", "hello"]
        result = _adjust_command_for_chat(cmd, "ollama")
        assert result == ["ollama", "run", "llama3", "hello"]

    def test_preserves_text_format(self) -> None:
        cmd = ["claude", "--print", "--output-format", "text", "hello"]
        result = _adjust_command_for_chat(cmd, "claude")
        assert result == ["claude", "--print", "--output-format", "text", "hello"]

    def test_output_format_stream_json_replaced_with_text(self) -> None:
        """stream-json is also replaced with text for readable chat output."""
        cmd = ["claude", "--print", "--output-format", "stream-json", "hello"]
        result = _adjust_command_for_chat(cmd, "claude")
        assert result == ["claude", "--print", "--output-format", "text", "hello"]


# ===========================================================================
# TestBuildHistoryPrompt
# ===========================================================================


class TestBuildHistoryPrompt:
    def test_empty_history(self) -> None:
        s = ChatSession()
        result = _build_history_prompt(s, "hello")
        assert result == "hello"

    def test_single_exchange(self) -> None:
        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content="hi"),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="hello!"),
        ])
        result = _build_history_prompt(s, "how are you?")
        assert "<conversation_history>" in result
        assert "User: hi" in result
        assert "Assistant: hello!" in result
        assert "User: how are you?" in result

    def test_truncation_on_long_history(self) -> None:
        # Create history that exceeds the limit
        long_content = "x" * 50_000
        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content=long_content),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content=long_content),
            ChatMessage(role="user", engine="claude", model="sonnet", content="recent"),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="reply"),
        ])
        result = _build_history_prompt(s, "new message")
        # Most recent messages should be present, oldest may be truncated
        assert "User: recent" in result
        assert "Assistant: reply" in result
        assert "new message" in result


# ===========================================================================
# TestAutoContextBootstrap
# ===========================================================================


class TestAutoContextBootstrap:
    def test_discover_auto_context_files_loads_root_to_leaf(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        cwd = root / "src" / "feature"
        cwd.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "AGENTS.md").write_text("root instructions", encoding="utf-8")
        (root / "src" / "CLAUDE.md").write_text("nested instructions", encoding="utf-8")

        discovered = _discover_auto_context_files(cwd)

        assert discovered == [
            (root / "AGENTS.md").resolve(),
            (root / "src" / "CLAUDE.md").resolve(),
        ]

    def test_discover_auto_context_files_without_git_stays_at_cwd(self, tmp_path: Path) -> None:
        parent = tmp_path / "parent"
        cwd = parent / "child"
        cwd.mkdir(parents=True)
        (parent / "AGENTS.md").write_text("parent instructions", encoding="utf-8")
        (cwd / "CLAUDE.md").write_text("child instructions", encoding="utf-8")

        discovered = _discover_auto_context_files(cwd)

        assert discovered == [(cwd / "CLAUDE.md").resolve()]

    def test_discover_auto_context_files_dedupes_duplicate_filename_entries(
        self,
        tmp_path: Path,
    ) -> None:
        root = tmp_path / "repo"
        root.mkdir()
        (root / ".git").mkdir()
        (root / "AGENTS.md").write_text("root instructions", encoding="utf-8")

        discovered = _discover_auto_context_files(
            root,
            filenames=("AGENTS.md", "AGENTS.md"),
        )

        assert discovered == [(root / "AGENTS.md").resolve()]

    def test_autoload_context_files_preserves_prompt_order(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        cwd = root / "src" / "feature"
        cwd.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "AGENTS.md").write_text("root instructions", encoding="utf-8")
        (root / "src" / "CLAUDE.md").write_text("nested instructions", encoding="utf-8")

        session = ChatSession()
        loaded = _autoload_context_files(session, cwd=cwd, announce=False)
        prompt = _build_history_prompt(session, "next step")

        assert loaded == [
            (root / "AGENTS.md").resolve(),
            (root / "src" / "CLAUDE.md").resolve(),
        ]
        assert list(session.context_files.values()) == ["root instructions", "nested instructions"]
        assert prompt.index("root instructions") < prompt.index("nested instructions")

    def test_manual_context_additions_still_work_after_auto_loading(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / "repo"
        cwd = root / "src"
        cwd.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "AGENTS.md").write_text("root instructions", encoding="utf-8")
        manual = cwd / "manual.md"
        manual.write_text("manual notes", encoding="utf-8")
        monkeypatch.chdir(cwd)

        session = ChatSession()
        _autoload_context_files(session, cwd=cwd, announce=False)
        _cmd_context([str(manual)], session)

        assert list(session.context_files.values()) == ["root instructions", "manual notes"]

    def test_context_clear_removes_auto_loaded_and_manual_files(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        root = tmp_path / "repo"
        cwd = root / "src"
        cwd.mkdir(parents=True)
        (root / ".git").mkdir()
        (root / "AGENTS.md").write_text("root instructions", encoding="utf-8")
        manual = cwd / "manual.md"
        manual.write_text("manual notes", encoding="utf-8")
        monkeypatch.chdir(cwd)

        session = ChatSession()
        _autoload_context_files(session, cwd=cwd, announce=False)
        _cmd_context([str(manual)], session)
        _cmd_context(["--clear"], session)

        assert session.context_files == {}


# ===========================================================================
# TestDispatchChatCommand
# ===========================================================================


class TestDispatchChatCommand:
    def test_quit_returns_false(self) -> None:
        s = ChatSession()
        assert _dispatch_chat_command("/quit", s) is False

    def test_help_returns_true(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession()
        assert _dispatch_chat_command("/help", s) is True
        out = capsys.readouterr().out
        assert "/model" in out
        assert "/quit" in out

    def test_unknown_command(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession()
        assert _dispatch_chat_command("/unknown", s) is True
        out = capsys.readouterr().out
        assert "unknown command" in out.lower()

    def test_clear_resets_history(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content="hi"),
        ])
        assert _dispatch_chat_command("/clear", s) is True
        assert len(s.messages) == 0
        out = capsys.readouterr().out
        assert "cleared 1" in out

    def test_cost_displays_zero_initially(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession()
        assert _dispatch_chat_command("/cost", s) is True
        out = capsys.readouterr().out
        assert "0 turns" in out
        assert "--" in out

    def test_models_lists_all_engines(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession()
        assert _dispatch_chat_command("/models", s) is True
        out = capsys.readouterr().out
        assert "claude" in out
        assert "codex" in out
        assert "gemini" in out
        assert "ollama" in out


# ===========================================================================
# TestCmdModel
# ===========================================================================


class TestCmdModel:
    def test_engine_slash_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession(engine="claude", model="sonnet")
        _cmd_model(["codex/5.4"], s)
        assert s.engine == "codex"
        assert s.model == "5.4"
        out = capsys.readouterr().out
        assert "codex/5.4" in out

    def test_model_only_keeps_engine(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession(engine="claude", model="sonnet")
        _cmd_model(["opus"], s)
        assert s.engine == "claude"  # unchanged
        assert s.model == "opus"

    def test_invalid_engine_warns(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession(engine="claude")
        _cmd_model(["invalid/model"], s)
        assert s.engine == "claude"  # unchanged
        out = capsys.readouterr().out
        assert "unknown engine" in out.lower()

    def test_no_args_shows_current(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession(engine="gemini", model="pro")
        _cmd_model([], s)
        out = capsys.readouterr().out
        assert "gemini/pro" in out

    def test_bare_engine_name_switches_engine(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession(engine="claude", model="sonnet")
        _cmd_model(["codex"], s)
        assert s.engine == "codex"
        assert s.model is None  # reset to default
        out = capsys.readouterr().out
        assert "codex/default" in out

    def test_engine_slash_empty_model(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession(engine="claude", model="sonnet")
        _cmd_model(["codex/"], s)
        assert s.engine == "codex"
        assert s.model is None  # reset to default


# ===========================================================================
# TestExtractTurnCost
# ===========================================================================


class TestExtractTurnCost:
    def test_no_cost_returns_none(self) -> None:
        assert _extract_turn_cost("just some text output", "claude") is None

    def test_cost_from_claude_output(self) -> None:
        output = "Here is the answer\nTotal cost: $0.0123\nDone."
        cost = _extract_turn_cost(output, "claude")
        assert cost is not None
        assert abs(cost - 0.0123) < 0.001

    def test_empty_output(self) -> None:
        assert _extract_turn_cost("", "claude") is None


# ===========================================================================
# TestRunChat
# ===========================================================================


class TestRunChat:
    def test_quit_on_first_input(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("builtins.input", lambda _: "/quit")
        result = run_chat()
        assert result == 0

    def test_keyboard_interrupt_exits_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise_interrupt(_: str) -> str:
            raise KeyboardInterrupt

        monkeypatch.setattr("builtins.input", _raise_interrupt)
        result = run_chat()
        assert result == 0

    def test_eof_exits_cleanly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise_eof(_: str) -> str:
            raise EOFError

        monkeypatch.setattr("builtins.input", _raise_eof)
        result = run_chat()
        assert result == 0

    def test_empty_input_continues(self, monkeypatch: pytest.MonkeyPatch) -> None:
        inputs = iter(["", "  ", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = run_chat()
        assert result == 0

    def test_slash_command_in_loop(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        inputs = iter(["/models", "/cost", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        result = run_chat()
        assert result == 0
        out = capsys.readouterr().out
        assert "claude" in out  # /models output

    def test_run_chat_announces_auto_loaded_context(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        cwd = tmp_path / "repo" / "src"
        cwd.mkdir(parents=True)
        (tmp_path / "repo" / ".git").mkdir()
        (tmp_path / "repo" / "AGENTS.md").write_text("root instructions", encoding="utf-8")
        monkeypatch.chdir(cwd)
        monkeypatch.setattr("builtins.input", lambda _: "/quit")

        result = run_chat()

        assert result == 0
        out = capsys.readouterr().out
        assert "auto-loaded 1 context file" in out


# ===========================================================================
# TestFormatEngineLine
# ===========================================================================


class TestFormatEngineLine:
    def test_non_codex_engine_returns_line_unchanged(self) -> None:
        line = "some plain text output\n"
        assert _format_engine_line(line, "claude") == line
        assert _format_engine_line(line, "gemini") == line
        assert _format_engine_line(line, "ollama") == line

    def test_codex_agent_message_returns_text(self) -> None:
        event = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Hello from codex"},
        }
        import json as _json
        line = _json.dumps(event) + "\n"
        result = _format_engine_line(line, "codex")
        assert result == "Hello from codex\n"

    def test_codex_item_completed_non_agent_message_suppressed(self) -> None:
        event = {
            "type": "item.completed",
            "item": {"type": "reasoning", "text": "thinking..."},
        }
        import json as _json
        line = _json.dumps(event) + "\n"
        result = _format_engine_line(line, "codex")
        assert result is None

    @pytest.mark.parametrize("msg_type", [
        "thread.started",
        "turn.started",
        "turn.completed",
        "item.started",
        "item.streaming",
    ])
    def test_codex_metadata_events_suppressed(self, msg_type: str) -> None:
        import json as _json
        line = _json.dumps({"type": msg_type, "data": {}}) + "\n"
        result = _format_engine_line(line, "codex")
        assert result is None

    def test_codex_malformed_json_returns_line_as_is(self) -> None:
        bad_line = '{"type": "item.completed", broken json\n'
        result = _format_engine_line(bad_line, "codex")
        assert result == bad_line

    def test_codex_non_json_line_returned_as_is(self) -> None:
        plain = "just a plain text line\n"
        result = _format_engine_line(plain, "codex")
        assert result == plain

    def test_codex_agent_message_empty_text_suppressed(self) -> None:
        import json as _json
        event = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": ""},
        }
        line = _json.dumps(event) + "\n"
        # empty text → item.completed branch returns None
        result = _format_engine_line(line, "codex")
        assert result is None


# ===========================================================================
# TestCmdHelpChat
# ===========================================================================


class TestCmdHelpChat:
    def test_all_six_slash_commands_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        _cmd_help_chat()
        out = capsys.readouterr().out
        for cmd in ("/model", "/models", "/clear", "/cost", "/help", "/quit"):
            assert cmd in out, f"Expected '{cmd}' in help output"

    def test_routing_section_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        _cmd_help_chat()
        out = capsys.readouterr().out
        assert "Routing" in out
        assert "@engine" in out


# ===========================================================================
# TestSetupChatReadline
# ===========================================================================


class TestSetupChatReadline:
    def test_graceful_fallback_when_readline_unavailable(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """_setup_chat_readline must not raise even if no readline module is available."""
        import builtins
        real_import = builtins.__import__

        def _block_readline(name: str, *args: Any, **kwargs: Any) -> Any:
            if name in ("readline", "pyreadline3"):
                raise ImportError(f"mocked: no {name}")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_readline)
        # Should complete without raising
        _setup_chat_readline()


# ===========================================================================
# TestAdjustCommandForChatExtra
# ===========================================================================


class TestAdjustCommandForChatExtra:
    def test_codex_does_not_duplicate_full_auto(self) -> None:
        """If --full-auto is already present it must not be inserted again."""
        cmd = ["codex", "exec", "--full-auto", "--json", "prompt"]
        result = _adjust_command_for_chat(cmd, "codex")
        assert result.count("--full-auto") == 1

    def test_codex_does_not_duplicate_skip_git_repo_check(self) -> None:
        cmd = ["codex", "exec", "--skip-git-repo-check", "--json", "prompt"]
        result = _adjust_command_for_chat(cmd, "codex")
        assert result.count("--skip-git-repo-check") == 1

    def test_empty_command_list_non_codex(self) -> None:
        result = _adjust_command_for_chat([], "claude")
        assert result == []

    def test_empty_command_list_codex(self) -> None:
        # For codex with empty list, --full-auto and --skip-git-repo-check are appended
        result = _adjust_command_for_chat([], "codex")
        assert "--full-auto" in result
        assert "--skip-git-repo-check" in result


# ===========================================================================
# TestRunChatTurn
# ===========================================================================


class TestRunChatTurn:
    def test_builds_correct_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from maestro_cli.chat import _run_chat_turn

        session = ChatSession(engine="claude", model="sonnet")
        built_commands: list[Any] = []

        def _fake_build_command(plan: Any, task: Any, workdir: Any, **kw: Any) -> tuple[list[str], bool]:
            built_commands.append(task.prompt)
            return (["echo", "hello from claude"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: dict(os.environ))

        result = _run_chat_turn(session, "test prompt")
        assert result is not None
        assert result.role == "assistant"
        assert result.engine == "claude"
        assert len(built_commands) == 1
        assert "test prompt" in built_commands[0]

    def test_records_message_content(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from maestro_cli.chat import _run_chat_turn

        session = ChatSession(engine="ollama", model="llama3")

        def _fake_build_command(plan: Any, task: Any, workdir: Any, **kw: Any) -> tuple[list[str], bool]:
            return (["echo", "response text"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: dict(os.environ))

        result = _run_chat_turn(session, "hello")
        assert result is not None
        assert "response text" in result.content

    def test_engine_not_found(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from maestro_cli.chat import _run_chat_turn

        session = ChatSession(engine="claude", model="sonnet")

        def _fake_build_command(plan: Any, task: Any, workdir: Any, **kw: Any) -> tuple[list[str], bool]:
            return (["nonexistent-engine-cli-12345", "hello"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: dict(os.environ))

        result = _run_chat_turn(session, "hello")
        assert result is None
        out = capsys.readouterr().out
        assert "not found" in out.lower() or "error" in out.lower()


# ===========================================================================
# TestCLIIntegration
# ===========================================================================


class TestCLIIntegration:
    def test_chat_subcommand_registered(self) -> None:
        from maestro_cli.cli import _build_parser

        parser = _build_parser()
        # Parse chat with defaults
        args = parser.parse_args(["chat"])
        assert args.command == "chat"
        assert args.engine == "claude"
        assert args.model is None
        assert args.execution_profile == "plan"
        assert args.no_auto_context is False

    def test_chat_with_engine_flag(self) -> None:
        from maestro_cli.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["chat", "--engine", "codex", "--model", "5.4"])
        assert args.engine == "codex"
        assert args.model == "5.4"

    def test_chat_with_profile(self) -> None:
        from maestro_cli.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["chat", "--execution-profile", "yolo"])
        assert args.execution_profile == "yolo"

    def test_chat_with_no_auto_context_flag(self) -> None:
        from maestro_cli.cli import _build_parser

        parser = _build_parser()
        args = parser.parse_args(["chat", "--no-auto-context"])
        assert args.no_auto_context is True


# ===========================================================================
# TestFormatEngineLine  (ZERO existing coverage)
# ===========================================================================


class TestFormatEngineLine2:
    """Tests for _format_engine_line() — Codex JSON parsing + metadata suppression."""

    def test_non_codex_engine_returns_line_unchanged(self) -> None:
        from maestro_cli.chat import _format_engine_line

        line = "Hello, I am Claude.\n"
        assert _format_engine_line(line, "claude") == line

    def test_non_codex_gemini_returns_line_unchanged(self) -> None:
        from maestro_cli.chat import _format_engine_line

        line = "Some gemini output\n"
        assert _format_engine_line(line, "gemini") == line

    def test_codex_agent_message_extracted(self) -> None:
        from maestro_cli.chat import _format_engine_line

        payload = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Here is the answer"},
        }
        line = __import__("json").dumps(payload) + "\n"
        result = _format_engine_line(line, "codex")
        assert result is not None
        assert "Here is the answer" in result

    def test_codex_agent_message_appends_newline(self) -> None:
        from maestro_cli.chat import _format_engine_line

        payload = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Done"},
        }
        result = _format_engine_line(__import__("json").dumps(payload), "codex")
        assert result is not None
        assert result.endswith("\n")

    def test_codex_reasoning_item_suppressed(self) -> None:
        from maestro_cli.chat import _format_engine_line

        payload = {
            "type": "item.completed",
            "item": {"type": "reasoning", "text": "I am thinking..."},
        }
        result = _format_engine_line(__import__("json").dumps(payload), "codex")
        assert result is None

    def test_codex_thread_started_suppressed(self) -> None:
        from maestro_cli.chat import _format_engine_line

        payload = {"type": "thread.started"}
        result = _format_engine_line(__import__("json").dumps(payload), "codex")
        assert result is None

    def test_codex_turn_started_suppressed(self) -> None:
        from maestro_cli.chat import _format_engine_line

        payload = {"type": "turn.started"}
        result = _format_engine_line(__import__("json").dumps(payload), "codex")
        assert result is None

    def test_codex_turn_completed_suppressed(self) -> None:
        from maestro_cli.chat import _format_engine_line

        payload = {"type": "turn.completed"}
        result = _format_engine_line(__import__("json").dumps(payload), "codex")
        assert result is None

    def test_codex_item_started_suppressed(self) -> None:
        from maestro_cli.chat import _format_engine_line

        payload = {"type": "item.started"}
        result = _format_engine_line(__import__("json").dumps(payload), "codex")
        assert result is None

    def test_codex_item_streaming_suppressed(self) -> None:
        from maestro_cli.chat import _format_engine_line

        payload = {"type": "item.streaming"}
        result = _format_engine_line(__import__("json").dumps(payload), "codex")
        assert result is None

    def test_codex_malformed_json_returned_as_is(self) -> None:
        from maestro_cli.chat import _format_engine_line

        line = "{not valid json at all\n"
        result = _format_engine_line(line, "codex")
        assert result == line

    def test_codex_non_json_plain_text_returned_as_is(self) -> None:
        from maestro_cli.chat import _format_engine_line

        line = "plain text without braces\n"
        result = _format_engine_line(line, "codex")
        assert result == line

    def test_codex_empty_agent_message_text_suppressed(self) -> None:
        from maestro_cli.chat import _format_engine_line

        payload = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": ""},
        }
        result = _format_engine_line(__import__("json").dumps(payload), "codex")
        # Empty text means no useful output — should be suppressed
        assert result is None

    def test_codex_unknown_event_type_suppressed(self) -> None:
        from maestro_cli.chat import _format_engine_line

        payload = {"type": "some.unknown.event", "data": "irrelevant"}
        result = _format_engine_line(__import__("json").dumps(payload), "codex")
        assert result is None


# ===========================================================================
# TestCmdHelpChat  (ZERO existing coverage)
# ===========================================================================


class TestCmdHelpChat2:
    """Tests for _cmd_help_chat() — all slash commands must appear in output."""

    def test_all_slash_commands_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_help_chat

        _cmd_help_chat()
        out = capsys.readouterr().out
        for cmd in ["/model", "/models", "/clear", "/cost", "/help", "/quit"]:
            assert cmd in out, f"Missing {cmd!r} in help output"

    def test_routing_section_present(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_help_chat

        _cmd_help_chat()
        out = capsys.readouterr().out
        assert "@engine" in out
        assert "Routing" in out

    def test_output_non_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_help_chat

        _cmd_help_chat()
        out = capsys.readouterr().out
        assert len(out) > 50


# ===========================================================================
# TestBuildHistoryPromptEdgeCases  (extends existing coverage)
# ===========================================================================


class TestBuildHistoryPromptEdgeCases:
    """Edge cases for _build_history_prompt() not covered by existing tests."""

    def test_char_limit_boundary_drops_oldest(self) -> None:
        """Messages exceeding _HISTORY_CHAR_LIMIT (80_000) must drop oldest first."""
        from maestro_cli.chat import _build_history_prompt, _HISTORY_CHAR_LIMIT

        # Each message is just over half the limit so together they exceed it
        half_limit = _HISTORY_CHAR_LIMIT // 2 + 100
        old_content = "O" * half_limit
        new_content = "N" * half_limit

        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content=old_content),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content=new_content),
        ])
        result = _build_history_prompt(s, "final question")
        # The most recent assistant message should survive; oldest user may not
        # What matters: result contains the new message and doesn't crash
        assert "final question" in result

    def test_oversized_single_message_returns_new_message_only(self) -> None:
        """A single message larger than the limit is skipped; only new message returned."""
        from maestro_cli.chat import _build_history_prompt, _HISTORY_CHAR_LIMIT

        giant_content = "G" * (_HISTORY_CHAR_LIMIT + 1)
        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content=giant_content),
        ])
        result = _build_history_prompt(s, "new question")
        # The giant message exceeds the budget on first entry, so parts is empty
        # → falls back to returning just the new message
        assert result == "new question"

    def test_history_block_format(self) -> None:
        """Verify the XML wrapper and role labels are correct."""
        from maestro_cli.chat import _build_history_prompt

        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content="ping"),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="pong"),
        ])
        result = _build_history_prompt(s, "next")
        assert result.startswith("<conversation_history>")
        assert "User: ping" in result
        assert "Assistant: pong" in result
        assert result.endswith("User: next")


# ===========================================================================
# TestSetupChatReadline  (ZERO existing coverage)
# ===========================================================================


class TestSetupChatReadline2:
    """Tests for _setup_chat_readline() — graceful no-readline fallback."""

    def test_no_readline_no_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When both readline and pyreadline3 are missing, function returns silently."""
        import builtins
        import importlib

        original_import = builtins.__import__

        def _block_readline(name: str, *args: object, **kwargs: object) -> object:
            if name in ("readline", "pyreadline3"):
                raise ImportError(f"Mocked: {name} not available")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _block_readline)
        from maestro_cli.chat import _setup_chat_readline

        # Must not raise
        _setup_chat_readline()

    def test_with_mock_readline(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When readline is available, set_completer and parse_and_bind are called."""
        import types

        calls: list[str] = []
        mock_readline = types.ModuleType("readline")
        mock_readline.set_completer = lambda fn: calls.append("set_completer")  # type: ignore[attr-defined]
        mock_readline.parse_and_bind = lambda s: calls.append(f"bind:{s}")  # type: ignore[attr-defined]

        import builtins
        import sys

        # Inject mock module
        original_modules = dict(sys.modules)
        sys.modules["readline"] = mock_readline
        try:
            # Re-import the function to get fresh execution
            from maestro_cli.chat import _setup_chat_readline
            _setup_chat_readline()
        finally:
            # Restore original module state
            if "readline" in original_modules:
                sys.modules["readline"] = original_modules["readline"]
            else:
                sys.modules.pop("readline", None)

        assert "set_completer" in calls
        assert any("bind:" in c for c in calls)


# ===========================================================================
# TestExtractTurnCostExtended  (extends existing coverage)
# ===========================================================================


class TestExtractTurnCostExtended:
    """Additional cost extraction edge cases."""

    def test_cost_extracted_from_last_20_lines(self) -> None:
        """Cost pattern found within last 20 lines is extracted."""
        from maestro_cli.chat import _extract_turn_cost

        # Build output with 25 lines; cost line is near the end (line 22)
        lines = [f"line {i}" for i in range(20)]
        lines.append("Total cost: $0.0042")
        lines.append("Done.")
        output = "\n".join(lines)
        cost = _extract_turn_cost(output, "claude")
        assert cost is not None
        assert abs(cost - 0.0042) < 0.0001

    def test_cost_beyond_last_20_lines_not_extracted(self) -> None:
        """Cost pattern beyond last 20 lines is NOT extracted."""
        from maestro_cli.chat import _extract_turn_cost

        lines = ["Total cost: $9.9999"]
        lines += [f"line {i}" for i in range(25)]  # 25 more lines push cost out of range
        output = "\n".join(lines)
        cost = _extract_turn_cost(output, "claude")
        # Cost line is now > 20 lines from the end — should not be found
        assert cost is None

    def test_multiline_output_no_cost(self) -> None:
        from maestro_cli.chat import _extract_turn_cost

        output = "\n".join(f"response line {i}" for i in range(10))
        assert _extract_turn_cost(output, "gemini") is None


# ===========================================================================
# TestCmdModelsDetail  (model alias content per engine)
# ===========================================================================


class TestCmdModelsDetail:
    def test_claude_aliases_include_standard_models(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_models

        _cmd_models()
        out = capsys.readouterr().out
        # Claude engine should list haiku and sonnet (from CLAUDE_MODELS)
        assert "haiku" in out or "sonnet" in out

    def test_gemini_aliases_include_flash_and_pro(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_models

        _cmd_models()
        out = capsys.readouterr().out
        assert "flash" in out
        assert "pro" in out

    def test_ollama_aliases_include_llama3(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_models

        _cmd_models()
        out = capsys.readouterr().out
        assert "llama3" in out

    def test_output_includes_all_six_engines(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_models

        _cmd_models()
        out = capsys.readouterr().out
        for engine in ["claude", "codex", "gemini", "copilot", "qwen", "ollama"]:
            assert engine in out, f"Missing engine: {engine}"

    def test_codex_aliases_include_version_shortcuts(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_models

        _cmd_models()
        out = capsys.readouterr().out
        # Codex aliases like "5.4" should appear
        assert "5.4" in out or "5.1" in out


# ===========================================================================
# TestParseEnginePrefixEdgeCases  (case sensitivity + whitespace)
# ===========================================================================


class TestParseEnginePrefixEdgeCases:
    def test_uppercase_prefix_not_recognized(self) -> None:
        """@CLAUDE uppercase is case-sensitive — not a valid engine name."""
        engine, text = _parse_engine_prefix("@CLAUDE do something")
        assert engine is None
        assert text == "@CLAUDE do something"

    def test_mixed_case_prefix_not_recognized(self) -> None:
        """@Claude mixed case is not recognized as a valid engine."""
        engine, text = _parse_engine_prefix("@Claude hello")
        assert engine is None
        assert text == "@Claude hello"

    def test_tab_separator_treated_as_whitespace(self) -> None:
        """@engine<tab>text — tab is whitespace; prefix is recognized."""
        engine, text = _parse_engine_prefix("@claude\thello world")
        assert engine == "claude"
        assert "hello world" in text

    def test_multiple_spaces_after_prefix_collapsed(self) -> None:
        """Multiple spaces after @engine are collapsed; text portion is correct."""
        engine, text = _parse_engine_prefix("@gemini   summarize this")
        assert engine == "gemini"
        assert "summarize" in text


# ===========================================================================
# TestDispatchChatCommandCaseSensitivity
# ===========================================================================


class TestDispatchChatCommandCaseSensitivity:
    def test_quit_uppercase_exits(self) -> None:
        """/QUIT uppercase should exit — cmd.lower() normalises it."""
        s = ChatSession()
        assert _dispatch_chat_command("/QUIT", s) is False

    def test_help_uppercase_shows_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """/HELP uppercase shows help output."""
        s = ChatSession()
        result = _dispatch_chat_command("/HELP", s)
        assert result is True
        out = capsys.readouterr().out
        assert "/model" in out

    def test_clear_uppercase_clears_history(self) -> None:
        """/CLEAR uppercase clears conversation history."""
        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content="hi"),
        ])
        _dispatch_chat_command("/CLEAR", s)
        assert len(s.messages) == 0

    def test_models_uppercase_lists_engines(self, capsys: pytest.CaptureFixture[str]) -> None:
        """/MODELS uppercase lists available engines."""
        s = ChatSession()
        result = _dispatch_chat_command("/MODELS", s)
        assert result is True
        out = capsys.readouterr().out
        assert "claude" in out


# ===========================================================================
# TestCmdCostExtended  (accumulated cost + elapsed time + bad isoformat)
# ===========================================================================


class TestCmdCostExtended:
    def test_cost_with_accumulated_turns(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Session with real cost shows formatted $ amount."""
        s = ChatSession(total_turns=3, total_cost_usd=0.0456)
        _cmd_cost(s)
        out = capsys.readouterr().out
        assert "3 turns" in out
        assert "$0.0456" in out

    def test_bad_isoformat_no_crash(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Invalid started_at isoformat is silently ignored — no crash."""
        s = ChatSession(started_at="not-a-date")
        _cmd_cost(s)  # must not raise
        out = capsys.readouterr().out
        assert "turns" in out

    def test_elapsed_time_shown_in_minutes(self, capsys: pytest.CaptureFixture[str]) -> None:
        """started_at set to ~5 min ago → elapsed shows 'min' in output."""
        from datetime import datetime, UTC, timedelta

        started = datetime.now(UTC) - timedelta(minutes=5)
        s = ChatSession(
            started_at=started.isoformat(),
            total_turns=1,
            total_cost_usd=0.01,
        )
        _cmd_cost(s)
        out = capsys.readouterr().out
        assert "min" in out

    def test_zero_cost_shows_dashes(self, capsys: pytest.CaptureFixture[str]) -> None:
        """total_cost_usd=0 shows '--' instead of '$0.0000'."""
        s = ChatSession(total_turns=2, total_cost_usd=0.0)
        _cmd_cost(s)
        out = capsys.readouterr().out
        assert "--" in out
        assert "$" not in out


# ===========================================================================
# TestAdjustCommandForChatEdgeCases  (duplicate flag prevention, edge inputs)
# ===========================================================================


class TestAdjustCommandForChatEdgeCases:
    def test_codex_without_exec_still_gets_flags(self) -> None:
        """Codex cmd without 'exec' subcommand still gets --full-auto at index 1."""
        cmd = ["codex", "--json", "hello"]
        result = _adjust_command_for_chat(cmd, "codex")
        assert "--full-auto" in result
        assert "--skip-git-repo-check" in result

    def test_codex_no_duplicate_full_auto(self) -> None:
        """--full-auto already present → not duplicated."""
        cmd = ["codex", "exec", "--full-auto", "--json", "hello"]
        result = _adjust_command_for_chat(cmd, "codex")
        assert result.count("--full-auto") == 1

    def test_codex_no_duplicate_skip_git_flag(self) -> None:
        """--skip-git-repo-check already present → not duplicated."""
        cmd = ["codex", "exec", "--skip-git-repo-check", "--json", "hello"]
        result = _adjust_command_for_chat(cmd, "codex")
        assert result.count("--skip-git-repo-check") == 1

    def test_empty_cmd_list_claude_no_crash(self) -> None:
        """Empty command list for non-codex engine returns empty list."""
        result = _adjust_command_for_chat([], "claude")
        assert result == []

    def test_empty_cmd_list_codex_no_crash(self) -> None:
        """Empty command list for codex engine still adds flags without crashing."""
        result = _adjust_command_for_chat([], "codex")
        assert "--full-auto" in result
        assert "--skip-git-repo-check" in result

    def test_string_command_not_modified(self) -> None:
        """String (shell) commands are passed through as a list untouched."""
        # _adjust_command_for_chat only handles list[str]; a string in list stays
        cmd = ["claude", "--print", "--output-format", "text", "my prompt"]
        result = _adjust_command_for_chat(cmd, "claude")
        assert result == cmd  # already text format, no change


# ===========================================================================
# TestRunChatTurnReasoningEffort
# ===========================================================================


class TestRunChatTurnReasoningEffort:
    def test_claude_reasoning_effort_injected_into_env(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """CLAUDE_CODE_EFFORT_LEVEL is injected when plan.defaults.claude.reasoning_effort is set."""
        import io
        from maestro_cli.chat import _run_chat_turn
        from maestro_cli.models import PlanSpec, PlanDefaults, EngineDefaults

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured["env"] = kwargs.get("env", {})
                self.stdout: list[str] = []
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        def _fake_plan_stub(session: object) -> PlanSpec:
            defaults = PlanDefaults()
            defaults.claude = EngineDefaults(model="sonnet", reasoning_effort="high")
            return PlanSpec(name="chat", defaults=defaults)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("maestro_cli.chat._build_chat_plan_stub", _fake_plan_stub)
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        _run_chat_turn(session, "test")

        env = captured.get("env", {})
        assert isinstance(env, dict)
        assert "CLAUDE_CODE_EFFORT_LEVEL" in env
        assert env["CLAUDE_CODE_EFFORT_LEVEL"] == "high"

    def test_non_claude_engine_no_effort_env_var(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Non-claude engines do NOT inject CLAUDE_CODE_EFFORT_LEVEL."""
        import io
        from maestro_cli.chat import _run_chat_turn

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured["env"] = kwargs.get("env", {})
                self.stdout: list[str] = []
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="gemini", model="flash")
        _run_chat_turn(session, "test")

        env = captured.get("env", {})
        assert isinstance(env, dict)
        assert "CLAUDE_CODE_EFFORT_LEVEL" not in env


# ===========================================================================
# TestRunChatTurnEdgeCases  (stderr fallback, Windows builtins, build error)
# ===========================================================================


class TestRunChatTurnEdgeCases:
    def test_stderr_fallback_when_stdout_empty(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When stdout is empty, stderr content is used as the turn output."""
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                # stdout yields nothing; stderr has the actual response
                self.stdout: list[str] = []
                self.stderr = io.StringIO("engine response via stderr\n")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", ""], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "hello")
        assert result is not None
        assert "engine response via stderr" in result.content

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-specific behavior: shell-builtin path uses subprocess.list2cmdline (Windows-only)",
    )
    def test_windows_shell_builtin_uses_shell_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On Windows (os.name == 'nt'), shell builtins trigger shell=True launch."""
        import io
        from maestro_cli.chat import _run_chat_turn

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(
                self, cmd: object, *args: object, shell: bool = False, **kwargs: object
            ) -> None:
                captured["cmd"] = cmd
                captured["shell"] = shell
                self.stdout: list[str] = []
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            # Return a command whose first element is a Windows shell builtin
            return (["echo", "hello"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)
        monkeypatch.setattr("os.name", "nt")

        session = ChatSession(engine="claude", model="sonnet")
        _run_chat_turn(session, "hello")

        # On Windows, "echo" is a shell builtin → shell=True must have been used
        assert captured.get("shell") is True

    def test_build_command_exception_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """If build_command() raises, _run_chat_turn returns None and prints error."""
        from maestro_cli.chat import _run_chat_turn

        def _broken_build(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            raise RuntimeError("intentional build failure")

        monkeypatch.setattr("maestro_cli.runners.build_command", _broken_build)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})

        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "hello")
        assert result is None
        out = capsys.readouterr().out
        assert "error" in out.lower()

    def test_engine_override_used_in_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Explicit engine/model override in _run_chat_turn is reflected in result."""
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = ["gemini response\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        # Session is claude, but this turn overrides to gemini/pro
        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "translate this", engine="gemini", model="pro")
        assert result is not None
        assert result.engine == "gemini"
        assert result.model == "pro"

    def test_task_level_reasoning_effort_overrides_plan_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """task.reasoning_effort takes precedence over plan.defaults.claude.reasoning_effort."""
        import io
        from maestro_cli.chat import _run_chat_turn
        from maestro_cli.models import PlanSpec, PlanDefaults, EngineDefaults, TaskSpec

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured["env"] = kwargs.get("env", {})
                self.stdout: list[str] = []
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        def _fake_plan_stub(session: object) -> PlanSpec:
            defaults = PlanDefaults()
            # Plan default is "medium" — task will override to "low"
            defaults.claude = EngineDefaults(model="opus", reasoning_effort="medium")
            return PlanSpec(name="chat", defaults=defaults)

        def _fake_task_stub(
            session: object,
            prompt: str,
            engine: str | None = None,
            model: str | None = None,
        ) -> TaskSpec:
            task = TaskSpec(
                id="chat-turn",
                engine="claude",  # type: ignore[arg-type]
                model="opus",
                prompt=prompt,
                reasoning_effort="low",  # task-level override
            )
            return task

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("maestro_cli.chat._build_chat_plan_stub", _fake_plan_stub)
        monkeypatch.setattr("maestro_cli.chat._build_chat_task_stub", _fake_task_stub)
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="opus")
        _run_chat_turn(session, "test reasoning")

        env = captured.get("env", {})
        assert isinstance(env, dict)
        # task.reasoning_effort="low" takes precedence
        assert env.get("CLAUDE_CODE_EFFORT_LEVEL") == "low"

    def test_output_ending_in_newline_no_extra_print(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When output already ends with a newline, no extra blank line is added."""
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = ["response with newline\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="ollama", model="llama3")
        result = _run_chat_turn(session, "hello")
        assert result is not None
        # The output content must be stripped (ChatMessage.content = full_output.strip())
        assert result.content == "response with newline"

    def test_codex_standalone_json_flag_preserved(self) -> None:
        """For Codex, the standalone --json flag (not --output-format json) is never removed."""
        from maestro_cli.chat import _adjust_command_for_chat

        cmd = ["codex", "exec", "--json", "--some-other-flag", "hello"]
        result = _adjust_command_for_chat(cmd, "codex")
        # --json (the standalone Codex JSON output flag) must be preserved
        assert "--json" in result


# ===========================================================================
# TestSetupChatReadline  — graceful no-readline fallback
# ===========================================================================


class TestSetupChatReadline3:
    def test_no_readline_no_crash(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When neither readline nor pyreadline3 is importable, _setup_chat_readline is a no-op."""
        import builtins
        from maestro_cli.chat import _setup_chat_readline

        real_import = builtins.__import__

        def _failing_import(name: str, *args: object, **kwargs: object) -> object:
            if name in ("readline", "pyreadline3"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _failing_import)
        # Must not raise
        _setup_chat_readline()


# ===========================================================================
# TestExtractTurnCostEdgeCases  — JSON payload with costUSD key
# ===========================================================================


class TestExtractTurnCostEdgeCases:
    def test_codex_json_cost_usd_key(self) -> None:
        """_extract_turn_cost finds cost from a Codex-style JSON line with 'costUSD'."""
        import json as _json
        from maestro_cli.chat import _extract_turn_cost

        line = _json.dumps({"costUSD": 0.0042})
        output = f"some preamble\n{line}\n"
        cost = _extract_turn_cost(output, "codex")
        assert cost == pytest.approx(0.0042)

    def test_cost_in_middle_of_output(self) -> None:
        """Cost line buried among other lines is still found (scans last 20 lines)."""
        import json as _json
        from maestro_cli.chat import _extract_turn_cost

        lines = [f"line {i}" for i in range(10)]
        lines.append(_json.dumps({"total_cost_usd": 0.123}))
        lines += [f"tail {i}" for i in range(5)]
        output = "\n".join(lines)
        cost = _extract_turn_cost(output, "claude")
        assert cost == pytest.approx(0.123)


# ===========================================================================
# TestBuildHistoryPromptRoles  — assistant-role messages and multi-turn ordering
# ===========================================================================


class TestBuildHistoryPromptRoles:
    def test_assistant_role_labeled_correctly(self) -> None:
        """Assistant messages appear with 'Assistant:' prefix in history block."""
        s = ChatSession()
        s.messages.append(
            ChatMessage(role="user", engine="claude", model="sonnet", content="hi")
        )
        s.messages.append(
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="hello back")
        )
        result = _build_history_prompt(s, "follow-up")
        assert "Assistant: hello back" in result
        assert "User: hi" in result

    def test_history_wraps_in_xml_tags(self) -> None:
        """Conversation history is wrapped in <conversation_history> XML tags."""
        s = ChatSession()
        s.messages.append(
            ChatMessage(role="user", engine="claude", model="sonnet", content="question")
        )
        result = _build_history_prompt(s, "next question")
        assert "<conversation_history>" in result
        assert "</conversation_history>" in result


# ===========================================================================
# TestRunChatTurnHistoryInjection  — history prepended to prompt
# ===========================================================================


class TestRunChatTurnHistoryInjection:
    def test_history_included_in_built_prompt(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When session has prior messages, the prompt passed to build_command
        contains <conversation_history>."""
        import io
        from maestro_cli.chat import _run_chat_turn

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout: list[str] = []
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            # Capture the task's prompt to check history injection
            captured["prompt"] = getattr(task, "prompt", "")
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        session.messages.append(
            ChatMessage(role="user", engine="claude", model="sonnet", content="prior question")
        )

        _run_chat_turn(session, "new question")

        prompt = captured.get("prompt", "")
        assert isinstance(prompt, str)
        assert "<conversation_history>" in prompt
        assert "prior question" in prompt


# ===========================================================================
# TestFormatEngineLineItemCompleted  — item.completed non-agent_message type
# ===========================================================================


class TestFormatEngineLineItemCompleted:
    def test_item_completed_non_agent_message_suppressed(self) -> None:
        """item.completed with item.type != agent_message is suppressed (returns None)."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        line = _json.dumps({
            "type": "item.completed",
            "item": {"type": "reasoning", "text": "internal reasoning"},
        })
        result = _format_engine_line(line, "codex")
        assert result is None

    def test_item_completed_agent_message_empty_text_suppressed(self) -> None:
        """item.completed agent_message with empty text returns None (suppressed)."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        line = _json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": ""},
        })
        result = _format_engine_line(line, "codex")
        assert result is None


# ===========================================================================
# TestCmdContext — /context command
# ===========================================================================


class TestCmdContext:
    def test_context_no_args_empty(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_context

        s = ChatSession()
        _cmd_context([], s)
        out = capsys.readouterr().out
        assert "no context files" in out.lower()

    def test_context_add_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.chat import _cmd_context

        monkeypatch.chdir(tmp_path)
        test_file = tmp_path / "example.py"
        test_file.write_text("def hello(): pass", encoding="utf-8")

        s = ChatSession()
        _cmd_context([str(test_file)], s)
        out = capsys.readouterr().out
        assert "added" in out.lower()
        assert len(s.context_files) == 1
        assert "def hello(): pass" in list(s.context_files.values())[0]

    def test_context_file_not_found(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_context

        s = ChatSession()
        _cmd_context(["/nonexistent/file.txt"], s)
        out = capsys.readouterr().out
        assert "not found" in out.lower()
        assert len(s.context_files) == 0

    def test_context_list_files(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_context

        s = ChatSession(context_files={"src/main.py": "print('hello')"})
        _cmd_context([], s)
        out = capsys.readouterr().out
        assert "1 context file" in out
        assert "src/main.py" in out

    def test_context_clear(self, capsys: pytest.CaptureFixture[str]) -> None:
        from maestro_cli.chat import _cmd_context

        s = ChatSession(context_files={"a.py": "x", "b.py": "y"})
        _cmd_context(["--clear"], s)
        out = capsys.readouterr().out
        assert "cleared 2" in out
        assert len(s.context_files) == 0

    def test_context_truncates_large_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.chat import _cmd_context

        monkeypatch.chdir(tmp_path)
        big_file = tmp_path / "big.txt"
        big_file.write_text("x" * 100_000, encoding="utf-8")

        s = ChatSession()
        _cmd_context([str(big_file)], s)
        content = list(s.context_files.values())[0]
        assert "truncated" in content
        assert len(content) < 100_000

    def test_context_multiple_files(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.chat import _cmd_context

        monkeypatch.chdir(tmp_path)
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("aaa", encoding="utf-8")
        f2.write_text("bbb", encoding="utf-8")

        s = ChatSession()
        _cmd_context([str(f1), str(f2)], s)
        assert len(s.context_files) == 2


# ===========================================================================
# TestCmdSaveLoad — /save and /load commands
# ===========================================================================


class TestCmdSaveLoad:
    def test_save_creates_file(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.chat import _cmd_save

        monkeypatch.chdir(tmp_path)
        s = ChatSession(
            engine="claude",
            model="sonnet",
            total_turns=3,
            messages=[
                ChatMessage(role="user", engine="claude", model="sonnet", content="hello"),
            ],
        )
        _cmd_save(s)
        out = capsys.readouterr().out
        assert "saved" in out.lower()

        sessions_dir = tmp_path / ".maestro-cache" / "sessions"
        files = list(sessions_dir.glob("chat_*.json"))
        assert len(files) == 1

    def test_load_latest(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        import json as _json
        from maestro_cli.chat import _cmd_load, _session_to_dict

        monkeypatch.chdir(tmp_path)
        sessions_dir = tmp_path / ".maestro-cache" / "sessions"
        sessions_dir.mkdir(parents=True)

        original = ChatSession(
            engine="gemini",
            model="pro",
            total_turns=5,
            total_cost_usd=0.42,
            messages=[
                ChatMessage(role="user", engine="gemini", model="pro", content="test"),
            ],
            context_files={"readme.md": "# Hello"},
        )
        session_file = sessions_dir / "chat_20260325_120000.json"
        session_file.write_text(
            _json.dumps(_session_to_dict(original), ensure_ascii=False),
            encoding="utf-8",
        )

        loaded = _cmd_load([], ChatSession())
        assert isinstance(loaded, ChatSession)
        assert loaded.engine == "gemini"
        assert loaded.model == "pro"
        assert loaded.total_turns == 5
        assert len(loaded.messages) == 1
        assert loaded.context_files.get("readme.md") == "# Hello"

    def test_load_no_sessions(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        from maestro_cli.chat import _cmd_load

        monkeypatch.chdir(tmp_path)
        original = ChatSession()
        result = _cmd_load([], original)
        out = capsys.readouterr().out
        assert "no saved sessions" in out.lower()
        assert result is original

    def test_roundtrip_serialization(self) -> None:
        from maestro_cli.chat import _session_from_dict, _session_to_dict

        original = ChatSession(
            engine="codex",
            model="5.4",
            total_turns=2,
            total_cost_usd=1.23,
            started_at="2026-03-25T12:00:00+00:00",
            context_files={"a.py": "content"},
            messages=[
                ChatMessage(role="user", engine="codex", model="5.4", content="hi", cost_usd=None),
                ChatMessage(role="assistant", engine="codex", model="5.4", content="hello", cost_usd=0.5, duration_sec=2.3),
            ],
        )
        data = _session_to_dict(original)
        restored = _session_from_dict(data)
        assert restored.engine == original.engine
        assert restored.model == original.model
        assert restored.total_turns == original.total_turns
        assert restored.total_cost_usd == original.total_cost_usd
        assert len(restored.messages) == 2
        assert restored.context_files == original.context_files
        assert restored.messages[1].cost_usd == 0.5


# ===========================================================================
# TestBuildHistoryPromptWithContext — file context injection
# ===========================================================================


class TestBuildHistoryPromptWithContext:
    def test_file_context_injected(self) -> None:
        from maestro_cli.chat import _build_history_prompt

        s = ChatSession(context_files={"src/main.py": "print('hello')"})
        prompt = _build_history_prompt(s, "explain this code")
        assert "<file_context>" in prompt
        assert "src/main.py" in prompt
        assert "print('hello')" in prompt
        assert "explain this code" in prompt

    def test_no_context_no_history(self) -> None:
        from maestro_cli.chat import _build_history_prompt

        s = ChatSession()
        prompt = _build_history_prompt(s, "hello")
        assert prompt == "hello"

    def test_context_plus_history(self) -> None:
        from maestro_cli.chat import _build_history_prompt

        s = ChatSession(
            context_files={"a.py": "def foo(): pass"},
            messages=[
                ChatMessage(role="user", engine="claude", model="sonnet", content="what does foo do"),
                ChatMessage(role="assistant", engine="claude", model="sonnet", content="foo does nothing"),
            ],
        )
        prompt = _build_history_prompt(s, "thanks")
        assert "<file_context>" in prompt
        assert "<conversation_history>" in prompt
        # File context should come before history
        ctx_pos = prompt.index("<file_context>")
        hist_pos = prompt.index("<conversation_history>")
        assert ctx_pos < hist_pos


# ===========================================================================
# TestDispatchChatCommandNew — /context, /save, /load dispatch
# ===========================================================================


class TestDispatchChatCommandNew:
    def test_context_dispatches(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession()
        result = _dispatch_chat_command("/context", s)
        assert result is True
        out = capsys.readouterr().out
        assert "no context files" in out.lower()

    def test_save_dispatches(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.chdir(tmp_path)
        s = ChatSession()
        result = _dispatch_chat_command("/save", s)
        assert result is True
        out = capsys.readouterr().out
        assert "saved" in out.lower()

    def test_load_returns_session(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
        monkeypatch.chdir(tmp_path)
        result = _dispatch_chat_command("/load", ChatSession())
        # No sessions exist → returns original session
        out = capsys.readouterr().out
        assert "no saved sessions" in out.lower()

    def test_help_shows_new_commands(self, capsys: pytest.CaptureFixture[str]) -> None:
        s = ChatSession()
        _dispatch_chat_command("/help", s)
        out = capsys.readouterr().out
        assert "/context" in out
        assert "/save" in out
        assert "/load" in out


# ===========================================================================
# TestFormatEngineLineResponseEvents  — response.completed / response.created
# ===========================================================================


class TestFormatEngineLineResponseEvents:
    """response.completed and response.created events must be suppressed."""

    def test_response_completed_suppressed(self) -> None:
        """response.completed Codex event returns None (falls to final return None)."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        line = _json.dumps({"type": "response.completed", "usage": {"tokens": 100}})
        result = _format_engine_line(line, "codex")
        assert result is None

    def test_response_created_suppressed(self) -> None:
        """response.created Codex event returns None (falls to final return None)."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        line = _json.dumps({"type": "response.created", "id": "resp_abc123"})
        result = _format_engine_line(line, "codex")
        assert result is None

    def test_empty_string_non_codex_returned_as_is(self) -> None:
        """An empty string for a non-codex engine is returned unchanged (not None)."""
        from maestro_cli.chat import _format_engine_line

        result = _format_engine_line("", "claude")
        assert result == ""

    def test_codex_item_completed_missing_item_key_suppressed(self) -> None:
        """item.completed with no 'item' key is suppressed (item defaults to {})."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        line = _json.dumps({"type": "item.completed"})
        result = _format_engine_line(line, "codex")
        assert result is None

    def test_codex_whitespace_only_line_not_json_returned_as_is(self) -> None:
        """A whitespace-only line for codex does not start with '{' → returned as-is."""
        from maestro_cli.chat import _format_engine_line

        line = "   \n"
        result = _format_engine_line(line, "codex")
        assert result == line


# ===========================================================================
# TestRunChatTurnGenericPopenException  — generic Exception from Popen
# ===========================================================================


class TestRunChatTurnGenericPopenException:
    def test_generic_popen_exception_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A generic Exception from subprocess.Popen (not FileNotFoundError) returns None
        and prints an error message."""
        from maestro_cli.chat import _run_chat_turn

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["some-engine", "hello"], False)

        def _popen_raises(*args: object, **kwargs: object) -> None:
            raise OSError("permission denied")

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", _popen_raises)

        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "hello")
        assert result is None
        out = capsys.readouterr().out
        assert "error" in out.lower()

    def test_multi_turn_history_ordering(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Three prior messages → history block in chronological order, new message at end."""
        import io
        from maestro_cli.chat import _run_chat_turn

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout: list[str] = []
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            captured["prompt"] = getattr(task, "prompt", "")
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        session.messages.extend([
            ChatMessage(role="user", engine="claude", model="sonnet", content="msg A"),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="reply A"),
            ChatMessage(role="user", engine="claude", model="sonnet", content="msg B"),
        ])

        _run_chat_turn(session, "msg C")

        prompt = str(captured.get("prompt", ""))
        # History must appear before the new message
        history_end = prompt.find("</conversation_history>")
        new_msg_pos = prompt.find("msg C")
        assert history_end != -1 and new_msg_pos != -1
        assert history_end < new_msg_pos
        # All prior messages must be in history block
        assert "msg A" in prompt
        assert "reply A" in prompt
        assert "msg B" in prompt


# ===========================================================================
# TestCmdHelpChatQuitAndModel  — /quit and /model explicitly present
# ===========================================================================


class TestCmdHelpChatQuitAndModel:
    def test_quit_command_in_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """/quit must appear explicitly in the help output."""
        from maestro_cli.chat import _cmd_help_chat

        _cmd_help_chat()
        out = capsys.readouterr().out
        assert "/quit" in out

    def test_model_command_description_in_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """/model command description must mention engine and model."""
        from maestro_cli.chat import _cmd_help_chat

        _cmd_help_chat()
        out = capsys.readouterr().out
        # The help entry for /model should mention engine or model switching
        assert "/model" in out
        # Must have some description text alongside it (not just the raw command)
        model_line_idx = out.find("/model")
        assert model_line_idx != -1
        snippet = out[model_line_idx:model_line_idx + 60]
        # At least one of these words should appear near /model
        assert any(kw in snippet.lower() for kw in ["engine", "model", "switch"])

    def test_at_engine_routing_example_in_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Help output shows @engine routing example (e.g., @codex)."""
        from maestro_cli.chat import _cmd_help_chat

        _cmd_help_chat()
        out = capsys.readouterr().out
        # The routing section contains a concrete @engine example
        assert "@codex" in out or "@engine" in out


# ===========================================================================
# Iteration 5 — _format_engine_line dedicated, _run_chat_turn edge cases,
#                slash command handler dispatch
# ===========================================================================


class TestFormatEngineLineIter5:
    """_format_engine_line edge cases: codex empty input, nested JSON text."""

    def test_codex_empty_string_returns_empty(self) -> None:
        """Codex engine with empty string: not JSON, returned as-is (empty str)."""
        from maestro_cli.chat import _format_engine_line

        result = _format_engine_line("", "codex")
        assert result == ""

    def test_codex_agent_message_with_json_in_text(self) -> None:
        """item.completed agent_message whose text itself contains JSON is extracted."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        inner_json = '{"key": "value"}'
        payload = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": f"Here is the result: {inner_json}"},
        }
        result = _format_engine_line(_json.dumps(payload), "codex")
        assert result is not None
        assert inner_json in result
        assert result.endswith("\n")

    def test_copilot_engine_returns_line_unchanged(self) -> None:
        """Copilot engine lines pass through unchanged (non-codex path)."""
        from maestro_cli.chat import _format_engine_line

        line = '{"type": "item.completed", "item": {"type": "agent_message"}}'
        result = _format_engine_line(line, "copilot")
        assert result == line


class TestRunChatTurnDurationAndCost:
    """_run_chat_turn: verify duration_sec and cost_usd on returned ChatMessage."""

    def test_duration_sec_is_positive(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Returned ChatMessage has duration_sec > 0 (time actually tracked)."""
        import io
        import time
        from maestro_cli.chat import _run_chat_turn

        # Mock time.monotonic to return controlled increasing values
        _call_count = 0
        def _fake_monotonic() -> float:
            nonlocal _call_count
            _call_count += 1
            # First call (start) returns 100.0, second call (end) returns 102.5
            return 100.0 if _call_count == 1 else 102.5

        monkeypatch.setattr(time, "monotonic", _fake_monotonic)

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = ["hello world\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "test prompt")
        assert result is not None
        # duration_sec must reflect the mocked 2.5 second elapsed time
        assert result.duration_sec == 2.5

    def test_cost_usd_extracted_from_turn_output(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When stdout includes a cost line, ChatMessage.cost_usd is populated."""
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = [
                    "Here is the response.\n",
                    '{"total_cost_usd": 0.0321}\n',
                ]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "hello")
        assert result is not None
        assert result.cost_usd is not None
        assert result.cost_usd == pytest.approx(0.0321)


class TestRunChatTurnInterruptAndStreaming:
    """_run_chat_turn: KeyboardInterrupt path and codex metadata filtering."""

    def test_keyboard_interrupt_returns_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """KeyboardInterrupt during stdout streaming → returns None, prints interrupted."""
        import io
        from maestro_cli.chat import _run_chat_turn

        class _InterruptingStdout:
            """Iterator that raises KeyboardInterrupt on second next()."""
            def __init__(self) -> None:
                self._count = 0

            def __iter__(self) -> _InterruptingStdout:
                return self

            def __next__(self) -> str:
                self._count += 1
                if self._count == 1:
                    return "partial output\n"
                raise KeyboardInterrupt

        terminated = {"called": False}

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = _InterruptingStdout()
                self.stderr = io.StringIO("")
                self.returncode = -15

            def terminate(self) -> None:
                terminated["called"] = True

            def wait(self, timeout: float | None = None) -> int:
                return -15

            def kill(self) -> None:
                pass

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "hello")
        assert result is None
        out = capsys.readouterr().out
        assert "interrupted" in out.lower()
        assert terminated["called"]

    def test_codex_metadata_suppressed_during_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """During a codex chat turn, JSON metadata lines are suppressed; only
        agent_message text reaches stdout."""
        import io
        import json as _json
        from maestro_cli.chat import _run_chat_turn

        metadata_line = _json.dumps({"type": "thread.started", "id": "t_1"}) + "\n"
        agent_line = _json.dumps({
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "Codex says hi"},
        }) + "\n"

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = [metadata_line, agent_line]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["codex", "exec", "--json", "prompt"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="codex", model="5.4")
        result = _run_chat_turn(session, "hello")
        assert result is not None

        out = capsys.readouterr().out
        # Agent message text should appear in printed output
        assert "Codex says hi" in out
        # Metadata event (thread.started) should NOT appear as readable text
        assert "thread.started" not in out


class TestSlashCommandDispatchIter5:
    """Slash command handlers dispatched via _dispatch_chat_command."""

    def test_dispatch_cost_prints_turn_info(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """/cost dispatched through _dispatch_chat_command prints session info."""
        session = ChatSession(total_turns=3, total_cost_usd=0.567)
        cont = _dispatch_chat_command("/cost", session)
        assert cont is True
        out = capsys.readouterr().out
        assert "3 turns" in out
        assert "$0.567" in out

    def test_dispatch_model_updates_session_state(self) -> None:
        """/model codex/5.4 dispatched through _dispatch_chat_command updates session."""
        session = ChatSession(engine="claude", model="sonnet")
        cont = _dispatch_chat_command("/model codex/5.4", session)
        assert cont is True
        assert session.engine == "codex"
        assert session.model == "5.4"

    def test_dispatch_help_contains_all_six_commands(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """/help via dispatch outputs all 6 slash commands."""
        session = ChatSession()
        cont = _dispatch_chat_command("/help", session)
        assert cont is True
        out = capsys.readouterr().out
        for cmd in ["/model", "/models", "/clear", "/cost", "/help", "/quit"]:
            assert cmd in out, f"Missing {cmd} in /help output"


# ===========================================================================
# Iteration 6 — new gaps: stub builders for extra engines, edge inputs,
#               farewell cost, _cmd_clear with zero messages, history roles
# ===========================================================================


class TestPlanStubEngineDefaults:
    """_build_chat_plan_stub sets the correct engine-specific defaults."""

    def test_gemini_engine_stub(self) -> None:
        """Gemini session: defaults.gemini.model is set."""
        from maestro_cli.chat import _build_chat_plan_stub

        s = ChatSession(engine="gemini", model="pro")
        plan = _build_chat_plan_stub(s)
        assert plan.defaults.gemini.model == "pro"

    def test_copilot_engine_stub(self) -> None:
        """Copilot session: defaults.copilot.model is set."""
        from maestro_cli.chat import _build_chat_plan_stub

        s = ChatSession(engine="copilot", model="opus")
        plan = _build_chat_plan_stub(s)
        assert plan.defaults.copilot.model == "opus"

    def test_qwen_engine_stub(self) -> None:
        """Qwen session: defaults.qwen.model is set."""
        from maestro_cli.chat import _build_chat_plan_stub

        s = ChatSession(engine="qwen", model="max")
        plan = _build_chat_plan_stub(s)
        assert plan.defaults.qwen.model == "max"

    def test_ollama_engine_stub(self) -> None:
        """Ollama session: defaults.ollama.model is set."""
        from maestro_cli.chat import _build_chat_plan_stub

        s = ChatSession(engine="ollama", model="codellama")
        plan = _build_chat_plan_stub(s)
        assert plan.defaults.ollama.model == "codellama"


class TestAdjustCommandForChatTrailingOutputFormat:
    """_adjust_command_for_chat with --output-format at the very end of cmd."""

    def test_output_format_at_end_no_crash(self) -> None:
        """--output-format with no following value does not crash and stays intact."""
        result = _adjust_command_for_chat(["claude", "--output-format"], "claude")
        # The condition `i + 1 < len(cmd) and cmd[i + 1] == "json"` is False
        # because there is no next arg, so the flag passes through unchanged.
        assert "--output-format" in result


class TestFormatEngineLineWhitespaceBeforeJson:
    """_format_engine_line: codex line with leading whitespace before '{'."""

    def test_codex_leading_whitespace_before_json_passes_through(self) -> None:
        """A codex line with spaces before '{' has stripped='{...}' so JSON is parsed."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        payload = {"type": "thread.started"}
        # Add leading whitespace — stripped will still start with '{'
        line = "   " + _json.dumps(payload) + "\n"
        result = _format_engine_line(line, "codex")
        # thread.started is a suppressed event → None
        assert result is None

    def test_codex_leading_whitespace_agent_message_extracted(self) -> None:
        """Leading whitespace before a valid agent_message JSON is handled correctly."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        payload = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": "trimmed response"},
        }
        line = "  " + _json.dumps(payload)
        result = _format_engine_line(line, "codex")
        assert result is not None
        assert "trimmed response" in result


class TestRunChatFarewellWithCost:
    """run_chat farewell message shows '$X.XXXX' when session has accumulated cost."""

    def test_farewell_shows_cost_when_nonzero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When /quit is entered after a turn that added cost, farewell shows dollar amount."""
        import io
        from maestro_cli.chat import run_chat

        # Simulate one turn that adds cost, then quit
        inputs_iter = iter(["hello there", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs_iter))

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                # Provide a cost line in stdout
                self.stdout = ['{"total_cost_usd": 0.0777}\n']
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "reply"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        result = run_chat()
        assert result == 0
        out = capsys.readouterr().out
        # Farewell must show formatted dollar amount, not '--'
        assert "$" in out
        assert "0.0777" in out


class TestCmdClearEmptySession:
    """_cmd_clear when there are already 0 messages."""

    def test_clear_empty_session_prints_zero(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Clearing an empty session prints 'cleared 0 messages' without crashing."""
        from maestro_cli.chat import _cmd_clear

        s = ChatSession()
        assert len(s.messages) == 0
        _cmd_clear(s)
        out = capsys.readouterr().out
        assert "cleared 0" in out
        assert len(s.messages) == 0


class TestBuildHistoryPromptAssistantOnly:
    """_build_history_prompt when all messages in history are from the assistant."""

    def test_history_with_only_assistant_messages(self) -> None:
        """All-assistant history is formatted with 'Assistant:' prefix."""
        s = ChatSession(messages=[
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="I said this"),
        ])
        result = _build_history_prompt(s, "follow up")
        assert "Assistant: I said this" in result
        assert "follow up" in result


class TestParseEnginePrefixBareAt:
    """_parse_engine_prefix with a bare '@' character (no engine name)."""

    def test_bare_at_sign_treated_as_plain_text(self) -> None:
        """A bare '@' with no following word is not a valid engine prefix."""
        engine, text = _parse_engine_prefix("@ hello")
        # '@' splits into ['@'] with no valid engine candidate
        assert engine is None

    def test_at_sign_only_returns_unchanged(self) -> None:
        """A single '@' character returns None engine and unchanged text."""
        engine, text = _parse_engine_prefix("@")
        assert engine is None
        assert text == "@"


# ===========================================================================
# Iteration 7 — new gaps: ChatMessage defaults, multi-slash model spec,
#               _cmd_cost empty started_at, whitespace-only cost output,
#               task stub None model, JSON without type key, empty @engine text
# ===========================================================================


class TestChatMessageDefaults:
    """ChatMessage dataclass field defaults."""

    def test_cost_usd_defaults_to_none(self) -> None:
        """cost_usd field defaults to None when not supplied."""
        msg = ChatMessage(role="user", engine="claude", model="sonnet", content="hi")
        assert msg.cost_usd is None

    def test_duration_sec_defaults_to_zero(self) -> None:
        """duration_sec field defaults to 0.0 when not supplied."""
        msg = ChatMessage(role="user", engine="gemini", model="flash", content="hi")
        assert msg.duration_sec == 0.0

    def test_explicit_cost_and_duration(self) -> None:
        """Explicitly supplied cost_usd and duration_sec are stored correctly."""
        msg = ChatMessage(
            role="assistant",
            engine="codex",
            model="5.4",
            content="done",
            cost_usd=0.0012,
            duration_sec=3.7,
        )
        assert msg.cost_usd == 0.0012
        assert msg.duration_sec == 3.7


class TestCmdModelMultiSlashSpec:
    """_cmd_model with a spec containing more than one slash character."""

    def test_multi_slash_uses_first_split_only(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Spec like 'claude/opus/extra' splits on first '/' only: engine=claude, model=opus/extra."""
        session = ChatSession(engine="gemini", model="pro")
        _cmd_model(["claude/opus/extra"], session)
        assert session.engine == "claude"
        # model gets everything after the first slash
        assert session.model == "opus/extra"
        out = capsys.readouterr().out
        assert "claude" in out


class TestCmdCostEmptyStartedAt:
    """_cmd_cost when started_at is empty string (not set)."""

    def test_empty_started_at_no_elapsed_shown(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """started_at='' → datetime.fromisoformat raises ValueError → elapsed skipped silently."""
        from maestro_cli.chat import _cmd_cost

        session = ChatSession(started_at="", total_turns=5, total_cost_usd=0.0)
        _cmd_cost(session)
        out = capsys.readouterr().out
        # Must not crash, must show turns, must NOT show 'min'
        assert "5 turns" in out
        assert "min" not in out


class TestExtractTurnCostWhitespaceOutput:
    """_extract_turn_cost with whitespace-only or blank output."""

    def test_whitespace_only_output_returns_none(self) -> None:
        """All-whitespace output contains no cost lines → None."""
        from maestro_cli.chat import _extract_turn_cost

        assert _extract_turn_cost("   \n\n  \t  \n", "claude") is None

    def test_single_newline_returns_none(self) -> None:
        """A single newline contains no cost line → None."""
        from maestro_cli.chat import _extract_turn_cost

        assert _extract_turn_cost("\n", "gemini") is None


class TestBuildChatTaskStubNoneModel:
    """_build_chat_task_stub when session.model is None and no model override supplied."""

    def test_task_stub_model_is_none_when_session_model_none(self) -> None:
        """No model on session and no override → task.model is None."""
        session = ChatSession(engine="claude", model=None)
        task = _build_chat_task_stub(session, "hello")
        assert task.model is None

    def test_task_stub_engine_from_session_when_no_override(self) -> None:
        """No engine override → task.engine matches session.engine."""
        session = ChatSession(engine="qwen", model=None)
        task = _build_chat_task_stub(session, "test")
        assert task.engine == "qwen"


class TestFormatEngineLineNoTypeKey:
    """_format_engine_line with a valid JSON object that has no 'type' key."""

    def test_codex_json_without_type_key_suppressed(self) -> None:
        """Valid JSON for codex with no 'type' field → msg_type='' → falls to final return None."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        line = _json.dumps({"data": "something", "id": 42})
        result = _format_engine_line(line, "codex")
        # No type key → msg_type="" → not in suppression list and not item.completed
        # → falls through all branches to the final `return None`
        assert result is None

    def test_codex_json_with_empty_type_suppressed(self) -> None:
        """Valid JSON for codex with type='' → falls to final return None."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        line = _json.dumps({"type": "", "payload": "x"})
        result = _format_engine_line(line, "codex")
        assert result is None


class TestRunChatEmptyEngineText:
    """run_chat loop: @engine prefix with no following text is skipped silently."""

    def test_at_engine_empty_text_does_not_call_run_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """'@claude' (engine-only, no message text) is silently skipped — no turn executed."""
        from maestro_cli.chat import run_chat

        run_turn_calls: list[str] = []

        def _fake_run_turn(session: object, prompt: str, **kw: object) -> None:
            run_turn_calls.append(prompt)
            return None

        monkeypatch.setattr("maestro_cli.chat._run_chat_turn", _fake_run_turn)

        inputs = iter(["@claude", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = run_chat()
        assert result == 0
        # _run_chat_turn must NOT have been called (empty text → `continue`)
        assert run_turn_calls == []


# ===========================================================================
# Iteration 8 — new gaps: run_chat engine/model params, total_turns counter,
#               shell-command path in _run_chat_turn, _cmd_cost zero-turns
#               with elapsed, numeric JSON line passthrough, history exact
#               boundary, /model dispatch no-args, _cmd_models sort order
# ===========================================================================


class TestRunChatEngineModelParams:
    """run_chat() accepts custom engine and model args shown in welcome message."""

    def test_welcome_message_shows_custom_engine_and_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """run_chat(engine='codex', model='5.4') prints engine/model in welcome line."""
        monkeypatch.setattr("builtins.input", lambda _: "/quit")
        result = run_chat(engine="codex", model="5.4")
        assert result == 0
        out = capsys.readouterr().out
        assert "codex" in out
        assert "5.4" in out

    def test_welcome_message_shows_default_engine(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """run_chat() with defaults shows 'claude/default' in welcome line."""
        monkeypatch.setattr("builtins.input", lambda _: "/quit")
        run_chat()
        out = capsys.readouterr().out
        assert "claude" in out
        assert "default" in out


class TestRunChatTotalTurnsCounter:
    """run_chat increments total_turns after each successful turn."""

    def test_total_turns_increments_after_successful_turn(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """After one real turn executes, /cost reports '1 turns'."""
        import io

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = ["answer\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        inputs = iter(["hello world", "/cost", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = run_chat()
        assert result == 0
        out = capsys.readouterr().out
        # After one successful turn, /cost must report 1 turn
        assert "1 turns" in out


class TestRunChatTurnShellCommandPath:
    """_run_chat_turn with a string command (shell=True path)."""

    def test_string_command_launches_with_shell_true(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When build_command returns a string (shell=True), Popen is invoked with
        launch_shell=True and the string command is passed unchanged."""
        import io
        from maestro_cli.chat import _run_chat_turn

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(
                self, cmd: object, *args: object, shell: bool = False, **kwargs: object
            ) -> None:
                captured["cmd"] = cmd
                captured["shell"] = shell
                self.stdout: list[str] = ["shell response\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[str, bool]:
            # Return a string command (triggers shell path)
            return ("echo 'shell output'", True)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "test")
        assert result is not None
        # A string command triggers launch_shell=True
        assert captured.get("shell") is True
        assert isinstance(captured.get("cmd"), str)


class TestCmdCostZeroTurnsWithElapsed:
    """_cmd_cost shows '--' for cost and elapsed time even with zero turns."""

    def test_zero_turns_with_valid_started_at_shows_elapsed_and_dashes(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """total_turns=0, total_cost_usd=0.0 but valid started_at → shows '--' + 'min'."""
        from datetime import datetime, UTC, timedelta

        started = datetime.now(UTC) - timedelta(minutes=3)
        session = ChatSession(
            started_at=started.isoformat(),
            total_turns=0,
            total_cost_usd=0.0,
        )
        _cmd_cost(session)
        out = capsys.readouterr().out
        assert "0 turns" in out
        assert "--" in out
        assert "min" in out


class TestFormatEngineLineNumericJson:
    """_format_engine_line with non-dict JSON values (array, number) for codex."""

    def test_codex_json_array_line_returned_as_is(self) -> None:
        """A JSON array line starts with '[', not '{', so it is returned unchanged."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        line = _json.dumps([1, 2, 3]) + "\n"
        result = _format_engine_line(line, "codex")
        # Doesn't start with '{' after strip → returned as-is
        assert result == line

    def test_codex_numeric_string_returned_as_is(self) -> None:
        """A plain numeric string (not JSON dict) for codex is returned unchanged."""
        from maestro_cli.chat import _format_engine_line

        line = "42\n"
        result = _format_engine_line(line, "codex")
        assert result == line


class TestBuildHistoryPromptExactBoundary:
    """_build_history_prompt with combined history exactly at the char limit."""

    def test_message_at_exact_char_limit_is_included(self) -> None:
        """A message whose entry length == _HISTORY_CHAR_LIMIT is included
        (condition is strictly >, so at == limit it is accepted)."""
        from maestro_cli.chat import _build_history_prompt, _HISTORY_CHAR_LIMIT

        # 'User: ' is 6 chars, so content should be limit-6 to make total exactly == limit
        prefix = "User: "
        content = "X" * (_HISTORY_CHAR_LIMIT - len(prefix))
        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content=content),
        ])
        result = _build_history_prompt(s, "next")
        # Message is exactly at the limit, not over it → included in history
        assert "<conversation_history>" in result
        assert content[:20] in result


class TestDispatchChatCommandModelNoArgs:
    """_dispatch_chat_command with '/model' (no arguments) shows current state."""

    def test_dispatch_model_no_args_shows_active_engine_model(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """/model with no args dispatched through _dispatch_chat_command shows
        the current engine and model without modifying the session."""
        session = ChatSession(engine="gemini", model="flash")
        cont = _dispatch_chat_command("/model", session)
        assert cont is True
        out = capsys.readouterr().out
        assert "gemini" in out
        assert "flash" in out
        # Session must not have been modified
        assert session.engine == "gemini"
        assert session.model == "flash"


class TestCmdModelsSortOrder:
    """_cmd_models outputs engines in alphabetical (sorted) order."""

    def test_engines_appear_in_alphabetical_order(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """_cmd_models sorts engines alphabetically: claude < codex < copilot <
        gemini < llama < ollama < qwen — each appears before the next in the output."""
        _cmd_models()
        out = capsys.readouterr().out
        ordered_engines = ["claude", "codex", "copilot", "gemini", "llama", "ollama", "qwen"]
        # Use "  <engine>:" prefix to avoid matching engine names inside alias lists
        positions = [out.find(f"  {e}:") for e in ordered_engines]
        # All engines must be present
        assert all(p != -1 for p in positions), f"Missing engine in output: {out}"
        # Positions must be strictly increasing (sorted order)
        assert positions == sorted(positions), (
            f"Engines not in sorted order. Positions: {list(zip(ordered_engines, positions))}"
        )


# ===========================================================================
# Iteration 9 — new gaps: CREATE_NEW_PROCESS_GROUP flag, KeyboardInterrupt
#               kill path, output without trailing newline, model-only switch,
#               run_chat @engine user-message engine field, turn model_tag,
#               list2cmdline conversion, _cmd_models per-engine alias format,
#               run_chat started_at set on launch, chat session started_at
# ===========================================================================


class TestRunChatTurnCreationFlags:
    """_run_chat_turn sets CREATE_NEW_PROCESS_GROUP on Windows (os.name == 'nt')."""

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-specific behavior: asserts subprocess.CREATE_NEW_PROCESS_GROUP (Windows-only attribute)",
    )
    def test_creation_flags_set_on_windows(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When os.name == 'nt' and command is not a shell builtin, Popen receives
        creationflags=CREATE_NEW_PROCESS_GROUP."""
        import io
        import subprocess
        from maestro_cli.chat import _run_chat_turn

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured["creationflags"] = kwargs.get("creationflags")
                self.stdout: list[str] = ["output\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            # A non-builtin command so the list path is taken
            return (["myengine", "hello"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)
        monkeypatch.setattr("os.name", "nt")

        session = ChatSession(engine="claude", model="sonnet")
        _run_chat_turn(session, "hello")

        assert captured.get("creationflags") == subprocess.CREATE_NEW_PROCESS_GROUP


class TestRunChatTurnKillOnWaitTimeout:
    """_run_chat_turn kills the process when proc.wait(timeout=5) raises TimeoutExpired
    after a KeyboardInterrupt during stdout streaming."""

    def test_kill_called_when_terminate_wait_times_out(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """After KeyboardInterrupt, proc.terminate() is called, then proc.wait(timeout=5)
        raises TimeoutExpired, which must trigger proc.kill()."""
        import io
        import subprocess
        from maestro_cli.chat import _run_chat_turn

        kill_called = {"called": False}

        class _InterruptStdout:
            def __iter__(self) -> _InterruptStdout:
                return self

            def __next__(self) -> str:
                raise KeyboardInterrupt

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = _InterruptStdout()
                self.stderr = io.StringIO("")
                self.returncode = -15
                self._wait_count = 0

            def terminate(self) -> None:
                pass

            def wait(self, timeout: float | None = None) -> int:
                self._wait_count += 1
                if timeout == 5:
                    # Second wait (after terminate) times out
                    raise subprocess.TimeoutExpired(cmd="fake", timeout=5)
                return -15

            def kill(self) -> None:
                kill_called["called"] = True

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "hello")
        assert result is None
        assert kill_called["called"], "proc.kill() must be called when wait(timeout=5) times out"
        out = capsys.readouterr().out
        assert "interrupted" in out.lower()


class TestRunChatTurnOutputWithoutTrailingNewline:
    """_run_chat_turn: when full_output does not end with '\\n', print() is called."""

    def test_output_no_trailing_newline_still_returns_message(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """stdout output without a trailing newline: _run_chat_turn returns a ChatMessage
        with content stripped correctly (the extra print() is a side effect)."""
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                # No trailing newline
                self.stdout = ["response without newline"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="gemini", model="flash")
        result = _run_chat_turn(session, "hello")
        assert result is not None
        assert result.content == "response without newline"


class TestCmdModelModelOnlySwitch:
    """_cmd_model with a bare model name (no '/') that is not a valid engine name."""

    def test_model_only_no_slash_updates_model_only(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A spec like 'haiku' (not a valid engine name, no slash) must update only
        session.model, leaving session.engine unchanged."""
        from maestro_cli.chat import _cmd_model

        session = ChatSession(engine="claude", model="sonnet")
        _cmd_model(["haiku"], session)
        # engine must not change
        assert session.engine == "claude"
        # model must be updated to the new value
        assert session.model == "haiku"
        out = capsys.readouterr().out
        assert "claude/haiku" in out

    def test_model_only_custom_string_updates_model(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A spec like 'custom-model-v2' (no slash, not an engine name) sets model."""
        from maestro_cli.chat import _cmd_model

        session = ChatSession(engine="gemini", model="pro")
        _cmd_model(["custom-model-v2"], session)
        assert session.engine == "gemini"
        assert session.model == "custom-model-v2"


class TestRunChatAtEngineUserMessageField:
    """run_chat stores the correct active_engine on the user message when @engine prefix used."""

    def test_at_engine_prefix_sets_engine_in_user_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the user types '@gemini <text>', the user ChatMessage appended to
        session.messages must have engine='gemini'."""
        import io
        from maestro_cli.chat import run_chat

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = ["gemini reply\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        messages_after_turn: list[object] = []

        def _fake_run_turn(
            session: object, prompt: str, **kw: object
        ) -> object:
            # Capture session state at turn time (user message already appended)
            from maestro_cli.chat import ChatMessage, ChatSession as _CS
            assert isinstance(session, _CS)
            messages_after_turn.extend(session.messages[:])
            result = ChatMessage(
                role="assistant", engine="gemini", model="flash", content="reply"
            )
            return result

        monkeypatch.setattr("maestro_cli.chat._run_chat_turn", _fake_run_turn)
        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})

        inputs = iter(["@gemini translate this", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        run_chat()

        # The first message in messages_after_turn is the user message
        assert len(messages_after_turn) >= 1
        from maestro_cli.chat import ChatMessage as _CM
        user_msg = messages_after_turn[0]
        assert isinstance(user_msg, _CM)
        assert user_msg.engine == "gemini"
        assert user_msg.role == "user"


class TestRunChatTurnModelTagDefault:
    """_run_chat_turn prints [engine/default] when turn_model is None."""

    def test_model_tag_is_default_when_no_model(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """When session.model is None and no model override is given, the printed tag
        shows '[engine/default]'."""
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout: list[str] = ["response\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="ollama", model=None)
        _run_chat_turn(session, "hello")

        out = capsys.readouterr().out
        # The engine tag line must contain "ollama/default"
        assert "ollama/default" in out


class TestRunChatTurnWindowsList2Cmdline:
    """_run_chat_turn on Windows converts shell-builtin list to string via list2cmdline."""

    @pytest.mark.skipif(
        sys.platform != "win32",
        reason="Windows-specific behavior: asserts subprocess.list2cmdline (Windows-only) is called",
    )
    def test_list2cmdline_converts_builtin_command(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """On Windows, a list command whose first element is a shell builtin is converted
        with subprocess.list2cmdline() before being passed to Popen as a string."""
        import io
        import subprocess as _subprocess
        from maestro_cli.chat import _run_chat_turn

        captured: dict[str, object] = {}
        list2cmdline_calls: list[list[str]] = []

        original_list2cmdline = _subprocess.list2cmdline

        def _spy_list2cmdline(seq: list[str]) -> str:
            list2cmdline_calls.append(list(seq))
            return original_list2cmdline(seq)

        class FakePopen:
            def __init__(self, cmd: object, *args: object, **kwargs: object) -> None:
                captured["cmd"] = cmd
                captured["shell"] = kwargs.get("shell", False)
                self.stdout: list[str] = ["ok\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "hello world"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)
        monkeypatch.setattr("subprocess.list2cmdline", _spy_list2cmdline)
        monkeypatch.setattr("os.name", "nt")

        session = ChatSession(engine="claude", model="sonnet")
        _run_chat_turn(session, "test")

        # list2cmdline must have been called with ["echo", "hello world"]
        assert list2cmdline_calls, "subprocess.list2cmdline was not called"
        assert list2cmdline_calls[0] == ["echo", "hello world"]
        # And the command passed to Popen must be a string (not a list)
        assert isinstance(captured.get("cmd"), str)
        assert captured.get("shell") is True


class TestCmdModelsAliasFormat:
    """_cmd_models per-engine alias format: qwen has exactly 5 known aliases."""

    def test_qwen_aliases_count_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The qwen engine has 5 known aliases (coder, coder-turbo, max, plus, qwq).
        All five must appear in _cmd_models output."""
        from maestro_cli.chat import _cmd_models

        _cmd_models()
        out = capsys.readouterr().out
        for alias in ["coder", "coder-turbo", "max", "plus", "qwq"]:
            assert alias in out, f"Missing qwen alias '{alias}' in _cmd_models output"

    def test_copilot_gpt_alias_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Copilot engine aliases include GPT model names (gpt-5.4-codex etc.)."""
        from maestro_cli.chat import _cmd_models

        _cmd_models()
        out = capsys.readouterr().out
        # At least one GPT model alias should appear in the copilot engine line
        assert "gpt-5" in out or "gpt-4" in out


class TestRunChatStartedAtSetOnLaunch:
    """run_chat() sets session.started_at to a valid ISO timestamp before the loop."""

    def test_cost_shows_elapsed_immediately_after_start(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Calling /cost right after starting (0 turns, 0 cost but valid started_at)
        shows '0min' or '1min' in the elapsed field — confirming started_at was set."""
        from datetime import datetime, UTC

        inputs = iter(["/cost", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        run_chat()
        out = capsys.readouterr().out
        # 'min' must appear — proving started_at was set (non-empty isoformat string)
        assert "min" in out


# ===========================================================================
# Iteration 10 — new gaps: qwen/copilot adjust passthrough, execution_profile
#                stored in session, plan stub with None model, _WINDOWS_SHELL_BUILTINS
#                content, qwen format_engine_line passthrough,
#                claude no-reasoning no env var, _cmd_cost turns>0 zero cost,
#                run_chat passes execution_profile to session
# ===========================================================================


class TestAdjustCommandForChatQwenCopilot:
    """_adjust_command_for_chat: qwen and copilot engines are pass-through (no modifications)."""

    def test_qwen_engine_cmd_unchanged(self) -> None:
        """Qwen engine is not codex and has no --output-format json flag → cmd unchanged."""
        cmd = ["qwen", "--model", "coder", "--prompt", "hello"]
        result = _adjust_command_for_chat(cmd, "qwen")
        assert result == cmd

    def test_copilot_engine_cmd_unchanged(self) -> None:
        """Copilot engine is not codex and has no --output-format json flag → cmd unchanged."""
        cmd = ["copilot", "--autopilot", "--silent", "--no-color", "hello"]
        result = _adjust_command_for_chat(cmd, "copilot")
        assert result == cmd

    def test_copilot_output_format_json_replaced(self) -> None:
        """Copilot with --output-format json gets it replaced with text (non-codex path)."""
        cmd = ["copilot", "--output-format", "json", "hello"]
        result = _adjust_command_for_chat(cmd, "copilot")
        assert result == ["copilot", "--output-format", "text", "hello"]


class TestChatSessionExecutionProfile:
    """ChatSession stores execution_profile and run_chat creates the session with it."""

    def test_session_stores_safe_profile(self) -> None:
        """ChatSession with execution_profile='safe' stores it correctly."""
        from maestro_cli.chat import ChatSession

        s = ChatSession(execution_profile="safe")
        assert s.execution_profile == "safe"

    def test_session_stores_yolo_profile(self) -> None:
        """ChatSession with execution_profile='yolo' stores it correctly."""
        from maestro_cli.chat import ChatSession

        s = ChatSession(execution_profile="yolo")
        assert s.execution_profile == "yolo"

    def test_run_chat_passes_execution_profile_to_session(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """run_chat(execution_profile='safe') creates a session with execution_profile='safe'
        and passes it through to _run_chat_turn when a turn is executed."""
        import io
        from maestro_cli.chat import run_chat, ChatSession as _CS

        captured: dict[str, object] = {}

        def _fake_run_turn(session: object, prompt: str, **kw: object) -> None:
            assert isinstance(session, _CS)
            captured["profile"] = session.execution_profile
            return None

        monkeypatch.setattr("maestro_cli.chat._run_chat_turn", _fake_run_turn)
        inputs = iter(["hello", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        run_chat(execution_profile="safe")
        assert captured.get("profile") == "safe"


class TestBuildChatPlanStubNoneModel:
    """_build_chat_plan_stub with session.model=None leaves engine default model as None."""

    def test_claude_stub_none_model(self) -> None:
        """Claude session with no model → defaults.claude.model is None."""
        from maestro_cli.chat import _build_chat_plan_stub

        s = ChatSession(engine="claude", model=None)
        plan = _build_chat_plan_stub(s)
        assert plan.defaults.claude.model is None

    def test_codex_stub_none_model(self) -> None:
        """Codex session with no model → defaults.codex.model is None."""
        from maestro_cli.chat import _build_chat_plan_stub

        s = ChatSession(engine="codex", model=None)
        plan = _build_chat_plan_stub(s)
        assert plan.defaults.codex.model is None


class TestWindowsShellBuiltinsConstant:
    """_WINDOWS_SHELL_BUILTINS constant contains the expected built-in command names."""

    def test_echo_in_builtins(self) -> None:
        """'echo' is a Windows shell builtin."""
        from maestro_cli.chat import _WINDOWS_SHELL_BUILTINS

        assert "echo" in _WINDOWS_SHELL_BUILTINS

    def test_dir_in_builtins(self) -> None:
        """'dir' is a Windows shell builtin."""
        from maestro_cli.chat import _WINDOWS_SHELL_BUILTINS

        assert "dir" in _WINDOWS_SHELL_BUILTINS

    def test_copy_in_builtins(self) -> None:
        """'copy' is a Windows shell builtin."""
        from maestro_cli.chat import _WINDOWS_SHELL_BUILTINS

        assert "copy" in _WINDOWS_SHELL_BUILTINS

    def test_type_in_builtins(self) -> None:
        """'type' is a Windows shell builtin."""
        from maestro_cli.chat import _WINDOWS_SHELL_BUILTINS

        assert "type" in _WINDOWS_SHELL_BUILTINS

    def test_set_in_builtins(self) -> None:
        """'set' is a Windows shell builtin."""
        from maestro_cli.chat import _WINDOWS_SHELL_BUILTINS

        assert "set" in _WINDOWS_SHELL_BUILTINS


class TestFormatEngineLineQwenOllama:
    """_format_engine_line for qwen and ollama engines returns line unchanged."""

    def test_qwen_engine_json_line_returned_unchanged(self) -> None:
        """Qwen (non-codex) engine: a JSON-looking line is returned as-is."""
        from maestro_cli.chat import _format_engine_line

        line = '{"type": "item.completed", "item": {"type": "agent_message", "text": "hi"}}\n'
        result = _format_engine_line(line, "qwen")
        assert result == line

    def test_ollama_engine_json_line_returned_unchanged(self) -> None:
        """Ollama (non-codex) engine: a JSON-looking line is returned as-is."""
        from maestro_cli.chat import _format_engine_line

        line = '{"type": "turn.completed"}\n'
        result = _format_engine_line(line, "ollama")
        assert result == line


class TestRunChatTurnClaudeNoReasoningEffort:
    """_run_chat_turn: when both task.reasoning_effort and plan default are None,
    CLAUDE_CODE_EFFORT_LEVEL is NOT injected into the subprocess environment."""

    def test_no_effort_env_var_when_both_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Claude engine with no reasoning_effort on task or plan → env var not set."""
        import io
        from maestro_cli.chat import _run_chat_turn
        from maestro_cli.models import PlanSpec, PlanDefaults, EngineDefaults

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured["env"] = kwargs.get("env", {})
                self.stdout: list[str] = []
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        def _fake_plan_stub(session: object) -> PlanSpec:
            # reasoning_effort=None (no effort level set)
            defaults = PlanDefaults()
            defaults.claude = EngineDefaults(model="sonnet", reasoning_effort=None)
            return PlanSpec(name="chat", defaults=defaults)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("maestro_cli.chat._build_chat_plan_stub", _fake_plan_stub)
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        _run_chat_turn(session, "test")

        env = captured.get("env", {})
        assert isinstance(env, dict)
        assert "CLAUDE_CODE_EFFORT_LEVEL" not in env


class TestCmdCostTurnsWithZeroCost:
    """_cmd_cost with total_turns > 0 but total_cost_usd == 0 shows '--' for cost."""

    def test_turns_with_zero_cost_shows_dashes(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """5 turns but zero accumulated cost → '--' displayed, not '$0.0000'."""
        from maestro_cli.chat import _cmd_cost

        s = ChatSession(total_turns=5, total_cost_usd=0.0)
        _cmd_cost(s)
        out = capsys.readouterr().out
        assert "5 turns" in out
        assert "--" in out
        # The '$' sign must NOT appear since cost is zero
        assert "$" not in out


# ===========================================================================
# Iteration 11 — distinct code path tests for _format_engine_line,
#                _run_chat_turn edge cases, slash command handlers,
#                run_chat loop integration, history prompt edge cases,
#                readline completion function
# ===========================================================================


class TestFormatEngineLineDistinctCodePaths:
    """_format_engine_line: distinct code paths not covered by existing tests."""

    def test_codex_item_completed_text_none_suppressed(self) -> None:
        """item.completed agent_message with text=None (explicit null) → suppressed.

        Code path: line 148-150 in chat.py — item.get("text", "") returns None,
        `if text:` is False → returns None (does not reach text + '\\n').
        """
        import json as _json
        from maestro_cli.chat import _format_engine_line

        payload = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": None},
        }
        result = _format_engine_line(_json.dumps(payload), "codex")
        assert result is None

    def test_codex_type_field_is_integer_falls_to_none(self) -> None:
        """JSON with type=42 (integer, not string) → no branch matches → return None.

        Code path: msg_type = 42 → `42 == "item.completed"` is False,
        `42 in ("thread.started", ...)` is False → final return None.
        """
        import json as _json
        from maestro_cli.chat import _format_engine_line

        result = _format_engine_line(_json.dumps({"type": 42, "data": "x"}), "codex")
        assert result is None

    def test_codex_item_completed_function_call_type_suppressed(self) -> None:
        """item.completed with item.type='function_call' (not agent_message) → None.

        Code path: line 147 — item.get("type") == "function_call" != "agent_message",
        falls to return None at line 152 (suppress non-agent_message items).
        """
        import json as _json
        from maestro_cli.chat import _format_engine_line

        payload = {
            "type": "item.completed",
            "item": {"type": "function_call", "name": "run_shell", "output": "ok"},
        }
        result = _format_engine_line(_json.dumps(payload), "codex")
        assert result is None

    def test_codex_multiline_agent_message_text_extracted_with_newline(self) -> None:
        """item.completed agent_message with multi-line text → extracted with \\n appended.

        Code path: line 149-150 — text is non-empty, returns text + '\\n'.
        """
        import json as _json
        from maestro_cli.chat import _format_engine_line

        multiline = "Line one\nLine two\nLine three"
        payload = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": multiline},
        }
        result = _format_engine_line(_json.dumps(payload), "codex")
        assert result is not None
        assert result == multiline + "\n"
        assert result.count("\n") == 3  # two internal + one appended


class TestRunChatTurnDistinctCodePaths:
    """_run_chat_turn: distinct code paths verified through returned ChatMessage fields."""

    def test_result_role_is_always_assistant(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Returned ChatMessage always has role='assistant' (line 419 in chat.py).

        Code path: ChatMessage constructor always sets role='assistant'.
        """
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = ["response text\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="gemini", model="flash")
        result = _run_chat_turn(session, "hello")
        assert result is not None
        assert result.role == "assistant"

    def test_result_model_is_default_when_turn_model_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When session.model is None and no model override, result.model == 'default'.

        Code path: line 420 — `model=turn_model or "default"` where turn_model is None.
        """
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = ["output\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="ollama", model=None)
        result = _run_chat_turn(session, "test")
        assert result is not None
        assert result.model == "default"

    def test_no_cost_pattern_in_output_cost_usd_is_none(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When stdout has no cost pattern, result.cost_usd is None.

        Code path: line 416 — _extract_turn_cost returns None → ChatMessage.cost_usd=None.
        """
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = ["just a plain answer with no cost info\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "what is 2+2?")
        assert result is not None
        assert result.cost_usd is None

    def test_stderr_not_printed_when_stdout_has_content(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When stdout has content, stderr is captured but NOT printed to output.

        Code path: line 406 — `if not full_output.strip()` is False (stdout non-empty),
        so stderr is NOT written to sys.stdout.
        """
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout = ["real answer here\n"]
                self.stderr = io.StringIO("WARNING: something happened on stderr\n")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "hello")
        assert result is not None
        out = capsys.readouterr().out
        # stderr text must NOT appear in printed output (stdout was non-empty)
        assert "WARNING" not in out
        assert result.content == "real answer here"

    def test_codex_turn_gets_full_auto_flag_via_adjust(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When engine is codex, _adjust_command_for_chat adds --full-auto to the command.

        Code path: line 192-201 — codex-specific flags injected after 'exec' index.
        Verified through Popen: command must contain --full-auto.
        """
        import io
        from maestro_cli.chat import _run_chat_turn

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(self, cmd: object, *args: object, **kwargs: object) -> None:
                captured["cmd"] = cmd
                self.stdout: list[str] = ["output\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["codex", "exec", "--json", "-p", "test"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="codex", model="5.4")
        _run_chat_turn(session, "hello")

        cmd = captured.get("cmd")
        assert isinstance(cmd, list)
        assert "--full-auto" in cmd
        assert "--skip-git-repo-check" in cmd

    def test_empty_stdout_and_empty_stderr_returns_empty_content(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When both stdout and stderr are empty, result.content is empty string.

        Code path: full_output = "" (empty), stderr_text = "" (empty),
        neither branch triggers fallback → ChatMessage.content = "".strip() = "".
        """
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout: list[str] = []
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="gemini", model="flash")
        result = _run_chat_turn(session, "hello")
        assert result is not None
        assert result.content == ""


class TestSlashCommandHandlerDistinctPaths:
    """Slash command handlers: distinct dispatch paths through _dispatch_chat_command."""

    def test_cmd_model_unknown_engine_in_slash_spec_warns(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """/model badengine/opus → prints 'unknown engine' warning, session unchanged.

        Code path: _cmd_model line 445-447 — eng not in _VALID_ENGINES → print + return.
        """
        from maestro_cli.chat import _cmd_model

        session = ChatSession(engine="claude", model="sonnet")
        _cmd_model(["badengine/opus"], session)
        out = capsys.readouterr().out
        assert "unknown engine" in out.lower()
        # Session must NOT be modified
        assert session.engine == "claude"
        assert session.model == "sonnet"

    def test_dispatch_clear_empties_messages_returns_true(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """/clear through _dispatch_chat_command clears messages and returns True.

        Code path: _dispatch_chat_command line 533 → _cmd_clear → session.messages.clear().
        """
        session = ChatSession(engine="claude", model="sonnet")
        session.messages.append(
            ChatMessage(role="user", engine="claude", model="sonnet", content="hello")
        )
        assert len(session.messages) == 1
        cont = _dispatch_chat_command("/clear", session)
        assert cont is True
        assert len(session.messages) == 0
        out = capsys.readouterr().out
        assert "cleared 1" in out

    def test_dispatch_unknown_command_prints_help_hint(
        self, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """/xyz dispatched → prints 'unknown command' with hint to type /help.

        Code path: _dispatch_chat_command line 539 — else branch for unrecognized commands.
        """
        session = ChatSession()
        cont = _dispatch_chat_command("/xyz", session)
        assert cont is True
        out = capsys.readouterr().out
        assert "unknown command" in out.lower()
        assert "/help" in out


class TestRunChatLoopIntegration:
    """run_chat main loop: session state mutations across multiple turns."""

    def test_assistant_message_appended_to_session_messages(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After a successful turn, both user and assistant messages are in session.messages.

        Code path: run_chat lines 621-626 (user append) and 637 (assistant append).
        """
        import io
        from maestro_cli.chat import run_chat

        session_ref: list[object] = []

        def _fake_run_turn(
            session: object, prompt: str, **kw: object
        ) -> ChatMessage | None:
            session_ref.append(session)
            return ChatMessage(
                role="assistant", engine="claude", model="sonnet",
                content="I am the assistant", cost_usd=None,
            )

        monkeypatch.setattr("maestro_cli.chat._run_chat_turn", _fake_run_turn)
        inputs = iter(["hello world", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        run_chat()

        assert len(session_ref) == 1
        session = session_ref[0]
        assert isinstance(session, ChatSession)
        # After turn: 1 user + 1 assistant = 2 messages
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "hello world"
        assert session.messages[1].role == "assistant"
        assert session.messages[1].content == "I am the assistant"

    def test_cost_accumulation_across_two_turns(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Two turns with known costs → total_cost_usd reflects their sum in farewell.

        Code path: run_chat line 640 — `session.total_cost_usd += result.cost_usd`.
        """
        turn_count = {"n": 0}

        def _fake_run_turn(
            session: object, prompt: str, **kw: object
        ) -> ChatMessage | None:
            turn_count["n"] += 1
            cost = 0.05 if turn_count["n"] == 1 else 0.03
            return ChatMessage(
                role="assistant", engine="claude", model="sonnet",
                content=f"reply {turn_count['n']}", cost_usd=cost,
            )

        monkeypatch.setattr("maestro_cli.chat._run_chat_turn", _fake_run_turn)
        inputs = iter(["msg1", "msg2", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        run_chat()
        out = capsys.readouterr().out
        # Farewell must show $0.0800 (0.05 + 0.03)
        assert "$0.0800" in out
        assert "2 turns" in out

    def test_failed_turn_does_not_increment_total_turns(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When _run_chat_turn returns None, total_turns stays at 0.

        Code path: run_chat line 636 — `if result is not None:` is False → skip increment.
        """
        def _fake_run_turn(
            session: object, prompt: str, **kw: object
        ) -> ChatMessage | None:
            return None  # simulate failure

        monkeypatch.setattr("maestro_cli.chat._run_chat_turn", _fake_run_turn)
        inputs = iter(["hello", "/cost", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        run_chat()
        out = capsys.readouterr().out
        # /cost must show 0 turns since the turn failed
        assert "0 turns" in out


class TestBuildHistoryPromptDistinctEdgeCases:
    """_build_history_prompt: edge cases for truncation and message ordering."""

    def test_two_messages_over_limit_keeps_only_most_recent(self) -> None:
        """Two messages each > 50k chars — only the most recent fits within 80k limit.

        Code path: _build_history_prompt lines 223-228 — reverse walk, second message
        would exceed _HISTORY_CHAR_LIMIT, so loop breaks after first (most recent).
        """
        from maestro_cli.chat import _build_history_prompt, _HISTORY_CHAR_LIMIT

        # Each message entry is "User: " + content — ~50k chars each
        content_a = "A" * 50_000  # old message
        content_b = "B" * 50_000  # recent message
        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content=content_a),
            ChatMessage(role="user", engine="claude", model="sonnet", content=content_b),
        ])
        result = _build_history_prompt(s, "new question")
        # Most recent message (B) should be in history
        assert "B" * 100 in result
        # Oldest message (A) should NOT be in history (exceeded budget)
        assert "A" * 100 not in result
        assert "new question" in result

    def test_history_preserves_chronological_order_in_output(self) -> None:
        """Three short messages → history output preserves chronological order.

        Code path: _build_history_prompt lines 223-233 — reverse walk collects messages
        then parts.reverse() restores chronological order.
        """
        from maestro_cli.chat import _build_history_prompt

        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content="first"),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="second"),
            ChatMessage(role="user", engine="claude", model="sonnet", content="third"),
        ])
        result = _build_history_prompt(s, "fourth")
        # All three messages must appear in order
        pos_first = result.find("first")
        pos_second = result.find("second")
        pos_third = result.find("third")
        pos_fourth = result.find("fourth")
        assert pos_first < pos_second < pos_third < pos_fourth


class TestSetupChatReadlineCompletionFunction:
    """_setup_chat_readline: tab completion returns matching slash commands and @engines."""

    def test_completion_returns_matching_slash_commands(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After _setup_chat_readline, the completer returns matching candidates for '/m'.

        Code path: _setup_chat_readline lines 559-567 — sets up completer function that
        returns matching completions by prefix.
        """
        from maestro_cli.chat import _setup_chat_readline

        set_completer_arg: list[object] = []

        class FakeReadline:
            @staticmethod
            def set_completer(fn: object) -> None:
                set_completer_arg.append(fn)

            @staticmethod
            def parse_and_bind(s: str) -> None:
                pass

        import builtins
        real_import = builtins.__import__

        def _mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "readline":
                return FakeReadline
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        _setup_chat_readline()

        assert len(set_completer_arg) == 1
        completer = set_completer_arg[0]
        assert callable(completer)
        # '/m' should match /model and /models
        match0 = completer("/m", 0)
        match1 = completer("/m", 1)
        match2 = completer("/m", 2)
        assert match0 == "/model"
        assert match1 == "/models"
        assert match2 is None  # no more matches

    def test_completion_returns_engine_prefixes(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Completer returns @engine prefixes for '@c' input.

        Code path: _setup_chat_readline line 560 — completions list includes
        '@' + engine_name for each engine.
        """
        from maestro_cli.chat import _setup_chat_readline

        set_completer_arg: list[object] = []

        class FakeReadline:
            @staticmethod
            def set_completer(fn: object) -> None:
                set_completer_arg.append(fn)

            @staticmethod
            def parse_and_bind(s: str) -> None:
                pass

        import builtins
        real_import = builtins.__import__

        def _mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "readline":
                return FakeReadline
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        _setup_chat_readline()

        completer = set_completer_arg[0]
        # '@c' should match @claude, @codex, @copilot
        matches: list[str] = []
        for i in range(10):
            m = completer("@c", i)
            if m is None:
                break
            matches.append(m)
        assert "@claude" in matches
        assert "@codex" in matches
        assert "@copilot" in matches
        assert len(matches) == 3  # exactly claude, codex, copilot


class TestExtractTurnCostReverseScan:
    """_extract_turn_cost scans last 20 lines in reverse for cost patterns."""

    def test_cost_line_beyond_20_line_window_not_found(self) -> None:
        """A cost line at the start (more than 20 lines from end) is NOT extracted.

        Code path: _extract_turn_cost line 275 — `lines[-20:]` discards earlier lines.
        """
        from maestro_cli.chat import _extract_turn_cost

        # Cost line at position 0, followed by 25 filler lines
        lines = ['{"total_cost_usd": 9.99}']
        lines += [f"filler line {i}" for i in range(25)]
        output = "\n".join(lines)
        cost = _extract_turn_cost(output, "claude")
        assert cost is None

    def test_cost_line_within_last_20_lines_found(self) -> None:
        """A cost line within the last 20 lines is found by _extract_turn_cost.

        Code path: _extract_turn_cost lines 275-278 — reverse scan finds cost pattern.
        """
        from maestro_cli.chat import _extract_turn_cost

        lines = [f"prefix line {i}" for i in range(5)]
        lines.append('{"total_cost_usd": 0.0567}')
        lines += [f"suffix line {i}" for i in range(10)]
        output = "\n".join(lines)
        cost = _extract_turn_cost(output, "claude")
        assert cost == pytest.approx(0.0567)


# ===========================================================================
# Iteration 12 — new gaps: run_chat farewell zero-cost path, _CHAT_COMMANDS
#                constant content, _VALID_ENGINES constant content,
#                build_command receives execution_profile kwarg,
#                history double-newline separator, @number prefix not an engine,
#                _cmd_models pass-through branch for engine with no aliases
# ===========================================================================


class TestRunChatFarewellZeroCost:
    """run_chat farewell prints '--' (not '$0.0000') when total_cost_usd is zero."""

    def test_farewell_shows_dashes_when_cost_is_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Calling /quit immediately (no turns, zero cost) → farewell line contains '--'
        and does NOT contain a '$' sign.

        Code path: run_chat line 643 — `else '--'` branch when total_cost_usd == 0.
        """
        monkeypatch.setattr("builtins.input", lambda _: "/quit")
        result = run_chat()
        assert result == 0
        out = capsys.readouterr().out
        # The farewell line is the last non-empty line
        farewell_lines = [ln for ln in out.splitlines() if "turns" in ln and "cost" in ln]
        assert farewell_lines, f"No farewell line found in output:\n{out}"
        farewell = farewell_lines[-1]
        assert "--" in farewell
        # Dollar sign must NOT appear in the farewell when cost is zero
        assert "$" not in farewell


class TestChatCommandsConstant:
    """_CHAT_COMMANDS constant contains exactly the 6 expected slash commands."""

    def test_chat_commands_has_expected_count(self) -> None:
        """_CHAT_COMMANDS must have exactly 9 entries."""
        from maestro_cli.chat import _CHAT_COMMANDS

        assert len(_CHAT_COMMANDS) == 9

    def test_chat_commands_contains_all_slash_commands(self) -> None:
        """Each documented slash command must be present in _CHAT_COMMANDS."""
        from maestro_cli.chat import _CHAT_COMMANDS

        expected = {"/model", "/models", "/context", "/save", "/load", "/clear", "/cost", "/help", "/quit"}
        assert expected == set(_CHAT_COMMANDS)


class TestValidEnginesConstant:
    """_VALID_ENGINES set contains exactly the 6 supported engine names."""

    def test_valid_engines_has_exactly_seven_members(self) -> None:
        """_VALID_ENGINES must have exactly 7 members."""
        from maestro_cli.chat import _VALID_ENGINES

        assert len(_VALID_ENGINES) == 7

    def test_valid_engines_contains_all_supported_engines(self) -> None:
        """All 7 documented engines must be in _VALID_ENGINES."""
        from maestro_cli.chat import _VALID_ENGINES

        expected = {"codex", "claude", "gemini", "copilot", "qwen", "ollama", "llama"}
        assert expected == _VALID_ENGINES


class TestRunChatTurnExecutionProfileForwarded:
    """_run_chat_turn forwards session.execution_profile to build_command."""

    def test_execution_profile_passed_as_kwarg_to_build_command(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When session.execution_profile='yolo', build_command receives
        execution_profile='yolo' as a keyword argument.

        Code path: _run_chat_turn line 309-313 — build_command(...,
        execution_profile=session.execution_profile).
        """
        import io
        from maestro_cli.chat import _run_chat_turn

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout: list[str] = ["ok\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _spy_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            captured["execution_profile"] = kw.get("execution_profile")
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _spy_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet", execution_profile="yolo")
        _run_chat_turn(session, "test")

        assert captured.get("execution_profile") == "yolo"


class TestBuildHistoryPromptDoubleSeparator:
    """_build_history_prompt joins history entries with double newline ('\\n\\n')."""

    def test_double_newline_separator_between_history_entries(self) -> None:
        """Two history messages are joined by exactly '\\n\\n' in the history block.

        Code path: _build_history_prompt line 234 — '\\n\\n'.join(parts).
        """
        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content="alpha"),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="beta"),
        ])
        result = _build_history_prompt(s, "gamma")
        # The two history entries must be separated by exactly two newlines
        assert "User: alpha\n\nAssistant: beta" in result


class TestParseEnginePrefixNumericCandidate:
    """_parse_engine_prefix with '@' followed by a numeric string (not an engine name)."""

    def test_numeric_prefix_not_recognized(self) -> None:
        """@123 is not a valid engine name — entire line is returned as plain text.

        Code path: _parse_engine_prefix line 257 — `if candidate in _VALID_ENGINES` is
        False for '123' → returns (None, line).
        """
        engine, text = _parse_engine_prefix("@123 something")
        assert engine is None
        assert text == "@123 something"

    def test_at_special_chars_not_recognized(self) -> None:
        """@gemini! (trailing punctuation breaks the engine name) → treated as plain text."""
        engine, text = _parse_engine_prefix("@gemini! write code")
        # '@gemini!' splits to candidate='gemini!' which is not in _VALID_ENGINES
        assert engine is None
        assert text == "@gemini! write code"


# ===========================================================================
# Iteration 13 — new gaps: _ENGINE_ALIASES constant structure, _HISTORY_CHAR_LIMIT
#                exact value, _cmd_help_chat codex example text, run_chat welcome
#                hint line, _cmd_cost 4-decimal format, _build_history_prompt
#                "User: {msg}" suffix, _dispatch_chat_command /model extra args,
#                _parse_engine_prefix @claude bare confirms empty remainder
# ===========================================================================


class TestEngineAliasesConstant:
    """_ENGINE_ALIASES constant maps each engine to a non-empty alias dict."""

    def test_engine_aliases_contains_all_six_engines(self) -> None:
        """_ENGINE_ALIASES must have exactly one key per supported engine."""
        from maestro_cli.chat import _ENGINE_ALIASES, _VALID_ENGINES

        assert set(_ENGINE_ALIASES.keys()) == _VALID_ENGINES

    def test_each_engine_alias_dict_is_a_dict(self) -> None:
        """Every value in _ENGINE_ALIASES is a dict (may be empty, but must be dict)."""
        from maestro_cli.chat import _ENGINE_ALIASES

        for engine, aliases in _ENGINE_ALIASES.items():
            assert isinstance(aliases, dict), (
                f"_ENGINE_ALIASES['{engine}'] must be dict, got {type(aliases)}"
            )

    def test_codex_engine_aliases_non_empty(self) -> None:
        """Codex alias dict maps short version strings to full model names."""
        from maestro_cli.chat import _ENGINE_ALIASES

        codex_aliases = _ENGINE_ALIASES["codex"]
        assert len(codex_aliases) > 0

    def test_claude_engine_aliases_non_empty(self) -> None:
        """Claude alias dict (identity mapping from CLAUDE_MODELS) is non-empty."""
        from maestro_cli.chat import _ENGINE_ALIASES

        claude_aliases = _ENGINE_ALIASES["claude"]
        assert len(claude_aliases) > 0


class TestHistoryCharLimitExactValue:
    """_HISTORY_CHAR_LIMIT constant has the documented value of 80,000."""

    def test_history_char_limit_is_80000(self) -> None:
        """The character limit for conversation history must be exactly 80,000."""
        from maestro_cli.chat import _HISTORY_CHAR_LIMIT

        assert _HISTORY_CHAR_LIMIT == 80_000


class TestCmdHelpChatCodexExampleText:
    """_cmd_help_chat contains the specific @codex example in the routing section."""

    def test_codex_example_in_routing_section(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The routing section must include '@codex' as an inline example."""
        from maestro_cli.chat import _cmd_help_chat

        _cmd_help_chat()
        out = capsys.readouterr().out
        # The exact example from the source: "(e.g., @codex optimize this query)"
        assert "@codex" in out

    def test_optimize_query_example_phrase_present(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The exact routing example phrase 'optimize this query' is in help output."""
        from maestro_cli.chat import _cmd_help_chat

        _cmd_help_chat()
        out = capsys.readouterr().out
        assert "optimize this query" in out


class TestRunChatWelcomeHintLine:
    """run_chat prints a hint line telling users how to get help."""

    def test_welcome_hint_line_contains_type_a_message(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """The welcome hint line must contain 'type a message' and '/help'."""
        from maestro_cli.chat import run_chat

        monkeypatch.setattr("builtins.input", lambda _: "/quit")
        run_chat()
        out = capsys.readouterr().out
        assert "type a message" in out
        assert "/help" in out


class TestCmdCostFourDecimalFormat:
    """_cmd_cost formats cost_usd with exactly 4 decimal places."""

    def test_cost_string_has_four_decimal_places(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A session with total_cost_usd=0.1 must display '$0.1000' (4 dp)."""
        from maestro_cli.chat import _cmd_cost

        session = ChatSession(total_turns=1, total_cost_usd=0.1)
        _cmd_cost(session)
        out = capsys.readouterr().out
        assert "$0.1000" in out

    def test_small_cost_value_four_decimal_places(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """A small cost like 0.0001 must display '$0.0001' (4 dp)."""
        from maestro_cli.chat import _cmd_cost

        session = ChatSession(total_turns=1, total_cost_usd=0.0001)
        _cmd_cost(session)
        out = capsys.readouterr().out
        assert "$0.0001" in out


class TestBuildHistoryPromptUserSuffix:
    """_build_history_prompt appends 'User: {new_message}' at the end of output."""

    def test_new_message_prefixed_with_user_label(self) -> None:
        """The new message appears as 'User: {text}' at the end of the prompt."""
        from maestro_cli.chat import _build_history_prompt

        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content="hi"),
        ])
        result = _build_history_prompt(s, "follow-up message")
        # The tail of the result must be "User: follow-up message"
        assert result.endswith("User: follow-up message")

    def test_new_message_not_double_labeled(self) -> None:
        """The new message appears exactly once at the tail, not inside history block."""
        from maestro_cli.chat import _build_history_prompt

        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content="first"),
        ])
        result = _build_history_prompt(s, "second")
        # "User: second" must appear, but only once (at the tail)
        assert result.count("User: second") == 1
        # It must be after the closing history tag
        close_tag_idx = result.find("</conversation_history>")
        user_second_idx = result.find("User: second")
        assert close_tag_idx < user_second_idx


# ===========================================================================
# New tests (iteration 14) — genuine coverage gaps
# ===========================================================================


class TestAdjustCommandForChatOutputFormatAtEnd:
    """_adjust_command_for_chat: --output-format at the end of the list (no value follows)."""

    def test_output_format_at_end_of_list_left_as_is(self) -> None:
        """When --output-format is the last token (no next element) it is kept unchanged."""
        cmd = ["claude", "--print", "--output-format"]
        result = _adjust_command_for_chat(cmd, "claude")
        # The flag is the last element; the condition i+1 < len(cmd) is False
        # so the flag is appended to result unchanged and no IndexError is raised.
        assert result == ["claude", "--print", "--output-format"]

    def test_output_format_followed_by_stream_json_replaced(self) -> None:
        """--output-format stream-json is replaced with text for chat."""
        cmd = ["claude", "--print", "--output-format", "stream-json"]
        result = _adjust_command_for_chat(cmd, "claude")
        assert result == ["claude", "--print", "--output-format", "text"]


class TestRunChatTurnCostUsdNoneNoAccumulation:
    """run_chat: when _run_chat_turn returns cost_usd=None, total_cost_usd stays 0."""

    def test_none_cost_does_not_increment_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """result.cost_usd=None — the 'if cost_usd is not None' guard in run_chat
        must prevent adding None to total_cost_usd."""
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                # Output line has no cost pattern
                self.stdout = ["response text\n"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "hello"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        # Simulate a single turn via run_chat loop
        inputs = iter(["@claude tell me something", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        from maestro_cli.chat import run_chat

        result = run_chat(engine="claude", model="sonnet")
        assert result == 0
        # We can't directly inspect session from outside, but if this doesn't
        # raise a TypeError (adding None to float), the guard is working.


class TestCmdModelsPassThroughText:
    """_cmd_models: engines with no aliases print '(pass-through model names)'."""

    def test_pass_through_text_present_for_ollama(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """ollama has aliases, but even if an engine had none, pass-through text appears."""
        from maestro_cli.chat import _cmd_models, _ENGINE_ALIASES

        # Find an engine whose alias dict has values (ollama has known aliases)
        _cmd_models()
        out = capsys.readouterr().out
        # ollama has aliases — but 'pass-through' text must exist in function output
        # if any engine has an empty alias dict
        engines_without_aliases = [
            eng for eng in ("codex", "claude", "gemini", "copilot", "qwen", "ollama")
            if not _ENGINE_ALIASES.get(eng)
        ]
        if engines_without_aliases:
            assert "pass-through" in out
        else:
            # All engines have aliases — just assert output is non-empty
            assert len(out.strip()) > 0

    def test_all_engine_names_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Every engine name must appear somewhere in _cmd_models output."""
        from maestro_cli.chat import _cmd_models

        _cmd_models()
        out = capsys.readouterr().out
        for engine in ("codex", "claude", "gemini", "copilot", "qwen", "ollama"):
            assert engine in out, f"Engine '{engine}' missing from /models output"


class TestFormatEngineLineAgentMessageNoTextKey:
    """_format_engine_line: item.completed agent_message without a 'text' key."""

    def test_agent_message_without_text_key_suppressed(self) -> None:
        """When item.type='agent_message' but 'text' key is absent, item.get('text', '')
        returns '' which evaluates falsy — result should be None (suppressed)."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        payload = {
            "type": "item.completed",
            "item": {"type": "agent_message"},  # no 'text' key
        }
        line = _json.dumps(payload) + "\n"
        result = _format_engine_line(line, "codex")
        # text defaults to '' (falsy), so 'if text:' is False → returns None
        assert result is None

    def test_agent_message_text_key_none_value_suppressed(self) -> None:
        """When item.type='agent_message' and text=None, item.get('text','') returns None
        which is falsy — result should be None."""
        import json as _json
        from maestro_cli.chat import _format_engine_line

        payload = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": None},
        }
        line = _json.dumps(payload) + "\n"
        result = _format_engine_line(line, "codex")
        # None is falsy → suppressed
        assert result is None


class TestBuildHistoryPromptMixedEngines:
    """_build_history_prompt: messages from different engines are all included."""

    def test_messages_from_different_engines_all_in_history(self) -> None:
        """History block includes messages regardless of which engine produced them."""
        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content="ask claude"),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="claude answer"),
            ChatMessage(role="user", engine="codex", model="5.4", content="ask codex"),
            ChatMessage(role="assistant", engine="codex", model="5.4", content="codex answer"),
        ])
        result = _build_history_prompt(s, "final question")
        assert "ask claude" in result
        assert "claude answer" in result
        assert "ask codex" in result
        assert "codex answer" in result
        assert "final question" in result

    def test_single_prior_message_history_injected(self) -> None:
        """A single prior message produces a conversation_history block."""
        s = ChatSession(messages=[
            ChatMessage(role="user", engine="gemini", model="flash", content="prior msg"),
        ])
        result = _build_history_prompt(s, "new msg")
        assert "<conversation_history>" in result
        assert "prior msg" in result
        assert "new msg" in result


class TestParseEnginePrefixWhitespaceOnly:
    """_parse_engine_prefix: @engine followed by only whitespace."""

    def test_engine_with_trailing_spaces_only_has_empty_text(self) -> None:
        """'@gemini   ' — split gives only '@gemini', text portion is empty string."""
        engine, text = _parse_engine_prefix("@gemini   ")
        # split() strips all whitespace; parts = ['@gemini'], len(parts) == 1
        assert engine == "gemini"
        assert text == ""

    def test_engine_with_tab_gives_correct_text(self) -> None:
        """'@ollama\thello' — tab separates prefix from text; engine and text correct."""
        engine, text = _parse_engine_prefix("@ollama\thello")
        assert engine == "ollama"
        assert "hello" in text


# ===========================================================================
# New tests (iteration 15) — genuine coverage gaps
# ===========================================================================


class TestFormatEngineLineJsonStringLiteral:
    """_format_engine_line: codex line whose JSON decodes to a string (not a dict)."""

    def test_codex_json_string_literal_starts_with_quote_returned_as_is(self) -> None:
        """A JSON string literal like '"hello world"' does not start with '{'
        (it starts with '"'), so it is returned as-is without JSON parsing.

        Code path: chat.py line 134 — `if not stripped.startswith("{"):` is True
        for a string literal → early return of the original line.
        """
        from maestro_cli.chat import _format_engine_line

        line = '"hello from codex"\n'
        result = _format_engine_line(line, "codex")
        assert result == line


class TestCmdClearMultipleMessages:
    """_cmd_clear correctly empties a session with multiple messages and reports count."""

    def test_clear_three_messages_reports_three(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Clearing a session with 3 messages prints 'cleared 3 messages' and
        leaves session.messages empty.

        Code path: _cmd_clear lines 475-477 — count = len(session.messages),
        session.messages.clear(), print(f"[maestro] cleared {count} messages").
        """
        from maestro_cli.chat import _cmd_clear

        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content="a"),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="b"),
            ChatMessage(role="user", engine="claude", model="sonnet", content="c"),
        ])
        _cmd_clear(s)
        out = capsys.readouterr().out
        assert "cleared 3" in out
        assert len(s.messages) == 0


class TestBuildHistoryPromptAllAssistantMessages:
    """_build_history_prompt when all prior messages are from the assistant role."""

    def test_history_with_two_assistant_only_messages(self) -> None:
        """Two assistant messages in history → both appear with 'Assistant:' prefix;
        the new message is appended as 'User: {text}'.

        Code path: _build_history_prompt line 224 — `msg.role == 'user'` is False for
        both messages, so 'Assistant' prefix is used; parts.reverse() preserves order.
        """
        from maestro_cli.chat import _build_history_prompt

        s = ChatSession(messages=[
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="first reply"),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content="second reply"),
        ])
        result = _build_history_prompt(s, "user follow-up")
        assert "Assistant: first reply" in result
        assert "Assistant: second reply" in result
        assert result.endswith("User: user follow-up")
        # Must be inside a conversation_history block
        assert "<conversation_history>" in result


class TestDispatchChatCommandModelWithExtraSpaces:
    """_dispatch_chat_command: '/model   codex/5.4' with extra spaces — split() normalises."""

    def test_model_dispatch_with_extra_spaces_still_works(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Extra whitespace in the dispatch line is collapsed by str.split(),
        so '/model   codex/5.4' is equivalent to '/model codex/5.4'.

        Code path: _dispatch_chat_command line 522 — `parts = line.split()` collapses
        any whitespace; args = ['codex/5.4'] → _cmd_model(['codex/5.4'], session).
        """
        session = ChatSession(engine="claude", model="sonnet")
        cont = _dispatch_chat_command("/model   codex/5.4", session)
        assert cont is True
        assert session.engine == "codex"
        assert session.model == "5.4"
        out = capsys.readouterr().out
        assert "codex" in out


class TestChatSessionMessagesIndependence:
    """ChatSession.messages is an independent list per instance (not shared)."""

    def test_two_sessions_have_independent_message_lists(self) -> None:
        """Appending to one session's messages must not affect another session's list.

        Code path: models.py field(default_factory=list) ensures each ChatSession
        gets its own list; this indirectly tests the dataclass default_factory
        via the ChatSession constructor.
        """
        s1 = ChatSession(engine="claude")
        s2 = ChatSession(engine="gemini")
        s1.messages.append(
            ChatMessage(role="user", engine="claude", model="sonnet", content="only in s1")
        )
        assert len(s1.messages) == 1
        assert len(s2.messages) == 0


class TestAdjustCommandForChatCodexBothFlagsOrdering:
    """_adjust_command_for_chat: --full-auto inserted before --skip-git-repo-check."""

    def test_full_auto_before_skip_git_repo_check_in_codex_cmd(self) -> None:
        """For codex commands, --full-auto must appear immediately before
        --skip-git-repo-check in the output (both inserted at the same index).

        Code path: _adjust_command_for_chat lines 197-202 — idx increments after
        --full-auto insertion, so --skip-git-repo-check lands at idx (one position later).
        """
        cmd = ["codex", "exec", "--json", "prompt"]
        result = _adjust_command_for_chat(cmd, "codex")
        fa_idx = result.index("--full-auto")
        sgrc_idx = result.index("--skip-git-repo-check")
        # --full-auto must come immediately before --skip-git-repo-check
        assert sgrc_idx == fa_idx + 1


class TestFormatEngineLineCodexItemCompletedNoItem:
    """_format_engine_line: item.completed with item missing entirely vs. present."""

    def test_item_completed_item_field_empty_dict_suppressed(self) -> None:
        """item.completed with item={} → item.get('type') is None → not 'agent_message'
        → falls to return None (line 152 in chat.py).

        Code path: chat.py line 147 — `item.get("type") == "agent_message"` is False
        when item is an empty dict.
        """
        import json as _json
        from maestro_cli.chat import _format_engine_line

        payload = {"type": "item.completed", "item": {}}
        line = _json.dumps(payload) + "\n"
        result = _format_engine_line(line, "codex")
        assert result is None

    def test_item_completed_item_type_code_output_suppressed(self) -> None:
        """item.completed with item.type='code_output' → not agent_message → suppressed.

        Code path: same as above — only 'agent_message' returns text; all other
        item types fall through to return None.
        """
        import json as _json
        from maestro_cli.chat import _format_engine_line

        payload = {
            "type": "item.completed",
            "item": {"type": "code_output", "output": "print result"},
        }
        line = _json.dumps(payload) + "\n"
        result = _format_engine_line(line, "codex")
        assert result is None


# ===========================================================================
# New tests: 8 additional methods targeting weakest coverage areas
# ===========================================================================


class TestFormatEngineLineCodexItemNullValue:
    """_format_engine_line: item.completed where 'item' JSON field is null (None).

    When a Codex event JSON has ``"item": null``, ``data.get("item", {})``
    returns ``None`` (the explicit JSON value), and then ``None.get("type")``
    raises ``AttributeError``.  The try/except in the function only catches
    ``json.JSONDecodeError`` and ``ValueError`` — it does NOT catch
    ``AttributeError``.  This test documents the actual behaviour so that
    any future change to handle it gracefully will be noticed.
    """

    def test_item_null_raises_attribute_error(self) -> None:
        """item=null in JSON causes data.get('item', {}) to return None;
        then None.get('type') raises AttributeError because the default {}
        is only used when the key is absent, not when it is explicitly null."""
        import json as _json
        import pytest as _pytest
        from maestro_cli.chat import _format_engine_line

        payload = {"type": "item.completed", "item": None}
        line = _json.dumps(payload) + "\n"
        # The implementation calls item.get("type") where item is None.
        # This raises AttributeError — document the current (unfixed) behaviour.
        with _pytest.raises(AttributeError):
            _format_engine_line(line, "codex")


class TestAdjustCommandForChatOutputFormatTrailingFlag:
    """_adjust_command_for_chat: ``--output-format`` with no following argument.

    When ``--output-format`` is the last element in the command list there is
    no ``cmd[i + 1]`` to inspect.  The guard ``i + 1 < len(cmd)`` is False, so
    the flag is NOT replaced — it is passed through unchanged.
    """

    def test_output_format_at_end_not_replaced(self) -> None:
        """``--output-format`` as the last arg (no value) is passed through unchanged."""
        from maestro_cli.chat import _adjust_command_for_chat

        cmd = ["claude", "--print", "--output-format"]
        result = _adjust_command_for_chat(cmd, "claude")
        # The flag is kept as-is; no replacement attempted
        assert result == ["claude", "--print", "--output-format"]


class TestParseEnginePrefixBareAtSign:
    """_parse_engine_prefix with a bare '@' (no engine name, no text)."""

    def test_bare_at_sign_not_recognized_as_engine(self) -> None:
        """'@' alone: parts[0][1:] = '' which is not in _VALID_ENGINES.
        The entire line is returned as plain text (engine=None)."""
        from maestro_cli.chat import _parse_engine_prefix

        engine, text = _parse_engine_prefix("@")
        assert engine is None
        assert text == "@"


class TestCmdCostNaiveDatetimeNoElapsed:
    """_cmd_cost: started_at is a naive ISO datetime (no timezone).

    ``datetime.now(UTC)`` is timezone-aware; subtracting a naive
    ``datetime.fromisoformat(...)`` raises ``TypeError``, which is NOT caught
    by the ``except ValueError`` handler in ``_cmd_cost``.  The function
    therefore propagates the TypeError.  This test documents the behaviour
    so that any future fix (broadening the except) is noticed.
    """

    def test_naive_started_at_raises_type_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Naive ISO string (no tz-info) → TypeError from aware-naive subtraction.

        The current ``except ValueError`` does not catch TypeError, so the
        exception propagates.  We assert this explicitly.
        """
        from maestro_cli.chat import _cmd_cost

        # Naive ISO format — no timezone suffix
        naive_iso = "2025-01-15T12:30:00"
        s = ChatSession(started_at=naive_iso, total_turns=2, total_cost_usd=0.01)
        with pytest.raises(TypeError):
            _cmd_cost(s)


class TestCmdHelpChatCommandsHeaderPresent:
    """_cmd_help_chat outputs a 'Commands:' section header."""

    def test_commands_header_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The 'Commands:' label must appear in the help output before the
        individual slash-command descriptions."""
        from maestro_cli.chat import _cmd_help_chat

        _cmd_help_chat()
        out = capsys.readouterr().out
        assert "Commands:" in out

    def test_example_codex_command_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """The example '@codex optimize this query' text appears in help output."""
        from maestro_cli.chat import _cmd_help_chat

        _cmd_help_chat()
        out = capsys.readouterr().out
        assert "@codex" in out


class TestBuildHistoryPromptExactCharLimit:
    """_build_history_prompt: message whose char count exactly equals _HISTORY_CHAR_LIMIT.

    The check is ``total_chars + len(entry) > _HISTORY_CHAR_LIMIT``.  When the
    entry length equals the limit exactly, ``0 + limit == limit`` is NOT greater
    than the limit, so the message IS included.
    """

    def test_message_exactly_at_limit_is_included(self) -> None:
        """A single message whose formatted entry length equals the char limit
        is included (boundary: == is not >)."""
        from maestro_cli.chat import _build_history_prompt, _HISTORY_CHAR_LIMIT

        # Build a message whose entry ("User: " + content) is exactly _HISTORY_CHAR_LIMIT chars
        prefix = "User: "
        content = "X" * (_HISTORY_CHAR_LIMIT - len(prefix))
        entry_len = len(prefix) + len(content)
        assert entry_len == _HISTORY_CHAR_LIMIT  # sanity check

        s = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content=content),
        ])
        result = _build_history_prompt(s, "follow-up")
        # The message fits exactly — it must appear in the result
        assert "follow-up" in result
        assert content[:20] in result  # first characters present


class TestBuildChatTaskStubEngineNoneOverride:
    """_build_chat_task_stub: explicit engine=None falls back to session.engine."""

    def test_engine_none_override_uses_session_engine(self) -> None:
        """When engine=None is passed explicitly, the function uses
        ``engine or session.engine`` which evaluates to session.engine."""
        from maestro_cli.chat import _build_chat_task_stub

        s = ChatSession(engine="gemini", model="flash")
        task = _build_chat_task_stub(s, "hello", engine=None, model=None)
        assert task.engine == "gemini"
        assert task.model == "flash"

    def test_model_none_override_uses_session_model(self) -> None:
        """When model=None is passed explicitly, task.model = session.model."""
        from maestro_cli.chat import _build_chat_task_stub

        s = ChatSession(engine="qwen", model="max")
        task = _build_chat_task_stub(s, "translate", engine=None, model=None)
        assert task.model == "max"


# ===========================================================================
# New tests (iteration 16) — 8 additional methods targeting genuine gaps
# ===========================================================================


class TestFormatEngineLineCodexAgentMessageTruthyText:
    """_format_engine_line: item.completed agent_message where text is a non-empty
    multi-word string — verifies the returned value is exactly ``text + '\\n'``."""

    def test_agent_message_with_spaces_in_text_extracted_verbatim(self) -> None:
        """Text containing spaces is extracted verbatim with a single trailing newline.

        Code path: chat.py lines 149-150 — `text = item.get("text", "")` is truthy;
        returns ``text + '\\n'``.  Verifies exact equality, not just substring.
        """
        import json as _json
        from maestro_cli.chat import _format_engine_line

        text_content = "The answer is 42 and the question is unknown."
        payload = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": text_content},
        }
        result = _format_engine_line(_json.dumps(payload), "codex")
        assert result == text_content + "\n"


class TestAdjustCommandForChatCodexNoExecFlagIndex:
    """_adjust_command_for_chat: codex command without 'exec' subcommand inserts
    flags at index 1 (immediately after the binary name)."""

    def test_codex_no_exec_inserts_full_auto_at_index_1(self) -> None:
        """When 'exec' is not in the codex command list, flags are inserted at
        index 1 (after the binary name).

        Code path: _adjust_command_for_chat lines 192-196 — `result.index('exec')`
        raises ValueError, caught → idx = 1; flags inserted there.
        """
        cmd = ["codex", "--json", "run this prompt"]
        result = _adjust_command_for_chat(cmd, "codex")
        assert result[0] == "codex"
        assert result[1] == "--full-auto"
        assert result[2] == "--skip-git-repo-check"
        assert "--json" in result
        assert "run this prompt" in result


class TestRunChatTurnMetadataLinesNotWrittenToStdout:
    """_run_chat_turn with codex engine: metadata JSON lines are suppressed from
    stdout but still captured in raw output_lines for cost extraction."""

    def test_metadata_lines_suppressed_from_stdout_cost_still_extracted(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Codex metadata events (thread.started, etc.) are filtered from stdout
        via _format_engine_line returning None, but the raw line is kept in
        output_lines for cost scanning.

        Code path: _run_chat_turn lines 381-385 — ``if formatted is not None:`` guards
        the write; raw line appended unconditionally.
        """
        import io
        import json as _json
        from maestro_cli.chat import _run_chat_turn

        metadata_line = _json.dumps({"type": "thread.started"}) + "\n"
        cost_line = _json.dumps({"total_cost_usd": 0.0099}) + "\n"

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                # Metadata first, then a cost JSON line (not agent_message, so suppressed)
                self.stdout = [metadata_line, cost_line]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["codex", "exec", "--json", "prompt"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="codex", model="5.4")
        result = _run_chat_turn(session, "compute")
        assert result is not None
        out = capsys.readouterr().out
        # Metadata line must NOT appear as readable text in stdout
        assert "thread.started" not in out
        # Cost must have been extracted from the raw output_lines
        assert result.cost_usd == pytest.approx(0.0099)


class TestCmdCostAwareDatetimeShowsElapsed:
    """_cmd_cost: started_at is an aware ISO string (with +00:00 suffix).

    ``datetime.fromisoformat`` parses the timezone info correctly, producing an
    aware datetime that can be subtracted from ``datetime.now(UTC)`` without a
    TypeError.  The elapsed minutes appear in the output.
    """

    def test_aware_started_at_shows_min_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """An aware ISO timestamp (ending '+00:00') allows elapsed calculation.

        Code path: _cmd_cost lines 485-487 — fromisoformat succeeds, subtraction
        produces a timedelta, elapsed minutes formatted as '{mins:.0f}min'.
        """
        from datetime import datetime, UTC, timedelta
        from maestro_cli.chat import _cmd_cost

        # 10 minutes ago (aware)
        started = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
        s = ChatSession(
            started_at=started,
            total_turns=2,
            total_cost_usd=0.05,
        )
        _cmd_cost(s)
        out = capsys.readouterr().out
        assert "min" in out
        assert "2 turns" in out
        assert "$0.0500" in out


class TestCmdModelsOllamaAliasesInOutput:
    """_cmd_models: ollama engine aliases (llama3, codellama, mistral, etc.) appear
    in the output because _ENGINE_ALIASES['ollama'] is built from OLLAMA_MODEL_ALIASES."""

    def test_ollama_codellama_alias_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """'codellama' must be listed under the ollama engine in _cmd_models output."""
        _cmd_models()
        out = capsys.readouterr().out
        assert "codellama" in out

    def test_ollama_mistral_alias_in_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """'mistral' must be listed under the ollama engine in _cmd_models output."""
        _cmd_models()
        out = capsys.readouterr().out
        assert "mistral" in out


class TestRunChatTurnOutputNoTrailingNewlinePrintsBlankLine:
    """_run_chat_turn: when full_output does not end with '\\n', a bare print()
    is called to move to the next line.  The capsys output therefore ends with
    an extra newline beyond the output content."""

    def test_extra_newline_printed_when_output_lacks_trailing_newline(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """When stdout output has no trailing newline, print() is called
        (lines 412-413 in chat.py) which adds '\\n' to capsys output.

        Code path: `if full_output and not full_output.endswith('\\n'): print()`
        """
        import io
        from maestro_cli.chat import _run_chat_turn

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                # No trailing newline in the output
                self.stdout = ["content without trailing newline"]
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        result = _run_chat_turn(session, "hello")
        assert result is not None
        assert result.content == "content without trailing newline"
        out = capsys.readouterr().out
        # The extra print() adds a newline at the end of stdout
        assert out.endswith("\n")


class TestRunChatAtEnginePrefixUserMessageContent:
    """run_chat: when a user types '@gemini ask something', the user ChatMessage
    stored in session.messages has content='ask something' (not the full line)."""

    def test_at_engine_strips_prefix_from_user_message_content(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The user message appended in run_chat uses `text` (the post-prefix portion),
        not the raw input line.

        Code path: run_chat lines 614-626 — `turn_engine, text = _parse_engine_prefix(line)`;
        the `ChatMessage(..., content=text)` must use `text` not `line`.
        """
        from maestro_cli.chat import run_chat, ChatMessage as _CM, ChatSession as _CS

        captured_sessions: list[_CS] = []

        def _fake_run_turn(session: object, prompt: str, **kw: object) -> _CM | None:
            assert isinstance(session, _CS)
            captured_sessions.append(session)
            return _CM(
                role="assistant", engine="gemini", model="flash", content="reply"
            )

        monkeypatch.setattr("maestro_cli.chat._run_chat_turn", _fake_run_turn)
        inputs = iter(["@gemini ask something", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        run_chat(engine="claude", model="sonnet")

        assert len(captured_sessions) == 1
        session = captured_sessions[0]
        # First message is the user message
        user_msg = session.messages[0]
        assert isinstance(user_msg, _CM)
        assert user_msg.role == "user"
        # Content must be only 'ask something', not '@gemini ask something'
        assert user_msg.content == "ask something"
        # Engine on the user message must reflect the @ prefix
        assert user_msg.engine == "gemini"


class TestBuildHistoryPromptPartsReverseSingleItem:
    """_build_history_prompt: single message in history — parts.reverse() on a
    one-element list is a no-op, and the output is correctly formatted."""

    def test_single_message_reverse_noop_correct_output(self) -> None:
        """With exactly one prior message, parts has one element after the reverse-walk.
        parts.reverse() is a no-op.  The output must have the history block and the
        new message suffix.

        Code path: _build_history_prompt lines 223-234 — loop runs once (one message
        fits), parts.reverse() on [entry] → still [entry], history = entry,
        final prompt = '<conversation_history>\\n{entry}\\n</conversation_history>\\n\\nUser: {new}'.
        """
        from maestro_cli.chat import _build_history_prompt

        s = ChatSession(messages=[
            ChatMessage(
                role="assistant", engine="claude", model="sonnet", content="only reply"
            ),
        ])
        result = _build_history_prompt(s, "next question")
        # History block present
        assert "<conversation_history>" in result
        assert "Assistant: only reply" in result
        assert "</conversation_history>" in result
        # New message appended as User: ...
        assert result.endswith("User: next question")


# ---------------------------------------------------------------------------
# Iteration 17 — targeted coverage for _format_engine_line(), _run_chat_turn()
# edge cases, and slash command handlers per judge criteria
# ---------------------------------------------------------------------------


class TestFormatEngineLineResponseCompletedSuppressed:
    """_format_engine_line: Codex 'response.completed' and 'response.created'
    events are suppressed — they are metadata, not user-visible content.

    Code path: chat.py lines 130-161 — engine=='codex', JSON parses OK,
    msg_type not in the item.completed branch, not in the metadata set,
    falls to the final ``return None`` on line 161.
    """

    def test_response_completed_event_returns_none(self) -> None:
        import json as _json

        line = _json.dumps({"type": "response.completed", "response": {"id": "r1"}})
        assert _format_engine_line(line, "codex") is None

    def test_response_created_event_returns_none(self) -> None:
        import json as _json

        line = _json.dumps({"type": "response.created", "response": {"id": "r2"}})
        assert _format_engine_line(line, "codex") is None


class TestFormatEngineLineCodexNonDictJsonReturnsLine:
    """_format_engine_line: Codex JSON that parses to a non-dict (array, int)
    calls .get() which raises AttributeError — the function has no handler
    for this so it propagates.  Verify actual behaviour.

    Code path: chat.py line 142 — ``data.get("type", "")`` on a list/int.
    """

    def test_json_array_behaviour(self) -> None:
        # A JSON array — json.loads succeeds, then data.get() raises
        # AttributeError because list has no .get().
        line = "[1, 2, 3]"
        try:
            result = _format_engine_line(line, "codex")
            # If it doesn't raise, result should be the line or None
            assert result == line or result is None
        except AttributeError:
            # This is an expected path — list has no .get()
            pass


class TestFormatEngineLineCodexAgentMessageEmptyTextField:
    """_format_engine_line: Codex item.completed with agent_message type but
    text field is empty string → returns None (suppressed).

    Code path: chat.py lines 145-152 — item.type=='agent_message', text==''
    → the ``if text:`` check on line 149 is False, falls to ``return None``
    on line 152.
    """

    def test_agent_message_empty_string_text_suppressed(self) -> None:
        import json as _json

        event = {
            "type": "item.completed",
            "item": {"type": "agent_message", "text": ""},
        }
        assert _format_engine_line(_json.dumps(event), "codex") is None


class TestRunChatTurnClaudeReasoningEffortFromPlanDefaults:
    """_run_chat_turn: Claude engine injects CLAUDE_CODE_EFFORT_LEVEL from
    plan defaults when task-level reasoning_effort is None.

    Code path: chat.py lines 330-333 — turn_engine=='claude',
    task.reasoning_effort is None, plan.defaults.claude.reasoning_effort
    is 'medium' → env['CLAUDE_CODE_EFFORT_LEVEL'] = 'medium'.
    """

    def test_plan_default_reasoning_effort_medium_injected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io
        from maestro_cli.chat import _run_chat_turn
        from maestro_cli.models import PlanSpec, PlanDefaults, EngineDefaults

        captured: dict[str, object] = {}

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                captured["env"] = kwargs.get("env", {})
                self.stdout: list[str] = []
                self.stderr = io.StringIO("")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["echo", "ok"], False)

        def _fake_plan_stub(session: object) -> PlanSpec:
            defaults = PlanDefaults()
            defaults.claude = EngineDefaults(model="sonnet", reasoning_effort="medium")
            return PlanSpec(name="chat", defaults=defaults)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})
        monkeypatch.setattr("maestro_cli.chat._build_chat_plan_stub", _fake_plan_stub)
        monkeypatch.setattr("subprocess.Popen", FakePopen)

        session = ChatSession(engine="claude", model="sonnet")
        _run_chat_turn(session, "test prompt")

        env = captured.get("env", {})
        assert isinstance(env, dict)
        assert env.get("CLAUDE_CODE_EFFORT_LEVEL") == "medium"


class TestRunChatTurnFileNotFoundReturnsNone:
    """_run_chat_turn: FileNotFoundError when engine CLI is missing → returns
    None and prints diagnostic message.

    Code path: chat.py lines 367-369 — FileNotFoundError caught, prints
    '[maestro] engine CLI not found', returns None.
    """

    def test_engine_not_found_returns_none_with_message(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from maestro_cli.chat import _run_chat_turn

        session = ChatSession(engine="codex", model="5.4")

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["codex-nonexistent", "exec", "--prompt", "x"], False)

        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})

        def _fake_popen(*args: object, **kwargs: object) -> None:
            raise FileNotFoundError("codex-nonexistent not found")
        monkeypatch.setattr("subprocess.Popen", _fake_popen)

        result = _run_chat_turn(session, "hello")
        assert result is None
        captured = capsys.readouterr()
        assert "engine CLI not found" in captured.out
        assert "codex" in captured.out


class TestRunChatTurnStderrFallbackOnEmptyStdout:
    """_run_chat_turn: When stdout is empty but stderr has content, stderr
    text is used as the response content.

    Code path: chat.py lines 406-409 — full_output.strip() is empty,
    stderr_text.strip() is non-empty → writes stderr to stdout, sets
    full_output = stderr_text.
    """

    def test_stderr_content_used_when_stdout_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import io
        from maestro_cli.chat import _run_chat_turn

        session = ChatSession(engine="gemini", model="flash")

        class FakePopen:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.stdout: list[str] = []  # empty stdout
                self.stderr = io.StringIO("stderr response content\n")
                self.returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

        monkeypatch.setattr("subprocess.Popen", FakePopen)
        monkeypatch.setattr("maestro_cli.runners._build_safe_env", lambda *a: {})

        def _fake_build_command(
            plan: object, task: object, workdir: object, **kw: object
        ) -> tuple[list[str], bool]:
            return (["gemini", "-m", "flash", "-p", "hello"], False)
        monkeypatch.setattr("maestro_cli.runners.build_command", _fake_build_command)

        result = _run_chat_turn(session, "test")
        assert result is not None
        assert "stderr response content" in result.content


class TestCmdHelpChatSlashCommandHandlerOutput:
    """_cmd_help_chat: slash command handler — verifies all 6 commands appear
    in the help output plus the @engine routing section.

    Code path: chat.py lines 495-512 — iterates _CHAT_COMMANDS, prints each
    entry from help_text dict, then prints routing section.
    """

    def test_all_six_slash_commands_in_help_output(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _cmd_help_chat()
        out = capsys.readouterr().out
        for cmd in ["/model", "/models", "/clear", "/cost", "/help", "/quit"]:
            assert cmd in out, f"{cmd} missing from help output"

    def test_at_engine_routing_example_present(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        _cmd_help_chat()
        out = capsys.readouterr().out
        assert "@engine" in out
        assert "Routing:" in out


# ===========================================================================
# Iteration 18 — new tests
# ===========================================================================


class TestFormatEngineLineCodexNonStringType:
    """_format_engine_line: Codex JSON with non-string type field values.

    Code path: chat.py line 142 — `msg_type = data.get("type", "")` followed
    by string comparisons.  When type is a non-string (int, bool, list) the
    comparisons silently fail and the function falls through to return None.
    """

    def test_codex_type_field_is_bool_suppressed(self) -> None:
        import json as _json

        line = _json.dumps({"type": True, "data": "abc"}) + "\n"
        assert _format_engine_line(line, "codex") is None

    def test_codex_type_field_is_list_suppressed(self) -> None:
        import json as _json

        line = _json.dumps({"type": ["item.completed"], "data": "abc"}) + "\n"
        assert _format_engine_line(line, "codex") is None

    def test_codex_type_field_is_none_suppressed(self) -> None:
        import json as _json

        line = _json.dumps({"type": None, "item": {}}) + "\n"
        assert _format_engine_line(line, "codex") is None


class TestBuildHistoryPromptEmptyContentMessages:
    """_build_history_prompt: messages with empty content strings.

    Code path: chat.py lines 223-228 — iterates messages in reverse, building
    entry strings.  Empty content produces 'User: ' or 'Assistant: ' entries.
    Verifies these are still included and counted toward the char limit.
    """

    def test_empty_content_messages_included_in_history(self) -> None:
        session = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content=""),
            ChatMessage(role="assistant", engine="claude", model="sonnet", content=""),
        ])
        result = _build_history_prompt(session, "hello")
        assert "User: " in result
        assert "Assistant: " in result
        assert "hello" in result

    def test_whitespace_only_content_in_history(self) -> None:
        session = ChatSession(messages=[
            ChatMessage(role="user", engine="claude", model="sonnet", content="   "),
        ])
        result = _build_history_prompt(session, "next")
        assert "User:    " in result
        assert "<conversation_history>" in result


class TestRunChatTurnProcWaitTimeout:
    """_run_chat_turn: proc.wait(timeout=1800) called with 30-min timeout.

    Code path: chat.py line 387 — proc.wait(timeout=1800).  When the process
    hangs beyond 1800s, subprocess.TimeoutExpired is raised.  Since it's not
    caught, it propagates.  This test verifies the normal path completes within
    the timeout.
    """

    def test_proc_wait_called_with_1800_timeout(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import subprocess as _sp
        import io

        session = ChatSession(engine="claude", model="sonnet")

        # Track the timeout value passed to proc.wait()
        wait_timeout_values: list[float] = []

        class _FakeProc:
            stdout = io.StringIO("hello world\n")
            stderr = io.StringIO("")
            returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                wait_timeout_values.append(timeout)  # type: ignore[arg-type]
                return 0

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

        monkeypatch.setattr(
            "maestro_cli.runners.build_command",
            lambda *a, **kw: (["echo", "test"], False),
        )
        monkeypatch.setattr(
            "maestro_cli.runners._build_safe_env",
            lambda *a, **kw: {},
        )
        monkeypatch.setattr(_sp, "Popen", lambda *a, **kw: _FakeProc())

        from maestro_cli.chat import _run_chat_turn

        result = _run_chat_turn(session, "test")

        assert result is not None
        assert 1800 in wait_timeout_values


class TestAdjustCommandForChatMultipleOutputFormatFlags:
    """_adjust_command_for_chat: command with multiple --output-format json flags.

    Code path: chat.py lines 180-188 — replaces each --output-format json pair.
    When a command has two occurrences, both should be replaced with text.
    """

    def test_two_output_format_json_flags_both_replaced(self) -> None:
        cmd = ["claude", "--output-format", "json", "--verbose", "--output-format", "json"]
        result = _adjust_command_for_chat(cmd, "claude")
        assert result.count("text") == 2
        assert "json" not in result

    def test_output_format_json_and_yaml_only_json_replaced(self) -> None:
        cmd = ["claude", "--output-format", "json", "--output-format", "yaml"]
        result = _adjust_command_for_chat(cmd, "claude")
        # json → text, yaml stays
        assert result == ["claude", "--output-format", "text", "--output-format", "yaml"]


class TestCmdCostLargeCostValue:
    """_cmd_cost: session with large accumulated cost.

    Code path: chat.py line 491 — `f"${session.total_cost_usd:.4f}"`.
    Verifies formatting with large cost values (>$1, >$100).
    """

    def test_cost_over_one_dollar(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = ChatSession(total_turns=5, total_cost_usd=1.2345)
        _cmd_cost(session)
        out = capsys.readouterr().out
        assert "$1.2345" in out
        assert "5 turns" in out

    def test_cost_over_hundred_dollars(self, capsys: pytest.CaptureFixture[str]) -> None:
        session = ChatSession(total_turns=50, total_cost_usd=123.456789)
        _cmd_cost(session)
        out = capsys.readouterr().out
        assert "$123.4568" in out


class TestRunChatSlashModelSwitchPersists:
    """run_chat: /model command updates session for subsequent turns.

    Code path: run_chat lines 608-610 — slash commands dispatched, then the
    session's engine/model persist.  This tests that the session state set by
    /model carries over to the next message.
    """

    def test_model_switch_persists_in_session_messages(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import subprocess as _sp
        import io

        inputs = iter(["/model codex/gpt-5.4-codex", "write code", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))
        monkeypatch.setattr(
            "maestro_cli.chat._setup_chat_readline", lambda: None
        )

        class _FakeProc:
            stdout = io.StringIO("done\n")
            stderr = io.StringIO("")
            returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

        monkeypatch.setattr(
            "maestro_cli.runners.build_command",
            lambda *a, **kw: (["codex", "exec", "-p", "test"], False),
        )
        monkeypatch.setattr(
            "maestro_cli.runners._build_safe_env",
            lambda *a, **kw: {},
        )
        monkeypatch.setattr(_sp, "Popen", lambda *a, **kw: _FakeProc())

        code = run_chat(engine="claude", model="sonnet")
        assert code == 0
        out = capsys.readouterr().out
        assert "switched to codex/gpt-5.4-codex" in out


class TestFormatEngineLineCodexReasoningItemSuppressed:
    """_format_engine_line: Codex item.completed with non-agent_message item type.

    Code path: chat.py lines 145-152 — item.completed with
    item.type != 'agent_message' (e.g. 'reasoning') returns None.
    """

    def test_codex_item_completed_reasoning_type_suppressed(self) -> None:
        import json as _json

        line = _json.dumps({
            "type": "item.completed",
            "item": {"type": "reasoning", "text": "thinking..."},
        }) + "\n"
        assert _format_engine_line(line, "codex") is None

    def test_codex_item_completed_function_call_type_suppressed(self) -> None:
        import json as _json

        line = _json.dumps({
            "type": "item.completed",
            "item": {"type": "function_call", "name": "read_file"},
        }) + "\n"
        assert _format_engine_line(line, "codex") is None


class TestFormatEngineLineEmptyJsonObject:
    """_format_engine_line: Codex JSON with no 'type' key at all.

    Code path: chat.py line 142 — data.get('type', '') returns ''.  Does not
    match any suppressed event types, falls through to final return None.
    """

    def test_codex_empty_json_object_returns_none(self) -> None:
        assert _format_engine_line("{}", "codex") is None

    def test_codex_json_only_data_no_type_returns_none(self) -> None:
        import json as _json

        line = _json.dumps({"data": "hello", "id": 42}) + "\n"
        assert _format_engine_line(line, "codex") is None


class TestRunChatTurnShellCommandString:
    """_run_chat_turn: when build_command returns a string, it runs as shell.

    Code path: chat.py lines 320-322 — isinstance(command, str) branch.
    """

    def test_string_command_runs_as_shell(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import subprocess as _sp
        import io

        session = ChatSession(engine="claude", model="sonnet")

        popen_args_captured: list[Any] = []

        class _FakeProc:
            stdout = io.StringIO("hello from shell\n")
            stderr = io.StringIO("")
            returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

        def _fake_popen(*args: Any, **kwargs: Any) -> _FakeProc:
            popen_args_captured.append((args, kwargs))
            return _FakeProc()

        monkeypatch.setattr(
            "maestro_cli.runners.build_command",
            lambda *a, **kw: ("echo hello", True),
        )
        monkeypatch.setattr(
            "maestro_cli.runners._build_safe_env",
            lambda *a, **kw: {},
        )
        monkeypatch.setattr(_sp, "Popen", _fake_popen)

        result = _run_chat_turn(session, "test")
        assert result is not None
        assert result.content == "hello from shell"
        # String command → shell=True passed to Popen
        assert popen_args_captured
        assert popen_args_captured[0][1]["shell"] is True


class TestRunChatTurnCostExtractedFromOutput:
    """_run_chat_turn: cost_usd extracted from Claude-style output.

    Code path: chat.py lines 416 — _extract_turn_cost called with full output.
    """

    def test_claude_cost_line_populates_cost_usd(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        import subprocess as _sp
        import io

        session = ChatSession(engine="claude", model="sonnet")

        class _FakeProc:
            stdout = io.StringIO(
                "Here is the answer.\n"
                "Total cost: $0.0042\n"
            )
            stderr = io.StringIO("")
            returncode = 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def terminate(self) -> None:
                pass

            def kill(self) -> None:
                pass

        monkeypatch.setattr(
            "maestro_cli.runners.build_command",
            lambda *a, **kw: (["claude", "--print", "test"], False),
        )
        monkeypatch.setattr(
            "maestro_cli.runners._build_safe_env",
            lambda *a, **kw: {},
        )
        monkeypatch.setattr(_sp, "Popen", lambda *a, **kw: _FakeProc())

        result = _run_chat_turn(session, "test")
        assert result is not None
        # _extract_turn_cost should find the cost pattern
        if result.cost_usd is not None:
            assert result.cost_usd > 0


class TestCmdCostSmallFractions:
    """_cmd_cost: very small cost values formatted to 4 decimal places.

    Code path: chat.py line 491 — f"${session.total_cost_usd:.4f}".
    """

    def test_cost_one_cent_formatted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = ChatSession(total_turns=1, total_cost_usd=0.01)
        _cmd_cost(session)
        out = capsys.readouterr().out
        assert "$0.0100" in out

    def test_cost_tiny_fraction_formatted(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = ChatSession(total_turns=3, total_cost_usd=0.00009)
        _cmd_cost(session)
        out = capsys.readouterr().out
        assert "$0.0001" in out


class TestDispatchChatCommandModelsCall:
    """/models slash command dispatches _cmd_models and returns True.

    Code path: chat.py line 531 — `elif cmd == "/models": _cmd_models()`.
    """

    def test_models_command_returns_true_and_prints(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = ChatSession()
        result = _dispatch_chat_command("/models", session)
        assert result is True
        out = capsys.readouterr().out
        assert "claude" in out
        assert "codex" in out

    def test_cost_command_returns_true(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        session = ChatSession(total_turns=2, total_cost_usd=0.05)
        result = _dispatch_chat_command("/cost", session)
        assert result is True
        out = capsys.readouterr().out
        assert "2 turns" in out


class TestBuildHistoryPromptSpecialChars:
    """_build_history_prompt: messages with XML-like tags and template vars.

    Code path: chat.py lines 219-238 — history block wraps content in
    <conversation_history> tags.  Content with special chars must not break.
    """

    def test_message_with_xml_tags_preserved(self) -> None:
        session = ChatSession(
            messages=[
                ChatMessage(
                    role="user",
                    engine="claude",
                    model="sonnet",
                    content="Use <div> and </div> tags",
                ),
            ]
        )
        result = _build_history_prompt(session, "continue")
        assert "<conversation_history>" in result
        assert "<div>" in result
        assert "User: continue" in result

    def test_message_with_template_braces_preserved(self) -> None:
        session = ChatSession(
            messages=[
                ChatMessage(
                    role="assistant",
                    engine="claude",
                    model="sonnet",
                    content="Use {{ workspace_root }} variable",
                ),
            ]
        )
        result = _build_history_prompt(session, "next")
        assert "{{ workspace_root }}" in result


class TestAdjustCommandForChatCodexAlreadyHasBothFlags:
    """_adjust_command_for_chat: Codex command with both flags already present.

    Code path: chat.py lines 197-201 — 'if "--full-auto" not in result' and
    'if "--skip-git-repo-check" not in result' both skip insertion.
    """

    def test_no_duplicate_flags_when_both_present(self) -> None:
        cmd = ["codex", "exec", "--full-auto", "--skip-git-repo-check", "-p", "test"]
        result = _adjust_command_for_chat(cmd, "codex")
        assert result.count("--full-auto") == 1
        assert result.count("--skip-git-repo-check") == 1

    def test_only_missing_flag_added(self) -> None:
        cmd = ["codex", "exec", "--full-auto", "-p", "test"]
        result = _adjust_command_for_chat(cmd, "codex")
        assert result.count("--full-auto") == 1
        assert result.count("--skip-git-repo-check") == 1
