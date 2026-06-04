"""Coverage backfill for src/maestro_cli/chat.py.

Targets the remaining uncovered branches: filesystem-root break in auto-context
discovery, OSError / dedupe handling during auto-load, the pass-through engine
branch in /models, error paths in /context, /save, /load, and the session
replacement path in run_chat.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from maestro_cli.chat import (
    ChatSession,
    _autoload_context_files,
    _cmd_context,
    _cmd_load,
    _cmd_models,
    _cmd_save,
    _discover_auto_context_files,
    run_chat,
)


# ===========================================================================
# _discover_auto_context_files — filesystem-root break (root unreachable)
# ===========================================================================


class TestDiscoverFilesystemRootBreak:
    def test_walk_reaches_filesystem_root_when_root_not_an_ancestor(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the discovered root is not an ancestor of cwd, the walk climbs
        all the way to the filesystem root and breaks on parent == cursor."""
        cwd = tmp_path / "a" / "b"
        cwd.mkdir(parents=True)

        # Force a root that the upward walk can never equal, so the loop only
        # terminates when it hits the actual filesystem root (parent == cursor).
        unreachable = (tmp_path / "elsewhere" / "deep" / "nonancestor").resolve()
        monkeypatch.setattr(
            "maestro_cli.chat._discover_context_root",
            lambda _current: unreachable,
        )

        discovered = _discover_auto_context_files(cwd)

        # No context files anywhere on the climbed path → empty result, but the
        # function must complete (it walked to the filesystem root).
        assert discovered == []

    def test_walk_collects_files_along_path_to_filesystem_root(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Same filesystem-root break, but with a real context file present so
        discovery still returns it."""
        cwd = tmp_path / "x" / "y"
        cwd.mkdir(parents=True)
        marker = cwd / "AGENTS.md"
        marker.write_text("leaf instructions", encoding="utf-8")

        unreachable = (tmp_path / "no" / "such" / "root").resolve()
        monkeypatch.setattr(
            "maestro_cli.chat._discover_context_root",
            lambda _current: unreachable,
        )

        discovered = _discover_auto_context_files(cwd)

        assert marker.resolve() in discovered


# ===========================================================================
# _autoload_context_files — dedupe continue + OSError handling
# ===========================================================================


class TestAutoloadEdgeCases:
    def test_skips_already_loaded_key(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A context key already present in the session is skipped (continue)."""
        cwd = tmp_path / "repo"
        cwd.mkdir()
        (cwd / "AGENTS.md").write_text("root instructions", encoding="utf-8")

        # Make discovery return the file, then pre-seed its display key so the
        # autoload loop hits the "key in session.context_files" continue.
        target = (cwd / "AGENTS.md").resolve()
        monkeypatch.setattr(
            "maestro_cli.chat._discover_auto_context_files",
            lambda *a, **k: [target],
        )

        from maestro_cli.chat import _context_display_key

        key = _context_display_key(target, cwd=cwd)
        session = ChatSession(context_files={key: "preloaded"})

        loaded = _autoload_context_files(session, cwd=cwd, announce=False)

        assert loaded == []  # nothing newly loaded
        assert session.context_files[key] == "preloaded"  # unchanged

    def test_read_oserror_is_reported_and_skipped(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An OSError while reading a discovered file is announced and skipped."""
        cwd = tmp_path / "repo"
        cwd.mkdir()
        target = (cwd / "AGENTS.md")
        target.write_text("data", encoding="utf-8")
        resolved = target.resolve()

        monkeypatch.setattr(
            "maestro_cli.chat._discover_auto_context_files",
            lambda *a, **k: [resolved],
        )

        def _boom(_path: Path, **_kw: object) -> str:
            raise OSError("permission denied")

        monkeypatch.setattr("maestro_cli.chat._read_context_file", _boom)

        session = ChatSession()
        loaded = _autoload_context_files(session, cwd=cwd, announce=True)

        assert loaded == []
        assert session.context_files == {}
        out = capsys.readouterr().out
        assert "auto-context read error" in out

    def test_read_oserror_silent_when_announce_false(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """OSError branch with announce=False does not print, still skips."""
        cwd = tmp_path / "repo"
        cwd.mkdir()
        target = (cwd / "CLAUDE.md")
        target.write_text("data", encoding="utf-8")
        resolved = target.resolve()

        monkeypatch.setattr(
            "maestro_cli.chat._discover_auto_context_files",
            lambda *a, **k: [resolved],
        )

        def _boom(_path: Path, **_kw: object) -> str:
            raise OSError("io error")

        monkeypatch.setattr("maestro_cli.chat._read_context_file", _boom)

        session = ChatSession()
        loaded = _autoload_context_files(session, cwd=cwd, announce=False)

        assert loaded == []
        out = capsys.readouterr().out
        assert "auto-context read error" not in out


# ===========================================================================
# _cmd_models — pass-through (no aliases) branch
# ===========================================================================


class TestCmdModelsPassThrough:
    def test_engine_without_aliases_shows_pass_through(
        self,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """An engine whose alias map is empty renders the pass-through note."""
        # Give ollama an empty alias dict so the else branch fires for it.
        patched = {"ollama": {}}
        monkeypatch.setattr("maestro_cli.chat._ENGINE_ALIASES", patched)

        _cmd_models()

        out = capsys.readouterr().out
        assert "ollama: (pass-through model names)" in out


# ===========================================================================
# _cmd_context — error reading an existing file
# ===========================================================================


class TestCmdContextReadError:
    def test_read_error_is_reported_and_continues(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A failure reading an existing file prints an error and continues."""
        good = tmp_path / "good.md"
        good.write_text("ok", encoding="utf-8")
        bad = tmp_path / "bad.md"
        bad.write_text("data", encoding="utf-8")

        real_reader = __import__(
            "maestro_cli.chat", fromlist=["_read_context_file"]
        )._read_context_file

        def _maybe_boom(path: Path, **kw: object) -> str:
            if path.name == "bad.md":
                raise OSError("cannot read")
            return real_reader(path, **kw)  # type: ignore[no-any-return]

        monkeypatch.setattr("maestro_cli.chat._read_context_file", _maybe_boom)

        session = ChatSession()
        _cmd_context([str(bad), str(good)], session)

        out = capsys.readouterr().out
        assert "error reading" in out.lower()
        # The good file still got added.
        assert any(v == "ok" for v in session.context_files.values())


# ===========================================================================
# _cmd_save — write error path
# ===========================================================================


class TestCmdSaveError:
    def test_write_error_is_reported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A failure writing the session file prints an error instead of raising."""
        monkeypatch.chdir(tmp_path)

        def _boom(self: Path, *a: object, **k: object) -> int:
            raise OSError("disk full")

        monkeypatch.setattr(Path, "write_text", _boom)

        session = ChatSession()
        _cmd_save(session)

        out = capsys.readouterr().out
        assert "error saving session" in out.lower()


# ===========================================================================
# _cmd_load — explicit path, empty dir, missing file, and load error
# ===========================================================================


class TestCmdLoad:
    def test_explicit_path_not_a_file(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """/load <path> where the path does not exist returns the session."""
        session = ChatSession(total_turns=7)
        missing = tmp_path / "does_not_exist.json"

        result = _cmd_load([str(missing)], session)

        assert result is session  # unchanged
        out = capsys.readouterr().out
        assert "session file not found" in out.lower()

    def test_no_args_empty_sessions_dir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """/load with no args and an existing but empty sessions dir."""
        monkeypatch.chdir(tmp_path)
        # Create the sessions dir but put no chat_*.json files in it.
        (tmp_path / ".maestro-cache" / "sessions").mkdir(parents=True)

        session = ChatSession(total_turns=3)
        result = _cmd_load([], session)

        assert result is session
        out = capsys.readouterr().out
        assert "no saved sessions found" in out.lower()

    def test_load_corrupt_file_reports_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A corrupt JSON session file triggers the load-error branch."""
        monkeypatch.chdir(tmp_path)
        sessions_dir = tmp_path / ".maestro-cache" / "sessions"
        sessions_dir.mkdir(parents=True)
        corrupt = sessions_dir / "chat_20260101_000000.json"
        corrupt.write_text("{ not valid json", encoding="utf-8")

        session = ChatSession(total_turns=9)
        # No args → finds most recent (the corrupt file) → json.loads raises.
        result = _cmd_load([], session)

        assert result is session  # falls back to original session
        out = capsys.readouterr().out
        assert "error loading session" in out.lower()

    def test_load_explicit_corrupt_path(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Explicit-path branch (Path(args[0])) plus a corrupt file body."""
        corrupt = tmp_path / "mine.json"
        corrupt.write_text("not json at all", encoding="utf-8")

        session = ChatSession(total_turns=2)
        result = _cmd_load([str(corrupt)], session)

        assert result is session
        out = capsys.readouterr().out
        assert "error loading session" in out.lower()


# ===========================================================================
# run_chat — /load replaces the active session (session = result)
# ===========================================================================


class TestRunChatSessionReplacement:
    def test_load_replaces_active_session(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """A successful /load inside the REPL swaps in the loaded session."""
        monkeypatch.chdir(tmp_path)

        # Pre-write a valid session file that /load (no args) will pick up.
        sessions_dir = tmp_path / ".maestro-cache" / "sessions"
        sessions_dir.mkdir(parents=True)
        saved = sessions_dir / "chat_20260101_000000.json"
        saved.write_text(
            '{"engine": "gemini", "model": "pro", "execution_profile": "plan",'
            ' "total_turns": 4, "total_cost_usd": 1.25, "messages": [],'
            ' "context_files": {}, "started_at": ""}',
            encoding="utf-8",
        )

        inputs = iter(["/load", "/cost", "/quit"])
        monkeypatch.setattr("builtins.input", lambda _: next(inputs))

        result = run_chat(auto_context=False)

        assert result == 0
        out = capsys.readouterr().out
        # /cost after the load must reflect the loaded session (4 turns).
        assert "4 turns" in out
        assert "loaded session" in out.lower()
