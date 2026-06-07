"""Interactive multi-model chat terminal.

``maestro chat`` provides a REPL that streams responses from any configured
engine (claude, codex, gemini, copilot, qwen, ollama) with ``@engine`` per-turn
routing, slash commands, conversation history, and cost tracking.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Literal

from .models import (
    EngineName,
    ExecutionProfile,
    EngineDefaults,
    PlanDefaults,
    PlanSpec,
    TaskSpec,
    CLAUDE_MODELS,
    CODEX_MODEL_ALIASES,
    COPILOT_MODEL_ALIASES,
    GEMINI_MODEL_ALIASES,
    LLAMA_MODEL_ALIASES,
    OLLAMA_MODEL_ALIASES,
    QWEN_MODEL_ALIASES,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_ENGINES: set[str] = {"codex", "claude", "gemini", "copilot", "qwen", "ollama", "llama"}
_WINDOWS_SHELL_BUILTINS: set[str] = {"echo", "dir", "copy", "type", "set"}

_HISTORY_CHAR_LIMIT = 80_000  # truncate oldest messages beyond this

_CHAT_COMMANDS: list[str] = [
    "/model",
    "/models",
    "/context",
    "/save",
    "/load",
    "/clear",
    "/cost",
    "/help",
    "/quit",
]

_SESSIONS_DIR = ".maestro-cache/sessions"
_AUTO_CONTEXT_FILENAMES: tuple[str, ...] = ("AGENTS.md", "CLAUDE.md")
_CONTEXT_FILE_CHAR_LIMIT = 50_000

_ENGINE_ALIASES: dict[str, dict[str, str]] = {
    "claude": dict.fromkeys(CLAUDE_MODELS, ""),  # alias → alias (identity for display)
    "codex": CODEX_MODEL_ALIASES,
    "gemini": GEMINI_MODEL_ALIASES,
    "copilot": COPILOT_MODEL_ALIASES,
    "qwen": QWEN_MODEL_ALIASES,
    "ollama": OLLAMA_MODEL_ALIASES,
    "llama": LLAMA_MODEL_ALIASES,
}

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    engine: str
    model: str
    content: str
    cost_usd: float | None = None
    duration_sec: float = 0.0


@dataclass
class ChatSession:
    engine: str = "claude"
    model: str | None = None  # None → engine default
    execution_profile: ExecutionProfile = "plan"
    messages: list[ChatMessage] = field(default_factory=list)
    context_files: dict[str, str] = field(default_factory=dict)  # path → content
    total_cost_usd: float = 0.0
    total_turns: int = 0
    started_at: str = ""


def _context_display_key(path: Path, *, cwd: Path | None = None) -> str:
    """Return a stable display key for a loaded context file."""
    base_dir = (cwd or Path.cwd()).resolve()
    resolved = path.resolve()
    if resolved.is_relative_to(base_dir):
        return str(resolved.relative_to(base_dir))
    return str(resolved)


def _read_context_file(path: Path, *, max_chars: int = _CONTEXT_FILE_CHAR_LIMIT) -> str:
    """Read a context file and truncate it to a prompt-safe size."""
    content = path.read_text(encoding="utf-8")
    if len(content) > max_chars:
        return content[:max_chars] + f"\n\n[... truncated at {max_chars:,} chars]"
    return content


def _discover_context_root(cwd: Path) -> Path:
    """Return git top-level when present, otherwise the current directory."""
    current = cwd.resolve()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return current


def _discover_auto_context_files(
    cwd: Path | None = None,
    *,
    filenames: tuple[str, ...] = _AUTO_CONTEXT_FILENAMES,
) -> list[Path]:
    """Discover hierarchical chat context files in root-to-leaf order."""
    current = (cwd or Path.cwd()).resolve()
    root = _discover_context_root(current)

    directories: list[Path] = []
    cursor = current
    while True:
        directories.append(cursor)
        if cursor == root:
            break
        parent = cursor.parent
        if parent == cursor:
            break
        cursor = parent
    directories.reverse()

    ordered_names = tuple(name.strip() for name in filenames if name and name.strip())
    discovered: list[Path] = []
    seen: set[Path] = set()
    for directory in directories:
        for name in ordered_names:
            candidate = directory / name
            if not candidate.is_file():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            discovered.append(resolved)
    return discovered


def _autoload_context_files(
    session: ChatSession,
    *,
    cwd: Path | None = None,
    filenames: tuple[str, ...] = _AUTO_CONTEXT_FILENAMES,
    announce: bool = True,
) -> list[Path]:
    """Load discovered context files into the session."""
    current = (cwd or Path.cwd()).resolve()
    loaded: list[Path] = []
    for path in _discover_auto_context_files(current, filenames=filenames):
        key = _context_display_key(path, cwd=current)
        if key in session.context_files:
            continue
        try:
            session.context_files[key] = _read_context_file(path)
        except OSError as exc:
            if announce:
                print(f"[maestro] auto-context read error: {path} ({exc})")
            continue
        loaded.append(path)

    if announce and loaded:
        rendered = ", ".join(str(path) for path in loaded)
        print(f"[maestro] auto-loaded {len(loaded)} context file(s): {rendered}")
    return loaded


# ---------------------------------------------------------------------------
# Stub builders — lightweight PlanSpec/TaskSpec for build_command()
# ---------------------------------------------------------------------------


def _build_chat_plan_stub(session: ChatSession) -> PlanSpec:
    """Create a minimal ``PlanSpec`` with engine defaults matching the session."""
    defaults = PlanDefaults()
    # Set model on the active engine's defaults so build_command() picks it up.
    engine_defaults = EngineDefaults(model=session.model)
    setattr(defaults, session.engine, engine_defaults)
    return PlanSpec(name="chat", defaults=defaults)


def _build_chat_task_stub(
    session: ChatSession,
    prompt: str,
    engine: str | None = None,
    model: str | None = None,
) -> TaskSpec:
    """Create a minimal ``TaskSpec`` with an inline prompt for one chat turn."""
    return TaskSpec(
        id="chat-turn",
        engine=engine or session.engine,  # type: ignore[arg-type]
        model=model or session.model,
        prompt=prompt,
    )


# ---------------------------------------------------------------------------
# Engine output formatting
# ---------------------------------------------------------------------------


def _format_engine_line(line: str, engine: str) -> str | None:
    """Extract human-readable text from an engine output line.

    For Codex (JSON output), extracts ``agent_message`` text and discards
    metadata/reasoning lines.  For other engines, returns the line as-is.

    Returns ``None`` to suppress the line entirely.
    """
    if engine != "codex":
        return line

    stripped = line.strip()
    if not stripped.startswith("{"):
        return line

    try:
        data = json.loads(stripped)
    except ValueError:
        return line

    msg_type = data.get("type", "")

    # item.completed with agent_message → the actual response text
    if msg_type == "item.completed":
        item = data.get("item", {})
        if item.get("type") == "agent_message":
            text = item.get("text", "")
            if text:
                return str(text) + "\n"
        # Suppress reasoning items and other internal events
        return None

    # Suppress metadata events (thread.started, turn.started, turn.completed, etc.)
    if msg_type in (
        "thread.started", "turn.started", "turn.completed",
        "item.started", "item.streaming",
    ):
        return None

    return None


# ---------------------------------------------------------------------------
# Command adjustment — text output for streaming
# ---------------------------------------------------------------------------


def _adjust_command_for_chat(cmd: list[str], engine: str) -> list[str]:
    """Replace ``--output-format json/stream-json`` with ``text`` for readable streaming.

    Claude emits ``--output-format stream-json`` by default (Gemini emits
    ``json``); for chat we want human-readable text output instead.  Codex
    keeps ``--json`` because it needs it to produce stdout output (without it,
    codex writes to files silently).  Codex also gets ``--full-auto`` so it
    runs non-interactively in chat mode.
    """
    result: list[str] = []
    skip_next = False
    for i, arg in enumerate(cmd):
        if skip_next:
            skip_next = False
            continue
        if arg == "--output-format" and i + 1 < len(cmd) and cmd[i + 1] in ("json", "stream-json"):
            result.append("--output-format")
            result.append("text")
            skip_next = True
            continue
        result.append(arg)

    # Codex needs --full-auto and --skip-git-repo-check for chat mode
    if engine == "codex":
        try:
            idx = result.index("exec") + 1
        except ValueError:
            idx = 1
        if "--full-auto" not in result:
            result.insert(idx, "--full-auto")
            idx += 1
        if "--skip-git-repo-check" not in result:
            result.insert(idx, "--skip-git-repo-check")

    return result


# ---------------------------------------------------------------------------
# Conversation history
# ---------------------------------------------------------------------------


def _build_history_prompt(session: ChatSession, new_message: str) -> str:
    """Build a prompt that includes file context, conversation history, and the new message.

    If history exceeds ``_HISTORY_CHAR_LIMIT``, oldest messages are trimmed.
    File context is prepended before conversation history.
    """
    preamble_parts: list[str] = []

    # File context block
    if session.context_files:
        ctx_parts: list[str] = []
        for fpath, content in session.context_files.items():
            ctx_parts.append(f"--- {fpath} ---\n{content}")
        file_context = "\n\n".join(ctx_parts)
        preamble_parts.append(
            f"<file_context>\n{file_context}\n</file_context>"
        )

    # Conversation history block
    if session.messages:
        history_parts: list[str] = []
        total_chars = 0
        for msg in reversed(session.messages):
            entry = f"{'User' if msg.role == 'user' else 'Assistant'}: {msg.content}"
            if total_chars + len(entry) > _HISTORY_CHAR_LIMIT:
                break
            history_parts.append(entry)
            total_chars += len(entry)

        if history_parts:
            history_parts.reverse()
            history = "\n\n".join(history_parts)
            preamble_parts.append(
                f"<conversation_history>\n{history}\n</conversation_history>"
            )

    if not preamble_parts:
        return new_message

    return "\n\n".join(preamble_parts) + f"\n\nUser: {new_message}"


# ---------------------------------------------------------------------------
# @engine prefix parsing
# ---------------------------------------------------------------------------


def _parse_engine_prefix(line: str) -> tuple[str | None, str]:
    """Parse ``@engine`` prefix from user input.

    Returns ``(engine_or_none, remaining_text)``.  Unknown prefixes are treated
    as plain text (no engine override).
    """
    if not line.startswith("@"):
        return None, line

    parts = line.split(None, 1)
    candidate = parts[0][1:]  # strip the @
    if candidate in _VALID_ENGINES:
        text = parts[1] if len(parts) > 1 else ""
        return candidate, text

    # Not a valid engine name → treat entire line as plain text
    return None, line


# ---------------------------------------------------------------------------
# Cost extraction
# ---------------------------------------------------------------------------


def _extract_turn_cost(output: str, engine: str) -> float | None:
    """Extract cost from a completed chat turn's output (best-effort)."""
    from .runners import _extract_cost_from_line

    # Scan last 20 lines for cost patterns
    lines = output.strip().splitlines()[-20:]
    for line in reversed(lines):
        cost = _extract_cost_from_line(line)
        if cost is not None:
            return cost
    return None


# ---------------------------------------------------------------------------
# Single chat turn execution
# ---------------------------------------------------------------------------


def _run_chat_turn(
    session: ChatSession,
    prompt: str,
    engine: str | None = None,
    model: str | None = None,
) -> ChatMessage | None:
    """Execute a single chat turn: build command, stream output, extract cost."""
    from .runners import _build_safe_env, build_command

    turn_engine = engine or session.engine
    turn_model = model or session.model

    # Build history-aware prompt
    full_prompt = _build_history_prompt(session, prompt)

    # Create stubs
    plan = _build_chat_plan_stub(session)
    task = _build_chat_task_stub(session, full_prompt, engine=turn_engine, model=turn_model)

    # Build command via runners infrastructure
    try:
        command, _ = build_command(
            plan,
            task,
            Path.cwd(),
            execution_profile=session.execution_profile,
        )
    except Exception as exc:
        print(f"[maestro] error building command: {exc}")
        return None

    # Ensure command is a list for adjustment
    if isinstance(command, str):
        # Shell command — run as-is
        cmd_list: list[str] | str = command
    else:
        cmd_list = _adjust_command_for_chat(command, turn_engine)

    # Build environment
    env = _build_safe_env({}, {})

    # Inject reasoning effort for Claude
    if turn_engine == "claude":
        reasoning = task.reasoning_effort or plan.defaults.claude.reasoning_effort
        if reasoning:
            env["CLAUDE_CODE_EFFORT_LEVEL"] = reasoning

    # Stream the response
    start = time.monotonic()
    output_lines: list[str] = []

    popen_kwargs: dict[str, Any] = {}
    if os.name == "nt":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP

    try:
        launch_command = cmd_list
        launch_shell = isinstance(cmd_list, str)
        if (
            os.name == "nt"
            and isinstance(cmd_list, list)
            and cmd_list
            and cmd_list[0].lower() in _WINDOWS_SHELL_BUILTINS
        ):
            launch_command = subprocess.list2cmdline(cmd_list)
            launch_shell = True

        proc = subprocess.Popen(
            launch_command,
            cwd=Path.cwd(),
            env=env,
            shell=launch_shell,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
    except FileNotFoundError:
        print(f"[maestro] engine CLI not found for '{turn_engine}'. Run 'maestro doctor' to check.")
        return None
    except Exception as exc:
        print(f"[maestro] error launching process: {exc}")
        return None

    # Print engine tag
    model_tag = turn_model or "default"
    print(f"\033[2m[{turn_engine}/{model_tag}]\033[0m ", end="", flush=True)

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            formatted = _format_engine_line(line, turn_engine)
            if formatted is not None:
                sys.stdout.write(formatted)
                sys.stdout.flush()
            output_lines.append(line)  # keep raw for cost extraction

        proc.wait(timeout=1800)
    except KeyboardInterrupt:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("\n[maestro] interrupted")
        return None

    # Capture stderr — show it if stdout was empty (some engines write there)
    stderr_text = ""
    if proc.stderr:
        stderr_text = proc.stderr.read()

    duration = time.monotonic() - start
    full_output = "".join(output_lines)

    # If stdout was empty, try stderr (codex and others may write output there)
    if not full_output.strip() and stderr_text.strip():
        sys.stdout.write(stderr_text)
        sys.stdout.flush()
        full_output = stderr_text

    # Ensure output ends with newline
    if full_output and not full_output.endswith("\n"):
        print()

    # Extract cost
    cost = _extract_turn_cost(full_output, turn_engine)

    msg = ChatMessage(
        role="assistant",
        engine=turn_engine,
        model=turn_model or "default",
        content=full_output.strip(),
        cost_usd=cost,
        duration_sec=round(duration, 1),
    )

    return msg


# ---------------------------------------------------------------------------
# Slash commands
# ---------------------------------------------------------------------------


def _cmd_model(args: list[str], session: ChatSession) -> None:
    """Switch engine/model.  Formats: ``/model engine/model``, ``/model model``."""
    if not args:
        print(f"[maestro] active: {session.engine}/{session.model or 'default'}")
        return

    spec = args[0]
    if "/" in spec:
        parts = spec.split("/", 1)
        eng, mdl = parts[0], parts[1]
        if eng not in _VALID_ENGINES:
            print(f"[maestro] unknown engine '{eng}'. Valid: {', '.join(sorted(_VALID_ENGINES))}")
            return
        session.engine = eng
        session.model = mdl if mdl else None
    elif spec in _VALID_ENGINES:
        # Bare engine name → switch engine, reset model to default
        session.engine = spec
        session.model = None
    else:
        session.model = spec

    print(f"[maestro] switched to {session.engine}/{session.model or 'default'}")


def _cmd_models() -> None:
    """Print available engines and model aliases."""
    print("[maestro] Available engines and models:\n")
    for engine in sorted(_VALID_ENGINES):
        aliases = _ENGINE_ALIASES.get(engine, {})
        if aliases:
            alias_list = ", ".join(sorted(aliases.keys()))
            print(f"  {engine}: {alias_list}")
        else:
            print(f"  {engine}: (pass-through model names)")
    print()


def _cmd_clear(session: ChatSession) -> None:
    """Clear conversation history."""
    count = len(session.messages)
    session.messages.clear()
    print(f"[maestro] cleared {count} messages")


def _cmd_cost(session: ChatSession) -> None:
    """Print session cost summary."""
    elapsed = ""
    if session.started_at:
        try:
            start = datetime.fromisoformat(session.started_at)
            mins = (datetime.now(UTC) - start).total_seconds() / 60
            elapsed = f" ({mins:.0f}min)"
        except ValueError:
            pass

    cost_str = f"${session.total_cost_usd:.4f}" if session.total_cost_usd > 0 else "--"
    print(f"[maestro] session: {session.total_turns} turns, cost: {cost_str}{elapsed}")


def _cmd_context(args: list[str], session: ChatSession) -> None:
    """Add or list file context.  ``/context <path>`` adds, ``/context`` lists, ``/context --clear`` removes all."""
    if not args:
        if not session.context_files:
            print("[maestro] no context files loaded. Use /context <path> to add.")
        else:
            print(f"[maestro] {len(session.context_files)} context file(s):")
            for fpath in session.context_files:
                size = len(session.context_files[fpath])
                print(f"  {fpath} ({size:,} chars)")
        return

    if args[0] == "--clear":
        count = len(session.context_files)
        session.context_files.clear()
        print(f"[maestro] cleared {count} context file(s)")
        return

    # Add file(s) to context
    for raw_path in args:
        path = Path(raw_path).resolve()
        if not path.is_file():
            print(f"[maestro] file not found: {raw_path}")
            continue
        try:
            content = _read_context_file(path)
        except Exception as exc:
            print(f"[maestro] error reading {raw_path}: {exc}")
            continue
        key = _context_display_key(path)
        session.context_files[key] = content
        print(f"[maestro] added {key} ({len(content):,} chars)")


def _session_to_dict(session: ChatSession) -> dict[str, Any]:
    """Serialize a ChatSession to a JSON-safe dict."""
    return {
        "engine": session.engine,
        "model": session.model,
        "execution_profile": session.execution_profile,
        "total_cost_usd": session.total_cost_usd,
        "total_turns": session.total_turns,
        "started_at": session.started_at,
        "context_files": session.context_files,
        "messages": [
            {
                "role": m.role,
                "engine": m.engine,
                "model": m.model,
                "content": m.content,
                "cost_usd": m.cost_usd,
                "duration_sec": m.duration_sec,
            }
            for m in session.messages
        ],
    }


def _session_from_dict(data: dict[str, Any]) -> ChatSession:
    """Deserialize a ChatSession from a JSON dict."""
    messages = [
        ChatMessage(
            role=m["role"],
            engine=m.get("engine", "claude"),
            model=m.get("model", "default"),
            content=m.get("content", ""),
            cost_usd=m.get("cost_usd"),
            duration_sec=m.get("duration_sec", 0.0),
        )
        for m in data.get("messages", [])
    ]
    return ChatSession(
        engine=data.get("engine", "claude"),
        model=data.get("model"),
        execution_profile=data.get("execution_profile", "plan"),
        messages=messages,
        context_files=data.get("context_files", {}),
        total_cost_usd=data.get("total_cost_usd", 0.0),
        total_turns=data.get("total_turns", 0),
        started_at=data.get("started_at", ""),
    )


def _cmd_save(session: ChatSession) -> None:
    """Save session to ``.maestro-cache/sessions/``."""
    sessions_dir = Path.cwd() / _SESSIONS_DIR
    sessions_dir.mkdir(parents=True, exist_ok=True)

    # Generate filename from timestamp
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    session_file = sessions_dir / f"chat_{ts}.json"

    try:
        session_file.write_text(
            json.dumps(_session_to_dict(session), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[maestro] session saved to {session_file.relative_to(Path.cwd())}")
    except Exception as exc:
        print(f"[maestro] error saving session: {exc}")


def _cmd_load(args: list[str], session: ChatSession) -> ChatSession:
    """Load a saved session.  ``/load`` loads latest, ``/load <path>`` loads specific file."""
    sessions_dir = Path.cwd() / _SESSIONS_DIR

    if args:
        session_file = Path(args[0])
    else:
        # Find most recent session file
        if not sessions_dir.is_dir():
            print("[maestro] no saved sessions found")
            return session
        files = sorted(sessions_dir.glob("chat_*.json"), reverse=True)
        if not files:
            print("[maestro] no saved sessions found")
            return session
        session_file = files[0]

    if not session_file.is_file():
        print(f"[maestro] session file not found: {session_file}")
        return session

    try:
        data = json.loads(session_file.read_text(encoding="utf-8"))
        loaded = _session_from_dict(data)
        print(
            f"[maestro] loaded session: {loaded.total_turns} turns, "
            f"{len(loaded.messages)} messages, "
            f"{len(loaded.context_files)} context file(s)"
        )
        return loaded
    except Exception as exc:
        print(f"[maestro] error loading session: {exc}")
        return session


def _cmd_help_chat() -> None:
    """Print chat help."""
    help_text = {
        "/model": "/model [engine/model]  Switch active engine and/or model",
        "/models": "/models              List available engines and models",
        "/context": "/context [path...]    Add file(s) to context (--clear to remove)",
        "/save": "/save                Save session to .maestro-cache/sessions/",
        "/load": "/load [path]          Load saved session (latest if no path)",
        "/clear": "/clear               Clear conversation history",
        "/cost": "/cost                Show session cost summary",
        "/help": "/help                Show this help",
        "/quit": "/quit                Exit chat",
    }
    print("Commands:")
    for cmd in _CHAT_COMMANDS:
        print(f"  {help_text[cmd]}")
    print()
    print("Routing:")
    print("  @engine <message>    Send to a specific engine for one turn")
    print("  (e.g., @codex optimize this query)")
    print()


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------


def _dispatch_chat_command(line: str, session: ChatSession) -> bool | ChatSession:
    """Handle slash commands.

    Returns ``False`` for ``/quit``, ``True`` to continue, or a new
    ``ChatSession`` when ``/load`` replaces the active session.
    """
    parts = line.split()
    cmd = parts[0].lower()
    args = parts[1:]

    if cmd == "/quit":
        return False
    elif cmd == "/model":
        _cmd_model(args, session)
    elif cmd == "/models":
        _cmd_models()
    elif cmd == "/context":
        _cmd_context(args, session)
    elif cmd == "/save":
        _cmd_save(session)
    elif cmd == "/load":
        return _cmd_load(args, session)
    elif cmd == "/clear":
        _cmd_clear(session)
    elif cmd == "/cost":
        _cmd_cost(session)
    elif cmd == "/help":
        _cmd_help_chat()
    else:
        print(f"[maestro] unknown command: {cmd}. Type /help for available commands.")

    return True


# ---------------------------------------------------------------------------
# Readline tab completion
# ---------------------------------------------------------------------------


def _setup_chat_readline() -> None:
    """Configure tab completion for chat commands and ``@engine`` prefixes."""
    try:
        import readline
    except ImportError:
        try:
            import pyreadline3 as readline  # type: ignore[import-not-found, no-redef]
        except ImportError:
            return

    completions: list[str] = list(_CHAT_COMMANDS) + [
        f"@{e}" for e in sorted(_VALID_ENGINES)
    ]

    def _complete(text: str, state: int) -> str | None:
        matches: list[str] = [c for c in completions if c.startswith(text)]
        if state < len(matches):
            return matches[state]
        return None

    readline.set_completer(_complete)  # type: ignore[attr-defined]
    readline.parse_and_bind("tab: complete")  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Main REPL
# ---------------------------------------------------------------------------


def run_chat(
    engine: str = "claude",
    model: str | None = None,
    execution_profile: ExecutionProfile = "plan",
    *,
    auto_context: bool = True,
) -> int:
    """Interactive multi-model chat terminal.  Returns exit code."""
    session = ChatSession(
        engine=engine,
        model=model,
        execution_profile=execution_profile,
        started_at=datetime.now(UTC).isoformat(),
    )
    _setup_chat_readline()

    # Welcome
    model_tag = session.model or "default"
    print(f"[maestro] chat mode — {session.engine}/{model_tag}")
    if auto_context:
        _autoload_context_files(session)
    print("[maestro] type a message, @engine to route, /help for commands, /quit to exit\n")

    while True:
        try:
            line = input("maestro> ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            break

        if not line:
            continue

        # Slash commands
        if line.startswith("/"):
            result = _dispatch_chat_command(line, session)
            if result is False:
                break
            if isinstance(result, ChatSession):
                session = result
            continue

        # Parse @engine prefix
        turn_engine, text = _parse_engine_prefix(line)
        if not text:
            continue

        # Record user message
        active_engine = turn_engine or session.engine
        active_model = session.model
        session.messages.append(ChatMessage(
            role="user",
            engine=active_engine,
            model=active_model or "default",
            content=text,
        ))

        # Execute turn
        turn_result = _run_chat_turn(
            session,
            text,
            engine=turn_engine,
            model=None,  # use session model (or engine default)
        )

        if turn_result is not None:
            session.messages.append(turn_result)
            session.total_turns += 1
            if turn_result.cost_usd is not None:
                session.total_cost_usd += turn_result.cost_usd

    # Farewell
    cost_str = f"${session.total_cost_usd:.4f}" if session.total_cost_usd > 0 else "--"
    print(f"[maestro] session: {session.total_turns} turns, cost: {cost_str}")
    return 0
