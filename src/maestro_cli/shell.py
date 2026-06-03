from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType


@dataclass
class ShellState:
    active_plan: Path | None = None
    last_run_dir: Path | None = None
    history: list[str] = field(default_factory=list)


_COMMANDS: list[str] = [
    "/run",
    "/validate",
    "/suggest",
    "/status",
    "/explain",
    "/plan",
    "/last",
    "/help",
    "/quit",
]


def _setup_readline() -> None:
    readline_module: ModuleType | None = None

    try:
        import readline as _rl  # type: ignore[import-not-found, unused-ignore]
        readline_module = _rl
    except Exception:
        try:
            import pyreadline3  # type: ignore[import-not-found, unused-ignore]  # noqa: F401
            import readline as _rl2  # type: ignore[import-not-found, unused-ignore]
            readline_module = _rl2
        except Exception:
            return

    if readline_module is None:
        return

    def _complete(text: str, state: int) -> str | None:
        matches: list[str] = []

        for command in _COMMANDS:
            if command.startswith(text):
                matches.append(command)

        try:
            for path in Path.cwd().iterdir():
                if path.suffix in {".yaml", ".yml"} and path.name.startswith(text):
                    matches.append(path.name)
        except OSError:
            pass

        if state < len(matches):
            return matches[state]
        return None

    readline_module.set_completer(_complete)  # type: ignore[attr-defined, unused-ignore]
    readline_module.parse_and_bind("tab: complete")  # type: ignore[attr-defined, unused-ignore]


def _print_help(command: str | None = None) -> None:
    help_text: dict[str, str] = {
        "/plan": "/plan <path>        Set active plan file",
        "/validate": "/validate           Validate active plan",
        "/run": "/run [--dry-run]    Run active plan",
        "/suggest": "/suggest            Suggest improvements for active plan",
        "/status": "/status             Show status for active plan",
        "/explain": "/explain            Explain active plan",
        "/last": "/last               Show last run directory",
        "/help": "/help [cmd]         Show help",
        "/quit": "/quit               Exit shell",
    }

    if command:
        msg = help_text.get(command)
        if msg:
            print(msg)
        else:
            print(f"[maestro] Unknown command for help: {command}")
        return

    print("Available commands:")
    for cmd in _COMMANDS:
        print(f"  {help_text[cmd]}")


def _dispatch_command(line: str, state: ShellState) -> bool:
    state.history.append(line)
    parts = line.split()
    command = parts[0]
    args = parts[1:]

    if command == "/quit":
        return False

    if command == "/plan":
        if not args:
            print("[maestro] Usage: /plan <path>")
            return True
        plan_path = Path(args[0]).expanduser()
        if not plan_path.exists():
            print(f"[maestro] Plan not found: {plan_path}")
            return True
        state.active_plan = plan_path
        print(f"[maestro] Active plan set: {state.active_plan}")
        return True

    if command == "/validate":
        _cmd_validate_in_shell(state)
        return True

    if command == "/run":
        dry_run = False
        for arg in args:
            if arg == "--dry-run":
                dry_run = True
            else:
                print(f"[maestro] Unknown option for /run: {arg}")
                return True
        run_path = _cmd_run_in_shell(state, dry_run=dry_run)
        if run_path is not None:
            state.last_run_dir = run_path
        return True

    if command == "/suggest":
        _cmd_suggest_in_shell(state)
        return True

    if command == "/status":
        _cmd_status_in_shell(state)
        return True

    if command == "/explain":
        _cmd_explain_in_shell(state)
        return True

    if command == "/last":
        if state.last_run_dir:
            print(state.last_run_dir)
        else:
            print("No runs yet")
        return True

    if command == "/help":
        _print_help(args[0] if args else None)
        return True

    print(f"[maestro] Unknown command: {command}. Type /help for commands.")
    return True


def _cmd_validate_in_shell(state: ShellState) -> None:
    if not state.active_plan:
        print("[maestro] No active plan. Use /plan <path> first.")
        return
    from .cli import _cmd_validate

    _cmd_validate(str(state.active_plan))


def _cmd_run_in_shell(state: ShellState, dry_run: bool = False) -> Path | None:
    if not state.active_plan:
        print("[maestro] No active plan. Use /plan <path> first.")
        return None
    import argparse
    from .cli import _cmd_run, _find_latest_run
    from .loader import load_plan

    args = argparse.Namespace(
        plan=[str(state.active_plan)],
        dry_run=dry_run,
        max_parallel=None,
        only=None,
        skip=None,
        tags=None,
        skip_tags=None,
        run_dir=None,
        execution_profile="plan",
        resume=None,
        resume_last=False,
        verbose=False,
        quiet=False,
        output="text",
        webhook=None,
        mask_secrets=False,
        auto_approve=False,
        no_cache=False,
        cache_dir=None,
        parallel=False,
    )
    rc = _cmd_run(args)
    if rc != 0:
        return None

    plan = load_plan(str(state.active_plan))
    return _find_latest_run(plan, run_dir=args.run_dir)


def _cmd_suggest_in_shell(state: ShellState) -> None:
    if not state.active_plan:
        print("[maestro] No active plan. Use /plan <path> first.")
        return
    from .loader import load_plan
    from .suggest import format_suggestions, suggest_plan

    plan = load_plan(str(state.active_plan))
    from pathlib import Path

    result = suggest_plan(plan, Path(plan.run_dir))
    print(format_suggestions(result))


def _cmd_status_in_shell(state: ShellState) -> None:
    if not state.active_plan:
        print("[maestro] No active plan. Use /plan <path> first.")
        return
    from .cli import _cmd_status
    import argparse

    args = argparse.Namespace(
        plan=str(state.active_plan),
        cache_dir=None,
        run_dir=None,
        json=False,
    )
    _cmd_status(args)


def _cmd_explain_in_shell(state: ShellState) -> None:
    if not state.active_plan:
        print("[maestro] No active plan. Use /plan <path> first.")
        return
    from .cli import _cmd_explain
    import argparse

    args = argparse.Namespace(
        plan=str(state.active_plan),
        cache_dir=None,
        json=False,
    )
    _cmd_explain(args)


def run_shell(plan_path: Path | None = None) -> int:
    state = ShellState(active_plan=plan_path)
    _setup_readline()

    print("[maestro] Interactive shell. Type /help for commands, /quit to exit.")
    if plan_path:
        print(f"[maestro] Active plan: {plan_path}")

    while True:
        try:
            line = input("maestro> ").strip()
            if not line:
                continue
            if not _dispatch_command(line, state):
                break
        except (KeyboardInterrupt, EOFError):
            print()
            break

    return 0


if __name__ == "__main__":
    plan = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    raise SystemExit(run_shell(plan))
