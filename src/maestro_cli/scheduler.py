from __future__ import annotations

import dataclasses
import json
import math
import os
import re
import shutil
import sys
import threading
import urllib.request
import uuid
from collections import deque
from collections.abc import Callable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from time import perf_counter
from typing import Any, cast

from .cache import cache_lookup, cache_store, compute_plan_hash, compute_task_hash
from .contracts import build_consistency_template_vars, build_contract_template_vars
from .eventsource import ChainState, compute_artefact_hash, emit_hashed_event
from .worktree import reset_merge_ledger
from .models import ExecutionProfile, KnowledgeRecord, OutputMode, PlanRunResult, PlanSpec, StructuredContext, TaskContract, TaskHistory, TaskResult, TaskSpec, TaskStatus, TokenUsage, Verbosity
from .plugins import PluginResolutionError, get_engine_plugin
from .relationships import build_consistency_group_members, clone_tasks_with_resolved_dependencies
from .runners import (
    _apply_progressive_compaction,
    _build_layered_context,
    _build_recursive_context,
    _build_secret_values,
    _build_selective_context,
    _build_structural_context,
    _build_knowledge_graph_context,
    _build_codebase_map_context,
    _build_scip_context,
    _compact_context,
    _filter_context_fields,
    _mask_secrets,
    _redact_output,
    _resolve_context_model,
    _run_map_reduce,
    _run_summarization,
    execute_task,
    kill_all_active,
    resolve_workdir,
)
from .policy import evaluate_policies
from .utils import evaluate_when_condition, extract_prompt_from_markdown, now_utc, resolve_path, sanitize_dirname

# ---------------------------------------------------------------------------
# ANSI color support
# ---------------------------------------------------------------------------

_NO_COLOR = os.environ.get("NO_COLOR") is not None
_IS_TTY = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _enable_win_ansi() -> None:
    """Enable ANSI/VT100 escape sequences on Windows consoles."""
    if os.name != "nt":
        return
    try:
        import ctypes
        kernel32 = getattr(ctypes, "windll").kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_ulong()
        kernel32.GetConsoleMode(handle, ctypes.byref(mode))
        kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except Exception:
        pass


_enable_win_ansi()


def _c(code: str, text: str) -> str:
    """Wrap *text* in an ANSI escape sequence (respects NO_COLOR and TTY)."""
    if _NO_COLOR or not _IS_TTY:
        return text
    return f"\033[{code}m{text}\033[0m"


def _dim(t: str) -> str:
    return _c("2", t)


def _bold(t: str) -> str:
    return _c("1", t)


def _cyan(t: str) -> str:
    return _c("36", t)


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("1;31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _magenta(t: str) -> str:
    return _c("35", t)


# ---------------------------------------------------------------------------
# Timestamp
# ---------------------------------------------------------------------------


def _local_timestamp() -> str:
    """Return current local time as HH:MM:SS for log lines."""
    from datetime import datetime as _dt
    return _dt.now().astimezone().strftime("%H:%M:%S")


def _event_timestamp() -> str:
    """Return current UTC time in ISO8601 with Z suffix for JSONL events."""
    return now_utc().isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_SUCCESS_LIKE = {"success", "soft_failed", "dry_run", "skipped"}

_print_lock = threading.Lock()


def _log(msg: str, progress: str = "") -> None:
    """Print a timestamped log line. *progress* adds a [X/N] counter."""
    ts = _local_timestamp()
    prefix = _dim(f"[maestro {ts}]")
    prog = f" {_bold(f'[{progress}]')}" if progress else ""
    with _print_lock:
        print(f"{prefix}{prog} {msg}", flush=True)


def _log_meta(msg: str) -> None:
    """Log a meta/header line (no progress counter, dimmed prefix)."""
    ts = _local_timestamp()
    prefix = _dim(f"[maestro {ts}]")
    with _print_lock:
        print(f"{prefix} {_dim(msg)}", flush=True)


# ---------------------------------------------------------------------------
# Model tag resolution
# ---------------------------------------------------------------------------

def _resolve_model(plan: PlanSpec, task: TaskSpec) -> str:
    """Resolve effective model for a task (task-level or plan default)."""
    if task.engine == "codex":
        return task.model or (plan.defaults.codex.model if plan.defaults.codex.model else "")
    if task.engine == "claude":
        return task.model or (plan.defaults.claude.model if plan.defaults.claude.model else "")
    if task.engine == "copilot":
        return task.model or (plan.defaults.copilot.model if plan.defaults.copilot.model else "")
    return task.model or ""


def _resolve_reasoning_effort(plan: PlanSpec, task: TaskSpec) -> str:
    """Resolve effective reasoning effort for a task (task-level or plan default)."""
    if task.engine == "codex":
        return task.reasoning_effort or (
            plan.defaults.codex.reasoning_effort if plan.defaults.codex.reasoning_effort else ""
        )
    if task.engine == "claude":
        return task.reasoning_effort or (
            plan.defaults.claude.reasoning_effort if plan.defaults.claude.reasoning_effort else ""
        )
    # Copilot CLI does not support reasoning_effort — always empty
    if task.engine == "copilot":
        return ""
    return task.reasoning_effort or ""


def _task_uses_exclusive_mcp_worktree_slot(plan: PlanSpec, task: TaskSpec) -> bool:
    if not task.worktree or not task.mcp_tools or not plan.mcp_servers:
        return False

    server_map = {server.name: server for server in plan.mcp_servers}
    for server_name in task.mcp_tools:
        server = server_map.get(server_name)
        if server is not None and server.is_concurrency_safe is False:
            return True
    return False


def _format_model_tag(plan: PlanSpec, task: TaskSpec) -> str:
    """Build display tag combining engine, model, and reasoning effort."""
    engine = task.engine or ""
    model = _resolve_model(plan, task)
    effort = _resolve_reasoning_effort(plan, task)

    # Build the inner part: model@effort, model, or just effort
    if model and effort:
        detail = f"{model}@{effort}"
    elif model:
        detail = model
    elif effort:
        detail = effort
    else:
        detail = ""

    # Always prefix with engine when available
    if engine and detail:
        return f"{engine}:{detail}"
    return engine or detail


def _model_suffix(plan: PlanSpec, task: TaskSpec) -> str:
    """Return colored model tag like ' [claude:sonnet]' or empty string."""
    tag = _format_model_tag(plan, task)
    return f" {_magenta(f'[{tag}]')}" if tag else ""


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------

def _fmt_duration(secs: float) -> str:
    """Format seconds into a human-readable string."""
    if secs < 60:
        return f"{secs:.0f}s"
    mins = int(secs // 60)
    remaining = int(secs % 60)
    return f"{mins}m{remaining:02d}s"


_TAIL_LINES = 8  # lines to show from failed task logs
_VERBOSITY_LEVELS: dict[str, int] = {"quiet": 0, "normal": 1, "verbose": 2}

# ---------------------------------------------------------------------------
# Token estimation (context budget awareness — F1)
# ---------------------------------------------------------------------------

_CHARS_PER_TOKEN_ESTIMATE = 4
_DECAY_BASE = 0.8

# ---------------------------------------------------------------------------
# Workspace-aware timeout estimation (T0.1)
# ---------------------------------------------------------------------------
_WSAT_BYTES_PER_TOKEN: float = 3.5        # typical Python/YAML source density
_WSAT_SECS_PER_TOKEN: float = 0.08        # 80 ms/token of file content (conservative)
_WSAT_BASE_SEC: int = 300                 # minimum floor for adjusted timeouts
_WSAT_MAX_SEC: int = 3600                 # hard cap (1 hour)
_WSAT_MIN_FILE_BYTES: int = 10_000        # ~3K tokens / ~200 lines; files below ignored
# Match relative file paths commonly referenced in prompts (src/, tests/, scripts/)
_WSAT_PATH_RE = re.compile(
    r"(?:src|tests|scripts)/[\w./\\-]+\.(?:py|yaml|yml|ts|js|rs|go|md|toml)"
)
_LAYERED_SECTION_RE = re.compile(r"^--- (?P<upstream_id>.+?) ---\n", re.MULTILINE)
_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "dare",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as",
    "into", "through", "during", "before", "after", "above", "below",
    "between", "out", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "each",
    "every", "both", "few", "more", "most", "other", "some", "such", "no",
    "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "because", "but", "and", "or", "if", "while", "that", "this", "these",
    "those", "it", "its", "my", "your", "his", "her", "our", "their",
    "what", "which", "who", "whom", "use", "using", "used",
})
_KEYWORD_RE = re.compile(r"[a-z0-9_]{2,}", re.IGNORECASE)
_SECTION_SPLIT_RE = re.compile(r"\n\s*\n+")


def _estimate_tokens(text: str) -> int:
    """Estimate token count using a conservative character-to-token heuristic."""
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


def _show_fail_tail(log_path: Path) -> None:
    """Print last N meaningful lines from a failed task's log file."""
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return

    lines = text.splitlines()
    # Strip header lines (task=, started_at=, workdir=, command=, blank)
    # and footer lines (status=, message=)
    body: list[str] = []
    in_body = False
    for line in lines:
        if not in_body:
            # Skip until we pass the blank line after the command header
            if line == "":
                in_body = True
            continue
        # Stop at footer
        if line.startswith(("status=", "message=")):
            continue
        body.append(line)

    if not body:
        return

    tail = body[-_TAIL_LINES:]
    with _print_lock:
        for line in tail:
            print(f"         {_dim('|')} {_dim(line)}", flush=True)


# ---------------------------------------------------------------------------
# Task selection
# ---------------------------------------------------------------------------

def _select_tasks(
    plan: PlanSpec,
    only: set[str] | None,
    skip: set[str] | None,
    tags: set[str] | None = None,
    skip_tags: set[str] | None = None,
) -> list[TaskSpec]:
    resolved_tasks = clone_tasks_with_resolved_dependencies(plan.tasks)
    task_map = {task.id: task for task in resolved_tasks}

    if only:
        selected: set[str] = set()

        def collect(task_id: str) -> None:
            if task_id in selected:
                return
            selected.add(task_id)
            for dep in task_map[task_id].depends_on:
                collect(dep)

        for task_id in only:
            if task_id not in task_map:
                raise ValueError(f"Unknown --only task: {task_id}")
            collect(task_id)
    else:
        selected = set(task_map)

    if skip:
        for task_id in skip:
            if task_id not in task_map:
                raise ValueError(f"Unknown --skip task: {task_id}")
        selected -= skip

    selected_tasks = [task for task in resolved_tasks if task.id in selected]

    if tags:
        selected_tasks = [t for t in selected_tasks if set(t.tags) & tags]
    if skip_tags:
        selected_tasks = [t for t in selected_tasks if not (set(t.tags) & skip_tags)]

    # Transitively add dependency tasks that were filtered out by tag filters
    if tags or skip_tags:
        selected_ids = {t.id for t in selected_tasks}
        pending = list(selected_ids)
        while pending:
            task_id = pending.pop()
            for dep in task_map[task_id].depends_on:
                if dep not in selected_ids:
                    selected_ids.add(dep)
                    pending.append(dep)
        selected_tasks = [task for task in resolved_tasks if task.id in selected_ids]

    return selected_tasks


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _new_skipped_result(
    task_id: str,
    run_path: Path,
    message: str,
) -> TaskResult:
    now = now_utc()
    log_path = run_path / f"{task_id}.log"
    result_path = run_path / f"{task_id}.result.json"

    result = TaskResult(
        task_id=task_id,
        status="skipped",
        exit_code=None,
        started_at=now,
        finished_at=now,
        duration_sec=0.0,
        command="",
        log_path=log_path,
        result_path=result_path,
        message=message,
    )

    log_path.write_text(f"status=skipped\nmessage={message}\n", encoding="utf-8")
    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


def _preflight_checks(
    plan: PlanSpec,
    selected_tasks: list[TaskSpec],
    dry_run: bool,
) -> None:
    if dry_run:
        return

    engines_needed: set[str] = set()
    for task in selected_tasks:
        if task.engine:
            engines_needed.add(task.engine)

    # Warn when running inside a Claude Code session — the Claude CLI may
    # refuse to start nested sessions, causing silent engine task failures.
    if "claude" in engines_needed and os.environ.get("CLAUDECODE"):
        _log_meta(
            "WARNING: CLAUDECODE env var detected — running inside a "
            "Claude Code session. engine: claude tasks may fail with "
            "'nested session' errors. Consider running from a clean terminal."
        )

    for engine in sorted(engines_needed):
        try:
            plugin = get_engine_plugin(engine)
        except PluginResolutionError as exc:
            raise ValueError(str(exc)) from None

        probe = plugin.doctor_probe
        executable = probe.executable if probe is not None else engine
        if shutil.which(executable):
            continue

        if probe is None or (probe.executable == engine and not probe.install_hint):
            raise ValueError(
                f"Engine '{engine}' not found on PATH. "
                f"Install it or check your PATH before running the plan."
            )

        detail = (
            f"Engine '{engine}' requires executable '{executable}', "
            "which was not found on PATH."
        )
        if probe.install_hint:
            detail += f" {probe.install_hint}"
        else:
            detail += " Install it or check your PATH before running the plan."
        raise ValueError(detail)

    # Detect UNC paths on Windows — CMD.EXE doesn't support them as cwd,
    # which causes verify_command / pre_command / guard_command failures.
    _unc_warned = False
    for task in selected_tasks:
        workdir = resolve_workdir(plan, task)
        if not workdir.exists():
            raise ValueError(
                f"Task '{task.id}' workdir does not exist: {workdir}"
            )
        if (
            not _unc_warned
            and os.name == "nt"
            and str(workdir).startswith("\\\\")
            and (task.verify_command or task.pre_command or task.guard_command)
        ):
            _unc_warned = True
            _log_meta(
                "WARNING: workspace resolves to a UNC path "
                f"({str(workdir)[:60]}...). CMD.EXE does not support UNC "
                "paths as working directory — string-format verify_command / "
                "pre_command / guard_command may fail. Use list-format "
                "commands with Git Bash or map the UNC path to a drive letter."
            )


def _load_prior_results(resume_path: Path) -> dict[str, str]:
    manifest_file = resume_path / "run_manifest.json"
    if not manifest_file.exists():
        raise ValueError(f"No run_manifest.json found in {resume_path}")

    raw = json.loads(manifest_file.read_text(encoding="utf-8"))
    task_results = raw.get("task_results", {})

    succeeded: dict[str, str] = {}
    for task_id, result in task_results.items():
        status = result.get("status")
        if status not in _SUCCESS_LIKE:
            continue
        # Dependency-failure and fail_fast-triggered skipped tasks should be
        # re-evaluated on resume, since the blocker task may now succeed.
        if status == "skipped":
            msg = result.get("message", "")
            if msg.startswith(("Skipped because dependency failed:", "fail_fast triggered by task")):
                continue
        succeeded[task_id] = status

    return succeeded


def _write_manifest(
    run_result: PlanRunResult,
    run_path: Path,
) -> Path:
    manifest_path = run_path / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(run_result.to_dict(), indent=2),
        encoding="utf-8",
    )
    return manifest_path


def _build_task_graph(tasks: list[TaskSpec]) -> dict[str, dict[str, Any]]:
    """Persist lightweight task metadata for post-run UI/API consumers."""
    graph: dict[str, dict[str, Any]] = {}
    for task in tasks:
        graph[task.id] = {
            "id": task.id,
            "description": task.description,
            "depends_on": list(task.depends_on),
            "agent": task.agent,
            "engine": task.engine,
            "model": task.model,
            "allow_failure": task.allow_failure,
            "worktree": task.worktree,
        }
    return graph


# PM3.2 — Test-count surfacing patterns. Each pattern matches a known test
# runner's summary line and yields a one-line label suitable for run_summary.md.
# Patterns are checked against stdout_tail; first match wins.
_TEST_SUMMARY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # pytest: "73 passed, 11 skipped, 10 warnings in 3.06s"
    # also supports failed / errors / xfailed / xpassed counts
    ("pytest", re.compile(
        r"(\d+)\s+passed"
        r"(?:,\s+(\d+)\s+failed)?"
        r"(?:,\s+(\d+)\s+skipped)?"
        r"(?:,\s+(\d+)\s+(?:error|errors))?"
        r"(?:,\s+(\d+)\s+xfailed)?"
        r"(?:,\s+(\d+)\s+xpassed)?"
        r"(?:,\s+(\d+)\s+warnings?)?"
        r"\s+in\s+[\d.]+",
    )),
    # jest: "Tests: 12 failed, 80 passed, 92 total"
    ("jest", re.compile(
        r"Tests?:\s*"
        r"(?:(\d+)\s+failed,\s*)?"
        r"(?:(\d+)\s+skipped,\s*)?"
        r"(\d+)\s+passed,\s*"
        r"(\d+)\s+total",
    )),
    # mocha: "94 passing" + optional "2 failing" / "5 pending"
    ("mocha", re.compile(
        r"(\d+)\s+passing"
        r"(?:.*?(\d+)\s+failing)?"
        r"(?:.*?(\d+)\s+pending)?",
        re.DOTALL,
    )),
)


def _extract_test_summary(stdout_tail: str) -> str | None:
    """Return a one-line test count summary if the output matches a known runner.

    Looks for pytest / jest / mocha summary lines in ``stdout_tail`` and emits
    a compact label like ``73 passed, 11 skipped`` (pytest) suitable for
    embedding in ``run_summary.md``. Returns ``None`` if no pattern matches —
    callers should treat that as "no test summary available".

    Added 2026-04-27 for PM3.2 (an internal post-mortem):
    a `command` task running pytest reported ``73 passed + 11 skipped`` but the
    summary just said ``success``, hiding how much was actually exercised.
    """
    if not stdout_tail:
        return None
    for runner, pattern in _TEST_SUMMARY_PATTERNS:
        match = pattern.search(stdout_tail)
        if not match:
            continue
        if runner == "pytest":
            passed = match.group(1)
            failed = match.group(2)
            skipped = match.group(3)
            errors = match.group(4)
            xfailed = match.group(5)
            xpassed = match.group(6)
            warnings = match.group(7)
            parts = [f"{passed} passed"]
            if failed and int(failed) > 0:
                parts.append(f"{failed} failed")
            if skipped and int(skipped) > 0:
                parts.append(f"{skipped} skipped")
            if errors and int(errors) > 0:
                parts.append(f"{errors} errors")
            if xfailed and int(xfailed) > 0:
                parts.append(f"{xfailed} xfailed")
            if xpassed and int(xpassed) > 0:
                parts.append(f"{xpassed} xpassed")
            if warnings and int(warnings) > 0:
                parts.append(f"{warnings} warnings")
            return ", ".join(parts) + " (pytest)"
        if runner == "jest":
            failed = match.group(1)
            skipped = match.group(2)
            passed = match.group(3)
            total = match.group(4)
            parts = [f"{passed} passed"]
            if failed and int(failed) > 0:
                parts.insert(0, f"{failed} failed")
            if skipped and int(skipped) > 0:
                parts.append(f"{skipped} skipped")
            parts.append(f"{total} total")
            return ", ".join(parts) + " (jest)"
        if runner == "mocha":
            passing = match.group(1)
            failing = match.group(2)
            pending = match.group(3)
            parts = [f"{passing} passing"]
            if failing and int(failing) > 0:
                parts.append(f"{failing} failing")
            if pending and int(pending) > 0:
                parts.append(f"{pending} pending")
            return ", ".join(parts) + " (mocha)"
    return None


def _compute_waves(
    plan: PlanSpec,
    run_result: PlanRunResult,
) -> list[list[str]]:
    """Compute execution waves (topological levels) from the plan DAG.

    Each wave contains tasks whose dependencies are all in prior waves.
    """
    resolved_tasks = clone_tasks_with_resolved_dependencies(plan.tasks)
    task_ids = [t.id for t in resolved_tasks if t.id in run_result.task_results]
    deps_map = {
        t.id: set(t.depends_on)
        for t in resolved_tasks
        if t.id in run_result.task_results
    }

    assigned: set[str] = set()
    waves: list[list[str]] = []
    remaining = set(task_ids)

    while remaining:
        wave = [
            tid for tid in task_ids
            if tid in remaining and deps_map[tid].issubset(assigned)
        ]
        if not wave:
            waves.append(sorted(remaining))
            break
        waves.append(wave)
        assigned.update(wave)
        remaining -= set(wave)

    return waves


def _write_summary(
    run_result: PlanRunResult,
    plan: PlanSpec,
    run_path: Path,
) -> Path:
    """Generate a human-readable run_summary.md in the run directory."""
    lines: list[str] = []

    status_label = "SUCCESS" if run_result.success else "FAILED"
    duration = (run_result.finished_at - run_result.started_at).total_seconds()

    ok_count = sum(1 for r in run_result.task_results.values() if r.status in {"success", "dry_run"})
    soft_count = sum(1 for r in run_result.task_results.values() if r.status == "soft_failed")
    fail_count = sum(1 for r in run_result.task_results.values() if r.status == "failed")
    skip_count = sum(1 for r in run_result.task_results.values() if r.status == "skipped")

    task_parts = [f"{ok_count} ok"]
    if soft_count:
        task_parts.append(f"{soft_count} soft_failed")
    task_parts.append(f"{fail_count} failed")
    task_parts.append(f"{skip_count} skipped")
    task_summary = " / ".join(task_parts)

    cost_str = (
        f"${run_result.total_cost_usd:.2f}"
        if run_result.total_cost_usd is not None
        else "---"
    )

    lines.append(f"# Run Summary: {run_result.plan_name}")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Run ID | `{run_result.run_id}` |")
    lines.append(f"| Status | **{status_label}** |")
    lines.append(f"| Duration | {_fmt_duration(duration)} |")
    lines.append(f"| Started | {run_result.started_at.isoformat()} |")
    lines.append(f"| Tasks | {task_summary} |")
    lines.append(f"| Cost | {cost_str} |")

    tokens_str = (
        f"{run_result.total_tokens:,}"
        if run_result.total_tokens is not None
        else "---"
    )
    lines.append(f"| Tokens | {tokens_str} |")

    if plan.max_cost_usd is not None:
        budget_status = "EXCEEDED" if run_result.budget_exceeded else "OK"
        lines.append(f"| Budget | ${plan.max_cost_usd:.2f} ({budget_status}) |")
    if run_result.sequential_duration_sec > 0:
        lines.append(
            f"| Parallelism | {_fmt_duration(duration)} wall / "
            f"{_fmt_duration(run_result.sequential_duration_sec)} seq "
            f"({run_result.parallelism_savings_pct:.0f}% saved) |"
        )
    lines.append(f"| Profile | {run_result.execution_profile} |")
    lines.append("")

    # Task table
    lines.append("## Tasks")
    lines.append("")
    lines.append("| Task | Status | Duration | Cost | Tokens | Engine |")
    lines.append("|------|--------|----------|------|--------|--------|")
    for task in plan.tasks:
        tr = run_result.task_results.get(task.id)
        if tr is None:
            continue
        cost_cell = f"${tr.cost_usd:.2f}" if tr.cost_usd is not None else "---"
        tokens_cell = (
            f"{tr.token_usage.total_tokens:,}" if tr.token_usage is not None else "---"
        )
        if task.group:
            engine_cell = f"group:{task.group}"
        elif task.engine:
            engine_cell = _format_model_tag(plan, task)
        else:
            engine_cell = "shell"
        lines.append(
            f"| {task.id} | {tr.status} | {_fmt_duration(tr.duration_sec)} "
            f"| {cost_cell} | {tokens_cell} | {engine_cell} |"
        )
    lines.append("")

    # Timeline
    waves = _compute_waves(plan, run_result)
    lines.append("## Timeline")
    lines.append("")
    # Waves are DAG topological levels (a task lands in wave N when all of its
    # depends_on are satisfied by waves < N), not runtime parallel slots. A wave
    # with more tasks than `max_parallel` was still scheduled correctly — it
    # just executed in multiple back-to-back batches under the hood.
    if any(len(wave) > plan.max_parallel for wave in waves):
        lines.append(
            f"_Waves are DAG topological levels, not runtime slots. "
            f"`max_parallel: {plan.max_parallel}` was respected at runtime even "
            f"when a wave lists more tasks._"
        )
        lines.append("")
    for wave_idx, wave_tasks in enumerate(waves):
        wave_results = [
            run_result.task_results[tid] for tid in wave_tasks
            if tid in run_result.task_results
        ]
        if not wave_results:
            continue
        wave_wall = max(r.duration_sec for r in wave_results)
        wave_cpu = sum(r.duration_sec for r in wave_results)
        task_list = ", ".join(
            f"{tid} ({_fmt_duration(run_result.task_results[tid].duration_sec)})"
            for tid in wave_tasks
            if tid in run_result.task_results
        )
        if len(wave_results) > 1:
            lines.append(
                f"- **Wave {wave_idx}**: {task_list} "
                f"— {_fmt_duration(wave_wall)} wall / {_fmt_duration(wave_cpu)} CPU"
            )
        else:
            lines.append(f"- **Wave {wave_idx}**: {task_list}")
    lines.append("")

    # Test results (PM3.2) — surface pytest / jest / mocha test counts when a
    # task's stdout_tail matches one of the runners. Even if the plan succeeds,
    # "73 passed, 11 skipped" tells the author how much was actually exercised
    # — the binary success bit alone doesn't.
    test_results: list[tuple[str, str]] = []
    for task in plan.tasks:
        if task.id not in run_result.task_results:
            continue
        tr = run_result.task_results[task.id]
        if not tr.stdout_tail:
            continue
        summary = _extract_test_summary(tr.stdout_tail)
        if summary:
            test_results.append((task.id, summary))
    if test_results:
        lines.append("## Test Results")
        lines.append("")
        for task_id, summary in test_results:
            lines.append(f"- **{task_id}**: {summary}")
        lines.append("")

    # Retried tasks (PM3.1) — show what each failed attempt saw when a later
    # attempt succeeded. Surfaces the diagnostic gap from an internal
    # post-mortem (2026-04-26): a `success` task
    # that retried once previously had no top-line indication of what the
    # agent corrected between attempts.
    retried_winners = [
        (task.id, run_result.task_results[task.id])
        for task in plan.tasks
        if task.id in run_result.task_results
        and run_result.task_results[task.id].status in ("success", "soft_failed")
        and run_result.task_results[task.id].failure_history
    ]
    if retried_winners:
        lines.append("## Retried Tasks")
        lines.append("")
        lines.append(
            "Tasks that recovered after one or more failed attempts. The "
            "`verify_tail` shows what each failed attempt saw — useful for "
            "understanding what the agent corrected without diffing log files."
        )
        lines.append("")
        for task_id, tr in retried_winners:
            lines.append(
                f"### {task_id} — {len(tr.failure_history)} failed attempt(s) "
                f"before {tr.status}"
            )
            lines.append("")
            for fr in tr.failure_history:
                duration_str = (
                    f", {_fmt_duration(fr.duration_sec)}"
                    if fr.duration_sec > 0.0 else ""
                )
                lines.append(
                    f"- **Attempt {fr.attempt}** "
                    f"(exit {fr.exit_code}, {fr.category}{duration_str}): "
                    f"{fr.message[:200]}"
                )
                if fr.verify_tail:
                    lines.append("  ```")
                    for vl in fr.verify_tail.splitlines():
                        lines.append(f"  {vl}")
                    lines.append("  ```")
            lines.append("")

    # Failed task details — show error messages and output tails
    failed_tasks = [
        (task.id, run_result.task_results[task.id])
        for task in plan.tasks
        if task.id in run_result.task_results
        and run_result.task_results[task.id].status == "failed"
    ]
    if failed_tasks:
        lines.append("## Failed Tasks")
        lines.append("")
        for task_id, tr in failed_tasks:
            lines.append(f"### {task_id}")
            lines.append("")
            if tr.message:
                lines.append(f"**Error**: {tr.message}")
                lines.append("")
            if tr.log_path:
                try:
                    rel_log = tr.log_path.relative_to(run_path)
                except ValueError:
                    rel_log = tr.log_path
                lines.append(f"**Log**: `{rel_log}`")
                lines.append("")
            if tr.stdout_tail:
                tail = tr.stdout_tail.strip().splitlines()[-5:]
                lines.append("**Output tail**:")
                lines.append("```")
                for tl in tail:
                    lines.append(tl)
                lines.append("```")
                lines.append("")

    summary_path = run_path / "run_summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def _post_completion_webhook(
    webhook_url: str,
    payload: dict[str, object],
) -> int:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        return int(getattr(response, "status", 200))


# ---------------------------------------------------------------------------
# Context budget helpers (F1)
# ---------------------------------------------------------------------------


def _extract_keywords(text: str) -> set[str]:
    return {
        m.group(0).lower()
        for m in _KEYWORD_RE.finditer(text)
        if m.group(0).lower() not in _STOPWORDS
    }


def _load_task_prompt_text(plan: PlanSpec, task: TaskSpec) -> str:
    if task.prompt:
        return task.prompt
    if task.prompt_file:
        prompt_path = resolve_path(plan.source_dir, task.prompt_file)
        if prompt_path is None or not prompt_path.exists():
            return ""
        try:
            return prompt_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
    if task.prompt_md_file and task.prompt_md_heading:
        md_path = resolve_path(plan.source_dir, task.prompt_md_file)
        if md_path is None or not md_path.exists():
            return ""
        try:
            md_text = md_path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return ""
        try:
            return extract_prompt_from_markdown(md_text, task.prompt_md_heading)
        except ValueError:
            return ""
    return ""


def _estimate_workspace_timeout(plan: PlanSpec, task: TaskSpec) -> int | None:
    """Return an adjusted timeout_sec if files mentioned in the prompt are large.

    Scans the task prompt for relative file paths, stats each against
    ``workspace_root``, sums their sizes, and derives a safe timeout.
    Returns ``None`` when no adjustment is warranted (files small / not found).
    """
    if task.engine is None or not plan.workspace_root:
        return None
    prompt_text = _load_task_prompt_text(plan, task)
    if not prompt_text:
        return None
    root = Path(plan.workspace_root)
    total_bytes = 0
    for rel_path in set(_WSAT_PATH_RE.findall(prompt_text)):
        try:
            sz = (root / rel_path).stat().st_size
            if sz >= _WSAT_MIN_FILE_BYTES:
                total_bytes += sz
        except OSError:
            pass
    if total_bytes < _WSAT_MIN_FILE_BYTES:
        return None
    estimated_tokens = total_bytes / _WSAT_BYTES_PER_TOKEN
    adjusted = int(_WSAT_BASE_SEC + estimated_tokens * _WSAT_SECS_PER_TOKEN)
    current = task.timeout_sec or plan.defaults.timeout_sec or 1800
    if adjusted <= current:
        return None
    return min(adjusted, _WSAT_MAX_SEC)


def _split_into_sections(text: str) -> list[str]:
    sections = [chunk.strip() for chunk in _SECTION_SPLIT_RE.split(text) if chunk.strip()]
    if len(sections) >= 2:
        return sections
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return []
    chunk_size = 8
    return [
        "\n".join(lines[idx: idx + chunk_size]).strip()
        for idx in range(0, len(lines), chunk_size)
    ]


def _compute_idf(sections: list[str]) -> dict[str, float]:
    if not sections:
        return {}

    doc_freq: dict[str, int] = {}
    for section in sections:
        for term in _extract_keywords(section):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    total_sections = len(sections)
    return {
        term: math.log(total_sections / (1 + freq))
        for term, freq in doc_freq.items()
    }


def _score_section(
    section: str,
    intent_keywords: set[str],
    idf: dict[str, float] | None = None,
    avg_section_len: float = 50.0,
) -> int:
    if not section or not intent_keywords:
        return 0

    if idf is None:
        return len(_extract_keywords(section) & intent_keywords)

    words = [m.group(0).lower() for m in _KEYWORD_RE.finditer(section)]
    if not words:
        return 0

    term_counts: dict[str, int] = {}
    for word in words:
        term_counts[word] = term_counts.get(word, 0) + 1

    section_len = len(words)
    norm_avg_len = avg_section_len if avg_section_len > 0 else 1.0
    k1 = 1.5
    b = 0.75
    score = 0.0

    for term in intent_keywords:
        tf = term_counts.get(term, 0)
        if tf <= 0:
            continue
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * section_len / norm_avg_len))
        score += idf.get(term, 1.0) * tf_norm

    if score <= 0:
        return 0
    return max(1, round(score))


def _filter_tail_by_intent(
    tail: str,
    intent_keywords: set[str],
    idf: dict[str, float] | None = None,
    avg_section_len: float = 50.0,
) -> tuple[str, int, list[str]]:
    if not tail or not intent_keywords:
        return tail, 0, []
    sections = _split_into_sections(tail)
    if not sections:
        return tail, 0, []

    def _score_sections(
        section_idf: dict[str, float] | None,
    ) -> list[tuple[str, int, set[str]]]:
        scored_sections: list[tuple[str, int, set[str]]] = []
        for section in sections:
            matched_keywords = _extract_keywords(section) & intent_keywords
            scored_sections.append(
                (
                    section,
                    _score_section(
                        section,
                        intent_keywords,
                        idf=section_idf,
                        avg_section_len=avg_section_len,
                    ),
                    matched_keywords,
                )
            )
        return scored_sections

    scored_sections = _score_sections(idf)
    kept_sections = [
        section
        for section, score, _matched_keywords in scored_sections
        if score > 0
    ]
    if not kept_sections and idf is not None:
        # Compatibility fallback: if BM25-style scoring yields no matches,
        # retry with legacy intersection scoring.
        scored_sections = _score_sections(None)
        kept_sections = [
            section
            for section, score, _matched_keywords in scored_sections
            if score > 0
        ]
    if not kept_sections:
        return tail, 0, []

    selection_score = sum(
        score
        for _section, score, _matched_keywords in scored_sections
        if score > 0
    )
    matched_keywords = sorted({
        keyword
        for _section, score, keywords in scored_sections
        if score > 0
        for keyword in keywords
    })
    return "\n\n".join(kept_sections), selection_score, matched_keywords


def _apply_intent_filtering(
    upstream: dict[str, TaskResult],
    intent_keywords: set[str] | None,
) -> tuple[
    dict[str, TaskResult],
    list[tuple[str, int, int]],
    dict[str, dict[str, object]],
]:
    """Filter upstream tails to sections matching downstream intent keywords.

    Returns:
        - Upstream results with filtered ``stdout_tail`` where applicable.
        - A list of ``(task_id, original_tokens, filtered_tokens)`` for every
          upstream entry reduced by filtering.
        - Per-upstream section selection metadata for explain/event output.
    """
    if not upstream or not intent_keywords:
        return upstream, [], {}

    all_sections: list[str] = []
    section_lengths: list[int] = []
    for task_result in upstream.values():
        for section in _split_into_sections(task_result.stdout_tail):
            all_sections.append(section)
            section_lengths.append(len([m.group(0) for m in _KEYWORD_RE.finditer(section)]))

    idf = _compute_idf(all_sections)
    avg_section_len = (
        sum(section_lengths) / len(section_lengths)
        if section_lengths
        else 50.0
    )

    result_map: dict[str, TaskResult] = {}
    filter_records: list[tuple[str, int, int]] = []
    selection_meta: dict[str, dict[str, object]] = {}

    for tid, task_result in upstream.items():
        original_tail = task_result.stdout_tail
        filtered_tail, selection_score, matched_keywords = _filter_tail_by_intent(
            original_tail,
            intent_keywords,
            idf=idf,
            avg_section_len=avg_section_len,
        )
        selection_meta[tid] = {
            "upstream_id": tid,
            "score": float(selection_score),
            "keywords_matched": matched_keywords,
        }
        if len(filtered_tail) < len(original_tail):
            orig_tokens = _estimate_tokens(original_tail)
            filtered_tokens = _estimate_tokens(filtered_tail) if filtered_tail else 0
            result_map[tid] = dataclasses.replace(task_result, stdout_tail=filtered_tail)
            filter_records.append((tid, orig_tokens, filtered_tokens))
        else:
            result_map[tid] = task_result

    if not filter_records:
        return upstream, [], selection_meta
    return result_map, filter_records, selection_meta


def _compute_hop_distances(
    task_id: str,
    context_from: list[str],
    all_tasks: dict[str, TaskSpec],
) -> dict[str, int]:
    """Compute graph hop distance from each context_from source to task_id.

    Direct dependencies = 1 hop. Dependencies of dependencies = 2 hops. Etc.
    Wildcard '*' entries get distance 1 (treated as direct).
    Returns {source_task_id: hop_count}.
    """
    task = all_tasks.get(task_id)
    if task is None:
        return {}

    sources: set[str] = set()
    for source in context_from:
        if source == "*":
            sources.update(task.depends_on)
        else:
            sources.add(source)
    if not sources:
        return {}

    distances: dict[str, int] = {}
    queue: deque[tuple[str, int]] = deque((dep, 1) for dep in task.depends_on)
    visited: set[str] = set()

    while queue:
        current, hops = queue.popleft()
        if current in visited:
            continue
        visited.add(current)
        if current in sources and current not in distances:
            distances[current] = hops
        current_task = all_tasks.get(current)
        if current_task is None:
            continue
        for dep in current_task.depends_on:
            if dep not in visited:
                queue.append((dep, hops + 1))

    for source in context_from:
        if source == "*":
            continue
        if source in task.depends_on:
            distances[source] = 1

    return distances


def _apply_hop_decay(
    upstream: dict[str, TaskResult],
    hop_distances: dict[str, int],
) -> dict[str, TaskResult]:
    """Apply decay factor to upstream stdout_tails based on hop distance.

    For hop distance h, keep only (DECAY_BASE^(h-1)) fraction of the tail.
    hop=1 (direct): keep 100%. hop=2: keep 80%. hop=3: keep 64%. etc.
    Returns new dict with trimmed copies (originals unchanged).
    """
    if not upstream:
        return upstream

    decayed: dict[str, TaskResult] = {}
    for tid, task_result in upstream.items():
        hops = hop_distances.get(tid, 1)
        if hops <= 1:
            decayed[tid] = task_result
            continue

        keep_fraction = _DECAY_BASE ** (hops - 1)
        keep_len = max(0, min(len(task_result.stdout_tail), math.floor(len(task_result.stdout_tail) * keep_fraction)))
        if keep_len >= len(task_result.stdout_tail):
            decayed[tid] = task_result
            continue
        decayed[tid] = dataclasses.replace(task_result, stdout_tail=task_result.stdout_tail[:keep_len])

    return decayed


def _rrf_score(
    bm25_scores: dict[str, float],
    hop_distances: dict[str, int],
    k: int = 60,
) -> dict[str, float]:
    """Reciprocal Rank Fusion of BM25 relevance scores and hop-distance proximity.

    Combines two ranking signals into a single score per upstream task.
    Higher scores indicate more relevant / closer upstreams.
    """
    all_ids = set(bm25_scores) | set(hop_distances)
    if not all_ids:
        return {}

    # Rank by BM25 score descending (best = rank 1)
    bm25_ranked = sorted(all_ids, key=lambda tid: bm25_scores.get(tid, 0.0), reverse=True)
    bm25_rank = {tid: rank for rank, tid in enumerate(bm25_ranked, 1)}

    # Rank by hop distance ascending (closest = rank 1)
    hop_ranked = sorted(all_ids, key=lambda tid: hop_distances.get(tid, 999))
    hop_rank = {tid: rank for rank, tid in enumerate(hop_ranked, 1)}

    return {
        tid: 1.0 / (k + bm25_rank.get(tid, len(all_ids))) + 1.0 / (k + hop_rank.get(tid, len(all_ids)))
        for tid in all_ids
    }


def _apply_context_budget(
    upstream: dict[str, TaskResult],
    budget_tokens: int,
    intent_keywords: set[str] | None = None,
    relevance_scores: dict[str, float] | None = None,
) -> tuple[
    dict[str, TaskResult],
    list[tuple[str, int, int]],
    dict[str, dict[str, object]],
]:
    """Trim upstream stdout_tail entries to fit within *budget_tokens*.

    If *intent_keywords* is provided and the upstream total exceeds budget,
    each tail is first filtered to sections that match the downstream intent.
    Any remaining overflow is trimmed by evicting least relevant upstream
    tails first.

    Returns:
        - A new dict with trimmed copies where needed (originals are unchanged).
        - A list of ``(task_id, original_tokens, trimmed_tokens)`` for every
          entry that was actually shortened.
        - Per-upstream section selection metadata for explain/event output.
    """
    items = [
        (tid, r, r.stdout_tail, _estimate_tokens(r.stdout_tail))
        for tid, r in upstream.items()
    ]
    total = sum(tok for _, _, _, tok in items)
    if total <= budget_tokens:
        if not intent_keywords:
            return upstream, [], {}
        _filtered_upstream, _filter_records, selection_meta = _apply_intent_filtering(
            upstream,
            intent_keywords,
        )
        return upstream, [], selection_meta

    filtered_upstream = upstream
    selection_meta = cast(dict[str, dict[str, object]], {})
    if intent_keywords:
        filtered_upstream, _filter_records, selection_meta = _apply_intent_filtering(
            upstream,
            intent_keywords,
        )

    prepared: list[tuple[str, TaskResult, str, int]] = [
        (tid, task_result, task_result.stdout_tail, _estimate_tokens(task_result.stdout_tail))
        for tid, task_result in filtered_upstream.items()
    ]

    prepared_total = sum(tok for _, _, _, tok in prepared)
    if prepared_total <= budget_tokens:
        result_map: dict[str, TaskResult] = {}
        trim_records: list[tuple[str, int, int]] = []
        for tid, _task_result, filtered_tail, _filtered_tokens in prepared:
            original_task_result = upstream[tid]
            orig_tokens = _estimate_tokens(original_task_result.stdout_tail)
            if len(filtered_tail) < len(original_task_result.stdout_tail):
                trimmed_tokens = _estimate_tokens(filtered_tail) if filtered_tail else 0
                result_map[tid] = dataclasses.replace(
                    original_task_result,
                    stdout_tail=filtered_tail,
                )
                trim_records.append((tid, orig_tokens, trimmed_tokens))
            else:
                result_map[tid] = original_task_result
        return result_map, trim_records, selection_meta

    if not intent_keywords:
        ratio = budget_tokens / prepared_total if prepared_total > 0 else 0.0
        result_map = cast(dict[str, TaskResult], {})
        trim_records = cast(list[tuple[str, int, int]], [])

        for tid, task_result, tail, filtered_tokens in prepared:
            orig_tokens = _estimate_tokens(task_result.stdout_tail)
            target_tokens = max(0, min(filtered_tokens, math.floor(filtered_tokens * ratio)))
            if target_tokens <= 0:
                trimmed_tail = ""
                trimmed_tokens = 0
            else:
                lo = 0
                hi = len(tail)
                best = 0
                while lo <= hi:
                    mid = (lo + hi) // 2
                    candidate = tail[:mid]
                    candidate_tokens = _estimate_tokens(candidate) if candidate else 0
                    if candidate_tokens <= target_tokens:
                        best = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                trimmed_tail = tail[:best]
                trimmed_tokens = _estimate_tokens(trimmed_tail) if trimmed_tail else 0

            if len(trimmed_tail) < len(task_result.stdout_tail):
                result_map[tid] = dataclasses.replace(task_result, stdout_tail=trimmed_tail)
                trim_records.append((tid, orig_tokens, trimmed_tokens))
            else:  # pragma: no cover - defensive: target_tokens<filtered always trims
                result_map[tid] = task_result

        return result_map, trim_records, {}

    overflow = prepared_total - budget_tokens
    scored_items: list[tuple[float, str, TaskResult, str, int]] = []
    for tid, task_result, tail, filtered_tokens in prepared:
        if relevance_scores and tid in relevance_scores:
            score = relevance_scores[tid]
        else:
            _raw_score = selection_meta.get(tid, {}).get("score", 0)
            score = float(cast(float, _raw_score))
        scored_items.append((score, tid, task_result, tail, filtered_tokens))

    scored_items.sort(key=lambda item: item[0])

    result_map = cast(dict[str, TaskResult], {})
    trim_records = cast(list[tuple[str, int, int]], [])
    for _score, tid, task_result, tail, filtered_tokens in scored_items:
        orig_tokens = _estimate_tokens(task_result.stdout_tail)
        if overflow <= 0 or not tail:
            result_map[tid] = task_result
            continue

        removable_tokens = min(overflow, filtered_tokens)
        target_tokens = filtered_tokens - removable_tokens
        if target_tokens <= 0:
            trimmed_tail = ""
            trimmed_tokens = 0
        else:
            lo = 0
            hi = len(tail)
            best = 0
            while lo <= hi:
                mid = (lo + hi) // 2
                candidate = tail[:mid]
                candidate_tokens = _estimate_tokens(candidate) if candidate else 0
                if candidate_tokens <= target_tokens:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            trimmed_tail = tail[:best]
            trimmed_tokens = _estimate_tokens(trimmed_tail) if trimmed_tail else 0

        overflow -= (filtered_tokens - trimmed_tokens)
        if len(trimmed_tail) < len(task_result.stdout_tail):
            result_map[tid] = dataclasses.replace(task_result, stdout_tail=trimmed_tail)
            trim_records.append((tid, orig_tokens, trimmed_tokens))
        else:  # pragma: no cover - defensive: removable>=1 means target<filtered always trims
            result_map[tid] = task_result

    return result_map, trim_records, selection_meta


def _parse_layered_context_sections(context_text: str) -> dict[str, str]:
    """Extract per-upstream section bodies from layered context output."""
    if not context_text:
        return {}

    matches = list(_LAYERED_SECTION_RE.finditer(context_text))
    if not matches:
        return {}

    sections: dict[str, str] = {}
    for index, match in enumerate(matches):
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(context_text)
        sections[match.group("upstream_id")] = context_text[body_start:body_end].strip()
    return sections


# ---------------------------------------------------------------------------
# RLM context helpers
# ---------------------------------------------------------------------------


def _prepare_summaries(
    upstream_results: dict[str, TaskResult],
    workdir: Path,
    model: str = "haiku",
) -> list[tuple[str, float]]:
    """Populate ``structured_context.summary`` for upstream results that lack one.

    Called by the scheduler before dispatching a downstream task with
    ``context_mode: summarized`` or ``context_mode: map_reduce``.
    Summaries are cached on the ``StructuredContext`` — if the same
    upstream is referenced by multiple downstream tasks, the haiku call
    happens only once.
    """
    summarized: list[tuple[str, float]] = []
    for tid, result in upstream_results.items():
        sc = result.structured_context
        if sc and not sc.summary:
            start = perf_counter()
            sc.summary = _run_summarization(tid, result.stdout_tail, sc, workdir, model=model)
            summarized.append((tid, perf_counter() - start))
    return summarized


# ---------------------------------------------------------------------------
# DAG structural helpers (for routing metadata)
# ---------------------------------------------------------------------------

def _compute_task_depth(task: TaskSpec, plan: PlanSpec) -> int:
    """Return the max dependency-chain depth for *task*. 0 = no deps."""
    if not task.depends_on:
        return 0
    task_map_local = {t.id: t for t in plan.tasks}
    seen: set[str] = set()

    def _depth(tid: str) -> int:
        if tid in seen or tid not in task_map_local:
            return 0
        seen.add(tid)
        t = task_map_local[tid]
        if not t.depends_on:
            return 1
        return 1 + max(_depth(d) for d in t.depends_on)

    return max(_depth(d) for d in task.depends_on)


def _compute_fan_out(task: TaskSpec, plan: PlanSpec) -> int:
    """Return the number of tasks that directly depend on *task*."""
    return sum(
        1 for t in plan.tasks
        if t.depends_on and task.id in t.depends_on
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _compute_task_hash_safe(
    task: TaskSpec,
    plan: PlanSpec,
    upstream_hashes: dict[str, str],
) -> str | None:
    """Compute the cache hash for a task, returning None on any error."""
    try:
        return compute_task_hash(task, plan, upstream_hashes)
    except Exception:
        return None


def _new_cached_result(
    task_id: str,
    run_path: Path,
    cached: dict[str, Any],
    task_hash: str,
    cache_dir: Path,
) -> TaskResult:
    """Reconstruct a TaskResult from a cache entry."""
    now = now_utc()
    log_path = run_path / f"{task_id}.log"
    result_path = run_path / f"{task_id}.result.json"

    tu_raw = cached.get("token_usage")
    token_usage: TokenUsage | None = None
    if tu_raw and isinstance(tu_raw, dict):
        token_usage = TokenUsage(
            input_tokens=int(tu_raw.get("input_tokens", 0)),
            cached_tokens=int(tu_raw.get("cached_tokens", 0)),
            output_tokens=int(tu_raw.get("output_tokens", 0)),
            cache_creation_tokens=int(tu_raw.get("cache_creation_tokens", 0)),
        )

    sc_raw = cached.get("structured_context")
    structured_context: StructuredContext | None = None
    if sc_raw and isinstance(sc_raw, dict):
        structured_context = StructuredContext(
            task_id=str(sc_raw.get("task_id", task_id)),
            status=str(sc_raw.get("status", "")),
            exit_code=sc_raw.get("exit_code"),
            duration_sec=float(sc_raw.get("duration_sec", 0.0)),
            files_changed=list(sc_raw.get("files_changed") or []),
            decisions=list(sc_raw.get("decisions") or []),
            errors=list(sc_raw.get("errors") or []),
            warnings=list(sc_raw.get("warnings") or []),
            cost_usd=sc_raw.get("cost_usd"),
            result_text=str(sc_raw.get("result_text", "")),
            summary=str(sc_raw.get("summary", "")),
        )

    contract_raw = cached.get("produced_contract")
    produced_contract: TaskContract | None = None
    if contract_raw and isinstance(contract_raw, dict):
        produced_contract = TaskContract(
            producer_task_id=str(contract_raw.get("producer_task_id", task_id)),
            contract_type=str(contract_raw.get("contract_type", "")),
            summary=str(contract_raw.get("summary", "")),
            body=str(contract_raw.get("body", "")),
            content_hash=str(contract_raw.get("content_hash", "")),
            metadata=dict(contract_raw.get("metadata") or {}),
        )

    result = TaskResult(
        task_id=task_id,
        status=cached["status"],
        exit_code=cached.get("exit_code"),
        started_at=now,
        finished_at=now,
        duration_sec=float(cached.get("duration_sec", 0.0)),
        command=str(cached.get("command", "")),
        log_path=log_path,
        result_path=result_path,
        message=f"Cache hit [{task_hash[:12]}]",
        stdout_tail=str(cached.get("stdout_tail", "")),
        cost_usd=cached.get("cost_usd"),
        token_usage=token_usage,
        structured_context=structured_context,
        produced_contract=produced_contract,
        retry_count=int(cached.get("retry_count", 0)),
        task_hash=task_hash,
        tainted=bool(cached.get("tainted", False)),
        tool_failure_count=int(cached.get("tool_failure_count", 0)),
    )

    cached_log = cache_dir / task_hash[:2] / task_hash / "task.log"
    if cached_log.exists():
        shutil.copy2(cached_log, log_path)
    else:
        log_path.write_text(
            f"status={result.status}\nmessage={result.message}\n",
            encoding="utf-8",
        )

    result_path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# Approval gates
# ---------------------------------------------------------------------------


def _request_approval(task_id: str, message: str | None, interactive: bool) -> bool:
    """Prompt user for approval. Returns True if approved."""
    if not interactive:
        return False
    msg = message or f"Task '{task_id}' requires approval."
    print(f"[maestro] {msg} [y/N] ", end="", flush=True)
    try:
        response = input().strip().lower()
        return response in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


# ---------------------------------------------------------------------------
# Untrusted context — Taint propagation
# ---------------------------------------------------------------------------


def _compute_tainted_tasks(
    plan: PlanSpec, selected_ids: set[str] | None = None,
) -> set[str]:
    """Compute the set of task IDs that are tainted via transitive propagation.

    A task is tainted if:
    1. It has ``context_trust: untrusted``, OR
    2. Any upstream in its ``context_from`` is tainted AND the task has
       neither ``guard_command`` nor ``verify_command`` to sanitize input.

    Taint is cleared by the presence of ``guard_command`` or ``verify_command``
    on the consuming task.
    """
    tasks = plan.tasks
    if selected_ids is not None:
        tasks = [t for t in tasks if t.id in selected_ids]
    task_map: dict[str, TaskSpec] = {t.id: t for t in tasks}
    tainted: set[str] = set()

    # Seed: explicitly untrusted tasks
    for task in tasks:
        if task.context_trust == "untrusted":
            tainted.add(task.id)

    # Propagate: iterate until stable (fixed-point)
    changed = True
    while changed:
        changed = False
        for task in tasks:
            if task.id in tainted:
                continue
            if not task.context_from:
                continue
            # Resolve context_from IDs
            ctx_ids: list[str] = []
            for entry in task.context_from:
                if entry == "*":
                    ctx_ids.extend(
                        dep for dep in task.depends_on if dep in task_map
                    )
                else:
                    ctx_ids.append(entry)
            # Check if any upstream is tainted
            if not any(uid in tainted for uid in ctx_ids):
                continue
            # Taint cleared by guard_command or verify_command
            if task.guard_command is not None or task.verify_command is not None:
                continue
            tainted.add(task.id)
            changed = True

    return tainted


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_plan(
    plan: PlanSpec,
    dry_run: bool = False,
    execution_profile: ExecutionProfile = "plan",
    max_parallel_override: int | None = None,
    only: set[str] | None = None,
    skip: set[str] | None = None,
    tags: set[str] | None = None,
    skip_tags: set[str] | None = None,
    run_dir_override: str | None = None,
    webhook_url: str | None = None,
    resume_path: Path | None = None,
    verbosity: Verbosity = "normal",
    output_mode: OutputMode = "text",
    cache_dir: Path | None = None,
    auto_approve: bool = False,
    event_callback: Callable[[str, dict[str, object]], None] | None = None,
    cancel_event: threading.Event | None = None,
    approval_handler: Callable[[str, str | None], bool] | None = None,
    extra_template_vars: dict[str, str] | None = None,
) -> PlanRunResult:
    """Execute a plan's task DAG with parallel scheduling.

    Uses a ``ThreadPoolExecutor`` with ``FIRST_COMPLETED`` wait strategy.
    Tasks are dispatched as their dependencies resolve.  Results are
    written to ``{run_path}/{task_id}.log`` and ``{task_id}.result.json``.

    Args:
        plan: Loaded and validated plan specification.
        dry_run: Build commands but skip actual execution.
        execution_profile: Safety mode (plan/safe/yolo).
        max_parallel_override: Override plan's ``max_parallel`` setting.
        only: Run only these task IDs (+ their transitive dependencies).
        skip: Exclude these task IDs from execution.
        run_dir_override: Override the output directory path.
        resume_path: Path to a prior run directory; succeeded tasks are skipped.
        verbosity: Output level — "quiet" (errors + final summary only),
            "normal" (default), or "verbose" (adds task output tails on success).
        output_mode: "text" (default human-readable) or "jsonl" (structured
            JSON Lines on stdout; suppresses all human-readable output).

    Returns:
        Aggregated ``PlanRunResult`` with per-task results and overall success.
    """
    started_at = now_utc()
    _jsonl_mode = output_mode == "jsonl"
    _v_level = _VERBOSITY_LEVELS.get(verbosity, 1)
    if _jsonl_mode:
        _v_level = -1  # suppress all human-readable output

    def _vlog(msg: str, progress: str = "", min_level: int = 1) -> None:
        if _v_level >= min_level:
            _log(msg, progress)

    def _vmeta(msg: str, min_level: int = 1) -> None:
        if _v_level >= min_level:
            _log_meta(msg)

    events_path: Path | None = None
    _event_callback = event_callback
    _secret_values: set[str] = set()
    if event_callback is not None and (plan.secrets or plan.secrets_auto):
        _secret_values = _build_secret_values(
            plan.secrets, plan.secrets_auto, plan.defaults.env, {}
        )

    chain_state = ChainState()

    def _emit(event_name: str, **data: object) -> None:
        """Emit a structured JSONL event to the run event log (and stdout in JSONL mode)."""
        payload = {"ts": _event_timestamp(), "event": event_name, "plan_name": plan.name, **data}
        with _print_lock:
            record = emit_hashed_event(payload, chain_state)
            payload["seq"] = record.sequence
            payload["prev_hash"] = record.prev_hash
            payload["hash"] = record.event_hash
            line = json.dumps(payload, default=str)
            if events_path is not None:
                with events_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            if _jsonl_mode:
                print(line, flush=True)
        if _event_callback is not None:
            try:
                if _secret_values:
                    masked_payload = {
                        k: _mask_secrets(v, _secret_values) if isinstance(v, str) else v
                        for k, v in payload.items()
                    }
                    _event_callback(event_name, masked_payload)
                else:
                    _event_callback(event_name, payload)
            except Exception:
                pass

    run_id = (
        f"{started_at.strftime('%Y%m%d_%H%M%S_%f')}_{uuid.uuid4().hex[:6]}"
    )

    run_root = run_dir_override or plan.run_dir
    run_root_path = resolve_path(plan.source_dir, run_root)
    if run_root_path is None:
        raise ValueError("Unable to resolve run directory")

    safe_name = sanitize_dirname(plan.name)
    run_path = (run_root_path / f"{run_id}_{safe_name}").resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    events_path = run_path / "events.jsonl"
    events_path.touch()

    # T1.2 — Reset merge ledger for fresh overlap tracking
    reset_merge_ledger()

    # Load historical task performance for predictive routing (T2.3)
    _task_histories: dict[str, TaskHistory] = {}
    if not dry_run:
        try:
            from .routing import load_task_histories
            _task_histories = load_task_histories(plan.name, run_root_path)
        except Exception:
            pass  # graceful degradation: no history = pure heuristic routing

    # Detect local hardware once for hardware-aware routing (v2.5.3) — only when
    # a task actually auto-routes a local engine, so non-local plans pay nothing.
    _hardware_info: object | None = None
    if not dry_run and any(
        t.model == "auto" and t.engine in ("ollama", "llama") for t in plan.tasks
    ):
        try:
            from .hardware import detect_hardware
            _hardware_info = detect_hardware()
        except Exception:
            pass  # graceful degradation: no hardware signal = tier default stands

    # Load cross-run knowledge for prompt injection (T1.3)
    _task_knowledge: dict[str, list[KnowledgeRecord]] = {}
    _knowledge_index = ""
    if not dry_run:
        try:
            from .knowledge import build_knowledge_index, load_knowledge
            _task_knowledge = load_knowledge(
                plan.name,
                plan.source_dir,
                max_per_task=None,
            )
            _knowledge_index = build_knowledge_index(plan.name, _task_knowledge)
        except Exception:
            pass  # graceful degradation: no knowledge = no injection

    selected_tasks = _select_tasks(plan, only=only, skip=skip, tags=tags, skip_tags=skip_tags)
    selected_group_members = build_consistency_group_members(selected_tasks)

    total = len(selected_tasks)

    # Cross-run budget gate
    if plan.budget_period and plan.max_cost_usd and not dry_run:
        from .budget import check_budget, _DEFAULT_LEDGER_PATH
        ledger_path = (plan.source_dir / _DEFAULT_LEDGER_PATH)
        allowed, spent, remaining = check_budget(
            ledger_path, plan.budget_period, plan.max_cost_usd,  # type: ignore[arg-type]
        )
        if not allowed:
            _vlog(
                f"[maestro] budget exceeded: ${spent:.2f} spent this {plan.budget_period} "
                f"(limit ${plan.max_cost_usd:.2f})",
                min_level=-1,
            )
            return PlanRunResult(
                plan_name=plan.name,
                run_id=run_id,
                success=False,
                task_results={},
                started_at=started_at,
                finished_at=now_utc(),
                sequential_duration_sec=0.0,
                run_path=run_path,
            )
        if _v_level >= 1:
            _vlog(
                f"budget: ${spent:.2f} / ${plan.max_cost_usd:.2f} this {plan.budget_period} "
                f"(${remaining:.2f} remaining)",
                min_level=1,
            )

    # Pre-flight checks (engine availability + workdir existence)
    _preflight_checks(plan, selected_tasks, dry_run)

    # Resume: load prior succeeded tasks
    prior_succeeded: dict[str, str] = {}
    if resume_path:
        prior_succeeded = _load_prior_results(resume_path)
        _vmeta(f"resume from={resume_path} prior_ok={len(prior_succeeded)}")

    task_map = {task.id: task for task in selected_tasks}
    selected_ids = set(task_map)

    deps_map: dict[str, set[str]] = {
        task.id: {dep for dep in task.depends_on if dep in selected_ids}
        for task in selected_tasks
    }
    dependents_map: dict[str, list[str]] = {task.id: [] for task in selected_tasks}
    for task in selected_tasks:
        for dep in deps_map[task.id]:
            dependents_map[dep].append(task.id)

    # Precompute DAG wave index for event metadata.
    task_waves: dict[str, int] = {}
    unassigned = [task.id for task in selected_tasks]
    while unassigned:
        progressed = False
        for task_id in list(unassigned):
            deps = deps_map[task_id]
            if deps and not deps.issubset(task_waves):
                continue
            task_waves[task_id] = 0 if not deps else max(task_waves[d] for d in deps) + 1
            unassigned.remove(task_id)
            progressed = True
        if not progressed:
            for task_id in unassigned:
                task_waves[task_id] = 0
            break

    results: dict[str, TaskResult] = {}
    completed: set[str] = set()
    pending: set[str] = set(task_map)
    task_hashes: dict[str, str] = {}  # task_id -> sha256 hash for cache keying

    # Untrusted context — compute tainted task set
    _tainted_tasks = _compute_tainted_tasks(plan, selected_ids)

    # T2.2 — Budget getter for mid-task signal queries
    _budget_lock = threading.Lock()

    def _budget_getter() -> tuple[float | None, float | None]:
        if plan.max_cost_usd is None:
            return None, None
        with _budget_lock:
            spent = sum(
                r.cost_usd for r in results.values() if r.cost_usd is not None
            )
            return max(0.0, plan.max_cost_usd - spent), plan.max_cost_usd

    def _progress() -> str:
        return f"{len(completed)}/{total}"

    # Pre-populate resumed tasks
    for task_id, prior_status in prior_succeeded.items():
        if task_id in pending:
            pending.remove(task_id)
            completed.add(task_id)
            now = now_utc()
            log_path = run_path / f"{task_id}.log"
            result_path = run_path / f"{task_id}.result.json"
            result = TaskResult(
                task_id=task_id,
                status=cast(TaskStatus, prior_status),
                exit_code=0,
                started_at=now,
                finished_at=now,
                duration_sec=0.0,
                command="",
                log_path=log_path,
                result_path=result_path,
                message=f"Resumed from prior run ({prior_status})",
            )
            results[task_id] = result
            log_path.write_text(
                f"status={prior_status}\nmessage=resumed from prior run\n",
                encoding="utf-8",
            )
            result_path.write_text(
                json.dumps(result.to_dict(), indent=2), encoding="utf-8"
            )
            _vlog(
                f"{_dim('resumed')} {_bold(task_id)} {_dim(f'(prior: {prior_status})')}",
                progress=_progress(),
            )
            _emit("task_complete", task_id=task_id, status=prior_status, duration_sec=0.0, cost_usd=None, tokens=None)

    tasks_with_when = {t.id for t in selected_tasks if t.when}

    def _evaluate_ready(tid: str) -> bool:
        """Check if a task should be added to the ready queue.

        For tasks with ``when``: evaluate the expression and skip/ready accordingly.
        For tasks without ``when``: require all deps succeeded (original behavior).
        Returns True if added to ready, False if skipped or not eligible.
        """
        task = task_map[tid]
        if tid in tasks_with_when:
            # Build when variables from completed deps
            when_vars: dict[str, str] = {}
            for dep_id in deps_map[tid]:
                if dep_id in results:
                    dep_r = results[dep_id]
                    when_vars[f"{dep_id}.status"] = dep_r.status
                    when_vars[f"{dep_id}.exit_code"] = str(dep_r.exit_code if dep_r.exit_code is not None else "")
            try:
                condition_met, rendered = evaluate_when_condition(task.when, when_vars)  # type: ignore[arg-type]
            except ValueError:
                # Invalid expression — skip task
                pending.discard(tid)
                skipped = _new_skipped_result(
                    tid, run_path,
                    message=f"Invalid when expression: {task.when}",
                )
                results[tid] = skipped
                completed.add(tid)
                _vlog(
                    f"{_yellow('skip')}    {_bold(tid)}: invalid when expression",
                    progress=_progress(),
                )
                _emit("task_skip", task_id=tid, reason=f"invalid when expression: {task.when}")
                return False

            if condition_met:
                return True
            else:
                pending.discard(tid)
                skipped = _new_skipped_result(
                    tid, run_path,
                    message=f"Condition not met: {rendered}",
                )
                results[tid] = skipped
                completed.add(tid)
                _vlog(
                    f"{_yellow('skip')}    {_bold(tid)}: condition not met ({rendered})",
                    progress=_progress(),
                )
                _emit("task_skip", task_id=tid, reason=f"condition not met: {rendered}")
                return False
        else:
            # Original behavior: require all deps succeeded
            dep_statuses = [results[dep].status for dep in deps_map[tid] if dep in results]
            return all(s in _SUCCESS_LIKE for s in dep_statuses) if dep_statuses else True

    def _handle_dependents(finished_task_id: str) -> None:
        """Update the ready queue after a task completes (or is cache-hit / skipped)."""
        for dependent_id in dependents_map.get(finished_task_id, []):
            if dependent_id not in pending:
                continue  # pragma: no cover
            deps = deps_map[dependent_id]
            if not deps.issubset(completed):
                continue
            if dependent_id in tasks_with_when:
                if _evaluate_ready(dependent_id):
                    ready.append(dependent_id)
            else:
                dep_statuses = [results[dep].status for dep in deps]
                if all(status in _SUCCESS_LIKE for status in dep_statuses):
                    ready.append(dependent_id)
                else:
                    pending.remove(dependent_id)
                    skipped = _new_skipped_result(
                        dependent_id,
                        run_path,
                        message=f"Skipped because dependency failed: {deps}",
                    )
                    results[dependent_id] = skipped
                    completed.add(dependent_id)
                    _vlog(
                        f"{_yellow('skip')}    {_bold(dependent_id)}: dependency failure",
                        progress=_progress(),
                    )
                    _emit("task_skip", task_id=dependent_id, reason="dependency failure")

    fail_fast_trigger = False
    fail_fast_reason = ""
    budget_exceeded = False
    budget_warned = False
    budget_reason = ""
    circuit_breaker_failure_count = 0

    def _apply_completion_state(task_id: str, result: TaskResult) -> None:
        nonlocal budget_exceeded
        nonlocal budget_warned
        nonlocal budget_reason
        nonlocal fail_fast_trigger
        nonlocal fail_fast_reason
        nonlocal circuit_breaker_failure_count

        if plan.max_cost_usd is not None and not budget_exceeded:
            running_cost = sum(
                r.cost_usd
                for r in results.values()
                if r.cost_usd is not None
            )
            if (
                not budget_warned
                and running_cost >= plan.max_cost_usd * budget_warning_pct
            ):
                pct = (
                    running_cost / plan.max_cost_usd
                    if plan.max_cost_usd > 0
                    else 1.0
                )
                _emit(
                    "budget_warning",
                    spent=round(running_cost, 2),
                    limit=plan.max_cost_usd,
                    pct=pct,
                )
                _vlog(
                    _yellow(
                        f"budget warning: ${running_cost:.2f} is "
                        f"{pct:.0%} of ${plan.max_cost_usd:.2f} limit"
                    ),
                    progress=_progress(),
                    min_level=0,
                )
                budget_warned = True
            if running_cost > plan.max_cost_usd:
                budget_exceeded = True
                budget_reason = (
                    f"Budget exceeded (${running_cost:.2f} / "
                    f"${plan.max_cost_usd:.2f} limit)"
                )
                _vlog(
                    _yellow(
                        f"budget exceeded: ${running_cost:.2f} > "
                        f"${plan.max_cost_usd:.2f} limit"
                    ),
                    progress=_progress(),
                    min_level=0,
                )
                _emit(
                    "budget_exceeded",
                    spent=round(running_cost, 2),
                    limit=plan.max_cost_usd,
                )

        if result.status == "failed" and plan.fail_fast:
            fail_fast_trigger = True
            fail_fast_reason = f"fail_fast triggered by task '{task_id}'"

        if result.status in ("failed", "soft_failed") and plan.circuit_breaker is not None:
            circuit_breaker_failure_count += 1
            cb = plan.circuit_breaker
            if circuit_breaker_failure_count >= cb.max_total_failures:
                _emit(
                    "circuit_breaker_tripped",
                    failure_count=circuit_breaker_failure_count,
                    max_total_failures=cb.max_total_failures,
                    action=cb.action,
                    task_id=task_id,
                )
                if cb.action == "fail":
                    fail_fast_trigger = True
                    fail_fast_reason = (
                        f"circuit breaker tripped: {circuit_breaker_failure_count} "
                        f"failure(s) reached max_total_failures={cb.max_total_failures}"
                    )
                else:  # "pause"
                    cb_message = (
                        f"Circuit breaker: {circuit_breaker_failure_count} failures reached. Continue?"
                    )
                    approved = False
                    if approval_handler is not None:
                        try:
                            approved = approval_handler("circuit_breaker", cb_message)
                        except Exception:
                            approved = False
                    elif not auto_approve:
                        approved = _request_approval(
                            "circuit_breaker", cb_message, sys.stdin.isatty()
                        )
                    else:
                        approved = True
                    if approved:
                        circuit_breaker_failure_count = 0
                    else:
                        fail_fast_trigger = True
                        fail_fast_reason = (
                            f"circuit breaker tripped (pause denied): "
                            f"{circuit_breaker_failure_count} failures"
                        )

    ready: deque[str] = deque()
    for task_id in (t.id for t in selected_tasks):
        if task_id not in pending or not deps_map[task_id].issubset(completed):
            continue
        if not deps_map[task_id]:
            # No deps — always ready (when evaluated after deps complete, but no deps = always)
            if task_id not in tasks_with_when:
                ready.append(task_id)
            else:
                # Tasks with when but no deps: evaluate immediately
                if _evaluate_ready(task_id):
                    ready.append(task_id)
        else:
            if _evaluate_ready(task_id):
                ready.append(task_id)

    max_parallel = max_parallel_override or plan.max_parallel
    budget_warning_pct = (
        plan.budget_warning_pct
        if plan.budget_warning_pct is not None
        else plan.defaults.budget_warning_pct
        if plan.defaults.budget_warning_pct is not None
        else 0.8
    )

    # Header
    _vmeta(f"run_id={run_id} plan={plan.name} tasks={total}")
    _vmeta(f"run_path={run_path}")
    _vmeta(
        f"max_parallel={max_parallel} dry_run={dry_run} "
        f"execution_profile={execution_profile}"
    )
    if plan.max_cost_usd is not None:
        _vmeta(f"budget=${plan.max_cost_usd:.2f}")
    _emit("run_start", run_id=run_id, plan=plan.name, goal=plan.goal or "", tasks=total, max_parallel=max_parallel, run_path=str(run_path))

    # Untrusted context — emit taint events for tracking
    if _tainted_tasks:
        for _tid in sorted(_tainted_tasks):
            _t = task_map[_tid]
            _src = "explicit" if _t.context_trust == "untrusted" else "propagated"
            _emit("taint_detected", task_id=_tid, source=_src)

    try:
        with ThreadPoolExecutor(max_workers=max_parallel) as pool:
            running: dict[Future[TaskResult], str] = {}
            # Per-task context metrics — keyed by task_id to avoid cross-task contamination
            _task_context_metrics: dict[str, tuple[int, int]] = {}  # task_id → (raw, final)

            while pending or running:
                if cancel_event is not None and cancel_event.is_set():
                    for tid in list(pending):
                        result = _new_skipped_result(tid, run_path, "Cancelled")
                        results[tid] = result
                        completed.add(tid)
                        _emit("task_skip", task_id=tid, reason="cancelled")
                    pending.clear()
                    ready.clear()
                    break

                _mcp_dispatch_blocked = 0
                while ready and len(running) < max_parallel and not fail_fast_trigger and not budget_exceeded and not (cancel_event is not None and cancel_event.is_set()):
                    task_id = ready.popleft()
                    if task_id not in pending:
                        continue  # pragma: no cover
                    task = task_map[task_id]
                    if _task_uses_exclusive_mcp_worktree_slot(plan, task):
                        blocked_by_running_mcp = any(
                            _task_uses_exclusive_mcp_worktree_slot(plan, task_map[running_id])
                            for running_id in running.values()
                        )
                        if blocked_by_running_mcp:
                            ready.append(task_id)
                            _mcp_dispatch_blocked += 1
                            if _mcp_dispatch_blocked >= len(ready):
                                break
                            continue
                    _mcp_dispatch_blocked = 0
                    pending.remove(task_id)

                    # --- Cache lookup ---
                    if cache_dir is not None and task.cache and not dry_run:
                        upstream_h = {
                            dep: task_hashes.get(dep, "")
                            for dep in deps_map[task_id]
                        }
                        task_hash = _compute_task_hash_safe(task, plan, upstream_h)
                        if task_hash is not None:
                            task_hashes[task_id] = task_hash
                            cached_data = cache_lookup(cache_dir, task_hash)
                            if cached_data is not None and isinstance(cached_data.get("status"), str):
                                result = _new_cached_result(
                                    task_id, run_path, cached_data, task_hash, cache_dir
                                )
                                results[task_id] = result
                                completed.add(task_id)
                                _vlog(
                                    f"{_green('CACHED')}  {_bold(task_id)}"
                                    f" {_dim(f'[{task_hash[:12]}]')}",
                                    progress=_progress(),
                                )
                                _emit(
                                    "task_complete",
                                    task_id=task_id,
                                    status=result.status,
                                    duration_sec=0.0,
                                    cost_usd=result.cost_usd,
                                    tokens=result.token_usage.total_tokens
                                    if result.token_usage is not None
                                    else None,
                                    from_cache=True,
                                )
                                _apply_completion_state(task_id, result)
                                _handle_dependents(task_id)
                                continue  # skip pool.submit

                    msuf = _model_suffix(plan, task)
                    desc = task.description or "(no description)"
                    _vlog(
                        f"{_cyan('starting')} {_bold(task_id)}{msuf}: {desc}",
                        progress=_progress(),
                    )
                    _emit(
                        "task_start",
                        task_id=task_id,
                        wave=task_waves.get(task_id, 0),
                        engine=task.engine,
                        model=_resolve_model(plan, task) or None,
                    )
                    # Approval gate
                    if task.requires_approval and not dry_run:
                        _emit("approval_required", task_id=task_id, message=task.approval_message)
                        if auto_approve:
                            _vlog(
                                f"[maestro] auto-approving '{task_id}'",
                                progress=_progress(),
                            )
                            _emit("approval_response", task_id=task_id, approved=True)
                        elif approval_handler is not None:
                            try:
                                approved = approval_handler(task_id, task.approval_message)
                            except Exception:
                                approved = False
                            _emit("approval_response", task_id=task_id, approved=approved)
                            if not approved:
                                result = _new_skipped_result(
                                    task_id, run_path, "Approval denied by handler"
                                )
                                results[task_id] = result
                                completed.add(task_id)
                                _vlog(
                                    f"{_yellow('denied')}  {_bold(task_id)}: approval denied",
                                    progress=_progress(),
                                )
                                _handle_dependents(task_id)
                                continue
                        elif not _request_approval(task_id, task.approval_message, sys.stdin.isatty()):
                            _emit("approval_response", task_id=task_id, approved=False)
                            result = _new_skipped_result(
                                task_id, run_path, "Approval denied or non-interactive"
                            )
                            results[task_id] = result
                            completed.add(task_id)
                            _vlog(
                                f"{_yellow('denied')}  {_bold(task_id)}: approval denied",
                                progress=_progress(),
                            )
                            _handle_dependents(task_id)
                            continue
                        else:
                            _emit("approval_response", task_id=task_id, approved=True)
                    # --- Policy enforcement ---
                    if plan.policies:
                        violations = evaluate_policies(plan.policies, task, plan)
                        blocked = [v for v in violations if v.action == "block"]
                        for v in violations:
                            _emit(
                                "policy_violation",
                                task_id=task_id,
                                policy_name=v.policy_name,
                                action=v.action,
                                message=v.message,
                            )
                            if v.action == "warn" and _v_level >= 1:
                                print(
                                    f"[maestro] policy warning '{v.policy_name}' on task '{task_id}': {v.message}"
                                )
                        if blocked:
                            msg = f"Blocked by policy '{blocked[0].policy_name}': {blocked[0].message}"
                            if _v_level >= 1:
                                print(f"[maestro] {msg}")
                            result = TaskResult(
                                task_id=task_id,
                                status="failed",
                                message=msg,
                                exit_code=1,
                                duration_sec=0.0,
                            )
                            results[task_id] = result
                            completed.add(task_id)
                            _vlog(
                                f"{_red('FAIL')}    {_bold(task_id)}: blocked by policy",
                                progress=_progress(),
                            )
                            _handle_dependents(task_id)
                            continue
                    # Also compute hash for cache miss (store after completion)
                    if cache_dir is not None and task.cache and not dry_run and task_id not in task_hashes:
                        upstream_h = {
                            dep: task_hashes.get(dep, "")
                            for dep in deps_map[task_id]
                        }
                        task_hash = _compute_task_hash_safe(task, plan, upstream_h)
                        if task_hash is not None:
                            task_hashes[task_id] = task_hash
                    # Build upstream context for tasks that declare context_from
                    upstream_for_task: dict[str, TaskResult] | None = None
                    selection_entries: dict[str, dict[str, object]] = {}
                    context_synthesis = ""
                    workspace_brief = ""
                    _context_raw_tokens = 0
                    _context_final_tokens = 0
                    effective_budget: int | None = None
                    selection_details_available = False
                    _task_prompt_text = _load_task_prompt_text(plan, task)
                    if task.context_from:
                        context_ids: set[str] = set()
                        for ctx_entry in task.context_from:
                            if ctx_entry == "*":
                                context_ids.update(deps_map[task_id])
                            else:
                                context_ids.add(ctx_entry)
                        upstream_for_task = {
                            tid: results[tid]
                            for tid in context_ids
                            if tid in results
                        }

                        hop_distances = _compute_hop_distances(
                            task_id,
                            task.context_from,
                            task_map,
                        )
                        if any(hop > 1 for hop in hop_distances.values()):
                            upstream_for_task = _apply_hop_decay(
                                upstream_for_task,
                                hop_distances,
                            )

                        for tid, task_result in upstream_for_task.items():
                            hop_distance = hop_distances.get(tid, 1)
                            selection_entries[tid] = {
                                "upstream_id": tid,
                                "score": 0.0,
                                "keywords_matched": [],
                                "hop_distance": hop_distance,
                                "hop_decay_factor": round(
                                    _DECAY_BASE ** max(0, hop_distance - 1),
                                    4,
                                ),
                                "tokens_raw": _estimate_tokens(task_result.stdout_tail),
                                "tokens_final": _estimate_tokens(task_result.stdout_tail),
                                "trimmed": False,
                                "trim_reason": "",
                            }

                        # Privacy-aware context filtering (v1.26.0)
                        if task.context_allowlist:
                            upstream_for_task = {
                                tid: _filter_context_fields(r, task.context_allowlist)
                                for tid, r in upstream_for_task.items()
                            }
                        # Apply upstream output_redact patterns: each upstream
                        # task's redact patterns are applied to its own output
                        # before it reaches any downstream consumer.
                        for tid in list(upstream_for_task):
                            _upstream_task = task_map.get(tid)
                            if _upstream_task and _upstream_task.output_redact:
                                _r = upstream_for_task[tid]
                                _redacted = _redact_output(
                                    _r.stdout_tail, _upstream_task.output_redact
                                )
                                if _redacted != _r.stdout_tail:
                                    upstream_for_task[tid] = dataclasses.replace(
                                        _r, stdout_tail=_redacted
                                    )

                        # Measure raw context tokens before any filtering
                        _context_raw_tokens = sum(
                            _estimate_tokens(r.stdout_tail)
                            for r in upstream_for_task.values()
                        )

                        intent_keywords = _extract_keywords(_task_prompt_text)

                        if task.context_mode == "selective" and upstream_for_task:
                            effective_budget = (
                                task.context_budget_tokens
                                or plan.defaults.context_budget_tokens
                                or 6000
                            )
                            _filtered_upstream, _filter_records, selection_meta = _apply_intent_filtering(
                                upstream_for_task,
                                intent_keywords,
                            )
                            selection_details_available = bool(selection_meta)
                            _sel_scores = {
                                uid: float(cast(float, meta.get("score", 0.0)))
                                for uid, meta in selection_meta.items()
                            }
                            _sel_texts = {
                                uid: r.stdout_tail
                                for uid, r in upstream_for_task.items()
                                if r.stdout_tail
                            }
                            context_synthesis = _build_selective_context(
                                _sel_texts,
                                effective_budget,
                                intent_keywords,
                                _sel_scores,
                            )
                            for uid, meta in selection_meta.items():
                                entry = selection_entries.setdefault(
                                    uid,
                                    {
                                        "upstream_id": uid,
                                        "score": 0.0,
                                        "keywords_matched": [],
                                        "hop_distance": hop_distances.get(uid, 1),
                                        "hop_decay_factor": round(
                                            _DECAY_BASE ** max(0, hop_distances.get(uid, 1) - 1),
                                            4,
                                        ),
                                        "tokens_raw": 0,
                                        "tokens_final": 0,
                                        "trimmed": False,
                                        "trim_reason": "",
                                    },
                                )
                                entry["score"] = float(cast(float, meta.get("score", 0.0)))
                                entry["keywords_matched"] = list(cast(list[str], meta.get("keywords_matched", [])))

                        elif task.context_mode == "layered" and upstream_for_task:
                            effective_budget = (
                                task.context_budget_tokens
                                or plan.defaults.context_budget_tokens
                                or 6000
                            )
                            _filtered_upstream, _filter_records, selection_meta = _apply_intent_filtering(
                                upstream_for_task,
                                intent_keywords,
                            )
                            selection_details_available = bool(selection_meta)
                            scores = {
                                upstream_id: float(cast(float, meta.get("score", 0.0)))
                                for upstream_id, meta in selection_meta.items()
                            }
                            upstream_texts = {
                                upstream_id: task_result.stdout_tail
                                for upstream_id, task_result in upstream_for_task.items()
                                if task_result.stdout_tail
                            }
                            context_synthesis = _build_layered_context(
                                upstream_texts,
                                effective_budget,
                                scores,
                            )
                            layered_sections = _parse_layered_context_sections(context_synthesis)

                            for upstream_id, meta in selection_meta.items():
                                entry = selection_entries.setdefault(
                                    upstream_id,
                                    {
                                        "upstream_id": upstream_id,
                                        "score": 0.0,
                                        "keywords_matched": [],
                                        "hop_distance": hop_distances.get(upstream_id, 1),
                                        "hop_decay_factor": round(
                                            _DECAY_BASE ** max(0, hop_distances.get(upstream_id, 1) - 1),
                                            4,
                                        ),
                                        "tokens_raw": 0,
                                        "tokens_final": 0,
                                        "trimmed": False,
                                        "trim_reason": "",
                                    },
                                )
                                entry["score"] = float(cast(float, meta.get("score", 0.0)))
                                entry["keywords_matched"] = list(cast(list[str], meta.get("keywords_matched", [])))

                            for upstream_id, task_result in upstream_for_task.items():
                                original_tokens = _estimate_tokens(task_result.stdout_tail)
                                section_tokens = _estimate_tokens(layered_sections.get(upstream_id, "")) if upstream_id in layered_sections else 0
                                if upstream_id in selection_entries:
                                    selection_entries[upstream_id]["tokens_final"] = section_tokens
                                if section_tokens < original_tokens:
                                    trim_entry = selection_entries.setdefault(
                                        upstream_id,
                                        {
                                            "upstream_id": upstream_id,
                                            "score": 0.0,
                                            "keywords_matched": [],
                                            "hop_distance": hop_distances.get(upstream_id, 1),
                                            "hop_decay_factor": round(
                                                _DECAY_BASE ** max(0, hop_distances.get(upstream_id, 1) - 1),
                                                4,
                                            ),
                                            "tokens_raw": original_tokens,
                                            "tokens_final": section_tokens,
                                            "trimmed": True,
                                            "trim_reason": "budget_trim",
                                        },
                                    )
                                    trim_entry["tokens_raw"] = original_tokens
                                    trim_entry["tokens_final"] = section_tokens
                                    trim_entry["trimmed"] = True
                                    trim_entry["trim_reason"] = "budget_trim"
                                    _emit(
                                        "context_budget_trim",
                                        task_id=task_id,
                                        upstream_id=upstream_id,
                                        original_tokens=original_tokens,
                                        trimmed_tokens=section_tokens,
                                        budget=effective_budget,
                                    )
                                    _vlog(
                                        f"{_dim('context trim')} {_bold(task_id)}"
                                        f" \u2190 {upstream_id}:"
                                        f" {original_tokens}\u2192{section_tokens}tok"
                                        f" (budget={effective_budget})",
                                        progress=_progress(),
                                        min_level=2,
                                    )

                        elif task.context_mode == "structural" and upstream_for_task:
                            effective_budget = (
                                task.context_budget_tokens
                                or plan.defaults.context_budget_tokens
                                or 6000
                            )
                            _struct_texts = {
                                uid: r.stdout_tail
                                for uid, r in upstream_for_task.items()
                                if r.stdout_tail
                            }
                            _struct_files = {
                                uid: (
                                    r.structured_context.files_changed
                                    if r.structured_context
                                    else []
                                )
                                for uid, r in upstream_for_task.items()
                            }
                            context_synthesis = _build_structural_context(
                                _struct_texts,
                                effective_budget,
                                _struct_files,
                                workspace_root=plan.workspace_root,
                            )

                        elif task.context_mode == "knowledge_graph" and upstream_for_task:
                            effective_budget = (
                                task.context_budget_tokens
                                or plan.defaults.context_budget_tokens
                                or 6000
                            )
                            _kg_texts = {
                                uid: r.stdout_tail
                                for uid, r in upstream_for_task.items()
                                if r.stdout_tail
                            }
                            context_synthesis = _build_knowledge_graph_context(
                                _kg_texts,
                                effective_budget,
                            )

                        elif task.context_mode == "codebase_map":
                            # Workspace-derived (not from upstream): read the
                            # pre-built Understand-Anything graph and score it
                            # against this task's prompt. Zero LLM cost; ""
                            # when no graph exists (task runs without it).
                            effective_budget = (
                                task.context_budget_tokens
                                or plan.defaults.context_budget_tokens
                                or 6000
                            )
                            context_synthesis = _build_codebase_map_context(
                                plan.workspace_root,
                                task.prompt or "",
                                effective_budget,
                            )

                        elif task.context_mode == "scip":
                            # Workspace-derived (not from upstream): read the
                            # pre-built SCIP index (JSON) and score its symbols
                            # against this task's prompt. Zero LLM cost; "" when
                            # no index exists (task runs without it).
                            effective_budget = (
                                task.context_budget_tokens
                                or plan.defaults.context_budget_tokens
                                or 6000
                            )
                            context_synthesis = _build_scip_context(
                                plan.workspace_root,
                                task.prompt or "",
                                effective_budget,
                            )

                        elif task.context_mode == "council" and task.council is not None:
                            from .council import run_council

                            # Build upstream context for the council
                            _council_upstream = ""
                            if upstream_for_task:
                                _u_parts = []
                                for uid, r in upstream_for_task.items():
                                    if r.stdout_tail:
                                        _u_parts.append(f"--- {uid} ---\n{r.stdout_tail[:4000]}")
                                _council_upstream = "\n\n".join(_u_parts)

                            # Load the task prompt
                            _council_prompt = _load_task_prompt_text(plan, task)

                            _emit(
                                "council_start",
                                task_id=task_id,
                                participants=len(task.council.participants),
                                rounds=task.council.rounds,
                                topology=task.council.topology,
                            )

                            _council_result = run_council(
                                task.council,
                                _council_prompt,
                                resolve_workdir(plan, task),
                                upstream_context=_council_upstream,
                                event_callback=lambda evt, **kw: _emit(evt, task_id=task_id, **kw),
                            )
                            context_synthesis = _council_result.synthesis

                            _emit(
                                "council_complete",
                                task_id=task_id,
                                rounds_completed=len(_council_result.rounds),
                                cost_usd=_council_result.total_cost_usd,
                            )

                        else:
                            # Context budget enforcement (F1)
                            effective_budget = (
                                task.context_budget_tokens
                                or plan.defaults.context_budget_tokens
                            )
                            if effective_budget is not None and upstream_for_task:
                                # RRF fusion: combine BM25 + hop-distance
                                rrf_scores: dict[str, float] | None = None
                                if intent_keywords and hop_distances:
                                    # Pre-compute BM25 scores for RRF
                                    _, _, _pre_meta = _apply_intent_filtering(
                                        upstream_for_task, intent_keywords,
                                    )
                                    bm25_scores = {
                                        tid: float(cast(float, m.get("score", 0)))
                                        for tid, m in _pre_meta.items()
                                    }
                                    rrf_scores = _rrf_score(bm25_scores, hop_distances)
                                upstream_for_task, trim_records, selection_meta = _apply_context_budget(
                                    upstream_for_task, effective_budget, intent_keywords,
                                    relevance_scores=rrf_scores,
                                )
                                selection_details_available = bool(selection_meta or trim_records)
                                for upstream_id, meta in selection_meta.items():
                                    entry = selection_entries.setdefault(
                                        upstream_id,
                                        {
                                            "upstream_id": upstream_id,
                                            "score": 0.0,
                                            "keywords_matched": [],
                                            "hop_distance": hop_distances.get(upstream_id, 1),
                                            "hop_decay_factor": round(
                                                _DECAY_BASE ** max(0, hop_distances.get(upstream_id, 1) - 1),
                                                4,
                                            ),
                                            "tokens_raw": 0,
                                            "tokens_final": 0,
                                            "trimmed": False,
                                            "trim_reason": "",
                                        },
                                    )
                                    entry["score"] = float(cast(float, meta.get("score", 0.0)))
                                    entry["keywords_matched"] = list(cast(list[str], meta.get("keywords_matched", [])))
                                for trim_tid, orig_tok, new_tok in trim_records:
                                    trim_entry = selection_entries.setdefault(
                                        trim_tid,
                                        {
                                            "upstream_id": trim_tid,
                                            "score": 0.0,
                                            "keywords_matched": [],
                                            "hop_distance": hop_distances.get(trim_tid, 1),
                                            "hop_decay_factor": round(
                                                _DECAY_BASE ** max(0, hop_distances.get(trim_tid, 1) - 1),
                                                4,
                                            ),
                                            "tokens_raw": orig_tok,
                                            "tokens_final": new_tok,
                                            "trimmed": True,
                                            "trim_reason": "budget_trim",
                                        },
                                    )
                                    trim_entry["tokens_raw"] = orig_tok
                                    trim_entry["tokens_final"] = new_tok
                                    trim_entry["trimmed"] = True
                                    trim_entry["trim_reason"] = "budget_trim"
                                    _emit(
                                        "context_budget_trim",
                                        task_id=task_id,
                                        upstream_id=trim_tid,
                                        original_tokens=orig_tok,
                                        trimmed_tokens=new_tok,
                                        budget=effective_budget,
                                    )
                                    _vlog(
                                        f"{_dim('context trim')} {_bold(task_id)}"
                                        f" \u2190 {trim_tid}:"
                                        f" {orig_tok}\u2192{new_tok}tok"
                                        f" (budget={effective_budget})",
                                        progress=_progress(),
                                        min_level=2,
                                    )
                                for upstream_id, task_result in upstream_for_task.items():
                                    if upstream_id in selection_entries:
                                        selection_entries[upstream_id]["tokens_final"] = _estimate_tokens(
                                            task_result.stdout_tail
                                        )

                        # Progressive / standard compaction (v1.26.0)
                        _effective_compaction = (
                            task.context_compaction
                            or plan.defaults.context_compaction
                        )
                        if _effective_compaction and _effective_compaction != "none" and upstream_for_task:
                            _compaction_budget = (
                                task.context_budget_tokens
                                or plan.defaults.context_budget_tokens
                                or 6000
                            )
                            if _effective_compaction == "progressive":
                                _upstream_texts = {
                                    uid: r.stdout_tail
                                    for uid, r in upstream_for_task.items()
                                    if r.stdout_tail
                                }
                                _intent_scores = {
                                    uid: float(cast(float, selection_entries.get(uid, {}).get("score", 0.0)))
                                    for uid in _upstream_texts
                                }
                                _compacted, _max_stage = _apply_progressive_compaction(
                                    _upstream_texts, _compaction_budget, _intent_scores,
                                    original_texts=dict(_upstream_texts),
                                    workdir=resolve_workdir(plan, task),
                                )
                                if _max_stage > 0:
                                    for uid, new_text in _compacted.items():
                                        if uid in upstream_for_task:
                                            upstream_for_task[uid] = dataclasses.replace(
                                                upstream_for_task[uid],
                                                stdout_tail=new_text,
                                            )
                                            if uid in selection_entries:
                                                selection_entries[uid]["tokens_final"] = _estimate_tokens(new_text)
                                                selection_entries[uid]["trimmed"] = True
                                                selection_entries[uid]["trim_reason"] = f"compaction_stage_{_max_stage}"
                                    _emit(
                                        "context_compaction",
                                        task_id=task_id,
                                        mode="progressive",
                                        max_stage=_max_stage,
                                        budget_tokens=_compaction_budget,
                                    )
                                    _vlog(
                                        f"{_dim('progressive compaction')} {_bold(task_id)}"
                                        f" stage={_max_stage} budget={_compaction_budget}tok",
                                        progress=_progress(),
                                        min_level=2,
                                    )
                            elif _effective_compaction == "standard":
                                for uid, r in upstream_for_task.items():
                                    if r.stdout_tail:
                                        compacted = _compact_context(r.stdout_tail)
                                        if len(compacted) < len(r.stdout_tail):
                                            upstream_for_task[uid] = dataclasses.replace(r, stdout_tail=compacted)
                                            if uid in selection_entries:
                                                selection_entries[uid]["tokens_final"] = _estimate_tokens(compacted)
                                                selection_entries[uid]["trimmed"] = True
                                                selection_entries[uid]["trim_reason"] = "compaction_standard"
                                _emit(
                                    "context_compaction",
                                    task_id=task_id,
                                    mode="standard",
                                    max_stage=1,
                                    budget_tokens=_compaction_budget,
                                )

                        # RLM context processing (summarized / map_reduce)
                        if upstream_for_task and task.context_mode == "summarized":
                            _vlog(
                                f"{_dim('summarizing context for')} {_bold(task_id)}",
                                progress=_progress(),
                            )
                            context_model = _resolve_context_model(task, plan)
                            summarized = _prepare_summaries(
                                upstream_for_task, resolve_workdir(plan, task),
                                model=context_model,
                            )
                            for upstream_id, duration_sec in summarized:
                                _emit(
                                    "context_summarize",
                                    task_id=task_id,
                                    upstream=upstream_id,
                                    duration_sec=round(duration_sec, 3),
                                )
                        elif upstream_for_task and task.context_mode == "map_reduce":
                            _vlog(
                                f"{_dim('map/reduce context for')} {_bold(task_id)}",
                                progress=_progress(),
                            )
                            wdir = resolve_workdir(plan, task)
                            context_model = _resolve_context_model(task, plan)
                            summarized = _prepare_summaries(
                                upstream_for_task, wdir, model=context_model
                            )
                            for upstream_id, duration_sec in summarized:
                                _emit(
                                    "context_summarize",
                                    task_id=task_id,
                                    upstream=upstream_id,
                                    duration_sec=round(duration_sec, 3),
                                )
                            context_synthesis = _run_map_reduce(
                                upstream_for_task, wdir, model=context_model
                            )

                    # Recursive context pipeline (workspace index → extract → brief)
                    if task.context_mode == "recursive":
                        _vlog(
                            f"{_dim('recursive context for')} {_bold(task_id)}",
                            progress=_progress(),
                        )
                        rc = _build_recursive_context(
                            plan, task, resolve_workdir(plan, task), dry_run
                        )
                        workspace_brief = rc.workspace_brief
                        _emit(
                            "context_recursive",
                            task_id=task_id,
                            stages=rc.stages,
                            reused_index=rc.reused_index,
                            duration_sec=round(rc.duration_sec or 0.0, 3),
                        )

                    # Measure final context tokens after filtering/summarization
                    if upstream_for_task and _context_raw_tokens > 0:
                        if task.context_mode in ("layered", "selective"):
                            _context_final_tokens = (
                                _estimate_tokens(context_synthesis)
                                if context_synthesis
                                else 0
                            )
                        else:
                            _context_final_tokens = sum(
                                _estimate_tokens(r.stdout_tail)
                                for r in upstream_for_task.values()
                            )
                            if context_synthesis:
                                _context_final_tokens += _estimate_tokens(context_synthesis)
                        _compression_ratio = (
                            round(1.0 - _context_final_tokens / _context_raw_tokens, 4)
                            if _context_raw_tokens > 0
                            else 0.0
                        )
                        selection_payload = (
                            list(selection_entries.values())
                            if selection_details_available
                            else []
                        )
                        if _compression_ratio > 0.0 or selection_payload:
                            event_data: dict[str, object] = {
                                "task_id": task_id,
                                "context_raw_tokens": _context_raw_tokens,
                                "context_final_tokens": _context_final_tokens,
                                "compression_ratio": _compression_ratio,
                            }
                            if effective_budget is not None:
                                event_data["budget_tokens"] = effective_budget
                            if selection_payload:
                                event_data["entries"] = sorted(
                                    selection_payload,
                                    key=lambda se: (
                                        -float(cast(float, se.get("score", 0.0))),
                                        int(cast(int, se.get("hop_distance", 0))),
                                        str(se.get("upstream_id", "")),
                                    ),
                                )
                            _emit("context_compression", **event_data)
                        _task_context_metrics[task_id] = (_context_raw_tokens, _context_final_tokens)

                    task_extra_template_vars: dict[str, str] = {}
                    task_extra_template_vars.update(
                        build_contract_template_vars(task, results)
                    )
                    task_extra_template_vars.update(
                        build_consistency_template_vars(
                            task,
                            results,
                            selected_group_members,
                        )
                    )
                    if extra_template_vars:
                        task_extra_template_vars.update(extra_template_vars)

                    if _knowledge_index:
                        task_extra_template_vars["knowledge_index"] = _knowledge_index

                    # Inject prompt-relevant cross-run knowledge (T1.3 + v2 index/detail retrieval)
                    if _task_knowledge:
                        from .knowledge import (
                            format_knowledge,
                            record_knowledge_retrievals,
                            select_relevant_knowledge,
                        )
                        _tk = select_relevant_knowledge(
                            _task_knowledge,
                            _task_prompt_text,
                            task_id=task_id,
                        )
                    else:
                        _tk = None
                    if _tk:
                        _alerts = record_knowledge_retrievals(
                            plan.name,
                            plan.source_dir,
                            _task_prompt_text,
                            _tk,
                        )
                        if _alerts:
                            alerted_signatures = {
                                (alert.task_id, alert.kind, alert.insight)
                                for alert in _alerts
                            }
                            _tk = [
                                rec
                                for rec in _tk
                                if (rec.task_id, rec.kind, rec.insight) not in alerted_signatures
                            ]
                            for alert in _alerts:
                                _existing_records = _task_knowledge.get(alert.task_id, [])
                                _task_knowledge[alert.task_id] = [
                                    rec
                                    for rec in _existing_records
                                    if (rec.task_id, rec.kind, rec.insight)
                                    != (alert.task_id, alert.kind, alert.insight)
                                ]
                                if not _task_knowledge[alert.task_id]:
                                    del _task_knowledge[alert.task_id]
                                _emit(
                                    "knowledge_poison_alert",
                                    task_id=task_id,
                                    source_task_id=alert.task_id,
                                    knowledge_kind=alert.kind,
                                    query_cluster=alert.query_cluster,
                                    retrieval_count=alert.retrieval_count,
                                    z_score=round(alert.z_score, 3),
                                    action=alert.action,
                                    signal=alert.signal,
                                )
                        if not _tk:
                            _formatted_knowledge = ""
                        else:
                            _formatted_knowledge = format_knowledge(_tk)
                        if _formatted_knowledge:
                            task_extra_template_vars["task_knowledge"] = _formatted_knowledge

                    # Build DAG structural metadata for difficulty-aware routing.
                    # Attached to the task object so runners.py can forward it to
                    # resolve_auto_model() without requiring extra execute_task params.
                    _dag_metadata: dict[str, float | int] = {
                        "fan_out": _compute_fan_out(task, plan),
                        "depth": _compute_task_depth(task, plan),
                        "upstream_failure_rate": (
                            sum(
                                1
                                for d in task.depends_on
                                if results.get(d) is not None
                                and results[d].status == "failed"
                            )
                            / max(len(task.depends_on), 1)
                        ) if task.depends_on else 0.0,
                    }
                    # Inject historical performance for predictive routing (T2.3)
                    _hist = _task_histories.get(task_id)
                    if _hist is not None:
                        _dag_metadata["task_history"] = _hist  # type: ignore[assignment]
                    # Inject cross-task data for adaptive temporal routing (v1.28.0)
                    if _task_histories:
                        _dag_metadata["all_histories"] = _task_histories  # type: ignore[assignment]
                        _dag_metadata["task_map"] = task_map  # type: ignore[assignment]
                    # Inject detected local hardware for hardware-aware routing (v2.5.3)
                    if _hardware_info is not None:
                        _dag_metadata["hardware"] = _hardware_info  # type: ignore[assignment]
                    task._dag_metadata = _dag_metadata  # type: ignore[attr-defined]

                    # --- Workspace-aware timeout adjustment (T0.1) ---
                    if not dry_run:
                        _wsat_adjusted = _estimate_workspace_timeout(plan, task)
                        if _wsat_adjusted is not None:
                            _wsat_original = task.timeout_sec
                            task = dataclasses.replace(task, timeout_sec=_wsat_adjusted)
                            task._dag_metadata = _dag_metadata  # type: ignore[attr-defined]
                            _emit(
                                "timeout_adjusted",
                                task_id=task_id,
                                original_timeout_sec=_wsat_original,
                                adjusted_timeout_sec=_wsat_adjusted,
                            )
                            _vlog(
                                f"{_dim('timeout')}↑ {_bold(task_id)}: "
                                f"{_wsat_original or 'default'}s → {_wsat_adjusted}s "
                                f"(large files detected in prompt)",
                                progress=_progress(),
                                min_level=1,
                            )

                    future = pool.submit(
                        execute_task,
                        plan,
                        task,
                        run_path,
                        dry_run,
                        execution_profile,
                        upstream_for_task,
                        context_synthesis,
                        workspace_brief,
                        event_callback=_event_callback,
                        extra_template_vars=task_extra_template_vars,
                        budget_getter=_budget_getter,
                    )
                    running[future] = task_id

                if not running:
                    break

                done, _ = wait(set(running), return_when=FIRST_COMPLETED)
                for future in done:
                    task_id = running.pop(future)
                    result = future.result()
                    # Attach compression metrics to result (per-task lookup)
                    _ctx_metrics = _task_context_metrics.pop(task_id, None)
                    if _ctx_metrics is not None:
                        _raw, _final = _ctx_metrics
                        result.context_raw_tokens = _raw
                        result.context_final_tokens = _final
                        result.context_compression_ratio = round(
                            1.0 - _final / _raw, 4
                        ) if _raw > 0 else 0.0

                    # Attach task hash for status/explain commands
                    if task_id in task_hashes:
                        result.task_hash = task_hashes[task_id]

                    # Untrusted context — mark tainted results for downstream
                    if task_id in _tainted_tasks:
                        result.tainted = True

                    results[task_id] = result
                    completed.add(task_id)

                    fin_task = task_map[task_id]
                    msuf = _model_suffix(plan, fin_task)
                    dur = _dim(f"({_fmt_duration(result.duration_sec)})")

                    retry_suf = f" (retried {result.retry_count}x)" if result.retry_count > 0 else ""

                    if result.status in _SUCCESS_LIKE:
                        label = _green("OK")
                        detail = f"{label}      {_bold(task_id)}{msuf} {dur}{retry_suf}"
                        _vlog(detail, progress=_progress())
                    else:
                        label = _red("FAIL")
                        exit_info = f"exit={result.exit_code}" if result.exit_code else ""
                        detail = f"{label}    {_bold(task_id)}{msuf} {exit_info} {dur}{retry_suf}"
                        _vlog(detail, progress=_progress(), min_level=0)

                    _failure_category = (
                        result.failure_history[-1].category
                        if result.failure_history
                        else None
                    )
                    _emit(
                        "task_complete",
                        task_id=task_id,
                        status=result.status,
                        duration_sec=round(result.duration_sec, 3),
                        cost_usd=result.cost_usd,
                        tokens=result.token_usage.total_tokens if result.token_usage is not None else None,
                        failure_category=_failure_category,
                        checkpoint_count=result.checkpoint_count if result.checkpoint_count else None,
                        judge_verdict=result.judge_result.verdict if result.judge_result else None,
                        log_hash=compute_artefact_hash(run_path / f"{task_id}.log"),
                        result_hash=compute_artefact_hash(run_path / f"{task_id}.result.json"),
                    )

                    # v0.6.0 supplementary events
                    if result.judge_result is not None:
                        _emit(
                            "judge_result",
                            task_id=task_id,
                            verdict=result.judge_result.verdict,
                            score=round(result.judge_result.overall_score, 3),
                            on_fail=fin_task.judge.on_fail if fin_task.judge else "fail",
                        )
                    if result.checkpoint_count > 0:
                        _emit(
                            "task_checkpoint",
                            task_id=task_id,
                            count=result.checkpoint_count,
                        )

                    # Trajectory guardrail evaluation (v1.26.0)
                    _tg = fin_task.trajectory_guard
                    if _tg is not None:
                        _violations: list[str] = []
                        # Check tool call count
                        if _tg.max_tool_calls is not None and result.tool_call_count > _tg.max_tool_calls:
                            _violations.append(
                                f"tool call count ({result.tool_call_count}) "
                                f"exceeds limit ({_tg.max_tool_calls})"
                            )
                        # Check retry without progress (same failure repeating)
                        if (
                            _tg.max_retries_without_progress is not None
                            and result.failure_history
                        ):
                            _consecutive = 0
                            _last_cat = None
                            for fh in result.failure_history:
                                cat = fh.category
                                if cat == _last_cat:
                                    _consecutive += 1
                                else:
                                    _consecutive = 1
                                    _last_cat = cat
                            if _consecutive >= _tg.max_retries_without_progress:
                                _violations.append(
                                    f"repeated failure category '{_last_cat}' "
                                    f"{_consecutive} times without progress"
                                )
                        # Check scope violation via regex on output
                        if _tg.scope_pattern and result.stdout_tail:
                            _scope_re = re.compile(_tg.scope_pattern)
                            _offending = _scope_re.findall(result.stdout_tail)
                            if _offending:
                                _violations.append(
                                    f"scope pattern matched in output: "
                                    f"{_offending[:3]}"
                                )
                        if _violations:
                            _emit(
                                "trajectory_violation",
                                task_id=task_id,
                                violations=_violations,
                                action=_tg.on_violation,
                            )
                            _msg = (
                                f"trajectory guardrail: {'; '.join(_violations)}"
                            )
                            if _tg.on_violation == "abort":
                                result = dataclasses.replace(
                                    result,
                                    status="failed",
                                    message=f"[trajectory guard] {_msg}",
                                )
                                results[task_id] = result
                                print(f"[maestro] {_msg} — task aborted")
                            elif _tg.on_violation == "warn":
                                print(f"[maestro] warning: {_msg}")

                    # Show last lines of log on failure; also on success in verbose mode
                    if result.status == "failed" or _v_level >= 2:
                        _show_fail_tail(result.log_path)

                    _apply_completion_state(task_id, result)

                    # Store cacheable result (success or short-lived negative entry)
                    if cache_dir is not None and task_map[task_id].cache:
                        stored_hash = task_hashes.get(task_id)
                        if stored_hash is not None:
                            cache_store(cache_dir, stored_hash, result, task=task_map[task_id])

                    _handle_dependents(task_id)

            if fail_fast_trigger and pending:
                for task_id in list(pending):
                    pending.remove(task_id)
                    skipped = _new_skipped_result(task_id, run_path, fail_fast_reason)
                    results[task_id] = skipped
                    completed.add(task_id)
                    _vlog(
                        f"{_yellow('skip')}    {_bold(task_id)}: {fail_fast_reason}",
                        progress=_progress(),
                    )
                    _emit("task_skip", task_id=task_id, reason=fail_fast_reason)

            if budget_exceeded and pending:
                for task_id in list(pending):
                    pending.remove(task_id)
                    skipped = _new_skipped_result(task_id, run_path, budget_reason)
                    results[task_id] = skipped
                    completed.add(task_id)
                    _vlog(
                        f"{_yellow('skip')}    {_bold(task_id)}: {budget_reason}",
                        progress=_progress(),
                    )
                    _emit("task_skip", task_id=task_id, reason=budget_reason)

    except KeyboardInterrupt:
        _vlog(_red("interrupted by user (Ctrl+C) -- killing active processes..."), min_level=0)
        kill_all_active()
        for task_id in list(pending):
            pending.remove(task_id)
            skipped = _new_skipped_result(
                task_id, run_path, "Interrupted by user"
            )
            results[task_id] = skipped
            completed.add(task_id)
            _vlog(
                f"{_yellow('skip')}    {_bold(task_id)}: interrupted",
                progress=_progress(),
            )
            _emit("task_skip", task_id=task_id, reason="interrupted by user")

    finished_at = now_utc()
    total_duration = (finished_at - started_at).total_seconds()
    success = all(
        result.status in _SUCCESS_LIKE
        for result in results.values()
    )

    # Parallelism metrics
    sequential_duration = sum(
        r.duration_sec for r in results.values() if r.status != "skipped"
    )
    parallelism_savings = (
        ((sequential_duration - total_duration) / sequential_duration * 100.0)
        if sequential_duration > total_duration > 0
        else 0.0
    )

    # Aggregate cost
    total_cost: float | None = None
    for r in results.values():
        if r.cost_usd is not None:
            total_cost = (total_cost or 0.0) + r.cost_usd

    # Aggregate tokens
    total_tokens: int | None = None
    for r in results.values():
        if r.token_usage is not None:
            total_tokens = (total_tokens or 0) + r.token_usage.total_tokens

    run_result = PlanRunResult(
        plan_name=plan.name,
        run_id=run_id,
        run_path=run_path,
        started_at=started_at,
        finished_at=finished_at,
        success=success,
        execution_profile=execution_profile,
        task_results=results,
        sequential_duration_sec=sequential_duration,
        parallelism_savings_pct=parallelism_savings,
        total_cost_usd=total_cost,
        total_tokens=total_tokens,
        budget_exceeded=budget_exceeded,
        task_graph=_build_task_graph(selected_tasks),
    )

    record_score_history = (
        not dry_run
        and only is None
        and skip is None
        and tags is None
        and skip_tags is None
        and len(selected_tasks) == len(plan.tasks)
    )
    if record_score_history:
        try:
            from .knowledge import build_score_record, store_score_history

            plan_hash = compute_plan_hash(plan)
            score_record = build_score_record(plan, run_result, plan_hash=plan_hash)
            run_result.plan_hash = score_record.plan_hash
            run_result.quality_score = score_record.quality_score
            if store_score_history(plan.name, plan.source_dir, score_record):
                _emit(
                    "score_recorded",
                    plan_hash=score_record.plan_hash,
                    quality_score=score_record.quality_score,
                    success=score_record.success,
                )
        except Exception:
            pass  # never fail the run for score history persistence

    manifest_path = _write_manifest(run_result, run_path)
    summary_plan = dataclasses.replace(plan, tasks=selected_tasks)
    summary_path = _write_summary(run_result, summary_plan, run_path)

    ok_count = sum(1 for r in results.values() if r.status in {"success", "dry_run"})
    soft_count = sum(1 for r in results.values() if r.status == "soft_failed")
    fail_count = sum(1 for r in results.values() if r.status == "failed")
    skip_count = sum(1 for r in results.values() if r.status == "skipped")

    # Summary line — always include ok/failed/skipped (matches _write_summary)
    summary_parts = [_green(f"{ok_count} ok") if ok_count else f"{ok_count} ok"]
    if soft_count:
        summary_parts.append(_yellow(f"{soft_count} soft_failed"))
    summary_parts.append(_red(f"{fail_count} failed") if fail_count else f"{fail_count} failed")
    summary_parts.append(_yellow(f"{skip_count} skipped") if skip_count else f"{skip_count} skipped")

    cost_str = f" ${total_cost:.2f}" if total_cost is not None else ""
    tokens_log = f" {total_tokens:,}tok" if total_tokens is not None else ""

    if sequential_duration > total_duration and total_duration > 0:
        timing_str = (
            f"({_fmt_duration(total_duration)} wall / "
            f"{_fmt_duration(sequential_duration)} seq "
            f"— {parallelism_savings:.0f}% saved)"
        )
    else:
        timing_str = f"({_fmt_duration(total_duration)})"

    budget_str = ""
    if run_result.budget_exceeded:
        budget_str = f" {_yellow('[BUDGET EXCEEDED]')}"

    status_word = _green("SUCCESS") if success else _red("FAILED")
    _vlog(
        f"{status_word} {' / '.join(summary_parts)}"
        f"{_dim(cost_str)}{_dim(tokens_log)} {_dim(timing_str)}{budget_str}",
        progress=_progress(),
        min_level=0,
    )
    _vmeta(f"manifest={manifest_path}")
    _vmeta(f"summary={summary_path}")

    effective_webhook_url = webhook_url if webhook_url is not None else plan.webhook_url
    if effective_webhook_url:
        summary_url: str | None = None
        try:
            summary_url = summary_path.resolve().as_uri()
        except ValueError:
            summary_url = None

        payload: dict[str, object] = {
            "plan_name": run_result.plan_name,
            "run_id": run_result.run_id,
            "success": run_result.success,
            "ok_count": ok_count,
            "failed_count": fail_count,
            "skipped_count": skip_count,
            "total_cost_usd": run_result.total_cost_usd,
            "total_tokens": run_result.total_tokens,
            "duration_sec": total_duration,
            "run_path": str(run_result.run_path),
            "summary_url": summary_url,
        }
        try:
            status_code = _post_completion_webhook(effective_webhook_url, payload)
            _emit(
                "webhook",
                status="delivered",
                url=effective_webhook_url,
                http_status=status_code,
            )
            _vmeta(
                f"webhook delivered status={status_code} url={effective_webhook_url}"
            )
        except (OSError, ValueError) as exc:
            _emit(
                "webhook",
                status="failed",
                url=effective_webhook_url,
                error=str(exc),
            )
            _vlog(
                _yellow(f"webhook delivery failed: {exc}"),
                min_level=0,
            )

    _emit(
        "run_complete",
        success=success,
        ok=ok_count,
        soft_failed=soft_count,
        failed=fail_count,
        skipped=skip_count,
        cost_usd=total_cost,
        tokens=total_tokens,
        duration_sec=round(total_duration, 3),
    )

    # Record cost in cross-run budget ledger
    if plan.budget_period and run_result.total_cost_usd and not dry_run:
        from .budget import record_cost, _DEFAULT_LEDGER_PATH
        ledger_path = (plan.source_dir / _DEFAULT_LEDGER_PATH)
        record_cost(ledger_path, plan.name, run_id, run_result.total_cost_usd)

    # Extract and store cross-run knowledge (T1.3)
    if not dry_run:
        _new_knowledge = []
        try:
            from .knowledge import extract_knowledge, store_knowledge_detailed
            _new_knowledge = extract_knowledge(run_result)
            if _new_knowledge:
                for outcome in store_knowledge_detailed(
                    plan.name,
                    plan.source_dir,
                    _new_knowledge,
                    source_type="task",
                    source_id=run_id,
                ):
                    _emit(
                        "memory_write",
                        task_id=outcome.task_id,
                        knowledge_kind=outcome.kind,
                        operation=outcome.operation,
                        outcome=outcome.outcome,
                        trust_label=outcome.trust_label,
                        instructionality_score=round(outcome.instructionality_score, 3),
                        source_type=outcome.source_type,
                        source_id=outcome.source_id,
                    )
        except Exception:
            for rec in _new_knowledge:
                _emit(
                    "memory_write",
                    task_id=rec.task_id,
                    knowledge_kind=rec.kind,
                    operation="store_failed",
                    outcome="rejected",
                    trust_label="unknown",
                    instructionality_score=0.0,
                    source_type="task",
                    source_id=f"{run_id}:{rec.task_id}",
                )
            pass  # never fail the run for knowledge extraction

    return run_result
