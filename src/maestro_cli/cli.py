from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable

from . import __version__
from .audit import (
    audit_plan,
    fix_plan as _audit_fix_plan,
    format_audit,
    format_audit_coverage,
    format_audit_coverage_json,
    format_audit_json,
)
from .eventsource import replay_events, verify_artefact_hashes, verify_chain
from .ci import generate_ci_yaml, write_ci_yaml, SUPPORTED_CI_PROVIDERS
from .cleanup import cleanup_runs
from .cost_backfill import backfill_run_costs, discover_run_roots
from .loader import compute_plan_density_score, load_plan
from .models import PlanRunResult, PlanSpec, Verbosity, WatchState
from .multi import run_multi_plan
from .mcts import DEFAULT_EXPLORATION_CONSTANT
from .replan import replan
from .report import generate_report
from .scheduler import run_plan
from .utils import resolve_path


# ── ASCII art banner with ANSI color gradient ────────────────────
# Each letter is a fixed-width column; "#" → U+2588 (full block).
# Programmatic join guarantees perfect alignment across all rows.
_LETTER_DATA: dict[str, tuple[str, ...]] = {
    "M": ("##   ##", "### ###", "## # ##", "##   ##", "##   ##"),
    "A": (" #### ", "##  ##", "######", "##  ##", "##  ##"),
    "E": ("######", "##    ", "##### ", "##    ", "######"),
    "S": (" ####", "##   ", " ### ", "   ##", "#### "),
    "T": ("######", "  ##  ", "  ##  ", "  ##  ", "  ##  "),
    "R": ("##### ", "##  ##", "##### ", "## ## ", "##  ##"),
    "O": (" #### ", "##  ##", "##  ##", "##  ##", " #### "),
}
_BANNER_LINES = tuple(
    "  " + " ".join(_LETTER_DATA[ch][row] for ch in "MAESTRO").replace("#", "\u2588")
    for row in range(5)
)
del _LETTER_DATA

# ANSI 256-color codes: warm gold→amber→orange gradient
_BANNER_COLORS = [220, 214, 208, 202, 166]


def _print_banner() -> None:
    """Print the colored Maestro CLI banner to stderr."""
    stream = sys.stderr
    use_color = hasattr(stream, "isatty") and stream.isatty()
    for i, line in enumerate(_BANNER_LINES):
        if use_color:
            color = _BANNER_COLORS[i % len(_BANNER_COLORS)]
            stream.write(f"\033[38;5;{color}m{line}\033[0m\n")
        else:
            stream.write(f"{line}\n")
    # Subtitle and description (mirroring the winapp CLI style)
    subtitle = f"Maestro CLI -- Version {__version__}"
    desc = "CLI orchestrator for multi-step AI execution plans (Codex, Claude, Gemini, Copilot, Qwen, Ollama)"
    if use_color:
        stream.write(f"\n  \033[1m{subtitle}\033[0m\n")
        stream.write(f"  \033[2m{desc}\033[0m\n\n")
    else:
        stream.write(f"\n  {subtitle}\n")
        stream.write(f"  {desc}\n\n")
    stream.flush()


def _print_commands() -> None:
    """Print a friendly command overview to stderr (no-args UX)."""
    stream = sys.stderr
    use_color = hasattr(stream, "isatty") and stream.isatty()

    _GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
        ("Execute", [
            ("run", "Run a plan (text, live, tui, or jsonl output)"),
            ("replan", "Re-plan and re-run after failures"),
            ("watch", "Autonomous iteration loop"),
            ("shell", "Interactive REPL mode"),
            ("chat", "Multi-model interactive terminal"),
        ]),
        ("Inspect", [
            ("validate", "Check plan syntax and structure"),
            ("explain", "Show cache hit/miss per task"),
            ("status", "Show task staleness vs last run"),
            ("diff", "Compare two runs side-by-side"),
            ("suggest", "Suggest optimizations from history"),
            ("estimate", "Estimate run cost before running"),
        ]),
        ("Generate", [
            ("scaffold", "Generate a plan from a brief"),
            ("ci", "Generate CI/CD YAML from a plan"),
            ("report", "HTML report from a run"),
            ("eval", "Batch judge evaluation on a run"),
        ]),
        ("Maintain", [
            ("doctor", "Diagnose the environment"),
            ("cleanup", "Delete old run directories"),
            ("backfill-costs", "Backfill costs for old runs"),
            ("ui", "Launch the web dashboard"),
        ]),
    ]

    if use_color:
        stream.write("  \033[1mCommands:\033[0m\n\n")
        for group_name, commands in _GROUPS:
            stream.write(f"    \033[1m{group_name}\033[0m\n")
            for name, desc in commands:
                padded = f"maestro {name}".ljust(28)
                stream.write(f"      \033[36m{padded}\033[0m \033[2m{desc}\033[0m\n")
            stream.write("\n")
        stream.write(
            "  \033[2mRun \033[0mmaestro <command> --help\033[2m "
            "for more information on a command.\033[0m\n\n"
        )
    else:
        stream.write("  Commands:\n\n")
        for group_name, commands in _GROUPS:
            stream.write(f"    {group_name}\n")
            for name, desc in commands:
                padded = f"maestro {name}".ljust(28)
                stream.write(f"      {padded} {desc}\n")
            stream.write("\n")
        stream.write("  Run maestro <command> --help for more information on a command.\n\n")
    stream.flush()


def _split_csv(values: Iterable[str] | None) -> set[str]:
    if not values:
        return set()
    out: set[str] = set()
    for chunk in values:
        for part in chunk.split(","):
            item = part.strip()
            if item:
                out.add(item)
    return out


def _parse_set_vars(raw: list[str]) -> dict[str, str]:
    """Parse ``--set key=value`` arguments into a template variable dict."""
    result: dict[str, str] = {}
    for entry in raw:
        if "=" not in entry:
            print(f"[maestro] error: --set value must be KEY=VALUE, got: {entry!r}")
            raise SystemExit(1)
        key, value = entry.split("=", 1)
        key = key.strip()
        if not key:
            print(f"[maestro] error: --set key must not be empty: {entry!r}")
            raise SystemExit(1)
        result[key] = value
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="maestro",
        description="Orchestrate multi-step AI execution plans from YAML (Codex, Claude, Gemini, Copilot, Qwen, Ollama)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    sub = parser.add_subparsers(dest="command")

    validate = sub.add_parser("validate", help="Validate a plan file")
    validate.add_argument("plan", help="Path to plan YAML")

    check = sub.add_parser(
        "check",
        help="Run validate + audit in one pass (single exit code)",
    )
    check.add_argument("plan", help="Path to plan YAML")
    check.add_argument(
        "--json", action="store_true", dest="json_output",
        help="Emit a structured JSON report instead of formatted text",
    )
    check.add_argument(
        "--with-suggest", action="store_true",
        help="Also include `maestro suggest` output (requires prior runs)",
    )
    check.add_argument(
        "--strict", action="store_true",
        help="Exit non-zero on any warning (validation OR audit), not just errors",
    )
    check.add_argument(
        "--run-dir", default=None,
        help="Override run directory for --with-suggest (default: plan.run_dir)",
    )

    run = sub.add_parser("run", help="Run a plan")
    run.add_argument("plan", nargs="+", help="Path(s) to plan YAML file(s)")
    run.add_argument(
        "--dry-run",
        action="store_true",
        help="Build commands and schedule tasks without executing",
    )
    run.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Override max_parallel from plan",
    )
    run.add_argument(
        "--only",
        action="append",
        default=None,
        help="Run only these task IDs (comma-separated or repeated), including dependencies",
    )
    run.add_argument(
        "--skip",
        action="append",
        default=None,
        help="Skip these task IDs (comma-separated or repeated)",
    )
    run.add_argument(
        "--run-dir",
        default=None,
        help="Override output run directory",
    )
    run.add_argument(
        "--webhook",
        metavar="URL",
        default=None,
        help="Webhook URL for run completion notifications (overrides plan webhook_url)",
    )
    run.add_argument(
        "--execution-profile",
        "--profile-mode",
        "--mode",
        choices=("plan", "safe", "yolo"),
        default="plan",
        help="Override runtime safety mode: plan (use YAML args), safe, or yolo",
    )
    resume_group = run.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        default=None,
        help="Path to a prior run directory to resume from (skip succeeded tasks)",
    )
    resume_group.add_argument(
        "--resume-last",
        action="store_true",
        default=False,
        help="Resume from the most recent run of this plan",
    )
    verbosity_group = run.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Show more output: task logs on success, extra detail",
    )
    verbosity_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        default=False,
        help="Suppress informational output; show only errors and final summary",
    )
    run.add_argument(
        "--output",
        choices=("text", "jsonl", "live", "tui"),
        default="text",
        help="Output format: text (default human-readable) or jsonl (structured JSON Lines on stdout)",
    )
    run.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Disable task result caching for this run",
    )
    run.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help="Cache directory (default: <run-dir>/.cache)",
    )
    run.add_argument(
        "--tags",
        type=str,
        default=None,
        help="Only run tasks with these tags (comma-separated)",
    )
    run.add_argument(
        "--skip-tags",
        type=str,
        default=None,
        help="Skip tasks with these tags (comma-separated)",
    )
    run.add_argument(
        "--mask-secrets",
        action="store_true",
        default=False,
        help="Auto-detect and mask secret values in logs",
    )
    run.add_argument(
        "--auto-approve",
        action="store_true",
        default=False,
        help="Auto-approve all approval gates",
    )
    run.add_argument(
        "--parallel",
        action="store_true",
        default=False,
        help="Run multiple plans in parallel (default: sequential)",
    )
    run.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="set_vars",
        help="Set template variable (repeatable, e.g. --set env=prod)",
    )

    cleanup_parser = sub.add_parser("cleanup", help="Delete old run directories")
    cleanup_parser.add_argument("plan", help="Path to plan YAML (to locate .maestro-runs)")
    cleanup_parser.add_argument(
        "--keep",
        type=int,
        default=10,
        help="Keep the N most recent runs (default: 10)",
    )
    cleanup_parser.add_argument(
        "--older-than",
        type=int,
        default=None,
        metavar="DAYS",
        help="Only delete runs older than DAYS days",
    )
    cleanup_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List directories that would be deleted without deleting them",
    )

    backfill_costs = sub.add_parser(
        "backfill-costs",
        help="Backfill missing task/run costs from historical log files",
    )
    backfill_costs.add_argument(
        "--root",
        default=".",
        help="Project root used to auto-discover .maestro-runs (default: current directory)",
    )
    backfill_costs.add_argument(
        "--run-root",
        action="append",
        default=None,
        help="Explicit .maestro-runs directory (repeatable). Overrides auto-discovery when provided.",
    )
    backfill_costs.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute updates without writing manifest/result files",
    )
    backfill_costs.add_argument(
        "--codex-pricing-file",
        default=None,
        help=(
            "Path to JSON pricing table for Codex token-based estimates. "
            "Loaded into MAESTRO_CODEX_PRICING_JSON for this run."
        ),
    )

    scaffold_parser = sub.add_parser("scaffold", help="Generate a plan from a brief")
    scaffold_parser.add_argument("brief", nargs="?", default=None, help="Path to brief YAML file")
    scaffold_parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output path for generated plan (default: stdout)",
    )
    scaffold_parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the generated plan after scaffolding",
    )
    scaffold_parser.add_argument(
        "--cost-check",
        action="store_true",
        help="Run cost safety checks on the generated plan",
    )
    scaffold_parser.add_argument(
        "--library",
        default=None,
        help="Workflow library to use (built-in name or path to library YAML)",
    )
    scaffold_parser.add_argument(
        "--list-libraries",
        action="store_true",
        help="List available built-in workflow libraries and exit",
    )
    scaffold_parser.add_argument(
        "--strict-defaults",
        action="store_true",
        help=(
            "Inject sane first-run defaults: timeout_sec=1500 (above W20's "
            "900s tight-timeout threshold), retry_delay_sec=[60, 120] "
            "(progressive backoff), max_cost_usd=10.0, budget_warning_pct=0.8. "
            "Recommended for first-time authors to skip the warning whack-a-mole."
        ),
    )

    ci_parser = sub.add_parser("ci", help="Generate CI/CD YAML from a plan")
    ci_parser.add_argument("plan", help="Path to plan YAML")
    ci_parser.add_argument(
        "--provider",
        choices=SUPPORTED_CI_PROVIDERS,
        default="github_actions",
        help="CI provider to target (default: github_actions)",
    )
    ci_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path for generated CI YAML (default: stdout)",
    )
    ci_parser.add_argument(
        "--workflow-name",
        default="Maestro CI",
        help="Workflow/pipeline name (default: Maestro CI)",
    )
    ci_parser.add_argument(
        "--python-version",
        default="3.11",
        help="Python version for generated CI jobs (default: 3.11)",
    )
    ci_parser.add_argument(
        "--test-command",
        default="python -m pytest -q",
        help="Normal test command for the generated test lane",
    )

    doctor_parser = sub.add_parser("doctor", help="Diagnose the Maestro environment")
    doctor_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    doctor_parser.add_argument(
        "--run-dir",
        default=".maestro-runs",
        help="Run directory to check for write access (default: .maestro-runs)",
    )
    doctor_parser.add_argument(
        "--full",
        action="store_true",
        help="Run extended integration checks (cache, knowledge, skills, plans, prior runs)",
    )
    doctor_parser.add_argument(
        "--hardware",
        action="store_true",
        help="Report local hardware (GPU/VRAM) and installed local models, then exit",
    )

    sub.add_parser("mcp-server", help="Launch the MCP protocol server (requires [mcp] extra)")

    ui_parser = sub.add_parser("ui", help="Launch the Maestro web UI")
    ui_parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    ui_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to listen on (default: 8000)",
    )
    ui_parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't auto-open the browser on startup",
    )
    ui_parser.add_argument(
        "--project-root",
        action="append",
        default=None,
        help=(
            "Additional project root to include in run discovery "
            "(repeatable; defaults to current directory)."
        ),
    )

    report_parser = sub.add_parser(
        "report",
        help="Generate a self-contained HTML report from a run directory",
    )
    report_parser.add_argument(
        "run_path",
        help="Path to a .maestro-runs/<run-id>/ directory",
    )
    report_parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output HTML path (default: <run-path>/report.html)",
    )

    diff_p = sub.add_parser("diff", help="Compare two run directories side-by-side")
    diff_p.add_argument("run_a", help="Path to first run directory")
    diff_p.add_argument("run_b", help="Path to second run directory")
    diff_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output in JSON format",
    )

    explain_p = sub.add_parser("explain", help="Show cache hit/miss status for each task")
    explain_p.add_argument("plan", help="Path to plan YAML")
    explain_p.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help="Cache directory (default: auto-detect from plan)",
    )
    explain_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output in JSON format",
    )
    explain_p.add_argument(
        "--context",
        action="store_true",
        default=False,
        help="Show context retrieval trajectory: BM25 scores, hop decay, and budget trim decisions per upstream per task",
    )

    status_p = sub.add_parser("status", help="Show task staleness vs last run")
    status_p.add_argument("plan", help="Path to plan YAML")
    status_p.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help="Cache directory (default: auto-detect from plan)",
    )
    status_p.add_argument(
        "--run-dir",
        default=None,
        metavar="DIR",
        help="Override run directory for finding latest run",
    )
    status_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output in JSON format",
    )

    estimate_p = sub.add_parser(
        "estimate",
        help="Estimate the cost of running a plan before running it",
    )
    estimate_p.add_argument("plan", help="Path to plan YAML")
    estimate_p.add_argument(
        "--run-dir",
        default=None,
        metavar="DIR",
        help="Override run directory for prior-run cost history",
    )
    estimate_p.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        dest="set_vars",
        help="Inject a template variable (repeatable) for prompt rendering",
    )
    estimate_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output in JSON format",
    )

    eval_p = sub.add_parser("eval", help="Run batch judge evaluation on a completed run")
    eval_p.add_argument("eval_yaml", help="Path to eval suite YAML")
    eval_p.add_argument("run_path", help="Path to a completed run directory")
    eval_p.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output in JSON format",
    )
    eval_p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Show detailed judge reasoning",
    )

    suggest_p = sub.add_parser("suggest", help="Suggest plan optimizations based on run history")
    suggest_p.add_argument("plan", help="Path to the plan YAML file")
    suggest_p.add_argument("--run-dir", help="Override run directory")
    suggest_p.add_argument("--min-runs", type=int, default=3, help="Minimum runs to analyze (default: 3)")
    suggest_p.add_argument("--json", action="store_true", help="Output as JSON")
    shell_p = sub.add_parser("shell", help="Launch interactive REPL mode")
    shell_p.add_argument("--plan", help="Path to initial plan YAML file")

    skill_p = sub.add_parser("skill", help="Discover and list available skills")
    skill_p.add_argument("action", nargs="?", default="list", choices=["list", "search", "recommend"], help="Action (default: list)")
    skill_p.add_argument("--query", "-q", default="", help="Search query for skill filtering")
    skill_p.add_argument("--json", action="store_true", help="Output as JSON")
    skill_p.add_argument("--dir", action="append", dest="skill_dirs", help="Additional skill directories to scan")

    chat_p = sub.add_parser("chat", help="Multi-model interactive chat terminal")
    chat_p.add_argument(
        "--engine",
        choices=("claude", "codex", "gemini", "copilot", "qwen", "ollama", "llama"),
        default="claude",
        help="Default engine (default: claude)",
    )
    chat_p.add_argument("--model", default=None, help="Default model (uses engine default if omitted)")
    chat_p.add_argument(
        "--execution-profile",
        choices=("plan", "safe", "yolo"),
        default="plan",
        help="Runtime safety mode (default: plan)",
    )
    chat_p.add_argument(
        "--no-auto-context",
        action="store_true",
        default=False,
        help="Disable startup auto-loading of AGENTS.md / CLAUDE.md context files",
    )

    replan_p = sub.add_parser("replan", help="Adaptively re-plan and re-run a failed plan")
    replan_p.add_argument("plan", help="Path to plan YAML")
    replan_p.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Maximum re-plan attempts (default: 3)",
    )
    replan_p.add_argument(
        "--model",
        default="opus",
        help="Model for failure analysis (default: opus)",
    )
    replan_p.add_argument(
        "--variants",
        type=int,
        default=1,
        help="Number of candidate plan variants to generate per failed round (default: 1)",
    )
    replan_p.add_argument(
        "--debug-prob",
        type=float,
        default=0.5,
        help="When searching multiple variants, probability of expanding an invalid leaf instead of the best valid leaf (default: 0.5)",
    )
    replan_p.add_argument(
        "--selection-policy",
        choices=("debug_prob", "ucb1"),
        default="debug_prob",
        help="Leaf selection policy for multi-variant search (default: debug_prob)",
    )
    replan_p.add_argument(
        "--exploration-constant",
        type=float,
        default=DEFAULT_EXPLORATION_CONSTANT,
        help="UCB1 exploration constant for multi-variant search (default: 1.41421356237)",
    )
    replan_p.add_argument(
        "--population-strategy",
        choices=("best", "tournament"),
        default="best",
        help="How to pick the next variant from the candidate population (default: best)",
    )
    replan_p.add_argument(
        "--tournament-size",
        type=int,
        default=2,
        help="Contestant count for tournament population selection (default: 2)",
    )
    replan_p.add_argument(
        "--elite-count",
        type=int,
        default=1,
        help="Number of top leaves preserved in the selection pool across rounds (default: 1)",
    )
    replan_p.add_argument(
        "--diversity-floor",
        type=float,
        default=0.25,
        help="Minimum mutation-signature distance between tournament contestants (default: 0.25; 0 disables)",
    )
    replan_p.add_argument(
        "--auto-approve",
        action="store_true",
        default=False,
        help="Auto-approve corrected plans without prompting",
    )
    replan_p.add_argument(
        "--execution-profile",
        "--profile-mode",
        "--mode",
        choices=("plan", "safe", "yolo"),
        default="plan",
        help="Runtime safety mode",
    )
    replan_p.add_argument("--dry-run", action="store_true", default=False)
    replan_p.add_argument("--verbose", "-v", action="store_true", default=False)
    replan_p.add_argument("--quiet", "-q", action="store_true", default=False)
    replan_p.add_argument("--output", choices=("text", "jsonl", "live", "tui"), default="text")
    replan_p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                          dest="set_vars", help="Set template variable (repeatable)")
    replan_p.set_defaults(func=_cmd_replan)

    watch_p = sub.add_parser("watch", help="Autonomous metric-driven iteration loop")
    watch_p.add_argument("plan", help="Path to plan YAML (must have watch: block)")
    watch_p.add_argument("--dry-run", action="store_true", help="Build commands without executing")
    watch_p.add_argument("--execution-profile", choices=["plan", "safe", "yolo"], default="plan")
    watch_p.add_argument("--max-parallel", type=int, default=None)
    watch_p.add_argument("--auto-approve", action="store_true")
    watch_p.add_argument("--verbose", "-v", action="store_true")
    watch_p.add_argument("--quiet", "-q", action="store_true")
    watch_p.add_argument("--output", choices=["text", "jsonl", "live", "tui"], default="text")
    watch_p.add_argument("--mask-secrets", action="store_true")
    watch_p.add_argument("--resume-last", action="store_true")
    watch_p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                         dest="set_vars", help="Set template variable (repeatable)")
    watch_p.set_defaults(func=_cmd_watch)

    p_audit = sub.add_parser("audit", help="Audit plan for security issues")
    p_audit.add_argument("plan", help="Plan YAML file")
    p_audit.add_argument("--json", action="store_true", dest="json_output",
                         help="Output findings as JSON")
    p_audit.add_argument("--fix", action="store_true", default=False,
                         help="Auto-fix safe, mechanical audit findings")
    p_audit.add_argument("--coverage", action="store_true", default=False,
                         help="Show per-category coverage breakdown for SEC001-SEC016 rules")
    p_audit.set_defaults(command="audit")

    p_verify = sub.add_parser("verify", help="Verify event chain integrity")
    p_verify.add_argument("run_path", help="Path to run directory")
    p_verify.add_argument("--json", action="store_true", dest="json_output",
                          help="Output result as JSON")
    p_verify.set_defaults(command="verify")

    p_blame = sub.add_parser("blame", help="Trace failure causality in a completed run")
    p_blame.add_argument("run_path", help="Path to a completed run directory")
    p_blame.add_argument("--json", action="store_true", dest="json_output",
                         help="Output blame chain as JSON")
    p_blame.set_defaults(command="blame")

    p_ci_analyze = sub.add_parser("ci-analyze", help="Analyze a failed CI run and suggest remediation")
    p_ci_analyze.add_argument("run_path", help="Path to a completed run directory")
    p_ci_analyze.add_argument("--json", action="store_true", dest="json_output",
                              help="Output analysis as JSON")
    p_ci_analyze.set_defaults(command="ci-analyze")

    p_budget = sub.add_parser("budget", help="Show cross-run budget spending")
    p_budget.add_argument("--root", default=None, help="Project root directory (default: cwd)")
    p_budget.set_defaults(command="budget")

    p_otel = sub.add_parser("export-otel", help="Export a run as OpenTelemetry spans")
    p_otel.add_argument("run_path", help="Path to a completed run directory")
    p_otel.add_argument("--endpoint", default=None, help="OTLP endpoint URL (default: JSON to stdout)")
    p_otel.add_argument("--json", action="store_true", dest="json_output",
                         help="Output span data as JSON")
    p_otel.add_argument(
        "--include-content",
        action="store_true",
        help="Attach captured task input/output previews to exported spans",
    )
    p_otel.add_argument(
        "--otel-mask-prompts",
        action="store_true",
        dest="otel_mask_prompts",
        help="Redact captured input/output content in exported spans",
    )
    p_otel.set_defaults(command="export-otel")

    return parser


def _find_latest_run(plan: PlanSpec, run_dir: str | None = None) -> Path | None:
    """Find the most recent run directory for a plan.

    Scans the run directory for subdirectories ending with
    ``_<plan_name>`` that contain a ``run_manifest.json``.
    Returns the latest (lexicographic sort on timestamp prefix)
    or ``None`` if no prior runs exist.
    """
    from .utils import sanitize_dirname

    run_root = resolve_path(plan.source_dir, run_dir or plan.run_dir)
    if run_root is None or not run_root.exists():
        return None

    safe_name = sanitize_dirname(plan.name)
    suffix = f"_{safe_name}"

    candidates: list[Path] = []
    for entry in run_root.iterdir():
        if (
            entry.is_dir()
            and entry.name.endswith(suffix)
            and (entry / "run_manifest.json").exists()
        ):
            candidates.append(entry)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def _print_warnings(plan_warnings: list[str]) -> None:
    if plan_warnings:
        print(f"\n[maestro] {len(plan_warnings)} warning(s):")
        for w in plan_warnings:
            print(f"  - {w}")


def _cmd_validate(plan_path: str) -> int:
    plan = load_plan(plan_path)
    print(f"Plan is valid: {Path(plan_path).resolve()}")
    print(f"- name: {plan.name}")
    print(f"- tasks: {len(plan.tasks)}")
    print(f"- max_parallel: {plan.max_parallel}")
    print(f"- fail_fast: {plan.fail_fast}")
    if plan.imports:
        print(f"- imports: {len(plan.imports)} file(s)")
    _density_score, _density_label, _density_factors = compute_plan_density_score(plan)
    print(f"- complexity: {_density_score:.2f} ({_density_label})")
    if _density_factors and _density_score >= 0.30:
        print(f"  factors: {_density_factors}")
    _print_warnings(plan.validation_warnings)
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    """Unified validate + audit (+ optional suggest) in one pass.

    Authors run validate and audit together 100% of the time; this command
    consolidates the two into a single command with a single exit code,
    addressing an internal post-mortem suggestion #3 (2026-04-26).

    Exit codes:
      - 0 if no errors and (without --strict) no warnings either escalate
      - 1 if validation fails OR audit reports any error finding
      - 1 if --strict and any warning is present
    """
    from .audit import audit_plan, format_audit, format_audit_json

    plan_path = Path(args.plan).resolve()
    json_mode = bool(getattr(args, "json_output", False))
    strict = bool(getattr(args, "strict", False))

    # ----- Validate -----
    try:
        plan = load_plan(plan_path)
    except Exception as exc:  # PlanValidationError or downstream
        if json_mode:
            import json as _json
            print(_json.dumps({
                "ok": False,
                "stage": "validate",
                "error": str(exc),
                "validation_warnings": [],
                "audit_findings": [],
            }, indent=2))
        else:
            print(f"[maestro] check: validation failed: {exc}")
        return 1

    # ----- Audit -----
    findings = audit_plan(plan)
    audit_errors = [f for f in findings if f.severity == "error"]
    audit_warnings = [f for f in findings if f.severity == "warning"]
    audit_info = [f for f in findings if f.severity == "info"]
    validation_warnings = list(plan.validation_warnings)

    # ----- Suggest (optional) -----
    suggestions = None
    if getattr(args, "with_suggest", False):
        try:
            from .suggest import suggest_plan
            run_dir = (
                Path(args.run_dir) if getattr(args, "run_dir", None) else Path(plan.run_dir)
            )
            suggestions = suggest_plan(plan, run_dir, min_runs=3)
        except Exception as exc:
            if not json_mode:
                print(f"[maestro] check: suggest skipped ({exc})")

    # ----- Output -----
    if json_mode:
        import json as _json
        report: dict[str, object] = {
            "plan": str(plan_path),
            "name": plan.name,
            "tasks": len(plan.tasks),
            "validation_warnings": validation_warnings,
            "audit_findings": _json.loads(format_audit_json(findings)),
            "summary": {
                "validation_warnings": len(validation_warnings),
                "audit_errors": len(audit_errors),
                "audit_warnings": len(audit_warnings),
                "audit_info": len(audit_info),
            },
        }
        if suggestions is not None:
            from .suggest import format_suggestions_json
            report["suggestions"] = _json.loads(format_suggestions_json(suggestions))
        report["ok"] = (
            len(audit_errors) == 0
            and (not strict or (len(validation_warnings) == 0 and len(audit_warnings) == 0))
        )
        print(_json.dumps(report, indent=2))
    else:
        print(f"Plan: {plan_path}")
        print(f"- name: {plan.name}")
        print(f"- tasks: {len(plan.tasks)}")
        print()
        print("== Validation ==")
        if validation_warnings:
            for w in validation_warnings:
                print(f"  [warn] {w}")
        else:
            print("  no warnings")
        print()
        print("== Audit ==")
        if findings:
            print(format_audit(findings))
        else:
            print("  no findings")
        print(
            f"\n[maestro] check: {len(validation_warnings)} validation warning(s), "
            f"{len(audit_errors)} audit error(s), "
            f"{len(audit_warnings)} audit warning(s), "
            f"{len(audit_info)} audit info"
        )
        if suggestions is not None:
            from .suggest import format_suggestions
            print()
            print("== Suggestions ==")
            print(format_suggestions(suggestions))

    if audit_errors:
        return 1
    if strict and (validation_warnings or audit_warnings):
        return 1
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    plan_args = list(args.plan) if isinstance(args.plan, (list, tuple)) else [args.plan]

    if args.output == "tui" and len(plan_args) > 1:
        print("[maestro] error: TUI mode does not support multi-plan execution yet")
        return 1

    if len(plan_args) > 1:
        verbosity_mp: Verbosity = "verbose" if args.verbose else "quiet" if args.quiet else "normal"
        mp_result = run_multi_plan(
            plan_args,
            parallel=getattr(args, "parallel", False),
            execution_profile=args.execution_profile,
            dry_run=bool(args.dry_run),
            verbosity=verbosity_mp,
            output_mode=getattr(args, "output", "text"),
            auto_approve=args.auto_approve,
        )
        return 0 if mp_result.success else 1

    plan = load_plan(plan_args[0])
    _jsonl = getattr(args, "output", "text") == "jsonl"
    _tui = getattr(args, "output", "text") == "tui"
    if not _jsonl and not _tui:
        _print_warnings(plan.validation_warnings)

    verbosity: Verbosity = "verbose" if args.verbose else "quiet" if args.quiet else "normal"

    only = _split_csv(args.only)
    skip = _split_csv(args.skip)
    tags = {t.strip() for t in args.tags.split(",")} if args.tags else None
    skip_tags = {t.strip() for t in args.skip_tags.split(",")} if args.skip_tags else None

    resume_path: Path | None = None
    if args.resume:
        resume_path = Path(args.resume).resolve()
        if not resume_path.exists():
            print(f"[maestro] error: resume path does not exist: {resume_path}")
            return 1
    elif args.resume_last:
        resume_path = _find_latest_run(plan, run_dir=args.run_dir)
        if resume_path is None:
            print(f"[maestro] error: no prior runs found for plan '{plan.name}'")
            return 1
        if verbosity != "quiet":
            print(f"[maestro] resuming from: {resume_path}")

    if args.mask_secrets:
        plan.secrets_auto = True

    set_vars = _parse_set_vars(args.set_vars) if args.set_vars else None

    cache_dir: Path | None = None
    if not args.no_cache:
        cache_dir_override = getattr(args, "cache_dir", None)
        cache_root = resolve_path(
            plan.source_dir,
            cache_dir_override if cache_dir_override else (args.run_dir or plan.run_dir),
        )
        if cache_root is not None:
            cache_dir = cache_root / ".cache"

    result: PlanRunResult | None
    if args.output == "live":
        try:
            from .live import create_live_callback
        except ImportError:
            print("[maestro] error: live output dependencies not installed.\n"
                  "  Install them with: pip install maestro-ai-cli[live]")
            return 1
        live_ctx, event_callback = create_live_callback(plan)
        with live_ctx:
            result = run_plan(
                plan,
                dry_run=bool(args.dry_run),
                execution_profile=args.execution_profile,
                max_parallel_override=args.max_parallel,
                only=only or None,
                skip=skip or None,
                run_dir_override=args.run_dir,
                webhook_url=args.webhook if args.webhook is not None else plan.webhook_url,
                resume_path=resume_path,
                verbosity="quiet",
                output_mode="text",
                cache_dir=cache_dir,
                tags=tags,
                skip_tags=skip_tags,
                auto_approve=args.auto_approve,
                event_callback=event_callback,
                extra_template_vars=set_vars,
            )
    elif args.output == "tui":
        try:
            from .tui import MaestroApp
        except ImportError:
            print("[maestro] error: TUI dependencies not installed. Run: pip install maestro-ai-cli[tui]")
            return 1
        app = MaestroApp(
            plan,
            dry_run=bool(args.dry_run),
            execution_profile=args.execution_profile,
            max_parallel_override=args.max_parallel,
            run_dir_override=args.run_dir,
            auto_approve=args.auto_approve,
            resume_path=resume_path,
            cache_dir=cache_dir,
            only=only or None,
            skip=skip or None,
            tags=tags,
            skip_tags=skip_tags,
            webhook_url=args.webhook if args.webhook is not None else plan.webhook_url,
            extra_template_vars=set_vars,
        )
        app.run()
        result = app._result
    else:
        result = run_plan(
            plan,
            dry_run=bool(args.dry_run),
            execution_profile=args.execution_profile,
            max_parallel_override=args.max_parallel,
            only=only or None,
            skip=skip or None,
            run_dir_override=args.run_dir,
            webhook_url=args.webhook if args.webhook is not None else plan.webhook_url,
            resume_path=resume_path,
            verbosity=verbosity,
            output_mode=args.output,
            cache_dir=cache_dir,
            tags=tags,
            skip_tags=skip_tags,
            auto_approve=args.auto_approve,
            extra_template_vars=set_vars,
        )

    if args.dry_run and not _jsonl and not _tui:
        print("\n[maestro] dry-run checklist (NOT validated):")
        print("  - [ ] Engine CLIs on PATH (claude, codex, gemini, copilot, qwen, ollama, llama-cli)")
        print("  - [ ] workdir directories exist")
        print("  - [ ] Network/API access available")
        print("  - [ ] Git worktree clean (if requires_clean_worktree)")
        if any(t.requires_approval for t in plan.tasks):
            print("  - [ ] Approval gates will be enforced interactively [would require approval]")

    if result is None:
        return 1
    return 0 if result.success else 1


def _cmd_cleanup(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)

    run_root = resolve_path(plan.source_dir, plan.run_dir)
    if run_root is None or not run_root.exists():
        print(f"[maestro] no run directory found: {plan.run_dir}")
        return 0

    deleted = cleanup_runs(
        run_root,
        keep=args.keep,
        older_than_days=args.older_than,
        dry_run=bool(args.dry_run),
    )

    action = "would delete" if args.dry_run else "deleted"
    for d in deleted:
        print(f"[maestro] {action}: {d.name}")

    print(f"[maestro] {action} {len(deleted)} run(s), kept {args.keep}")
    return 0


def _cmd_backfill_costs(args: argparse.Namespace) -> int:
    if args.codex_pricing_file:
        pricing_path = Path(args.codex_pricing_file)
        if not pricing_path.exists():
            print(f"[maestro] error: pricing file does not exist: {pricing_path}")
            return 1
        try:
            os.environ["MAESTRO_CODEX_PRICING_JSON"] = pricing_path.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"[maestro] error: failed to read pricing file: {exc}")
            return 1

    if args.run_root:
        run_roots = [Path(p).resolve() for p in args.run_root]
    else:
        run_roots = discover_run_roots(Path(args.root).resolve())

    summary = backfill_run_costs(run_roots=run_roots, write=not bool(args.dry_run))
    mode = "dry-run" if args.dry_run else "write"
    print(
        "[maestro] backfill-costs "
        f"mode={mode} roots={summary.run_roots} runs_scanned={summary.runs_scanned} "
        f"runs_updated={summary.runs_updated} tasks_updated={summary.tasks_updated} "
        f"result_files_updated={summary.result_files_updated} "
        f"manifests_failed={summary.manifests_failed}"
    )
    return 0 if summary.manifests_failed == 0 else 1


def _cmd_doctor(args: argparse.Namespace) -> int:
    if getattr(args, "hardware", False):
        from .hardware import detect_hardware, format_hardware, format_hardware_json

        info = detect_hardware()
        print(format_hardware_json(info) if args.json else format_hardware(info))
        return 0

    from .doctor import run_doctor

    results = run_doctor(
        run_dir=args.run_dir,
        json_output=bool(args.json),
        full=bool(getattr(args, "full", False)),
    )
    return 1 if any(status == "fail" for _, _, status in results) else 0


def _cmd_mcp_server(args: argparse.Namespace) -> int:
    try:
        from .mcp_server import main as mcp_main
    except ImportError:
        print(
            "[maestro] error: MCP dependencies not installed.\n"
            "  Install them with: pip install maestro-ai-cli[mcp]"
        )
        return 1
    mcp_main()
    return 0


def _cmd_ui(args: argparse.Namespace) -> int:
    try:
        from .web import create_app  # noqa: F811
    except ImportError:
        print(
            "[maestro] error: web dependencies not installed.\n"
            "  Install them with: pip install maestro-ai-cli[web]"
        )
        return 1

    import uvicorn  # noqa: F811

    host = args.host
    port = args.port

    if not args.no_browser:
        import threading
        import webbrowser

        url = f"http://{host}:{port}"

        def _open_browser() -> None:
            import time
            time.sleep(1)
            webbrowser.open(url)

        threading.Thread(target=_open_browser, daemon=True).start()

    from pathlib import Path as _Path

    project_roots: list[_Path] = [_Path.cwd()]
    if args.project_root:
        for raw in args.project_root:
            project_roots.append(_Path(raw).resolve())

    # De-duplicate while preserving order
    deduped_roots: list[_Path] = []
    seen_roots: set[_Path] = set()
    for root in project_roots:
        if root in seen_roots:
            continue
        seen_roots.add(root)
        deduped_roots.append(root)

    print(f"[maestro] starting web UI at http://{host}:{port}")
    if len(deduped_roots) == 1:
        print(f"[maestro] project root: {deduped_roots[0]}")
    else:
        print("[maestro] project roots:")
        for root in deduped_roots:
            print(f"[maestro]  - {root}")

    app = create_app(project_roots=deduped_roots)
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    run_path = Path(args.run_path).resolve()
    if not run_path.exists() or not run_path.is_dir():
        print(f"[maestro] error: run path is not a directory: {run_path}")
        return 1

    output_path = Path(args.output).resolve() if args.output else (run_path / "report.html")
    report_path = generate_report(run_path, output_path=output_path)
    print(f"[maestro] report written to {report_path}")
    return 0


def _cmd_diff(args: argparse.Namespace) -> int:
    from .diff import diff_runs, format_diff, format_diff_json

    run_dir_a = Path(args.run_a).resolve()
    run_dir_b = Path(args.run_b).resolve()

    for run_dir in (run_dir_a, run_dir_b):
        if not run_dir.exists() or not run_dir.is_dir():
            print(f"[maestro] error: run path is not a directory: {run_dir}")
            return 1

    try:
        diff = diff_runs(run_dir_a, run_dir_b)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[maestro] error: {exc}")
        return 1

    if args.json:
        print(format_diff_json(diff))
    else:
        print(format_diff(diff))

    return 1 if diff.regressions else 0


def _cmd_explain(args: argparse.Namespace) -> int:
    from .explain import explain_plan, format_explain, format_explain_json

    plan = load_plan(args.plan)

    if getattr(args, "context", False):
        from .explain import (
            explain_context_trajectory,
            format_context_trajectory,
            format_context_trajectory_json,
        )

        latest_run = _find_latest_run(plan)
        if latest_run is None:
            print("[maestro] error: no previous run found for context trajectory")
            return 1
        reports = explain_context_trajectory(latest_run)
        if args.json:
            print(format_context_trajectory_json(reports))
        else:
            print(format_context_trajectory(reports))
        return 0

    cache_dir = _resolve_cache_dir(plan, args.cache_dir)

    explanation = explain_plan(plan, cache_dir)
    if args.json:
        print(format_explain_json(explanation))
    else:
        print(format_explain(explanation))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    from .status import plan_status, format_status, format_status_json

    plan = load_plan(args.plan)
    cache_dir = _resolve_cache_dir(plan, args.cache_dir)
    latest_run = _find_latest_run(plan, run_dir=args.run_dir)

    ps = plan_status(plan, latest_run, cache_dir)
    if args.json:
        print(format_status_json(ps))
    else:
        print(format_status(ps))
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    from .eval import run_eval, format_eval, format_eval_json

    run_path = Path(args.run_path).resolve()
    if not run_path.exists() or not run_path.is_dir():
        print(f"[maestro] error: run path is not a directory: {run_path}")
        return 1

    suite = run_eval(Path(args.eval_yaml), run_path)

    if args.json:
        print(format_eval_json(suite))
    else:
        print(format_eval(suite))

    return 0 if suite.overall_pass else 1


def _cmd_suggest(args: argparse.Namespace) -> int:
    from .loader import load_plan
    from .suggest import suggest_plan, format_suggestions, format_suggestions_json

    plan = load_plan(args.plan)
    run_dir = Path(args.run_dir) if args.run_dir else Path(plan.run_dir)
    result = suggest_plan(plan, run_dir, min_runs=args.min_runs)

    if args.json:
        print(format_suggestions_json(result))
    else:
        print(format_suggestions(result))

    return 0


def _cmd_estimate(args: argparse.Namespace) -> int:
    from .estimate import estimate_plan, format_estimate, format_estimate_json

    plan = load_plan(args.plan)
    run_dir = Path(args.run_dir) if args.run_dir else Path(plan.run_dir)
    extra_vars = _parse_set_vars(getattr(args, "set_vars", []))
    report = estimate_plan(plan, run_dir, extra_template_vars=extra_vars)

    if args.json:
        print(format_estimate_json(report))
    else:
        print(format_estimate(report))

    return 0


def _cmd_ci_analyze(args: argparse.Namespace) -> int:
    from .ci_agent import analyze_ci_failure, format_ci_analysis

    run_path = Path(args.run_path)
    if not run_path.is_dir():
        print(f"[maestro] error: run directory not found: {run_path}")
        return 1
    analysis = analyze_ci_failure(run_path)
    if args.json_output:
        import json
        print(json.dumps(analysis.to_dict(), indent=2))
    else:
        print(format_ci_analysis(analysis))
    return 0


def _cmd_skill(args: argparse.Namespace) -> int:
    from .skill_registry import (
        discover_skills,
        format_skill_recommendations,
        format_skill_recommendations_json,
        format_skills,
        format_skills_json,
        recommend_skills,
        search_skills,
    )

    search_dirs = [Path.cwd() / ".claude" / "skills"]
    if args.skill_dirs:
        search_dirs.extend(Path(d) for d in args.skill_dirs)
    skills = discover_skills(search_dirs)

    query = args.query or ""
    if args.action in {"search", "recommend"} and not query:
        print(f"[maestro] error: --query is required for {args.action} action")
        return 1

    if args.action == "search":
        skills = search_skills(skills, query)
        if args.json:
            print(format_skills_json(skills))
        else:
            print(format_skills(skills))
        return 0

    if args.action == "recommend":
        recommendations = recommend_skills(skills, query)
        if args.json:
            print(format_skill_recommendations_json(recommendations))
        else:
            print(format_skill_recommendations(recommendations))
        return 0

    if args.json:
        print(format_skills_json(skills))
    else:
        print(format_skills(skills))
    return 0


def _cmd_shell(args: argparse.Namespace) -> int:
    from .shell import run_shell

    plan_path = Path(args.plan) if args.plan else None
    return run_shell(plan_path)


def _cmd_chat(args: argparse.Namespace) -> int:
    from .chat import run_chat

    return run_chat(
        engine=args.engine,
        model=args.model,
        execution_profile=args.execution_profile,
        auto_context=not bool(args.no_auto_context),
    )


def _cmd_replan(args: argparse.Namespace) -> int:
    set_vars = _parse_set_vars(args.set_vars) if args.set_vars else None
    verbosity: Verbosity = "verbose" if args.verbose else "quiet" if args.quiet else "normal"
    if args.output == "tui":
        print("[maestro] error: TUI mode is not supported for replan yet")
        return 1
    if args.output == "live":
        try:
            from .live import create_live_callback
        except ImportError:
            print("[maestro] error: live output dependencies not installed.\n"
                  "  Install them with: pip install maestro-ai-cli[live]")
            return 1
        plan = load_plan(args.plan)
        live_ctx, event_callback = create_live_callback(plan)
        with live_ctx:
            state = replan(
                args.plan,
                max_attempts=args.max_attempts,
                analysis_model=args.model,
                dry_run=args.dry_run,
                execution_profile=args.execution_profile,
                verbosity="quiet",
                output_mode="text",
                auto_approve=args.auto_approve,
                event_callback=event_callback,
                extra_template_vars=set_vars,
                variants=args.variants,
                debug_prob=args.debug_prob,
                selection_policy=args.selection_policy,
                exploration_constant=args.exploration_constant,
                population_strategy=args.population_strategy,
                tournament_size=args.tournament_size,
                elite_count=args.elite_count,
                diversity_floor=args.diversity_floor,
            )
    else:
        state = replan(
            args.plan,
            max_attempts=args.max_attempts,
            analysis_model=args.model,
            dry_run=args.dry_run,
            execution_profile=args.execution_profile,
            verbosity=verbosity,
            output_mode=args.output,
            auto_approve=args.auto_approve,
            extra_template_vars=set_vars,
            variants=args.variants,
            debug_prob=args.debug_prob,
            selection_policy=args.selection_policy,
            exploration_constant=args.exploration_constant,
            population_strategy=args.population_strategy,
            tournament_size=args.tournament_size,
            elite_count=args.elite_count,
            diversity_floor=args.diversity_floor,
        )
    return 0 if state.final_success else 1


def _cmd_watch(args: argparse.Namespace) -> int:
    from .watch import watch
    from .loader import load_plan

    set_vars = _parse_set_vars(args.set_vars) if args.set_vars else None
    verbosity: Verbosity = "verbose" if args.verbose else "quiet" if args.quiet else "normal"

    state: WatchState | None
    if args.output == "tui":
        print("[maestro] error: TUI output mode is not yet supported for watch.\n"
              "  Use --output text, --output jsonl, or --output live instead.")
        return 1
    elif args.output == "live":
        try:
            from .live import create_live_callback
        except ImportError:
            print("[maestro] error: live output dependencies not installed.\n"
                  "  Install them with: pip install maestro-ai-cli[live]")
            return 1
        plan = load_plan(args.plan)
        live_ctx, event_callback = create_live_callback(plan)
        with live_ctx:
            state = watch(
                args.plan,
                max_parallel_override=args.max_parallel,
                dry_run=args.dry_run,
                execution_profile=args.execution_profile,
                verbosity="quiet",
                output_mode="text",
                auto_approve=args.auto_approve,
                event_callback=event_callback,
                extra_template_vars=set_vars,
            )
    else:
        event_callback = None
        if args.output == "jsonl":
            import json

            def event_callback(event_type: str, data: dict[str, object]) -> None:
                payload = {"event": event_type, **data}
                print(json.dumps(payload, default=str), flush=True)

        state = watch(
            args.plan,
            max_parallel_override=args.max_parallel,
            dry_run=args.dry_run,
            execution_profile=args.execution_profile,
            verbosity=verbosity,
            output_mode=args.output,
            auto_approve=args.auto_approve,
            event_callback=event_callback,
            extra_template_vars=set_vars,
        )

    if state is None:
        return 1

    if args.output != "jsonl":
        print(f"\n[maestro] watch complete: {state.status}")
        print(f"[maestro]   iterations: {state.total_iterations}")
        if state.best_metric is not None:
            print(f"[maestro]   best metric: {state.best_metric} (iteration {state.best_iteration})")
        if state.total_cost_usd:
            print(f"[maestro]   total cost: ${state.total_cost_usd:.2f}")

    return 0 if state.status in ("improved", "max_iterations", "plateau", "target_reached") else 1


def _resolve_cache_dir(plan: PlanSpec, cache_dir_override: str | None) -> Path | None:
    """Resolve the cache directory from CLI args or plan defaults."""
    if cache_dir_override:
        return Path(cache_dir_override).resolve()
    run_root = resolve_path(plan.source_dir, plan.run_dir)
    if run_root is not None:
        return run_root / ".cache"
    return None


def _cmd_scaffold(args: argparse.Namespace) -> int:
    from .scaffold import (
        list_workflow_libraries,
        load_brief,
        scaffold_plan,
        validate_plan_cost_safety,
    )

    if args.list_libraries:
        libs = list_workflow_libraries()
        if not libs:
            print("[maestro] no built-in workflow libraries available")
            return 0
        print("[maestro] built-in workflow libraries:\n")
        for lib in libs:
            print(f"  {lib['name']:20s} {lib['description']}")
        print(f"\nUsage: maestro scaffold <brief.yaml> --library {libs[0]['name']}")
        return 0

    if not args.brief:
        print("[maestro] error: brief file is required (unless --list-libraries)")
        return 1

    brief = load_brief(
        args.brief,
        library_override=args.library,
        strict_defaults=getattr(args, "strict_defaults", False),
    )
    plan_yaml = scaffold_plan(brief)

    if args.cost_check:
        warnings = validate_plan_cost_safety(plan_yaml)
        for w in warnings:
            print(f"[maestro] cost-warning: {w}")

    if args.validate:
        import tempfile
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False,
        ) as tmp_file:
            tmp_file.write(plan_yaml)
            tmp_path = Path(tmp_file.name)
        try:
            load_plan(tmp_path)
            print("[maestro] generated plan is valid")
        finally:
            tmp_path.unlink(missing_ok=True)

    if args.output:
        Path(args.output).write_text(plan_yaml, encoding="utf-8")
        print(f"[maestro] plan written to {args.output}")
    else:
        print(plan_yaml)

    return 0


def _cmd_ci(args: argparse.Namespace) -> int:
    ci_yaml = generate_ci_yaml(
        args.plan,
        provider=args.provider,
        workflow_name=args.workflow_name,
        python_version=args.python_version,
        test_command=args.test_command,
    )

    if args.output:
        output_path = write_ci_yaml(ci_yaml, args.output)
        print(f"[maestro] CI config written to {output_path}")
    else:
        print(ci_yaml, end="")

    return 0


def _cmd_blame(args: argparse.Namespace) -> int:
    from .blame import blame_run, format_blame, format_blame_json

    run_path = Path(args.run_path).resolve()
    if not run_path.exists() or not run_path.is_dir():
        print(f"[maestro] error: run path is not a directory: {run_path}")
        return 1

    chain = blame_run(run_path)
    if args.json_output:
        print(format_blame_json(chain))
    else:
        print(format_blame(chain))

    return 1 if chain.nodes else 0


def _cmd_budget(args: argparse.Namespace) -> int:
    from .budget import format_budget, _DEFAULT_LEDGER_PATH

    root = Path(args.root).resolve() if hasattr(args, "root") and args.root else Path.cwd()
    ledger_path = root / _DEFAULT_LEDGER_PATH
    print(format_budget(ledger_path))
    return 0


def _cmd_export_otel(args: argparse.Namespace) -> int:
    from .otel import export_run

    run_path = Path(args.run_path).resolve()
    if not run_path.exists() or not run_path.is_dir():
        print(f"[maestro] error: run path is not a directory: {run_path}")
        return 1

    result = export_run(
        run_path,
        endpoint=args.endpoint if hasattr(args, "endpoint") else None,
        json_output=args.json_output,
        include_content=bool(getattr(args, "include_content", False)),
        mask_content=bool(getattr(args, "otel_mask_prompts", False)),
    )

    if not result.get("success"):
        print(f"[maestro] error: {result.get('error', 'export failed')}")
        return 1

    if "json" in result:
        print(result["json"])
    else:
        fmt = result.get("format", "unknown")
        endpoint = result.get("endpoint", "console")
        print(f"[maestro] exported {result['span_count']} spans ({fmt}) to {endpoint}")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    import json as _json

    run_path = Path(args.run_path).resolve()
    if not run_path.exists() or not run_path.is_dir():
        print(f"[maestro] error: run path is not a directory: {run_path}")
        return 1
    events_path = run_path / "events.jsonl"
    if not events_path.exists():
        print(f"[maestro] error: events.jsonl not found in {run_path}")
        return 1
    events = replay_events(events_path)
    status = verify_chain(events)

    # Artefact integrity check
    artefact_mismatches = verify_artefact_hashes(run_path, events)
    artefact_status = "tampered" if artefact_mismatches else "valid"
    overall = "valid" if status == "valid" and not artefact_mismatches else (
        "tampered" if status == "tampered" or artefact_mismatches else status
    )

    if args.json_output:
        result = {
            "status": overall,
            "chain_status": status,
            "artefact_status": artefact_status,
            "events": len(events),
        }
        if artefact_mismatches:
            result["artefact_mismatches"] = artefact_mismatches
        print(_json.dumps(result))
    else:
        print(f"[maestro] verify: chain={status}, artefacts={artefact_status} ({len(events)} events)")
        for m in artefact_mismatches:
            print(f"[maestro]   tampered: {m}")
    return 0 if overall == "valid" else 1


def _cmd_audit(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan)
    findings = audit_plan(plan)

    if getattr(args, "coverage", False):
        if args.json_output:
            print(format_audit_coverage_json(findings))
        else:
            print(format_audit_coverage(findings))
        return 0

    if args.json_output:
        print(format_audit_json(findings))
    else:
        print(format_audit(findings))
    errors = sum(1 for f in findings if f.severity == "error")
    warnings = sum(1 for f in findings if f.severity == "warning")
    info = sum(1 for f in findings if f.severity == "info")
    print(
        f"[maestro] audit: {len(findings)} findings"
        f" ({errors} errors, {warnings} warnings, {info} info)"
    )
    if args.fix and findings:
        fixes = _audit_fix_plan(Path(args.plan), findings)
        if fixes:
            if args.json_output:
                import json as _json
                print(_json.dumps(fixes))
            else:
                for desc in fixes:
                    print(f"[maestro] fix: {desc}")
        else:
            if not args.json_output:
                print("[maestro] No auto-fixable findings")
    return 1 if errors > 0 else 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    effective = argv if argv is not None else sys.argv[1:]

    # Show banner on top-level --help (not subcommand --help)
    _SUBCOMMANDS = {
        "validate", "check", "run", "cleanup", "backfill-costs",
        "scaffold", "ci", "doctor", "ui", "report", "diff",
        "explain", "status", "eval", "suggest", "estimate", "shell", "chat",
        "replan", "watch", "audit", "verify", "blame", "export-otel",
    }
    if any(a in ("-h", "--help") for a in effective) and not (
        _SUBCOMMANDS & set(effective)
    ):
        _print_banner()
        args = parser.parse_args(argv)
        return 0  # pragma: no cover

    # No arguments → show banner + friendly command list (not an error)
    if not effective:
        _print_banner()
        _print_commands()
        return 0

    args = parser.parse_args(argv)

    try:
        if args.command == "validate":
            return _cmd_validate(args.plan)
        if args.command == "check":
            return _cmd_check(args)
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "cleanup":
            return _cmd_cleanup(args)
        if args.command == "backfill-costs":
            return _cmd_backfill_costs(args)
        if args.command == "scaffold":
            return _cmd_scaffold(args)
        if args.command == "ci":
            return _cmd_ci(args)
        if args.command == "doctor":
            return _cmd_doctor(args)
        if args.command == "mcp-server":
            return _cmd_mcp_server(args)
        if args.command == "ui":
            return _cmd_ui(args)
        if args.command == "report":
            return _cmd_report(args)
        if args.command == "diff":
            return _cmd_diff(args)
        if args.command == "explain":
            return _cmd_explain(args)
        if args.command == "status":
            return _cmd_status(args)
        if args.command == "eval":
            return _cmd_eval(args)
        if args.command == "suggest":
            return _cmd_suggest(args)
        if args.command == "estimate":
            return _cmd_estimate(args)
        if args.command == "shell":
            return _cmd_shell(args)
        if args.command == "skill":
            return _cmd_skill(args)
        if args.command == "ci-analyze":
            return _cmd_ci_analyze(args)
        if args.command == "chat":
            return _cmd_chat(args)
        if args.command == "replan":
            return _cmd_replan(args)
        if args.command == "watch":
            return _cmd_watch(args)
        if args.command == "audit":
            return _cmd_audit(args)
        if args.command == "verify":
            return _cmd_verify(args)
        if args.command == "blame":
            return _cmd_blame(args)
        if args.command == "budget":
            return _cmd_budget(args)
        if args.command == "export-otel":
            return _cmd_export_otel(args)

        parser.error(f"Unknown command: {args.command}")  # pragma: no cover
        return 2  # pragma: no cover
    except Exception as exc:
        print(f"[maestro] error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
