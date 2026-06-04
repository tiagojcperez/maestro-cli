"""MCP (Model Context Protocol) server for Maestro CLI.

Exposes Maestro's 12 CLI subcommands as MCP tools, run artefacts as
resources, and plan templates as prompts.  Any MCP-compatible client
(Claude Code, VS Code, Cursor, Zed) can consume these.

Install: ``pip install maestro-ai-cli[mcp]``
Run:     ``maestro mcp-server`` or ``python -m maestro_cli.mcp_server``

See https://modelcontextprotocol.io and ``docs/PROTOCOL-ROADMAP.md``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .loader import load_plan
from .errors import PlanValidationError

try:
    from mcp.server.fastmcp import FastMCP
    mcp = FastMCP("maestro-cli")
    _HAS_MCP = True
except ImportError:
    # MCP SDK not installed — define a no-op decorator for import compatibility
    _HAS_MCP = False

    class _NoOpDecorator:
        def tool(self, **kw):  # type: ignore[no-untyped-def]
            def _wrap(fn):  # type: ignore[no-untyped-def]
                return fn
            return _wrap

        def resource(self, *a, **kw):  # type: ignore[no-untyped-def]
            def _wrap(fn):  # type: ignore[no-untyped-def]
                return fn
            return _wrap

        def prompt(self, **kw):  # type: ignore[no-untyped-def]
            def _wrap(fn):  # type: ignore[no-untyped-def]
                return fn
            return _wrap

    mcp = _NoOpDecorator()  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DEFAULT_RUN_ROOT = ".maestro-runs"


def _find_run_root(base: Path | None = None) -> Path:
    """Locate the .maestro-runs directory from a base path."""
    root = base or Path.cwd()
    run_root = root / _DEFAULT_RUN_ROOT
    if run_root.is_dir():
        return run_root
    return run_root


def _list_run_dirs(run_root: Path) -> list[Path]:
    """List run directories sorted newest first."""
    if not run_root.is_dir():
        return []
    return sorted(
        [d for d in run_root.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )


def _find_run(run_id: str, run_root: Path | None = None) -> Path | None:
    """Find a run directory by ID (exact or prefix match)."""
    root = run_root or _find_run_root()
    if not root.is_dir():
        return None
    exact = root / run_id
    if exact.is_dir():
        return exact
    # Prefix match
    for d in root.iterdir():
        if d.is_dir() and d.name.startswith(run_id):
            return d
    return None


def _list_plan_files(base: Path | None = None) -> list[Path]:
    """List YAML plan files in the workspace (max 2 levels)."""
    root = base or Path.cwd()
    plans: list[Path] = []
    _yaml = {".yaml", ".yml"}
    try:
        for item in sorted(root.iterdir()):
            if item.is_file() and item.suffix in _yaml:
                plans.append(item)
            elif item.is_dir() and not item.name.startswith("."):
                try:
                    for child in sorted(item.iterdir()):
                        if child.is_file() and child.suffix in _yaml:
                            plans.append(child)
                except OSError:
                    pass
    except OSError:
        pass
    return plans


# ---------------------------------------------------------------------------
# Tools (12 — wrapping CLI subcommands)
# ---------------------------------------------------------------------------


@mcp.tool()
def validate_plan(plan_path: str) -> dict[str, Any]:
    """Validate a Maestro YAML plan for structural errors.

    Returns the plan summary on success or validation errors on failure.
    """
    try:
        plan = load_plan(plan_path)
        return {
            "valid": True,
            "name": plan.name,
            "task_count": len(plan.tasks),
            "task_ids": [t.id for t in plan.tasks],
            "max_parallel": plan.max_parallel,
            "fail_fast": plan.fail_fast,
            "max_cost_usd": plan.max_cost_usd,
        }
    except PlanValidationError as exc:
        return {"valid": False, "error": str(exc)}


@mcp.tool()
def run_plan_tool(
    plan_path: str,
    dry_run: bool = True,
    execution_profile: str = "plan",
    max_parallel: int | None = None,
    only: list[str] | None = None,
    skip: list[str] | None = None,
) -> dict[str, Any]:
    """Execute a Maestro plan. Defaults to dry_run=True for safety.

    Set dry_run=False to actually execute tasks.
    Returns the run result summary with per-task status, cost, and tokens.
    """
    from .scheduler import run_plan

    try:
        plan = load_plan(plan_path)
    except PlanValidationError as exc:
        return {"success": False, "error": str(exc)}

    try:
        result = run_plan(
            plan,
            dry_run=dry_run,
            execution_profile=execution_profile,  # type: ignore[arg-type]
            max_parallel_override=max_parallel,
            only=set(only) if only else None,
            skip=set(skip) if skip else None,
            verbosity="quiet",
            output_mode="jsonl",
        )
        return result.to_dict()
    except Exception as exc:
        return {"success": False, "error": str(exc)}


@mcp.tool()
def audit_plan(plan_path: str, fix: bool = False) -> dict[str, Any]:
    """Security audit a Maestro plan (SEC001-SEC018).

    Returns findings with severity, rule ID, and remediation guidance.
    Set fix=True to auto-remediate common issues.
    """
    from .audit import audit_plan as _audit_plan, fix_plan as _fix_plan

    try:
        plan = load_plan(plan_path)
    except PlanValidationError as exc:
        return {"error": str(exc)}

    findings = _audit_plan(plan)
    result: dict[str, Any] = {
        "findings": [f.to_dict() for f in findings],
        "total": len(findings),
        "errors": sum(1 for f in findings if f.severity == "error"),
        "warnings": sum(1 for f in findings if f.severity == "warning"),
    }

    if fix and findings:
        fixes = _fix_plan(Path(plan_path), findings)
        result["fixes_applied"] = fixes

    return result


@mcp.tool()
def blame_run(run_path: str) -> dict[str, Any]:
    """Trace failure causality in a completed Maestro run.

    Walks the dependency graph backward from failed tasks to identify
    root causes, cascading failures, and suggested fixes.
    """
    from .blame import blame_run as _blame_run

    rp = _find_run(run_path) or Path(run_path)
    if not rp.is_dir():
        return {"error": f"Run directory not found: {run_path}"}

    try:
        chain = _blame_run(rp)
        return chain.to_dict()
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def diff_runs(run_a: str, run_b: str) -> dict[str, Any]:
    """Compare two Maestro runs (status, cost, token, duration deltas).

    Returns per-task diffs and overall regression status.
    """
    from .diff import diff_runs as _diff_runs, format_diff_json

    path_a = _find_run(run_a) or Path(run_a)
    path_b = _find_run(run_b) or Path(run_b)

    try:
        diff = _diff_runs(path_a, path_b)
        result: dict[str, Any] = json.loads(format_diff_json(diff))
        return result
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def explain_plan(plan_path: str) -> dict[str, Any]:
    """Show cache hit/miss status per task in a Maestro plan.

    Explains why each task would or would not use cached results.
    """
    from .explain import explain_plan as _explain_plan, format_explain_json

    try:
        plan = load_plan(plan_path)
    except PlanValidationError as exc:
        return {"error": str(exc)}

    try:
        explanation = _explain_plan(plan, cache_dir=None)
        result: dict[str, Any] = json.loads(format_explain_json(explanation))
        return result
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def plan_status(plan_path: str) -> dict[str, Any]:
    """Show task staleness vs last run for a Maestro plan.

    Returns per-task freshness status (stale, fresh, never-run, deps-changed).
    """
    from .status import plan_status as _plan_status, format_status_json

    try:
        plan = load_plan(plan_path)
    except PlanValidationError as exc:
        return {"error": str(exc)}

    try:
        ps = _plan_status(plan, latest_run_path=None, cache_dir=None)
        result: dict[str, Any] = json.loads(format_status_json(ps))
        return result
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def suggest_plan(plan_path: str, min_runs: int = 3) -> dict[str, Any]:
    """Analyze run history and suggest plan optimizations.

    Requires at least min_runs prior runs to generate meaningful suggestions.
    """
    from .suggest import suggest_plan as _suggest_plan

    try:
        plan = load_plan(plan_path)
    except PlanValidationError as exc:
        return {"error": str(exc)}

    run_root = _find_run_root(Path(plan_path).parent)
    try:
        result = _suggest_plan(plan, run_root, min_runs=min_runs)
        return result.to_dict()
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def doctor() -> list[dict[str, str]]:
    """Diagnose the Maestro CLI environment.

    Checks Python version, PyYAML, engine CLIs, Git, and dependencies.
    """
    from .doctor import run_doctor

    results = run_doctor(json_output=True)
    return [
        {"check": name, "detail": detail, "status": status}
        for name, detail, status in results
    ]


@mcp.tool()
def scaffold_plan(brief_path: str, validate: bool = True) -> str:
    """Generate a Maestro plan YAML from a brief description.

    Returns the generated YAML as text.
    """
    from .scaffold import load_brief, scaffold_plan as _scaffold

    try:
        brief = load_brief(brief_path)
        yaml_text = _scaffold(brief)
        if validate:
            try:
                import tempfile
                tmp = Path(tempfile.mktemp(suffix=".yaml"))
                tmp.write_text(yaml_text, encoding="utf-8")
                load_plan(tmp)
                tmp.unlink(missing_ok=True)
            except PlanValidationError as exc:
                return f"# Generated plan has validation errors: {exc}\n\n{yaml_text}"
        return yaml_text
    except Exception as exc:
        return f"Error: {exc}"


@mcp.tool()
def verify_events(run_path: str) -> dict[str, Any]:
    """Verify event chain integrity for a Maestro run.

    Checks SHA-256 hash chain and artefact integrity.
    """
    from .eventsource import replay_events, verify_chain, verify_artefact_hashes

    rp = _find_run(run_path) or Path(run_path)
    events_file = rp / "events.jsonl"
    if not events_file.exists():
        return {"error": f"No events.jsonl in {run_path}"}

    try:
        events = replay_events(events_file)
        status = verify_chain(events)
        artefact_issues = verify_artefact_hashes(rp, events)
        return {
            "chain_status": status,
            "event_count": len(events),
            "artefact_issues": artefact_issues,
            "valid": status == "valid" and not artefact_issues,
        }
    except Exception as exc:
        return {"error": str(exc)}


@mcp.tool()
def cleanup_runs(
    plan_path: str,
    keep: int = 10,
    older_than_days: int | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Clean up old Maestro run directories.

    Defaults to dry_run=True for safety. Returns list of affected directories.
    """
    from .cleanup import cleanup_runs as _cleanup_runs
    from .utils import resolve_path

    try:
        plan = load_plan(plan_path)
    except PlanValidationError as exc:
        return {"error": str(exc)}

    run_root = resolve_path(plan.source_dir, plan.run_dir) or (
        Path(plan_path).parent / _DEFAULT_RUN_ROOT
    )

    try:
        deleted = _cleanup_runs(
            run_root, keep=keep, older_than_days=older_than_days, dry_run=dry_run,
        )
        return {
            "dry_run": dry_run,
            "affected": [str(p) for p in deleted],
            "count": len(deleted),
        }
    except Exception as exc:
        return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Resources (run artefacts + plans)
# ---------------------------------------------------------------------------


@mcp.resource("maestro://runs")
def list_runs() -> str:
    """List all Maestro run directories (newest first)."""
    run_root = _find_run_root()
    dirs = _list_run_dirs(run_root)
    runs = []
    for d in dirs[:50]:  # cap at 50
        manifest = d / "run_manifest.json"
        info: dict[str, Any] = {"id": d.name, "path": str(d)}
        if manifest.exists():
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                info["plan_name"] = data.get("plan_name")
                info["success"] = data.get("success")
                info["started_at"] = data.get("started_at")
            except (json.JSONDecodeError, OSError):
                pass
        runs.append(info)
    return json.dumps(runs, indent=2, default=str)


@mcp.resource("maestro://runs/{run_id}/manifest")
def read_manifest(run_id: str) -> str:
    """Read the run manifest for a completed Maestro run."""
    rp = _find_run(run_id)
    if rp is None:
        return json.dumps({"error": f"Run '{run_id}' not found"})
    manifest = rp / "run_manifest.json"
    if not manifest.exists():
        return json.dumps({"error": "No run_manifest.json"})
    return manifest.read_text(encoding="utf-8")


@mcp.resource("maestro://runs/{run_id}/summary")
def read_summary(run_id: str) -> str:
    """Read the human-readable run summary."""
    rp = _find_run(run_id)
    if rp is None:
        return f"Run '{run_id}' not found"
    summary = rp / "run_summary.md"
    if not summary.exists():
        return "No run_summary.md"
    return summary.read_text(encoding="utf-8")


@mcp.resource("maestro://runs/{run_id}/events")
def read_events(run_id: str) -> str:
    """Read the event log for a Maestro run."""
    rp = _find_run(run_id)
    if rp is None:
        return f"Run '{run_id}' not found"
    events = rp / "events.jsonl"
    if not events.exists():
        return "No events.jsonl"
    return events.read_text(encoding="utf-8")


@mcp.resource("maestro://runs/{run_id}/tasks/{task_id}/log")
def read_task_log(run_id: str, task_id: str) -> str:
    """Read the execution log for a specific task."""
    rp = _find_run(run_id)
    if rp is None:
        return f"Run '{run_id}' not found"
    log = rp / f"{task_id}.log"
    if not log.exists():
        return f"No log for task '{task_id}'"
    return log.read_text(encoding="utf-8")


@mcp.resource("maestro://runs/{run_id}/tasks/{task_id}/result")
def read_task_result(run_id: str, task_id: str) -> str:
    """Read the structured result for a specific task."""
    rp = _find_run(run_id)
    if rp is None:
        return json.dumps({"error": f"Run '{run_id}' not found"})
    result = rp / f"{task_id}.result.json"
    if not result.exists():
        return json.dumps({"error": f"No result for task '{task_id}'"})
    return result.read_text(encoding="utf-8")


@mcp.resource("maestro://plans")
def list_plans() -> str:
    """List YAML plan files in the workspace."""
    plans = _list_plan_files()
    return json.dumps(
        [{"name": p.name, "path": str(p)} for p in plans],
        indent=2,
        default=str,
    )


@mcp.resource("maestro://plans/{name}")
def read_plan(name: str) -> str:
    """Read a plan YAML file by name."""
    # Try direct path first
    p = Path(name)
    if p.exists():
        return p.read_text(encoding="utf-8")
    # Search in workspace
    for plan_path in _list_plan_files():
        if plan_path.name == name or plan_path.stem == name:
            return plan_path.read_text(encoding="utf-8")
    return f"Plan '{name}' not found"


# ---------------------------------------------------------------------------
# Prompts (reusable templates)
# ---------------------------------------------------------------------------


@mcp.prompt(title="Debug Failed Run")
def debug_run(run_path: str) -> str:
    """Analyze a failed Maestro run and suggest fixes."""
    rp = _find_run(run_path)
    if rp is None:
        return f"Run '{run_path}' not found. Use maestro://runs to list available runs."

    parts: list[str] = [
        f"Analyze this failed Maestro run and suggest fixes.\n",
        f"Run: {rp.name}\n",
    ]

    # Load manifest
    manifest = rp / "run_manifest.json"
    if manifest.exists():
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            parts.append(f"Plan: {data.get('plan_name')}")
            parts.append(f"Success: {data.get('success')}")
            parts.append(f"Profile: {data.get('execution_profile')}")
            parts.append(f"\nTask results:")
            for tid, tr in data.get("task_results", {}).items():
                status = tr.get("status", "unknown")
                msg = tr.get("message", "")
                parts.append(f"  - {tid}: {status}" + (f" ({msg})" if msg else ""))
        except (json.JSONDecodeError, OSError):
            parts.append("(Could not read manifest)")

    # Load events
    events_file = rp / "events.jsonl"
    if events_file.exists():
        try:
            lines = events_file.read_text(encoding="utf-8").strip().split("\n")
            parts.append(f"\nEvent count: {len(lines)}")
            # Show last 10 events
            parts.append("\nLast 10 events:")
            for line in lines[-10:]:
                parts.append(f"  {line}")
        except OSError:
            pass

    parts.append("\nPlease identify:\n1. Root cause of the failure")
    parts.append("2. Which task failed first (root cause vs cascade)")
    parts.append("3. Suggested fixes (plan changes, timeout adjustments, etc.)")

    return "\n".join(parts)


@mcp.prompt(title="Review Plan")
def review_plan(plan_path: str) -> str:
    """Review a Maestro plan for issues and suggest improvements."""
    p = Path(plan_path)
    if not p.exists():
        return f"Plan file not found: {plan_path}"

    yaml_content = p.read_text(encoding="utf-8")
    return (
        f"Review this Maestro CLI plan YAML for issues.\n\n"
        f"```yaml\n{yaml_content}\n```\n\n"
        f"Check for:\n"
        f"1. Missing or incorrect dependencies\n"
        f"2. Cost optimization (model selection, reasoning effort)\n"
        f"3. Missing quality gates (verify_command, judge, guard_command)\n"
        f"4. Security issues (context_trust, secrets, approval gates)\n"
        f"5. Reliability (timeout_sec, max_retries, escalation)\n"
        f"6. DAG structure (parallelism opportunities, unnecessary sequencing)\n"
    )


@mcp.prompt(title="Create Plan")
def create_plan(description: str) -> str:
    """Generate a Maestro plan YAML from a description."""
    return (
        f"Create a Maestro CLI plan YAML for the following:\n\n"
        f"{description}\n\n"
        f"Requirements:\n"
        f"- version: 1\n"
        f"- Use appropriate engines (claude/codex/gemini) and models\n"
        f"- Include verify_command for testable tasks\n"
        f"- Set reasonable timeout_sec values\n"
        f"- Use depends_on for task ordering\n"
        f"- Use context_from to pass information between tasks\n"
        f"- Include max_cost_usd budget\n"
        f"- Follow the model routing policy (haiku for trivial, sonnet for standard, opus for complex)\n"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the MCP server (stdio transport by default)."""
    if not _HAS_MCP:
        print("[maestro] error: MCP SDK not installed.")
        print("  Install with: pip install maestro-ai-cli[mcp]")
        raise SystemExit(1)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
