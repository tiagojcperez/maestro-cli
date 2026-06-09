"""Offline cost/token preflight for a plan (``maestro estimate``).

Estimates what a plan will cost to run **before** running it, read-only and
deterministic — no engine is ever invoked.  Two sources feed each engine task,
in priority order:

1. **history** — the average of the task's *actual* ``cost_usd`` across prior
   ``run_manifest.json`` files for the same plan.  Trustworthy: it is real
   spend, and it covers explicit-model tasks (unlike auto-route history).
2. **heuristic** — a rough lower bound derived from the prompt size and the
   per-model pricing tables, used when a task has no prior-run cost.  It cannot
   see the agent's own context (file reads, tool results) or the real output
   length, so it under-counts; it exists to flag the *relatively* expensive
   tasks, not to be exact.

Shell (``command``) tasks cost nothing; ``ollama``/``llama`` run locally
(zero cost); ``copilot`` is subscription-based (cost unknown).  ``group`` tasks
point at a sub-plan — run ``maestro estimate`` on that sub-plan directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .models import PlanSpec, TaskSpec
from .utils import extract_prompt_from_markdown, render_template, resolve_path

# Rough heuristic knobs (documented in the printed output).
_CHARS_PER_TOKEN = 4
_PROMPT_OVERHEAD_TOKENS = 400  # system prompt + tool framing the CLI adds
_HEURISTIC_OUTPUT_RATIO = 0.5  # assumed output ≈ half the input
_HEURISTIC_MIN_OUTPUT_TOKENS = 400

# Engines that never carry a per-token cost.
_LOCAL_ENGINES = frozenset({"ollama", "llama"})
_SUBSCRIPTION_ENGINES = frozenset({"copilot"})


@dataclass
class TaskEstimate:
    """Per-task cost estimate."""

    task_id: str
    kind: str  # "engine" | "command" | "group"
    engine: str | None
    model: str | None
    source: str  # history | heuristic | shell | local | subscription | unpriced
    input_tokens: int
    output_tokens: int
    cost_usd: float | None
    history_runs: int
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "engine": self.engine,
            "model": self.model,
            "source": self.source,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cost_usd": (
                round(self.cost_usd, 6) if self.cost_usd is not None else None
            ),
            "history_runs": self.history_runs,
            "note": self.note,
        }


@dataclass
class PlanEstimate:
    """Aggregated plan cost estimate."""

    plan_name: str
    task_estimates: list[TaskEstimate] = field(default_factory=list)
    total_cost_usd: float = 0.0
    by_engine: dict[str, float] = field(default_factory=dict)
    history_tasks: int = 0
    heuristic_tasks: int = 0
    unpriced_tasks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "by_engine": {k: round(v, 6) for k, v in self.by_engine.items()},
            "history_tasks": self.history_tasks,
            "heuristic_tasks": self.heuristic_tasks,
            "unpriced_tasks": self.unpriced_tasks,
            "tasks": [t.to_dict() for t in self.task_estimates],
        }


# ---------------------------------------------------------------------------
# Prior-run cost history
# ---------------------------------------------------------------------------

def _load_task_cost_history(plan_name: str, run_dir: Path) -> dict[str, list[float]]:
    """Collect actual per-task ``cost_usd`` values from prior run manifests."""
    history: dict[str, list[float]] = {}
    if not run_dir.exists() or not run_dir.is_dir():
        return history

    for candidate in run_dir.glob(f"*_{plan_name}"):
        manifest_path = candidate / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        task_results = payload.get("task_results")
        if not isinstance(task_results, dict):
            continue
        for task_id, result in task_results.items():
            if not isinstance(result, dict):
                continue
            cost = result.get("cost_usd")
            if isinstance(cost, (int, float)) and cost > 0:
                history.setdefault(task_id, []).append(float(cost))
    return history


# ---------------------------------------------------------------------------
# Heuristic token / cost estimation
# ---------------------------------------------------------------------------

def _default_model_for_engine(plan: PlanSpec, engine: str) -> str | None:
    defaults = plan.defaults
    table = {
        "codex": defaults.codex.model,
        "claude": defaults.claude.model,
        "gemini": defaults.gemini.model,
        "copilot": defaults.copilot.model,
        "qwen": defaults.qwen.model,
        "ollama": defaults.ollama.model,
        "llama": defaults.llama.model,
    }
    return table.get(engine)


def _resolve_prompt_text(
    task: TaskSpec, plan: PlanSpec, extra_template_vars: dict[str, str]
) -> str | None:
    """Best-effort rendered prompt text for a task (``None`` if unresolvable)."""
    raw: str | None = None
    if task.prompt:
        raw = task.prompt
    elif task.prompt_file:
        path = resolve_path(
            Path(plan.workspace_root) if plan.workspace_root else plan.source_dir,
            task.prompt_file,
        )
        if path is None or not path.is_file():
            path = resolve_path(plan.source_dir, task.prompt_file)
        if path is not None and path.is_file():
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                raw = None
    elif task.prompt_md_file and task.prompt_md_heading:
        path = resolve_path(
            Path(plan.workspace_root) if plan.workspace_root else plan.source_dir,
            task.prompt_md_file,
        )
        if path is None or not path.is_file():
            path = resolve_path(plan.source_dir, task.prompt_md_file)
        if path is not None and path.is_file():
            try:
                md_text = path.read_text(encoding="utf-8")
                raw = extract_prompt_from_markdown(md_text, task.prompt_md_heading)
            except OSError:
                raw = None

    if raw is None:
        return None

    variables: dict[str, str] = {
        "workspace_root": plan.workspace_root or "",
        "plan_name": plan.name,
        "task_id": task.id,
        "goal": plan.goal,
    }
    variables.update(extra_template_vars)
    return render_template(raw, variables)


def _pricing_table_for(engine: str) -> dict[str, tuple[float, float, float]]:
    from .runners import (
        _load_claude_pricing_table,
        _load_codex_pricing_table,
        _load_gemini_pricing_table,
        _load_qwen_pricing_table,
    )

    loaders = {
        "codex": _load_codex_pricing_table,
        "claude": _load_claude_pricing_table,
        "gemini": _load_gemini_pricing_table,
        "qwen": _load_qwen_pricing_table,
    }
    loader = loaders.get(engine)
    return loader() if loader is not None else {}


def _heuristic_estimate(
    engine: str, model: str | None, prompt_text: str | None
) -> tuple[int, int, float | None, str]:
    """Return ``(input_tokens, output_tokens, cost_usd, note)``."""
    from .runners import _estimate_cost_from_tokens, _normalize_model_for_pricing

    if prompt_text is None:
        return 0, 0, None, "prompt not resolvable; tokens not estimated"

    input_tokens = _PROMPT_OVERHEAD_TOKENS + len(prompt_text) // _CHARS_PER_TOKEN
    output_tokens = max(
        _HEURISTIC_MIN_OUTPUT_TOKENS, int(input_tokens * _HEURISTIC_OUTPUT_RATIO)
    )

    pricing = _pricing_table_for(engine)
    lookup_model = _normalize_model_for_pricing(model) or (model or "default")
    cost = _estimate_cost_from_tokens(
        model=lookup_model,
        input_tokens=input_tokens,
        cached_tokens=0,
        output_tokens=output_tokens,
        pricing=pricing,
    )
    note = "lower bound: excludes agent context + real output length"
    return input_tokens, output_tokens, cost, note


def estimate_plan(
    plan: PlanSpec,
    run_dir: Path,
    *,
    extra_template_vars: dict[str, str] | None = None,
) -> PlanEstimate:
    """Estimate the cost of running *plan* without running it."""
    extra = extra_template_vars or {}
    cost_history = _load_task_cost_history(plan.name, run_dir)
    report = PlanEstimate(plan_name=plan.name)

    for task in plan.tasks:
        if task.group is not None:
            report.task_estimates.append(
                TaskEstimate(
                    task_id=task.id,
                    kind="group",
                    engine=None,
                    model=None,
                    source="group",
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=None,
                    history_runs=0,
                    note="sub-plan — estimate it directly",
                )
            )
            report.unpriced_tasks += 1
            continue

        if task.engine is None:
            report.task_estimates.append(
                TaskEstimate(
                    task_id=task.id,
                    kind="command",
                    engine=None,
                    model=None,
                    source="shell",
                    input_tokens=0,
                    output_tokens=0,
                    cost_usd=0.0,
                    history_runs=0,
                    note="shell command — no model cost",
                )
            )
            continue

        engine = task.engine
        model = task.model or _default_model_for_engine(plan, engine)
        prior_costs = cost_history.get(task.id, [])

        if prior_costs:
            avg_cost = sum(prior_costs) / len(prior_costs)
            estimate = TaskEstimate(
                task_id=task.id,
                kind="engine",
                engine=engine,
                model=model,
                source="history",
                input_tokens=0,
                output_tokens=0,
                cost_usd=avg_cost,
                history_runs=len(prior_costs),
                note=f"avg of {len(prior_costs)} prior run(s)",
            )
            report.history_tasks += 1
        elif engine in _LOCAL_ENGINES:
            estimate = TaskEstimate(
                task_id=task.id,
                kind="engine",
                engine=engine,
                model=model,
                source="local",
                input_tokens=0,
                output_tokens=0,
                cost_usd=0.0,
                history_runs=0,
                note="local execution — zero cost",
            )
        elif engine in _SUBSCRIPTION_ENGINES:
            estimate = TaskEstimate(
                task_id=task.id,
                kind="engine",
                engine=engine,
                model=model,
                source="subscription",
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                history_runs=0,
                note="subscription-based — no per-token cost",
            )
            report.unpriced_tasks += 1
        else:
            prompt_text = _resolve_prompt_text(task, plan, extra)
            in_tok, out_tok, cost, note = _heuristic_estimate(engine, model, prompt_text)
            estimate = TaskEstimate(
                task_id=task.id,
                kind="engine",
                engine=engine,
                model=model,
                source="heuristic" if cost is not None else "unpriced",
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost_usd=cost,
                history_runs=0,
                note=note,
            )
            if cost is not None:
                report.heuristic_tasks += 1
            else:
                report.unpriced_tasks += 1

        report.task_estimates.append(estimate)
        if estimate.cost_usd:
            report.total_cost_usd += estimate.cost_usd
            if engine:
                report.by_engine[engine] = (
                    report.by_engine.get(engine, 0.0) + estimate.cost_usd
                )

    return report


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_tokens(value: int) -> str:
    if value <= 0:
        return "—"
    if value >= 1000:
        return f"{value / 1000:.1f}k"
    return str(value)


def _fmt_cost(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:.4f}"


def format_estimate(report: PlanEstimate) -> str:
    lines: list[str] = []
    lines.append(f"Cost estimate: {report.plan_name}")
    lines.append("")

    header = f"  {'TASK':<22} {'ENGINE':<8} {'MODEL':<10} {'SOURCE':<12} {'TOK in/out':<14} {'EST COST':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for est in report.task_estimates:
        tok = (
            f"{_fmt_tokens(est.input_tokens)}/{_fmt_tokens(est.output_tokens)}"
            if est.input_tokens
            else "—"
        )
        lines.append(
            f"  {est.task_id[:22]:<22} {(est.engine or '—'):<8} "
            f"{(est.model or '—')[:10]:<10} {est.source:<12} {tok:<14} "
            f"{_fmt_cost(est.cost_usd):>10}"
        )

    lines.append("")
    lines.append(f"  Estimated total: {_fmt_cost(report.total_cost_usd)}")
    if report.by_engine:
        parts = ", ".join(
            f"{eng} {_fmt_cost(cost)}" for eng, cost in sorted(report.by_engine.items())
        )
        lines.append(f"  By engine: {parts}")
    lines.append(
        f"  Sources: {report.history_tasks} history, "
        f"{report.heuristic_tasks} heuristic, {report.unpriced_tasks} unpriced"
    )
    lines.append("")
    lines.append("  Notes:")
    lines.append("  - history = average of actual cost from prior runs (trustworthy)")
    lines.append(
        "  - heuristic = rough lower bound from prompt size; excludes agent context,"
    )
    lines.append("    tool use, and real output length — treat as a floor, not a quote")
    lines.append(
        "  - ollama/llama are zero-cost (local); copilot is subscription-based"
    )
    return "\n".join(lines)


def format_estimate_json(report: PlanEstimate) -> str:
    return json.dumps(report.to_dict(), indent=2)
