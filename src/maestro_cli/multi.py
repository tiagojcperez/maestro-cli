from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from .errors import PlanValidationError, TaskExecutionError
from .loader import load_plan
from .models import (
    ExecutionProfile,
    MultiPlanResult,
    OutputMode,
    PlanRunResult,
    PlanSpec,
    TaskResult,
    TaskStatus,
    Verbosity,
)
from .scheduler import run_plan
from .utils import resolve_path, sanitize_dirname


def run_multi_plan(
    plan_paths: list[str],
    *,
    parallel: bool = False,
    event_callback: Callable[[str, dict[str, object]], None] | None = None,
    max_cost_usd: float | None = None,
    execution_profile: ExecutionProfile = "plan",
    dry_run: bool = False,
    verbosity: Verbosity = "normal",
    output_mode: OutputMode = "text",
    cache_dir: Path | None = None,
    auto_approve: bool = False,
) -> MultiPlanResult:
    """Run multiple plans sequentially or in parallel with optional shared budget."""
    started_at = datetime.now(UTC)
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(".maestro-runs") / f"{timestamp}_multi"
    output_dir.mkdir(parents=True, exist_ok=True)

    loaded_plans: list[PlanSpec] = []
    loaded_plan_paths: list[str] = []
    preload_results: list[PlanRunResult] = []

    for raw_path in plan_paths:
        resolved_path = resolve_path(Path.cwd(), raw_path)
        if resolved_path is None:
            message = f"invalid plan path: {raw_path}"
            print(f"[maestro] {message}")
            preload_results.append(
                _new_plan_result(
                    plan_name=Path(raw_path).stem or raw_path,
                    run_path=output_dir,
                    success=False,
                    message=message,
                    status="failed",
                )
            )
            continue

        try:
            plan = load_plan(resolved_path)
            loaded_plans.append(plan)
            loaded_plan_paths.append(str(resolved_path))
        except (PlanValidationError, TaskExecutionError, OSError, ValueError) as exc:
            print(f"[maestro] failed to load plan '{raw_path}': {exc}")
            preload_results.append(
                _new_plan_result(
                    plan_name=resolved_path.stem,
                    run_path=output_dir,
                    success=False,
                    message=f"plan load failed: {exc}",
                    status="failed",
                )
            )

    run_result = MultiPlanResult(plan_results=[])
    if loaded_plans:
        if parallel:
            run_result = _run_parallel(
                loaded_plans,
                loaded_plan_paths,
                event_callback=event_callback,
                max_cost_usd=max_cost_usd,
                execution_profile=execution_profile,
                dry_run=dry_run,
                verbosity=verbosity,
                output_mode=output_mode,
                cache_dir=cache_dir,
                auto_approve=auto_approve,
            )
        else:
            run_result = _run_sequential(
                loaded_plans,
                loaded_plan_paths,
                event_callback=event_callback,
                max_cost_usd=max_cost_usd,
                execution_profile=execution_profile,
                dry_run=dry_run,
                verbosity=verbosity,
                output_mode=output_mode,
                cache_dir=cache_dir,
                auto_approve=auto_approve,
            )

    all_results = preload_results + run_result.plan_results
    final_result = _aggregate_results(all_results)
    final_result.started_at = started_at
    final_result.finished_at = datetime.now(UTC)
    final_result.budget_exceeded = run_result.budget_exceeded

    if max_cost_usd is not None and final_result.total_cost_usd is not None and final_result.total_cost_usd > max_cost_usd:
        final_result.budget_exceeded = True

    try:
        _write_multi_summary(final_result, output_dir)
        print(f"[maestro] multi summary: {output_dir / 'summary.md'}")
    except OSError as exc:
        print(f"[maestro] failed to write multi summary: {exc}")

    return final_result


def _run_sequential(
    plans: list[PlanSpec],
    plan_paths: list[str],
    *,
    event_callback: Callable[[str, dict[str, object]], None] | None = None,
    max_cost_usd: float | None = None,
    execution_profile: ExecutionProfile = "plan",
    dry_run: bool = False,
    verbosity: Verbosity = "normal",
    output_mode: OutputMode = "text",
    cache_dir: Path | None = None,
    auto_approve: bool = False,
) -> MultiPlanResult:
    """Run plans one at a time and stop dispatch when shared budget is exhausted."""
    started_at = datetime.now(UTC)
    remaining_budget = max_cost_usd
    budget_exceeded = False
    results: list[PlanRunResult] = []

    for idx, plan in enumerate(plans):
        plan_path = plan_paths[idx]

        if remaining_budget is not None and remaining_budget <= 0:
            print("[maestro] budget exceeded, skipping remaining plans")
            for skipped_idx in range(idx, len(plans)):
                skipped_plan = plans[skipped_idx]
                skipped_path = plan_paths[skipped_idx]
                results.append(
                    _new_plan_result(
                        plan_name=skipped_plan.name,
                        run_path=Path(skipped_path).parent,
                        success=False,
                        message="skipped due to shared budget exceeded",
                        status="skipped",
                    )
                )
            budget_exceeded = True
            break

        try:
            result = run_plan(
                plan,
                event_callback=event_callback,
                dry_run=dry_run,
                execution_profile=execution_profile,
                verbosity=verbosity,
                output_mode=output_mode,
                cache_dir=cache_dir,
                auto_approve=auto_approve,
            )
        except (TaskExecutionError, PlanValidationError, OSError, ValueError) as exc:
            result = _new_plan_result(
                plan_name=plan.name,
                run_path=Path(plan_path).parent,
                success=False,
                message=f"plan run failed: {exc}",
                status="failed",
            )

        results.append(result)

        if remaining_budget is not None:
            remaining_budget -= result.total_cost_usd or 0.0
            if remaining_budget <= 0:
                budget_exceeded = True

    aggregated = _aggregate_results(results)
    aggregated.started_at = started_at
    aggregated.finished_at = datetime.now(UTC)
    aggregated.budget_exceeded = budget_exceeded
    return aggregated


def _run_parallel(
    plans: list[PlanSpec],
    plan_paths: list[str],
    *,
    event_callback: Callable[[str, dict[str, object]], None] | None = None,
    max_cost_usd: float | None = None,
    execution_profile: ExecutionProfile = "plan",
    dry_run: bool = False,
    verbosity: Verbosity = "normal",
    output_mode: OutputMode = "text",
    cache_dir: Path | None = None,
    auto_approve: bool = False,
) -> MultiPlanResult:
    """Run plans concurrently and apply atomic shared-budget deduction per completed plan."""
    started_at = datetime.now(UTC)
    lock = threading.Lock()
    remaining_budget = max_cost_usd
    budget_exceeded = False
    warned_budget_exceeded = False

    indexed_results: dict[int, PlanRunResult] = {}

    def _worker(index: int, plan: PlanSpec, plan_path: str) -> tuple[int, PlanRunResult]:
        try:
            result = run_plan(
                plan,
                event_callback=event_callback,
                dry_run=dry_run,
                execution_profile=execution_profile,
                verbosity=verbosity,
                output_mode=output_mode,
                cache_dir=cache_dir,
                auto_approve=auto_approve,
            )
        except (TaskExecutionError, PlanValidationError, OSError, ValueError) as exc:
            result = _new_plan_result(
                plan_name=plan.name,
                run_path=Path(plan_path).parent,
                success=False,
                message=f"plan run failed: {exc}",
                status="failed",
            )

        nonlocal remaining_budget, budget_exceeded
        with lock:
            if remaining_budget is not None:
                remaining_budget -= result.total_cost_usd or 0.0
                if remaining_budget <= 0:
                    budget_exceeded = True

        return index, result

    with ThreadPoolExecutor(max_workers=max(1, len(plans))) as executor:
        futures = [
            executor.submit(_worker, idx, plan, plan_paths[idx])
            for idx, plan in enumerate(plans)
        ]

        for future in as_completed(futures):
            index, result = future.result()
            indexed_results[index] = result

            with lock:
                if budget_exceeded and not warned_budget_exceeded:
                    print("[maestro] budget exceeded, skipping remaining plans")
                    warned_budget_exceeded = True

    ordered_results = [indexed_results[idx] for idx in range(len(plans))]
    aggregated = _aggregate_results(ordered_results)
    aggregated.started_at = started_at
    aggregated.finished_at = datetime.now(UTC)
    aggregated.budget_exceeded = budget_exceeded
    return aggregated


def _aggregate_results(results: list[PlanRunResult]) -> MultiPlanResult:
    """Aggregate plan run results into a single multi-plan result."""
    total_cost_usd: float | None = None
    total_tokens: int | None = None

    for result in results:
        if result.total_cost_usd is not None:
            total_cost_usd = (total_cost_usd or 0.0) + result.total_cost_usd
        if result.total_tokens is not None:
            total_tokens = (total_tokens or 0) + result.total_tokens

    success = all(result.success for result in results)
    budget_exceeded = any(result.budget_exceeded for result in results)

    if results:
        started_at = min(result.started_at for result in results)
        finished_at = max(result.finished_at for result in results)
    else:
        started_at = datetime.now(UTC)
        finished_at = started_at

    return MultiPlanResult(
        plan_results=results,
        total_cost_usd=total_cost_usd,
        total_tokens=total_tokens,
        budget_exceeded=budget_exceeded,
        success=success,
        started_at=started_at,
        finished_at=finished_at,
    )


def _write_multi_summary(result: MultiPlanResult, output_dir: Path) -> None:
    """Write markdown summary for a multi-plan run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.md"

    total_duration_sec = (result.finished_at - result.started_at).total_seconds()

    lines: list[str] = []
    lines.append("# Multi-Plan Run Summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|-------|-------|")
    lines.append(f"| Plans | {len(result.plan_results)} |")
    lines.append(f"| Success | {result.success} |")
    lines.append(f"| Budget Exceeded | {result.budget_exceeded} |")
    lines.append(f"| Started | {result.started_at.isoformat()} |")
    lines.append(f"| Finished | {result.finished_at.isoformat()} |")
    lines.append(f"| Duration (sec) | {total_duration_sec:.1f} |")

    total_cost_str = f"${result.total_cost_usd:.2f}" if result.total_cost_usd is not None else "---"
    total_tokens_str = f"{result.total_tokens:,}" if result.total_tokens is not None else "---"
    lines.append(f"| Total Cost | {total_cost_str} |")
    lines.append(f"| Total Tokens | {total_tokens_str} |")
    lines.append("")

    lines.append("## Plans")
    lines.append("")
    lines.append("| Plan | Status | Cost | Duration (sec) | Tokens |")
    lines.append("|------|--------|------|----------------|--------|")

    for plan_result in result.plan_results:
        duration_sec = (plan_result.finished_at - plan_result.started_at).total_seconds()
        status = "success" if plan_result.success else "failed"
        if plan_result.budget_exceeded:
            status = f"{status} (budget exceeded)"

        cost_cell = f"${plan_result.total_cost_usd:.2f}" if plan_result.total_cost_usd is not None else "---"
        tokens_cell = f"{plan_result.total_tokens:,}" if plan_result.total_tokens is not None else "---"

        lines.append(
            f"| {plan_result.plan_name} | {status} | {cost_cell} | {duration_sec:.1f} | {tokens_cell} |"
        )

    lines.append(
        f"| **TOTAL** | **{'success' if result.success else 'failed'}** | **{total_cost_str}** | **{total_duration_sec:.1f}** | **{total_tokens_str}** |"
    )
    lines.append("")

    summary_path.write_text("\n".join(lines), encoding="utf-8")


def _new_plan_result(
    *,
    plan_name: str,
    run_path: Path,
    success: bool,
    message: str,
    status: TaskStatus,
) -> PlanRunResult:
    """Create a synthetic PlanRunResult for load/run failures and skips."""
    now = datetime.now(UTC)
    safe_name = sanitize_dirname(plan_name)
    task_result = TaskResult(task_id=f"{safe_name}:multi", status=status, message=message)

    return PlanRunResult(
        plan_name=plan_name,
        run_id=f"multi-{safe_name}-{now.strftime('%Y%m%d%H%M%S')}",
        run_path=run_path,
        started_at=now,
        finished_at=now,
        success=success,
        task_results={task_result.task_id: task_result},
        total_cost_usd=None,
        total_tokens=None,
        budget_exceeded=False,
    )
