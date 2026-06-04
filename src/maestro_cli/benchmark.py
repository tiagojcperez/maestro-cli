from __future__ import annotations

import argparse
import io
import re
import shutil
from contextlib import contextmanager, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import fmean, median
from tempfile import mkdtemp
from time import perf_counter
from typing import Any, Callable, Literal, Sequence, TypeAlias, cast

from . import replan as replan_module
from .cache import compute_task_hash
from .knowledge import build_score_record, store_knowledge, store_score_history
from .loader import load_plan
from .memory import close_all_connections, close_connection
from .mcts import score_record_fitness
from .models import JudgeResult, KnowledgeRecord, PlanRunResult, PlanSpec, ReplanState, TaskResult
from .replan import replan
from .scheduler import run_plan

BenchmarkMetric: TypeAlias = bool | int | float | str
BenchmarkName: TypeAlias = Literal[
    "loader",
    "cache",
    "scheduler",
    "replan_pruning",
    "replan_population",
    "replan_novelty",
    "replan_guidance",
]
_DEFAULT_CASES: tuple[BenchmarkName, ...] = (
    "loader",
    "cache",
    "scheduler",
    "replan_pruning",
    "replan_population",
    "replan_novelty",
    "replan_guidance",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    iterations: int = 5
    warmups: int = 1
    task_count: int = 200
    max_parallel: int = 8


@dataclass(frozen=True)
class BenchmarkResult:
    name: BenchmarkName
    iterations: int
    warmups: int
    samples_ms: tuple[float, ...]
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    metrics: dict[str, BenchmarkMetric] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSample:
    cleanup_path: Path | None = None
    metrics: dict[str, BenchmarkMetric] = field(default_factory=dict)


Runner: TypeAlias = Callable[[], BenchmarkSample | Path | None]


def _build_plan_text(task_count: int, max_parallel: int) -> str:
    lines = [
        "version: 1",
        "name: benchmark-plan",
        f"max_parallel: {max_parallel}",
        "tasks:",
    ]
    for index in range(task_count):
        task_id = f"task-{index:04d}"
        depends_on: list[str] = []
        if index > 0:
            depends_on.append(f"task-{index - 1:04d}")
        if index > 2 and index % 3 == 0:
            depends_on.append(f"task-{index - 3:04d}")

        lines.append(f"  - id: {task_id}")
        lines.append(f"    description: synthetic benchmark task {index}")
        if depends_on:
            lines.append(f"    depends_on: [{', '.join(depends_on)}]")
        lines.append('    command: ["python", "-c", "print(\'ok\')"]')
    return "\n".join(lines) + "\n"


def _write_fixture(base_dir: Path, config: BenchmarkConfig) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    plan_path = base_dir / "benchmark-plan.yaml"
    plan_path.write_text(
        _build_plan_text(config.task_count, config.max_parallel),
        encoding="utf-8",
    )
    return plan_path


def _build_replan_plan_text() -> str:
    return (
        "version: 1\n"
        "name: benchmark-replan\n"
        "max_parallel: 1\n"
        "tasks:\n"
        "  - id: t1\n"
        "    command: echo hello\n"
    )


def _write_replan_fixture(base_dir: Path) -> Path:
    base_dir.mkdir(parents=True, exist_ok=True)
    plan_path = base_dir / "plan.yaml"
    plan_path.write_text(_build_replan_plan_text(), encoding="utf-8")
    return plan_path


def _format_metric_value(value: BenchmarkMetric) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _candidate_number(plan: PlanSpec) -> int | None:
    source_path = plan.source_path
    if source_path is None:
        return None
    match = re.search(r"candidate-(\d+)-", source_path.name)
    if match is None:
        return None
    return int(match.group(1))


def _scenario_candidate_number(plan: PlanSpec, scenario_dir: Path) -> int | None:
    source_path = plan.source_path
    if source_path is None:
        return None
    if source_path.parent == scenario_dir:
        return None
    return _candidate_number(plan) or 1


def _candidate_fitness(
    sample_dir: Path,
    *,
    candidate_yaml: str,
    run_result: PlanRunResult,
) -> float:
    candidate_path = sample_dir / "selected-candidate.yaml"
    candidate_path.write_text(candidate_yaml, encoding="utf-8")
    candidate_plan = load_plan(candidate_path)
    score_record = build_score_record(candidate_plan, run_result)
    return round(score_record_fitness(score_record), 6)


def _memory_db_path(source_dir: Path, *, plan_name: str = "benchmark-replan") -> Path:
    return source_dir / ".maestro-cache" / "memory" / f"{plan_name}.db"


def _close_memory_connections(*source_dirs: Path) -> None:
    for source_dir in source_dirs:
        close_connection(_memory_db_path(source_dir))
    # The replan flow loads candidate plans from unpredictable ``mkdtemp``
    # directories, each of which opens its own memory-store connection.  Those
    # source dirs are not known to the caller, so close every remaining
    # thread-local connection to release Windows file handles before cleanup.
    close_all_connections()


def _make_run_result(
    plan_name: str,
    run_path: Path,
    *,
    run_id: str,
    success: bool,
    task_id: str = "t1",
    duration_sec: float = 5.0,
    total_cost_usd: float = 0.1,
    judge_score: float | None = None,
    message: str = "",
) -> PlanRunResult:
    run_path.mkdir(parents=True, exist_ok=True)
    finished_at = datetime.now(timezone.utc)
    started_at = finished_at - timedelta(seconds=duration_sec)
    task_result = TaskResult(
        task_id=task_id,
        status="success" if success else "failed",
        exit_code=0 if success else 1,
        started_at=started_at,
        finished_at=finished_at,
        duration_sec=duration_sec,
        message=message,
    )
    if judge_score is not None:
        task_result.judge_result = JudgeResult(
            verdict="pass" if success else "fail",
            overall_score=judge_score,
            reasoning="benchmark fixture",
        )
    return PlanRunResult(
        plan_name=plan_name,
        run_id=run_id,
        run_path=run_path,
        started_at=started_at,
        finished_at=finished_at,
        success=success,
        task_results={task_id: task_result},
        total_cost_usd=total_cost_usd,
        total_tokens=128,
    )


def _selected_candidate_summary(state: ReplanState) -> dict[str, Any]:
    attempt = state.attempts[-1]
    assert attempt.selected_candidate_id is not None
    for candidate in attempt.candidate_variants:
        if candidate.get("node_id") == attempt.selected_candidate_id:
            return candidate
    raise AssertionError("selected candidate summary was not recorded")


@contextmanager
def _patched_attributes(
    patches: Sequence[tuple[object, str, Any]],
) -> Any:
    originals: list[tuple[object, str, Any]] = []
    try:
        for target, attribute, replacement in patches:
            originals.append((target, attribute, getattr(target, attribute)))
            setattr(target, attribute, replacement)
        yield
    finally:
        for target, attribute, original in reversed(originals):
            setattr(target, attribute, original)


def _run_replan_with_patches(
    plan_path: Path,
    *,
    max_attempts: int = 2,
    variants: int,
    patches: Sequence[tuple[object, str, Any]],
    exploration_constant: float = 1.41421356237,
    population_strategy: str = "best",
    tournament_size: int = 2,
    elite_count: int = 1,
    diversity_floor: float = 0.25,
) -> ReplanState:
    with _patched_attributes(patches), io.StringIO() as stdout_buffer, redirect_stdout(stdout_buffer):
        return replan(
            plan_path,
            max_attempts=max_attempts,
            auto_approve=True,
            variants=variants,
            debug_prob=0.0,
            selection_policy="ucb1",
            exploration_constant=exploration_constant,
            population_strategy=cast(Literal["best", "tournament"], population_strategy),
            tournament_size=tournament_size,
            elite_count=elite_count,
            diversity_floor=diversity_floor,
        )


def _run_pruning_scenario(sample_dir: Path, *, preload_history: bool) -> dict[str, BenchmarkMetric]:
    plan_path = _write_replan_fixture(sample_dir)
    candidate_dirs = [sample_dir / "candidate-1", sample_dir / "candidate-2"]
    for candidate_dir in candidate_dirs:
        candidate_dir.mkdir(parents=True, exist_ok=True)

    candidate_yamls = (
        (
            "version: 1\n"
            "name: benchmark-replan\n"
            "max_parallel: 1\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    timeout_sec: 30\n"
        ),
        (
            "version: 1\n"
            "name: benchmark-replan\n"
            "max_parallel: 2\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    checkpoint: true\n"
        ),
    )

    if preload_history:
        candidate_path = candidate_dirs[0] / "candidate-1-plan.yaml"
        candidate_path.write_text(candidate_yamls[0], encoding="utf-8")
        candidate_plan = load_plan(candidate_path)
        for idx in range(5):
            score_record = build_score_record(
                candidate_plan,
                _make_run_result(
                    candidate_plan.name,
                    candidate_dirs[0] / "history" / f"run-{idx}",
                    run_id=f"history-{idx}",
                    success=False,
                    total_cost_usd=0.2,
                    message="historical failure",
                ),
            )
            store_score_history(candidate_plan.name, candidate_dirs[0], score_record)

    analysis_calls = {"n": 0}
    run_calls = {"candidate": 0}
    candidate_dir_iter = iter(str(path) for path in candidate_dirs)

    def _fake_analysis(prompt: str, model: str) -> str:
        del prompt, model
        response = candidate_yamls[analysis_calls["n"]]
        analysis_calls["n"] += 1
        return f"```yaml\n{response}```"

    def _fake_mkdtemp(prefix: str) -> str:
        del prefix
        return next(candidate_dir_iter)

    def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
        del args, kwargs
        candidate_number = _scenario_candidate_number(plan, sample_dir)
        if candidate_number is None:
            return _make_run_result(
                plan.name,
                sample_dir / "root-run",
                run_id="root-run",
                success=False,
                total_cost_usd=0.2,
                message="root failure",
            )
        run_calls["candidate"] += 1
        return _make_run_result(
            plan.name,
            sample_dir / "runs" / f"candidate-{candidate_number}",
            run_id=f"candidate-{candidate_number}",
            success=(candidate_number == 2),
            total_cost_usd=0.1,
            message="" if candidate_number == 2 else "candidate failure",
        )

    try:
        state = _run_replan_with_patches(
            plan_path,
            variants=2,
            patches=(
                (replan_module, "_call_analysis_model", _fake_analysis),
                (replan_module, "run_plan", _fake_run_plan),
                (replan_module.tempfile, "mkdtemp", _fake_mkdtemp),  # type: ignore[attr-defined]
            ),
        )

        attempt = state.attempts[-1]
        pruned_candidates = sum(1 for candidate in attempt.candidate_variants if candidate.get("pruned"))
        return {
            "candidate_simulations": run_calls["candidate"],
            "pruned_candidates": pruned_candidates,
            "final_success": state.final_success,
        }
    finally:
        _close_memory_connections(sample_dir, *candidate_dirs)


def _benchmark_replan_pruning(config: BenchmarkConfig, base_dir: Path) -> BenchmarkResult:
    counter = {"n": 0}

    def runner() -> BenchmarkSample:
        sample_dir = base_dir / f"replan-pruning-{counter['n']}"
        counter["n"] += 1
        sample_dir.mkdir(parents=True, exist_ok=True)
        baseline = _run_pruning_scenario(sample_dir / "baseline", preload_history=False)
        pruned = _run_pruning_scenario(sample_dir / "pruned", preload_history=True)
        return BenchmarkSample(
            cleanup_path=sample_dir,
            metrics={
                "baseline_simulations": int(baseline["candidate_simulations"]),
                "pruned_simulations": int(pruned["candidate_simulations"]),
                "saved_simulations": int(baseline["candidate_simulations"]) - int(pruned["candidate_simulations"]),
                "pruned_candidates": int(pruned["pruned_candidates"]),
            },
        )

    return _run_samples(
        "replan_pruning",
        runner,
        iterations=config.iterations,
        warmups=config.warmups,
    )


def _run_population_scenario(
    sample_dir: Path,
    *,
    variants: int,
    population_strategy: str,
) -> dict[str, BenchmarkMetric]:
    plan_path = _write_replan_fixture(sample_dir)
    analysis_calls = {"n": 0}
    candidate_yamls = (
        (
            "version: 1\n"
            "name: benchmark-replan\n"
            "max_parallel: 1\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    timeout_sec: 45\n"
        ),
        (
            "version: 1\n"
            "name: benchmark-replan\n"
            "max_parallel: 3\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    checkpoint: true\n"
            "    max_retries: 2\n"
        ),
        (
            "version: 1\n"
            "name: benchmark-replan\n"
            "max_parallel: 2\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    stdout_tail_lines: 40\n"
        ),
    )
    candidate_scores = {
        1: 0.55,
        2: 0.92,
        3: 0.78,
    }
    run_calls = {"candidate": 0}

    def _fake_analysis(prompt: str, model: str) -> str:
        del prompt, model
        response = candidate_yamls[analysis_calls["n"]]
        analysis_calls["n"] += 1
        return f"```yaml\n{response}```"

    def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
        del args, kwargs
        candidate_number = _scenario_candidate_number(plan, sample_dir)
        if candidate_number is None:
            return _make_run_result(
                plan.name,
                sample_dir / "root-run",
                run_id="root-run",
                success=False,
                total_cost_usd=0.2,
                message="root failure",
            )
        run_calls["candidate"] += 1
        return _make_run_result(
            plan.name,
            sample_dir / "runs" / f"candidate-{candidate_number}",
            run_id=f"candidate-{candidate_number}",
            success=True,
            judge_score=candidate_scores[candidate_number],
            total_cost_usd=0.1,
        )

    try:
        state = _run_replan_with_patches(
            plan_path,
            variants=variants,
            population_strategy=population_strategy,
            patches=(
                (replan_module, "_call_analysis_model", _fake_analysis),
                (replan_module, "run_plan", _fake_run_plan),
            ),
        )
        attempt = state.attempts[-1]
        result = attempt.run_result
        assert result is not None
        if variants == 1:
            selected_run_id = result.run_id
            selected_score = _candidate_fitness(
                sample_dir,
                candidate_yaml=candidate_yamls[0],
                run_result=result,
            )
        else:
            selected = _selected_candidate_summary(state)
            selected_run_id = str(selected["run_id"])
            match = re.search(r"candidate-(\d+)", selected_run_id)
            assert match is not None
            selected_candidate_number = int(match.group(1))
            selected_score = _candidate_fitness(
                sample_dir,
                candidate_yaml=candidate_yamls[selected_candidate_number - 1],
                run_result=result,
            )
        return {
            "selected_score": selected_score,
            "selected_run_id": selected_run_id,
            "candidate_simulations": run_calls["candidate"],
        }
    finally:
        _close_memory_connections(sample_dir)


def _benchmark_replan_population(config: BenchmarkConfig, base_dir: Path) -> BenchmarkResult:
    counter = {"n": 0}

    def runner() -> BenchmarkSample:
        sample_dir = base_dir / f"replan-population-{counter['n']}"
        counter["n"] += 1
        sample_dir.mkdir(parents=True, exist_ok=True)
        single = _run_population_scenario(
            sample_dir / "single",
            variants=1,
            population_strategy="best",
        )
        tournament = _run_population_scenario(
            sample_dir / "tournament",
            variants=3,
            population_strategy="tournament",
        )
        single_score = float(single["selected_score"])
        tournament_score = float(tournament["selected_score"])
        return BenchmarkSample(
            cleanup_path=sample_dir,
            metrics={
                "single_score": single_score,
                "tournament_score": tournament_score,
                "score_gain": round(tournament_score - single_score, 6),
                "single_run_id": str(single["selected_run_id"]),
                "tournament_run_id": str(tournament["selected_run_id"]),
            },
        )

    return _run_samples(
        "replan_population",
        runner,
        iterations=config.iterations,
        warmups=config.warmups,
    )


def _run_novelty_scenario(
    sample_dir: Path,
    *,
    novelty_enabled: bool,
) -> dict[str, BenchmarkMetric]:
    plan_path = _write_replan_fixture(sample_dir)
    analysis_calls = {"n": 0}
    candidate_yamls = (
        (
            "version: 1\n"
            "name: benchmark-replan\n"
            "max_parallel: 1\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    timeout_sec: 35\n"
        ),
        (
            "version: 1\n"
            "name: benchmark-replan\n"
            "max_parallel: 3\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    checkpoint: true\n"
            "    max_retries: 2\n"
            "    stdout_tail_lines: 40\n"
            "    tags: [novel]\n"
        ),
    )
    candidate_scores = {
        1: 0.83,
        2: 0.82,
    }

    def _fake_analysis(prompt: str, model: str) -> str:
        del prompt, model
        response = candidate_yamls[analysis_calls["n"]]
        analysis_calls["n"] += 1
        return f"```yaml\n{response}```"

    def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
        del args, kwargs
        candidate_number = _scenario_candidate_number(plan, sample_dir)
        if candidate_number is None:
            return _make_run_result(
                plan.name,
                sample_dir / "root-run",
                run_id="root-run",
                success=False,
                total_cost_usd=0.2,
                message="root failure",
            )
        return _make_run_result(
            plan.name,
            sample_dir / "runs" / f"candidate-{candidate_number}",
            run_id=f"candidate-{candidate_number}",
            success=True,
            judge_score=candidate_scores[candidate_number],
            total_cost_usd=0.1,
        )

    patches: list[tuple[object, str, Any]] = [
        (replan_module, "_call_analysis_model", _fake_analysis),
        (replan_module, "run_plan", _fake_run_plan),
    ]
    if not novelty_enabled:
        patches.append(
            (
                replan_module,
                "_score_replan_variant_novelty",
                lambda candidate_yaml, baseline_plan_yaml, prior_tree_rows: (0.0, []),
            )
        )

    try:
        state = _run_replan_with_patches(
            plan_path,
            variants=2,
            patches=tuple(patches),
        )
        selected = _selected_candidate_summary(state)
        selected_run_id = str(selected["run_id"])
        match = re.search(r"candidate-(\d+)", selected_run_id)
        assert match is not None
        selected_candidate_number = int(match.group(1))
        result = state.attempts[-1].run_result
        assert result is not None
        attempt = state.attempts[-1]
        novelty_bonus = max(
            float(candidate.get("novelty_bonus", 0.0) or 0.0)
            for candidate in attempt.candidate_variants
        )
        return {
            "selected_score": _candidate_fitness(
                sample_dir,
                candidate_yaml=candidate_yamls[selected_candidate_number - 1],
                run_result=result,
            ),
            "selected_run_id": selected_run_id,
            "max_novelty_bonus": novelty_bonus,
        }
    finally:
        _close_memory_connections(sample_dir)


def _benchmark_replan_novelty(config: BenchmarkConfig, base_dir: Path) -> BenchmarkResult:
    counter = {"n": 0}

    def runner() -> BenchmarkSample:
        sample_dir = base_dir / f"replan-novelty-{counter['n']}"
        counter["n"] += 1
        sample_dir.mkdir(parents=True, exist_ok=True)
        disabled = _run_novelty_scenario(sample_dir / "disabled", novelty_enabled=False)
        enabled = _run_novelty_scenario(sample_dir / "enabled", novelty_enabled=True)
        return BenchmarkSample(
            cleanup_path=sample_dir,
            metrics={
                "without_novelty": str(disabled["selected_run_id"]),
                "with_novelty": str(enabled["selected_run_id"]),
                "selected_changed": str(disabled["selected_run_id"]) != str(enabled["selected_run_id"]),
                "max_novelty_bonus": float(enabled["max_novelty_bonus"]),
                "score_gain": round(
                    float(enabled["selected_score"]) - float(disabled["selected_score"]),
                    6,
                ),
            },
        )

    return _run_samples(
        "replan_novelty",
        runner,
        iterations=config.iterations,
        warmups=config.warmups,
    )


def _seed_guidance_bridge(sample_dir: Path) -> None:
    now = "2026-04-09T00:00:00+00:00"
    store_knowledge(
        "benchmark-replan",
        sample_dir,
        [
            KnowledgeRecord(
                task_id="t1",
                kind="failure_pattern",
                insight="Increase timeout_sec to 120, enable checkpoint, and set stable retries for flaky validation.",
                confidence=0.92,
                occurrences=4,
                first_seen=now,
                last_seen=now,
            ),
            KnowledgeRecord(
                task_id="t1",
                kind="model_pattern",
                insight="Model gpt-5.4-codex succeeds with timeout_sec 120 and checkpoint stable validation.",
                confidence=0.88,
                occurrences=3,
                first_seen=now,
                last_seen=now,
            ),
        ],
    )
    history_path = sample_dir / "historical-guided.yaml"
    history_path.write_text(
        "version: 1\n"
        "name: benchmark-replan\n"
        "max_parallel: 4\n"
        "tasks:\n"
        "  - id: t1\n"
        "    command: echo hello\n"
        "    model: gpt-5.4-codex\n"
        "    timeout_sec: 120\n"
        "    checkpoint: true\n"
        "    max_retries: 2\n"
        "    tags: [stable]\n",
        encoding="utf-8",
    )
    history_plan = load_plan(history_path)
    history_score = build_score_record(
        history_plan,
        _make_run_result(
            history_plan.name,
            sample_dir / "history-guided-run",
            run_id="history-guided",
            success=True,
            duration_sec=8.0,
            total_cost_usd=0.08,
            judge_score=0.94,
        ),
    )
    store_score_history(history_plan.name, sample_dir, history_score)


def _guidance_candidate_yaml(candidate_number: int, *, guided: bool) -> str:
    if candidate_number == 1:
        return (
            "version: 1\n"
            "name: benchmark-replan\n"
            "max_parallel: 1\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    timeout_sec: 45\n"
            "    tags: [local]\n"
        )
    if guided:
        return (
            "version: 1\n"
            "name: benchmark-replan\n"
            "max_parallel: 3\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    model: gpt-5.4-codex\n"
            "    timeout_sec: 120\n"
            "    checkpoint: true\n"
            "    max_retries: 2\n"
            "    tags: [stable]\n"
        )
    return (
        "version: 1\n"
        "name: benchmark-replan\n"
        "max_parallel: 2\n"
        "tasks:\n"
        "  - id: t1\n"
        "    command: echo hello\n"
        "    timeout_sec: 60\n"
        "    tags: [generic]\n"
    )


def _guidance_retry_yaml(candidate_number: int) -> str:
    if candidate_number == 1:
        return (
            "version: 1\n"
            "name: benchmark-replan\n"
            "max_parallel: 1\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            "    timeout_sec: 75\n"
            "    tags: [retry_a]\n"
        )
    return (
        "version: 1\n"
        "name: benchmark-replan\n"
        "max_parallel: 1\n"
        "tasks:\n"
        "  - id: t1\n"
        "    command: echo hello\n"
        "    timeout_sec: 80\n"
        "    tags: [retry_b]\n"
    )


def _guidance_finish_yaml(candidate_number: int, *, guided: bool) -> str:
    if candidate_number == 1:
        return (
            "version: 1\n"
            "name: benchmark-replan\n"
            "max_parallel: 3\n"
            "tasks:\n"
            "  - id: t1\n"
            "    command: echo hello\n"
            f"    timeout_sec: {120 if guided else 90}\n"
            "    verify_command: echo ok\n"
            f"    checkpoint: {'true' if guided else 'false'}\n"
            f"    max_retries: {2 if guided else 1}\n"
            f"    tags: [{'guided_finish' if guided else 'eventual'}]\n"
        )
    return (
        "version: 1\n"
        "name: benchmark-replan\n"
        "max_parallel: 2\n"
        "tasks:\n"
        "  - id: t1\n"
        "    command: echo hello\n"
        "    timeout_sec: 85\n"
        "    tags: [retry_c]\n"
    )


def _run_guidance_scenario(
    sample_dir: Path,
    *,
    bridge_enabled: bool,
) -> dict[str, BenchmarkMetric]:
    plan_path = _write_replan_fixture(sample_dir)
    if bridge_enabled:
        _seed_guidance_bridge(sample_dir)

    prompts: list[str] = []
    run_calls = {"candidate": 0}

    def _fake_analysis(prompt: str, model: str) -> str:
        del model
        prompts.append(prompt)
        match = re.search(r"candidate (\d+) of (\d+)", prompt)
        assert match is not None
        candidate_number = int(match.group(1))
        guided = "HISTORICAL KNOWLEDGE HINTS" in prompt
        if "tags: [retry_a]" in prompt:
            response = _guidance_finish_yaml(candidate_number, guided=False)
        elif "tags: [stable]" in prompt and "checkpoint: true" in prompt:
            response = _guidance_finish_yaml(candidate_number, guided=True)
        elif "tags: [local]" in prompt:
            response = _guidance_retry_yaml(candidate_number)
        else:
            response = _guidance_candidate_yaml(candidate_number, guided=guided)
        return f"```yaml\n{response}```"

    def _fake_run_plan(plan: PlanSpec, *args: object, **kwargs: object) -> PlanRunResult:
        del args, kwargs
        source_name = plan.source_path.name if plan.source_path is not None else ""
        if "candidate-" not in source_name:
            return _make_run_result(
                plan.name,
                sample_dir / "root-run",
                run_id="root-run",
                success=False,
                task_id="t1_root",
                duration_sec=5.0,
                total_cost_usd=0.2,
                message="root failure",
            )

        run_calls["candidate"] += 1
        task = plan.tasks[0]
        timeout = task.timeout_sec or 0
        tags = {tag.lower() for tag in task.tags}
        if timeout == 120 and task.checkpoint and task.verify_command:
            return _make_run_result(
                plan.name,
                sample_dir / "runs" / source_name.replace(".yaml", ""),
                run_id="guided-finish",
                success=True,
                task_id="t1_guided_finish",
                duration_sec=7.0,
                total_cost_usd=0.08,
                judge_score=0.95,
            )
        if timeout == 90 and task.verify_command:
            return _make_run_result(
                plan.name,
                sample_dir / "runs" / source_name.replace(".yaml", ""),
                run_id="delayed-finish",
                success=True,
                task_id="t1_delayed_finish",
                duration_sec=9.0,
                total_cost_usd=0.1,
                judge_score=0.89,
            )
        if timeout == 45 and "local" in tags:
            return _make_run_result(
                plan.name,
                sample_dir / "runs" / source_name.replace(".yaml", ""),
                run_id="candidate1-initial",
                success=False,
                task_id="t1_local",
                duration_sec=12.0,
                total_cost_usd=0.05,
                message="candidate1 failure",
            )
        if timeout == 60 and "generic" in tags:
            return _make_run_result(
                plan.name,
                sample_dir / "runs" / source_name.replace(".yaml", ""),
                run_id="candidate2-generic",
                success=False,
                task_id="t1_generic",
                duration_sec=70.0,
                total_cost_usd=0.14,
                message="candidate2 failure",
            )
        if timeout == 120 and task.checkpoint and "stable" in tags:
            return _make_run_result(
                plan.name,
                sample_dir / "runs" / source_name.replace(".yaml", ""),
                run_id="candidate2-guided",
                success=False,
                task_id="t1_guided",
                duration_sec=70.0,
                total_cost_usd=0.14,
                message="candidate2 failure",
            )
        if timeout == 75 and "retry_a" in tags:
            return _make_run_result(
                plan.name,
                sample_dir / "runs" / source_name.replace(".yaml", ""),
                run_id="candidate1-retry-a",
                success=False,
                task_id="t1_retry_a",
                duration_sec=15.0,
                total_cost_usd=0.06,
                message="retry a failure",
            )
        if timeout == 80 and "retry_b" in tags:
            return _make_run_result(
                plan.name,
                sample_dir / "runs" / source_name.replace(".yaml", ""),
                run_id="candidate1-retry-b",
                success=False,
                task_id="t1_retry_b",
                duration_sec=30.0,
                total_cost_usd=0.09,
                message="retry b failure",
            )
        if timeout == 85 and "retry_c" in tags:
            return _make_run_result(
                plan.name,
                sample_dir / "runs" / source_name.replace(".yaml", ""),
                run_id="candidate-retry-c",
                success=False,
                task_id="t1_retry_c",
                duration_sec=20.0,
                total_cost_usd=0.11,
                message="retry c failure",
            )
        raise AssertionError(f"unexpected plan source: {source_name}")  # pragma: no cover

    try:
        state = _run_replan_with_patches(
            plan_path,
            max_attempts=3,
            variants=2,
            exploration_constant=0.0,
            patches=(
                (replan_module, "_call_analysis_model", _fake_analysis),
                (replan_module, "run_plan", _fake_run_plan),
                (
                    replan_module,
                    "_score_replan_variant_novelty",
                    lambda candidate_yaml, baseline_plan_yaml, prior_tree_rows: (0.0, []),
                ),
            ),
        )
        final_attempt = state.attempts[-1]
        selected = _selected_candidate_summary(state)
        first_attempt = state.attempts[0]
        return {
            "candidate_simulations": run_calls["candidate"],
            "attempts": len(state.attempts),
            "selected_run_id": str(selected["run_id"]),
            "knowledge_bonus": max(
                float(candidate.get("knowledge_bonus", 0.0) or 0.0)
                for candidate in first_attempt.candidate_variants
            ),
            "historical_bonus": max(
                float(candidate.get("historical_fitness_bonus", 0.0) or 0.0)
                for candidate in first_attempt.candidate_variants
            ),
            "final_success": state.final_success,
            "final_selected_node": str(final_attempt.selected_candidate_id or ""),
            "guidance_prompted": any("HISTORICAL KNOWLEDGE HINTS" in prompt for prompt in prompts),
            "first_attempt_selected_node": str(first_attempt.selected_candidate_id or ""),
        }
    finally:
        _close_memory_connections(sample_dir)


def _benchmark_replan_guidance(config: BenchmarkConfig, base_dir: Path) -> BenchmarkResult:
    counter = {"n": 0}

    def runner() -> BenchmarkSample:
        sample_dir = base_dir / f"replan-guidance-{counter['n']}"
        counter["n"] += 1
        sample_dir.mkdir(parents=True, exist_ok=True)
        baseline = _run_guidance_scenario(sample_dir / "baseline", bridge_enabled=False)
        bridged = _run_guidance_scenario(sample_dir / "bridged", bridge_enabled=True)
        return BenchmarkSample(
            cleanup_path=sample_dir,
            metrics={
                "baseline_attempts": int(baseline["attempts"]),
                "bridged_attempts": int(bridged["attempts"]),
                "baseline_simulations": int(baseline["candidate_simulations"]),
                "bridged_simulations": int(bridged["candidate_simulations"]),
                "saved_simulations": int(baseline["candidate_simulations"]) - int(bridged["candidate_simulations"]),
                "knowledge_bonus": float(bridged["knowledge_bonus"]),
                "historical_bonus": float(bridged["historical_bonus"]),
                "guidance_prompted": bool(bridged["guidance_prompted"]),
                "selected_changed": str(baseline["first_attempt_selected_node"]) != str(bridged["first_attempt_selected_node"]),
                "baseline_success": bool(baseline["final_success"]),
                "bridged_success": bool(bridged["final_success"]),
            },
        )

    return _run_samples(
        "replan_guidance",
        runner,
        iterations=config.iterations,
        warmups=config.warmups,
    )


def _run_samples(
    name: BenchmarkName,
    runner: Runner,
    *,
    iterations: int,
    warmups: int,
) -> BenchmarkResult:
    def _coerce_sample(result: BenchmarkSample | Path | None) -> BenchmarkSample:
        if isinstance(result, BenchmarkSample):
            return result
        if isinstance(result, Path):
            return BenchmarkSample(cleanup_path=result)
        return BenchmarkSample()

    for _ in range(warmups):
        sample = _coerce_sample(runner())
        if sample.cleanup_path is not None:
            shutil.rmtree(sample.cleanup_path, ignore_errors=True)

    samples_ms: list[float] = []
    sample_metrics: dict[str, BenchmarkMetric] = {}
    for _ in range(iterations):
        start = perf_counter()
        sample = _coerce_sample(runner())
        elapsed_ms = (perf_counter() - start) * 1000.0
        samples_ms.append(elapsed_ms)
        sample_metrics = dict(sample.metrics)
        if sample.cleanup_path is not None:
            shutil.rmtree(sample.cleanup_path, ignore_errors=True)

    sample_tuple = tuple(samples_ms)
    return BenchmarkResult(
        name=name,
        iterations=iterations,
        warmups=warmups,
        samples_ms=sample_tuple,
        mean_ms=fmean(sample_tuple),
        median_ms=median(sample_tuple),
        min_ms=min(sample_tuple),
        max_ms=max(sample_tuple),
        metrics=sample_metrics,
    )


def _benchmark_loader(plan_path: Path, config: BenchmarkConfig) -> BenchmarkResult:
    def runner() -> BenchmarkSample:
        load_plan(plan_path)
        return BenchmarkSample()

    return _run_samples(
        "loader",
        runner,
        iterations=config.iterations,
        warmups=config.warmups,
    )


def _benchmark_cache(plan_path: Path, config: BenchmarkConfig) -> BenchmarkResult:
    plan = load_plan(plan_path)

    def runner() -> BenchmarkSample:
        upstream_hashes: dict[str, str] = {}
        for task in plan.tasks:
            task_upstream_hashes = {
                dependency_id: upstream_hashes[dependency_id]
                for dependency_id in task.depends_on
            }
            upstream_hashes[task.id] = compute_task_hash(task, plan, task_upstream_hashes)
        return BenchmarkSample()

    return _run_samples(
        "cache",
        runner,
        iterations=config.iterations,
        warmups=config.warmups,
    )


def _benchmark_scheduler(plan_path: Path, config: BenchmarkConfig, base_dir: Path) -> BenchmarkResult:
    plan = load_plan(plan_path)
    runs_dir = base_dir / "runs"

    def runner() -> BenchmarkSample:
        with io.StringIO() as stdout_buffer, redirect_stdout(stdout_buffer):
            result = run_plan(
                plan,
                dry_run=True,
                run_dir_override=str(runs_dir),
                verbosity="quiet",
            )
        return BenchmarkSample(cleanup_path=result.run_path)

    return _run_samples(
        "scheduler",
        runner,
        iterations=config.iterations,
        warmups=config.warmups,
    )


def run_benchmarks(
    config: BenchmarkConfig | None = None,
    *,
    cases: Sequence[BenchmarkName] = _DEFAULT_CASES,
) -> list[BenchmarkResult]:
    benchmark_config = config or BenchmarkConfig()
    selected_cases = tuple(dict.fromkeys(cases))

    # Manage the scratch directory manually instead of via ``TemporaryDirectory``
    # so cleanup is tolerant of leftover handles.  The replan scenarios open
    # SQLite memory-store connections (WAL ``.db``/``-wal``/``-shm`` files); on
    # Windows a residual handle would make a strict ``__exit__`` rmtree raise
    # ``PermissionError [WinError 32]``.  We close every connection first, then
    # rmtree with ``ignore_errors=True``.
    temp_dir = mkdtemp(prefix="maestro-benchmark-")
    base_dir = Path(temp_dir)
    try:
        plan_path = _write_fixture(base_dir, benchmark_config)

        results: list[BenchmarkResult] = []
        for case_name in selected_cases:
            if case_name == "loader":
                results.append(_benchmark_loader(plan_path, benchmark_config))
            elif case_name == "cache":
                results.append(_benchmark_cache(plan_path, benchmark_config))
            elif case_name == "scheduler":
                results.append(_benchmark_scheduler(plan_path, benchmark_config, base_dir))
            elif case_name == "replan_pruning":
                results.append(_benchmark_replan_pruning(benchmark_config, base_dir))
            elif case_name == "replan_population":
                results.append(_benchmark_replan_population(benchmark_config, base_dir))
            elif case_name == "replan_guidance":
                results.append(_benchmark_replan_guidance(benchmark_config, base_dir))
            else:
                results.append(_benchmark_replan_novelty(benchmark_config, base_dir))
        return results
    finally:
        close_all_connections()
        shutil.rmtree(base_dir, ignore_errors=True)


def _format_result(result: BenchmarkResult) -> str:
    text = (
        f"{result.name:<10} "
        f"mean={result.mean_ms:8.2f}ms "
        f"median={result.median_ms:8.2f}ms "
        f"min={result.min_ms:8.2f}ms "
        f"max={result.max_ms:8.2f}ms"
    )
    if result.metrics:
        metric_text = " ".join(
            f"{key}={_format_metric_value(value)}"
            for key, value in sorted(result.metrics.items())
        )
        return f"{text} {metric_text}"
    return text


def _parse_case_names(raw_cases: Sequence[str]) -> tuple[BenchmarkName, ...]:
    if not raw_cases or "all" in raw_cases:
        return _DEFAULT_CASES
    unique_cases = tuple(dict.fromkeys(raw_cases))
    return tuple(case for case in _DEFAULT_CASES if case in unique_cases)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m maestro_cli.benchmark",
        description=(
            "Run deterministic local benchmarks for loader, cache hashing, dry-run "
            "scheduling, and Phase 3 replan/search scenarios."
        ),
    )
    parser.add_argument("--iterations", type=int, default=5, help="Measured runs per case.")
    parser.add_argument("--warmups", type=int, default=1, help="Warm-up runs per case.")
    parser.add_argument(
        "--task-count",
        type=int,
        default=200,
        help="Synthetic task count for generated benchmark plans.",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=8,
        help="max_parallel written into the generated benchmark plan.",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=(
            "all",
            "loader",
            "cache",
            "scheduler",
            "replan_pruning",
            "replan_population",
            "replan_novelty",
            "replan_guidance",
        ),
        help="Benchmark case to run. Repeat to select multiple cases.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.iterations < 1:
        parser.error("--iterations must be >= 1")
    if args.warmups < 0:
        parser.error("--warmups must be >= 0")
    if args.task_count < 1:
        parser.error("--task-count must be >= 1")
    if args.max_parallel < 1:
        parser.error("--max-parallel must be >= 1")

    config = BenchmarkConfig(
        iterations=args.iterations,
        warmups=args.warmups,
        task_count=args.task_count,
        max_parallel=args.max_parallel,
    )
    cases = _parse_case_names(args.case or [])
    results = run_benchmarks(config, cases=cases)

    print(
        f"maestro benchmark: task_count={config.task_count} "
        f"iterations={config.iterations} warmups={config.warmups}"
    )
    for result in results:
        print(_format_result(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
