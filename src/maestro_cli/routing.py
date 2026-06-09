from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from typing import Literal

from .models import ModelRecord, PlanSpec, TaskHistory, TaskSpec

TierName = Literal["low", "medium", "high"]

# Tag signals: tag pattern -> complexity score delta
_HIGH_TIER_TAGS: frozenset[str] = frozenset({
    "security",
    "architecture",
    "critical",
    "audit",
})
_LOW_TIER_TAGS: frozenset[str] = frozenset({
    "trivial",
    "typo",
    "config",
    "docs",
    "rename",
})
_MEDIUM_BOOST_TAGS: frozenset[str] = frozenset({
    "review",
    "qa",
    "refactor",
    "complex",
    "algorithm",
})

# Model tier tables per engine
_MODEL_TIERS: dict[str, dict[TierName, str]] = {
    # Refreshed 2026-06 for GPT-5.5 and Claude Opus 4.8 (Anthropic). Claude
    # `opus` alias auto-resolves to claude-opus-4-8.
    # Codex high tier moved from 5.4 to 5.5; medium kept at 5.4 since 5.5
    # is materially more expensive ($30/M output vs $15) and the medium
    # tier is meant to be a sensible price/quality midpoint.
    "claude": {"low": "haiku", "medium": "sonnet", "high": "opus"},
    "codex": {"low": "5-mini", "medium": "5.4", "high": "5.5"},
    "gemini": {"low": "flash-lite", "medium": "flash", "high": "pro"},
    "copilot": {"low": "haiku", "medium": "sonnet", "high": "opus"},
    "qwen": {"low": "coder-turbo", "medium": "coder", "high": "max"},
    "ollama": {"low": "phi3", "medium": "llama3", "high": "mixtral"},
    "llama": {"low": "llama-3.2-3b", "medium": "llama-3-8b", "high": "codellama-13b"},
}

# Engines whose auto-routed model is refined against detected local hardware.
_LOCAL_ROUTING_ENGINES: frozenset[str] = frozenset({"ollama", "llama"})

# Routing strategy cost adjustments per tier
# cost_optimized: push scores UP → cheaper tiers selected
# quality_first: push scores DOWN → more capable tiers selected
# balanced: no adjustment (default)
_COST_WEIGHTS: dict[str, dict[TierName, float]] = {
    "cost_optimized":  {"low": 0.0,   "medium": 0.15,  "high": 0.3},
    "quality_first":   {"low": -0.15, "medium": 0.0,   "high": -0.15},
    "balanced":        {"low": 0.0,   "medium": 0.0,   "high": 0.0},
}

# Predictive routing constants (T2.3)
_HISTORY_MIN_CONFIDENCE_RUNS = 5   # full confidence at 5+ runs
_HISTORY_MAX_ADJUSTMENT = 0.20     # never shift score more than this
_HISTORY_MAX_MANIFESTS = 20        # cap manifests read for performance
_HISTORY_MIN_MODEL_RUNS = 3        # min runs for per-model signal

# Adaptive Temporal Routing constants (v1.28.0)
_RECENCY_DECAY_FACTOR = 0.85      # each older manifest weighted × 0.85
_TREND_WINDOW = 3                  # look at last N runs for trend detection
_CROSS_TASK_MAX_ADJUSTMENT = 0.10  # max signal from similar tasks
_CROSS_TASK_MIN_SIMILARITY = 0.3   # min similarity score to transfer


def resolve_auto_model(
    task: TaskSpec,
    plan: PlanSpec,
    engine: str,
    *,
    routing_strategy: str | None = None,
    dag_metadata: dict[str, Any] | None = None,
    evidence: dict[str, Any] | None = None,
) -> str:
    engine_tiers = _MODEL_TIERS.get(engine)
    if engine_tiers is None:
        return "auto"
    score = _score_task_complexity(
        task, plan, routing_strategy=routing_strategy, dag_metadata=dag_metadata,
    )
    # Adaptive temporal routing: cross-task signal (v1.28.0)
    if dag_metadata:
        _all_hist = dag_metadata.get("all_histories")
        _task_map_ref = dag_metadata.get("task_map")
        if _all_hist and _task_map_ref:
            score = apply_cross_task_routing(
                task, _task_map_ref, _all_hist, engine, score,
            )
    tier = _tier_from_score(score)
    model = engine_tiers.get(tier, engine_tiers["medium"])

    # Hardware-aware adjustment for local engines (v2.5.3): land on a model the
    # user actually has installed and that fits available VRAM.  The scheduler
    # detects hardware once and passes it via dag_metadata; absent that, this is
    # a no-op and the tier default stands.
    if engine in _LOCAL_ROUTING_ENGINES and dag_metadata:
        hardware = dag_metadata.get("hardware")
        if hardware is not None:
            from .hardware import select_local_model

            adjusted = select_local_model(
                engine, tier, cast("dict[str, str]", engine_tiers), hardware
            )
            if adjusted is not None and adjusted != model:
                if evidence is not None:
                    evidence["hardware_adjusted_from"] = model
                model = adjusted

    if evidence is not None:
        evidence["complexity_score"] = score
        evidence["tier"] = tier
        _hist = (dag_metadata or {}).get("task_history")
        evidence["historical_runs"] = _hist.total_runs if _hist else 0
    return model


def _score_task_complexity(
    task: TaskSpec,
    plan: PlanSpec,
    *,
    routing_strategy: str | None = None,
    dag_metadata: dict[str, Any] | None = None,
) -> float:
    del plan
    score = 0.5
    tags = {tag.strip().lower() for tag in task.tags}

    if tags & _HIGH_TIER_TAGS:
        score = max(score, 0.8)
    if tags & _LOW_TIER_TAGS:
        score = min(score, 0.2)
    if tags & _MEDIUM_BOOST_TAGS:
        score += 0.15

    prompt_length = len(task.prompt or "")
    if prompt_length > 2000:
        score += 0.15
    elif prompt_length > 1000:
        score += 0.10
    elif prompt_length > 500:
        score += 0.05
    elif prompt_length < 100:
        score -= 0.10

    if len(task.depends_on) > 3:
        score += 0.10
    if len(task.context_from) > 2:
        score += 0.10
    if task.context_mode == "recursive":
        score += 0.15
    if task.judge is not None:
        score += 0.10

    # DAG structural signals (only when metadata provided by scheduler)
    if dag_metadata:
        fan_out = dag_metadata.get("fan_out", 0)
        depth = dag_metadata.get("depth", 0)
        upstream_failure_rate = dag_metadata.get("upstream_failure_rate", 0.0)
        if fan_out > 3:
            score += 0.10
        if depth > 4:
            score += 0.05
        if upstream_failure_rate > 0.3:
            score += 0.15

    # Routing strategy cost adjustment
    strategy = routing_strategy or "balanced"
    tier = _tier_from_score(score)
    score += _COST_WEIGHTS.get(strategy, _COST_WEIGHTS["balanced"]).get(tier, 0.0)

    # Historical performance signal (T2.3 — predictive routing)
    if dag_metadata:
        _task_hist = dag_metadata.get("task_history")
        if _task_hist is not None:
            score = _apply_historical_signal(score, _task_hist, task.engine or "")

    return max(0.0, min(1.0, score))


def _tier_from_score(score: float) -> TierName:
    if score < 0.3:
        return "low"
    if score > 0.7:
        return "high"
    return "medium"


# ---------------------------------------------------------------------------
# Predictive routing — historical performance (T2.3)
# ---------------------------------------------------------------------------

def load_task_histories(
    plan_name: str,
    run_dir: Path,
    min_runs: int = 3,
) -> dict[str, TaskHistory]:
    """Load per-task model performance from prior run manifests.

    Scans *run_dir* for directories matching ``*_{plan_name}`` and reads
    ``run_manifest.json`` from each.  Only tasks with ``auto_routed_model``
    set are included (explicit-model tasks are skipped).

    Returns a mapping of ``task_id`` → ``TaskHistory``.  Empty dict when
    fewer than *min_runs* matching manifests are found.
    """
    if not run_dir.exists() or not run_dir.is_dir():
        return {}

    loaded: list[tuple[str, dict[str, Any]]] = []
    for candidate in run_dir.glob(f"*_{plan_name}"):
        if not candidate.is_dir():
            continue
        manifest_path = candidate / "run_manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            loaded.append((candidate.name, payload))

    if len(loaded) < min_runs:
        return {}

    # Take most recent manifests (directory name starts with timestamp)
    loaded.sort(key=lambda item: item[0])
    manifests = [m for _, m in loaded[-_HISTORY_MAX_MANIFESTS:]]

    # Aggregate: (task_id, model) → accumulator
    accum: dict[str, dict[str, _ModelAccum]] = {}

    for manifest_idx, manifest in enumerate(manifests):
        task_results = manifest.get("task_results")
        if not isinstance(task_results, dict):
            continue
        for task_id, result in task_results.items():
            if not isinstance(result, dict):
                continue
            model = result.get("auto_routed_model")
            if not model or not isinstance(model, str):
                continue

            if task_id not in accum:
                accum[task_id] = {}
            if model not in accum[task_id]:
                accum[task_id][model] = _ModelAccum()

            acc = accum[task_id][model]
            acc.runs += 1
            status = result.get("status", "")
            if status == "success":
                acc.successes += 1
            elif status == "failed":
                acc.failures += 1
            exit_code = result.get("exit_code")
            if exit_code == 124:
                acc.timeouts += 1
            acc.recent_outcomes.append(status)
            duration = result.get("duration_sec")
            if isinstance(duration, (int, float)):
                acc.total_duration += float(duration)
            cost = result.get("cost_usd")
            if isinstance(cost, (int, float)):
                acc.total_cost += float(cost)
                acc.cost_count += 1

    # Build TaskHistory objects
    histories: dict[str, TaskHistory] = {}
    for task_id, model_accums in accum.items():
        total_runs = sum(a.runs for a in model_accums.values())
        records: dict[str, ModelRecord] = {}
        for model, acc in model_accums.items():
            records[model] = ModelRecord(
                model=model,
                runs=acc.runs,
                successes=acc.successes,
                failures=acc.failures,
                timeouts=acc.timeouts,
                avg_duration_sec=acc.total_duration / acc.runs if acc.runs else 0.0,
                avg_cost_usd=acc.total_cost / acc.cost_count if acc.cost_count else None,
                recent_outcomes=list(acc.recent_outcomes),
            )
        histories[task_id] = TaskHistory(
            task_id=task_id,
            total_runs=total_runs,
            records=records,
        )

    return histories


class _ModelAccum:
    """Mutable accumulator for building ModelRecord."""
    __slots__ = ("runs", "successes", "failures", "timeouts",
                 "total_duration", "total_cost", "cost_count",
                 "recent_outcomes")

    def __init__(self) -> None:
        self.runs = 0
        self.successes = 0
        self.failures = 0
        self.timeouts = 0
        self.total_duration = 0.0
        self.total_cost = 0.0
        self.cost_count = 0
        self.recent_outcomes: list[str] = []  # ordered newest-first


def _apply_historical_signal(
    score: float,
    history: TaskHistory,
    engine: str,
) -> float:
    """Adjust complexity score using historical model performance.

    Confidence scales linearly with the number of runs up to
    ``_HISTORY_MIN_CONFIDENCE_RUNS``.  The total adjustment is clamped to
    ±\\ ``_HISTORY_MAX_ADJUSTMENT``.
    """
    if history.total_runs == 0:
        return score

    engine_tiers = _MODEL_TIERS.get(engine)
    if engine_tiers is None:
        return score

    confidence = min(history.total_runs / _HISTORY_MIN_CONFIDENCE_RUNS, 1.0)
    adjustment = 0.0

    low_model = engine_tiers["low"]
    medium_model = engine_tiers["medium"]

    low_rec = history.records.get(low_model)
    medium_rec = history.records.get(medium_model)

    # Rule 1: cheap model succeeds 100% (≥3 runs) → push cheaper
    if low_rec and low_rec.runs >= _HISTORY_MIN_MODEL_RUNS:
        if low_rec.successes == low_rec.runs:
            adjustment -= 0.15
        # Rule 2: cheap model fails ≥50% → push stronger
        elif low_rec.failures / low_rec.runs >= 0.5:
            adjustment += 0.15

    # Rule 3: medium model 100% success, no cheap data → modest push cheaper
    if low_rec is None and medium_rec and medium_rec.runs >= _HISTORY_MIN_MODEL_RUNS:
        if medium_rec.successes == medium_rec.runs:
            adjustment -= 0.10

    # Rule 4: any model with ≥40% timeout rate → push stronger
    for rec in history.records.values():
        if rec.runs >= 2 and rec.timeouts / rec.runs >= 0.4:
            adjustment += 0.10
            break

    # Rule 5 (ATR): trend detection — degrading model → push stronger
    for rec in history.records.values():
        if rec.recent_outcomes and len(rec.recent_outcomes) >= _TREND_WINDOW * 2:
            trend = _detect_trend(rec.recent_outcomes)
            if trend == "degrading":
                adjustment += 0.10
            elif trend == "improving" and rec.model == low_model:
                adjustment -= 0.05

    # Scale by confidence and clamp
    adjustment *= confidence
    adjustment = max(-_HISTORY_MAX_ADJUSTMENT, min(_HISTORY_MAX_ADJUSTMENT, adjustment))

    return max(0.0, min(1.0, score + adjustment))


# ---------------------------------------------------------------------------
# Adaptive Temporal Routing — trend detection + cross-task affinity (v1.28.0)
# ---------------------------------------------------------------------------


def _detect_trend(recent_outcomes: list[str], window: int = _TREND_WINDOW) -> str:
    """Detect performance trend from recent outcomes.

    Returns ``"improving"``, ``"degrading"``, or ``"stable"``.
    """
    if len(recent_outcomes) < window * 2:
        return "stable"
    # Compare last `window` outcomes vs previous `window`
    recent = recent_outcomes[-window:]
    previous = recent_outcomes[-window * 2:-window]

    recent_success_rate = sum(1 for o in recent if o == "success") / len(recent)
    previous_success_rate = sum(1 for o in previous if o == "success") / len(previous)

    delta = recent_success_rate - previous_success_rate
    if delta > 0.3:
        return "improving"
    if delta < -0.3:
        return "degrading"
    return "stable"


def _compute_task_similarity(
    task_a: TaskSpec,
    task_b: TaskSpec,
) -> float:
    """Compute similarity score (0.0-1.0) between two tasks for routing transfer.

    Uses tag overlap, engine match, and structural features.
    """
    score = 0.0
    weights_total = 0.0

    # Engine match (weight: 3)
    weights_total += 3.0
    if task_a.engine == task_b.engine:
        score += 3.0

    # Tag overlap (weight: 2)
    tags_a = {t.lower() for t in task_a.tags}
    tags_b = {t.lower() for t in task_b.tags}
    weights_total += 2.0
    if tags_a and tags_b:
        overlap = len(tags_a & tags_b)
        union = len(tags_a | tags_b)
        score += 2.0 * (overlap / union) if union > 0 else 0.0

    # Judge presence (weight: 1)
    weights_total += 1.0
    if (task_a.judge is not None) == (task_b.judge is not None):
        score += 1.0

    # Context mode match (weight: 1)
    weights_total += 1.0
    if task_a.context_mode == task_b.context_mode:
        score += 1.0

    # Dependency count similarity (weight: 0.5)
    weights_total += 0.5
    dep_diff = abs(len(task_a.depends_on) - len(task_b.depends_on))
    if dep_diff <= 1:
        score += 0.5
    elif dep_diff <= 3:
        score += 0.25

    return score / weights_total if weights_total > 0 else 0.0


def apply_cross_task_routing(
    task: TaskSpec,
    task_map: dict[str, TaskSpec],
    histories: dict[str, TaskHistory],
    engine: str,
    base_score: float,
) -> float:
    """Adjust routing score using similar tasks' historical performance.

    Finds tasks with similarity ≥ ``_CROSS_TASK_MIN_SIMILARITY``, then
    applies their model history as a weak signal (capped at ±0.10).
    """
    engine_tiers = _MODEL_TIERS.get(engine)
    if engine_tiers is None:
        return base_score

    low_model = engine_tiers["low"]

    weighted_signal = 0.0
    total_weight = 0.0

    for other_id, other_task in task_map.items():
        if other_id == task.id:
            continue
        other_hist = histories.get(other_id)
        if other_hist is None or other_hist.total_runs < _HISTORY_MIN_MODEL_RUNS:
            continue

        similarity = _compute_task_similarity(task, other_task)
        if similarity < _CROSS_TASK_MIN_SIMILARITY:
            continue

        # Check what tier worked for the similar task
        low_rec = other_hist.records.get(low_model)
        if low_rec and low_rec.runs >= 2:
            if low_rec.successes / low_rec.runs >= 0.8:
                weighted_signal -= 0.10 * similarity
            elif low_rec.failures / low_rec.runs >= 0.5:
                weighted_signal += 0.10 * similarity
            total_weight += similarity

    if total_weight <= 0:
        return base_score

    adjustment = weighted_signal / total_weight
    adjustment = max(
        -_CROSS_TASK_MAX_ADJUSTMENT,
        min(_CROSS_TASK_MAX_ADJUSTMENT, adjustment),
    )
    return max(0.0, min(1.0, base_score + adjustment))
