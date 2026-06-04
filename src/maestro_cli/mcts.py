from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .cache import compute_plan_hash
from .knowledge import (
    build_score_record,
    get_historical_pruning_decision,
    load_score_history,
)
from .models import (
    ExecutionProfile,
    HistoricalPruningDecision,
    MctsSelectionPolicy,
    OutputMode,
    PlanRunResult,
    PlanSpec,
    ScoreRecord,
    VariantType,
    Verbosity,
    WorkflowVariant,
)
from .scheduler import run_plan

_TREE_FILENAME = "tree.jsonl"
DEFAULT_EXPLORATION_CONSTANT = 1.41421356237
_VARIANT_PRIOR_KEYS = ("knowledge_prior", "novelty_prior", "historical_fitness_prior")


def classify_variant_type(parent: WorkflowVariant | None) -> VariantType:
    """Classify the next child expansion using the roadmap trichotomy."""
    if parent is None:
        return "draft"
    if not parent.is_valid:
        return "debug"
    return "improve"


def create_workflow_variant(
    plan_spec: PlanSpec,
    *,
    parent: WorkflowVariant | None = None,
    run_result: PlanRunResult | None = None,
    score: float = 0.0,
    is_valid: bool | None = None,
    mutation_desc: str = "",
    node_id: str | None = None,
    plan_hash: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> WorkflowVariant:
    """Create a workflow-search node and attach it to the parent tree."""
    result_plan_hash = run_result.plan_hash if run_result is not None else None
    resolved_plan_hash = plan_hash or result_plan_hash or compute_plan_hash(plan_spec)
    resolved_node_id = node_id or _variant_node_id(
        resolved_plan_hash,
        parent=parent,
        mutation_desc=mutation_desc,
    )
    variant = WorkflowVariant(
        node_id=resolved_node_id,
        plan_spec=plan_spec,
        run_result=run_result,
        score=score,
        is_valid=bool(run_result.success) if is_valid is None and run_result is not None else bool(is_valid),
        parent=parent,
        variant_type=classify_variant_type(parent),
        mutation_desc=mutation_desc,
        plan_hash=resolved_plan_hash,
        metadata=dict(metadata or {}),
    )
    if parent is not None:
        parent.children.append(variant)
    return variant


def _variant_total_prior(variant: WorkflowVariant) -> float:
    return sum(float(variant.metadata.get(key, 0.0) or 0.0) for key in _VARIANT_PRIOR_KEYS)


def _apply_variant_prior(
    variant: WorkflowVariant,
    *,
    metadata_key: str,
    signal_key: str,
    bonus: float,
    signals: list[dict[str, Any]] | None = None,
    max_abs_bonus: float,
) -> float:
    bounded = max(-max_abs_bonus, min(max_abs_bonus, float(bonus)))
    variant.metadata[metadata_key] = round(bounded, 6)
    if signals:
        variant.metadata[signal_key] = signals
    if variant.run_result is None:
        variant.score = max(variant.score, max(0.0, min(1.0, _variant_total_prior(variant))))
    return bounded


def apply_variant_knowledge_prior(
    variant: WorkflowVariant,
    *,
    bonus: float,
    signals: list[dict[str, Any]] | None = None,
    max_abs_bonus: float = 0.1,
) -> float:
    """Attach a small knowledge-derived prior to a variant.

    The prior is persisted in variant metadata and blended into the aggregate
    score during backpropagation, so knowledge can bias both leaf selection
    and post-simulation ranking without overpowering observed execution data.
    """
    return _apply_variant_prior(
        variant,
        metadata_key="knowledge_prior",
        signal_key="knowledge_signals",
        bonus=bonus,
        signals=signals,
        max_abs_bonus=max_abs_bonus,
    )


def apply_variant_novelty_prior(
    variant: WorkflowVariant,
    *,
    bonus: float,
    signals: list[dict[str, Any]] | None = None,
    max_abs_bonus: float = 0.08,
) -> float:
    """Attach a bounded novelty prior derived from variant diversity signals."""
    return _apply_variant_prior(
        variant,
        metadata_key="novelty_prior",
        signal_key="novelty_signals",
        bonus=bonus,
        signals=signals,
        max_abs_bonus=max_abs_bonus,
    )


def apply_variant_historical_fitness_prior(
    variant: WorkflowVariant,
    *,
    bonus: float,
    signals: list[dict[str, Any]] | None = None,
    max_abs_bonus: float = 0.08,
) -> float:
    """Attach a bounded prior bootstrapped from similar historical score records."""
    return _apply_variant_prior(
        variant,
        metadata_key="historical_fitness_prior",
        signal_key="historical_fitness_signals",
        bonus=bonus,
        signals=signals,
        max_abs_bonus=max_abs_bonus,
    )


def iter_variants(root: WorkflowVariant) -> list[WorkflowVariant]:
    """Return the full tree rooted at *root* in depth-first order."""
    ordered: list[WorkflowVariant] = []
    stack = [root]
    seen: set[str] = set()

    while stack:
        current = stack.pop()
        if current.node_id in seen:
            continue
        seen.add(current.node_id)
        ordered.append(current)
        stack.extend(reversed(current.children))

    return ordered


def select_expansion_parent(
    root: WorkflowVariant,
    *,
    selection_policy: MctsSelectionPolicy = "debug_prob",
    debug_prob: float = 0.5,
    exploration_constant: float = DEFAULT_EXPLORATION_CONSTANT,
    rng: random.Random | Any | None = None,
) -> WorkflowVariant | None:
    """Pick the next leaf to expand using the configured selection strategy."""
    leaves = [
        node
        for node in iter_variants(root)
        if not node.children and not node.pruned
    ]
    return select_variant_from_pool(
        leaves,
        selection_policy=selection_policy,
        debug_prob=debug_prob,
        exploration_constant=exploration_constant,
        rng=rng,
    )


def select_variant_from_pool(
    candidates: list[WorkflowVariant],
    *,
    selection_policy: MctsSelectionPolicy = "debug_prob",
    debug_prob: float = 0.5,
    exploration_constant: float = DEFAULT_EXPLORATION_CONSTANT,
    rng: random.Random | Any | None = None,
) -> WorkflowVariant | None:
    """Pick one variant from a pre-selected candidate pool."""
    if selection_policy not in {"debug_prob", "ucb1"}:
        raise ValueError("selection_policy must be 'debug_prob' or 'ucb1'")
    if debug_prob < 0.0 or debug_prob > 1.0:
        raise ValueError("debug_prob must be between 0.0 and 1.0")
    if exploration_constant < 0.0:
        raise ValueError("exploration_constant must be >= 0.0")

    chooser = rng or random.Random()
    if not candidates:
        return None

    unexecuted = [
        node
        for node in candidates
        if node.run_result is None and "score_record" not in node.metadata
    ]
    if unexecuted:
        return max(
            unexecuted,
            key=lambda node: (node.score, -_variant_depth(node), node.node_id),
        )

    if selection_policy == "ucb1":
        return max(
            candidates,
            key=lambda node: (
                _ucb1_score(node, exploration_constant=exploration_constant),
                node.score,
                -_variant_depth(node),
                node.node_id,
            ),
        )

    invalid = [node for node in candidates if not node.is_valid]
    valid = [node for node in candidates if node.is_valid]

    if invalid and (not valid or float(chooser.random()) < debug_prob):
        return chooser.choice(invalid)
    if valid:
        return max(valid, key=lambda node: (node.score, -_variant_depth(node), node.node_id))
    if invalid:  # pragma: no cover
        return chooser.choice(invalid)  # pragma: no cover
    return candidates[0]  # pragma: no cover


def simulate_variant(
    variant: WorkflowVariant,
    *,
    runner: Callable[..., PlanRunResult] | None = None,
    event_callback: Any = None,
    dry_run: bool = False,
    execution_profile: ExecutionProfile = "plan",
    verbosity: Verbosity = "normal",
    output_mode: OutputMode = "text",
    cache_dir: str | Path | None = None,
    auto_approve: bool = False,
    extra_template_vars: dict[str, str] | None = None,
) -> PlanRunResult:
    """Execute the candidate plan through the normal scheduler."""
    execute = runner or run_plan
    result = execute(
        variant.plan_spec,
        event_callback=event_callback,
        dry_run=dry_run,
        execution_profile=execution_profile,
        verbosity=verbosity,
        output_mode=output_mode,
        cache_dir=Path(cache_dir) if isinstance(cache_dir, str) else cache_dir,
        auto_approve=auto_approve,
        extra_template_vars=extra_template_vars,
    )
    variant.run_result = result
    variant.is_valid = result.success
    variant.plan_hash = result.plan_hash or variant.plan_hash or compute_plan_hash(variant.plan_spec)
    variant.metadata["run_id"] = result.run_id
    return result


def apply_historical_pruning(
    variant: WorkflowVariant,
    *,
    threshold: float = 0.8,
    min_runs: int = 5,
    recent_runs: int = 20,
    horizon_days: int = 30,
) -> HistoricalPruningDecision:
    """Consult score history and mark the node pruned when warranted."""
    plan_hash = variant.plan_hash or compute_plan_hash(variant.plan_spec)
    decision = get_historical_pruning_decision(
        variant.plan_spec.name,
        variant.plan_spec.source_dir,
        plan_hash,
        threshold=threshold,
        min_runs=min_runs,
        recent_runs=recent_runs,
        horizon_days=horizon_days,
    )
    variant.plan_hash = plan_hash
    variant.pruned = decision.prune
    variant.metadata["historical_pruning"] = decision.to_dict()
    return decision


def backpropagate_variant(
    variant: WorkflowVariant,
    *,
    score_record: ScoreRecord | None = None,
    history_limit: int = 20,
    horizon_days: int = 30,
    anchor_time: datetime | None = None,
    score_discount: float = 1.0,
) -> float:
    """Blend the current simulation with historical scores and update lineage."""
    if score_discount <= 0.0 or score_discount > 1.0:
        raise ValueError("score_discount must be between 0.0 and 1.0")
    resolved_score_record = score_record
    if resolved_score_record is None and variant.run_result is not None:
        resolved_score_record = build_score_record(
            variant.plan_spec,
            variant.run_result,
            plan_hash=variant.plan_hash or None,
        )

    plan_hash = (
        resolved_score_record.plan_hash
        if resolved_score_record is not None
        else variant.plan_hash or compute_plan_hash(variant.plan_spec)
    )
    history = load_score_history(
        variant.plan_spec.name,
        variant.plan_spec.source_dir,
        plan_hash=plan_hash,
        limit=history_limit,
    )

    if resolved_score_record is not None:
        history = [record for record in history if record.run_id != resolved_score_record.run_id]

    records = ([resolved_score_record] if resolved_score_record is not None else []) + history
    if not records:
        raise ValueError("Cannot backpropagate without a score record or prior score history")

    anchor = anchor_time or datetime.now(timezone.utc)
    aggregate_score = sum(
        _score_record_value(record, anchor=anchor, horizon_days=horizon_days)
        * (
            score_discount
            if (
                resolved_score_record is not None
                and record.run_id == resolved_score_record.run_id
            )
            else 1.0
        )
        for record in records
    ) / len(records)
    prior_bonus = _variant_total_prior(variant)
    aggregate_score = max(0.0, min(1.0, aggregate_score + prior_bonus))

    variant.plan_hash = plan_hash
    if resolved_score_record is not None:
        variant.is_valid = resolved_score_record.success
        variant.metadata["score_record"] = resolved_score_record.to_dict()
    if score_discount != 1.0:
        variant.metadata["score_discount"] = round(score_discount, 6)
    variant.metadata["history_samples"] = len(records)
    for prior_key in _VARIANT_PRIOR_KEYS:
        prior_value = float(variant.metadata.get(prior_key, 0.0) or 0.0)
        if prior_value:
            variant.metadata[prior_key] = round(prior_value, 6)
    variant.metadata["aggregate_score"] = round(aggregate_score, 6)

    current: WorkflowVariant | None = variant
    while current is not None:
        total = current.score * current.visits
        current.visits += 1
        current.score = (total + aggregate_score) / current.visits
        current = current.parent

    return aggregate_score


def score_record_fitness(
    record: ScoreRecord,
    *,
    anchor_time: datetime | None = None,
    horizon_days: int = 30,
) -> float:
    """Convert a score record into the normalized fitness used by MCTS."""
    anchor = anchor_time or datetime.now(timezone.utc)
    return _score_record_value(record, anchor=anchor, horizon_days=horizon_days)


def append_tree_node(run_path: str | Path, variant: WorkflowVariant) -> Path:
    """Append a JSONL snapshot of a workflow variant to tree.jsonl."""
    tree_path = _resolve_tree_path(run_path)
    tree_path.parent.mkdir(parents=True, exist_ok=True)
    with tree_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(variant.to_dict(), ensure_ascii=True, sort_keys=True) + "\n")
        handle.flush()
    return tree_path


def load_tree_index(run_path: str | Path) -> list[dict[str, Any]]:
    """Load persisted tree snapshots without rehydrating full PlanSpec objects."""
    tree_path = _resolve_tree_path(run_path)
    if not tree_path.exists():
        return []

    rows: list[dict[str, Any]] = []
    try:
        for raw_line in tree_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                parsed = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                rows.append(parsed)
    except OSError:
        return []
    return rows


def _variant_node_id(
    plan_hash: str,
    *,
    parent: WorkflowVariant | None,
    mutation_desc: str,
) -> str:
    child_index = len(parent.children) if parent is not None else 0
    seed = f"{parent.node_id if parent is not None else 'root'}:{plan_hash}:{mutation_desc}:{child_index}"
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


def _variant_depth(variant: WorkflowVariant) -> int:
    depth = 0
    current = variant.parent
    while current is not None:
        depth += 1
        current = current.parent
    return depth


def _resolve_tree_path(run_path: str | Path) -> Path:
    path = Path(run_path)
    if path.name == _TREE_FILENAME:
        return path
    return path / _TREE_FILENAME


def _score_record_value(
    record: ScoreRecord,
    *,
    anchor: datetime,
    horizon_days: int,
) -> float:
    quality = record.quality_score
    if quality is None:
        quality = 1.0 if record.success else 0.0
    success_score = 1.0 if record.success else 0.0
    cost_score = 1.0 / (1.0 + max(record.cost_usd or 0.0, 0.0))
    duration_score = 1.0 / (1.0 + max(record.duration_sec, 0.0) / 60.0)
    base_score = (
        0.60 * quality
        + 0.25 * success_score
        + 0.10 * cost_score
        + 0.05 * duration_score
    )

    if horizon_days <= 0:
        return base_score

    timestamp = _parse_timestamp(record.timestamp)
    age_days = max(0.0, (anchor - timestamp).total_seconds() / 86400.0)
    decay = 0.5 ** (age_days / float(horizon_days))
    return float(base_score * decay)


def _ucb1_score(
    variant: WorkflowVariant,
    *,
    exploration_constant: float,
) -> float:
    if variant.visits <= 0:
        return float("inf")

    parent_visits = variant.parent.visits if variant.parent is not None else variant.visits
    parent_visits = max(parent_visits, variant.visits, 1)
    if parent_visits <= 1 or exploration_constant == 0.0:
        return variant.score

    return variant.score + exploration_constant * math.sqrt(
        math.log(parent_visits) / float(variant.visits)
    )


def _parse_timestamp(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
