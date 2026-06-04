from __future__ import annotations

import difflib
import json
import random
import re
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .audit import AuditFinding, audit_plan
from .cache import compute_simulation_plan_hash
from .eventsource import ChainState, emit_hashed_event, replay_events
from .errors import PlanValidationError
from .knowledge import build_score_record
from .knowledge import (
    format_knowledge,
    load_knowledge,
    load_simulation_cache_record,
    load_score_history,
    plan_topology_signature,
    record_knowledge_retrievals,
    select_relevant_knowledge,
)
from .loader import load_plan
from .mcts import (
    DEFAULT_EXPLORATION_CONSTANT,
    apply_variant_historical_fitness_prior,
    apply_variant_knowledge_prior,
    apply_variant_novelty_prior,
    append_tree_node,
    apply_historical_pruning,
    backpropagate_variant,
    create_workflow_variant,
    iter_variants,
    load_tree_index,
    select_expansion_parent,
    select_variant_from_pool,
    simulate_variant,
    score_record_fitness,
)
from .knowledge_graph import extract_entities
from .scheduler import run_plan
from .models import (
    ExecutionProfile,
    KnowledgeRecord,
    MctsSelectionPolicy,
    OutputMode,
    PlanRunResult,
    PlanSpec,
    ReplanPopulationStrategy,
    ReplanAttempt,
    ReplanState,
    ScoreRecord,
    Verbosity,
    WorkflowVariant,
)
from .runners import _run_firewall_pass2
from .watch import _save_stepping_stone

_BLOCKING_GENERATED_PLAN_SEVERITIES = {"error", "warning"}
_REPLAN_GUIDANCE_KINDS = {"failure_pattern", "model_pattern"}
_REPLAN_HISTORY_BOOTSTRAP_MAX_BONUS = 0.08
_REPLAN_HISTORY_BOOTSTRAP_MIN_SIMILARITY = 0.35
_REPLAN_HISTORY_BOOTSTRAP_LIMIT = 48
_REPLAN_NOVELTY_MAX_BONUS = 0.06
_REPLAN_TOURNAMENT_DIVERSITY_FLOOR = 0.25
_SIMULATION_CACHE_SCORE_DISCOUNT = 0.9
_SIMULATION_CACHE_SCAN_LIMIT = 128


@dataclass
class _GeneratedPlanSecurityDecision:
    blocked: bool
    introduced_findings: list[AuditFinding]
    firewall_verdict: str = "allow"
    firewall_category: str = ""
    firewall_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "blocked": self.blocked,
            "introduced_findings": [
                finding.to_dict() for finding in self.introduced_findings
            ],
            "firewall_verdict": self.firewall_verdict,
        }
        if self.firewall_category:
            data["firewall_category"] = self.firewall_category
        if self.firewall_reason:
            data["firewall_reason"] = self.firewall_reason
        return data


@dataclass
class _ReplanAuditTrail:
    plan_name: str
    events_path: Path
    chain_state: ChainState


@dataclass
class _ReplanDuplicateDecision:
    duplicate: bool
    source: str = ""
    existing_node_id: str = ""
    existing_run_id: str = ""
    score_record: ScoreRecord | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "duplicate": self.duplicate,
            "source": self.source,
        }
        if self.existing_node_id:
            data["existing_node_id"] = self.existing_node_id
        if self.existing_run_id:
            data["existing_run_id"] = self.existing_run_id
        if self.score_record is not None:
            data["score_record"] = self.score_record.to_dict()
        return data


def _rehydrate_cached_run_result(
    variant: WorkflowVariant,
    score_record: ScoreRecord,
) -> PlanRunResult:
    metadata = score_record.metadata
    run_path = Path(str(metadata.get("run_path") or variant.plan_spec.source_dir))
    timestamp = datetime.fromisoformat(score_record.timestamp)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    else:
        timestamp = timestamp.astimezone(timezone.utc)
    execution_profile: ExecutionProfile = "plan"
    raw_execution_profile = str(metadata.get("execution_profile") or "plan")
    if raw_execution_profile == "safe":
        execution_profile = "safe"
    elif raw_execution_profile == "yolo":
        execution_profile = "yolo"
    raw_total_tokens = metadata.get("total_tokens")
    total_tokens = raw_total_tokens if isinstance(raw_total_tokens, int) else None
    return PlanRunResult(
        plan_name=variant.plan_spec.name,
        run_id=score_record.run_id,
        run_path=run_path,
        started_at=timestamp,
        finished_at=timestamp,
        success=score_record.success,
        execution_profile=execution_profile,
        task_results={},
        total_cost_usd=None,
        total_tokens=total_tokens,
        plan_hash=variant.plan_hash or score_record.plan_hash,
        quality_score=score_record.quality_score,
    )


def _record_simulation_cache_hit(
    state: ReplanState,
    variant: WorkflowVariant,
    score_record: ScoreRecord,
    *,
    search_tree_dir: Path,
    score_discount: float,
    simulation_plan_hash: str,
) -> None:
    variant.run_result = _rehydrate_cached_run_result(variant, score_record)
    variant.is_valid = score_record.success
    variant.metadata["simulation_cache"] = {
        "hit": True,
        "source_run_id": score_record.run_id,
        "source_plan_hash": score_record.plan_hash,
        "simulation_plan_hash": simulation_plan_hash,
        "score_discount": round(score_discount, 6),
    }
    backpropagate_variant(
        variant,
        score_record=score_record,
        score_discount=score_discount,
    )
    append_tree_node(search_tree_dir, variant)
    if state.search_tree_path is None:
        state.search_tree_path = str(search_tree_dir / "tree.jsonl")


def replan(
    plan_path: str | Path,
    *,
    max_attempts: int = 3,
    analysis_model: str = "opus",
    event_callback: Callable[[str, dict[str, object]], None] | None = None,
    dry_run: bool = False,
    execution_profile: ExecutionProfile = "plan",
    verbosity: Verbosity = "normal",
    output_mode: OutputMode = "text",
    cache_dir: str | Path | None = None,
    auto_approve: bool = False,
    extra_template_vars: dict[str, str] | None = None,
    variants: int = 1,
    debug_prob: float = 0.5,
    selection_policy: MctsSelectionPolicy = "debug_prob",
    exploration_constant: float = DEFAULT_EXPLORATION_CONSTANT,
    population_strategy: ReplanPopulationStrategy = "best",
    tournament_size: int = 2,
    elite_count: int = 1,
    diversity_floor: float = _REPLAN_TOURNAMENT_DIVERSITY_FLOOR,
) -> ReplanState:
    if variants < 1:
        raise ValueError("variants must be >= 1")
    if debug_prob < 0.0 or debug_prob > 1.0:
        raise ValueError("debug_prob must be between 0.0 and 1.0")
    if selection_policy not in {"debug_prob", "ucb1"}:
        raise ValueError("selection_policy must be 'debug_prob' or 'ucb1'")
    if exploration_constant < 0.0:
        raise ValueError("exploration_constant must be >= 0.0")
    if population_strategy not in {"best", "tournament"}:
        raise ValueError("population_strategy must be 'best' or 'tournament'")
    if tournament_size < 1:
        raise ValueError("tournament_size must be >= 1")
    if elite_count < 0:
        raise ValueError("elite_count must be >= 0")
    if diversity_floor < 0.0 or diversity_floor > 1.0:
        raise ValueError("diversity_floor must be between 0.0 and 1.0")

    if variants > 1:
        return _replan_search(
            plan_path,
            max_attempts=max_attempts,
            analysis_model=analysis_model,
            event_callback=event_callback,
            dry_run=dry_run,
            execution_profile=execution_profile,
            verbosity=verbosity,
            output_mode=output_mode,
            cache_dir=cache_dir,
            auto_approve=auto_approve,
            extra_template_vars=extra_template_vars,
            variants=variants,
            debug_prob=debug_prob,
            selection_policy=selection_policy,
            exploration_constant=exploration_constant,
            population_strategy=population_strategy,
            tournament_size=tournament_size,
            elite_count=elite_count,
            diversity_floor=diversity_floor,
        )

    return _replan_single(
        plan_path,
        max_attempts=max_attempts,
        analysis_model=analysis_model,
        event_callback=event_callback,
        dry_run=dry_run,
        execution_profile=execution_profile,
        verbosity=verbosity,
        output_mode=output_mode,
        cache_dir=cache_dir,
        auto_approve=auto_approve,
        extra_template_vars=extra_template_vars,
    )


def _replan_single(
    plan_path: str | Path,
    *,
    max_attempts: int,
    analysis_model: str,
    event_callback: Callable[[str, dict[str, object]], None] | None,
    dry_run: bool,
    execution_profile: ExecutionProfile,
    verbosity: Verbosity,
    output_mode: OutputMode,
    cache_dir: str | Path | None,
    auto_approve: bool,
    extra_template_vars: dict[str, str] | None,
) -> ReplanState:
    current_plan_path = Path(plan_path).resolve()
    resolved_cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
    temp_dirs: list[Path] = []
    trusted_firewall_model: str | None = None

    state = ReplanState(
        plan_path=str(current_plan_path),
        max_attempts=max_attempts,
        analysis_model=analysis_model,
    )

    for attempt_number in range(1, max_attempts + 1):
        if _detect_exit_loop(state.attempts):
            print("[maestro] Circuit breaker triggered: repeated failed task set.")
            state.status = "circuit_breaker"
            break

        attempt = ReplanAttempt(attempt_number=attempt_number)

        try:
            plan_yaml = current_plan_path.read_text(encoding="utf-8")
            plan: PlanSpec = load_plan(current_plan_path)
            attempt.plan_yaml = plan_yaml
            if trusted_firewall_model is None and plan.firewall_model:
                trusted_firewall_model = plan.firewall_model
        except (PlanValidationError, OSError) as exc:
            attempt.error_summary = f"Failed to load plan: {exc}"
            state.attempts.append(attempt)
            print(f"[maestro] {attempt.error_summary}")
            continue

        try:
            result = run_plan(
                plan,
                event_callback=event_callback,
                dry_run=dry_run,
                execution_profile=execution_profile,
                verbosity=verbosity,
                output_mode=output_mode,
                cache_dir=resolved_cache_dir,
                auto_approve=auto_approve,
                extra_template_vars=extra_template_vars,
            )
        except Exception as exc:  # pragma: no cover - orchestration safety net
            attempt.error_summary = f"Plan execution failed: {exc}"
            state.attempts.append(attempt)
            print(f"[maestro] {attempt.error_summary}")
            continue

        attempt.run_result = result
        if result.total_cost_usd is not None:
            state.total_cost_usd += result.total_cost_usd
        if result.total_tokens is not None:
            state.total_tokens += result.total_tokens

        if result.success:
            state.final_success = True
            state.status = "success"
            state.attempts.append(attempt)
            break

        failed_state = _extract_failed_state(result)
        attempt.failed_task_ids = list(failed_state["failed_task_ids"])

        knowledge_records = _select_replan_knowledge_records(
            plan,
            failed_state,
            plan_yaml=plan_yaml,
        )
        knowledge_guidance = _build_replan_knowledge_guidance(knowledge_records)
        prompt = _build_analysis_prompt(
            plan_yaml,
            failed_state,
            knowledge_guidance=knowledge_guidance,
        )
        try:
            attempt.analysis_response = _call_analysis_model(prompt, analysis_model)
        except Exception as exc:
            attempt.analysis_error = str(exc)
            attempt.error_summary = f"Analysis failed: {exc}"
            state.attempts.append(attempt)
            print(f"[maestro] {attempt.error_summary}")
            continue

        corrected_yaml = _parse_corrected_yaml(attempt.analysis_response)
        if corrected_yaml is None:
            attempt.analysis_error = "No YAML code fence found in analysis response."
            attempt.error_summary = attempt.analysis_error
            state.attempts.append(attempt)
            print(f"[maestro] {attempt.error_summary}")
            continue

        attempt.corrected_plan_yaml = corrected_yaml
        attempt.diff_summary = _show_plan_diff(plan_yaml, corrected_yaml)

        if auto_approve:
            attempt.approved = True
        else:
            approval = input("[maestro] Approve? [y/n]: ").strip().lower()
            attempt.approved = approval == "y"

        state.attempts.append(attempt)

        if not attempt.approved:
            print("[maestro] Re-plan rejected by user.")
            state.status = "circuit_breaker"
            break

        # .resolve() canonicalizes 8.3 short names (e.g. RUNNER~1) so candidate
        # paths compare equal to the resolved scenario/plan paths on Windows.
        temp_dir = Path(tempfile.mkdtemp(prefix="maestro-replan-")).resolve()
        temp_dirs.append(temp_dir)
        corrected_path = temp_dir / current_plan_path.name

        try:
            corrected_path.write_text(corrected_yaml, encoding="utf-8")
            candidate_plan = load_plan(corrected_path)
        except (PlanValidationError, OSError) as exc:
            attempt.error_summary = f"Corrected plan invalid: {exc}"
            print(f"[maestro] {attempt.error_summary}")
            continue

        security = _evaluate_generated_plan_security(
            candidate_plan,
            corrected_yaml,
            baseline_plan=plan,
            firewall_model=trusted_firewall_model,
            workdir=plan.source_dir,
        )
        if security.blocked:
            attempt.error_summary = _format_security_gate_message(security)
            print(f"[maestro] {attempt.error_summary}")
            continue

        current_plan_path = corrected_path
        if trusted_firewall_model is None and candidate_plan.firewall_model:
            trusted_firewall_model = candidate_plan.firewall_model

    for td in temp_dirs:
        shutil.rmtree(td, ignore_errors=True)

    return state


def _replan_search(
    plan_path: str | Path,
    *,
    max_attempts: int,
    analysis_model: str,
    event_callback: Callable[[str, dict[str, object]], None] | None,
    dry_run: bool,
    execution_profile: ExecutionProfile,
    verbosity: Verbosity,
    output_mode: OutputMode,
    cache_dir: str | Path | None,
    auto_approve: bool,
    extra_template_vars: dict[str, str] | None,
    variants: int,
    debug_prob: float,
    selection_policy: MctsSelectionPolicy,
    exploration_constant: float,
    population_strategy: ReplanPopulationStrategy,
    tournament_size: int,
    elite_count: int,
    diversity_floor: float,
) -> ReplanState:
    current_plan_path = Path(plan_path).resolve()
    resolved_cache_dir = Path(cache_dir).resolve() if cache_dir is not None else None
    temp_dirs: list[Path] = []
    search_root: WorkflowVariant | None = None
    current_variant: WorkflowVariant | None = None
    search_approved = auto_approve
    trusted_firewall_model: str | None = None
    audit_trail: _ReplanAuditTrail | None = None

    state = ReplanState(
        plan_path=str(current_plan_path),
        max_attempts=max_attempts,
        analysis_model=analysis_model,
    )

    for attempt_number in range(1, max_attempts + 1):
        if _detect_exit_loop(state.attempts):
            print("[maestro] Circuit breaker triggered: repeated failed task set.")
            state.status = "circuit_breaker"
            break

        attempt = ReplanAttempt(attempt_number=attempt_number)

        try:
            plan_yaml = current_plan_path.read_text(encoding="utf-8")
            plan: PlanSpec = load_plan(current_plan_path)
            attempt.plan_yaml = plan_yaml
            if trusted_firewall_model is None and plan.firewall_model:
                trusted_firewall_model = plan.firewall_model
        except (PlanValidationError, OSError) as exc:
            attempt.error_summary = f"Failed to load plan: {exc}"
            state.attempts.append(attempt)
            print(f"[maestro] {attempt.error_summary}")
            continue

        if current_variant is None:
            current_variant = create_workflow_variant(plan, mutation_desc="root search variant")
            search_root = current_variant
        else:
            current_variant.plan_spec = plan

        reused_result = current_variant.run_result is not None
        if reused_result:
            result = current_variant.run_result
            assert result is not None
        else:
            try:
                result = run_plan(
                    plan,
                    event_callback=event_callback,
                    dry_run=dry_run,
                    execution_profile=execution_profile,
                    verbosity=verbosity,
                    output_mode=output_mode,
                    cache_dir=resolved_cache_dir,
                    auto_approve=auto_approve,
                    extra_template_vars=extra_template_vars,
                )
            except Exception as exc:  # pragma: no cover - orchestration safety net
                attempt.error_summary = f"Plan execution failed: {exc}"
                state.attempts.append(attempt)
                print(f"[maestro] {attempt.error_summary}")
                continue
            _record_variant_result(
                state,
                current_variant,
                result,
                search_tree_dir=result.run_path,
            )

        if state.search_tree_path is None:
            state.search_tree_path = str(result.run_path / "tree.jsonl")  # pragma: no cover
        if audit_trail is None:
            audit_trail = _open_replan_audit_trail(plan.name, result.run_path)
            _append_replan_audit_event(
                audit_trail,
                "replan_search_start",
                event_callback=event_callback,
                analysis_model=analysis_model,
                variants=variants,
                debug_prob=debug_prob,
                selection_policy=selection_policy,
                exploration_constant=round(exploration_constant, 6),
                population_strategy=population_strategy,
                tournament_size=tournament_size,
                elite_count=elite_count,
                diversity_floor=round(diversity_floor, 6),
                root_node_id=current_variant.node_id,
                root_run_id=result.run_id,
                run_path=str(result.run_path),
            )

        attempt.run_result = result
        if result.success:
            state.final_success = True
            state.status = "success"
            state.attempts.append(attempt)
            break

        failed_state = _extract_failed_state(result)
        attempt.failed_task_ids = list(failed_state["failed_task_ids"])
        baseline_attempt_score = current_variant.score
        knowledge_records = _select_replan_knowledge_records(
            plan,
            failed_state,
            plan_yaml=plan_yaml,
        )
        knowledge_guidance = _build_replan_knowledge_guidance(knowledge_records)
        if audit_trail is not None:
            _append_replan_audit_event(
                audit_trail,
                "replan_round_start",
                event_callback=event_callback,
                attempt_number=attempt_number,
                current_node_id=current_variant.node_id,
                failed_task_ids=attempt.failed_task_ids,
                knowledge_guidance=bool(knowledge_guidance),
            )

        if not search_approved:
            approval = input(
                "[maestro] Multi-variant search will execute generated plans automatically. Approve? [y/n]: "
            ).strip().lower()
            attempt.approved = approval == "y"
            if not attempt.approved:
                state.attempts.append(attempt)
                print("[maestro] Re-plan rejected by user.")
                state.status = "circuit_breaker"
                break
            search_approved = True
        else:
            attempt.approved = True

        candidate_records: dict[str, dict[str, str]] = {}
        prior_tree_rows = (
            load_tree_index(Path(state.search_tree_path).parent)
            if state.search_tree_path is not None
            else []
        )
        tree_plan_hash_index = _index_tree_rows_by_plan_hash(prior_tree_rows)
        historical_score_cache: dict[str, ScoreRecord | None] = {}
        historical_bootstrap_records = load_score_history(
            plan.name,
            plan.source_dir,
            limit=_REPLAN_HISTORY_BOOTSTRAP_LIMIT,
        )

        for candidate_number in range(1, variants + 1):
            prompt = _build_analysis_prompt(
                plan_yaml,
                failed_state,
                candidate_number=candidate_number,
                total_candidates=variants,
                knowledge_guidance=knowledge_guidance,
            )
            candidate_summary: dict[str, Any] = {
                "candidate_number": candidate_number,
                "variant_type": "debug" if not current_variant.is_valid else "improve",
            }

            try:
                response = _call_analysis_model(prompt, analysis_model)
            except Exception as exc:
                candidate_summary["error"] = f"Analysis failed: {exc}"
                attempt.candidate_variants.append(candidate_summary)
                continue

            corrected_yaml = _parse_corrected_yaml(response)
            if corrected_yaml is None:
                candidate_summary["error"] = "No YAML code fence found in analysis response."
                attempt.candidate_variants.append(candidate_summary)
                continue

            # .resolve() canonicalizes 8.3 short names (e.g. RUNNER~1) so candidate
            # paths compare equal to the resolved scenario/plan paths on Windows.
            temp_dir = Path(tempfile.mkdtemp(prefix="maestro-replan-")).resolve()
            temp_dirs.append(temp_dir)
            corrected_path = temp_dir / f"candidate-{candidate_number}-{current_plan_path.name}"

            try:
                corrected_path.write_text(corrected_yaml, encoding="utf-8")
                candidate_plan = load_plan(corrected_path)
            except (PlanValidationError, OSError) as exc:
                candidate_summary["error"] = f"Corrected plan invalid: {exc}"
                attempt.candidate_variants.append(candidate_summary)
                continue

            variant = create_workflow_variant(
                candidate_plan,
                parent=current_variant,
                mutation_desc=f"candidate {candidate_number}/{variants}",
                metadata={"tainted": True},
            )
            simulation_plan_hash = compute_simulation_plan_hash(candidate_plan)
            variant.metadata["simulation_plan_hash"] = simulation_plan_hash
            candidate_summary["node_id"] = variant.node_id
            candidate_summary["plan_hash"] = variant.plan_hash
            candidate_summary["simulation_plan_hash"] = simulation_plan_hash
            candidate_summary["tainted"] = True
            candidate_records[variant.node_id] = {
                "analysis_response": response,
                "corrected_yaml": corrected_yaml,
                "diff_summary": _show_plan_diff(plan_yaml, corrected_yaml, print_lines=False),
                "path": str(corrected_path),
            }
            if audit_trail is not None:
                _append_replan_audit_event(
                    audit_trail,
                    "replan_candidate_generated",
                    event_callback=event_callback,
                    attempt_number=attempt_number,
                    candidate_number=candidate_number,
                    node_id=variant.node_id,
                    parent_id=current_variant.node_id,
                    variant_type=variant.variant_type,
                    mutation_desc=variant.mutation_desc,
                    plan_hash=variant.plan_hash,
                    tainted=True,
                )

            knowledge_bonus, knowledge_signals = _score_replan_variant_with_knowledge(
                corrected_yaml,
                knowledge_records,
                baseline_plan_yaml=plan_yaml,
            )
            if knowledge_bonus:
                apply_variant_knowledge_prior(
                    variant,
                    bonus=knowledge_bonus,
                    signals=knowledge_signals,
                )
            candidate_summary["knowledge_bonus"] = round(knowledge_bonus, 6)
            if knowledge_signals:
                candidate_summary["knowledge_hits"] = knowledge_signals
            novelty_bonus, novelty_signals = _score_replan_variant_novelty(
                corrected_yaml,
                baseline_plan_yaml=plan_yaml,
                prior_tree_rows=prior_tree_rows,
            )
            if novelty_bonus:
                apply_variant_novelty_prior(
                    variant,
                    bonus=novelty_bonus,
                    signals=novelty_signals,
                    max_abs_bonus=_REPLAN_NOVELTY_MAX_BONUS,
                )
            candidate_summary["novelty_bonus"] = round(novelty_bonus, 6)
            if novelty_signals:
                candidate_summary["novelty_hits"] = novelty_signals
            historical_fitness_bonus, historical_fitness_signals = _score_replan_variant_with_history(
                variant,
                historical_bootstrap_records,
            )
            if historical_fitness_bonus:
                apply_variant_historical_fitness_prior(
                    variant,
                    bonus=historical_fitness_bonus,
                    signals=historical_fitness_signals,
                    max_abs_bonus=_REPLAN_HISTORY_BOOTSTRAP_MAX_BONUS,
                )
            candidate_summary["historical_fitness_bonus"] = round(historical_fitness_bonus, 6)
            if historical_fitness_signals:
                candidate_summary["historical_fitness_hits"] = historical_fitness_signals
            security = _evaluate_generated_plan_security(
                candidate_plan,
                corrected_yaml,
                baseline_plan=plan,
                firewall_model=trusted_firewall_model,
                workdir=plan.source_dir,
            )
            candidate_summary["security_blocked"] = security.blocked
            candidate_summary["firewall_verdict"] = security.firewall_verdict
            if security.introduced_findings:
                candidate_summary["security_findings"] = [
                    finding.to_dict() for finding in security.introduced_findings
                ]
            if security.firewall_category:
                candidate_summary["firewall_category"] = security.firewall_category
            variant.metadata["security_validation"] = security.to_dict()
            if security.blocked:
                variant.pruned = True
                candidate_summary["pruned"] = True
                candidate_summary["error"] = _format_security_gate_message(security)
                if state.search_tree_path is not None:
                    append_tree_node(Path(state.search_tree_path).parent, variant)
                if audit_trail is not None:
                    _append_replan_audit_event(
                        audit_trail,
                        "replan_candidate_blocked",
                        event_callback=event_callback,
                        attempt_number=attempt_number,
                        candidate_number=candidate_number,
                        node_id=variant.node_id,
                        plan_hash=variant.plan_hash,
                        security=security.to_dict(),
                        knowledge_bonus=round(knowledge_bonus, 6),
                        novelty_bonus=round(novelty_bonus, 6),
                        historical_fitness_bonus=round(historical_fitness_bonus, 6),
                    )
                attempt.candidate_variants.append(candidate_summary)
                tree_plan_hash_index[variant.plan_hash] = variant.to_dict()
                continue

            duplicate = _detect_replan_duplicate(
                variant,
                plan_name=plan.name,
                source_dir=plan.source_dir,
                tree_plan_hash_index=tree_plan_hash_index,
                historical_score_cache=historical_score_cache,
            )
            if duplicate.duplicate:
                candidate_summary["deduplicated"] = True
                candidate_summary["duplicate_source"] = duplicate.source
                if duplicate.existing_node_id:
                    candidate_summary["duplicate_node_id"] = duplicate.existing_node_id
                if duplicate.existing_run_id:
                    candidate_summary["duplicate_run_id"] = duplicate.existing_run_id
                variant.metadata["deduplication"] = duplicate.to_dict()
                if duplicate.source == "search_tree":
                    variant.pruned = True
                    candidate_summary["pruned"] = True
                    candidate_summary["error"] = "Candidate deduplicated against existing search-tree variant."
                elif duplicate.score_record is not None:
                    backpropagate_variant(variant, score_record=duplicate.score_record)
                    candidate_summary["historical_success"] = duplicate.score_record.success
                    candidate_summary["score"] = round(variant.score, 6)
                if state.search_tree_path is not None:
                    append_tree_node(Path(state.search_tree_path).parent, variant)
                if audit_trail is not None:
                    _append_replan_audit_event(
                        audit_trail,
                        "replan_candidate_deduplicated",
                        event_callback=event_callback,
                        attempt_number=attempt_number,
                        candidate_number=candidate_number,
                        node_id=variant.node_id,
                        plan_hash=variant.plan_hash,
                        duplicate=duplicate.to_dict(),
                        knowledge_bonus=round(knowledge_bonus, 6),
                        novelty_bonus=round(novelty_bonus, 6),
                        historical_fitness_bonus=round(historical_fitness_bonus, 6),
                    )
                attempt.candidate_variants.append(candidate_summary)
                tree_plan_hash_index[variant.plan_hash] = variant.to_dict()
                continue

            simulation_cache_record = load_simulation_cache_record(
                plan.name,
                plan.source_dir,
                simulation_plan_hash=simulation_plan_hash,
                limit=_SIMULATION_CACHE_SCAN_LIMIT,
            )
            if simulation_cache_record is not None:
                _record_simulation_cache_hit(
                    state,
                    variant,
                    simulation_cache_record,
                    search_tree_dir=Path(state.search_tree_path).parent,
                    score_discount=_SIMULATION_CACHE_SCORE_DISCOUNT,
                    simulation_plan_hash=simulation_plan_hash,
                )
                candidate_summary["simulation_cache_hit"] = True
                candidate_summary["simulation_cache_source_run_id"] = simulation_cache_record.run_id
                candidate_summary["simulation_cache_source_plan_hash"] = simulation_cache_record.plan_hash
                candidate_summary["simulation_cache_discount"] = _SIMULATION_CACHE_SCORE_DISCOUNT
                candidate_summary["success"] = simulation_cache_record.success
                candidate_summary["run_id"] = simulation_cache_record.run_id
                candidate_summary["score"] = round(variant.score, 6)
                if audit_trail is not None:
                    _append_replan_audit_event(
                        audit_trail,
                        "replan_candidate_cache_hit",
                        event_callback=event_callback,
                        attempt_number=attempt_number,
                        candidate_number=candidate_number,
                        node_id=variant.node_id,
                        plan_hash=variant.plan_hash,
                        simulation_plan_hash=simulation_plan_hash,
                        source_run_id=simulation_cache_record.run_id,
                        source_plan_hash=simulation_cache_record.plan_hash,
                        score=round(variant.score, 6),
                        score_discount=round(_SIMULATION_CACHE_SCORE_DISCOUNT, 6),
                        knowledge_bonus=round(knowledge_bonus, 6),
                        novelty_bonus=round(novelty_bonus, 6),
                        historical_fitness_bonus=round(historical_fitness_bonus, 6),
                    )
                attempt.candidate_variants.append(candidate_summary)
                tree_plan_hash_index[variant.plan_hash] = variant.to_dict()
                continue

            pruning = apply_historical_pruning(variant)
            candidate_summary["pruned"] = pruning.prune
            candidate_summary["failure_rate"] = round(pruning.failure_rate, 4)
            if pruning.prune:
                if state.search_tree_path is not None:
                    append_tree_node(Path(state.search_tree_path).parent, variant)
                if audit_trail is not None:
                    _append_replan_audit_event(
                        audit_trail,
                        "replan_candidate_pruned",
                        event_callback=event_callback,
                        attempt_number=attempt_number,
                        candidate_number=candidate_number,
                        node_id=variant.node_id,
                        plan_hash=variant.plan_hash,
                        pruning=pruning.to_dict(),
                        knowledge_bonus=round(knowledge_bonus, 6),
                        novelty_bonus=round(novelty_bonus, 6),
                        historical_fitness_bonus=round(historical_fitness_bonus, 6),
                    )
                attempt.candidate_variants.append(candidate_summary)
                tree_plan_hash_index[variant.plan_hash] = variant.to_dict()
                continue

            try:
                variant_result = simulate_variant(
                    variant,
                    runner=run_plan,
                    event_callback=event_callback,
                    dry_run=dry_run,
                    execution_profile=execution_profile,
                    verbosity=verbosity,
                    output_mode=output_mode,
                    cache_dir=resolved_cache_dir,
                    auto_approve=auto_approve,
                    extra_template_vars=extra_template_vars,
                )
            except Exception as exc:  # pragma: no cover - orchestration safety net
                variant.pruned = True
                candidate_summary["error"] = f"Simulation failed: {exc}"
                if state.search_tree_path is not None:
                    append_tree_node(Path(state.search_tree_path).parent, variant)
                attempt.candidate_variants.append(candidate_summary)
                tree_plan_hash_index[variant.plan_hash] = variant.to_dict()
                continue

            _record_variant_result(
                state,
                variant,
                variant_result,
                search_tree_dir=Path(state.search_tree_path).parent if state.search_tree_path else variant_result.run_path,
            )

            candidate_summary["success"] = variant_result.success
            candidate_summary["run_id"] = variant_result.run_id
            candidate_summary["score"] = round(variant.score, 6)
            if audit_trail is not None:
                _append_replan_audit_event(
                    audit_trail,
                    "replan_candidate_simulated",
                    event_callback=event_callback,
                    attempt_number=attempt_number,
                    candidate_number=candidate_number,
                    node_id=variant.node_id,
                    plan_hash=variant.plan_hash,
                    run_id=variant_result.run_id,
                    run_path=str(variant_result.run_path),
                    success=variant_result.success,
                    score=round(variant.score, 6),
                    knowledge_bonus=round(knowledge_bonus, 6),
                    novelty_bonus=round(novelty_bonus, 6),
                    historical_fitness_bonus=round(historical_fitness_bonus, 6),
                )
            attempt.candidate_variants.append(candidate_summary)
            tree_plan_hash_index[variant.plan_hash] = variant.to_dict()

        if not attempt.candidate_variants:
            attempt.error_summary = "No candidate variants were generated."
            state.attempts.append(attempt)
            print(f"[maestro] {attempt.error_summary}")
            break

        assert search_root is not None
        selected_variant = _select_replan_search_variant(
            search_root,
            selection_policy=selection_policy,
            debug_prob=debug_prob,
            exploration_constant=exploration_constant,
            population_strategy=population_strategy,
            tournament_size=tournament_size,
            elite_count=elite_count,
            diversity_floor=diversity_floor,
        )
        if selected_variant is None or selected_variant is current_variant:
            attempt.error_summary = "No selectable candidate variants remained after pruning."
            state.attempts.append(attempt)
            print(f"[maestro] {attempt.error_summary}")
            break

        attempt.selected_candidate_id = selected_variant.node_id
        if audit_trail is not None:
            _append_replan_audit_event(
                audit_trail,
                "replan_candidate_selected",
                event_callback=event_callback,
                attempt_number=attempt_number,
                selected_node_id=selected_variant.node_id,
                plan_hash=selected_variant.plan_hash,
                selection_policy=selection_policy,
                debug_prob=debug_prob,
                exploration_constant=round(exploration_constant, 6),
                population_strategy=population_strategy,
                tournament_size=tournament_size,
                elite_count=elite_count,
                diversity_floor=round(diversity_floor, 6),
                knowledge_bonus=round(float(selected_variant.metadata.get("knowledge_prior", 0.0) or 0.0), 6),
                novelty_bonus=round(float(selected_variant.metadata.get("novelty_prior", 0.0) or 0.0), 6),
                historical_fitness_bonus=round(float(selected_variant.metadata.get("historical_fitness_prior", 0.0) or 0.0), 6),
                success=selected_variant.run_result.success if selected_variant.run_result is not None else None,
                run_id=selected_variant.run_result.run_id if selected_variant.run_result is not None else None,
            )
        selected_record = candidate_records.get(selected_variant.node_id)
        if selected_record is not None:
            attempt.analysis_response = selected_record["analysis_response"]
            attempt.corrected_plan_yaml = selected_record["corrected_yaml"]
            attempt.diff_summary = _show_plan_diff(
                plan_yaml,
                selected_record["corrected_yaml"],
                print_lines=True,
            )
        if selected_variant.plan_spec.source_path is not None:
            current_plan_path = selected_variant.plan_spec.source_path

        selected_result = selected_variant.run_result
        if selected_result is not None:
            selected_failed_state = _extract_failed_state(selected_result)
            attempt.failed_task_ids = list(selected_failed_state["failed_task_ids"])
            if selected_result.success:
                stepping_lessons: list[dict[str, Any]] = []
                if selected_record is not None:
                    stepping_lessons.append(
                        {
                            "source": "replan",
                            "attempt_number": attempt_number,
                            "variant_type": selected_variant.variant_type,
                            "mutation_desc": selected_variant.mutation_desc,
                            "diff_summary": selected_record["diff_summary"],
                        }
                    )
                stepping_stone = None
                if selected_variant.plan_spec.source_path is not None:
                    selected_metric = float(
                        selected_variant.metadata.get("aggregate_score", selected_variant.score) or 0.0
                    )
                    stepping_stone = _save_stepping_stone(
                        plan_path=selected_variant.plan_spec.source_path,
                        plan_name=plan.name,
                        metric_value=selected_metric,
                        metric_name="replan_fitness",
                        iteration=attempt_number,
                        git_commit=None,
                        lessons_path=None,
                        lessons=stepping_lessons,
                        watch_run_path=str(selected_result.run_path),
                        total_cost_usd=state.total_cost_usd,
                        archive_source_dir=plan.source_dir,
                        source_type="replan",
                        metadata={
                            "attempt_number": attempt_number,
                            "selected_node_id": selected_variant.node_id,
                            "selected_run_id": selected_result.run_id,
                            "parent_node_id": current_variant.node_id,
                            "parent_plan_hash": current_variant.plan_hash,
                            "variant_type": selected_variant.variant_type,
                            "mutation_desc": selected_variant.mutation_desc,
                            "baseline_score": round(baseline_attempt_score, 6),
                            "selected_score": round(selected_variant.score, 6),
                            "fitness_gain": round(selected_variant.score - baseline_attempt_score, 6),
                            "knowledge_bonus": round(float(selected_variant.metadata.get("knowledge_prior", 0.0) or 0.0), 6),
                            "novelty_bonus": round(float(selected_variant.metadata.get("novelty_prior", 0.0) or 0.0), 6),
                            "historical_fitness_bonus": round(float(selected_variant.metadata.get("historical_fitness_prior", 0.0) or 0.0), 6),
                        },
                    )
                if stepping_stone is not None and audit_trail is not None:
                    _append_replan_audit_event(
                        audit_trail,
                        "replan_stepping_stone_saved",
                        event_callback=event_callback,
                        attempt_number=attempt_number,
                        selected_node_id=selected_variant.node_id,
                        selected_run_id=selected_result.run_id,
                        plan_hash=selected_variant.plan_hash,
                        metric_name=stepping_stone.metric_name,
                        metric_value=round(stepping_stone.metric_value, 6),
                        source_type=stepping_stone.source_type,
                    )
                attempt.run_result = selected_result
                state.final_success = True
                state.status = "success"
                state.attempts.append(attempt)
                break

        current_variant = selected_variant
        if trusted_firewall_model is None and selected_variant.plan_spec.firewall_model:
            trusted_firewall_model = selected_variant.plan_spec.firewall_model
        state.attempts.append(attempt)

    for td in temp_dirs:
        shutil.rmtree(td, ignore_errors=True)

    if audit_trail is not None:
        _append_replan_audit_event(
            audit_trail,
            "replan_search_complete",
            event_callback=event_callback,
            status=state.status,
            final_success=state.final_success,
            attempts=len(state.attempts),
            total_cost_usd=round(state.total_cost_usd, 6),
            total_tokens=state.total_tokens,
            search_tree_path=state.search_tree_path,
        )

    return state


def _record_variant_result(
    state: ReplanState,
    variant: WorkflowVariant,
    result: PlanRunResult,
    *,
    search_tree_dir: Path,
) -> None:
    variant.run_result = result
    variant.is_valid = result.success
    if result.total_cost_usd is not None:
        state.total_cost_usd += result.total_cost_usd
    if result.total_tokens is not None:
        state.total_tokens += result.total_tokens
    score_record = build_score_record(
        variant.plan_spec,
        result,
        plan_hash=variant.plan_hash or None,
    )
    backpropagate_variant(variant, score_record=score_record)
    append_tree_node(search_tree_dir, variant)
    if state.search_tree_path is None:
        state.search_tree_path = str(search_tree_dir / "tree.jsonl")


def _select_replan_search_variant(
    root: WorkflowVariant,
    *,
    selection_policy: MctsSelectionPolicy,
    debug_prob: float,
    exploration_constant: float,
    population_strategy: ReplanPopulationStrategy,
    tournament_size: int,
    elite_count: int,
    diversity_floor: float,
) -> WorkflowVariant | None:
    if population_strategy == "best":
        return select_expansion_parent(
            root,
            selection_policy=selection_policy,
            debug_prob=debug_prob,
            exploration_constant=exploration_constant,
        )

    leaves = [
        node
        for node in iter_variants(root)
        if not node.children and not node.pruned
    ]
    if not leaves:
        return None

    chooser = random.Random()
    elites = _select_replan_elites(leaves, elite_count=elite_count)
    non_elites = [node for node in leaves if node.node_id not in {elite.node_id for elite in elites}]
    required_contestants = max(tournament_size, len(elites))
    challengers = _select_replan_diverse_challengers(
        non_elites,
        anchors=elites,
        target_count=max(0, required_contestants - len(elites)),
        root=root,
        diversity_floor=diversity_floor,
        chooser=chooser,
    )
    contestants = elites + challengers
    if not contestants:
        contestants = leaves

    return select_variant_from_pool(
        contestants,
        selection_policy=selection_policy,
        debug_prob=debug_prob,
        exploration_constant=exploration_constant,
        rng=chooser,
    )


def _select_replan_elites(
    candidates: list[WorkflowVariant],
    *,
    elite_count: int,
) -> list[WorkflowVariant]:
    if elite_count <= 0 or not candidates:
        return []
    ranked = sorted(
        candidates,
        key=_replan_elite_sort_key,
        reverse=True,
    )
    return ranked[:elite_count]


def _select_replan_diverse_challengers(
    candidates: list[WorkflowVariant],
    *,
    anchors: list[WorkflowVariant],
    target_count: int,
    root: WorkflowVariant,
    diversity_floor: float,
    chooser: random.Random,
) -> list[WorkflowVariant]:
    if target_count <= 0 or not candidates:
        return []

    remaining = list(candidates)
    selected: list[WorkflowVariant] = []
    baseline_signature = _replan_variant_signature(root)

    while remaining and len(selected) < target_count:
        comparison_pool = anchors + selected
        eligible = _filter_replan_diverse_candidates(
            remaining,
            comparisons=comparison_pool,
            baseline_signature=baseline_signature,
            diversity_floor=diversity_floor,
        )
        pool = eligible or remaining
        chosen = chooser.choice(pool)
        selected.append(chosen)
        remaining = [node for node in remaining if node.node_id != chosen.node_id]

    return selected


def _filter_replan_diverse_candidates(
    candidates: list[WorkflowVariant],
    *,
    comparisons: list[WorkflowVariant],
    baseline_signature: set[str],
    diversity_floor: float,
) -> list[WorkflowVariant]:
    if diversity_floor <= 0.0 or not comparisons:
        return list(candidates)
    return [
        candidate
        for candidate in candidates
        if _replan_min_diversity_distance(
            candidate,
            comparisons=comparisons,
            baseline_signature=baseline_signature,
        ) >= diversity_floor
    ]


def _replan_min_diversity_distance(
    candidate: WorkflowVariant,
    *,
    comparisons: list[WorkflowVariant],
    baseline_signature: set[str],
) -> float:
    if not comparisons:
        return 1.0
    candidate_mutation = _replan_variant_mutation_signature(
        candidate,
        baseline_signature=baseline_signature,
    )
    return min(
        _replan_signature_distance(
            candidate_mutation,
            _replan_variant_mutation_signature(
                other,
                baseline_signature=baseline_signature,
            ),
        )
        for other in comparisons
    )


def _replan_variant_mutation_signature(
    variant: WorkflowVariant,
    *,
    baseline_signature: set[str],
) -> set[str]:
    return _replan_variant_signature(variant) ^ baseline_signature


def _replan_variant_signature(variant: WorkflowVariant) -> set[str]:
    plan_payload = variant.to_dict().get("plan_spec")
    if not isinstance(plan_payload, dict):
        return set()
    stable_payload = dict(plan_payload)
    stable_payload.pop("source_path", None)
    stable_payload.pop("validation_warnings", None)
    return _replan_signature(json.dumps(stable_payload, sort_keys=True, ensure_ascii=False))


def _replan_signature_distance(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 0.0
    return 1.0 - _replan_jaccard_similarity(left, right)


def _replan_elite_sort_key(variant: WorkflowVariant) -> tuple[int, int, float, int, int, str]:
    has_evidence = int(variant.run_result is not None or "score_record" in variant.metadata)
    return (
        has_evidence,
        int(variant.is_valid),
        variant.score,
        variant.visits,
        -_variant_depth_for_replan(variant),
        variant.node_id,
    )


def _variant_depth_for_replan(variant: WorkflowVariant) -> int:
    depth = 0
    current = variant.parent
    while current is not None:
        depth += 1
        current = current.parent
    return depth


def _index_tree_rows_by_plan_hash(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        plan_hash = row.get("plan_hash")
        if isinstance(plan_hash, str) and plan_hash:
            indexed[plan_hash] = row
    return indexed


def _detect_replan_duplicate(
    variant: WorkflowVariant,
    *,
    plan_name: str,
    source_dir: Path,
    tree_plan_hash_index: dict[str, dict[str, Any]],
    historical_score_cache: dict[str, ScoreRecord | None],
) -> _ReplanDuplicateDecision:
    existing_row = tree_plan_hash_index.get(variant.plan_hash)
    if existing_row is not None:
        run_result = existing_row.get("run_result")
        existing_run_id = ""
        if isinstance(run_result, dict):
            raw_run_id = run_result.get("run_id")
            if isinstance(raw_run_id, str):
                existing_run_id = raw_run_id
        return _ReplanDuplicateDecision(
            duplicate=True,
            source="search_tree",
            existing_node_id=str(existing_row.get("node_id") or ""),
            existing_run_id=existing_run_id,
        )

    cached = historical_score_cache.get(variant.plan_hash)
    if variant.plan_hash not in historical_score_cache:
        records = load_score_history(
            plan_name,
            source_dir,
            plan_hash=variant.plan_hash,
            limit=1,
        )
        cached = records[0] if records else None
        historical_score_cache[variant.plan_hash] = cached
    if cached is not None:
        return _ReplanDuplicateDecision(
            duplicate=True,
            source="score_history",
            existing_run_id=cached.run_id,
            score_record=cached,
        )

    return _ReplanDuplicateDecision(duplicate=False)


def _select_replan_knowledge_records(
    plan: PlanSpec,
    failed_state: dict[str, Any],
    *,
    plan_yaml: str,
    max_records: int = 6,
) -> list[KnowledgeRecord]:
    try:
        knowledge = load_knowledge(plan.name, plan.source_dir, max_per_task=None)
    except Exception:
        return []
    if not knowledge:
        return []

    failed_task_ids = [
        str(task_id)
        for task_id in failed_state.get("failed_task_ids", [])
        if isinstance(task_id, str) and task_id.strip()
    ]

    selected: list[KnowledgeRecord] = []
    seen: set[tuple[str, str, str]] = set()

    def _append_record(record: KnowledgeRecord) -> None:
        signature = (record.task_id, record.kind, record.insight)
        if signature in seen:
            return
        if record.kind not in _REPLAN_GUIDANCE_KINDS:
            return
        selected.append(record)
        seen.add(signature)

    for task_id in failed_task_ids:
        for record in knowledge.get(task_id, []):
            _append_record(record)

    if len(selected) < max_records:
        prompt_text = (
            f"failed tasks: {', '.join(failed_task_ids)}\n"
            f"errors: {json.dumps(failed_state.get('error_messages', {}), ensure_ascii=False)}\n"
            f"{plan_yaml}"
        )
        for record in select_relevant_knowledge(
            knowledge,
            prompt_text,
            max_records=max_records * 2,
        ):
            _append_record(record)
            if len(selected) >= max_records:
                break

    if not selected:
        return []

    try:
        alerts = record_knowledge_retrievals(
            plan.name,
            plan.source_dir,
            "\n".join(failed_task_ids) or plan_yaml,
            selected,
        )
    except Exception:
        alerts = []

    if alerts:
        alerted_signatures = {
            (alert.task_id, alert.kind, alert.insight)
            for alert in alerts
        }
        selected = [
            record
            for record in selected
            if (record.task_id, record.kind, record.insight) not in alerted_signatures
        ]

    return selected[:max_records]


def _build_replan_knowledge_guidance(records: list[KnowledgeRecord]) -> str:
    if not records:
        return ""
    return format_knowledge(records, include_task_id=True)


def _score_replan_variant_with_knowledge(
    candidate_yaml: str,
    records: list[KnowledgeRecord],
    *,
    baseline_plan_yaml: str,
    max_bonus: float = 0.08,
) -> tuple[float, list[dict[str, Any]]]:
    if not records:
        return 0.0, []

    candidate_tokens = _replan_tokens(candidate_yaml)
    baseline_tokens = _replan_tokens(baseline_plan_yaml)
    added_tokens = candidate_tokens - baseline_tokens
    if not added_tokens:
        return 0.0, []

    raw_score = 0.0
    signals: list[dict[str, Any]] = []

    for record in records:
        record_tokens = _replan_tokens(record.insight)
        if not record_tokens:
            continue
        overlap = added_tokens & record_tokens
        if not overlap:
            continue
        overlap_ratio = len(overlap) / len(record_tokens)
        confidence_weight = max(0.0, min(1.0, record.confidence))
        occurrence_weight = min(record.occurrences, 5) / 5.0
        direction = 1.0
        if record.kind == "model_pattern":
            lowered = record.insight.lower()
            if "fail" in lowered or "error" in lowered:
                direction = -1.0
        contribution = direction * overlap_ratio * (0.7 * confidence_weight + 0.3 * occurrence_weight)
        raw_score += contribution
        signals.append(
            {
                "task_id": record.task_id,
                "kind": record.kind,
                "matched_terms": sorted(overlap),
                "direction": "positive" if direction > 0 else "negative",
                "confidence": round(record.confidence, 3),
                "occurrences": record.occurrences,
                "contribution": round(contribution, 6),
            }
        )

    if not signals:
        return 0.0, []

    bonus = max(-max_bonus, min(max_bonus, raw_score * max_bonus))
    return round(bonus, 6), signals


def _score_replan_variant_with_history(
    variant: WorkflowVariant,
    records: list[ScoreRecord],
    *,
    max_bonus: float = _REPLAN_HISTORY_BOOTSTRAP_MAX_BONUS,
    min_similarity: float = _REPLAN_HISTORY_BOOTSTRAP_MIN_SIMILARITY,
    max_records: int = 4,
    horizon_days: int = 90,
    anchor_time: datetime | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    if not records:
        return 0.0, []

    candidate_signature = {
        term.strip().lower()
        for term in plan_topology_signature(variant.plan_spec)
        if term.strip()
    }
    if not candidate_signature:
        return 0.0, []

    anchor = anchor_time or datetime.now(timezone.utc)
    matched: list[tuple[float, float, ScoreRecord]] = []
    for record in records:
        if record.plan_hash == variant.plan_hash:
            continue
        record_signature = _score_record_signature(record)
        if not record_signature:
            continue
        similarity = _replan_jaccard_similarity(candidate_signature, record_signature)
        if similarity < min_similarity:
            continue
        matched.append(
            (
                similarity,
                score_record_fitness(record, anchor_time=anchor, horizon_days=horizon_days),
                record,
            )
        )

    if not matched:
        return 0.0, []

    matched.sort(
        key=lambda item: (item[0], item[1], item[2].timestamp, item[2].run_id),
        reverse=True,
    )
    selected = matched[:max_records]
    total_weight = sum(item[0] for item in selected)
    if total_weight <= 0.0:
        return 0.0, []  # pragma: no cover

    weighted_fitness = sum(
        similarity * fitness
        for similarity, fitness, _record in selected
    ) / total_weight
    average_similarity = sum(item[0] for item in selected) / len(selected)
    confidence = min(1.0, average_similarity * (len(selected) / max(1, max_records)))
    centered_fitness = (weighted_fitness - 0.5) * 2.0
    bonus = round(
        max(
            -max_bonus,
            min(max_bonus, max_bonus * centered_fitness * confidence),
        ),
        6,
    )

    nearest_similarity, nearest_fitness, nearest_record = selected[0]
    signals: list[dict[str, Any]] = [
        {
            "matched_records": len(selected),
            "best_similarity": round(nearest_similarity, 4),
            "avg_similarity": round(average_similarity, 4),
            "bootstrapped_fitness": round(weighted_fitness, 4),
            "confidence": round(confidence, 4),
            "nearest_run_id": nearest_record.run_id,
            "nearest_plan_hash": nearest_record.plan_hash,
            "nearest_fitness": round(nearest_fitness, 4),
        }
    ]
    return bonus, signals


def _score_replan_variant_novelty(
    candidate_yaml: str,
    *,
    baseline_plan_yaml: str,
    prior_tree_rows: list[dict[str, Any]],
    max_bonus: float = _REPLAN_NOVELTY_MAX_BONUS,
    max_prior_rows: int = 24,
) -> tuple[float, list[dict[str, Any]]]:
    baseline_signature = _replan_signature(baseline_plan_yaml)
    candidate_signature = _replan_signature(candidate_yaml)
    mutation_signature = candidate_signature ^ baseline_signature
    if not mutation_signature:
        return 0.0, []

    compared = 0
    best_similarity = 0.0
    nearest_row: dict[str, Any] | None = None
    for row in prior_tree_rows[-max_prior_rows:]:
        plan_spec = row.get("plan_spec")
        if not isinstance(plan_spec, dict):
            continue
        prior_signature = _replan_signature(
            json.dumps(plan_spec, sort_keys=True, ensure_ascii=False),
        )
        prior_mutation_signature = prior_signature ^ baseline_signature
        if not prior_mutation_signature:
            continue
        compared += 1
        similarity = _replan_jaccard_similarity(
            mutation_signature,
            prior_mutation_signature,
        )
        if similarity >= best_similarity:
            best_similarity = similarity
            nearest_row = row

    mutation_magnitude = len(mutation_signature) / max(1, len(candidate_signature))
    if compared:
        novelty_distance = 1.0 - best_similarity
    else:
        novelty_distance = min(1.0, mutation_magnitude * 4.0)
    bounded_magnitude = min(1.0, mutation_magnitude * 3.0)
    bonus = round(max(0.0, min(max_bonus, max_bonus * novelty_distance * bounded_magnitude)), 6)
    signals: list[dict[str, Any]] = [
        {
            "compared_variants": compared,
            "nearest_similarity": round(best_similarity, 4),
            "novelty_distance": round(novelty_distance, 4),
            "mutation_terms": sorted(mutation_signature)[:8],
        }
    ]
    if nearest_row is not None:
        nearest_signal = signals[0]
        nearest_signal["nearest_node_id"] = nearest_row.get("node_id")
        nearest_signal["nearest_plan_hash"] = nearest_row.get("plan_hash")
    return bonus, signals


def _replan_tokens(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in re.finditer(r"[a-z0-9_]{2,}", text)
    }


def _score_record_signature(record: ScoreRecord) -> set[str]:
    terms = record.metadata.get("plan_signature_terms")
    if not isinstance(terms, list):
        return set()
    return {
        str(term).strip().lower()
        for term in terms
        if isinstance(term, str) and str(term).strip()
    }


def _replan_signature(text: str) -> set[str]:
    signature = _replan_tokens(text)
    entities, _ = extract_entities(text, "replan-variant")
    for entity in entities:
        signature.add(f"{entity.entity_type}:{entity.name.lower()}")
    return signature


def _replan_jaccard_similarity(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    union = left | right
    if not union:  # pragma: no cover
        return 0.0
    return len(left & right) / len(union)


def _open_replan_audit_trail(plan_name: str, run_path: Path) -> _ReplanAuditTrail:
    events_path = run_path / "events.jsonl"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    events_path.touch(exist_ok=True)
    records = replay_events(events_path)
    if records:
        last = records[-1]
        sequence = last.sequence + 1
        prev_hash = last.event_hash or last.prev_hash
    else:
        sequence = 0
        prev_hash = "0" * 16
    return _ReplanAuditTrail(
        plan_name=plan_name,
        events_path=events_path,
        chain_state=ChainState(sequence=sequence, prev_hash=prev_hash),
    )


def _append_replan_audit_event(
    trail: _ReplanAuditTrail,
    event_name: str,
    *,
    event_callback: Callable[[str, dict[str, object]], None] | None,
    **data: object,
) -> None:
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event_name,
        "plan_name": trail.plan_name,
        **data,
    }
    record = emit_hashed_event(payload, trail.chain_state)
    payload["seq"] = record.sequence
    payload["prev_hash"] = record.prev_hash
    payload["hash"] = record.event_hash
    with trail.events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, default=str) + "\n")
        handle.flush()
    if event_callback is not None:
        try:
            event_callback(event_name, payload)
        except Exception:
            pass


def _evaluate_generated_plan_security(
    candidate_plan: PlanSpec,
    candidate_yaml: str,
    *,
    baseline_plan: PlanSpec,
    firewall_model: str | None,
    workdir: Path,
) -> _GeneratedPlanSecurityDecision:
    baseline_findings = {
        _audit_finding_fingerprint(finding)
        for finding in audit_plan(baseline_plan)
        if finding.severity in _BLOCKING_GENERATED_PLAN_SEVERITIES
    }
    introduced_findings = [
        finding
        for finding in audit_plan(candidate_plan)
        if (
            finding.severity in _BLOCKING_GENERATED_PLAN_SEVERITIES
            and _audit_finding_fingerprint(finding) not in baseline_findings
        )
    ]

    firewall_verdict = "allow"
    firewall_category = ""
    firewall_reason = ""
    resolved_firewall_model = (firewall_model or "").strip()
    if resolved_firewall_model and candidate_yaml.strip():
        try:
            firewall_decision = _run_firewall_pass2(
                resolved_firewall_model,
                "generated_plan_yaml",
                candidate_yaml,
                workdir=workdir,
            )
            firewall_verdict = firewall_decision.verdict
            firewall_category = firewall_decision.category
            firewall_reason = firewall_decision.reason
        except Exception:
            firewall_verdict = "allow"
            firewall_category = ""
            firewall_reason = ""

    blocked = bool(introduced_findings) or firewall_verdict in {"block", "rewrite"}
    return _GeneratedPlanSecurityDecision(
        blocked=blocked,
        introduced_findings=introduced_findings,
        firewall_verdict=firewall_verdict,
        firewall_category=firewall_category,
        firewall_reason=firewall_reason,
    )


def _audit_finding_fingerprint(finding: AuditFinding) -> tuple[str, str, str]:
    return (finding.rule, finding.task_id or "", finding.message)


def _format_security_gate_message(decision: _GeneratedPlanSecurityDecision) -> str:
    parts = ["Generated plan blocked by security gate"]
    if decision.introduced_findings:
        findings = ", ".join(
            f.rule if not f.task_id else f"{f.rule}:{f.task_id}"
            for f in decision.introduced_findings[:3]
        )
        if len(decision.introduced_findings) > 3:
            findings += ", ..."
        parts.append(f"new findings: {findings}")
    if decision.firewall_verdict in {"block", "rewrite"}:
        category = decision.firewall_category or "suspicious_content"
        parts.append(f"pass-2 verdict: {decision.firewall_verdict}/{category}")
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} ({'; '.join(parts[1:])})"


def _extract_failed_state(result: PlanRunResult) -> dict[str, Any]:
    failed_task_ids: list[str] = []
    passed_task_ids: list[str] = []
    error_messages: dict[str, str] = {}
    stdout_tails: dict[str, str] = {}

    for task_id, task_result in result.task_results.items():
        if task_result.status == "failed":
            failed_task_ids.append(task_id)
            message = (task_result.message or "").strip()
            error_messages[task_id] = message or f"exit_code={task_result.exit_code}"
            if task_result.stdout_tail:
                stdout_tails[task_id] = task_result.stdout_tail
        elif task_result.status in {"success", "soft_failed", "dry_run"}:
            passed_task_ids.append(task_id)

    return {
        "failed_task_ids": failed_task_ids,
        "passed_task_ids": passed_task_ids,
        "error_messages": error_messages,
        "stdout_tails": stdout_tails,
    }


def _build_analysis_prompt(
    plan_yaml: str,
    failed_state: dict[str, Any],
    *,
    candidate_number: int = 1,
    total_candidates: int = 1,
    knowledge_guidance: str = "",
) -> str:
    candidate_rules = ""
    if total_candidates > 1:
        candidate_rules = (
            f"4) This is candidate {candidate_number} of {total_candidates}. "
            "Produce a materially distinct correction strategy from the other candidates.\n"
            "5) Vary at least one of: verification flow, retry/timeout policy, task ordering, model choice, or prompt structure.\n"
        )
    knowledge_block = ""
    if knowledge_guidance.strip():
        knowledge_block = (
            "HISTORICAL KNOWLEDGE HINTS (advisory, derived from prior runs):\n"
            f"{knowledge_guidance.rstrip()}\n\n"
        )
    return (
        "You are fixing a failed Maestro execution plan.\n"
        "Analyze the failures and produce a corrected plan.\n"
        "Rules:\n"
        "1) Keep valid parts unchanged whenever possible.\n"
        "2) Address all failed tasks and dependency implications.\n"
        "3) Return the corrected plan in exactly one ```yaml fenced block.\n\n"
        f"{candidate_rules}"
        f"{knowledge_block}"
        "FAILED EXECUTION STATE:\n"
        f"{json.dumps(failed_state, indent=2, ensure_ascii=False)}\n\n"
        "ORIGINAL PLAN YAML:\n"
        "```yaml\n"
        f"{plan_yaml.rstrip()}\n"
        "```\n"
    )


def _call_analysis_model(prompt: str, model: str) -> str:
    try:
        completed = subprocess.run(
            ["claude", "--print", "--model", model, "-p", prompt],
            capture_output=True,
            text=True,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Analysis command timed out after 120s: {exc}") from exc
    except OSError as exc:
        raise RuntimeError(f"Failed to invoke analysis command: {exc}") from exc

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        raise RuntimeError(f"Analysis command failed (exit={completed.returncode}): {stderr}")

    stdout = (completed.stdout or "").strip()
    if not stdout:
        raise RuntimeError("Analysis command returned empty stdout")

    return stdout


def _parse_corrected_yaml(response: str) -> str | None:
    patterns = [
        r"```yaml\s*(.*?)```",
        r"```yml\s*(.*?)```",
        r"```\s*(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, flags=re.DOTALL | re.IGNORECASE)
        if match:
            parsed = match.group(1).strip()
            if parsed:
                return parsed
    return None


def _show_plan_diff(original: str, corrected: str, *, print_lines: bool = True) -> str:
    diff_lines = list(
        difflib.unified_diff(
            original.splitlines(),
            corrected.splitlines(),
            fromfile="original.yaml",
            tofile="corrected.yaml",
            lineterm="",
        )
    )
    diff_text = "\n".join(diff_lines)

    if not diff_text:
        if print_lines:
            print("[maestro] No plan changes detected.")
        return ""

    if print_lines:
        for line in diff_lines:
            print(f"[maestro] {line}")
    return diff_text


def _detect_exit_loop(attempts: list[ReplanAttempt]) -> bool:
    if len(attempts) < 2:
        return False

    previous = set(attempts[-2].failed_task_ids)
    latest = set(attempts[-1].failed_task_ids)
    return bool(previous) and previous == latest
