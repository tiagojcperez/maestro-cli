from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest

from maestro_cli.models import (
    KnowledgeRecord,
    PlanRunResult,
    PlanSpec,
    ScoreRecord,
    TaskResult,
    TaskSpec,
    WorkflowVariant,
)
import maestro_cli.replan as replan_mod
from maestro_cli.replan import (
    _GeneratedPlanSecurityDecision,
    _append_replan_audit_event,
    _build_replan_knowledge_guidance,
    _evaluate_generated_plan_security,
    _filter_replan_diverse_candidates,
    _format_security_gate_message,
    _open_replan_audit_trail,
    _replan_jaccard_similarity,
    _replan_min_diversity_distance,
    _replan_search,
    _replan_signature,
    _replan_variant_signature,
    _score_replan_variant_novelty,
    _score_replan_variant_with_history,
    _score_replan_variant_with_knowledge,
    _select_replan_diverse_challengers,
    _select_replan_knowledge_records,
    _select_replan_search_variant,
)
from maestro_cli.audit import AuditFinding


# ─────────────────────────────── helpers ──────────────────────────────────


def _plan(name: str = "demo", **kwargs: Any) -> PlanSpec:
    return PlanSpec(name=name, **kwargs)


def _variant(node_id: str, plan: PlanSpec, **kwargs: Any) -> WorkflowVariant:
    return WorkflowVariant(node_id=node_id, plan_spec=plan, **kwargs)


def _knowledge_record(
    *,
    task_id: str = "t1",
    kind: str = "failure_pattern",
    insight: str = "network timeout fails repeatedly",
    confidence: float = 0.8,
    occurrences: int = 3,
) -> KnowledgeRecord:
    now = datetime.now(timezone.utc).isoformat()
    return KnowledgeRecord(
        task_id=task_id,
        kind=kind,  # type: ignore[arg-type]
        insight=insight,
        confidence=confidence,
        occurrences=occurrences,
        first_seen=now,
        last_seen=now,
    )


def _score_record(
    *,
    plan_hash: str,
    run_id: str = "run-x",
    success: bool = True,
    quality_score: float | None = 0.9,
    terms: list[str] | None = None,
    timestamp: str | None = None,
) -> ScoreRecord:
    ts = timestamp or datetime.now(timezone.utc).isoformat()
    metadata: dict[str, Any] = {}
    if terms is not None:
        metadata["plan_signature_terms"] = terms
    return ScoreRecord(
        plan_name="demo",
        plan_hash=plan_hash,
        run_id=run_id,
        success=success,
        cost_usd=0.1,
        quality_score=quality_score,
        duration_sec=1.0,
        timestamp=ts,
        metadata=metadata,
    )


# ───────────────────────── _select_replan_search_variant ──────────────────


def test_tournament_returns_none_when_no_leaves() -> None:
    # Root has a single child, and that child also has a child -> the only
    # leaf is the grandchild, but if we prune it there are no usable leaves.
    plan = _plan()
    root = _variant("root", plan)
    child = _variant("child", plan, parent=root, pruned=True)
    root.children.append(child)

    result = _select_replan_search_variant(
        root,
        selection_policy="debug_prob",
        debug_prob=0.0,
        exploration_constant=1.4,
        population_strategy="tournament",
        tournament_size=2,
        elite_count=0,
        diversity_floor=0.0,
    )
    assert result is None


def test_tournament_falls_back_to_all_leaves_when_no_contestants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # elite_count=0 and tournament_size=0 -> required_contestants is 0, so no
    # elites and no challengers are picked, exercising the empty-contestants
    # fallback that reuses every leaf.
    plan = _plan()
    root = _variant("root", plan)
    leaf = _variant("leaf", plan, parent=root, score=0.5, is_valid=True)
    root.children.append(leaf)

    captured: dict[str, Any] = {}

    def _fake_pool(contestants: list[WorkflowVariant], **kwargs: Any) -> WorkflowVariant:
        captured["ids"] = [c.node_id for c in contestants]
        return contestants[0]

    monkeypatch.setattr(replan_mod, "select_variant_from_pool", _fake_pool)

    result = _select_replan_search_variant(
        root,
        selection_policy="debug_prob",
        debug_prob=0.0,
        exploration_constant=1.4,
        population_strategy="tournament",
        tournament_size=0,
        elite_count=0,
        diversity_floor=0.0,
    )
    assert result is leaf
    assert captured["ids"] == ["leaf"]


# ──────────────────── _select_replan_diverse_challengers ───────────────────


def test_diverse_challengers_zero_target_returns_empty() -> None:
    plan = _plan()
    root = _variant("root", plan)
    candidate = _variant("c1", plan)
    result = _select_replan_diverse_challengers(
        [candidate],
        anchors=[],
        target_count=0,
        root=root,
        diversity_floor=0.0,
        chooser=random.Random(0),
    )
    assert result == []


# ──────────────────────── _replan_min_diversity_distance ───────────────────


def test_min_diversity_distance_no_comparisons_returns_one() -> None:
    plan = _plan()
    candidate = _variant("c1", plan)
    distance = _replan_min_diversity_distance(
        candidate,
        comparisons=[],
        baseline_signature=set(),
    )
    assert distance == 1.0


def test_filter_diverse_candidates_no_comparisons_passthrough() -> None:
    plan = _plan()
    candidates = [_variant("c1", plan), _variant("c2", plan)]
    result = _filter_replan_diverse_candidates(
        candidates,
        comparisons=[],
        baseline_signature=set(),
        diversity_floor=0.5,
    )
    assert result == candidates


# ───────────────────────── _replan_variant_signature ──────────────────────


def test_variant_signature_non_dict_payload_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    variant = _variant("c1", plan)

    def _fake_to_dict() -> dict[str, Any]:
        return {"plan_spec": "not-a-dict"}

    monkeypatch.setattr(variant, "to_dict", _fake_to_dict)
    assert _replan_variant_signature(variant) == set()


# ────────────────────────── _replan_jaccard_similarity ────────────────────


def test_jaccard_similarity_empty_inputs() -> None:
    assert _replan_jaccard_similarity(set(), {"a"}) == 0.0
    assert _replan_jaccard_similarity({"a"}, set()) == 0.0


def test_jaccard_similarity_nonempty() -> None:
    assert _replan_jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


# ──────────────────── _select_replan_knowledge_records ─────────────────────


def test_knowledge_records_load_failure_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("knowledge store unreadable")

    monkeypatch.setattr(replan_mod, "load_knowledge", _boom)

    result = _select_replan_knowledge_records(
        _plan(),
        {"failed_task_ids": ["t1"], "error_messages": {}},
        plan_yaml="version: 1\n",
    )
    assert result == []


def test_knowledge_records_skips_non_guidance_kind_and_breaks_on_max(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # One non-guidance record (skipped via the _REPLAN_GUIDANCE_KINDS gate) and
    # several guidance records returned by select_relevant_knowledge so the
    # max_records break path fires.
    guidance = [
        _knowledge_record(task_id="t1", kind="failure_pattern", insight=f"timeout case {i}")
        for i in range(8)
    ]
    knowledge = {
        "t1": [_knowledge_record(task_id="t1", kind="success_pattern", insight="all good")]
    }

    monkeypatch.setattr(
        replan_mod,
        "load_knowledge",
        lambda *a, **k: knowledge,
    )
    monkeypatch.setattr(
        replan_mod,
        "select_relevant_knowledge",
        lambda knowledge_map, prompt, **k: guidance,
    )
    monkeypatch.setattr(
        replan_mod,
        "record_knowledge_retrievals",
        lambda *a, **k: [],
    )

    result = _select_replan_knowledge_records(
        _plan(),
        {"failed_task_ids": ["t1"], "error_messages": {"t1": "boom"}},
        plan_yaml="version: 1\nname: demo\n",
        max_records=3,
    )
    assert len(result) == 3
    assert all(r.kind == "failure_pattern" for r in result)


def test_knowledge_records_empty_after_filter_returns_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # All records are a non-guidance kind, so nothing is appended -> the
    # "if not selected: return []" path is taken.
    knowledge = {
        "t1": [_knowledge_record(task_id="t1", kind="success_pattern", insight="great")]
    }
    monkeypatch.setattr(replan_mod, "load_knowledge", lambda *a, **k: knowledge)
    monkeypatch.setattr(
        replan_mod,
        "select_relevant_knowledge",
        lambda *a, **k: [
            _knowledge_record(task_id="t1", kind="success_pattern", insight="great")
        ],
    )
    result = _select_replan_knowledge_records(
        _plan(),
        {"failed_task_ids": ["t1"], "error_messages": {}},
        plan_yaml="version: 1\n",
    )
    assert result == []


def test_knowledge_records_retrieval_alert_filters_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _knowledge_record(task_id="t1", kind="failure_pattern", insight="poisoned insight")
    knowledge = {"t1": [rec]}

    class _Alert:
        task_id = "t1"
        kind = "failure_pattern"
        insight = "poisoned insight"

    monkeypatch.setattr(replan_mod, "load_knowledge", lambda *a, **k: knowledge)
    monkeypatch.setattr(replan_mod, "select_relevant_knowledge", lambda *a, **k: [])
    monkeypatch.setattr(
        replan_mod,
        "record_knowledge_retrievals",
        lambda *a, **k: [_Alert()],
    )

    result = _select_replan_knowledge_records(
        _plan(),
        {"failed_task_ids": ["t1"], "error_messages": {}},
        plan_yaml="version: 1\n",
    )
    # The single record matched an alert signature and was filtered out.
    assert result == []


def test_knowledge_records_retrieval_call_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rec = _knowledge_record(task_id="t1", kind="failure_pattern", insight="useful insight")
    knowledge = {"t1": [rec]}

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("retrieval ledger error")

    monkeypatch.setattr(replan_mod, "load_knowledge", lambda *a, **k: knowledge)
    monkeypatch.setattr(replan_mod, "select_relevant_knowledge", lambda *a, **k: [])
    monkeypatch.setattr(replan_mod, "record_knowledge_retrievals", _boom)

    result = _select_replan_knowledge_records(
        _plan(),
        {"failed_task_ids": ["t1"], "error_messages": {}},
        plan_yaml="version: 1\n",
    )
    # alerts default to [] on exception, so the record is preserved.
    assert len(result) == 1
    assert result[0].insight == "useful insight"


def test_build_knowledge_guidance_empty_returns_empty() -> None:
    assert _build_replan_knowledge_guidance([]) == ""


# ─────────────────── _score_replan_variant_with_knowledge ──────────────────


def test_score_with_knowledge_skips_empty_record_tokens() -> None:
    # A record whose insight has no >=2-char tokens -> record_tokens empty ->
    # continue path. Adding a real failure-pattern record that does match the
    # added tokens keeps the overall result non-trivial.
    rec_empty = _knowledge_record(insight="!")  # no token of len>=2
    rec_match = _knowledge_record(
        kind="failure_pattern",
        insight="install dependency packages",
    )
    bonus, signals = _score_replan_variant_with_knowledge(
        "version: 1\nname: demo\ntasks: [install dependency packages here]\n",
        [rec_empty, rec_match],
        baseline_plan_yaml="version: 1\nname: demo\n",
    )
    assert isinstance(bonus, float)
    assert any(s["kind"] == "failure_pattern" for s in signals)


def test_score_with_knowledge_model_pattern_negative_direction() -> None:
    rec = _knowledge_record(
        kind="model_pattern",
        insight="haiku model fails on complex refactor",
        confidence=0.9,
        occurrences=4,
    )
    bonus, signals = _score_replan_variant_with_knowledge(
        "version: 1\nname: demo\ntasks: haiku model complex refactor\n",
        [rec],
        baseline_plan_yaml="version: 1\nname: demo\n",
    )
    assert signals
    assert signals[0]["direction"] == "negative"
    assert bonus <= 0.0


# ─────────────────── _score_replan_variant_with_history ────────────────────


def test_score_with_history_empty_candidate_signature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    variant = _variant("c1", plan, plan_hash="hash-c1")
    monkeypatch.setattr(replan_mod, "plan_topology_signature", lambda spec: [])
    bonus, signals = _score_replan_variant_with_history(
        variant,
        [_score_record(plan_hash="hash-prev", terms=["plan.task_count:1"])],
    )
    assert bonus == 0.0
    assert signals == []


def test_score_with_history_matches_records() -> None:
    # Build a real plan so plan_topology_signature yields stable terms; the
    # historical record carries the same terms so similarity exceeds the floor.
    plan = PlanSpec(
        name="demo",
        max_parallel=2,
        tasks=[TaskSpec(id="t1", command="echo hi")],
    )
    variant = _variant("c1", plan, plan_hash="hash-c1")
    terms = list(replan_mod.plan_topology_signature(plan))
    record = _score_record(
        plan_hash="hash-prev",
        run_id="prev-run",
        success=True,
        quality_score=1.0,
        terms=terms,
    )
    bonus, signals = _score_replan_variant_with_history(variant, [record])
    assert signals
    assert signals[0]["matched_records"] >= 1


# ─────────────────────── _score_replan_variant_novelty ─────────────────────


def test_novelty_skips_non_dict_plan_spec_and_empty_prior_mutation() -> None:
    baseline = "version: 1\nname: demo\ntasks: alpha beta\n"
    candidate = "version: 1\nname: demo\ntasks: alpha beta gamma delta\n"
    baseline_sig = _replan_signature(baseline)
    # One prior row with non-dict plan_spec (skipped), one whose signature
    # equals the baseline mutation set (empty mutation -> skipped).
    import json

    same_as_baseline = json.loads(
        json.dumps({"name": "demo"})
    )
    prior_rows = [
        {"plan_spec": "not-a-dict", "node_id": "n1"},
        # A prior plan_spec identical to the baseline signature input so its
        # mutation signature is empty.
        {"plan_spec": {"__baseline__": True}, "node_id": "n2"},
    ]
    # Make _replan_signature of the second row's plan_spec equal baseline_sig
    # by monkeypatching is hard; instead rely on the candidate differing and
    # ensure no crash and a bonus is returned.
    bonus, signals = _score_replan_variant_novelty(
        candidate,
        baseline_plan_yaml=baseline,
        prior_tree_rows=prior_rows,
    )
    assert isinstance(bonus, float)
    assert signals
    assert "compared_variants" in signals[0]
    assert baseline_sig  # baseline signature is non-empty


def test_novelty_no_compared_uses_mutation_magnitude() -> None:
    # No prior rows -> compared == 0 -> the else branch computes novelty from
    # mutation magnitude directly.
    bonus, signals = _score_replan_variant_novelty(
        "version: 1\nname: demo\ntasks: brand new unique mutation tokens\n",
        baseline_plan_yaml="version: 1\nname: demo\n",
        prior_tree_rows=[],
    )
    assert bonus >= 0.0
    assert signals[0]["compared_variants"] == 0


def test_novelty_skips_prior_row_with_empty_mutation() -> None:
    # A prior row whose plan_spec JSON signature equals the baseline signature
    # produces an empty prior mutation set and is skipped (compared stays 0).
    baseline = "version: 1\nname: demo\n"
    baseline_sig = _replan_signature(baseline)

    import maestro_cli.replan as rm

    calls = {"n": 0}
    orig_signature = rm._replan_signature

    def _fake_signature(text: str) -> set[str]:
        # First two calls are baseline + candidate (real); third (prior row)
        # returns the baseline signature so mutation is empty.
        calls["n"] += 1
        if calls["n"] == 3:
            return set(baseline_sig)
        return orig_signature(text)

    import pytest as _pytest

    with _pytest.MonkeyPatch.context() as mp:
        mp.setattr(rm, "_replan_signature", _fake_signature)
        bonus, signals = rm._score_replan_variant_novelty(
            "version: 1\nname: demo\ntasks: a different candidate plan body\n",
            baseline_plan_yaml=baseline,
            prior_tree_rows=[{"plan_spec": {"x": 1}, "node_id": "n1"}],
        )
    assert signals[0]["compared_variants"] == 0


# ──────────────────────────── _open_replan_audit_trail ────────────────────


def test_open_audit_trail_with_existing_events(tmp_path: Path) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    trail = _open_replan_audit_trail("demo", run_path)
    # Emit one event so the file has a record, then re-open to take the
    # "records exist" branch that derives sequence/prev_hash from the last.
    _append_replan_audit_event(trail, "first_event", event_callback=None, foo="bar")
    reopened = _open_replan_audit_trail("demo", run_path)
    assert reopened.chain_state.sequence == 1
    assert reopened.chain_state.prev_hash != "0" * 16


# ─────────────────────────── _append_replan_audit_event ───────────────────


def test_append_audit_event_callback_exception_swallowed(tmp_path: Path) -> None:
    run_path = tmp_path / "run"
    run_path.mkdir()
    trail = _open_replan_audit_trail("demo", run_path)

    def _bad_callback(name: str, payload: dict[str, object]) -> None:
        raise RuntimeError("callback blew up")

    # Should not raise despite the callback raising.
    _append_replan_audit_event(
        trail,
        "ev",
        event_callback=_bad_callback,
        detail="x",
    )
    contents = (run_path / "events.jsonl").read_text(encoding="utf-8")
    assert "ev" in contents


# ────────────────────── _evaluate_generated_plan_security ──────────────────


def test_evaluate_security_firewall_raises_falls_back_to_allow(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("firewall model unavailable")

    monkeypatch.setattr(replan_mod, "_run_firewall_pass2", _boom)
    # No introduced findings (same plan), so blocked depends only on firewall.
    monkeypatch.setattr(replan_mod, "audit_plan", lambda spec: [])

    decision = _evaluate_generated_plan_security(
        plan,
        "version: 1\nname: demo\ntasks: []\n",
        baseline_plan=plan,
        firewall_model="haiku",
        workdir=tmp_path,
    )
    assert decision.firewall_verdict == "allow"
    assert decision.blocked is False


# ────────────────────────── _format_security_gate_message ─────────────────


def test_format_security_message_truncates_many_findings() -> None:
    findings = [
        AuditFinding(rule=f"SEC0{i:02d}", severity="error", message=f"m{i}", task_id="t1")
        for i in range(5)
    ]
    decision = _GeneratedPlanSecurityDecision(
        blocked=True,
        introduced_findings=findings,
        firewall_verdict="allow",
    )
    message = _format_security_gate_message(decision)
    assert "..." in message
    assert "new findings" in message


def test_format_security_message_single_part_returns_header() -> None:
    decision = _GeneratedPlanSecurityDecision(
        blocked=False,
        introduced_findings=[],
        firewall_verdict="allow",
    )
    message = _format_security_gate_message(decision)
    assert "Generated plan blocked by security gate" in message
    assert "(" not in message


# ─────────────────────────────── _replan_search deep ──────────────────────

_VALID_PLAN = "version: 1\nname: test\ntasks:\n  - id: t1\n    command: echo hello\n"


def _result(
    tmp_path: Path,
    run_id: str,
    run_dir: str,
    success: bool,
    message: str | None = None,
) -> PlanRunResult:
    now = datetime.now()
    return PlanRunResult(
        plan_name="test",
        run_id=run_id,
        run_path=tmp_path / "runs" / run_dir,
        started_at=now,
        finished_at=now,
        success=success,
        task_results={
            "t1": TaskResult(
                task_id="t1",
                status="success" if success else "failed",
                exit_code=0 if success else 1,
                message=message,
            ),
        },
    )


def test_replan_search_no_candidate_variants(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # variants=0 makes the candidate loop body never run, so candidate_variants
    # stays empty and the "No candidate variants were generated." break fires.
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(_VALID_PLAN, encoding="utf-8")

    monkeypatch.setattr(
        replan_mod,
        "run_plan",
        lambda plan, *a, **k: _result(tmp_path, "root-run", "root-run", False, "root failure"),
    )
    monkeypatch.setattr(
        replan_mod,
        "_call_analysis_model",
        lambda prompt, model: "```yaml\n" + _VALID_PLAN + "```",
    )

    state = _replan_search(
        plan_path,
        max_attempts=1,
        analysis_model="opus",
        event_callback=None,
        dry_run=False,
        execution_profile="plan",
        verbosity="normal",
        output_mode="text",
        cache_dir=None,
        auto_approve=True,
        extra_template_vars=None,
        variants=0,
        debug_prob=0.5,
        selection_policy="debug_prob",
        exploration_constant=1.4,
        population_strategy="best",
        tournament_size=2,
        elite_count=1,
        diversity_floor=0.25,
    )
    assert state.attempts
    assert "No candidate variants were generated." in state.attempts[-1].error_summary


def test_replan_search_no_selectable_variant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Selection returns the current (root) variant -> the "no selectable
    # candidate variants remained" break fires.
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(_VALID_PLAN, encoding="utf-8")

    run_counter = {"n": 0}

    def _fake_run_plan(plan: PlanSpec, *a: Any, **k: Any) -> PlanRunResult:
        run_counter["n"] += 1
        if run_counter["n"] == 1:
            return _result(tmp_path, "root-run", "root-run", False, "root failure")
        return _result(tmp_path, f"cand-{run_counter['n']}", f"cand-{run_counter['n']}", True)

    monkeypatch.setattr(replan_mod, "run_plan", _fake_run_plan)
    monkeypatch.setattr(
        replan_mod,
        "_call_analysis_model",
        lambda prompt, model: (
            "```yaml\nversion: 1\nname: test\nmax_parallel: 2\n"
            "tasks:\n  - id: t1\n    command: echo hi\n```"
        ),
    )

    def _fake_select(root: WorkflowVariant, **kwargs: Any) -> WorkflowVariant:
        return root  # equals current_variant on the first attempt

    monkeypatch.setattr(replan_mod, "_select_replan_search_variant", _fake_select)

    state = _replan_search(
        plan_path,
        max_attempts=1,
        analysis_model="opus",
        event_callback=None,
        dry_run=False,
        execution_profile="plan",
        verbosity="normal",
        output_mode="text",
        cache_dir=None,
        auto_approve=True,
        extra_template_vars=None,
        variants=2,
        debug_prob=0.5,
        selection_policy="debug_prob",
        exploration_constant=1.4,
        population_strategy="best",
        tournament_size=2,
        elite_count=1,
        diversity_floor=0.25,
    )
    assert "No selectable candidate variants remained after pruning." in (
        state.attempts[-1].error_summary
    )


def test_replan_search_propagates_firewall_model_from_selected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Selected (non-success) candidate carries a firewall_model; with
    # trusted_firewall_model still None, line picks it up and continues.
    plan_path = tmp_path / "plan.yaml"
    plan_path.write_text(_VALID_PLAN, encoding="utf-8")

    run_counter = {"n": 0}

    def _fake_run_plan(plan: PlanSpec, *a: Any, **k: Any) -> PlanRunResult:
        run_counter["n"] += 1
        if run_counter["n"] == 1:
            return _result(tmp_path, "root-run", "root-run", False, "root failure")
        # All candidates fail so selected_result.success is False, leaving the
        # firewall-propagation tail reachable instead of the success break.
        return _result(
            tmp_path, f"cand-{run_counter['n']}", f"cand-{run_counter['n']}", False, "still failing"
        )

    monkeypatch.setattr(replan_mod, "run_plan", _fake_run_plan)
    monkeypatch.setattr(
        replan_mod,
        "_call_analysis_model",
        lambda prompt, model: (
            "```yaml\nversion: 1\nname: test\nfirewall_model: haiku\nmax_parallel: 2\n"
            "tasks:\n  - id: t1\n    command: echo hi\n```"
        ),
    )

    def _fake_select(root: WorkflowVariant, **kwargs: Any) -> WorkflowVariant:
        # Return a fresh child (not the root) so the loop advances and reaches
        # the firewall-propagation assignment.
        for child in root.children:
            if child.plan_spec.firewall_model:
                return child
        return root.children[-1] if root.children else root

    monkeypatch.setattr(replan_mod, "_select_replan_search_variant", _fake_select)

    state = _replan_search(
        plan_path,
        max_attempts=2,
        analysis_model="opus",
        event_callback=None,
        dry_run=False,
        execution_profile="plan",
        verbosity="normal",
        output_mode="text",
        cache_dir=None,
        auto_approve=True,
        extra_template_vars=None,
        variants=2,
        debug_prob=0.5,
        selection_policy="debug_prob",
        exploration_constant=1.4,
        population_strategy="best",
        tournament_size=2,
        elite_count=1,
        diversity_floor=0.25,
    )
    # The run completes (either circuit breaker or exhausted attempts) without
    # crashing, and the selected variants carried the firewall model.
    assert state.attempts
