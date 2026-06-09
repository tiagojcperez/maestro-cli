from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .cache import (
    SIMULATION_CACHE_POLICY_VERSION,
    compute_plan_hash,
    compute_simulation_plan_hash,
    simulation_model_families,
)
from .fts import relevance_by_rank
from .models import (
    HistoricalPruningDecision,
    KnowledgeRecord,
    KnowledgeWriteOutcome,
    PlanRunResult,
    PlanSpec,
    ScoreRecord,
)

if TYPE_CHECKING:
    from .memory import RetrievalDominanceAlert

_KNOWLEDGE_DIR = ".maestro-cache/knowledge"
_CONFIDENCE_DECAY_HALF_LIFE_DAYS = 30.0
_MAX_RECORDS_PER_TASK = 5
# Scale factor that lifts the FTS5 rank-position relevance (0.0-1.0) into the
# same magnitude band as the in-Python BM25 base score, so the domain boosts
# below (task match +2.0, confidence, occurrences) stay proportionate whichever
# lexical ranker is in play.
_FTS_RELEVANCE_SCALE = 5.0
_INITIAL_CONFIDENCE = 0.5
_CONFIDENCE_PER_OCCURRENCE = 0.1
_MAX_CONFIDENCE = 1.0
_KNOWLEDGE_INDEX_MAX_RECORDS = 200
_KNOWLEDGE_INDEX_SUMMARY_MAX_CHARS = 120
_KNOWLEDGE_KEYWORD_RE = re.compile(r"[a-z0-9_]{2,}", re.IGNORECASE)
_KNOWLEDGE_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can", "need", "to", "of",
    "in", "for", "on", "with", "at", "by", "from", "as", "into", "after",
    "before", "over", "under", "again", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "not", "only", "own",
    "same", "so", "than", "too", "very", "just", "because", "but", "and",
    "or", "if", "while", "that", "this", "these", "those", "it", "its",
    "my", "your", "his", "her", "our", "their", "what", "which", "who",
    "whom", "use", "using", "used",
})

# Remediation advice per failure category (mirrors suggest.py)
_FAILURE_REMEDIATION: dict[str, str] = {
    "timeout": "Increase timeout_sec or split the task into smaller parts.",
    "compilation_error": "Add verify_command to check syntax before committing.",
    "test_failure": "Add guard_command or verify_command with targeted test suite.",
    "permission_error": "Check file permissions or add pre_command to verify access.",
    "validation_error": "Add guard_command to validate output format.",
    "context_exceeded": "Reduce context_budget_tokens or use context_mode: layered.",
    "rate_limited": "Add retry_delay_sec or reduce max_parallel.",
    "runtime_error": "Add verify_command with targeted assertions.",
    "dependency_missing": "Add pre_command to verify required tools.",
    "output_format_error": "Add guard_command or use judge type: json-schema.",
    "cascading_failure": "Add context_budget_tokens to limit upstream propagation.",
    "miscommunication": "Improve prompt clarity with description field.",
    "role_confusion": "Scope agent role more narrowly.",
    "verification_gap": "Strengthen verify_command assertions.",
}


def _insight_key(insight: str) -> str:
    """Stable hash of the first 100 chars of an insight for dedup."""
    return hashlib.sha256(insight[:100].encode("utf-8")).hexdigest()[:12]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _apply_decay(confidence: float, last_seen: str) -> float:
    """Apply time-decay to a confidence value."""
    try:
        ts = datetime.fromisoformat(last_seen)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
        decay = 0.5 ** (age_days / _CONFIDENCE_DECAY_HALF_LIFE_DAYS)
        return float(max(0.0, min(1.0, confidence * decay)))
    except (ValueError, TypeError):
        return confidence


def _sort_records(records: list[KnowledgeRecord]) -> list[KnowledgeRecord]:
    """Sort records by usefulness for prompt priming."""
    return sorted(
        records,
        key=lambda r: (
            r.confidence,
            r.occurrences,
            r.last_seen,
            r.task_id,
            r.kind,
        ),
        reverse=True,
    )


def _flatten_knowledge(
    knowledge: dict[str, list[KnowledgeRecord]],
) -> list[KnowledgeRecord]:
    return [rec for records in knowledge.values() for rec in records]


def _extract_keywords(text: str) -> set[str]:
    return {
        match.group(0).lower()
        for match in _KNOWLEDGE_KEYWORD_RE.finditer(text)
        if match.group(0).lower() not in _KNOWLEDGE_STOPWORDS
    }


def _tokenize_words(text: str) -> list[str]:
    return [match.group(0).lower() for match in _KNOWLEDGE_KEYWORD_RE.finditer(text)]


def _knowledge_record_text(record: KnowledgeRecord) -> str:
    return f"{record.task_id} {record.kind} {record.insight}"


def _knowledge_fts_enabled() -> bool:
    """Whether to use the SQLite FTS5 lexical ranker for knowledge retrieval.

    On by default (when the sqlite3 build supports FTS5); set
    ``MAESTRO_KNOWLEDGE_FTS=0`` to force the legacy in-Python BM25 ranker, e.g.
    to reproduce pre-FTS5 ranking exactly.
    """
    return os.environ.get("MAESTRO_KNOWLEDGE_FTS", "1").strip().lower() not in {
        "0",
        "false",
        "no",
        "off",
    }


def _compute_idf(documents: list[str]) -> dict[str, float]:
    if not documents:
        return {}

    doc_freq: dict[str, int] = {}
    for document in documents:
        for term in _extract_keywords(document):
            doc_freq[term] = doc_freq.get(term, 0) + 1

    total_docs = len(documents)
    return {
        term: math.log(total_docs / (1 + freq))
        for term, freq in doc_freq.items()
    }


def _score_document(
    document: str,
    intent_keywords: set[str],
    *,
    idf: dict[str, float] | None,
    avg_doc_len: float,
) -> float:
    if not document or not intent_keywords:
        return 0.0

    if idf is None:
        return float(len(_extract_keywords(document) & intent_keywords))

    words = _tokenize_words(document)
    if not words:
        return 0.0

    term_counts: dict[str, int] = {}
    for word in words:
        term_counts[word] = term_counts.get(word, 0) + 1

    doc_len = len(words)
    norm_avg_len = avg_doc_len if avg_doc_len > 0 else 1.0
    k1 = 1.5
    b = 0.75
    score = 0.0

    for term in intent_keywords:
        tf = term_counts.get(term, 0)
        if tf <= 0:
            continue
        tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * doc_len / norm_avg_len))
        score += idf.get(term, 1.0) * tf_norm
    return score


def _one_line_summary(text: str, max_chars: int = _KNOWLEDGE_INDEX_SUMMARY_MAX_CHARS) -> str:
    summary = " ".join(text.split())
    if len(summary) <= max_chars:
        return summary
    return summary[: max_chars - 3].rstrip() + "..."


# ---------------------------------------------------------------------------
# Extract knowledge from a completed run
# ---------------------------------------------------------------------------

def extract_knowledge(run_result: PlanRunResult) -> list[KnowledgeRecord]:
    """Extract learnable patterns from a completed run result."""
    records: list[KnowledgeRecord] = []
    now = _now_iso()

    for task_id, result in run_result.task_results.items():
        if result.status in ("skipped", "dry_run"):
            continue

        # Failure pattern: task failed with a classified failure category
        if result.status == "failed" and result.failure_history:
            for fr in result.failure_history:
                cat = fr.category
                if cat == "unknown":
                    continue
                remediation = _FAILURE_REMEDIATION.get(cat, "")
                insight = f"Fails with {cat}"
                if fr.exit_code is not None:
                    insight += f" (exit_code={fr.exit_code})"
                if remediation:
                    insight += f". {remediation}"
                records.append(KnowledgeRecord(
                    task_id=task_id,
                    kind="failure_pattern",
                    insight=insight,
                    confidence=_INITIAL_CONFIDENCE,
                    occurrences=1,
                    first_seen=now,
                    last_seen=now,
                ))

        # Timeout hint: task timed out
        if result.exit_code == 124:
            insight = (
                f"Times out (ran {result.duration_sec:.0f}s). "
                f"Consider increasing timeout_sec or splitting the task."
            )
            records.append(KnowledgeRecord(
                task_id=task_id,
                kind="timeout_hint",
                insight=insight,
                confidence=_INITIAL_CONFIDENCE,
                occurrences=1,
                first_seen=now,
                last_seen=now,
            ))

        # Success pattern: clean success with judge pass and high score
        if (
            result.status == "success"
            and result.retry_count == 0
            and result.judge_result is not None
            and result.judge_result.verdict == "pass"
            and result.judge_result.overall_score > 0.9
        ):
            insight = (
                f"Reliably succeeds with clean pass "
                f"(judge score {result.judge_result.overall_score:.2f})."
            )
            records.append(KnowledgeRecord(
                task_id=task_id,
                kind="success_pattern",
                insight=insight,
                confidence=_INITIAL_CONFIDENCE,
                occurrences=1,
                first_seen=now,
                last_seen=now,
            ))

        # Cost pattern: track cost per task (helps model routing decisions)
        if result.cost_usd is not None and result.cost_usd > 0 and result.status == "success":
            insight = f"Costs ${result.cost_usd:.4f} on success."
            records.append(KnowledgeRecord(
                task_id=task_id,
                kind="cost_pattern",
                insight=insight,
                confidence=_INITIAL_CONFIDENCE,
                occurrences=1,
                first_seen=now,
                last_seen=now,
            ))

        # Duration pattern: track duration for timeout estimation
        if result.duration_sec and result.duration_sec > 10 and result.status in ("success", "failed"):
            insight = f"Takes {result.duration_sec:.0f}s ({result.status})."
            records.append(KnowledgeRecord(
                task_id=task_id,
                kind="duration_pattern",
                insight=insight,
                confidence=_INITIAL_CONFIDENCE,
                occurrences=1,
                first_seen=now,
                last_seen=now,
            ))

        # Retry pattern: track retry behaviour
        if result.retry_count > 0:
            outcome = "succeeded after" if result.status == "success" else "failed after"
            insight = f"Needed retries: {outcome} {result.retry_count} retries."
            records.append(KnowledgeRecord(
                task_id=task_id,
                kind="retry_pattern",
                insight=insight,
                confidence=_INITIAL_CONFIDENCE,
                occurrences=1,
                first_seen=now,
                last_seen=now,
            ))

        # Model pattern: track model effectiveness
        if hasattr(result, "auto_routed_model") and result.auto_routed_model:
            model = result.auto_routed_model
            outcome = "success" if result.status == "success" else result.status
            insight = f"Model {model}: {outcome}."
            records.append(KnowledgeRecord(
                task_id=task_id,
                kind="model_pattern",
                insight=insight,
                confidence=_INITIAL_CONFIDENCE,
                occurrences=1,
                first_seen=now,
                last_seen=now,
            ))

        # Meta-Policy Reflexion: extract persistent rules from repeated failures
        # When a task fails with judge verdict + specific criteria, create a rule
        if (
            result.status == "failed"
            and result.judge_result is not None
            and result.judge_result.verdict == "fail"
        ):
            reasoning = result.judge_result.reasoning or ""
            # Extract the failure reason as a policy rule
            if len(reasoning) > 20:
                rule = reasoning[:200].strip()
                insight = (
                    f"POLICY RULE: When running this task, ensure: {rule}. "
                    f"Previous judge score: {result.judge_result.overall_score:.2f}."
                )
                records.append(KnowledgeRecord(
                    task_id=task_id,
                    kind="policy_rule",
                    insight=insight,
                    confidence=_INITIAL_CONFIDENCE,
                    occurrences=1,
                    first_seen=now,
                    last_seen=now,
                ))

        # Meta-Policy: extract rules from tasks that failed then succeeded on retry
        if (
            result.status == "success"
            and result.retry_count > 0
            and result.failure_history
        ):
            # The last failure before success is the most informative
            last_failure = result.failure_history[-1]
            if last_failure.category != "unknown" and last_failure.message:
                fix_hint = last_failure.message[:150].strip()
                insight = (
                    f"POLICY RULE: Watch for {last_failure.category}. "
                    f"Previously failed with: {fix_hint}. "
                    f"Succeeded after {result.retry_count} retries."
                )
                records.append(KnowledgeRecord(
                    task_id=task_id,
                    kind="policy_rule",
                    insight=insight,
                    confidence=_INITIAL_CONFIDENCE + 0.1,  # slightly higher — confirmed fix
                    occurrences=1,
                    first_seen=now,
                    last_seen=now,
                ))

    return records


# ---------------------------------------------------------------------------
# Store / merge knowledge
# ---------------------------------------------------------------------------

def _knowledge_path(plan_name: str, source_dir: Path) -> Path:
    return source_dir / _KNOWLEDGE_DIR / f"{plan_name}.jsonl"


def _load_raw(path: Path) -> list[KnowledgeRecord]:
    """Load all records from a JSONL file."""
    if not path.is_file():
        return []
    records: list[KnowledgeRecord] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                records.append(KnowledgeRecord(
                    task_id=d["task_id"],
                    kind=d["kind"],
                    insight=d["insight"],
                    confidence=float(d.get("confidence", _INITIAL_CONFIDENCE)),
                    occurrences=int(d.get("occurrences", 1)),
                    first_seen=d.get("first_seen", ""),
                    last_seen=d.get("last_seen", ""),
                ))
            except (KeyError, TypeError, ValueError):
                continue
    except OSError:
        pass
    return records


def store_knowledge(
    plan_name: str,
    source_dir: Path,
    new_records: list[KnowledgeRecord],
) -> None:
    """Merge new records into the knowledge store."""
    store_knowledge_detailed(plan_name, source_dir, new_records)


def _normalized_source_id(source_type: str, source_id: str, task_id: str) -> str:
    """Attach task ids to task-scoped provenance pointers when available."""
    if source_type != "task" or not source_id:
        return source_id
    suffix = f":{task_id}"
    if source_id.endswith(suffix):
        return source_id
    return f"{source_id}{suffix}"


def store_knowledge_detailed(
    plan_name: str,
    source_dir: Path,
    new_records: list[KnowledgeRecord],
    *,
    source_type: str = "task",
    source_id: str = "",
) -> list[KnowledgeWriteOutcome]:
    """Merge new records into the knowledge store.

    Delegates to :mod:`memory` (SQLite backend) when available, with
    JSONL fallback for backward compatibility.
    """
    try:
        from .memory import store_records_detailed

        return store_records_detailed(
            plan_name,
            source_dir,
            new_records,
            source_type=source_type,
            source_id=source_id,
        )
    except Exception:
        pass

    # JSONL fallback
    if not new_records:
        return []

    path = _knowledge_path(plan_name, source_dir)
    existing = _load_raw(path)

    index: dict[str, KnowledgeRecord] = {}
    for rec in existing:
        key = f"{rec.task_id}:{rec.kind}:{_insight_key(rec.insight)}"
        index[key] = rec

    outcomes: list[KnowledgeWriteOutcome] = []
    for rec in new_records:
        key = f"{rec.task_id}:{rec.kind}:{_insight_key(rec.insight)}"
        operation = "inserted"
        if key in index:
            old = index[key]
            old.occurrences += 1
            old.last_seen = rec.last_seen
            old.confidence = min(
                _INITIAL_CONFIDENCE + old.occurrences * _CONFIDENCE_PER_OCCURRENCE,
                _MAX_CONFIDENCE,
            )
            operation = "merged"
        else:
            index[key] = rec

        outcomes.append(
            KnowledgeWriteOutcome(
                task_id=rec.task_id,
                kind=rec.kind,
                operation=operation,
                outcome="accepted",
                trust_label="trusted" if source_type == "task" else "untrusted",
                instructionality_score=0.0,
                source_type=source_type,
                source_id=_normalized_source_id(source_type, source_id, rec.task_id),
            )
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in index.values()]
    path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")
    return outcomes


# ---------------------------------------------------------------------------
# Load knowledge for prompt injection
# ---------------------------------------------------------------------------

def load_knowledge(
    plan_name: str,
    source_dir: Path,
    *,
    max_per_task: int | None = _MAX_RECORDS_PER_TASK,
    include_quarantined: bool = False,
) -> dict[str, list[KnowledgeRecord]]:
    """Load knowledge for a plan, grouped by task_id.

    Delegates to :mod:`memory` (SQLite backend) when available, with
    JSONL fallback for backward compatibility.
    """
    try:
        from .memory import load_records
        return load_records(
            plan_name,
            source_dir,
            max_per_task=max_per_task,
            include_quarantined=include_quarantined,
        )
    except Exception:
        pass

    # JSONL fallback
    path = _knowledge_path(plan_name, source_dir)
    raw = _load_raw(path)

    for rec in raw:
        rec.confidence = _apply_decay(rec.confidence, rec.last_seen)

    grouped: dict[str, list[KnowledgeRecord]] = {}
    for rec in raw:
        grouped.setdefault(rec.task_id, []).append(rec)

    for task_id in grouped:
        grouped[task_id].sort(key=lambda r: r.confidence, reverse=True)
        if max_per_task is not None:
            grouped[task_id] = grouped[task_id][:max_per_task]

    return grouped


def build_knowledge_index(
    plan_name: str,
    knowledge: dict[str, list[KnowledgeRecord]],
    *,
    max_records: int = _KNOWLEDGE_INDEX_MAX_RECORDS,
) -> str:
    """Build a lightweight index suitable for ``{{ knowledge_index }}``.

    The index stays compact: one line per record with plan name once at the
    top, then task pattern, category, and a one-line summary.
    """
    flat = _sort_records(_flatten_knowledge(knowledge))
    if not flat or max_records <= 0:
        return ""

    lines = [f"Plan: {plan_name}"]
    for rec in flat[:max_records]:
        kind_label = _KIND_ICONS.get(rec.kind, rec.kind.upper())
        lines.append(
            f"- [task={rec.task_id}] [{kind_label}] {_one_line_summary(rec.insight)}"
        )
    return "\n".join(lines)


def select_relevant_knowledge(
    knowledge: dict[str, list[KnowledgeRecord]],
    prompt_text: str,
    *,
    task_id: str | None = None,
    max_records: int = _MAX_RECORDS_PER_TASK,
) -> list[KnowledgeRecord]:
    """Select the most relevant records for a downstream task prompt.

    Uses lightweight BM25-style scoring over record text.  Exact task matches
    receive a conservative boost so existing per-task behaviour remains stable.
    """
    if max_records <= 0:
        return []

    flat = _flatten_knowledge(knowledge)
    if not flat:
        return []

    task_records = knowledge.get(task_id, []) if task_id else []
    fallback = _sort_records(task_records or flat)[:max_records]

    intent_keywords = _extract_keywords(prompt_text)
    if not prompt_text.strip() or not intent_keywords:
        return fallback

    documents = [_knowledge_record_text(rec) for rec in flat]

    # Prefer SQLite FTS5 for the lexical ranking when the build supports it
    # (indexed, standard BM25); an empty mapping means "FTS5 unavailable, no
    # matches, or disabled via MAESTRO_KNOWLEDGE_FTS=0" and we transparently
    # fall back to the in-Python BM25 below.  Feed FTS5 the *stopword-filtered*
    # keywords (sorted for deterministic term selection) so both rankers operate
    # on the same term set — passing the raw prompt would let common words like
    # "the" create spurious matches.
    fts_relevance: dict[int, float] = {}
    if _knowledge_fts_enabled():
        fts_query = " ".join(sorted(intent_keywords))
        fts_relevance = relevance_by_rank(documents, fts_query)

    idf: dict[str, float] | None = None
    avg_doc_len = 0.0
    if not fts_relevance:
        idf = _compute_idf(documents)
        avg_doc_len = (
            sum(max(1, len(_tokenize_words(doc))) for doc in documents)
            / len(documents)
        )

    scored: list[tuple[float, KnowledgeRecord]] = []

    for index, rec in enumerate(flat):
        if fts_relevance:
            score = fts_relevance.get(index, 0.0) * _FTS_RELEVANCE_SCALE
        else:
            score = _score_document(
                documents[index],
                intent_keywords,
                idf=idf,
                avg_doc_len=avg_doc_len,
            )
        if task_id and rec.task_id == task_id:
            score += 2.0
        score += rec.confidence * 0.5
        score += min(rec.occurrences, 5) * 0.05
        if score > 0.0 or (task_id and rec.task_id == task_id):
            scored.append((score, rec))

    if not scored:
        return fallback

    scored.sort(
        key=lambda item: (
            item[0],
            item[1].confidence,
            item[1].occurrences,
            item[1].last_seen,
            item[1].task_id,
            item[1].kind,
        ),
        reverse=True,
    )
    return [rec for _score, rec in scored[:max_records]]


def record_knowledge_retrievals(
    plan_name: str,
    source_dir: Path,
    prompt_text: str,
    records: list[KnowledgeRecord],
) -> list[RetrievalDominanceAlert]:
    """Persist retrieval counts and return any dominance alerts."""
    if not records:
        return []
    try:
        from .memory import record_retrievals
        return record_retrievals(plan_name, source_dir, prompt_text, records)
    except Exception:
        return []


def get_poisoning_alerts(
    plan_name: str,
    source_dir: Path,
) -> list[RetrievalDominanceAlert]:
    """Load persisted poisoning alerts for a plan."""
    try:
        from .memory import get_poisoning_alerts as _get_poisoning_alerts
        return _get_poisoning_alerts(plan_name, source_dir)
    except Exception:
        return []


def run_poisoning_harness(
    plan_name: str,
    source_dir: Path,
    prompts: list[str],
    *,
    task_id: str | None = None,
    max_records: int = 1,
) -> list[RetrievalDominanceAlert]:
    """Replay retrieval prompts against stored knowledge and surface alerts."""
    alerts: list[RetrievalDominanceAlert] = []
    for prompt_text in prompts:
        knowledge = load_knowledge(plan_name, source_dir, max_per_task=None)
        selected = select_relevant_knowledge(
            knowledge,
            prompt_text,
            task_id=task_id,
            max_records=max_records,
        )
        if not selected:
            continue
        alerts.extend(
            record_knowledge_retrievals(
                plan_name,
                source_dir,
                prompt_text,
                selected,
            )
        )
    return alerts


# ---------------------------------------------------------------------------
# Score history — Phase 2 -> Phase 3 bridge
# ---------------------------------------------------------------------------

def _run_quality_score(run_result: PlanRunResult) -> tuple[float, str]:
    judge_scores = [
        float(result.judge_result.overall_score)
        for result in run_result.task_results.values()
        if result.judge_result is not None and result.judge_result.overall_score is not None
    ]
    if judge_scores:
        return sum(judge_scores) / len(judge_scores), "judge"
    return (1.0 if run_result.success else 0.0), "outcome"


def plan_topology_signature(plan: PlanSpec) -> list[str]:
    """Return a compact structural signature for cross-run plan comparisons."""
    signature: set[str] = {
        f"plan.max_parallel:{plan.max_parallel}",
        f"plan.fail_fast:{int(plan.fail_fast)}",
        f"plan.task_count:{len(plan.tasks)}",
    }
    if plan.routing_strategy is not None:
        signature.add(f"plan.routing:{plan.routing_strategy}")
    if plan.firewall_model:
        signature.add("plan.firewall:1")

    for task in plan.tasks:
        signature.add(f"task.id:{task.id.lower()}")
        if task.engine:
            signature.add(f"task.engine:{task.engine}")
        if task.agent:
            signature.add(f"task.agent:{task.agent}")
        if task.model:
            signature.add(f"task.model:{task.model}")
        if task.context_mode:
            signature.add(f"task.context_mode:{task.context_mode}")
        if task.timeout_sec is not None:
            signature.add(f"task.timeout:{task.timeout_sec}")
        if task.max_retries:
            signature.add(f"task.retries:{task.max_retries}")
        if task.allow_failure:
            signature.add("task.allow_failure:1")
        if task.worktree:
            signature.add("task.worktree:1")
        if task.checkpoint:
            signature.add("task.checkpoint:1")
        if task.dynamic_group:
            signature.add("task.dynamic_group:1")
        if task.deliberation:
            signature.add("task.deliberation:1")
        if task.group:
            signature.add("task.group:1")
        if task.batch is not None:
            signature.add("task.batch:1")
        if task.judge is not None:
            signature.add("task.judge:1")
        for dependency in task.depends_on:
            signature.add(f"task.dep:{dependency.lower()}")
        for tag in task.tags:
            signature.add(f"task.tag:{tag.lower()}")

    return sorted(signature)


def build_score_record(
    plan: PlanSpec,
    run_result: PlanRunResult,
    *,
    plan_hash: str | None = None,
) -> ScoreRecord:
    """Build a plan-level score artifact from a completed run."""
    resolved_plan_hash = plan_hash or compute_plan_hash(plan)
    quality_score, quality_source = _run_quality_score(run_result)
    failed_tasks = sorted(
        task_id
        for task_id, result in run_result.task_results.items()
        if result.status == "failed"
    )
    timestamp = run_result.finished_at.astimezone(timezone.utc).isoformat()
    return ScoreRecord(
        plan_name=run_result.plan_name,
        plan_hash=resolved_plan_hash,
        run_id=run_result.run_id,
        success=run_result.success,
        cost_usd=run_result.total_cost_usd,
        quality_score=quality_score,
        duration_sec=max(
            0.0,
            (run_result.finished_at - run_result.started_at).total_seconds(),
        ),
        timestamp=timestamp,
        valid_from=timestamp,
        recorded_at=timestamp,
        source_id=f"{run_result.run_id}:score",
        metadata={
            "quality_source": quality_source,
            "judge_task_count": sum(
                1
                for result in run_result.task_results.values()
                if result.judge_result is not None and result.judge_result.overall_score is not None
            ),
            "task_count": len(run_result.task_results),
            "failed_tasks": failed_tasks,
            "execution_profile": run_result.execution_profile,
            "budget_exceeded": run_result.budget_exceeded,
            "run_path": str(run_result.run_path),
            "total_tokens": run_result.total_tokens,
            "simulation_plan_hash": compute_simulation_plan_hash(plan),
            "simulation_model_families": simulation_model_families(plan),
            "simulation_cache_policy_version": SIMULATION_CACHE_POLICY_VERSION,
            "plan_signature_terms": plan_topology_signature(plan),
        },
    )


def store_score_history(
    plan_name: str,
    source_dir: Path,
    score_record: ScoreRecord,
) -> bool:
    """Persist a score record via the memory backend."""
    try:
        from .memory import store_score_record
        return store_score_record(plan_name, source_dir, score_record)
    except Exception:
        return False


def load_score_history(
    plan_name: str,
    source_dir: Path,
    *,
    plan_hash: str | None = None,
    since: str | None = None,
    limit: int | None = None,
) -> list[ScoreRecord]:
    """Load score records for a plan or a concrete plan hash."""
    try:
        from .memory import load_score_records
        return load_score_records(
            plan_name,
            source_dir,
            plan_hash=plan_hash,
            since=since,
            limit=limit,
        )
    except Exception:
        return []


def load_simulation_cache_record(
    plan_name: str,
    source_dir: Path,
    *,
    simulation_plan_hash: str,
    limit: int = 128,
) -> ScoreRecord | None:
    """Return the newest successful score record matching a simulation-cache key."""
    for record in load_score_history(plan_name, source_dir, limit=limit):
        if not record.success:
            continue
        if record.metadata.get("simulation_plan_hash") != simulation_plan_hash:
            continue
        return record
    return None


def get_historical_pruning_decision(
    plan_name: str,
    source_dir: Path,
    plan_hash: str,
    *,
    threshold: float = 0.8,
    min_runs: int = 5,
    recent_runs: int = 20,
    horizon_days: int = 30,
) -> HistoricalPruningDecision:
    """Return whether recent history suggests pruning this plan variant."""
    try:
        from .memory import historical_pruning_decision
        return historical_pruning_decision(
            plan_name,
            source_dir,
            plan_hash,
            threshold=threshold,
            min_runs=min_runs,
            recent_runs=recent_runs,
            horizon_days=horizon_days,
        )
    except Exception:
        return HistoricalPruningDecision(
            plan_hash=plan_hash,
            sample_size=0,
            failures=0,
            failure_rate=0.0,
            threshold=threshold,
            min_runs=min_runs,
            prune=False,
            horizon_days=horizon_days,
            recent_runs=recent_runs,
        )


# ---------------------------------------------------------------------------
# Format knowledge for prompt injection
# ---------------------------------------------------------------------------

_KIND_ICONS: dict[str, str] = {
    "failure_pattern": "FAIL",
    "timeout_hint": "TIME",
    "success_pattern": "OK",
    "cost_pattern": "COST",
    "duration_pattern": "DUR",
    "retry_pattern": "RETRY",
    "model_pattern": "MODEL",
    "policy_rule": "RULE",
}


def format_knowledge(
    records: list[KnowledgeRecord],
    *,
    include_task_id: bool | None = None,
) -> str:
    """Format knowledge records as a bullet list for prompt injection."""
    if not records:
        return ""
    if include_task_id is None:
        include_task_id = len({rec.task_id for rec in records}) > 1
    lines: list[str] = []
    for rec in sorted(records, key=lambda r: r.confidence, reverse=True):
        conf = f"{rec.confidence:.0%}"
        kind_label = _KIND_ICONS.get(rec.kind, rec.kind.upper())
        task_label = f" [task={rec.task_id}]" if include_task_id else ""
        lines.append(f"- [{conf}] [{kind_label}]{task_label} {rec.insight}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Knowledge consolidation — strategic lessons from repeated patterns
# ---------------------------------------------------------------------------

_CONSOLIDATION_MIN_OCCURRENCES = 3
_CONSOLIDATION_MIN_CONFIDENCE = 0.4
_CONSOLIDATION_MAX_INSTRUCTIONALITY = 0.4
_CONSOLIDATION_MAX_UNTRUSTED_RATIO = 0.5


@dataclass
class ConsolidatedLesson:
    """A strategic lesson derived from repeated knowledge patterns."""

    category: str
    lesson: str
    evidence_count: int
    confidence: float
    task_ids: list[str]
    recommendation: str = ""
    source_trust_labels: list[str] = field(default_factory=list)
    avg_instructionality: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "category": self.category,
            "lesson": self.lesson,
            "evidence_count": self.evidence_count,
            "confidence": round(self.confidence, 3),
            "task_ids": self.task_ids,
            "recommendation": self.recommendation,
            "source_trust_labels": self.source_trust_labels,
            "avg_instructionality": round(self.avg_instructionality, 3),
        }


def consolidate_knowledge(
    plan_name: str,
    source_dir: Path,
    min_occurrences: int = _CONSOLIDATION_MIN_OCCURRENCES,
) -> list[ConsolidatedLesson]:
    """Aggregate knowledge records into strategic lessons.

    Groups records by kind and failure category, then produces
    consolidated lessons when enough evidence exists (>= *min_occurrences*
    total occurrences across tasks with sufficient confidence).

    Safety gates reject buckets where the evidence is predominantly
    untrusted (> 50%) or has high average instructionality score
    (>= 0.4), preventing poisoned knowledge from being promoted
    into reusable lessons.
    """
    from .memory import load_records_detailed, RecordWithTrust

    try:
        detailed = load_records_detailed(plan_name, source_dir)
    except Exception:
        # Fallback: load without trust metadata (JSONL-only installs)
        all_records = load_knowledge(plan_name, source_dir)
        detailed = [
            RecordWithTrust(record=r)
            for recs in all_records.values() for r in recs
        ]
    if not detailed:
        return []

    lessons: list[ConsolidatedLesson] = []

    # Group by (kind, first word of insight for semantic bucket)
    buckets: dict[str, list[RecordWithTrust]] = {}
    for rw in detailed:
        bucket_key = f"{rw.record.kind}:{_insight_key(rw.record.insight)}"
        buckets.setdefault(bucket_key, []).append(rw)

    for bucket_items in buckets.values():
        bucket_records = [rw.record for rw in bucket_items]
        total_occurrences = sum(r.occurrences for r in bucket_records)
        if total_occurrences < min_occurrences:
            continue

        avg_confidence = sum(r.confidence for r in bucket_records) / len(bucket_records)
        if avg_confidence < _CONSOLIDATION_MIN_CONFIDENCE:
            continue

        # Safety gate: reject high-instructionality evidence
        trust_labels = [rw.trust_label for rw in bucket_items]
        scores = [rw.instructionality_score for rw in bucket_items]
        avg_instr = sum(scores) / len(scores) if scores else 0.0
        if avg_instr >= _CONSOLIDATION_MAX_INSTRUCTIONALITY:
            continue

        # Safety gate: reject predominantly untrusted evidence
        untrusted_count = sum(1 for t in trust_labels if t != "trusted")
        if len(trust_labels) > 0 and untrusted_count / len(trust_labels) > _CONSOLIDATION_MAX_UNTRUSTED_RATIO:
            continue

        # Use the highest-confidence record as the representative
        representative = max(bucket_records, key=lambda r: r.confidence)
        task_ids = sorted({r.task_id for r in bucket_records})
        recommendation = _FAILURE_REMEDIATION.get(
            representative.kind.replace("_pattern", "").replace("_hint", ""),
            "",
        )

        lessons.append(ConsolidatedLesson(
            category=representative.kind,
            lesson=representative.insight,
            evidence_count=total_occurrences,
            confidence=avg_confidence,
            task_ids=task_ids,
            recommendation=recommendation,
            source_trust_labels=trust_labels,
            avg_instructionality=avg_instr,
        ))

    lessons.sort(key=lambda l: (-l.confidence, -l.evidence_count))
    return lessons


def compact_knowledge(
    plan_name: str,
    source_dir: Path,
    max_records_per_task: int = _MAX_RECORDS_PER_TASK,
) -> int:
    """Remove low-confidence and duplicate records, keeping only the most valuable.

    Delegates to :mod:`memory` (SQLite backend) when available, with
    JSONL fallback for backward compatibility.

    Returns the number of records removed.
    """
    try:
        from .memory import compact_records
        return compact_records(plan_name, source_dir, max_per_task=max_records_per_task)
    except Exception:
        pass

    path = _knowledge_path(plan_name, source_dir)
    raw = _load_raw(path)
    if not raw:
        return 0

    original_count = len(raw)

    # Apply time-decay
    for rec in raw:
        rec.confidence = _apply_decay(rec.confidence, rec.last_seen)

    # Dedup by (task_id, kind, insight_key)
    deduped: dict[str, KnowledgeRecord] = {}
    for rec in raw:
        key = f"{rec.task_id}:{rec.kind}:{_insight_key(rec.insight)}"
        existing = deduped.get(key)
        if existing is None:
            deduped[key] = rec
        elif rec.confidence > existing.confidence:
            rec.occurrences = max(rec.occurrences, existing.occurrences)
            deduped[key] = rec
        else:
            existing.occurrences = max(rec.occurrences, existing.occurrences)

    # Group by task_id, keep top N per task
    grouped: dict[str, list[KnowledgeRecord]] = {}
    for rec in deduped.values():
        grouped.setdefault(rec.task_id, []).append(rec)

    final: list[KnowledgeRecord] = []
    for task_id in grouped:
        grouped[task_id].sort(key=lambda r: (-r.confidence, -r.occurrences))
        final.extend(grouped[task_id][:max_records_per_task])

    # Remove records below minimum confidence
    final = [r for r in final if r.confidence >= 0.05]

    removed = original_count - len(final)
    if removed > 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [json.dumps(r.to_dict(), ensure_ascii=False) for r in final]
        path.write_text("\n".join(lines) + "\n" if lines else "", encoding="utf-8")

    return removed


def format_consolidated_lessons(lessons: list[ConsolidatedLesson]) -> str:
    """Format consolidated lessons for human-readable output."""
    if not lessons:
        return "[maestro] No consolidated lessons found."
    lines = ["[maestro] Consolidated lessons:", ""]
    for lesson in lessons:
        lines.append(
            f"  [{lesson.confidence:.0%}] [{lesson.category}] {lesson.lesson}"
        )
        lines.append(f"    Evidence: {lesson.evidence_count} occurrences across {len(lesson.task_ids)} tasks")
        if lesson.recommendation:
            lines.append(f"    Recommendation: {lesson.recommendation}")
        lines.append("")
    return "\n".join(lines)
