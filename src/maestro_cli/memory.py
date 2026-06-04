"""SQLite-backed persistent memory for Knowledge + Memory v2.

Replaces the JSONL-based storage in :mod:`knowledge` with a SQLite database
that supports:

- **WAL mode** for concurrent read/write (readers never block writers)
- **Bi-temporal model**: ``valid_from``/``valid_to`` (when true in reality) +
  ``recorded_at`` (when the system learned it)
- **Provenance fields**: ``source_type``, ``source_id``, ``trust_label``
- **Write validation**: instructionality scoring via regex
- **Automatic JSONL migration**: existing ``.jsonl`` files are imported on
  first access
- **Session snapshots**: durable watch-run summaries for future
  session-memory extraction

Thread safety: each thread gets its own connection via ``_get_connection()``.
Write serialization is handled by SQLite's WAL mode (single writer at a time).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, cast

from .models import (
    HistoricalPruningDecision,
    KnowledgeKind,
    KnowledgeRecord,
    KnowledgeWriteOutcome,
    ScoreRecord,
    SessionSnapshot,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SCHEMA_VERSION = 4
_MEMORY_DIR = ".maestro-cache/memory"
_KNOWLEDGE_JSONL_DIR = ".maestro-cache/knowledge"
_RETRIEVAL_DOMINANCE_ZSCORE = 3.0
_RETRIEVAL_DOMINANCE_MIN_COUNT = 5
_QUERY_CLUSTER_MAX_CHARS = 160
_SCORE_HISTORY_DEFAULT_MIN_RUNS = 5
_SCORE_HISTORY_DEFAULT_THRESHOLD = 0.8
_SCORE_HISTORY_DEFAULT_RECENT_RUNS = 20
_SCORE_HISTORY_DEFAULT_HORIZON_DAYS = 30
_QUERY_KEYWORD_RE = re.compile(r"[a-z0-9_]{2,}", re.IGNORECASE)
_QUERY_STOPWORDS: frozenset[str] = frozenset({
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
_FAILURE_KIND_RE = re.compile(r"^Fails with\s+([a-z0-9_./-]+)", re.IGNORECASE)
_MODEL_PATTERN_RE = re.compile(r"^Model\s+([a-zA-Z0-9_.:-]+):", re.IGNORECASE)

# Instructionality scoring patterns
_INSTRUCTIONAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\b(execute|run|invoke|call|trigger|perform|use)\s+(the\s+)?(tool|command|function|bash|shell)", re.I),
    re.compile(r"\b(ignore|override|bypass|skip|disable)\s+(all\s+)?(rules?|policies?|restrictions?|checks?)", re.I),
    re.compile(r"\b(you\s+must|you\s+should|always|never)\s+", re.I),
    re.compile(r"\b(delete|remove|drop|truncate|destroy)\s+(all|every|the)\s+", re.I),
    re.compile(r"\b(inject|insert|append)\s+(into|to)\s+(the\s+)?(prompt|context|memory)", re.I),
]
_INSTRUCTIONALITY_THRESHOLD = 0.4

# Thread-local storage for connections
_local = threading.local()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class MemoryRecord:
    """A single memory entry with bi-temporal and provenance fields."""

    id: int | None = None
    task_id: str = ""
    plan_name: str = ""
    kind: str = ""
    insight: str = ""
    insight_key: str = ""
    conflict_key: str = ""
    confidence: float = 0.5
    occurrences: int = 1
    first_seen: str = ""
    last_seen: str = ""
    # Bi-temporal
    valid_from: str = ""
    valid_to: str | None = None  # None = still valid
    recorded_at: str = ""
    # Provenance
    source_type: str = "task"  # task | tool | web | file
    source_id: str = ""  # run_id:task_id
    trust_label: str = "trusted"  # trusted | untrusted | quarantined
    instructionality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task_id": self.task_id,
            "plan_name": self.plan_name,
            "kind": self.kind,
            "insight": self.insight,
            "insight_key": self.insight_key,
            "conflict_key": self.conflict_key,
            "confidence": round(self.confidence, 3),
            "occurrences": self.occurrences,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "recorded_at": self.recorded_at,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "trust_label": self.trust_label,
            "instructionality_score": round(self.instructionality_score, 3),
        }

    def to_knowledge_record(self) -> KnowledgeRecord:
        """Convert to legacy KnowledgeRecord for backward compatibility."""
        return KnowledgeRecord(
            task_id=self.task_id,
            kind=self.kind,  # type: ignore[arg-type]
            insight=self.insight,
            confidence=self.confidence,
            occurrences=self.occurrences,
            first_seen=self.first_seen,
            last_seen=self.last_seen,
        )


@dataclass
class RetrievalDominanceAlert:
    """A suspicious retrieval pattern detected for a memory record."""

    record_id: int
    task_id: str
    kind: str
    insight: str
    insight_key: str
    query_cluster: str
    retrieval_count: int
    cluster_mean: float
    cluster_stddev: float
    z_score: float
    action: str = "quarantine"
    signal: str = "retrieval_dominance"

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task_id": self.task_id,
            "kind": self.kind,
            "insight": self.insight,
            "insight_key": self.insight_key,
            "query_cluster": self.query_cluster,
            "retrieval_count": self.retrieval_count,
            "cluster_mean": round(self.cluster_mean, 3),
            "cluster_stddev": round(self.cluster_stddev, 3),
            "z_score": round(self.z_score, 3),
            "action": self.action,
            "signal": self.signal,
        }


def _normalized_source_id(source_type: str, source_id: str, task_id: str) -> str:
    """Attach the task id to task-scoped provenance pointers when missing."""
    if source_type != "task" or not source_id:
        return source_id
    suffix = f":{task_id}"
    if source_id.endswith(suffix):
        return source_id
    return f"{source_id}{suffix}"


def _merge_trust_label(existing: str, incoming: str) -> str:
    """Preserve the most restrictive trust label seen for a record."""
    ranks = {
        "trusted": 0,
        "untrusted": 1,
        "quarantined": 2,
    }
    if ranks.get(incoming, 0) > ranks.get(existing, 0):
        return incoming
    return existing


# ---------------------------------------------------------------------------
# SQLite connection management
# ---------------------------------------------------------------------------

def _db_path(plan_name: str, source_dir: Path) -> Path:
    """Return the path to the SQLite database for a plan."""
    base = source_dir / _MEMORY_DIR
    base.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w\-.]", "_", plan_name)
    return base / f"{safe_name}.db"


def _get_connection(db_path: Path) -> sqlite3.Connection:
    """Get a thread-local SQLite connection with WAL mode.

    Each thread gets its own connection to avoid cross-thread sharing.
    WAL mode ensures readers never block writers and vice versa.
    """
    key = f"_memory_conn_{db_path}"
    conn: sqlite3.Connection | None = getattr(_local, key, None)
    if conn is not None:
        return conn

    conn = sqlite3.connect(str(db_path), timeout=10.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    setattr(_local, key, conn)
    return conn


def close_connection(db_path: Path) -> None:
    """Explicitly close a thread-local connection (for testing)."""
    key = f"_memory_conn_{db_path}"
    conn = getattr(_local, key, None)
    if conn is not None:
        conn.close()
        delattr(_local, key)


def close_all_connections() -> None:
    """Close every cached thread-local SQLite connection for this thread.

    Connection lifetimes are otherwise tied to the thread-local cache and live
    until process exit.  On Windows an open handle keeps the underlying ``.db``
    (and WAL/SHM sidecar) files locked, so a later ``rmtree`` of a scratch
    directory raises ``PermissionError [WinError 32]``.  Callers that create
    short-lived memory stores in temporary directories (e.g. the benchmark
    harness) use this to release all handles deterministically before cleanup,
    regardless of which ``source_dir`` opened the connection.
    """
    keys = [name for name in vars(_local) if name.startswith("_memory_conn_")]
    for key in keys:
        conn = getattr(_local, key, None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        try:
            delattr(_local, key)
        except AttributeError:
            pass


def _init_db(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS schema_version (
            version INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id TEXT NOT NULL,
            plan_name TEXT NOT NULL,
            kind TEXT NOT NULL,
            insight TEXT NOT NULL,
            insight_key TEXT NOT NULL,
            conflict_key TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0.5,
            occurrences INTEGER NOT NULL DEFAULT 1,
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            recorded_at TEXT NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'task',
            source_id TEXT NOT NULL DEFAULT '',
            trust_label TEXT NOT NULL DEFAULT 'trusted',
            instructionality_score REAL NOT NULL DEFAULT 0.0
        );

        CREATE INDEX IF NOT EXISTS idx_knowledge_plan_task
            ON knowledge(plan_name, task_id);
        CREATE INDEX IF NOT EXISTS idx_knowledge_dedup
            ON knowledge(plan_name, task_id, kind, insight_key);
        CREATE INDEX IF NOT EXISTS idx_knowledge_conflict
            ON knowledge(plan_name, task_id, kind, conflict_key, trust_label);
        CREATE INDEX IF NOT EXISTS idx_knowledge_valid
            ON knowledge(valid_to);
        CREATE INDEX IF NOT EXISTS idx_knowledge_trust
            ON knowledge(trust_label);

        CREATE TABLE IF NOT EXISTS retrieval_stats (
            plan_name TEXT NOT NULL,
            query_cluster TEXT NOT NULL,
            record_id INTEGER NOT NULL,
            retrieval_count INTEGER NOT NULL DEFAULT 0,
            first_retrieved_at TEXT NOT NULL,
            last_retrieved_at TEXT NOT NULL,
            PRIMARY KEY (plan_name, query_cluster, record_id),
            FOREIGN KEY(record_id) REFERENCES knowledge(id)
        );

        CREATE INDEX IF NOT EXISTS idx_retrieval_stats_cluster
            ON retrieval_stats(plan_name, query_cluster);

        CREATE TABLE IF NOT EXISTS poisoning_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_name TEXT NOT NULL,
            record_id INTEGER NOT NULL,
            query_cluster TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            retrieval_count INTEGER NOT NULL,
            cluster_mean REAL NOT NULL,
            cluster_stddev REAL NOT NULL,
            z_score REAL NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            created_at TEXT NOT NULL,
            FOREIGN KEY(record_id) REFERENCES knowledge(id)
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_poisoning_alert_open
            ON poisoning_alerts(plan_name, record_id, query_cluster, alert_type, status);

        CREATE TABLE IF NOT EXISTS score_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_name TEXT NOT NULL,
            plan_hash TEXT NOT NULL,
            run_id TEXT NOT NULL,
            success INTEGER NOT NULL DEFAULT 0,
            cost_usd REAL,
            quality_score REAL,
            duration_sec REAL NOT NULL DEFAULT 0.0,
            timestamp TEXT NOT NULL,
            valid_from TEXT NOT NULL,
            valid_to TEXT,
            recorded_at TEXT NOT NULL,
            source_id TEXT NOT NULL DEFAULT '',
            metadata TEXT NOT NULL DEFAULT '{}'
        );

        CREATE UNIQUE INDEX IF NOT EXISTS idx_score_records_run
            ON score_records(plan_name, run_id);
        CREATE INDEX IF NOT EXISTS idx_score_records_plan_hash
            ON score_records(plan_name, plan_hash, timestamp DESC);
        CREATE INDEX IF NOT EXISTS idx_score_records_valid
            ON score_records(plan_name, valid_to);

        CREATE TABLE IF NOT EXISTS session_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            plan_name TEXT NOT NULL,
            watch_run_path TEXT NOT NULL,
            snapshot_kind TEXT NOT NULL DEFAULT 'watch',
            iteration_from INTEGER NOT NULL DEFAULT 0,
            iteration_to INTEGER NOT NULL DEFAULT 0,
            best_metric REAL,
            snapshot_text TEXT NOT NULL DEFAULT '',
            recent_tail_count INTEGER NOT NULL DEFAULT 0,
            recorded_at TEXT NOT NULL,
            source_type TEXT NOT NULL DEFAULT 'watch',
            source_id TEXT NOT NULL DEFAULT '',
            trust_label TEXT NOT NULL DEFAULT 'trusted',
            instructionality_score REAL NOT NULL DEFAULT 0.0,
            metadata TEXT NOT NULL DEFAULT '{}'
        );

        CREATE INDEX IF NOT EXISTS idx_session_snapshots_plan_run
            ON session_snapshots(plan_name, watch_run_path, snapshot_kind, iteration_to DESC);
        CREATE INDEX IF NOT EXISTS idx_session_snapshots_trust
            ON session_snapshots(plan_name, trust_label);
    """)
    cols = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(knowledge)").fetchall()
    }
    if "conflict_key" not in cols:
        conn.execute(  # pragma: no cover
            "ALTER TABLE knowledge ADD COLUMN conflict_key TEXT NOT NULL DEFAULT ''"
        )
        conn.execute(  # pragma: no cover
            "UPDATE knowledge SET conflict_key = '' WHERE conflict_key IS NULL"
        )

    # Ensure schema version row exists
    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        conn.execute("INSERT INTO schema_version (version) VALUES (?)", (_SCHEMA_VERSION,))
    elif int(row["version"]) < _SCHEMA_VERSION:
        conn.execute("UPDATE schema_version SET version = ?", (_SCHEMA_VERSION,))
    conn.commit()


# ---------------------------------------------------------------------------
# Write validation (instructionality scoring)
# ---------------------------------------------------------------------------

def compute_instructionality(text: str) -> float:
    """Score how "instruction-like" a text is (0.0 = data, 1.0 = commands).

    Uses regex patterns to detect imperative tool-control language.
    """
    if not text:
        return 0.0
    hits = sum(1 for p in _INSTRUCTIONAL_PATTERNS if p.search(text))
    return min(1.0, hits / max(1, len(_INSTRUCTIONAL_PATTERNS)))


def _validate_write(record: MemoryRecord) -> MemoryRecord:
    """Validate a record before writing.  May set trust_label to quarantined."""
    score = compute_instructionality(record.insight)
    record.instructionality_score = score
    if score >= _INSTRUCTIONALITY_THRESHOLD:
        record.trust_label = "quarantined"
    return record


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _insight_key(insight: str) -> str:
    """Stable hash of the first 100 chars for deduplication."""
    return hashlib.sha256(insight[:100].encode("utf-8")).hexdigest()[:12]


def _extract_query_keywords(text: str) -> list[str]:
    keywords = {
        match.group(0).lower()
        for match in _QUERY_KEYWORD_RE.finditer(text)
        if match.group(0).lower() not in _QUERY_STOPWORDS
    }
    return sorted(keywords)


def _query_cluster(text: str) -> str:
    """Return a stable, human-readable query cluster label."""
    keywords = _extract_query_keywords(text)
    if not keywords:
        return "<empty>"
    cluster = "|".join(keywords[:12])
    if len(cluster) <= _QUERY_CLUSTER_MAX_CHARS:
        return cluster
    return cluster[: _QUERY_CLUSTER_MAX_CHARS - 3].rstrip("|") + "..."


def _compute_conflict_key(record: KnowledgeRecord) -> str:
    """Compute a stable conflict family for records that can supersede each other."""
    if record.kind == "failure_pattern":
        match = _FAILURE_KIND_RE.match(record.insight.strip())
        if match:
            return f"{record.task_id}:{record.kind}:{match.group(1).lower()}"
    if record.kind == "model_pattern":
        match = _MODEL_PATTERN_RE.match(record.insight.strip())
        if match:
            return f"{record.task_id}:{record.kind}:{match.group(1).lower()}"
    if record.kind in {
        "timeout_hint",
        "success_pattern",
        "cost_pattern",
        "duration_pattern",
        "retry_pattern",
    }:
        return f"{record.task_id}:{record.kind}"
    return f"{record.task_id}:{record.kind}:{_insight_key(record.insight)}"


def _computed_trust_label(source_type: str, insight: str) -> str:
    iscore = compute_instructionality(insight)
    if iscore >= _INSTRUCTIONALITY_THRESHOLD:
        return "quarantined"
    if source_type in {"web", "tool"}:
        return "untrusted"
    return "trusted"


def _record_sort_key(record: MemoryRecord) -> tuple[str, str, str]:
    return (
        record.valid_from or "",
        record.recorded_at or "",
        record.last_seen or "",
    )


def _resolve_memory_rows(rows: list[sqlite3.Row]) -> list[MemoryRecord]:
    """Collapse same-tier conflicts while preserving cross-tier alternatives."""
    resolved: dict[tuple[str, str, str, str], MemoryRecord] = {}
    for row in rows:
        rec = _row_to_memory_record(row)
        key = (
            rec.task_id,
            rec.kind,
            rec.conflict_key or f"{rec.task_id}:{rec.kind}:{rec.insight_key}",
            rec.trust_label,
        )
        existing = resolved.get(key)
        if existing is None or _record_sort_key(rec) > _record_sort_key(existing):
            resolved[key] = rec
    return list(resolved.values())


def store_records(
    plan_name: str,
    source_dir: Path,
    records: list[KnowledgeRecord],
    *,
    source_type: str = "task",
    source_id: str = "",
) -> int:
    """Store knowledge records in SQLite, merging duplicates.

    Returns the number of new records inserted (not merged).
    """
    return sum(
        1 for outcome in store_records_detailed(
            plan_name,
            source_dir,
            records,
            source_type=source_type,
            source_id=source_id,
        )
        if outcome.operation == "inserted"
    )


def store_records_detailed(
    plan_name: str,
    source_dir: Path,
    records: list[KnowledgeRecord],
    *,
    source_type: str = "task",
    source_id: str = "",
) -> list[KnowledgeWriteOutcome]:
    """Store knowledge records in SQLite and return write outcomes."""
    if not records:
        return []

    path = _db_path(plan_name, source_dir)
    _auto_migrate_jsonl(plan_name, source_dir, path)
    conn = _get_connection(path)
    now = _now_iso()
    outcomes: list[KnowledgeWriteOutcome] = []

    for rec in records:
        key = _insight_key(rec.insight)
        iscore = compute_instructionality(rec.insight)
        trust = _computed_trust_label(source_type, rec.insight)
        conflict_key = _compute_conflict_key(rec)
        effective_source_id = _normalized_source_id(source_type, source_id, rec.task_id)

        # Check for existing record (dedup)
        existing = conn.execute(
            "SELECT id, occurrences, confidence, trust_label, instructionality_score "
            "FROM knowledge "
            "WHERE plan_name = ? AND task_id = ? AND kind = ? AND insight_key = ? "
            "AND valid_to IS NULL",
            (plan_name, rec.task_id, rec.kind, key),
        ).fetchone()

        if existing:
            new_occ = existing["occurrences"] + 1
            new_conf = min(0.5 + new_occ * 0.1, 1.0)
            merged_trust = _merge_trust_label(existing["trust_label"], trust)
            merged_iscore = max(float(existing["instructionality_score"] or 0.0), iscore)
            conn.execute(
                "UPDATE knowledge SET occurrences = ?, confidence = ?, "
                "last_seen = ?, trust_label = ?, instructionality_score = ? "
                "WHERE id = ?",
                (new_occ, new_conf, rec.last_seen, merged_trust, merged_iscore, existing["id"]),
            )
            outcomes.append(
                KnowledgeWriteOutcome(
                    task_id=rec.task_id,
                    kind=rec.kind,
                    operation="merged",
                    outcome="quarantined" if merged_trust == "quarantined" else "accepted",
                    trust_label=merged_trust,
                    instructionality_score=merged_iscore,
                    source_type=source_type,
                    source_id=effective_source_id,
                )
            )
        else:
            conn.execute(
                "INSERT INTO knowledge "
                "(task_id, plan_name, kind, insight, insight_key, conflict_key, confidence, "
                "occurrences, first_seen, last_seen, valid_from, recorded_at, "
                "source_type, source_id, trust_label, instructionality_score) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    rec.task_id, plan_name, rec.kind, rec.insight, key,
                    conflict_key,
                    rec.confidence, rec.occurrences, rec.first_seen,
                    rec.last_seen, rec.first_seen, now,
                    source_type, effective_source_id, trust, iscore,
                ),
            )
            outcomes.append(
                KnowledgeWriteOutcome(
                    task_id=rec.task_id,
                    kind=rec.kind,
                    operation="inserted",
                    outcome="quarantined" if trust == "quarantined" else "accepted",
                    trust_label=trust,
                    instructionality_score=iscore,
                    source_type=source_type,
                    source_id=effective_source_id,
                )
            )

    conn.commit()
    return outcomes


def load_records(
    plan_name: str,
    source_dir: Path,
    *,
    max_per_task: int | None = 5,
    include_quarantined: bool = False,
) -> dict[str, list[KnowledgeRecord]]:
    """Load knowledge records, grouped by task_id.

    Applies time-decay, excludes quarantined by default, and optionally
    limits to *max_per_task* per task (sorted by confidence descending).
    """
    path = _db_path(plan_name, source_dir)
    _auto_migrate_jsonl(plan_name, source_dir, path)
    conn = _get_connection(path)

    trust_filter = "" if include_quarantined else "AND trust_label != 'quarantined'"
    rows = conn.execute(
        f"SELECT * FROM knowledge WHERE plan_name = ? AND valid_to IS NULL "
        f"{trust_filter} ORDER BY task_id, confidence DESC",
        (plan_name,),
    ).fetchall()

    resolved_rows = _resolve_memory_rows(rows)
    grouped: dict[str, list[KnowledgeRecord]] = {}
    for rec_row in resolved_rows:
        rec = KnowledgeRecord(
            task_id=rec_row.task_id,
            kind=cast(KnowledgeKind, rec_row.kind),
            insight=rec_row.insight,
            confidence=_apply_decay(rec_row.confidence, rec_row.last_seen),
            occurrences=rec_row.occurrences,
            first_seen=rec_row.first_seen,
            last_seen=rec_row.last_seen,
        )
        grouped.setdefault(rec.task_id, []).append(rec)

    # Sort by decayed confidence and cap
    for task_id in grouped:
        grouped[task_id].sort(key=lambda r: r.confidence, reverse=True)
        if max_per_task is not None:
            grouped[task_id] = grouped[task_id][:max_per_task]

    return grouped


@dataclass
class RecordWithTrust:
    """KnowledgeRecord paired with trust metadata from the memory backend."""

    record: KnowledgeRecord
    trust_label: str = "trusted"
    instructionality_score: float = 0.0


def load_records_detailed(
    plan_name: str,
    source_dir: Path,
    *,
    include_quarantined: bool = False,
) -> list[RecordWithTrust]:
    """Load knowledge records with trust labels and instructionality scores.

    Unlike :func:`load_records`, this returns a flat list with metadata
    for consolidation safety gates.
    """
    path = _db_path(plan_name, source_dir)
    _auto_migrate_jsonl(plan_name, source_dir, path)
    conn = _get_connection(path)

    trust_filter = "" if include_quarantined else "AND trust_label != 'quarantined'"
    rows = conn.execute(
        f"SELECT * FROM knowledge WHERE plan_name = ? AND valid_to IS NULL "
        f"{trust_filter} ORDER BY task_id, confidence DESC",
        (plan_name,),
    ).fetchall()

    resolved_rows = _resolve_memory_rows(rows)
    results: list[RecordWithTrust] = []
    for rec_row in resolved_rows:
        rec = KnowledgeRecord(
            task_id=rec_row.task_id,
            kind=cast(KnowledgeKind, rec_row.kind),
            insight=rec_row.insight,
            confidence=_apply_decay(rec_row.confidence, rec_row.last_seen),
            occurrences=rec_row.occurrences,
            first_seen=rec_row.first_seen,
            last_seen=rec_row.last_seen,
        )
        results.append(RecordWithTrust(
            record=rec,
            trust_label=rec_row.trust_label,
            instructionality_score=rec_row.instructionality_score,
        ))
    return results


def record_retrievals(
    plan_name: str,
    source_dir: Path,
    query_text: str,
    records: list[KnowledgeRecord],
) -> list[RetrievalDominanceAlert]:
    """Record memory retrievals for a query cluster and flag dominance outliers."""
    if not records:
        return []

    path = _db_path(plan_name, source_dir)
    _auto_migrate_jsonl(plan_name, source_dir, path)
    conn = _get_connection(path)
    cluster = _query_cluster(query_text)
    now = _now_iso()
    matched_rows: list[sqlite3.Row] = []

    for rec in records:
        row = conn.execute(
            "SELECT id, task_id, kind, insight, insight_key FROM knowledge "
            "WHERE plan_name = ? AND task_id = ? AND kind = ? AND insight_key = ? "
            "AND valid_to IS NULL ORDER BY confidence DESC LIMIT 1",
            (plan_name, rec.task_id, rec.kind, _insight_key(rec.insight)),
        ).fetchone()
        if row is None:
            continue
        matched_rows.append(row)
        conn.execute(
            "INSERT INTO retrieval_stats "
            "(plan_name, query_cluster, record_id, retrieval_count, first_retrieved_at, last_retrieved_at) "
            "VALUES (?, ?, ?, 1, ?, ?) "
            "ON CONFLICT(plan_name, query_cluster, record_id) DO UPDATE SET "
            "retrieval_count = retrieval_count + 1, "
            "last_retrieved_at = excluded.last_retrieved_at",
            (plan_name, cluster, row["id"], now, now),
        )

    if not matched_rows:
        conn.commit()
        return []

    candidate_rows = conn.execute(
        "SELECT k.id, k.task_id, k.kind, k.insight, k.insight_key, "
        "COALESCE(rs.retrieval_count, 0) AS retrieval_count "
        "FROM knowledge k "
        "LEFT JOIN retrieval_stats rs "
        "ON rs.record_id = k.id AND rs.plan_name = k.plan_name AND rs.query_cluster = ? "
        "WHERE k.plan_name = ? AND k.valid_to IS NULL AND k.trust_label != 'quarantined'",
        (cluster, plan_name),
    ).fetchall()

    counts = [int(row["retrieval_count"]) for row in candidate_rows]
    if len(counts) < 2:
        conn.commit()
        return []

    mean = sum(counts) / len(counts)
    variance = sum((count - mean) ** 2 for count in counts) / len(counts)
    stddev = math.sqrt(variance)
    if stddev <= 0.0:
        conn.commit()
        return []

    matched_ids = {int(row["id"]) for row in matched_rows}
    alerts: list[RetrievalDominanceAlert] = []

    for row in candidate_rows:
        record_id = int(row["id"])
        if record_id not in matched_ids:
            continue
        retrieval_count = int(row["retrieval_count"])
        if retrieval_count < _RETRIEVAL_DOMINANCE_MIN_COUNT:
            continue
        z_score = (retrieval_count - mean) / stddev
        threshold = mean + _RETRIEVAL_DOMINANCE_ZSCORE * stddev
        if retrieval_count <= threshold or z_score < _RETRIEVAL_DOMINANCE_ZSCORE:
            continue

        conn.execute(
            "UPDATE knowledge SET trust_label = 'quarantined' "
            "WHERE id = ? AND trust_label != 'quarantined'",
            (record_id,),
        )
        conn.execute(
            "INSERT OR IGNORE INTO poisoning_alerts "
            "(plan_name, record_id, query_cluster, alert_type, retrieval_count, "
            "cluster_mean, cluster_stddev, z_score, status, created_at) "
            "VALUES (?, ?, ?, 'retrieval_dominance', ?, ?, ?, ?, 'open', ?)",
            (
                plan_name,
                record_id,
                cluster,
                retrieval_count,
                mean,
                stddev,
                z_score,
                now,
            ),
        )
        alerts.append(
            RetrievalDominanceAlert(
                record_id=record_id,
                task_id=str(row["task_id"]),
                kind=str(row["kind"]),
                insight=str(row["insight"]),
                insight_key=str(row["insight_key"]),
                query_cluster=cluster,
                retrieval_count=retrieval_count,
                cluster_mean=mean,
                cluster_stddev=stddev,
                z_score=z_score,
            )
        )

    conn.commit()
    return alerts


def get_poisoning_alerts(
    plan_name: str,
    source_dir: Path,
    *,
    status: str = "open",
) -> list[RetrievalDominanceAlert]:
    """Return recorded poisoning alerts for a plan."""
    path = _db_path(plan_name, source_dir)
    _auto_migrate_jsonl(plan_name, source_dir, path)
    conn = _get_connection(path)
    rows = conn.execute(
        "SELECT a.record_id, k.task_id, k.kind, k.insight, k.insight_key, "
        "a.query_cluster, a.retrieval_count, a.cluster_mean, a.cluster_stddev, a.z_score "
        "FROM poisoning_alerts a "
        "JOIN knowledge k ON k.id = a.record_id "
        "WHERE a.plan_name = ? AND a.status = ? "
        "ORDER BY a.created_at DESC, a.id DESC",
        (plan_name, status),
    ).fetchall()
    return [
        RetrievalDominanceAlert(
            record_id=int(row["record_id"]),
            task_id=str(row["task_id"]),
            kind=str(row["kind"]),
            insight=str(row["insight"]),
            insight_key=str(row["insight_key"]),
            query_cluster=str(row["query_cluster"]),
            retrieval_count=int(row["retrieval_count"]),
            cluster_mean=float(row["cluster_mean"]),
            cluster_stddev=float(row["cluster_stddev"]),
            z_score=float(row["z_score"]),
        )
        for row in rows
    ]


def invalidate_record(
    plan_name: str,
    source_dir: Path,
    record_id: int,
) -> bool:
    """Invalidate a record by setting valid_to (immutable-with-windows).

    Returns True if the record was found and invalidated.
    """
    path = _db_path(plan_name, source_dir)
    conn = _get_connection(path)
    now = _now_iso()
    cursor = conn.execute(
        "UPDATE knowledge SET valid_to = ? WHERE id = ? AND valid_to IS NULL",
        (now, record_id),
    )
    conn.commit()
    return cursor.rowcount > 0


def point_in_time_query(
    plan_name: str,
    source_dir: Path,
    as_of: str,
    *,
    resolve_conflicts: bool = True,
) -> list[MemoryRecord]:
    """Return what the system knew at time *as_of* (ISO timestamp).

    Uses ``recorded_at <= as_of`` for transaction-time filtering.
    """
    path = _db_path(plan_name, source_dir)
    conn = _get_connection(path)
    rows = conn.execute(
        "SELECT * FROM knowledge WHERE plan_name = ? AND recorded_at <= ? "
        "AND (valid_to IS NULL OR valid_to > ?) "
        "ORDER BY confidence DESC",
        (plan_name, as_of, as_of),
    ).fetchall()

    if not resolve_conflicts:
        return [_row_to_memory_record(row) for row in rows]
    resolved = _resolve_memory_rows(rows)
    resolved.sort(key=lambda rec: (rec.confidence, rec.recorded_at), reverse=True)
    return resolved


def get_record_count(plan_name: str, source_dir: Path) -> int:
    """Return total record count for a plan (active only)."""
    path = _db_path(plan_name, source_dir)
    conn = _get_connection(path)
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM knowledge "
        "WHERE plan_name = ? AND valid_to IS NULL",
        (plan_name,),
    ).fetchone()
    return row["cnt"] if row else 0


# ---------------------------------------------------------------------------
# Session snapshots
# ---------------------------------------------------------------------------

def _normalize_session_snapshot_trust_label(
    source_type: str,
    snapshot_text: str,
    incoming: str | None = None,
) -> tuple[str, float]:
    """Return effective trust label and instructionality for a snapshot."""
    score = compute_instructionality(snapshot_text)
    trust = incoming or ("untrusted" if source_type in {"web", "tool"} else "trusted")
    if source_type in {"web", "tool"} and trust == "trusted":
        trust = "untrusted"
    if score >= _INSTRUCTIONALITY_THRESHOLD:
        trust = "quarantined"
    return trust, score


def store_session_snapshot(
    plan_name: str,
    source_dir: Path,
    snapshot: SessionSnapshot,
) -> int:
    """Persist a session snapshot and return its row id."""
    path = _db_path(plan_name, source_dir)
    _auto_migrate_jsonl(plan_name, source_dir, path)
    conn = _get_connection(path)

    recorded_at = snapshot.recorded_at or _now_iso()
    source_id = snapshot.source_id or snapshot.watch_run_path
    metadata_json = json.dumps(snapshot.metadata, ensure_ascii=True, sort_keys=True)
    trust_label, instructionality_score = _normalize_session_snapshot_trust_label(
        snapshot.source_type,
        snapshot.snapshot_text,
        snapshot.trust_label,
    )

    cursor = conn.execute(
        "INSERT INTO session_snapshots "
        "(plan_name, watch_run_path, snapshot_kind, iteration_from, iteration_to, "
        "best_metric, snapshot_text, recent_tail_count, recorded_at, source_type, "
        "source_id, trust_label, instructionality_score, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            plan_name,
            snapshot.watch_run_path,
            snapshot.snapshot_kind,
            snapshot.iteration_from,
            snapshot.iteration_to,
            snapshot.best_metric,
            snapshot.snapshot_text,
            snapshot.recent_tail_count,
            recorded_at,
            snapshot.source_type,
            source_id,
            trust_label,
            instructionality_score,
            metadata_json,
        ),
    )
    conn.commit()
    return int(cursor.lastrowid or 0)


def load_session_snapshots(
    plan_name: str,
    source_dir: Path,
    *,
    watch_run_path: str | None = None,
    snapshot_kind: str | None = None,
    include_quarantined: bool = False,
    limit: int | None = None,
) -> list[SessionSnapshot]:
    """Load durable session snapshots for a plan, newest first."""
    path = _db_path(plan_name, source_dir)
    _auto_migrate_jsonl(plan_name, source_dir, path)
    conn = _get_connection(path)

    clauses = ["plan_name = ?"]
    params: list[Any] = [plan_name]
    if watch_run_path is not None:
        clauses.append("watch_run_path = ?")
        params.append(watch_run_path)
    if snapshot_kind is not None:
        clauses.append("snapshot_kind = ?")
        params.append(snapshot_kind)
    if not include_quarantined:
        clauses.append("trust_label != 'quarantined'")

    query = (
        "SELECT * FROM session_snapshots WHERE "
        + " AND ".join(clauses)
        + " ORDER BY iteration_to DESC, recorded_at DESC, id DESC"
    )
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, tuple(params)).fetchall()
    return [_row_to_session_snapshot(row) for row in rows]


def load_latest_session_snapshot(
    plan_name: str,
    source_dir: Path,
    *,
    watch_run_path: str | None = None,
    snapshot_kind: str = "watch",
    include_quarantined: bool = False,
) -> SessionSnapshot | None:
    """Return the latest available session snapshot for a plan/run."""
    snapshots = load_session_snapshots(
        plan_name,
        source_dir,
        watch_run_path=watch_run_path,
        snapshot_kind=snapshot_kind,
        include_quarantined=include_quarantined,
        limit=1,
    )
    if not snapshots:
        return None
    return snapshots[0]


def prune_session_snapshots(
    plan_name: str,
    source_dir: Path,
    *,
    watch_run_path: str | None = None,
    snapshot_kind: str = "watch",
    keep: int = 5,
) -> int:
    """Delete older session snapshots, keeping only the newest N rows."""
    if keep < 0:
        raise ValueError("keep must be >= 0")

    path = _db_path(plan_name, source_dir)
    _auto_migrate_jsonl(plan_name, source_dir, path)
    conn = _get_connection(path)

    clauses = ["plan_name = ?", "snapshot_kind = ?"]
    params: list[Any] = [plan_name, snapshot_kind]
    if watch_run_path is not None:
        clauses.append("watch_run_path = ?")
        params.append(watch_run_path)

    rows = conn.execute(
        "SELECT id FROM session_snapshots WHERE "
        + " AND ".join(clauses)
        + " ORDER BY iteration_to DESC, recorded_at DESC, id DESC",
        tuple(params),
    ).fetchall()
    if len(rows) <= keep:
        return 0

    to_delete = [int(row["id"]) for row in rows[keep:]]
    conn.executemany(
        "DELETE FROM session_snapshots WHERE id = ?",
        [(row_id,) for row_id in to_delete],
    )
    conn.commit()
    return len(to_delete)


# ---------------------------------------------------------------------------
# Score history
# ---------------------------------------------------------------------------

def store_score_record(
    plan_name: str,
    source_dir: Path,
    score_record: ScoreRecord,
) -> bool:
    """Persist a plan-level score artifact for later search and pruning."""
    path = _db_path(plan_name, source_dir)
    _auto_migrate_jsonl(plan_name, source_dir, path)
    conn = _get_connection(path)
    now = _now_iso()
    timestamp = score_record.timestamp or now
    valid_from = score_record.valid_from or timestamp
    recorded_at = score_record.recorded_at or now
    source_id = score_record.source_id or f"{score_record.run_id}:score"
    metadata_json = json.dumps(score_record.metadata, ensure_ascii=True, sort_keys=True)

    conn.execute(
        "INSERT INTO score_records "
        "(plan_name, plan_hash, run_id, success, cost_usd, quality_score, duration_sec, "
        "timestamp, valid_from, valid_to, recorded_at, source_id, metadata) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(plan_name, run_id) DO UPDATE SET "
        "plan_hash = excluded.plan_hash, "
        "success = excluded.success, "
        "cost_usd = excluded.cost_usd, "
        "quality_score = excluded.quality_score, "
        "duration_sec = excluded.duration_sec, "
        "timestamp = excluded.timestamp, "
        "valid_from = excluded.valid_from, "
        "valid_to = excluded.valid_to, "
        "recorded_at = excluded.recorded_at, "
        "source_id = excluded.source_id, "
        "metadata = excluded.metadata",
        (
            plan_name,
            score_record.plan_hash,
            score_record.run_id,
            1 if score_record.success else 0,
            score_record.cost_usd,
            score_record.quality_score,
            score_record.duration_sec,
            timestamp,
            valid_from,
            score_record.valid_to,
            recorded_at,
            source_id,
            metadata_json,
        ),
    )
    conn.commit()
    return True


def load_score_records(
    plan_name: str,
    source_dir: Path,
    *,
    plan_hash: str | None = None,
    since: str | None = None,
    limit: int | None = None,
    include_invalidated: bool = False,
) -> list[ScoreRecord]:
    """Load persisted score records for a plan or concrete plan hash."""
    path = _db_path(plan_name, source_dir)
    _auto_migrate_jsonl(plan_name, source_dir, path)
    conn = _get_connection(path)

    clauses = ["plan_name = ?"]
    params: list[Any] = [plan_name]
    if plan_hash is not None:
        clauses.append("plan_hash = ?")
        params.append(plan_hash)
    if since is not None:
        clauses.append("timestamp >= ?")
        params.append(since)
    if not include_invalidated:
        clauses.append("valid_to IS NULL")

    query = (
        "SELECT * FROM score_records WHERE "
        + " AND ".join(clauses)
        + " ORDER BY timestamp DESC, id DESC"
    )
    if limit is not None:
        query += " LIMIT ?"
        params.append(limit)

    rows = conn.execute(query, tuple(params)).fetchall()
    return [_row_to_score_record(row) for row in rows]


def historical_pruning_decision(
    plan_name: str,
    source_dir: Path,
    plan_hash: str,
    *,
    threshold: float = _SCORE_HISTORY_DEFAULT_THRESHOLD,
    min_runs: int = _SCORE_HISTORY_DEFAULT_MIN_RUNS,
    recent_runs: int = _SCORE_HISTORY_DEFAULT_RECENT_RUNS,
    horizon_days: int = _SCORE_HISTORY_DEFAULT_HORIZON_DAYS,
    as_of: str | None = None,
) -> HistoricalPruningDecision:
    """Return whether a plan variant should be pruned from future expansion."""
    anchor = _parse_iso_timestamp(as_of) or datetime.now(timezone.utc)
    since_iso = (anchor - timedelta(days=horizon_days)).isoformat()
    records = load_score_records(
        plan_name,
        source_dir,
        plan_hash=plan_hash,
        since=since_iso,
        limit=recent_runs,
    )
    failures = sum(1 for record in records if not record.success)
    sample_size = len(records)
    failure_rate = failures / sample_size if sample_size else 0.0
    prune = sample_size >= min_runs and failure_rate >= threshold
    return HistoricalPruningDecision(
        plan_hash=plan_hash,
        sample_size=sample_size,
        failures=failures,
        failure_rate=failure_rate,
        threshold=threshold,
        min_runs=min_runs,
        prune=prune,
        horizon_days=horizon_days,
        recent_runs=recent_runs,
    )


# ---------------------------------------------------------------------------
# Time-decay
# ---------------------------------------------------------------------------

_CONFIDENCE_DECAY_HALF_LIFE_DAYS = 30.0


def _apply_decay(confidence: float, last_seen: str) -> float:
    """Apply exponential time-decay to a confidence value."""
    try:
        ts = datetime.fromisoformat(last_seen)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
        decay = 0.5 ** (age_days / _CONFIDENCE_DECAY_HALF_LIFE_DAYS)
        return float(max(0.0, min(1.0, confidence * decay)))
    except (ValueError, TypeError):
        return confidence


def _parse_iso_timestamp(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        ts = datetime.fromisoformat(value)
    except ValueError:
        return None
    if ts.tzinfo is None:
        return ts.replace(tzinfo=timezone.utc)
    return ts.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# JSONL migration
# ---------------------------------------------------------------------------

_migration_done: set[str] = set()


def _auto_migrate_jsonl(plan_name: str, source_dir: Path, db_path: Path) -> None:
    """One-time migration from JSONL to SQLite."""
    cache_key = f"{plan_name}:{source_dir}"
    if cache_key in _migration_done:
        return
    _migration_done.add(cache_key)

    safe_name = re.sub(r"[^\w\-.]", "_", plan_name)
    jsonl_path = source_dir / _KNOWLEDGE_JSONL_DIR / f"{safe_name}.jsonl"
    if not jsonl_path.exists():
        return

    conn = _get_connection(db_path)

    # Check if already migrated (has any records for this plan)
    row = conn.execute(
        "SELECT COUNT(*) as cnt FROM knowledge WHERE plan_name = ?",
        (plan_name,),
    ).fetchone()
    if row and row["cnt"] > 0:
        return  # Already has data, skip migration

    # Read JSONL
    now = _now_iso()
    count = 0
    try:
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                key = _insight_key(d.get("insight", ""))
                insight = d.get("insight", "")
                iscore = compute_instructionality(insight)
                conflict_key = _compute_conflict_key(KnowledgeRecord(
                    task_id=d.get("task_id", ""),
                    kind=d.get("kind", ""),
                    insight=insight,
                    confidence=float(d.get("confidence", 0.5)),
                    occurrences=int(d.get("occurrences", 1)),
                    first_seen=d.get("first_seen", now),
                    last_seen=d.get("last_seen", now),
                ))
                first_seen = d.get("first_seen", now)
                conn.execute(
                    "INSERT INTO knowledge "
                    "(task_id, plan_name, kind, insight, insight_key, conflict_key, confidence, "
                    "occurrences, first_seen, last_seen, valid_from, recorded_at, "
                    "source_type, source_id, trust_label, instructionality_score) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        d.get("task_id", ""),
                        plan_name,
                        d.get("kind", ""),
                        insight,
                        key,
                        conflict_key,
                        d.get("confidence", 0.5),
                        d.get("occurrences", 1),
                        first_seen,
                        d.get("last_seen", now),
                        first_seen,
                        now,
                        "task",
                        "",
                        _computed_trust_label("task", insight),
                        iscore,
                    ),
                )
                count += 1
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
        conn.commit()
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def compact_records(
    plan_name: str,
    source_dir: Path,
    max_per_task: int = 5,
    min_confidence: float = 0.05,
) -> int:
    """Remove low-confidence records, keeping top N per task.

    Returns the number of records deleted.
    """
    path = _db_path(plan_name, source_dir)
    _auto_migrate_jsonl(plan_name, source_dir, path)
    conn = _get_connection(path)

    # Get all active records
    rows = conn.execute(
        "SELECT id, task_id, confidence, last_seen FROM knowledge "
        "WHERE plan_name = ? AND valid_to IS NULL "
        "ORDER BY task_id, confidence DESC",
        (plan_name,),
    ).fetchall()

    if not rows:
        return 0

    # Apply decay and find records to delete
    to_delete: list[int] = []
    task_counts: dict[str, int] = {}

    for row in rows:
        decayed = _apply_decay(row["confidence"], row["last_seen"])
        tid = row["task_id"]
        count = task_counts.get(tid, 0)

        if decayed < min_confidence or count >= max_per_task:
            to_delete.append(row["id"])
        else:
            task_counts[tid] = count + 1

    if to_delete:
        # Use valid_to invalidation (immutable pattern)
        now = _now_iso()
        conn.executemany(
            "UPDATE knowledge SET valid_to = ? WHERE id = ?",
            [(now, rid) for rid in to_delete],
        )
        conn.commit()

    return len(to_delete)


def _row_to_memory_record(row: sqlite3.Row) -> MemoryRecord:
    """Convert a sqlite3.Row to a MemoryRecord."""
    return MemoryRecord(
        id=row["id"],
        task_id=row["task_id"],
        plan_name=row["plan_name"],
        kind=row["kind"],
        insight=row["insight"],
        insight_key=row["insight_key"],
        conflict_key=row["conflict_key"],
        confidence=row["confidence"],
        occurrences=row["occurrences"],
        first_seen=row["first_seen"],
        last_seen=row["last_seen"],
        valid_from=row["valid_from"],
        valid_to=row["valid_to"],
        recorded_at=row["recorded_at"],
        source_type=row["source_type"],
        source_id=row["source_id"],
        trust_label=row["trust_label"],
        instructionality_score=row["instructionality_score"],
    )


def _row_to_score_record(row: sqlite3.Row) -> ScoreRecord:
    """Convert a sqlite3.Row to a ScoreRecord."""
    metadata_raw = row["metadata"]
    metadata: dict[str, Any] = {}
    if isinstance(metadata_raw, str) and metadata_raw:
        try:
            parsed = json.loads(metadata_raw)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            metadata = parsed
    return ScoreRecord(
        plan_name=str(row["plan_name"]),
        plan_hash=str(row["plan_hash"]),
        run_id=str(row["run_id"]),
        success=bool(row["success"]),
        cost_usd=float(row["cost_usd"]) if row["cost_usd"] is not None else None,
        quality_score=float(row["quality_score"]) if row["quality_score"] is not None else None,
        duration_sec=float(row["duration_sec"]),
        timestamp=str(row["timestamp"]),
        valid_from=str(row["valid_from"]),
        valid_to=str(row["valid_to"]) if row["valid_to"] is not None else None,
        recorded_at=str(row["recorded_at"]),
        source_id=str(row["source_id"]),
        metadata=metadata,
    )


def _row_to_session_snapshot(row: sqlite3.Row) -> SessionSnapshot:
    """Convert a sqlite3.Row to a SessionSnapshot."""
    metadata_raw = row["metadata"]
    metadata: dict[str, Any] = {}
    if isinstance(metadata_raw, str) and metadata_raw:
        try:
            parsed = json.loads(metadata_raw)
        except json.JSONDecodeError:
            parsed = {}
        if isinstance(parsed, dict):
            metadata = parsed

    return SessionSnapshot(
        id=int(row["id"]),
        plan_name=str(row["plan_name"]),
        watch_run_path=str(row["watch_run_path"]),
        snapshot_kind=str(row["snapshot_kind"]),
        iteration_from=int(row["iteration_from"]),
        iteration_to=int(row["iteration_to"]),
        best_metric=float(row["best_metric"]) if row["best_metric"] is not None else None,
        snapshot_text=str(row["snapshot_text"]),
        recent_tail_count=int(row["recent_tail_count"]),
        recorded_at=str(row["recorded_at"]),
        source_type=str(row["source_type"]),
        source_id=str(row["source_id"]),
        trust_label=str(row["trust_label"]),
        instructionality_score=float(row["instructionality_score"]),
        metadata=metadata,
    )
