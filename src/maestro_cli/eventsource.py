from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fnmatch

from .models import EventRecord, OutputEnvelope, VerifyStatus


def compute_artefact_hash(path: Path) -> str | None:
    """SHA-256 (first 16 hex chars) of a file's contents.

    Returns ``None`` if the file does not exist or cannot be read.
    """
    try:
        data = path.read_bytes()
        return hashlib.sha256(data).hexdigest()[:16]
    except (OSError, IOError):
        return None


def verify_artefact_hashes(
    run_path: Path,
    events: list[EventRecord],
) -> list[str]:
    """Check artefact hashes recorded in ``task_complete`` events.

    Returns a list of mismatch descriptions (empty = all OK).
    """
    mismatches: list[str] = []
    for record in events:
        if record.event_type != "task_complete":
            continue
        payload = record.payload
        task_id = payload.get("task_id", "")
        expected_log = payload.get("log_hash")
        expected_result = payload.get("result_hash")
        if expected_log is not None:
            actual = compute_artefact_hash(run_path / f"{task_id}.log")
            if actual != expected_log:
                mismatches.append(
                    f"{task_id}.log: expected {expected_log}, got {actual}"
                )
        if expected_result is not None:
            actual = compute_artefact_hash(run_path / f"{task_id}.result.json")
            if actual != expected_result:
                mismatches.append(
                    f"{task_id}.result.json: expected {expected_result}, got {actual}"
                )
    return mismatches


def compute_event_hash(event_dict: dict[str, Any], prev_hash: str) -> str:
    """SHA-256 of ``json.dumps(event_dict, sort_keys=True) + prev_hash``.

    Returns the first 16 hex characters for compactness.
    """
    raw = json.dumps(event_dict, sort_keys=True) + prev_hash
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


@dataclass
class ChainState:
    sequence: int = 0
    prev_hash: str = "0" * 16


def emit_hashed_event(event_dict: dict[str, Any], state: ChainState) -> EventRecord:
    """Compute hash chain entry for *event_dict*, advance *state*, return EventRecord.

    The entire original *event_dict* is stored as ``payload`` so that
    ``verify_chain`` can recompute the hash deterministically without
    reconstructing the dict from individual fields.
    """
    event_hash = compute_event_hash(event_dict, state.prev_hash)
    record = EventRecord(
        sequence=state.sequence,
        event_type=event_dict.get("event", event_dict.get("type", "")),
        timestamp=event_dict.get("ts", ""),
        payload=event_dict,          # original dict stored verbatim
        prev_hash=state.prev_hash,
        event_hash=event_hash,
    )
    state.prev_hash = event_hash
    state.sequence += 1
    return record


def replay_events(events_path: Path) -> list[EventRecord]:
    """Read *events_path* (events.jsonl), parse each line, rebuild hash chain.

    Lines that have no ``hash`` field (legacy events written before hash-chain
    was introduced) are included in the result but hash verification is skipped
    for them.
    """
    records: list[EventRecord] = []
    seq = 0
    prev_hash = "0" * 16

    with events_path.open(encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            try:
                data: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                continue

            stored_hash: str | None = data.get("hash")

            # Determine the original event payload.
            # to_dict() format: {"seq":…,"type":…,"ts":…,"payload":{…},"prev_hash":…,"hash":…}
            # Legacy flat format: {"event":…,"ts":…,…}  (no separate "payload" key)
            if "payload" in data and isinstance(data["payload"], dict):
                payload: dict[str, Any] = data["payload"]
                stored_prev: str = data.get("prev_hash", prev_hash)
            else:
                # Legacy: everything except chain-metadata fields IS the payload
                payload = {k: v for k, v in data.items() if k not in {"seq", "hash", "prev_hash"}}
                stored_prev = prev_hash

            record = EventRecord(
                sequence=data.get("seq", seq),
                event_type=data.get("type", data.get("event", "")),
                timestamp=data.get("ts", ""),
                payload=payload,
                prev_hash=stored_prev,
                event_hash=stored_hash or "",
            )
            records.append(record)

            if stored_hash:
                prev_hash = stored_hash
            seq += 1

    return records


def verify_chain(events: list[EventRecord]) -> VerifyStatus:
    """Walk *events* and verify hash-chain integrity.

    Returns:
        ``"valid"``      — every hashed event checks out and the prev_hash
                           chain is internally consistent.
        ``"tampered"``   — a hash or prev_hash mismatch was detected.
        ``"incomplete"`` — no hashed events were found (all legacy / empty).
    """
    if not events:
        return "incomplete"

    hashed_count = 0
    prev_hash = "0" * 16

    for record in events:
        if not record.event_hash:
            # Legacy event — skip verification, keep prev_hash position
            continue

        hashed_count += 1

        # The payload stored by emit_hashed_event IS the original event_dict
        expected = compute_event_hash(record.payload, prev_hash)

        if record.prev_hash != prev_hash:
            return "tampered"
        if record.event_hash != expected:
            return "tampered"

        prev_hash = record.event_hash

    return "valid" if hashed_count > 0 else "incomplete"


# ---------------------------------------------------------------------------
# Output envelope — scope verification + output hashing
# ---------------------------------------------------------------------------


def compute_output_hash(output: str) -> str:
    """SHA-256 (first 16 hex chars) of task output text."""
    return hashlib.sha256(output.encode("utf-8")).hexdigest()[:16]


def check_scope_violations(
    files_changed: list[str],
    output_scope: list[str],
) -> list[str]:
    """Check which changed files fall outside the declared output scope.

    Args:
        files_changed: Actual files modified (from git diff or structured context).
        output_scope: Glob patterns declaring allowed output paths.

    Returns:
        List of file paths that violate the declared scope.
    """
    if not output_scope:
        return []  # no scope declared = no violations

    violations: list[str] = []
    for f in files_changed:
        normalized = f.replace("\\", "/")
        matched = any(fnmatch.fnmatch(normalized, pat) for pat in output_scope)
        if not matched:
            violations.append(f)
    return violations


def build_output_envelope(
    stdout_tail: str,
    output_scope: list[str],
    files_changed: list[str] | None = None,
) -> OutputEnvelope:
    """Build an output envelope for a task result.

    Args:
        stdout_tail: The task's stdout output (last N lines).
        output_scope: Declared scope globs from task.output_scope.
        files_changed: Optional list of actually changed files (from
            worktree merge, structured context, or git diff).

    Returns:
        OutputEnvelope with hash, scope status, and any violations.
    """
    output_hash = compute_output_hash(stdout_tail or "")
    violations = check_scope_violations(files_changed or [], output_scope)

    return OutputEnvelope(
        output_hash=output_hash,
        scope_declared=list(output_scope),
        scope_violations=violations,
        scope_verified=len(violations) == 0,
    )


_TERMINAL_STATUSES: frozenset[str] = frozenset(
    {"success", "failed", "soft_failed", "skipped", "dry_run"}
)


def replay_run_state(events: list[EventRecord]) -> dict[str, Any]:
    """Reconstruct a minimal run state from an event list.

    Returns a dict with:
    - ``tasks``           — ``dict[task_id, last_known_status]``
    - ``completed_tasks`` — set of task IDs that have reached a terminal status
    - ``total_cost_usd``  — sum of ``cost_usd`` from ``task_complete`` events
    """
    tasks: dict[str, str] = {}
    completed_tasks: set[str] = set()
    total_cost_usd: float = 0.0

    for record in events:
        etype = record.event_type
        payload = record.payload

        if etype == "task_start":
            task_id: str = payload.get("task_id", "")
            if task_id:
                tasks[task_id] = "running"

        elif etype == "task_complete":
            task_id = payload.get("task_id", "")
            status: str = payload.get("status", "")
            cost: Any = payload.get("cost_usd")
            if task_id:
                tasks[task_id] = status
                if status in _TERMINAL_STATUSES:
                    completed_tasks.add(task_id)
            if cost is not None:
                try:
                    total_cost_usd += float(cost)
                except (TypeError, ValueError):
                    pass

        elif etype == "task_skip":
            task_id = payload.get("task_id", "")
            if task_id:
                tasks[task_id] = "skipped"
                completed_tasks.add(task_id)

    return {
        "tasks": tasks,
        "completed_tasks": completed_tasks,
        "total_cost_usd": total_cost_usd,
    }
