from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .models import BlameCategory, BlameChain, BlameNode


def _safe_str(value: object) -> str:
    if isinstance(value, str):
        return value
    return ""


def _safe_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _safe_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _classify_failed_task(
    exit_code: int | None,
    message: str,
    depends_on: list[str],
    failed_set: set[str],
) -> BlameCategory:
    """Classify a failed task into a blame category."""
    # Dependency cascade: at least one dep also failed
    if any(dep in failed_set for dep in depends_on):
        return "dependency_cascade"

    # Root cause classification by exit code
    if exit_code == 124:
        return "timeout_propagation"

    # Root cause classification by message keywords
    msg_lower = message.lower()
    if any(kw in msg_lower for kw in ("budget", "cost", "skipped", "max_cost")):
        return "budget_exhaustion"
    if any(kw in msg_lower for kw in ("context", "token", "context window")):
        return "context_corruption"

    return "root_cause"


def _confidence_for_category(category: BlameCategory) -> float:
    _CONFIDENCE: dict[str, float] = {
        "root_cause": 0.9,
        "timeout_propagation": 0.85,
        "budget_exhaustion": 0.9,
        "dependency_cascade": 0.95,
        "context_corruption": 0.85,
        "unknown": 0.6,
    }
    return _CONFIDENCE.get(category, 0.6)


def _suggested_fix(category: BlameCategory, root_task_id: str = "") -> str:
    if category == "timeout_propagation":
        return "Increase timeout_sec or split into smaller tasks"
    if category == "root_cause":
        return "Check task logs and verify_command output"
    if category == "budget_exhaustion":
        return "Increase max_cost_usd or use cheaper models"
    if category == "context_corruption":
        return "Add context_budget_tokens or guard_command"
    if category == "dependency_cascade":
        root = root_task_id or "the root cause task"
        return f"Fix the root cause task first: {root}"
    return "Investigate task logs for error details"


def _load_events_evidence(run_path: Path) -> dict[str, list[str]]:
    """Load task_complete events from events.jsonl and return per-task evidence."""
    evidence_map: dict[str, list[str]] = {}
    events_path = run_path / "events.jsonl"
    if not events_path.is_file():
        return evidence_map

    try:
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event: Any = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue

            event_name = _safe_str(event.get("event"))
            task_id = _safe_str(event.get("task_id"))
            if not task_id or event_name != "task_complete":
                continue

            status = _safe_str(event.get("status"))
            if status not in ("failed", "soft_failed"):
                continue

            evidence: list[str] = evidence_map.setdefault(task_id, [])
            evidence.append(f"status={status}")
            exit_code = event.get("exit_code")
            if exit_code is not None:
                evidence.append(f"exit_code={exit_code}")
            msg = event.get("message") or event.get("error")
            if isinstance(msg, str) and msg:
                evidence.append(f"message={msg[:120]}")
    except OSError:
        pass

    return evidence_map


def blame_run(run_path: Path) -> BlameChain:
    """Analyze a completed run to trace failure causality back to root causes."""
    manifest_path = run_path / "run_manifest.json"
    if not manifest_path.is_file():
        return BlameChain(
            root_task_id="",
            nodes=[],
            suggested_fixes=[f"run_manifest.json not found in: {run_path}"],
        )

    try:
        raw: Any = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return BlameChain(
            root_task_id="",
            nodes=[],
            suggested_fixes=[f"Failed to load run_manifest.json: {exc}"],
        )

    if not isinstance(raw, dict):
        return BlameChain(
            root_task_id="",
            nodes=[],
            suggested_fixes=["run_manifest.json is not a valid JSON object"],
        )

    task_results_raw = raw.get("task_results")
    if not isinstance(task_results_raw, dict):
        return BlameChain(
            root_task_id="",
            nodes=[],
            suggested_fixes=["run_manifest.json has no task_results"],
        )

    # Parse task results — keyed by task_id
    task_data: dict[str, dict[str, Any]] = {
        tid: result
        for tid, result in task_results_raw.items()
        if isinstance(result, dict)
    }

    # Identify all failed tasks
    failed_tasks: set[str] = {
        tid for tid, r in task_data.items()
        if _safe_str(r.get("status")) == "failed"
    }

    if not failed_tasks:
        return BlameChain(root_task_id="", nodes=[], suggested_fixes=[])

    # Build dependency graph from task result data.
    # depends_on is not part of TaskResult.to_dict() by default; if it is
    # present (e.g. injected by future schema updates), use it — otherwise
    # treat every failed task as an independent root candidate.
    deps_map: dict[str, list[str]] = {}
    for task_id, result in task_data.items():
        raw_deps = result.get("depends_on")
        if isinstance(raw_deps, list):
            deps_map[task_id] = [str(d) for d in raw_deps if isinstance(d, str)]
        else:
            deps_map[task_id] = []

    # Load supplementary evidence from events.jsonl
    evidence_map = _load_events_evidence(run_path)

    # Classify each failed task
    nodes: list[BlameNode] = []
    for task_id in sorted(failed_tasks):
        result = task_data[task_id]
        exit_code = _safe_int(result.get("exit_code"))
        message = _safe_str(result.get("message"))
        depends_on = deps_map.get(task_id, [])

        # Exclude self from failed_set when checking cascade
        peer_failures = failed_tasks - {task_id}
        category = _classify_failed_task(exit_code, message, depends_on, peer_failures)
        confidence = _confidence_for_category(category)

        # Identify the first failed dependency for cascade nodes
        caused_by: str | None = None
        if category == "dependency_cascade":
            for dep in depends_on:
                if dep in failed_tasks:
                    caused_by = dep
                    break

        # Build evidence: start with events.jsonl data, add fallback from manifest
        evidence: list[str] = list(evidence_map.get(task_id, []))
        if not evidence:
            parts: list[str] = []
            if exit_code is not None:
                parts.append(f"exit_code={exit_code}")
            if message:
                parts.append(f"message={message[:120]}")
            if parts:
                evidence.append(", ".join(parts))

        nodes.append(BlameNode(
            task_id=task_id,
            category=category,
            confidence=confidence,
            message=message or f"Task failed (exit_code={exit_code})",
            caused_by=caused_by,
            evidence=evidence,
        ))

    if not nodes:
        return BlameChain(root_task_id="", nodes=[], suggested_fixes=[])  # pragma: no cover

    # Select root cause: prefer non-cascade nodes with highest confidence
    root_candidates = [n for n in nodes if n.category != "dependency_cascade"]
    if not root_candidates:
        root_candidates = nodes
    root_node = max(root_candidates, key=lambda n: n.confidence)
    root_task_id = root_node.task_id

    # Build deduplicated suggested_fixes list
    seen_fixes: set[str] = set()
    suggested_fixes: list[str] = []

    def _add_fix(fix: str) -> None:
        if fix not in seen_fixes:
            seen_fixes.add(fix)
            suggested_fixes.append(fix)

    # Root cause fix comes first
    _add_fix(_suggested_fix(root_node.category))

    # Cascade fix (if any cascades exist)
    if any(n.category == "dependency_cascade" for n in nodes):
        _add_fix(_suggested_fix("dependency_cascade", root_task_id))

    # Additional fixes for other unique categories
    for node in sorted(nodes, key=lambda n: -n.confidence):
        if node.task_id == root_node.task_id:
            continue
        _add_fix(_suggested_fix(node.category, root_task_id))

    return BlameChain(
        root_task_id=root_task_id,
        nodes=nodes,
        suggested_fixes=suggested_fixes,
    )


def format_blame(chain: BlameChain) -> str:
    """Format a BlameChain as human-readable text."""
    if not chain.nodes:
        return "[maestro blame] No failures found."

    lines: list[str] = []

    # Root cause section
    root = next(
        (n for n in chain.nodes if n.task_id == chain.root_task_id),
        chain.nodes[0],
    )
    lines.append(
        f"[maestro blame] Root cause: {root.task_id} "
        f"({root.category}, {root.confidence:.0%} confidence)"
    )
    if root.message:
        lines.append(f"[maestro blame]   {root.message}")
    if root.evidence:
        lines.append("[maestro blame]   Evidence:")
        for ev in root.evidence:
            lines.append(f"[maestro blame]     - {ev}")

    # Cascade section
    cascades = [n for n in chain.nodes if n.category == "dependency_cascade"]
    if cascades:
        lines.append("")
        lines.append(f"[maestro blame] Cascade ({len(cascades)} tasks affected):")
        for node in cascades:
            caused = f" <- {node.caused_by}" if node.caused_by else ""
            lines.append(f"[maestro blame]   {node.task_id}{caused} ({node.category})")

    # Other non-cascade, non-root failures
    others = [
        n for n in chain.nodes
        if n.task_id != chain.root_task_id and n.category != "dependency_cascade"
    ]
    if others:
        lines.append("")
        lines.append(f"[maestro blame] Additional root causes ({len(others)} tasks):")
        for node in others:
            lines.append(
                f"[maestro blame]   {node.task_id} "
                f"({node.category}, {node.confidence:.0%} confidence)"
            )
            if node.message:
                lines.append(f"[maestro blame]     {node.message}")

    # Suggested fixes section
    if chain.suggested_fixes:
        lines.append("")
        lines.append("[maestro blame] Suggested fixes:")
        for idx, fix in enumerate(chain.suggested_fixes, 1):
            lines.append(f"[maestro blame]   {idx}. {fix}")

    return "\n".join(lines)


def format_blame_json(chain: BlameChain) -> str:
    """Format a BlameChain as JSON."""
    return json.dumps(chain.to_dict(), indent=2)
