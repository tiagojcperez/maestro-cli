from __future__ import annotations

from collections import deque
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .runners import _extract_cost_and_tokens_from_log, _extract_cost_from_log

_MAESTRO_RUNS_DIR = ".maestro-runs"
_DISCOVERY_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "venv",
}
_DEFAULT_DISCOVERY_MAX_DEPTH = 8


@dataclass
class BackfillSummary:
    run_roots: int = 0
    runs_scanned: int = 0
    runs_updated: int = 0
    tasks_updated: int = 0
    manifests_failed: int = 0
    result_files_updated: int = 0


def discover_run_roots(
    project_root: Path,
    max_depth: int = _DEFAULT_DISCOVERY_MAX_DEPTH,
) -> list[Path]:
    roots: list[Path] = []
    seen: set[Path] = set()

    try:
        root = project_root.resolve()
    except OSError:
        root = project_root

    if not root.exists() or not root.is_dir():
        return roots

    queue: deque[tuple[Path, int]] = deque([(root, 0)])
    while queue:
        current, depth = queue.popleft()

        candidate = current / _MAESTRO_RUNS_DIR
        if candidate.exists() and candidate.is_dir():
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                roots.append(resolved)

        if depth >= max_depth:
            continue

        try:
            children = sorted(current.iterdir())
        except OSError:
            continue

        for child in children:
            if not child.is_dir():
                continue
            name = child.name
            if name == _MAESTRO_RUNS_DIR:
                continue
            if name in _DISCOVERY_SKIP_DIRS:
                continue
            if name.startswith("."):
                continue
            queue.append((child, depth + 1))

    return roots


def _coerce_cost(value: object) -> float | None:
    if not isinstance(value, (int, float, str)):
        return None
    try:
        cost = float(value)
    except (TypeError, ValueError):
        return None
    if cost < 0:
        return None
    return cost


def _resolve_task_log_path(run_dir: Path, task_id: str, task_result: dict[str, Any]) -> Path | None:
    raw = task_result.get("log_path")
    if isinstance(raw, str) and raw.strip():
        p = Path(raw)
        if not p.is_absolute():
            p = run_dir / p
        if p.exists():
            return p

    fallback = run_dir / f"{task_id}.log"
    if fallback.exists():
        return fallback
    return None


def _resolve_result_path(run_dir: Path, task_id: str, task_result: dict[str, Any]) -> Path | None:
    raw = task_result.get("result_path")
    if isinstance(raw, str) and raw.strip():
        p = Path(raw)
        if not p.is_absolute():
            p = run_dir / p
        return p
    return run_dir / f"{task_id}.result.json"


def _infer_engine(command: str) -> str | None:
    """Best-effort engine detection from the command string in a manifest."""
    cmd_lower = command.lower()
    if "codex " in cmd_lower or cmd_lower.startswith("codex"):
        return "codex"
    if "claude " in cmd_lower or cmd_lower.startswith("claude"):
        return "claude"
    if "gemini " in cmd_lower or cmd_lower.startswith("gemini"):
        return "gemini"
    if "copilot " in cmd_lower or cmd_lower.startswith("copilot"):
        return "copilot"
    return None


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding="utf-8")


def _backfill_single_run(run_dir: Path, *, write: bool) -> tuple[bool, int, int]:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.exists():
        return False, 0, 0

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False, 0, 0

    task_results = manifest.get("task_results")
    if not isinstance(task_results, dict):
        return False, 0, 0

    manifest_changed = False
    tasks_updated = 0
    result_files_updated = 0

    for task_id, task_result in task_results.items():
        if not isinstance(task_id, str) or not isinstance(task_result, dict):
            continue

        existing_cost = _coerce_cost(task_result.get("cost_usd"))
        existing_tokens = task_result.get("token_usage")

        # Skip if both cost and tokens are already populated
        if existing_cost is not None and existing_tokens is not None:
            continue

        log_path = _resolve_task_log_path(run_dir, task_id, task_result)
        if log_path is None:
            continue

        # Try the new combined extractor (needs engine info)
        engine = _infer_engine(task_result.get("command", ""))
        cost_and_tokens = None
        if engine:
            cost_and_tokens = _extract_cost_and_tokens_from_log(log_path, engine, None)

        # Update cost if missing
        if existing_cost is None:
            inferred_cost = (
                cost_and_tokens.cost_usd if cost_and_tokens else None
            ) or _extract_cost_from_log(log_path)
            if inferred_cost is not None:
                task_result["cost_usd"] = inferred_cost
                manifest_changed = True
                tasks_updated += 1

        # Update token_usage if missing
        if existing_tokens is None and cost_and_tokens and cost_and_tokens.token_usage:
            task_result["token_usage"] = cost_and_tokens.token_usage.to_dict()
            manifest_changed = True

        result_path = _resolve_result_path(run_dir, task_id, task_result)
        if result_path and result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                payload = None
            if isinstance(payload, dict):
                result_file_changed = False
                if _coerce_cost(payload.get("cost_usd")) is None and task_result.get("cost_usd") is not None:
                    payload["cost_usd"] = task_result["cost_usd"]
                    result_file_changed = True
                if payload.get("token_usage") is None and task_result.get("token_usage") is not None:
                    payload["token_usage"] = task_result["token_usage"]
                    result_file_changed = True
                if result_file_changed:
                    if write:
                        _write_json(result_path, payload)
                    result_files_updated += 1

    costs: list[float] = []
    tokens: list[int] = []
    for tr in task_results.values():
        if not isinstance(tr, dict):
            continue
        c = _coerce_cost(tr.get("cost_usd"))
        if c is not None:
            costs.append(c)
        tu = tr.get("token_usage")
        if isinstance(tu, dict):
            total_tok = tu.get("total_tokens")
            if isinstance(total_tok, int) and total_tok > 0:
                tokens.append(total_tok)

    new_total: float | None = sum(costs) if costs else None
    old_total = _coerce_cost(manifest.get("total_cost_usd"))
    if (new_total is None and old_total is not None) or (new_total is not None and old_total != new_total):
        manifest["total_cost_usd"] = new_total
        manifest_changed = True

    new_total_tokens: int | None = sum(tokens) if tokens else None
    old_total_tokens = manifest.get("total_tokens")
    if new_total_tokens != old_total_tokens:
        manifest["total_tokens"] = new_total_tokens
        manifest_changed = True

    if manifest_changed and write:
        _write_json(manifest_path, manifest)

    return manifest_changed, tasks_updated, result_files_updated


def backfill_run_costs(
    *,
    run_roots: list[Path],
    write: bool = True,
) -> BackfillSummary:
    summary = BackfillSummary(run_roots=len(run_roots))

    for run_root in run_roots:
        if not run_root.exists() or not run_root.is_dir():
            continue
        for run_dir in run_root.iterdir():
            if not run_dir.is_dir():
                continue
            summary.runs_scanned += 1
            try:
                changed, task_updates, result_updates = _backfill_single_run(run_dir, write=write)
            except Exception:
                summary.manifests_failed += 1
                continue

            if changed:
                summary.runs_updated += 1
            summary.tasks_updated += task_updates
            summary.result_files_updated += result_updates

    return summary
