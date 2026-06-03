"""REST API routes for Maestro Web UI."""
from __future__ import annotations

import json
import re
import shutil
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .. import __version__
from ..cleanup import cleanup_runs
from ..cost_backfill import discover_run_roots
from ..errors import PlanValidationError
from ..eventsource import replay_events
from ..loader import load_plan
from ..models import ExecutionProfile, PlanRunResult
from ..scheduler import run_plan
from ..utils import now_utc, resolve_path
from .state import (
    RunState,
    get_project_root,
    get_project_roots,
    get_run,
    list_active_runs,
    register_run,
    remove_run,
)

router = APIRouter()

_MODEL_FLAG_CLAUDE = re.compile(r"(?:^|\s)--model(?:=|\s+)(\"[^\"]+\"|'[^']+'|[^\s]+)")
_MODEL_FLAG_CODEX = re.compile(r"(?:^|\s)-m\s+(\"[^\"]+\"|'[^']+'|[^\s]+)")
_SUCCESS_LIKE = {"success", "dry_run"}
_TERMINAL_STATUSES = {"success", "failed", "soft_failed", "skipped", "dry_run"}
_ACTIVITY_LIMIT = 8


def _discover_run_roots() -> list[Path]:
    """Find all discoverable .maestro-runs directories across project roots."""
    roots: list[Path] = []
    seen: set[Path] = set()
    for base in get_project_roots():
        for candidate in discover_run_roots(base):
            if candidate in seen:
                continue
            seen.add(candidate)
            roots.append(candidate)
    return roots


def _is_dry_run_from_manifest(manifest: dict[str, Any]) -> bool:
    explicit = manifest.get("dry_run")
    if isinstance(explicit, bool):
        return explicit

    task_results = manifest.get("task_results")
    if not isinstance(task_results, dict) or not task_results:
        return False

    statuses: list[str] = []
    for tr in task_results.values():
        if not isinstance(tr, dict):
            continue
        status = tr.get("status")
        if isinstance(status, str):
            statuses.append(status)

    if not statuses:
        return False

    has_dry_run = any(s == "dry_run" for s in statuses)
    has_real_exec = any(s in {"success", "failed", "soft_failed"} for s in statuses)
    return has_dry_run and not has_real_exec


def _duration_from_manifest(manifest: dict[str, Any]) -> float | None:
    started_at = manifest.get("started_at")
    finished_at = manifest.get("finished_at")
    if not isinstance(started_at, str) or not isinstance(finished_at, str):
        return None

    try:
        start_dt = datetime.fromisoformat(started_at)
        finish_dt = datetime.fromisoformat(finished_at)
    except (ValueError, TypeError):
        return None

    return max(0.0, (finish_dt - start_dt).total_seconds())


def _total_cost_from_manifest(manifest: dict[str, Any]) -> float | None:
    total = manifest.get("total_cost_usd")
    if total is not None:
        try:
            return float(total)
        except (TypeError, ValueError):
            pass

    task_results = manifest.get("task_results")
    if not isinstance(task_results, dict):
        return None

    costs: list[float] = []
    for tr in task_results.values():
        if not isinstance(tr, dict):
            continue
        cost = tr.get("cost_usd")
        if cost is None:
            continue
        try:
            costs.append(float(cost))
        except (TypeError, ValueError):
            continue

    if not costs:
        return None
    return sum(costs)


def _total_tokens_from_manifest(manifest: dict[str, Any]) -> int | None:
    total = manifest.get("total_tokens")
    if total is not None:
        try:
            return int(total)
        except (TypeError, ValueError):
            pass

    task_results = manifest.get("task_results")
    if not isinstance(task_results, dict):
        return None

    tokens: list[int] = []
    for tr in task_results.values():
        if not isinstance(tr, dict):
            continue
        tu = tr.get("token_usage")
        if not isinstance(tu, dict):
            continue
        t = tu.get("total_tokens")
        if t is None:
            continue
        try:
            tokens.append(int(t))
        except (TypeError, ValueError):
            continue

    if not tokens:
        return None
    return sum(tokens)


def _clean_cli_token(value: str) -> str:
    token = value.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in {"'", '"'}:
        token = token[1:-1]
    return token.strip()


def _infer_model_from_task_result(task_result: dict[str, Any]) -> str | None:
    command = task_result.get("command")
    if not isinstance(command, str) or not command:
        return None

    match = _MODEL_FLAG_CLAUDE.search(command)
    if match:
        model = _clean_cli_token(match.group(1))
        if model:
            return model

    match = _MODEL_FLAG_CODEX.search(command)
    if match:
        model = _clean_cli_token(match.group(1))
        if model:
            return model

    lowered = command.lower()
    if "codex" in lowered:
        return "codex"
    if "claude" in lowered:
        return "claude"
    return None


def _build_task_graph_from_tasks(tasks: list[Any]) -> dict[str, dict[str, Any]]:
    graph: dict[str, dict[str, Any]] = {}
    for task in tasks:
        task_id = getattr(task, "id", None)
        if not isinstance(task_id, str) or not task_id:
            continue
        depends_on_raw = getattr(task, "depends_on", [])
        depends_on = [dep for dep in depends_on_raw if isinstance(dep, str)]
        graph[task_id] = {
            "id": task_id,
            "description": getattr(task, "description", "") or "",
            "depends_on": depends_on,
            "agent": getattr(task, "agent", None),
            "engine": getattr(task, "engine", None),
            "model": getattr(task, "model", None),
            "allow_failure": bool(getattr(task, "allow_failure", False)),
            "worktree": bool(getattr(task, "worktree", False)),
        }
    return graph


def _normalize_task_graph(task_graph: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(task_graph, dict):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for raw_task_id, raw_meta in task_graph.items():
        if not isinstance(raw_task_id, str) or not raw_task_id:
            continue
        meta = raw_meta if isinstance(raw_meta, dict) else {}
        depends_on_raw = meta.get("depends_on", [])
        depends_on = [dep for dep in depends_on_raw if isinstance(dep, str)]
        normalized[raw_task_id] = {
            "id": raw_task_id,
            "description": meta.get("description") if isinstance(meta.get("description"), str) else "",
            "depends_on": depends_on,
            "agent": meta.get("agent") if isinstance(meta.get("agent"), str) and meta.get("agent") else None,
            "engine": meta.get("engine") if isinstance(meta.get("engine"), str) and meta.get("engine") else None,
            "model": meta.get("model") if isinstance(meta.get("model"), str) and meta.get("model") else None,
            "allow_failure": bool(meta.get("allow_failure", False)),
            "worktree": bool(meta.get("worktree", False)),
        }
    return normalized


def _ordered_task_ids(
    task_ids: list[str] | None,
    task_results: dict[str, Any],
    task_graph: dict[str, dict[str, Any]],
) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for source in (
        task_ids or [],
        list(task_graph.keys()),
        list(task_results.keys()),
    ):
        for task_id in source:
            if not isinstance(task_id, str) or not task_id or task_id in seen:
                continue
            seen.add(task_id)
            ordered.append(task_id)
    return ordered


def _read_task_results(run_path: Path, task_ids: list[str]) -> dict[str, Any]:
    task_results: dict[str, Any] = {}
    for task_id in task_ids:
        result_file = run_path / f"{task_id}.result.json"
        if not result_file.exists():
            continue
        try:
            task_results[task_id] = json.loads(result_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
    return task_results


def _task_status(task_results: dict[str, Any], task_id: str) -> str:
    result = task_results.get(task_id)
    if not isinstance(result, dict):
        return "pending"
    status = result.get("status")
    return status if isinstance(status, str) and status else "pending"


def _task_owner(task_meta: dict[str, Any]) -> str | None:
    owner = task_meta.get("agent")
    return owner if isinstance(owner, str) and owner else None


def _task_runtime(task_meta: dict[str, Any], task_result: dict[str, Any] | None) -> str | None:
    engine = task_meta.get("engine")
    model = task_meta.get("model")
    if isinstance(engine, str) and engine:
        if isinstance(model, str) and model:
            return f"{engine}:{model}"
        return engine
    if isinstance(task_result, dict):
        inferred = _infer_model_from_task_result(task_result)
        if inferred:
            return inferred
    return None


def _format_activity_message(event_type: str, payload: dict[str, Any]) -> str | None:
    task_id = payload.get("task_id")
    task_label = task_id if isinstance(task_id, str) and task_id else "task"

    if event_type == "task_start":
        return f"{task_label} started"
    if event_type == "task_complete":
        status = payload.get("status")
        if isinstance(status, str) and status:
            return f"{task_label} finished as {status}"
        return f"{task_label} finished"
    if event_type == "task_skip":
        reason = payload.get("reason")
        if isinstance(reason, str) and reason:
            return f"{task_label} skipped: {reason}"
        return f"{task_label} skipped"
    if event_type == "task_progress":
        pct = payload.get("pct")
        step = payload.get("step")
        if isinstance(pct, (int, float)):
            message = f"{task_label} progress {int(pct)}%"
        else:
            message = f"{task_label} progress update"
        if isinstance(step, str) and step:
            message += f" ({step})"
        return message
    if event_type == "approval_required":
        return f"{task_label} is waiting for approval"
    if event_type == "approval_response":
        approved = payload.get("approved")
        if approved is True:
            return f"{task_label} approved"
        if approved is False:
            return f"{task_label} denied"
        return f"{task_label} approval updated"
    if event_type == "budget_exceeded":
        return "Budget exceeded"
    if event_type == "policy_violation":
        action = payload.get("action")
        if isinstance(action, str) and action:
            return f"Policy violation ({action})"
        return "Policy violation detected"
    return None


def _load_activity_feed(
    run_path: Path,
    task_graph: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    events_path = run_path / "events.jsonl"
    if not events_path.exists():
        return []

    try:
        records = replay_events(events_path)
    except OSError:
        return []

    activity: list[dict[str, Any]] = []
    for record in records:
        message = _format_activity_message(record.event_type, record.payload)
        if not message:
            continue
        task_id = record.payload.get("task_id")
        task_meta = (
            task_graph.get(task_id)
            if isinstance(task_id, str)
            else None
        ) or {}
        activity.append({
            "timestamp": record.timestamp,
            "event": record.event_type,
            "task_id": task_id if isinstance(task_id, str) else None,
            "owner": _task_owner(task_meta),
            "message": message,
        })

    if len(activity) <= _ACTIVITY_LIMIT:
        return activity
    return activity[-_ACTIVITY_LIMIT:]


def _build_collaboration(
    run_path: Path,
    task_ids: list[str],
    task_results: dict[str, Any],
    task_graph: dict[str, dict[str, Any]],
    *,
    include_activity: bool,
) -> dict[str, Any]:
    owners: dict[str, dict[str, Any]] = {}
    blocked_tasks: list[dict[str, Any]] = []
    unassigned_tasks: list[str] = []
    task_entries: dict[str, dict[str, Any]] = {}

    for task_id in task_ids:
        task_meta = task_graph.get(task_id, {})
        task_result = task_results.get(task_id)
        status = _task_status(task_results, task_id)
        depends_on = list(task_meta.get("depends_on", []))
        blocked_by: list[dict[str, str]] = []
        for dep_id in depends_on:
            dep_status = _task_status(task_results, dep_id)
            if dep_status not in _SUCCESS_LIKE:
                blocked_by.append({"task_id": dep_id, "status": dep_status})

        is_blocked = (
            status not in _TERMINAL_STATUSES
            and status != "running"
            and bool(blocked_by)
        )
        owner = _task_owner(task_meta)
        runtime = _task_runtime(task_meta, task_result if isinstance(task_result, dict) else None)
        last_progress_pct = None
        if isinstance(task_result, dict):
            progress_raw = task_result.get("last_progress_pct")
            if isinstance(progress_raw, int):
                last_progress_pct = progress_raw

        task_entries[task_id] = {
            "task_id": task_id,
            "description": task_meta.get("description", ""),
            "owner": owner,
            "runtime": runtime,
            "depends_on": depends_on,
            "blocked_by": blocked_by,
            "is_blocked": is_blocked,
            "allow_failure": bool(task_meta.get("allow_failure", False)),
            "worktree": bool(task_meta.get("worktree", False)),
            "last_progress_pct": last_progress_pct,
        }

        if owner:
            bucket = owners.setdefault(owner, {
                "label": owner,
                "task_count": 0,
                "completed_count": 0,
                "active_count": 0,
                "blocked_count": 0,
            })
            bucket["task_count"] += 1
            if status in _SUCCESS_LIKE:
                bucket["completed_count"] += 1
            if status not in _TERMINAL_STATUSES:
                bucket["active_count"] += 1
            if is_blocked:
                bucket["blocked_count"] += 1
        else:
            unassigned_tasks.append(task_id)

        if is_blocked:
            blocked_tasks.append({
                "task_id": task_id,
                "owner": owner,
                "blocked_by": blocked_by,
            })

    owner_list = sorted(
        owners.values(),
        key=lambda item: (-int(item["task_count"]), str(item["label"])),
    )
    activity = _load_activity_feed(run_path, task_graph) if include_activity else []

    return {
        "owner_count": len(owner_list),
        "owners": owner_list,
        "blocked_count": len(blocked_tasks),
        "blocked_tasks": blocked_tasks[:12],
        "unassigned_count": len(unassigned_tasks),
        "unassigned_tasks": unassigned_tasks[:12],
        "activity": activity,
        "activity_count": len(activity),
        "tasks": task_entries,
    }


def _build_collaboration_summary(
    run_path: Path,
    task_ids: list[str],
    task_results: dict[str, Any],
    task_graph: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    collaboration = _build_collaboration(
        run_path,
        task_ids,
        task_results,
        task_graph,
        include_activity=False,
    )
    owners = collaboration.get("owners", [])
    top_owners = [
        owner.get("label")
        for owner in owners[:2]
        if isinstance(owner, dict) and isinstance(owner.get("label"), str)
    ]
    return {
        "owner_count": collaboration["owner_count"],
        "blocked_count": collaboration["blocked_count"],
        "unassigned_count": collaboration["unassigned_count"],
        "top_owners": top_owners,
    }


def _enrich_run_payload(
    payload: dict[str, Any],
    run_path: Path,
    *,
    task_ids: list[str] | None = None,
    task_graph: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    task_results = payload.get("task_results")
    if not isinstance(task_results, dict):
        task_results = {}
        payload["task_results"] = task_results

    normalized_graph = _normalize_task_graph(payload.get("task_graph"))
    if not normalized_graph and task_graph:
        normalized_graph = _normalize_task_graph(task_graph)
    if normalized_graph:
        payload["task_graph"] = normalized_graph

    explicit_task_ids = task_ids
    if explicit_task_ids is None:
        raw_task_ids = payload.get("task_ids")
        explicit_task_ids = (
            [task_id for task_id in raw_task_ids if isinstance(task_id, str)]
            if isinstance(raw_task_ids, list)
            else None
        )
    ordered_task_ids = _ordered_task_ids(explicit_task_ids, task_results, normalized_graph)
    if ordered_task_ids:
        payload["task_ids"] = ordered_task_ids

    collaboration = _build_collaboration(
        run_path,
        ordered_task_ids,
        task_results,
        normalized_graph,
        include_activity=True,
    )
    payload["collaboration"] = collaboration
    payload["collaboration_summary"] = _build_collaboration_summary(
        run_path,
        ordered_task_ids,
        task_results,
        normalized_graph,
    )
    return payload


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ValidateRequest(BaseModel):
    yaml_content: str | None = None
    path: str | None = None


class RunRequest(BaseModel):
    plan_path: str | None = None
    yaml_content: str | None = None
    dry_run: bool = False
    execution_profile: str = "plan"
    max_parallel: int | None = None
    only: list[str] | None = None
    skip: list[str] | None = None


class CleanupRequest(BaseModel):
    plan_path: str
    keep: int = 10
    older_than_days: int | None = None
    dry_run: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@router.get("/plans/examples")
async def list_example_plans() -> list[dict[str, str]]:
    """List YAML plan files from examples/ and plans/ directories."""
    results: list[dict[str, str]] = []
    base = get_project_root()
    for dirname in ("examples", "plans"):
        d = base / dirname
        if not d.exists() or not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.is_file() and f.suffix in (".yaml", ".yml"):
                results.append({
                    "name": f.stem,
                    "path": str(f.relative_to(base)).replace("\\", "/"),
                })
    return results


@router.post("/plans/validate")
async def validate_plan(req: ValidateRequest) -> dict[str, Any]:
    if req.path:
        plan_path = Path(req.path)
    elif req.yaml_content:
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False,
        )
        try:
            tmp.write(req.yaml_content)
            tmp.close()
            plan_path = Path(tmp.name)
        except Exception:
            Path(tmp.name).unlink(missing_ok=True)
            raise
    else:
        raise HTTPException(status_code=400, detail="Provide 'yaml_content' or 'path'")

    try:
        plan = load_plan(plan_path)
        return {
            "valid": True,
            "plan": {
                "name": plan.name,
                "tasks": len(plan.tasks),
                "task_ids": [t.id for t in plan.tasks],
                "max_parallel": plan.max_parallel,
                "fail_fast": plan.fail_fast,
                "task_details": [
                    {
                        "id": t.id,
                        "description": t.description or "",
                        "engine": t.engine,
                        "model": t.model,
                        "has_command": bool(t.command),
                        "depends_on": t.depends_on,
                        "allow_failure": t.allow_failure,
                    }
                    for t in plan.tasks
                ],
            },
        }
    except PlanValidationError as exc:
        return {"valid": False, "error": str(exc)}
    finally:
        if req.yaml_content and plan_path.exists():
            plan_path.unlink(missing_ok=True)


@router.get("/files/browse")
async def browse_files() -> list[dict[str, str]]:
    """List YAML plan files from project root (max 2 levels deep)."""
    results: list[dict[str, str]] = []
    base = get_project_root()
    _yaml_suffixes = {".yaml", ".yml"}
    try:
        for item in sorted(base.iterdir()):
            if item.is_file() and item.suffix in _yaml_suffixes:
                results.append({
                    "name": item.name,
                    "path": str(item.relative_to(base)).replace("\\", "/"),
                    "dir": "",
                })
            elif item.is_dir() and not item.name.startswith("."):
                try:
                    for child in sorted(item.iterdir()):
                        if child.is_file() and child.suffix in _yaml_suffixes:
                            results.append({
                                "name": child.name,
                                "path": str(child.relative_to(base)).replace("\\", "/"),
                                "dir": item.name,
                            })
                except OSError:
                    pass
    except OSError:
        pass
    return results


@router.post("/runs")
async def start_run(req: RunRequest) -> dict[str, Any]:
    import tempfile as _tempfile

    if req.yaml_content and not req.plan_path:
        # Drag & drop: write content to temp file for loading
        tmp = _tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", encoding="utf-8", delete=False,
        )
        tmp.write(req.yaml_content)
        tmp.close()
        plan_path: str | Path = Path(tmp.name)
    elif req.plan_path:
        plan_path = req.plan_path
    else:
        raise HTTPException(
            status_code=400, detail="Provide 'plan_path' or 'yaml_content'",
        )

    try:
        plan = load_plan(plan_path)
    except PlanValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    task_ids = [t.id for t in plan.tasks]
    only = set(req.only) if req.only else None
    skip = set(req.skip) if req.skip else None

    # We need to capture the run_path from inside the thread.
    # run_plan returns PlanRunResult which contains the run_path.
    state_holder: dict[str, Any] = {}

    def _run() -> None:
        try:
            result = run_plan(
                plan,
                dry_run=req.dry_run,
                execution_profile=cast(ExecutionProfile, req.execution_profile),
                max_parallel_override=req.max_parallel,
                only=only,
                skip=skip,
            )
            state_holder["result"] = result
            # The API uses directory-name IDs (e.g. "<run_id>_<plan_name>"),
            # while scheduler result.run_id is the short ID. Resolve robustly.
            rs = get_run(result.run_path.name) or get_run(result.run_id)
            if rs is None:
                for candidate in list_active_runs():
                    if candidate.thread is threading.current_thread():
                        rs = candidate
                        break
            if rs:
                rs.execution_profile = result.execution_profile
                rs.result = result
        except Exception as exc:
            # Store error on the run state if available
            for rs in list_active_runs():
                if rs.thread is threading.current_thread():
                    rs.error = str(exc)
                    break

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    # Give the scheduler a moment to create the run directory
    thread.join(timeout=2)

    run_id = "unknown"
    run_path = Path(".")
    initial_result = state_holder.get("result")

    if isinstance(initial_result, PlanRunResult):
        result_run_path = initial_result.run_path
        result_run_name = result_run_path.name
        if result_run_name and result_run_name != ".":
            run_path = result_run_path
            run_id = result_run_name
    if run_id == "unknown":
        # Find the run state — the scheduler creates the run dir with a timestamped name.
        # We detect it by finding the newest dir in the run root.
        run_dir_base = resolve_path(plan.source_dir, plan.run_dir)

        if run_dir_base and run_dir_base.exists():
            dirs = sorted(run_dir_base.iterdir(), key=lambda d: d.stat().st_mtime, reverse=True)
            if dirs:
                run_path = dirs[0]
                run_id = run_path.name

    state = RunState(
        run_id=run_id,
        plan_name=plan.name,
        task_ids=task_ids,
        run_path=run_path,
        started_at=now_utc(),
        thread=thread,
        execution_profile=req.execution_profile,
        dry_run=req.dry_run,
        result=initial_result if isinstance(initial_result, PlanRunResult) else None,
        task_graph=_build_task_graph_from_tasks(plan.tasks),
    )
    register_run(state)

    return {
        "run_id": run_id,
        "plan_name": plan.name,
        "tasks": task_ids,
        "run_path": str(run_path),
    }


@router.get("/runs")
async def list_runs(plan_path: str | None = None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []

    # Active runs from state
    for rs in list_active_runs():
        summary = rs.to_summary()
        task_results = _read_task_results(rs.run_path, rs.task_ids)
        summary["collaboration_summary"] = _build_collaboration_summary(
            rs.run_path,
            rs.task_ids,
            task_results,
            _normalize_task_graph(rs.task_graph),
        )
        runs.append(summary)

    # Historical runs from filesystem
    run_roots: list[Path | None] = []
    if plan_path:
        try:
            plan = load_plan(plan_path)
            run_roots = [resolve_path(plan.source_dir, plan.run_dir)]
        except Exception:
            pass
    else:
        run_roots.extend(_discover_run_roots())

    active_ids = {rs.run_id for rs in list_active_runs()}
    all_dirs: list[tuple[float, Path]] = []
    for run_root in run_roots:
        if run_root is None or not run_root.exists():
            continue
        for d in run_root.iterdir():
            if d.is_dir() and d.name not in active_ids:
                try:
                    all_dirs.append((d.stat().st_mtime, d))
                except OSError:
                    pass
    all_dirs.sort(key=lambda x: x[0], reverse=True)

    for _mtime, d in all_dirs:
        manifest_path = d / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                task_results = manifest.get("task_results", {})
                if not isinstance(task_results, dict):
                    task_results = {}
                task_graph = _normalize_task_graph(manifest.get("task_graph"))
                task_ids = _ordered_task_ids(None, task_results, task_graph)
                runs.append({
                    "run_id": d.name,
                    "plan_name": manifest.get("plan_name", ""),
                    "started_at": manifest.get("started_at", ""),
                    "finished_at": manifest.get("finished_at"),
                    "duration_sec": _duration_from_manifest(manifest),
                    "success": manifest.get("success"),
                    "active": False,
                    "task_count": len(task_results),
                    "execution_profile": manifest.get("execution_profile", "plan"),
                    "dry_run": _is_dry_run_from_manifest(manifest),
                    "total_cost_usd": _total_cost_from_manifest(manifest),
                    "run_path": str(d),
                    "collaboration_summary": _build_collaboration_summary(
                        d,
                        task_ids,
                        task_results,
                        task_graph,
                    ),
                })
            except (json.JSONDecodeError, OSError):
                pass

    return runs


@router.get("/runs/roots")
async def list_run_roots() -> dict[str, Any]:
    project_roots = get_project_roots()
    roots = _discover_run_roots()
    return {
        "project_root": str(project_roots[0]) if project_roots else "",
        "project_roots": [str(r) for r in project_roots],
        "count": len(roots),
        "run_roots": [str(r) for r in roots],
    }


@router.get("/runs/stats")
async def get_runs_stats(plan_path: str | None = None) -> dict[str, Any]:
    """Aggregate metrics across all completed runs."""
    from datetime import datetime

    run_roots: list[Path | None] = []
    if plan_path:
        try:
            plan = load_plan(plan_path)
            run_roots = [resolve_path(plan.source_dir, plan.run_dir)]
        except Exception:
            pass
    else:
        run_roots.extend(_discover_run_roots())

    # Collect all run dirs across all roots, sorted by mtime desc
    all_dirs: list[tuple[float, Path]] = []
    for run_root in run_roots:
        if run_root is None or not run_root.exists():
            continue
        for d in run_root.iterdir():
            if d.is_dir():
                try:
                    all_dirs.append((d.stat().st_mtime, d))
                except OSError:
                    pass
    all_dirs.sort(key=lambda x: x[0], reverse=True)

    manifests: list[dict[str, Any]] = []
    for _mtime, d in all_dirs:
        manifest_path = d / "run_manifest.json"
        if manifest_path.exists():
            try:
                manifests.append(
                    json.loads(manifest_path.read_text(encoding="utf-8"))
                )
            except (json.JSONDecodeError, OSError):
                pass

    total_runs = len(manifests)
    success_count = sum(1 for m in manifests if m.get("success") is True)
    failed_count = sum(1 for m in manifests if m.get("success") is False)

    # Duration aggregation
    durations: list[float] = []
    for m in manifests:
        sa = m.get("started_at")
        fa = m.get("finished_at")
        if sa and fa:
            try:
                start = datetime.fromisoformat(sa)
                end = datetime.fromisoformat(fa)
                durations.append((end - start).total_seconds())
            except (ValueError, TypeError):
                pass
    avg_duration_sec = sum(durations) / len(durations) if durations else 0.0

    # Cost aggregation
    costs: list[float] = []
    for m in manifests:
        c = m.get("total_cost_usd")
        if c is not None:
            costs.append(float(c))
    total_cost_usd = sum(costs) if costs else None

    # Token aggregation
    tokens_list: list[int] = []
    for m in manifests:
        t = _total_tokens_from_manifest(m)
        if t is not None:
            tokens_list.append(t)
    total_tokens: int | None = sum(tokens_list) if tokens_list else None
    avg_tokens_per_run: int | None = (
        round(total_tokens / len(tokens_list))
        if tokens_list and total_tokens is not None
        else None
    )

    # Status distribution across all tasks
    status_dist: dict[str, int] = {}
    model_cost_map: dict[str, dict[str, Any]] = {}
    model_token_map: dict[str, dict[str, Any]] = {}
    for m in manifests:
        for tr in (m.get("task_results") or {}).values():
            if not isinstance(tr, dict):
                continue
            s = tr.get("status", "unknown")
            status_dist[s] = status_dist.get(s, 0) + 1

            model = _infer_model_from_task_result(tr) or "unknown"

            c = tr.get("cost_usd")
            if c is not None:
                try:
                    cost_val = float(c)
                except (TypeError, ValueError):
                    cost_val = 0.0
                if cost_val > 0:
                    bucket = model_cost_map.setdefault(model, {
                        "model": model,
                        "total_cost_usd": 0.0,
                        "task_count": 0,
                    })
                    bucket["total_cost_usd"] += cost_val
                    bucket["task_count"] += 1

            tu = tr.get("token_usage")
            if isinstance(tu, dict):
                tok = tu.get("total_tokens")
                if tok is not None:
                    try:
                        tok_val = int(tok)
                    except (TypeError, ValueError):
                        tok_val = 0
                    if tok_val > 0:
                        tbucket = model_token_map.setdefault(model, {
                            "model": model,
                            "total_tokens": 0,
                            "task_count": 0,
                        })
                        tbucket["total_tokens"] += tok_val
                        tbucket["task_count"] += 1

    # Recent runs (latest 20)
    recent_runs: list[dict[str, Any]] = []
    for m in manifests[:20]:
        sa = m.get("started_at")
        fa = m.get("finished_at")
        dur = None
        if sa and fa:
            try:
                dur = (
                    datetime.fromisoformat(fa) - datetime.fromisoformat(sa)
                ).total_seconds()
            except (ValueError, TypeError):
                pass
        recent_runs.append({
            "run_id": m.get("run_id", ""),
            "plan_name": m.get("plan_name", ""),
            "success": m.get("success"),
            "duration_sec": dur,
            "total_cost_usd": m.get("total_cost_usd"),
            "total_tokens": _total_tokens_from_manifest(m),
            "task_count": len(m.get("task_results", {})),
            "started_at": sa,
        })

    # Cost by run (latest 20 with cost)
    cost_by_run: list[dict[str, Any]] = []
    for m in manifests:
        c = m.get("total_cost_usd")
        if c is not None:
            cost_by_run.append({
                "run_id": m.get("run_id", ""),
                "plan_name": m.get("plan_name", ""),
                "total_cost_usd": c,
                "started_at": m.get("started_at"),
            })
            if len(cost_by_run) >= 20:
                break

    # Cost by model (top 12 by total cost)
    cost_by_model: list[dict[str, Any]] = []
    for item in sorted(
        model_cost_map.values(),
        key=lambda x: float(x.get("total_cost_usd", 0.0)),
        reverse=True,
    ):
        task_count = int(item.get("task_count", 0))
        if task_count <= 0:
            continue
        total = float(item.get("total_cost_usd", 0.0))
        cost_by_model.append({
            "model": item.get("model", "unknown"),
            "total_cost_usd": total,
            "task_count": task_count,
            "avg_cost_usd": total / task_count,
        })
        if len(cost_by_model) >= 12:
            break

    # Tokens by model (top 12 by total tokens)
    tokens_by_model: list[dict[str, Any]] = []
    for item in sorted(
        model_token_map.values(),
        key=lambda x: int(x.get("total_tokens", 0)),
        reverse=True,
    ):
        task_count = int(item.get("task_count", 0))
        if task_count <= 0:
            continue
        tok_total = int(item.get("total_tokens", 0))
        tokens_by_model.append({
            "model": item.get("model", "unknown"),
            "total_tokens": tok_total,
            "task_count": task_count,
            "avg_tokens": tok_total // task_count,
        })
        if len(tokens_by_model) >= 12:
            break

    return {
        "total_runs": total_runs,
        "success_count": success_count,
        "failed_count": failed_count,
        "total_cost_usd": total_cost_usd,
        "total_tokens": total_tokens,
        "avg_tokens_per_run": avg_tokens_per_run,
        "avg_duration_sec": round(avg_duration_sec, 2),
        "recent_runs": recent_runs,
        "cost_by_run": cost_by_run,
        "cost_by_model": cost_by_model,
        "tokens_by_model": tokens_by_model,
        "status_distribution": status_dist,
    }


@router.get("/runs/{run_id}")
async def get_run_detail(run_id: str) -> dict[str, Any]:
    # Check active runs first
    rs = get_run(run_id)
    if rs:
        if rs.result:
            return _enrich_run_payload(
                rs.result.to_dict(),
                rs.run_path,
                task_ids=rs.task_ids,
                task_graph=cast(dict[str, dict[str, Any]], rs.task_graph),
            )
        if rs.is_finished:
            manifest_path = rs.run_path / "run_manifest.json"
            if manifest_path.exists():
                try:
                    result: dict[str, Any] = json.loads(
                        manifest_path.read_text(encoding="utf-8"),
                    )
                    return _enrich_run_payload(
                        result,
                        rs.run_path,
                        task_ids=rs.task_ids,
                        task_graph=cast(dict[str, dict[str, Any]], rs.task_graph),
                    )
                except (json.JSONDecodeError, OSError):
                    pass
        # Run is still active — return partial info
        summary = rs.to_summary()
        summary["task_results"] = _read_task_results(rs.run_path, rs.task_ids)
        return _enrich_run_payload(
            summary,
            rs.run_path,
            task_ids=rs.task_ids,
            task_graph=cast(dict[str, dict[str, Any]], rs.task_graph),
        )

    # Check filesystem for completed run
    for base in _discover_run_roots():
        run_dir = base / run_id
        if run_dir.exists():
            manifest_path = run_dir / "run_manifest.json"
            if manifest_path.exists():
                data: dict[str, Any] = json.loads(
                    manifest_path.read_text(encoding="utf-8"),
                )
                return _enrich_run_payload(data, run_dir)

    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


@router.get("/runs/{run_id}/tasks/{task_id}/log")
async def get_task_log(run_id: str, task_id: str) -> dict[str, str]:
    # Find the log file
    rs = get_run(run_id)
    if rs:
        log_path = rs.run_path / f"{task_id}.log"
    else:
        log_path = None
        for base in _discover_run_roots():
            candidate = base / run_id / f"{task_id}.log"
            if candidate.exists():
                log_path = candidate
                break

    if log_path is None or not log_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Log for task '{task_id}' in run '{run_id}' not found",
        )

    content = log_path.read_text(encoding="utf-8", errors="replace")
    return {"task_id": task_id, "content": content}


@router.delete("/runs/{run_id}")
async def delete_run(run_id: str) -> dict[str, bool]:
    # Don't allow deleting active runs
    rs = get_run(run_id)
    if rs and not rs.is_finished:
        raise HTTPException(status_code=409, detail="Cannot delete an active run")

    if rs:
        remove_run(run_id)
        if rs.run_path.exists():
            shutil.rmtree(rs.run_path)
        return {"deleted": True}

    # Try filesystem
    for base in _discover_run_roots():
        run_dir = base / run_id
        if run_dir.exists():
            shutil.rmtree(run_dir)
            return {"deleted": True}

    raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")


@router.post("/cleanup")
async def cleanup(req: CleanupRequest) -> dict[str, Any]:
    try:
        plan = load_plan(req.plan_path)
    except PlanValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    run_root = resolve_path(plan.source_dir, plan.run_dir)
    if run_root is None or not run_root.exists():
        return {"deleted": [], "count": 0}

    deleted = cleanup_runs(
        run_root,
        keep=req.keep,
        older_than_days=req.older_than_days,
        dry_run=req.dry_run,
    )

    return {
        "deleted": [str(d) for d in deleted],
        "count": len(deleted),
        "dry_run": req.dry_run,
    }
