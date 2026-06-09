from __future__ import annotations

import hashlib
import json
import re
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .models import (
    CODEX_MODEL_ALIASES,
    COPILOT_MODEL_ALIASES,
    GEMINI_MODEL_ALIASES,
    LLAMA_MODEL_ALIASES,
    OLLAMA_MODEL_ALIASES,
    PlanSpec,
    QWEN_MODEL_ALIASES,
    TaskSpec,
)
from .utils import extract_prompt_from_markdown, render_template, resolve_path

_CODEX_MODEL_ALIASES = CODEX_MODEL_ALIASES
_GEMINI_MODEL_ALIASES = GEMINI_MODEL_ALIASES
_COPILOT_MODEL_ALIASES = COPILOT_MODEL_ALIASES
_QWEN_MODEL_ALIASES = QWEN_MODEL_ALIASES
_OLLAMA_MODEL_ALIASES = OLLAMA_MODEL_ALIASES
_LLAMA_MODEL_ALIASES = LLAMA_MODEL_ALIASES
_CLAUDE_MODEL_ALIASES: dict[str, str] = {
    "haiku": "haiku",
    "sonnet": "sonnet",
    "opus": "opus",
    "opusplan": "opusplan",
}

_CODEX_DANGEROUS_FLAG = "--dangerously-bypass-approvals-and-sandbox"
_CLAUDE_DANGEROUS_FLAG = "--dangerously-skip-permissions"
_GEMINI_DANGEROUS_FLAG_OPTION = "--approval-mode"
_GEMINI_DANGEROUS_FLAG_VALUE = "yolo"
_CACHE_SCHEMA_VERSION = 1
_CACHE_POLICY_VERSION = 1
SIMULATION_CACHE_POLICY_VERSION = 1
_DEFAULT_NEGATIVE_CACHE_TTL_SEC = 300
_PLAN_HASH_SCHEMA_VERSION = 1
_SIMULATION_PLAN_HASH_SCHEMA_VERSION = 1


def _resolve_codex_model(model: str | None) -> str | None:
    if model is None:
        return None
    return _CODEX_MODEL_ALIASES.get(model, model)


def _resolve_gemini_model(model: str | None) -> str | None:
    if model is None:
        return None
    return _GEMINI_MODEL_ALIASES.get(model, model)


def _resolve_copilot_model(model: str | None) -> str | None:
    if model is None:
        return None
    return _COPILOT_MODEL_ALIASES.get(model, model)


def _resolve_qwen_model(model: str | None) -> str | None:
    if model is None:
        return None
    return _QWEN_MODEL_ALIASES.get(model, model)


def _resolve_ollama_model(model: str | None) -> str | None:
    if model is None:
        return None
    return _OLLAMA_MODEL_ALIASES.get(model, model)


def _resolve_claude_model(model: str | None) -> str | None:
    if model is None:
        return None
    return _CLAUDE_MODEL_ALIASES.get(model, model)


def _resolve_llama_model(model: str | None) -> str | None:
    if model is None:
        return None
    return _LLAMA_MODEL_ALIASES.get(model, model)


def _resolve_model_for_engine(engine: str, model: str | None) -> str | None:
    if engine == "codex":
        return _resolve_codex_model(model)
    if engine == "claude":
        return _resolve_claude_model(model)
    if engine == "gemini":
        return _resolve_gemini_model(model)
    if engine == "copilot":
        return _resolve_copilot_model(model)
    if engine == "qwen":
        return _resolve_qwen_model(model)
    if engine == "ollama":
        return _resolve_ollama_model(model)
    if engine == "llama":
        return _resolve_llama_model(model)
    return model


def _stem_model_family(model: str) -> str:
    token = model.strip().lower().split(":", 1)[0]
    token = token.rsplit("/", 1)[-1]
    token = re.sub(r"[^a-z0-9]+", "-", token)
    token = re.sub(r"-?\d+(?:[a-z]+)?$", "", token).strip("-")
    return token or model.strip().lower()


def model_family_for_engine(engine: str, model: str | None) -> str | None:
    resolved = _resolve_model_for_engine(engine, model)
    if resolved is None:
        return None

    normalized = str(resolved).strip().lower()
    if not normalized:
        return None

    if engine in {"codex", "claude", "gemini", "qwen"}:
        return engine

    if engine == "copilot":
        if any(token in normalized for token in ("claude", "haiku", "sonnet", "opus")):
            return "anthropic"
        if "gemini" in normalized:
            return "gemini"
        if "grok" in normalized:
            return "grok"
        if any(token in normalized for token in ("gpt", "codex", "o1", "o3", "o4")):
            return "openai"
        return "copilot"

    if engine in {"ollama", "llama"}:
        return _stem_model_family(normalized)

    return normalized


def simulation_model_families(plan: PlanSpec) -> list[str]:
    """Return per-task model-family labels for simulation-cache lookups."""
    families: list[str] = []
    for task in sorted(plan.tasks, key=lambda item: item.id):
        engine = task.engine or ""
        if not engine:
            continue
        config = _effective_engine_config(task, plan, use_model_family=True)
        family = config.get("model_family")
        if family:
            families.append(f"{task.id}:{family}")
    return families


def _normalize_codex_args(args: list[str]) -> list[str]:
    normalized: list[str] = []
    has_dangerous = False

    for arg in args:
        if arg == "--yolo":
            arg = _CODEX_DANGEROUS_FLAG
        if arg == _CODEX_DANGEROUS_FLAG:
            has_dangerous = True
        normalized.append(arg)

    if not has_dangerous:
        return normalized

    out: list[str] = []
    seen_dangerous = False
    for arg in normalized:
        if arg == _CODEX_DANGEROUS_FLAG:
            if seen_dangerous:
                continue
            seen_dangerous = True
        out.append(arg)
    return out


def _normalize_claude_args(args: list[str]) -> list[str]:
    out: list[str] = []
    seen_dangerous = False

    for arg in args:
        if arg == _CLAUDE_DANGEROUS_FLAG:
            if seen_dangerous:
                continue
            seen_dangerous = True
        out.append(arg)

    return out


def _normalize_gemini_args(args: list[str]) -> list[str]:
    expanded: list[str] = []
    for arg in args:
        if arg == "--yolo":
            expanded += [_GEMINI_DANGEROUS_FLAG_OPTION, _GEMINI_DANGEROUS_FLAG_VALUE]
        else:
            expanded.append(arg)

    out: list[str] = []
    seen_approval_mode = False
    skip_next = False
    for arg in expanded:
        if skip_next:
            skip_next = False
            continue
        if arg == _GEMINI_DANGEROUS_FLAG_OPTION:
            if seen_approval_mode:
                skip_next = True
                continue
            seen_approval_mode = True
        out.append(arg)
    return out


def _normalize_copilot_args(args: list[str]) -> list[str]:
    out: list[str] = []
    seen_yolo = False
    for arg in args:
        if arg in ("--yolo", "--allow-all"):
            if seen_yolo:
                continue
            seen_yolo = True
            out.append("--yolo")
        else:
            out.append(arg)
    return out


def _resolve_edit_policy(plan: PlanSpec, task: TaskSpec) -> str:
    return task.edit_policy or plan.defaults.edit_policy


def _resolve_append_system_prompt(plan: PlanSpec, task: TaskSpec, engine: str) -> str | None:
    custom = task.append_system_prompt
    if custom is None and engine == "codex":
        custom = plan.defaults.codex.append_system_prompt
    elif custom is None and engine == "claude":
        custom = plan.defaults.claude.append_system_prompt
    elif custom is None and engine == "gemini":
        custom = plan.defaults.gemini.append_system_prompt
    elif custom is None and engine == "copilot":
        custom = plan.defaults.copilot.append_system_prompt
    elif custom is None and engine == "qwen":
        custom = plan.defaults.qwen.append_system_prompt
    elif custom is None and engine == "ollama":
        custom = plan.defaults.ollama.append_system_prompt
    return custom


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _resolve_prompt_path(plan: PlanSpec, relative_path: str) -> Path | None:
    """Resolve prompt_file / prompt_md_file — workspace_root first, then source_dir."""
    if not relative_path:
        return None
    p = Path(relative_path)
    if p.is_absolute():
        return p
    if plan.workspace_root:
        ws_path = (Path(plan.workspace_root).resolve() / p).resolve()
        if ws_path.exists():
            return ws_path
    return resolve_path(plan.source_dir, relative_path)


def _load_prompt_content(task: TaskSpec, plan: PlanSpec) -> dict[str, str]:
    source_type: str
    raw_content: str
    source_ref: str

    if task.prompt is not None:
        source_type = "inline"
        raw_content = task.prompt
        source_ref = "inline"
    elif task.prompt_file:
        source_type = "file"
        prompt_path = _resolve_prompt_path(plan, task.prompt_file)
        if prompt_path is None or not prompt_path.exists():
            raise FileNotFoundError(f"Task '{task.id}' prompt_file not found: {task.prompt_file}")
        raw_content = _read_text_file(prompt_path)
        source_ref = str(prompt_path)
    elif task.prompt_md_file and task.prompt_md_heading:
        source_type = "markdown"
        md_path = _resolve_prompt_path(plan, task.prompt_md_file)
        if md_path is None or not md_path.exists():
            raise FileNotFoundError(f"Task '{task.id}' prompt_md_file not found: {task.prompt_md_file}")
        md_text = _read_text_file(md_path)
        raw_content = extract_prompt_from_markdown(md_text, task.prompt_md_heading)
        source_ref = f"{md_path}#{task.prompt_md_heading}"
    else:
        raise ValueError(f"Task '{task.id}' has no prompt source")

    rendered_content = render_template(
        raw_content,
        {
            "workspace_root": str(Path(plan.workspace_root).resolve()) if plan.workspace_root else "",
            "plan_name": plan.name,
            "task_id": task.id,
        },
    )

    return {
        "source_type": source_type,
        "source_ref": source_ref,
        "rendered_content": rendered_content,
    }


def _effective_engine_config(
    task: TaskSpec,
    plan: PlanSpec,
    *,
    use_model_family: bool = False,
) -> dict[str, Any]:
    engine = task.engine or ""
    model_key = "model_family" if use_model_family else "model"
    config: dict[str, Any] = {
        "engine": engine,
        "agent": task.agent,
        "edit_policy": _resolve_edit_policy(plan, task),
        "append_system_prompt": _resolve_append_system_prompt(plan, task, engine),
    }

    if engine == "codex":
        resolved_model = _resolve_codex_model(task.model or plan.defaults.codex.model)
        config[model_key] = (
            model_family_for_engine(engine, resolved_model)
            if use_model_family
            else resolved_model
        )
        config["reasoning_effort"] = task.reasoning_effort or plan.defaults.codex.reasoning_effort
        config["args"] = sorted(_normalize_codex_args(plan.defaults.codex.args + task.args))
        return config

    if engine == "claude":
        resolved_model = _resolve_claude_model(task.model or plan.defaults.claude.model)
        config[model_key] = (
            model_family_for_engine(engine, resolved_model)
            if use_model_family
            else resolved_model
        )
        config["reasoning_effort"] = task.reasoning_effort or plan.defaults.claude.reasoning_effort
        config["args"] = sorted(_normalize_claude_args(plan.defaults.claude.args + task.args))
        return config

    if engine == "gemini":
        resolved_model = _resolve_gemini_model(task.model or plan.defaults.gemini.model)
        config[model_key] = (
            model_family_for_engine(engine, resolved_model)
            if use_model_family
            else resolved_model
        )
        config["reasoning_effort"] = task.reasoning_effort or plan.defaults.gemini.reasoning_effort
        config["args"] = sorted(_normalize_gemini_args(plan.defaults.gemini.args + task.args))
        return config

    if engine == "copilot":
        resolved_model = _resolve_copilot_model(task.model or plan.defaults.copilot.model)
        config[model_key] = (
            model_family_for_engine(engine, resolved_model)
            if use_model_family
            else resolved_model
        )
        config["reasoning_effort"] = None  # Copilot CLI does not support reasoning_effort
        config["args"] = sorted(_normalize_copilot_args(plan.defaults.copilot.args + task.args))
        return config

    if engine == "qwen":
        resolved_model = _resolve_qwen_model(task.model or plan.defaults.qwen.model)
        config[model_key] = (
            model_family_for_engine(engine, resolved_model)
            if use_model_family
            else resolved_model
        )
        config["reasoning_effort"] = None
        config["args"] = sorted(plan.defaults.qwen.args + task.args)
        return config

    if engine == "ollama":
        resolved_model = _resolve_ollama_model(task.model or plan.defaults.ollama.model or "llama3")
        config[model_key] = (
            model_family_for_engine(engine, resolved_model)
            if use_model_family
            else resolved_model
        )
        config["reasoning_effort"] = None
        config["args"] = sorted(plan.defaults.ollama.args + task.args)
        return config

    if engine == "llama":
        resolved_model = _resolve_llama_model(task.model or plan.defaults.llama.model)
        config[model_key] = (
            model_family_for_engine(engine, resolved_model)
            if use_model_family
            else resolved_model
        )
        config["reasoning_effort"] = None
        config["args"] = sorted(plan.defaults.llama.args + task.args)
        return config

    config[model_key] = (
        model_family_for_engine(engine, task.model)
        if use_model_family
        else task.model
    )
    config["reasoning_effort"] = task.reasoning_effort
    config["args"] = sorted(task.args)
    return config


def _serialize_task_input_payload(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _effective_negative_cache_ttl_sec(
    task: TaskSpec | None,
    explicit_ttl_sec: int | None = None,
) -> int:
    if explicit_ttl_sec is not None:
        return max(0, explicit_ttl_sec)
    if task is not None and task.negative_cache_ttl_sec is not None:
        return max(0, task.negative_cache_ttl_sec)
    return _DEFAULT_NEGATIVE_CACHE_TTL_SEC


def _parse_cache_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _has_partial_output(result: Any) -> bool:
    handoff_report = getattr(result, "handoff_report", None)
    if handoff_report is None:
        return False
    partial_output = getattr(handoff_report, "partial_output", "")
    return bool(str(partial_output).strip())


def _has_tool_failures(result: Any) -> bool:
    return int(getattr(result, "tool_failure_count", 0) or 0) > 0


def _is_cacheable_result(task: TaskSpec | None, result: Any) -> bool:
    if bool(getattr(result, "tainted", False)):
        return False
    if task is not None and task.context_trust == "untrusted":
        return False
    if _has_partial_output(result):
        return False
    if _has_tool_failures(result):
        return False
    return True


def _compute_task_hash(
    task: TaskSpec,
    plan: PlanSpec,
    upstream_hashes: dict[str, str] | None = None,
    *,
    use_model_family: bool = False,
    cache_policy_version: int = _CACHE_POLICY_VERSION,
) -> str:
    """Compute SHA-256 hash of task inputs for caching (Merkle DAG)."""
    merged_env = dict(plan.defaults.env)
    merged_env.update(task.env)

    timeout_sec = (
        task.timeout_sec if task.timeout_sec is not None else plan.defaults.timeout_sec
    )
    requires_clean_worktree = (
        task.requires_clean_worktree
        if task.requires_clean_worktree is not None
        else plan.defaults.requires_clean_worktree
    )
    stdout_tail_lines = (
        task.stdout_tail_lines
        if task.stdout_tail_lines is not None
        else plan.defaults.stdout_tail_lines
    )
    retry_delay_sec: list[float] | float | None = (
        task.retry_delay_sec
        if task.retry_delay_sec is not None
        else plan.defaults.retry_delay_sec
    )

    resolved_upstream_hashes = upstream_hashes or {}
    dependency_ids = sorted(set(task.depends_on))
    dependency_hashes = {dep_id: resolved_upstream_hashes.get(dep_id, "") for dep_id in dependency_ids}

    if task.command is not None:
        command_input: dict[str, Any] = {
            "kind": "command",
            "command": task.command,
            "shell": task.shell if task.shell is not None else isinstance(task.command, str),
        }
    else:
        command_input = {
            "kind": "engine",
            "prompt": _load_prompt_content(task, plan),
            "engine_config": _effective_engine_config(
                task,
                plan,
                use_model_family=use_model_family,
            ),
        }

    policy_input: dict[str, Any] = {
        "cache_policy_version": cache_policy_version,
        "context_trust": task.context_trust,
        "negative_cache_ttl_sec": _effective_negative_cache_ttl_sec(task),
        "observation_block": task.observation_block,
        "honeypot": task.honeypot,
        "mcp_tools": sorted(task.mcp_tools),
        "allowed_tools": sorted(task.allowed_tools) if task.allowed_tools is not None else None,
        "on_grant_violation": task.on_grant_violation,
        "output_scope": sorted(task.output_scope),
        "output_redact": sorted(task.output_redact),
        "context_allowlist": sorted(task.context_allowlist),
        "phantom_workspace": task.phantom_workspace,
        "trajectory_guard": task.trajectory_guard.to_dict() if task.trajectory_guard else None,
    }

    payload: dict[str, Any] = {
        "cache_schema_version": _CACHE_SCHEMA_VERSION,
        "task_id": task.id,
        "depends_on": dependency_ids,
        "consistency_group": list(task.consistency_group),
        "reconcile_after": list(task.reconcile_after),
        "context_from": list(task.context_from),
        "context_mode": task.context_mode,
        "command_input": command_input,
        "env": {k: merged_env[k] for k in sorted(merged_env)},
        "upstream_hashes": dependency_hashes,
        "pre_command": task.pre_command,
        "verify_command": task.verify_command,
        "guard_command": task.guard_command,
        "assertions": list(task.assertions),
        "contract_type": task.contract_type,
        "consumes_contracts": list(task.consumes_contracts),
        "timeout_sec": timeout_sec,
        "allow_failure": task.allow_failure,
        "max_retries": task.max_retries,
        "max_iterations": task.max_iterations,
        "retry_delay_sec": retry_delay_sec,
        "retry_strategy": task.retry_strategy,
        "requires_clean_worktree": requires_clean_worktree,
        "workdir": task.workdir or plan.workspace_root,
        "stdout_tail_lines": stdout_tail_lines,
        "when": task.when,
        "matrix_values": getattr(task, "matrix_values", None),
        "batch": task.batch.to_dict() if task.batch else None,
        "policy_input": policy_input,
    }
    hasher = hashlib.sha256()
    hasher.update(_serialize_task_input_payload(payload))

    # v0.6.0 fields
    hasher.update(b"checkpoint:")
    hasher.update(str(task.checkpoint).encode())

    if task.context_budget_tokens is not None:
        hasher.update(b"context_budget_tokens:")
        hasher.update(str(task.context_budget_tokens).encode())

    if task.judge is not None:
        hasher.update(b"judge:")
        # Includes v0.10.0 fields (`method`, `aggregation`, `preset`) via JudgeSpec.to_dict().
        hasher.update(_serialize_task_input_payload(task.judge.to_dict()))

    # v0.7.0 fields
    if task.workspace_index_exclude:
        hasher.update(b"workspace_index_exclude:")
        hasher.update(json.dumps(sorted(task.workspace_index_exclude)).encode())

    return hasher.hexdigest()


def compute_task_hash(
    task: TaskSpec,
    plan: PlanSpec,
    upstream_hashes: dict[str, str] | None = None,
) -> str:
    return _compute_task_hash(task, plan, upstream_hashes)


def compute_plan_hash(plan: PlanSpec) -> str:
    """Compute a normalized hash for the overall plan configuration."""
    task_map = {task.id: task for task in plan.tasks}
    task_hashes: dict[str, str] = {}
    visiting: set[str] = set()

    def _hash_task(task_id: str) -> str:
        existing = task_hashes.get(task_id)
        if existing is not None:
            return existing
        if task_id in visiting:
            raise ValueError(f"Dependency cycle detected while hashing plan at task '{task_id}'")
        task = task_map.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task '{task_id}' while hashing plan")

        visiting.add(task_id)
        upstream = {
            dep_id: _hash_task(dep_id)
            for dep_id in sorted(set(task.depends_on))
        }
        visiting.remove(task_id)
        task_hash = compute_task_hash(task, plan, upstream)
        task_hashes[task_id] = task_hash
        return task_hash

    for task_id in sorted(task_map):
        _hash_task(task_id)

    imports_payload = sorted(
        (imp.to_dict() for imp in plan.imports),
        key=lambda item: (
            str(item.get("prefix", "")),
            str(item.get("path", "")),
            json.dumps(item.get("overrides", {}), ensure_ascii=True, sort_keys=True),
        ),
    )
    policies_payload = sorted(
        (policy.to_dict() for policy in plan.policies),
        key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True),
    )
    mcp_servers_payload = sorted(
        (server.to_dict() for server in plan.mcp_servers),
        key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True),
    )

    payload: dict[str, Any] = {
        "plan_hash_schema_version": _PLAN_HASH_SCHEMA_VERSION,
        "goal": plan.goal,
        "firewall_model": plan.firewall_model,
        "max_parallel": plan.max_parallel,
        "fail_fast": plan.fail_fast,
        "max_cost_usd": plan.max_cost_usd,
        "budget_warning_pct": plan.budget_warning_pct,
        "routing_strategy": plan.routing_strategy,
        "control_flow_integrity": plan.control_flow_integrity,
        "budget_period": plan.budget_period,
        "audit_packs": sorted(plan.audit_packs),
        "imports": imports_payload,
        "policies": policies_payload,
        "mcp_servers": mcp_servers_payload,
        "watch": plan.watch.to_dict() if plan.watch else None,
        "circuit_breaker": plan.circuit_breaker.to_dict() if plan.circuit_breaker else None,
        "task_hashes": {task_id: task_hashes[task_id] for task_id in sorted(task_hashes)},
    }
    hasher = hashlib.sha256()
    hasher.update(_serialize_task_input_payload(payload))
    return hasher.hexdigest()


def compute_simulation_plan_hash(plan: PlanSpec) -> str:
    """Compute a normalized plan hash for simulation-cache reuse."""
    task_map = {task.id: task for task in plan.tasks}
    task_hashes: dict[str, str] = {}
    visiting: set[str] = set()

    def _hash_task(task_id: str) -> str:
        existing = task_hashes.get(task_id)
        if existing is not None:
            return existing
        if task_id in visiting:
            raise ValueError(f"Dependency cycle detected while hashing plan at task '{task_id}'")
        task = task_map.get(task_id)
        if task is None:
            raise ValueError(f"Unknown task '{task_id}' while hashing plan")

        visiting.add(task_id)
        upstream = {
            dep_id: _hash_task(dep_id)
            for dep_id in sorted(set(task.depends_on))
        }
        visiting.remove(task_id)
        task_hash = _compute_task_hash(
            task,
            plan,
            upstream,
            use_model_family=True,
            cache_policy_version=SIMULATION_CACHE_POLICY_VERSION,
        )
        task_hashes[task_id] = task_hash
        return task_hash

    for task_id in sorted(task_map):
        _hash_task(task_id)

    imports_payload = sorted(
        (imp.to_dict() for imp in plan.imports),
        key=lambda item: (
            str(item.get("prefix", "")),
            str(item.get("path", "")),
            json.dumps(item.get("overrides", {}), ensure_ascii=True, sort_keys=True),
        ),
    )
    policies_payload = sorted(
        (policy.to_dict() for policy in plan.policies),
        key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True),
    )
    mcp_servers_payload = sorted(
        (server.to_dict() for server in plan.mcp_servers),
        key=lambda item: json.dumps(item, ensure_ascii=True, sort_keys=True),
    )

    payload: dict[str, Any] = {
        "simulation_plan_hash_schema_version": _SIMULATION_PLAN_HASH_SCHEMA_VERSION,
        "simulation_cache_policy_version": SIMULATION_CACHE_POLICY_VERSION,
        "goal": plan.goal,
        "firewall_model": plan.firewall_model,
        "max_parallel": plan.max_parallel,
        "fail_fast": plan.fail_fast,
        "max_cost_usd": plan.max_cost_usd,
        "budget_warning_pct": plan.budget_warning_pct,
        "routing_strategy": plan.routing_strategy,
        "control_flow_integrity": plan.control_flow_integrity,
        "budget_period": plan.budget_period,
        "audit_packs": sorted(plan.audit_packs),
        "imports": imports_payload,
        "policies": policies_payload,
        "mcp_servers": mcp_servers_payload,
        "watch": plan.watch.to_dict() if plan.watch else None,
        "circuit_breaker": plan.circuit_breaker.to_dict() if plan.circuit_breaker else None,
        "simulation_model_families": simulation_model_families(plan),
        "task_hashes": {task_id: task_hashes[task_id] for task_id in sorted(task_hashes)},
    }
    hasher = hashlib.sha256()
    hasher.update(_serialize_task_input_payload(payload))
    return hasher.hexdigest()


def _cache_entry_dir(cache_dir: Path, task_hash: str) -> Path:
    return cache_dir / task_hash[:2] / task_hash


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cache_lookup(cache_dir: Path, task_hash: str) -> dict[str, Any] | None:
    """Look up a cached task result. Returns the stored result dict or None."""
    result_path = _cache_entry_dir(cache_dir, task_hash) / "result.json"
    try:
        if not result_path.exists():
            return None
        raw = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    if not isinstance(raw, dict):
        return None
    if raw.get("_cache_kind") == "negative":
        expires_at = _parse_cache_timestamp(raw.get("_cache_expires_at"))
        if expires_at is None or expires_at <= datetime.now(timezone.utc):
            try:
                shutil.rmtree(result_path.parent, ignore_errors=True)
            except OSError:
                pass
            return None
    return raw


def _classify_cache_reason(result: Any) -> str:
    """Derive a semantic 'why' for a cache entry from the task result."""
    status = str(getattr(result, "status", ""))
    if status == "success":
        return "success"
    exit_code = getattr(result, "exit_code", None)
    message = str(getattr(result, "message", "")).lower()
    if exit_code == 124:
        return "negative:timeout"
    if "rate" in message and ("limit" in message or "throttl" in message):
        return "negative:rate_limit"
    if "verify" in message and ("fail" in message or "error" in message):
        return "negative:verify_fail"
    if "judge" in message and ("fail" in message or "reject" in message):
        return "negative:judge_fail"
    return "negative:generic"


def cache_store(
    cache_dir: Path,
    task_hash: str,
    result: Any,
    task: TaskSpec | None = None,
    *,
    negative_cache_ttl_sec: int | None = None,
) -> None:
    """Store a task result in the cache."""
    try:
        status = str(getattr(result, "status", ""))
        if status not in {"success", "failed", "soft_failed"}:
            return
        if not _is_cacheable_result(task, result):
            return

        payload = result.to_dict()
        payload["_cached_at"] = _utc_now_iso()
        payload["_cache_policy_version"] = _CACHE_POLICY_VERSION
        payload["_cache_why"] = _classify_cache_reason(result)

        if status == "success":
            payload["_cache_kind"] = "success"
        else:
            ttl_sec = _effective_negative_cache_ttl_sec(task, negative_cache_ttl_sec)
            if ttl_sec <= 0:
                return
            payload["_cache_kind"] = "negative"
            payload["_negative_cache_ttl_sec"] = ttl_sec
            payload["_cache_expires_at"] = (
                datetime.now(timezone.utc) + timedelta(seconds=ttl_sec)
            ).isoformat()

        entry_dir = _cache_entry_dir(cache_dir, task_hash)
        entry_dir.mkdir(parents=True, exist_ok=True)

        tmp_path = entry_dir / "result.json.tmp"
        result_path = entry_dir / "result.json"
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=True, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        tmp_path.replace(result_path)

        log_path = Path(getattr(result, "log_path", ""))
        if log_path.exists() and log_path.is_file():
            shutil.copy2(log_path, entry_dir / "task.log")
    except Exception:
        return


def cache_stats(cache_dir: Path) -> dict[str, Any]:
    """Return cache statistics: entry count, total size, oldest/newest entries."""
    try:
        if not cache_dir.exists():
            return {
                "entries": 0,
                "total_size_bytes": 0,
                "oldest": None,
                "newest": None,
            }

        entries = 0
        total_size = 0
        oldest_ts: float | None = None
        newest_ts: float | None = None

        for result_path in cache_dir.glob("*/*/result.json"):
            if not result_path.is_file():
                continue
            entries += 1
            stat = result_path.stat()
            total_size += stat.st_size
            mtime = stat.st_mtime
            oldest_ts = mtime if oldest_ts is None else min(oldest_ts, mtime)
            newest_ts = mtime if newest_ts is None else max(newest_ts, mtime)

            log_path = result_path.parent / "task.log"
            if log_path.exists() and log_path.is_file():
                total_size += log_path.stat().st_size

        return {
            "entries": entries,
            "total_size_bytes": total_size,
            "oldest": datetime.fromtimestamp(oldest_ts, timezone.utc).isoformat()
            if oldest_ts is not None
            else None,
            "newest": datetime.fromtimestamp(newest_ts, timezone.utc).isoformat()
            if newest_ts is not None
            else None,
        }
    except Exception:
        return {
            "entries": 0,
            "total_size_bytes": 0,
            "oldest": None,
            "newest": None,
        }


def cache_clear(cache_dir: Path) -> int:
    """Remove all cache entries. Returns number of entries removed."""
    removed = 0
    try:
        if not cache_dir.exists():
            return 0

        for shard_dir in cache_dir.iterdir():
            if not shard_dir.is_dir():
                continue
            for entry_dir in shard_dir.iterdir():
                if not entry_dir.is_dir():
                    continue
                removed += 1
                shutil.rmtree(entry_dir, ignore_errors=True)
            if not any(shard_dir.iterdir()):
                shard_dir.rmdir()
    except Exception:
        return removed
    return removed
