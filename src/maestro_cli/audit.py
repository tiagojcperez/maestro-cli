from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from .models import PlanSpec
from .utils import resolve_path
from .workspace_assertions import evaluate_workspace_assertion, normalize_workspace_assertion

AuditSeverity = Literal["error", "warning", "info"]
AuditCategory = Literal[
    "Agent-Tool Coupling",
    "Data Leakage",
    "Injection",
    "Identity/Provenance",
    "Memory Poisoning",
    "Non-Determinism",
    "Trust Exploitation",
    "Timing/Monitoring",
    "Workflow Architecture",
]

# Maps built-in SEC rules to their risk category (from "Security Considerations
# for Multi-agent Systems" — 193 risks across 9 categories).
_RULE_CATEGORIES: dict[str, AuditCategory] = {
    "SEC001": "Workflow Architecture",   # no cost budget
    "SEC002": "Trust Exploitation",      # yolo without approval
    "SEC003": "Data Leakage",            # secret env vars unmasked
    "SEC004": "Workflow Architecture",   # allow_failure on security tasks
    "SEC005": "Data Leakage",            # hardcoded API key in prompt
    "SEC006": "Workflow Architecture",   # prod path without guard
    "SEC007": "Data Leakage",            # secrets undeclared
    "SEC008": "Agent-Tool Coupling",     # destructive command without approval
    "SEC009": "Agent-Tool Coupling",     # engine yolo without worktree
    "SEC010": "Memory Poisoning",        # deep context chain without budget
    "SEC011": "Timing/Monitoring",       # escalation without cost budget
    "SEC012": "Trust Exploitation",      # fallback yolo propagation
    "SEC013": "Timing/Monitoring",       # watch loop without bounds
    "SEC014": "Data Leakage",            # cloud credentials without secrets config
    "SEC015": "Injection",               # when expr references raw output
    "SEC016": "Injection",               # context_from raw output without guard
    "SEC017": "Memory Poisoning",        # context_from without explicit context_trust
    "SEC018": "Memory Poisoning",        # tainted task without guard/verify
    "SEC019": "Injection",               # untrusted context without honeypot
    "SEC020": "Data Leakage",            # context_from without output_redact on PII-likely patterns
    "SEC021": "Agent-Tool Coupling",     # destructive command without phantom workspace
    "SEC022": "Workflow Architecture",   # contract consumer without verify_command
    "SEC023": "Agent-Tool Coupling",     # untrusted context without allowed_tools
}

_ALL_CATEGORIES: tuple[AuditCategory, ...] = (
    "Agent-Tool Coupling",
    "Data Leakage",
    "Injection",
    "Identity/Provenance",
    "Memory Poisoning",
    "Non-Determinism",
    "Trust Exploitation",
    "Timing/Monitoring",
    "Workflow Architecture",
)

_SECRET_ENV_PATTERN = re.compile(r"KEY|SECRET|TOKEN|PASSWORD|CREDENTIAL", re.IGNORECASE)
_API_KEY_PATTERNS = [
    re.compile(r"\bsk-[A-Za-z0-9]{20,}"),           # OpenAI-style
    re.compile(r"\bghp_[A-Za-z0-9]{36,}"),           # GitHub PAT
    re.compile(r"\bAKIA[A-Z0-9]{16}"),               # AWS access key
    re.compile(r"\bAIza[A-Za-z0-9_\-]{35}"),         # Google API key
    re.compile(r"\bxoxb-[0-9]{11,13}-[0-9]{11,13}-[A-Za-z0-9]{24}"),  # Slack bot token
    re.compile(r"\bEY[A-Za-z0-9_\-]{20,}"),          # JWT-style
]
_PROD_PATH_PATTERN = re.compile(r"\b(prod|production|deploy)\b", re.IGNORECASE)
_YOLO_FLAGS = {"--dangerously-bypass-approvals-and-sandbox", "--yolo", "--allow-all"}
_DESTRUCTIVE_PATTERNS = re.compile(
    r"rm\s+-rf|DROP\s+TABLE|DROP\s+DATABASE|git\s+push\s+(-f|--force)"
    r"|git\s+reset\s+--hard|DELETE\s+FROM|TRUNCATE",
    re.IGNORECASE,
)
_CLOUD_CRED_PATTERN = re.compile(
    r"AWS_SECRET|AWS_ACCESS_KEY|GOOGLE_APPLICATION_CREDENTIALS"
    r"|AZURE_CLIENT_SECRET|DATABASE_URL|PRIVATE_KEY",
    re.IGNORECASE,
)


@dataclass
class AuditFinding:
    severity: AuditSeverity
    rule: str
    message: str
    task_id: str | None = None
    category: str | None = None  # populated for SEC001-SEC023; None for audit pack rules

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "severity": self.severity,
            "rule": self.rule,
            "message": self.message,
        }
        if self.task_id:
            d["task_id"] = self.task_id
        if self.category:
            d["category"] = self.category
        return d


def audit_plan(plan: PlanSpec) -> list[AuditFinding]:
    findings: list[AuditFinding] = []

    # SEC001 — max_cost_usd not set
    if plan.max_cost_usd is None:
        findings.append(AuditFinding(
            severity="error",
            rule="SEC001",
            message="max_cost_usd is not set; plan has unbounded spend risk.",
        ))

    # Collect declared secret env var names for SEC003
    declared_secrets: set[str] = set()
    if isinstance(plan.secrets, list):
        declared_secrets.update(plan.secrets)

    for task in plan.tasks:
        task_tags = set(task.tags or [])

        # SEC002 — yolo usage without approval
        _check_sec002(task, findings)

        # SEC003 — env vars matching secret patterns without secrets declaration
        _check_sec003(task, declared_secrets, findings)

        # SEC004 — allow_failure on security/critical tasks
        if task.allow_failure and task_tags & {"security", "critical"}:
            findings.append(AuditFinding(
                severity="info",
                rule="SEC004",
                message=(
                    f"Task has allow_failure: true but is tagged "
                    f"{sorted(task_tags & {'security', 'critical'})}; failures may go undetected."
                ),
                task_id=task.id,
            ))

        # SEC005 — hardcoded API key patterns in prompt
        _check_sec005(task, findings)

        # SEC006 — writing to prod paths without guard_command
        _check_sec006(task, findings)

    # SEC007 — secrets declared but mask-secrets not enforceable from plan
    if plan.secrets:
        findings.append(AuditFinding(
            severity="warning",
            rule="SEC007",
            message=(
                "secrets: is declared in the plan, but --mask-secrets must be passed at "
                "runtime to enforce redaction; consider documenting this requirement."
            ),
        ))

    # SEC008 — destructive commands without approval
    for task in plan.tasks:
        _check_sec008(task, findings)

    # SEC009 — engine task without worktree isolation (yolo args only)
    if plan.workspace_root:
        for task in plan.tasks:
            _check_sec009(task, findings)

    # SEC010 — deep context chain without budget
    _check_sec010(plan, findings)

    # SEC011 — escalation without cost budget
    if plan.max_cost_usd is None:
        for task in plan.tasks:
            if task.escalation:
                findings.append(AuditFinding(
                    severity="warning",
                    rule="SEC011",
                    message=(
                        "Task has escalation: configured but plan has no max_cost_usd; "
                        "escalation retries with higher-tier models have unbounded cost."
                    ),
                    task_id=task.id,
                ))

    # SEC012 — fallback with yolo propagation
    for task in plan.tasks:
        if task.fallback_engine and _YOLO_FLAGS & set(task.args or []):
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC012",
                message=(
                    f"Task has fallback_engine: '{task.fallback_engine}' and uses "
                    "yolo/bypass flags in args; fallback engine will also run without sandbox."
                ),
                task_id=task.id,
            ))

    # SEC013 — watch loop without bounds
    if plan.watch is not None and plan.watch.max_cost_usd is None:
        findings.append(AuditFinding(
            severity="error",
            rule="SEC013",
            message=(
                "Plan has a watch: block but no max_cost_usd budget; the autonomous "
                "iteration loop has unbounded spend risk. Set watch.max_cost_usd."
            ),
        ))

    # SEC014 — cloud credentials in env without secrets configuration
    if not plan.secrets and not plan.secrets_auto:
        for task in plan.tasks:
            _check_sec014(task, findings)

    # SEC015 — when expression references unbounded upstream output
    for task in plan.tasks:
        _check_sec015(task, findings)

    # SEC016 — context_from pulls raw engine output without guard_command
    task_map_for_ctx: dict[str, Any] = {t.id: t for t in plan.tasks}
    for task in plan.tasks:
        _check_sec016(task, task_map_for_ctx, findings)
        _check_sec017(task, task_map_for_ctx, findings)

    # SEC018: tainted tasks without guard/verify (plan-level check)
    _check_sec018(plan, findings)

    # SEC019: untrusted context without honeypot (per-task)
    _check_sec019(plan, findings)

    # SEC020: context_from passes engine output without redaction (per-task)
    _check_sec020(plan, findings)

    # SEC021: destructive commands without phantom workspace
    _check_sec021(plan, findings)

    # SEC022: contract consumer without verify_command
    _check_sec022(plan, findings)

    # SEC023: untrusted context without allowed_tools
    _check_sec023(plan, findings)

    _apply_audit_packs(plan, findings)

    # Inject category for all built-in SEC rules (audit pack rules stay None)
    for finding in findings:
        if finding.category is None and finding.rule in _RULE_CATEGORIES:
            finding.category = _RULE_CATEGORIES[finding.rule]

    return findings


def _check_sec002(task: Any, findings: list[AuditFinding]) -> None:
    from .models import TaskSpec  # local import to avoid circularity

    if not isinstance(task, TaskSpec):
        return

    yolo_in_args = False
    args: list[str] = list(task.args or [])
    if task.engine and hasattr(task, "args") and args:
        yolo_in_args = bool(_YOLO_FLAGS & set(args))

    if yolo_in_args and not task.requires_approval:
        findings.append(AuditFinding(
            severity="warning",
            rule="SEC002",
            message=(
                "Task uses yolo/bypass flags in args without requires_approval: true; "
                "dangerous operations may execute without confirmation."
            ),
            task_id=task.id,
        ))


def _check_sec003(task: Any, declared_secrets: set[str], findings: list[AuditFinding]) -> None:
    env: dict[str, str] = dict(task.env or {})
    for var_name in env:
        if _SECRET_ENV_PATTERN.search(var_name) and var_name not in declared_secrets:
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC003",
                message=(
                    f"env var '{var_name}' looks like a secret (matches KEY/SECRET/TOKEN/"
                    f"PASSWORD/CREDENTIAL) but is not listed in plan secrets:."
                ),
                task_id=task.id,
            ))


def _check_sec005(task: Any, findings: list[AuditFinding]) -> None:
    prompt_text: str = task.prompt or ""
    if not prompt_text:
        return
    for pattern in _API_KEY_PATTERNS:
        if pattern.search(prompt_text):
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC005",
                message=(
                    "Prompt appears to contain a hardcoded API key or credential "
                    f"(matched pattern: {pattern.pattern!r}); use secrets: instead."
                ),
                task_id=task.id,
            ))
            break  # one finding per task is enough


def _check_sec006(task: Any, findings: list[AuditFinding]) -> None:
    # Check command and verify_command for prod/production/deploy paths
    texts: list[str] = []
    if task.command:
        cmd = task.command
        if isinstance(cmd, list):
            texts.extend(cmd)
        else:
            texts.append(str(cmd))
    if task.verify_command:
        vc = task.verify_command
        if isinstance(vc, list):
            texts.extend(vc)
        else:
            texts.append(str(vc))

    for text in texts:
        if _PROD_PATH_PATTERN.search(text) and not task.guard_command:
            findings.append(AuditFinding(
                severity="info",
                rule="SEC006",
                message=(
                    "Task writes to a path containing 'prod'/'production'/'deploy' "
                    "but has no guard_command to validate output before applying."
                ),
                task_id=task.id,
            ))
            break


def _check_sec008(task: Any, findings: list[AuditFinding]) -> None:
    texts: list[str] = []
    for attr in ("command", "pre_command", "verify_command"):
        val = getattr(task, attr, None)
        if val is None:
            continue
        if isinstance(val, list):
            texts.append(" ".join(str(v) for v in val))
        else:
            texts.append(str(val))

    if not texts:
        return

    combined = " ".join(texts)
    if _DESTRUCTIVE_PATTERNS.search(combined) and not task.requires_approval:
        findings.append(AuditFinding(
            severity="warning",
            rule="SEC008",
            message=(
                "Task contains a destructive command (rm -rf, DROP TABLE/DATABASE, "
                "git push -f/--force, git reset --hard, DELETE FROM, TRUNCATE) "
                "but has no requires_approval: true."
            ),
            task_id=task.id,
        ))


def _check_sec009(task: Any, findings: list[AuditFinding]) -> None:
    if task.engine is None:
        return
    has_yolo = bool(_YOLO_FLAGS & set(task.args or []))
    if has_yolo and not task.worktree:
        findings.append(AuditFinding(
            severity="info",
            rule="SEC009",
            message=(
                "Engine task uses yolo/bypass flags with a workspace_root but no "
                "worktree: true; filesystem changes are not isolated to a git worktree."
            ),
            task_id=task.id,
        ))


def _check_sec010(plan: Any, findings: list[AuditFinding]) -> None:
    """Flag tasks with a context chain deeper than 3 hops and no context_budget_tokens."""
    task_map: dict[str, Any] = {t.id: t for t in plan.tasks}

    def _chain_depth(task_id: str, visited: set[str]) -> int:
        if task_id in visited:
            return 0
        visited.add(task_id)
        task = task_map.get(task_id)
        if task is None or not task.context_from:
            return 0
        return 1 + max(
            (_chain_depth(upstream, visited) for upstream in task.context_from),
            default=0,
        )

    for task in plan.tasks:
        if not task.context_from:
            continue
        depth = _chain_depth(task.id, set())
        if depth > 3 and task.context_budget_tokens is None:
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC010",
                message=(
                    f"Task has a context chain {depth} hops deep but no "
                    "context_budget_tokens; deep context chains can silently exhaust "
                    "the model's context window."
                ),
                task_id=task.id,
            ))


def _check_sec015(task: Any, findings: list[AuditFinding]) -> None:
    """SEC015: when expression references unbounded upstream output."""
    if not task.when:
        return
    dangerous_fields = ["stdout_tail", "log"]
    for f in dangerous_fields:
        if f in task.when:
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC015",
                message=(
                    f"when expression references '{f}' which may contain "
                    f"adversarial content — consider adding guard_command to "
                    f"the upstream task or using structured fields instead"
                ),
                task_id=task.id,
            ))


def _check_sec016(task: Any, task_map: dict[str, Any], findings: list[AuditFinding]) -> None:
    """SEC016: context_from passes raw upstream output without guard_command on the upstream.

    Triggers only on `context_mode: raw` (direct passthrough). LLM-mediated modes
    (`summarized`, `map_reduce`, `recursive`, `council`, `knowledge_graph`) and
    heuristic-extraction modes (`selective`, `layered`, `structural`) provide at
    least partial injection resistance and are exempt — adding `guard_command`
    on top of those is redundant and was the #1 false-positive friction reported
    in an internal 2026-04-26 post-mortem.
    """
    if not task.context_from:
        return
    context_mode: str = task.context_mode or "raw"
    if context_mode != "raw":
        return
    upstreams = task.context_from if task.context_from != ["*"] else []
    for upstream_id in upstreams:
        upstream = task_map.get(upstream_id)
        if upstream is None:
            continue
        if upstream.engine and not upstream.guard_command:
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC016",
                message=(
                    f"context_from includes engine task '{upstream_id}' without a "
                    f"guard_command; raw LLM output injected into prompts may carry "
                    f"adversarial instructions — add guard_command to '{upstream_id}' "
                    f"or switch to context_mode: summarized / map_reduce / recursive / "
                    f"layered for at least partial injection resistance"
                ),
                task_id=task.id,
            ))


# -- Untrusted context: Patterns suggesting external / user-controlled data --
_EXTERNAL_DATA_INDICATORS = re.compile(
    r"user[\s_-]*(input|data|request|submission)|"
    r"external[\s_-]*(api|service|source|data)|"
    r"web[\s_-]*(scrape|crawl|fetch)|"
    r"upload|download|ingest|parse[\s_-]*input|"
    r"untrusted|third[\s_-]*party",
    re.IGNORECASE,
)


def _check_sec017(task: Any, task_map: dict[str, Any], findings: list[AuditFinding]) -> None:
    """SEC017: context_from references tasks with external data indicators but no context_trust."""
    if not task.context_from:
        return
    upstreams = task.context_from if task.context_from != ["*"] else []
    for upstream_id in upstreams:
        upstream = task_map.get(upstream_id)
        if upstream is None:
            continue
        if upstream.context_trust is not None:
            continue  # explicitly set — no warning needed
        prompt_text = upstream.prompt or ""
        command_text = ""
        if isinstance(upstream.command, str):
            command_text = upstream.command
        elif isinstance(upstream.command, list):
            command_text = " ".join(upstream.command)
        combined = f"{prompt_text} {command_text} {upstream.description or ''}"
        if _EXTERNAL_DATA_INDICATORS.search(combined):
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC017",
                message=(
                    f"context_from includes task '{upstream_id}' which appears to "
                    f"handle external data but has no context_trust set — consider "
                    f"adding context_trust: untrusted to '{upstream_id}'"
                ),
                task_id=task.id,
            ))


def _check_sec018(plan: PlanSpec, findings: list[AuditFinding]) -> None:
    """SEC018: tainted task has no guard_command or verify_command."""
    task_map: dict[str, Any] = {t.id: t for t in plan.tasks}
    tainted: set[str] = set()

    # Seed: explicitly untrusted tasks
    for task in plan.tasks:
        if task.context_trust == "untrusted":
            tainted.add(task.id)

    # Propagate transitively
    changed = True
    while changed:
        changed = False
        for task in plan.tasks:
            if task.id in tainted:
                continue
            if not task.context_from:
                continue
            ctx_ids: list[str] = []
            for entry in task.context_from:
                if entry == "*":
                    ctx_ids.extend(
                        dep for dep in task.depends_on if dep in task_map
                    )
                else:
                    ctx_ids.append(entry)
            if any(uid in tainted for uid in ctx_ids):
                if task.guard_command is None and task.verify_command is None:
                    tainted.add(task.id)
                    changed = True

    # Report: tasks that inherited taint (not the explicit source)
    for task in plan.tasks:
        if task.id in tainted and task.context_trust != "untrusted":
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC018",
                message=(
                    f"Task inherits tainted context from upstream but has no "
                    f"guard_command or verify_command to sanitize — add "
                    f"validation or set context_trust explicitly"
                ),
                task_id=task.id,
            ))


def _check_sec019(plan: Any, findings: list[AuditFinding]) -> None:
    """SEC019: untrusted context_from without honeypot decoys."""
    for task in plan.tasks:
        if task.context_trust == "untrusted" and not task.honeypot:
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC019",
                message=(
                    "Task has context_trust: untrusted but honeypot: true is not "
                    "set — honeypot decoys help detect prompt injection by planting "
                    "trap values that only injected instructions would access"
                ),
                task_id=task.id,
            ))


_PII_KEYWORDS = re.compile(
    r"(email|password|ssn|social.?security|credit.?card|phone|address|"
    r"api.?key|secret|token|bearer|authorization|credentials)",
    re.IGNORECASE,
)


def _check_sec020(plan: Any, findings: list[AuditFinding]) -> None:
    """SEC020: engine tasks producing PII-like output consumed downstream without redaction."""
    task_map = {t.id: t for t in plan.tasks}
    for task in plan.tasks:
        if not task.context_from:
            continue
        for upstream_id in task.context_from:
            if upstream_id == "*":
                continue
            upstream = task_map.get(upstream_id)
            if upstream is None:
                continue
            if upstream.engine is None:
                continue
            # Check if upstream prompt mentions PII-like fields
            prompt_text = upstream.prompt or ""
            if not _PII_KEYWORDS.search(prompt_text):
                continue
            if upstream.output_redact:
                continue
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC020",
                message=(
                    f"Upstream task '{upstream_id}' prompt references PII-like "
                    f"fields but has no output_redact patterns — downstream task "
                    f"'{task.id}' may receive sensitive data via context_from"
                ),
                task_id=task.id,
            ))


_DESTRUCTIVE_CMD_PATTERN = re.compile(
    r"\b(rm\s+-rf|rmdir|del\s+/[sS]|DROP\s+TABLE|TRUNCATE|DELETE\s+FROM|"
    r"git\s+reset\s+--hard|git\s+clean\s+-[fd])\b",
    re.IGNORECASE,
)


def _check_sec021(plan: Any, findings: list[AuditFinding]) -> None:
    """SEC021: destructive commands without phantom workspace or approval."""
    for task in plan.tasks:
        cmd_text = ""
        if isinstance(task.command, str):
            cmd_text = task.command
        elif isinstance(task.command, list):
            cmd_text = " ".join(str(c) for c in task.command)
        prompt_text = task.prompt or ""
        combined = cmd_text + " " + prompt_text
        if not _DESTRUCTIVE_CMD_PATTERN.search(combined):
            continue
        if task.phantom_workspace or task.requires_approval:
            continue
        findings.append(AuditFinding(
            severity="warning",
            rule="SEC021",
            message=(
                f"Task uses destructive command patterns but has neither "
                f"phantom_workspace: true nor requires_approval: true"
            ),
            task_id=task.id,
        ))


def _check_sec022(plan: Any, findings: list[AuditFinding]) -> None:
    """SEC022: task consumes contracts but has no verify_command to validate them."""
    for task in plan.tasks:
        if not task.consumes_contracts:
            continue
        if task.verify_command or task.guard_command:
            continue
        findings.append(AuditFinding(
            severity="warning",
            rule="SEC022",
            message=(
                f"Task consumes contracts from {task.consumes_contracts} but has "
                f"no verify_command or guard_command to validate contract integrity"
            ),
            task_id=task.id,
        ))


def _check_sec023(plan: PlanSpec, findings: list[AuditFinding]) -> None:
    """SEC023: engine task processes untrusted context without allowed_tools."""
    for task in plan.tasks:
        if (
            task.engine is not None
            and task.context_trust == "untrusted"
            and task.allowed_tools is None
        ):
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC023",
                message=(
                    f"Task '{task.id}' processes untrusted context without allowed_tools; "
                    "consider restricting tool access for prompt injection containment"
                ),
                task_id=task.id,
                category=_RULE_CATEGORIES.get("SEC023"),
            ))


def _check_sec014(task: Any, findings: list[AuditFinding]) -> None:
    env: dict[str, str] = dict(task.env or {})
    for var_name in env:
        if _CLOUD_CRED_PATTERN.search(var_name):
            findings.append(AuditFinding(
                severity="warning",
                rule="SEC014",
                message=(
                    f"env var '{var_name}' looks like a cloud credential but plan has "
                    "no secrets: configuration; value may appear in logs unredacted."
                ),
                task_id=task.id,
            ))
            break  # one finding per task is enough


def _apply_audit_packs(plan: PlanSpec, findings: list[AuditFinding]) -> None:
    base_dir = resolve_path(plan.source_dir, plan.workspace_root) or plan.source_dir
    for pack_ref in plan.audit_packs:
        pack_path = resolve_path(plan.source_dir, pack_ref)
        if pack_path is None:
            findings.append(AuditFinding(
                severity="error",
                rule="PACK001",
                message=f"audit pack path could not be resolved: {pack_ref}",
            ))
            continue
        _apply_single_audit_pack(pack_path, base_dir, findings)


def _apply_single_audit_pack(
    pack_path: Path,
    base_dir: Path,
    findings: list[AuditFinding],
) -> None:
    try:
        raw = yaml.safe_load(pack_path.read_text(encoding="utf-8"))
    except OSError as exc:
        findings.append(AuditFinding(
            severity="error",
            rule="PACK001",
            message=f"Failed to read audit pack {pack_path}: {exc}",
        ))
        return
    except yaml.YAMLError as exc:
        findings.append(AuditFinding(
            severity="error",
            rule="PACK002",
            message=f"Invalid YAML in audit pack {pack_path}: {exc}",
        ))
        return

    rules_raw: Any = raw
    if isinstance(raw, dict):
        rules_raw = raw.get("rules")
    if not isinstance(rules_raw, list):
        findings.append(AuditFinding(
            severity="error",
            rule="PACK002",
            message=f"Audit pack {pack_path} must contain a top-level 'rules' list",
        ))
        return

    for idx, item in enumerate(rules_raw):
        field_name = f"{pack_path.name}.rules[{idx}]"
        try:
            rule = normalize_workspace_assertion(item, field_name)
        except ValueError as exc:
            findings.append(AuditFinding(
                severity="error",
                rule="PACK003",
                message=str(exc),
            ))
            continue

        passed, reason = evaluate_workspace_assertion(rule, base_dir)
        if passed:
            continue

        findings.append(AuditFinding(
            severity=cast(AuditSeverity, str(rule.get("severity", "error"))),
            rule=str(rule.get("rule", f"PACK{idx + 1:03d}")),
            message=str(rule.get("message", reason)),
            task_id=str(rule["task_id"]) if rule.get("task_id") is not None else None,
        ))


_FIXABLE_RULES: set[str] = {"SEC001", "SEC003", "SEC014"}


def fix_plan(
    plan_path: Path,
    findings: list[AuditFinding],
    *,
    dry_run: bool = False,
) -> list[str]:
    """Apply safe auto-fixes for audit findings. Returns descriptions."""
    fixable = [f for f in findings if f.rule in _FIXABLE_RULES]
    if not fixable:
        return []

    data: dict[str, Any] = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    applied: list[str] = []

    rules_present = {f.rule for f in fixable}

    if "SEC001" in rules_present and data.get("max_cost_usd") is None:
        data["max_cost_usd"] = 10.0
        applied.append("SEC001: Added max_cost_usd: 10.0")

    if rules_present & {"SEC003", "SEC014"} and data.get("secrets") is None:
        data["secrets"] = "auto"
        label = "/".join(sorted(rules_present & {"SEC003", "SEC014"}))
        applied.append(f"{label}: Added secrets: auto")

    if not applied:
        return []

    if not dry_run:
        shutil.copy2(plan_path, plan_path.with_suffix(".yaml.bak"))
        plan_path.write_text(
            yaml.safe_dump(data, default_flow_style=False, sort_keys=False),
            encoding="utf-8",
        )

    return applied


def format_audit(findings: list[AuditFinding]) -> str:
    if not findings:
        return "[audit] No findings — plan looks clean."

    _prefix = {
        "error": "[ERROR]",
        "warning": "[WARN] ",
        "info": "[INFO] ",
    }
    lines: list[str] = []
    for f in findings:
        prefix = _prefix.get(f.severity, "[?]   ")
        location = f" (task: {f.task_id})" if f.task_id else ""
        lines.append(f"{prefix} {f.rule}{location}: {f.message}")

    counts: dict[str, int] = {"error": 0, "warning": 0, "info": 0}
    for f in findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1

    summary_parts = []
    if counts["error"]:
        summary_parts.append(f"{counts['error']} error(s)")
    if counts["warning"]:
        summary_parts.append(f"{counts['warning']} warning(s)")
    if counts["info"]:
        summary_parts.append(f"{counts['info']} info")
    lines.append(f"\n[audit] {', '.join(summary_parts)} found.")
    return "\n".join(lines)


def format_audit_json(findings: list[AuditFinding]) -> str:
    return json.dumps([f.to_dict() for f in findings], indent=2)


@dataclass
class CategoryCoverage:
    category: str
    rules: list[str]      # all built-in rules in this category
    triggered: list[str]  # rules that fired in this audit run
    coverage_pct: float   # triggered / total rules * 100


def compute_audit_coverage(findings: list[AuditFinding]) -> list[CategoryCoverage]:
    """Aggregate findings by category and compute per-category coverage."""
    # Build category → all rules mapping
    category_rules: dict[AuditCategory, list[str]] = {cat: [] for cat in _ALL_CATEGORIES}
    for rule, cat in _RULE_CATEGORIES.items():
        category_rules[cat].append(rule)
    for cat_key in category_rules:
        category_rules[cat_key] = sorted(category_rules[cat_key])

    # Collect triggered SEC rules from findings
    triggered_by_cat: dict[AuditCategory, list[str]] = {cat: [] for cat in _ALL_CATEGORIES}
    seen: set[str] = set()
    for f in findings:
        if f.category and f.rule not in seen and f.rule in _RULE_CATEGORIES:
            triggered_by_cat[cast(AuditCategory, f.category)].append(f.rule)
            seen.add(f.rule)

    result: list[CategoryCoverage] = []
    for cat in _ALL_CATEGORIES:
        rules = category_rules[cat]
        triggered = sorted(triggered_by_cat[cat])
        pct = (len(triggered) / len(rules) * 100) if rules else 0.0
        result.append(CategoryCoverage(
            category=cat,
            rules=rules,
            triggered=triggered,
            coverage_pct=round(pct, 1),
        ))
    return result


def format_audit_coverage(findings: list[AuditFinding]) -> str:
    """Human-readable per-category coverage table."""
    coverage = compute_audit_coverage(findings)
    lines: list[str] = ["[audit] Security category coverage (SEC001-SEC023):", ""]

    col_w = max(len(c.category) for c in coverage) + 2
    header = f"  {'Category':<{col_w}}  Rules   Triggered  Coverage"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for c in coverage:
        bar_filled = int(c.coverage_pct / 10)
        bar = "█" * bar_filled + "░" * (10 - bar_filled)
        triggered_str = ", ".join(c.triggered) if c.triggered else "—"
        lines.append(
            f"  {c.category:<{col_w}}  {len(c.rules):<7} {len(c.triggered):<10} "
            f"{c.coverage_pct:5.1f}%  {bar}"
        )
        if c.triggered:
            lines.append(f"  {'':>{col_w}}         triggered: {triggered_str}")

    total_rules = sum(len(c.rules) for c in coverage)
    total_triggered = sum(len(c.triggered) for c in coverage)
    total_pct = round(total_triggered / total_rules * 100, 1) if total_rules else 0.0
    lines.append("")
    lines.append(
        f"  Overall: {total_triggered}/{total_rules} rules triggered ({total_pct}% coverage)"
    )
    lines.append("")
    lines.append(
        "  Note: coverage shows which risk categories were detected in this plan.\n"
        "  Zero-coverage categories (Identity/Provenance, Non-Determinism) have no\n"
        "  SEC rules yet — see docs/ROADMAP.md for planned additions."
    )
    return "\n".join(lines)


def format_audit_coverage_json(findings: list[AuditFinding]) -> str:
    """JSON-encoded coverage report."""
    coverage = compute_audit_coverage(findings)
    total_rules = sum(len(c.rules) for c in coverage)
    total_triggered = sum(len(c.triggered) for c in coverage)
    return json.dumps(
        {
            "categories": [
                {
                    "category": c.category,
                    "rules": c.rules,
                    "triggered": c.triggered,
                    "coverage_pct": c.coverage_pct,
                }
                for c in coverage
            ],
            "total_rules": total_rules,
            "total_triggered": total_triggered,
            "overall_pct": round(total_triggered / total_rules * 100, 1) if total_rules else 0.0,
        },
        indent=2,
    )
