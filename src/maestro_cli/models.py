from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from pathlib import Path
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .workspace_index import WorkspaceIndex as _WsIndex

EngineName = Literal["codex", "claude", "gemini", "copilot", "qwen", "ollama", "llama"]
ExecutionProfile = Literal["plan", "safe", "yolo"]
TaskStatus = Literal["success", "failed", "soft_failed", "skipped", "dry_run"]
EditPolicy = Literal["default", "efficient", "strict"]
ContextMode = Literal["raw", "summarized", "map_reduce", "recursive", "layered", "selective", "structural", "council", "knowledge_graph", "codebase_map", "scip"]
Verbosity = Literal["quiet", "normal", "verbose"]
OutputMode = Literal["text", "jsonl"]
WorktreeMergeStatus = Literal["merged", "conflict", "empty", "error"]
MergeReviewVerdict = Literal["safe", "resolvable", "conflict", "error"]
RetryStrategy = Literal["constant", "linear", "exponential"]
CircuitBreakerAction = Literal["pause", "fail"]

# v0.8.0 -- Resilience & Recovery types
FailureCategory = Literal[
    "timeout", "compilation_error", "test_failure",
    "validation_error", "permission_error", "runtime_error",
    "context_exceeded", "rate_limited", "unknown",
    "miscommunication", "role_confusion", "verification_gap",
    "cascading_failure", "deadlock", "dependency_missing",
    "output_format_error",
]
JudgeOnFail = Literal["fail", "warn", "retry"]
JudgeVerdict = Literal["pass", "fail", "warn", "error"]
JudgeMethod = Literal["direct", "g_eval", "debate", "reflection"]
ScoreAggregation = Literal["mean", "min", "weighted_mean"]
VerifyStatus = Literal["valid", "tampered", "incomplete"]
PolicyAction = Literal["block", "warn", "audit"]
RoutingStrategy = Literal["cost_optimized", "quality_first", "balanced"]
QuorumStrategy = Literal["majority", "unanimous", "any"]
ContextTrust = Literal["trusted", "untrusted"]
ContextCompaction = Literal["none", "standard", "progressive"]
TrajectoryGuardAction = Literal["warn", "abort", "escalate"]
PopulationStrategy = Literal["best", "first_passing", "majority"]
VariantType = Literal["draft", "debug", "improve"]
MctsSelectionPolicy = Literal["debug_prob", "ucb1"]
ReplanPopulationStrategy = Literal["best", "tournament"]
BlameCategory = Literal[
    "root_cause", "dependency_cascade", "context_corruption",
    "timeout_propagation", "budget_exhaustion", "unknown",
]
SignalType = Literal[
    "progress", "metric", "log", "artifact",
    "timeout_extend", "budget_query", "checkpoint",
]

# v2.0 -- SDK public callback type
EventCallback = Callable[[str, dict[str, object]], None]

# v0.7.0 -- Recursive Context types
RecursiveContextStage = Literal["index", "extract", "brief"]
KnowledgeKind = Literal[
    "failure_pattern",    # recurring failure category across runs
    "timeout_hint",       # task tends to timeout
    "success_pattern",    # task reliably succeeds with specific config
    "cost_pattern",       # cost per task on success (helps model routing)
    "duration_pattern",   # duration tracking (helps timeout estimation)
    "retry_pattern",      # retry behaviour (which tasks need retries)
    "model_pattern",      # model effectiveness (auto-routed model outcomes)
    "policy_rule",        # meta-policy reflexion: persistent failure rule
]

SuggestionCategory = Literal[
    "downgrade_model", "upgrade_model",
    "add_judge", "remove_judge",
    "add_retry", "reduce_retry",
    "adjust_effort", "add_review_task",
    "add_checkpoint", "reduce_context_budget",
    "fix_failure_pattern", "tune_timeout",
]

# Valid reasoning_effort values per engine
CODEX_REASONING_EFFORTS: set[str] = {"none", "minimal", "low", "medium", "high", "xhigh"}
# Claude effort levels expanded 2026-04-27: `xhigh` for Opus 4.7/4.8, `max` for
# Opus 4.6/4.7/4.8 + Sonnet 4.6. Per-model defaults: Opus 4.8 = high, Opus 4.7 =
# xhigh, Sonnet 4.6 = high; Haiku ignores effort. `max` is reserved for genuinely
# frontier problems.
CLAUDE_REASONING_EFFORTS: set[str] = {"low", "medium", "high", "xhigh", "max"}
CLAUDE_MODELS: set[str] = {"haiku", "sonnet", "opus", "opusplan"}
GEMINI_MODELS: set[str] = {
    "flash", "flash-lite", "pro", "auto",
    "flash-3", "pro-3", "pro-3.1",
}
CODEX_MODEL_ALIASES: dict[str, str] = {
    # GPT-5.5 (April 2026) and 5.4 onwards drop the historical `-codex` suffix:
    # the standard model now powers Codex CLI workloads. The 5.0-5.3 aliases
    # still target the `-codex` variants, which are what those generations
    # shipped with.
    "5.5": "gpt-5.5",
    "5.4": "gpt-5.4-codex",
    "5.4-mini": "gpt-5.4-mini",
    "5.3": "gpt-5.3-codex",
    "5.2": "gpt-5.2-codex",
    "5.1": "gpt-5.1-codex",
    "5": "gpt-5-codex",
    "5-mini": "gpt-5-codex-mini",
}
GEMINI_MODEL_ALIASES: dict[str, str] = {
    "flash": "gemini-2.5-flash",
    "flash-lite": "gemini-2.5-flash-lite",
    "pro": "gemini-2.5-pro",
    "flash-3": "gemini-3-flash-preview",
    "pro-3": "gemini-3.1-pro-preview",
    "pro-3.1": "gemini-3.1-pro-preview",
}
COPILOT_MODEL_ALIASES: dict[str, str] = {
    # Claude
    "opus": "claude-opus-4.6",
    "opus-fast": "claude-opus-4.6-fast",
    "opus-4.5": "claude-opus-4.5",
    "sonnet": "claude-sonnet-4.6",
    "sonnet-4.5": "claude-sonnet-4.5",
    "sonnet-4": "claude-sonnet-4",
    "haiku": "claude-haiku-4.5",
    # GPT
    "gpt-5.4-codex": "gpt-5.4-codex",
    "gpt-5.3-codex": "gpt-5.3-codex",
    "gpt-5.2-codex": "gpt-5.2-codex",
    "gpt-5.1-codex": "gpt-5.1-codex",
    "gpt-5.1-codex-mini": "gpt-5.1-codex-mini",
    "gpt-5.1-codex-max": "gpt-5.1-codex-max",
    "gpt-5.2": "gpt-5.2",
    "gpt-5.1": "gpt-5.1",
    "gpt-5-mini": "gpt-5-mini",
    "gpt-4.1": "gpt-4.1",
    # Gemini
    "gemini-pro": "gemini-2.5-pro",
    "gemini-3-pro": "gemini-3-pro-preview",
}
QWEN_MODEL_ALIASES: dict[str, str] = {
    "coder": "qwen-coder-plus",
    "coder-turbo": "qwen-coder-turbo",
    "max": "qwen-max",
    "plus": "qwen-plus",
    "qwq": "qwq-plus",
}
OLLAMA_MODEL_ALIASES: dict[str, str] = {
    "llama3": "llama3",
    "llama3.1": "llama3.1",
    "llama3.2": "llama3.2",
    "codellama": "codellama",
    "mistral": "mistral",
    "mixtral": "mixtral",
    "phi3": "phi3",
    "qwen2": "qwen2",
    "qwen2.5-coder": "qwen2.5-coder",
    "deepseek-coder": "deepseek-coder",
    "deepseek-coder-v2": "deepseek-coder-v2",
    "starcoder2": "starcoder2",
    # 2026 refresh (all available via `ollama pull`)
    "llama4": "llama4",
    "qwen3": "qwen3",
    "qwen3-coder": "qwen3-coder",
    "deepseek-r1": "deepseek-r1",
    "deepseek-v3": "deepseek-v3",
    "gemma3": "gemma3",
    "phi4": "phi4",
    "gpt-oss": "gpt-oss",
}
LLAMA_MODEL_ALIASES: dict[str, str] = {
    "llama3": "llama-3-8b",
    "llama3.1": "llama-3.1-8b",
    "llama3.2": "llama-3.2-3b",
    "codellama": "codellama-13b",
    "phi3": "phi-3-mini",
    "mistral": "mistral-7b",
    "qwen2.5-coder": "qwen2.5-coder-7b",
    # Llama 4 (MoE) -- needs aggressive quantization to run locally
    "llama4-scout": "llama-4-scout-17b-16e",
    "llama4-maverick": "llama-4-maverick-17b-128e",
}
COPILOT_MODELS: set[str] = set(COPILOT_MODEL_ALIASES)
QWEN_MODELS: dict[str, str] = dict(QWEN_MODEL_ALIASES)
OLLAMA_MODELS: dict[str, str] = dict(OLLAMA_MODEL_ALIASES)
LLAMA_MODELS: dict[str, str] = dict(LLAMA_MODEL_ALIASES)
EDIT_POLICIES: set[str] = {"default", "efficient", "strict"}
CONTEXT_MODES: set[str] = {"raw", "summarized", "map_reduce", "recursive", "layered", "selective", "structural", "council", "knowledge_graph", "codebase_map", "scip"}
MAX_RETRIES_LIMIT = 3
JUDGE_ON_FAIL_VALUES: set[str] = {"fail", "warn", "retry"}
ASSERTION_TYPES: set[str] = {"contains", "regex", "is-json", "json-schema", "llm-rubric", "cost_under", "duration_under", "rubric"}
SIGNAL_TYPES: frozenset[str] = frozenset({
    "progress", "metric", "log", "artifact",
    "timeout_extend", "budget_query", "checkpoint",
    "compress",
})
WORKSPACE_ASSERTION_TYPES: set[str] = {
    "file_contains",
    "file_contains_count",
    "file_not_contains",
    "file_regex",
    "file_regex_absent",
    "glob_exists",
    "json_path_exists",
    "composer_package_present",
    "npm_package_present",
}
CONTRACT_TYPES: set[str] = {
    "sql-schema",
    "dependency-manifest",
    "conventions-doc",
    "file-inventory",
    "api-schema",
    "test-manifest",
}
JUDGE_METHODS: set[str] = {"direct", "g_eval", "debate", "reflection"}
SCORE_AGGREGATIONS: set[str] = {"mean", "min", "weighted_mean"}
QUORUM_STRATEGIES: set[str] = {"majority", "unanimous", "any"}
JUDGE_DIVERSITY_TIERS: list[str] = ["haiku", "sonnet", "opus"]
CONTEXT_TRUST_VALUES: set[str] = {"trusted", "untrusted"}
CONTEXT_COMPACTION_VALUES: set[str] = {"none", "standard", "progressive"}
POPULATION_STRATEGIES: set[str] = {"best", "first_passing", "majority"}

# Named criteria presets — pre-defined criteria sets with calibrated thresholds
JUDGE_PRESETS: dict[str, dict[str, Any]] = {
    "code_quality": {
        "criteria": [
            {"type": "rubric", "name": "Correctness", "levels": [
                {"score": 1, "description": "Broken, does not compile or run"},
                {"score": 2, "description": "Runs but produces wrong results"},
                {"score": 3, "description": "Mostly correct with minor issues"},
                {"score": 4, "description": "Correct, handles common cases"},
                {"score": 5, "description": "Correct, handles edge cases, well-tested"},
            ], "min_score": 3, "weight": 2.0},
            {"type": "rubric", "name": "Code Style", "levels": [
                {"score": 1, "description": "Unreadable, no conventions followed"},
                {"score": 3, "description": "Readable, follows basic conventions"},
                {"score": 5, "description": "Clean, idiomatic, well-documented"},
            ], "min_score": 3, "weight": 1.0},
            {"type": "rubric", "name": "Error Handling", "levels": [
                {"score": 1, "description": "No error handling, crashes on bad input"},
                {"score": 3, "description": "Basic error handling present"},
                {"score": 5, "description": "Comprehensive error handling, graceful degradation"},
            ], "min_score": 2, "weight": 1.5},
        ],
        "pass_threshold": 0.6,
        "aggregation": "weighted_mean",
    },
    "security_audit": {
        "criteria": [
            {"type": "rubric", "name": "Input Validation", "levels": [
                {"score": 1, "description": "No input validation"},
                {"score": 3, "description": "Basic validation, some gaps"},
                {"score": 5, "description": "All inputs validated and sanitized"},
            ], "min_score": 4, "weight": 2.0},
            {"type": "rubric", "name": "Authentication", "levels": [
                {"score": 1, "description": "No auth or hardcoded credentials"},
                {"score": 3, "description": "Auth present but has weaknesses"},
                {"score": 5, "description": "Secure auth with proper session management"},
            ], "min_score": 4, "weight": 2.0},
            {"type": "rubric", "name": "Data Protection", "levels": [
                {"score": 1, "description": "Sensitive data exposed or logged"},
                {"score": 3, "description": "Basic protection, some data exposure risks"},
                {"score": 5, "description": "All sensitive data encrypted, no leaks"},
            ], "min_score": 3, "weight": 1.5},
        ],
        "pass_threshold": 0.7,
        "aggregation": "min",
    },
    "ai_slop_detection": {
        "criteria": [
            {"type": "rubric", "name": "No Filler Preamble", "levels": [
                {"score": 1, "description": "Starts with filler ('Sure!', 'Great question!', 'I'd be happy to...')"},
                {"score": 3, "description": "Minor preamble before substance"},
                {"score": 5, "description": "Gets straight to the point, no unnecessary opening"},
            ], "min_score": 3, "weight": 1.5},
            {"type": "rubric", "name": "No Hedging", "levels": [
                {"score": 1, "description": "Excessive hedging ('It's worth noting', 'It should be mentioned', 'arguably')"},
                {"score": 3, "description": "Occasional hedging but mostly direct"},
                {"score": 5, "description": "Direct, confident statements with appropriate caveats only"},
            ], "min_score": 3, "weight": 1.0},
            {"type": "rubric", "name": "No Repetition", "levels": [
                {"score": 1, "description": "Same idea restated 3+ times in different words"},
                {"score": 3, "description": "Minor repetition, mostly distinct points"},
                {"score": 5, "description": "Each sentence adds new information"},
            ], "min_score": 3, "weight": 1.5},
            {"type": "rubric", "name": "Specific Over Generic", "levels": [
                {"score": 1, "description": "Vague platitudes ('best practices', 'robust solution', 'comprehensive approach')"},
                {"score": 3, "description": "Mix of specific and generic statements"},
                {"score": 5, "description": "Concrete, specific, actionable content throughout"},
            ], "min_score": 3, "weight": 2.0},
            {"type": "rubric", "name": "No Trailing Summary", "levels": [
                {"score": 1, "description": "Ends with unnecessary recap of what was just said"},
                {"score": 3, "description": "Brief closing that adds minor value"},
                {"score": 5, "description": "Ends when the content is delivered, no padding"},
            ], "min_score": 3, "weight": 1.0},
        ],
        "pass_threshold": 0.6,
        "aggregation": "weighted_mean",
    },
    # -----------------------------------------------------------------------
    # CWE Security Profiles — targeted vulnerability category presets
    # -----------------------------------------------------------------------
    "cwe_injection": {
        "criteria": [
            {"type": "rubric", "name": "SQL Injection (CWE-89)", "levels": [
                {"score": 1, "description": "Raw string concatenation in SQL queries"},
                {"score": 3, "description": "Parameterized queries with some gaps"},
                {"score": 5, "description": "All queries parameterized, ORM or prepared statements throughout"},
            ], "min_score": 4, "weight": 2.0},
            {"type": "rubric", "name": "Command Injection (CWE-78)", "levels": [
                {"score": 1, "description": "User input passed to shell/exec without sanitization"},
                {"score": 3, "description": "Some input sanitization, shell=True still used"},
                {"score": 5, "description": "No shell=True, all arguments passed as lists, inputs validated"},
            ], "min_score": 4, "weight": 2.0},
            {"type": "rubric", "name": "XSS (CWE-79)", "levels": [
                {"score": 1, "description": "User input rendered without escaping"},
                {"score": 3, "description": "Template engine with auto-escape, some raw HTML"},
                {"score": 5, "description": "All output escaped, CSP headers, no innerHTML with user data"},
            ], "min_score": 4, "weight": 1.5},
            {"type": "rubric", "name": "Path Traversal (CWE-22)", "levels": [
                {"score": 1, "description": "User input used directly in file paths"},
                {"score": 3, "description": "Basic checks but bypasses possible (../, encoding)"},
                {"score": 5, "description": "Path canonicalized and confined to allowed directory"},
            ], "min_score": 4, "weight": 1.5},
        ],
        "pass_threshold": 0.8,
        "aggregation": "min",
    },
    "cwe_auth": {
        "criteria": [
            {"type": "rubric", "name": "Broken Authentication (CWE-287)", "levels": [
                {"score": 1, "description": "No authentication or hardcoded credentials"},
                {"score": 3, "description": "Authentication present but weak (plain text, no MFA)"},
                {"score": 5, "description": "Strong authentication, secure session management, MFA support"},
            ], "min_score": 4, "weight": 2.0},
            {"type": "rubric", "name": "Broken Access Control (CWE-284)", "levels": [
                {"score": 1, "description": "No authorization checks, all users can access everything"},
                {"score": 3, "description": "Role-based access with some enforcement gaps"},
                {"score": 5, "description": "Consistent authorization, principle of least privilege, tested"},
            ], "min_score": 4, "weight": 2.0},
            {"type": "rubric", "name": "Credential Storage (CWE-256)", "levels": [
                {"score": 1, "description": "Plaintext passwords or reversible encryption"},
                {"score": 3, "description": "Hashed but weak algorithm (MD5, SHA1)"},
                {"score": 5, "description": "bcrypt/scrypt/argon2 with proper salt and work factor"},
            ], "min_score": 4, "weight": 1.5},
            {"type": "rubric", "name": "Session Management (CWE-384)", "levels": [
                {"score": 1, "description": "Predictable session IDs or no session expiry"},
                {"score": 3, "description": "Random session IDs but missing rotation or expiry"},
                {"score": 5, "description": "Cryptographic session IDs, rotation on auth, proper expiry"},
            ], "min_score": 3, "weight": 1.5},
        ],
        "pass_threshold": 0.8,
        "aggregation": "min",
    },
    "cwe_data_exposure": {
        "criteria": [
            {"type": "rubric", "name": "Sensitive Data Exposure (CWE-200)", "levels": [
                {"score": 1, "description": "Secrets, PII, or tokens in logs/responses/errors"},
                {"score": 3, "description": "Most sensitive data protected, some leak vectors remain"},
                {"score": 5, "description": "No sensitive data in logs/errors, proper redaction everywhere"},
            ], "min_score": 4, "weight": 2.0},
            {"type": "rubric", "name": "Cryptographic Failures (CWE-327)", "levels": [
                {"score": 1, "description": "No encryption or broken algorithms (DES, RC4, MD5 for security)"},
                {"score": 3, "description": "Modern algorithms but weak configuration (short keys, ECB)"},
                {"score": 5, "description": "AES-256/ChaCha20, proper IV/nonce, TLS 1.2+ enforced"},
            ], "min_score": 3, "weight": 1.5},
            {"type": "rubric", "name": "Error Information Leakage (CWE-209)", "levels": [
                {"score": 1, "description": "Stack traces and internal details exposed to users"},
                {"score": 3, "description": "Generic errors in production, but debug mode easily enabled"},
                {"score": 5, "description": "Structured error responses, no internals exposed, centralized handling"},
            ], "min_score": 3, "weight": 1.0},
        ],
        "pass_threshold": 0.7,
        "aggregation": "min",
    },
    "cwe_top_25": {
        "criteria": [
            {"type": "rubric", "name": "Injection Prevention (CWE-89/78/79)", "levels": [
                {"score": 1, "description": "No input sanitization, raw concatenation"},
                {"score": 3, "description": "Partial sanitization, some unprotected paths"},
                {"score": 5, "description": "All inputs validated, parameterized, and escaped"},
            ], "min_score": 4, "weight": 2.0},
            {"type": "rubric", "name": "Access Control (CWE-284/287/862)", "levels": [
                {"score": 1, "description": "No auth or authorization enforcement"},
                {"score": 3, "description": "Auth present but inconsistent enforcement"},
                {"score": 5, "description": "Complete auth + authz coverage, tested, least privilege"},
            ], "min_score": 4, "weight": 2.0},
            {"type": "rubric", "name": "Data Protection (CWE-200/327/502)", "levels": [
                {"score": 1, "description": "Sensitive data unprotected, unsafe deserialization"},
                {"score": 3, "description": "Basic protection with some gaps"},
                {"score": 5, "description": "Encryption at rest/transit, safe serialization, no leaks"},
            ], "min_score": 3, "weight": 1.5},
            {"type": "rubric", "name": "Resource Management (CWE-400/476/787)", "levels": [
                {"score": 1, "description": "No resource limits, null derefs, buffer issues"},
                {"score": 3, "description": "Basic limits and null checks"},
                {"score": 5, "description": "Proper resource limits, null safety, bounds checking"},
            ], "min_score": 3, "weight": 1.0},
            {"type": "rubric", "name": "Configuration Security (CWE-798/732/434)", "levels": [
                {"score": 1, "description": "Hardcoded secrets, permissive defaults, unrestricted upload"},
                {"score": 3, "description": "Secrets externalized but weak file permissions"},
                {"score": 5, "description": "Secrets in vault, least-privilege defaults, upload validation"},
            ], "min_score": 3, "weight": 1.5},
        ],
        "pass_threshold": 0.75,
        "aggregation": "min",
    },
}

# CWE profile names for documentation and validation
CWE_SECURITY_PROFILES: set[str] = {"cwe_injection", "cwe_auth", "cwe_data_exposure", "cwe_top_25"}

RECURSIVE_CONTEXT_STAGES: set[str] = {"index", "extract", "brief"}

# Context window sizes (tokens) per canonical model name — informational.
CONTEXT_WINDOWS: dict[str, int] = {
    # Claude
    "haiku": 200_000,
    "sonnet": 200_000,
    "opus": 200_000,
    "opusplan": 200_000,
    # Codex
    "gpt-5.3-codex": 400_000,
    "gpt-5.2-codex": 400_000,
    "gpt-5.1-codex": 400_000,
    "gpt-5-codex": 400_000,
    "gpt-5-codex-mini": 400_000,
    # Gemini
    "gemini-2.5-flash": 1_000_000,
    "gemini-2.5-flash-lite": 1_000_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-3-flash-preview": 1_000_000,
    "gemini-3-pro-preview": 1_000_000,
    "gemini-3.1-pro-preview": 1_000_000,
}


@dataclass
class CircuitBreakerSpec:
    max_total_failures: int = 5
    action: CircuitBreakerAction = "fail"

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_total_failures": self.max_total_failures,
            "action": self.action,
        }


TRAJECTORY_GUARD_ACTIONS: set[str] = {"warn", "abort", "escalate"}

# -- Capability-Based Tool Access (v2.0) --

# Known tool names per engine (for allowed_tools validation)
CLAUDE_TOOLS: frozenset[str] = frozenset({
    "Read", "Write", "Edit", "Bash", "Glob", "Grep",
    "WebSearch", "WebFetch", "TodoWrite",
})

CODEX_SANDBOX_LEVELS: frozenset[str] = frozenset({
    "workspace-write", "workspace-read-only", "network-off",
})

# Shorthand categories that expand to per-engine tool lists
TOOL_CATEGORIES: dict[str, dict[str, list[str]]] = {
    "read-only": {
        "claude": ["Read", "Glob", "Grep", "WebSearch", "WebFetch"],
        "codex": ["workspace-read-only"],
    },
    "no-shell": {
        "claude": ["Read", "Write", "Edit", "Glob", "Grep",
                   "WebSearch", "WebFetch", "TodoWrite"],
        "codex": ["workspace-read-only"],
    },
    "git-only": {
        "claude": ["Read", "Glob", "Grep", "Bash(git *)"],
        "codex": ["workspace-read-only"],
    },
    "src-scoped": {
        "claude": ["Read(src/*)", "Edit(src/*)", "Glob", "Grep",
                   "Bash(git *)", "WebSearch", "WebFetch"],
        "codex": ["workspace-write"],
    },
}

# Regex for wildcard tool patterns: ToolName(pattern)
_TOOL_PATTERN_RE_STR = r"^([A-Za-z]\w*)\((.+)\)$"


@dataclass
class TrajectoryGuardSpec:
    """Real-time trajectory guardrail evaluated during task execution."""

    max_tool_calls: int | None = None
    max_retries_without_progress: int | None = None
    scope_pattern: str | None = None
    on_violation: TrajectoryGuardAction = "warn"

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_tool_calls": self.max_tool_calls,
            "max_retries_without_progress": self.max_retries_without_progress,
            "scope_pattern": self.scope_pattern,
            "on_violation": self.on_violation,
        }


@dataclass
class PopulationSpec:
    """Run N model candidates per task, pick the best output."""

    candidates: list[str] = field(default_factory=list)
    strategy: PopulationStrategy = "best"
    parallel: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": self.candidates,
            "strategy": self.strategy,
            "parallel": self.parallel,
        }


MCPTransport = Literal["stdio", "http", "sse"]
MCP_TRANSPORTS: set[str] = {"stdio", "http", "sse"}


@dataclass
class MCPServerSpec:
    """Declaration of an MCP server available to tasks."""

    name: str
    command: list[str] = field(default_factory=list)
    description: str = ""
    url: str | None = None
    transport: MCPTransport = "stdio"
    env: dict[str, str] = field(default_factory=dict)
    allowed_task_roles: list[str] = field(default_factory=list)
    is_concurrency_safe: bool | None = None
    timeout_sec: int = 30

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "command": self.command,
            "description": self.description,
            "url": self.url,
            "transport": self.transport,
            "env": self.env,
            "allowed_task_roles": self.allowed_task_roles,
            "timeout_sec": self.timeout_sec,
        }
        if self.is_concurrency_safe is not None:
            payload["is_concurrency_safe"] = self.is_concurrency_safe
        return payload


@dataclass
class PolicySpec:
    """Declarative runtime policy evaluated at task dispatch time."""

    name: str
    rule: str
    action: PolicyAction = "warn"
    message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rule": self.rule,
            "action": self.action,
            "message": self.message,
        }


@dataclass
class PolicyViolation:
    """Record of a policy violation detected at task dispatch time."""

    policy_name: str
    task_id: str
    action: PolicyAction
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_name": self.policy_name,
            "task_id": self.task_id,
            "action": self.action,
            "message": self.message,
        }


@dataclass
class BlameNode:
    task_id: str
    category: BlameCategory
    confidence: float
    message: str
    caused_by: str | None = None
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "confidence": self.confidence,
            "message": self.message,
            "caused_by": self.caused_by,
            "evidence": self.evidence,
        }


@dataclass
class BlameChain:
    root_task_id: str = ""
    nodes: list[BlameNode] = field(default_factory=list)
    suggested_fixes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "root_task_id": self.root_task_id,
            "nodes": [n.to_dict() for n in self.nodes],
            "suggested_fixes": self.suggested_fixes,
        }


@dataclass
class ContextSelectionEntry:
    """One upstream's context selection decision for a downstream task."""

    upstream_id: str
    score: float = 0.0                    # BM25/IDF score
    keywords_matched: list[str] = field(default_factory=list)
    hop_distance: int = 0                 # 0 = direct dep, 1+ = transitive
    hop_decay_factor: float = 1.0         # decay applied (0.8^hops)
    tokens_raw: int = 0                   # tokens before trimming
    tokens_final: int = 0                 # tokens after trimming
    trimmed: bool = False                 # was this upstream trimmed?
    trim_reason: str = ""                 # e.g. "budget_eviction", "hop_decay"

    def to_dict(self) -> dict[str, Any]:
        return {
            "upstream_id": self.upstream_id,
            "score": round(self.score, 4),
            "keywords_matched": self.keywords_matched,
            "hop_distance": self.hop_distance,
            "hop_decay_factor": round(self.hop_decay_factor, 4),
            "tokens_raw": self.tokens_raw,
            "tokens_final": self.tokens_final,
            "trimmed": self.trimmed,
            "trim_reason": self.trim_reason,
        }


@dataclass
class ContextTrajectoryReport:
    """Context selection trajectory for a single task execution."""

    task_id: str
    entries: list[ContextSelectionEntry] = field(default_factory=list)
    total_tokens_raw: int = 0
    total_tokens_final: int = 0
    budget_tokens: int | None = None
    upstreams_evicted: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "entries": [e.to_dict() for e in self.entries],
            "total_tokens_raw": self.total_tokens_raw,
            "total_tokens_final": self.total_tokens_final,
            "budget_tokens": self.budget_tokens,
            "upstreams_evicted": self.upstreams_evicted,
        }


@dataclass
class EngineDefaults:
    model: str | None = None
    reasoning_effort: str | None = None
    args: list[str] = field(default_factory=list)
    append_system_prompt: str | None = None
    # v1.3.0 -- Resilience defaults (set via plan-level defaults.<engine>)
    escalation: list[str] = field(default_factory=list)
    fallback_engine: str | None = None
    fallback_model: str | None = None
    # v1.13.0 -- Model for context operations (summarized/map_reduce/recursive)
    context_model: str | None = None
    # v2.0 -- Capability-Based Tool Access
    allowed_tools: list[str] | None = None


@dataclass
class FailureRecord:
    """Record of a single failed attempt during retry.

    ``verify_tail`` and ``duration_sec`` were added 2026-04-27 in response to
    an internal post-mortem (PM3.1) — they let
    authors see what each failed attempt saw without diffing log files
    manually when a later retry succeeds. Both default to empty / 0.0 for
    backward compatibility with deserialized historical manifests.
    """

    attempt: int
    category: FailureCategory
    exit_code: int | None
    message: str
    verify_tail: str = ""
    duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "attempt": self.attempt,
            "category": self.category,
            "exit_code": self.exit_code,
            "message": self.message,
        }
        if self.verify_tail:
            d["verify_tail"] = self.verify_tail
        if self.duration_sec > 0.0:
            d["duration_sec"] = round(self.duration_sec, 2)
        return d


@dataclass
class RubricLevel:
    """Single level in a Likert rubric scale."""

    score: int  # 1-5
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {"score": self.score, "description": self.description}


@dataclass
class RubricCriterion:
    """Named criterion with Likert-scale rubric."""

    name: str
    levels: list[RubricLevel]  # Ordered 1-5 (or subset)
    min_score: int = 3  # Minimum passing score
    weight: float = 1.0  # For weighted_mean aggregation

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "levels": [lv.to_dict() for lv in self.levels],
            "min_score": self.min_score,
            "weight": self.weight,
        }


@dataclass
class JudgeSpec:
    """Configuration for LLM-as-Judge quality evaluation."""

    criteria: list[str | dict[str, Any]] = field(default_factory=list)
    pass_threshold: float = 0.7
    on_fail: JudgeOnFail = "fail"
    model: str = "haiku"
    method: JudgeMethod = "direct"
    aggregation: ScoreAggregation = "mean"
    preset: str | None = None
    timeout_sec: int | None = None
    quorum: int | None = None
    quorum_strategy: QuorumStrategy | None = None
    quorum_diversity: bool = False
    debate_rounds: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "criteria": self.criteria,
            "pass_threshold": self.pass_threshold,
            "on_fail": self.on_fail,
            "model": self.model,
            "method": self.method,
            "aggregation": self.aggregation,
            "preset": self.preset,
            "timeout_sec": self.timeout_sec,
            "quorum": self.quorum,
            "quorum_strategy": self.quorum_strategy,
            "quorum_diversity": self.quorum_diversity,
            "debate_rounds": self.debate_rounds,
        }


@dataclass
class CriterionScore:
    """Score for a single evaluation criterion."""

    criterion: str
    passed: bool
    score: float
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "passed": self.passed,
            "score": self.score,
            "reasoning": self.reasoning,
        }


@dataclass
class JudgeResult:
    """Result of LLM-as-Judge evaluation."""

    verdict: JudgeVerdict
    overall_score: float
    criterion_scores: list[CriterionScore] = field(default_factory=list)
    reasoning: str = ""
    eval_steps: list[str] = field(default_factory=list)
    previous_score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "overall_score": self.overall_score,
            "criterion_scores": [c.to_dict() for c in self.criterion_scores],
            "reasoning": self.reasoning,
            "eval_steps": self.eval_steps,
            "previous_score": self.previous_score,
        }


@dataclass
class BatchSpec:
    """Specification for batch processing within a single engine task."""

    items: list[str] = field(default_factory=list)
    template: str = ""
    max_per_call: int = 5

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": self.items,
            "template": self.template,
            "max_per_call": self.max_per_call,
        }


@dataclass
class BatchItemResult:
    """Result for a single item within a batch task."""

    item: str
    chunk_index: int
    output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "item": self.item,
            "chunk_index": self.chunk_index,
            "output": self.output,
        }


@dataclass
class TaskSignal:
    """A mid-task signal sent from a running engine process to the scheduler."""

    signal_type: SignalType
    timestamp: str  # ISO format
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "signal_type": self.signal_type,
            "timestamp": self.timestamp,
            "payload": self.payload,
        }


@dataclass
class HandoffReport:
    """Summary artifact for manual continuation after unrecoverable failure."""

    failure_category: FailureCategory = "runtime_error"
    partial_output: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "failure_category": self.failure_category,
            "partial_output": self.partial_output,
            "summary": self.summary,
        }


@dataclass
class WorkspaceExtraction:
    """Result of the extract pass -- relevant files identified for a task."""

    relevant_files: list[str] = field(default_factory=list)
    snippets: dict[str, str] = field(default_factory=dict)
    reasoning: str = ""
    token_estimate: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevant_files": list(self.relevant_files),
            "snippets": dict(self.snippets),
            "reasoning": self.reasoning,
            "token_estimate": self.token_estimate,
        }


# Backward-compatible alias used by previous internal prototypes.
ContextExtraction = WorkspaceExtraction


@dataclass
class WorkspaceBrief:
    """Result of the brief pass -- focused context document for the agent."""

    brief_text: str = ""
    token_estimate: int = 0
    files_referenced: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "brief_text": self.brief_text,
            "token_estimate": self.token_estimate,
            "files_referenced": list(self.files_referenced),
        }


@dataclass
class RecursiveContext:
    """Full recursive context artifact: index -> extract -> brief."""

    stages: list[RecursiveContextStage] = field(default_factory=list)
    index: _WsIndex | None = None
    extraction: WorkspaceExtraction | None = None
    brief: WorkspaceBrief | None = None
    workspace_brief: str = ""
    duration_sec: float = 0.0
    reused_index: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": self.stages,
            "index": self.index.to_dict() if self.index else None,
            "extraction": self.extraction.to_dict() if self.extraction else None,
            "brief": self.brief.to_dict() if self.brief else None,
            "workspace_brief": self.workspace_brief,
            "duration_sec": self.duration_sec,
            "reused_index": self.reused_index,
        }


@dataclass
class Suggestion:
    """A single optimization suggestion for a task."""
    task_id: str
    category: SuggestionCategory
    severity: Literal["high", "medium", "low", "info"]
    reason: str
    current_value: str
    suggested_value: str
    confidence: float  # 0.0-1.0
    estimated_savings_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "severity": self.severity,
            "reason": self.reason,
            "current_value": self.current_value,
            "suggested_value": self.suggested_value,
            "confidence": self.confidence,
            "estimated_savings_pct": self.estimated_savings_pct,
        }


@dataclass
class PlanSuggestions:
    """Aggregated suggestions for an entire plan."""
    plan_name: str
    runs_analyzed: int
    suggestions: list[Suggestion] = field(default_factory=list)
    total_estimated_savings_pct: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "runs_analyzed": self.runs_analyzed,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "total_estimated_savings_pct": self.total_estimated_savings_pct,
        }


@dataclass
class ModelRecord:
    """Per-model aggregated performance stats for a task across runs."""
    model: str
    runs: int
    successes: int
    failures: int
    timeouts: int
    avg_duration_sec: float
    avg_cost_usd: float | None
    recent_outcomes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "runs": self.runs,
            "successes": self.successes,
            "failures": self.failures,
            "timeouts": self.timeouts,
            "avg_duration_sec": round(self.avg_duration_sec, 3),
            "avg_cost_usd": round(self.avg_cost_usd, 4) if self.avg_cost_usd is not None else None,
            "recent_outcomes": self.recent_outcomes[-10:] if self.recent_outcomes else [],
        }


@dataclass
class TaskHistory:
    """Historical performance profile for a task across prior runs."""
    task_id: str
    total_runs: int
    records: dict[str, ModelRecord] = field(default_factory=dict)  # model -> stats

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "total_runs": self.total_runs,
            "records": {k: v.to_dict() for k, v in self.records.items()},
        }


@dataclass
class ScoreRecord:
    """Cross-run score artifact for a concrete plan variant."""

    plan_name: str
    plan_hash: str
    run_id: str
    success: bool
    cost_usd: float | None
    quality_score: float | None
    duration_sec: float
    timestamp: str
    valid_from: str = ""
    valid_to: str | None = None
    recorded_at: str = ""
    source_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "plan_hash": self.plan_hash,
            "run_id": self.run_id,
            "success": self.success,
            "cost_usd": self.cost_usd,
            "quality_score": self.quality_score,
            "duration_sec": self.duration_sec,
            "timestamp": self.timestamp,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "recorded_at": self.recorded_at,
            "source_id": self.source_id,
            "metadata": dict(self.metadata),
        }


@dataclass
class HistoricalPruningDecision:
    """Decision helper for pruning historically bad plan variants."""

    plan_hash: str
    sample_size: int
    failures: int
    failure_rate: float
    threshold: float
    min_runs: int
    prune: bool
    horizon_days: int | None = None
    recent_runs: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_hash": self.plan_hash,
            "sample_size": self.sample_size,
            "failures": self.failures,
            "failure_rate": round(self.failure_rate, 4),
            "threshold": self.threshold,
            "min_runs": self.min_runs,
            "prune": self.prune,
            "horizon_days": self.horizon_days,
            "recent_runs": self.recent_runs,
        }


@dataclass
class TaskContract:
    producer_task_id: str
    contract_type: str
    summary: str
    body: str
    content_hash: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "producer_task_id": self.producer_task_id,
            "contract_type": self.contract_type,
            "summary": self.summary,
            "body": self.body,
            "content_hash": self.content_hash,
            "metadata": dict(self.metadata),
        }


@dataclass
class PlanDefaults:
    env: dict[str, str] = field(default_factory=dict)
    secrets: list[str] = field(default_factory=list)
    secrets_auto: bool = False
    timeout_sec: int | None = None
    requires_clean_worktree: bool = False
    stdout_tail_lines: int = 50
    edit_policy: EditPolicy = "default"
    retry_delay_sec: list[float] | float | None = None
    context_budget_tokens: int | None = None
    budget_warning_pct: float | None = None
    workspace_index_exclude: list[str] = field(default_factory=list)
    codex: EngineDefaults = field(default_factory=EngineDefaults)
    claude: EngineDefaults = field(default_factory=EngineDefaults)
    gemini: EngineDefaults = field(default_factory=EngineDefaults)
    copilot: EngineDefaults = field(default_factory=EngineDefaults)
    qwen: EngineDefaults = field(default_factory=EngineDefaults)
    ollama: EngineDefaults = field(default_factory=EngineDefaults)
    llama: EngineDefaults = field(default_factory=EngineDefaults)
    context_compaction: ContextCompaction | None = None
    signals: bool = False


@dataclass
class TaskSpec:
    id: str
    description: str = ""
    tags: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    consistency_group: list[str] = field(default_factory=list)
    reconcile_after: list[str] = field(default_factory=list)

    engine: EngineName | None = None
    command: str | list[str] | None = None
    shell: bool | None = None

    # Engine settings
    agent: str | None = None
    model: str | None = None
    reasoning_effort: str | None = None
    args: list[str] = field(default_factory=list)

    # Prompt sources
    prompt: str | None = None
    prompt_file: str | None = None
    prompt_md_file: str | None = None
    prompt_md_heading: str | None = None

    workdir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    timeout_sec: int | None = None
    allow_failure: bool = False
    max_retries: int = 0
    retry_delay_sec: list[float] | float | None = None
    requires_clean_worktree: bool | None = None
    worktree: bool = False
    pre_command: str | list[str] | None = None
    verify_command: str | list[str] | None = None

    # Inter-task context passing
    context_from: list[str] = field(default_factory=list)
    context_mode: ContextMode = "raw"
    context_budget_tokens: int | None = None
    context_compact: bool = False  # deprecated — use context_compaction
    context_compaction: ContextCompaction | None = None
    workspace_index_exclude: list[str] = field(default_factory=list)
    # v1.13.0 -- override model for context operations (summarized/map_reduce/recursive)
    context_model: str | None = None

    # Output capture
    stdout_tail_lines: int | None = None

    # System prompt injection + edit policy
    append_system_prompt: str | None = None
    edit_policy: EditPolicy | None = None

    # Conditional execution
    when: str | None = None

    # Caching
    cache: bool = True  # Enable caching for this task (default: True)
    negative_cache_ttl_sec: int | None = None  # None => use default short TTL, 0 => disable

    # Group task — references another plan file that runs as a nested DAG
    group: str | None = None

    # Matrix expansion fields
    matrix: dict[str, list[str]] | None = None
    matrix_parent: str | None = None
    matrix_values: dict[str, str] | None = None

    # v0.6.0 -- Context Intelligence
    checkpoint: bool = False
    judge: JudgeSpec | None = None

    # v0.9.0 -- Smart Evaluation
    guard_command: str | list[str] | None = None
    assertions: list[dict[str, Any]] = field(default_factory=list)
    contract_type: str | None = None
    consumes_contracts: list[str] = field(default_factory=list)
    max_iterations: int | None = None

    # v1.15.0 -- Structured task outputs (T1.1)
    output_schema: dict[str, Any] | None = None  # JSON Schema for structured output

    # v0.8.0 -- Approval Gates
    requires_approval: bool = False
    approval_message: str | None = None

    # v1.3.0 -- Resilience
    escalation: list[str] = field(default_factory=list)
    fallback_engine: EngineName | None = None
    fallback_model: str | None = None

    # v1.6.0 -- Configurable fault tolerance
    retry_strategy: RetryStrategy | None = None

    # v1.10.0 -- Control Flow Integrity
    observation_block: bool = False
    # Untrusted Context Detection (v1.21.0)
    context_trust: ContextTrust | None = None

    # v1.12.0 -- Batch processing
    batch: BatchSpec | None = None

    # v1.12.0 -- Frozen tasks (immutable harness for mode: improve)
    frozen: bool = False
    compress_before: bool = False  # Trigger context compaction before task runs
    honeypot: bool = False  # Inject decoy variables into untrusted context
    output_scope: list[str] = field(default_factory=list)  # Allowed output file globs

    # v1.14.0 -- Deliberation Gate
    deliberation: bool = False
    deliberation_threshold: float = 0.5

    # T2.1 -- Dynamic Task Decomposition
    dynamic_group: bool = False

    # T2.2 -- Mid-task Signals
    signals: bool = False

    # v1.24.0 -- Event-Driven System Reminders
    reminders: list[dict[str, str]] | None = None

    # v1.26.0 -- Privacy-Aware Context Pipeline
    output_redact: list[str] = field(default_factory=list)
    context_allowlist: list[str] = field(default_factory=list)

    # v1.26.0 -- Trajectory-Level Guardrails
    trajectory_guard: TrajectoryGuardSpec | None = None

    # v1.26.0 -- Phantom Output Interception
    phantom_workspace: bool = False

    # v1.28.0 -- Population-Based Search
    population: PopulationSpec | None = None

    # v1.29.0 -- MCP-Native Tool Orchestration
    mcp_tools: list[str] = field(default_factory=list)

    # v2.0 -- Capability-Based Tool Access
    allowed_tools: list[str] | None = None

    # v1.36.0 -- Council Mode
    council: Any | None = None  # CouncilSpec from council.py (avoids circular import)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "tags": self.tags,
            "depends_on": self.depends_on,
            "consistency_group": self.consistency_group,
            "reconcile_after": self.reconcile_after,
            "engine": self.engine,
            "command": self.command,
            "shell": self.shell,
            "agent": self.agent,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "args": self.args,
            "prompt": self.prompt,
            "prompt_file": self.prompt_file,
            "prompt_md_file": self.prompt_md_file,
            "prompt_md_heading": self.prompt_md_heading,
            "workdir": self.workdir,
            "env": self.env,
            "timeout_sec": self.timeout_sec,
            "allow_failure": self.allow_failure,
            "max_retries": self.max_retries,
            "retry_delay_sec": self.retry_delay_sec,
            "requires_clean_worktree": self.requires_clean_worktree,
            "worktree": self.worktree,
            "pre_command": self.pre_command,
            "verify_command": self.verify_command,
            "context_from": self.context_from,
            "context_mode": self.context_mode,
            "context_budget_tokens": self.context_budget_tokens,
            "context_compact": self.context_compact,
            "context_compaction": self.context_compaction,
            "workspace_index_exclude": self.workspace_index_exclude,
            "context_model": self.context_model,
            "stdout_tail_lines": self.stdout_tail_lines,
            "append_system_prompt": self.append_system_prompt,
            "edit_policy": self.edit_policy,
            "when": self.when,
            "cache": self.cache,
            "negative_cache_ttl_sec": self.negative_cache_ttl_sec,
            "group": self.group,
            "matrix": self.matrix,
            "matrix_parent": self.matrix_parent,
            "matrix_values": self.matrix_values,
            "checkpoint": self.checkpoint,
            "judge": self.judge.to_dict() if self.judge else None,
            "guard_command": self.guard_command,
            "assert": self.assertions,
            "contract_type": self.contract_type,
            "consumes_contracts": self.consumes_contracts,
            "max_iterations": self.max_iterations,
            "requires_approval": self.requires_approval,
            "approval_message": self.approval_message,
            "escalation": self.escalation,
            "fallback_engine": self.fallback_engine,
            "fallback_model": self.fallback_model,
            "retry_strategy": self.retry_strategy,
            "observation_block": self.observation_block,
            "context_trust": self.context_trust,
            "batch": self.batch.to_dict() if self.batch else None,
            "frozen": self.frozen,
            "deliberation": self.deliberation,
            "deliberation_threshold": self.deliberation_threshold,
            "dynamic_group": self.dynamic_group,
            "signals": self.signals,
            "reminders": self.reminders,
            "output_redact": self.output_redact,
            "context_allowlist": self.context_allowlist,
            "trajectory_guard": self.trajectory_guard.to_dict() if self.trajectory_guard else None,
            "phantom_workspace": self.phantom_workspace,
            "population": self.population.to_dict() if self.population else None,
            "mcp_tools": self.mcp_tools,
            "allowed_tools": self.allowed_tools,
        }


@dataclass
class PlanImport:
    """Represents an imported task template file."""

    path: str
    prefix: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"path": self.path, "prefix": self.prefix}
        if self.overrides:
            d["overrides"] = self.overrides
        return d


@dataclass
class PlanSpec:
    name: str
    version: int = 1
    webhook_url: str | None = None
    goal: str = ""
    firewall_model: str | None = None
    secrets: list[str] = field(default_factory=list)
    secrets_auto: bool = False
    workspace_root: str | None = None
    max_parallel: int = 1
    fail_fast: bool = True
    run_dir: str = ".maestro-runs"
    max_cost_usd: float | None = None
    budget_warning_pct: float | None = None
    routing_strategy: RoutingStrategy | None = None
    control_flow_integrity: bool = False
    defaults: PlanDefaults = field(default_factory=PlanDefaults)
    tasks: list[TaskSpec] = field(default_factory=list)
    imports: list[PlanImport] = field(default_factory=list)
    audit_packs: list[str] = field(default_factory=list)
    policies: list[PolicySpec] = field(default_factory=list)
    mcp_servers: list[MCPServerSpec] = field(default_factory=list)
    watch: WatchSpec | None = None
    circuit_breaker: CircuitBreakerSpec | None = None
    budget_period: str | None = None  # daily | weekly | monthly
    source_path: Path | None = None
    validation_warnings: list[str] = field(default_factory=list)

    @property
    def source_dir(self) -> Path:
        if self.source_path:
            return self.source_path.parent
        return Path.cwd()

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": self.version,
            "name": self.name,
            "webhook_url": self.webhook_url,
            "goal": self.goal,
            "firewall_model": self.firewall_model,
            "secrets": self.secrets,
            "secrets_auto": self.secrets_auto,
            "workspace_root": self.workspace_root,
            "max_parallel": self.max_parallel,
            "fail_fast": self.fail_fast,
            "run_dir": self.run_dir,
            "max_cost_usd": self.max_cost_usd,
            "budget_warning_pct": self.budget_warning_pct,
            "defaults": self.defaults,
            "tasks": self.tasks,
            "imports": [imp.to_dict() for imp in self.imports],
            "audit_packs": self.audit_packs,
        }
        if self.control_flow_integrity:
            d["control_flow_integrity"] = self.control_flow_integrity
        if self.routing_strategy is not None:
            d["routing_strategy"] = self.routing_strategy
        if self.policies:
            d["policies"] = [p.to_dict() for p in self.policies]
        if self.mcp_servers:
            d["mcp_servers"] = [s.to_dict() for s in self.mcp_servers]
        if self.watch:
            d["watch"] = self.watch.to_dict()
        if self.circuit_breaker is not None:
            d["circuit_breaker"] = self.circuit_breaker.to_dict()
        return d


@dataclass
class StructuredContext:
    """Structured information extracted from a task's output."""

    task_id: str
    status: str
    exit_code: int | None
    duration_sec: float
    files_changed: list[str] = field(default_factory=list)
    decisions: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    cost_usd: float | None = None
    result_text: str = ""
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status,
            "exit_code": self.exit_code,
            "duration_sec": self.duration_sec,
            "files_changed": self.files_changed,
            "decisions": self.decisions,
            "errors": self.errors,
            "warnings": self.warnings,
            "cost_usd": self.cost_usd,
            "result_text": self.result_text,
            "summary": self.summary,
        }


@dataclass
class TokenUsage:
    """Token counts extracted from an engine task's output."""

    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.cached_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "cached_tokens": self.cached_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_tokens": self.cache_creation_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass
class TaskResult:
    task_id: str
    status: TaskStatus
    exit_code: int | None = None
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime = field(default_factory=datetime.now)
    duration_sec: float = 0.0
    command: str = ""
    log_path: Path = field(default_factory=Path)
    result_path: Path = field(default_factory=Path)
    message: str = ""
    stdout_tail: str = ""
    cost_usd: float | None = None
    token_usage: TokenUsage | None = None
    structured_context: StructuredContext | None = None
    retry_count: int = 0
    failure_history: list[FailureRecord] = field(default_factory=list)
    checkpoint_count: int = 0
    tool_call_count: int = 0
    tool_failure_count: int = 0
    judge_result: JudgeResult | None = None
    handoff_report: HandoffReport | None = None
    produced_contract: TaskContract | None = None
    context_raw_tokens: int = 0
    context_final_tokens: int = 0
    context_compression_ratio: float = 0.0
    workspace_brief: WorkspaceBrief | None = None
    context_raw_bytes: int = 0
    context_final_bytes: int = 0
    compression_ratio: float = 0.0
    task_hash: str | None = None
    worktree_merge: WorktreeMergeResult | None = None
    auto_routed_model: str | None = None
    context_trajectory: ContextTrajectoryReport | None = None
    output_envelope: OutputEnvelope | None = None

    # v1.12.0 -- Batch processing
    batch_results: list[BatchItemResult] = field(default_factory=list)
    batch_chunks_total: int = 0
    batch_items_total: int = 0

    # v1.14.0 -- Deliberation Gate
    deliberation_skipped: bool = False

    # v1.15.0 -- Structured task outputs (T1.1)
    structured_output: dict[str, Any] | None = None  # validated output from output_schema
    dynamic_subplan_result: dict[str, Any] | None = None  # T2.1 sub-plan summary

    # Untrusted Context Detection (v1.21.0)
    tainted: bool = False
    # T2.2 -- Mid-task Signals
    signals_received: list[TaskSignal] = field(default_factory=list)
    artifacts: list[dict[str, str]] = field(default_factory=list)
    last_progress_pct: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "duration_sec": self.duration_sec,
            "command": self.command,
            "log_path": str(self.log_path),
            "result_path": str(self.result_path),
            "message": self.message,
            "stdout_tail": self.stdout_tail,
            "cost_usd": self.cost_usd,
            "token_usage": self.token_usage.to_dict() if self.token_usage else None,
            "structured_context": self.structured_context.to_dict() if self.structured_context else None,
            "retry_count": self.retry_count,
            "failure_history": [f.to_dict() for f in self.failure_history],
            "checkpoint_count": self.checkpoint_count,
            "tool_call_count": self.tool_call_count,
            "judge_result": self.judge_result.to_dict() if self.judge_result else None,
            "handoff_report": self.handoff_report.to_dict() if self.handoff_report else None,
            "produced_contract": self.produced_contract.to_dict() if self.produced_contract else None,
            "context_raw_tokens": self.context_raw_tokens,
            "context_final_tokens": self.context_final_tokens,
            "context_compression_ratio": self.context_compression_ratio,
            "context_raw_bytes": self.context_raw_bytes,
            "context_final_bytes": self.context_final_bytes,
            "compression_ratio": self.compression_ratio,
            "task_hash": self.task_hash,
        }
        if self.workspace_brief is not None:
            d["workspace_brief"] = self.workspace_brief.to_dict()
        if self.worktree_merge is not None:
            d["worktree_merge"] = self.worktree_merge.to_dict()
        if self.auto_routed_model is not None:
            d["auto_routed_model"] = self.auto_routed_model
        if self.context_trajectory is not None:
            d["context_trajectory"] = self.context_trajectory.to_dict()
        if self.output_envelope is not None:
            d["output_envelope"] = self.output_envelope.to_dict()
        if self.batch_results:
            d["batch_results"] = [r.to_dict() for r in self.batch_results]
            d["batch_chunks_total"] = self.batch_chunks_total
            d["batch_items_total"] = self.batch_items_total
        if self.deliberation_skipped:
            d["deliberation_skipped"] = True
        if self.structured_output is not None:
            d["structured_output"] = self.structured_output
        if self.dynamic_subplan_result is not None:
            d["dynamic_subplan_result"] = self.dynamic_subplan_result
        if self.tainted:
            d["tainted"] = True
        if self.tool_failure_count:
            d["tool_failure_count"] = self.tool_failure_count
        if self.signals_received:
            d["signals_received"] = [s.to_dict() for s in self.signals_received]
        if self.artifacts:
            d["artifacts"] = self.artifacts
        if self.last_progress_pct is not None:
            d["last_progress_pct"] = self.last_progress_pct
        return d


@dataclass
class PlanRunResult:
    plan_name: str
    run_id: str
    run_path: Path
    started_at: datetime
    finished_at: datetime
    success: bool
    execution_profile: ExecutionProfile = "plan"
    task_results: dict[str, TaskResult] = field(default_factory=dict)
    sequential_duration_sec: float = 0.0
    parallelism_savings_pct: float = 0.0
    total_cost_usd: float | None = None
    total_tokens: int | None = None
    budget_exceeded: bool = False
    plan_hash: str | None = None
    quality_score: float | None = None
    task_graph: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "plan_name": self.plan_name,
            "run_id": self.run_id,
            "run_path": str(self.run_path),
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
            "success": self.success,
            "execution_profile": self.execution_profile,
            "task_results": {k: v.to_dict() for k, v in self.task_results.items()},
            "sequential_duration_sec": self.sequential_duration_sec,
            "parallelism_savings_pct": self.parallelism_savings_pct,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "budget_exceeded": self.budget_exceeded,
        }
        if self.plan_hash is not None:
            d["plan_hash"] = self.plan_hash
        if self.quality_score is not None:
            d["quality_score"] = self.quality_score
        if self.task_graph:
            d["task_graph"] = self.task_graph
        return d


def _serialize_model_value(value: Any) -> Any:
    """Convert nested dataclass content into JSON-safe primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {
            field_info.name: _serialize_model_value(getattr(value, field_info.name))
            for field_info in fields(value)
        }
    if isinstance(value, list):
        return [_serialize_model_value(item) for item in value]
    if isinstance(value, dict):
        return {
            str(key): _serialize_model_value(item)
            for key, item in value.items()
        }
    return value


@dataclass
class WorkflowVariant:
    """A single node in the Phase 3 workflow-search tree."""

    node_id: str
    plan_spec: PlanSpec
    run_result: PlanRunResult | None = None
    score: float = 0.0
    is_valid: bool = False
    parent: WorkflowVariant | None = field(default=None, repr=False)
    children: list[WorkflowVariant] = field(default_factory=list, repr=False)
    variant_type: VariantType = "draft"
    mutation_desc: str = ""
    plan_hash: str = ""
    visits: int = 0
    pruned: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "parent_id": self.parent.node_id if self.parent is not None else None,
            "child_ids": [child.node_id for child in self.children],
            "variant_type": self.variant_type,
            "mutation_desc": self.mutation_desc,
            "plan_hash": self.plan_hash,
            "plan_spec": _serialize_model_value(self.plan_spec),
            "run_result": _serialize_model_value(self.run_result),
            "score": self.score,
            "is_valid": self.is_valid,
            "visits": self.visits,
            "pruned": self.pruned,
            "metadata": _serialize_model_value(self.metadata),
        }


@dataclass
class ReplanAttempt:
    attempt_number: int
    plan_yaml: str = ""
    corrected_plan_yaml: str | None = None
    diff_summary: str = ""
    approved: bool = False
    run_result: PlanRunResult | None = None
    failed_task_ids: list[str] = field(default_factory=list)
    error_summary: str = ""
    analysis_response: str | None = None
    analysis_error: str | None = None
    candidate_variants: list[dict[str, Any]] = field(default_factory=list)
    selected_candidate_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "attempt_number": self.attempt_number,
            "plan_yaml": self.plan_yaml,
            "corrected_plan_yaml": self.corrected_plan_yaml,
            "diff_summary": self.diff_summary,
            "approved": self.approved,
            "run_result": self.run_result.to_dict() if self.run_result else None,
            "failed_task_ids": self.failed_task_ids,
            "error_summary": self.error_summary,
            "analysis_response": self.analysis_response,
            "analysis_error": self.analysis_error,
        }
        if self.candidate_variants:
            d["candidate_variants"] = self.candidate_variants
        if self.selected_candidate_id is not None:
            d["selected_candidate_id"] = self.selected_candidate_id
        return d


@dataclass
class ReplanState:
    plan_path: str = ""
    max_attempts: int = 3
    attempts: list[ReplanAttempt] = field(default_factory=list)
    status: str = "max_attempts_exceeded"
    final_success: bool = False
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    analysis_model: str = "opus"
    search_tree_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "plan_path": self.plan_path,
            "max_attempts": self.max_attempts,
            "attempts": [attempt.to_dict() for attempt in self.attempts],
            "status": self.status,
            "final_success": self.final_success,
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "analysis_model": self.analysis_model,
        }
        if self.search_tree_path is not None:
            d["search_tree_path"] = self.search_tree_path
        return d


@dataclass
class MultiPlanResult:
    plan_results: list[PlanRunResult] = field(default_factory=list)
    total_cost_usd: float | None = None
    total_tokens: int | None = None
    budget_exceeded: bool = False
    success: bool = True
    started_at: datetime = field(default_factory=datetime.now)
    finished_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_results": [result.to_dict() for result in self.plan_results],
            "total_cost_usd": self.total_cost_usd,
            "total_tokens": self.total_tokens,
            "budget_exceeded": self.budget_exceeded,
            "success": self.success,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Watch (v1.4.0) — autonomous iteration loop
# ---------------------------------------------------------------------------

MetricDirection = Literal["lower_is_better", "higher_is_better"]
MetricSource = Literal["stdout_regex", "verify_command", "guard_command", "json_field", "manifest"]
WatchMode = Literal["custom", "improve"]
OnRegression = Literal["rollback", "revert", "keep"]
PlateauAction = Literal["stop", "escalate_model", "notify"]
WatchStatus = Literal[
    "improved", "plateau", "budget_exceeded",
    "max_iterations", "interrupted", "error",
    "target_reached", "step_limit_reached",
]


@dataclass
class WatchSpec:
    """Plan-level configuration for ``maestro watch``."""

    metric: str
    max_iterations: int = 100
    iteration_budget_sec: int | None = None
    metric_direction: MetricDirection = "lower_is_better"
    metric_source: MetricSource = "stdout_regex"
    metric_pattern: str | None = None
    metric_task: str | None = None
    metric_json_path: str | None = None
    on_regression: OnRegression = "rollback"
    program_md: str | None = None
    warmup_iterations: int = 1
    plateau_threshold: int = 5
    plateau_action: PlateauAction = "stop"
    max_cost_usd: float | None = None
    consolidate_model: str | None = None
    consolidate_every: int = 3
    consolidate_prompt: str | None = None
    target_metric: float | None = None
    blame_plan: str | None = None
    mode: WatchMode = "custom"
    improve_model: str | None = None
    max_total_steps: int | None = None
    stepping_stones: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "max_iterations": self.max_iterations,
            "iteration_budget_sec": self.iteration_budget_sec,
            "metric_direction": self.metric_direction,
            "metric_source": self.metric_source,
            "metric_pattern": self.metric_pattern,
            "metric_task": self.metric_task,
            "metric_json_path": self.metric_json_path,
            "on_regression": self.on_regression,
            "program_md": self.program_md,
            "warmup_iterations": self.warmup_iterations,
            "plateau_threshold": self.plateau_threshold,
            "plateau_action": self.plateau_action,
            "max_cost_usd": self.max_cost_usd,
            "consolidate_model": self.consolidate_model,
            "consolidate_every": self.consolidate_every,
            "consolidate_prompt": self.consolidate_prompt,
            "target_metric": self.target_metric,
            "blame_plan": self.blame_plan,
            "mode": self.mode,
            "improve_model": self.improve_model,
            "max_total_steps": self.max_total_steps,
            "stepping_stones": self.stepping_stones,
        }


@dataclass
class SteppingStone:
    """Snapshot of a successful watch iteration for reuse by future runs."""

    plan_name: str
    plan_hash: str
    metric_value: float
    metric_name: str
    iteration: int
    git_commit: str | None = None
    plan_yaml: str = ""
    lessons: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = ""
    watch_run_path: str = ""
    total_cost_usd: float = 0.0
    source_type: str = "watch"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_name": self.plan_name,
            "plan_hash": self.plan_hash,
            "metric_value": self.metric_value,
            "metric_name": self.metric_name,
            "iteration": self.iteration,
            "git_commit": self.git_commit,
            "plan_yaml": self.plan_yaml,
            "lessons": self.lessons,
            "timestamp": self.timestamp,
            "watch_run_path": self.watch_run_path,
            "total_cost_usd": self.total_cost_usd,
            "source_type": self.source_type,
            "metadata": dict(self.metadata),
        }


@dataclass
class WatchIteration:
    """Per-iteration record for the experiment ledger."""

    iteration: int
    metric_value: float | None = None
    best_metric: float | None = None
    improved: bool = False
    action: str = ""
    cost_usd: float | None = None
    duration_sec: float = 0.0
    git_commit: str | None = None
    error: str | None = None
    timestamp: str = ""
    fix_summary: str | None = None
    manifest_excerpt: str | None = None
    blame_excerpt: str | None = None
    consolidated_excerpt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "metric_value": self.metric_value,
            "best_metric": self.best_metric,
            "improved": self.improved,
            "action": self.action,
            "cost_usd": self.cost_usd,
            "duration_sec": self.duration_sec,
            "git_commit": self.git_commit,
            "error": self.error,
            "timestamp": self.timestamp,
            "fix_summary": self.fix_summary,
            "manifest_excerpt": self.manifest_excerpt,
            "blame_excerpt": self.blame_excerpt,
            "consolidated_excerpt": self.consolidated_excerpt,
        }


@dataclass
class LessonRecord:
    """A semantic lesson extracted from a watch/improve iteration."""

    iteration: int
    task_id: str
    category: str  # timeout_fix, guard_fix, path_fix, escalation, etc.
    lesson: str  # human-readable description of what was tried and outcome
    confidence: float = 0.8  # 0.0-1.0, decays over time
    timestamp: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "iteration": self.iteration,
            "task_id": self.task_id,
            "category": self.category,
            "lesson": self.lesson,
            "confidence": self.confidence,
            "timestamp": self.timestamp,
        }


@dataclass
class KnowledgeRecord:
    """A piece of cross-run knowledge extracted from run history."""
    task_id: str
    kind: KnowledgeKind
    insight: str              # human-readable description
    confidence: float         # 0.0-1.0, increases with occurrences
    occurrences: int          # how many times observed
    first_seen: str           # ISO timestamp
    last_seen: str            # ISO timestamp

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "insight": self.insight,
            "confidence": round(self.confidence, 3),
            "occurrences": self.occurrences,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
        }


@dataclass
class KnowledgeWriteOutcome:
    """Outcome metadata for a persisted knowledge write attempt."""

    task_id: str
    kind: KnowledgeKind
    operation: str
    outcome: str
    trust_label: str
    instructionality_score: float
    source_type: str
    source_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "kind": self.kind,
            "operation": self.operation,
            "outcome": self.outcome,
            "trust_label": self.trust_label,
            "instructionality_score": round(self.instructionality_score, 3),
            "source_type": self.source_type,
            "source_id": self.source_id,
        }


@dataclass
class WatchState:
    """Overall state for a ``maestro watch`` session."""

    plan_path: str = ""
    iterations: list[WatchIteration] = field(default_factory=list)
    status: WatchStatus = "max_iterations"
    best_metric: float | None = None
    best_iteration: int | None = None
    total_cost_usd: float = 0.0
    total_iterations: int = 0
    plateau_count: int = 0
    total_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_path": self.plan_path,
            "iterations": [it.to_dict() for it in self.iterations],
            "status": self.status,
            "best_metric": self.best_metric,
            "best_iteration": self.best_iteration,
            "total_cost_usd": self.total_cost_usd,
            "total_iterations": self.total_iterations,
            "plateau_count": self.plateau_count,
            "total_steps": self.total_steps,
        }


@dataclass
class SessionSnapshot:
    """Durable session-memory snapshot for long-horizon watch runs."""

    id: int | None = None
    plan_name: str = ""
    watch_run_path: str = ""
    snapshot_kind: str = "watch"
    iteration_from: int = 0
    iteration_to: int = 0
    best_metric: float | None = None
    snapshot_text: str = ""
    recent_tail_count: int = 0
    recorded_at: str = ""
    source_type: str = "watch"
    source_id: str = ""
    trust_label: str = "trusted"
    instructionality_score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "plan_name": self.plan_name,
            "watch_run_path": self.watch_run_path,
            "snapshot_kind": self.snapshot_kind,
            "iteration_from": self.iteration_from,
            "iteration_to": self.iteration_to,
            "best_metric": self.best_metric,
            "snapshot_text": self.snapshot_text,
            "recent_tail_count": self.recent_tail_count,
            "recorded_at": self.recorded_at,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "trust_label": self.trust_label,
            "instructionality_score": self.instructionality_score,
            "metadata": dict(self.metadata),
        }


@dataclass
class MergeOverlap:
    """Files that overlap between this task and previously-merged tasks."""

    file: str
    merged_by: list[str] = field(default_factory=list)


@dataclass
class MergeReview:
    """Pre-merge intelligence for a worktree task."""

    verdict: MergeReviewVerdict
    overlapping_files: list[MergeOverlap] = field(default_factory=list)
    conflict_files: list[str] = field(default_factory=list)
    resolution_suggestion: str | None = None
    auto_resolved: bool = False
    review_model: str | None = None
    review_duration_sec: float = 0.0
    review_cost_usd: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "verdict": self.verdict,
            "overlapping_files": [
                {"file": o.file, "merged_by": o.merged_by}
                for o in self.overlapping_files
            ],
            "conflict_files": self.conflict_files,
            "auto_resolved": self.auto_resolved,
        }
        if self.resolution_suggestion:
            d["resolution_suggestion"] = self.resolution_suggestion
        if self.review_model:
            d["review_model"] = self.review_model
        if self.review_duration_sec > 0:
            d["review_duration_sec"] = round(self.review_duration_sec, 2)
        if self.review_cost_usd is not None:
            d["review_cost_usd"] = self.review_cost_usd
        return d


@dataclass
class WorktreeMergeResult:
    status: WorktreeMergeStatus
    files_changed: list[str] = field(default_factory=list)
    conflict_files: list[str] = field(default_factory=list)
    merge_commit: str | None = None
    error: str | None = None
    review: MergeReview | None = None

    verification: DualVerificationResult | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "status": self.status,
            "files_changed": self.files_changed,
            "conflict_files": self.conflict_files,
            "merge_commit": self.merge_commit,
            "error": self.error,
        }
        if self.review is not None:
            d["review"] = self.review.to_dict()
        if self.verification is not None:
            d["verification"] = self.verification.to_dict()
        return d


@dataclass
class DualVerificationResult:
    """Result of comparing agent-reported changes vs actual git diff.

    Inspired by CIBER's dual verification: cross-check textual claims
    against environmental evidence to detect verification gaps.
    """
    verified: bool  # True if agent claims match reality
    files_in_diff: list[str] = field(default_factory=list)  # actual changes
    files_claimed: list[str] = field(default_factory=list)  # agent-reported
    unclaimed_files: list[str] = field(default_factory=list)  # changed but not mentioned
    phantom_files: list[str] = field(default_factory=list)  # claimed but not changed
    overlap_ratio: float = 0.0  # |intersection| / |union|

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "files_in_diff": self.files_in_diff,
            "files_claimed": self.files_claimed,
            "unclaimed_files": self.unclaimed_files,
            "phantom_files": self.phantom_files,
            "overlap_ratio": self.overlap_ratio,
        }


@dataclass
class OutputEnvelope:
    """Signed envelope for task output integrity and scope verification.

    Captures a SHA-256 hash of the task output, the declared scope, and
    whether the task's actual file changes stayed within scope.
    """
    output_hash: str  # SHA-256 (first 16 hex) of stdout_tail
    scope_declared: list[str] = field(default_factory=list)  # from task.output_scope
    scope_violations: list[str] = field(default_factory=list)  # files outside scope
    scope_verified: bool = True  # True if no violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "output_hash": self.output_hash,
            "scope_declared": self.scope_declared,
            "scope_violations": self.scope_violations,
            "scope_verified": self.scope_verified,
        }


# ---------------------------------------------------------------------------
# Plan scaffolding models
# ---------------------------------------------------------------------------

TaskType = Literal[
    "shell", "trivial-fix", "implementation", "complex-implementation",
    "code-review", "qa-verification", "build-verify", "security-audit",
    "branch-setup",
]

Topology = Literal["linear", "fan-out", "diamond", "pipeline"]


@dataclass
class TaskBrief:
    """High-level task description for plan scaffolding."""
    id: str
    description: str = ""
    task_type: TaskType = "implementation"
    depends_on: list[str] = field(default_factory=list)
    engine: EngineName | None = None
    agent: str | None = None
    workdir: str | None = None
    prompt_hint: str = ""
    auto_split: bool = True  # split into read-plan+apply when prompt mentions large files


@dataclass
class PlanBrief:
    """High-level plan description for scaffolding."""
    name: str
    goal: str = ""
    workspace_root: str | None = None
    branch_name: str | None = None
    tasks: list[TaskBrief] = field(default_factory=list)
    topology: Topology = "pipeline"
    include_quality_gates: bool = True
    include_build_verify: bool = True
    max_parallel: int = 3
    fail_fast: bool = True
    library: str | None = None  # built-in name or path to workflow library YAML
    # When true, scaffold injects first-run sane defaults: timeout_sec=1500,
    # retry_delay_sec=[60, 120], max_cost_usd=10.0, budget_warning_pct=0.8.
    # Inspired by an internal post-mortem (2026-04-26): authors derived these from
    # warnings instead of getting them up front.
    strict_defaults: bool = False


@dataclass
class EventRecord:
    """Immutable event record with hash chain for tamper detection."""
    sequence: int
    event_type: str
    timestamp: str
    payload: dict[str, Any]
    prev_hash: str
    event_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.sequence,
            "type": self.event_type,
            "ts": self.timestamp,
            "payload": self.payload,
            "prev_hash": self.prev_hash,
            "hash": self.event_hash,
        }


# ---------------------------------------------------------------------------
# Shared display constants
# ---------------------------------------------------------------------------

STATUS_STYLES: dict[str, tuple[str, str]] = {
    "pending": ("[..]", "dim"),
    "running": ("[>>]", "bold cyan"),
    "success": ("[ok]", "bold green"),
    "failed": ("[!!]", "bold red"),
    "soft_failed": ("[~~]", "bold yellow"),
    "skipped": ("[--]", "dim"),
    "dry_run": ("[ok]", "bold green"),
}

TERMINAL_STATUSES: set[str] = {"success", "failed", "soft_failed", "skipped", "dry_run"}
