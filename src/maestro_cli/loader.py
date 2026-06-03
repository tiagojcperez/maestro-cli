from __future__ import annotations

import itertools
import math
import os
import re
from pathlib import Path, PurePosixPath
from typing import Any, cast

import yaml

from .errors import (
    E001,
    E002,
    E003,
    E004,
    E005,
    E006,
    E007,
    E008,
    E010,
    E011,
    E012,
    E013,
    E014,
    E015,
    E016,
    E017,
    E018,
    E019,
    E020,
    E021,
    E022,
    E023,
    E024,
    E025,
    E026,
    E027,
    E028,
    E029,
    E032,
    E033,
    E034,
    E035,
    E036,
    E037,
    E038,
    E039,
    E040,
    E041,
    E042,
    E043,
    E044,
    E045,
    E046,
    E047,
    E048,
    E052,
    E053,
    E054,
    E055,
    E056,
    E057,
    E058,
    E060,
    E062,
    E063,
    E064,
    E066,
    E067,
    E068,
    E069,
    E070,
    E071,
    E072,
    PlanValidationError,
)
from .models import (
    ASSERTION_TYPES,
    CLAUDE_MODELS,
    CLAUDE_REASONING_EFFORTS,
    CLAUDE_TOOLS,
    CODEX_SANDBOX_LEVELS,
    CONTRACT_TYPES,
    CODEX_REASONING_EFFORTS,
    CONTEXT_MODES,
    CONTEXT_COMPACTION_VALUES,
    CONTEXT_TRUST_VALUES,
    MCP_TRANSPORTS,
    POPULATION_STRATEGIES,
    TOOL_CATEGORIES,
    TRAJECTORY_GUARD_ACTIONS,
    COPILOT_MODELS,
    EDIT_POLICIES,
    GEMINI_MODELS,
    JUDGE_METHODS,
    JUDGE_ON_FAIL_VALUES,
    JUDGE_PRESETS,
    SCORE_AGGREGATIONS,
    QUORUM_STRATEGIES,
    MAX_RETRIES_LIMIT,
    BatchSpec,
    CircuitBreakerSpec,
    ContextCompaction,
    ContextMode,
    EditPolicy,
    EngineName,
    EngineDefaults,
    JudgeMethod,
    JudgeOnFail,
    JudgeSpec,
    MetricDirection,
    MetricSource,
    OnRegression,
    PlateauAction,
    PlanDefaults,
    PlanImport,
    PlanSpec,
    PolicyAction,
    PolicySpec,
    MCPServerSpec,
    PopulationSpec,
    QuorumStrategy,
    RetryStrategy,
    RoutingStrategy,
    ScoreAggregation,
    TaskSpec,
    TrajectoryGuardSpec,
    WatchMode,
    WatchSpec,
)
from .plugins import PluginResolutionError, get_engine_plugin
from .policy import compile_policy
from .relationships import build_consistency_group_members, clone_tasks_with_resolved_dependencies
from .runners import (
    _CODEX_MODEL_ALIASES,
    _COPILOT_MODEL_ALIASES,
    _ENV_ALLOWLIST,
    _GEMINI_MODEL_ALIASES,
)
from .utils import _TEMPLATE_RE, command_to_string, resolve_path
from .workspace_assertions import normalize_workspace_assertion

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-\./]*$")
_VALID_ENGINES = {"codex", "claude", "gemini", "copilot", "qwen", "ollama", "llama"}


def _to_str_dict(value: Any, field_name: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise PlanValidationError(f"{field_name} must be an object", code=E018)
    out: dict[str, str] = {}
    for k, v in value.items():
        out[str(k)] = str(v)
    return out


def _to_str_list(value: Any, field_name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        raise PlanValidationError(f"{field_name} must be a list or string", code=E018)
    return [str(item) for item in value]


def _set_engine_resilience_defaults(
    engine_defaults: EngineDefaults,
    engine_raw: dict[str, Any],
    field_prefix: str,
) -> None:
    setattr(
        engine_defaults,
        "escalation",
        _to_str_list(engine_raw.get("escalation"), f"{field_prefix}.escalation"),
    )
    setattr(
        engine_defaults,
        "fallback_engine",
        str(engine_raw["fallback_engine"]) if engine_raw.get("fallback_engine") is not None else None,
    )
    setattr(
        engine_defaults,
        "fallback_model",
        str(engine_raw["fallback_model"]) if engine_raw.get("fallback_model") is not None else None,
    )
    setattr(
        engine_defaults,
        "context_model",
        str(engine_raw["context_model"]) if engine_raw.get("context_model") is not None else None,
    )
    if "allowed_tools" in engine_raw:
        setattr(
            engine_defaults,
            "allowed_tools",
            _to_str_list(engine_raw["allowed_tools"], f"{field_prefix}.allowed_tools")
            if engine_raw["allowed_tools"] is not None
            else None,
        )


def _to_int_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(f"{field_name} must be an integer", code=E018) from exc


def _to_context_budget_or_none(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(f"{field_name} must be an integer", code=E019) from exc
    if parsed < 1:
        raise PlanValidationError(f"{field_name} must be >= 1", code=E019)
    return parsed


def _to_contract_type_or_none(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        raise PlanValidationError(f"{field_name} must be a non-empty string", code=E018)
    if text not in CONTRACT_TYPES:
        raise PlanValidationError(
            f"{field_name} '{text}' is not valid. Allowed: {sorted(CONTRACT_TYPES)}",
            code=E018,
        )
    return text


def _to_workspace_assertions(value: Any, field_name: str) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise PlanValidationError(f"{field_name} must be a list", code=E018)

    normalized: list[dict[str, Any]] = []
    for i, item in enumerate(value):
        try:
            normalized.append(normalize_workspace_assertion(item, f"{field_name}[{i}]"))
        except ValueError as exc:
            raise PlanValidationError(str(exc), code=E018) from exc
    return normalized


_KNOWN_CRITERIA_FIELDS: dict[str, set[str]] = {
    "contains": {"type", "value"},
    "regex": {"type", "pattern", "value"},
    "is-json": {"type"},
    "json-schema": {"type", "schema", "schema_file"},
    "llm-rubric": {"type", "value"},
    "cost_under": {"type", "value"},
    "duration_under": {"type", "value"},
    "rubric": {"type", "name", "levels", "min_score", "weight"},
}


def _check_unknown_criteria_fields(
    item: dict[str, Any], assertion_type: str, field_name: str,
) -> None:
    """Raise E020 if a typed judge criterion contains unrecognised fields."""
    known = _KNOWN_CRITERIA_FIELDS.get(assertion_type)
    if known is None:
        return  # unknown type already caught upstream
    unknown = set(item.keys()) - known
    if unknown:
        raise PlanValidationError(
            f"{field_name}: unknown field(s) {sorted(unknown)} "
            f"for criterion type '{assertion_type}'. "
            f"Allowed: {sorted(known)}",
            code=E020,
        )


def _to_judge_spec(value: Any, field_name: str) -> JudgeSpec | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PlanValidationError(f"{field_name} must be an object", code=E020)

    preset = value.get("preset")
    if preset is not None:
        if not isinstance(preset, str) or not preset.strip():
            raise PlanValidationError(
                f"{field_name}.preset must be a non-empty string",
                code=E020,
            )
        preset = preset.strip()
        if preset not in JUDGE_PRESETS:
            raise PlanValidationError(
                f"{field_name}.preset '{preset}' is not valid. "
                f"Allowed: {sorted(JUDGE_PRESETS)}",
                code=E020,
            )

    criteria_raw = value.get("criteria", [])
    if preset is not None and not criteria_raw:
        criteria_raw = JUDGE_PRESETS[preset]["criteria"]
    if not isinstance(criteria_raw, list) or not criteria_raw:
        raise PlanValidationError(
            f"{field_name}.criteria must be a non-empty list",
            code=E020,
        )
    criteria: list[str | dict[str, Any]] = []
    for i, item in enumerate(criteria_raw):
        if isinstance(item, dict):
            assertion_type = str(item.get("type", "")).strip()
            if not assertion_type:
                raise PlanValidationError(
                    f"{field_name}.criteria[{i}].type is required",
                    code=E020,
                )
            if assertion_type not in ASSERTION_TYPES:
                raise PlanValidationError(
                    f"{field_name}.criteria[{i}].type '{assertion_type}' is not valid. "
                    f"Allowed: {sorted(ASSERTION_TYPES)}",
                    code=E020,
                )
            if assertion_type == "json-schema":
                has_schema = "schema" in item
                has_schema_file = "schema_file" in item
                if has_schema and has_schema_file:
                    raise PlanValidationError(
                        f"[E020] Judge criterion 'json-schema' must have 'schema' or 'schema_file', not both",
                        code=E020,
                    )
                if not has_schema and not has_schema_file:
                    raise PlanValidationError(
                        f"[E020] Judge criterion 'json-schema' requires 'schema' (dict) or 'schema_file' (str)",
                        code=E020,
                    )
                if has_schema and not isinstance(item["schema"], dict):
                    raise PlanValidationError(
                        f"{field_name}.criteria[{i}].schema must be a dict",
                        code=E020,
                    )
                if has_schema_file and not isinstance(item["schema_file"], str):
                    raise PlanValidationError(
                        f"{field_name}.criteria[{i}].schema_file must be a string",
                        code=E020,
                    )
            if assertion_type == "rubric":
                name = item.get("name")
                if not isinstance(name, str) or not name.strip():
                    raise PlanValidationError(
                        f"{field_name}.criteria[{i}].name is required for rubric and must be a non-empty string",
                        code=E020,
                    )
                levels = item.get("levels")
                if not isinstance(levels, list) or not levels:
                    raise PlanValidationError(
                        f"{field_name}.criteria[{i}].levels is required for rubric and must be a non-empty list",
                        code=E020,
                    )
                for j, level in enumerate(levels):
                    if not isinstance(level, dict):
                        raise PlanValidationError(
                            f"{field_name}.criteria[{i}].levels[{j}] must be an object",
                            code=E020,
                        )
                    if "score" not in level or "description" not in level:
                        raise PlanValidationError(
                            f"{field_name}.criteria[{i}].levels[{j}] must contain 'score' and 'description'",
                            code=E020,
                        )
                    score = level.get("score")
                    if not isinstance(score, int) or score < 1 or score > 5:
                        raise PlanValidationError(
                            f"{field_name}.criteria[{i}].levels[{j}].score must be an integer 1-5",
                            code=E020,
                        )
                    description = level.get("description")
                    if not isinstance(description, str) or not description.strip():
                        raise PlanValidationError(
                            f"{field_name}.criteria[{i}].levels[{j}].description must be a non-empty string",
                            code=E020,
                        )
                min_score_raw = item.get("min_score", 3)
                if not isinstance(min_score_raw, int):
                    raise PlanValidationError(
                        f"{field_name}.criteria[{i}].min_score must be an integer",
                        code=E020,
                    )
                weight_raw = item.get("weight", 1.0)
                if not isinstance(weight_raw, (int, float)):
                    raise PlanValidationError(
                        f"{field_name}.criteria[{i}].weight must be a number",
                        code=E020,
                    )
            _check_unknown_criteria_fields(item, assertion_type, f"{field_name}.criteria[{i}]")
            criteria.append(item)
            continue

        criterion = str(item).strip()
        if not criterion:
            raise PlanValidationError(
                f"{field_name}.criteria entries must be non-empty strings",
                code=E020,
            )
        criteria.append(criterion)

    threshold_raw = value.get("pass_threshold", 0.7)
    if preset is not None and "pass_threshold" not in value:
        threshold_raw = JUDGE_PRESETS[preset]["pass_threshold"]
    try:
        pass_threshold = float(threshold_raw)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(
            f"{field_name}.pass_threshold must be a number",
            code=E020,
        ) from exc
    if pass_threshold < 0 or pass_threshold > 1:
        raise PlanValidationError(
            f"{field_name}.pass_threshold must be between 0 and 1",
            code=E020,
        )

    method = str(value.get("method", "direct"))
    if method not in JUDGE_METHODS:
        raise PlanValidationError(
            f"{field_name}.method '{method}' is not valid. "
            f"Allowed: {sorted(JUDGE_METHODS)}",
            code=E020,
        )

    aggregation = str(value.get("aggregation", "mean"))
    if preset is not None and "aggregation" not in value:
        aggregation = str(JUDGE_PRESETS[preset]["aggregation"])
    if aggregation not in SCORE_AGGREGATIONS:
        raise PlanValidationError(
            f"{field_name}.aggregation '{aggregation}' is not valid. "
            f"Allowed: {sorted(SCORE_AGGREGATIONS)}",
            code=E020,
        )

    on_fail = str(value.get("on_fail", "fail"))
    if on_fail not in JUDGE_ON_FAIL_VALUES:
        raise PlanValidationError(
            f"{field_name}.on_fail '{on_fail}' is not valid. "
            f"Allowed: {sorted(JUDGE_ON_FAIL_VALUES)}",
            code=E020,
        )

    model = str(value.get("model", "haiku")).strip()
    if not model:
        raise PlanValidationError(
            f"{field_name}.model must be a non-empty string",
            code=E020,
        )

    timeout_sec: int | None = None
    timeout_raw = value.get("timeout_sec")
    if timeout_raw is not None:
        try:
            timeout_sec = int(timeout_raw)
        except (TypeError, ValueError) as exc:
            raise PlanValidationError(
                f"{field_name}.timeout_sec must be a positive integer",
                code=E020,
            ) from exc
        if timeout_sec < 10:
            raise PlanValidationError(
                f"{field_name}.timeout_sec must be >= 10 (got {timeout_sec})",
                code=E020,
            )

    quorum: int | None = None
    quorum_raw = value.get("quorum")
    if quorum_raw is not None:
        try:
            quorum = int(quorum_raw)
        except (TypeError, ValueError) as exc:
            raise PlanValidationError(
                f"{field_name}.quorum must be a positive integer >= 2",
                code=E054,
            ) from exc
        if quorum < 2:
            raise PlanValidationError(
                f"{field_name}.quorum must be >= 2 (got {quorum})",
                code=E054,
            )

    quorum_strategy: str | None = None
    quorum_strategy_raw = value.get("quorum_strategy")
    if quorum_strategy_raw is not None:
        quorum_strategy = str(quorum_strategy_raw).strip()
        if quorum_strategy not in QUORUM_STRATEGIES:
            raise PlanValidationError(
                f"{field_name}.quorum_strategy '{quorum_strategy}' is not valid. "
                f"Allowed: {sorted(QUORUM_STRATEGIES)}",
                code=E055,
            )
    if quorum_strategy is not None and quorum is None:
        raise PlanValidationError(
            f"{field_name}.quorum_strategy requires quorum to be set",
            code=E056,
        )

    quorum_diversity = bool(value.get("quorum_diversity", False))

    return JudgeSpec(
        criteria=criteria,
        pass_threshold=pass_threshold,
        on_fail=cast(JudgeOnFail, on_fail),
        model=model,
        method=cast(JudgeMethod, method),
        aggregation=cast(ScoreAggregation, aggregation),
        preset=preset if preset else None,
        timeout_sec=timeout_sec,
        quorum=quorum,
        quorum_strategy=cast(QuorumStrategy | None, quorum_strategy),
        quorum_diversity=quorum_diversity,
        debate_rounds=_parse_debate_rounds(value, field_name),
    )


def _parse_debate_rounds(value: dict[str, Any], field_name: str) -> int:
    """Parse debate_rounds from judge block (default 2, minimum 1)."""
    raw = value.get("debate_rounds")
    if raw is None:
        return 2
    try:
        rounds = int(raw)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(
            f"{field_name}.debate_rounds must be a positive integer >= 1",
            code=E020,
        ) from exc
    if rounds < 1:
        raise PlanValidationError(
            f"{field_name}.debate_rounds must be >= 1 (got {rounds})",
            code=E020,
        )
    return rounds


def _to_float_or_none(value: Any, field_name: str) -> float | None:
    """Parse an optional positive float field."""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(f"{field_name} must be a number", code=E018) from exc
    if result <= 0:
        raise PlanValidationError(f"{field_name} must be > 0", code=E012)
    return result


def _to_pct_or_none(value: Any, field_name: str) -> float | None:
    """Parse an optional percentage float (0.0–1.0 range validated later)."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(f"{field_name} must be a number", code=E018) from exc


def _to_delay_spec(value: Any, field_name: str) -> list[float] | float | None:
    """Parse ``retry_delay_sec``: accepts a number or list of numbers (>= 0)."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if value < 0:
            raise PlanValidationError(f"{field_name} must be >= 0", code=E013)
        return float(value)
    if isinstance(value, list):
        result: list[float] = []
        for i, item in enumerate(value):
            try:
                v = float(item)
            except (TypeError, ValueError) as exc:
                raise PlanValidationError(
                    f"{field_name}[{i}] must be a number", code=E013
                ) from exc
            if v < 0:
                raise PlanValidationError(f"{field_name}[{i}] must be >= 0", code=E013)
            result.append(v)
        return result
    raise PlanValidationError(f"{field_name} must be a number or list of numbers", code=E013)


_RETRY_STRATEGIES = ("constant", "linear", "exponential")


def _parse_retry_strategy(
    task_value: Any, default_value: Any, task_label: str
) -> RetryStrategy | None:
    """Parse ``retry_strategy`` for a task, falling back to plan default."""
    value = task_value if task_value is not None else default_value
    if value is None:
        return None
    if value not in _RETRY_STRATEGIES:
        raise PlanValidationError(
            f"{task_label}: retry_strategy must be constant/linear/exponential, got '{value}'",
            code="E051",
        )
    return cast(RetryStrategy, str(value))


def _parse_context_compaction(
    value: Any, field_name: str
) -> ContextCompaction | None:
    """Parse ``context_compaction`` field (none/standard/progressive)."""
    if value is None:
        return None
    value = str(value)
    if value not in CONTEXT_COMPACTION_VALUES:
        raise PlanValidationError(
            f"{field_name} must be one of {sorted(CONTEXT_COMPACTION_VALUES)}, "
            f"got '{value}'",
            code=E068,
        )
    return cast(ContextCompaction, value)


def _to_trajectory_guard(
    value: Any, field_name: str,
) -> TrajectoryGuardSpec | None:
    """Parse ``trajectory_guard`` block."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PlanValidationError(
            f"{field_name} must be an object", code=E018,
        )
    on_violation = str(value.get("on_violation", "warn"))
    if on_violation not in TRAJECTORY_GUARD_ACTIONS:
        raise PlanValidationError(
            f"{field_name}.on_violation must be one of "
            f"{sorted(TRAJECTORY_GUARD_ACTIONS)}, got '{on_violation}'",
            code=E008,
        )
    max_tool_calls = value.get("max_tool_calls")
    if max_tool_calls is not None:
        max_tool_calls = int(max_tool_calls)
        if max_tool_calls < 1:
            raise PlanValidationError(
                f"{field_name}.max_tool_calls must be >= 1",
                code=E012,
            )
    max_retries = value.get("max_retries_without_progress")
    if max_retries is not None:
        max_retries = int(max_retries)
        if max_retries < 1:
            raise PlanValidationError(
                f"{field_name}.max_retries_without_progress must be >= 1",
                code=E012,
            )
    scope_pattern = value.get("scope_pattern")
    if scope_pattern is not None:
        scope_pattern = str(scope_pattern)
        try:
            re.compile(scope_pattern)
        except re.error as e:
            raise PlanValidationError(
                f"{field_name}.scope_pattern is not a valid regex: {e}",
                code=E008,
            )
    return TrajectoryGuardSpec(
        max_tool_calls=max_tool_calls,
        max_retries_without_progress=max_retries,
        scope_pattern=scope_pattern,
        on_violation=on_violation,  # type: ignore[arg-type]
    )


def _to_population_spec(
    value: Any, field_name: str,
) -> PopulationSpec | None:
    """Parse ``population`` block."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PlanValidationError(
            f"{field_name} must be an object", code=E018,
        )
    candidates = value.get("candidates")
    if not candidates or not isinstance(candidates, list):
        raise PlanValidationError(
            f"{field_name}.candidates must be a non-empty list of model names",
            code=E012,
        )
    candidates = [str(c) for c in candidates]
    if len(candidates) < 2:
        raise PlanValidationError(
            f"{field_name}.candidates must have at least 2 models",
            code=E012,
        )
    strategy = str(value.get("strategy", "best"))
    if strategy not in POPULATION_STRATEGIES:
        raise PlanValidationError(
            f"{field_name}.strategy must be one of "
            f"{sorted(POPULATION_STRATEGIES)}, got '{strategy}'",
            code=E008,
        )
    parallel = bool(value.get("parallel", True))
    return PopulationSpec(
        candidates=candidates,
        strategy=strategy,  # type: ignore[arg-type]
        parallel=parallel,
    )


def _to_council_spec(value: Any, field_name: str) -> Any:
    """Parse ``council`` block into a CouncilSpec.

    Returns ``None`` if *value* is ``None``.
    """
    from .council import CouncilParticipant, CouncilSpec, _VALID_TOPOLOGIES

    if value is None:
        return None
    if not isinstance(value, dict):
        raise PlanValidationError(
            f"{field_name} must be an object", code=E018,
        )
    participants_raw = value.get("participants")
    if not participants_raw or not isinstance(participants_raw, list):
        raise PlanValidationError(
            f"{field_name}.participants must be a non-empty list",
            code=E012,
        )
    if len(participants_raw) < 2:
        raise PlanValidationError(
            f"{field_name}.participants must have at least 2 entries",
            code=E012,
        )
    participants: list[CouncilParticipant] = []
    valid_engines = {"codex", "claude", "gemini", "copilot", "qwen", "ollama", "llama"}
    for i, p in enumerate(participants_raw):
        if not isinstance(p, dict):
            raise PlanValidationError(
                f"{field_name}.participants[{i}] must be an object",
                code=E018,
            )
        eng = str(p.get("engine", ""))
        if eng not in valid_engines:
            raise PlanValidationError(
                f"{field_name}.participants[{i}].engine must be a valid engine, got '{eng}'",
                code=E006,
            )
        participants.append(CouncilParticipant(
            engine=eng,
            model=str(p["model"]) if p.get("model") else None,
            role=str(p.get("role", "")),
        ))
    rounds = int(value.get("rounds", 1))
    if rounds < 1 or rounds > 5:
        raise PlanValidationError(
            f"{field_name}.rounds must be 1-5, got {rounds}",
            code=E012,
        )
    topology = str(value.get("topology", "star"))
    if topology not in _VALID_TOPOLOGIES:
        raise PlanValidationError(
            f"{field_name}.topology must be one of {sorted(_VALID_TOPOLOGIES)}, got '{topology}'",
            code=E008,
        )
    threshold = float(value.get("consensus_threshold", 0.7))
    if not 0.0 < threshold <= 1.0:
        raise PlanValidationError(
            f"{field_name}.consensus_threshold must be (0.0, 1.0], got {threshold}",
            code=E012,
        )

    # Parse connections for graph topology
    connections_raw = value.get("connections")
    connections: dict[str, list[str]] = {}
    if connections_raw is not None:
        if not isinstance(connections_raw, dict):
            raise PlanValidationError(
                f"{field_name}.connections must be a dict mapping role to list of roles",
                code=E072,
            )
        role_names = {p.role for p in participants}
        for key, val in connections_raw.items():
            key_str = str(key)
            if key_str not in role_names:
                raise PlanValidationError(
                    f"{field_name}.connections key '{key_str}' does not match "
                    f"any participant role",
                    code=E072,
                )
            if not isinstance(val, list):
                raise PlanValidationError(
                    f"{field_name}.connections['{key_str}'] must be a list of role names",
                    code=E072,
                )
            targets: list[str] = []
            for t in val:
                t_str = str(t)
                if t_str not in role_names:
                    raise PlanValidationError(
                        f"{field_name}.connections['{key_str}'] references unknown role '{t_str}'",
                        code=E072,
                    )
                targets.append(t_str)
            connections[key_str] = targets

    # Graph topology requires non-empty connections
    if topology == "graph" and not connections:
        raise PlanValidationError(
            f"{field_name}: topology 'graph' requires a non-empty 'connections' map",
            code=E072,
        )

    # Graph topology requires all participants to have non-empty roles
    if topology == "graph":
        for i, p in enumerate(participants):
            if not p.role:
                raise PlanValidationError(
                    f"{field_name}.participants[{i}]: topology 'graph' requires "
                    f"all participants to have a non-empty 'role'",
                    code=E072,
                )

    return CouncilSpec(
        participants=participants,
        rounds=rounds,
        topology=topology,
        consensus_threshold=threshold,
        connections=connections,
    )


def _to_matrix(value: Any, field_name: str) -> dict[str, list[str]] | None:
    """Parse a ``matrix`` block: must be a dict mapping string keys to non-empty lists."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PlanValidationError(f"{field_name} must be an object", code=E018)
    result: dict[str, list[str]] = {}
    for k, v in value.items():
        key = str(k)
        if not isinstance(v, list):
            raise PlanValidationError(f"{field_name}.{key} must be a list", code=E018)
        if not v:
            raise PlanValidationError(f"{field_name}.{key} must be a non-empty list", code=E018)
        result[key] = [str(item) for item in v]
    if not result:
        raise PlanValidationError(f"{field_name} must have at least one dimension", code=E018)
    return result


def _to_batch_spec(value: Any, field_name: str) -> BatchSpec | None:
    """Parse an optional ``batch`` block into a :class:`BatchSpec`."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise PlanValidationError(f"{field_name} batch must be an object", code=E057)

    # items — required, non-empty list of strings
    items_raw = value.get("items")
    if items_raw is None:
        raise PlanValidationError(
            f"{field_name} batch.items is required",
            code=E057,
        )
    if not isinstance(items_raw, list) or not items_raw:
        raise PlanValidationError(
            f"{field_name} batch.items must be a non-empty list",
            code=E057,
        )
    items = [str(i) for i in items_raw]

    # template — required, must contain {{ batch.item }}
    template_raw = value.get("template")
    if template_raw is None:
        raise PlanValidationError(
            f"{field_name} batch.template is required",
            code=E057,
        )
    template = str(template_raw)
    if "{{ batch.item }}" not in template and "{{batch.item}}" not in template:
        raise PlanValidationError(
            f"{field_name} batch.template must contain '{{{{ batch.item }}}}'",
            code=E057,
        )

    # max_per_call — optional, default 5, must be >= 1
    max_per_call_raw = value.get("max_per_call", 5)
    try:
        max_per_call = int(max_per_call_raw)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError(
            f"{field_name} batch.max_per_call must be an integer",
            code=E058,
        ) from exc
    if max_per_call < 1:
        raise PlanValidationError(
            f"{field_name} batch.max_per_call must be >= 1 (got {max_per_call})",
            code=E058,
        )

    return BatchSpec(items=items, template=template, max_per_call=max_per_call)


def _sanitize_id_part(value: str) -> str:
    """Sanitize a matrix key/value for inclusion in a task ID."""
    return re.sub(r"[^a-zA-Z0-9_\-\.]", "_", value)


def _expand_matrix_tasks(tasks: list[TaskSpec]) -> list[TaskSpec]:
    """Expand matrix tasks into individual ``TaskSpec`` instances.

    Tasks without a ``matrix`` pass through unchanged.  Matrix tasks are
    replaced by N expanded tasks (one per Cartesian-product combination).
    Other tasks' ``depends_on`` and ``context_from`` lists are updated to
    reference the expanded IDs instead of the original matrix-task ID.
    """
    # First pass: expand matrix tasks and record the mapping parent → children.
    matrix_expansions: dict[str, list[str]] = {}
    expanded: list[TaskSpec] = []

    for task in tasks:
        if task.matrix is None:
            expanded.append(task)
            continue

        keys = list(task.matrix.keys())
        value_lists = [task.matrix[k] for k in keys]
        child_ids: list[str] = []

        for combo in itertools.product(*value_lists):
            parts = [
                f"{_sanitize_id_part(k)}-{_sanitize_id_part(v)}"
                for k, v in zip(keys, combo)
            ]
            child_id = f"{task.id}.{'.'.join(parts)}"
            matrix_values = {k: v for k, v in zip(keys, combo)}

            child = TaskSpec(
                id=child_id,
                description=task.description,
                depends_on=list(task.depends_on),
                consistency_group=list(task.consistency_group),
                reconcile_after=list(task.reconcile_after),
                engine=task.engine,
                command=task.command,
                shell=task.shell,
                agent=task.agent,
                model=task.model,
                reasoning_effort=task.reasoning_effort,
                args=list(task.args),
                prompt=task.prompt,
                prompt_file=task.prompt_file,
                prompt_md_file=task.prompt_md_file,
                prompt_md_heading=task.prompt_md_heading,
                workdir=task.workdir,
                env=dict(task.env),
                timeout_sec=task.timeout_sec,
                allow_failure=task.allow_failure,
                max_retries=task.max_retries,
                retry_delay_sec=task.retry_delay_sec,
                requires_clean_worktree=task.requires_clean_worktree,
                pre_command=task.pre_command,
                verify_command=task.verify_command,
                context_from=list(task.context_from),
                context_mode=task.context_mode,
                context_budget_tokens=task.context_budget_tokens,
                context_model=task.context_model,
                workspace_index_exclude=list(task.workspace_index_exclude),
                stdout_tail_lines=task.stdout_tail_lines,
                append_system_prompt=task.append_system_prompt,
                edit_policy=task.edit_policy,
                when=task.when,
                cache=task.cache,
                negative_cache_ttl_sec=task.negative_cache_ttl_sec,
                matrix=None,
                matrix_parent=task.id,
                matrix_values=matrix_values,
                group=task.group,
                checkpoint=task.checkpoint,
                guard_command=task.guard_command,
                assertions=[dict(a) for a in task.assertions],
                contract_type=task.contract_type,
                consumes_contracts=list(task.consumes_contracts),
                tags=task.tags[:],
                max_iterations=task.max_iterations,
                requires_approval=task.requires_approval,
                approval_message=task.approval_message,
                escalation=list(task.escalation),
                fallback_engine=task.fallback_engine,
                fallback_model=task.fallback_model,
                judge=(
                    JudgeSpec(
                        criteria=list(task.judge.criteria),
                        pass_threshold=task.judge.pass_threshold,
                        on_fail=task.judge.on_fail,
                        model=task.judge.model,
                        method=task.judge.method,
                        aggregation=task.judge.aggregation,
                        preset=task.judge.preset,
                        quorum=task.judge.quorum,
                        quorum_strategy=task.judge.quorum_strategy,
                    )
                    if task.judge is not None
                    else None
                ),
            )
            child.context_compact = task.context_compact
            child.context_compaction = task.context_compaction
            child.observation_block = task.observation_block
            child.context_trust = task.context_trust
            child.output_schema = task.output_schema
            child_ids.append(child_id)
            expanded.append(child)

        matrix_expansions[task.id] = child_ids

    # Second pass: rewrite depends_on and context_from in all tasks.
    if matrix_expansions:
        for task in expanded:
            new_deps: list[str] = []
            for dep in task.depends_on:
                if dep in matrix_expansions:
                    new_deps.extend(matrix_expansions[dep])
                else:
                    new_deps.append(dep)
            task.depends_on = new_deps

            new_ctx: list[str] = []
            for ctx in task.context_from:
                if ctx in matrix_expansions:
                    new_ctx.extend(matrix_expansions[ctx])
                else:
                    new_ctx.append(ctx)
            task.context_from = new_ctx

    return expanded


_CURRENT_SCHEMA_VERSION = 1
_SUPPORTED_SCHEMA_VERSIONS = {1}
_IMPORT_MAX_DEPTH: int = 5
_IMPORT_PREFIX_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")


def _migrate_plan(raw: dict[str, object], from_version: int) -> dict[str, object]:
    """Apply sequential migrations to bring a plan to the current schema version.

    Raises PlanValidationError if the version is unsupported or too new.
    """
    if from_version > _CURRENT_SCHEMA_VERSION:
        raise PlanValidationError(
            f"Plan uses schema version {from_version}, but this version of Maestro "
            f"only supports up to version {_CURRENT_SCHEMA_VERSION}. "
            f"Please upgrade Maestro CLI.",
            code=E002,
        )

    if from_version == _CURRENT_SCHEMA_VERSION:
        return raw

    # Future migrations go here:
    # if from_version < 2:
    #     raw = _migrate_v1_to_v2(raw)
    #     from_version = 2

    # If we reach here, no migration path exists
    if from_version not in _SUPPORTED_SCHEMA_VERSIONS:
        raise PlanValidationError(
            f"Unsupported schema version: {from_version}. "
            f"Supported versions: {sorted(_SUPPORTED_SCHEMA_VERSIONS)}",
            code=E002,
        )

    return raw


def _resolve_imports(
    raw: dict[str, Any],
    plan_dir: Path,
    seen: set[str] | None = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    """Load and prefix imported task definitions. Returns list of raw task dicts."""
    if seen is None:
        seen = set()

    imports = raw.get("imports", [])
    if imports is None:
        imports = []
    if not isinstance(imports, list):
        raise PlanValidationError("'imports' must be a list", code=E026)
    if not imports:
        return []
    if depth > _IMPORT_MAX_DEPTH:
        raise PlanValidationError(f"Import depth exceeds {_IMPORT_MAX_DEPTH}", code=E025)

    imported_tasks: list[dict[str, Any]] = []
    seen_prefixes: set[str] = set()

    for imp in imports:
        if not isinstance(imp, dict) or "path" not in imp or "prefix" not in imp:
            raise PlanValidationError(
                "Each import must have 'path' and 'prefix' fields",
                code=E026,
            )
        prefix = str(imp["prefix"])
        if not _IMPORT_PREFIX_RE.match(prefix):
            raise PlanValidationError(
                f"Import prefix '{prefix}' must match [a-z0-9][a-z0-9-]*",
                code=E028,
            )
        if prefix in seen_prefixes:
            raise PlanValidationError(f"Duplicate import prefix: '{prefix}'", code=E027)
        seen_prefixes.add(prefix)

        imp_path = (plan_dir / str(imp["path"])).resolve()
        path_key = str(imp_path)
        if path_key in seen:
            raise PlanValidationError(f"Circular import detected: {imp_path}", code=E025)
        seen.add(path_key)

        if not imp_path.is_file():
            raise PlanValidationError(f"Imported file not found: {imp_path}", code=E026)

        try:
            fragment = yaml.safe_load(imp_path.read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise PlanValidationError(
                f"Invalid YAML in imported file '{imp_path}': {exc}",
                code=E026,
            ) from exc

        if not isinstance(fragment, dict):
            raise PlanValidationError(
                f"Imported file must contain an object root: {imp_path}",
                code=E026,
            )

        fragment_tasks = fragment.get("tasks")
        if not isinstance(fragment_tasks, list):
            raise PlanValidationError(
                f"Imported file must contain a 'tasks' list: {imp_path}",
                code=E026,
            )

        nested = _resolve_imports(fragment, imp_path.parent, seen, depth + 1)
        imported_tasks.extend(nested)

        overrides = imp.get("overrides", {})
        if overrides is None:
            overrides = {}
        if not isinstance(overrides, dict):
            raise PlanValidationError(
                f"Import overrides must be an object: {imp_path}",
                code=E026,
            )

        for task_index, task_raw in enumerate(fragment_tasks):
            if not isinstance(task_raw, dict):
                raise PlanValidationError(
                    f"Imported task at index {task_index} must be an object: {imp_path}",
                    code=E026,
                )
            if "id" not in task_raw:
                raise PlanValidationError(
                    f"Imported task at index {task_index} missing 'id': {imp_path}",
                    code=E026,
                )
            base_task_id = str(task_raw["id"]).strip()
            if not base_task_id:
                raise PlanValidationError(
                    f"Imported task at index {task_index} has empty 'id': {imp_path}",
                    code=E026,
                )

            task = dict(task_raw)
            task["id"] = f"{prefix}/{base_task_id}"

            if "depends_on" in task:
                deps = task["depends_on"]
                if isinstance(deps, str):
                    deps = [deps]
                if not isinstance(deps, list):
                    raise PlanValidationError(
                        f"Imported task '{base_task_id}' has invalid depends_on in {imp_path}",
                        code=E026,
                    )
                task["depends_on"] = [
                    f"{prefix}/{str(d)}" if "/" not in str(d) else str(d) for d in deps
                ]

            if "context_from" in task:
                ctx = task["context_from"]
                if isinstance(ctx, str):
                    ctx = [ctx]
                if not isinstance(ctx, list):
                    raise PlanValidationError(
                        f"Imported task '{base_task_id}' has invalid context_from in {imp_path}",
                        code=E026,
                    )
                task["context_from"] = [
                    f"{prefix}/{str(c)}"
                    if str(c) != "*" and "/" not in str(c)
                    else str(c)
                    for c in ctx
                ]

            for key, val in overrides.items():
                if key == "env" and isinstance(val, dict):
                    env_current = task.get("env")
                    if env_current is None:
                        env_current = {}
                    if not isinstance(env_current, dict):
                        raise PlanValidationError(
                            f"Imported task '{base_task_id}' has non-object env in {imp_path}",
                            code=E026,
                        )
                    env_current.update(val)
                    task["env"] = env_current
                else:
                    task[key] = val

            imported_tasks.append(task)

    return imported_tasks


def load_plan(path: str | Path) -> PlanSpec:
    plan_path = Path(path).resolve()
    if not plan_path.exists():
        raise PlanValidationError(f"Plan file not found: {plan_path}")

    try:
        raw = yaml.safe_load(plan_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PlanValidationError(f"Invalid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise PlanValidationError("Plan root must be an object", code=E018)

    raw_version = int(raw.get("version", 1))
    raw = _migrate_plan(raw, raw_version)
    plan_dir = plan_path.parent.resolve()
    imported_raw_tasks = _resolve_imports(raw, plan_dir)
    main_tasks_raw = raw.get("tasks", [])
    if main_tasks_raw is None:
        main_tasks_raw = []
    if not isinstance(main_tasks_raw, list):
        raise PlanValidationError("tasks must be a list", code=E018)
    raw_tasks = imported_raw_tasks + main_tasks_raw

    raw_secrets = raw.get("secrets", [])
    plan_secrets: list[str] = []
    plan_secrets_auto = False
    if raw_secrets == "auto":
        plan_secrets_auto = True
    elif isinstance(raw_secrets, list):
        plan_secrets = _to_str_list(raw_secrets, "secrets")
    elif raw_secrets is not None:
        raise PlanValidationError(
            f"'secrets' must be a list of variable names or 'auto', got: {type(raw_secrets).__name__}",
            code=E024,
        )

    defaults_raw = raw.get("defaults", {}) or {}
    if not isinstance(defaults_raw, dict):
        raise PlanValidationError("defaults must be an object", code=E018)

    _default_retry_strategy = defaults_raw.get("retry_strategy")
    if _default_retry_strategy is not None and _default_retry_strategy not in (
        "constant",
        "linear",
        "exponential",
    ):
        raise PlanValidationError(
            f"defaults.retry_strategy must be constant/linear/exponential, got '{_default_retry_strategy}'",
            code="E051",
        )

    defaults_secrets = _to_str_list(defaults_raw.get("secrets"), "defaults.secrets")
    defaults_secrets_auto_raw = defaults_raw.get("secrets_auto", False)
    if not isinstance(defaults_secrets_auto_raw, bool):
        raise PlanValidationError("defaults.secrets_auto must be a boolean", code=E018)
    defaults_secrets_auto = defaults_secrets_auto_raw

    codex_raw = defaults_raw.get("codex", {}) or {}
    claude_raw = defaults_raw.get("claude", {}) or {}
    gemini_raw = defaults_raw.get("gemini", {}) or {}
    copilot_raw = defaults_raw.get("copilot", {}) or {}
    qwen_raw = defaults_raw.get("qwen", {}) or {}
    ollama_raw = defaults_raw.get("ollama", {}) or {}
    llama_raw = defaults_raw.get("llama", {}) or {}
    if (
        not isinstance(codex_raw, dict)
        or not isinstance(claude_raw, dict)
        or not isinstance(gemini_raw, dict)
        or not isinstance(copilot_raw, dict)
        or not isinstance(qwen_raw, dict)
        or not isinstance(ollama_raw, dict)
        or not isinstance(llama_raw, dict)
    ):
        raise PlanValidationError(
            "defaults.codex, defaults.claude, defaults.gemini, defaults.copilot, defaults.qwen, defaults.ollama and defaults.llama must be objects",
            code=E018,
        )

    defaults = PlanDefaults(
        env=_to_str_dict(defaults_raw.get("env"), "defaults.env"),
        secrets=defaults_secrets,
        secrets_auto=defaults_secrets_auto,
        timeout_sec=_to_int_or_none(defaults_raw.get("timeout_sec"), "defaults.timeout_sec"),
        requires_clean_worktree=bool(defaults_raw.get("requires_clean_worktree", False)),
        stdout_tail_lines=int(defaults_raw.get("stdout_tail_lines", 50)),
        edit_policy=cast(EditPolicy, str(defaults_raw.get("edit_policy", "default"))),
        retry_delay_sec=_to_delay_spec(
            defaults_raw.get("retry_delay_sec"), "defaults.retry_delay_sec"
        ),
        budget_warning_pct=_to_pct_or_none(
            defaults_raw.get("budget_warning_pct"), "defaults.budget_warning_pct"
        ),
        context_budget_tokens=_to_context_budget_or_none(
            defaults_raw.get("context_budget_tokens"),
            "defaults.context_budget_tokens",
        ),
        workspace_index_exclude=_to_str_list(
            defaults_raw.get("workspace_index_exclude"),
            "defaults.workspace_index_exclude",
        ),
        signals=bool(defaults_raw.get("signals", False)),
        context_compaction=_parse_context_compaction(
            defaults_raw.get("context_compaction"), "defaults.context_compaction"
        ),
        codex=EngineDefaults(
            model=str(codex_raw["model"]) if codex_raw.get("model") is not None else None,
            reasoning_effort=(
                str(codex_raw["reasoning_effort"]) if codex_raw.get("reasoning_effort") is not None else None
            ),
            args=_to_str_list(codex_raw.get("args"), "defaults.codex.args"),
            append_system_prompt=(
                str(codex_raw["append_system_prompt"]) if codex_raw.get("append_system_prompt") is not None else None
            ),
        ),
        claude=EngineDefaults(
            model=str(claude_raw["model"]) if claude_raw.get("model") is not None else None,
            reasoning_effort=(
                str(claude_raw["reasoning_effort"]) if claude_raw.get("reasoning_effort") is not None else None
            ),
            args=_to_str_list(claude_raw.get("args"), "defaults.claude.args"),
            append_system_prompt=(
                str(claude_raw["append_system_prompt"]) if claude_raw.get("append_system_prompt") is not None else None
            ),
        ),
        gemini=EngineDefaults(
            model=str(gemini_raw["model"]) if gemini_raw.get("model") is not None else None,
            reasoning_effort=(
                str(gemini_raw["reasoning_effort"]) if gemini_raw.get("reasoning_effort") is not None else None
            ),
            args=_to_str_list(gemini_raw.get("args"), "defaults.gemini.args"),
            append_system_prompt=(
                str(gemini_raw["append_system_prompt"]) if gemini_raw.get("append_system_prompt") is not None else None
            ),
        ),
        copilot=EngineDefaults(
            model=str(copilot_raw["model"]) if copilot_raw.get("model") is not None else None,
            reasoning_effort=(
                str(copilot_raw["reasoning_effort"]) if copilot_raw.get("reasoning_effort") is not None else None
            ),
            args=_to_str_list(copilot_raw.get("args"), "defaults.copilot.args"),
            append_system_prompt=(
                str(copilot_raw["append_system_prompt"]) if copilot_raw.get("append_system_prompt") is not None else None
            ),
        ),
        qwen=EngineDefaults(
            model=str(qwen_raw["model"]) if qwen_raw.get("model") is not None else None,
            reasoning_effort=(
                str(qwen_raw["reasoning_effort"]) if qwen_raw.get("reasoning_effort") is not None else None
            ),
            args=_to_str_list(qwen_raw.get("args"), "defaults.qwen.args"),
            append_system_prompt=(
                str(qwen_raw["append_system_prompt"]) if qwen_raw.get("append_system_prompt") is not None else None
            ),
        ),
        ollama=EngineDefaults(
            model=str(ollama_raw["model"]) if ollama_raw.get("model") is not None else None,
            reasoning_effort=(
                str(ollama_raw["reasoning_effort"]) if ollama_raw.get("reasoning_effort") is not None else None
            ),
            args=_to_str_list(ollama_raw.get("args"), "defaults.ollama.args"),
            append_system_prompt=(
                str(ollama_raw["append_system_prompt"]) if ollama_raw.get("append_system_prompt") is not None else None
            ),
        ),
        llama=EngineDefaults(
            model=str(llama_raw["model"]) if llama_raw.get("model") is not None else None,
            reasoning_effort=(
                str(llama_raw["reasoning_effort"]) if llama_raw.get("reasoning_effort") is not None else None
            ),
            args=_to_str_list(llama_raw.get("args"), "defaults.llama.args"),
            append_system_prompt=(
                str(llama_raw["append_system_prompt"]) if llama_raw.get("append_system_prompt") is not None else None
            ),
        ),
    )
    _set_engine_resilience_defaults(defaults.codex, codex_raw, "defaults.codex")
    _set_engine_resilience_defaults(defaults.claude, claude_raw, "defaults.claude")
    _set_engine_resilience_defaults(defaults.gemini, gemini_raw, "defaults.gemini")
    _set_engine_resilience_defaults(defaults.copilot, copilot_raw, "defaults.copilot")
    _set_engine_resilience_defaults(defaults.qwen, qwen_raw, "defaults.qwen")
    _set_engine_resilience_defaults(defaults.ollama, ollama_raw, "defaults.ollama")
    _set_engine_resilience_defaults(defaults.llama, llama_raw, "defaults.llama")
    audit_packs = _to_str_list(raw.get("audit_packs"), "audit_packs")

    if not raw_tasks:
        raise PlanValidationError("tasks must be a non-empty list", code=E001)

    tasks: list[TaskSpec] = []
    for idx, item in enumerate(raw_tasks):
        if not isinstance(item, dict):
            raise PlanValidationError(f"tasks[{idx}] must be an object", code=E018)

        task_id = str(item.get("id", "")).strip()
        if not task_id:
            raise PlanValidationError(f"tasks[{idx}].id is required", code=E001)

        task = TaskSpec(
            id=task_id,
            description=str(item.get("description", "") or ""),
            depends_on=_to_str_list(item.get("depends_on"), f"tasks[{idx}].depends_on"),
            consistency_group=_to_str_list(
                item.get("consistency_group"),
                f"tasks[{idx}].consistency_group",
            ),
            reconcile_after=_to_str_list(
                item.get("reconcile_after"),
                f"tasks[{idx}].reconcile_after",
            ),
            engine=cast(EngineName, str(item["engine"])) if item.get("engine") is not None else None,
            command=item.get("command"),
            shell=bool(item["shell"]) if "shell" in item else None,
            agent=str(item["agent"]) if item.get("agent") is not None else None,
            model=str(item["model"]) if item.get("model") is not None else None,
            reasoning_effort=(
                str(item["reasoning_effort"]) if item.get("reasoning_effort") is not None else None
            ),
            args=_to_str_list(item.get("args"), f"tasks[{idx}].args"),
            prompt=str(item["prompt"]) if item.get("prompt") is not None else None,
            prompt_file=str(item["prompt_file"]) if item.get("prompt_file") is not None else None,
            prompt_md_file=str(item["prompt_md_file"]) if item.get("prompt_md_file") is not None else None,
            prompt_md_heading=str(item["prompt_md_heading"]) if item.get("prompt_md_heading") is not None else None,
            workdir=str(item["workdir"]) if item.get("workdir") is not None else None,
            env=_to_str_dict(item.get("env"), f"tasks[{idx}].env"),
            timeout_sec=_to_int_or_none(item.get("timeout_sec"), f"tasks[{idx}].timeout_sec"),
            allow_failure=bool(item.get("allow_failure", False)),
            max_retries=int(item.get("max_retries", 0)),
            retry_delay_sec=_to_delay_spec(
                item.get("retry_delay_sec"), f"tasks[{idx}].retry_delay_sec"
            ),
            requires_clean_worktree=(
                bool(item["requires_clean_worktree"]) if "requires_clean_worktree" in item else None
            ),
            worktree=bool(item.get("worktree", False)),
            pre_command=item.get("pre_command"),
            verify_command=item.get("verify_command"),
            guard_command=item.get("guard_command"),
            assertions=_to_workspace_assertions(item.get("assert"), f"tasks[{idx}].assert"),
            contract_type=_to_contract_type_or_none(
                item.get("contract_type"),
                f"tasks[{idx}].contract_type",
            ),
            consumes_contracts=_to_str_list(
                item.get("consumes_contracts"),
                f"tasks[{idx}].consumes_contracts",
            ),
            max_iterations=_to_int_or_none(
                item.get("max_iterations"), f"tasks[{idx}].max_iterations"
            ),
            escalation=_to_str_list(item.get("escalation"), f"tasks[{idx}].escalation"),
            fallback_engine=(
                cast(EngineName, str(item["fallback_engine"])) if item.get("fallback_engine") is not None else None
            ),
            fallback_model=(
                str(item["fallback_model"]) if item.get("fallback_model") is not None else None
            ),
            context_from=_to_str_list(item.get("context_from"), f"tasks[{idx}].context_from"),
            context_mode=cast(ContextMode, str(item.get("context_mode", "raw"))),
            context_budget_tokens=_to_context_budget_or_none(
                item.get("context_budget_tokens"),
                f"tasks[{idx}].context_budget_tokens",
            ),
            context_model=(
                str(item["context_model"]) if item.get("context_model") is not None else None
            ),
            workspace_index_exclude=_to_str_list(
                item.get("workspace_index_exclude"), f"tasks[{idx}].workspace_index_exclude"
            ),
            stdout_tail_lines=_to_int_or_none(
                item.get("stdout_tail_lines"), f"tasks[{idx}].stdout_tail_lines"
            ),
            append_system_prompt=(
                str(item["append_system_prompt"]) if item.get("append_system_prompt") is not None else None
            ),
            edit_policy=(
                cast(EditPolicy, str(item["edit_policy"])) if item.get("edit_policy") is not None else None
            ),
            when=str(item["when"]) if item.get("when") is not None else None,
            cache=bool(item.get("cache", True)),
            negative_cache_ttl_sec=_to_int_or_none(
                item.get("negative_cache_ttl_sec"),
                f"tasks[{idx}].negative_cache_ttl_sec",
            ),
            matrix=_to_matrix(item.get("matrix"), f"tasks[{idx}].matrix"),
            batch=_to_batch_spec(item.get("batch"), f"tasks[{idx}]"),
            group=str(item["group"]) if item.get("group") is not None else None,
            checkpoint=bool(item.get("checkpoint", False)),
            judge=_to_judge_spec(item.get("judge"), f"tasks[{idx}].judge"),
            tags=_to_str_list(item.get("tags", []), f"tasks[{idx}].tags"),
            requires_approval=bool(item.get("requires_approval", False)),
            approval_message=str(item["approval_message"]) if item.get("approval_message") is not None else None,
            retry_strategy=_parse_retry_strategy(item.get("retry_strategy"), _default_retry_strategy, f"tasks[{idx}]"),
        )
        task.context_compact = bool(item.get("context_compact", False))
        _ctx_compaction = item.get("context_compaction")
        if _ctx_compaction is not None:
            _ctx_compaction = str(_ctx_compaction)
            if _ctx_compaction not in CONTEXT_COMPACTION_VALUES:
                raise PlanValidationError(
                    f"tasks[{idx}].context_compaction must be one of "
                    f"{sorted(CONTEXT_COMPACTION_VALUES)}, got '{_ctx_compaction}'",
                    code="E068",
                )
            task.context_compaction = _ctx_compaction  # type: ignore[assignment]
        elif task.context_compact:
            task.context_compaction = "standard"
        task.observation_block = bool(item.get("observation_block", False))
        if task.negative_cache_ttl_sec is not None and task.negative_cache_ttl_sec < 0:
            raise PlanValidationError(
                f"tasks[{idx}].negative_cache_ttl_sec must be >= 0 "
                f"(got {task.negative_cache_ttl_sec})",
                code=E013,
            )
        _ctx_trust = item.get("context_trust")
        if _ctx_trust is not None:
            _ctx_trust = str(_ctx_trust)
            if _ctx_trust not in CONTEXT_TRUST_VALUES:
                raise PlanValidationError(
                    f"tasks[{idx}].context_trust must be 'trusted' or 'untrusted', "
                    f"got '{_ctx_trust}'",
                    code="E065",
                )
            task.context_trust = _ctx_trust  # type: ignore[assignment]
        task.frozen = bool(item.get("frozen", False))
        task.compress_before = bool(item.get("compress_before", False))
        task.honeypot = bool(item.get("honeypot", False))
        _output_scope_raw = item.get("output_scope")
        if _output_scope_raw is not None:
            if isinstance(_output_scope_raw, list):
                task.output_scope = [str(s) for s in _output_scope_raw]
            elif isinstance(_output_scope_raw, str):
                task.output_scope = [_output_scope_raw]
        task.signals = bool(item.get("signals", False))
        task.deliberation = bool(item.get("deliberation", False))
        _output_schema = item.get("output_schema")
        if _output_schema is not None:
            if not isinstance(_output_schema, dict):
                raise PlanValidationError(
                    f"tasks[{idx}].output_schema must be an object (JSON Schema dict)",
                    code=E018,
                )
            task.output_schema = _output_schema
        _delib_thresh = item.get("deliberation_threshold")
        if _delib_thresh is not None:
            try:
                task.deliberation_threshold = float(_delib_thresh)
            except (TypeError, ValueError):
                raise PlanValidationError(
                    f"tasks[{idx}].deliberation_threshold must be a number between 0.0 and 1.0",
                    code=E008,
                )
            if not 0.0 <= task.deliberation_threshold <= 1.0:
                raise PlanValidationError(
                    f"tasks[{idx}].deliberation_threshold must be between 0.0 and 1.0 "
                    f"(got {task.deliberation_threshold})",
                    code=E008,
                )

        task.dynamic_group = bool(item.get("dynamic_group", False))

        # v1.24.0 -- Event-Driven System Reminders
        _reminders_raw = item.get("reminders")
        if _reminders_raw is not None:
            if not isinstance(_reminders_raw, list):
                raise PlanValidationError(
                    f"tasks[{idx}].reminders must be a list of objects",
                    code=E018,
                )
            for ri, rem in enumerate(_reminders_raw):
                if not isinstance(rem, dict):
                    raise PlanValidationError(
                        f"tasks[{idx}].reminders[{ri}] must be an object "
                        f"with 'trigger' and 'message' keys",
                        code=E018,
                    )
                if "trigger" not in rem or "message" not in rem:
                    raise PlanValidationError(
                        f"tasks[{idx}].reminders[{ri}] must have both "
                        f"'trigger' and 'message' keys",
                        code=E067,
                    )
                _trig = str(rem["trigger"]).strip()
                _msg = str(rem["message"]).strip()
                if not _trig:
                    raise PlanValidationError(
                        f"tasks[{idx}].reminders[{ri}].trigger must be a "
                        f"non-empty string",
                        code=E067,
                    )
                if not _msg:
                    raise PlanValidationError(
                        f"tasks[{idx}].reminders[{ri}].message must be a "
                        f"non-empty string",
                        code=E067,
                    )
            task.reminders = [
                {"trigger": str(r["trigger"]).strip(), "message": str(r["message"]).strip()}
                for r in _reminders_raw
            ]

        # v1.26.0 -- Privacy-Aware Context Pipeline
        _output_redact_raw = item.get("output_redact")
        if _output_redact_raw is not None:
            if isinstance(_output_redact_raw, list):
                task.output_redact = [str(p) for p in _output_redact_raw]
            elif isinstance(_output_redact_raw, str):
                task.output_redact = [_output_redact_raw]
            else:
                raise PlanValidationError(
                    f"tasks[{idx}].output_redact must be a list of regex patterns",
                    code=E018,
                )
            for pi, pat in enumerate(task.output_redact):
                try:
                    re.compile(pat)
                except re.error as e:
                    raise PlanValidationError(
                        f"tasks[{idx}].output_redact[{pi}] is not a valid regex: {e}",
                        code=E008,
                    )
        _ctx_allowlist_raw = item.get("context_allowlist")
        if _ctx_allowlist_raw is not None:
            task.context_allowlist = _to_str_list(
                _ctx_allowlist_raw, f"tasks[{idx}].context_allowlist"
            )

        # v1.26.0 -- Trajectory-Level Guardrails
        task.trajectory_guard = _to_trajectory_guard(
            item.get("trajectory_guard"), f"tasks[{idx}].trajectory_guard"
        )

        # v1.26.0 -- Phantom Output Interception
        task.phantom_workspace = bool(item.get("phantom_workspace", False))

        # v1.28.0 -- Population-Based Search
        task.population = _to_population_spec(
            item.get("population"), f"tasks[{idx}].population"
        )

        # v1.29.0 -- MCP-Native Tool Orchestration
        task.mcp_tools = _to_str_list(
            item.get("mcp_tools", []), f"tasks[{idx}].mcp_tools"
        )

        # v2.0 -- Capability-Based Tool Access
        _allowed_tools_raw = item.get("allowed_tools")
        if _allowed_tools_raw is not None:
            task.allowed_tools = _to_str_list(
                _allowed_tools_raw, f"tasks[{idx}].allowed_tools"
            )

        # v1.36.0 -- Council Mode
        task.council = _to_council_spec(
            item.get("council"), f"tasks[{idx}].council"
        )

        if task.command is not None:
            if not isinstance(task.command, (str, list)):
                raise PlanValidationError(
                    f"tasks[{idx}].command must be a string or list of strings",
                    code=E018,
                )
            if isinstance(task.command, list) and not all(
                isinstance(cmd_part, str) for cmd_part in task.command
            ):
                raise PlanValidationError(
                    f"tasks[{idx}].command list must contain only strings",
                    code=E018,
                )

        if task.pre_command is not None:
            if not isinstance(task.pre_command, (str, list)):
                raise PlanValidationError(
                    f"tasks[{idx}].pre_command must be a string or list of strings",
                    code=E018,
                )
            if isinstance(task.pre_command, list) and not all(
                isinstance(cmd_part, str) for cmd_part in task.pre_command
            ):
                raise PlanValidationError(
                    f"tasks[{idx}].pre_command list must contain only strings",
                    code=E018,
                )

        if task.verify_command is not None:
            if not isinstance(task.verify_command, (str, list)):
                raise PlanValidationError(
                    f"tasks[{idx}].verify_command must be a string or list of strings",
                    code=E018,
                )
            if isinstance(task.verify_command, list) and not all(
                isinstance(cmd_part, str) for cmd_part in task.verify_command
            ):
                raise PlanValidationError(
                    f"tasks[{idx}].verify_command list must contain only strings",
                    code=E018,
                )

        if task.guard_command is not None:
            if not isinstance(task.guard_command, (str, list)):
                raise PlanValidationError(
                    f"tasks[{idx}].guard_command must be a string or list of strings",
                    code=E018,
                )
            if isinstance(task.guard_command, list) and not all(
                isinstance(cmd_part, str) for cmd_part in task.guard_command
            ):
                raise PlanValidationError(
                    f"tasks[{idx}].guard_command list must contain only strings",
                    code=E018,
                )

        tasks.append(task)

    for task in tasks:
        if task.engine == "codex":
            engine_defaults = defaults.codex
        elif task.engine == "claude":
            engine_defaults = defaults.claude
        elif task.engine == "gemini":
            engine_defaults = defaults.gemini
        elif task.engine == "copilot":
            engine_defaults = defaults.copilot
        elif task.engine == "qwen":
            engine_defaults = defaults.qwen
        elif task.engine == "ollama":
            engine_defaults = defaults.ollama
        elif task.engine == "llama":
            engine_defaults = defaults.llama
        else:
            continue

        if not task.escalation:
            task.escalation = list(getattr(engine_defaults, "escalation", []))
        if task.fallback_engine is None:
            task.fallback_engine = getattr(engine_defaults, "fallback_engine", None)
        if task.fallback_model is None:
            task.fallback_model = getattr(engine_defaults, "fallback_model", None)
        if task.allowed_tools is None and engine_defaults is not None:
            _eng_allowed = getattr(engine_defaults, "allowed_tools", None)
            if _eng_allowed is not None:
                task.allowed_tools = list(_eng_allowed)

    tasks = clone_tasks_with_resolved_dependencies(_expand_matrix_tasks(tasks))

    # -- Parse optional watch block --
    raw_watch = raw.get("watch")
    watch_spec: WatchSpec | None = None
    if raw_watch is not None:
        if not isinstance(raw_watch, dict):
            raise PlanValidationError("'watch' must be a mapping", code=E032)

        # Parse mode first — it controls defaults for other fields
        watch_mode = str(raw_watch.get("mode", "custom"))
        valid_modes = {"custom", "improve"}
        if watch_mode not in valid_modes:
            raise PlanValidationError(
                f"watch.mode '{watch_mode}' is not valid. "
                f"Allowed: {sorted(valid_modes)}",
                code=E048,
            )

        # mode: improve auto-sets defaults when not explicitly provided
        is_improve = watch_mode == "improve"
        metric = raw_watch.get("metric")
        if is_improve and not metric:
            metric = "tasks_passed"
        if not metric or not isinstance(metric, str):
            raise PlanValidationError(
                "'watch.metric' is required and must be a non-empty string",
                code=E032,
            )

        improve_defaults: dict[str, object] = {}
        if is_improve:
            improve_defaults = {
                "metric_direction": "higher_is_better",
                "metric_source": "manifest",
                "warmup_iterations": 0,
                "on_regression": "rollback",
                "plateau_threshold": 3,
                "plateau_action": "stop",
            }

        def _wg(key: str, default: object) -> object:
            """Get from raw_watch, falling back to improve defaults then default."""
            if key in raw_watch:
                return raw_watch[key]
            return improve_defaults.get(key, default)

        watch_spec = WatchSpec(
            metric=metric,
            max_iterations=int(raw_watch.get("max_iterations", 100)),
            iteration_budget_sec=int(raw_watch["iteration_budget_sec"])
            if raw_watch.get("iteration_budget_sec") is not None
            else None,
            metric_direction=cast(MetricDirection, str(_wg("metric_direction", "lower_is_better"))),
            metric_source=cast(MetricSource, str(_wg("metric_source", "stdout_regex"))),
            metric_pattern=str(raw_watch["metric_pattern"])
            if raw_watch.get("metric_pattern") is not None
            else None,
            metric_task=str(raw_watch["metric_task"])
            if raw_watch.get("metric_task") is not None
            else None,
            metric_json_path=str(raw_watch["metric_json_path"])
            if raw_watch.get("metric_json_path") is not None
            else None,
            on_regression=cast(OnRegression, str(_wg("on_regression", "rollback"))),
            program_md=str(raw_watch["program_md"])
            if raw_watch.get("program_md") is not None
            else None,
            warmup_iterations=int(cast(int | str, _wg("warmup_iterations", 1))),
            plateau_threshold=int(cast(int | str, _wg("plateau_threshold", 5))),
            plateau_action=cast(PlateauAction, str(_wg("plateau_action", "stop"))),
            max_cost_usd=float(raw_watch["max_cost_usd"])
            if raw_watch.get("max_cost_usd") is not None
            else None,
            consolidate_model=str(raw_watch["consolidate_model"])
            if raw_watch.get("consolidate_model") is not None
            else None,
            consolidate_every=int(raw_watch.get("consolidate_every", 3)),
            consolidate_prompt=str(raw_watch["consolidate_prompt"])
            if raw_watch.get("consolidate_prompt") is not None
            else None,
            target_metric=float(raw_watch["target_metric"])
            if raw_watch.get("target_metric") is not None
            else None,
            blame_plan=str(raw_watch["blame_plan"])
            if raw_watch.get("blame_plan") is not None
            else None,
            mode=cast(WatchMode, watch_mode),
            improve_model=str(raw_watch["improve_model"])
            if raw_watch.get("improve_model") is not None
            else None,
            max_total_steps=int(raw_watch["max_total_steps"])
            if raw_watch.get("max_total_steps") is not None
            else None,
            stepping_stones=bool(raw_watch.get("stepping_stones", False)),
        )

    circuit_breaker: CircuitBreakerSpec | None = None
    cb_raw = raw.get("circuit_breaker")
    if cb_raw is not None:
        if not isinstance(cb_raw, dict):
            raise PlanValidationError(
                "circuit_breaker must be a mapping", code="E050"
            )
        max_fail = cb_raw.get("max_total_failures", 5)
        if not isinstance(max_fail, int) or max_fail < 1:
            raise PlanValidationError(
                "circuit_breaker.max_total_failures must be a positive integer",
                code="E050",
            )
        action = cb_raw.get("action", "fail")
        if action not in ("pause", "fail"):
            raise PlanValidationError(
                f"circuit_breaker.action must be 'pause' or 'fail', got '{action}'",
                code="E050",
            )
        circuit_breaker = CircuitBreakerSpec(max_total_failures=max_fail, action=action)

    plan = PlanSpec(
        version=int(cast(int | str, raw.get("version", 1))),
        name=str(raw.get("name", "")).strip() or "unnamed-plan",
        goal=str(raw.get("goal", "")),
        firewall_model=(
            str(raw["firewall_model"]).strip()
            if raw.get("firewall_model") is not None and str(raw.get("firewall_model", "")).strip()
            else None
        ),
        webhook_url=(
            str(raw["webhook_url"]) if raw.get("webhook_url") is not None else None
        ),
        secrets=list(dict.fromkeys([*defaults_secrets, *plan_secrets])),
        secrets_auto=(defaults_secrets_auto or plan_secrets_auto),
        workspace_root=str(raw["workspace_root"]) if raw.get("workspace_root") is not None else None,
        max_parallel=int(cast(int | str, raw.get("max_parallel", 1))),
        fail_fast=bool(raw.get("fail_fast", True)),
        run_dir=str(raw.get("run_dir", ".maestro-runs")),
        max_cost_usd=_to_float_or_none(raw.get("max_cost_usd"), "max_cost_usd"),
        budget_warning_pct=_to_pct_or_none(raw.get("budget_warning_pct"), "budget_warning_pct"),
        defaults=defaults,
        tasks=tasks,
        audit_packs=audit_packs,
        source_path=plan_path,
        watch=watch_spec,
        circuit_breaker=circuit_breaker,
        budget_period=str(raw["budget_period"]) if raw.get("budget_period") else None,
    )
    raw_imports: list[Any] = cast(list[Any], raw.get("imports") or [])
    plan.imports = [
        PlanImport(
            path=str(imp["path"]),
            prefix=str(imp["prefix"]),
            overrides=imp.get("overrides", {}),
        )
        for imp in raw_imports
    ]

    raw_policies: list[Any] = cast(list[Any], raw.get("policies", []) or [])
    policies: list[PolicySpec] = []
    for rp in raw_policies:
        if not isinstance(rp, dict):
            continue
        policies.append(PolicySpec(
            name=str(rp.get("name", "")),
            rule=str(rp.get("rule", "")),
            action=cast(PolicyAction, str(rp.get("action", "warn"))),
            message=str(rp.get("message", "")),
        ))
    plan.policies = policies

    # v1.29.0 -- MCP-Native Tool Orchestration (plan-level servers)
    raw_mcp_servers: list[Any] = cast(list[Any], raw.get("mcp_servers", []) or [])
    mcp_servers: list[MCPServerSpec] = []
    for ms_idx, ms in enumerate(raw_mcp_servers):
        if not isinstance(ms, dict):
            raise PlanValidationError(
                f"mcp_servers[{ms_idx}] must be an object",
                code=E069,
            )
        ms_name = str(ms.get("name", "")).strip()
        if not ms_name:
            raise PlanValidationError(
                f"mcp_servers[{ms_idx}].name is required and must be non-empty",
                code=E069,
            )
        ms_command = ms.get("command")
        ms_url = ms.get("url")
        ms_transport = str(ms.get("transport", "stdio"))
        if ms_transport not in MCP_TRANSPORTS:
            raise PlanValidationError(
                f"mcp_servers[{ms_idx}].transport must be one of "
                f"{sorted(MCP_TRANSPORTS)}, got '{ms_transport}'",
                code=E069,
            )
        if ms_transport == "stdio" and not ms_command:
            raise PlanValidationError(
                f"mcp_servers[{ms_idx}]: stdio transport requires 'command'",
                code=E069,
            )
        if ms_transport in ("http", "sse") and not ms_url:
            raise PlanValidationError(
                f"mcp_servers[{ms_idx}]: {ms_transport} transport requires 'url'",
                code=E069,
            )
        cmd_list: list[str] = []
        if ms_command:
            if isinstance(ms_command, list):
                cmd_list = [str(c) for c in ms_command]
            elif isinstance(ms_command, str):
                cmd_list = [ms_command]
        ms_description = str(ms.get("description", "") or "")
        ms_env = {str(k): str(v) for k, v in ms.get("env", {}).items()} if isinstance(ms.get("env"), dict) else {}
        ms_allowed_task_roles = _to_str_list(
            ms.get("allowed_task_roles"),
            f"mcp_servers[{ms_idx}].allowed_task_roles",
        )
        ms_has_concurrency_flag = "is_concurrency_safe" in ms or "isConcurrencySafe" in ms
        ms_concurrency_raw = (
            ms.get("is_concurrency_safe")
            if "is_concurrency_safe" in ms
            else ms.get("isConcurrencySafe")
        )
        ms_is_concurrency_safe = (
            bool(ms_concurrency_raw)
            if ms_has_concurrency_flag and ms_concurrency_raw is not None
            else None
        )
        ms_timeout = int(ms.get("timeout_sec", 30))
        mcp_servers.append(MCPServerSpec(
            name=ms_name,
            command=cmd_list,
            description=ms_description,
            url=str(ms_url) if ms_url else None,
            transport=ms_transport,  # type: ignore[arg-type]
            env=ms_env,
            allowed_task_roles=ms_allowed_task_roles,
            is_concurrency_safe=ms_is_concurrency_safe,
            timeout_sec=ms_timeout,
        ))
    # Check for duplicate names
    _mcp_names = [s.name for s in mcp_servers]
    if len(_mcp_names) != len(set(_mcp_names)):
        raise PlanValidationError(
            "mcp_servers names must be unique",
            code=E069,
        )
    plan.mcp_servers = mcp_servers

    raw_routing_strategy = raw.get("routing_strategy")
    if raw_routing_strategy is not None:
        plan.routing_strategy = cast(RoutingStrategy, str(raw_routing_strategy))
    else:
        plan.routing_strategy = None

    cfi = raw.get("control_flow_integrity", False)
    if isinstance(cfi, bool):
        plan.control_flow_integrity = cfi
    elif isinstance(cfi, str) and cfi.lower() in ("true", "false"):
        plan.control_flow_integrity = cfi.lower() == "true"

    validate_plan(plan)
    _collect_warnings(plan)
    return plan


_TOOL_PATTERN_RE = re.compile(r"^([A-Za-z]\w*)\((.+)\)$")


def _extract_tool_name(entry: str) -> str:
    """Extract the base tool name from an entry, handling wildcard patterns.

    ``"Read"`` → ``"Read"``, ``"Bash(git *)"`` → ``"Bash"``.
    """
    m = _TOOL_PATTERN_RE.match(entry)
    return m.group(1) if m else entry


def _validate_allowed_tools(task: TaskSpec, plan: PlanSpec) -> None:
    """Validate ``allowed_tools`` on a task, emitting E071 or W27 as needed.

    Supports bare tool names (``Read``) and wildcard patterns
    (``Bash(git *)``).  The base tool name is validated; the argument
    pattern is passed through without validation.
    """
    if task.allowed_tools is None:
        return

    # E071: allowed_tools on a non-engine task
    if task.engine is None:
        raise PlanValidationError(
            f"Task '{task.id}' has allowed_tools but no engine "
            "(allowed_tools requires an engine task)",
            code=E071,
        )

    engine = task.engine
    ws = plan.validation_warnings

    if engine == "claude":
        for entry in task.allowed_tools:
            name = _extract_tool_name(entry)
            if (
                name not in CLAUDE_TOOLS
                and not name.startswith("mcp__")
                and entry not in TOOL_CATEGORIES
            ):
                ws.append(
                    f"W27: task '{task.id}': allowed_tools entry '{entry}' "
                    f"is not a recognised Claude tool, MCP reference, or tool category"
                )
    elif engine == "codex":
        for entry in task.allowed_tools:
            name = _extract_tool_name(entry)
            if name not in CODEX_SANDBOX_LEVELS and entry not in TOOL_CATEGORIES:
                ws.append(
                    f"W27: task '{task.id}': allowed_tools entry '{entry}' "
                    f"is not a recognised Codex sandbox level or tool category"
                )
    elif engine == "ollama":
        ws.append(
            f"W27: task '{task.id}': Ollama has no tool restriction support; "
            f"allowed_tools is advisory only"
        )
    elif engine == "llama":
        ws.append(
            f"W27: task '{task.id}': Llama has no tool restriction support; "
            f"allowed_tools is advisory only"
        )
    elif engine in ("gemini", "copilot", "qwen"):
        ws.append(
            f"W27: task '{task.id}': tool restriction for {engine} "
            f"is system-prompt-enforced only"
        )


def validate_plan(plan: PlanSpec) -> None:
    if plan.version != 1:
        raise PlanValidationError("Only version: 1 is supported", code=E002)
    if not plan.name:
        raise PlanValidationError("Plan name must not be empty", code=E001)
    if not _SAFE_ID_RE.match(plan.name):
        raise PlanValidationError(
            f"Plan name '{plan.name}' contains invalid characters "
            "(allowed: alphanumeric, hyphen, underscore, dot; must start with alphanumeric)",
            code=E017,
        )
    if plan.max_parallel < 1:
        raise PlanValidationError("max_parallel must be >= 1", code=E012)
    if plan.defaults.stdout_tail_lines < 1:
        raise PlanValidationError("defaults.stdout_tail_lines must be >= 1", code=E012)
    if plan.budget_warning_pct is not None and not (0.0 < plan.budget_warning_pct < 1.0):
        raise PlanValidationError("budget_warning_pct must be between 0 and 1", code=E023)
    if plan.budget_period is not None:
        from .budget import BUDGET_PERIODS
        if plan.budget_period not in BUDGET_PERIODS:
            raise PlanValidationError(
                f"budget_period '{plan.budget_period}' is not valid. "
                f"Allowed: {sorted(BUDGET_PERIODS)}",
                code=E014,
            )
    if plan.routing_strategy is not None:
        _valid_routing = ("cost_optimized", "quality_first", "balanced")
        if plan.routing_strategy not in _valid_routing:
            raise PlanValidationError(
                f"Invalid routing_strategy '{plan.routing_strategy}'; "
                f"must be one of: {', '.join(_valid_routing)}",
                code=E053,
            )
    # Validate mcp_tools reference valid mcp_servers
    _mcp_servers_by_name = {s.name: s for s in plan.mcp_servers}
    _mcp_server_names = set(_mcp_servers_by_name)
    for task in plan.tasks:
        for tool_ref in task.mcp_tools:
            if tool_ref not in _mcp_server_names:
                raise PlanValidationError(
                    f"Task '{task.id}' references MCP server '{tool_ref}' "
                    f"in mcp_tools but it is not defined in plan mcp_servers. "
                    f"Available: {sorted(_mcp_server_names) or '(none)'}",
                    code=E070,
                )
            server = _mcp_servers_by_name[tool_ref]
            if server.allowed_task_roles:
                task_role = (task.agent or "").strip()
                if not task_role or task_role not in server.allowed_task_roles:
                    raise PlanValidationError(
                        f"Task '{task.id}' references MCP server '{tool_ref}' "
                        f"but task.agent='{task_role or '(none)'}' is not in "
                        f"allowed_task_roles={server.allowed_task_roles}",
                        code=E070,
                    )
    if (
        plan.defaults.budget_warning_pct is not None
        and not (0.0 < plan.defaults.budget_warning_pct < 1.0)
    ):
        raise PlanValidationError(
            "defaults.budget_warning_pct must be between 0 and 1", code=E023
        )
    if (
        plan.defaults.context_budget_tokens is not None
        and plan.defaults.context_budget_tokens < 1
    ):
        raise PlanValidationError(
            "defaults.context_budget_tokens must be >= 1",
            code=E019,
        )
    if plan.defaults.edit_policy not in EDIT_POLICIES:
        raise PlanValidationError(
            f"defaults.edit_policy '{plan.defaults.edit_policy}' is not valid. "
            f"Allowed: {sorted(EDIT_POLICIES)}",
            code=E008,
        )

    # Validate defaults-level reasoning_effort
    if plan.defaults.codex.reasoning_effort:
        if plan.defaults.codex.reasoning_effort not in CODEX_REASONING_EFFORTS:
            raise PlanValidationError(
                f"defaults.codex.reasoning_effort '{plan.defaults.codex.reasoning_effort}' is not valid. "
                f"Allowed: {sorted(CODEX_REASONING_EFFORTS)}",
                code=E008,
            )
    if plan.defaults.claude.reasoning_effort:
        if plan.defaults.claude.reasoning_effort not in CLAUDE_REASONING_EFFORTS:
            raise PlanValidationError(
                f"defaults.claude.reasoning_effort '{plan.defaults.claude.reasoning_effort}' is not valid. "
                f"Allowed: {sorted(CLAUDE_REASONING_EFFORTS)}",
                code=E008,
            )
    if plan.defaults.gemini.reasoning_effort:
        plan.validation_warnings.append(
            "defaults.gemini.reasoning_effort is set but Gemini CLI does not currently support it. "
            "Value will be ignored."
        )
    if plan.defaults.copilot.reasoning_effort:
        plan.validation_warnings.append(
            "defaults.copilot.reasoning_effort is set but Copilot CLI does not support it. "
            "Use model routing (opus vs sonnet vs haiku) for capability tiers."
        )
    if plan.defaults.qwen.reasoning_effort:
        plan.validation_warnings.append(
            "defaults.qwen.reasoning_effort is set but Qwen CLI does not support reasoning_effort. "
            "Value will be ignored."
        )

    ids = [t.id for t in plan.tasks]
    if len(ids) != len(set(ids)):
        raise PlanValidationError("Task IDs must be unique", code=E003)

    for pack in plan.audit_packs:
        pack_path = resolve_path(plan.source_dir, pack)
        if pack_path is None or not pack_path.exists() or not pack_path.is_file():
            raise PlanValidationError(
                f"audit_packs entry does not resolve to an existing file: {pack}",
                code=E018,
            )

    id_set = set(ids)
    task_by_id = {task.id: task for task in plan.tasks}
    group_members = build_consistency_group_members(plan.tasks)

    for task in plan.tasks:
        if not _SAFE_ID_RE.match(task.id):
            raise PlanValidationError(
                f"Task ID '{task.id}' contains invalid characters "
                "(allowed: alphanumeric, hyphen, underscore, dot; must start with alphanumeric)",
                code=E017,
            )

        for dep in task.depends_on:
            if dep not in id_set:
                raise PlanValidationError(
                    f"Task '{task.id}' depends on unknown task '{dep}'",
                    code=E005,
                )

        _task_type_count = sum([
            task.command is not None,
            task.engine is not None,
            task.group is not None,
        ])
        if _task_type_count == 0:
            raise PlanValidationError(
                f"Task '{task.id}' must define 'command', 'engine', or 'group'",
                code=E001,
            )
        if _task_type_count > 1:
            raise PlanValidationError(
                f"Task '{task.id}' defines more than one of 'command', 'engine', 'group' "
                f"(use exactly one)",
                code=E011,
            )

        if task.group is not None:
            if task.prompt or task.prompt_file or task.prompt_md_file:
                raise PlanValidationError(
                    f"Task '{task.id}': group tasks cannot have prompt fields",
                    code=E011,
                )
            if task.assertions:
                raise PlanValidationError(
                    f"Task '{task.id}': group tasks cannot use assert",
                    code=E011,
                )
            if task.contract_type is not None:
                raise PlanValidationError(
                    f"Task '{task.id}': group tasks cannot produce typed contracts",
                    code=E011,
                )

        if task.batch is not None:
            if task.engine is None or task.command is not None or task.group is not None:
                raise PlanValidationError(
                    f"Task '{task.id}': batch is only allowed on engine tasks",
                    code=E060,
                )
            if task.matrix is not None:
                raise PlanValidationError(
                    f"Task '{task.id}': batch and matrix are mutually exclusive",
                    code=E062,
                )

        # T2.1 -- dynamic_group validation
        if task.dynamic_group:
            if task.engine is None:
                raise PlanValidationError(
                    f"Task '{task.id}': dynamic_group requires engine (not command/group)",
                    code=E063,
                )
            if task.output_schema is None:
                raise PlanValidationError(
                    f"Task '{task.id}': dynamic_group requires output_schema",
                    code=E063,
                )
            if task.group is not None:
                raise PlanValidationError(
                    f"Task '{task.id}': dynamic_group and group are mutually exclusive",
                    code=E064,
                )
            if task.batch is not None:
                raise PlanValidationError(
                    f"Task '{task.id}': dynamic_group and batch are mutually exclusive",
                    code=E064,
                )
            if task.matrix is not None:
                raise PlanValidationError(
                    f"Task '{task.id}': dynamic_group and matrix are mutually exclusive",
                    code=E064,
                )
            # Force cache: false — dynamic tasks depend on workspace state
            task.cache = False

        for contract_id in task.consumes_contracts:
            if contract_id not in id_set:
                raise PlanValidationError(
                    f"Task '{task.id}' consumes_contracts references unknown task '{contract_id}'",
                    code=E018,
                )
            if contract_id == task.id:
                raise PlanValidationError(
                    f"Task '{task.id}' cannot consume its own contract",
                    code=E018,
                )
            producer = task_by_id[contract_id]
            if producer.contract_type is None:
                raise PlanValidationError(
                    f"Task '{task.id}' consumes_contracts references '{contract_id}', "
                    "which does not declare contract_type",
                    code=E018,
                )

        for group_name in task.reconcile_after:
            if group_name not in group_members:
                raise PlanValidationError(
                    f"Task '{task.id}' reconcile_after references unknown consistency_group '{group_name}'",
                    code=E018,
                )

        if task.engine is not None:
            try:
                get_engine_plugin(task.engine)
            except PluginResolutionError as exc:
                raise PlanValidationError(
                    f"Task '{task.id}' has unsupported engine '{task.engine}'. {exc}",
                    code=E006,
                ) from None

        if task.engine is not None and task.command is None:
            has_prompt = bool(task.prompt or task.prompt_file or task.prompt_md_file or task.batch)
            if not has_prompt:
                raise PlanValidationError(
                    f"Task '{task.id}' uses engine '{task.engine}' but has no prompt source",
                    code=E007,
                )

        if bool(task.prompt_md_file) != bool(task.prompt_md_heading):
            raise PlanValidationError(
                f"Task '{task.id}' must define both prompt_md_file and prompt_md_heading",
                code=E011,
            )

        if task.engine == "codex" and task.model:
            if not task.model.startswith("gpt-") and task.model not in _CODEX_MODEL_ALIASES:
                plan.validation_warnings.append(
                    f"Task '{task.id}': Codex model '{task.model}' may not be valid. "
                    f"Known short aliases: {list(_CODEX_MODEL_ALIASES.keys())}"
                )

        if task.engine == "claude" and task.model:
            if task.model not in CLAUDE_MODELS:
                plan.validation_warnings.append(
                    f"Task '{task.id}': Claude model '{task.model}' may not be valid. "
                    f"Known models: {sorted(CLAUDE_MODELS)}"
                )

        if task.engine == "gemini" and task.model:
            if (
                not task.model.startswith("gemini-")
                and task.model not in GEMINI_MODELS
                and task.model not in _GEMINI_MODEL_ALIASES
            ):
                plan.validation_warnings.append(
                    f"Task '{task.id}': Gemini model '{task.model}' may not be valid. "
                    f"Known models: {sorted(GEMINI_MODELS)}"
                )

        if task.engine == "copilot" and task.model:
            if task.model not in COPILOT_MODELS and task.model not in _COPILOT_MODEL_ALIASES:
                plan.validation_warnings.append(
                    f"Task '{task.id}': Copilot model '{task.model}' is not a known alias. "
                    f"Known: {sorted(COPILOT_MODELS)}. Will pass through as-is."
                )

        # Validate reasoning_effort against allowed values
        effort = task.reasoning_effort
        if effort:
            if task.engine == "codex" and effort not in CODEX_REASONING_EFFORTS:
                raise PlanValidationError(
                    f"Task '{task.id}': reasoning_effort '{effort}' is not valid for Codex. "
                    f"Allowed: {sorted(CODEX_REASONING_EFFORTS)}",
                    code=E008,
                )
            if task.engine == "claude" and effort not in CLAUDE_REASONING_EFFORTS:
                raise PlanValidationError(
                    f"Task '{task.id}': reasoning_effort '{effort}' is not valid for Claude. "
                    f"Allowed: {sorted(CLAUDE_REASONING_EFFORTS)}",
                    code=E008,
                )
            if task.engine == "gemini":
                plan.validation_warnings.append(
                    f"Task '{task.id}': Gemini CLI does not currently support reasoning_effort. "
                    f"Value '{effort}' will be ignored."
                )
            if task.engine == "copilot":
                plan.validation_warnings.append(
                    f"Task '{task.id}': Copilot CLI does not support reasoning_effort. "
                    f"Use model routing (opus vs sonnet vs haiku) for capability tiers. "
                    f"Value '{effort}' will be ignored."
                )
            if task.engine == "qwen":
                plan.validation_warnings.append(
                    f"Task '{task.id}': Qwen CLI does not support reasoning_effort. "
                    f"Value '{effort}' will be ignored."
                )
            if task.engine == "ollama":
                plan.validation_warnings.append(
                    f"Task '{task.id}' sets reasoning_effort but Ollama does not support it"
                    " — model selection serves a similar purpose"
                )
            if task.engine == "llama":
                plan.validation_warnings.append(
                    f"Task '{task.id}' sets reasoning_effort but Llama does not support it"
                    " — model selection serves a similar purpose"
                )

        if task.id in task.depends_on:
            raise PlanValidationError(
                f"Task '{task.id}' cannot depend on itself", code=E016
            )

        for ctx in task.context_from:
            if ctx == "*":
                continue
            if ctx not in id_set:
                raise PlanValidationError(
                    f"Task '{task.id}' context_from references unknown task '{ctx}'",
                    code=E010,
                )
            if ctx not in task.depends_on:
                raise PlanValidationError(
                    f"Task '{task.id}' context_from lists '{ctx}' which is not in depends_on "
                    f"(context is only available from direct dependencies)",
                    code=E010,
                )

        if task.max_retries < 0 or task.max_retries > MAX_RETRIES_LIMIT:
            raise PlanValidationError(
                f"Task '{task.id}' max_retries must be 0-{MAX_RETRIES_LIMIT}, "
                f"got {task.max_retries}",
                code=E012,
            )
        # -- Escalation & fallback (v1.3.0) --
        if task.escalation:
            if task.engine is None:
                raise PlanValidationError(
                    f"Task '{task.id}' has escalation but no engine",
                    code="E031",
                )
            for m in task.escalation:
                if not isinstance(m, str) or not m.strip():
                    raise PlanValidationError(
                        f"Task '{task.id}' escalation entries must be non-empty strings",
                        code="E031",
                    )

        if task.fallback_engine is not None:
            if task.engine is None:
                raise PlanValidationError(
                    f"Task '{task.id}' has fallback_engine but no engine",
                    code="E030",
                )
            if task.fallback_engine not in _VALID_ENGINES:
                raise PlanValidationError(
                    f"Task '{task.id}' fallback_engine '{task.fallback_engine}' is not a valid engine",
                    code="E030",
                )

        if task.fallback_model is not None and task.fallback_engine is None:
            raise PlanValidationError(
                f"Task '{task.id}' has fallback_model without fallback_engine",
                code="E030",
            )
        # -- Capability-Based Tool Access (v2.0) --
        _validate_allowed_tools(task, plan)

        if task.max_iterations is not None and task.max_iterations < 1:
            raise PlanValidationError(
                f"Task '{task.id}' max_iterations must be >= 1",
                code=E022,
            )
        if task.approval_message and not task.requires_approval:
            raise PlanValidationError(
                f"Task '{task.id}' has 'approval_message' without 'requires_approval: true'",
                code=E029,
            )

        if task.stdout_tail_lines is not None and task.stdout_tail_lines < 1:
            raise PlanValidationError(
                f"Task '{task.id}' stdout_tail_lines must be >= 1", code=E012
            )

        if task.context_budget_tokens is not None and task.context_budget_tokens < 1:
            raise PlanValidationError(
                f"Task '{task.id}' context_budget_tokens must be >= 1",
                code=E019,
            )

        if task.judge is not None:
            if not task.judge.criteria:
                raise PlanValidationError(
                    f"Task '{task.id}': judge.criteria must be a non-empty list",
                    code=E020,
                )
            if any(not str(criterion).strip() for criterion in task.judge.criteria):
                raise PlanValidationError(
                    f"Task '{task.id}': judge.criteria entries must be non-empty strings",
                    code=E020,
                )
            if task.judge.pass_threshold < 0 or task.judge.pass_threshold > 1:
                raise PlanValidationError(
                    f"Task '{task.id}': judge.pass_threshold must be between 0 and 1",
                    code=E020,
                )
            if task.judge.on_fail not in JUDGE_ON_FAIL_VALUES:
                raise PlanValidationError(
                    f"Task '{task.id}': judge.on_fail '{task.judge.on_fail}' is not valid. "
                    f"Allowed: {sorted(JUDGE_ON_FAIL_VALUES)}",
                    code=E020,
                )
            if not task.judge.model.strip():
                raise PlanValidationError(
                    f"Task '{task.id}': judge.model must be a non-empty string",
                    code=E020,
                )
            if task.judge.method == "g_eval" and task.judge.model == "haiku":
                plan.validation_warnings.append(
                    f"Task '{task.id}': judge.method 'g_eval' works best with a more capable model "
                    "than 'haiku' (sonnet recommended)."
                )

            # -- W22: Judge timeout potentially insufficient for method/criteria/quorum --
            _j = task.judge
            _j_criteria_count = len(_j.criteria)
            _j_explicit = _j.timeout_sec is not None
            if _j.method == "g_eval":
                _j_min = 120
                if _j_criteria_count > 4:
                    _j_min += (_j_criteria_count - 4) * 15
                if _j.quorum is not None and _j.quorum >= 2:
                    _j_min *= _j.quorum
                if _j_explicit and _j.timeout_sec < _j_min:  # type: ignore[operator]
                    plan.validation_warnings.append(
                        f"W22: Task '{task.id}': judge.method 'g_eval' makes 2 LLM calls"
                        f" ({_j_criteria_count} criteria"
                        f"{f', quorum={_j.quorum}' if _j.quorum and _j.quorum >= 2 else ''}"
                        f"); timeout_sec={_j.timeout_sec} may be insufficient"
                        f" (recommend >= {_j_min})"
                    )
                elif not _j_explicit and _j_criteria_count > 4:
                    plan.validation_warnings.append(
                        f"W22: Task '{task.id}': judge.method 'g_eval' with"
                        f" {_j_criteria_count} criteria — auto-scaled timeout"
                        f" to {_j_min}s (set judge.timeout_sec explicitly to override)"
                    )
            elif _j.method == "debate":
                _j_rounds = max(1, min(_j.debate_rounds, 4))
                _j_min = 60 * _j_rounds * 2
                if _j_criteria_count > 4:
                    _j_min += (_j_criteria_count - 4) * 15
                if _j.quorum is not None and _j.quorum >= 2:
                    _j_min *= _j.quorum
                if _j_explicit and _j.timeout_sec < _j_min:  # type: ignore[operator]
                    plan.validation_warnings.append(
                        f"W22: Task '{task.id}': judge.method 'debate' with"
                        f" {_j_rounds} rounds makes {_j_rounds * 2} LLM calls"
                        f"; timeout_sec={_j.timeout_sec} may be insufficient"
                        f" (recommend >= {_j_min})"
                    )
            elif _j.method == "reflection":
                _j_min = 120  # 2 LLM calls (critique + scoring)
                if _j_criteria_count > 4:
                    _j_min += (_j_criteria_count - 4) * 15
                if _j.quorum is not None and _j.quorum >= 2:
                    _j_min *= _j.quorum
                if _j_explicit and _j.timeout_sec < _j_min:  # type: ignore[operator]
                    plan.validation_warnings.append(
                        f"W22: Task '{task.id}': judge.method 'reflection' makes"
                        f" 2 LLM calls (critique + scoring"
                        f"{f', quorum={_j.quorum}' if _j.quorum and _j.quorum >= 2 else ''}"
                        f"); timeout_sec={_j.timeout_sec} may be insufficient"
                        f" (recommend >= {_j_min})"
                    )
            elif _j.quorum is not None and _j.quorum >= 2:
                _j_min = 60 * _j.quorum
                if _j_criteria_count > 4:
                    _j_min += (_j_criteria_count - 4) * 15
                if _j_explicit and _j.timeout_sec < _j_min:  # type: ignore[operator]
                    plan.validation_warnings.append(
                        f"W22: Task '{task.id}': judge.quorum={_j.quorum} runs"
                        f" {_j.quorum} sequential evaluations"
                        f"; timeout_sec={_j.timeout_sec} may be insufficient"
                        f" (recommend >= {_j_min})"
                    )
            # -- W24: large quorum degrades consensus reliability --
            if _j.quorum is not None and _j.quorum > 3:
                plan.validation_warnings.append(
                    f"W24: Task '{task.id}': judge.quorum={_j.quorum} — "
                    f"LLM consensus reliability degrades beyond 3 evaluators. "
                    f"Consider quorum: 3 with '{_j.quorum_strategy or 'majority'}' strategy"
                )
            # -- W25: quorum_diversity without quorum --
            if _j.quorum_diversity and (_j.quorum is None or _j.quorum < 2):
                plan.validation_warnings.append(
                    f"W25: Task '{task.id}': judge.quorum_diversity has no effect "
                    f"without quorum >= 2"
                )

        if task.edit_policy is not None and task.edit_policy not in EDIT_POLICIES:
            raise PlanValidationError(
                f"Task '{task.id}': edit_policy '{task.edit_policy}' is not valid. "
                f"Allowed: {sorted(EDIT_POLICIES)}",
                code=E008,
            )

        if task.edit_policy is not None and task.engine is None and task.command is not None:
            plan.validation_warnings.append(
                f"Task '{task.id}': edit_policy has no effect on shell command tasks"
            )
        if task.guard_command is not None and task.engine is None and task.command is None:
            plan.validation_warnings.append(
                f"Task '{task.id}': guard_command on task without engine or command"
            )

        if task.context_mode not in CONTEXT_MODES:
            raise PlanValidationError(
                f"Task '{task.id}': context_mode '{task.context_mode}' is not valid. "
                f"Allowed: {sorted(CONTEXT_MODES)}",
                code=E008,
            )

        if task.context_mode in {"summarized", "map_reduce", "layered", "structural", "knowledge_graph"} and not task.context_from:
            raise PlanValidationError(
                f"Task '{task.id}': context_mode '{task.context_mode}' requires "
                f"non-empty context_from",
                code=E001,
            )
        if task.context_mode == "recursive":
            workspace_root = resolve_path(plan.source_dir, plan.workspace_root)
            if workspace_root is None or not workspace_root.exists() or not workspace_root.is_dir():
                raise PlanValidationError(
                    f"Task '{task.id}': context_mode 'recursive' requires plan.workspace_root "
                    f"to resolve to an existing directory",
                    code=E021,
                )

        # Council mode validation
        if task.context_mode == "council" and task.council is None:
            raise PlanValidationError(
                f"Task '{task.id}': context_mode 'council' requires a 'council' block "
                f"with participants",
                code=E001,
            )
        if task.council is not None and task.context_mode != "council":
            raise PlanValidationError(
                f"Task '{task.id}': 'council' block requires context_mode: council",
                code=E001,
            )

        # -- W28: council topology warnings --
        if task.council is not None:
            if task.council.connections and task.council.topology != "graph":
                plan.validation_warnings.append(
                    f"W28: Task '{task.id}': council.connections provided but "
                    f"topology is '{task.council.topology}', not 'graph' — "
                    f"connections will be ignored"
                )
            if task.council.topology == "chain" and task.council.rounds > 1:
                plan.validation_warnings.append(
                    f"W28: Task '{task.id}': council.rounds={task.council.rounds} "
                    f"with chain topology — chain is a single-pass pipeline, "
                    f"extra rounds have no effect"
                )

        # Validate 'when' expression references
        if task.when:
            referenced: set[str] = set()
            for match in re.finditer(
                r"\{\{\s*([a-zA-Z0-9_\-]+)\.[a-zA-Z0-9_]+\s*\}\}", task.when
            ):
                referenced.add(match.group(1))
            for ref_id in referenced:
                if ref_id in ("workspace_root", "plan_name", "task_id"):
                    continue
                if ref_id not in id_set:
                    raise PlanValidationError(
                        f"Task '{task.id}': when expression references "
                        f"unknown task '{ref_id}'",
                        code=E015,
                    )
                if ref_id not in task.depends_on:
                    raise PlanValidationError(
                        f"Task '{task.id}': when expression references "
                        f"task '{ref_id}' which is not in depends_on",
                        code=E015,
                    )

    # Cycle detection (DFS)
    graph = {task.id: task.depends_on for task in plan.tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> None:
        if node in visited:
            return
        if node in visiting:
            raise PlanValidationError(
                f"Dependency cycle detected at task '{node}'", code=E004
            )

        visiting.add(node)
        for dep in graph[node]:
            dfs(dep)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        dfs(node)

    # -- Watch block validation --
    watch = plan.watch
    if watch is not None:
        valid_directions = {"lower_is_better", "higher_is_better"}
        if watch.metric_direction not in valid_directions:
            raise PlanValidationError(
                f"watch.metric_direction must be one of {valid_directions}, "
                f"got '{watch.metric_direction}'",
                code=E033,
            )
        valid_sources = {"stdout_regex", "verify_command", "guard_command", "json_field", "manifest"}
        if watch.metric_source not in valid_sources:
            raise PlanValidationError(
                f"watch.metric_source must be one of {valid_sources}, "
                f"got '{watch.metric_source}'",
                code=E033,
            )
        if watch.metric_source == "stdout_regex":
            if not watch.metric_pattern:
                raise PlanValidationError(
                    "watch.metric_pattern is required when metric_source is 'stdout_regex'",
                    code=E034,
                )
            try:
                compiled = re.compile(watch.metric_pattern)
            except re.error as exc:
                raise PlanValidationError(
                    f"watch.metric_pattern is not a valid regex: {exc}",
                    code=E034,
                )
            if compiled.groups != 1:
                raise PlanValidationError(
                    f"watch.metric_pattern must have exactly 1 capture group, "
                    f"got {compiled.groups}",
                    code=E034,
                )
        if watch.metric_source == "json_field" and not watch.metric_json_path:
            raise PlanValidationError(
                "watch.metric_json_path is required when metric_source is 'json_field'",
                code=E035,
            )
        if watch.max_iterations < 1:
            raise PlanValidationError(
                f"watch.max_iterations must be >= 1, got {watch.max_iterations}",
                code=E036,
            )
        if watch.warmup_iterations < 0 or watch.warmup_iterations >= watch.max_iterations:
            raise PlanValidationError(
                f"watch.warmup_iterations must be >= 0 and < max_iterations "
                f"({watch.max_iterations}), got {watch.warmup_iterations}",
                code=E037,
            )
        if watch.plateau_threshold < 1:
            raise PlanValidationError(
                f"watch.plateau_threshold must be >= 1, got {watch.plateau_threshold}",
                code=E038,
            )
        if watch.max_cost_usd is not None and watch.max_cost_usd <= 0:
            raise PlanValidationError(
                f"watch.max_cost_usd must be positive, got {watch.max_cost_usd}",
                code=E039,
            )
        task_ids = {t.id for t in plan.tasks}
        if watch.metric_task and watch.metric_task not in task_ids:
            raise PlanValidationError(
                f"watch.metric_task '{watch.metric_task}' does not reference a valid task ID",
                code=E040,
            )
        valid_regressions = {"rollback", "revert", "keep"}
        if watch.on_regression not in valid_regressions:
            raise PlanValidationError(
                f"watch.on_regression must be one of {valid_regressions}, "
                f"got '{watch.on_regression}'",
                code=E041,
            )
        if watch.program_md:
            program_path = (
                (plan.source_dir / watch.program_md)
                if plan.source_path
                else Path(watch.program_md)
            )
            if not program_path.exists():
                raise PlanValidationError(
                    f"watch.program_md file not found: {program_path}",
                    code=E042,
                )
        valid_plateau_actions = {"stop", "escalate_model", "notify"}
        if watch.plateau_action not in valid_plateau_actions:
            raise PlanValidationError(
                f"watch.plateau_action must be one of {valid_plateau_actions}, "
                f"got '{watch.plateau_action}'",
                code=E043,
            )
        if watch.iteration_budget_sec is not None and watch.iteration_budget_sec <= 0:
            raise PlanValidationError(
                f"watch.iteration_budget_sec must be positive, "
                f"got {watch.iteration_budget_sec}",
                code=E044,
            )
        if watch.max_total_steps is not None and watch.max_total_steps < 1:
            raise PlanValidationError(
                f"watch.max_total_steps must be >= 1, got {watch.max_total_steps}",
                code=E066,
            )
        # E047: mode: improve requires workspace_root
        if watch.mode == "improve":
            ws = plan.workspace_root
            if not ws:
                raise PlanValidationError(
                    "watch.mode 'improve' requires a resolvable workspace_root",
                    code=E047,
                )

    # --- Worktree validation ---
    for task in plan.tasks:
        if not task.worktree:
            continue
        # E045: worktree requires workspace_root
        ws_root = plan.workspace_root
        if not ws_root:
            raise PlanValidationError(
                f"[{E045}] Task '{task.id}' has worktree: true but plan has no workspace_root",
                code=E045,
            )
        # E046: worktree not valid on group/command tasks
        if task.group or (task.command and not task.engine):
            raise PlanValidationError(
                f"[{E046}] Task '{task.id}' has worktree: true but is a group/command task "
                "(worktree isolation only works with engine tasks)",
                code=E046,
            )

    # --- Policies validation ---
    _valid_policy_actions = {"block", "warn", "audit"}
    seen_policy_names: set[str] = set()
    for policy in plan.policies:
        if not policy.name:
            raise PlanValidationError(
                "Policy entry is missing required 'name' field",
                code=E052,
            )
        if not policy.rule:
            raise PlanValidationError(
                f"Policy '{policy.name}' is missing required 'rule' field",
                code=E052,
            )
        if policy.action not in _valid_policy_actions:
            raise PlanValidationError(
                f"Policy '{policy.name}' has invalid action '{policy.action}'; "
                f"must be one of {sorted(_valid_policy_actions)}",
                code=E052,
            )
        if policy.name in seen_policy_names:
            raise PlanValidationError(
                f"Duplicate policy name: '{policy.name}'",
                code=E052,
            )
        seen_policy_names.add(policy.name)
        try:
            compile_policy(policy)
        except (ValueError, SyntaxError) as exc:
            raise PlanValidationError(
                f"Policy '{policy.name}' has invalid rule syntax: {exc}",
                code=E052,
            )


# ---------------------------------------------------------------------------
# Non-blocking validation warnings
# ---------------------------------------------------------------------------

_MAX_TIMEOUT_WARNINGS = 3

# Known global template variables (not prefixed with a task-id).
_KNOWN_GLOBAL_VARS: set[str] = {
    "workspace_root", "plan_name", "task_id",
    "upstream_synthesis", "workspace_brief",
    "contracts_summary", "consistency_summary",
    "goal",
    "batch.item",
    "watch.history", "watch.program", "watch.iteration",
    "watch.best_metric", "watch.last_metric",
    "watch.blame", "watch.manifest", "watch.lessons",
    "watch.experiments_summary",
    "improve.plan_path", "improve.total_tasks", "improve.frozen_tasks",
    "task_knowledge",
}

# Known suffixes for ``<task-id>.<suffix>`` context variables.
_KNOWN_CONTEXT_SUFFIXES: set[str] = {
    "status", "exit_code", "stdout_tail", "log", "duration",
    "files_changed", "decisions", "errors", "warnings",
    "result_text", "summary",
    # v1.15.0 — output_schema structured outputs: full suffix is "output.<field>"
    # validated separately via startswith("output.") in the W3 check
}
_KNOWN_CONTRACT_SUFFIXES: set[str] = {
    "producer", "type", "summary", "body", "hash", "metadata_json",
}
_KNOWN_CONSISTENCY_SUFFIXES: set[str] = {
    "tasks", "statuses", "summaries", "contracts",
}

# Bash special variables that should not trigger the env-reference warning.
_BASH_SPECIAL_VARS: set[str] = {
    "?", "!", "@", "*", "#", "$", "_", "-",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "LINENO", "RANDOM", "SECONDS", "BASHPID", "BASH_SOURCE",
    "PWD", "OLDPWD", "HOME", "IFS",
}

_HEREDOC_RE = re.compile(r"<<\s*['\"]?\w+")
_PROC_SUB_RE = re.compile(r"[<>]\(")
_ENV_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")
_TSC_NO_EMIT_RE = re.compile(
    r"(?:^|[\s;&|])(?:(?:npx|npm\s+exec|pnpm(?:\s+exec)?|yarn(?:\s+run)?|bunx)\s+)?"
    r"tsc(?:\s+|$).*--noemit\b",
    re.IGNORECASE,
)


def _command_matches_repo_wide_typescript_gate(command: str | list[str] | None) -> bool:
    if not command:
        return False
    rendered = command if isinstance(command, str) else command_to_string(command)
    return bool(_TSC_NO_EMIT_RE.search(rendered))


def _collect_warnings(plan: PlanSpec) -> None:
    """Populate ``plan.validation_warnings`` with non-blocking advisories."""
    ws = plan.validation_warnings
    task_ids = {t.id for t in plan.tasks}
    tasks_with_dependents = {dep for task in plan.tasks for dep in task.depends_on}
    group_names = set(build_consistency_group_members(plan.tasks))

    # -- Pitfall 1 & W1 & W5: Windows shell execution --
    if os.name == "nt":
        for task in plan.tasks:
            # W1: check string commands (shell=True) — includes guard_command
            for field_name, field_val in [
                ("command", task.command),
                ("pre_command", task.pre_command),
                ("verify_command", task.verify_command),
                ("guard_command", task.guard_command),
            ]:
                if isinstance(field_val, str) and field_val:
                    ws.append(
                        f"Task '{task.id}': string {field_name} uses "
                        f"shell=True (cmd.exe on Windows). "
                        f"Consider list format with Git Bash."
                    )
                    # W5: detect bash-only syntax in string commands
                    if _HEREDOC_RE.search(field_val):
                        ws.append(
                            f"Task '{task.id}': {field_name} contains "
                            f"heredoc syntax (<<) which will fail with "
                            f"cmd.exe shell=True. Use list-format command."
                        )
                    if _PROC_SUB_RE.search(field_val):
                        ws.append(
                            f"Task '{task.id}': {field_name} contains "
                            f"process substitution (<() or >()) which will "
                            f"fail with cmd.exe. Use list-format command."
                        )
                    # W-pipes: pipe in string command requires cmd.exe shell
                    if (
                        " | " in field_val
                        or field_val.startswith("| ")
                        or field_val.endswith(" |")
                    ):
                        ws.append(
                            f"Task '{task.id}': {field_name} contains pipe '|' which requires "
                            f"shell=True (cmd.exe on Windows). Consider list-format command."
                        )

            # W1: check list commands for wrong bash binary
            for field_name, field_val in [
                ("command", task.command),
                ("pre_command", task.pre_command),
                ("verify_command", task.verify_command),
                ("guard_command", task.guard_command),
            ]:
                if isinstance(field_val, list) and field_val:
                    exe = field_val[0].replace("\\", "/").lower()
                    if "usr/bin/bash" in exe and "git" in exe:
                        ws.append(
                            f"Task '{task.id}': {field_name} uses "
                            f"Git\\usr\\bin\\bash.exe (no PATH setup). "
                            f"Use Git\\bin\\bash.exe instead."
                        )

    # -- W-multiline-string-verify: multiline py -c in string verify_command --
    for task in plan.tasks:
        if isinstance(task.verify_command, str) and task.verify_command:
            vc = task.verify_command
            if "\n" in vc and ("py -c" in vc or "python -c" in vc):
                ws.append(
                    f"Task '{task.id}': verify_command is a string with multiline "
                    f"'py -c' / 'python -c'. Use list format ['py', '-c', '...'] instead."
                )

    # -- W2: prompt_md_heading starts with '#' --
    for task in plan.tasks:
        if task.prompt_md_heading:
            stripped = task.prompt_md_heading.lstrip()
            if stripped.startswith("#"):
                ws.append(
                    f"Task '{task.id}': prompt_md_heading starts with '#' "
                    f"-- the loader prepends '## ' automatically. "
                    f"Remove the '#' prefix."
                )

    # -- Pitfall 3: Unicode in prompt_md_heading --
    for task in plan.tasks:
        if task.prompt_md_heading:
            try:
                task.prompt_md_heading.encode("ascii")
            except UnicodeEncodeError:
                ws.append(
                    f"Task '{task.id}': prompt_md_heading contains non-ASCII "
                    f"characters which may cause heading mismatch. "
                    f"Use ASCII equivalents (e.g., '--' instead of em-dash)."
                )

    # -- Pitfall 4 & W4: Backslashes in path fields --
    if plan.workspace_root and "\\" in plan.workspace_root:
        ws.append(
            f"Plan workspace_root contains backslashes. "
            f"Use forward slashes instead (e.g., 'C:/path/to/dir')."
        )
    if plan.run_dir and "\\" in plan.run_dir:
        ws.append(
            f"Plan run_dir contains backslashes. "
            f"Use forward slashes instead."
        )
    for task in plan.tasks:
        for field_name, field_val in [
            ("workdir", task.workdir),
            ("prompt_file", task.prompt_file),
            ("prompt_md_file", task.prompt_md_file),
            ("group", task.group),
        ]:
            if field_val and "\\" in field_val:
                ws.append(
                    f"Task '{task.id}': {field_name} contains backslashes. "
                    f"Use forward slashes instead."
                )

    # -- W3: Unrecognised template variables --
    for task in plan.tasks:
        text_fields: list[str] = []
        if task.prompt:
            text_fields.append(task.prompt)
        if isinstance(task.command, str) and task.command:
            text_fields.append(task.command)
        if isinstance(task.pre_command, str) and task.pre_command:
            text_fields.append(task.pre_command)
        if isinstance(task.verify_command, str) and task.verify_command:
            text_fields.append(task.verify_command)

        for text in text_fields:
            for match in _TEMPLATE_RE.finditer(text):
                var = match.group(1)
                if var in _KNOWN_GLOBAL_VARS:
                    continue
                if var.startswith("matrix."):
                    continue
                if var.startswith("contract."):
                    parts = var.split(".", 2)
                    if len(parts) == 3:
                        _contract, producer_id, suffix = parts
                        if (
                            producer_id in set(task.consumes_contracts)
                            and suffix in _KNOWN_CONTRACT_SUFFIXES
                        ):
                            continue
                if var.startswith("consistency."):
                    parts = var.split(".", 2)
                    if len(parts) == 3:
                        _consistency, group_name, suffix = parts
                        if (
                            group_name in set(task.reconcile_after) or group_name in group_names
                        ) and suffix in _KNOWN_CONSISTENCY_SUFFIXES:
                            continue
                if "." in var:
                    prefix, suffix = var.split(".", 1)
                    ctx_sources = set(task.context_from or [])
                    deps = set(task.depends_on or [])
                    refs = ctx_sources | deps
                    if prefix in refs and suffix in _KNOWN_CONTEXT_SUFFIXES:
                        continue
                    if prefix in task_ids and suffix in _KNOWN_CONTEXT_SUFFIXES:
                        # Valid var but task not in depends_on/context_from
                        # (already caught by E010 if truly missing)
                        continue
                    # output_schema structured outputs: {{ task-id.output.field }}
                    if prefix in (refs | task_ids) and suffix.startswith("output."):
                        continue
                ws.append(
                    f"Task '{task.id}': template variable "
                    f"'{{{{{var}}}}}' does not match any known pattern. "
                    f"Check spelling."
                )

    # -- W6: retry_delay_sec list shorter than max_retries --
    for task in plan.tasks:
        if (
            isinstance(task.retry_delay_sec, list)
            and task.max_retries
            and len(task.retry_delay_sec) < task.max_retries
        ):
            ws.append(
                f"Task '{task.id}': retry_delay_sec has "
                f"{len(task.retry_delay_sec)} value(s) but max_retries "
                f"is {task.max_retries}. Last value will be reused for "
                f"remaining retries."
            )

    # -- W7: Environment variable references in commands --
    plan_env_keys = set(plan.defaults.env.keys()) if plan.defaults.env else set()
    for task in plan.tasks:
        task_env_keys = set(task.env.keys()) if task.env else set()
        available = _ENV_ALLOWLIST | plan_env_keys | task_env_keys
        str_fields: list[tuple[str, str]] = []
        if isinstance(task.command, str) and task.command:
            str_fields.append(("command", task.command))
        if isinstance(task.pre_command, str) and task.pre_command:
            str_fields.append(("pre_command", task.pre_command))
        if isinstance(task.verify_command, str) and task.verify_command:
            str_fields.append(("verify_command", task.verify_command))
        if isinstance(task.guard_command, str) and task.guard_command:
            str_fields.append(("guard_command", task.guard_command))

        for field_name, field_val in str_fields:
            for m in _ENV_REF_RE.finditer(field_val):
                var_name = m.group(1) or m.group(2)
                if var_name in _BASH_SPECIAL_VARS:
                    continue
                if var_name in available:
                    continue
                ws.append(
                    f"Task '{task.id}': {field_name} references "
                    f"${var_name} which is not in the env allowlist "
                    f"or task/plan env. It may be empty at runtime."
                )

    # -- W8: Tags with whitespace --
    for task in plan.tasks:
        for tag in task.tags:
            if " " in tag or "\t" in tag:
                ws.append(
                    f"Task '{task.id}' tag '{tag}' contains whitespace — use hyphens instead"
                )

    # -- Pitfall 8: Implicit timeout defaults --
    has_plan_timeout = plan.defaults.timeout_sec is not None
    if not has_plan_timeout:
        no_timeout_tasks = [
            t.id for t in plan.tasks if t.timeout_sec is None
        ]
        if no_timeout_tasks:
            shown = no_timeout_tasks[:_MAX_TIMEOUT_WARNINGS]
            for tid in shown:
                ws.append(
                    f"Task '{tid}': no explicit timeout_sec "
                    f"(will use hardcoded default of 1800s / 30min)."
                )
            remaining = len(no_timeout_tasks) - len(shown)
            if remaining > 0:
                ws.append(
                    f"... and {remaining} more task(s) without explicit timeout."
                )

    # -- W-no-retry-with-verify: verify_command set but max_retries=0 --
    for task in plan.tasks:
        if task.verify_command is not None and task.max_retries == 0:
            ws.append(
                f"Task '{task.id}': has verify_command but max_retries=0. "
                f"Set max_retries >= 1 so verify failures can trigger a retry with feedback."
            )

    # -- W-assert-no-retry: assert set but max_retries=0 --
    for task in plan.tasks:
        if task.assertions and task.max_retries == 0 and task.engine is not None:
            ws.append(
                f"Task '{task.id}': has assert rules but max_retries=0. "
                f"Set max_retries >= 1 if you want deterministic assertion failures to trigger retry."
            )

    # -- W-judge-retry-no-iterations: judge on_fail=retry without max_iterations --
    for task in plan.tasks:
        if (
            task.judge is not None
            and task.judge.on_fail == "retry"
            and task.max_iterations is None
        ):
            ws.append(
                f"Task '{task.id}': judge on_fail='retry' without max_iterations. "
                f"Set max_iterations (e.g. 3-5) to prevent infinite retry spirals."
            )

    # -- W20: retries without an escape valve --
    # Replaces the legacy W20 + W21 + W-timeout-retry-futility chain. Retries
    # only help if SOMETHING differs between attempts. When max_retries > 0 but
    # the task has no feedback signal, no progressive backoff, no model
    # escalation, and no fallback engine, retries reproduce the same conditions
    # under the same timeout and likely fail identically.
    #
    # Escape valves (any one silences the warning):
    #   - verify_command / guard_command / assertions / judge → retries get feedback
    #   - retry_delay_sec as a list → progressive backoff (helps with rate/transient)
    #   - escalation → next retry uses a stronger model (engine tasks only)
    #   - fallback_engine → engine-level failures swap engines (engine tasks only)
    #
    # Applies to both engine and command tasks: a `sleep 999 timeout 60 retries
    # 2` shell loop is just as futile as the engine equivalent. Engine-only
    # valves are still listed in the message so authors know what's available.
    #
    # W21 (task-level timeout_sec + retries) was eliminated: retry-with-feedback
    # is a valid pattern even under a fixed timeout.
    _RETRY_TIGHT_TIMEOUT_SEC = 900
    _plan_default_retry_delay = plan.defaults.retry_delay_sec
    _plan_default_timeout = plan.defaults.timeout_sec
    for task in plan.tasks:
        if task.max_retries == 0:
            continue
        # Group tasks have no command/retry semantics of their own — skip.
        if task.group is not None:
            continue

        # Effective timeout mirrors runtime resolution: task > plan default > 1800.
        if task.timeout_sec is not None:
            effective_timeout = task.timeout_sec
        elif _plan_default_timeout is not None:
            effective_timeout = _plan_default_timeout
        else:
            effective_timeout = 1800

        # Any positive retry_delay_sec (float or list) signals the author expects
        # transient failure modes — accept as an escape valve. A list also enables
        # progressive backoff via retry_strategy.
        def _has_delay_signal(value: object) -> bool:
            if isinstance(value, list) and len(value) > 0:
                return True
            if isinstance(value, (int, float)) and value > 0:
                return True
            return False

        has_progressive_delay = (
            _has_delay_signal(task.retry_delay_sec)
            or _has_delay_signal(_plan_default_retry_delay)
        )
        has_escalation = bool(task.escalation)
        has_fallback = bool(task.fallback_engine)
        has_feedback = (
            task.verify_command is not None
            or task.guard_command is not None
            or bool(task.assertions)
            or task.judge is not None
        )

        if has_progressive_delay or has_escalation or has_fallback or has_feedback:
            continue

        urgency = (
            "tight" if effective_timeout < _RETRY_TIGHT_TIMEOUT_SEC else "fixed"
        )
        # Engine-only valves only suggested to engine tasks.
        engine_valves = (
            " (b) add `escalation: [haiku, sonnet, opus]` to upgrade the model on retry;"
            " (d) add `fallback_engine` for engine-level failures;"
        ) if task.engine is not None else ""
        ws.append(
            f"W20: Task '{task.id}' has max_retries={task.max_retries} with "
            f"a {urgency} effective timeout of {effective_timeout}s and no "
            f"retry escape valve — every attempt runs under identical conditions "
            f"and is likely to fail the same way. Pick at least one: "
            f"(a) add `verify_command` / `guard_command` / `assert` / `judge` so "
            f"retries get feedback;"
            f"{engine_valves}"
            f" (c) set `retry_delay_sec: [60, 120]` for progressive backoff "
            f"(helps with rate limits / transient errors); "
            f"or set `max_retries: 0` if this task is best-effort."
        )

    # -- W-context-no-budget: context_from without context_budget_tokens --
    for task in plan.tasks:
        if (
            task.context_from
            and task.context_budget_tokens is None
            and plan.defaults.context_budget_tokens is None
        ):
            ws.append(
                f"Task '{task.id}': has context_from but no context_budget_tokens. "
                f"Set context_budget_tokens (e.g. 4000-8000) to limit context costs."
            )

    # -- W-judge-contains-engine: judge 'contains' assertion on engine task --
    for task in plan.tasks:
        if task.engine is not None:
            if task.judge is not None:
                for criterion in task.judge.criteria:
                    if isinstance(criterion, dict):
                        ctype = criterion.get("type")
                        if ctype in {"contains", "regex"}:
                            ws.append(
                                f"Task '{task.id}': judge '{ctype}' assertion on engine task. "
                                f"Judge checks engine stdout (JSON), not file contents. "
                                f"Use verify_command exit code or guard_command instead."
                            )
                            break

    for task in plan.tasks:
        if task.fallback_engine is not None and task.fallback_engine == task.engine:
            ws.append(f"W13: Task '{task.id}' fallback_engine is same as engine (redundant)")

        if task.escalation and len(task.escalation) != len(set(task.escalation)):
            ws.append(f"W14: Task '{task.id}' escalation list has duplicates")

        if task.escalation and task.max_retries == 0:
            ws.append(
                f"W15: Task '{task.id}' has escalation but max_retries=0 (escalation needs retries)"
            )

    # W16: worktree on task with no parallel siblings
    worktree_tasks = [t for t in plan.tasks if t.worktree]
    if len(worktree_tasks) == 1:
        t = worktree_tasks[0]
        ws.append(
            f"Task '{t.id}': worktree: true but only one worktree task in plan "
            "(worktree isolation is most useful with parallel tasks)"
        )

    # -- W-observation-block-no-context: observation_block without context_from --
    for task in plan.tasks:
        if task.observation_block and not task.context_from:
            ws.append(
                f"Task '{task.id}': observation_block: true has no effect without context_from"
            )

    # -- W23: codex engine without explicit reasoning_effort --
    _codex_default_re = plan.defaults.codex.reasoning_effort
    for task in plan.tasks:
        if task.engine == "codex" and task.reasoning_effort is None and _codex_default_re is None:
            ws.append(
                f"W23: Task '{task.id}': engine codex without explicit reasoning_effort. "
                f"The user's ~/.codex/config.toml may inject an incompatible value "
                f"(e.g. 'xhigh' on gpt-5-codex-mini). Set reasoning_effort explicitly "
                f"or in defaults.codex.reasoning_effort."
            )

    # -- W29: fail-fast codex plans without fallback still depend on runtime entitlements --
    _codex_no_fallback = [
        task.id for task in plan.tasks
        if task.engine == "codex"
        and (task.fallback_engine or plan.defaults.codex.fallback_engine) is None
    ]
    if plan.fail_fast and _codex_no_fallback:
        ids = ", ".join(_codex_no_fallback[:5])
        suffix = f" (+{len(_codex_no_fallback) - 5} more)" if len(_codex_no_fallback) > 5 else ""
        ws.append(
            f"W29: fail_fast=true with Codex task(s) without fallback_engine "
            f"({ids}{suffix}). Runtime account/model entitlement is not validated offline, "
            f"so an unsupported Codex model can abort the DAG even when the CLI is installed. "
            f"Add fallback_engine or smoke-test the chosen model before large runs."
        )

    # -- W30: repo-wide tsc --noEmit gates can fail on pre-existing baseline errors --
    for task in plan.tasks:
        if task.allow_failure:
            continue
        if not (plan.fail_fast or task.id in tasks_with_dependents):
            continue
        fields = [
            field_name for field_name, field_val in [
                ("command", task.command),
                ("verify_command", task.verify_command),
            ]
            if _command_matches_repo_wide_typescript_gate(field_val)
        ]
        if not fields:
            continue
        field_list = ", ".join(fields)
        ws.append(
            f"W30: Task '{task.id}' uses repo-wide TypeScript compile gate in "
            f"{field_list} (`tsc --noEmit`) while failures can block downstream work. "
            f"Pre-existing baseline errors can produce false negatives even when this "
            f"plan's changes are correct. Consider a baseline compile before the DAG, "
            f"scoping the check, or allow_failure: true plus review."
        )

    # -- W26: overlapping output_scope between tasks increases omission risk --
    for left, right in itertools.combinations(plan.tasks, 2):
        overlaps = _find_output_scope_overlaps(left.output_scope, right.output_scope)
        if overlaps:
            examples = ", ".join(overlaps[:3])
            if len(overlaps) > 3:
                examples += ", ..."
            ws.append(
                f"W26: Tasks '{left.id}' and '{right.id}' have potentially overlapping "
                f"output_scope patterns ({examples}). Consider merging them or narrowing "
                f"scope to preserve the one-task-one-file rule."
            )

    # -- W17/W18/W19: Plan density warnings (AgentConductor S_complex) --
    density_score, density_label, _ = compute_plan_density_score(plan)
    task_count = len(plan.tasks)
    if task_count >= 4:
        edge_count = sum(len(t.depends_on) for t in plan.tasks)
        max_edges = task_count * (task_count - 1) / 2
        edge_density = edge_count / max_edges if max_edges > 0 else 0.0
        if edge_density > 0.6:
            ws.append(
                f"W17: High dependency density ({edge_density:.0%}) — "
                f"consider reducing cross-task dependencies for better parallelism"
            )
        seq_depth = _compute_sequential_depth(plan.tasks)
        parallelism = 1.0 - (seq_depth / task_count) if task_count > 0 else 1.0
        if parallelism < 0.3 and task_count > 3:
            ws.append(
                f"W18: Low parallelism (sequential depth {seq_depth}/{task_count}) — "
                f"most tasks are sequential; consider restructuring the DAG"
            )
    if density_score > 0.8:
        ws.append(
            f"W19: Plan complexity is {density_label} (score {density_score:.2f}) — "
            f"consider splitting into smaller sub-plans or using group tasks"
        )


_SCOPE_GLOB_CHARS = frozenset("*?[{")


def _find_output_scope_overlaps(
    left_scope: list[str],
    right_scope: list[str],
) -> list[str]:
    """Return representative overlap examples for two output_scope lists.

    This is a heuristic warning, not an exact glob-intersection solver. It is
    intentionally biased toward surfacing likely overlaps for manual review.
    """
    overlaps: list[str] = []
    seen: set[str] = set()
    for left, right in itertools.product(left_scope, right_scope):
        if not left or not right:
            continue
        if not _output_scope_patterns_overlap(left, right):
            continue
        example = left if _normalize_scope_pattern(left) == _normalize_scope_pattern(right) else (
            f"{left} <-> {right}"
        )
        if example not in seen:
            seen.add(example)
            overlaps.append(example)
    return overlaps


def _output_scope_patterns_overlap(left: str, right: str) -> bool:
    left_norm = _normalize_scope_pattern(left)
    right_norm = _normalize_scope_pattern(right)
    if not left_norm or not right_norm:
        return False
    if left_norm == right_norm:
        return True

    left_literal = _is_literal_scope_pattern(left_norm)
    right_literal = _is_literal_scope_pattern(right_norm)
    if left_literal and right_literal:
        return left_norm == right_norm
    if left_literal:
        return _scope_glob_matches_path(right_norm, left_norm)
    if right_literal:
        return _scope_glob_matches_path(left_norm, right_norm)

    left_prefix = _scope_literal_prefix(left_norm)
    right_prefix = _scope_literal_prefix(right_norm)
    if left_prefix and right_prefix:
        if not (
            left_prefix.startswith(right_prefix)
            or right_prefix.startswith(left_prefix)
        ):
            return False

    left_suffix = _scope_extension_hint(left_norm)
    right_suffix = _scope_extension_hint(right_norm)
    if left_suffix and right_suffix and left_suffix != right_suffix:
        return False

    return True


def _normalize_scope_pattern(pattern: str) -> str:
    return pattern.replace("\\", "/").strip()


def _is_literal_scope_pattern(pattern: str) -> bool:
    return not any(char in pattern for char in _SCOPE_GLOB_CHARS)


def _scope_glob_matches_path(glob_pattern: str, raw_path: str) -> bool:
    path = raw_path.lstrip("./")
    pattern = glob_pattern.lstrip("./")
    try:
        pure_path = PurePosixPath(path)
        return pure_path.match(pattern)
    except ValueError:
        return False


def _scope_literal_prefix(pattern: str) -> str:
    prefix_parts: list[str] = []
    for part in pattern.split("/"):
        if any(char in part for char in _SCOPE_GLOB_CHARS):
            break
        prefix_parts.append(part)
    if not prefix_parts:
        return ""
    return "/".join(prefix_parts) + "/"


def _scope_extension_hint(pattern: str) -> str | None:
    leaf = pattern.split("/")[-1]
    if "." not in leaf:
        return None
    suffix = "." + leaf.rsplit(".", 1)[-1]
    if any(char in suffix for char in "[]{}"):
        return None
    return suffix


def _dag_max_depth(plan: PlanSpec) -> int:
    """Longest DAG path length (root tasks = 0, each hop adds 1)."""
    task_map: dict[str, TaskSpec] = {t.id: t for t in plan.tasks}
    memo: dict[str, int] = {}

    def _depth(tid: str) -> int:
        if tid in memo:
            return memo[tid]
        t = task_map.get(tid)
        if t is None or not t.depends_on:
            memo[tid] = 0
            return 0
        memo[tid] = 1 + max(_depth(d) for d in t.depends_on)
        return memo[tid]

    return max((_depth(t.id) for t in plan.tasks), default=0)


def compute_plan_density(plan: PlanSpec) -> dict[str, int | float]:
    """DAG density metrics: nodes, edges, depth, s_node, s_edge, s_depth, s_complex.

    S_complex is inspired by AgentConductor topology density score — lower = sparser.
    """
    import math as _math

    n = len(plan.tasks)
    if n == 0:
        return {
            "nodes": 0, "edges": 0, "depth": 0,
            "s_node": 0.0, "s_edge": 0.0, "s_depth": 0.0, "s_complex": 0.0,
        }

    edges = sum(len(t.depends_on) for t in plan.tasks)
    depth = _dag_max_depth(plan)

    n_max = 10.0
    s_node = _math.exp(-n / n_max)
    s_edge = _math.exp(-edges / (n * (n - 0.5))) if n > 1 else 1.0
    s_depth = 1.0 - (depth / n) if n > 0 else 0.0
    s_complex = _math.exp(s_node + 2 * s_edge + s_depth)

    return {
        "nodes": n,
        "edges": edges,
        "depth": depth,
        "s_node": round(s_node, 3),
        "s_edge": round(s_edge, 3),
        "s_depth": round(s_depth, 3),
        "s_complex": round(s_complex, 3),
    }


def _compute_sequential_depth(tasks: list[TaskSpec]) -> int:
    """Compute the longest dependency chain length (sequential depth) in the DAG."""
    if not tasks:
        return 0
    graph: dict[str, list[str]] = {t.id: list(t.depends_on) for t in tasks}
    cache: dict[str, int] = {}

    def _depth(node: str) -> int:
        if node in cache:
            return cache[node]
        deps = graph.get(node, [])
        d = 1 + max((_depth(dep) for dep in deps), default=0)
        cache[node] = d
        return d

    return max((_depth(t_id) for t_id in graph), default=1)


def compute_plan_density_score(
    plan: PlanSpec,
) -> tuple[float, str, str]:
    """Compute S_complex density score for a plan (AgentConductor-inspired).

    Uses the AgentConductor S_complex formula components:
    - S_node: exp(-|V| / N_max) — node count vs capacity
    - S_edge: exp(-|E| / (|V| * (|V| - 0.5))) — edge density
    - S_depth: 1 - s/|V| — parallelism ratio

    Blended with Maestro-specific factors (context, judges, resilience).

    Returns (score, label, factors_description):
    - score: 0.0-1.0 float (higher = more complex)
    - label: low, moderate, high, or very_high
    - factors: human-readable description of contributing factors
    """
    n = len(plan.tasks)
    if n == 0:
        return 0.0, "low", ""

    # -- AgentConductor S_complex components (inverted: 1-S = complexity) --
    n_max = 10.0  # node capacity for complex plans
    s_node = math.exp(-n / n_max)
    node_complexity = 1.0 - s_node  # more tasks → higher complexity

    total_edges = sum(len(t.depends_on) for t in plan.tasks)
    edge_denom = n * (n - 0.5) if n > 1 else 1.0
    s_edge = math.exp(-total_edges / edge_denom)
    edge_complexity = 1.0 - s_edge  # more edges → higher complexity

    seq_depth = _compute_sequential_depth(plan.tasks)
    s_depth = 1.0 - (seq_depth / n) if n > 0 else 1.0
    depth_complexity = 1.0 - s_depth  # deeper chain → higher complexity

    # S_complex uses 2× weight on edges (per paper)
    s_complex_raw = node_complexity + 2.0 * edge_complexity + depth_complexity
    s_complex = min(s_complex_raw / 4.0, 1.0)  # normalize to 0-1

    # -- Maestro-specific factors --
    advanced_context = sum(
        1 for t in plan.tasks
        if t.context_mode in ("summarized", "map_reduce", "recursive", "layered")
    )
    context_ratio = advanced_context / n

    judge_count = sum(1 for t in plan.tasks if t.judge is not None)
    judge_ratio = judge_count / n

    engines = {t.engine for t in plan.tasks if t.engine is not None}
    engine_diversity = min(len(engines) / 4.0, 1.0)

    resilience_count = sum(
        1 for t in plan.tasks
        if t.max_retries > 0 or t.escalation or t.fallback_engine
    )
    resilience_ratio = resilience_count / n

    # Weighted blend: 50% S_complex (DAG topology) + 50% Maestro factors
    score = (
        0.50 * s_complex
        + 0.15 * context_ratio
        + 0.15 * judge_ratio
        + 0.10 * engine_diversity
        + 0.10 * resilience_ratio
    )
    score = min(score, 1.0)

    if score >= 0.70:
        label = "very_high"
    elif score >= 0.50:
        label = "high"
    elif score >= 0.30:
        label = "moderate"
    else:
        label = "low"

    factors_parts: list[str] = []
    factors_parts.append(f"S_complex={s_complex:.2f}")
    max_edges = n * (n - 1) / 2 if n > 1 else 1
    dep_density = min(total_edges / max_edges, 1.0) if max_edges > 0 else 0.0
    if dep_density >= 0.3:
        factors_parts.append(f"dep_density={dep_density:.2f}")
    if seq_depth > 1:
        factors_parts.append(f"depth={seq_depth}/{n}")
    if context_ratio >= 0.3:
        factors_parts.append(f"advanced_context={advanced_context}/{n}")
    if judge_ratio >= 0.3:
        factors_parts.append(f"judges={judge_count}/{n}")
    if engine_diversity >= 0.5:
        factors_parts.append(f"engines={len(engines)}")
    if resilience_ratio >= 0.3:
        factors_parts.append(f"resilience={resilience_count}/{n}")

    return score, label, ", ".join(factors_parts)
