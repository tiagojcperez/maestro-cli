from __future__ import annotations

# ---------------------------------------------------------------------------
# Validation error codes (E001-E099)
# ---------------------------------------------------------------------------

E001 = "E001"  # Missing required field
E002 = "E002"  # Invalid schema version
E003 = "E003"  # Duplicate task ID
E004 = "E004"  # Circular dependency
E005 = "E005"  # Unknown dependency reference
E006 = "E006"  # Invalid engine name
E007 = "E007"  # Missing prompt source
E008 = "E008"  # Invalid field value (reasoning_effort, edit_policy, etc.)
E009 = "E009"  # Invalid model name
E010 = "E010"  # Invalid context_from reference
E011 = "E011"  # Mutually exclusive fields conflict
E012 = "E012"  # Value out of range (max_retries, max_parallel, etc.)
E013 = "E013"  # Invalid delay specification
E014 = "E014"  # Invalid budget value
E015 = "E015"  # Invalid when expression
E016 = "E016"  # Self-dependency
E017 = "E017"  # Invalid characters in name/ID
E018 = "E018"  # Type mismatch (expected dict, got list, etc.)
E019 = "E019"  # Context budget value out of range
E020 = "E020"  # Invalid judge configuration
E021 = "E021"  # context_mode: recursive without workspace root
E022 = "E022"  # Invalid max_iterations value
E023 = "E023"  # Invalid budget_warning_pct value
E024 = "E024"  # Invalid secrets configuration
E025 = "E025"  # Circular import or max depth exceeded
E026 = "E026"  # Invalid import structure
E027 = "E027"  # Duplicate import prefix
E028 = "E028"  # Invalid import prefix format
E029 = "E029"  # approval_message without requires_approval
E032 = "E032"  # Invalid watch block structure
E033 = "E033"  # Invalid watch.metric_direction or metric_source
E034 = "E034"  # Invalid watch.metric_pattern
E035 = "E035"  # Missing watch.metric_json_path for json_field source
E036 = "E036"  # Invalid watch.max_iterations
E037 = "E037"  # Invalid watch.warmup_iterations
E038 = "E038"  # Invalid watch.plateau_threshold
E039 = "E039"  # Invalid watch.max_cost_usd
E040 = "E040"  # Invalid watch.metric_task reference
E041 = "E041"  # Invalid watch.on_regression
E042 = "E042"  # watch.program_md file not found
E043 = "E043"  # Invalid watch.plateau_action
E044 = "E044"  # Invalid watch.iteration_budget_sec
E045 = "E045"  # worktree: true requires workspace_root
E046 = "E046"  # worktree: true not valid on group/command tasks
E047 = "E047"  # watch mode: improve requires resolvable workspace_root
E048 = "E048"  # invalid watch.mode value
E057 = "E057"  # Invalid batch configuration (missing items/template, empty items)
E058 = "E058"  # Invalid batch.max_per_call (must be >= 1)
E060 = "E060"  # batch not allowed on command/group tasks (engine only)
E062 = "E062"  # batch and matrix are mutually exclusive

E063 = "E063"  # dynamic_group requires engine + output_schema
E064 = "E064"  # dynamic_group conflicts with group/batch/matrix

E066 = "E066"  # Invalid watch.max_total_steps (must be >= 1)
E067 = "E067"  # Invalid reminders configuration (missing trigger/message, empty values)
E068 = "E068"  # Invalid context_compaction value
E069 = "E069"  # Invalid MCP server configuration
E070 = "E070"  # Unknown MCP server reference in mcp_tools
E071 = "E071"  # allowed_tools on command/group task (engine only)
E072 = "E072"  # Invalid council graph topology connections

E052 = "E052"  # Invalid policy configuration (missing name/rule, bad action, duplicate name, bad rule syntax)
E053 = "E053"  # Invalid routing_strategy value
E054 = "E054"  # Invalid judge.quorum value (must be integer >= 2)
E055 = "E055"  # Invalid judge.quorum_strategy value
E056 = "E056"  # quorum_strategy requires quorum to be set

# ---------------------------------------------------------------------------
# Runtime error codes (E100-E199)
# ---------------------------------------------------------------------------

E100 = "E100"  # Prompt file not found
E101 = "E101"  # Markdown heading not found
E102 = "E102"  # Unsupported engine
E103 = "E103"  # No engine specified
E104 = "E104"  # Workdir resolution failed
E105 = "E105"  # Command build failure
E106 = "E106"  # Group sub-plan not found or failed to load
E107 = "E107"  # Judge evaluation failed
E108 = "E108"  # Workspace index build failure
E109 = "E109"  # Workspace extraction LLM call failure
E110 = "E110"  # Workspace brief LLM call failure


class PlanValidationError(Exception):
    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        msg = super().__str__()
        if self.code:
            return f"[{self.code}] {msg}"
        return msg


class TaskExecutionError(Exception):
    def __init__(self, message: str, code: str | None = None) -> None:
        super().__init__(message)
        self.code = code

    def __str__(self) -> str:
        msg = super().__str__()
        if self.code:
            return f"[{self.code}] {msg}"
        return msg
