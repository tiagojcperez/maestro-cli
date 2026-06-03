# Rule: YAML Plan Schema

## Scope
Plan YAML files plus `src/maestro_cli/loader.py` and the dataclasses in
`src/maestro_cli/models.py`.

## Contract Posture
- Authored plans stay on `version: 1`.
- The frozen `1.x` contract is narrower than the full implementation surface.
- `maestro ci` output shape and custom engine plugin authoring remain
  implemented-but-unfrozen in `1.0.0`.

## Required Root Shape
- `version` is required and must equal `1`
- `name` is required and must be non-empty
- `tasks` is required and must be a non-empty list

## Plan-Level Rules
- `max_parallel >= 1`
- `fail_fast` is boolean
- `run_dir` is a path string
- `workspace_root` is optional
- `max_cost_usd` is positive float or null
- `budget_warning_pct` is float in `(0.0, 1.0]` or null
- `secrets` is either a list of env-var names or `"auto"`
- `imports` entries require `path` + `prefix`; overrides are optional

## Task-Level Core Rules
- `id` must be unique
- `depends_on` must reference existing task IDs
- no self-dependencies or cycles
- each task must define exactly one of:
  - `command`
  - `engine`
  - `group`
- allowed engines are:
  - `codex`
  - `claude`
  - `gemini`
  - `copilot`
  - `qwen`
  - `ollama`
  - `llama`

## Prompt Rules
- Engine tasks need a prompt source:
  - `prompt`
  - `prompt_file`
  - `prompt_md_file` + `prompt_md_heading`
- `prompt_md_file` and `prompt_md_heading` must be used together
- Markdown heading matching is exact

## Execution and Reliability Rules
- `command`, `pre_command`, `verify_command`, and `guard_command` accept string
  or list format
- `max_retries` is `0..3`
- `retry_delay_sec` is positive float, list of positive floats, or null
- `max_iterations` is optional and caps total retry/judge loops
- `allow_failure` permits `soft_failed`
- `when` must reference known upstream task fields
- `cache` is boolean
- `requires_approval` is boolean; `approval_message` is only valid with it
- `escalation` is a list of model name strings for auto-escalation on failure;
  inheritable via `defaults.<engine>.escalation`; E030 validation
- `fallback_engine` is a known engine name; `fallback_model` requires
  `fallback_engine`; E031 validation
- escalation is disabled after fallback triggers

## Context Rules
- `context_from` must reference dependencies, except wildcard `"*"`
- `context_mode` is one of `raw`, `summarized`, `map_reduce`, `recursive`,
  `layered`, `selective`, `structural`, `council`, `knowledge_graph`
- `context_budget_tokens` is null or integer `>= 100`
- `context_compact` is boolean
- `workspace_index_exclude` is a list of globs
- `recursive` requires `workspace_root`
- `summarized`, `map_reduce`, and `recursive` are not zero-cost: they trigger
  extra model work and should be chosen intentionally

## Composition Rules
- `matrix` is a dict of key -> list[str] used for task expansion
- `group` points to a nested plan
- imported task IDs are prefixed during `imports` expansion

## Defaults-Level Rules
- `defaults.env` is a string->string dict
- `defaults.secrets` / `defaults.secrets_auto` follow the plan-level secret model
- `defaults.timeout_sec`, `stdout_tail_lines`, `edit_policy`,
  `retry_delay_sec`, `context_budget_tokens`, and
  `workspace_index_exclude` must match their runtime types
- per-engine defaults exist for all current engines:
  - `defaults.codex`
  - `defaults.claude`
  - `defaults.gemini`
  - `defaults.copilot`
  - `defaults.qwen`
  - `defaults.ollama`
  - `defaults.llama`

## Practical Authoring Guidance
- Prefer list-form shell commands for cross-platform safety
- Prefer deterministic `verify_command` / `guard_command` over subjective judge
  logic when possible
- Use `tags` so reruns can stay surgical
- Treat `cache: false` as expensive
- When updating the schema, use `models.py` as the field inventory and
  `validate_plan()` as the enforcement point

## When Adding a New Field
1. Add it to the dataclass in `models.py` with a safe default.
2. Parse it in `loader.py`.
3. Validate it in `validate_plan()` if constraints exist.
4. Wire it into the runtime path that actually uses it.
5. Add loader/runtime tests before documenting it.
