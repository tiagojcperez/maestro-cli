# Maestro CLI v1.0.0 Public Contract

This document defines the stable public contract for Maestro CLI v1.0.0.
It is the source of truth for what must remain backward-compatible for all
`1.x` releases. Anything not listed here is not frozen by v1.0.0.
It turns the v1.0.0 roadmap milestone into explicit regression targets.
Roadmap bullets, including the historical v1.0.0 roadmap list, README
summaries, and incidental help text are informative; this file is the
normative v1 freeze and regression fixture.
Implemented features may exist outside this contract; they are not frozen until
they are listed here.

## Scope

The v1.0.0 freeze covers three machine-relevant surfaces:

- Authored YAML plans with `version: 1`
- The documented CLI command and flag surface
- On-disk run artifacts intended for automation and post-processing

This freeze is intentionally narrower than the broader v1.0.0 milestone. A
feature may ship in v1.0.0 without being frozen unless it is explicitly listed
below as part of the contract.

The freeze does not silently bless every current implementation detail. Where
current behavior is useful but not stable enough to promise for `1.x`, it is
called out below as out of contract.

## Contract Reading Rules

This document is intended to be regression-testable. Read it with these rules:

- Items listed below are the frozen public contract for `1.x`.
- The enumerated keys, command names, option spellings, artifact filenames, and
  JSON keys below are the normative regression targets.
- "Documented" means the surfaces named in this file, not every currently
  implemented code path or parser-accepted variant.
- A regression suite may treat the tables and bullet lists below as allowlists
  for frozen names and keys; omitted behavior is intentionally not frozen.
- `1.x` may add optional YAML keys, new CLI flags or subcommands, and new JSON
  keys, but it must not remove or rename the frozen items listed here.
- If a behavior is omitted from the frozen lists below, or is explicitly called
  out as out of contract, it is not part of the v1.0.0 stability promise.

The regression targets are organized into six contract sections below: Stable
YAML Schema, Stable CLI Surface, Stable Run Artifacts, Compatibility Promise
for `1.x`, Explicit Non-Goals for `v1.0.0`, and Migration Posture from `0.x`
to `1.0.0`.

## Stable YAML Schema

### Schema version

- The authored plan schema version remains `version: 1` throughout `1.x`.
- `1.x` releases may add new optional fields, but they must not require
  `version: 2`.
- A change that requires renumbering the plan schema is deferred to `2.0.0`.

### Stable top-level plan keys

The following authored top-level keys are frozen for `1.x`:

| Key | Notes |
|-----|-------|
| `version` | Must remain `1` |
| `name` | Stable plan identifier |
| `webhook_url` | Optional run-complete webhook |
| `secrets` | Explicit secret names to redact or literal `auto` |
| `workspace_root` | Optional workspace root |
| `max_parallel` | Plan-level concurrency |
| `fail_fast` | Stop dispatching after failure |
| `run_dir` | Run root directory |
| `max_cost_usd` | Soft plan budget |
| `budget_warning_pct` | Budget warning threshold |
| `imports` | Reusable task imports |
| `defaults` | Plan-level defaults |
| `tasks` | Task list |

### Stable `defaults` keys

The following authored `defaults` keys are frozen for `1.x`:

- `env`
- `secrets`
- `timeout_sec`
- `edit_policy`
- `retry_delay_sec`
- `context_budget_tokens`
- `budget_warning_pct`
- `workspace_index_exclude`
- `codex`
- `claude`
- `gemini`
- `copilot`
- `qwen`
- `ollama`

For each engine block under `defaults`, the stable child keys are:

- `model`
- `reasoning_effort`
- `args`

### Stable `imports` shape

Each object under `imports` is frozen with these authored keys:

- `path`
- `prefix`
- `overrides`

### Stable task keys

The following authored task keys are frozen for `1.x`:

- `id`
- `description`
- `tags`
- `depends_on`
- `engine`
- `command`
- `shell`
- `agent`
- `model`
- `reasoning_effort`
- `args`
- `prompt`
- `prompt_file`
- `prompt_md_file`
- `prompt_md_heading`
- `workdir`
- `env`
- `timeout_sec`
- `allow_failure`
- `max_retries`
- `retry_delay_sec`
- `pre_command`
- `verify_command`
- `context_from`
- `context_mode`
- `context_budget_tokens`
- `context_compact`
- `workspace_index_exclude`
- `edit_policy`
- `when`
- `cache`
- `group`
- `matrix`
- `checkpoint`
- `judge`
- `guard_command`
- `max_iterations`
- `requires_approval`
- `approval_message`

For `judge`, the stable child keys are:

- `criteria`
- `pass_threshold`
- `on_fail`
- `model`
- `method`
- `aggregation`
- `preset`

### Stable authored value shapes

The following authored value shapes are part of the `1.x` contract:

- `secrets` accepts either a list of environment-variable names or the literal
  `auto`
- `matrix` is a map of string keys to lists of string values
- `imports` is a list of objects with `path`, `prefix`, and optional
  `overrides`
- `judge.criteria` accepts the documented criterion families from v1.0.0,
  including string criteria and object criteria with `type` values
  `contains`, `regex`, `is-json`, `llm-rubric`, `cost_under`,
  `duration_under`, and `rubric`

Only the documented authored keys and value families listed in this section are
frozen. Acceptance of extra keys, coercions, or parser leniency is not.

### Stable enumerations and value families

The following value families are part of the `1.x` contract:

- Engines: `codex`, `claude`, `gemini`, `copilot`, `qwen`, `ollama`
- Execution profiles: `plan`, `safe`, `yolo`
- Context modes: `raw`, `summarized`, `map_reduce`, `recursive`, `layered`, `selective`
- Edit policies: `default`, `efficient`, `strict`
- Task status values: `success`, `failed`, `soft_failed`, `skipped`, `dry_run`

### Explicitly out of YAML contract

The following are intentionally not frozen by v1.0.0:

- Internal/generated task fields such as `matrix_parent` and `matrix_values`
- Loader/runtime-only fields such as `source_path` and `validation_warnings`
- Undocumented authored knobs currently accepted by the loader, including raw
  `secrets_auto`, `requires_clean_worktree`, `stdout_tail_lines`, and
  `append_system_prompt`
- Exact warning text, warning ordering, and warning count
- Model alias catalogs, vendor model name mappings, and pricing tables
- Acceptance of undocumented keys or parser leniency around malformed input

## Stable CLI Surface

### Stable top-level entrypoints

The following top-level CLI entrypoints are frozen for `1.x`:

- `maestro --help`
- `maestro --version`
- `maestro validate`
- `maestro run`
- `maestro cleanup`
- `maestro backfill-costs`
- `maestro scaffold`
- `maestro doctor`
- `maestro ui`
- `maestro report`
- `maestro diff`
- `maestro explain`
- `maestro status`
- `maestro eval`
- `maestro suggest`
- `maestro shell`
- `maestro replan`

The documented `maestro ci` command remains supported in v1.0.0, but it is
intentionally outside the frozen `1.x` CLI contract and is therefore excluded
from the list above.
Shipping additional commands in `1.x` does not freeze them unless they are
explicitly added to this document in a later minor release.

### Stable command signatures

The stable command surface at v1.0.0 is:

| Command | Stable arguments/options |
|---------|--------------------------|
| `maestro validate <plan.yaml>` | positional plan |
| `maestro run <plan.yaml> [more-plans...]` | `--dry-run`, `--max-parallel`, `--only`, `--skip`, `--run-dir`, `--webhook`, `--execution-profile` and aliases `--profile-mode`/`--mode`, `--resume`, `--resume-last`, `--verbose`/`-v`, `--quiet`/`-q`, `--output`, `--no-cache`, `--cache-dir`, `--tags`, `--skip-tags`, `--mask-secrets`, `--auto-approve`, `--parallel` |
| `maestro cleanup <plan.yaml>` | `--keep`, `--older-than`, `--dry-run` |
| `maestro backfill-costs` | `--root`, `--run-root`, `--dry-run`, `--codex-pricing-file` |
| `maestro scaffold <brief.yaml>` | `-o`/`--output`, `--validate`, `--cost-check` |
| `maestro doctor` | `--json`, `--run-dir` |
| `maestro ui` | `--host`, `--port`, `--no-browser`, `--project-root` |
| `maestro report <run-path>` | `-o`/`--output` |
| `maestro diff <run_a> <run_b>` | `--json` |
| `maestro explain <plan.yaml>` | `--cache-dir`, `--json` |
| `maestro status <plan.yaml>` | `--cache-dir`, `--run-dir`, `--json` |
| `maestro eval <eval.yaml> <run-path>` | `--json`, `--verbose`/`-v` |
| `maestro suggest <plan.yaml>` | `--run-dir`, `--min-runs`, `--json` |
| `maestro shell` | `--plan` |
| `maestro replan <plan.yaml>` | `--max-attempts`, `--model`, `--auto-approve`, `--execution-profile` and aliases `--profile-mode`/`--mode`, `--dry-run`, `--verbose`/`-v`, `--quiet`/`-q`, `--output` |

The table above, not incidental `argparse` help layout or no-argument banner
output, defines the frozen CLI contract for `1.x`.

### CLI compatibility rules for `1.x`

- Existing documented commands and flags above must not be removed or renamed
  in `1.x`.
- New subcommands and new flags may be added in minor releases.
- Additive aliases may be added in `1.x`, but existing names must keep working.
- Human-readable output may gain fields or wording, but machine-oriented JSON
  options and the on-disk artifacts below remain the preferred automation
  contract.

### Explicitly out of CLI contract

The following current behaviors are not frozen by v1.0.0:

- ASCII banner artwork, colors, and no-arg command overview text
- Exact help text wording and section ordering
- Human-readable stdout/stderr formatting in `text` mode
- Numeric exit-code taxonomy beyond documented success vs non-zero failure
- The documented-but-intentionally-unfrozen `maestro ci` command, its
  `--provider` values, and the generated CI YAML contents
- Custom engine entry-point discovery and plugin authoring conventions
- Interactive shell prompt wording, autocomplete behavior, transcript format,
  and slash-command UX details
- Web UI HTTP routes, SSE payloads, and HTML structure
- HTML report markup and styling

## Stable Run Artifacts

### Stable run directory contract

Each started execution from `maestro run` or the run phase of `maestro replan`
creates a run directory under the configured run root. For `1.x`, consumers may
rely on the presence and purpose of these artifact names:

- `run_manifest.json`
- `run_summary.md`
- `events.jsonl`
- `<task-id>.log`
- `<task-id>.result.json`
- `.cache/` when task-result caching is enabled

Consumers must not parse semantics from the run directory name itself beyond it
being a unique run directory for one execution.

Additional files may appear in the run directory during `1.x`. Consumers should
ignore files that are not part of the frozen artifact list above.
Regression checks should assert the frozen filenames and JSON keys below, not
incidental path formatting, file ordering, or extra files.

The workspace-index cache under `<workspace_root>/.maestro-cache/index/` is an
implementation detail used by recursive context and is not part of the frozen
run-artifact set.

### Stable `run_manifest.json` top-level keys

The top-level manifest object is frozen with these keys:

- `plan_name`
- `run_id`
- `run_path`
- `started_at`
- `finished_at`
- `success`
- `execution_profile`
- `task_results`
- `sequential_duration_sec`
- `parallelism_savings_pct`
- `total_cost_usd`
- `total_tokens`
- `budget_exceeded`

`task_results` is a map keyed by task id. Each value follows the stable
task-result shape below. These keys are the JSON regression targets for
`run_manifest.json` in `1.x`.

### Stable `<task-id>.result.json` keys

Each task result object is frozen with these keys:

- `task_id`
- `status`
- `exit_code`
- `started_at`
- `finished_at`
- `duration_sec`
- `command`
- `log_path`
- `result_path`
- `message`
- `stdout_tail`
- `cost_usd`
- `token_usage`
- `structured_context`
- `retry_count`
- `failure_history`
- `checkpoint_count`
- `judge_result`
- `handoff_report`
- `context_raw_tokens`
- `context_final_tokens`
- `context_compression_ratio`
- `context_raw_bytes`
- `context_final_bytes`
- `compression_ratio`
- `task_hash`

`workspace_brief` is also a stable optional key when recursive context is used.

Stable task status values for `1.x` are:

- `success`
- `failed`
- `soft_failed`
- `skipped`
- `dry_run`

### Stable nested result objects

Where present, these nested objects are part of the frozen contract:

- `failure_history[]`: `attempt`, `category`, `exit_code`, `message`
- `judge_result`: `verdict`, `overall_score`, `criterion_scores`, `reasoning`,
  `eval_steps`, `previous_score`
- `judge_result.criterion_scores[]`: `criterion`, `passed`, `score`,
  `reasoning`
- `handoff_report`: `failure_category`, `partial_output`, `summary`
- `token_usage`: `input_tokens`, `cached_tokens`, `output_tokens`,
  `cache_creation_tokens`, `total_tokens`
- `structured_context`: `task_id`, `status`, `exit_code`, `duration_sec`,
  `files_changed`, `decisions`, `errors`, `warnings`, `cost_usd`,
  `result_text`, `summary`
- `workspace_brief`: `brief_text`, `token_estimate`, `files_referenced`

### Stable `run_summary.md` contract

`run_summary.md` is a stable human-readable artifact. For `1.x`, consumers may
rely on:

- The file existing in every completed run directory
- The file being Markdown
- The file containing a run-level summary and a task table

Consumers should not scrape exact heading text, table formatting, or prose from
`run_summary.md`. Machine consumers should prefer `run_manifest.json`.

### Stable `events.jsonl` contract

For `1.x`, `events.jsonl` is frozen only at this level:

- The file exists in each run directory
- It is newline-delimited JSON
- Each line is a JSON object
- Each event object contains `ts` and `event`

Event-specific payload keys, payload ordering, event names beyond current use,
and event emission timing are not frozen by v1.0.0.

### JSON compatibility rules for `1.x`

- Existing JSON keys listed above must not be removed or renamed in `1.x`.
- New keys may be added in minor releases.
- Consumers must ignore unknown keys.
- JSON key ordering and whitespace are not part of the contract.

## Compatibility Promise for 1.x

Maestro CLI `1.x` makes the following promise:

- Plans authored against the frozen `version: 1` contract continue to validate
  and run across `1.x`, unless they rely on behavior explicitly marked out of
  contract here.
- Existing documented CLI commands and flags continue to work across `1.x`.
- Existing frozen run-artifact filenames and JSON keys remain available across
  `1.x`.
- New features in `1.x` are additive: new optional YAML keys, new flags, new
  subcommands, and new JSON fields are allowed.
- A breaking change to any frozen surface is deferred to `2.0.0`.

Bug fixes are allowed in `1.x` even when they stop honoring clearly erroneous,
undocumented, or contradictory behavior from `0.x`.

## Freeze Promotions (v1.30.0+)

The following surfaces, originally deferred from the v1.0.0 freeze, are now
promoted to the frozen `1.x` contract:

### `maestro ci` command surface (promoted v1.31.0)

The `maestro ci` command and its stable arguments are now frozen:

| Command | Stable arguments |
|---------|------------------|
| `maestro ci <plan.yaml>` | `--provider`, `--output`, `--workflow-name`, `--python-version`, `--test-command` |

Stable `--provider` values: `github_actions`, `gitlab_ci`, `github`, `gitlab`.

The generated CI YAML shape is frozen at the structural level:
- **GitHub Actions**: `validate_linux`, `validate_windows`, `validate_macos`,
  `test_linux`, `test_windows`, `test_macos`, `maestro_real_engine` jobs
- **GitLab CI**: `validate_linux`, `validate_windows`, `validate_macos`,
  `test_linux`, `test_windows`, `test_macos`, `run_maestro_real_engine` jobs

Minor formatting changes (whitespace, comments) are allowed. Adding new
jobs/stages is allowed. Removing or renaming existing jobs/stages is not.

### Custom engine plugin API (promoted v1.31.0)

The following plugin authoring surface is frozen for `1.x`:

- Entry-point group: `maestro_cli.engines`
- `EnginePlugin` dataclass: `name`, `build_command`, `model_aliases`,
  `doctor_probe`, `load_pricing_table`, `resolve_pricing_model`,
  `get_default_model`, `extract_cost`
- `DoctorProbe` dataclass: `executable`, `check_name`, `install_hint`
- `EngineCommandContext` dataclass: `plan`, `task`, `workdir`, `prompt_text`,
  `execution_profile`, `retry_feedback`
- Built-in engines cannot be overridden by entry points
- `discover_engine_plugins()` returns custom plugins only
- `get_engine_plugin()` resolves built-in first, then custom

New optional fields may be added to these dataclasses in `1.x`. Existing
fields must not be removed or renamed.

---

## Explicit Non-Goals for v1.0.0

The following were not required for the v1.0.0 freeze. Items marked *(promoted)*
have been promoted to the contract in a later `1.x` release:

- *(promoted v1.31.0)* ~~Plugin system or custom engine entry points~~
- *(promoted v1.31.0)* ~~CI/CD file generation~~
- Engine-specific release gates, including Claude-dependent release criteria or
  mandatory live-engine calls
- End-to-end tests with real engine calls as a release gate
- Performance benchmark suite as a release gate
- Security audit as a release gate
- `mypy --strict` as a release gate
- Cross-platform certification beyond normal support expectations
- Output-schema validation for task outputs
- Inter-run knowledge features
- Consensus voting
- Smart model routing from prior run history
- Full TUI or agentic terminal mode
- `llama-cpp` / `llama-server` support
- Freezing internal Python modules or dataclass APIs as a supported SDK

## Migration Posture from 0.x to 1.0.0

The move from `0.x` to `1.0.0` is a contract freeze, not a plan-schema reset.

- Existing plans that already use `version: 1` should not need a mechanical
  schema migration just to run on `1.0.0`.
- The main migration task is behavioral: stop depending on undocumented YAML
  keys, parser leniency, warning text, human-readable console formatting, or
  event-payload details.
- Where older `0.x` plans used loader-only knobs such as raw `secrets_auto`,
  migrate to the documented authored form (`secrets: auto`).
- Automation that currently scrapes stdout/stderr should migrate to
  `run_manifest.json`, `<task-id>.result.json`, and documented `--json` /
  `--output jsonl` modes.
- Consumers of run artifacts should ignore unknown JSON keys so `1.x` additive
  expansion does not break them.
- Any compatibility cleanup needed between late `0.x` and `1.0.0` should prefer
  documenting the stable contract over preserving accidental behavior.
- When `0.x` roadmap bullets and `1.0.0` contract language differ, treat this
  document as authoritative for regression testing.
