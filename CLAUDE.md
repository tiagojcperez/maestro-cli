# Maestro CLI — Claude Code Instructions

## Project Overview

**Maestro CLI** (`maestro`) is a Python CLI orchestrator for multi-step AI execution plans.
It schedules tasks as a DAG (Directed Acyclic Graph), running them via `codex`, `claude`, `gemini`, `copilot`, `qwen`, `ollama`, `llama`, or raw shell commands — with parallel execution, dependency resolution, and deterministic logging.

- **Package name**: `maestro-ai-cli` (importable as `maestro_cli`)
- **CLI entry point**: `maestro` (or `py -m maestro_cli`)
- **Version**: 2.5.2
- **Python**: >=3.11 (uses PEP 604 `X | Y` unions, `from __future__ import annotations`)
- **Dependencies**: PyYAML core only; optional `[live]` (rich), `[tui]` (textual), `[web]` (fastapi+uvicorn), `[agui]` (ag-ui-protocol), `[mcp]` (mcp SDK), `[otel]` (OpenTelemetry OTLP exporter)

## v1.0.0 Contract Posture

For `1.x`, the frozen public contract is the documented `version: 1` plan
schema, the documented `maestro` CLI commands and flags, and the stable run
artifacts listed in `docs/V1_API_FREEZE.md`.

Implemented but intentionally not frozen by v1.0.0:

- `maestro ci` provider behavior and rendered YAML shape
- Custom engine plugin authoring details behind the `maestro_cli.engines`
  entry-point group
- Broader strict mypy coverage, benchmark thresholds, and stronger release
  gates around security/process tooling

Migration guidance for users coming from `0.x` lives in
`docs/MIGRATING_TO_V1.md`.

## Documentation Map

| File | Purpose |
|------|---------|
| `README.md` | Storefront + router for GitHub visitors (compact, links to docs/) |
| `CLAUDE.md` | Full codebase instructions for Claude Code (this file) |
| `CODEX.md` | Compact project instructions for Codex CLI tasks |
| `AGENTS.md` | Agent role catalog with collaboration map |
| `CHANGELOG.md` | Full release history |
| `docs/AGENT_OPS.md` | **Operations manual for AI agents** — decision trees, pitfalls, checklists |
| `docs/CLI_REFERENCE.md` | Full CLI flag tables for all 28 subcommands |
| `docs/PLAN_GUIDE.md` | Plan schema, context, reliability, judge, policies (authoring guide) |
| `docs/PLAYBOOK.md` | Curated recipes by task type (12 recipes, anti-patterns, cost checklist) |
| `docs/MODELS.md` | Engine model alias tables, routing, pricing |
| `docs/CONTRIBUTING.md` | Testing, CI, mypy, benchmarks, dev setup |
| `docs/PITFALLS.md` | Windows-specific and general gotchas |
| `docs/V1_API_FREEZE.md` | Frozen v1 contract definition |
| `docs/ROADMAP.md` | Planned features |
| `docs/SECURITY.md` | Threat model and mitigation |

## Architecture

```
src/maestro_cli/
├── cli.py            # argparse CLI — 28 shipped subcommands (`ci`, `verify`, `audit`, `chat`, `blame`, `check`, `estimate` included), 16 frozen in the v1 contract, plus `--version` and the ASCII art banner
├── models.py         # Dataclasses: PlanSpec, TaskSpec, TaskResult, PlanRunResult, WorkflowVariant, TokenUsage, JudgeSpec, JudgeResult, RubricLevel, RubricCriterion, FailureRecord, HandoffReport, WorkspaceExtraction, WorkspaceBrief, PlanBrief, TaskBrief, PlanImport, Suggestion, PlanSuggestions, ReplanAttempt, ReplanState, MultiPlanResult, WatchSpec, WatchIteration, WatchState, EventRecord, VerifyStatus, CircuitBreakerSpec, PolicySpec, PolicyViolation, BlameNode, BlameChain, ContextSelectionEntry, ContextTrajectoryReport
├── loader.py         # YAML parsing + validation + cycle detection (DFS) + matrix expansion + plan imports resolution + circuit_breaker/retry_strategy parsing
├── runners.py        # Command building + task execution (subprocess) + pricing tables + recursive context pipeline + handoff reports + context compression + context compaction + guard_command + typed assertions + rubric eval + G-Eval + comparative eval + score aggregation + secrets masking + auto-escalation + cross-engine fallback + Codex token extraction (4 strategies)
├── scheduler.py      # DAG scheduler (ThreadPoolExecutor) + group execution + intent filtering + BM25 scoring + RRF fusion + priority eviction + graph decay + JSONL events + summary + tag filtering + approval gates
├── plugins.py        # Custom engine plugin discovery (`maestro_cli.engines`) + EnginePlugin contract
├── ci.py             # CI generator entrypoints (`maestro ci`) + provider dispatch
├── ci_github_actions.py # GitHub Actions renderer
├── ci_gitlab_ci.py   # GitLab CI renderer
├── ci_agent.py       # CI failure-analysis agent (`maestro ci-analyze`)
├── benchmark.py      # Deterministic local benchmark harness (`maestro-benchmark`)
├── replan.py         # Adaptive re-planning + knowledge/history-guided scoring + multi-variant search + generated-plan security gate + search audit trail
├── mcts.py           # Phase 3 workflow-search foundations (WorkflowVariant tree, selection, simulation, pruning, backpropagation, tree persistence)
├── watch.py          # Autonomous watch loop (metric extraction, git commit/rollback, experiments.jsonl, consolidation agent)
├── worktree.py       # Git worktree isolation (create/merge/cleanup per task)
├── eventsource.py    # Event sourcing: hash chain, replay engine, verify_chain, output envelope
├── audit.py          # Plan security scanner (SEC001-SEC023, AuditFinding, fix_plan)
├── policy.py         # Declarative runtime policy engine (safe AST-based evaluation, block/warn/audit)
├── blame.py          # Causal failure attribution (dependency graph backward walk, blame chain)
├── contracts.py      # Typed contract normalization (6 types + generic fallback), template variable injection
├── workspace_assertions.py # Workspace assertion normalization + evaluation (file_contains, file_regex, package-present, etc.)
├── relationships.py  # Consistency groups, dependency resolution (consumes_contracts + reconcile_after → deps)
├── budget.py         # Cross-run budget tracking (ledger, budget check, maestro budget CLI)
├── dynamic.py        # Dynamic task decomposition (dynamic_group: true, LLM sub-plan generation)
├── knowledge_graph.py # Entity extraction + graph-based context for context_mode: knowledge_graph
├── codebase_graph.py  # AST-backed codebase graph + blast radius analysis
├── council.py        # Multi-model deliberation for context_mode: council (star/chain/graph topologies)
├── symbols.py        # Regex-based code symbol extraction for context_mode: structural (10 languages)
├── routing.py        # Semantic model routing (auto model selection by complexity/tags/DAG structure/routing_strategy/historical performance)
├── knowledge.py      # Cross-run knowledge accumulation (extract/load/store/format, auto-inject into prompts)
├── fts.py            # Zero-dep SQLite FTS5 ranked full-text search (rank_documents, relevance_by_rank, fts5_available) — indexed BM25 lexical ranker for knowledge retrieval
├── memory.py         # SQLite-backed Knowledge + Memory v2 (WAL, provenance, poisoning alerts, score history)
├── multi.py          # Multi-plan execution (sequential/parallel, shared budget, aggregated summary)
├── live.py           # Rich live output renderer + event-driven progress table
├── tui/              # Textual TUI package (--output tui)
│   ├── __init__.py   # MaestroApp factory
│   ├── app.py        # MaestroApp(App) — compose, worker, event dispatch, approval handler bridge
│   ├── widgets.py    # PlanHeader, DAGPanel, DetailPanel, EventFeed, ApprovalModal, TaskState
│   └── app.tcss      # CSS styling
├── suggest.py        # Run history analysis + optimization heuristics (maestro suggest)
├── chat.py           # Multi-model interactive terminal (maestro chat)
├── shell.py          # Interactive REPL with slash commands and autocomplete (maestro shell)
├── workspace_index.py # Workspace file tree indexing with caching (FileEntry, WorkspaceIndex)
├── scaffold.py       # Plan scaffolding from brief YAML (model routing, quality gates, workflow libraries)
├── skill_registry.py # Skill discovery from `.claude/skills/*/SKILL.md` (frontmatter parsing, searchable registry)
├── cache.py          # Task/plan hashing + semantic cache policy (positive + negative cache)
├── diff.py           # Run comparison (RunDiff, TaskDiff, status/cost/token/duration deltas)
├── explain.py        # Cache explain (TaskExplanation, PlanExplanation, hit/miss/disabled)
├── status.py         # Pipeline status (TaskPipelineStatus, PlanPipelineStatus, stale detection)
├── estimate.py       # Offline cost preflight (maestro estimate) — history-aware + token heuristic, reuses pricing tables
├── eval.py           # Batch judge evaluation (EvalResult, EvalSuiteResult, DimensionResult, multi-dimensional eval)
├── doctor.py         # Environment diagnostics (maestro doctor)
├── report.py         # Self-contained HTML report generation (maestro report)
├── cost_backfill.py  # Historical cost/token backfill
├── cleanup.py        # Run directory cleanup
├── ag_ui.py          # AG-UI protocol adapter (event translation, state tracking, SSE format)
├── mcp_server.py     # MCP protocol server (12 tools, 8 resources, 3 prompts)
├── otel.py           # OTLP exporter (run → OpenTelemetry spans, JSON fallback)
├── utils.py          # Helpers: paths, templates, markdown extraction
├── errors.py         # PlanValidationError (E001-E072), TaskExecutionError (E100-E110)
├── __init__.py       # Version + public SDK API (`__all__` exports)
├── __main__.py       # Module entry (py -m maestro_cli)
└── web/              # FastAPI backend + vanilla HTML/CSS/JS dashboard
    ├── app.py        # FastAPI app factory
    ├── routes_api.py # REST API (/runs, /runs/stats, /runs/{id})
    ├── routes_agui.py # AG-UI protocol SSE endpoint (POST /api/agui/runs)
    ├── routes_sse.py # Server-Sent Events for real-time updates
    └── state.py      # Run discovery and caching
```

**Data flow**: YAML → `loader.load_plan()` → `PlanSpec` → `scheduler.run_plan()` → `runners.execute_task()` per task → `.maestro-runs/<run>/` output

**Key design decisions**:
- Local-first persistence: run artifacts stay in JSON/log files and cross-run memory lives in per-plan SQLite under `.maestro-cache/memory/`
- Engine-agnostic: shells out to `codex exec` / `claude --print` / `gemini` / `copilot` / `qwen` / `ollama` / `llama-cli` / any command
- Minimal core deps: only PyYAML — everything else is stdlib unless the optional `[live]` extra is installed
- Strong typing: ~95% type hint coverage, dataclasses everywhere
- Python SDK: `__init__.py` exports a stable public API via `__all__` (core workflow functions, specs, results, type aliases, exceptions) for programmatic use

## Claude Code Configuration

See `AGENTS.md` for the full agent role catalog with expertise and collaboration map.

```
.claude/
├── agents/                        # Subagent role definitions (11 roles — see AGENTS.md)
├── rules/                         # Coding rule documents
│   ├── code-style.md
│   ├── error-handling.md
│   ├── testing.md
│   ├── type-safety.md
│   └── yaml-schema.md
├── skills/                        # Task-oriented skills (slash commands)
│   ├── create-plan/SKILL.md       # /create-plan — scaffold a new YAML plan
│   ├── create-dynamic-plan/SKILL.md # /create-dynamic-plan — scaffold a dynamic_group plan
│   ├── add-engine/SKILL.md        # /add-engine — add a new AI engine
│   ├── add-plan-field/SKILL.md    # /add-plan-field — extend the plan schema
│   ├── debug-run/SKILL.md         # /debug-run — diagnose failed runs
│   └── write-tests/SKILL.md       # /write-tests — write pytest tests
└── settings.json                  # Project-level permission settings
```

**Skills format** (per [Anthropic docs](https://code.claude.com/docs/en/skills)):
- Each skill is a **directory** with a `SKILL.md` entrypoint (NOT a flat `.md` file)
- `SKILL.md` has YAML frontmatter (`name`, `description`, `disable-model-invocation`, `argument-hint`, etc.)
- Markdown body contains the instructions Claude follows when the skill is invoked
- `$ARGUMENTS` placeholder receives arguments from the user (e.g., `/create-plan deploy pipeline`)
- Skills with `disable-model-invocation: true` are only invoked manually via `/name`

## Key Conventions

### Python Style
- `from __future__ import annotations` in ALL files
- PEP 604 union syntax: `str | None` (NOT `Optional[str]`)
- Dataclasses for all data models (NOT dicts, NOT Pydantic)
- Private helpers prefixed with `_` (e.g., `_split_csv`, `_normalize_codex_args`)
- No classes except dataclasses — functional style with module-level functions
- f-strings for all string formatting
- `Path` from pathlib (NOT os.path) for all path operations
- UTF-8 encoding explicitly passed to all file I/O

### Type System
- `Literal` types for enums: `EngineName`, `ExecutionProfile`, `TaskStatus`
- `EventCallback` type alias: `Callable[[str, dict[str, object]], None]` — used for scheduler/runner event callbacks
- `field(default_factory=...)` for mutable defaults in dataclasses
- All function signatures fully annotated (params + return type)
- No `Any` except in serialization boundaries (`to_dict()`)

### Error Handling
- Two custom exceptions: `PlanValidationError` (loader), `TaskExecutionError` (runner)
- Validation errors = immediate exit (fail early)
- Runtime errors = captured in `TaskResult.status` + `.message`
- Timeouts → exit_code=124 (Unix convention)
- `allow_failure: true` → `soft_failed` status (doesn't block dependents)

### Logging
- Console: `print(f"[maestro] ...")` format — no structured logging library
- File: per-task `.log` (transcript) + `.result.json` (structured)
- Run-level: `run_manifest.json` (aggregated results)

### YAML Plan Schema
- Version must be `1`
- Tasks must have unique IDs
- Each task needs exactly one of `command`, `engine`, or `group`
- Engine tasks need a prompt source: `prompt`, `prompt_file`, or `prompt_md_file` + `prompt_md_heading`
- Group tasks specify a sub-plan YAML path: `group: "path/to/sub-plan.yaml"`
- No circular dependencies (DFS validation)
- Template variables: `{{ workspace_root }}`, `{{ plan_name }}`, `{{ task_id }}`, `{{ contracts_summary }}`, `{{ consistency_summary }}`
- `goal`: optional plan-level string injected into engine task prompts; also available as `{{ goal }}`
- Context variables from upstream tasks: `{{ task-id.status }}`, `{{ task-id.stdout_tail }}`, `{{ task-id.exit_code }}`, `{{ task-id.log }}`, `{{ task-id.duration }}`
- Structured context variables (zero cost, auto-extracted): `{{ task-id.files_changed }}`, `{{ task-id.errors }}`, `{{ task-id.warnings }}`, `{{ task-id.decisions }}`, `{{ task-id.result_text }}`, `{{ task-id.summary }}`
- Structured output variables (T1.1): `{{ task-id.output.FIELD }}` — available when the upstream task declares `output_schema` and its output validated successfully; access any top-level field by name
- `context_from` entries must be in `depends_on` (except wildcard `"*"`)
- `context_mode`: `raw` (default), `summarized` (haiku summary per upstream), `map_reduce` (map+reduce synthesis via `{{ upstream_synthesis }}`), `recursive` (index→extract→brief pipeline via `{{ workspace_brief }}`), `layered` (budget-aware L0/L1/L2 tiers — zero LLM cost, 40-65% token savings), `selective` (BM25 chunk-level selection — splits upstream into chunks, scores by keyword relevance to downstream prompt, greedily selects within budget — zero LLM cost, more precise than `raw`), `structural` (code symbol extraction via regex — extracts function/class/import definitions from upstream diff/code, scores chunks by blast radius relevance, greedily selects within budget — zero LLM cost, language-aware for Python/JS/TS/Go/Rust/PHP/Java/Ruby/C/C++; `symbols.py` module; inspired by code-review-graph 6.8× fewer tokens), `council` (multi-model deliberation — N participants discuss over R rounds before task execution; 3 topologies: `star` (all see each other's responses), `chain` (single-pass pipeline, each sees only predecessor), `graph` (visibility constrained by `connections` map); consolidation via haiku; requires `council` block with `participants`, `rounds`, `topology`, `consensus_threshold`, optional `connections` (required for `graph` topology); `council.py` module), `knowledge_graph` (entity extraction — extracts files, functions, classes, decisions, errors, dependencies from upstream output into a typed graph with relationships; formats as structured context within budget; zero LLM cost; `knowledge_graph.py` module; inspired by HippoRAG 2 + MemoRAG), `codebase_map` (interop — reads a pre-built Understand-Anything knowledge graph at `<workspace_root>/.understand-anything/knowledge-graph.json`, scores its nodes by keyword relevance to the task prompt, and injects a budget-bounded codebase map as `{{ upstream_synthesis }}`; zero LLM cost; workspace-derived, requires `workspace_root` (E021); degrades to empty when no graph is present; `codebase_map.py` module; format-interop only — no dependency on Understand-Anything)
- `context_model`: model override for LLM context operations (`summarized`, `map_reduce`, `recursive`); default `haiku`; task-level or via `defaults.<engine>.context_model`; takes priority: task > engine default > haiku; `_resolve_context_model()` in runners.py
- `workspace_index_exclude`: list of glob patterns to exclude from workspace indexing (task + plan defaults level); combined with built-in excludes (`.git`, `node_modules`, `__pycache__`, etc.)
- `context_budget_tokens`: token limit for context operations (task + plan level); auto-summarizes/truncates near threshold; prevents expensive context operations when budget exhausted
- `max_cost_usd`: soft budget limit (float); remaining tasks skipped when exceeded
- `retry_delay_sec`: float (constant) or list[float] (per-retry backoff); plan and task level
- `when`: conditional expression (`{{ task.status }} == value`); changes deps to wait-for-completion semantics
- `judge`: quality gate block with `criteria` (list[str | dict]), `pass_threshold` (0.0-1.0), `on_fail` (fail|warn|retry); structured evaluation via haiku; criteria can be plain strings (LLM-evaluated), typed assertions (`{type: contains/regex/is-json/llm-rubric/cost_under/duration_under/rubric}`), or Likert-scale rubrics (`{type: rubric, name, levels: [{score, description}], min_score, weight}`)
- `judge.method`: `direct` (default), `g_eval` (two-phase: generate evaluation steps then score), `debate` (adversarial multi-round deliberation), or `reflection` (self-critique pass) — all aim to improve consistency over `direct`
- `judge.aggregation`: `mean` (default), `min` (all criteria must pass individually), or `weighted_mean` (per-criterion weights from rubric)
- `judge.preset`: named criteria catalogue (`code_quality`, `security_audit`, `ai_slop_detection`, `cwe_injection`, `cwe_auth`, `cwe_data_exposure`, `cwe_top_25`); provides default criteria, pass_threshold, and aggregation — explicit YAML values override preset defaults; CWE profiles map to specific vulnerability categories (CWE-89/78/79/22 for injection, CWE-287/284/256/384 for auth, CWE-200/327/209 for data exposure); all CWE presets use `aggregation: min` for strictest evaluation; `CWE_SECURITY_PROFILES` constant in models.py for discoverability
- `judge.timeout_sec`: configurable timeout for judge LLM calls (minimum 10s, default auto-scaled); E020 if < 10; when not set, `_compute_judge_timeout()` auto-scales based on method (g_eval=120s, debate=rounds×120s, direct=60s), criteria count (+15s per criterion over 4), and quorum (multiplied); W22 warns when explicit value is below recommended minimum
- `judge.quorum`: integer >= 2 — run N independent judge evaluations; `judge.quorum_strategy`: `majority` (default) | `unanimous` | `any`; E054 (invalid quorum), E055 (invalid strategy), E056 (strategy without quorum)
- `judge.quorum_diversity`: bool (default false) — when true, each quorum slot uses a different model tier (cycles through `JUDGE_DIVERSITY_TIERS`: haiku, sonnet, opus); reduces groupthink by getting genuinely different perspectives; W25 if set without `quorum >= 2`; reasoning summary includes `[model]` tags per judge
- Comparative/pairwise evaluation: on judge retry (`on_fail: retry`), the next attempt is also compared against the previous attempt for relative assessment
- `guard_command`: lightweight output validator — shell command that receives stdout_tail via stdin; exit 0 = pass; runs after verify_command, before judge
- `max_iterations`: hard cap on total task execution attempts (initial + retries + judge retries); prevents infinite retry spirals
- `budget_warning_pct`: plan-level threshold (0.0-1.0, default 0.8) for budget warning events
- `checkpoint: true`: creates `MAESTRO_CHECKPOINT_DIR`, persists progress for long-running tasks, auto-injects checkpoint context on retry
- Error feedback auto-injected into engine task retry prompts when `verify_command` + `max_retries > 0`
- `--resume-last`: automatically finds and resumes the most recent prior run
- `webhook_url`: plan-level URL for POST notification on run completion; also `--webhook URL` CLI flag
- `--output jsonl`: emit structured JSON Lines events to stdout; suppresses all `[maestro]` text output
- `--output tui`: interactive Textual TUI with DAG panel, detail panel, event feed, keyboard navigation, approval modal; install with `pip install maestro-ai-cli[tui]`; requires `textual>=1.0.0,<9.0.0`; does not support multi-plan execution yet
- `escalation`: task-level list of model names for auto-escalation on failure (e.g., `[haiku, sonnet, opus]`); each retry uses the next tier; inheritable via `defaults.<engine>.escalation`
- `fallback_engine`: task-level alternative engine for infrastructure failures (CLI not found, API down, rate limit); only triggers on engine-level failures detected by `_is_engine_failure()`
- `fallback_model`: model to use with `fallback_engine`; inheritable via `defaults.<engine>.fallback_model`
- Per-run `events.jsonl` records timestamped events; all events include `plan_name` field automatically via `_emit()` auto-injection
- **Scheduler events** (scheduler.py): `run_start` (includes `goal`), `task_start`, `task_complete`, `task_skip`, `budget_warning`, `budget_exceeded`, `approval_required`, `approval_response`, `policy_violation`, `judge_result`, `task_checkpoint`, `context_budget_trim`, `context_summarize`, `context_recursive`, `context_compression`, `context_compaction` (`task_id`, `mode`, `max_stage`, `budget_tokens`), `circuit_breaker_tripped`, `taint_detected` (`task_id`, `source`: explicit|propagated), `trajectory_violation` (`task_id`, `violations`, `action`), `council_start` (`task_id`, `participants`, `rounds`, `topology`), `council_complete` (`task_id`, `rounds_completed`, `cost_usd`), `memory_write` (`task_id`, `operation`, `outcome`, `trust_label`, `instructionality_score`, `source_id`), `knowledge_poison_alert` (`task_id`, `signal`, `z_score`, `query_cluster`, `action`), `score_recorded` (`plan_hash`, `quality_score`, `source_id`), `webhook`, `run_complete`
- **Runner events** (runners.py via callback): `task_retry` (`task_id`, `attempt`, `max_retries`), `task_output`, `verify_failure`, `judge_start` (`task_id`, `criteria_count`, `method`), `engine_fallback`, `task_escalation`, `worktree_create`, `worktree_merge`, `worktree_cleanup`, `batch_chunk_complete` (`task_id`, `chunk`, `total_chunks`, `items_in_chunk`, `exit_code`), `task_tool_call` (`task_id`, `tool`, `input_preview`) — emitted per tool_use in claude stream-json events, `deliberation_skip` (`task_id`, `score`, `threshold`) — emitted when deliberation gate decides task is self-answerable, `dynamic_subplan_start` (`task_id`, `sub_plan_name`, `sub_task_count`) — emitted when dynamic_group Phase 2 begins, `dynamic_subplan_complete` (`task_id`, `success`, `sub_task_count`, `total_cost_usd`) — emitted when dynamic sub-plan finishes, `honeypot_triggered` (`task_id`, `triggered_decoys`) — emitted when honeypot decoy access is detected in task output, `phantom_commit` (`task_id`, `files_committed`) — emitted when phantom workspace files are committed to real directory, `population_selected` (`task_id`, `strategy`, `winner`, `candidates`) — emitted when population search picks a winner model
- **Watch events** (watch.py): `watch_start`, `iteration_start`, `iteration_complete`, `metric_recorded`, `regression_detected`, `rollback_executed`, `plateau_detected`, `target_reached`, `watch_step_limit`, `watch_complete`, `stepping_stone_saved` (`iteration`, `metric_value`, `plan_hash`), `stepping_stone_applied` (`metric_value`, `source_iteration`, `source_run`)
- **Replan events** (replan.py): `replan_search_start`, `replan_round_start`, `replan_candidate_generated`, `replan_candidate_blocked`, `replan_candidate_deduplicated`, `replan_candidate_pruned`, `replan_candidate_simulated`, `replan_candidate_selected`, `replan_stepping_stone_saved`, `replan_search_complete` — appended to the root search run's `events.jsonl` during multi-variant replan, continuing the existing hash chain when present

### YAML Anchors and Aliases
- Standard YAML anchors (`&name`) and aliases (`*name`) are supported via `yaml.safe_load()`
- Merge keys (`<<: *name`) work for sharing task defaults and reducing duplication
- Coercion helpers (`_to_str_dict`, `_to_str_list`, etc.) work seamlessly with resolved anchor data
- **Example**: Define common engine settings in an anchor and reuse across multiple tasks:
  ```yaml
  _impl_defaults: &impl_defaults
    engine: claude
    model: sonnet
    edit_policy: efficient
    max_retries: 1

  tasks:
    - id: task-a
      <<: *impl_defaults
      prompt: "Implement feature A"
    - id: task-b
      <<: *impl_defaults
      prompt: "Implement feature B"
    - id: qa-check
      engine: claude
      model: opus
      prompt: "Review implementations"
      depends_on: [task-a, task-b]
  ```

## CLI Commands

```powershell
# Validate plan structure
maestro validate <plan.yaml>

# Validate + audit in one pass with a single exit code (preferred for first-run / CI)
maestro check <plan.yaml> [--json] [--with-suggest] [--strict] [--run-dir DIR]

# Run plan (accepts multiple plans for multi-plan execution)
maestro run <plan.yaml> [plan2.yaml ...] [--parallel] [--dry-run] [--max-parallel N] [--only t1,t2] [--skip t1] [--tags t1,t2] [--skip-tags t1] [--run-dir DIR] [--execution-profile plan|safe|yolo] [--resume PATH | --resume-last] [--verbose | --quiet] [--output text|jsonl|live|tui] [--webhook URL] [--mask-secrets] [--auto-approve] [--set KEY=VALUE]

# Generate CI/CD YAML from a plan
maestro ci <plan.yaml> [--provider github_actions|gitlab_ci|github|gitlab] [--output PATH] [--workflow-name TEXT] [--python-version X.Y] [--test-command CMD]

# Adaptive re-planning (analyze failures + corrected plan search)
maestro replan <plan.yaml> [--max-attempts N] [--model MODEL] [--variants N] [--selection-policy debug_prob|ucb1] [--debug-prob P] [--exploration-constant C] [--population-strategy best|tournament] [--tournament-size N] [--elite-count N] [--diversity-floor F] [--auto-approve] [--dry-run] [--execution-profile plan|safe|yolo] [--verbose | --quiet] [--output text|jsonl|live|tui]

# Scaffold a plan from a brief
maestro scaffold <brief.yaml> [-o output.yaml] [--validate] [--cost-check] [--library NAME_OR_PATH] [--list-libraries] [--strict-defaults]

# Clean up old runs
maestro cleanup <plan.yaml> [--keep N] [--older-than DAYS] [--dry-run]

# Diagnose environment
maestro doctor [--json] [--run-dir DIR]

# Verify event chain integrity
maestro verify <run-path> [--json]

# Security audit a plan
maestro audit <plan.yaml> [--json] [--fix] [--coverage]

# Trace failure causality in a completed run
maestro blame <run-path> [--json]

`maestro doctor` includes custom engine plugin discovery from the
`maestro_cli.engines` entry-point group. Built-in engines stay in the internal registry
and cannot be overridden by entry points.

Typical JSON detail from `maestro doctor --json`:

```json
{"check":"engine_plugins","detail":"no custom engine plugins discovered in 'maestro_cli.engines'","status":"ok"}
{"check":"engine_plugins","detail":"discovered 1 custom engine plugin(s) in 'maestro_cli.engines': acme","status":"ok"}
```

Minimal custom engine contract:

```toml
[project.entry-points."maestro_cli.engines"]
acme = "acme_maestro.plugin:plugin"
```

```python
from maestro_cli.plugins import DoctorProbe, EnginePlugin

plugin = EnginePlugin(
    name="acme",
    build_command=lambda ctx: (["acme-cli", "--prompt", ctx.prompt_text], False),
    doctor_probe=DoctorProbe(executable="acme-cli"),
)
```

The plugin extension path is implemented for v1.0.0, but it is not part of the
frozen `1.x` public contract. Treat it as an extension surface that may be
refined in `1.1.0+`.

### CI generator

- `maestro ci` validates a plan and generates starter YAML for GitHub Actions or GitLab CI
- Supported `--provider` values: `github_actions`, `gitlab_ci`, `github`, `gitlab`
- CLI-exposed knobs are `--output`, `--workflow-name`, `--python-version`, and `--test-command`
- `maestro run --output live` shows a real-time Rich progress table during execution and requires the optional `[live]` dependency extra
- Generated CI keeps normal validate/test lanes offline-first and puts real-engine plan runs behind an explicit manual lane
- Current rendered real-engine jobs are `maestro_real_engine` for GitHub Actions and `run_maestro_real_engine` for GitLab CI

Like the plugin API, the CI generator is implemented but intentionally not frozen
by the v1.0.0 compatibility contract. Any hard freeze for provider behavior is
deferred to `1.1.0+`.

### Tooling posture

- Real-engine coverage lives in `tests/test_e2e_real_engines.py`, is marked `real_engine`, and only runs when `MAESTRO_RUN_REAL_ENGINE_TESTS=1`
- Strict mypy is configured in `pyproject.toml` and runs over the full `src/maestro_cli/` package (`[tool.mypy] files = ["src/maestro_cli/"]`, `strict = true`)
- The current strict mypy invocation is `python -m mypy`
- Deterministic local performance checks live behind the `maestro-benchmark` entry point
- Security guidance is documented in `docs/SECURITY.md` and `docs/SECURITY_BASELINE.md`
- No built-in security scanner or mandatory security audit gate is frozen into the v1.0.0 contract

Broader mypy coverage, benchmark thresholds, and any stronger release gating are
deferred to `1.1.0+`.

# Autonomous metric-driven iteration loop
maestro watch <plan.yaml> [--dry-run] [--execution-profile plan|safe|yolo] [--max-parallel N] [--auto-approve] [--verbose | --quiet] [--output text|jsonl|live|tui] [--mask-secrets] [--resume-last]

# Launch Web UI
maestro ui [--host HOST] [--port PORT] [--no-browser] [--project-root PATH]

# Backfill costs for old runs
maestro backfill-costs [--root PATH] [--run-root PATH] [--dry-run]

# Generate HTML report from a run
maestro report <run-path> [-o output.html]

# Compare two runs
maestro diff <run_a> <run_b> [--json]

# Show cache hit/miss status per task
maestro explain <plan.yaml> [--cache-dir DIR] [--json]

# Show task staleness vs last run
maestro status <plan.yaml> [--cache-dir DIR] [--run-dir DIR] [--json]

# Batch judge evaluation on a completed run
maestro eval <eval.yaml> <run-path> [--json] [--verbose]

# Analyze run history and suggest plan optimizations
maestro suggest <plan.yaml> [--min-runs N] [--json]

# Estimate run cost before running (read-only, offline)
maestro estimate <plan.yaml> [--run-dir DIR] [--set KEY=VALUE] [--json]

# Multi-model interactive chat
maestro chat [--engine ENGINE] [--model MODEL] [--execution-profile PROFILE]

# Launch interactive REPL
maestro shell [--plan <path>]
```

**Execution profiles**:
- `plan` — Use YAML args exactly as written
- `safe` — Strip dangerous flags, add sandbox/approval gates
- `yolo` — Ensure dangerous bypass flags are present

## Model & Reasoning Reference

### Available Engines, Models, and Reasoning Levels

#### Engine: `claude` (Claude Code CLI)

| Model Alias | Full Name | Best For | Cost |
|-------------|-----------|----------|------|
| `haiku` | Haiku 4.5 | Simple tasks, quick checks, linting | $ |
| `sonnet` | Sonnet 4.6 | Daily coding, implementation, reviews | $$ |
| `opus` | Opus 4.8 | Complex reasoning, architecture, agentic coding (latest, since 2026-06) | $$$ |
| `opusplan` | Opus→Sonnet | Plan with Opus, execute with Sonnet | $$-$$$ |

The `opus` alias resolves to the latest Opus shipped via the Claude CLI (currently
Opus 4.8, requires Claude Code v2.1.154+). Pin `claude-opus-4-7` (or `claude-opus-4-6`)
explicitly if you need a previous generation. Opus pricing is unchanged from 4.7.

**Reasoning effort** (Opus 4.6 / 4.7 / 4.8 / Sonnet 4.6 — ignored on Haiku):

| Level | Behaviour | Use When |
|-------|-----------|----------|
| `low` | Most efficient, scoped tasks | Subagents, simple lookups, latency-sensitive work |
| `medium` | Balanced cost/quality | Standard agentic workflows, the average task |
| `high` | Deep reasoning (default for Opus 4.8 / 4.6 + Sonnet 4.6) | Complex coding, nuanced analysis, tool-heavy work |
| `xhigh` | Extended capability for long-horizon work (Opus 4.7 / 4.8) | Agentic coding > 30 min, repeated tool calling, exploratory search. **Default for Opus 4.7; recommended for hard Opus 4.8 coding/agentic** |
| `max` | Absolute maximum capability (Opus 4.6 / 4.7 / 4.8 / Sonnet 4.6) | Genuinely frontier problems where evals show measurable headroom over `xhigh` |

**Opus 4.7 / 4.8 specifics**: adaptive thinking is the only thinking-on mode (extended
thinking removed). Sampling parameters (`temperature`, `top_p`, `top_k`) are
rejected — return 400 if non-default. New tokenizer uses ~1.0–1.35× more tokens
for the same text (varies by content); update `max_tokens` to leave headroom.
1M context window at standard pricing, 128k max output, knowledge cutoff Jan 2026.

**CLI mechanism**: `CLAUDE_CODE_EFFORT_LEVEL=<level>` (env var, injected automatically by Maestro).

#### Engine: `codex` (OpenAI Codex CLI)

| Model ID | Best For | Cost |
|----------|----------|------|
| `gpt-5.5` | Most capable; 1.05M context, 128k output (latest, since 2026-04-23) | $$$$ |
| `gpt-5.4` | Standard frontier coding | $$$ |
| `gpt-5.4-mini` | Responsive coding tasks, subagents | $ |
| `gpt-5.3-codex` | Complex software engineering (last `-codex` suffixed model) | $$ |
| `gpt-5.3-codex-spark` | Near-instant real-time iteration (research preview) | varies |
| `gpt-5.2-codex` | Advanced coding (previous gen) | $$ |
| `gpt-5.1-codex` | Stable, well-tested | $$ |
| `gpt-5-codex-mini` | Fast, lightweight tasks | $ |

> **Naming change**: Starting with `gpt-5.4`, OpenAI dropped the `-codex` suffix
> from canonical model IDs — the standard model now powers Codex CLI workloads
> directly. Maestro's `"5.4"` alias still maps to `gpt-5.4-codex` for backward
> compatibility, but new aliases (`"5.5"`, `"5.4-mini"`) target the unsuffixed
> names. Pin the explicit model ID if you need predictability.

**Reasoning effort** (all models; `none` and `xhigh` are GPT-5.5 additions):

| Level | Behaviour | Use When |
|-------|-----------|----------|
| `none` | No reasoning step (GPT-5.5+) | Pure generation tasks, latency-sensitive |
| `minimal` | Barely reasons, fastest | Trivial edits, file renames |
| `low` | Quick reasoning | Simple implementations |
| `medium` | Default, balanced | Standard coding tasks |
| `high` | Deep analysis | Complex logic, algorithms, refactors |
| `xhigh` | Maximum reasoning | Non-latency-sensitive, hardest problems |

**CLI mechanism**: `-c model_reasoning_effort=<level>` (config override flag, injected automatically by Maestro).

#### Engine: `gemini` (Google Gemini CLI)

| Model Alias | Full Name | Best For | Cost |
|-------------|-----------|----------|------|
| `flash` | gemini-2.5-flash | Fast tasks, budget-friendly | $ |
| `pro` | gemini-2.5-pro | Complex reasoning | $$ |
| `flash-3` | gemini-3-flash-preview | Next-gen fast | $$ |
| `pro-3` | gemini-3.1-pro-preview | Next-gen capable (3-pro-preview retired, redirects to 3.1) | $$$ |
| `pro-3.1` | gemini-3.1-pro-preview | Latest preview | $$$ |
| `flash-lite` | gemini-2.5-flash-lite | Cheapest, simple tasks | $ |
| `auto` | (system routes) | Auto-selection | varies |

Gemini CLI does not expose reasoning effort control. Model routing (Pro vs Flash) serves a similar purpose.

**CLI mechanism**: model set via `-m <model>` flag (injected automatically by Maestro).

#### Engine: `copilot` (GitHub Copilot CLI)

Multi-model access via GitHub Copilot subscription (premium requests). Supports Claude, GPT, and Gemini models.

| Model Alias | Full Name | Provider | Best For |
|-------------|-----------|----------|----------|
| `sonnet` | Claude Sonnet 4.6 | Anthropic | Daily coding, implementation |
| `opus` | Claude Opus 4.6 | Anthropic | Complex reasoning, architecture |
| `opus-fast` | Claude Opus 4.6 (fast) | Anthropic | Fast deep reasoning |
| `haiku` | Claude Haiku 4.5 | Anthropic | Simple tasks, quick checks |
| `gpt-5.4-codex` | GPT-5.4-Codex | OpenAI | Most capable, 1M context |
| `gpt-5.3-codex` | GPT-5.3-Codex | OpenAI | Frontier coding |
| `gpt-5.1-codex` | GPT-5.1-Codex | OpenAI | Stable coding |
| `gemini-pro` | Gemini 2.5 Pro | Google | Complex reasoning |

Copilot CLI does not expose reasoning effort control. Model routing serves a similar purpose.

**CLI mechanism**: `copilot --autopilot --silent --no-color --model <model> -p <prompt>`. Always runs in autopilot mode. `--yolo` for unrestricted execution.

**Environment variables**: `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN` (auth), `COPILOT_MODEL` (default model), `COPILOT_ALLOW_ALL` (yolo mode).

**Cost model**: Subscription-based (premium requests), not per-token. `cost_usd` returns `None`.

#### Engine: `qwen` (Qwen Code CLI)

| Model Alias | Full Name | Best For | Cost |
|-------------|-----------|----------|------|
| `coder` | `qwen-coder-plus` | Standard coding tasks | $$ |
| `coder-turbo` | `qwen-coder-turbo` | Fast coding | $ |
| `max` | `qwen-max` | Most capable, complex tasks | $$$ |
| `plus` | `qwen-plus` | Balanced coding | $$ |
| `qwq` | `qwq-plus` | Reasoning-focused | $$ |

Qwen CLI does not expose reasoning effort control. Model routing serves a similar purpose.

**CLI mechanism**: `qwen --model <model> --prompt "<prompt>"`. `--yolo` for unrestricted execution.

**Environment variables**: `DASHSCOPE_API_KEY` (auth).

**Cost model**: Per-token pricing via `MAESTRO_QWEN_PRICING_JSON` env var override.

#### Engine: `ollama` (Ollama local models)

| Model Alias | Full Name | Best For | Cost |
|-------------|-----------|----------|------|
| `llama3` | llama3 | General purpose | Free |
| `codellama` | codellama | Code generation | Free |
| `mistral` | mistral | Fast general tasks | Free |
| `mixtral` | mixtral | Complex reasoning | Free |
| `phi3` | phi3 | Lightweight tasks | Free |
| `qwen2` | qwen2 | Multilingual | Free |
| `deepseek-coder` | deepseek-coder | Code generation | Free |
| `llama4` | llama4 | General purpose (latest gen) | Free |
| `qwen3-coder` | qwen3-coder | Code generation | Free |
| `deepseek-r1` | deepseek-r1 | Reasoning | Free |

All models run locally — zero API cost. Unknown model names are passed through as-is (supports any model available via `ollama pull`).

**CLI mechanism**: `ollama run <model> "<prompt>"`. No reasoning effort control.

**Environment variables**: `OLLAMA_HOST` (server URL, default `http://localhost:11434`).

**Cost model**: Zero cost (local execution).

#### Engine: `llama` (llama.cpp / llama-cli)

| Model Alias | Full Model Name | Best For | Cost |
|-------------|----------------|----------|------|
| `llama3` | llama-3-8b | General purpose | Free |
| `llama3.1` | llama-3.1-8b | General purpose (updated) | Free |
| `llama3.2` | llama-3.2-3b | Lightweight tasks | Free |
| `codellama` | codellama-13b | Code generation | Free |
| `phi3` | phi-3-mini | Lightweight tasks | Free |
| `mistral` | mistral-7b | Fast general tasks | Free |
| `qwen2.5-coder` | qwen2.5-coder-7b | Code generation | Free |
| `llama4-scout` | llama-4-scout-17b-16e | Latest gen (needs quantization) | Free |
| `llama4-maverick` | llama-4-maverick-17b-128e | Latest gen, larger | Free |

All models run locally via llama.cpp — zero API cost. Unknown model names are passed through as-is. If `LLAMA_MODEL_DIR` is set, relative model paths are resolved against it.

**CLI mechanism**: `llama-cli -m <model> -p "<prompt>" --no-display-prompt`. No reasoning effort control.

**Environment variables**: `LLAMA_MODEL_DIR` (directory containing model files).

**Cost model**: Zero cost (local execution).

### YAML Configuration

#### Plan-level defaults
```yaml
defaults:
  codex:
    model: "5.4"
    reasoning_effort: medium       # minimal|low|medium|high|xhigh
    args: []
  claude:
    model: sonnet
    reasoning_effort: high         # low|medium|high (Opus only)
    args: []
  gemini:
    model: flash
    args: []
  copilot:
    model: sonnet
    args: ["--max-autopilot-continues", "15"]
  qwen:
    model: coder
    args: []
  ollama:
    model: llama3
    args: []
  llama:
    model: llama3
    args: []
```

#### Task-level overrides
```yaml
tasks:
  - id: simple-fix
    engine: claude
    model: haiku                   # Override: use cheaper model
    # reasoning_effort omitted → inherits plan default
    prompt: "Fix the typo in README"

  - id: security-audit
    engine: claude
    model: opus                    # Override: use most capable
    reasoning_effort: high         # Override: max reasoning
    agent: security-engineer
    prompt: "Audit authentication flow"

  - id: codex-hard-problem
    engine: codex
    reasoning_effort: xhigh        # Override: maximum reasoning
    prompt: "Optimize the DAG scheduler"

  - id: copilot-review
    engine: copilot
    model: gpt-5.4-codex           # GPT via Copilot subscription
    agent: code-review
    prompt: "Review the DAG scheduler changes"
```

#### Resolution priority
1. **Task-level** `model` / `reasoning_effort` (highest priority)
2. **Plan defaults** `defaults.{engine}.model` / `defaults.{engine}.reasoning_effort`
3. **Empty** (engine's own default)

### Log Display
Maestro logs show `[model@effort]` tags:
```
[maestro] starting security-audit [opus@high]: Audit authentication flow
[maestro] starting simple-fix [haiku]: Fix the typo in README
[maestro] starting codex-task [gpt-5.4-codex@xhigh]: Optimize scheduler
[maestro] starting copilot-review [gpt-5.4-codex]: Review changes
```

---

## Agent Policy (Cost + Quality)

This policy is mandatory for agents creating or editing Maestro plans.

### Model Routing Strategy

| Task Type | Recommended Engine + Model | Reasoning Effort |
|-----------|---------------------------|-----------------|
| **Trivial fix** (typo, rename, config) | `claude` + `haiku` | — |
| **Standard implementation** (CRUD, views, services) | `claude` + `sonnet` | — |
| **Complex implementation** (algorithms, state machines) | `claude` + `sonnet` | — |
| **Code review** | `claude` + `sonnet` | — |
| **QA validation** | `claude` + `sonnet` | — |
| **Architecture / security audit** | `claude` + `opus` | `high` |
| **Cross-module refactor** | `claude` + `opus` | `medium` |
| **Codex standard task** | `codex` + `5.4` | `medium` |
| **Codex complex problem** | `codex` + `5.4` | `high` or `xhigh` |
| **Codex trivial task** | `codex` + `5.4` | `low` or `minimal` |
| **Gemini standard task** | `gemini` + `flash` | — |
| **Gemini complex task** | `gemini` + `pro` | — |
| **Copilot standard task** | `copilot` + `sonnet` | — |
| **Copilot multi-model** | `copilot` + `gpt-5.4-codex` | — |
| **Copilot quick task** | `copilot` + `haiku` | — |
| **Qwen standard task** | `qwen` + `coder` | — |
| **Qwen complex task** | `qwen` + `max` | — |
| **Qwen fast task** | `qwen` + `coder-turbo` | — |
| **Ollama local task** | `ollama` + `llama3` | — |
| **Ollama code generation** | `ollama` + `codellama` | — |
| **Llama local task** | `llama` + `llama3` | — |
| **Llama code generation** | `llama` + `codellama` | — |

### Recommended Defaults Snippet
```yaml
defaults:
  codex:
    model: "5.4"
    reasoning_effort: medium
  claude:
    model: sonnet
    # reasoning_effort omitted — sonnet doesn't use it
  gemini:
    model: flash
  copilot:
    model: sonnet
    # no reasoning_effort — model routing serves same purpose
  qwen:
    model: coder
    # no reasoning_effort — model routing serves same purpose
  ollama:
    model: llama3
    # local execution — zero cost, no reasoning effort
  llama:
    model: llama3
    # local execution via llama.cpp — zero cost, no reasoning effort
```

### Escalation Rules
Escalate model tier only when at least one condition applies:
- security-critical changes with unclear impact → `opus` + `high`
- concurrency/scheduler invariants at risk → `opus` + `high`
- large cross-module refactors with repeated low-tier failure → `opus` + `medium`
- quality gate failures remain after one scoped retry → escalate model or effort
- codex task fails at `medium` → retry at `high` before escalating to `xhigh`
- if Claude tokens are unavailable, prefer Codex `raw` context plus deterministic
  `verify_command` / `guard_command` before reaching for `summarized`,
  `map_reduce`, or subjective judge flows

### Quality Gates Required for Cost-Optimized Plans
When lower-cost models are used, plans must include verification tasks:
1. test validation task (typically `qa-engineer`)
2. code review task (typically `code-reviewer` and/or `quality-gatekeeper`)
3. explicit pass/fail outcome based on findings

### Agent Roles for This Policy
- `cost-optimizer`: defines model routing and token budget tactics
- `quality-gatekeeper`: validates quality is preserved under lower-cost model selection

## Development

```powershell
# Install in editable mode
cd C:\py\maestro-cli
py -m pip install -e .

# Run directly
py -m maestro_cli validate examples\demo_plan.yaml
py -m maestro_cli run examples\demo_plan.yaml --dry-run

# Run tests (when available)
py -m pytest tests/ -v
```

## File Purposes

| File | Purpose | Key functions |
|------|---------|--------------|
| `cli.py` | CLI entry, argparse (28 shipped subcommands; 16 frozen in the v1 contract), `--version`, ASCII art banner | `main()`, `_build_parser()`, `_print_banner()`, `_parse_set_vars()`, `_cmd_validate()`, `_cmd_check()`, `_cmd_run()`, `_cmd_ci()`, `_cmd_ci_analyze()`, `_cmd_replan()`, `_cmd_scaffold()`, `_cmd_cleanup()`, `_cmd_backfill_costs()`, `_cmd_doctor()`, `_cmd_mcp_server()`, `_cmd_ui()`, `_cmd_report()`, `_cmd_diff()`, `_cmd_explain()`, `_cmd_status()`, `_cmd_eval()`, `_cmd_suggest()`, `_cmd_estimate()`, `_cmd_skill()`, `_cmd_shell()`, `_cmd_chat()`, `_cmd_watch()`, `_cmd_verify()`, `_cmd_audit()`, `_cmd_blame()`, `_cmd_budget()`, `_cmd_export_otel()` |
| `models.py` | Data models (39+ dataclasses) | `PlanSpec`, `TaskSpec`, `TaskResult`, `PlanRunResult`, `TokenUsage`, `JudgeSpec`, `JudgeResult`, `RubricLevel`, `RubricCriterion`, `CriterionScore`, `FailureRecord`, `HandoffReport`, `WorkspaceExtraction`, `WorkspaceBrief`, `PlanBrief`, `TaskBrief`, `PlanImport`, `Suggestion`, `PlanSuggestions`, `SuggestionCategory`, `ShellState`, `ReplanAttempt`, `ReplanState`, `MultiPlanResult`, `WatchSpec`, `WatchIteration`, `WatchState`, `EventRecord`, `VerifyStatus`, `CircuitBreakerSpec`, `PolicySpec`, `PolicyViolation`, `BlameNode`, `BlameChain` |
| `loader.py` | YAML → PlanSpec + validation + matrix expansion + imports + circuit_breaker/retry_strategy parsing | `load_plan()`, `validate_plan()`, `_expand_matrix_tasks()`, `_to_judge_spec()`, `_resolve_imports()` |
| `runners.py` | Build + execute commands + pricing + judge + retry + recursive context + handoff reports + guard_command + typed assertions + rubric eval + G-Eval + comparative eval + score aggregation + secrets masking + auto-escalation + cross-engine fallback + Codex token extraction (4 strategies) | `build_command()`, `execute_task()`, `_classify_failure()`, `_next_escalation_model()`, `_is_engine_failure()`, `_run_judge_evaluation()`, `_evaluate_typed_assertion()`, `_run_guard_command()`, `_evaluate_rubric_criteria()`, `_format_rubric_criteria()`, `_generate_eval_steps()`, `_aggregate_scores()`, `_run_comparative_evaluation()`, `_build_recursive_context()`, `_run_workspace_extraction()`, `_run_workspace_brief()`, `_build_smart_retry_feedback()`, `_generate_handoff_report()`, `_compress_context_for_retry()`, `_build_secret_values()`, `_mask_secrets()`, `_extract_codex_cumulative_usage()`, `_compute_retry_delay()` |
| `scheduler.py` | DAG scheduler + group execution + context budget + intent filtering + BM25 scoring + RRF fusion + priority eviction + graph decay + budget warning + tag filtering + approval gates + run_summary.md test-count surfacing (PM3.2) + retried-tasks section (PM3.1) | `run_plan()`, `_select_tasks()`, `_execute_group_task()`, `_estimate_tokens()`, `_apply_context_budget()`, `_extract_keywords()`, `_apply_intent_filtering()`, `_compute_idf()`, `_score_section()`, `_rrf_score()`, `_compute_hop_distances()`, `_apply_hop_decay()`, `_request_approval()`, `_extract_test_summary()` |
| `replan.py` | Adaptive re-planning + knowledge/history-guided prompts/scoring + multi-variant search rounds + generated-plan security gate + search audit trail | `replan()`, `_replan_single()`, `_replan_search()`, `_extract_failed_state()`, `_build_analysis_prompt()`, `_call_analysis_model()`, `_parse_corrected_yaml()`, `_show_plan_diff()`, `_detect_exit_loop()` |
| `mcts.py` | Phase 3 workflow-search foundations (WorkflowVariant tree, selection, simulation, pruning, backpropagation, tree persistence) | `WorkflowVariant`, `select_leaf()`, `simulate_variant()`, `backpropagate_variant()`, `apply_historical_pruning()`, `append_tree_node()`, `load_tree_index()` |
| `watch.py` | Autonomous watch loop (metric extraction, git ops, experiments, consolidation) | `watch()`, `_extract_metric()`, `_is_improvement()`, `_git_commit_changes()`, `_git_rollback()`, `_load_program()`, `_build_history_text()`, `_write_experiment()`, `_resume_watch_state()`, `_run_consolidation()` |
| `eventsource.py` | Event sourcing: hash chain, replay, verify, output envelope | `compute_event_hash()`, `ChainState`, `emit_hashed_event()`, `replay_events()`, `verify_chain()`, `compute_output_hash()`, `check_scope_violations()`, `build_output_envelope()` |
| `audit.py` | Plan security scanner (23 rules: SEC001-SEC023) + auto-fix | `audit_plan()`, `fix_plan()`, `format_audit()`, `format_audit_json()`, `AuditFinding`, `AuditSeverity` |
| `multi.py` | Multi-plan execution (sequential/parallel + shared budget) | `run_multi_plan()`, `_run_sequential()`, `_run_parallel()`, `_aggregate_results()`, `_write_multi_summary()` |
| `workspace_index.py` | Workspace file tree indexing with caching | `build_workspace_index()`, `load_cached_index()`, `save_index()`, `quick_root_hash()`, `build_tree_summary()`, `FileEntry`, `WorkspaceIndex` |
| `worktree.py` | Git worktree isolation | `create_worktree()`, `merge_worktree()`, `cleanup_worktree()`, `get_base_branch()` |
| `policy.py` | Declarative runtime policy engine | `compile_policy()`, `evaluate_policies()`, `format_violations()`, `format_violations_json()`, `_SafeEvaluator` |
| `blame.py` | Causal failure attribution | `blame_run()`, `format_blame()`, `format_blame_json()`, `_classify_failed_task()`, `_load_events_evidence()` |
| `contracts.py` | Typed contract normalization + template var injection | `normalize_task_contract()`, `build_contract_template_vars()`, `build_consistency_template_vars()`, `_normalize_sql_schema()`, `_normalize_api_schema()`, `_normalize_test_manifest()`, `_extract_primary_output()` |
| `relationships.py` | Consistency groups + dependency resolution | `build_consistency_group_members()`, `resolve_task_dependencies()`, `clone_tasks_with_resolved_dependencies()` |
| `budget.py` | Cross-run budget tracking | `load_budget_ledger()`, `record_budget_entry()`, `check_budget()` |
| `dynamic.py` | Dynamic task decomposition | `build_dynamic_sub_plan()`, `run_dynamic_group()` |
| `knowledge_graph.py` | Entity extraction + graph-based context | `build_knowledge_graph()`, `extract_entities()`, `Entity`, `Relation`, `KnowledgeGraph` |
| `council.py` | Multi-model deliberation (star/chain/graph topologies) | `run_council()`, `_call_participant()`, `_build_round_prompt()`, `_build_consolidation_prompt()`, `CouncilParticipant`, `CouncilSpec`, `CouncilRound`, `CouncilResult`, `CouncilTopology` |
| `symbols.py` | Regex-based code symbol extraction | `extract_symbols()`, `extract_changed_symbols()`, `build_structural_context()`, `detect_language_from_text()`, `detect_language_from_path()` |
| `routing.py` | Semantic model routing + difficulty-aware routing + predictive routing | `resolve_auto_model()`, `_score_task_complexity()`, `_tier_from_score()`, `_COST_WEIGHTS`, `load_task_histories()`, `_apply_historical_signal()` |
| `knowledge.py` | Cross-run knowledge accumulation | `extract_knowledge()`, `load_knowledge()`, `store_knowledge()`, `format_knowledge()`, `consolidate_knowledge()`, `select_relevant_knowledge()`, `ConsolidatedLesson` |
| `fts.py` | Zero-dep SQLite FTS5 ranked full-text search (indexed BM25 lexical ranker, graceful fallback) | `rank_documents()`, `relevance_by_rank()`, `fts5_available()`, `FtsHit` |
| `scaffold.py` | Brief → plan YAML generation + workflow libraries | `scaffold_plan()`, `load_brief()`, `validate_plan_cost_safety()`, `list_workflow_libraries()`, `_load_library()`, `_merge_library_into_brief()`, `WORKFLOW_LIBRARY_NAMES` |
| `suggest.py` | Run history analysis + optimization heuristics | `suggest_plan()`, `format_suggestions()`, `format_suggestions_json()`, `_load_run_history()`, `_analyze_task()` |
| `chat.py` | Multi-model interactive terminal | `run_chat()`, `_run_chat_turn()`, `_build_chat_plan_stub()`, `_build_chat_task_stub()`, `_adjust_command_for_chat()`, `_format_engine_line()`, `_build_history_prompt()`, `_parse_engine_prefix()`, `_dispatch_chat_command()`, `_cmd_model()`, `_cmd_models()`, `_cmd_context()`, `_cmd_save()`, `_cmd_load()`, `_cmd_clear()`, `_cmd_cost()`, `_cmd_help_chat()`, `_setup_chat_readline()`, `_extract_turn_cost()`, `_session_to_dict()`, `_session_from_dict()`, `ChatMessage`, `ChatSession` |
| `shell.py` | Interactive REPL with slash commands | `run_shell()`, `_dispatch_command()`, `_setup_readline()`, `ShellState` |
| `cache.py` | Content-addressable caching | `compute_task_hash()`, `cache_lookup()`, `cache_store()`, `cache_stats()`, `cache_clear()` |
| `diff.py` | Run comparison | `diff_runs()`, `format_diff()`, `format_diff_json()`, `RunDiff`, `TaskDiff` |
| `explain.py` | Cache explain per task | `explain_plan()`, `format_explain()`, `format_explain_json()`, `TaskExplanation`, `PlanExplanation` |
| `status.py` | Pipeline staleness detection | `plan_status()`, `format_status()`, `format_status_json()`, `TaskPipelineStatus`, `PlanPipelineStatus` |
| `estimate.py` | Offline cost preflight (history-aware + token heuristic) | `estimate_plan()`, `format_estimate()`, `format_estimate_json()`, `TaskEstimate`, `PlanEstimate` |
| `eval.py` | Batch judge evaluation + multi-dimensional eval | `load_eval_spec()`, `run_eval()`, `format_eval()`, `format_eval_json()`, `EvalResult`, `EvalSuiteResult`, `DimensionResult` |
| `doctor.py` | Environment diagnostics | `run_doctor()` |
| `report.py` | HTML report generation | `generate_report()` |
| `cost_backfill.py` | Historical cost/token backfill | `backfill_costs()` |
| `cleanup.py` | Run directory cleanup | `cleanup_runs()` |
| `utils.py` | Path, template, markdown utils | `resolve_path()`, `render_template()`, `extract_prompt_from_markdown()` |
| `ag_ui.py` | AG-UI protocol adapter | `translate_event()`, `AgUiRunState`, `format_sse()` |
| `mcp_server.py` | MCP protocol server (12 tools, 8 resources, 3 prompts) | `validate_plan()`, `run_plan_tool()`, `audit_plan()`, `blame_run()`, `diff_runs()`, `explain_plan()`, `plan_status()`, `suggest_plan()`, `doctor()`, `scaffold_plan()`, `verify_events()`, `cleanup_runs()`, `main()` |
| `otel.py` | OTLP exporter (run → OpenTelemetry spans) | `build_span_data()`, `export_to_otlp()`, `format_otel_json()`, `export_run()` |
| `errors.py` | Custom exceptions with error codes | `PlanValidationError(code=)`, `TaskExecutionError(code=)` |

## Output Structure

```
.maestro-runs/
└── <timestamp>_<plan-name>/
    ├── run_manifest.json      # Aggregated results (status, cost, tokens per task)
    ├── run_summary.md         # Human-readable summary with cost/timing/tokens; auto-includes ## Test Results when stdout matches pytest/jest/mocha (PM3.2) and ## Retried Tasks when a task succeeded after failed attempts (PM3.1, surfaces verify_tail per failed attempt)
    ├── events.jsonl           # Structured event log (task start/complete/skip)
    ├── report.html            # Self-contained HTML report (if `maestro report` run)
    ├── <task-id>.log          # Execution transcript
    └── <task-id>.result.json  # Structured result (status, exit_code, duration, cost, token_usage)
```

### Token & Cost Tracking

Each engine task result includes:
- `cost_usd`: direct cost from CLI output, or estimated from token pricing tables
- `token_usage`: `TokenUsage` dataclass with `input_tokens`, `cached_tokens`, `output_tokens`, `cache_creation_tokens`, `total_tokens`

**Cost extraction priority**:
1. Direct cost from CLI (Claude's `total_cost_usd`, Codex's `costUSD`)
2. Token-based estimation using per-engine pricing tables
3. None (shell commands, missing data)

**Pricing tables** (defined in `runners.py`, overridable via env vars):

| Env Var | Engine | Default Models |
|---------|--------|---------------|
| `MAESTRO_CODEX_PRICING_JSON` | Codex | `default` (conservative fallback) |
| `MAESTRO_CLAUDE_PRICING_JSON` | Claude | `haiku`, `sonnet`, `opus`, `opusplan` |
| `MAESTRO_GEMINI_PRICING_JSON` | Gemini | `gemini-2.5-flash`, `gemini-2.5-pro`, etc. |
| (none) | Copilot | N/A — subscription-based premium requests, no per-token pricing |
| `MAESTRO_QWEN_PRICING_JSON` | Qwen | `qwen-coder-plus`, `qwen-max`, etc. |

Override format: `'{"model":{"input_per_million":X,"cached_input_per_million":Y,"output_per_million":Z}}'`

**Plan-level aggregation**: `PlanRunResult.total_tokens` and `PlanRunResult.total_cost_usd`
**Summary output**: `run_summary.md` includes Tokens row in header + Tokens column in task table
**Backfill**: `maestro backfill-costs` also populates `token_usage` for old runs (best-effort engine inference)

## Codex Model Aliases

Short aliases for Codex model names (defined in `models.py:CODEX_MODEL_ALIASES`):

| Alias | Full Model Name |
|-------|----------------|
| `"5.5"` | `gpt-5.5` (latest, since 2026-04-23) |
| `"5.4"` | `gpt-5.4-codex` (kept for back-compat; canonical OpenAI ID is now `gpt-5.4`) |
| `"5.4-mini"` | `gpt-5.4-mini` |
| `"5.3"` | `gpt-5.3-codex` |
| `"5.2"` | `gpt-5.2-codex` |
| `"5.1"` | `gpt-5.1-codex` |
| `"5"` | `gpt-5-codex` |
| `"5-mini"` | `gpt-5-codex-mini` |

Use in YAML: `model: "5.5"` instead of `model: "gpt-5.5"`.

**Pricing per million tokens** (defined in `runners.py:_DEFAULT_CODEX_PRICING_RAW`):

| Model | Input | Cached Input | Output |
|-------|-------|--------------|--------|
| `gpt-5.5` | $5.00 | $0.50 | $30.00 |
| `gpt-5.4` (and `gpt-5.4-codex` via alias) | $2.50 | $0.25 | $15.00 |
| `gpt-5.4-mini` | $0.75 | $0.075 | $4.50 |
| `gpt-5.3-codex` | $1.75 | $0.175 | $14.00 |
| `default` (fallback for any other model) | $2.00 | $0.50 | $8.00 |

Override per-model with `MAESTRO_CODEX_PRICING_JSON`. Prompts > 272K input
tokens are billed at 2× input / 1.5× output by OpenAI for the full session;
override the table if your workloads regularly exceed that threshold.

## Gemini Model Aliases

Short aliases for Gemini model names (defined in `runners.py`):

| Alias | Full Model Name |
|-------|----------------|
| `flash` | `gemini-2.5-flash` |
| `flash-lite` | `gemini-2.5-flash-lite` |
| `pro` | `gemini-2.5-pro` |
| `flash-3` | `gemini-3-flash-preview` |
| `pro-3` | `gemini-3.1-pro-preview` |
| `pro-3.1` | `gemini-3.1-pro-preview` |

Use in YAML: `model: flash` instead of `model: "gemini-2.5-flash"`.

## Copilot Model Aliases

Short aliases for Copilot model names (defined in `runners.py`). Copilot provides multi-model access via GitHub subscription.

**Claude:**

| Alias | Full Model Name |
|-------|----------------|
| `opus` | `claude-opus-4.6` |
| `opus-fast` | `claude-opus-4.6-fast` |
| `opus-4.5` | `claude-opus-4.5` |
| `sonnet` | `claude-sonnet-4.6` |
| `sonnet-4.5` | `claude-sonnet-4.5` |
| `sonnet-4` | `claude-sonnet-4` |
| `haiku` | `claude-haiku-4.5` |

**GPT:**

| Alias | Full Model Name |
|-------|----------------|
| `gpt-5.4-codex` | `gpt-5.4-codex` |
| `gpt-5.3-codex` | `gpt-5.3-codex` |
| `gpt-5.2-codex` | `gpt-5.2-codex` |
| `gpt-5.1-codex` | `gpt-5.1-codex` |
| `gpt-5.1-codex-mini` | `gpt-5.1-codex-mini` |
| `gpt-5.1-codex-max` | `gpt-5.1-codex-max` |
| `gpt-5.2` | `gpt-5.2` |
| `gpt-5.1` | `gpt-5.1` |
| `gpt-5-mini` | `gpt-5-mini` |
| `gpt-4.1` | `gpt-4.1` |

**Gemini & Other:**

| Alias | Full Model Name |
|-------|----------------|
| `gemini-pro` | `gemini-2.5-pro` |
| `gemini-3-pro` | `gemini-3-pro-preview` |

Use in YAML: `model: sonnet` instead of `model: "claude-sonnet-4.6"`. Unknown models are passed through as-is. (Grok was removed: `grok-code-fast-1` was retired from Copilot on 2026-05-15.)

## Qwen Model Aliases

Short aliases for Qwen model names (defined in `runners.py`):

| Alias | Full Model Name |
|-------|----------------|
| `coder` | `qwen-coder-plus` |
| `coder-turbo` | `qwen-coder-turbo` |
| `max` | `qwen-max` |
| `plus` | `qwen-plus` |
| `qwq` | `qwq-plus` |

Use in YAML: `model: coder` instead of `model: "qwen-coder-plus"`.

## Ollama Model Aliases

Short aliases for Ollama model names (defined in `runners.py`):

| Alias | Full Model Name |
|-------|----------------|
| `llama3` | `llama3` |
| `llama3.1` | `llama3.1` |
| `llama3.2` | `llama3.2` |
| `codellama` | `codellama` |
| `mistral` | `mistral` |
| `mixtral` | `mixtral` |
| `phi3` | `phi3` |
| `qwen2` | `qwen2` |
| `qwen2.5-coder` | `qwen2.5-coder` |
| `deepseek-coder` | `deepseek-coder` |
| `deepseek-coder-v2` | `deepseek-coder-v2` |
| `starcoder2` | `starcoder2` |
| `llama4` | `llama4` |
| `qwen3` | `qwen3` |
| `qwen3-coder` | `qwen3-coder` |
| `deepseek-r1` | `deepseek-r1` |
| `deepseek-v3` | `deepseek-v3` |
| `gemma3` | `gemma3` |
| `phi4` | `phi4` |
| `gpt-oss` | `gpt-oss` |

Use in YAML: `model: llama3`. Unknown models are passed through as-is (Ollama supports any model available via `ollama pull`).

Ollama runs locally — zero API cost. Set `OLLAMA_HOST` to override the default server address (`http://localhost:11434`).

## Llama Model Aliases

Short aliases for Llama model names (defined in `models.py`):

| Alias | Full Model Name |
|-------|----------------|
| `llama3` | `llama-3-8b` |
| `llama3.1` | `llama-3.1-8b` |
| `llama3.2` | `llama-3.2-3b` |
| `codellama` | `codellama-13b` |
| `phi3` | `phi-3-mini` |
| `mistral` | `mistral-7b` |
| `qwen2.5-coder` | `qwen2.5-coder-7b` |
| `llama4-scout` | `llama-4-scout-17b-16e` |
| `llama4-maverick` | `llama-4-maverick-17b-128e` |

Use in YAML: `model: llama3`. Unknown models are passed through as-is. Set `LLAMA_MODEL_DIR` to the directory containing model files; relative model names are resolved against it.

Llama runs locally via llama.cpp — zero API cost.

## Environment Variable Isolation

Maestro builds a **clean environment** for each task subprocess.
Only these system variables are inherited (defined in `_ENV_ALLOWLIST` in `runners.py`):

```
PATH, HOME, USER, LOGNAME, SHELL, LANG, LC_ALL, TERM,
USERPROFILE, SYSTEMROOT, SYSTEMDRIVE, COMSPEC, PATHEXT,
TEMP, TMP, APPDATA, LOCALAPPDATA, PROGRAMFILES,
PROGRAMFILES(X86), WINDIR, HOMEDRIVE, HOMEPATH,
PYTHONUTF8, PYTHONIOENCODING,
GEMINI_API_KEY, GOOGLE_API_KEY, GOOGLE_APPLICATION_CREDENTIALS,
GOOGLE_CLOUD_PROJECT, GOOGLE_CLOUD_LOCATION, GOOGLE_GENAI_USE_VERTEXAI,
COPILOT_GITHUB_TOKEN, GH_TOKEN, GITHUB_TOKEN,
COPILOT_MODEL, COPILOT_ALLOW_ALL,
DASHSCOPE_API_KEY,
OLLAMA_HOST,
LLAMA_MODEL_DIR
```

All other variables must be explicitly set via `defaults.env` or `task.env`.

## Important Notes

- `.maestro-runs/` directories are gitignored -- never commit run outputs
- `codex`, `claude`, `gemini`, `copilot`, `qwen`, `ollama`, and `llama-cli` must be on PATH for engine tasks
- The `--yolo` flag is normalized to `--dangerously-bypass-approvals-and-sandbox` for codex
- The `--yolo` flag for Gemini is normalized to `--approval-mode yolo`
- The `--yolo` flag for Copilot is `--yolo` (native); `--allow-all` is normalized to `--yolo`
- Dangerous flags are de-duplicated automatically
- Template variable `{{ workspace_root }}` resolves to empty string if not set in plan
- Windows paths in YAML should use forward slashes (`C:/path/to/dir`)
- `pre_command` failures prevent the main command from running
- `verify_command` runs after main command success/soft_fail; failure marks task as failed
- `max_retries` (0-3) retries main command + verify_command on failure (not pre_command)
- Engine task retries auto-inject verify failure feedback into the prompt (error feedback injection)
- `retry_delay_sec` adds delay between retries: float (constant) or list (per-retry backoff)
- `max_cost_usd` is a soft budget: running task completes, but pending tasks are skipped when exceeded
- `when` expressions change dependency semantics: deps wait for completion (not success); the expression decides execution
- `--resume-last` finds the most recent prior run for the plan and resumes from it
- `requires_clean_worktree` checks `git status --porcelain` in the task's workdir
- `matrix:` defines key→list[str] dict for Cartesian product expansion; generates tasks with IDs `parent@key=val,key=val`
- `{{ matrix.KEY }}` template vars are available in prompt, command, and verify_command for matrix-expanded tasks
- `group:` points to a sub-plan YAML path; executed via recursive `run_plan()` with inherited workspace_root
- Group tasks cannot have `prompt`, `command`, or `engine` — they are a distinct task type
- `cache: false` on a task disables caching for that task; `--no-cache` CLI flag disables globally
- Only `success` results are cached; `soft_failed` and other non-success results are never stored or served from cache
- Pre-hash normalization: `_effective_engine_config()` in cache.py resolves model aliases for all 7 engines (including Claude and Llama), sorts normalized args lists, and uses `sort_keys=True` JSON serialization — semantically identical tasks with different arg order or alias spelling produce the same cache key
- Eviction "why" fields: `_classify_cache_reason()` in cache.py classifies entries into 6 categories (`success`, `negative:timeout`, `negative:rate_limit`, `negative:verify_fail`, `negative:judge_fail`, `negative:generic`); `_cache_why` field stored in every cache entry payload alongside `_cached_at` and `_cache_kind`
- Task result cache storage defaults to `<run-dir>/.cache` with SHA-256
  content-addressed entries
- Workspace indices are stored separately under
  `<workspace_root>/.maestro-cache/index/`
- `webhook_url` must start with `http://` or `https://` (validated in loader); uses `urllib.request` (zero deps)
- Webhook failure never affects the run result (logged as warning)
- `--output jsonl` suppresses all `[maestro]` output, validation warnings, and dry-run checklist
- `--output live` renders a real-time Rich table with task status, duration, cost, and last output line; handles `task_output`, `task_escalation`, `engine_fallback` events; install via `pip install maestro-ai-cli[live]`
- `events.jsonl` is flushed after each write for real-time observability
- `maestro report` generates self-contained HTML (no external deps, works with `file://` protocol)
- Error codes: E001-E072 (validation), E100-E110 (runtime); displayed as `[E001] message`
  - E001: missing required field
  - E002: invalid schema version
  - E003: duplicate task ID
  - E004: circular dependency
  - E005: unknown dependency reference
  - E006: invalid engine name
  - E007: missing prompt source
  - E008: invalid field value (reasoning_effort, edit_policy, etc.)
  - E009: invalid model name
  - E010: invalid context_from reference
  - E011: mutually exclusive fields conflict
  - E012: value out of range (max_retries, max_parallel, etc.)
  - E013: invalid delay specification
  - E014: invalid budget value
  - E015: invalid when expression
  - E016: self-dependency
  - E017: invalid characters in name/ID
  - E018: type mismatch (expected dict, got list, etc.) or unknown fields in `assert:` rules
  - E019: context budget range validation error
  - E020: judge block configuration error (includes `timeout_sec < 10`, unknown fields on typed criteria)
  - E021: context_mode: recursive without resolvable workspace root
  - E022: max_iterations value out of range (< 1)
  - E023: budget_warning_pct value out of range (must be 0.0-1.0 exclusive)
  - E024: invalid secrets configuration
  - E025: import circular dependency or max depth exceeded
  - E026: import file structure invalid
  - E027: duplicate task prefix in imports
  - E028: invalid prefix format in imports
  - E029: approval_message set without requires_approval
  - E030: invalid escalation list (must be non-empty list of model name strings)
  - E031: invalid fallback configuration (fallback_engine must be a known engine; fallback_model requires fallback_engine)
  - E032: watch block missing required `metric` field
  - E033: invalid `metric_direction` or `metric_source` value
  - E034: `metric_pattern` required for stdout_regex, must have exactly 1 capture group
  - E035: `metric_json_path` required for json_field source
  - E036: `max_iterations` must be >= 1
  - E037: `warmup_iterations` must be < `max_iterations`
  - E038: `plateau_threshold` must be >= 1
  - E039: `max_cost_usd` must be positive if set
  - E040: `metric_task` must reference an existing task ID
  - E041: invalid `on_regression` value
  - E042: `program_md` file must exist
  - E043: invalid `plateau_action` value
  - E044: `iteration_budget_sec` must be positive if set
  - E045: `worktree: true` requires resolvable `workspace_root`
  - E046: `worktree: true` not valid on group/command tasks
  - E047: `watch.mode: improve` requires resolvable `workspace_root`
  - E048: invalid `watch.mode` value (must be `custom` or `improve`)
  - E050: invalid `circuit_breaker` configuration (max_failures < 1 or reset_after_sec < 0)
  - E051: invalid `retry_strategy` value (must be constant, linear, or exponential)
  - E052: invalid policy configuration (missing name/rule, bad action, duplicate name, bad rule syntax)
  - E053: invalid `routing_strategy` value (must be cost_optimized, quality_first, or balanced)
  - E054: invalid `judge.quorum` value (must be integer >= 2)
  - E055: invalid `judge.quorum_strategy` value (must be majority, unanimous, or any)
  - E056: `quorum_strategy` requires `quorum` to be set
  - E057: invalid `batch` configuration (missing items/template, empty items, template without `{{ batch.item }}`)
  - E058: `batch.max_per_call` must be >= 1
  - E060: `batch` not allowed on command/group tasks (engine only)
  - E062: `batch` and `matrix` are mutually exclusive on the same task
  - E063: `dynamic_group` requires `engine` + `output_schema`
  - E064: `dynamic_group` conflicts with `group`, `batch`, or `matrix`
  - E065: invalid `context_trust` value (must be `trusted` or `untrusted`)
  - E066: invalid `watch.max_total_steps` value (must be >= 1)
  - E067: invalid `reminders` configuration (missing trigger/message keys, empty values)
  - E068: invalid `context_compaction` value (must be none/standard/progressive)
  - E069: invalid MCP server configuration (missing name, wrong transport/command/url)
  - E070: unknown MCP server reference in `mcp_tools` (not defined in plan `mcp_servers`)
  - E071: `allowed_tools` set on a non-engine task (command or group tasks cannot restrict tools)
  - E072: invalid council `graph` topology connections (missing, invalid roles, empty)
  - SEC022: task consumes contracts via `consumes_contracts` but has no `verify_command` or `guard_command` to validate contract integrity — warning severity
  - SEC023: engine task processes untrusted context without `allowed_tools` restriction — warning severity
  - E100: prompt file not found
  - E101: markdown heading not found
  - E102: unsupported engine
  - E103: no engine specified
  - E104: workdir resolution failed
  - E105: command build failure
  - E106: group sub-plan not found or failed to load
  - E107: judge execution/evaluation runtime error
  - E108: workspace index build failure
  - E109: workspace extraction LLM call failure
  - E110: workspace brief LLM call failure
- Warning codes (loader.py, non-fatal, printed with `[maestro]` prefix):
  - W1: string command uses `shell=True` (cmd.exe on Windows); list commands using wrong Git bash binary (`usr/bin/bash` instead of `bin/bash`)
  - W2: `prompt_md_heading` starts with `#` (loader prepends `## ` automatically)
  - W3: unrecognised template variable (check spelling or add to `_KNOWN_GLOBAL_VARS`)
  - W4: backslashes in path fields (`workspace_root`, `run_dir`, `workdir`, `prompt_file`, etc.)
  - W5: bash-only syntax (heredoc `<<`, process substitution `<()`) in string commands on Windows
  - W6: `retry_delay_sec` list shorter than `max_retries` (last value reused)
  - W7: environment variable reference (`$VAR`) not in env allowlist or task/plan env
  - W8: tag contains whitespace (use hyphens)
  - W13: `fallback_engine` same as primary engine (redundant)
  - W14: `escalation` list has duplicate model names
  - W15: `escalation` set but `max_retries=0` (escalation needs retries)
  - W16: `worktree: true` with only one worktree task (isolation most useful with parallel tasks)
  - W17: high dependency density (edge_density > 60%) — too many cross-task dependencies reduce parallelism
  - W18: low parallelism (sequential depth / task_count > 70%, task_count > 3) — most tasks are sequential; consider restructuring the DAG
  - W19: plan complexity score > 0.8 (S_complex from `compute_plan_density`) — consider splitting into sub-plans or using group tasks
  - W20: task has `max_retries > 0` and no retry escape valve — every attempt runs under identical conditions and likely fails the same way; consolidated 2026-04-26 from legacy W20 + W21 + verify-no-retry chain that contradicted itself (an internal post-mortem); silenced by ANY of: `verify_command` / `guard_command` / `assert` / `judge` (feedback for retries), `escalation` (engine tasks, model upgrade per retry), `fallback_engine` (engine-level swap), positive `retry_delay_sec` (float or list); engine-only valves omitted from message on shell tasks
  - W21: retired 2026-04-26 — folded into the unified W20 above
  - W22: judge timeout insufficient for method/criteria/quorum — warns when explicit `timeout_sec` is too low for the judge configuration (g_eval needs ≥120s, debate needs ≥rounds×120s, quorum multiplies); also informs when auto-scaling kicks in for g_eval with 5+ criteria
  - W23: codex engine task without explicit `reasoning_effort` — user's `~/.codex/config.toml` may inject incompatible value (e.g. `xhigh` on `gpt-5-codex-mini`)
  - W24: `judge.quorum > 3` — LLM consensus reliability degrades beyond 3 evaluators (Byzantine consensus research); recommend `quorum: 3` with `majority` strategy
  - W25: `judge.quorum_diversity` has no effect without `quorum >= 2`
  - W26: tasks have potentially overlapping `output_scope` patterns — raises omission/conflict risk; consider merging them or narrowing scope to preserve the one-task-one-file rule
  - W27: `allowed_tools` validation warnings — unknown tool name for engine, ollama/llama advisory-only, gemini/copilot/qwen system-prompt-enforced
  - W28: council topology warnings — `connections` provided but topology is not `graph`; `rounds > 1` with `chain` topology
  - W29: `fail_fast: true` plans with Codex tasks lacking `fallback_engine` — unsupported-model/account-entitlement failures are only known at runtime and can abort the DAG
  - W30: blocking repo-wide TypeScript compile gates (`tsc --noEmit`) — pre-existing baseline errors can fail the run even when the plan's changes are correct
  - Unnumbered warnings: no explicit `timeout_sec` (default 30min), `verify_command` without `max_retries`, `assert` without `max_retries`, `judge on_fail=retry` without `max_iterations`, `context_from` without `context_budget_tokens`, judge `contains`/`regex` assertion on engine task, `observation_block` without `context_from`, multiline `py -c` in string verify_command, pipe in string command on Windows, non-ASCII in `prompt_md_heading`
- `timeout_adjusted` scheduler event: emitted when workspace-aware timeout estimation (T0.1) increases a task's timeout before dispatch; fields: `task_id`, `original_timeout_sec`, `adjusted_timeout_sec`; formula: `max(plan_default, 300 + file_bytes/3.5 * 0.08)` capped at 3600s; only triggers when plan has `workspace_root` and the task prompt references source files ≥ 10 KB
- `context_mode: recursive` activates the index→extract→brief pipeline; requires `workspace_root` (E021 if missing); uses cheap haiku calls for extraction and brief generation
- `workspace_index_exclude` patterns are combined with `_DEFAULT_EXCLUDES` (`.git`, `node_modules`, `__pycache__`, `.maestro-runs`, `.maestro-cache`, `*.pyc`, etc.); max 5000 files indexed
- Workspace indices are cached in
  `<workspace_root>/.maestro-cache/index/{root_hash}.json` with stat-only quick
  validation (~50ms for 1000 files)
- `{{ workspace_brief }}` template variable is injected when `context_mode: recursive`; contains a focused context document for the agent
- `maestro doctor` checks Python, PyYAML, engine CLIs, custom plugin discovery, Git, and run directory access
- Intent-driven context filtering: `_extract_keywords()`, `_apply_intent_filtering()` in scheduler.py; uses downstream task prompt keywords to score and filter upstream output sections; zero-token, keyword-based scoring
- `context_exceeded` and `rate_limited` failure categories in `_classify_failure()` for context window and API throttling errors
- Smart retry context compression: `_compress_context_for_retry()` compresses upstream context on `context_exceeded` retries; `_CONCISENESS_HINT` injected into retry prompts
- Handoff reports: `HandoffReport` dataclass with failure analysis, partial output, and suggested next steps; `_generate_handoff_report()` creates structured reports after unrecoverable failures
- Context compression metrics: `context_raw_tokens`, `context_final_tokens`, `context_compression_ratio` fields on `TaskResult`; `context_compression` event emitted when filtering reduces context
- Progressive compaction LLM tier: `_apply_progressive_compaction()` in runners.py includes Stage 2.5 (LLM summarization) between section pruning (Stage 2) and truncation (Stage 3); uses `_run_summarization()` with the 9-section structured template + scratchpad-then-strip; respects summarization circuit breaker; `workdir` param enables subprocess calls; scheduler passes `resolve_workdir(plan, task)` through
- Run success calculation includes `"skipped"` tasks — `when`-expression skipped tasks no longer mark the entire run as failed
- Summary format: `N ok / M soft_failed / F failed / S skipped` — `soft_failed` shown as a distinct category (not counted in "ok"); `ok`, `failed`, and `skipped` counts are always printed (even when 0) for reliable regex metric extraction; `soft_failed` only shown when non-zero
- `--resume-last` excludes dependency-failure skipped tasks from the resumed set — they re-run when their dependency succeeds
- Live display shows `--` instead of `$0.00` when no tasks report cost data
- Typed assertion criteria: `judge.criteria` accepts plain strings (LLM-evaluated) or typed dicts (`{type: contains, value: "..."}`, `{type: regex, pattern: "..."}`, `{type: is-json}`, `{type: json-schema, schema: {...}}`, `{type: llm-rubric, value: "..."}`, `{type: cost_under, value: 1.0}`, `{type: duration_under, value: 60.0}`); deterministic checks run at zero cost; `ASSERTION_TYPES` constant in models.py
- Workspace assertions (`assert:` / audit packs) include `file_contains_count` for literal occurrence checks; supports exact `count` or lower-bound `min_count` against a file's substring content
- `json-schema` assertion: validates task output against a JSON Schema definition; supports inline `schema` dict or `schema_file` path; recursive stdlib-only validation (type, properties, required, items, enum, minLength, maxLength); depth limit of 20; `_validate_json_schema()` in runners.py
- BM25-style intent scoring: `_score_section()` uses IDF weighting + TF saturation via `_compute_idf()`; rare terms score higher; backward compatible (falls back to intersection count without IDF)
- RRF Fusion for context eviction: `_rrf_score()` in scheduler.py combines BM25 keyword ranking with graph-distance hop ranking via Reciprocal Rank Fusion (k=60); `_apply_context_budget()` accepts optional `relevance_scores: dict[str, float]` for pre-computed fusion ordering; when both `intent_keywords` and `hop_distances` are available, RRF scores drive eviction priority instead of BM25-only
- Priority-based context eviction: `_apply_context_budget()` trims least-relevant upstreams first (greedy knapsack) instead of proportional trimming
- `guard_command:` field on TaskSpec — shell command that validates output via stdin pipe; exit 0 = pass, non-zero = fail; runs after verify_command, before judge; `_run_guard_command()` in runners.py
- `budget_warning_pct` field (plan + defaults level, default 0.8) — emits `budget_warning` event when running cost approaches `max_cost_usd`
- `max_iterations` per task — hard cap on total attempts (initial + retries + judge retries); prevents infinite retry spirals
- Graph-distance decay: `_compute_hop_distances()` + `_apply_hop_decay()` in scheduler.py; direct deps keep 100%, transitive decay by `0.8^(hops-1)`
- Likert-scale rubrics: `type: rubric` criterion with named levels (1-5), `min_score`, `weight`; `RubricLevel`, `RubricCriterion` dataclasses; `_format_rubric_criteria()`, `_evaluate_rubric_criteria()` in runners.py
- G-Eval two-phase: `judge.method: g_eval` generates evaluation steps then scores; `_generate_eval_steps()`, `_GEVAL_STEPS_PROMPT_TEMPLATE`, `_GEVAL_SCORE_PROMPT_TEMPLATE`; falls back to direct on error
- Score aggregation: `judge.aggregation` (`mean`/`min`/`weighted_mean`); `_aggregate_scores()` in runners.py; `weighted_mean` uses rubric `weight` fields
- Comparative/pairwise eval: `_run_comparative_evaluation()`, `_COMPARATIVE_JUDGE_PROMPT_TEMPLATE`; runs on judge retry comparing new vs previous attempt; `JudgeResult.previous_score`
- Named presets: `judge.preset` (`code_quality`/`security_audit`); `JUDGE_PRESETS` constant; provides default criteria + thresholds; explicit YAML overrides preset
- CLI banner: `_print_banner()` with ANSI 256-color gold/amber gradient; shown on `--help` and no-args; `--version` flag added
- `secrets:` plan field — list of env var names to redact from logs/manifests, or `auto` to detect by name pattern (KEY/SECRET/TOKEN); `--mask-secrets` CLI flag; values ≥3 chars are masked; sorted longest-first to prevent partial masking
- `imports:` plan field — list of YAML paths providing reusable task templates; prefix-based ID namespacing prevents collisions; circular import detection (depth limit 5); nested imports supported; E025-E028 validation
- `tags:` field on TaskSpec — list of semantic labels; `--tags`/`--skip-tags` CLI flags for filtering; dependency auto-inclusion ensures tagged tasks' deps are included; W8 whitespace warning for tags with spaces
- `requires_approval:` field on TaskSpec — interactive pause before task execution; non-interactive mode auto-skips; `--auto-approve` CLI flag bypasses all approval gates; E029 if `approval_message` set without `requires_approval: true`
- `escalation:` field on TaskSpec — list of model names for auto-escalation on failure; each retry attempts the next tier in the list; respects `max_retries`; inheritable via `defaults.<engine>.escalation`; `_next_escalation_model()` in runners.py; E030 validation; W13 warning for escalation without max_retries; `task_escalation` event emitted
- `fallback_engine:` / `fallback_model:` fields on TaskSpec — alternative engine for infrastructure failures (CLI missing, API down, rate limit, unsupported-model/account-entitlement failures); `_is_engine_failure()` detects exit codes 127/9009, auth/quota/rate-limit patterns, and Codex unsupported-model/account-access errors; E031 validation; W14 warning for fallback_model without fallback_engine; W15 warning for fallback_engine same as primary; `engine_fallback` event emitted
- Escalation is disabled after fallback triggers (prevents ValueError from looking up fallback_model in original engine's escalation list)
- Fallback engine args clearing: when `engine_override` changes the engine in `build_command()`, `task.args` are cleared to prevent engine-specific flags from crashing the fallback engine (e.g., codex `--full-auto` passed to claude)
- TUI Phase B2-B5: DetailPanel (live log tail, 500ms polling), keyboard navigation (Up/Down/PgUp/PgDn, Enter/Escape, f=filter, t=follow, q=quit), ApprovalModal (threading.Event bridge for interactive approval), `maestro doctor` checks for textual/rich availability
- `watch:` plan-level block for autonomous iteration loops — `metric` (name), `metric_direction` (lower_is_better/higher_is_better), `metric_source` (stdout_regex/verify_command/guard_command/json_field/manifest), `metric_pattern` (regex with 1 capture group), `metric_json_path`, `metric_task`, `max_iterations`, `warmup_iterations`, `plateau_threshold`, `plateau_action` (stop/escalate_model/notify), `on_regression` (rollback/revert/keep), `program_md`, `max_cost_usd`, `iteration_budget_sec`, `target_metric` (stop when metric reaches target value), `blame_plan` (path to target plan for blame/manifest injection)
- `watch.mode`: `custom` (default, user supplies all config) or `improve` (built-in plan improvement loop); `mode: improve` auto-sets metric=tasks_passed, metric_source=manifest, metric_direction=higher_is_better, warmup=0, on_regression=rollback, plateau_threshold=3; requires `workspace_root` (E047)
- `watch.improve_model`: model for the improve agent (default: sonnet); only used when `mode: improve`
- `metric_source: manifest`: counts tasks with `success`/`dry_run` status directly from `PlanRunResult` — no regex needed; available independently of `mode: improve`
- `mode: improve` two-phase execution: Phase 0 (first iteration) runs target plan for baseline; subsequent iterations run improve agent (modifies plan YAML) then target plan (measures metric); auto-computes `target_metric` from task count; strips target plan's watch block to prevent recursion; overrides fail_fast to false
- `{{ improve.plan_path }}` and `{{ improve.total_tasks }}` template variables injected in improve mode
- `{{ batch.item }}` template variable in `batch.template` — replaced per-item when building chunk prompts
- `{{ watch.lessons }}` — formatted lessons from knowledge archive (time-decayed confidence), injected in improve mode
- `{{ watch.experiments_summary }}` — semantic analysis of experiment history (successes, failures, plateau alerts); auto-populated in watch loops; includes "Approaches that WORKED/FAILED" sections and escalating urgency near plateau threshold; helps improve agent avoid repeating failed strategies
- `{{ improve.frozen_tasks }}` — comma-separated list of frozen task IDs, injected in improve mode
- `{{ task_knowledge }}` — cross-run knowledge auto-injected into engine task prompts (T1.3); formatted bullet list of historical insights (failure patterns, timeout hints, success patterns) with confidence percentages; populated by `knowledge.py` from `.maestro-cache/knowledge/<plan_name>.jsonl`; auto-prepended to prompt as "## Previous Run Insights" section; zero config. `select_relevant_knowledge()` ranks the lexical relevance with SQLite FTS5 BM25 (`fts.py`) when the build supports it — fed the stopword-filtered keyword set, deterministic (`ORDER BY rank, rowid`), tokenizer pinned (`unicode61 remove_diacritics 2`) — and falls back byte-for-byte to the in-Python BM25 otherwise; set `MAESTRO_KNOWLEDGE_FTS=0` to force the legacy ranker. Affects advisory prompt context only — never enters cache keys or the event hash chain.
- `_build_improve_plan()` generates internal 1-task PlanSpec with embedded improvement rules (priority table, fix classification, safety constraints); `_watch_improve()` orchestrates the two-phase loop
- `maestro watch` CLI subcommand — runs watch loop with full output mode support (text, jsonl, live, tui); `--resume-last` to continue from experiments.jsonl
- Watch template variables: `{{ watch.iteration }}`, `{{ watch.best_metric }}`, `{{ watch.last_metric }}`, `{{ watch.history }}` (formatted table), `{{ watch.program }}` (program.md content), `{{ watch.blame }}` (JSON blame analysis from target plan's last run — requires `blame_plan`), `{{ watch.manifest }}` (compact task status summary from target plan's last run — requires `blame_plan`) — injected via `extra_template_vars` parameter
- `extra_template_vars` parameter on `run_plan()`, `execute_task()`, `build_command()`, `_load_prompt()` — merged into template variables before rendering; used by watch for iteration context injection
- `--set KEY=VALUE` CLI flag on `maestro run`, `maestro replan`, `maestro watch` — repeatable; parsed by `_parse_set_vars()` in cli.py; injects into `extra_template_vars`; also passed through TUI via `MaestroApp`; values available as `{{ key }}` in task prompts; CLI vars are base layer — watch/replan internal vars override on collision
- Watch events: `watch_start`, `iteration_start`, `iteration_complete`, `metric_recorded`, `regression_detected`, `rollback_executed`, `plateau_detected`, `target_reached`, `watch_complete`
- `experiments.jsonl` in watch run directory — one JSON line per iteration with metric_value, best_metric, improved, action, cost_usd, duration_sec, git_commit, timestamp
- Watch git operations: commit on improvement (`git add -A` + `git commit`), rollback on regression (`git reset --hard HEAD~1` or `git revert --no-edit HEAD` or keep)
- `WatchSpec`, `WatchIteration`, `WatchState` dataclasses in models.py; `MetricDirection`, `MetricSource`, `OnRegression`, `PlateauAction`, `WatchStatus`, `WatchMode` Literal types; `WatchStatus` includes `"target_reached"` for `target_metric` convergence and `"step_limit_reached"` for `max_total_steps` safety cap
- `watch.max_total_steps`: hard cap on total task executions across all iterations; prevents runaway loops beyond plateau detection; `_count_executed_tasks()` in watch.py counts non-skipped tasks per iteration; `watch_step_limit` event emitted when limit reached; E066 validation (must be >= 1)
- `watch.stepping_stones`: bool (default false) — when true, saves plan YAML + lessons to `.maestro-cache/stepping/<plan>/stones.jsonl` on each improvement; future watch runs start from the best prior stepping stone; `SteppingStone` dataclass in models.py; `_save_stepping_stone()`, `_load_best_stepping_stone()`, `_apply_stepping_stone()` in watch.py; auto-compacts to 20 stones per metric, stores `source_type` + free-form provenance metadata, and shares the same archive with successful multi-variant `replan` completions (`metric_name=replan_fitness`); `stepping_stone_saved`/`stepping_stone_applied` events; respects `metric_direction` when selecting best stone
- `reminders:` per-task field — list of `{trigger, message}` dicts for context-aware retry hints; 4 built-in triggers always active: `repeated_error` (same error 2+×), `timeout` (exit 124), `context_pressure` (token limit keywords), `stuck_loop` (attempt >= 3 same category); custom triggers match as substring in stdout_tail or failure messages; `_evaluate_reminders()` in runners.py evaluates triggers and injects matching messages as `## Reminders` section in retry prompts; E067 validation for invalid config
- `worktree: true` on TaskSpec — runs engine task in isolated git worktree (`.maestro-worktrees/<task-id>`); auto-creates worktree branch `maestro/<task-id>`, merges back on success, reports conflicts on failure, cleans up after
- `WorktreeMergeResult` dataclass: `status` (merged/conflict/empty/error), `files_changed`, `conflict_files`, `merge_commit`, `error`
- `worktree_merge` field on TaskResult: populated after worktree merge attempt
- Worktree events: `worktree_create` (task_id, worktree_path, branch), `worktree_merge` (task_id, status, files_changed, conflict_files), `worktree_verification` (task_id, verified, overlap_ratio, unclaimed_files, phantom_files), `worktree_cleanup` (task_id)
- Dual verification: `verify_worktree_output()` in worktree.py compares actual `files_changed` (from git diff) against agent-claimed file modifications extracted from stdout; `DualVerificationResult` dataclass with `verified` (bool), `overlap_ratio` (0.0-1.0), `unclaimed_files` (changed but not mentioned), `phantom_files` (claimed but not changed); `WorktreeMergeResult.verification` populated after successful merge; threshold default 0.5; inspired by CIBER's textual + environmental dual verification
- `model: "auto"` on engine tasks — triggers semantic model routing via `routing.py`; resolves to concrete model based on task complexity score (tags, prompt length, deps, context mode, judge presence)
- `auto_routed_model` field on TaskResult: the model selected by auto-routing
- `model_routed` event: task_id, engine, requested, resolved, complexity_score, historical_runs — emitted from runners.py via event_callback when `model: auto` resolves
- Routing tier tables per engine: claude (haiku/sonnet/opus), codex (5-mini/5.4/5.4), gemini (flash-lite/flash/pro), copilot (haiku/sonnet/opus), qwen (coder-turbo/coder/max), ollama (phi3/llama3/mixtral), llama (llama-3.2-3b/llama-3-8b/codellama-13b)
- Tag signals for routing: `security`/`architecture`/`critical`/`audit` → high tier (+0.4), `trivial`/`typo`/`config`/`docs` → low tier (-0.3), `review`/`qa`/`refactor`/`complex` → medium boost (+0.2)
- `retry_strategy` field on TaskSpec: `Literal["constant", "linear", "exponential"]` — controls how `retry_delay_sec` scales with attempt number; `constant` (default), `linear` (base × attempt), `exponential` (base × 2^attempt)
- `circuit_breaker:` block on TaskSpec — `CircuitBreakerSpec(max_failures: int, reset_after_sec: float)`; trips after N consecutive failures, auto-resets after cooldown; `circuit_breaker_tripped` event emitted; E050 validation
- Event sourcing: `eventsource.py` provides hash-chained events with SHA-256 tamper detection; `EventRecord` dataclass (sequence, event_type, timestamp, payload, prev_hash, event_hash); `ChainState` tracks chain progress; `replay_events()` reconstructs from `events.jsonl`; `verify_chain()` validates integrity; `maestro verify <run-path>` CLI command
- Security audit: `audit.py` scans plans for security risks; 23 rules: SEC001 (no budget), SEC002 (secrets in prompts), SEC003 (secrets in env), SEC004 (yolo/bypass flags), SEC005 (production paths without approval), SEC006 (no verify_command), SEC007 (exposed API keys), SEC008 (destructive commands without approval), SEC009 (engine yolo without worktree), SEC010 (deep context chain without budget), SEC011 (escalation without cost budget), SEC012 (fallback with yolo propagation), SEC013 (watch loop without bounds), SEC014 (cloud credentials without secrets); `maestro audit <plan.yaml> [--json] [--fix]`
- `maestro audit --fix`: auto-remediation for SEC001 (adds `max_cost_usd: 10.0`), SEC003/SEC014 (adds `secrets: auto`); creates `.yaml.bak` backup; `fix_plan()` in audit.py
- `maestro audit --coverage`: per-category coverage breakdown for SEC001-SEC023 rules; maps built-in rules to 9 risk categories from "Security Considerations for Multi-agent Systems"; `_RULE_CATEGORIES`, `compute_audit_coverage()`, `format_audit_coverage()`, `format_audit_coverage_json()` in audit.py; `AuditFinding.category` field auto-populated; categories: Agent-Tool Coupling, Data Leakage, Injection, Identity/Provenance, Memory Poisoning, Non-Determinism, Trust Exploitation, Timing/Monitoring, Workflow Architecture
- `contract_type`: per-task field declaring what kind of output the task produces; accepted values: `sql-schema`, `dependency-manifest`, `conventions-doc` (extracts heading_count + headings), `file-inventory`, `api-schema` (OpenAPI 3.0/Swagger 2.0 JSON — extracts path_count, schema_count, openapi_version), `test-manifest` (pytest/jest JSON or plain text — extracts passed/failed/skipped/total counts); unknown types get generic fallback (line_count, char_count); `CONTRACT_TYPES` set in models.py; normalizers in contracts.py; `_extract_primary_output()` extracts contract body from task log (skips `[maestro]` header lines, stops at `## ` section markers)
- `consumes_contracts`: per-task list of task IDs whose contracts this task depends on; creates implicit dependency edges (merged into `depends_on` by `resolve_task_dependencies()` in relationships.py); referenced tasks must have `contract_type` set (E018 validation); injects `{{ contract.<id>.producer }}`, `{{ contract.<id>.type }}`, `{{ contract.<id>.summary }}`, `{{ contract.<id>.body }}`, `{{ contract.<id>.hash }}`, `{{ contract.<id>.metadata_json }}` template vars; also injects `{{ contracts_summary }}` (one-line per consumed contract)
- `consistency_group`: per-task string label grouping tasks that must maintain shared invariants; tasks in the same group get `{{ consistency.<group>.tasks }}`, `{{ consistency.<group>.statuses }}`, `{{ consistency.<group>.summaries }}`, `{{ consistency.<group>.contracts }}` template vars; also injects `{{ consistency_summary }}`
- `reconcile_after`: per-task list of consistency group names; creates implicit dependency on all tasks in those groups (resolved by `resolve_task_dependencies()`); the task runs after all group members complete, receiving their contract summaries for cross-cutting validation; E018 validation for unknown group names
- `FailureCategory` expanded to 16 categories (9 original + 7 new): `dependency_missing`, `output_format_error`, `cascading_failure`, `deadlock`, `miscommunication`, `role_confusion`, `verification_gap`; regex patterns in `_FAILURE_PATTERNS`; category-specific remediation in `_FAILURE_REMEDIATION` dict in suggest.py
- Watch consolidation: `consolidate_model`, `consolidate_every`, `consolidate_prompt` fields on WatchSpec; `_run_consolidation()` in watch.py; `{{ watch.consolidated }}` template variable
- Consolidation safety gates: `_run_consolidation()` applies pass-1 `_strip_injection_patterns()`, optional pass-2 `_run_firewall_pass2()` (when `plan.firewall_model` set), and `compute_instructionality()` check before injecting into `{{ watch.consolidated }}`; `consolidate_knowledge()` rejects buckets with avg instructionality >= 0.4 or > 50% untrusted evidence; `ConsolidatedLesson` carries `source_trust_labels` and `avg_instructionality` provenance fields; `load_records_detailed()` in memory.py returns `RecordWithTrust` tuples
- Codex token extraction: `_extract_codex_cumulative_usage()` in runners.py with 4 strategies: (1) `response.completed` events, (2) `turn.completed` events, (3) `item.completed` events, (4) byte-length estimation fallback (output ≈ bytes/4)
- Claude stream-json: `_build_claude_command()` uses `--output-format stream-json` (one JSON event per line, not buffered); `_parse_claude_stream_event(line)` parses event lines; `_extract_stream_json_result_text(output)` extracts the human-readable `result` field from the final `{"type":"result"}` event — used to replace raw JSON events in `stdout_tail` / `feedback_output` / judge / guard_command; `task_tool_call` events fired per `tool_use` item in `assistant` events; `_claude_json_is_success()` and cost/token extraction run on raw tail before replacement; `_adjust_command_for_chat()` in chat.py replaces `stream-json` → `text` for interactive mode
- Policy engine: `policy.py` implements safe AST-based expression evaluation; `compile_policy()` parses rules via `ast.parse(mode="eval")` and builds `_SafeEvaluator`; NEVER uses eval()/exec(); whitelisted task fields (`id`, `engine`, `model`, `tags`, `timeout_sec`, `max_retries`, `allow_failure`, `requires_approval`, `cache`, `description`, `allowed_tools`, `has_allowed_tools`) + computed (`cost_usd`, `has_judge`, `execution_profile`); whitelisted plan fields (`name`, `max_cost_usd`, `max_parallel`, `execution_profile`, `fail_fast`); operators: `==`, `!=`, `<`, `>`, `<=`, `>=`, `and`, `or`, `not`, `in`, `not in`
- `policies:` plan-level block — list of `PolicySpec` with `name` (required, unique), `rule` (required, AST-evaluable expression), `action` (`block`/`warn`/`audit`, default `warn`), `message` (optional); evaluated at task dispatch time in scheduler.py; `block` prevents execution, `warn` emits event + prints, `audit` only emits event; `policy_violation` event (task_id, policy_name, action, message)
- `routing_strategy:` plan-level field — `cost_optimized` | `quality_first` | `balanced` (default None = balanced); affects `model: auto` routing via `_COST_WEIGHTS` table; `cost_optimized` pushes towards cheaper models, `quality_first` towards more capable
- Difficulty-aware routing: `_score_task_complexity()` in routing.py now accepts `routing_strategy` and `dag_metadata` params; DAG metadata includes `fan_out` (>3 → +0.10), `depth` (>4 → +0.05), `upstream_failure_rate` (>0.3 → +0.15); `_compute_task_depth()` and `_compute_fan_out()` helpers in scheduler.py
- Predictive routing (T2.3): `load_task_histories()` reads prior `run_manifest.json` files, aggregates per-task per-model performance (`ModelRecord`: runs, successes, failures, timeouts, avg_duration_sec, avg_cost_usd); `_apply_historical_signal()` adjusts complexity score ±0.20 max based on evidence — cheap model 100% success → −0.15, cheap model ≥50% failure → +0.15, any model ≥40% timeouts → +0.10; confidence scales linearly to `_HISTORY_MIN_CONFIDENCE_RUNS` (5); `_HISTORY_MAX_MANIFESTS` (20) caps I/O; `TaskHistory` dataclass in models.py; scheduler loads histories before dispatch loop; `resolve_auto_model()` accepts optional `evidence: dict` for score/tier/historical_runs export; zero config — automatic when `model: auto` + prior runs exist
- Blame attribution: `blame.py` analyzes `run_manifest.json` + `events.jsonl` to trace failure causality; `blame_run()` walks dependency graph backward from failed tasks; `BlameCategory`: `root_cause`, `dependency_cascade`, `context_corruption`, `timeout_propagation`, `budget_exhaustion`, `unknown`; confidence scores 0.6–0.95; `format_blame()` for human output, `format_blame_json()` for CI; `maestro blame <run-path> [--json]`
- `allowed_tools`: per-task list of tool names the engine agent is allowed to use; also settable via `defaults.<engine>.allowed_tools`; for Claude, maps to `--disallowedTools` (inverse); for Codex, maps to sandbox levels; for other engines (gemini, copilot, qwen), enforced via system prompt injection; for ollama/llama, advisory only (W27); `CLAUDE_TOOLS` constant defines known Claude tools (`Read`, `Write`, `Edit`, `Bash`, `Glob`, `Grep`, `WebSearch`, `WebFetch`, `TodoWrite`); `TOOL_CATEGORIES` provides shorthand categories: `read-only` (no write/edit/bash), `no-shell` (everything except Bash); categories expand to per-engine tool lists via `_expand_tool_categories()`; E071 if set on command/group tasks; `has_allowed_tools` available in policy engine rules
- `context_mode: layered` — budget-aware L0/L1/L2 context tiers: L0 (one-line summary, ~50 tokens), L1 (section headings + key findings, ~200 tokens), L2 (full raw content); zero LLM cost (heuristic extraction); `_build_layered_context()`, `_extract_l0_summary()`, `_extract_l1_sections()` in runners.py; `_L0_TARGET_TOKENS = 50`, `_L1_TARGET_TOKENS = 200` constants; most-relevant upstreams promoted first within budget; 40-65% token savings for tasks with 3+ upstreams
- `control_flow_integrity: true` — plan-level opt-in flag; sandboxes `context_from` content into `<observation>` XML blocks via `_sandbox_observation()` in runners.py; `observation_block: bool` per-task field for granular control; prevents prompt injection via upstream output
- `context_trust: trusted | untrusted` — per-task field; when `untrusted`: upstream output is stripped of common injection patterns (`_strip_injection_patterns()` in runners.py) and auto-wrapped in `<observation>` tags regardless of plan-level CFI; taint propagation: if an upstream task is untrusted, all downstream tasks consuming it via `context_from` inherit `tainted: true` unless they have `guard_command` or `verify_command`; `_compute_tainted_tasks()` in scheduler.py computes taint set via fixed-point iteration; `TaskResult.tainted: bool` tracks taint status; `taint_detected` event emitted at run start; `context_trust` accessible in policy engine rules; E065 validation for invalid values
- SEC015: `when:` expression references unbounded upstream output fields (`stdout_tail`, `log`) — warning severity
- SEC016: `context_from` pulls raw engine output without `guard_command` validation — warning severity; **only triggers on `context_mode: raw`** (refined 2026-04-26 in response to an internal post-mortem); LLM-mediated modes (`summarized`, `map_reduce`, `recursive`, `council`, `knowledge_graph`) and heuristic-extraction modes (`selective`, `layered`, `structural`) are exempt — adding `guard_command` on top of those was the #1 false-positive friction
- SEC017: `context_from` references tasks with external data indicators (user input, web scrape, etc.) but no `context_trust` set — warning severity
- SEC018: task inherits tainted context from upstream via transitive propagation but has no `guard_command` or `verify_command` to sanitize — warning severity
- SEC019: `context_trust: untrusted` without `honeypot: true` — honeypot decoys help detect prompt injection by planting trap values that only injected instructions would access — warning severity
- SEC020: upstream engine task prompt references PII-like fields (email, password, token, etc.) but has no `output_redact` patterns — downstream tasks may receive sensitive data via `context_from` — warning severity
- SEC021: task uses destructive command patterns (`rm -rf`, `DROP TABLE`, `git reset --hard`, etc.) without `phantom_workspace: true` or `requires_approval: true` — warning severity
- `judge.quorum` — run N independent judge evaluations; `quorum: int` (>= 2) on JudgeSpec; `quorum_strategy: majority | unanimous | any`; `_run_judge_quorum()` in runners.py; `QuorumStrategy` Literal type + `QUORUM_STRATEGIES` constant in models.py; majority = more than half pass, unanimous = all pass, any = at least one passes; score averaged across valid evaluations
- Context retrieval trajectory: `maestro explain --context` shows per-task context selection decisions; `explain_context_trajectory()` in explain.py reads `events.jsonl` for context_compression events; `ContextSelectionEntry` (upstream_id, score, keywords_matched, hop_distance, hop_decay_factor, tokens_raw, tokens_final, trimmed, trim_reason) and `ContextTrajectoryReport` (task_id, entries, totals) dataclasses; `context_trajectory` field on `TaskResult`; JSON output via `format_context_trajectory_json()`
- `output_schema:` (T1.1 — structured task outputs) — JSON Schema dict on a task; after success/soft_failed the runner parses stdout_tail as JSON (tries direct parse, then markdown code block, then first `{...}` block) and validates against the schema using `_validate_json_schema()`; if valid: `TaskResult.structured_output` is populated; if invalid: task fails with `"output_schema validation failed: ..."` message; downstream tasks access individual fields via `{{ task-id.output.field_name }}` template variables (nested objects and arrays are JSON-encoded; strings and numbers passed as-is); `_extract_json_from_text()` and `_validate_task_output_schema()` helpers in runners.py; W3 warning suppressed for `task-id.output.*` patterns; `TaskSpec.output_schema` + `TaskResult.structured_output` fields added to models.py

## Pitfalls & Gotchas

See [docs/PITFALLS.md](docs/PITFALLS.md) for a comprehensive list.  Key ones:

### Markdown Prompt Extraction
- `prompt_md_heading` must NOT include `## ` -- the loader prepends it automatically
- Code fences (` ```text `) are preferred; if no fence is found, **all prose text** under the heading is extracted instead
- Use ASCII only in headings -- a Unicode em-dash (the long `—`, U+2014) in a `prompt_md_heading` causes encoding mismatches; use two ASCII hyphens (`--`) instead

### Prompt File Resolution
- `prompt_file` and `prompt_md_file` resolve relative to `workspace_root` first (when set), then fall back to the plan's source directory
- Absolute paths bypass this logic and are used directly

### Windows Shell Execution
- `shell: true` uses `cmd.exe` on Windows -- use list-format commands with Git Bash instead
- Use `C:\Program Files\Git\bin\bash.exe` (NOT `Git\usr\bin\bash.exe`)
- Inside bash scripts, use forward slashes (`/c/projects/...` not `C:\projects\...`)
- MySQL/MariaDB binaries need full paths (e.g., `/c/tools/mysql/bin/mysql`)
- UNC paths (`\\server\share\...`) are not supported by CMD.EXE as working directory -- map to a drive letter or use list-format commands

### Workflow Libraries
- `PlanBrief.library` field references a built-in library name or path to external YAML
- Built-in libraries: `rest-api`, `refactor`, `security-review` (see `WORKFLOW_LIBRARY_NAMES` constant in scaffold.py)
- Library tasks form the base; brief tasks override by matching ID or extend with new IDs
- Library metadata (goal, topology, include_quality_gates, include_build_verify) provides defaults the brief can override
- `--library NAME` CLI flag overrides the YAML `library` field
- `--list-libraries` lists available built-in workflow libraries
- External libraries: any YAML file with `description`, `tasks` list, and optional metadata
- `_merge_library_into_brief()` handles the merge: override index built from brief task IDs, library tasks merged with overrides, extra tasks appended

### Engine Session Conflicts
- Running `maestro run` from inside a Claude Code terminal may fail silently -- the `CLAUDECODE` env var triggers nested session detection in the Claude CLI
- Maestro now warns at preflight and surfaces stderr in failure messages for diagnosis
