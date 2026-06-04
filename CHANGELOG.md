# Changelog

All notable changes to Maestro CLI are documented here.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html). The
v1 compatibility contract is defined in [VERSIONING.md](docs/VERSIONING.md) and
[docs/V1_API_FREEZE.md](docs/V1_API_FREEZE.md).

> **A note on dates.** Maestro was developed privately and entries below carry
> their internal development dates. The project was **first published publicly as
> v2.4.0 on 2026-06-04**; earlier entries document the internal history leading
> up to that release rather than separate public releases.

---

## [Unreleased]

### Added
- **Session memory extraction** — `maestro watch` now extracts durable session snapshots into the plan SQLite memory store, injects `watch.session_memory` plus recent verbatim outputs into improve prompts, reuses the latest snapshot on resume, and prunes stale snapshots so long watch runs retain continuity without unbounded prompt growth.
- **Chat context bootstrap** — `maestro chat` now auto-discovers hierarchical `AGENTS.md` / `CLAUDE.md` files from repo root to cwd, loads them into the existing context-file pipeline at startup, announces the resolved files, and supports opt-out via `--no-auto-context`.
- **Skill registry v2** — `maestro skill` now supports deterministic recommendations driven by explicit skill metadata (`triggers`, `recommended-when`, `recommended-chain`) and updated built-in skill frontmatter, while keeping execution fully inspectable and manual.
- **Web UI collaboration surfaces** — the Web UI now exposes owner/blocker/activity surfaces derived from `run_manifest.json`, `events.jsonl`, and active task results. Run detail includes owners, blocked tasks, recent activity, and richer task identity chips; the dashboard shows compact collaboration summaries per run.
- **Redundant coverage backends** — CI now uploads the same coverage to Coveralls (tokenless, badge in README) alongside Codecov, plus gated-on-secret integrations for SonarCloud (`sonarcloud.yml`) and Codacy. Every uploader is non-blocking, so CI stays green until each service is enabled. Setup steps + badge snippets in [docs/COVERAGE_PLATFORMS.md](docs/COVERAGE_PLATFORMS.md).
- **Tested-guarantees doc + security-audit demo** — new [docs/TESTED_GUARANTEES.md](docs/TESTED_GUARANTEES.md) states honestly which engines have real end-to-end tests (codex/ollama) versus unit+integration only, and what runs in default CI. A new `docs/assets/demo-audit.tape` + `examples/risky_plan.yaml` record a `maestro audit` GIF showing a structurally valid plan flagged as unsafe. Demo GIFs/tapes moved out of the repo root into `docs/assets/`.

### Changed
- **Roadmap baseline** — the full post-`v2.4.0` local-first follow-on tranche is now completed on main: session memory extraction -> chat context bootstrap -> skill registry v2 -> Web UI collaboration surfaces. The next major item remains Remote Execution Backends as a deliberate prototype branch, not a baseline runtime rewrite.

---

## [2.4.0] — 2026-04-09 _(first published publicly 2026-06-04)_

### Added
- **Simulation cache for Phase 3 replan search** — successful `ScoreRecord` entries now persist a model-family-normalized `simulation_plan_hash`, and multi-variant `replan` reuses matching successful simulations with a bounded confidence discount instead of re-executing equivalent topologies. Cache hits are recorded in candidate metadata, `tree.jsonl`, and the replan audit trail via `replan_candidate_cache_hit`.
- **Phase 3 integration benchmarks** — `maestro-benchmark` now includes deterministic `replan_pruning`, `replan_population`, and `replan_novelty` cases. The suite demonstrates saved simulations from historical pruning, higher selected fitness for N>1 tournament search versus single-shot replan, and novelty-driven selection changes that prevent near-duplicate convergence.
- **Knowledge bridge benchmark** — `maestro-benchmark --case replan_guidance` now demonstrates that Phase 2 `failure_pattern` / `model_pattern` guidance plus similar-history bootstrap produce positive priors and reduce the number of candidate simulations needed to reach a successful replan branch.

### Fixed
- **Codex `5.4` alias normalization** — restored the short alias to the canonical `gpt-5.4-codex` target so model resolution stays aligned across `models.py`, runner command building, pricing normalization, and cache hashing.

### Tests
- Added regression coverage for simulation-cache hashing reuse across same-family model variants, cache-hit reuse in multi-variant `replan`, and discounted MCTS backpropagation of cached scores.
- Added a Phase 3 red-team replan regression showing that poisoned `failure_pattern` knowledge can bias a harmful candidate (`rm -rf build`) into generation, but the generated-plan security gate blocks it before simulation and search selects the safe alternative.
- Stabilized time-sensitive memory tests by generating current timestamps in the helper fixture instead of relying on an aging fixed date.
- Forced TUI anyio tests onto the `asyncio` backend so Textual `run_test()` coverage remains deterministic in environments where the `trio` backend lacks a running asyncio loop.

### Documentation
- Release mirrors now reflect `v2.4.0`, and the roadmap marks Phase 3 as completed while keeping Phase 4 explicitly deferred post-release.

---

## [2.3.0] — 2026-04-05

### Added
- **MCTS workflow-search foundations** — new `mcts.py` module with `WorkflowVariant` tree nodes, draft/debug/improve classification, MVP `debug_prob` parent selection, optional UCB1 leaf selection with configurable exploration constant, `run_plan()` simulation helper, historical-pruning consumption, score backpropagation blended with decayed `ScoreRecord` history, and `tree.jsonl` persistence helpers.
- **Multi-variant replan search** — `maestro replan` now supports `--variants N`, `--debug-prob P`, `--selection-policy {debug_prob,ucb1}`, and `--exploration-constant C`; when `N > 1`, replan evaluates multiple candidate corrected plans per failed round, persists the search tree, and uses the configured MCTS leaf-selection policy to continue from the best next variant.
- **Generated-plan security gate for replan** — machine-generated plan variants now inherit Phase 2 safety checks before execution: `replan` audits each generated candidate against the current trusted plan, blocks mutations that introduce new SEC error/warning findings, records per-candidate security verdicts, and applies the trusted plan's optional `firewall_model` pass-2 to generated YAML.
- **Replan search audit trail** — multi-variant `replan` now appends hashed `replan_search_*` / `replan_candidate_*` events to the root search run's `events.jsonl`, so mutation, pruning, simulation, and selection decisions are preserved alongside the normal run event stream with hash-chain continuation.
- **Knowledge-guided replan prompts** — `replan` now retrieves relevant `failure_pattern` and `model_pattern` records from the plan knowledge store, filters retrieval-dominance alerts the same way as runtime task prompting, and injects the surviving hints into analysis prompts as advisory historical guidance.
- **Knowledge-biased variant scoring** — multi-variant `replan` candidates now derive a bounded `knowledge_prior` from retrieved `failure_pattern` / `model_pattern` hints; the prior is persisted in variant metadata and blended into MCTS aggregate scores so tied simulations prefer historically aligned fixes without overriding observed run results.
- **Novelty-biased variant scoring** — multi-variant `replan` candidates now also derive a bounded `novelty_prior` from mutation-signature distance against the current baseline plan and previously persisted `tree.jsonl` variants; this encourages search diversity and helps break ties away from near-duplicate fixes.
- **Tournament population selection for replan** — multi-variant `replan` now supports `--population-strategy tournament` and `--tournament-size N`; when enabled, the next continuation leaf is selected from a sampled pool of candidate leaves using the configured MCTS leaf-selection policy instead of always choosing the global best leaf directly.
- **Replan variant deduplication** — multi-variant `replan` now deduplicates exact `plan_hash` repeats against both the active search tree and historical `ScoreRecord` history. Search-tree duplicates are pruned immediately; historical duplicates reuse prior score evidence for ranking without spending an extra simulation in the same round.
- **Elitism for replan search** — multi-variant `replan` now supports `--elite-count N`; in tournament population mode, top-ranked leaves are preserved in the contestant pool across rounds before challenger sampling, reducing the chance that strong variants disappear due to tournament randomness.
- **Diversity-preserving tournament pools** — multi-variant `replan` now supports `--diversity-floor F`; in tournament mode, challenger leaves are filtered by minimum mutation-signature distance from the current elite/pool before fallback sampling, reducing near-duplicate contestant pools and slowing premature convergence.
- **Cross-run fitness bootstrap for replan** — completed runs now persist compact topology signatures inside `ScoreRecord.metadata`, and multi-variant `replan` derives a bounded `historical_fitness_prior` from similar historical signatures so tied simulations can lean toward variants whose topology resembles previously successful runs without needing exact `plan_hash` matches.
- **Replan stepping-stone persistence** — successful multi-variant `replan` completions now save `replan_fitness` stepping stones into the shared stepping-stone archive, carrying mutation provenance (`selected_node_id`, parent hash, mutation description, priors, fitness gain) alongside the corrected plan YAML so future evolution history is preserved outside `tree.jsonl`.
- **Post-mortem validation guardrails** — `validate` now warns when `fail_fast: true` plans use Codex tasks without `fallback_engine` (runtime account/model entitlement is still unknown offline) and when blocking tasks use repo-wide `tsc --noEmit` gates that can fail on pre-existing baseline errors. `_is_engine_failure()` now also classifies unsupported-model/account-access errors so `fallback_engine` can recover from Codex entitlement mismatches.
- **`--set KEY=VALUE` CLI flag** — inject template variables at runtime for `maestro run`, `maestro replan`, and `maestro watch` (repeatable; e.g. `--set env=prod --set region=eu`); also wired through TUI mode via `MaestroApp`.
- **RRF Fusion for intent filtering** — `_rrf_score()` in `scheduler.py` combines BM25 keyword score with graph-distance hop score via Reciprocal Rank Fusion (k=60) for more balanced context eviction; `_apply_context_budget()` accepts optional `relevance_scores` for pre-computed fusion ordering.
- **Pre-hash normalization in cache** — `_effective_engine_config()` now resolves Claude and Llama model aliases (matching the other 5 engines), sorts normalized args lists for deterministic hashing, and adds an explicit `llama` engine branch; `_resolve_claude_model()` and `_resolve_llama_model()` functions added.
- **Consolidation safety gates** — `ConsolidatedLesson` now carries `source_trust_labels` and `avg_instructionality` provenance fields; `consolidate_knowledge()` rejects buckets with avg instructionality >= 0.4 or > 50% untrusted evidence; `_run_consolidation()` in watch.py now applies pass-1 injection stripping, optional pass-2 `firewall_model` classification, and instructionality scoring before injecting `{{ watch.consolidated }}`; new `load_records_detailed()` in memory.py exposes trust metadata for consolidation.
- **LLM tier in compaction pipeline** — `_apply_progressive_compaction()` now includes Stage 2.5 (LLM summarization via `_run_summarization()`) between section pruning and lossy truncation; uses the existing 9-section structured template + scratchpad-then-strip; respects the summarization circuit breaker; `workdir` parameter added for subprocess calls.
- **Eviction "why" fields in cache** — `_classify_cache_reason()` classifies cache entries into 6 semantic categories (`success`, `negative:timeout`, `negative:rate_limit`, `negative:verify_fail`, `negative:judge_fail`, `negative:generic`); `_cache_why` field stored alongside every cache entry.
- **Semantic Firewalls MVP (MCP path)** — deterministic pass-1 sanitization for MCP metadata, semantic-firewall system prompt for `mcp_tools`, optional `mcp_servers[].description`, and `allowed_task_roles` filtering so tool access can be scoped to declared task roles.
- **Semantic Firewalls pass-2 (opt-in)** — top-level `firewall_model` now enables a lightweight classifier for MCP descriptions and tainted upstream text (`stdout_tail`, structured `result_text`, structured `summary`), with `allow` / `rewrite` / `block` verdicts and fail-open fallback to pass-1 sanitization.
- **MCP concurrency safety metadata** — `mcp_servers[].is_concurrency_safe` can now serialize parallel `worktree: true` tasks that rely on side-effectful MCP servers, reducing the risk of shared-tool writes escaping task-local worktrees.
- **Observability v2 export enrichment** — `maestro export-otel` now emits `gen_ai.*` attributes for engine spans, can optionally capture task input/output previews via `--include-content`, supports privacy-preserving redaction via `--otel-mask-prompts`, and exports `knowledge_poison_alert` events for downstream tracing.
- **Observability v2 memory-write tracing** — post-run knowledge persistence now emits `memory_write` task events with accepted / rejected / quarantined outcomes, instructionality scores, and `source_id` provenance pointers so quarantine decisions and storage failures are visible in OTEL traces.
- **Codebase graph package-aware scoring** — `codebase_graph.py` now normalizes `pkg/__init__.py` as the package module, resolves Tier B package re-exports for `from pkg import symbol` and `pkg.symbol()`, records relative imports as absolute module dependencies in the cached graph, and persists PageRank centrality so structural context ranking can prioritize shared call hubs.
- **Semantic cache tool-failure exclusion** — successful task results with structured tool failures are no longer cached; the runner now records `tool_failure_count` from engine stream events so flaky or partially degraded tool-assisted runs do not become reusable cache entries.

### Tests
- Added regression coverage for `WorkflowVariant` creation, debug/improve selection, scheduler-backed simulation, historical pruning, score backpropagation, and `tree.jsonl` persistence.
- Added regression coverage for optional UCB1 leaf selection and CLI forwarding of `--selection-policy` / `--exploration-constant`.
- Added CLI, model, and orchestration regression coverage for multi-variant `replan` search, selected candidate tracking, approval gating, and search-tree persistence.
- Added regression coverage for novelty-prior scoring and selection of more diverse replan variants when simulation results tie.
- Added regression coverage for tournament-based replan population selection and CLI forwarding of `--population-strategy` / `--tournament-size`.
- Added regression coverage for search-tree duplicate pruning and score-history duplicate reuse in multi-variant `replan`.
- Added regression coverage for elite-preserving tournament selection and CLI forwarding of `--elite-count`.
- Added regression coverage for diversity-aware tournament pool construction and CLI forwarding of `--diversity-floor`.
- Added regression coverage for persisted score-record topology signatures and history-bootstrapped replan selection when simulations tie.
- Added regression coverage for replan-sourced stepping-stone persistence, metric-scoped stepping-stone compaction, and stepping-stone provenance metadata.
- Added regression coverage for Codex unsupported-model engine-failure classification and the new W29/W30 post-mortem validation warnings.
- Added regression coverage for indirect prompt injection patterns in MCP metadata and for role-filtered MCP server access.
- Added runtime regression coverage for pass-2 blocking, tainted structured-context handling, fail-open behavior, and `firewall_model` plan hashing/loading.
- Added loader, serialization, and scheduler regression coverage for `is_concurrency_safe` parsing and worktree concurrency gating.
- Added OTEL and CLI regression coverage for `gen_ai.*` attributes, optional content capture/redaction, and observability export flag plumbing.
- Added memory, scheduler, and OTEL regression coverage for `memory_write` events, provenance pointers, and rejected write instrumentation.
- Added codebase graph regression coverage for package `__init__` re-exports, package-attribute calls, PageRank centrality, and parser cache version updates.
- Added runner, cache, scheduler, and model regression coverage for `tool_failure_count` propagation and semantic-cache exclusion of tool-failure results.
- Added CLI regression coverage for `_parse_set_vars()` (11 tests: valid/malformed/empty/duplicates, parser flag presence on run/replan/watch).
- Added cache regression coverage for `_resolve_claude_model()` and `_resolve_llama_model()` alias resolution (11 tests), args sorting in `_effective_engine_config()` (4 tests).
- Added scheduler regression coverage for `_rrf_score()` (5 tests: empty, single, ranking, disjoint, k-parameter) and `_apply_context_budget()` with pre-computed `relevance_scores` (1 test).
- Added knowledge consolidation regression coverage for provenance fields (`source_trust_labels`, `avg_instructionality`), `to_dict()` serialization, and high-instructionality bucket rejection (3 tests).
- Added cache regression coverage for `_classify_cache_reason()` across all 6 categories and `_cache_why` field presence in stored/negative entries (8 tests).

---

## [2.2.0] — 2026-04-01

### Added
- **Context Pipeline v2** — structured compact templates, scratchpad-then-strip handling, summarization circuit breaker fallback, and post-compact restoration for higher-signal downstream context.
- **Codebase Graph** — `codebase_graph.py` with AST-backed graph extraction, blast-radius analysis, Tarjan SCC detection, and cache-aware graph reuse for code-heavy plans.
- **Wildcard Tool Permissions** — pattern-based `allowed_tools:` entries, built-in categories such as `git-only` / `src-scoped`, and hybrid CLI+prompt enforcement for tighter least-privilege execution.
- **Knowledge + Memory v2 core** — SQLite-backed per-plan memory in `.maestro-cache/memory/<plan>.db` with WAL, automatic JSONL migration, bi-temporal records, provenance/trust labels, point-in-time queries, conflict resolution, relation confidence, prompt-relevant `{{ task_knowledge }}` retrieval, lightweight `{{ knowledge_index }}`, retrieval-dominance quarantine, and poisoning alert persistence.
- **Score history bridge for Phase 3** — `compute_plan_hash()`, `ScoreRecord`, score history persistence/load APIs, historical pruning decisions, `score_recorded` events, and `plan_hash` / `quality_score` fields on full-run manifests.

### Changed
- **Semantic cache core** — cache keys are now policy-versioned, failed runs can be short-lived negative-cache entries via `negative_cache_ttl_sec`, and untrusted / tainted / partial outputs are excluded from cache writes.
- **Scheduler knowledge injection** — task memory retrieval is prompt-relevant rather than task-id-only, and records that trip poisoning alerts are quarantined and removed from subsequent injections in the same run.
- **Release accounting** — the `2.1.x -> 2.2.0` implementation cycle is now folded into an explicit minor release entry instead of a generic `Unreleased` bucket.

### Documentation
- Updated `README.md`, `docs/PLAN_GUIDE.md`, `docs/FEATURE-READINESS.md`, `docs/VERSIONING.md`, `CLAUDE.md`, and `CODEX.md` to reflect the `2.2.0` release and the move from stateless JSONL-only memory to SQLite-backed persistent memory.

---

## [2.1.0] — 2026-03-29

### Added
- **Council Topology Extensions** — `chain` (sequential pipeline, no consolidation) and `graph` (peer-to-peer with explicit `connections:` adjacency). 3 topologies total: star, chain, graph. New `connections` field on CouncilSpec. E072 validation for graph connections. W28 warnings. `council_chain_step` event. 39 new tests.
- **Llama Engine** — 7th engine via `llama-cli` (llama-cpp). Local inference, zero API cost. 7 model aliases (llama3, codellama, phi3, mistral, etc.). `LLAMA_MODEL_DIR` env var. Routing tier table, chat integration. 52 new tests.

---

## [2.0.0] — 2026-03-29

### Added
- **Python SDK** — stable programmatic API via `import maestro_cli`. 25 exports in `__all__`: core functions (`load_plan`, `run_plan`, `validate_plan`, `scaffold_plan`, `blame_run`, `diff_runs`, `audit_plan`), dataclasses, type aliases, exceptions. `py.typed` PEP 561 marker. `EventCallback` type alias. 42 new tests.
- **Capability-Based Tool Access** — `allowed_tools:` per-task list restricting which tools each engine task can invoke. `CLAUDE_TOOLS`, `CODEX_SANDBOX_LEVELS`, `TOOL_CATEGORIES` constants. Claude: translates to `--disallowedTools`. Codex: maps to `--sandbox` levels. Gemini/Copilot/Qwen/Ollama/Llama: system-prompt injection. Policy engine: `has_allowed_tools` + `allowed_tools` fields. Audit: SEC023. E071 validation. W27 warnings. Inheritable via `defaults.<engine>.allowed_tools`. 77 new tests.

---

## [1.37.1] — 2026-03-28

### Fixed
- **mypy strict: 100% source coverage** — expanded from 3 to 61 source files
  under `mypy --strict`. Fixed ~230 type errors across 31 modules (Literal casts,
  type narrowing, missing annotations). Caught 5 latent runtime bugs:
  - `runners.py`: `_run_guard_command` called with wrong arguments in batch path
  - `watch.py`: undefined variable `run_dir` (should be `watch_run_path`)
  - `mcp_server.py`: called `diff.to_dict()` which doesn't exist on `RunDiff`
  - `tui/widgets.py`: `_running` name conflict with Textual base class
  - `cli.py`: dead code referencing non-existent `tui.create_tui_callback`
- **Dead code removal** — removed unimplemented TUI watch path (`maestro watch
  --output tui` now shows a clear "not yet supported" error instead of crashing)
- **pyproject.toml** simplified to `files = ["src/maestro_cli/"]` (covers all modules)

### Added
- **+622 tests** (10949 → 11571) — targeted coverage push across 11 modules:
  - `web/routes_agui.py`: 27% → 97% (+41 tests)
  - `tui/widgets.py`: 63% → 99% (+128 tests)
  - `council.py`: 64% → 98% (+26 tests)
  - `doctor.py`: 66% → 95% (+39 tests)
  - `tui/app.py`: 71% → 87% (+128 tests shared with widgets)
  - `watch.py`: 73% → 86% (+21 tests)
  - `otel.py`: 75% → 96% (+20 tests)
  - `shell.py`: 77% → 97% (+8 tests)
  - `cli.py`: 78% → 98% (+51 tests)
  - `runners.py`: 82% → 86% (+211 tests)
  - `scheduler.py`: 84% → 93% (+77 tests)
- **v2.0 phased roadmap** — formalized into 5 dependency-ordered phases
  (Foundation → Execution → Intelligence → Search → Interop) with
  dependency graph, YAML examples, and per-feature research references
- **121 research references** across 14 categories in ROADMAP.md (was ~79)

---

## [1.37.0] — 2026-03-25

### Added
- **Knowledge Graph Context Mode** (`context_mode: knowledge_graph`) — extracts
  structured entities (files, functions, classes, decisions, errors, dependencies)
  from upstream task output into a typed graph with relationships. Downstream tasks
  receive a focused, structured context instead of raw text. New module
  `knowledge_graph.py` with `Entity`, `Relation`, `KnowledgeGraph` dataclasses,
  `extract_entities()` regex-based extractor, `build_knowledge_graph()` builder,
  and multi-hop graph traversal (`get_related()`, `subgraph()`). Zero LLM cost.
  Inspired by HippoRAG 2 (associative retrieval) and MemoRAG (global memory).
  27 new tests.

- **Meta-Policy Reflexion** — persistent failure rules extracted from judge
  failures and retry-then-success patterns. New `policy_rule` KnowledgeKind
  that captures actionable lessons (e.g., "watch for SQL injection", "check
  syntax before commit"). Rules auto-accumulate in `.maestro-cache/knowledge/`
  and are injected into future task prompts via `{{ task_knowledge }}`.
  Inspired by Meta-Policy Reflexion (reusable reflective memory). 4 new tests.

---

## [1.36.0] — 2026-03-25

### Added
- **Council Mode** (`context_mode: council`) — multi-model deliberation
  before task execution. N participants discuss over R rounds (star topology),
  then a consolidation step synthesizes consensus for the downstream task.
  New module `council.py` with `CouncilParticipant`, `CouncilSpec`,
  `CouncilRound`, `CouncilResult` dataclasses and `run_council()` orchestrator.
  - `council:` block on tasks with `participants` (list of engine/model/role),
    `rounds` (1-5), `topology` (star), `consensus_threshold` (0.0-1.0)
  - Star topology: all participants see prompt + each other's responses per round
  - Consolidation via haiku (cheap synthesis of all perspectives)
  - Upstream context passed to participants when task has `context_from`
  - Events: `council_start`, `council_turn`, `council_consolidation`, `council_complete`
  - Loader validation: requires 2+ participants, valid engines, rounds 1-5
  - 20 new tests

---

## [1.35.0] — 2026-03-25

### Added
- **Chat enhancements** — three new slash commands for `maestro chat`:
  - `/context <path...>` — add file(s) to conversation context; content is
    prepended as `<file_context>` block in every turn prompt; supports
    multiple files, `--clear` to remove all, truncation at 50K chars
  - `/save` — persist session (messages, context files, cost, settings)
    to `.maestro-cache/sessions/chat_<timestamp>.json`
  - `/load [path]` — restore a saved session; loads latest if no path given;
    restores messages, context files, engine/model, cost tracking
  - `ChatSession.context_files` field (dict of path→content)
  - Session serialization: `_session_to_dict()`, `_session_from_dict()`
  - `_build_history_prompt()` now injects file context before conversation
    history, ensuring models see referenced files every turn
  - Tab completion includes new commands automatically
  - 18 new tests

### Fixed
- Roadmap: marked Layered/Selective context modes, Plan Density Score, and
  Deliberation Gate as shipped (were using inconsistent notation)

---

## [1.34.0] — 2026-03-25

### Added
- **Structural Context Mode** (`context_mode: structural`) — regex-based code
  symbol extraction for blast radius filtering. Extracts function, class, import,
  and type definitions from upstream diff/code output using language-aware patterns
  (Python, JavaScript, TypeScript, Go, Rust, PHP, Java, Ruby, C, C++). Scores
  downstream context chunks by symbol reference density and greedily selects
  within budget. Zero LLM cost. Inspired by code-review-graph (6.8× fewer tokens,
  8.8/10 vs 7.2/10 review quality). New module `symbols.py` with `Symbol`
  dataclass, `extract_symbols()`, `extract_changed_symbols()`,
  `build_structural_context()`, and language auto-detection (diff headers, code
  fences, shebangs, keyword density). 54 new tests.

---

## [1.33.0] — 2026-03-25

### Added
- **Stepping Stones Archive** (`watch.stepping_stones: true`) — when a watch
  iteration improves the metric, the full plan state is snapshotted to
  `.maestro-cache/stepping/<plan>/stones.jsonl`. Future watch runs start from
  the best prior stepping stone, inheriting the plan YAML and lessons. Auto-
  compacts to 20 stones. `_save_stepping_stone()`, `_load_best_stepping_stone()`,
  `_apply_stepping_stone()` in watch.py. `SteppingStone` dataclass in models.py.
  Events: `stepping_stone_saved`, `stepping_stone_applied`.
- **Consistency Groups documentation** — contracts.py and relationships.py added
  to CLAUDE.md architecture diagram and file purposes table. Full documentation
  of `consumes_contracts`, `consistency_group`, `reconcile_after` fields and
  their template variables (`{{ contract.<id>.* }}`, `{{ consistency.<group>.* }}`).
- **PLAN_GUIDE.md** — fixed `conventions-doc` metadata (was "N/A", now
  `heading_count` + `headings`). Added generic fallback behavior note and
  contract body extraction explanation.
- 21 new tests (stepping stones).

---

## [1.32.0] — 2026-03-25

### Added
- **Quorum Diversity** (`judge.quorum_diversity: true`) — each quorum slot uses a
  different model tier (cycles through haiku → sonnet → opus), reducing groupthink
  and producing genuine disagreement diversity. Reasoning summary includes `[model]`
  tags per judge. W25 warning if set without `quorum >= 2`. New constant
  `JUDGE_DIVERSITY_TIERS` in models.py.
- **Meta-Watch Adaptive Improvement Prompt** — when `mode: improve` encounters
  plateau pressure, the improve agent receives a semantic analysis of experiment
  history via `{{ watch.experiments_summary }}`. Includes "Approaches that WORKED"
  and "Approaches that FAILED (do NOT repeat)" sections, with escalating urgency
  near the plateau threshold ("CRITICAL: last chance — try a fundamentally different
  approach"). New function `_build_experiments_summary()` in watch.py.
- **10 new research references** in ROADMAP.md: HippoRAG 2, LightMem, Reflexion,
  MINJA, BIPIA, ST-WebAgentBench, Darwin Gödel Machine, WISE-Flow, Terminal-Bench
  2.0, τ²-Bench-Verified.
- 24 new tests (12 quorum diversity + 12 Meta-Watch).

---

## [1.31.1] — 2026-03-24

### Added
- **W24: `quorum > 3` warning** — LLM consensus reliability degrades beyond 3
  evaluators (informed by Byzantine consensus research, ETH Zurich). Emits a
  loader warning recommending `quorum: 3` with `majority` strategy.
- **Timeout-aware quorum voting** — error/timeout judge evaluations are now
  excluded from the quorum vote instead of counting as failures. A quorum of 3
  where 1 judge times out now votes on the 2 valid results (2/2 pass = unanimous
  pass, 1/2 pass = majority fail). Previously, any error forced `unanimous` to
  fail even when all valid evaluations passed.
- **ROADMAP.md** — 7 new research-inspired items: Quorum Reliability (W24 +
  timeout-aware), Meta-Watch (HyperAgents), Structural Context Mode
  (code-review-graph), Quorum Diversity, 3 new research references

### Fixed
- **Unknown fields on `assert:` rules now rejected** — workspace assertion rules
  with unrecognised fields (e.g. `negate: true`) are caught at validation time
  (E018) instead of being silently dropped. Previously, `{type: file_contains,
  negate: true}` would silently ignore `negate` and run a positive check —
  the opposite of the author's intent.
- **Unknown fields on typed judge criteria now rejected** — judge criteria dicts
  with unrecognised fields (e.g. `{type: contains, value: "x", negate: true}`)
  are caught at validation time (E020). Each criterion type validates its own
  known field set: `contains` (`value`), `regex` (`pattern`/`value`),
  `json-schema` (`schema`/`schema_file`), `rubric` (`name`/`levels`/`min_score`/`weight`), etc.
- **PITFALLS.md** — new entry #34: Unknown Assert Fields Silently Invert Logic
- **AGENT_OPS.md** — pre-flight checklist: 2 new verification items for assert field validation
- **PLAN_GUIDE.md** — workspace assertions section: documented allowed fields and E018 rejection

---

## [1.31.0] — 2026-03-22

### Added
- **PLAN_GUIDE.md expansion** — new sections: `context_mode: selective` (usage,
  comparison with layered/compaction), Trajectory-Level Guardrails (detection
  patterns, actions, `tool_call_count`), Contracts best practices (when to use
  contracts vs context_from, SEC022 guidance), E068-E070 error codes table
- **`maestro doctor --full`** — extended integration checks: cache directory,
  knowledge store, skill registry discovery, plan scanning in cwd, prior run
  count; 5 new check functions in doctor.py
- **Freeze promotions** — `maestro ci` provider surface and custom engine plugin
  API promoted to frozen v1.x contract in V1_API_FREEZE.md; `layered` and
  `selective` context modes added to frozen enumerations

### Fixed
- **TUI DetailPanel rendering** — replaced `Static.update()` pattern with
  `render()` method + `refresh()` calls; fixes blank right panel when selecting
  tasks in `--output tui` mode

---

## [1.30.0] — 2026-03-22

### Added
- **Consistency Groups polish** — `contract_type` and `has_consistency_group`
  added to policy engine whitelist; SEC022 audit rule (contract consumer
  without verify_command or guard_command)
- **`context_mode: selective`** — BM25 chunk-level context selection; splits
  upstream output into fixed-size chunks, scores each by keyword relevance to
  downstream prompt, greedily selects highest-scoring within budget; zero LLM
  cost, more precise than `raw`, cheaper than `summarized`;
  `_build_selective_context()` and `_score_chunk_bm25()` in runners.py

### Changed
- **ROADMAP.md cleanup** — marked 11 shipped features with strikethrough,
  updated release history table (v1.26-v1.29)

### Stats
- Tests: 10691 → 10710 (+19 new)
- New audit rule: SEC022
- New context mode: `selective`
- New policy fields: `task.contract_type`, `task.has_consistency_group`

---

## [1.29.0] — 2026-03-22

### Added
- **MCP-Native Tool Orchestration** — plan-level `mcp_servers` block declares
  MCP server providers (stdio/http/sse transports); task-level `mcp_tools`
  references servers by name; `MCPServerSpec` dataclass with name, command, url,
  transport, env, timeout_sec; `_build_mcp_config()` generates temporary JSON
  config for Claude CLI `--mcp-config` flag; E069 (invalid server config), E070
  (unknown server reference) validation; duplicate server name detection

### Stats
- Tests: 10667 → 10691 (+24 new)
- New error codes: E069, E070
- New dataclass: `MCPServerSpec`
- New fields: `mcp_servers` on PlanSpec, `mcp_tools` on TaskSpec

---

## [1.28.0] — 2026-03-22

### Added
- **Adaptive Temporal Routing** — trend detection (`_detect_trend()` —
  improving/degrading/stable from recent outcomes), cross-task affinity
  (`_compute_task_similarity()` — engine/tags/judge/context_mode matching,
  `apply_cross_task_routing()` — transfers model knowledge between similar
  tasks), `recent_outcomes` field on `ModelRecord`, recency-weighted history
  accumulation; Rule 5 in `_apply_historical_signal()`: degrading model trend
  pushes toward stronger tier
- **Population-Based Search** — `population` block on tasks with `candidates`
  (list of models), `strategy` (best/first_passing/majority), `parallel` flag;
  `PopulationSpec` dataclass; `_run_population_search()` dispatches N model
  variants and selects winner by judge score, success rate, or cost;
  `population_selected` event emitted with winner details

### Stats
- Tests: 10639 → 10667 (+28 new)
- New events: `population_selected`
- New dataclasses: `PopulationSpec`
- New functions: `_detect_trend()`, `_compute_task_similarity()`,
  `apply_cross_task_routing()`, `_run_population_search()`

---

## [1.27.0] — 2026-03-22

### Added
- **Skill Registry** — `skill_registry.py` module with `discover_skills()`,
  `search_skills()`, `SkillEntry` dataclass; parses `.claude/skills/*/SKILL.md`
  YAML frontmatter; `maestro skill list|search` CLI subcommand with `--query`,
  `--json`, `--dir` flags; keyword-based search with name/description/tag scoring
- **Knowledge Consolidation & Compaction** — `consolidate_knowledge()` aggregates
  repeated patterns into strategic `ConsolidatedLesson` dataclass; `compact_knowledge()`
  removes low-confidence/duplicate records with time-decay; `format_consolidated_lessons()`
  for human output; configurable `min_occurrences` threshold
- **CI Agentic Workflows** — `ci_agent.py` module with `analyze_ci_failure()`,
  `CiFailureAnalysis` and `CiRemediationAction` dataclasses; automatic failure
  categorization (timeout → increase_timeout, rate_limited → add_delay,
  context_exceeded → progressive compaction, test_failure → escalate_model);
  blame-based root cause identification; `maestro ci-analyze` CLI subcommand

### Stats
- Tests: 10637 → 10639 (+34 new, -32 deduped from prior)
- New modules: `skill_registry.py`, `ci_agent.py`
- New CLI subcommands: `maestro skill`, `maestro ci-analyze`

---

## [1.26.0] — 2026-03-22

### Added
- **Staged Progressive Compaction** — 5-stage context degradation pipeline
  (`context_compaction: progressive|standard|none` on task/defaults);
  stages: structural compaction → section pruning → truncation with markers →
  L1 extraction → L0 summary; applied sequentially until within budget;
  activates dormant `_compact_context()` function; `context_compaction` event
  emitted; E068 validation; backward compat with `context_compact: true`
- **Privacy-Aware Context Pipeline** — `output_redact` per-task regex patterns
  to strip sensitive data before downstream consumption; `context_allowlist`
  restricts which upstream fields downstream tasks can access; SEC020 audit
  rule (PII-like prompts without redaction); `_redact_output()` and
  `_filter_context_fields()` in runners.py
- **Trajectory-Level Guardrails** — `trajectory_guard` block on tasks with
  `max_tool_calls`, `max_retries_without_progress`, `scope_pattern` (regex),
  `on_violation` (warn/abort/escalate); `TrajectoryGuardSpec` dataclass;
  `tool_call_count` field on TaskResult; `trajectory_violation` event;
  real-time evaluation after task completion in scheduler
- **Phantom Output Interception** — `phantom_workspace: true` routes file
  operations to shadow directory; on success, files committed to real target;
  on failure, auto-cleanup; SEC021 audit rule (destructive commands without
  phantom or approval); `phantom_commit` event; inspired by Kavach pattern

### Stats
- Tests: 10570 → 10637 (+67 new)
- New error code: E068
- New audit rules: SEC020, SEC021
- New events: `context_compaction`, `trajectory_violation`, `phantom_commit`
- New dataclasses: `TrajectoryGuardSpec`
- New fields: `context_compaction`, `output_redact`, `context_allowlist`,
  `trajectory_guard`, `phantom_workspace`, `tool_call_count`

---

## [1.25.0] — 2026-03-22

### Added
- **Workflow Libraries** — reusable task template catalogs for scaffold;
  3 built-in libraries (`rest-api`, `refactor`, `security-review`);
  `PlanBrief.library` field references built-in name or external YAML path;
  library tasks form base, brief tasks override by ID or extend with new IDs;
  library metadata (goal, topology, quality gates) provides defaults
- **`--library` flag** on `maestro scaffold` — CLI override for library selection
- **`--list-libraries` flag** on `maestro scaffold` — lists available built-in
  workflow libraries with descriptions
- **External workflow libraries** — custom YAML files with `description`,
  `tasks`, and optional metadata; loaded via `library: path/to/file.yaml`
- Example files: `examples/library-brief.yaml`, `examples/custom-workflow-library.yaml`
- **CWE Security Profiles** — 4 new judge presets mapped to CWE vulnerability
  categories: `cwe_injection` (CWE-89/78/79/22), `cwe_auth` (CWE-287/284/256/384),
  `cwe_data_exposure` (CWE-200/327/209), `cwe_top_25` (broad OWASP coverage);
  all use `aggregation: min` for strictest evaluation; `CWE_SECURITY_PROFILES`
  constant for discoverability
- **Playbook** — `docs/PLAYBOOK.md` with 7 curated recipes for common tasks
  (feature implementation, refactoring, security audit, test backfill, bug fix,
  multi-engine, watch mode); anti-patterns section; cost optimisation checklist;
  verification stack guide; based on 41+ real runs totalling ~$297
- **2 new workflow libraries** — `bug-fix` (reproduce + fix + regression) and
  `test-backfill` (parallel test writing + coverage gate); now 5 built-in
  libraries total; `playbook_ref` field cross-references Playbook recipes
- **Multi-Dimensional Eval** — `dimensions` block in eval YAML for independent
  assessment across correctness, security, efficiency, etc.; `DimensionResult`
  dataclass with per-dimension pass/fail/score; each dimension can have its own
  judge spec, task patterns, and exclusions; `EvalSuiteResult.dimensions` field;
  `format_eval` shows dimension breakdown; `format_eval_json` includes dimension
  data; `overall_pass` requires all dimensions to pass; MASEval-inspired
- **Security Contracts (Envelope)** — `output_scope` field on tasks declares
  allowed output file globs; `OutputEnvelope` dataclass captures SHA-256 hash
  of task output + scope verification; `check_scope_violations()` and
  `build_output_envelope()` in eventsource.py; `scope_violation` event emitted
  when task modifies files outside declared scope; integrates with worktree
  `files_changed` and structured context `files_changed`
- **Dual Verification for Worktrees** — cross-checks agent-reported output
  against actual `git diff` in worktrees; `verify_worktree_output()` compares
  `files_changed` with file paths extracted from agent stdout; flags
  `unclaimed_files` (changed but not mentioned) and `phantom_files` (claimed
  but not changed); `DualVerificationResult` dataclass with `overlap_ratio`;
  `worktree_verification` event emitted after successful merge; inspired by
  CIBER's textual + environmental dual verification

---

## [1.24.0] — 2026-03-21

### Added
- **OTLP Exporter** — `otel.py` converts completed runs into OpenTelemetry
  spans; root span per run, child span per task with engine/model/cost/tokens
  attributes; task events (retry, escalation, judge) attached as span events
- **`maestro export-otel`** subcommand — exports runs to OTLP endpoints (gRPC
  or HTTP) or JSON stdout; falls back to JSON when SDK not installed
- **Optional `[otel]` extra** — `pip install maestro-ai-cli[otel]` adds
  `opentelemetry-api`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp`
- **Watch Step Counter** — `watch.max_total_steps` field provides a hard cap on
  total task executions across all iterations; prevents runaway loops beyond
  plateau detection; `watch_step_limit` event emitted; E066 validation
- **Event-Driven Reminders** — `reminders` field on tasks with `{trigger, message}`
  dicts; 4 built-in triggers always active: `repeated_error`, `timeout`,
  `context_pressure`, `stuck_loop`; custom triggers match as substring;
  injected as `## Reminders` section in retry prompts; E067 validation
- **Agent-Triggered Context Compression** — `compress` signal type lets agents
  request context compression mid-task; `compress_before: bool` on tasks triggers
  compression before retries; integrates with existing `_compress_context_for_retry`
- **Run Knowledge Expansion** — 4 new knowledge pattern types: `cost_pattern`,
  `duration_pattern`, `retry_pattern`, `model_pattern`; `_KIND_ICONS` dict for
  readable format output with `[COST]/[DUR]/[RETRY]/[MODEL]` labels
- **Doctor check** — `maestro doctor` reports OpenTelemetry SDK availability
- 39 new tests for OTLP exporter, 120+ new tests for other v1.24 features
- **Massive test expansion** — 8500 tests total (+1774 from v1.23.0); 24 modules
  improved, `benchmark.py` ratio dropped from 135.5 to 5.0
- **Feature Readiness Matrix** — `docs/FEATURE-READINESS.md` scores 25 pending
  features on coverage, integration, history, and security dimensions

### Fixed
- `replan.py` — missing `PlanValidationError` import (NameError on plan failures)

---

## [1.23.0] — 2026-03-20

### Added
- **MCP Protocol Server** — `mcp_server.py` exposes Maestro as an MCP server
  for Claude Code, VS Code, Cursor, and other MCP-compatible clients
- **12 MCP tools** — validate_plan, run_plan, audit_plan, blame_run, diff_runs,
  explain_plan, plan_status, suggest_plan, doctor, scaffold_plan, verify_events,
  cleanup_runs
- **8 MCP resource patterns** — `maestro://runs`, `maestro://runs/{id}/manifest`,
  `/summary`, `/events`, `/tasks/{id}/log`, `/tasks/{id}/result`,
  `maestro://plans`, `maestro://plans/{name}`
- **3 MCP prompts** — debug_run, review_plan, create_plan
- **`maestro mcp-server`** subcommand — launches MCP server via stdio transport
- **Optional `[mcp]` extra** — `pip install maestro-ai-cli[mcp]` adds `mcp>=1.26.0`
- **Doctor check** — `maestro doctor` reports MCP SDK availability
- **Graceful degradation** — mcp_server.py works without SDK (no-op decorators)
  so tests run without the optional dependency
- 25 new tests covering all tools, resources, and prompts

---

## [1.22.0] — 2026-03-20

### Added
- **AG-UI Protocol Adapter** — `ag_ui.py` translates Maestro's 48+ event types
  into the AG-UI wire protocol (SSE over HTTP POST)
- **AG-UI SSE endpoint** — `POST /api/agui/runs` accepts `RunAgentInput` JSON,
  streams AG-UI events (`RUN_STARTED`, `STEP_STARTED/FINISHED`,
  `TEXT_MESSAGE_*`, `TOOL_CALL_*`, `STATE_SNAPSHOT/DELTA`, `CUSTOM`)
- **AG-UI approval companion** — `POST /api/agui/runs/{id}/approve` for
  human-in-the-loop approval flow
- **State management** — `AgUiRunState` tracks progress, costs, tokens;
  emits `STATE_SNAPSHOT` (initial) + `STATE_DELTA` (RFC 6902 JSON Patch)
- **Optional `[agui]` extra** — `pip install maestro-ai-cli[web,agui]` adds
  `ag-ui-protocol>=0.1.14`; endpoint auto-mounts when available
- **Doctor check** — `maestro doctor` reports AG-UI protocol availability
- 43 new tests covering event translation, state management, SSE format,
  suppressed events, watch/dynamic lifecycle, and domain event mapping

---

## [1.21.0] — 2026-03-20

### Added
- **Untrusted Context Detection** — `context_trust: trusted | untrusted`
  per-task field with transitive taint propagation across `context_from` chains
- **Injection stripping** — `_strip_injection_patterns()` removes common prompt
  injection patterns (system prompt overrides, role reassignment, XML injection
  tags, delimiter-based attacks) from untrusted upstream output before injection
- **Taint propagation** — `_compute_tainted_tasks()` computes tainted task set
  via fixed-point iteration on the DAG; taint clears at `guard_command` or
  `verify_command` boundaries; `TaskResult.tainted` tracks taint status
- **Auto-sandboxing** — untrusted upstreams are automatically wrapped in
  `<observation>` tags regardless of plan-level `control_flow_integrity`
- **SEC017 audit rule** — warns when `context_from` references tasks with
  external data indicators but no `context_trust` set
- **SEC018 audit rule** — warns when a task inherits tainted context without
  `guard_command` or `verify_command` to sanitize
- **`taint_detected` event** — emitted at run start for each tainted task
  (source: explicit or propagated)
- **Policy engine** — `context_trust` accessible in declarative policy rules
- **E065 validation** — invalid `context_trust` values rejected at load time
- 35 new tests covering taint propagation, injection stripping, SEC017/SEC018,
  loader validation, TaskResult serialization, and policy engine access

---

## [1.20.0] — 2026-03-20

### Added
- **Merge Intelligence (T1.2)** — worktree merge operations are now serialized
  via `_merge_lock` to prevent concurrent git state corruption. Pre-merge
  conflict detection uses `git merge --no-commit --no-ff` preview-then-commit
  pattern. Cross-task file overlap tracking via in-memory merge ledger detects
  when parallel tasks modify the same files. LLM merge review (haiku) analyses
  conflict diffs and suggests resolution strategies (additive/non_overlapping/
  true_conflict). Enhanced `WorktreeMergeResult` includes `MergeReview` with
  verdict, overlapping files, and resolution suggestions.
- **`MergeReview`**, **`MergeOverlap`** dataclasses and `MergeReviewVerdict`
  Literal type in models.py.
- **`worktree_review` event** — emitted when merge conflicts are detected,
  with verdict, conflict files, overlaps, and resolution suggestion. TUI
  EventFeed renders review events with color-coded verdicts.
- **`worktree_merge` event enriched** — now includes `review_verdict` and
  `overlapping_files` fields.
- **Enhanced conflict failure messages** — include overlapping task IDs and
  LLM resolution suggestions when available.

### Changed
- `merge_worktree()` signature extended with optional `review_model` and
  `review_callback` parameters (backwards compatible).
- Merge strategy changed from direct `git merge --no-ff` to preview-then-commit
  (`git merge --no-commit --no-ff` + `git commit`), enabling pre-merge analysis.
- `reset_merge_ledger()` called at the start of each `run_plan()`.

---

## [1.19.0] — 2026-03-19

### Added
- **Mid-task Signals (T2.2)** — running engine tasks can now send structured
  signals back to the scheduler via `[MAESTRO_SIGNAL] {json}` lines in stdout.
  7 signal types: `progress` (pct + step), `metric` (name + value), `log`
  (level + message), `artifact` (path + label), `timeout_extend` (request more
  time), `budget_query` (ask remaining budget), `checkpoint` (named checkpoint
  with data). Enable per-task with `signals: true` or plan-wide via
  `defaults.signals: true`.
- **New events**: `task_progress`, `task_metric`, `task_signal_log`,
  `task_artifact`, `timeout_extended`, `budget_query`, `task_checkpoint_signal`
  — all flow through events.jsonl, live display, and TUI.
- **Timeout extension**: tasks can request up to +1800s before timing out;
  `_stream_process` uses a polling loop when signals are enabled.
- **Budget query**: tasks can ask remaining budget via signal; scheduler
  provides thread-safe budget getter.
- **Signal security**: rate limiting (10/sec, 1000 total), 4KB line cap,
  path validation (no absolute paths, no `..` traversal), type allowlist.
- **`MAESTRO_SIGNALS=1`**, `MAESTRO_TASK_ID` env vars injected when signals
  are enabled.
- **TUI/Live progress**: `task_progress` events render in DAGPanel and
  EventFeed; `task_artifact` and `timeout_extended` events rendered.
- **40 new tests** in `tests/test_signals.py` covering parsing, handler,
  rate limiting, security, loader, and serialization.

### Changed
- `execute_task()` accepts optional `budget_getter` parameter for signal queries.
- `_stream_process()` accepts optional `deadline_ref` for dynamic timeout extension.
- `_on_line` callback now intercepts signal lines before `task_output` events.

---

## [1.18.0] — 2026-03-19

### Added
- **Dynamic Task Decomposition (T2.1)** — `dynamic_group: true` on engine
  tasks enables two-phase execution: Phase 1 generates a sub-plan via
  `output_schema`, Phase 2 builds a `PlanSpec` from the LLM output and
  executes it as a nested DAG. Strict 7-field allowlist (`id`, `engine`,
  `prompt`, `model`, `depends_on`, `description`, `tags`) — all other
  TaskSpec fields are silently ignored. Security hardening: CFI forced
  `True`, execution_profile forced `"safe"`, budget inherited, no
  recursion, no commands, no prompt_file. Raw LLM output logged to
  `_dynamic/raw_output.json` for forensics. New `dynamic.py` module.
- **`dynamic_subplan_start` / `dynamic_subplan_complete` events** — emitted
  during Phase 2 with sub-plan name, task count, success, and cost.
  Sub-task events tagged with `dynamic_parent` for TUI visibility.
- **`dynamic-planner` agent role** — dedicated `.claude/agents/` role for
  generating allowlist-safe sub-plans (defense in depth).
- **`/create-dynamic-plan` skill** — scaffolding skill with template,
  checklist, and security notes.
- **E063 / E064 error codes** — validation for `dynamic_group` requirements
  (engine + output_schema) and mutual exclusivity (group/batch/matrix).

---

## [1.17.0] — 2026-03-19

### Added
- **Cross-run Knowledge Accumulation (T1.3)** — new `knowledge.py` module
  extracts learnable patterns from completed runs and stores them in
  `.maestro-cache/knowledge/<plan_name>.jsonl`. Three knowledge kinds:
  `failure_pattern` (recurring failure categories with remediation advice),
  `timeout_hint` (tasks that tend to timeout), `success_pattern` (tasks
  with reliable clean passes). Knowledge is auto-injected into engine task
  prompts as a "## Previous Run Insights" section before dispatch — zero
  config required. Time-decay (30-day half-life) ensures stale knowledge
  fades, confidence increases with occurrences. `{{ task_knowledge }}`
  template variable also available for explicit use.
- **`KnowledgeRecord` / `KnowledgeKind`** dataclasses in `models.py` —
  task_id, kind, insight, confidence, occurrences, timestamps.

---

## [1.16.0] — 2026-03-18

### Added
- **Predictive Model Routing (T2.3)** — `model: auto` now learns from
  historical run data. `load_task_histories()` in `routing.py` reads prior
  `run_manifest.json` files and aggregates per-task, per-model performance
  (success rate, failure rate, timeout rate, cost, duration). The
  `_apply_historical_signal()` function adjusts the complexity score by up
  to ±0.20 based on evidence: cheap model 100% success → push cheaper
  (−0.15), cheap model ≥50% failures → push stronger (+0.15), any model
  ≥40% timeouts → push stronger (+0.10). Confidence scales linearly with
  run count (full at 5+ runs). Zero config — automatic when `model: auto`
  and prior runs exist.
- **`model_routed` event** (bug fix) — now emitted from `runners.py` via
  `event_callback` with `task_id`, `engine`, `requested`, `resolved`,
  `complexity_score`, `historical_runs`. Previously referenced in TUI
  `widgets.py` but never emitted.
- **`evidence` parameter on `resolve_auto_model()`** — optional dict that
  receives `complexity_score`, `tier`, and `historical_runs` for
  observability and event emission.
- **`ModelRecord` / `TaskHistory` dataclasses** in `models.py` — per-model
  aggregated stats (runs, successes, failures, timeouts, avg duration/cost)
  and per-task historical profile.

---

## [1.14.0] — 2026-03-18

### Added
- **Plan Density Score** (AgentConductor-inspired) — `maestro validate` now
  prints a DAG complexity report with S_complex score (nodes, edges, depth).
  Public function `compute_plan_density(plan)` returns raw metrics dict.
  `compute_plan_density_score(plan)` returns `(score, label, factors)` where
  label is `low|moderate|high|very_high`. W17 warns on high edge density
  (>60%), W18 warns on low parallelism (>70% sequential), W19 warns when
  S_complex > 0.8.
- **Deliberation Gate** (DOVA-inspired) — new `deliberation: bool` (default
  `false`) and `deliberation_threshold: float` (default `0.5`) fields on
  `TaskSpec`. When enabled, a cheap haiku pre-call checks if the task is
  self-answerable from upstream context before dispatching the main engine.
  Score < threshold → task skipped with message
  `"deliberation: self-answerable from context"`. Fail-open: any LLM error
  lets the engine call proceed normally. `deliberation_skip` event emitted on
  skip.
- **Adversarial Debate Judge Mode** (DOVA-inspired) — `judge.method: debate`
  runs bull+bear agents across N rounds. Bull advocates, bear critiques
  conditioned on bull's assessment. Score averaged across all rounds.
  `judge.debate_rounds: int` (default 2, range 1–4, capped at 4). E020
  validation: debate_rounds must be >= 1.
- **T0.3 — Streaming subprocess output** — `claude` engine now uses
  `--output-format stream-json --verbose`; per-tool-call `task_tool_call`
  events fire in real time so the TUI DetailPanel shows Read/Edit/Bash
  actions as they happen. `_extract_stream_json_result_text()` extracts the
  human-readable result from the stream for judge/guard/stdout_tail.
- **T0.4 — Scaffold auto-split heuristic** — `maestro scaffold` now
  automatically splits `implementation`/`complex-implementation` tasks that
  mention files ≥ 300 lines into a `{id}-read-plan` (haiku, read-only,
  produces JSON change plan) + `{id}-apply` (sonnet, applies the plan).
  Opt-out per task with `auto_split: false` in the brief YAML.
  `_detect_large_files()`, `_generate_split_tasks()` in `scaffold.py`;
  `TaskBrief.auto_split` field added.

## [1.15.0] — 2026-03-18

### Added
- **T1.1 — Structured task outputs** — tasks can declare `output_schema:` (JSON
  Schema dict) to get validated, typed outputs. After success/soft_failed the
  runner parses `stdout_tail` as JSON (direct → markdown code block → first
  `{...}` block) and validates it against the schema. If valid:
  `TaskResult.structured_output` is populated and downstream tasks access
  individual fields via `{{ task-id.output.FIELD }}` template variables.
  If invalid: task fails with `"output_schema validation failed: ..."`.
  `_extract_json_from_text()` and `_validate_task_output_schema()` helpers in
  `runners.py`. W3 warning suppressed for `task-id.output.*` patterns.
  `TaskSpec.output_schema` and `TaskResult.structured_output` fields added.
- **T0.4 — Scaffold auto-split already released in v1.14.0** (see above).

## [1.13.0] — 2026-03-17

### Added
- **`context_model` field** — decouples the model used for LLM context
  operations (`summarized`, `map_reduce`, `recursive`) from the task's
  execution model. Set at task level or via `defaults.<engine>.context_model`.
  Priority: task > engine default > `haiku` (unchanged default). Enables
  using cheaper models (e.g. `flash-lite`) for context compression while
  running tasks with more capable models.
- **`contract_type: api-schema`** — normalises OpenAPI 3.0 / Swagger 2.0
  JSON output. Extracts `path_count`, `schema_count`, `openapi_version` into
  contract metadata. Falls back to generic contract for non-JSON input.
- **`contract_type: test-manifest`** — normalises test run reports. Accepts
  pytest/jest JSON or plain text (`N passed, M failed`). Extracts
  `passed`/`failed`/`skipped`/`total` counts. Falls back to generic for
  unrecognised formats.
- **`maestro audit --coverage`** — per-category security coverage breakdown.
  Maps SEC001-SEC016 to 9 risk categories from "Security Considerations for
  Multi-agent Systems". Shows triggered rules per category and overall
  coverage %. JSON output with `--json`. `AuditFinding.category` field
  populated for all built-in rules.
- **`examples/audit-packs/security-baseline-demo.yaml`** — runnable demo
  plan showing `audit_packs:`, `contract_type:`, and `assert:` together.

## [1.12.0] — 2026-03-17

### Added
- **Batch task mode** — `batch:` block on engine tasks groups multiple items
  into fewer LLM calls. Items chunked by `max_per_call` (default 5). Per-item
  result parsing via `### Item N` markers. E057-E062 validation. Validated:
  11 files in 3 calls ($0.14) instead of 11 calls.
- **`judge.preset: ai_slop_detection`** — 5-rubric preset for detecting
  LLM-characteristic output (filler preamble, hedging, repetition, vague
  platitudes, trailing summary). Weighted mean aggregation, 0.6 threshold.
- **`frozen: true` on tasks** — prevents `mode: improve` agent from modifying
  frozen tasks. Prompt enforcement + post-edit validation with rollback.
- **Knowledge archive** — `LessonRecord` extracted after each improve iteration,
  persisted to `lessons.jsonl` with 30-day time-decay. `{{ watch.lessons }}`
  template var provides semantic memory to the improve agent.
- **Cross-run budget tracking** — `budget_period: daily | weekly | monthly`
  on plans. Budget ledger in `.maestro-cache/budget_ledger.jsonl`. Pre-run
  gate refuses execution if period budget exceeded. `maestro budget` subcommand.

## [1.11.1] — 2026-03-17

### Added
- **Structured summarization fields** — `context_mode: summarized` now
  requests `**Intent:**`, `**Findings:**`, `**Next steps:**` structured
  format instead of free-form bullet points. Inspired by Deep Agents
  (+13 Terminal Bench 2.0 points).
- **Artefact integrity verification** — `task_complete` events now include
  `log_hash` and `result_hash` (SHA-256, 16 hex chars) for `.log` and
  `.result.json` files. `maestro verify` validates both chain integrity
  AND artefact integrity. Protects `--resume` against memory poisoning.
  JSON output includes `chain_status`, `artefact_status`, and `artefact_mismatches`.

## [1.11.0] — 2026-03-17

### Added
- **`watch.mode: improve`** — built-in plan auto-improvement loop. Add
  `watch: { mode: improve }` to any plan and Maestro handles: target plan
  execution, `tasks_passed` extraction from manifest, blame + manifest
  injection as template vars, validate gates, and git commit/rollback.
  Auto-derives `metric`, `metric_source`, `metric_direction`, `target_metric`,
  and embeds improvement rules (priority table). Validated: 5-task plan
  converges 1→5/5 in 4 iterations ($0.84); 8-task stress test converges
  3→8/8 in 10 iterations ($2.85).
- **`watch.improve_model`** — override the model used by the improve agent
  (default: sonnet). Allows using opus for harder plans.
- **`metric_source: manifest`** — count tasks with `success`/`dry_run` status
  directly from `PlanRunResult`, without regex. Works independently of
  `mode: improve`.
- **`watch.target_metric`** — stop the watch loop when the metric reaches a
  target value. Uses `>=` comparison for `higher_is_better` and `<=` for
  `lower_is_better`. New `"target_reached"` watch status + event.
- **`watch.blame_plan`** — path to a target plan whose runs should be analyzed
  for blame injection. Enables `{{ watch.blame }}` (JSON blame analysis) and
  `{{ watch.manifest }}` (compact task status summary) template variables.
- **`{{ improve.plan_path }}`** and **`{{ improve.total_tasks }}`** — template
  variables injected in improve mode for the agent prompt.
- **Regression guards** — 10 reference plans in `tests/fixtures/plans/`
  exercising all schema features + 29 pytest tests in 3 tiers (loads,
  no-warnings, feature assertions). Catches loader/audit breakage in 0.56s.
- **Documentation-code sync tests** — 10 pytest tests verifying error codes,
  event types, SEC rules, warning codes, CLI subcommands, engine names, and
  context modes are documented. Runs in 0.12s.

### Fixed
- **Console summary always prints all counts** — the `[maestro]` summary line
  now always includes `ok`, `failed`, and `skipped` counts (even when 0),
  matching the `run_summary.md` format.
- **Improve agent yolo profile** — improve agent runs in `yolo` execution
  profile to write plan files (permission_denials fix).
- **Run path tracking** — `_watch_improve()` tracks target run paths directly
  instead of searching by name suffix (avoids `*_improve-X` matching `*_X`).
- **Keep lateral fixes** — in improve mode, `metric == best` now keeps instead
  of rolling back, allowing incremental progress when multiple independent
  root causes exist.

### Fixed
- **Console summary always prints all counts** — the `[maestro]` summary line
  now always includes `ok`, `failed`, and `skipped` counts (even when 0),
  matching the `run_summary.md` format. Previously, zero counts were omitted,
  causing `metric_source: stdout_regex` patterns like `"(\d+) ok"` to return
  `None` when all tasks failed (or `"(\d+) failed"` when all passed).

### Improvements
- **Verify/guard output in failure messages** — when `verify_command` or
  `guard_command` fails, the last 300 characters of their output are appended
  to `TaskResult.message` (e.g. `"verify_command failed with exit code 1
  (verify output: assert 'page-break' in html failed)"`)
- **Failed Tasks section in `run_summary.md`** — failed tasks now get a
  dedicated section at the end of the summary showing the error message,
  log file path, and last 5 lines of stdout output
- **`verify_failure` event** — structured event emitted when `verify_command`
  fails, with `task_id`, `exit_code`, and `output_snippet`; handled by
  live display and TUI event feed
- **`tune_timeout` suggestion** — `maestro suggest` now detects tasks with
  2+ timeout failures (exit code 124) across runs and recommends a new
  `timeout_sec` at 1.5× the maximum observed duration
- **Stderr surfacing in failure messages** — when an engine task fails with
  little or no stdout, the last 20 lines of stderr are included in the
  `TaskResult.message` field (e.g. `"Task failed with exit code 1 (stderr:
  Error: Claude Code cannot be launched inside another session.)"`)
  - `_stream_process()` now returns a 3-tuple `(returncode, stdout_tail, stderr_tail)`
  - Stderr is also fed into `_classify_failure()` and `_is_engine_failure()`
    for better fallback engine detection
- **CLAUDECODE env var detection** — `_preflight_checks()` warns when
  `CLAUDECODE` is set in the environment and the plan uses `engine: claude`,
  preventing silent nested-session failures
- **Prose fallback for `prompt_md_heading`** — `extract_prompt_from_markdown()`
  now extracts all prose text under a heading when no code fence is found,
  instead of raising an error. Code-fenced content is still preferred when
  present.
- **`prompt_file` / `prompt_md_file` workspace_root resolution** — relative
  prompt paths are now resolved against `workspace_root` first (if set), then
  fall back to the plan's source directory. This lets plans stored outside the
  workspace reference prompt files inside it.
- **UNC path warning on Windows** — `_preflight_checks()` detects when a task's
  working directory resolves to a UNC path (`\\server\share\...`) and warns
  that `CMD.EXE` does not support UNC paths as working directory — string-format
  `verify_command` / `pre_command` / `guard_command` may fail.

---

## v1.10.0 — Context Intelligence + Hardening

### New Features
- **Layered Context Loading** (`context_mode: layered`) — budget-aware tiered
  context resolution with three levels: L0 (one-line summary, ~50 tokens),
  L1 (section headings + key findings, ~200 tokens), L2 (full content).
  Most-relevant upstreams promoted first within `context_budget_tokens`.
  Zero LLM cost (heuristic extraction). Estimated 40-65% context token savings
  for tasks with 3+ upstreams.
  - New functions: `_build_layered_context()`, `_extract_l0_summary()`,
    `_extract_l1_sections()` in runners.py
  - `"layered"` added to `ContextMode` Literal and `CONTEXT_MODES` constant
  - Wired into scheduler.py budget pipeline
- **Control Flow Integrity** (`control_flow_integrity: true`) — plan-level
  opt-in flag that sandboxes upstream context into `<observation>` blocks,
  preventing prompt injection via `context_from` data.
  - `_sandbox_observation()` in runners.py wraps context with XML tags
  - `observation_block: bool` field on `TaskSpec` for per-task control
  - `PlanSpec.control_flow_integrity` field with serialization in `to_dict()`
  - SEC015: detect `when:` expressions referencing unbounded upstream output
  - SEC016: detect `context_from` chains without `guard_command` validation
- **Judge Quorum** (`judge.quorum`) — run N independent judge evaluations and
  require majority/unanimous/any consensus for pass/fail verdict. Reduces
  single-judge variance for high-stakes tasks.
  - `quorum: int` (>= 2) and `quorum_strategy: majority | unanimous | any`
    on `JudgeSpec`
  - `_run_judge_quorum()` in runners.py with configurable aggregation
  - `QuorumStrategy` Literal type + `QUORUM_STRATEGIES` constant in models.py
  - Loader validation: E054 (invalid quorum), E055 (invalid strategy),
    E056 (strategy without quorum)
- **Context Retrieval Trajectory** (`maestro explain --context`) — shows why
  each piece of context was selected: BM25 keywords, hop distance, budget
  trimming, compression ratio per upstream.
  - `explain_context_trajectory()`, `format_context_trajectory()`,
    `format_context_trajectory_json()` in explain.py
  - `ContextSelectionEntry`, `ContextTrajectoryReport` dataclasses in models.py
  - `context_trajectory` field on `TaskResult` with serialization
  - `--context` flag wired in `_cmd_explain()` in cli.py

### Bug Fixes (review)
- Fix duplicate `ContextSelectionEntry` class definition in models.py
- Fix undefined `quiet` variable in scheduler.py (`NameError` on policy warn)
- Fix context metrics cross-task contamination (per-task dict instead of shared
  variables in scheduler's inner/outer loop)
- Wire `dag_metadata` through to `resolve_auto_model()` — difficulty-aware
  routing now functional (was computed but not passed)
- Fix `_apply_context_budget()` and `_apply_intent_filtering()` return value
  unpacking in test files (3-tuple returns)

### New Error Codes
- `E054` — invalid `judge.quorum` value (must be integer >= 2)
- `E055` — invalid `judge.quorum_strategy` value
- `E056` — `quorum_strategy` requires `quorum` to be set

### New Audit Rules
- `SEC015` — `when:` expression references unbounded upstream output fields
- `SEC016` — `context_from` pulls raw engine output without `guard_command`

### Tests
- 4544 total tests (+963 from v1.9.0)
- New test files: `test_layered_context.py`, `test_cfi.py`,
  `test_judge_quorum.py`, `test_explain_trajectory.py`,
  `test_contracts.py`, `test_workspace_assertions.py`, `test_relationships.py`
- New test classes in `test_loader.py`: quorum validation tests
- Watch-driven: `qa-watch-contracts` grew 316 tests from zero (contracts, assertions, relationships)
- Watch-driven: `qa-watch-suggest-status` grew 365 tests (suggest 11→195, status 11→192)

---

## v1.9.0 — Smart Prevention (Policy Engine + Difficulty-Aware Routing + Blame Attribution)

### New Features
- **Policy Engine** — declarative `policies:` block with `block`/`warn`/`audit`
  actions evaluated at task dispatch time in scheduler.py. Safe AST-based
  expression evaluation (NEVER eval/exec). Supports `task.*` and `plan.*` field
  access, comparisons, boolean operators, and `in`/`not in` for tags. Complements
  `maestro audit` (static, pre-run) with runtime enforcement.
  - New module: `policy.py` — `compile_policy()`, `evaluate_policies()`,
    `format_violations()`, `_SafeEvaluator` class
  - New dataclasses: `PolicySpec`, `PolicyViolation`
  - New Literal: `PolicyAction = Literal["block", "warn", "audit"]`
  - `PlanSpec.policies` field
  - Loader parsing + validation (E052)
  - Scheduler enforcement: `policy_violation` event emitted before task dispatch
- **Difficulty-Aware Model Routing** — extends `model: auto` with cost/latency
  weights, `routing_strategy:` plan-level field (`cost_optimized` | `quality_first`
  | `balanced`), and DAG structural signals (fan_out, depth, upstream_failure_rate).
  - New constant: `_COST_WEIGHTS` in routing.py
  - `_score_task_complexity()` extended with `routing_strategy` and `dag_metadata` params
  - `resolve_auto_model()` extended with same params
  - `_compute_task_depth()`, `_compute_fan_out()` helpers in scheduler.py
  - New Literal: `RoutingStrategy = Literal["cost_optimized", "quality_first", "balanced"]`
  - `PlanSpec.routing_strategy` field
  - Loader parsing + validation (E053)
- **Blame Attribution** — `maestro blame <run-path>` CLI command for causal failure
  tracing via dependency graph backward walk from failed tasks.
  - New module: `blame.py` — `blame_run()`, `format_blame()`, `format_blame_json()`
  - New dataclasses: `BlameNode`, `BlameChain`
  - New Literal: `BlameCategory = Literal["root_cause", "dependency_cascade",
    "context_corruption", "timeout_propagation", "budget_exhaustion", "unknown"]`
  - Classifies root causes with confidence scores (0.6–0.95)
  - Category-specific suggested fixes
  - Evidence extraction from `events.jsonl`
  - JSON output for CI integration (`--json`)

### New CLI Commands
- `maestro blame <run-path> [--json]` — trace failure causality in completed runs

### New Error Codes
- `E052` — invalid policy configuration (missing name/rule, bad action, duplicate name, bad syntax)
- `E053` — invalid `routing_strategy` value

### New Events
- `policy_violation` — emitted for each policy violation at task dispatch time
  (fields: `task_id`, `policy_name`, `action`, `message`)

### Tests
- 3581 total tests (+164 from v1.8.0)
- New test files: `test_policy.py`, `test_routing_strategy.py`, `test_blame.py`
- New test classes in `test_loader.py`: `TestPoliciesValidation`, `TestRoutingStrategyValidation`

---

## v1.8.0 — Structured Validation + Security Hardening + Diagnostics

### New Features
- **`json-schema` assertion type** — validate engine task output against a JSON
  Schema definition. Supports inline `schema` dict or external `schema_file`.
  Recursive stdlib-only validation (no dependencies) covering `type`,
  `properties`, `required`, `items`, `enum`, `minLength`, `maxLength`, with
  depth limit of 20. Added to `ASSERTION_TYPES` in models.py, wired into
  `_evaluate_typed_assertion()` in runners.py, validated in loader.py (E020).
- **STPA security audit rules SEC008-SEC014** — 7 new hazard detection rules in
  `maestro audit`:
  - SEC008: Destructive commands (`rm -rf`, `DROP TABLE`, `git push -f`, etc.)
    without `requires_approval`
  - SEC009: Engine tasks with yolo flags + workspace_root but no worktree
    isolation
  - SEC010: Context chains deeper than 3 hops without `context_budget_tokens`
  - SEC011: `escalation` configured without `max_cost_usd` budget
  - SEC012: `fallback_engine` with yolo flag propagation
  - SEC013: `watch` loop without cost bounds (severity: error)
  - SEC014: Cloud credentials in env without `secrets` configuration
- **`maestro audit --fix`** — auto-remediation for safe, mechanical findings.
  Fixes SEC001 (adds `max_cost_usd: 10.0`), SEC003/SEC014 (adds
  `secrets: auto`). Creates `.yaml.bak` backup before modifying. Supports
  `--dry-run` via `fix_plan(dry_run=True)`.
- **Expanded failure taxonomy** — `FailureCategory` expanded from 9 to 16
  categories with 7 new classifications: `dependency_missing`,
  `output_format_error`, `cascading_failure`, `deadlock`, `miscommunication`,
  `role_confusion`, `verification_gap`. Regex patterns added to
  `_FAILURE_PATTERNS` in runners.py. Category-specific remediation advice in
  `_FAILURE_REMEDIATION` dict in suggest.py.

### New CLI Flags
- `maestro audit --fix` — auto-fix safe audit findings (SEC001, SEC003, SEC014)

### New Functions
- `_validate_json_schema()` in runners.py — recursive JSON Schema validation
- `fix_plan()` in audit.py — auto-remediation engine with backup + dry-run

### Tests
- 3417 total tests (+45 from v1.7.0)
- New test classes: `TestJsonSchemaValidation`, `TestJsonSchemaLoaderValidation`,
  `TestSTPAHazardRules`, `TestExpandedFailureClassification`, `TestAuditFix`

---

## v1.7.0 — `maestro chat` (Multi-Model Interactive Terminal)

### New Features
- **`maestro chat`** — interactive terminal that talks to any of the 6
  configured engines (claude, codex, gemini, copilot, qwen, ollama) from a
  single prompt. Reuses `build_command()` and `_build_safe_env()` from
  runners.py via lightweight PlanSpec/TaskSpec stubs.
- **Multi-model routing** — `@engine` prefix routes a single turn to a
  different engine without changing the session default (e.g.,
  `@codex optimize this`).
- **Conversation history** — previous messages prepended as transcript into
  each turn's prompt. Truncated at ~80k chars to stay within context limits.
- **Slash commands** — `/model engine/model`, `/models`, `/clear`, `/cost`,
  `/help`, `/quit` for session management.
- **Streaming output** — real-time line-by-line streaming from all engines via
  `subprocess.Popen` + stdout pipe. Codex JSON events parsed and formatted
  via `_format_engine_line()`. Stderr fallback when stdout is empty.
- **Cost tracking** — per-turn and cumulative session cost extraction using
  existing `_extract_cost_from_line()` regex patterns.
- **Tab completion** — readline-based autocomplete for slash commands and
  `@engine` prefixes.

### New Modules
- `src/maestro_cli/chat.py` — `ChatMessage`, `ChatSession`, `run_chat()`,
  `_run_chat_turn()`, `_build_chat_plan_stub()`, `_build_chat_task_stub()`,
  `_adjust_command_for_chat()`, `_format_engine_line()`,
  `_build_history_prompt()`, `_parse_engine_prefix()`,
  `_dispatch_chat_command()`, `_cmd_model()`, `_cmd_models()`, `_cmd_clear()`,
  `_cmd_cost()`, `_cmd_help_chat()`, `_setup_chat_readline()`,
  `_extract_turn_cost()`

### New CLI Commands
- `maestro chat [--engine ENGINE] [--model MODEL] [--execution-profile PROFILE]`

### Bug Fixes
- **Claude exit code 3 false positive** — `_claude_json_is_success()` detects
  successful Claude runs that report non-zero exit codes by checking for
  `"result": "success"` or `session_cost` in the JSON output.
- **Rate limit detection** — 5 new patterns added to `_is_engine_failure()`
  for subscription limits, spending limits, and generic rate limit messages.

### Tests
- 3372 total tests (+69 from v1.6.0)
- New test file: `test_chat.py` (52 tests)
- New runner tests: `_claude_json_is_success` unit tests, Claude exit code
  override integration tests, subscription limit detection tests

---

## v1.6.0 — Event Sourcing + Security Audit + Fault Tolerance (Wave 1)

### New Features
- **Event sourcing** (`eventsource.py`) — hash-chained `events.jsonl` with
  SHA-256 tamper detection. Each event carries `prev_hash` + `event_hash` for
  integrity verification. Replay engine reconstructs `EventRecord` list from
  events alone. `maestro verify <run-path>` validates hash chain integrity.
- **Security audit** (`audit.py`) — `maestro audit <plan.yaml>` scans plans for
  security risks across 7 rules (SEC001-SEC007): missing budget limits, exposed
  secrets in prompts/env, yolo/bypass flags, production paths without approval
  gates, missing verification commands. JSON output with `--json`.
- **Configurable retry strategies** — `retry_strategy` field on tasks:
  `constant` (default), `linear` (base × attempt), `exponential` (base × 2^attempt).
  New `_compute_retry_delay()` function with proper plan-defaults fallback.
- **Circuit breaker** — `circuit_breaker:` block on tasks with `max_failures`
  and `reset_after_sec`. Trips after N consecutive failures, auto-resets after
  cooldown. `circuit_breaker_tripped` event emitted. `CircuitBreakerSpec`
  dataclass.
- **Robust Codex token extraction** — 4-strategy extraction in
  `_extract_codex_cumulative_usage()`: `response.completed` events,
  `turn.completed` events, `item.completed` events, byte-length estimation
  fallback. Handles Codex CLI output format changes gracefully.
- **Watch consolidation agent** — `_run_consolidation()` in watch.py: periodic
  LLM-driven analysis of experiment history. Configurable via `consolidate_model`,
  `consolidate_every`, `consolidate_prompt`. Output injected as
  `{{ watch.consolidated }}` template variable.

### New Modules
- `src/maestro_cli/eventsource.py` — `compute_event_hash()`, `ChainState`,
  `emit_hashed_event()`, `replay_events()`, `verify_chain()`
- `src/maestro_cli/audit.py` — `AuditFinding`, `AuditSeverity`, `audit_plan()`,
  `format_audit()`, `format_audit_json()`

### New CLI Commands
- `maestro verify <run-path>` — validate event chain integrity (19 frozen → 2 new unfrozen)
- `maestro audit <plan.yaml> [--json]` — security audit a plan

### Schema
- New `retry_strategy` field on TaskSpec: `Literal["constant", "linear", "exponential"]`
- New `circuit_breaker` block on TaskSpec: `CircuitBreakerSpec(max_failures, reset_after_sec)`
- New `consolidate_model`, `consolidate_every`, `consolidate_prompt` fields on WatchSpec
- New dataclasses: `EventRecord`, `VerifyStatus`, `CircuitBreakerSpec`
- New validation: E050 (invalid circuit_breaker config), E051 (invalid retry_strategy)

### Bug Fixes
- **Fallback engine args inheritance** (P15) — when `fallback_engine` triggers,
  engine-specific `args` (e.g., `--full-auto`) are now cleared instead of being
  passed to the fallback engine CLI, which would crash it.
- **`_compute_retry_delay` plan defaults** — new retry delay function now properly
  falls back to `plan.defaults.retry_delay_sec` when task has no override.

### Tests
- 3303 total tests (+83 from v1.5.1)
- New test files: `test_eventsource.py`, `test_audit.py`, `test_fault_tolerance.py`
- New runner tests: fallback args clearing, byte-length token estimation

---

## v1.5.1 — Bug Fixes + Judge Timeout Configuration

### Bug Fixes
- **`--resume-last` retries dependency-failed tasks** — Previously, tasks skipped
  due to upstream dependency failure were treated as "success-like" on resume and
  never re-evaluated. Now `_load_prior_results()` excludes them from the resumed
  set, so they re-run when their dependency succeeds.
- **`soft_failed` shown separately in summary** — Console, markdown summary, and
  `run_complete` event now report `soft_failed` as a distinct category instead of
  counting them as "ok". Format: `9 ok / 4 soft_failed / 0 failed / 0 skipped`.
- **Cache no longer stores `soft_failed` results** — Timed-out tasks with
  `allow_failure: true` were cached and replayed on re-runs, preventing actual
  re-execution. Only `success` results are now cached.
- **Cache lookup rejects stale `soft_failed` entries** — Existing cached
  `soft_failed` entries from prior runs are ignored on lookup.
- **Live display `$0.00` → `--`** — When no tasks report cost data, the live
  header now shows `--` instead of `$0.00`.
- **Live display resize improvement** — Explicit `Console(force_terminal=True)`
  and `vertical_overflow="visible"` to reduce duplication on Windows Terminal
  resize.
- **Timeout+retry warning deduplicated** — The per-task `timeout_sec with
  max_retries` warning is now grouped into a single summary line instead of
  N individual warnings.

### New Features
- **`judge.timeout_sec`** — Configurable timeout for LLM-as-Judge evaluations
  (minimum 10s, default 60s). Prevents false FAIL on large outputs where the
  judge cannot evaluate within the default 60s window.

### Schema
- New `timeout_sec: int | None` field on `JudgeSpec` (default: None → 60s).
- Validation: E020 if `judge.timeout_sec < 10`.

### Tests
- 3220 total tests (+1150 from v1.5.0).
- New resume tests: dependency-failure skipped tasks re-run, when-condition
  skipped tasks stay skipped.
- Updated summary tests for new `soft_failed` category.
- Updated cache tests: `soft_failed` no longer stored.

---

## v1.5.0 — Git Worktree Isolation + Semantic Model Routing

### New Features
- **Git Worktree Isolation** (`worktree: true`) — per-task isolated git worktrees
  for safe parallel file editing. Automatically creates a worktree at
  `.maestro-worktrees/<task-id>`, runs the task inside it, merges changes back
  on success, and reports conflicts on failure. Cleanup is automatic.
- **Semantic Model Routing** (`model: auto`) — automatic model selection based
  on task complexity, tags, prompt length, dependency count, and context mode.
  Supports all 6 engines with per-engine tier tables (low/medium/high).
  Tag signals: `security`/`architecture`/`critical` → high tier,
  `trivial`/`typo`/`docs` → low tier.

### New Modules
- `src/maestro_cli/worktree.py` — git worktree create/merge/cleanup operations.
  All git commands use `subprocess.run` with list args (no shell injection).
  Functions: `create_worktree()`, `merge_worktree()`, `cleanup_worktree()`,
  `get_base_branch()`.
- `src/maestro_cli/routing.py` — pure-function model routing heuristics.
  Functions: `resolve_auto_model()`, `_score_task_complexity()`,
  `_tier_from_score()`. Tag-based and complexity-based scoring with
  per-engine model tier tables.

### Schema
- New `worktree: bool` field on TaskSpec (default: false).
- `model: "auto"` resolves to a concrete model via routing heuristics.
- New `WorktreeMergeResult` dataclass with `status` (merged/conflict/empty/error),
  `files_changed`, `conflict_files`, `merge_commit`, `error`.
- New `auto_routed_model` field on TaskResult.
- New error codes: E045 (worktree requires workspace_root), E046 (worktree
  not valid on group/command tasks).
- New warning: W16 (worktree on task with no parallel siblings).

### Events
- `worktree_create` — task_id, worktree_path, branch
- `worktree_merge` — task_id, status, files_changed, conflict_files
- `worktree_cleanup` — task_id
- `model_routed` — task_id, engine, requested, resolved, complexity_score

### TUI
- Worktree and routing event formatters in EventFeed.
- Restored `priority=True` on navigation bindings (Up/Down/Enter/Escape).
- `can_focus = False` on EventFeed to prevent keyboard stealing.

### Tests
- 2070 total tests (45 worktree, 19 routing, 5 loader v1.5.0).
- Watch QA grew worktree tests from 12 to 45 over 10 iterations ($1.68).

---

## v1.4.0 — Watch Mode

### New Features
- **`maestro watch`**: Autonomous metric-driven iteration loop. Runs a plan
  repeatedly, extracts a numeric metric from task output, compares to previous
  best, and keeps (git commit) or rolls back (git reset/revert) changes
  automatically. Stops on max iterations, plateau, budget, or interrupt.
- **Metric extraction**: Four sources — `stdout_regex` (capture group from
  stdout), `verify_command` (from verify output section in log), `guard_command`
  (from guard output section), `json_field` (navigate JSON path in result file).
- **Experiment ledger**: Per-watch-session `experiments.jsonl` with iteration
  number, metric value, best metric, action (keep/rollback/revert), cost, duration,
  git commit SHA, and timestamp.
- **Watch template variables**: `{{ watch.iteration }}`, `{{ watch.best_metric }}`,
  `{{ watch.last_metric }}`, `{{ watch.history }}` (formatted table of recent
  iterations), `{{ watch.program }}` (program.md content) — injected into all
  task prompts via `extra_template_vars`.
- **Program-as-prompt**: Optional `program_md` field points to a markdown file
  with high-level directives for the agent across iterations.
- **Resume support**: `--resume-last` or `resume_from` parameter to continue
  a watch session from `experiments.jsonl`.
- **Watch events**: `watch_start`, `iteration_start`, `iteration_complete`,
  `metric_recorded`, `regression_detected`, `rollback_executed`,
  `plateau_detected`, `watch_complete`.

### Schema
- New `watch:` plan-level block with fields: `metric`, `metric_direction`
  (lower_is_better/higher_is_better), `metric_source` (stdout_regex/
  verify_command/guard_command/json_field), `metric_pattern`, `metric_json_path`,
  `metric_task`, `max_iterations`, `warmup_iterations`, `plateau_threshold`,
  `plateau_action` (stop/escalate_model/notify), `on_regression`
  (rollback/revert/keep), `program_md`, `max_cost_usd`, `iteration_budget_sec`.
- New `WatchSpec`, `WatchIteration`, `WatchState` dataclasses in models.py.
- New error codes: E032-E044 (watch block validation).
- New Literal types: `MetricDirection`, `MetricSource`, `OnRegression`,
  `PlateauAction`, `WatchStatus`.

### Architecture
- New module: `src/maestro_cli/watch.py` (520 lines) — iteration loop, metric
  extraction, git operations, experiment logging, resume.
- `extra_template_vars` parameter threaded through `run_plan()` → `execute_task()`
  → `build_command()` → `_load_prompt()` for watch variable injection.
- CLI integration: `maestro watch` subcommand with full output mode support
  (text, jsonl, live, tui).

### Stats
- 2001 tests (up from 1959 in v1.3.0)
- New error codes: E032-E044
- New module: watch.py
- 17th CLI subcommand: `watch`

---

## v1.3.0 — TUI Completion + Resilience

### New Features
- **Auto-escalation policy**: `escalation: [haiku, sonnet, opus]` field on tasks —
  automatically retry with a higher-tier model on failure before exhausting retries.
  Configurable per-task or via plan-level `defaults.<engine>.escalation`. Emits
  `task_escalation` event with `from_model`/`to_model` payload.
- **Cross-engine fallback**: `fallback_engine` + `fallback_model` fields on tasks —
  switch to an alternative engine on infrastructure failures (CLI not found, auth
  errors, rate limits, API outages). Detected via `_is_engine_failure()` with exit
  code checks (127/9009) and error pattern matching. One fallback attempt per task.
  Emits `engine_fallback` event.
- **DetailPanel (TUI B2)**: split-pane detail view showing selected task metadata
  (engine, model, duration, cost, exit code, retries) and live log tail with 500ms
  polling. Press Enter on a task to open, Escape to close.
- **Keyboard navigation (TUI B3)**: Up/Down/j/k to scroll task list, Enter to select,
  Escape to deselect/clear filter, `f` to cycle filter (all/running/failed/completed),
  `t` to toggle follow mode (auto-scroll to latest running task), `q` to quit with
  double-press confirmation.
- **Approval modal (TUI B4)**: interactive approval gate in TUI mode. When a task has
  `requires_approval: true`, a modal shows task ID and approval message with `y` to
  approve, `n` to deny. Blocks executor thread via `threading.Event`.
- **TUI doctor checks (B5)**: `maestro doctor` now checks for `textual` and `rich`
  availability with version info. Reports "info" status if missing with install hint.
- **Live output enhancements**: `task_output` in live table, `task_escalation` and
  `engine_fallback` event handlers, judge event display, retry counter in live output.

### Schema
- New `TaskSpec` fields: `escalation: list[str]`, `fallback_engine: EngineName | None`,
  `fallback_model: str | None` (all optional, backward-compatible)
- New `EngineDefaults` fields: `escalation`, `fallback_engine`, `fallback_model` for
  plan-level defaults inheritance
- New error codes: E030 (invalid fallback configuration), E031 (invalid escalation)
- New warning codes: W13 (redundant fallback — same as primary engine), W14 (duplicate
  entries in escalation list), W15 (escalation without retries)

### Architecture
- `_next_escalation_model()` in runners.py: linear progression through escalation chain
- `_is_engine_failure()` in runners.py: detect infrastructure failures vs task failures
- Escalation disabled after engine fallback (`not _fallback_used` guard) to prevent
  cross-engine escalation chain confusion
- `build_command()` accepts `engine_override` and `model_override` parameters
- TUI widgets: `DetailPanel`, `ApprovalModal` added to widgets.py
- TUI app: approval handler with `threading.Event` bridge, quit confirmation UX

### Bug Fixes
- `EngineDefaults` now explicitly declares `escalation`/`fallback_engine`/`fallback_model`
  fields instead of using dynamic `setattr()` — fixes mypy strict mode violations
- Escalation correctly disabled after engine fallback to prevent invalid model lookup
  in the original engine's escalation chain

### Stats
- 1959 tests (up from 1903 in v1.2.0)
- New error codes: E030, E031
- New warning codes: W13, W14, W15
- New events: `task_escalation`, `engine_fallback`

---

## v1.2.0 — Interactive TUI

### New Features
- **`--output tui`**: Interactive Textual TUI with DAG panel, header, event feed, and
  keyboard navigation. Install with `pip install maestro-ai-cli[tui]`. Requires
  `textual>=1.0.0,<9.0.0`.
- **`PlanHeader` widget**: Real-time progress bar, completed/total count, cost accumulation,
  budget warning display.
- **`DAGPanel` widget**: Rich Table with status icons, task descriptions, engine/model,
  duration, cost, and live output in the Info column. Updates in real time as tasks run.
- **`EventFeed` widget**: Scrolling RichLog showing timestamped events (task start/complete/
  skip, budget warnings, run complete) with live stdout streaming via `├─` tree connectors.
- **Live task output streaming**: `task_output` events emitted per stdout line from running
  tasks; displayed in both DAGPanel (Info column) and EventFeed (tree connector lines).
  Throttled to ~4 refreshes/sec to avoid terminal flooding.
- **Local time display**: EventFeed timestamps converted from UTC to local timezone.
- **TUI demo plan**: `examples/tui_demo.yaml` — 8-task DAG with progressive output for
  visual testing.

### Architecture
- New package: `src/maestro_cli/tui/` (app.py, widgets.py, app.tcss, __init__.py)
- Threading model: `run_worker(thread=True)` + `call_from_thread()` bridges executor
  threads to Textual's async event loop
- `cancel_event` (threading.Event) wired from TUI quit to scheduler dispatch loop
- DAGPanel uses `Static` + Rich `Table` instead of Textual `DataTable` to avoid cursor
  and row-indexing bugs in Textual 8.x
- `_stream_process()` accepts optional `line_callback` for real-time stdout streaming
- `execute_task()` emits `task_output` events via `event_callback` when provided

### Refactored
- Extracted `format_duration()`, `format_cost()` from `live.py` to `utils.py` (shared)
- Extracted `STATUS_STYLES`, `TERMINAL_STATUSES` from `live.py` to `models.py` (shared)

### CLI
- `--output` choices extended to `text`, `jsonl`, `live`, `tui` (both `run` and `replan`)
- Multi-plan + `--output tui` blocked with clear error message
- `--output tui` on `replan` blocked with clear error message
- Updated banner: shows all 6 engines, output modes in run description

### Stats
- 1903 tests (up from 1891 in v1.1.2)
- New optional dependency group: `[tui]` (textual>=1.0.0,<9.0.0)

---

## v1.1.2 — Event Enhancements

### New Events
- **`task_retry`**: Emitted in runners.py when a task retries (attempt number, max_retries)
- **`judge_start`**: Emitted before judge evaluation (criteria_count, method)
- **`plan_name` auto-injection**: All events now include `plan_name` field automatically
  via `_emit()` — no per-call-site changes needed
- **`goal` in `run_start`**: `run_start` event now includes `goal` field

### Internal
- `execute_task()` accepts optional `event_callback` parameter for task-level event emission
- Scheduler passes `event_callback` as keyword arg to `execute_task()`
- All mock `execute_task` signatures across 7 test files updated with `**kwargs` for
  forward compatibility

### Stats
- 1891 tests (up from 1888 in v1.1.1)

---

## v1.1.1 — Scheduler Wiring (Phase B Blockers)

### Fixed
- **`cancel_event` wired**: `threading.Event` now checked at top of scheduler dispatch
  loop and after `wait()` — pending tasks marked as skipped on cancellation
- **`approval_handler` wired**: Inserted between `auto_approve` and `_request_approval()`
  fallback; handler exceptions treated as denial
- **Secrets masking in event callbacks**: `_build_secret_values()` called once before
  thread pool; string values in event payloads masked before reaching callback consumers

### Stats
- 1888 tests (up from 1881 in v1.1.0)
- Maestro-executed plan: 7/7 tasks SUCCESS, $3.03, 10m40s

---

## v1.1.0 — Event Callback + Live Output + Goal Field

### New Features
- **`--output live`**: Real-time Rich table display during plan execution showing task
  progress, cost, duration, and status. Install with `pip install maestro-ai-cli[live]`.
- **`goal:` plan field**: Optional string injected as context into all engine task prompts.
  Also available as `{{ goal }}` template variable.
- **Event callback**: `run_plan()` accepts `event_callback` parameter for programmatic
  event consumption. Foundation for future TUI mode.

### Internal
- `run_plan()` signature extended with `event_callback`, `cancel_event`, `approval_handler`
  (all optional, backwards compatible)
- Event callback propagated to `run_multi_plan()` and `replan()`
- New module: `src/maestro_cli/live.py`
- New optional dependency group: `[live]` (rich>=13.0.0)

---

## [1.0.0] -- 2026-03-06

### Added
- **v1 migration guide**: `docs/MIGRATING_TO_V1.md` covering concrete cleanup
  steps for users coming from `0.x`, including the `maestro replan --model`
  switch, cache path distinctions, opt-in real-engine policy, and a move away
  from undocumented loader knobs and scraped console output

### Changed
- **Release docs completed for v1.0.0** across `README.md`,
  `CHANGELOG.md`, `docs/VERSIONING.md`, `CLAUDE.md`, and
  `docs/MIGRATING_TO_V1.md`
- README now states exactly what is frozen in `1.x`: documented `version: 1`
  plan schema, documented CLI commands/flags, and stable run artifacts named in
  `docs/V1_API_FREEZE.md`
- README now links the concrete `0.x` upgrade path in
  `docs/MIGRATING_TO_V1.md`, and the roadmap section now reflects the v1.0.0
  release baseline instead of the pre-1.0 wording
- README and CLAUDE now document the implemented plugin extension path via the
  `maestro_cli.engines` entry-point group, while explicitly noting that the
  custom plugin API is not frozen by `1.0.0`
- Release docs now name the current plugin authoring surface precisely as
  `maestro_cli.plugins.EnginePlugin` / `DoctorProbe`, and CLAUDE now matches
  the implemented cache split of `<run-dir>/.cache` vs
  `<workspace_root>/.maestro-cache/index/`
- README and CLAUDE now document the implemented `maestro ci` workflow
  generator, including supported `--provider` aliases, output flags, explicit
  cross-platform lanes, and the current manual real-engine jobs
  `maestro_real_engine` / `run_maestro_real_engine`
- Versioning docs now clarify that CI generator behavior, custom engine plugin
  API stability, strict mypy expansion, and benchmark release gates are
  intentionally deferred to `1.1.0+`

### Security
- README now points to `docs/SECURITY.md` and `docs/SECURITY_BASELINE.md` as the
  operational security guidance for v1.0.0
- Release docs now state that default CI remains offline-first and that
  real-engine tests in `tests/test_e2e_real_engines.py` stay opt-in via
  `MAESTRO_RUN_REAL_ENGINE_TESTS=1`
- Release docs now explicitly state that no built-in security scanner or
  mandatory security audit gate is frozen into the v1.0.0 contract

### Tooling
- README and CLAUDE now document the current mypy posture from `pyproject.toml`
  and the local `maestro-benchmark` entry point, including the current
  `python -m mypy` invocation, without overstating them as mandatory v1.0.0
  release gates

---

## [0.15.0] -- 2026-03-05

### Added
- **7 new warning detectors**: `W-pipes`, `W-multiline-verify`,
  `W-no-retry-verify`, `W-judge-retry-iterations`, `W-timeout-retry`,
  `W-context-no-budget`, and `W-judge-codex-contains`

### Fixed
- **Conditional skip success**: tasks skipped via `when:` no longer mark the run as
  failed
- **Interactive shell `/run`**: the shell now passes the same CLI contract expected by
  `_cmd_run()` (`plan=[...]`, `parallel=False`, `cache_dir=None`) and no longer fails
  on missing attributes
- **Interactive shell `/last`**: now reports the actual latest run directory instead of
  the active plan directory
- **Cache hashing/model normalisation**: cache resolution now shares the same alias
  source of truth as runtime execution for Codex, Qwen, and Ollama models

### Changed
- **Model aliases centralised** in `models.py` and reused by runners + cache logic
- **CLI banner** reorganised into 4 command groups for the no-args/help UX

### Documentation
- Updated `README.md`, `docs/VERSIONING.md`, and CLI reference to match the real
  command surface and cache paths
- `maestro replan` now documented with the correct `--model` flag
- Clarified the distinction between task result cache (`<run-dir>/.cache`) and
  workspace index cache (`<workspace_root>/.maestro-cache/index/`)
- Fixed Codex badge (Simple Icons removed openai; now uses inline SVG)

### Stats
- 1762 tests collected
- 15 CLI subcommands
- 6 engines

---

## [0.14.0] -- 2026-03-05

### Added
- **Adaptive re-planning (`maestro replan`)**: persist failed plan state, call frontier
  model to analyze failures and generate corrected plan YAML, diff-based human approval
  before re-run, circuit breaker (max 3 re-plans, repeated failure set detection);
  new module `replan.py` with `replan()`, `_extract_failed_state()`,
  `_build_analysis_prompt()`, `_call_analysis_model()`, `_parse_corrected_yaml()`,
  `_show_plan_diff()`, `_detect_exit_loop()`
- **Multi-plan execution**: `maestro run plan1.yaml plan2.yaml` runs multiple plans
  sequentially (default) or in parallel (`--parallel` flag); shared budget tracking
  across plans (`max_cost_usd` applies globally); aggregated summary with per-plan
  status, cost, tokens, and duration; new module `multi.py` with `run_multi_plan()`,
  `_run_sequential()`, `_run_parallel()`, `_aggregate_results()`,
  `_write_multi_summary()`
- **Ollama engine**: 6th engine (`engine: ollama`) for local model execution via
  `ollama run <model> "<prompt>"`; 12 model aliases including `llama3.1`, `llama3.2`,
  `qwen2.5-coder`, `deepseek-coder-v2`, and `starcoder2`; zero API cost; `OLLAMA_HOST`
  in env allowlist; execution profiles (plan passthrough, safe/yolo no-op for local);
  `maestro doctor` checks `ollama --version`
- New dataclasses: `ReplanAttempt`, `ReplanState`, `MultiPlanResult`
- New CLI subcommand: `maestro replan` with `--max-attempts`, `--model`,
  `--auto-approve` flags
- `run` subcommand accepts multiple plan paths (`nargs="+"`) with `--parallel` flag

### Fixed
- `cli.py`: `run_multi_plan()` call expanded individual kwargs instead of passing
  raw `argparse.Namespace` (TypeError bug)
- `replan.py`: temp directories from corrected plans are now cleaned up on exit
  (was leaking `maestro-replan-*` dirs in system temp)
- `models.py`: `ReplanState.total_cost_usd` default changed from `None` to `0.0`
  to match `+=` accumulation in replan loop

### Stats
- 1722 tests (up from 1685 in v0.13.0)
- 15 CLI subcommands (up from 14)
- 6 engines (up from 5)

---

## [0.13.0] -- 2026-03-05

### Added
- **`maestro suggest`**: run history analysis with optimization heuristics;
  new module `suggest.py` with `suggest_plan()`, `format_suggestions()`;
  deterministic heuristics (model downgrade/upgrade, cost/duration outliers,
  retry tuning) — no LLM calls; `--min-runs` and `--json` flags
- **`maestro shell`**: interactive REPL with slash commands (`/run`, `/validate`,
  `/suggest`, `/status`, `/explain`, `/plan`, `/last`, `/help`, `/quit`);
  readline autocomplete for commands and YAML files; graceful Windows fallback;
  new module `shell.py` with `run_shell()`, `ShellState` dataclass
- **Context compaction**: `context_compact` field on TaskSpec (bool, default false);
  `_compact_context()` in runners.py strips git diff noise, summarizes long
  stack traces, minifies JSON, normalizes whitespace
- **Qwen Code CLI engine**: 5th engine (`engine: qwen`); model aliases
  (`coder`, `coder-turbo`, `max`, `plus`, `qwq`); per-token pricing table
  with `MAESTRO_QWEN_PRICING_JSON` env var; execution profiles (plan/safe/yolo);
  `DASHSCOPE_API_KEY` in env allowlist; `maestro doctor` checks qwen CLI
- New dataclasses: `Suggestion`, `PlanSuggestions`, `SuggestionCategory`, `ShellState`

### Stats
- 1685 tests (up from 1638 in v0.12.0)
- 14 CLI subcommands (up from 12)
- 5 engines (up from 4)

---

## [0.12.0] -- 2026-03-04

### Added
- **Secrets masking**: `secrets:` plan field + `--mask-secrets` CLI flag;
  redacts env vars and API keys from task logs and manifests;
  supports explicit list or `auto` detection by name pattern
- **Plan imports**: `imports:` field for reusable task template composition;
  prefix-based ID namespacing; circular import detection; nested imports
- **Tags**: `tags:` field on tasks + `--tags`/`--skip-tags` CLI flags
  for semantic task filtering; dependency auto-inclusion
- **Approval gates**: `requires_approval:` field + `--auto-approve` CLI flag;
  interactive pause before risky tasks; non-interactive = auto-skip
- New error codes: E024 (invalid secrets), E025-E028 (import validation),
  E029 (approval misconfiguration)

### Stats
- 1638 tests (up from 1548 in v0.11.0)

---

## [0.11.0] -- 2026-03-04

### Added
- **`maestro explain`**: show cache hit/miss status for each task in a plan — reports whether each task would run or be cached, with hash diagnostics; `explain.py` module with `TaskExplanation`, `PlanExplanation`, `explain_plan()`, table and JSON formatters
- **`maestro status`**: show pipeline staleness vs last run — compares current task hashes against the last run manifest to report up-to-date/stale/never-run/failed/skipped state per task; `status.py` module with `TaskPipelineStatus`, `PlanPipelineStatus`, `plan_status()`
- **`maestro eval`**: batch judge evaluation on completed runs — loads an eval YAML suite, resolves task patterns (glob/exclude), runs judge evaluation per task with per-task overrides, returns CI-friendly exit codes; `eval.py` module with `EvalResult`, `EvalSuiteResult`, `load_eval_spec()`, `run_eval()`
- **Eval YAML schema**: reusable judge suites with `name`, `tasks` (glob patterns), `exclude`, `judge` block, `overrides` per task, `timeout_sec`
- **`task_hash` on TaskResult**: persisted in `result.json` and `run_manifest.json` for post-run staleness detection
- CLI: 12 subcommands (up from 9), updated banner command list
- 7 new pitfall warning detectors in `_collect_warnings()`: guard_command shell warnings, prompt_md_heading `#` prefix, template variable validation, run_dir backslash check, bash-only syntax detection (Windows), retry_delay_sec length check, env var reference cross-check

### Stats
- 1548 tests (up from 1487 in v0.10.0)

---

## [0.10.0] -- 2026-03-04

### Added
- **Likert-scale rubrics**: `judge.criteria` now accepts `type: rubric` dicts with named criteria, discrete 1-5 score levels with anchored descriptions, `min_score`, and `weight`; LLMs score more consistently against anchored descriptions than free-form 0.0-1.0; `RubricLevel` and `RubricCriterion` dataclasses; `_format_rubric_criteria()` and `_evaluate_rubric_criteria()` in runners.py
- **G-Eval two-phase scoring**: `judge.method: g_eval` generates evaluation steps first (Phase 1), then scores using those steps (Phase 2) — significantly improves judge consistency; `_generate_eval_steps()` in runners.py; `_GEVAL_STEPS_PROMPT_TEMPLATE` and `_GEVAL_SCORE_PROMPT_TEMPLATE`; falls back to direct scoring on error
- **Score aggregation strategies**: `judge.aggregation` field (`mean` | `min` | `weighted_mean`); `mean` (default, backward-compatible), `min` (all criteria must pass — strict mode for security audits), `weighted_mean` (per-criterion weights from rubric); `_aggregate_scores()` in runners.py
- **Comparative/pairwise evaluation on retry**: when judge fails with `on_fail: retry`, the next attempt is also compared against the previous attempt ("Is this better?"); `_run_comparative_evaluation()` with `_COMPARATIVE_JUDGE_PROMPT_TEMPLATE`; `JudgeResult.previous_score` tracks comparison; relative comparison is easier for LLMs than absolute scoring
- **Named criteria presets**: `judge.preset` field (`code_quality` | `security_audit`); `JUDGE_PRESETS` constant with pre-defined rubric criteria and calibrated thresholds; preset provides default criteria, pass_threshold, and aggregation — explicit YAML values override preset defaults
- **CLI banner**: ASCII art banner with warm gold/amber ANSI color gradient on `--help` and no-args invocation; `--version` flag

### Stats
- 1487 tests (up from 1427 in v0.9.0)

---

## [0.9.0] -- 2026-03-04

### Added
- **Typed assertion criteria**: `judge.criteria` now accepts both plain strings (LLM-evaluated) and typed assertion dicts (`contains`, `regex`, `is-json`, `llm-rubric`, `cost_under`, `duration_under`); deterministic checks run locally at zero cost, LLM only invoked for subjective criteria; `ASSERTION_TYPES` constant; `_evaluate_typed_assertion()` in runners.py
- **BM25-style intent scoring**: `_score_section()` upgraded with IDF weighting and TF saturation; `_compute_idf()` computes inverse document frequency across all upstream sections; rare terms score higher, repeated terms saturate; ~40 lines pure Python, zero deps
- **Priority-based context eviction**: `_apply_context_budget()` now trims least-relevant upstreams first (greedy knapsack) instead of proportional trimming; preserves best context intact
- **`guard_command:` field**: lightweight alternative to `judge:` -- shell command that validates task output via stdin pipe; exit 0 = pass, non-zero = fail; runs after verify_command, before judge; `_run_guard_command()` in runners.py
- **Budget warning threshold**: `budget_warning_pct` field (plan + defaults level, default 0.8); emits `budget_warning` event when running cost approaches `max_cost_usd` limit; one-shot warning to avoid spam
- **`max_iterations` per task**: hard cap on total execution attempts (initial + retries + judge retries); prevents infinite retry spirals; validated via E022
- **Graph-distance decay**: `_compute_hop_distances()` (BFS) and `_apply_hop_decay()` in scheduler.py; direct deps keep 100% context, transitive deps decay by `0.8^(hops-1)`; applied before context budget enforcement

### Error Codes
- **E022**: max_iterations value out of range (< 1)
- **E023**: budget_warning_pct value out of range (must be 0.0-1.0 exclusive)

### Events
- `budget_warning` -- running cost approaching max_cost_usd limit (spent, limit, pct)

### Stats
- 1427 tests (up from 1386 in v0.8.0)

---

## [0.8.0] -- 2026-03-04

### Added
- **Intent-driven context filtering**: `_extract_keywords()`, `_split_into_sections()`, `_score_section()`, `_apply_intent_filtering()` in scheduler.py; uses downstream task prompt keywords to score and filter upstream output sections instead of blind proportional truncation; zero-token, keyword-based scoring
- **Context exhaustion detection**: `context_exceeded` failure category with per-engine regex patterns for context window errors (token limit, input too long, context length exceeded)
- **Rate limiting detection**: `rate_limited` failure category for API throttling errors (429, quota exceeded, too many requests, retry after, overloaded)
- **Smart retry context compression**: `_compress_context_for_retry()`, `_compress_upstream_context_for_retry()` compress upstream context on `context_exceeded` retries; `_CONCISENESS_HINT` injected into retry prompts when context was the bottleneck
- **Handoff report generation**: `HandoffReport` dataclass with failure analysis, partial output, and suggested next steps; `_generate_handoff_report()` creates structured reports for manual pickup after unrecoverable failures
- **Context compression metrics**: `context_raw_tokens`, `context_final_tokens`, `context_compression_ratio` fields on `TaskResult`; `context_compression` event emitted when filtering reduces context

### Fixed
- **Timezone bug**: `_local_timestamp()` used `time.daylight` (whether DST is defined) instead of checking if DST is currently active; replaced with `datetime.now().astimezone()` which handles DST correctly
- **Run success with conditional skips**: `_SUCCESS_LIKE` set now includes `"skipped"` in run-level success calculation; `when`-expression skipped tasks no longer mark the entire run as failed

### Events
- `context_compression` -- context filtering reduced upstream tokens (raw/final/ratio)

### Stats
- 1386 tests (up from 1314 in v0.7.0)

---

## [0.7.0] -- 2026-03-03

### Added
- **Recursive context pipeline**: `context_mode: recursive` with three-pass index→extract→brief pipeline; cheap haiku calls map workspace structure, extract relevant file snippets, and produce focused briefs before engine dispatch; `{{ workspace_brief }}` template variable for prompt injection
- **Persistent workspace indexing**: `<workspace_root>/.maestro-cache/index/` stores workspace structural maps with hash/mtime invalidation; stat-only quick validation (~50ms for 1000 files); cross-task and cross-run reuse when workspace unchanged
- **Workspace index exclusion**: `workspace_index_exclude` field (task + plan defaults level) for custom glob patterns to exclude from indexing

### Error Codes
- **E021**: context_mode: recursive without resolvable workspace root
- **E108**: Workspace index build failure
- **E109**: Workspace extraction LLM call failure
- **E110**: Workspace brief LLM call failure

### Events
- `workspace_brief` — recursive context pipeline completed (token_estimate, files, duration)

### Stats
- 1314 tests (up from 1165 in v0.6.0)

---

## [0.6.0] -- 2026-03-03

### Added
- **Context budget awareness**: `context_budget_tokens` field (task + plan level) to track and enforce token limits; auto-summarize/truncate near token limits; prevents expensive context operations when budget is exhausted; `context_budget_applied` event on threshold breach
- **LLM-as-Judge quality gates**: `judge:` block with structured criteria, `pass_threshold` (0.0-1.0), and `on_fail` action (fail/warn/retry); uses haiku for fast structured evaluation; returns `pass` or `fail` verdict; integrates with quality gate workflow
- **Smart retry with failure analysis**: `_classify_failure()` categorizes errors (timeout, test, compilation, integration, etc.) from task output; tracks failure history across retries; auto-generates escalation hints on repeated patterns; `task_failure_analysis` event on classification
- **Checkpoint protocol**: `checkpoint: true` on task creates `MAESTRO_CHECKPOINT_DIR`, persists progress across retries, auto-injects checkpoint context on retry; enables safe resumption for long-running tasks; `task_checkpoints` event on checkpoint write

### Error Codes
- **E019**: context budget range validation error
- **E020**: judge block configuration error
- **E107**: judge execution/evaluation runtime error

### Events
- `context_budget_applied` — context_budget_tokens threshold breached, auto-summarized/truncated
- `judge_verdict` — LLM judge returned structured pass/fail verdict
- `task_failure_analysis` — failure classified (timeout/test/compilation/integration)
- `task_checkpoints` — checkpoint created in MAESTRO_CHECKPOINT_DIR

### Stats
- 1165 tests (up from 991 in v0.5.0)

---

## [0.5.0] -- 2026-03-02

### Added
- **Matrix tasks**: `matrix:` field for parameterized task expansion; Cartesian product of matrix dimensions; supports string lists and template variable interpolation; auto-generated task IDs with `matrix_index` context variable
- **Task groups**: `group:` field for nested sub-plan execution; enables hierarchical task organization; groups can depend on tasks and be depended upon; group status determined by member task outcomes (failed if any member failed, soft_failed if any soft_failed)
- **Content-addressable caching**: Merkle DAG cache system for deterministic task deduplication; `cache:` per-task control (enabled/disabled); `--no-cache` CLI flag to bypass cache entirely; auto-computes content hash from task spec; cache hits skip task execution and reuse prior output
- **`maestro diff` subcommand**: side-by-side comparison of two runs (by run ID or directory path); compares task status, exit code, cost, token usage, and duration; tabular output highlighting deltas; useful for validating model changes, cost optimizations, and performance improvements
- **GitHub Copilot CLI engine**: 4th AI engine (`engine: copilot`) providing multi-model access via GitHub Copilot subscription; 22 model aliases spanning Claude, GPT, Gemini, and Grok; `--autopilot --silent --no-color` base flags; `--yolo` profile normalization; system prompt prepended to prompt (like Gemini); `defaults.copilot` plan-level config; `maestro doctor` checks copilot CLI availability; cost tracking deferred (subscription-based premium requests, not per-token); env allowlist includes `COPILOT_GITHUB_TOKEN`, `GH_TOKEN`, `GITHUB_TOKEN`; 48 dedicated tests

---

## [0.4.0] -- 2026-03-01

### Added
- **Web UI token display**: token usage metrics card on run detail page, per-task token counts in task cards, token breakdown tooltip (input/cached/output/cache_creation); dashboard stats cards show total tokens; `/runs/stats` API returns `total_tokens`, `avg_tokens_per_run`, `tokens_by_model`
- **Cost/token breakdown charts**: per-task token usage bar chart, cost/token by engine donut charts, efficiency metrics (cost per 1K tokens, cache hit rate), token trend on dashboard (`charts.js`)
- **`--output jsonl` streaming mode**: structured JSON Lines events to stdout during execution (`run_start`, `task_start`, `task_complete`, `task_skip`, `run_complete`); suppresses `[maestro]` human-readable output; compatible with `jq` and `--dry-run`
- **Webhook notifications**: `webhook_url` plan-level field + `--webhook URL` CLI flag; POSTs JSON payload on run completion (plan_name, run_id, success, cost, tokens, duration); uses `urllib.request` (zero deps); failure never affects run result
- **`maestro report` subcommand**: generates self-contained HTML report from run directory; includes header, metrics cards, sortable task table, Gantt timeline, cost/token charts, expandable task details; works with `file://` protocol; `-o` flag for custom output path
- **Per-run `events.jsonl`**: timestamped structured event log in run directory; events: `run_start`, `task_start`, `task_complete`, `task_skip`, `context_summarize`, `budget_exceeded`, `run_complete`; flushed after each write for real-time observability

### Fixed
- **JSONL mode text leak**: `--output jsonl` now suppresses validation warnings and dry-run checklist that were printed to stdout outside the scheduler

### Changed
- `run_plan()` accepts `output_mode` parameter (`"text"` or `"jsonl"`, default `"text"`)
- CLI `_cmd_run()` passes `output_mode` to scheduler and suppresses non-JSONL output
- `/runs/stats` API response now includes token aggregation fields

---

## [0.3.0] — 2026-03-01

### Added
- **Structured error codes**: E001-E018 (validation) and E100-E105 (runtime) with `code=` parameter on `PlanValidationError` and `TaskExecutionError`; errors display as `[E001] message`
- **`maestro doctor` subcommand**: diagnoses environment health (Python version, PyYAML, engine CLIs on PATH, Git); supports `--json` for machine-readable output
- **Schema version migration**: `_CURRENT_SCHEMA_VERSION` constant in loader, `_migrate_plan()` infrastructure for future schema evolution
- **Git Bash auto-detection**: `_find_git_bash()` searches standard Windows install paths for Git Bash binary
- **`--verbose` / `--quiet` flags**: control output verbosity on `maestro run`; `Verbosity` type in models; scheduler respects verbosity levels
- **YAML anchor documentation**: docs and tests confirming `&anchor`, `*alias`, and `<<: *merge` support via `yaml.safe_load()`

### Fixed
- **`allow_failure` + timeout bug**: tasks with `allow_failure: true` that timed out (exit=124) were incorrectly marked as `failed` instead of `soft_failed`; dependents were blocked instead of proceeding

### Changed
- Loader validation messages now include structured error codes (`[E001]`, `[E002]`, etc.)
- Scheduler output respects verbosity levels (quiet suppresses non-essential output)

---

## [0.2.0] — 2026-03-01

### Added
- **Gemini engine**: third AI engine alongside Codex and Claude, with model aliases (flash/pro/flash-3/pro-3/pro-3.1), `--approval-mode yolo` normalization, system prompt prepend, env var isolation
- **Token usage tracking**: `TokenUsage` dataclass with `input_tokens`, `cached_tokens`, `output_tokens`, `cache_creation_tokens`, and `total_tokens` property
- **Per-engine pricing tables**: Claude (haiku/sonnet/opus/opusplan), Gemini (6 models), Codex (default fallback) — all overridable via `MAESTRO_*_PRICING_JSON` env vars
- **Generalized cost estimation**: token-based cost estimation for all 3 engines when CLI doesn't report cost directly
- **Token aggregation**: `PlanRunResult.total_tokens`, per-task `TaskResult.token_usage`
- **Summary tokens**: `run_summary.md` now includes Tokens row in header table and Tokens column in per-task table
- **Token backfill**: `maestro backfill-costs` now also populates `token_usage` for old runs (best-effort engine inference from command string)
- **Context window constants**: `CONTEXT_WINDOWS` dict in models.py (informational, for future scaffold cost estimation)
- CHANGELOG.md and versioning rules

### Changed
- `_extract_cost_from_log` now infers engine from log's `command=` line for token-based estimation (backward compatible)
- `_normalize_codex_pricing_table` renamed to `_normalize_pricing_table` (alias preserved)
- `_estimate_codex_cost` renamed to `_estimate_cost_from_tokens` (alias preserved)

---

## [0.1.0] — 2026-03-01

Initial release. Full-featured CLI orchestrator for multi-step AI execution plans.

### Core
- **DAG scheduler**: topological sort, `ThreadPoolExecutor`, parallel execution with `max_parallel`, fail-fast semantics
- **3 AI engines**: Codex (OpenAI), Claude (Anthropic), Gemini (Google) — plus raw shell commands
- **YAML plan format**: version 1 schema with validation, cycle detection (DFS), duplicate ID checks
- **Execution profiles**: `plan` (as-is), `safe` (strip dangerous flags), `yolo` (add bypass flags)
- **CLI**: 6 subcommands — `validate`, `run`, `cleanup`, `scaffold`, `ui`, `backfill-costs`

### Task Features
- `depends_on` — DAG dependency declarations
- `pre_command` — setup command (failure prevents main execution)
- `verify_command` — post-execution verification (failure marks task as failed)
- `max_retries` (0-3) — retry main+verify loop on failure (pre_command NOT retried)
- `retry_delay_sec` — constant (float) or per-retry (list) backoff delays
- `allow_failure` — soft failure mode (`soft_failed` status, dependents proceed)
- `timeout_sec` — per-task timeout (exit_code=124 on timeout)
- `when` — conditional execution expressions (`{{ task.status }} == value`)
- `context_from` + `context_mode` — inter-task output passing (raw/summarized/map_reduce)
- `edit_policy` — efficient/strict editing instructions injected into prompts
- `append_system_prompt` — custom system prompt additions per engine/task
- Error feedback injection — verify failures auto-injected into engine retry prompts

### Plan Features
- `max_cost_usd` — soft budget limit (running task completes, pending tasks skipped)
- `defaults` — plan-level engine config (model, reasoning_effort, args, system prompt)
- `workspace_root` — base directory for all tasks
- Template variables: `{{ workspace_root }}`, `{{ plan_name }}`, `{{ task_id }}`
- Context variables: `{{ task-id.status }}`, `{{ task-id.stdout_tail }}`, `{{ task-id.log }}`, etc.
- Structured context: `{{ task-id.files_changed }}`, `{{ task-id.errors }}`, `{{ task-id.decisions }}`, etc.

### Scaffolding
- `maestro scaffold` — generate full YAML plans from brief descriptions
- `PlanBrief` / `TaskBrief` dataclasses for simplified input
- Automatic model routing (task_type -> engine + model)
- Quality gates generation (code-review, QA verification, build verify)
- Anti-stalling prompt injection for implementation tasks
- Cost safety checks (`--cost-check` flag)

### Output & Observability
- Per-task `.log` (transcript) and `.result.json` (structured result)
- `run_manifest.json` — aggregated run results
- `run_summary.md` — human-readable summary with tasks table and timeline waves
- Cost extraction from engine output (regex patterns + JSON parsing)
- Parallelism metrics (wall time vs sequential time, savings percentage)
- `--resume` / `--resume-last` — resume from prior run (skip succeeded tasks)
- `maestro backfill-costs` — retroactively populate cost data in old runs
- `maestro cleanup` — prune old run directories

### Web UI
- `maestro ui` — FastAPI backend + vanilla HTML/CSS/JS dashboard
- Dashboard: stats cards, status donut chart, cost trend, runs table with filtering
- Run detail: metrics cards, task duration/cost charts, CSS Gantt timeline, task cards
- Log viewer: line numbers, syntax highlighting, search, level filter, keyboard shortcuts
- SSE streaming: real-time task completion events via filesystem polling

### Validation
- Blocking: `PlanValidationError` for schema violations (fail-fast)
- Non-blocking: 7 warning detectors (Windows shell, wrong bash, non-ASCII, backslash paths, missing timeouts, unknown models, edit_policy on shell)
- Dry-run checklist: items NOT validated (CLI tools, workdir, network, git state)
- Environment isolation: `_ENV_ALLOWLIST` controls inherited env vars

### Engines
- **Codex**: model aliases (5.3/5.2/5.1/5/5-mini), reasoning_effort (minimal-xhigh), `--dangerously-bypass-approvals-and-sandbox` normalization
- **Claude**: model aliases (haiku/sonnet/opus/opusplan), reasoning_effort (low/medium/high, Opus only via `CLAUDE_CODE_EFFORT_LEVEL`), `--dangerously-skip-permissions` normalization
- **Gemini**: model aliases (flash/flash-lite/pro/flash-3/pro-3/pro-3.1), `--approval-mode yolo` normalization, system prompt prepended to user prompt
