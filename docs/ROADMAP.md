# Maestro CLI — Roadmap

v1.x is **feature-complete** as of v1.37.0. v2.x development has begun.
This document focuses on the v2.0+ horizon and tracks what was shipped
in v1.x and v2.x for reference.

For detailed release notes, see [CHANGELOG.md](../CHANGELOG.md).

---

## v1.x Feature Freeze

v1.37.0 marks the **feature freeze** for the 1.x line. Future 1.x
releases (v1.37.1, v1.38.0, etc.) are limited to:
- Bug fixes
- Documentation improvements
- Test coverage expansion
- Performance optimizations
- Broader strict mypy coverage

New features go to v2.x.

---

## v2.0 Horizon

v2.0 is organised in **5 phases** ordered by dependency and value.
Each phase builds on the previous one. Features within a phase are
independent and can ship in any order.

```
                    ┌─────────────────────────┐
                    │  Phase 0 — Foundation    │
                    │  Python SDK              │
                    │  Capability-Based Access  │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │  Phase 1 — Execution     │
                    │  Remote Backends          │
                    │  Council Topologies       │
                    │  llama-cpp Engine         │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │  Phase 2 — Intelligence   │
                    │  Knowledge + Memory v2    │
                    │  Semantic Firewalls       │
                    │  Observability v2         │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │  Phase 3 — Search         │
                    │  MCTS Workflow Search     │
                    │  Self-Evolving Replan     │
                    └────────┬────────────────┘
                             │
                    ┌────────▼────────────────┐
                    │  Phase 4 — Interop        │
                    │  A2A Protocol Layer       │
                    │  Meta-Agent Generation    │
                    └──────────────────────────┘
```

### Dependency graph

```
Capability-Based Access ───┐
                           ├──► Remote Execution ──► A2A Protocol
Python SDK ────────────────┘         │                    │
                                     │                    ▼
                                     │           Meta-Agent Generation
                                     ▼
                          Knowledge + Memory v2 ──► MCTS ──► Self-Evolving
                                     │
                                     ▼
                              Semantic Firewalls
```

### Immediate next steps (recommended sequence)

The roadmap below is phase-structured for long-term planning. For
day-to-day prioritisation, the recommended execution order is:

#### Sprints 1-3 — **Shipped in v2.3.0**

All three sprints completed and tagged:

1. ~~**Tag a release**~~ — v2.3.0 consolidates everything shipped on
   main after v2.2.0: Semantic Firewalls (MCP-path + pass-2),
   Codebase Graph (PageRank, `__init__` re-exports), Observability v2
   (GenAI attrs, memory-write tracing), Semantic Cache (tool-failure
   exclusion), `--set KEY=VALUE`, RRF Fusion, pre-hash normalization,
   consolidation safety gates, LLM tier in compaction, eviction "why"
   fields, MCTS foundations, self-evolving replan foundations, and
   cross-phase security contract foundations.

2. ~~**`--set key=value`**~~ — **Shipped in v2.3.0**.
3. ~~**Pre-hash normalization**~~ — **Shipped in v2.3.0**.
4. ~~**RRF Fusion**~~ — **Shipped in v2.3.0**.
5. ~~**Consolidation safety gates**~~ — **Shipped in v2.3.0**.
6. ~~**LLM tier in compaction**~~ — **Shipped in v2.3.0**.
7. ~~**Eviction "why" fields**~~ — **Shipped in v2.3.0**.
8. ~~**MCTS Workflow Search foundations**~~ — **Shipped in v2.3.0**.
9. ~~**Cross-Phase Security Contract foundations**~~ — **Shipped in v2.3.0**.
10. ~~**Self-Evolving Replan foundations**~~ — **Shipped in v2.3.0**.

#### Sprint 4 — Finish Phase 3 — **Shipped in v2.4.0**

Focus was: **close the remaining Phase 3 "Done When" gaps.**

11. ~~**Simulation cache**~~ — **Shipped in v2.4.0**:
    successful simulations now persist a normalized, model-family-aware
    `simulation_plan_hash`, and multi-variant `replan` reuses matching
    successful `ScoreRecord` entries with a confidence discount instead
    of re-executing equivalent topologies.

12. ~~**Phase 3 integration benchmarks**~~ — **Shipped in v2.4.0**:
    `maestro-benchmark` now ships deterministic `replan_pruning`,
    `replan_population`, and `replan_novelty` cases demonstrating
    saved simulations from historical pruning, higher selected fitness
    for N>1 tournament search versus single-shot replan, and
    novelty-driven candidate selection shifts.

13. ~~**Red-team test for safety contract**~~ — **Shipped in v2.4.0**:
    integration coverage now seeds poisoned `failure_pattern`
    knowledge, verifies that it appears in replan guidance, confirms
    the resulting harmful mutation is blocked by the generated-plan
    security gate before simulation, and checks that search continues
    with a safe candidate instead.

#### Post-v2.4.0 — Re-baseline before opening v2.5.0

No v2.5.0 sprint is active yet, but the planning baseline is now
explicit. If a v2.5.0 sprint is opened, it should stay local-first and
follow this order:

1. ~~**Session memory extraction**~~ — MVP landed on main. The
   deterministic path now ships extraction/injection plus resume
   coverage on top of the existing `WatchIteration` excerpts and SQLite
   `session_snapshots`. Optional follow-up remains: LLM-polished
   consolidation and reuse of the same snapshot shape in chat hot-cache
   bootstrap.

2. ~~**Chat context bootstrap**~~ — MVP landed on main. `maestro chat`
   now auto-loads hierarchical `AGENTS.md` / `CLAUDE.md` files in
   root-to-leaf order at startup, reusing the existing
   `session.context_files` path, with `--no-auto-context` as an escape
   hatch. Deferred follow-up remains: reuse watch session snapshots for
   chat hot-cache bootstrap rather than inventing a second memory path.

3. ~~**Skill registry v2**~~ — MVP landed on main. `maestro skill`
   now goes beyond list/search into deterministic recommendations driven
   by explicit trigger metadata and small inspectable chain hints. It
   borrows the good part of `claude-code-on-steroids` (trigger/chain
   UX) without adding opaque routing or provider-specific hooks.

4. ~~**Web UI collaboration surfaces**~~ — MVP landed on main. The Web
   UI now exposes ownership, blockers, recent activity from
   `events.jsonl`, and task/runtime identity in run detail, plus compact
   collaboration summaries in the dashboard. This intentionally stays a
   local-first presentation layer over run manifests and event logs
   rather than turning Maestro into a daemon or issue tracker.

5. **Remote Execution Backends** — biggest upside and biggest refactor;
   reconsider only after the local-first backlog above is complete, and
   then only as a deliberate branch/prototype with security review.
6. **A2A + Meta-Agent generation** — keep deferred until remote
   execution, firewall coverage, and traceability are mature enough to
   support networked delegation safely.

#### Explicitly deferred

- **Remote Execution Backends** — biggest refactor in v2, but does
  not block the local-first v2.5.0 backlog above. Current users are
  local-first. Re-evaluate only if those constraints start to hurt.
- **Embedding-assisted cache lookup** — requires external dependency
  (`[memory]` extra), uncertain ROI vs. pre-hash normalization.
- **Multi-plan TUI** — high effort, low usage. Text/JSONL output
  covers multi-plan execution adequately.
- **Tree-sitter for `symbols.py`** — regex fallback covers 10
  languages. Revisit when non-Python codebase graph is needed.
- **A2A Protocol + Meta-Agent Generation** — Phase 4 remains
  deferred until Remote Execution, firewall coverage, and
  traceability are mature.

#### External Repo Scans (2026-04)

Net takeaways:

- **`aaif-goose/goose` + `milla-jovovich/mempalace`** — still validate
  `maestro chat` context-file auto-discovery and `watch` session-memory
  extraction as the strongest local-first follow-ons.
- **`AgriciDaniel/claude-obsidian` + `GadaaLabs/claude-code-on-steroids`**
  — reinforce the same direction from a different angle: hot-cache
  bootstrap, recent-context-first retrieval, trigger-based skill
  routing, and chain recipes.
- **`multica-ai/multica` + `win4r/ClawTeam-OpenClaw`** — useful as
  product and operational references for ownership/blockers/agent
  identity plus per-agent worktree/session ergonomics, but not as
  core-runtime architecture for Maestro.
- **Re-prioritisation result** — the post-v2.4.0 local-first backlog
  has now landed end-to-end: session memory extraction -> chat context
  bootstrap -> skill registry v2 -> Web UI collaboration surfaces. This
  still does not reprioritise Maestro toward daemon/cloud/A2A work; the
  next major item remains Remote Execution Backends as an explicit
  exploratory branch, not a baseline rewrite.

---

### Phase 0 — Foundation (v2.0.0) -- SHIPPED

> Prerequisites for everything else. Without these, subsequent phases
> build on unstable ground.

#### Python SDK -- Shipped in v2.0.0

Freeze internal module APIs as a supported programmatic interface.
Enable `import maestro_cli` as a library, not just a CLI tool.

- Public API surface: `load_plan()`, `run_plan()`, `validate_plan()`,
  `scaffold_plan()`, `blame_run()`, `diff_runs()`, `audit_plan()`
- Typed return values (existing dataclasses)
- Versioned API contract with deprecation policy
- `py.typed` marker for downstream type checking
- 29 exports in `__all__`, `EventCallback` type alias

**Why first**: every v2 feature (remote exec, MCTS, meta-agent) needs a
stable programmatic interface to compose against. Today the codebase is
CLI-first -- module functions work but are not contracted.

#### Capability-Based Tool Access -- Shipped in v2.0.0

Per-task `allowed_tools:` list for prompt injection containment.
Restricts which tools each engine task can invoke.

```yaml
tasks:
  - id: write-code
    engine: claude
    allowed_tools: [Read, Write, Edit, Bash]
    prompt: "Implement the feature"

  - id: review-code
    engine: claude
    allowed_tools: [Read, Grep, Glob]   # no write access
    prompt: "Review the implementation"
```

- Loader validation: tool names against known sets per engine
- Policy engine integration: `has_allowed_tools` + `allowed_tools` fields
- Audit rule: SEC023 for engine tasks without `allowed_tools` in
  untrusted contexts
- Claude: `--disallowedTools`. Codex: `--sandbox` levels.
  Gemini/Copilot/Qwen/Ollama/Llama: system-prompt injection.
- Inheritable via `defaults.<engine>.allowed_tools`

**Why first**: security prerequisite for remote execution and A2A.
Without tool restrictions, sandboxed agents can still invoke arbitrary
tools. References: Progent (DSL), MiniScope (least-privilege), AC4A.

---

### Phase 1 — Execution Model (v2.1.0)

> The leap from "local CLI tool" to "execution platform".

#### Remote Execution Backends

`executor:` per-task field (local/docker/ssh/cloud). Plan-level
`executors:` block for named executor definitions. Fundamentally changes
the execution model from local shell-out to sandboxed/remote dispatch.

```yaml
executors:
  docker-sandbox:
    type: docker
    image: "python:3.14-slim"
    volumes: ["./:/workspace"]
  remote-gpu:
    type: ssh
    host: "gpu-box.internal"
    user: "maestro"

tasks:
  - id: run-tests
    executor: docker-sandbox
    command: ["pytest", "tests/"]
  - id: train-model
    executor: remote-gpu
    engine: claude
    prompt: "Train the classifier"
```

- Backend abstraction: `runners.py` → `runners/` package with
  `LocalBackend`, `DockerBackend`, `SSHBackend`, `CloudBackend`
- Executor lifecycle: create → execute → collect artifacts → cleanup
- Integration with `allowed_tools` for sandboxed tool access
- Integration with `worktree` for isolated workspaces in containers
- FSA-based state recovery for remote task failure/reconnection

**Architecture change**: largest refactor in v2. References: Agent
Sandbox (k8s-sigs), Firecracker, gVisor, Kata Containers, nsjail.

#### Council Topology Extensions -- Shipped in v2.1.0

Extend `context_mode: council` with `chain` (sequential pass-through)
and `graph` (concurrent peer-to-peer discussion) topologies beyond
the existing `star` topology.

- `chain`: participants form a pipeline, each refining the previous
  response (like a review chain); no consolidation step
- `graph`: concurrent peer-to-peer discussion where participants
  exchange messages directly via explicit `connections:` adjacency
- 3 topologies total: star, chain, graph
- E072 validation for graph connections, W28 warnings
- `council_chain_step` event

References: G-Designer, Graph-of-Agents, AgentPrune, MARBLE.

#### llama-cpp / llama-server Engine -- Shipped in v2.1.0

Local inference engine via `llama-cli` (llama-cpp). Zero API cost,
full privacy. Complements remote execution: expensive tasks run
remotely, cheap tasks run locally.

- 7th engine: `llama`
- 7 model aliases (llama3, codellama, phi3, mistral, etc.)
- `LLAMA_MODEL_DIR` env var
- Routing tier table, chat integration
- Token counting for budget tracking
- Integration with `model: auto` routing

---

### Phase 2 — Intelligence (v2.2.0)

> From "execute what I tell you" to "learn and improve".
>
> Design principle: treat **persistence, provenance, time, and
> security** as first-class requirements, not features. Every learned
> fact must be attributable, have a lifecycle, and be reversible.

#### Knowledge + Memory v2 — Partially Shipped

SQLite-backed persistent memory replacing the current JSONL-based
`knowledge.py`. The architectural shift from stateless to stateful.
Module: `memory.py` (~400 lines).

**Persistence layer (SQLite)** — **Shipped in v2.2.0**:
- WAL mode enabled by default (readers never block writers)
- Connection-per-thread lifecycle via `threading.local`
- Write serialization handled by SQLite WAL (single writer)
- Schema versioning (`schema_version` table) for safe migrations
- Automatic JSONL→SQLite migration on first access
- `knowledge.py` delegates store/load/compact to `memory.py`
  with JSONL fallback on any exception

**Semantic cache** (subsystem 1):
- Content-addressed task results with **pre-hash normalization**:
  strip whitespace, normalize model aliases, sort environment keys,
  deduplicate args — increases hit rate on paraphrased/reordered
  inputs without external dependencies (zero-cost improvement over
  current raw SHA-256 in `cache.py`)
- Optional embedding similarity lookup via `[memory]` extra for
  cross-phrasing matches (two-stage: fast candidate retrieval,
  then exact verification before reuse)
- Cache key schema: task signature + engine config + model family +
  **policy version** + content hash (policy changes invalidate cache)
  — core policy-versioned keys **Shipped in v2.2.0**
- **Negative caching**: short-TTL entries (default 5min, configurable
  per task via `negative_cache_ttl_sec`) for repeated failures on
  expensive tasks — prevents redundant retries; current short-TTL
  `failed` / `soft_failed` cache entries **Shipped in v2.2.0**
- Contamination control: never cache untrusted-context outputs
  (`context_trust: untrusted`), partial outputs, tainted results,
  or tool failures — untrusted / tainted / partial exclusion
  **Shipped in v2.2.0**; explicit tool-failure classification and
  runtime `tool_failure_count` exclusion **Shipped in v2.3.0**
- Eviction with "why" fields (latency win, expensive tool avoided,
  safety-reviewed) so consolidation can compress intelligently

**Temporal knowledge graph** (subsystem 2):
- Bi-temporal model: `valid_from`/`valid_to` (when true in reality)
  + `recorded_at` (when the system learned it); SQLite columns, not
  a full temporal DB engine
- **Immutable-with-windows**: edges are never mutated; invalidation
  creates a new record with updated `valid_to` + replacement record
  with new `valid_from` (history table pattern for auditability)
- **Conflict resolution**: contradictory facts coexist with different
  provenance and trust tiers; "latest wins" only within the same
  trust tier; consolidation resolves cross-tier conflicts —
  **Shipped in v2.2.0**
- **Event time extraction boundary**: explicit separation between
  what extraction can infer (timestamps from logs, commit dates)
  vs. what remains uncertain until confirmed (relative dates,
  ambiguous references)
- Entity confidence scores on relations (CALLS > MENTIONS); not all
  edges are equal — **Shipped in v2.2.0**
- Point-in-time queries: "what did the agent believe at time T?"
  via transaction-time filtering

**Memory write policies** (subsystem 3 — security perimeter):
- **Mandatory provenance fields**: source_type (task/tool/web/file),
  source_id (run + task ID), pipeline_version, trust label
  (**computed**, not user-provided)
- **Staged write validation**:
  (1) syntactic: schema conformance
  (2) security: firewall-style detection of instruction-like
      payloads in "facts" (instructionality score — regex for
      imperative tool-control language)
  Semantic consistency checks (vs. existing KG) deferred to v2.4+
- **Poisoning detection signals**:
  - *Retrieval dominance* — **Shipped in v2.2.0**: track per-memory-block retrieval count
    per distinct query cluster; flag blocks exceeding 3σ from mean
    as suspicious — zero embedding cost (pure frequency counting)
  - *Instructionality score*: regex for imperative tool-control
    language (contains imperative verbs + tool names)
  - *Embedding outlier influence*: deferred to v2.4+ (requires
    embedding infra from `[memory]` extra)
- **Quarantine**: suspicious writes are segregated, not rejected —
  flagged for review, excluded from normal retrieval
- **Poisoning harness** — **Shipped in v2.2.0**: replay prompt
  retrieval patterns against stored knowledge to validate dominance
  detection and quarantine behaviour
- **Rollback**: memory writes are reversible by transaction time
  (the operational payoff of bitemporality)

**Index-and-detail retrieval** (query interface) — **Partially shipped in v2.2.0**:
- Lightweight index always available: plan-name + task-pattern +
  category + one-line summary per record (~200 entries max);
  injected into prompt as `{{ knowledge_index }}`
- Full records loaded on demand via BM25 relevance scoring
  against downstream task prompt keywords via lightweight local
  scoring — zero LLM cost
- Remaining: optional LLM-based relevance selection via cheap
  sideQuery when BM25 confidence is low (< 0.3 threshold)
- **Exclusion rule**: never store what's derivable from the
  codebase or git history — memory stores what the code cannot
  tell you (role preferences, failure context, project decisions)

**Consolidation** (offline pipeline):
- Background "sleep-time" processing to merge, compress, and
  extract rules from accumulated knowledge
- Compression artifacts preserve citations to original memory
  blocks — never "rewrite history," only layer interpretations
  above immutable records
- Rule extraction artifacts (learned patterns with confidence +
  provenance) are **separately gated** — a powerful injection
  surface because a poisoned memory can promote itself into a
  global rule via consolidation
- **Consolidation safety gates**: same semantic firewall policies
  apply to consolidation outputs as to tool outputs, because
  consolidation transforms untrusted text into privileged,
  reusable directives

**Why here**: prerequisite for MCTS (needs run history), self-evolving
(needs pattern learning), and meta-agent (needs plan templates).
References: Graphiti/Zep (temporal KG, provenance, validity windows),
MAGMA (multi-graph), LightMem (sleep-time consolidation, 117×
compression), Graph-based Agent Memory survey, AgentPoison (retrieval
dominance detection), Memory Poisoning Attack (write gate design),
Unit 42 Memory Poisoning PoC (persistent compromise via benign
artifacts).

#### Codebase Graph (`codebase_graph.py`) — Shipped in v2.2.0

AST-based call graph and cross-file dependency analysis for Python
codebases.  Upgrades `context_mode: structural` from regex chunk
scoring to precise blast-radius-aware context selection.  Zero
additional dependencies — uses stdlib `ast` module (~480 lines).

**Resolution tiers** (all shipped):
- **Tier A**: direct function calls with `ast.Name` nodes in the
  same module; intra-file symbol resolution; explicit imports
- **Tier B**: cross-file `from .X import Y` aliasing; module
  imports; `self.method()` calls within class context
- **Tier C**: `getattr`-based calls, dynamic dispatch — flagged
  as `uncertain: true` edges (do not participate in blast-radius
  scoring)

**Shipped implementation**:
- `_ASTVisitor(ast.NodeVisitor)`: extracts `FunctionDef`,
  `ImportRef`, `CallSite` dataclasses from `.py` files
- Call graph: `dict[str, set[str]]` with precomputed reverse edges
- `blast_radius()`: BFS over call + reverse edges for impact sets
- `find_clusters()`: Tarjan's SCC algorithm (O(V+E))
- `build_ast_structural_context()`: drop-in replacement that
  dispatches to AST graph for Python, regex fallback for others
- Cache: JSON in `.maestro-cache/codebase_graph/{snapshot_id}.json`
  with `schema_version`, `parser_version`, `python_version` fields;
  invalidated by file stat changes (SHA-256 of path+mtime tuples)
- Integration: `_build_structural_context()` in `runners.py` gains
  `workspace_root` param; `scheduler.py` passes it through (1-line)
- `symbols.py` unchanged — serves as regex fallback for non-Python

**Shipped in v2.3.0**:
- Package `__init__` modules now normalize to the package name
  instead of `pkg.__init__`, which keeps graph nodes aligned with
  actual import semantics
- Tier B resolution now follows package re-exports so
  `from pkg import symbol` and `pkg.symbol()` can resolve to the
  underlying implementation in package graphs
- Relative import file dependencies are normalized to absolute
  module names before being stored in the graph cache
- PageRank-based centrality scoring now ships in the graph cache and
  feeds structural-context chunk ranking, so hot utility nodes and
  shared call hubs receive higher review priority

**Remaining** (future improvements):
- Tree-sitter for non-Python languages (see Across All Phases)

References: code-review-graph (DiGraph cache, PageRank risk
scoring, 8.2× fewer tokens — MIT, Python, primary reference),
Codemem (SCIP confidence fusion, 9-component scoring), codegraph
(dataflow + CFG analysis, CI gates), Joern (CPG).

#### Context Pipeline v2 — Partially Shipped

Upgrade context compression and summarization for the increased
state carried by persistent memory and cross-run knowledge.

**Structured compact template** — **Shipped in v2.2.0**:
- 9-section structured template in `build_summarization_prompt()`:
  (1) Primary Request, (2) Key Technical Concepts, (3) Files and
  Code, (4) Errors and Fixes, (5) Problem Solving, (6) Outputs,
  (7) Pending Issues, (8) Current State, (9) Next Steps
- **Scratchpad-then-strip**: `<analysis>` block where the LLM
  reasons before producing the summary; `_strip_analysis_block()`
  strips it from the final output — zero extra downstream tokens
- Reduce prompt (`build_reduce_prompt()`) also uses scratchpad +
  structured sections (Progress, Cross-cutting, Errors, Outputs,
  Verdict)
- Prompt now includes warnings and decisions from StructuredContext

**Circuit breaker** — **Shipped in v2.2.0**:
- `_summarization_consecutive_failures` counter in `runners.py`
- After 3 consecutive LLM failures, falls back to mechanical L1
  extraction (`_extract_l1_sections()`) — zero LLM cost fallback
- Counter resets to 0 on any successful summarization call

**Post-compact restoration** — **Shipped in v2.2.0**:
- After progressive compaction stage >= 3 (aggressive truncation),
  re-inject top-5 scored upstreams at L1 detail from original text
  if budget allows (`_POST_COMPACT_RESTORE_MAX = 5`,
  `_POST_COMPACT_RESTORE_MAX_CHARS = 20000`)
- `_apply_progressive_compaction()` accepts `original_texts` param;
  scheduler passes pre-compaction copy to enable restoration

**LLM tier in compaction** — **Shipped in v2.3.0**:
- Stage 2.5 in `_apply_progressive_compaction()` uses
  `_run_summarization()` (9-section structured template +
  scratchpad-then-strip) between section pruning and lossy
  truncation; respects circuit breaker; `workdir` param added

**Remaining** (not yet shipped):
- **Session memory extraction**: top post-v2.4.0 local-first priority.
  For watch loops and long-horizon tasks, extract a state snapshot and
  truncate history while keeping the N most recent iteration outputs
  verbatim. `mempalace` is the main design reference for transcript
  ingest, layered wake-up memory, and diary-style persistence;
  `claude-obsidian` reinforces the hot-cache/session-bootstrap side.
  Maestro should keep all of this inside Memory v2's
  SQLite/provenance/trust pipeline. After the watch path lands, reuse
  the same snapshot shape for `maestro chat` bootstrap rather than
  adding a second memory subsystem.

#### Semantic Firewalls

LLM-based input/output validation beyond static regex patterns.
Complements v1.x's `_strip_injection_patterns()` and audit rules.
Defined around a concrete adversary model.

**Adversary model** (what we defend against):
1. **Indirect prompt injection** via upstream output — partially
   covered by v1.x (4 regex patterns + CFI + taint propagation)
2. **Tool-selection hijacking** via MCP tool descriptions and
   plugin metadata — **new gap**: `mcp_servers` currently treated
   as trusted; tool docs must be treated as untrusted inputs
3. **Adaptive attacks** against static defenses — **new gap**:
   current pattern list is hardcoded and not learnable

**Two-pass validation**:
- Pass 1 (fast, deterministic): known injection patterns,
  suspicious tool-call syntaxes, disallowed URL schemes, oversized
  payloads — extends current `_INJECTION_PATTERNS` in runners.py
- Pass 2 (model-based, opt-in): structured classifier prompt via
  lightweight model (haiku) outputting machine-checkable verdict
  (allow/rewrite/block + rationale category); logged as auditable
  span event; enabled via `firewall_model: haiku` in plan config

**Adaptive defense with guardrails**:
- System can learn new **detectors and test cases** from observed
  injection patterns (honeypot triggers, failed attacks)
- **Invariant**: learned patterns can extend detectors and test
  suites but CANNOT modify allowlists, write policies, or trust
  labels without explicit operator approval — prevents memory
  poisoning from steering defense evolution
- Closed-loop: honeypot trigger → new signature → regression test

**Wildcard tool permission patterns** — **Shipped in v2.2.0**:
- `allowed_tools` supports glob patterns: `Bash(git *)`,
  `Edit(src/*)`, `Read(*)` — backward-compatible: bare names
  still work as exact matches
- `parse_tool_pattern()`: parses `ToolName(pattern)` syntax
- `_split_tool_permissions()`: separates fully allowed tools
  from argument-restricted tools
- Claude: `--disallowedTools` blocks entire tools; argument
  restrictions injected as prompt text (hybrid approach)
- Other engines: full tool + argument restrictions in prompt
- `TOOL_CATEGORIES` expanded with `git-only` and `src-scoped`
- Loader validates base tool name from patterns (no false W27)

**Role-based filtering** — **Shipped in v2.3.0**:
- `mcp_servers[].allowed_task_roles` constrains which `task.agent`
  roles may reference a server via `mcp_tools`
- Enforced in both plan validation and runtime MCP config generation

**MCP tool description validation** — **Shipped in v2.3.0**:
- `mcp_servers[].description` is treated as untrusted metadata
- Deterministic pass-1 sanitization strips prompt-injection markers,
  suspicious tool-call syntax, dangerous URL schemes, and secret
  exfiltration language before metadata is re-injected into prompts
- Prompt adds an explicit trust-boundary reminder: tool metadata,
  schemas, and outputs are instructions-shaped data, not authority

**Two-pass validation** — **Shipped in v2.3.0**:
- Optional top-level `firewall_model` enables a lightweight pass-2
  classifier for MCP descriptions and tainted upstream text
  (`stdout_tail`, structured `result_text`, structured `summary`)
- Verdicts are machine-checkable (`allow` / `rewrite` / `block`);
  classifier failures fail open to deterministic pass-1 sanitization

**Concurrency safety declarations** — **Shipped in v2.3.0**:
- `mcp_servers[].is_concurrency_safe` lets plan authors declare
  whether a server is safe to share across parallel `worktree: true`
  tasks
- Scheduler serialises worktree tasks that use MCP servers explicitly
  marked `is_concurrency_safe: false`, reducing the risk of shared
  tool side effects escaping task-local worktrees

**Remaining** (future refinement):
- Per-tool concurrency metadata beyond the current server-level
  declaration, for MCP servers that expose a mix of read-only and
  side-effectful tools

**Scope extensions**:
- Expand pass-2 coverage beyond current tainted text fields if needed
- Workflow-level fuzzing integration (ChainFuzzer-style) — v2.4+

**Test strategy** (minimum viable for v2.2):
- Indirect prompt injection regression set (static corpus based on
  current `_INJECTION_PATTERNS` + honeypot patterns) — **Shipped in v2.3.0**
- Adaptive attacker suite and workflow fuzzing deferred to v2.4+

References: Adaptive Attacks Break IPI (NAACL 2025), ToolHijacker
(tool-selection via injected docs), ChainFuzzer (workflow-level
multi-tool vulnerabilities), VIGIL + SIREN (verify-before-commit),
WASP (agents follow malicious instructions even without completing
them), Prompt Injection on Agentic Coding Assistants (trust boundary
failures).

#### Observability v2

Enhanced monitoring for the increased complexity of remote execution,
persistent memory, and semantic firewalls. Observability as an **API
contract**, not a debugging afterthought.

**Trace model** (public contract):
- Core span hierarchy: Agent Run → Plan → Step →
  Tool/Retrieval/Memory/Guardrail
- Span kinds aligned with OpenInference: Agent, Tool, Retriever,
  Guardrail, Evaluator (replaces current flat `INTERNAL` kind)
- GenAI semantic conventions: `gen_ai.system`, `gen_ai.model.id`,
  `gen_ai.usage.completion_tokens`, `gen_ai.usage.prompt_tokens`
  on LLM-calling spans

**Memory instrumentation**:
- Every memory write attempt is a span event with outcome
  (accepted/rejected/quarantined), trust score, and provenance
  pointer — essential for debugging poisoning defenses
- Quarantine events surfaced in TUI and OTLP export

**Privacy controls**:
- `--otel-mask-prompts` flag: configurable masking of prompt and
  output content in OTLP spans (extends existing `--mask-secrets`)
- Prompt/response capture is valuable for debugging but must be
  gatable for production environments with PII sensitivity

**Shipped in v2.3.0**:
- `maestro export-otel` emits `gen_ai.system`, `gen_ai.model.id`,
  `gen_ai.usage.prompt_tokens`, and
  `gen_ai.usage.completion_tokens` for engine task spans
- Optional task input/output capture via `--include-content`
  with privacy-preserving redaction via `--otel-mask-prompts`
- Post-run knowledge persistence emits `memory_write` task events with
  accepted / rejected / quarantined outcomes, instructionality scores,
  and provenance pointers for downstream tracing
- `knowledge_poison_alert` events are exported as task span events,
  making quarantine decisions visible in downstream tracing

**TUI enhancements**:
- Watch Mode: MetricPanel with sparkline charts for step latency,
  cache hit rate, memory write acceptance rate, quarantine events,
  and tool error rate — each derivable from trace attributes
- Multi-plan TUI: per-plan tabs or split view

**Workflow authoring UX**:
- `--set key=value` for plan overrides (template variable injection
  from CLI without editing YAML)

References: OpenTelemetry GenAI semantic conventions, OpenInference
spec (span kinds + payload attributes), Arize Phoenix (OTel-to-
OpenInference translation), OTLP stable specification.

#### Definition of Done — Phase 2

Each module ships with measurable acceptance criteria and explicit
failure modes. Phase 2 is complete when all criteria are met.

| Module | Done When |
|--------|-----------|
| **Persistent memory** | ~~WAL mode~~ DONE; ~~writes serialized~~ DONE; ~~every write transactional~~ DONE; ~~point-in-time query~~ DONE; ~~index-and-detail BM25 selective loading~~ DONE |
| **Temporal KG** | ~~valid_from/valid_to/recorded_at columns~~ DONE; ~~immutable-with-windows (invalidate_record)~~ DONE; ~~provenance fields (source_type, source_id)~~ DONE; ~~conflict resolution~~ DONE; ~~entity confidence scores~~ DONE |
| **Write policies** | ~~instructionality scoring (5 regex patterns)~~ DONE; ~~auto-quarantine on high score~~ DONE; ~~quarantined excluded from normal load~~ DONE; ~~retrieval dominance tracking~~ DONE; ~~poisoning harness~~ DONE |
| **Semantic cache** | ~~policy-versioned cache keys~~ DONE; ~~short-TTL negative cache (`negative_cache_ttl_sec`)~~ DONE; ~~untrusted / tainted / partial outputs excluded~~ DONE; ~~explicit tool-failure exclusion~~ DONE in v2.3.0; ~~pre-hash normalization~~ DONE in v2.3.0; ~~eviction `why` fields~~ DONE in v2.3.0; remaining: embedding-assisted semantic lookup |
| **Consolidation** | ~~Rule extraction artifacts carry provenance tracebacks~~ DONE in v2.3.0 (`ConsolidatedLesson.source_trust_labels` + `avg_instructionality`); ~~consolidation outputs pass same firewall policies as tool outputs~~ DONE in v2.3.0 (pass-1 + pass-2 + instructionality check in `_run_consolidation()`) |
| **Context pipeline v2** | ~~9-section structured template~~ DONE; ~~circuit breaker (3 failures → L1 fallback)~~ DONE; ~~post-compact restoration (top-5 by score)~~ DONE; ~~LLM tier in compaction pipeline~~ DONE in v2.3.0 (Stage 2.5 in `_apply_progressive_compaction()`); remaining: session memory extraction for watch loops (top post-v2.4.0 item; deterministic extraction/injection + resume now landed on main (unreleased) on top of `WatchIteration` excerpts + SQLite `session_snapshots`; remaining follow-up: LLM-polished consolidation), then reuse the same snapshot shape for `maestro chat` hot-cache/bootstrap |
| **Codebase graph** | ~~AST visitor with Tier A/B/C resolution~~ DONE; ~~blast radius BFS + Tarjan SCC~~ DONE; ~~cache with version fields~~ DONE; ~~integration via workspace_root~~ DONE; ~~`__init__` re-export resolution~~ DONE in v2.3.0; ~~PageRank scoring~~ DONE in v2.3.0 |
| **Semantic firewalls** | ~~wildcard tool patterns backward-compatible~~ DONE; ~~new categories (git-only, src-scoped)~~ DONE; ~~IPI regression tests~~ DONE in v2.3.0; ~~MCP tool description validation~~ DONE in v2.3.0; ~~role-based filtering~~ DONE in v2.3.0; ~~two-pass validation (`firewall_model`)~~ DONE in v2.3.0; ~~server-level MCP concurrency declarations (`is_concurrency_safe`)~~ DONE in v2.3.0; future refinement: per-tool concurrency metadata |
| **Observability v2** | OTLP export stable; ~~GenAI semantic attributes on LLM spans~~ DONE in v2.3.0; ~~prompt/response capture configurable with masking~~ DONE in v2.3.0; ~~memory write spans emitted~~ DONE in v2.3.0 |

---

### Phase 3 — Search (v2.3.0 foundations, completed in v2.4.0)

> Automatic workflow optimisation.

#### MCTS Workflow Search

Tree search over DAG topologies for plan evolution. Treats the
space of possible plans (task ordering, model selection, context
modes, dependency structure) as a search tree.

**Foundations shipped in v2.3.0**:
- `WorkflowVariant` dataclass now exists in `models.py`
- `mcts.py` provides draft/debug/improve classification and MVP
  `debug_prob` selection over leaf variants plus optional UCB1 with
  configurable exploration constant
- `simulate_variant()` executes candidates through `run_plan()`
- `apply_historical_pruning()` consumes the existing Phase 2
  historical-pruning API
- `backpropagate_variant()` blends current and historical
  `ScoreRecord` values with time decay before updating lineage
- `append_tree_node()` / `load_tree_index()` persist tree state in
  `tree.jsonl`

**`WorkflowVariant` tree node**:
- Dataclass: `plan_spec` (PlanSpec), `run_result` (PlanRunResult
  | None), `score` (float), `is_valid` (bool), `parent`
  (WorkflowVariant | None), `children` (list[WorkflowVariant]),
  `variant_type` (draft | debug | improve), `mutation_desc` (str)
- Each node holds the full plan + execution result + lineage
- Tree persisted in `tree.jsonl` (one JSON line per node with
  parent_id reference) alongside `events.jsonl`

**Expansion trichotomy** (from AI Scientist v2):
- **draft**: scaffold a new plan topology from scratch (no parent)
- **debug**: fix a failing variant — keep topology, correct errors
  (maps to existing `maestro replan` with error feedback)
- **improve**: enhance a working variant — mutate model, context
  mode, ordering, or dependency structure (keep what works)
- Classification: `parent is None` = draft, `parent.is_valid is
  False` = debug, else = improve

**Selection strategy** (two-phase):
- MVP: `debug_prob` parameter (default 0.5) — with probability P,
  pick a random invalid leaf for debugging; otherwise pick the
  best-scoring valid node for improvement.  Simple, effective,
  avoids local optima without full UCB implementation
- Foundations shipped in v2.3.0: optional UCB1 with configurable exploration
  constant `C` — `score + C * sqrt(ln(parent.visits) / visits)` —
  is now available via `maestro replan --selection-policy ucb1
  --exploration-constant C`; `debug_prob` remains the default
  compatibility policy until more visit-depth data is collected

- Simulation: execute candidate plan, measure cost/quality/time
- Backpropagation: update node scores based on execution results
- Integration with Knowledge v2 for cross-run score accumulation

**Knowledge v2 interface** (Phase 2 → Phase 3 bridge):
- `ScoreRecord`: dataclass with `plan_hash`, `cost_usd`,
  `quality_score` (from judge), `duration_sec`, `timestamp`;
  stored in temporal KG with validity windows; MCTS
  backpropagation writes ScoreRecords after each simulation
  — core storage + scheduler persistence **Shipped in v2.2.0**
- **Simulation cache**: completed plan executions cached by
  normalized plan hash (pre-hash normalization from semantic
  cache subsystem); plans with same DAG topology but different
  models share cached simulations with confidence discount
  (avoids re-executing known-good topologies) — **implemented on main**
  for successful multi-variant `replan` simulations using
  model-family normalization + policy-versioned invalidation
- **Historical pruning**: MCTS expansion consults Knowledge v2
  to skip branches with >80% historical failure rate across
  recent runs (configurable threshold); saves simulation cost
  by avoiding repeatedly failed configurations — pruning
  decision API **Shipped in v2.2.0**, MCTS consumer **shipped in v2.3.0**
- **Temporal scoring**: ScoreRecords carry validity windows —
  stale scores (older than configurable horizon) receive
  decay discount during backpropagation, preventing outdated
  performance data from dominating search decisions

Based on AFlow. Additional references: ToolTree (MCTS for tool
decisions), MermaidFlow (verifiable IR), Workflow Optimization survey,
AI Scientist v2 (WorkflowVariant node dataclass, expansion trichotomy,
debug_prob selection, BFTS as MVP before full UCB1).

#### Self-Evolving Workflow Search (replan v2)

Evolve `maestro replan` from single-shot correction to iterative
workflow optimisation. Generate N variant plans, evaluate each,
keep the best.

**Foundations shipped in v2.3.0**:
- `maestro replan` supports multi-variant search rounds via
  `--variants N`
- `--debug-prob` exposes the MVP leaf-selection policy from the MCTS
  roadmap
- `--selection-policy ucb1` + `--exploration-constant C` expose the
  roadmap's optional UCB1 leaf-selection path without changing the
  default MVP behaviour
- search rounds persist `tree.jsonl` and record the selected
  candidate variant ID in replan state
- candidate plans are executed once during search and the next round
  continues from the selected leaf rather than re-running the same
  corrected plan blindly

- Mutation operators classified by the MCTS trichotomy:
  - **debug** mutations: error feedback injection, timeout increase,
    verify_command fix, prompt clarification (preserve topology)
  - **improve** mutations: model swap, context mode change, task
    reordering, dependency restructuring, reasoning effort tuning
  - **draft** mutations: task insertion, task deletion, group
    extraction, topology restructuring (new topology)
- Fitness function: weighted score of cost, quality, duration +
  **novelty bonus** from KG distance to previously tried variants
  (prevents convergence to local optima)
  - Foundations shipped in v2.3.0: `replan` now computes an initial
    concept-signature novelty bonus from diff-like plan mutations
    against the current baseline and persisted `tree.jsonl` history;
    the resulting `novelty_prior` is blended into candidate scores as
    a bounded tie-breaker rather than a replacement for observed
    simulation quality
- Integration: watch (loop) + replan (mutation) + diff (comparison)
  + MCTS (search) + knowledge (history)
- Population management: tournament selection, elitism, diversity
  preservation
  - Foundations shipped in v2.3.0: tournament selection now exists as an
    optional population strategy for multi-variant `replan`; elitism
    is now partially implemented via `--elite-count N`, which keeps
    top-scoring leaves in the selection pool during tournament rounds;
    explicit diversity preservation now also exists via
    `--diversity-floor F`, which filters tournament challengers by
    minimum mutation-signature distance before falling back to generic
    sampling when the pool gets too homogeneous

**Safety contract** (mutations as untrusted inputs):
- Generated mutations pass through semantic firewall (Phase 2)
  before execution — prevents adversarial mutation injection
  where a compromised knowledge base steers evolution towards
  harmful plan configurations
- Foundations shipped in v2.3.0: `replan` now treats generated variants as
  tainted inputs, blocks candidates that introduce new blocking SEC
  findings relative to the current trusted plan, and applies the
  trusted plan's optional `firewall_model` pass-2 to generated YAML
  before simulation
- All machine-generated plan variants are treated as `tainted:
  true` until validated by firewall + judge evaluation
- Population diversity tracked via temporal KG point-in-time
  queries ("have we tried this variant before?") — deduplicate
  against historical ScoreRecords to avoid wasted simulations
  - Foundations shipped in v2.3.0: multi-variant `replan` now deduplicates
    exact `plan_hash` repeats against both the current `tree.jsonl`
    search history and prior `ScoreRecord` history; search-tree
    duplicates are pruned immediately, while historical duplicates
    reuse stored score evidence for ranking without re-simulating in
    the same round

**Knowledge v2 integration**:
- Mutation guidance: Knowledge v2 `failure_pattern` and
  `model_pattern` records inform which mutations are likely
  productive (e.g., "sonnet fails on security tasks" → prefer
  opus mutation for security-tagged tasks)
  - Foundations shipped in v2.3.0: `replan` now retrieves relevant
    `failure_pattern` / `model_pattern` records for failed tasks and
    injects them into the analysis prompt as advisory historical
    hints, after the same retrieval-dominance filtering used by the
    runtime knowledge path
  - Foundations shipped in v2.3.0: search candidates also receive a small
    `knowledge_prior` score bonus when their mutations introduce
    tokens aligned with the retrieved historical hints, so tied
    simulations prefer variants that better match known productive
    fixes
- Cross-run fitness: fitness function consults historical
  ScoreRecords for similar plan topologies, bootstrapping
  evaluation for new variants without full simulation
  - Foundations shipped in v2.3.0: completed `replan` variants now persist a
    compact topology signature in `ScoreRecord.metadata`, and
    multi-variant search derives a bounded
    `historical_fitness_prior` from similar historical signatures to
    bias tied simulations toward variants whose topology matches
    previously successful runs
- Stepping stone continuity: successful mutations persisted as
  stepping stones (existing `watch.stepping_stones` mechanism)
  with KG provenance linking mutation → fitness improvement
  - Foundations shipped in v2.3.0: successful multi-variant `replan`
    completions now save `replan_fitness` stepping stones into the
    shared stepping-stone archive with mutation provenance
    (`selected_node_id`, parent hash, bonuses, fitness gain), so the
    archive now records both watch-driven and replan-driven
    improvements without mixing compaction across different metrics

References: HyEvo, EvoAgentX, Darwin Godel Machine, AI Scientist v2
(best-first tree search, draft/debug/improve trichotomy, debug_prob
exploration, stage-based progression with LLM-generated sub-stages).

#### Cross-Phase Security Contract

Phase 3 features generate and execute plan variants automatically.
All Phase 2 security gates apply transitively:

- **Write policy inheritance**: ScoreRecords and mutation-derived
  knowledge entries pass through the same staged write validation
  (syntactic → security) as any other memory write
- **Firewall coverage**: machine-generated plans are validated by
  the semantic firewall before execution — both deterministic
  Pass 1 (injection patterns, destructive commands) and optional
  model-based Pass 2
- **Consolidation safety**: rule extraction from MCTS/evolution
  history passes through consolidation safety gates — prevents
  a high-scoring but compromised variant from promoting itself
  into a persistent "learned best practice"
- **Audit trail**: every mutation, simulation, and selection
  decision is an event in `events.jsonl` with hash-chain
  integrity (existing eventsource.py mechanism)
  - Foundations shipped in v2.3.0: multi-variant `replan` now appends
    `replan_search_*` / `replan_candidate_*` events to the root
    search run's `events.jsonl`, continuing the existing hash chain
    when the initial run already emitted scheduler events
  - Foundations shipped in v2.3.0: explicit `replan_candidate_deduplicated`
    events now record when a generated mutation is skipped because
    its `plan_hash` was already seen in the active search tree or in
    historical score history

#### Definition of Done — Phase 3

| Module | Done When | Status |
|--------|-----------|--------|
| **MCTS search** | Candidate plans execute and backpropagate scores; historical pruning demonstrably reduces simulation count vs. exhaustive search; ScoreRecords stored in temporal KG with validity windows | **Shipped in v2.4.0** (`maestro-benchmark --case replan_pruning` reports saved candidate simulations under historical pruning) |
| **Simulation cache** | Plans with identical normalized topology share cached results; cache invalidation triggers on policy version or model family change | **Shipped in v2.4.0** (successful `replan` simulation reuse with confidence discount, model-family normalization, policy-versioned invalidation) |
| **Self-evolving replan** | N>1 variant population with tournament selection converges to measurably better fitness (cost×quality×duration) than single-shot replan; novelty bonus prevents premature convergence | **Shipped in v2.4.0** (`maestro-benchmark --case replan_population` and `--case replan_novelty` demonstrate selected-fitness uplift and novelty-driven selection changes) |
| **Safety contract** | Generated mutations pass firewall validation; tainted variants blocked before execution; poisoned knowledge base cannot steer evolution towards harmful configurations (red-team test) | **Shipped in v2.4.0** (integration red-team coverage proves poisoned replan guidance can generate a harmful variant, which is then blocked before simulation while search selects a safe alternative) |
| **Knowledge bridge** | Phase 2 `failure_pattern` and `model_pattern` records demonstrably guide mutation selection; cross-run fitness bootstrapping reduces simulation cost | **Shipped in v2.4.0** (`maestro-benchmark --case replan_guidance` demonstrates positive knowledge/history bonuses and lower simulation count when the bridge is enabled) |

---

### Phase 4 — Interop (Deferred post-v2.4.0)

> From isolated tool to networked agent node.

#### A2A Protocol Layer

Agent-to-agent interop for cross-system communication. Shift from
local shell-out to networked agent communication.

- Agent cards: capability advertisement and discovery
- Task delegation: remote plan execution via A2A messages
- Streaming: real-time progress via SSE (builds on AG-UI adapter)
- Trust: capability negotiation + `allowed_tools` enforcement
- Protocol validation: ProtocolBench-informed testing

Based on A2A Protocol. Additional references: ProtocolBench, ACP.

#### Meta-Agent Plan Generation

LLM that generates Maestro YAML plans from natural language
objectives, iterating on plan design based on execution feedback.

- Natural language → YAML plan generation
- Iterative refinement: execute plan, analyse results, regenerate
- Template library: learn from Knowledge v2 accumulated patterns
- Constraint satisfaction: budget limits, model preferences, security
  requirements
- Integration with MCTS for plan space exploration

**Coordinator-as-prompt pattern**:
- The meta-agent is NOT a separate orchestration subsystem — it is
  a regular Maestro task with a coordinator-aware system prompt and
  access to plan generation tools (scaffold, validate, MCTS)
- Workers (sub-plans) report back via structured `events.jsonl`
  entries, not a custom IPC protocol — the coordinator reads
  `run_manifest.json` results as its "worker notifications"
- Worker context isolation is deliberate: the coordinator
  synthesizes findings into self-contained prompts for each
  generated plan; workers never inherit the coordinator's full
  conversation context
- This pattern avoids building a separate "meta-orchestrator" —
  the DAG scheduler IS the orchestrator; the meta-agent is just
  another task that happens to generate and execute plans

The ultimate abstraction: describe what you want, Maestro figures
out how.

References: A2A Protocol.

---

### Across All Phases (incremental)

These features are independent and can ship alongside any phase.

| Feature | Earliest Phase | Notes |
|---------|---------------|-------|
| Chat Extensions (`/run`, auto-routing, TUI chat) | v2.1+ | Incremental additions to existing `chat.py` |
| FSA-based State Recovery | v2.1+ | Pairs with remote execution for reconnection |
| SOP Repository | v2.2+ | Pairs with Knowledge v2 (persist success patterns) |
| Spec-Driven Scaffold (`--from-spec RFC.md`) | v2.2+ | LLM-based, standalone |
| Runtime Watcher Agent | v2.1+ | Pairs with Observability v2 |
| Environment Drift Benchmark | v2.3+ | Pairs with MCTS (detect drift between runs) |
| Visual DAG Editor | v2.4+ | Frontend-heavy, lowest technical priority |
| Benchmark Harness Improvements | v2.0+ | Can ship anytime |
| Tree-sitter for `symbols.py` | v2.2+ | Optional `[ast]` extra with py-tree-sitter; regex fallback stays; inspired by GitNexus 14-language pipeline |
| ~~RRF Fusion for Intent Filtering~~ | ~~v2.2+~~ | **Shipped in v2.3.0** — `_rrf_score()` in scheduler.py; combines BM25 + hop-distance via RRF (k=60); integrated into `_apply_context_budget()` |
| ~~Confidence-weighted Knowledge Graph~~ | ~~v2.2+~~ | **Shipped in v2.2.0** — confidence scores on `knowledge_graph.py` relations (CALLS > MENTIONS); integrated into context scoring |
| Community Detection for `suggest.py` | v2.2+ | Detect task clusters sharing upstreams/context_from; suggest `group:` sub-plans; connected-components (not full Leiden) |
| MCP Tool Ecosystem Recipes | v2.1+ | PLAYBOOK recipes for external MCP tools (GitNexus, etc.) as opt-in deep analysis layer |

---

## Shipped Features (summary)

130+ features shipped across v1.0.0 -- v2.4.0. v1.37.1 is the
validation sprint release (mypy 100%, +622 tests, 5 bugs fixed).
v2.0.0 adds the Python SDK and capability-based tool access.
v2.1.0 adds council topology extensions and the llama engine.
v2.2.0 adds Context Pipeline v2, Codebase Graph, wildcard tool
permissions, and Knowledge + Memory v2 (partial): SQLite+WAL persistent
memory with bi-temporal model, provenance, instructionality scoring,
and auto JSONL migration. v2.3.0 closes Phase 2 gaps (Semantic
Firewalls MCP-path + pass-2, Codebase Graph PageRank, Observability v2
GenAI attrs, consolidation safety gates, LLM compaction tier, RRF
fusion, pre-hash normalization, eviction "why" fields) and ships Phase 3
foundations: MCTS workflow search (`mcts.py`), self-evolving replan
(multi-variant search, tournament, elitism, diversity, novelty/knowledge
priors, deduplication, stepping stones), and cross-phase security
contract (generated-plan audit gate, firewall, taint blocking, search
audit trail). v2.4.0 closes the remaining Phase 3 gaps with simulation
cache reuse, deterministic search benchmarks, demonstrable
knowledge-bridge guidance, and red-team coverage for poisoned replan
guidance.

### Context Pipeline (11 modes + v2 upgrades)
- `raw` (default), `summarized`, `map_reduce`, `recursive`, `layered`,
  `selective`, `structural`, `council`, `knowledge_graph`
- BM25 intent filtering, graph-distance decay, progressive compaction,
  trajectory guardrails, privacy pipeline, taint propagation
- **v2**: 9-section structured compact, `<analysis>` scratchpad-then-strip,
  summarization circuit breaker (3 failures → L1 fallback),
  post-compact restoration (top-5 by relevance score),
  AST-based codebase graph (blast radius, Tarjan SCC, cached)

### Quality + Evaluation
- `verify_command`, `guard_command`, typed assertions, judge (4 methods:
  direct, g_eval, debate, reflection), quorum with diversity, CWE
  presets, multi-dimensional eval, rubric scoring

### Retry + Resilience
- Auto-escalation, cross-engine fallback, circuit breaker, event-driven
  reminders, population-based search, checkpoint/resume

### Watch + Self-Improvement
- Autonomous iteration loop, metric extraction, git commit/rollback,
  stepping stones, adaptive improvement prompt, consolidation agent,
  Meta-Policy Reflexion (persistent failure rules)

### Security (23 audit rules + wildcard tool patterns)
- SEC001-SEC023, CFI, honeypot decoys, output scope envelopes, dual
  verification, taint propagation, phantom workspace, privacy pipeline
- **v2**: wildcard `allowed_tools` patterns (`Bash(git *)`),
  `git-only`/`src-scoped` categories, hybrid CLI+prompt enforcement

### Protocols
- AG-UI adapter, MCP server (12 tools), MCP client orchestration, OTLP
  exporter

### Routing + Knowledge
- Difficulty-aware auto-routing, predictive routing, temporal learning,
  cross-task affinity, cross-run knowledge (8 pattern kinds including
  policy_rule), knowledge graph entities

### Extensibility
- Custom engine plugins (frozen API), CI generator (frozen), 5 workflow
  libraries, skill registry, dynamic task decomposition

### CLI (27 subcommands)
- `run`, `validate`, `check`, `ci`, `replan`, `scaffold`, `watch`,
  `chat`, `shell`, `ui`, `report`, `diff`, `explain`, `status`,
  `eval`, `suggest`, `blame`, `verify`, `audit`, `doctor`, `cleanup`,
  `backfill-costs`, `export-otel`, `mcp-server`, `ci-analyze`,
  `skill`, `budget`

### Release History

| Version | Theme | Highlights |
|---------|-------|------------|
| v2.4.0 | Phase 3 Completion | Simulation-cache reuse for replan search, deterministic `maestro-benchmark` Phase 3 cases (`replan_pruning`, `replan_population`, `replan_novelty`, `replan_guidance`), and red-team validation that poisoned replan guidance is blocked before simulation |
| v2.3.0 | Phase 3 Foundations | MCTS workflow search (`mcts.py`), self-evolving replan (multi-variant, tournament, elitism, diversity, novelty/knowledge priors, stepping stones), cross-phase security (audit gate, firewall, taint, audit trail), Phase 2 completion (firewalls, consolidation safety, LLM compaction, RRF, pre-hash normalization, eviction "why"), W29/W30 post-mortem validation |
| v2.2.0 | Phase 2 Foundations | Context Pipeline v2, Codebase Graph, wildcard tool patterns, Knowledge + Memory v2 (SQLite+WAL, bi-temporal, provenance, write validation, JSONL migration), semantic cache hardening, score history |
| v2.1.0 | Council Topologies + Llama | chain/graph topologies, llama engine, 91 new tests |
| v2.0.0 | Python SDK + Tool Access | Programmatic API, `allowed_tools:`, SEC023, E071, 119 new tests |
| v1.37.0 | Knowledge Graph + Meta-Policy | `context_mode: knowledge_graph`, Meta-Policy Reflexion, 31 new tests |
| v1.36.0 | Council Mode | `context_mode: council` (star topology), 20 new tests |
| v1.35.0 | Chat Enhancements | `/context`, `/save`, `/load`, session persistence, 18 new tests |
| v1.34.0 | Structural Context | `context_mode: structural` (10 languages), 54 new tests |
| v1.33.0 | Stepping Stones | Watch archive, contracts docs polish, 21 new tests |
| v1.32.0 | Judge Intelligence | Quorum diversity, Meta-Watch prompt, 24 new tests |
| v1.31.x | Stability | Quorum reliability, freeze promotions, docs |
| v1.30.0 | Selective Context | `context_mode: selective`, consistency groups |
| v1.29.0 | MCP Tools | `mcp_servers` + `mcp_tools` for engine tasks |
| v1.28.0 | Adaptive Routing | Temporal routing, population search |
| v1.27.0 | Skill Registry | `maestro skill`, CI agent, knowledge consolidation |
| v1.26.0 | Context Safety | Progressive compaction, privacy pipeline, phantom workspace |
| v1.25.0 | Production Hardening | Workflow libraries, CWE presets, dual verification, playbook |
| v1.24.0 | Quick Wins | OTLP, reminders, compression signals, +1774 tests |
| v1.23.0 | MCP Server | 12 tools, 8 resources, 3 prompts |
| v1.22.0 | AG-UI Protocol | Event translation, SSE endpoint |
| v1.21.0 | Untrusted Context | Taint propagation, injection stripping |
| v1.20.0 | Merge Intelligence | LLM conflict review for worktrees |
| v1.19.0 | Mid-task Signals | `[MAESTRO_SIGNAL]` protocol, 7 types |
| v1.18.0 | Dynamic Decomposition | `dynamic_group: true` |
| v1.17.0 | Cross-run Knowledge | `{{ task_knowledge }}` auto-inject |
| v1.16.0 | Predictive Routing | `model: auto` learns from history |
| v1.15.0 | Structured Outputs | `output_schema`, `{{ task-id.output.FIELD }}` |
| v1.14.0 | Research Features | Plan density, deliberation gate, debate judge |
| v1.13.0 | Audit + Context | `context_model`, contract types, `--coverage` |
| v1.12.0 | Batch + Budget | Batch mode, frozen tasks, ai_slop, knowledge archive |
| v1.11.x | Self-Improvement | `mode: improve`, structured summaries, integrity |
| v1.10.0 | Context Intelligence | Layered context, CFI, quorum, trajectory |
| v1.9.0 | Smart Prevention | Policy engine, routing, blame |
| v1.8.0 | Validation | `json-schema`, STPA, failure taxonomy |
| v1.7.0 | Chat | `maestro chat`, 6 engines, slash commands |
| v1.6.0 | Event Sourcing | Hash chain, `maestro verify`, circuit breaker |
| v1.5.x | Worktree + Routing | Git isolation, `model: auto` |
| v1.4.0 | Watch Loop | `maestro watch`, experiments.jsonl |
| v1.3.0 | TUI + Resilience | TUI panels, escalation, fallback |
| v1.1-2.x | Foundation | Live output, TUI base, events |

---

## Research References

Features in this roadmap were informed by these projects and papers.

### Context + Retrieval + Memory
- **[OpenViking](https://github.com/volcengine/OpenViking)** -- L0/L1/L2 tiered context loading
- **[FlashRAG](https://github.com/RUC-NLPIR/FlashRAG)** -- selective context refiners, IRCoT reasoning-based RAG
- **[ContextBench](https://arxiv.org/abs/2602.05892)** -- validates BM25 over embedding-based retrieval for coding agents
- **[MemoryOS](https://github.com/BAI-LAB/MemoryOS)** -- hierarchical memory OS, 48% F1 improvement (EMNLP 2025 Oral)
- **[ReMe](https://github.com/agentscope-ai/ReMe)** -- "memory as files" paradigm, cross-session recall
- **[SimpleMem](https://github.com/aiming-lab/SimpleMem)** -- lossless semantic compression for lifelong memory
- **[MemOS](https://github.com/MemTensor/MemOS)** -- unified memory API, memory cubes
- **[AgeMem](https://arxiv.org/abs/2601.01885)** -- unified STM+LTM as learnable policy
- **[A-MEM](https://github.com/agiresearch/A-mem)** -- agent-organised memory
- **[MAGMA](https://arxiv.org/abs/2601.03236)** -- multi-graph memory (semantic, temporal, causal, entity)
- **[SwiftMem](https://arxiv.org/abs/2601.08160)** -- query-aware indexing for low-latency retrieval
- **[HippoRAG 2](https://arxiv.org/abs/2502.14802)** -- associative multi-hop retrieval via Personalized PageRank; informed `context_mode: knowledge_graph`
- **[LightMem](https://arxiv.org/abs/2510.18866)** -- three-stage memory with "sleep-time" consolidation, 117x token reduction
- **[Reflexion](https://arxiv.org/abs/2303.11366)** -- verbal reinforcement learning via self-reflection, 91% pass@1 on HumanEval; informed `judge.method: reflection` and Meta-Policy Reflexion
- **[MemoRAG](https://arxiv.org/abs/2409.05591)** -- global memory + clue-guided retrieval; draft-retrieve-verify pipeline; informed knowledge_graph context briefing
- **[Temporal Semantic Memory](https://arxiv.org/abs/2601.07468)** -- temporal validity for facts, timeline reasoning; informs v2 temporal knowledge graph
- **[EvolMem](https://arxiv.org/abs/2601.03543)** -- cognitive-driven multi-session memory benchmark; declarative + non-declarative memory evaluation
- **[MemGPT/Letta](https://arxiv.org/abs/2310.08560)** -- hierarchical memory with OS-style paging; foundational for stateful agents
- **[MIRIX](https://arxiv.org/abs/2507.07957)** -- multi-agent memory with 6 types; multimodal; reference for v2 structured memory blocks
- **[Meta-Policy Reflexion](https://arxiv.org/abs/2509.03990)** -- reflections as reusable policy rules; informed `policy_rule` KnowledgeKind
- **[Graphiti](https://github.com/getzep/graphiti)** -- temporal context graph engine with governed retrieval; primary reference for v2 temporal KG
- **[Zep](https://github.com/getzep/zep)** ([paper](https://arxiv.org/abs/2501.13956)) -- temporal KG architecture for agent memory; validity windows, retrieval assembly
- **[Graph-based Agent Memory survey](https://arxiv.org/abs/2602.05665)** -- 2026 taxonomy of graph-based memory for agents; entity drift, routing over memory graphs
- **[MemPalace](https://github.com/milla-jovovich/mempalace)** -- local-first transcript/project memory with exchange-pair chunking, layered wake-up recall (`L0/L1/L2/L3`), agent diaries, and temporal KG; strongest external reference so far for deferred `session memory extraction`, but not a replacement for Maestro Memory v2
- **[claude-obsidian](https://github.com/AgriciDaniel/claude-obsidian)** -- persistent hot-cache plus session bootstrap around a compounding local knowledge base; useful reference for `maestro chat` bootstrap and recent-context-first retrieval, not for Obsidian-specific integration
- **[LLMLingua-2](https://arxiv.org/abs/2403.12968)** -- prompt compression via token classification; learned compressor beyond heuristic summarization
- **[Influence Guided Context Selection](https://neurips.cc/virtual/2025/poster/115474)** -- NeurIPS 2025; context filtering to mitigate "lost-in-the-middle"

### Security + Safety
- **CaMeLs** -- control flow integrity
- **AgentDojo** -- prompt injection in agent tools
- **[CIBER](https://arxiv.org/abs/2602.19547)** -- dual verification, memory poisoning, zero-trust taint
- **[Security Considerations for MAS](https://arxiv.org/abs/2603.09002)** -- 193 risks across 9 categories
- **[TrinityGuard](https://arxiv.org/abs/2603.15408)** -- unified safeguarding, 20 risk types
- **[AgentDoG](https://github.com/AI45Lab/AgentDoG)** -- trajectory-level diagnostic guardrail
- **[MINJA](https://arxiv.org/abs/2503.03704)** -- practical memory injection via query-only interaction; critical for `--resume` and watch loops
- **[BIPIA](https://arxiv.org/abs/2312.14197)** -- indirect prompt injection benchmark; baseline for CFI and taint propagation
- **[ST-WebAgentBench](https://arxiv.org/abs/2410.06703)** -- safety/trustworthiness with policy-paired evaluation; Completion Under Policy metric
- **[WASP](https://arxiv.org/abs/2504.18575)** -- web agent prompt injection benchmark; agents begin following malicious instructions even without completing them
- **[RedTeamCUA](https://arxiv.org/abs/2505.21936)** -- adversarial testing for computer-use agents in hybrid web-OS environments
- **[OS-Harm](https://arxiv.org/abs/2506.14866)** -- safety benchmark for computer-use agents; misuse, injection, misbehaviour
- **[ToolHijacker](https://arxiv.org/abs/2504.19793)** -- tool selection hijacking via injected tool documents; informs plugin trust boundaries
- **[Adaptive Attacks Break IPI Defenses](https://aclanthology.org/2025.findings-naacl.395/)** -- NAACL 2025; adaptive attackers bypass common IPI defenses; informs semantic firewalls
- **[AdapTools](https://arxiv.org/abs/2602.20720)** -- tool-oriented indirect prompt injection via third-party tool outputs
- **[Prompt Injection on Agentic Coding Assistants](https://arxiv.org/abs/2601.17548)** -- trust boundary failures in code agents; informs honeypot + taint design
- **[Memory Poisoning Attack and Defense](https://arxiv.org/abs/2601.05504)** -- persistent memory poisoning; informs v2 memory write policies and provenance
- **[Memory Injection via Query-Only Access](https://openreview.net/forum?id=QINnsnppv8)** -- query-only attacker drives harmful memory writes; informs capability-based access
- **[VIGIL + SIREN](https://arxiv.org/abs/2601.05755)** -- verify-before-commit protocol + tool-stream injection benchmark; validates scope envelopes
- **[ChainFuzzer](https://arxiv.org/abs/2603.12614)** -- greybox fuzzing for workflow-level multi-tool vulnerabilities; unique for DAG security
- **[ToolFuzz](https://github.com/eth-sri/ToolFuzz)** -- ETH SRI fuzzing framework for agent tools; adversarial input generation
- **[Unit 42 Memory Poisoning PoC](https://unit42.paloaltonetworks.com/indirect-prompt-injection-poisons-ai-longterm-memory/)** -- real-world IPI to long-term memory; practical mitigation reference

### Self-Improvement + Meta-Learning
- **[HyperAgents](https://github.com/facebookresearch/Hyperagents)** -- self-referential agents, stepping stones archive
- **[code-review-graph](https://github.com/nicobailon/code-review-graph)** -- Tree-sitter blast radius, 6.8x fewer tokens
- **[Darwin Godel Machine](https://arxiv.org/abs/2505.22954)** -- self-improving coding agents, SWE-bench 20%->50%
- **[WISE-Flow](https://arxiv.org/abs/2601.08158)** -- interaction-to-workflow distillation
- **[HyEvo](https://arxiv.org/abs/2603.19639)** -- self-evolving hybrid agentic workflows; variation operators over DAGs; informs v2 replan
- **[EvoAgentX](https://github.com/EvoAgentX/EvoAgentX)** -- open-source evolving workflow framework; mutation/selection/experiment tracking
- **[MermaidFlow](https://openreview.net/forum?id=bhPaXhWVKG)** -- verifiable IR (Mermaid graphs) for safety-constrained workflow evolution
- **[ToolTree](https://arxiv.org/abs/2603.12740)** -- MCTS for tool-use decision making; search tree over tool choices + parameters
- **[Workflow Optimization for LLM Agents survey](https://arxiv.org/abs/2603.22386)** -- 2026 survey mapping search-based and learning-based workflow optimisation

### Orchestration + Workflow
- **[Kong](https://github.com/amruth-sn/kong)** -- batch LLM chunking, $0.019/function
- **[Deep Agents](https://github.com/langchain-ai/deepagents)** -- autonomous compression, structured summaries
- **[OpenDev](https://arxiv.org/abs/2603.05344)** -- 5-stage progressive compaction, event reminders
- **[AgentConductor](https://arxiv.org/abs/2602.17100)** -- topology evolution, Plan Density Score
- **[DOVA](https://arxiv.org/abs/2603.13327)** -- deliberation-first, adversarial debate
- **[AFlow](https://github.com/FoundationAgents/AFlow)** -- MCTS over workflow space
- **[MARBLE/MultiAgentBench](https://github.com/ulab-uiuc/MARBLE)** -- star/chain/graph topologies (ACL 2025)
- **[LLMSched](https://arxiv.org/abs/2504.03444)** -- DAG-structured LLM inference scheduler; resource-aware scheduling and queueing
- **[Nalar](https://arxiv.org/abs/2601.05109)** -- serving framework for agent workflows; performance instrumentation + scheduling hooks
- **[Efficient LLM Serving for Agentic Workflows](https://arxiv.org/abs/2603.16104)** -- throughput/latency trade-offs for multi-step agent call patterns
- **[Batch Query Processing for Agentic Workflows](https://arxiv.org/abs/2509.02121)** -- cross-call batching/optimization; prefix sharing
- **[Murakkab](https://arxiv.org/abs/2502.17350)** -- decoupled async execution inside LLM workflows; preemption and work-stealing
- **[MaAS](https://github.com/bingreeky/MaAS)** -- multi-agent architecture search via agentic supernet; automated topology selection
- **[goose](https://github.com/aaif-goose/goose)** -- open-source agent host combining ACP providers, MCP extensions, recipes, and interactive session UX; validates Maestro's low-priority ACP stance while informing future `maestro chat` context-file auto-loading and MCP Apps ergonomics
- **[Multica](https://github.com/multica-ai/multica)** -- managed agents platform with issue board, blockers, agent identity, reusable skills, and local daemon runtimes; strong product/UI reference for future dashboard collaboration surfaces, not a core architecture match for Maestro's run-and-exit scheduler
- **[ClawTeam-OpenClaw](https://github.com/win4r/ClawTeam-OpenClaw)** -- git-worktree-per-agent swarm orchestration with Windows subprocess fallback; useful operational reference for worker identity/session isolation and monitoring ergonomics, not a replacement for Maestro DAG plans
- **[Claude Code on Steroids](https://github.com/GadaaLabs/claude-code-on-steroids)** -- extended obra/superpowers distribution with trigger tables, skill chains, context compression, and mechanical-task routing; useful reference for `maestro skill` evolution toward recommendation/trigger/chain UX while keeping Maestro engine-agnostic

### Benchmarks + Evaluation
- **[Terminal-Bench 2.0](https://arxiv.org/abs/2601.11868)** -- 89 hard CLI tasks, frontier <65%; most aligned with Maestro's terminal-first nature
- **[tau2-Bench-Verified](https://arxiv.org/abs/2506.07982)** -- dual-control benchmark; tool-agent-user interaction
- **[SWE-EVO](https://arxiv.org/abs/2512.18470)** -- long-horizon software evolution; 44pp gap from single-shot; validates watch + worktree design
- **[MLE-bench](https://arxiv.org/abs/2410.07095)** -- ML engineering agents on Kaggle tasks; budget/time/iteration constraints
- **[WorkArena](https://arxiv.org/abs/2403.07718)** -- enterprise knowledge worker tasks; operational robustness
- **[WebChoreArena](https://arxiv.org/abs/2506.01952)** -- long, tedious web tasks; tests consistency under boredom
- **[ToolDial](https://arxiv.org/abs/2503.00564)** -- multi-turn tool-augmented dialogue; parameter filling + API navigation
- **[Can AI Agents Agree?](https://arxiv.org/abs/2603.01213)** -- Byzantine consensus with LLMs
- **[DECKBench](https://arxiv.org/abs/2602.13318)** -- multi-agent workflow benchmark; multi-turn instruction-following and compliance
- **[Judge Reliability Harness](https://github.com/RANDCorporation/judge-reliability-harness)** -- RAND; reliability tests for LLM judges (formatting invariance, verbosity bias, calibration)
- **[Diagnosing LLM-as-Judge via IRT](https://arxiv.org/abs/2602.00521)** -- Item Response Theory for judge consistency and human alignment
- **[LLM-as-a-Judge survey](https://www.cell.com/the-innovation/pdf/S2666-6758%2825%2900456-4.pdf)** -- systematizes judge construction, bias, reliability issues
- **[Sage](https://openreview.net/forum?id=JFTSZa2stt)** -- judge evaluation suite without human annotation ground truth
- **[ProtocolBench](https://arxiv.org/abs/2510.17149)** -- benchmarks multi-agent protocol choices (latency, overhead, robustness under failure)
- **[Benchmarking multi-agent orchestration](https://arxiv.org/abs/2603.22651)** -- compares pipeline/fan-out/supervisor/self-correcting architectures empirically

### Books
- **[Agentic Design Patterns](https://github.com/DanieleSalatti/AgenticDesignPatterns)** (Gulli, 2025, 424 pg) -- 21 patterns, Maestro covers 100% of resource-aware + guardrails chapters

### Inter-Agent Communication
- **[AgentPrune](https://github.com/yanweiyue/AgentPrune)** -- message pruning on agent graphs; cost control for multi-agent topologies
- **[G-Designer](https://arxiv.org/abs/2410.11782)** -- task-aware topology design via GNNs; informs council topology selection
- **[Graph-of-Agents](https://openreview.net/forum?id=34cANdsHKV)** -- graph-based agent selection and edge construction by relevance
- **[ACP](https://agentcommunicationprotocol.dev/introduction/welcome)** -- Agent Communication Protocol specification; handoff semantics

### Code Analysis
- **[Joern](https://github.com/joernio/joern)** -- code property graph platform; multi-language graph queries; informs v2 structural context
- **[Code Property Graph spec](https://github.com/ShiftLeftSecurity/codepropertygraph)** -- language-agnostic CPG representation for incremental code analysis
- **[Tree-sitter](https://github.com/tree-sitter/tree-sitter)** -- incremental AST parsing; candidate to replace regex-based symbol extraction in `symbols.py`
- **[GitNexus](https://github.com/abhigyanpatwari/GitNexus)** -- codebase → knowledge graph + MCP server; Tree-sitter AST (14 langs), Leiden community detection, BM25+semantic+RRF hybrid search, process tracing; consumable as MCP tool for deep structural analysis; inspires RRF fusion, confidence-weighted relations, community detection for suggest.py; PolyForm Noncommercial licence
- **[code-review-graph](https://github.com/tirth8205/code-review-graph)** -- Python/SQLite/NetworkX persistent codebase map, 22 MCP tools, Tree-sitter (18 langs), blast radius with PageRank (8.2× fewer tokens benchmarked), Leiden community detection, execution flow tracing; **MIT licence**; primary reference for `codebase_graph.py` patterns (DiGraph cache, risk scoring, flow detection)
- **[codegraph](https://github.com/optave/codegraph)** -- TypeScript code intelligence CLI, 30+ MCP tools, Tree-sitter (17 langs), function-level call graph with dataflow + CFG, CI quality gates (`--no-new-cycles`, `--max-blast-radius`), Leiden clustering, co-change analysis, architecture boundaries; **Apache-2.0**
- **[Codemem](https://github.com/cogniplex/codemem)** -- Rust single-binary memory engine, 32 MCP tools, graph-vector hybrid (petgraph + HNSW), bi-temporal model (Zep-inspired), SCIP cross-references fused with ast-grep, 9-component hybrid scoring, PageRank blast radius; **Apache-2.0**; informs confidence fusion and temporal tracking for v2 Knowledge+Memory

### Execution Isolation
- **[Agent Sandbox (k8s-sigs)](https://github.com/kubernetes-sigs/agent-sandbox)** -- Kubernetes-native sandbox CRD for untrusted agent code; primary v2 backend reference
- **[Firecracker](https://github.com/firecracker-microvm/firecracker)** -- minimalist KVM-based microVM; high-isolation per-task execution backend
- **[gVisor](https://github.com/google/gvisor)** -- OCI-compatible container sandbox; middle ground between containers and microVMs
- **[Kata Containers](https://github.com/kata-containers/kata-containers)** -- lightweight VM containers; Docker UX with VM-grade isolation
- **[nsjail](https://github.com/google/nsjail)** -- Linux namespace sandbox; local sandboxing without full container plumbing
- **[Wasmtime](https://github.com/bytecodealliance/wasmtime)** -- WebAssembly runtime with capability-style sandbox; deterministic task runners

### Model Routing
- **[LLMRouterBench](https://github.com/ynulihao/LLMRouterBench)** -- large-scale routing benchmark + unified framework; validates predictive routing
- **[Cascade routing (ETH SRI)](https://github.com/eth-sri/cascade-routing)** -- unified routing+cascading strategy; informs auto-escalation
- **[RouteLLM](https://github.com/lm-sys/routellm)** -- strong-vs-weak model selection with cost thresholds; learned/heuristic routers
- **[LiteLLM Router](https://docs.litellm.ai/docs/proxy/configs)** -- production routing strategies (least-busy, usage-based, latency-based, fallbacks)
- **[Bifrost](https://github.com/maximhq/bifrost)** -- high-performance AI gateway; unified provider access with failover and caching

### Capability + Tool Access Control
- **[Progent](https://arxiv.org/abs/2504.11703)** -- programmable privilege control DSL for tool calls; informs `allowed_tools:` design
- **[MiniScope](https://arxiv.org/abs/2512.11147)** -- least-privilege framework for agent authorization; prevents permission creep
- **[AC4A](https://arxiv.org/abs/2603.20933)** -- practical permission framework for agents across APIs; code released
- **[Agent Access Control (AAC)](https://arxiv.org/abs/2510.11108)** -- information-flow-based access control for agents; provenance and scoped outputs
- **[OPA](https://github.com/open-policy-agent/opa)** -- general-purpose policy engine (Rego); mature external policy language reference
- **[Cedar](https://github.com/cedar-policy/cedar)** -- authorization policy language; analyzability and audit-friendly rules
- **[Verifiably Safe Tool Use (ICSE 2026)](https://conf.researchr.org/details/icse-2026/icse-2026-nier/41/)** -- research direction for verifiable safety in tool use

### State Recovery + Durability
- **[Temporal](https://github.com/temporalio/temporal)** -- durable execution with event history replay; gold standard for "resume after failure"
- **[Restate](https://github.com/restatedev/restate)** -- durable execution runtime; long-running process semantics and exactly-once recovery
- **[DBOS Transact](https://github.com/dbos-inc/dbos-transact-py)** -- Postgres-backed durable workflows; embedded durability for Python
- **[Hatchet](https://github.com/hatchet-dev/hatchet)** -- durable task queue + checkpointed execution; DAG/durable-task hybrid patterns

### Observability
- **[OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)** -- standardized agent span attributes; aligns OTLP exporter
- **[OpenInference spec](https://arize-ai.github.io/openinference/spec/)** -- LLM call/agent step/tool trace format; complement to OTel
- **[Arize Phoenix](https://github.com/arize-ai/phoenix)** -- open-source agent tracing/eval on OTel/OpenInference
- **[Langfuse](https://github.com/langfuse/langfuse)** -- self-hostable LLM observability; trace-eval-iterate loops

### Inference Optimization
- **[TurboQuant](https://arxiv.org/abs/2504.19874)** (ICLR 2026) -- training-free KV cache quantization to 3 bits via PolarQuant + QJL; 6× compression, zero accuracy loss; benefits local engines (ollama/llama via llama.cpp); orthogonal to Maestro's context compression (before-model vs inside-model)

### Protocols
- **[A2A Protocol](https://github.com/a2aproject/A2A)** -- agent-to-agent interop
- **[MCP Specification](https://modelcontextprotocol.io/specification/2025-11-25)** -- Model Context Protocol
- **[AG-UI](https://github.com/ag-ui-protocol/ag-ui)** -- Agent-User Interaction Protocol
